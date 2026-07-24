from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autoalpha.security.guard import CandidatePolicy, validate_candidate_source

_RESULT_PREFIX = "AUTOALPHA_RESULT="


class SandboxError(RuntimeError):
    """Candidate failed or violated the execution sandbox."""


@dataclass(frozen=True)
class SandboxLimits:
    timeout_seconds: float = 30.0
    cpu_seconds: int = 30
    max_output_bytes: int = 1_000_000

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0 or self.cpu_seconds <= 0 or self.max_output_bytes <= 0:
            raise ValueError("Sandbox limits must be positive")


class IsolatedCandidateRunner:
    def __init__(
        self,
        *,
        policy: CandidatePolicy | None = None,
        limits: SandboxLimits | None = None,
    ) -> None:
        self.policy = policy or CandidatePolicy()
        self.limits = limits or SandboxLimits()

    def run(self, source: str, payload: dict[str, Any] | None = None) -> Any:
        validate_candidate_source(source, self.policy)
        with tempfile.TemporaryDirectory(prefix="autoalpha-candidate-") as directory:
            root = Path(directory)
            candidate = root / "candidate.py"
            request = root / "request.json"
            worker = root / "worker.py"
            candidate.write_text(source, encoding="utf-8")
            request.write_text(json.dumps(payload or {}), encoding="utf-8")
            worker.write_text(_worker_source(), encoding="utf-8")
            command = [sys.executable, "-I", str(worker), str(candidate), str(request)]
            try:
                process = subprocess.run(
                    command,
                    cwd=root,
                    env=_minimal_environment(),
                    capture_output=True,
                    text=True,
                    timeout=self.limits.timeout_seconds,
                    check=False,
                    preexec_fn=_resource_limiter(self.limits.cpu_seconds),
                )
            except subprocess.TimeoutExpired as error:
                raise SandboxError(
                    f"Candidate exceeded {self.limits.timeout_seconds}s wall-clock limit"
                ) from error
            combined_size = len(process.stdout.encode()) + len(process.stderr.encode())
            if combined_size > self.limits.max_output_bytes:
                raise SandboxError("Candidate exceeded output limit")
            if process.returncode != 0:
                detail = process.stderr.strip().splitlines()[-1] if process.stderr.strip() else ""
                raise SandboxError(
                    f"Candidate process failed with exit {process.returncode}: {detail}"
                )
            result_lines = [
                line.removeprefix(_RESULT_PREFIX)
                for line in process.stdout.splitlines()
                if line.startswith(_RESULT_PREFIX)
            ]
            if len(result_lines) != 1:
                raise SandboxError("Candidate did not return exactly one result")
            return json.loads(result_lines[0])


def _minimal_environment() -> dict[str, str]:
    allowed = ("PATH", "SYSTEMROOT", "TMPDIR", "TEMP", "TMP")
    environment = {name: os.environ[name] for name in allowed if name in os.environ}
    environment.update(
        {
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "AUTOALPHA_SANDBOX": "1",
        }
    )
    return environment


def _resource_limiter(cpu_seconds: int):
    if sys.platform == "win32":
        return None

    def limit() -> None:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
        resource.setrlimit(resource.RLIMIT_FSIZE, (10_000_000, 10_000_000))
        if hasattr(resource, "RLIMIT_NPROC"):
            resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))

    return limit


def _worker_source() -> str:
    return f'''\
import json
import runpy
import socket
import sys
from pathlib import Path

candidate = Path(sys.argv[1]).resolve()
request = Path(sys.argv[2]).resolve()
root = candidate.parent
runtime_roots = tuple(Path(item).resolve() for item in sys.path if item)

def audit(event, args):
    if event.startswith("socket.") or event.startswith("subprocess.") or event == "os.system":
        raise PermissionError("network and child processes are disabled")
    if event == "open" and args:
        target = args[0]
        if isinstance(target, (str, bytes)):
            path = Path(target).resolve()
            mode = args[1] if len(args) > 1 and isinstance(args[1], str) else "r"
            inside_root = path == root or root in path.parents
            inside_runtime = any(path == base or base in path.parents for base in runtime_roots)
            if any(flag in mode for flag in "wax+") and not inside_root:
                raise PermissionError("writes outside sandbox are disabled")
            if not inside_root and not inside_runtime:
                raise PermissionError("reads outside sandbox and runtime are disabled")

sys.addaudithook(audit)
namespace = runpy.run_path(str(candidate), run_name="autoalpha_candidate")
entrypoint = namespace.get("main")
if not callable(entrypoint):
    raise TypeError("candidate must define main(payload)")
payload = json.loads(request.read_text(encoding="utf-8"))
result = entrypoint(payload)
print("{_RESULT_PREFIX}" + json.dumps(result, sort_keys=True, separators=(",", ":")))
'''
