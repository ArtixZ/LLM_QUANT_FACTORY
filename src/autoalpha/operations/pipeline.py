from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TaskResult:
    task_key: str
    output: dict[str, Any]
    cached: bool


class IdempotentPipeline:
    def __init__(self, root: Path) -> None:
        self.root = root

    def run(
        self,
        task_name: str,
        inputs: dict[str, Any],
        execute: Callable[[], dict[str, Any]],
    ) -> TaskResult:
        task_key = _task_key(task_name, inputs)
        directory = self.root / task_name
        result_path = directory / f"{task_key}.json"
        if result_path.exists():
            return TaskResult(task_key, _read_result(result_path, task_key), True)
        directory.mkdir(parents=True, exist_ok=True)
        lock_path = directory / f"{task_key}.lock"
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            raise RuntimeError(f"Task {task_key} is already running") from error
        os.close(descriptor)
        try:
            output = execute()
            envelope = {"task_key": task_key, "inputs": inputs, "output": output}
            _atomic_json(result_path, envelope)
            return TaskResult(task_key, output, False)
        finally:
            lock_path.unlink(missing_ok=True)


def _task_key(task_name: str, inputs: dict[str, Any]) -> str:
    canonical = json.dumps(
        {"task": task_name, "inputs": inputs},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:24]


def _read_result(path: Path, expected_key: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("task_key") != expected_key:
        raise RuntimeError(f"Pipeline result key mismatch: {path}")
    return payload["output"]


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
