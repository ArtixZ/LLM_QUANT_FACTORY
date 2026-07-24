from __future__ import annotations

import pytest

from autoalpha.security.sandbox import IsolatedCandidateRunner, SandboxError, SandboxLimits


def test_sandbox_runs_pure_candidate_with_json_contract() -> None:
    source = """
def main(payload):
    return {"score": payload["x"] * 2}
"""

    result = IsolatedCandidateRunner().run(source, {"x": 3})

    assert result == {"score": 6}


def test_sandbox_stops_wall_clock_overrun() -> None:
    source = """
def main(payload):
    while True:
        pass
"""
    runner = IsolatedCandidateRunner(limits=SandboxLimits(timeout_seconds=0.2, cpu_seconds=1))

    with pytest.raises(SandboxError, match="wall-clock limit"):
        runner.run(source)
