from __future__ import annotations

import pytest

from autoalpha.security.guard import CandidatePolicy, PolicyViolation, validate_candidate_source


def test_guard_accepts_numeric_candidate() -> None:
    source = """
import numpy as np

def factor(values):
    return np.log1p(values)
"""
    tree = validate_candidate_source(source)

    assert tree is not None


@pytest.mark.parametrize(
    "source, expected",
    [
        ("import os\n", "import 'os'"),
        ("open('secret')\n", "name 'open'"),
        ("import prepare\nprepare._load_test_panel()\n", "attribute '_load_test_panel'"),
        ("getattr(object(), '_' + 'secret')\n", "name 'getattr'"),
    ],
)
def test_guard_rejects_capability_escalation(source: str, expected: str) -> None:
    with pytest.raises(PolicyViolation, match=expected):
        validate_candidate_source(source)


def test_guard_enforces_line_budget() -> None:
    policy = CandidatePolicy(max_lines=2)
    with pytest.raises(PolicyViolation, match="maximum is 2"):
        validate_candidate_source("x = 1\ny = 2\nz = 3\n", policy)
