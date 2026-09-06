from __future__ import annotations

from pathlib import Path

import pytest

from autoalpha.governance.audit import HashChainAuditLog
from autoalpha.governance.holdout import (
    ApprovalAuthority,
    HoldoutJudge,
    HoldoutPermissionError,
)


def test_holdout_approval_is_candidate_bound_and_single_use(tmp_path: Path) -> None:
    authority = ApprovalAuthority(b"a" * 32)
    judge = HoldoutJudge(authority, HashChainAuditLog(tmp_path / "holdout.jsonl"))
    token = authority.issue("candidate-hash", "generation-v1")

    result = judge.evaluate(
        token=token,
        candidate_hash="candidate-hash",
        generation="generation-v1",
        trusted_evaluator=lambda: {
            "decision": "APPROVED",
            "rank_ic_mean": 0.04,
            "private_daily_labels": [1, 2, 3],
        },
    )

    assert result["decision"] == "APPROVED"
    assert set(result) == {"decision", "evidence_hash"}
    assert len(result["evidence_hash"]) == 64
    with pytest.raises(HoldoutPermissionError, match="already consumed"):
        judge.evaluate(
            token=token,
            candidate_hash="candidate-hash",
            generation="generation-v1",
            trusted_evaluator=lambda: {},
        )


def test_holdout_approval_rejects_another_candidate(tmp_path: Path) -> None:
    authority = ApprovalAuthority(b"b" * 32)
    judge = HoldoutJudge(authority, HashChainAuditLog(tmp_path / "holdout.jsonl"))
    token = authority.issue("candidate-a", "generation-v1")

    with pytest.raises(HoldoutPermissionError, match="another candidate"):
        judge.evaluate(
            token=token,
            candidate_hash="candidate-b",
            generation="generation-v1",
            trusted_evaluator=lambda: {},
        )
