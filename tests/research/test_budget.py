from __future__ import annotations

from pathlib import Path

import pytest

from autoalpha.research.budget import BudgetExceeded, ExperimentBudgetLedger


def test_budget_and_gate_decisions_are_machine_enforced(tmp_path: Path) -> None:
    ledger = ExperimentBudgetLedger(
        tmp_path / "experiments.jsonl",
        max_generation=3,
        max_family=2,
    )
    first = ledger.reserve("e1", "g1", "momentum")
    second = ledger.reserve("e2", "g1", "momentum")

    rejected = ledger.close(first, admission_decision="REJECTED", reason="capacity")
    retained = ledger.close(second, admission_decision="RESEARCH")

    assert rejected.status == "REJECTED"
    assert rejected.admission_decision == "REJECTED"
    assert retained.status == "RETAINED"
    assert retained.admission_decision == "RESEARCH"
    with pytest.raises(ValueError, match="already closed"):
        ledger.close(second, admission_decision="APPROVED_FOR_PAPER")
    with pytest.raises(BudgetExceeded, match="Family"):
        ledger.reserve("e3", "g1", "momentum")


def test_experiment_ids_are_unique(tmp_path: Path) -> None:
    ledger = ExperimentBudgetLedger(
        tmp_path / "experiments.jsonl",
        max_generation=2,
        max_family=2,
    )
    ledger.reserve("e1", "g1", "value")

    with pytest.raises(ValueError, match="Duplicate"):
        ledger.reserve("e1", "g1", "value")
