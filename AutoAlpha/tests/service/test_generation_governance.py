from __future__ import annotations

from pathlib import Path

import pytest

from autoalpha.service.store import ServiceStore


def test_generation_budgets_and_blind_evidence_are_candidate_bound(tmp_path: Path) -> None:
    store = ServiceStore(tmp_path / "service.sqlite3")
    store.ensure_generation(
        generation_id="g1",
        protocol_version="v3",
        maximum_candidates=2,
        maximum_holdout_attempts=1,
    )
    state = store.reserve_generation_experiment(
        candidate_hash="factor-a",
        generation_id="g1",
        factor_id="factor-a",
        family="liquidity",
        iteration=1,
        maximum_family_candidates=1,
    )
    assert state["candidate_attempts"] == 1
    store.close_generation_experiment(
        "factor-a", status="REJECTED", public_verdict="PUBLIC_GATES_FAILED"
    )

    with pytest.raises(PermissionError, match="family"):
        store.reserve_generation_experiment(
            candidate_hash="factor-b",
            generation_id="g1",
            factor_id="factor-b",
            family="liquidity",
            iteration=2,
            maximum_family_candidates=1,
        )

    record = store.record_blind_evaluation(
        candidate_hash="portfolio-a",
        generation_id="g1",
        iteration=1,
        holdout_verdict="BLIND_GENERALIZATION_PASSED",
        holdout_passed=True,
        holdout_evidence_hash="a" * 64,
    )
    assert record["holdout_passed"] is True
    assert "metrics" not in record
    store.update_blind_capital(
        "portfolio-a",
        capital_verdict="CAPITAL_SIMULATION_PASSED",
        capital_passed=True,
        capital_evidence_hash="b" * 64,
    )
    with pytest.raises(PermissionError, match="holdout budget"):
        store.record_blind_evaluation(
            candidate_hash="portfolio-b",
            generation_id="g1",
            iteration=2,
            holdout_verdict="BLIND_NO_GENERALIZATION",
            holdout_passed=False,
            holdout_evidence_hash="c" * 64,
        )


def test_blind_evaluations_can_be_isolated_by_generation(tmp_path: Path) -> None:
    store = ServiceStore(tmp_path / "service.sqlite3")
    for generation_id, candidate_hash in (("task-a", "portfolio-a"), ("task-b", "portfolio-b")):
        store.ensure_generation(
            generation_id=generation_id,
            protocol_version="v1",
            maximum_candidates=2,
            maximum_holdout_attempts=1,
        )
        store.record_blind_evaluation(
            candidate_hash=candidate_hash,
            generation_id=generation_id,
            iteration=1,
            holdout_verdict="BLIND_GENERALIZATION_PASSED",
            holdout_passed=True,
            holdout_evidence_hash=generation_id * 8,
        )

    records = store.blind_evaluations(generation_id="task-b")

    assert [record["candidate_hash"] for record in records] == ["portfolio-b"]


def test_exhausted_generation_rolls_to_public_only_without_reusing_holdout(tmp_path: Path) -> None:
    store = ServiceStore(tmp_path / "service.sqlite3")
    first = store.ensure_running_generation(
        base_generation_id="g1",
        protocol_version="v3",
        maximum_candidates=1,
        maximum_holdout_attempts=2,
    )
    store.reserve_generation_experiment(
        candidate_hash="factor-a",
        generation_id=first["generation_id"],
        factor_id="factor-a",
        family="test",
        iteration=1,
        maximum_family_candidates=2,
    )

    second = store.ensure_running_generation(
        base_generation_id="g1",
        protocol_version="v3",
        maximum_candidates=1,
        maximum_holdout_attempts=2,
    )

    assert second["generation_id"] == "g1-public-002"
    assert second["maximum_holdout_attempts"] == 0
    assert store.generation_state("g1")["status"] == "SEALED"  # type: ignore[index]


def test_protocol_change_seals_previous_active_generation(tmp_path: Path) -> None:
    store = ServiceStore(tmp_path / "service.sqlite3")
    store.ensure_running_generation(
        base_generation_id="institutional-v3",
        protocol_version="v3",
        maximum_candidates=10,
        maximum_holdout_attempts=2,
    )

    current = store.ensure_running_generation(
        base_generation_id="institutional-v4",
        protocol_version="v4",
        maximum_candidates=10,
        maximum_holdout_attempts=2,
    )

    assert current["generation_id"] == "institutional-v4"
    previous = store.generation_state("institutional-v3")
    assert previous is not None and previous["status"] == "SEALED"


def test_underused_same_protocol_chain_recovers_from_cross_scope_seal(tmp_path: Path) -> None:
    store = ServiceStore(tmp_path / "service.sqlite3")
    store.ensure_generation(
        generation_id="g-task",
        protocol_version="v1",
        maximum_candidates=10,
        maximum_holdout_attempts=2,
    )
    store.ensure_generation(
        generation_id="g-task-public-002",
        protocol_version="v1",
        maximum_candidates=10,
        maximum_holdout_attempts=0,
    )
    with store.connection() as connection:
        connection.execute("UPDATE research_generations SET status='SEALED', candidate_attempts=2")

    recovered = store.ensure_running_generation(
        base_generation_id="g-task",
        protocol_version="v1",
        maximum_candidates=10,
        maximum_holdout_attempts=2,
    )

    assert recovered["generation_id"] == "g-task-public-002"
    assert recovered["status"] == "ACTIVE"
    assert recovered["recovered_cross_scope_seal"]
    assert recovered["maximum_candidates"] == 8
    assert recovered["maximum_holdout_attempts"] == 2


def test_restart_reconciles_reserved_generation_experiment(tmp_path: Path) -> None:
    store = ServiceStore(tmp_path / "service.sqlite3")
    store.ensure_generation(
        generation_id="g1",
        protocol_version="v3",
        maximum_candidates=2,
        maximum_holdout_attempts=1,
    )
    store.update_state(run_id="run-1", iteration=1)
    store.begin_iteration("run-1", 1)
    store.reserve_generation_experiment(
        candidate_hash="factor-a",
        generation_id="g1",
        factor_id="factor-a",
        family="liquidity",
        iteration=1,
        maximum_family_candidates=2,
    )

    assert store.reconcile_orphaned_iterations()
    reconciled = store.reconcile_orphaned_generation_experiments()

    assert reconciled == [{"candidate_hash": "factor-a", "generation_id": "g1", "iteration": 1}]
    with store.connection() as connection:
        experiment = connection.execute(
            "SELECT status, public_verdict FROM generation_experiments WHERE candidate_hash=?",
            ("factor-a",),
        ).fetchone()
    assert dict(experiment) == {
        "status": "CRASHED",
        "public_verdict": "SERVICE_RESTART_INTERRUPTED",
    }
    assert store.generation_state("g1")["candidate_attempts"] == 1  # type: ignore[index]
