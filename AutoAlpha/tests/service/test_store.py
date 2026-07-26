from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from autoalpha.service.store import ServiceStore


def test_store_persists_memory_metrics_and_hash_chained_logs(tmp_path: Path) -> None:
    store = ServiceStore(tmp_path / "service.sqlite3")
    store.save_settings({"model": "research-model"})
    store.update_state(state="RUNNING", run_id="run-1", iteration=1)
    store.begin_iteration("run-1", 1)
    store.append_event(
        "research",
        "EVALUATED",
        "candidate evaluated",
        "deterministic metrics produced",
        run_id="run-1",
        iteration=1,
        payload={"candidate_id": "F_1"},
    )
    store.finish_iteration(
        "run-1",
        1,
        status="COMPLETED",
        candidate_id="F_1",
        metrics={"incremental_net_ir": 0.4},
        decision="RESEARCH_ONLY_DATA_BLOCKED",
    )
    store.remember("run-1", 1, "evaluation", "F_1 net_ir=0.4")

    assert store.settings()["model"] == "research-model"
    assert store.metric_history()[0]["incremental_net_ir"] == 0.4
    assert store.recent_memories()[0]["content"] == "F_1 net_ir=0.4"
    assert store.verify_events() == 1


def test_store_versions_settings_atomically_without_duplicate_revisions(tmp_path: Path) -> None:
    store = ServiceStore(tmp_path / "service.sqlite3")
    store.save_settings({"model": "baseline", "data_path": "/data"})

    revision = store.save_settings_revision(
        {"model": "next", "data_path": "/data"},
        change_note="switch model",
        metadata={"api_key_replaced": True},
    )

    assert revision is not None
    assert revision["changed_keys"] == ["model"]
    assert revision["previous_values"]["model"] == "baseline"
    assert revision["values"]["model"] == "next"
    assert revision["metadata"] == {"api_key_replaced": True}
    assert "api_key" not in revision["values"]
    assert store.settings_revision(revision["id"])["fingerprint"] == revision["fingerprint"]
    assert store.save_settings_revision(
        {"model": "next"}, change_note="no change"
    ) is None
    assert len(store.settings_revisions()) == 1
    with pytest.raises(ValueError, match="Secrets cannot be stored"):
        store.save_settings_revision({"api_key": "secret"}, change_note="invalid")


def test_store_persists_generic_favorites_with_context(tmp_path: Path) -> None:
    path = tmp_path / "service.sqlite3"
    store = ServiceStore(path)

    created = store.set_favorite(
        "factor",
        "F_1",
        favorite=True,
        label="Volume Stability",
        context={"source_task_id": "task-a"},
    )
    store.set_favorite("research_task", "task-a", favorite=True, label="2026 research")

    assert created is not None
    assert created["context"] == {"source_task_id": "task-a"}
    assert store.favorite_ids("factor") == {"F_1"}
    assert [item["entity_id"] for item in store.favorites(entity_type="research_task")] == [
        "task-a"
    ]

    reopened = ServiceStore(path)
    assert reopened.favorite("factor", "F_1")["label"] == "Volume Stability"
    assert reopened.set_favorite("factor", "F_1", favorite=False) is None
    assert reopened.favorite("factor", "F_1") is None
    with pytest.raises(ValueError, match="32 KiB"):
        reopened.set_favorite(
            "screener_preset", "too-large", favorite=True, context={"payload": "x" * 40_000}
        )


def test_store_supplements_existing_metrics_without_losing_evidence(tmp_path: Path) -> None:
    store = ServiceStore(tmp_path / "service.sqlite3")
    store.begin_iteration("run-1", 1)
    store.finish_iteration(
        "run-1",
        1,
        status="COMPLETED",
        candidate_id="F_1",
        metrics={"incremental_net_ir": 0.4},
        decision="RESEARCH_ONLY_DATA_BLOCKED",
    )

    merged = store.supplement_iteration_metrics("F_1", {"sharpe_ratio": 0.4})

    assert merged == {"incremental_net_ir": 0.4, "sharpe_ratio": 0.4}
    assert store.metric_history()[0]["decision"] == "RESEARCH_ONLY_DATA_BLOCKED"


def test_store_preserves_staged_candidate_when_iteration_fails(tmp_path: Path) -> None:
    store = ServiceStore(tmp_path / "service.sqlite3")
    store.begin_iteration("run-1", 2)
    proposal = {"name": "candidate", "expression": {"operator": "field"}}
    store.stage_iteration_candidate("run-1", 2, candidate_id="F_candidate", proposal=proposal)

    store.finish_iteration("run-1", 2, status="FAILED", error="semantic rejection")

    record = store.iteration_record("run-1", 2)
    assert record is not None
    assert record["candidate_id"] == "F_candidate"
    assert record["proposal"] == proposal
    assert store.iteration_stats() == {
        "total": 1,
        "completed": 0,
        "failed": 1,
        "running": 0,
        "success_rate": 0.0,
    }


def test_store_can_restore_legacy_failed_iteration_evidence(tmp_path: Path) -> None:
    store = ServiceStore(tmp_path / "service.sqlite3")
    store.begin_iteration("run-legacy", 6)
    store.finish_iteration("run-legacy", 6, status="FAILED", error="legacy failure")

    restored = store.restore_failed_iteration_evidence(
        "run-legacy",
        6,
        candidate_id="F_restored",
        proposal={"name": "restored"},
    )

    assert restored
    assert store.iteration_record("run-legacy", 6)["candidate_id"] == "F_restored"


def test_store_reconciles_orphaned_running_iterations(tmp_path: Path) -> None:
    store = ServiceStore(tmp_path / "service.sqlite3")
    store.begin_iteration("run-interrupted", 17)

    reconciled = store.reconcile_orphaned_iterations()

    assert reconciled[0]["iteration"] == 17
    record = store.iteration_record("run-interrupted", 17)
    assert record is not None
    assert record["status"] == "FAILED"
    assert record["finished_at"] is not None
    assert "ServiceRestartInterrupted" in record["error"]


def test_direction_reconciliation_isolated_by_run_and_refunds_budget(tmp_path: Path) -> None:
    store = ServiceStore(tmp_path / "service.sqlite3")
    for generation_id in ("g-a", "g-b"):
        store.ensure_generation(
            generation_id=generation_id,
            protocol_version="v1",
            maximum_candidates=10,
            maximum_holdout_attempts=1,
        )
    campaign_a = store.start_direction_campaign(
        generation_id="g-a",
        direction="EXPLORE_NEW_MECHANISM",
        title="a",
        objective="a",
        diagnostic_score=1.0,
        rationale=[],
        evidence={},
        baseline={},
        maximum_attempts=3,
        started_iteration=1,
    )
    campaign_b = store.start_direction_campaign(
        generation_id="g-b",
        direction="EXPLORE_NEW_MECHANISM",
        title="b",
        objective="b",
        diagnostic_score=1.0,
        rationale=[],
        evidence={},
        baseline={},
        maximum_attempts=3,
        started_iteration=1,
    )
    store.begin_iteration("run-a", 1)
    store.finish_iteration("run-a", 1, status="FAILED")
    store.begin_iteration("run-b", 1)
    attempt_a = store.reserve_direction_attempt(
        campaign_id=campaign_a["id"],
        run_id="run-a",
        iteration=1,
        baseline={},
    )
    store.reserve_direction_attempt(
        campaign_id=campaign_b["id"],
        run_id="run-b",
        iteration=1,
        baseline={},
    )

    reconciled = store.reconcile_orphaned_direction_attempts(
        early_stop_consecutive_misses=3
    )

    assert reconciled == [
        {
            "attempt_id": attempt_a["id"],
            "run_id": "run-a",
            "iteration": 1,
            "campaign_id": campaign_a["id"],
            "budget_refunded": True,
        }
    ]
    assert store.direction_attempt(1, run_id="run-a")["status"] == "CANCELLED_OPERATIONAL"
    assert store.direction_attempt(1, run_id="run-b")["status"] == "RESERVED"
    assert store.direction_campaign(campaign_a["id"])["attempts_used"] == 0
    assert store.direction_campaign(campaign_b["id"])["attempts_used"] == 1


def test_store_closes_legacy_active_campaign_with_exhausted_budget(tmp_path: Path) -> None:
    store = ServiceStore(tmp_path / "service.sqlite3")
    store.ensure_generation(
        generation_id="g1",
        protocol_version="v1",
        maximum_candidates=10,
        maximum_holdout_attempts=1,
    )
    campaign = store.start_direction_campaign(
        generation_id="g1",
        direction="EXPLORE_NEW_MECHANISM",
        title="legacy",
        objective="legacy",
        diagnostic_score=1.0,
        rationale=[],
        evidence={},
        baseline={},
        maximum_attempts=1,
        started_iteration=1,
    )
    with store.connection() as connection:
        connection.execute(
            "UPDATE direction_campaigns SET attempts_used=1 WHERE id=?",
            (campaign["id"],),
        )

    recovered = store.reconcile_exhausted_direction_campaigns()

    assert recovered[0]["id"] == campaign["id"]
    updated = store.direction_campaign(campaign["id"])
    assert updated["status"] == "EXHAUSTED"
    assert updated["closure_reason"] == "RECOVERED_EXHAUSTED_CAMPAIGN"


def test_iteration_history_returns_proposal_metrics_and_failures(tmp_path: Path) -> None:
    store = ServiceStore(tmp_path / "service.sqlite3")
    store.begin_iteration("run-history", 1)
    store.finish_iteration(
        "run-history",
        1,
        status="COMPLETED",
        candidate_id="F_1",
        proposal={"name": "factor"},
        metrics={"sharpe_ratio": 1.2},
        decision="RESEARCH_ONLY",
    )

    history = store.iteration_history()

    assert history[0]["proposal"]["name"] == "factor"
    assert history[0]["metrics"]["sharpe_ratio"] == 1.2


def test_iteration_views_can_be_isolated_by_research_run(tmp_path: Path) -> None:
    store = ServiceStore(tmp_path / "service.sqlite3")
    for run_id, iteration, status in (
        ("run-a", 1, "COMPLETED"),
        ("run-b", 7, "FAILED"),
    ):
        store.begin_iteration(run_id, iteration)
        store.finish_iteration(run_id, iteration, status=status)

    assert [row["run_id"] for row in store.iteration_history(run_id="run-a")] == ["run-a"]
    assert store.iteration_stats(run_id="run-a") == {
        "total": 1,
        "completed": 1,
        "failed": 0,
        "running": 0,
        "success_rate": 1.0,
    }
    assert store.iteration_stats(run_id="run-b")["failed"] == 1


def test_store_detects_event_tampering(tmp_path: Path) -> None:
    store = ServiceStore(tmp_path / "service.sqlite3")
    store.append_event("audit", "START", "start", "original")
    with store.connection() as connection:
        connection.execute("UPDATE events SET message='modified' WHERE id=1")

    with pytest.raises(RuntimeError, match="hash chain failed"):
        store.verify_events()


def test_task_event_view_combines_run_and_pre_run_task_events(tmp_path: Path) -> None:
    store = ServiceStore(tmp_path / "service.sqlite3")
    store.append_event(
        "action",
        "TASK_CREATED",
        "created",
        "before first run",
        payload={"task_id": "task-a"},
    )
    store.append_event(
        "research",
        "ITERATION_COMPLETED",
        "completed",
        "run evidence",
        run_id="run-a",
    )
    store.append_event("action", "GLOBAL", "global", "control-plane event")

    events = store.events(run_id="run-a", task_id="task-a")

    assert [event["event"] for event in events] == [
        "TASK_CREATED",
        "ITERATION_COMPLETED",
    ]


def test_metric_history_is_continuous_across_store_instances(tmp_path: Path) -> None:
    path = tmp_path / "service.sqlite3"
    first = ServiceStore(path)
    first.update_state(state="RUNNING", run_id="run-1", iteration=3)
    first.begin_iteration("run-1", 3)
    first.finish_iteration("run-1", 3, status="COMPLETED", metrics={"coverage": 0.91})

    second = ServiceStore(path)

    assert second.state()["iteration"] == 3
    assert second.metric_history()[0]["coverage"] == 0.91


def test_store_migrates_phase_into_existing_service_state(tmp_path: Path) -> None:
    path = tmp_path / "legacy.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute(
        """CREATE TABLE service_state (
        singleton INTEGER PRIMARY KEY, state TEXT NOT NULL, run_id TEXT,
        iteration INTEGER NOT NULL, stop_requested INTEGER NOT NULL,
        updated_at TEXT NOT NULL, last_error TEXT)"""
    )
    connection.execute(
        """INSERT INTO service_state VALUES
        (1, 'WAITING_CONFIGURATION', NULL, 0, 0, '2026-07-15T00:00:00Z', NULL)"""
    )
    connection.commit()
    connection.close()

    store = ServiceStore(path)

    assert store.state()["phase"] == "CONFIGURE"


def test_factor_pool_and_portfolio_versions_preserve_last_accepted_champion(
    tmp_path: Path,
) -> None:
    store = ServiceStore(tmp_path / "service.sqlite3")
    proposal = {
        "name": "factor",
        "family": "reversal",
        "hypothesis": "test",
        "expected_direction": 1,
        "expression": {
            "operator": "field",
            "arguments": [],
            "parameters": {"name": "adj_close"},
        },
    }
    store.upsert_factor_pool(
        factor_id="F_1",
        source_iteration=1,
        proposal=proposal,
        metrics={"sharpe_ratio": 1.0},
        status="ELIGIBLE",
        status_reason="passed",
    )
    accepted = store.record_portfolio_decision(
        run_id="run-1",
        iteration=1,
        action="BOOTSTRAP",
        candidate_id="F_1",
        removed_factor_id=None,
        accepted=True,
        reason="initial",
        metrics={"portfolio_sharpe_ratio": 1.0},
        members=[("F_1", 1.0)],
    )
    store.record_portfolio_decision(
        run_id="run-1",
        iteration=2,
        action="HOLD",
        candidate_id="F_2",
        removed_factor_id=None,
        accepted=False,
        reason="no improvement",
        metrics={"portfolio_sharpe_ratio": 1.0},
        members=[("F_1", 1.0)],
    )

    champion = store.active_portfolio()

    assert champion is not None
    assert champion["id"] == accepted
    assert champion["members"][0]["factor_id"] == "F_1"
    assert len(store.portfolio_history()) == 2


def test_factor_reevaluation_batch_is_atomic(tmp_path: Path) -> None:
    store = ServiceStore(tmp_path / "service.sqlite3")
    proposal = {
        "name": "factor",
        "family": "reversal",
        "hypothesis": "test",
        "expected_direction": 1,
        "expression": {
            "operator": "field",
            "arguments": [],
            "parameters": {"name": "adj_close"},
        },
    }
    store.upsert_factor_pool(
        factor_id="F_1",
        source_iteration=1,
        proposal=proposal,
        metrics={"evaluation_protocol": "old"},
        status="ACTIVE",
        status_reason="old",
    )

    count = store.apply_factor_reevaluations(
        [
            {
                "factor_id": "F_1",
                "metrics": {"evaluation_protocol": "current", "sharpe_ratio": 1.2},
                "status": "ELIGIBLE",
                "status_reason": "current protocol passed",
            }
        ]
    )

    assert count == 1
    assert store.factor_pool_record("F_1")["metrics"]["evaluation_protocol"] == "current"
    with pytest.raises(KeyError, match="F_missing"):
        store.apply_factor_reevaluations(
            [
                {
                    "factor_id": "F_1",
                    "metrics": {"evaluation_protocol": "corrupted"},
                    "status": "SCREENED_OUT",
                    "status_reason": "should roll back",
                },
                {
                    "factor_id": "F_missing",
                    "metrics": {},
                    "status": "SCREENED_OUT",
                    "status_reason": "missing",
                },
            ]
        )
    assert store.factor_pool_record("F_1")["metrics"]["evaluation_protocol"] == "current"


def test_manual_backtest_records_are_persistent_and_isolated(tmp_path: Path) -> None:
    store = ServiceStore(tmp_path / "service.sqlite3")
    request = {
        "factor_ids": ["F_1", "F_2"],
        "weights": [1.0, 1.0],
        "start_date": "2020-01-02",
        "end_date": "2024-12-31",
        "initial_cash_cny": 1_000_000,
        "gross_exposure": 0.5,
        "holding_period_days": 5,
        "portfolio_mode": "long_short",
    }

    backtest_id = store.create_manual_backtest(request)
    store.complete_manual_backtest(
        backtest_id,
        metrics={"sharpe_ratio": 1.2, "simple_annual_return": 0.1},
        artifact_path="/tmp/manual.json",
        result_hash="a" * 64,
    )

    record = store.manual_backtest(backtest_id)
    assert record is not None
    assert record["status"] == "COMPLETED"
    assert record["request"]["factor_ids"] == ["F_1", "F_2"]
    assert record["metrics"]["sharpe_ratio"] == 1.2
    assert store.active_portfolio() is None

    favorite = store.update_manual_backtest_metadata(
        backtest_id,
        favorite=True,
        title="stable pair",
        notes="review after costs",
        tags=["low-turnover", "reviewed"],
    )
    assert favorite["favorite"]
    assert favorite["title"] == "stable pair"
    assert favorite["tags"] == ["low-turnover", "reviewed"]
    assert store.manual_backtests(favorite_only=True)[0]["id"] == backtest_id


def test_manual_holdout_exposure_is_hashed_and_blocks_all_generations(tmp_path: Path) -> None:
    store = ServiceStore(tmp_path / "service.sqlite3")
    proposal = {
        "name": "factor",
        "family": "reversal",
        "hypothesis": "test",
        "expected_direction": 1,
        "expression": {
            "operator": "field",
            "arguments": [],
            "parameters": {"name": "adj_close"},
        },
    }
    store.upsert_factor_pool(
        factor_id="F_1",
        source_iteration=1,
        proposal=proposal,
        metrics={},
        status="ELIGIBLE",
        status_reason="passed",
    )
    backtest_id = store.create_manual_backtest({"factor_ids": ["F_1"]})
    records = store.record_manual_research_exposures(
        backtest_id=backtest_id,
        generation_id="generation-v1",
        factor_ids=["F_1"],
        period_start="2020-01-01",
        period_end="2026-01-01",
        holdout_start="2025-01-01",
        holdout_end="2026-12-31",
    )

    assert records[0]["contaminated"]
    assert len(records[0]["evidence_hash"]) == 64
    assert store.contaminated_factor_ids("generation-v1") == {"F_1"}
    assert store.portfolio_contamination("generation-v1", ["F_1"])[0]["factor_id"] == "F_1"
    assert store.contaminated_factor_ids("generation-v2") == {"F_1"}
    assert store.portfolio_contamination("generation-v2", ["F_1"])[0]["factor_id"] == "F_1"
    with (
        pytest.raises(sqlite3.IntegrityError, match="append-only"),
        store.connection() as connection,
    ):
        connection.execute("DELETE FROM manual_research_exposures")


def test_service_lifecycle_persists_qualified_shadow_paper_transitions(tmp_path: Path) -> None:
    store = ServiceStore(tmp_path / "service.sqlite3")
    store.upsert_factor_pool(
        factor_id="F_1",
        source_iteration=1,
        proposal={"name": "factor", "family": "test"},
        metrics={},
        status="ELIGIBLE",
        status_reason="public gates passed",
    )

    shadow = store.transition_factor_lifecycle(
        "F_1", "SHADOW", actor="reviewer", reason="qualified for forward observation"
    )
    paper = store.transition_factor_lifecycle(
        "F_1", "PAPER", actor="reviewer", reason="shadow evidence accepted"
    )

    assert shadow["previous_state"] == "QUALIFIED"
    assert paper["state"] == "PAPER"
    assert len(store.factor_lifecycle_history("F_1")) == 3


def test_llm_artifacts_are_append_only_and_factor_knowledge_is_queryable(
    tmp_path: Path,
) -> None:
    store = ServiceStore(tmp_path / "service.sqlite3")
    proposal = {"name": "factor", "family": "reversal", "hypothesis": "test"}
    store.upsert_factor_pool(
        factor_id="F_1",
        source_iteration=1,
        proposal=proposal,
        metrics={},
        status="ELIGIBLE",
        status_reason="passed",
        source_task_id="task-one",
    )
    store.upsert_factor_pool(
        factor_id="F_2",
        source_iteration=2,
        proposal={"name": "peer", "family": "reversal"},
        metrics={},
        status="OBSERVE",
        status_reason="library",
        source_task_id="task-one",
    )
    artifact = store.record_llm_role_artifact(
        task_id="task-one",
        run_id="run-one",
        iteration=1,
        candidate_id="F_1",
        role="INDEPENDENT_REVIEWER",
        stage="PRE_EVALUATION",
        status="COMPLETED",
        artifact={"verdict": "CLEAR"},
        usage={"total_tokens": 42},
        prompt_hash="prompt",
        response_hash="response",
    )
    knowledge = store.upsert_factor_knowledge(
        factor_id="F_1",
        canonical_mechanism="PRICE_REVERSAL",
        mechanism_summary="Short-horizon reversal.",
        tags=["reversal"],
        review={"verdict": "CLEAR"},
        falsification={"results": []},
        related_factors=[
            {
                "factor_id": "F_2",
                "relation": "SAME_MECHANISM",
                "confidence": "high",
                "rationale": "Shared reversal mechanism",
            }
        ],
    )

    assert artifact["artifact"] == {"verdict": "CLEAR"}
    assert store.llm_role_summary(task_id="task-one")["roles"][
        "INDEPENDENT_REVIEWER"
    ]["total_tokens"] == 42
    assert store.llm_role_artifacts(candidate_id="F_1")[0]["usage"]["total_tokens"] == 42
    assert knowledge["edges"][0]["target_factor_id"] == "F_2"
    assert knowledge["edges"][0]["confidence"] == 0.9
    assert store.factor_knowledge_catalog(task_id="task-one")[0][
        "canonical_mechanism"
    ] == "PRICE_REVERSAL"
    with (
        pytest.raises(sqlite3.IntegrityError, match="append-only"),
        store.connection() as connection,
    ):
        connection.execute("DELETE FROM llm_role_artifacts")
