from __future__ import annotations

import json
from pathlib import Path

from autoalpha.service.autocombine_store import AutoCombineStore
from autoalpha.service.quantcombine_store import QuantCombineStore
from autoalpha.service.store import ServiceStore
from autoalpha.service.strategy_bus import stable_experiment_id
from autoalpha.service.system_jobs import (
    SUPPORTED_SYSTEM_JOB_TYPES,
    SystemJobRunner,
    _clamp_repair_protocol_to_capacity,
    build_gate_funnel_diagnostics,
)


def _proposal(name: str, field: str = "turnover_rate") -> dict[str, object]:
    return {
        "name": name,
        "family": "Liquidity",
        "canonical_mechanism": "liquidity",
        "hypothesis": f"{field} behavior",
        "expected_direction": 1,
        "expression": {"operator": "field", "parameters": {"name": field}, "arguments": []},
    }


def _store(tmp_path: Path) -> ServiceStore:
    store = ServiceStore(tmp_path / "autoalpha.sqlite3")
    for index in range(2):
        store.upsert_factor_pool(
            factor_id=f"F_{index}",
            source_iteration=index + 1,
            source_task_id="task-aa",
            proposal=_proposal(f"Factor {index}"),
            metrics={"long_only_sharpe_ratio": 1.0 + index / 10},
            status="ELIGIBLE",
            status_reason="test",
        )
    return store


def test_system_job_runner_backfills_factor_homogeneity(tmp_path: Path) -> None:
    store = _store(tmp_path)
    runtime = tmp_path / "runtime"
    behavior_root = runtime / "factor-behavior"
    behavior_root.mkdir(parents=True)
    (behavior_root / "latest.json").write_text(
        json.dumps(
            {
                "status": "COMPLETED",
                "snapshot_id": "behavior-test",
                "factors": {
                    "F_0": {
                        "behavior_cluster_id": "B_TEST",
                        "behavior_cluster_size": 2,
                        "behavior_cluster_role": "LEADER",
                        "behavior_nearest_factor_id": "F_1",
                        "behavior_nearest_similarity": 0.81,
                        "behavior_redundancy": "RELATED",
                    },
                    "F_1": {
                        "behavior_cluster_id": "B_TEST",
                        "behavior_cluster_size": 2,
                        "behavior_cluster_role": "MEMBER",
                        "behavior_nearest_factor_id": "F_0",
                        "behavior_nearest_similarity": 0.81,
                        "behavior_redundancy": "RELATED",
                    },
                },
            }
        )
    )
    runner = SystemJobRunner(
        store,
        autocombine_store=AutoCombineStore(store),
        quantcombine_store=QuantCombineStore(store),
        runtime_root=runtime,
    )
    store.enqueue_system_job(
        job_id="job-homogeneity",
        queue="system",
        job_type="factor_homogeneity_backfill",
        payload={},
        progress_total=2,
    )

    result = runner.run_next(queue="system")

    assert result["claimed"] is True
    assert result["job"]["status"] == "COMPLETED"
    knowledge = store.factor_knowledge("F_0")
    assert knowledge is not None
    assert knowledge["canonical_mechanism"] == "TURNOVER_LIQUIDITY"
    assert knowledge["review"]["raw_canonical_mechanism"] == "liquidity"
    assert knowledge["review"]["canonical_mechanism"] == "TURNOVER_LIQUIDITY"
    assert knowledge["review"]["behavior_cluster_id"] == "B_TEST"
    factor_metrics = store.factor_pool_record("F_0")["metrics"]  # type: ignore[index]
    assert factor_metrics["long_only_sharpe_ratio"] == 1.0
    assert factor_metrics["canonical_mechanism"] == "TURNOVER_LIQUIDITY"
    assert factor_metrics["homogeneity_cluster_id"] == "B_TEST"
    assert factor_metrics["homogeneity_nearest_factor_id"] == "F_1"
    assert factor_metrics["homogeneity_nearest_similarity"] == 0.81
    assert factor_metrics["behavior_cluster_role"] == "LEADER"
    assert store.materialized_snapshot("factor_homogeneity_backfill") is not None


def test_system_job_runner_materializes_factor_knowledge_map(tmp_path: Path) -> None:
    store = _store(tmp_path)
    runner = SystemJobRunner(
        store,
        autocombine_store=AutoCombineStore(store),
        quantcombine_store=QuantCombineStore(store),
        runtime_root=tmp_path / "runtime",
    )
    store.enqueue_system_job(
        job_id="job-knowledge-map",
        queue="system",
        job_type="factor_knowledge_map_sync",
        payload={},
        progress_total=2,
    )

    result = runner.run_next(queue="system")
    snapshot = store.materialized_snapshot("factor_knowledge_map")

    assert "factor_knowledge_map_sync" in SUPPORTED_SYSTEM_JOB_TYPES
    assert result["claimed"] is True
    assert result["job"]["status"] == "COMPLETED"
    assert result["job"]["result"]["protocol"] == "MATERIALIZED_FACTOR_KNOWLEDGE_MAP_V1"
    assert snapshot is not None
    assert snapshot["payload"]["factor_count"] == 2
    assert snapshot["payload"]["primary_metric_policy"] == "long_only_first"
    assert snapshot["source"] == "job:job-knowledge-map"
    assert snapshot["expires_at"] is not None
    assert snapshot["cache_state"]["status"] == "FRESH"
    assert snapshot["cache_state"]["source"] == "job:job-knowledge-map"


def test_system_job_runner_materializes_factor_library_with_injected_builder(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    runner = SystemJobRunner(
        store,
        autocombine_store=AutoCombineStore(store),
        quantcombine_store=QuantCombineStore(store),
        runtime_root=tmp_path / "runtime",
        factor_library_builder=lambda: {
            "summary": {"factor_count": 2},
            "factors": [{"factor_id": "F_0"}, {"factor_id": "F_1"}],
            "research_tasks": [],
            "data": {},
            "knowledge_integrity": {
                "protocol": "AUTOALPHA_FACTOR_KNOWLEDGE_INTEGRITY_V1",
                "complete": True,
            },
        },
    )
    store.enqueue_system_job(
        job_id="job-factor-library",
        queue="system",
        job_type="factor_library_refresh",
        payload={},
        progress_total=1,
    )

    result = runner.run_next(queue="system")
    snapshot = store.materialized_snapshot("factor_library")

    assert "factor_library_refresh" in SUPPORTED_SYSTEM_JOB_TYPES
    assert result["claimed"] is True
    assert result["job"]["status"] == "COMPLETED"
    assert result["job"]["result"]["protocol"] == "MATERIALIZED_FACTOR_LIBRARY_REFRESH_V1"
    assert result["job"]["result"]["processed_count"] == 2
    assert snapshot is not None
    assert snapshot["payload"]["api_payload_protocol"] == "MATERIALIZED_FACTOR_LIBRARY_API_V1"
    assert snapshot["source"] == "job:job-factor-library"
    assert snapshot["cache_state"]["status"] == "FRESH"


def test_system_job_runner_dispatches_market_data_sync_job(tmp_path: Path) -> None:
    store = ServiceStore(tmp_path / "autoalpha.sqlite3")
    calls: list[dict[str, object]] = []

    def fake_market_data_sync(job: dict[str, object]) -> dict[str, object]:
        calls.append(job)
        return {
            "protocol": "AUTOALPHA_MARKET_DATA_SYNC_JOB_V1",
            "status": "COMPLETED",
            "panel_rebuilt": True,
            "download_returncode": 0,
        }

    runner = SystemJobRunner(
        store,
        autocombine_store=AutoCombineStore(store),
        quantcombine_store=QuantCombineStore(store),
        runtime_root=tmp_path / "runtime",
        market_data_sync_runner=fake_market_data_sync,
    )
    store.enqueue_system_job(
        job_id="job-market-sync",
        queue="system",
        job_type="market_data_sync",
        payload={"dataset_ids": ["core_market"]},
        progress_total=1,
    )

    result = runner.run_next(queue="system")

    assert "market_data_sync" in SUPPORTED_SYSTEM_JOB_TYPES
    assert result["claimed"] is True
    assert result["job"]["status"] == "COMPLETED"
    assert result["job"]["result"]["protocol"] == "AUTOALPHA_MARKET_DATA_SYNC_JOB_V1"
    assert calls[0]["job_id"] == "job-market-sync"


def test_system_job_commands_pause_resume_and_cancel_queued_job(tmp_path: Path) -> None:
    store = ServiceStore(tmp_path / "autoalpha.sqlite3")
    store.enqueue_system_job(
        job_id="job-control",
        queue="system",
        job_type="factor_knowledge_map_sync",
        payload={},
    )

    paused = store.command_system_job(
        "job-control",
        command="pause",
        actor="tester",
        reason="operator pause",
    )
    resumed = store.command_system_job("job-control", command="resume", actor="tester")
    cancelled = store.command_system_job(
        "job-control",
        command="cancel",
        actor="tester",
        reason="operator cancel",
    )

    assert paused["status"] == "PAUSED"
    assert paused["lease_owner"] is None
    assert resumed["status"] == "QUEUED"
    assert cancelled["status"] == "CANCELLED"
    assert cancelled["finished_at"] is not None
    assert store.claim_system_job(queue="system", worker_id="worker") is None
    logs = store.system_job_logs("job-control")
    assert [log["event"] for log in logs[:3]] == [
        "COMMAND_CANCEL",
        "COMMAND_RESUME",
        "COMMAND_PAUSE",
    ]


def test_system_job_commands_request_running_job_cancellation(tmp_path: Path) -> None:
    store = ServiceStore(tmp_path / "autoalpha.sqlite3")
    store.enqueue_system_job(
        job_id="job-running-control",
        queue="system",
        job_type="factor_knowledge_map_sync",
        payload={},
    )
    claimed = store.claim_system_job(queue="system", worker_id="worker")
    assert claimed is not None

    requested = store.command_system_job(
        "job-running-control",
        command="cancel",
        actor="tester",
        reason="stop after checkpoint",
    )

    assert requested["status"] == "CANCEL_REQUESTED"
    assert requested["lease_owner"] == "worker"
    assert requested["lease_expires_at"] is not None
    assert store.claim_system_job(queue="system", worker_id="other") is None
    recent_logs = store.system_job_logs_for_jobs(["job-running-control"], limit_per_job=2)
    assert [log["event"] for log in recent_logs["job-running-control"]] == [
        "COMMAND_CANCEL",
        "CLAIMED",
    ]


def test_system_job_claim_honors_queue_and_global_capacity(tmp_path: Path) -> None:
    store = ServiceStore(tmp_path / "autoalpha.sqlite3")
    store.enqueue_system_job(
        job_id="job-a",
        queue="system",
        job_type="factor_knowledge_map_sync",
        payload={},
        priority=1,
        resource_group="cpu-heavy",
        max_workers=4,
    )
    store.enqueue_system_job(
        job_id="job-b",
        queue="system",
        job_type="factor_knowledge_map_sync",
        payload={},
        priority=2,
        resource_group="independent",
        max_workers=4,
    )
    store.enqueue_system_job(
        job_id="job-c",
        queue="batch",
        job_type="factor_knowledge_map_sync",
        payload={},
        priority=1,
        resource_group="independent",
        max_workers=4,
    )

    first = store.claim_system_job(
        queue="system",
        worker_id="worker-a",
        max_queue_running=1,
        max_global_running=2,
    )
    assert first is not None
    assert first["job_id"] == "job-a"
    assert (
        store.claim_system_job(
            queue="system",
            worker_id="worker-b",
            max_queue_running=1,
            max_global_running=2,
        )
        is None
    )
    queue_logs = store.system_job_logs("job-b")
    assert queue_logs[0]["event"] == "CLAIM_DEFERRED_QUEUE_CAPACITY"

    batch = store.claim_system_job(
        queue="batch",
        worker_id="worker-c",
        max_queue_running=1,
        max_global_running=2,
    )
    assert batch is not None
    assert batch["job_id"] == "job-c"
    assert (
        store.claim_system_job(
            queue="system",
            worker_id="worker-d",
            max_queue_running=2,
            max_global_running=2,
        )
        is None
    )
    global_logs = store.system_job_logs("job-b")
    assert global_logs[0]["event"] == "CLAIM_DEFERRED_GLOBAL_CAPACITY"

    store.update_system_job("job-a", status="COMPLETED", lease_owner=None, lease_expires_at=None)
    next_job = store.claim_system_job(
        queue="system",
        worker_id="worker-b",
        max_queue_running=2,
        max_global_running=2,
    )
    assert next_job is not None
    assert next_job["job_id"] == "job-b"


def test_system_job_summary_exposes_resource_utilization(tmp_path: Path) -> None:
    store = ServiceStore(tmp_path / "autoalpha.sqlite3")
    store.enqueue_system_job(
        job_id="job-running",
        queue="system",
        job_type="factor_knowledge_map_sync",
        payload={},
        resource_group="sqlite-writer",
        max_workers=1,
    )
    store.enqueue_system_job(
        job_id="job-queued",
        queue="system",
        job_type="strategy_bus_sync",
        payload={},
        resource_group="sqlite-writer",
        max_workers=1,
    )
    claimed = store.claim_system_job(queue="system", worker_id="worker-a")
    assert claimed is not None

    summary = store.system_job_summary()
    resource = summary["resource_utilization"][0]

    assert resource["queue"] == "system"
    assert resource["resource_group"] == "sqlite-writer"
    assert resource["capacity"] == 1
    assert resource["running"] == 1
    assert resource["queued"] == 1
    assert resource["saturated"] is True
    assert summary["saturated_resources"][0]["resource_group"] == "sqlite-writer"


def test_system_job_runner_honors_cancel_request_before_completion(tmp_path: Path) -> None:
    store = _store(tmp_path)
    runner = SystemJobRunner(
        store,
        autocombine_store=AutoCombineStore(store),
        quantcombine_store=QuantCombineStore(store),
        runtime_root=tmp_path / "runtime",
    )
    store.enqueue_system_job(
        job_id="job-cancel-before-complete",
        queue="system",
        job_type="factor_knowledge_map_sync",
        payload={},
        progress_total=2,
    )
    job = store.claim_system_job(
        queue="system",
        worker_id=runner.worker_id,
        lease_seconds=300,
    )
    assert job is not None
    store.command_system_job(
        "job-cancel-before-complete",
        command="cancel",
        actor="tester",
        reason="cancel during run",
    )

    result = runner.run_claimed(job)

    assert result["status"] == "CANCELLED"
    assert result["result"]["partial_result"]["protocol"] == (
        "MATERIALIZED_FACTOR_KNOWLEDGE_MAP_V1"
    )
    assert result["lease_owner"] is None
    assert store.system_job_logs("job-cancel-before-complete")[0]["message"] == (
        "CANCEL_REQUESTED -> CANCELLED"
    )


def test_materialized_snapshot_exposes_ttl_state(tmp_path: Path) -> None:
    store = ServiceStore(tmp_path / "autoalpha.sqlite3")

    fresh = store.upsert_materialized_snapshot(
        "fresh",
        {"value": 1},
        ttl_seconds=60,
        source="test.fresh",
    )
    no_ttl = store.upsert_materialized_snapshot("no_ttl", {"value": 2})
    stale = store.upsert_materialized_snapshot(
        "stale",
        {"value": 3},
        ttl_seconds=1,
        source="test.stale",
    )
    with store.connection() as connection:
        connection.execute(
            "UPDATE materialized_snapshots SET expires_at=? WHERE key='stale'",
            ("2000-01-01T00:00:00+00:00",),
        )
    stale = store.materialized_snapshot("stale")

    assert fresh["cache_state"]["status"] == "FRESH"
    assert fresh["cache_state"]["stale"] is False
    assert fresh["source"] == "test.fresh"
    assert no_ttl["cache_state"]["status"] == "NO_TTL"
    assert stale is not None
    assert stale["cache_state"]["status"] == "STALE"
    assert stale["cache_state"]["stale"] is True


def test_system_job_runner_seeds_formal_strategy_library(tmp_path: Path) -> None:
    store = _store(tmp_path)
    experiment_id = stable_experiment_id("TEST", "seed-candidate", "COMBINATION_CANDIDATE")
    store.upsert_strategy_experiment_object(
        experiment_id=experiment_id,
        stage="COMBINATION_CANDIDATE",
        object_type="factor_combination",
        source_system="TEST",
        source_id="seed-candidate",
        title="Seed candidate",
        status="RESEARCH_LEADER",
        market="CN_A",
        metrics={
            "portfolio_sharpe_ratio": 1.2,
            "portfolio_simple_annual_return": 0.18,
            "portfolio_max_drawdown": -0.08,
            "portfolio_walk_forward_worst_sharpe": 0.35,
        },
        evidence={"factor_ids": ["F_0", "F_1"], "weights": [0.55, 0.45]},
    )
    runner = SystemJobRunner(
        store,
        autocombine_store=AutoCombineStore(store),
        quantcombine_store=QuantCombineStore(store),
        runtime_root=tmp_path / "runtime",
    )
    store.enqueue_system_job(
        job_id="job-seed-strategy",
        queue="system",
        job_type="strategy_library_seed",
        payload={"limit": 5},
        progress_total=1,
    )

    result = runner.run_next(queue="system")
    strategies = store.formal_strategy_versions(limit=10)

    assert "strategy_library_seed" in SUPPORTED_SYSTEM_JOB_TYPES
    assert result["claimed"] is True
    assert result["job"]["status"] == "COMPLETED"
    assert result["job"]["result"]["protocol"] == "AUTOALPHA_STRATEGY_LIBRARY_SEED_V1"
    assert result["job"]["result"]["created_count"] == 1
    assert len(strategies) == 1
    assert strategies[0]["source_experiment_id"] == experiment_id
    assert strategies[0]["lifecycle"] == "RESEARCH"


def test_system_job_runner_freezes_only_public_validation_ready_strategies(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    passed_id = stable_experiment_id("TEST", "freeze-passed", "COMBINATION_CANDIDATE")
    blocked_id = stable_experiment_id("TEST", "freeze-blocked", "COMBINATION_CANDIDATE")
    for experiment_id, title, status, gate_status in (
        (passed_id, "Freeze passed", "QUALIFIED_CHAMPION", "PASSED"),
        (blocked_id, "Freeze blocked", "RESEARCH_LEADER", "REJECTED"),
    ):
        store.upsert_strategy_experiment_object(
            experiment_id=experiment_id,
            stage="COMBINATION_CANDIDATE",
            object_type="factor_combination",
            source_system="TEST",
            source_id=title,
            title=title,
            status=status,
            market="CN_A",
            evidence={
                "factor_ids": ["F_0", "F_1"],
                "weights": [0.55, 0.45],
                "gate_status": gate_status,
            },
        )
    runner = SystemJobRunner(
        store,
        autocombine_store=AutoCombineStore(store),
        quantcombine_store=QuantCombineStore(store),
        runtime_root=tmp_path / "runtime",
    )
    store.enqueue_system_job(
        job_id="job-seed-before-freeze",
        queue="system",
        job_type="strategy_library_seed",
        payload={"limit": 5},
        progress_total=2,
    )
    runner.run_next(queue="system")
    store.enqueue_system_job(
        job_id="job-freeze-public-validation",
        queue="system",
        job_type="strategy_public_validation_freeze",
        payload={"limit": 10},
        progress_total=2,
    )

    result = runner.run_next(queue="system")
    strategies = store.formal_strategy_versions(limit=10)
    by_source = {item["source_experiment_id"]: item for item in strategies}

    assert "strategy_public_validation_freeze" in SUPPORTED_SYSTEM_JOB_TYPES
    assert result["claimed"] is True
    assert result["job"]["status"] == "COMPLETED"
    assert result["job"]["result"]["protocol"] == (
        "AUTOALPHA_STRATEGY_PUBLIC_VALIDATION_FREEZE_V1"
    )
    assert result["job"]["result"]["frozen_count"] == 1
    assert result["job"]["result"]["skipped_count"] == 1
    skipped = result["job"]["result"]["skipped"][0]
    assert skipped["public_validation_gap"]["gate_status"] == "REJECTED"
    assert skipped["public_validation_gap"]["root_causes"] == ["MISSING_GATE_TELEMETRY"]
    assert skipped["public_validation_gap"]["operator_hint"] == (
        "INSPECT_SOURCE_CANDIDATE_GATE_TELEMETRY"
    )
    assert by_source[passed_id]["lifecycle"] == "FROZEN"
    assert by_source[blocked_id]["lifecycle"] == "RESEARCH"
    assert by_source[passed_id]["evidence"]["public_validation_passed"] is True


def test_gate_funnel_diagnostics_counts_failure_categories() -> None:
    class FakeAuto:
        def tasks(self) -> list[dict[str, str]]:
            return [{"task_id": "auto-1"}]

        def experiments(self, task_id: str, *, limit: int) -> list[dict[str, object]]:
            assert task_id == "auto-1"
            assert limit > 0
            return [
                {
                    "id": 1,
                    "gate_status": "REJECTED",
                    "qualification": "EVALUATED",
                    "failed_gates": ["max_drawdown_too_large", "correlation_too_high"],
                    "metrics": {},
                }
            ]

    class FakeQuant:
        def tasks(self) -> list[dict[str, str]]:
            return [{"task_id": "quant-1"}]

        def candidates(self, task_id: str, *, limit: int) -> list[dict[str, object]]:
            assert task_id == "quant-1"
            assert limit > 0
            return [
                {
                    "id": 2,
                    "gate_status": "PASSED",
                    "qualification": "QUALIFIED_CHAMPION",
                    "failed_gates": [],
                    "metrics": {},
                }
            ]

    result = build_gate_funnel_diagnostics(FakeAuto(), FakeQuant())  # type: ignore[arg-type]

    assert result["total_candidates"] == 2
    assert result["passed_candidates"] == 1
    assert result["protocol"] == "AUTOALPHA_GATE_FUNNEL_DIAGNOSTICS_V2"
    assert result["failure_categories"][0]["key"] in {
        "DRAWDOWN_OR_RISK",
        "CORRELATION_OR_INDEPENDENCE",
    }
    assert {item["key"] for item in result["root_causes"]} >= {
        "RISK_CONSTRAINT_BREACH",
        "FACTOR_INDEPENDENCE_INSUFFICIENT",
    }
    assert result["operator_actions"][0]["action"] == (
        "PROMOTE_GATE_PASSING_CANDIDATES_TO_STRATEGY_LIBRARY"
    )
    assert result["by_system"]["AUTOCOMBINE"]["root_causes"]


def test_gate_funnel_operator_action_prioritizes_sample_and_trial_repairs() -> None:
    class FakeAuto:
        def tasks(self) -> list[dict[str, str]]:
            return [{"task_id": "auto-1"}]

        def experiments(self, task_id: str, *, limit: int) -> list[dict[str, object]]:
            assert task_id == "auto-1"
            assert limit > 0
            return [
                {
                    "id": 1,
                    "gate_status": "REJECTED",
                    "qualification": "EVALUATED",
                    "failed_gates": ["insufficient_sample_folds", "deflated_sharpe_probability"],
                    "metrics": {},
                }
            ]

    class FakeQuant:
        def tasks(self) -> list[dict[str, str]]:
            return []

        def candidates(self, task_id: str, *, limit: int) -> list[dict[str, object]]:
            raise AssertionError("no quant tasks")

    result = build_gate_funnel_diagnostics(FakeAuto(), FakeQuant())  # type: ignore[arg-type]

    assert result["passed_candidates"] == 0
    assert [item["action"] for item in result["operator_actions"][:2]] == [
        "REPAIR_WALK_FORWARD_CAPACITY_OR_COVERAGE",
        "REDUCE_EFFECTIVE_TRIAL_COUNT_AND_DEDUPLICATE_SEARCH_SPACE",
    ]
    assert result["diagnosis"][0]["class"] == "NO_GATE_PASSING_COMBINATIONS"


def test_system_job_runner_materializes_gate_feedback_policy(tmp_path: Path) -> None:
    store = _store(tmp_path)

    class FakeAuto:
        def tasks(self) -> list[dict[str, str]]:
            return [{"task_id": "auto-1"}]

        def experiments(self, task_id: str, *, limit: int) -> list[dict[str, object]]:
            assert task_id == "auto-1"
            assert limit > 0
            return [
                {
                    "id": 1,
                    "gate_status": "REJECTED",
                    "qualification": "EVALUATED",
                    "failed_gates": ["deflated_sharpe_probability"],
                    "metrics": {},
                }
            ]

    class FakeQuant:
        def tasks(self) -> list[dict[str, str]]:
            return []

        def candidates(self, task_id: str, *, limit: int) -> list[dict[str, object]]:
            raise AssertionError("no quant tasks")

    runner = SystemJobRunner(
        store,
        autocombine_store=FakeAuto(),  # type: ignore[arg-type]
        quantcombine_store=FakeQuant(),  # type: ignore[arg-type]
        runtime_root=tmp_path / "runtime",
    )
    store.enqueue_system_job(
        job_id="job-gate-feedback",
        queue="system",
        job_type="gate_feedback_policy_sync",
        payload={},
        progress_total=1,
    )

    result = runner.run_next(queue="system")
    snapshot = store.materialized_snapshot("gate_feedback_policy")

    assert "gate_feedback_policy_sync" in SUPPORTED_SYSTEM_JOB_TYPES
    assert result["claimed"] is True
    assert result["job"]["status"] == "COMPLETED"
    assert result["job"]["result"]["protocol"] == "GATE_FUNNEL_FEEDBACK_POLICY_V1"
    assert "REDUCE_EFFECTIVE_TRIAL_COUNT_AND_DEDUPLICATE_SEARCH_SPACE" in result["job"][
        "result"
    ]["action_ids"]
    assert result["job"]["result"]["root_cause_intensity"][
        "TRIAL_BUDGET_AND_OVERFITTING_PENALTY"
    ] == 1
    assert result["job"]["result"]["recommendations"][0]["action"] == (
        "REDUCE_EFFECTIVE_TRIAL_COUNT_AND_DEDUPLICATE_SEARCH_SPACE"
    )
    assert snapshot is not None


def test_system_job_runner_seeds_quantcombine_repair_task_from_gate_feedback(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)

    class FakeAuto:
        def tasks(self) -> list[dict[str, str]]:
            return [{"task_id": "auto-1"}]

        def experiments(self, task_id: str, *, limit: int) -> list[dict[str, object]]:
            assert task_id == "auto-1"
            assert limit > 0
            return [
                {
                    "id": 1,
                    "gate_status": "REJECTED",
                    "qualification": "EVALUATED",
                    "failed_gates": [
                        "deflated_sharpe_probability",
                        "correlation_too_high",
                        "max_drawdown_too_large",
                    ],
                    "metrics": {},
                }
            ]

    quant_store = QuantCombineStore(store)
    runner = SystemJobRunner(
        store,
        autocombine_store=FakeAuto(),  # type: ignore[arg-type]
        quantcombine_store=quant_store,
        runtime_root=tmp_path / "runtime",
    )
    store.enqueue_system_job(
        job_id="job-quant-repair",
        queue="system",
        job_type="quantcombine_repair_task_seed",
        payload={
            "data_path": str(tmp_path),
            "protocol": {
                "exploration_start": "2010-01-01",
                "exploration_end": "2017-12-31",
                "validation_start": "2018-01-01",
                "validation_end": "2024-12-31",
                "holdout_start": "2025-01-01",
                "holdout_end": "2026-07-16",
                "minimum_folds": 1,
            },
        },
        progress_total=1,
    )

    result = runner.run_next(queue="system")
    task = quant_store.task(result["job"]["result"]["task_id"])

    assert "quantcombine_repair_task_seed" in SUPPORTED_SYSTEM_JOB_TYPES
    assert result["claimed"] is True
    assert result["job"]["status"] == "COMPLETED"
    assert result["job"]["result"]["protocol"] == "AUTOALPHA_QUANTCOMBINE_REPAIR_TASK_SEED_V1"
    assert task is not None
    assert result["job"]["result"]["task_url"].endswith(
        f"/tasks/{result['job']['result']['task_id']}"
    )
    assert result["job"]["result"]["auto_start_requested"] is True
    assert task["objective"]["profile"] == "DIVERSIFICATION_FIRST"
    assert task["construction"]["maximum_same_family"] == 1
    assert task["objective"]["maximum_drawdown"] == 0.18
    assert task["objective"]["maximum_factor_correlation"] == 0.6
    assert task["budget"]["maximum_evaluations"] == 120
    assert "gate-feedback:GATE_FUNNEL_FEEDBACK_POLICY_V1" in task["notes"]
    assert "[quantcombine-autostart:AUTOALPHA_REPAIR_V1]" in task["notes"]


def test_quantcombine_repair_protocol_guard_excludes_fragile_edge_fold() -> None:
    protocol = {
        "exploration_start": "2010-01-04",
        "exploration_end": "2018-04-16",
        "validation_start": "2018-04-17",
        "validation_end": "2023-04-05",
        "holdout_start": "2023-04-06",
        "holdout_end": "2026-07-28",
        "minimum_folds": 6,
    }
    capacity = {
        "minimum_observations_per_fold": 60,
        "maximum_folds": 6,
        "evaluable_years": [2018, 2019, 2020, 2021, 2022, 2023],
        "observations_by_year": {
            2018: 175,
            2019: 244,
            2020: 243,
            2021: 243,
            2022: 242,
            2023: 61,
        },
    }

    guarded, guard = _clamp_repair_protocol_to_capacity(protocol, capacity)

    assert guarded["minimum_folds"] == 5
    assert guard["status"] == "ADJUSTED"
    assert guard["requested_minimum_folds"] == 6
    assert guard["applied_minimum_folds"] == 5
    assert guard["safe_evaluable_years"] == [2018, 2019, 2020, 2021, 2022]
