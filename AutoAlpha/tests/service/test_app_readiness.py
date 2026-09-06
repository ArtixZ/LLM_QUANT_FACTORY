from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from autoalpha.service import app as service_app
from autoalpha.service.app import (
    _factor_research_map,
    _materialized_snapshot_policy,
    _production_data_readiness,
)
from autoalpha.service.store import ServiceStore
from autoalpha.service.strategy_bus import stable_experiment_id


def test_production_readiness_blocks_non_pit_research_proxy() -> None:
    readiness = _production_data_readiness(
        institutional_pit_ready=False,
        execution_basis={
            "capital_ledger_ready": False,
            "capital_ledger_proxy_ready": True,
            "blockers": [
                "cash execution requires point-in-time market state: ['is_st']",
            ],
            "proxy_blockers": [],
        },
    )

    assert readiness["production_trading_allowed"] is False
    assert readiness["strict_pit_capital_ledger_ready"] is False
    assert readiness["non_pit_proxy_allowed"] is True
    assert "institutional_pit_workspace_not_ready" in readiness["blockers"]
    assert "non_pit_proxy_is_research_and_paper_trading_only" in readiness["policy"]


def test_production_readiness_allows_only_strict_pit_capital_ledger() -> None:
    readiness = _production_data_readiness(
        institutional_pit_ready=True,
        execution_basis={
            "capital_ledger_ready": True,
            "capital_ledger_proxy_ready": True,
            "blockers": [],
            "proxy_blockers": [],
        },
    )

    assert readiness["production_trading_allowed"] is True
    assert readiness["strict_pit_capital_ledger_ready"] is True
    assert readiness["blockers"] == []


def test_ready_exposes_module_level_data_capability_matrix(monkeypatch) -> None:
    class FakeWorkspace:
        price_research_ready = True
        institutional_pit_ready = False
        first_trade_date = "2020-01-02"
        last_trade_date = "2026-07-28"
        panel_path = "/tmp/panel"
        fingerprint = "test-data-fingerprint"

        def to_dict(self) -> dict[str, object]:
            return {
                "price_research_ready": self.price_research_ready,
                "institutional_pit_ready": self.institutional_pit_ready,
                "first_trade_date": self.first_trade_date,
                "last_trade_date": self.last_trade_date,
                "warnings": [],
                "blockers": [],
            }

    class FakeExecutionBasis:
        def to_dict(self) -> dict[str, object]:
            return {
                "capital_ledger_proxy_ready": True,
                "capital_ledger_ready": False,
                "proxy_blockers": [],
                "blockers": ["cash execution requires point-in-time market state"],
            }

    monkeypatch.setattr(service_app, "inspect_data_workspace", lambda _: FakeWorkspace())
    monkeypatch.setattr(service_app, "inspect_execution_data_basis", lambda _: FakeExecutionBasis())

    with TestClient(service_app.app) as client:
        ready = client.get("/ready").json()

    matrix = ready["data_capability_matrix"]
    by_module = {row["module_id"]: row for row in matrix["rows"]}
    assert matrix["protocol"] == "AUTOALPHA_DATA_CAPABILITY_MATRIX_V1"
    assert matrix["summary"]["research_ready"] is True
    assert matrix["summary"]["non_pit_proxy_ready"] is True
    assert matrix["summary"]["production_allowed"] is False
    assert by_module["auto_research"]["allowed"] is True
    assert by_module["manual_backtest_proxy"]["level"] == "PROXY_BACKTEST_READY"
    assert by_module["paper_trading"]["level"] == "PROXY_PAPER_READY"
    assert by_module["strict_capital_ledger"]["level"] == "PRODUCTION_BLOCKED"
    assert by_module["strict_capital_ledger"]["required_fields"]


def test_factor_research_map_folds_mechanisms_and_near_duplicates() -> None:
    factors = [
        {
            "factor_id": "F_A",
            "name": "Leader",
            "canonical_mechanism": "LIQUIDITY_STABILITY",
            "behavior_cluster_id": "B001",
            "behavior_redundancy": "LEADER",
            "ranking_values": {"long_only_overall": 1.0},
            "metric_summary": {
                "long_only_walk_forward_folds": [
                    {"validation_start": "2020-01-02", "annual_return": 0.10},
                    {"validation_start": "2021-01-04", "annual_return": -0.03},
                ]
            },
        },
        {
            "factor_id": "F_B",
            "name": "Duplicate",
            "canonical_mechanism": "LIQUIDITY_STABILITY",
            "behavior_cluster_id": "B001",
            "behavior_redundancy": "NEAR_DUPLICATE",
            "behavior_nearest_factor_id": "F_A",
            "behavior_nearest_similarity": 0.95,
            "ranking_values": {"long_only_overall": 0.8},
            "metric_summary": {
                "long_only_walk_forward_folds": [
                    {"validation_start": "2020-01-02", "annual_return": 0.20},
                    {"validation_start": "2021-01-04", "annual_return": 0.01},
                ]
            },
        },
    ]

    research_map = _factor_research_map(factors)

    assert research_map["protocol"] == "AUTOALPHA_FACTOR_RESEARCH_MAP_V2"
    assert research_map["research_map_protocol"] == "AUTOALPHA_FACTOR_RESEARCH_MAP_V2"
    assert research_map["mechanism_cluster_count"] == 1
    assert research_map["near_duplicate_count"] == 1
    assert research_map["parameter_family_count"] == 1
    assert research_map["parameter_families"][0]["parameter_family"] == "NO_EXPLICIT_LOOKBACK"
    assert research_map["mechanism_clusters"][0]["leader_factor_id"] == "F_A"
    assert research_map["mechanism_map"][0]["leader_factor_id"] == "F_A"
    assert research_map["crowded_clusters"][0]["cluster_id"] == "B001"
    assert research_map["homogeneity_fold_groups"][0]["cluster_id"] == "B001"
    assert research_map["annual_heatmap"]["years"] == [2020, 2021]
    assert research_map["annual_heatmap"]["rows"][0]["annual_returns"][2020] == pytest.approx(
        0.15
    )


def test_health_and_ready_survive_unavailable_credential_backend(monkeypatch) -> None:
    class BrokenCredentialStore:
        def get(self) -> str | None:
            raise RuntimeError("Keychain backend unavailable")

        def set(self, value: str) -> None:
            raise RuntimeError("Keychain backend unavailable")

    monkeypatch.delenv("AUTOALPHA_API_KEY", raising=False)
    monkeypatch.setattr(service_app.vault, "credential_store", BrokenCredentialStore())

    with TestClient(service_app.app) as client:
        health = client.get("/health").json()
        ready = client.get("/ready").json()

    assert health["status"] == "ok"
    assert ready["status"] in {"ready", "degraded"}
    assert "production_data" in ready
    assert health["system_job_scheduler"]["status"] in {
        "disabled",
        "not_started",
        "running",
    }
    assert "system_job_scheduler" in ready
    assert ready["metric_convention"]["protocol"] == "LONG_ONLY_METRIC_CONVENTION_CHECK_V1"
    assert health["runtime"]["journal_mode"].lower() == "wal"
    assert "busy_timeout_ms" in health["runtime"]
    assert "materialized_snapshots" in health["runtime"]
    assert "system_jobs" in ready["runtime"]


def test_platform_doctor_exposes_routes_snapshots_jobs_and_processes() -> None:
    with TestClient(service_app.app) as client:
        client.cookies.set("autoalpha_session", "local")
        response = client.get("/api/platform/doctor")

    assert response.status_code == 200
    doctor = response.json()
    assert doctor["protocol"] == "AUTOALPHA_PLATFORM_DOCTOR_V1"
    assert doctor["status"] in {"OK", "ATTENTION"}
    assert "/api/platform/doctor" in doctor["expected_routes"]
    assert "/api/strategy-bus/sync" in doctor["expected_routes"]
    assert "missing_routes" in doctor
    assert "snapshot_policy" in doctor
    assert "job_summary" in doctor
    assert "system_job_scheduler" in doctor
    assert {item["name"] for item in doctor["processes"]} == {
        "autoalpha",
        "autocombine",
        "quantcombine",
    }


def test_ready_reports_postgres_migration_blockers(monkeypatch) -> None:
    monkeypatch.setenv("AUTOALPHA_DATABASE_BACKEND", "postgresql")
    monkeypatch.setenv(
        "AUTOALPHA_DATABASE_URL",
        "postgresql://user:secret@localhost:5432/autoalpha",
    )

    with TestClient(service_app.app) as client:
        ready = client.get("/ready").json()

    assert ready["runtime"]["backend"]["backend"] == "postgresql"
    assert ready["runtime"]["backend"]["url"] == "postgresql://***@localhost:5432/autoalpha"
    assert ready["runtime"]["backend"]["adapter_capabilities"]["system_jobs"] == (
        "postgresql_available"
    )
    assert ready["status"] == "degraded"
    assert any(
        "Full PostgreSQL ServiceStore adapter migration is not enabled yet" in blocker
        for blocker in ready["blockers"]
    )


def test_ready_degrades_when_system_job_scheduler_failed(monkeypatch) -> None:
    monkeypatch.setattr(
        service_app,
        "_system_job_scheduler_status",
        lambda: {
            "enabled": True,
            "alive": False,
            "status": "failed",
            "queue": "system",
            "poll_seconds": {"claimed": 2, "idle": 15, "after_error": 30},
            "supported_job_types": [],
            "failure": "RuntimeError: test failure",
        },
    )

    with TestClient(service_app.app) as client:
        ready = client.get("/ready").json()

    assert ready["status"] == "degraded"
    assert "system_job_scheduler_failed" in ready["blockers"]
    assert ready["system_job_scheduler"]["failure"] == "RuntimeError: test failure"


def test_system_job_command_endpoint_controls_queued_jobs(
    tmp_path, monkeypatch
) -> None:
    test_store = ServiceStore(tmp_path / "autoalpha.sqlite3")
    monkeypatch.setattr(service_app, "store", test_store)
    test_store.enqueue_system_job(
        job_id="job-api-control",
        queue="system",
        job_type="factor_knowledge_map_sync",
        payload={},
    )

    with TestClient(service_app.app) as client:
        paused = client.post(
            "/api/jobs/job-api-control/pause",
            json={"actor": "tester", "reason": "api pause"},
        )
        resumed = client.post(
            "/api/jobs/job-api-control/resume",
            json={"actor": "tester", "reason": "api resume"},
        )
        logs = client.get("/api/jobs/job-api-control/logs")

    assert paused.status_code == 200
    assert paused.json()["status"] == "PAUSED"
    assert resumed.status_code == 200
    assert resumed.json()["status"] == "QUEUED"
    assert logs.status_code == 200
    assert logs.json()["logs"][0]["event"] == "COMMAND_RESUME"
    assert logs.json()["job"]["job_id"] == "job-api-control"


def test_data_sync_start_queues_market_data_job_by_default(
    tmp_path, monkeypatch
) -> None:
    test_store = ServiceStore(tmp_path / "autoalpha.sqlite3")
    test_store.save_settings({"data_path": str(tmp_path), "market_data_root": str(tmp_path)})
    monkeypatch.setattr(service_app, "store", test_store)
    monkeypatch.setattr(
        type(service_app.data_sync_worker),
        "alive",
        property(lambda self: False),
    )
    monkeypatch.setattr(service_app.data_sync_worker, "token_configured", lambda: True)
    monkeypatch.setattr(
        service_app.data_sync_worker,
        "status",
        lambda: {"state": "IDLE", "running": False},
    )

    with TestClient(service_app.app) as client:
        first = client.post(
            "/api/data-sync/start",
            json={"dataset_ids": ["core_market"], "start_date": "2020-01-01"},
        )
        second = client.post(
            "/api/data-sync/start",
            json={"dataset_ids": ["core_market"], "start_date": "2020-01-01"},
        )

    assert first.status_code == 200
    assert first.json()["mode"] == "queued"
    assert first.json()["job"]["job_type"] == "market_data_sync"
    assert first.json()["job"]["resource_group"] == "market-data-sync"
    assert second.status_code == 200
    assert second.json()["deduplicated"] is True
    assert len(test_store.system_jobs(status="QUEUED")) == 1


def test_ready_degrades_when_system_jobs_have_expired_leases(monkeypatch) -> None:
    monkeypatch.setattr(
        service_app,
        "_runtime_database_health",
        lambda: {
            "system_jobs": {
                "total": 1,
                "expired_running_count": 1,
                "expired_running": [
                    {"queue": "system", "resource_group": "sqlite-writer", "count": 1}
                ],
            },
            "materialized_snapshots": {
                "total": 0,
                "states": {},
                "stale_count": 0,
                "no_ttl_count": 0,
                "stale_keys": [],
                "no_ttl_keys": [],
            },
        },
    )

    with TestClient(service_app.app) as client:
        ready = client.get("/ready").json()

    assert ready["status"] == "degraded"
    assert "system_jobs_have_expired_leases" in ready["blockers"]
    assert ready["runtime"]["system_jobs"]["expired_running_count"] == 1


def test_jobs_api_exposes_scheduler_status(monkeypatch) -> None:
    monkeypatch.setattr(
        service_app,
        "_system_job_scheduler_status",
        lambda: {
            "enabled": True,
            "alive": True,
            "status": "running",
            "queue": "system",
            "poll_seconds": {"claimed": 2, "idle": 15, "after_error": 30},
            "supported_job_types": ["factor_homogeneity_backfill"],
            "failure": None,
        },
    )

    with TestClient(service_app.app) as client:
        client.cookies.set("autoalpha_session", "local")
        response = client.get("/api/jobs")

    assert response.status_code == 200
    payload = response.json()
    assert payload["scheduler"]["status"] == "running"
    assert payload["scheduler"]["supported_job_types"] == ["factor_homogeneity_backfill"]
    assert "recent_logs" in payload
    assert payload["database"]["journal_mode"].lower() == "wal"
    assert payload["resource_policy"]["job_log_policy"] == "structured_system_job_logs_by_job_id"
    assert payload["resource_policy"]["snapshot_policy"].startswith("read materialized")
    assert payload["snapshot_policy"]["protocol"] == (
        "AUTOALPHA_MATERIALIZED_SNAPSHOT_POLICY_V1"
    )
    assert payload["resource_policy"]["claim_quota"] == (
        "global_capacity_queue_capacity_and_resource_group_capacity"
    )


def test_materialized_snapshot_policy_marks_missing_and_stale_views() -> None:
    policy = _materialized_snapshot_policy(
        {
            "snapshots": [
                {
                    "key": "factor_library",
                    "source": "api.refresh",
                    "updated_at": "2026-07-29T01:00:00Z",
                    "expires_at": "2026-07-29T01:15:00Z",
                    "cache_state": {"status": "FRESH"},
                },
                {
                    "key": "strategy_bus",
                    "source": "job:test",
                    "updated_at": "2026-07-29T00:00:00Z",
                    "expires_at": "2026-07-29T00:05:00Z",
                    "cache_state": {"status": "STALE"},
                },
            ],
        }
    )
    by_key = {row["key"]: row for row in policy["rows"]}

    assert policy["protocol"] == "AUTOALPHA_MATERIALIZED_SNAPSHOT_POLICY_V1"
    assert policy["status"] == "ATTENTION"
    assert "strategy_bus" in policy["stale_keys"]
    assert "factor_knowledge_map" in policy["missing_keys"]
    assert by_key["factor_library"]["refresh_endpoint"] == "POST /api/factors/refresh"
    assert by_key["strategy_bus"]["refresh_job_type"] == "strategy_bus_sync"


def test_jobs_page_is_served() -> None:
    with TestClient(service_app.app) as client:
        response = client.get("/jobs")

    assert response.status_code == 200
    assert "AutoAlpha 作业中心" in response.text
    assert "资源配额矩阵" in response.text


def test_factor_library_page_exposes_parameter_family_research_map() -> None:
    with TestClient(service_app.app) as client:
        response = client.get("/factors")

    assert response.status_code == 200
    assert "参数家族" in response.text
    assert "parameterFamilyList" in response.text


def test_strategy_experiment_lineage_api_returns_404_for_missing_experiment() -> None:
    with TestClient(service_app.app) as client:
        client.cookies.set("autoalpha_session", "local")
        response = client.get("/api/strategy-experiments/EXP_DOES_NOT_EXIST/lineage")

    assert response.status_code == 404


def test_gate_feedback_api_exposes_policy() -> None:
    with TestClient(service_app.app) as client:
        client.cookies.set("autoalpha_session", "local")
        response = client.get("/api/gate-feedback?refresh=true")

    assert response.status_code == 200
    payload = response.json()
    assert payload["protocol"] == "GATE_FUNNEL_FEEDBACK_POLICY_V1"
    assert "adjustments" in payload
    assert "action_ids" in payload


def test_gate_feedback_can_enqueue_quant_repair_job_idempotently(monkeypatch) -> None:
    calls = {"count": 0}

    def fake_gate_funnel(*_args, **_kwargs):
        calls["count"] += 1
        return {
            "protocol": "AUTOALPHA_GATE_FUNNEL_DIAGNOSTICS_V2_TEST",
            "created_at": f"2026-07-29T00:00:0{calls['count']}Z",
            "total_candidates": 1,
            "passed_candidates": 0,
            "rejected_candidates": 1,
            "operator_actions": [
                {
                    "priority": "P0",
                    "action": "REDUCE_EFFECTIVE_TRIAL_COUNT_AND_DEDUPLICATE_SEARCH_SPACE",
                    "reason": "test",
                    "evidence_count": 1,
                }
            ],
            "root_causes": [
                {"key": "TRIAL_BUDGET_AND_OVERFITTING_PENALTY", "count": 1}
            ],
        }

    monkeypatch.setattr(service_app, "build_gate_funnel_diagnostics", fake_gate_funnel)

    with TestClient(service_app.app) as client:
        client.cookies.set("autoalpha_session", "local")
        first = client.post("/api/gate-feedback/seed-quant-repair", json={})
        second = client.post("/api/gate-feedback/seed-quant-repair", json={})

    assert first.status_code == 200
    first_payload = first.json()
    assert first_payload["status"] in {"ENQUEUED", "EXISTING"}
    assert first_payload["job"]["job_type"] == "quantcombine_repair_task_seed"
    assert first_payload["job"]["resource_group"] == "sqlite-writer"
    assert "feedback_source_fingerprint" in first_payload["job"]["payload"]
    assert second.status_code == 200
    assert second.json()["status"] == "EXISTING"
    assert second.json()["job"]["job_id"] == first_payload["job"]["job_id"]
    service_app.store.update_system_job(
        first_payload["job"]["job_id"],
        status="COMPLETED",
        result={
            "protocol": "AUTOALPHA_QUANTCOMBINE_REPAIR_TASK_SEED_V1",
            "task_id": "qcombine-test",
            "task_url": "http://127.0.0.1:8889/tasks/qcombine-test",
        },
    )
    with TestClient(service_app.app) as client:
        client.cookies.set("autoalpha_session", "local")
        third = client.post("/api/gate-feedback/seed-quant-repair", json={})
    assert third.status_code == 200
    assert third.json()["status"] == "EXISTING"
    assert third.json()["repair_task_id"] == "qcombine-test"
    if first_payload["status"] == "ENQUEUED":
        service_app.store.update_system_job(
            first_payload["job"]["job_id"],
            status="FAILED",
            error="test cleanup: fake gate feedback repair seed",
        )


def test_strategy_library_seed_endpoint_enqueues_idempotent_job(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        service_app,
        "strategy_promotion_candidates",
        lambda store, *, limit: [
            {
                "experiment_id": "EXP_A",
                "candidate_class": "RESEARCH_LEADER",
                "title": "A",
            },
            {"experiment_id": "EXP_B", "candidate_class": "OBSERVATION", "title": "B"},
        ],
    )
    monkeypatch.setattr(
        service_app.store,
        "system_job",
        lambda job_id: (_ for _ in ()).throw(KeyError(job_id)),
    )

    def fake_enqueue(**kwargs):
        captured.update(kwargs)
        return {
            "job_id": kwargs["job_id"],
            "job_type": kwargs["job_type"],
            "payload": kwargs["payload"],
            "status": "QUEUED",
        }

    monkeypatch.setattr(service_app.store, "enqueue_system_job", fake_enqueue)
    monkeypatch.setattr(service_app.store, "append_event", lambda *args, **kwargs: None)

    result = asyncio.run(
        service_app.seed_strategy_library_candidates(
            service_app.StrategyLibrarySeedJobRequest(limit=10)
        )
    )

    assert result["status"] == "ENQUEUED"
    assert result["candidate_count"] == 1
    assert captured["job_type"] == "strategy_library_seed"
    assert captured["payload"]["candidate_ids"] == ["EXP_A"]
    assert captured["resource_group"] == "sqlite-writer"


def test_strategy_freeze_ready_endpoint_skips_when_nothing_ready(monkeypatch) -> None:
    monkeypatch.setattr(
        service_app.store,
        "formal_strategy_versions",
        lambda limit: [
            {"strategy_uid": "STR_A", "version": 1, "lifecycle": "RESEARCH"}
        ],
    )
    monkeypatch.setattr(
        service_app,
        "strategy_lifecycle_readiness",
        lambda store, strategy_uid, version: {
            "next_lifecycle": "FROZEN",
            "ready": False,
            "missing_evidence": ["public_validation_passed"],
        },
    )

    result = asyncio.run(
        service_app.freeze_public_validation_ready_strategies(
            service_app.StrategyFreezeReadyJobRequest(limit=10)
        )
    )

    assert result == {
        "status": "SKIPPED",
        "reason": "NO_PUBLIC_VALIDATION_READY_STRATEGIES",
        "ready_count": 0,
    }


def test_strategy_library_can_create_paper_portfolio_from_execution_seed(
    tmp_path, monkeypatch
) -> None:
    test_store = ServiceStore(tmp_path / "service.sqlite3")
    test_store.create_research_task(
        task_id="task-seed",
        name="Seed Task",
        market="US",
        data_path=str(tmp_path / "panel"),
        data_start="2020-01-01",
        data_end="2026-07-29",
        snapshot_hash="hash",
        status="READY",
        protocol={},
    )
    for factor_id in ("F_A", "F_B"):
        test_store.upsert_factor_pool(
            factor_id=factor_id,
            source_iteration=1,
            source_task_id="task-seed",
            proposal={
                "name": factor_id,
                "family": "Liquidity",
                "hypothesis": "test",
                "expected_direction": 1,
                "expression": {
                    "operator": "field",
                    "parameters": {"name": "close"},
                    "arguments": [],
                },
            },
            metrics={"long_only_sharpe_ratio": 1.0},
            status="ELIGIBLE",
            status_reason="test",
        )
    experiment_id = stable_experiment_id("TEST", "paper-seed", "COMBINATION_CANDIDATE")
    test_store.upsert_strategy_experiment_object(
        experiment_id=experiment_id,
        stage="COMBINATION_CANDIDATE",
        object_type="factor_combination",
        source_system="TEST",
        source_id="paper-seed",
        title="Seed Strategy",
        status="QUALIFIED_CHAMPION",
        market="US",
        metrics={"portfolio_capacity_usd": 1_000_000},
        evidence={
            "factor_ids": ["F_A", "F_B"],
            "weights": [0.7, 0.3],
            "maximum_positions": 12,
            "target_gross_exposure": 0.88,
            "gate_status": "PASSED",
        },
    )
    strategy = service_app.create_formal_strategy_from_experiment(test_store, experiment_id)
    captured: dict[str, object] = {}

    def fake_create(self, spec):  # noqa: ANN001, ANN202
        captured["data_path"] = str(self.data_path)
        captured["spec"] = spec
        return {
            "id": 42,
            "name": spec.name,
            "config": {
                "factor_ids": spec.factor_ids,
                "weights": spec.weights,
                "execution_protocol": {
                    "protocol": "US_EQUITY_PAPER_NEXT_OPEN_PROXY_EXECUTION_V2"
                },
            },
            "positions": [],
            "trades": [],
        }

    monkeypatch.setattr(service_app, "store", test_store)
    monkeypatch.setattr(
        type(service_app.data_sync_worker),
        "alive",
        property(lambda self: False),
    )
    monkeypatch.setattr(service_app.PaperTradingEngine, "create", fake_create)
    monkeypatch.setattr(service_app, "_paper_event", lambda *_, **__: None)

    with TestClient(service_app.app) as client:
        client.cookies.set("autoalpha_session", "local")
        response = client.post(
            f"/api/strategy-library/{strategy['strategy_uid']}/versions/{strategy['version']}/paper-portfolio",
            json={
                "initial_cash_usd": 2_000_000,
                "as_of_date": "2026-07-29",
                "name": "Seeded Paper",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    spec = captured["spec"]
    assert payload["source_strategy"]["strategy_uid"] == strategy["strategy_uid"]
    assert payload["source_strategy"]["execution_protocol"] == (
        "US_EQUITY_PAPER_NEXT_OPEN_PROXY_EXECUTION_V2"
    )
    assert spec.name == "Seeded Paper"
    assert spec.factor_ids == ["F_A", "F_B"]
    assert spec.weights == [0.7, 0.3]
    assert spec.initial_cash_usd == 2_000_000
    assert spec.selection_count == 12
    assert spec.gross_exposure == 0.88
    assert spec.as_of_date.isoformat() == "2026-07-29"
    assert captured["data_path"] == str(tmp_path / "panel")


def test_strategy_bus_get_is_read_only_even_when_sync_query_is_set(monkeypatch) -> None:
    calls: list[bool] = []

    monkeypatch.setattr(service_app.store, "materialized_snapshot", lambda key: None)

    def fake_snapshot(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        calls.append(bool(kwargs["sync"]))
        return {"summary": {}, "objects": [], "edges": [], "protocol": {}}

    monkeypatch.setattr(service_app, "build_strategy_bus_snapshot", fake_snapshot)

    with TestClient(service_app.app) as client:
        client.cookies.set("autoalpha_session", "local")
        response = client.get("/api/strategy-bus?sync=true")

    assert response.status_code == 200
    payload = response.json()
    assert calls == [False]
    assert payload["read_only"] is True
    assert payload["materialized"] is False
    assert payload["sync_ignored"] is True
    assert "POST /api/strategy-bus/sync" in payload["sync_hint"]


def test_strategy_bus_get_uses_materialized_snapshot_without_rebuild(monkeypatch) -> None:
    monkeypatch.setattr(
        service_app.store,
        "materialized_snapshot",
        lambda key: {
            "payload": {"summary": {"total": 1}, "objects": [], "edges": [], "protocol": {}},
            "updated_at": "2026-07-29T00:00:00+00:00",
            "fingerprint": "fingerprint-test",
            "source": "job:test",
            "status": "READY",
            "expires_at": "2026-07-29T00:05:00+00:00",
            "cache_state": {
                "status": "FRESH",
                "stale": False,
                "age_seconds": 12,
                "expires_at": "2026-07-29T00:05:00+00:00",
                "source": "job:test",
                "snapshot_status": "READY",
            },
        },
    )

    def fail_rebuild(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError("GET /api/strategy-bus must not rebuild when cache exists")

    monkeypatch.setattr(service_app, "build_strategy_bus_snapshot", fail_rebuild)

    with TestClient(service_app.app) as client:
        client.cookies.set("autoalpha_session", "local")
        response = client.get("/api/strategy-bus")

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"] == {"total": 1}
    assert payload["materialized"] is True
    assert payload["read_only"] is True
    assert payload["materialized_fingerprint"] == "fingerprint-test"
    assert payload["materialized_source"] == "job:test"
    assert payload["cache_state"]["status"] == "FRESH"


def test_strategy_bus_sync_post_queues_job_by_default(monkeypatch) -> None:
    enqueued: list[dict[str, object]] = []

    monkeypatch.setattr(service_app.store, "system_jobs", lambda **_: [])

    def enqueue_job(**kwargs):  # noqa: ANN003, ANN202
        enqueued.append(kwargs)
        return {
            "job_id": kwargs["job_id"],
            "queue": kwargs["queue"],
            "job_type": kwargs["job_type"],
            "status": "QUEUED",
            "priority": kwargs["priority"],
        }

    def fail_sync(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError("POST /api/strategy-bus/sync should enqueue by default")

    monkeypatch.setattr(service_app.store, "enqueue_system_job", enqueue_job)
    monkeypatch.setattr(service_app.store, "append_event", lambda *_, **__: None)
    monkeypatch.setattr(service_app, "build_strategy_bus_snapshot", fail_sync)

    with TestClient(service_app.app) as client:
        client.cookies.set("autoalpha_session", "local")
        response = client.post("/api/strategy-bus/sync", json={})

    assert response.status_code == 200
    payload = response.json()
    assert payload["queued"] is True
    assert payload["deduplicated"] is False
    assert payload["job"]["job_type"] == "strategy_bus_sync"
    assert enqueued[0]["resource_group"] == "strategy_bus"


def test_strategy_bus_sync_post_deduplicates_existing_job(monkeypatch) -> None:
    existing = {
        "job_id": "job-existing",
        "queue": "system",
        "job_type": "strategy_bus_sync",
        "status": "RUNNING",
    }

    monkeypatch.setattr(
        service_app.store,
        "system_jobs",
        lambda **kwargs: [existing] if kwargs.get("status") == "RUNNING" else [],
    )
    monkeypatch.setattr(
        service_app.store,
        "enqueue_system_job",
        lambda **_: (_ for _ in ()).throw(AssertionError("must not enqueue duplicate")),
    )

    with TestClient(service_app.app) as client:
        client.cookies.set("autoalpha_session", "local")
        response = client.post("/api/strategy-bus/sync", json={})

    assert response.status_code == 200
    payload = response.json()
    assert payload["queued"] is True
    assert payload["deduplicated"] is True
    assert payload["job"]["job_id"] == "job-existing"


def test_factor_knowledge_map_get_refresh_is_read_only(monkeypatch) -> None:
    monkeypatch.setattr(service_app.store, "materialized_snapshot", lambda key: None)
    monkeypatch.setattr(
        service_app.store,
        "upsert_materialized_snapshot",
        lambda *_, **__: (_ for _ in ()).throw(AssertionError("GET must not write")),
    )
    monkeypatch.setattr(
        service_app,
        "factor_knowledge_map",
        lambda *_, **__: (_ for _ in ()).throw(AssertionError("GET refresh must not rebuild")),
    )

    with TestClient(service_app.app) as client:
        client.cookies.set("autoalpha_session", "local")
        response = client.get("/api/factor-knowledge-map?refresh=true")

    assert response.status_code == 200
    payload = response.json()
    assert payload["read_only"] is True
    assert payload["materialized"] is False
    assert payload["refresh_ignored"] is True
    assert "POST /api/factor-knowledge-map/sync" in payload["refresh_hint"]


def test_factor_knowledge_map_sync_post_queues_job_by_default(monkeypatch) -> None:
    enqueued: list[dict[str, object]] = []

    monkeypatch.setattr(service_app.store, "system_jobs", lambda **_: [])

    def enqueue_job(**kwargs):  # noqa: ANN003, ANN202
        enqueued.append(kwargs)
        return {
            "job_id": kwargs["job_id"],
            "queue": kwargs["queue"],
            "job_type": kwargs["job_type"],
            "status": "QUEUED",
            "priority": kwargs["priority"],
        }

    monkeypatch.setattr(service_app.store, "enqueue_system_job", enqueue_job)
    monkeypatch.setattr(service_app.store, "append_event", lambda *_, **__: None)
    monkeypatch.setattr(
        service_app,
        "factor_knowledge_map",
        lambda *_, **__: (_ for _ in ()).throw(AssertionError("must enqueue by default")),
    )

    with TestClient(service_app.app) as client:
        client.cookies.set("autoalpha_session", "local")
        response = client.post("/api/factor-knowledge-map/sync", json={})

    assert response.status_code == 200
    payload = response.json()
    assert payload["queued"] is True
    assert payload["deduplicated"] is False
    assert payload["job"]["job_type"] == "factor_knowledge_map_sync"
    assert enqueued[0]["resource_group"] == "sqlite-writer"


def test_factor_knowledge_map_sync_post_deduplicates_existing_job(monkeypatch) -> None:
    existing = {
        "job_id": "job-knowledge-existing",
        "queue": "system",
        "job_type": "factor_knowledge_map_sync",
        "status": "QUEUED",
    }

    monkeypatch.setattr(
        service_app.store,
        "system_jobs",
        lambda **kwargs: [existing] if kwargs.get("status") == "QUEUED" else [],
    )
    monkeypatch.setattr(
        service_app.store,
        "enqueue_system_job",
        lambda **_: (_ for _ in ()).throw(AssertionError("must not enqueue duplicate")),
    )

    with TestClient(service_app.app) as client:
        client.cookies.set("autoalpha_session", "local")
        response = client.post("/api/factor-knowledge-map/sync", json={})

    assert response.status_code == 200
    payload = response.json()
    assert payload["queued"] is True
    assert payload["deduplicated"] is True
    assert payload["job"]["job_id"] == "job-knowledge-existing"


def test_factor_library_get_is_read_only_and_does_not_rebuild(monkeypatch) -> None:
    monkeypatch.setattr(service_app.store, "materialized_snapshot", lambda key: None)

    def fail_rebuild():  # noqa: ANN202
        raise AssertionError("GET /api/factors must not rebuild factor library")

    monkeypatch.setattr(service_app, "_build_factor_library_payload", fail_rebuild)

    with TestClient(service_app.app) as client:
        client.cookies.set("autoalpha_session", "local")
        response = client.get("/api/factors?refresh=true")

    assert response.status_code == 200
    payload = response.json()
    assert payload["read_only"] is True
    assert payload["materialized"] is False
    assert payload["cache_status"] == "MISSING"
    assert payload["refresh_ignored"] is True
    assert "POST /api/factors/refresh" in payload["refresh_hint"]


def test_factor_library_refresh_post_queues_job_by_default(monkeypatch) -> None:
    enqueued: list[dict[str, object]] = []

    monkeypatch.setattr(service_app.store, "system_jobs", lambda **_: [])

    def enqueue_job(**kwargs):  # noqa: ANN003, ANN202
        enqueued.append(kwargs)
        return {
            "job_id": kwargs["job_id"],
            "queue": kwargs["queue"],
            "job_type": kwargs["job_type"],
            "status": "QUEUED",
            "priority": kwargs["priority"],
        }

    monkeypatch.setattr(service_app.store, "enqueue_system_job", enqueue_job)
    monkeypatch.setattr(service_app.store, "append_event", lambda *_, **__: None)
    monkeypatch.setattr(
        service_app,
        "_build_factor_library_payload",
        lambda: (_ for _ in ()).throw(AssertionError("must enqueue by default")),
    )

    with TestClient(service_app.app) as client:
        client.cookies.set("autoalpha_session", "local")
        response = client.post("/api/factors/refresh", json={})

    assert response.status_code == 200
    payload = response.json()
    assert payload["queued"] is True
    assert payload["deduplicated"] is False
    assert payload["job"]["job_type"] == "factor_library_refresh"
    assert enqueued[0]["resource_group"] == "sqlite-writer"


def test_factor_library_refresh_post_deduplicates_existing_job(monkeypatch) -> None:
    existing = {
        "job_id": "job-factor-library-existing",
        "queue": "system",
        "job_type": "factor_library_refresh",
        "status": "RUNNING",
    }

    monkeypatch.setattr(
        service_app.store,
        "system_jobs",
        lambda **kwargs: [existing] if kwargs.get("status") == "RUNNING" else [],
    )
    monkeypatch.setattr(
        service_app.store,
        "enqueue_system_job",
        lambda **_: (_ for _ in ()).throw(AssertionError("must not enqueue duplicate")),
    )

    with TestClient(service_app.app) as client:
        client.cookies.set("autoalpha_session", "local")
        response = client.post("/api/factors/refresh", json={})

    assert response.status_code == 200
    payload = response.json()
    assert payload["queued"] is True
    assert payload["deduplicated"] is True
    assert payload["job"]["job_id"] == "job-factor-library-existing"


def test_factor_library_refresh_post_run_now_materializes_payload(monkeypatch) -> None:
    snapshots: dict[str, dict[str, object]] = {}

    monkeypatch.setattr(
        service_app,
        "_build_factor_library_payload",
        lambda: {
            "summary": {"factor_count": 1},
            "factors": [{"factor_id": "F_1"}],
            "research_tasks": [],
            "data": {},
            "knowledge_integrity": {
                "protocol": "AUTOALPHA_FACTOR_KNOWLEDGE_INTEGRITY_V1",
                "complete": True,
            },
        },
    )

    def fake_upsert(key, payload, **kwargs):  # noqa: ANN001, ANN003, ANN202
        snapshots[key] = payload
        return {
            "payload": payload,
            "updated_at": "2026-07-29T00:00:00+00:00",
            "fingerprint": "factor-library-fingerprint",
            "source": kwargs.get("source", "unknown"),
            "status": "READY",
            "expires_at": "2026-07-29T00:15:00+00:00",
            "cache_state": {
                "status": "FRESH",
                "stale": False,
                "age_seconds": 0,
                "expires_at": "2026-07-29T00:15:00+00:00",
                "source": kwargs.get("source", "unknown"),
                "snapshot_status": "READY",
            },
        }

    monkeypatch.setattr(service_app.store, "upsert_materialized_snapshot", fake_upsert)

    with TestClient(service_app.app) as client:
        client.cookies.set("autoalpha_session", "local")
        response = client.post("/api/factors/refresh", json={"run_now": True})

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"] == {"factor_count": 1}
    assert payload["materialized"] is True
    assert payload["read_only"] is False
    assert payload["materialized_fingerprint"] == "factor-library-fingerprint"
    assert snapshots["factor_library"]["api_payload_protocol"] == (
        "MATERIALIZED_FACTOR_LIBRARY_API_V1"
    )


def test_factor_library_cache_miss_retains_workspace_metadata(monkeypatch) -> None:
    class FakeWorkspace:
        root_path = "/tmp/data"
        panel_path = "/tmp/data/processed/daily_panel"
        first_trade_date = "2020-01-02"
        last_trade_date = "2026-09-02"
        fingerprint = "workspace-fingerprint"
        price_research_ready = True
        institutional_pit_ready = False

    class FakeExecutionBasis:
        def to_dict(self) -> dict[str, object]:
            return {
                "capital_ledger_proxy_ready": True,
                "capital_ledger_ready": False,
            }

    monkeypatch.setattr(
        service_app.store,
        "research_tasks",
        lambda: [{"task_id": "primary-us-equity", "name": "US", "market": "US"}],
    )
    monkeypatch.setattr(
        service_app.store,
        "settings",
        lambda: {"data_path": "/tmp/data"},
    )
    monkeypatch.setattr(service_app, "inspect_data_workspace", lambda _: FakeWorkspace())
    monkeypatch.setattr(
        service_app,
        "inspect_execution_data_basis",
        lambda _: FakeExecutionBasis(),
    )

    payload = service_app._factor_library_cache_miss_payload(None)

    assert payload["summary"]["factor_count"] == 0
    assert payload["research_tasks"][0]["task_id"] == "primary-us-equity"
    assert payload["data"]["first_trade_date"] == "2020-01-02"
    assert payload["data"]["last_trade_date"] == "2026-09-02"
    assert payload["data"]["execution_basis"]["capital_ledger_ready"] is False
