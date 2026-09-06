from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from autoalpha.config import ResearchConfig
from autoalpha.service.research_manager import ResearchTaskManager
from autoalpha.service.research_protocol import default_task_protocol, protocol_fingerprint
from autoalpha.service.store import ServiceStore
from autoalpha.service.worker import (
    ContinuousResearchWorker,
    SecretVault,
    _candidate_level_failure_reason,
)


def _task(
    store: ServiceStore,
    task_id: str,
    *,
    market: str = "US",
    start: str = "2010-01-04",
    end: str = "2026-07-16",
) -> None:
    protocol = default_task_protocol(
        start, end, ResearchConfig.from_toml(Path("config/research.toml"))
    )
    store.create_research_task(
        task_id=task_id,
        name=task_id,
        market=market,
        data_path="/data/panel",
        data_start=start,
        data_end=end,
        snapshot_hash="frozen",
        status="READY",
        protocol=protocol,
        protocol_hash=protocol_fingerprint(protocol),
    )


def test_task_workers_keep_runtime_state_and_history_isolated(tmp_path: Path) -> None:
    store = ServiceStore(tmp_path / "service.sqlite3")
    _task(store, "task-a")
    _task(store, "task-b")
    worker_a = ContinuousResearchWorker(
        store,
        SecretVault(api_key="test"),
        config_path=Path("config/research.toml"),
        artifact_root=tmp_path / "artifacts-a",
        task_id="task-a",
    )
    worker_b = ContinuousResearchWorker(
        store,
        SecretVault(api_key="test"),
        config_path=Path("config/research.toml"),
        artifact_root=tmp_path / "artifacts-b",
        task_id="task-b",
    )

    worker_a._update_state(state="RUNNING", run_id="run-a", iteration=3)
    worker_b._update_state(state="STOPPED", run_id="run-b", iteration=8)
    for run_id, value in (("run-a", 1.2), ("run-b", 2.4)):
        store.begin_iteration(run_id, 1)
        store.finish_iteration(run_id, 1, status="COMPLETED", metrics={"sharpe_ratio": value})

    assert worker_a._state()["iteration"] == 3
    assert worker_b._state()["iteration"] == 8
    assert store.state()["iteration"] == 0
    assert store.metric_history(run_id="run-a")[0]["sharpe_ratio"] == 1.2
    assert store.metric_history(run_id="run-b")[0]["sharpe_ratio"] == 2.4


def test_task_scoped_generation_does_not_seal_another_task(tmp_path: Path) -> None:
    store = ServiceStore(tmp_path / "service.sqlite3")
    for generation in ("protocol-v3--task-a", "protocol-v3--task-b"):
        store.ensure_running_generation(
            base_generation_id=generation,
            protocol_version="v3",
            maximum_candidates=10,
            maximum_holdout_attempts=2,
        )

    store.ensure_running_generation(
        base_generation_id="protocol-v4--task-a",
        protocol_version="v4",
        maximum_candidates=10,
        maximum_holdout_attempts=2,
    )

    assert store.generation_state("protocol-v3--task-a")["status"] == "SEALED"  # type: ignore[index]
    assert store.generation_state("protocol-v3--task-b")["status"] == "ACTIVE"  # type: ignore[index]


def test_legacy_generation_does_not_seal_task_scoped_generation(tmp_path: Path) -> None:
    store = ServiceStore(tmp_path / "service.sqlite3")
    store.ensure_running_generation(
        base_generation_id="protocol-v3--task-a",
        protocol_version="v3-task",
        maximum_candidates=10,
        maximum_holdout_attempts=2,
    )

    store.ensure_running_generation(
        base_generation_id="protocol-v4",
        protocol_version="v4",
        maximum_candidates=10,
        maximum_holdout_attempts=2,
    )

    assert store.generation_state("protocol-v3--task-a")["status"] == "ACTIVE"  # type: ignore[index]
    assert store.generation_state("protocol-v4")["status"] == "ACTIVE"  # type: ignore[index]


def test_direction_attempt_iteration_numbers_can_overlap_between_runs(tmp_path: Path) -> None:
    store = ServiceStore(tmp_path / "service.sqlite3")
    campaign_ids = []
    for generation in ("g-a", "g-b"):
        store.ensure_generation(
            generation_id=generation,
            protocol_version="v1",
            maximum_candidates=10,
            maximum_holdout_attempts=1,
        )
        campaign = store.start_direction_campaign(
            generation_id=generation,
            direction="RESTORE_STABILITY",
            title=generation,
            objective="test",
            diagnostic_score=1.0,
            rationale=["test"],
            evidence={},
            baseline={},
            maximum_attempts=3,
            started_iteration=1,
        )
        campaign_ids.append(campaign["id"])

    store.reserve_direction_attempt(
        campaign_id=campaign_ids[0], iteration=1, baseline={}, run_id="run-a"
    )
    store.reserve_direction_attempt(
        campaign_id=campaign_ids[1], iteration=1, baseline={}, run_id="run-b"
    )

    assert store.direction_attempt(1, run_id="run-a")["campaign_id"] == campaign_ids[0]  # type: ignore[index]
    assert store.direction_attempt(1, run_id="run-b")["campaign_id"] == campaign_ids[1]  # type: ignore[index]


def test_manager_reports_protocol_and_market_readiness(tmp_path: Path, monkeypatch) -> None:
    store = ServiceStore(tmp_path / "service.sqlite3")
    _task(store, "task-ready")
    _task(store, "task-short", start="2026-01-01")
    _task(store, "task-cn", market="CN_A")
    workspace = SimpleNamespace(fingerprint="panel-fingerprint", panel_path=str(tmp_path))
    monkeypatch.setattr(
        "autoalpha.service.research_manager.inspect_data_workspace", lambda _: workspace
    )
    monkeypatch.setattr(
        "autoalpha.service.research_manager.inspect_execution_data_basis",
        lambda _: SimpleNamespace(capital_ledger_proxy_ready=True, proxy_blockers=()),
    )
    monkeypatch.setattr(
        "autoalpha.service.research_manager.protocol_data_blockers", lambda *_: []
    )
    monkeypatch.setattr(
        "autoalpha.service.research_manager.panel_validation_fold_capacity",
        lambda *_: {"maximum_folds": 6},
    )
    manager = ResearchTaskManager(
        store,
        SecretVault(api_key="test"),
        config_path=Path("config/research.toml"),
        artifact_root=tmp_path / "artifacts",
        maximum_concurrent_iterations=3,
    )

    ready = manager.readiness("task-ready")
    short = manager.readiness("task-short")
    china = manager.readiness("task-cn")

    assert ready["runnable"] is True
    assert ready["maximum_concurrent_iterations"] == 3
    assert ready["snapshot_changed"] is True
    assert any("至少需要" in blocker for blocker in short["blockers"])
    assert any("US equities only" in blocker for blocker in china["blockers"])


def test_candidate_level_failure_classifier_covers_metric_and_coverage_errors() -> None:
    assert _candidate_level_failure_reason(
        ValueError("Evaluation produced non-finite metrics")
    ) == "NON_FINITE_SINGLE_FACTOR_METRICS"
    assert _candidate_level_failure_reason(
        ValueError("severe coverage shortfall")
    ) == "SEVERE_COVERAGE_SHORTFALL"
    assert _candidate_level_failure_reason(
        ValueError("zero observations after alignment")
    ) == "INSUFFICIENT_OBSERVATIONS"
    assert _candidate_level_failure_reason(ConnectionError("db connection lost")) is None
