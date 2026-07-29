from __future__ import annotations

import json
from typing import Any

from autoalpha.service.postgres_system_jobs import PostgresSystemJobStore


def _job_row(job_id: str = "job-pg", *, status: str = "QUEUED") -> dict[str, Any]:
    return {
        "job_id": job_id,
        "queue": "system",
        "job_type": "factor_knowledge_map_sync",
        "status": status,
        "priority": 100,
        "resource_group": "default",
        "max_workers": 1,
        "progress_current": 0,
        "progress_total": 1,
        "payload_json": json.dumps({"x": 1}),
        "checkpoint_json": "{}",
        "result_json": "{}",
        "error": None,
        "attempts": 0,
        "max_attempts": 3,
        "lease_owner": None,
        "lease_expires_at": None,
        "heartbeat_at": None,
        "created_at": "2026-07-29T00:00:00+00:00",
        "updated_at": "2026-07-29T00:00:00+00:00",
        "started_at": None,
        "finished_at": None,
    }


def _formal_strategy_row(*, lifecycle: str = "RESEARCH", version: int = 1) -> dict[str, Any]:
    return {
        "strategy_uid": "STR_pg",
        "version": version,
        "source_experiment_id": "EXP_pg",
        "name": "PG Strategy",
        "market": "CN_A",
        "lifecycle": lifecycle,
        "signal_policy_json": json.dumps({"factor_ids": ["F_1"]}),
        "rebalance_policy_json": json.dumps({"frequency": "WEEKLY"}),
        "execution_policy_json": json.dumps({"fill": "NEXT_OPEN"}),
        "risk_policy_json": json.dumps({"max_positions": 50}),
        "cost_policy_json": json.dumps({"commission_bps": 3}),
        "monitoring_policy_json": json.dumps({"max_drawdown": -0.12}),
        "evidence_json": json.dumps({"public_validation_passed": lifecycle != "RESEARCH"}),
        "specification_hash": "hash-pg",
        "created_at": "2026-07-29T00:00:00+00:00",
    }


def _factor_knowledge_row(factor_id: str = "F_pg") -> dict[str, Any]:
    return {
        "factor_id": factor_id,
        "canonical_mechanism": "TURNOVER_LIQUIDITY",
        "mechanism_summary": "Volume stability proxy",
        "tags_json": json.dumps(["AUTOALPHA", "TURNOVER_LIQUIDITY"]),
        "review_json": json.dumps({"behavior_cluster_id": "B1"}),
        "falsification_json": json.dumps({"near_duplicate": False}),
        "updated_at": "2026-07-29T00:00:00+00:00",
    }


def _factor_knowledge_catalog_row(factor_id: str = "F_pg") -> dict[str, Any]:
    row = _factor_knowledge_row(factor_id)
    row.update(
        {
            "name": "Volume Stability",
            "family": "Liquidity",
            "source_task_id": "task-aa",
            "source_iteration": 7,
        }
    )
    return row


def _factor_edge_row() -> dict[str, Any]:
    return {
        "source_factor_id": "F_pg",
        "target_factor_id": "F_peer",
        "relation": "RELATED",
        "confidence": 0.82,
        "rationale": "similar behavior",
        "created_at": "2026-07-29T00:00:00+00:00",
        "target_name": "Peer Factor",
    }


def _factor_pool_row(factor_id: str = "F_pg", *, source_iteration: int = 7) -> dict[str, Any]:
    return {
        "factor_id": factor_id,
        "source_task_id": "task-aa",
        "source_iteration": source_iteration,
        "name": "Volume Stability",
        "family": "Liquidity",
        "proposal_json": json.dumps({"name": "Volume Stability", "family": "Liquidity"}),
        "metrics_json": json.dumps({"long_only_sharpe_ratio": 1.23}),
        "status": "ELIGIBLE",
        "status_reason": "passed",
        "created_at": "2026-07-29T00:00:00+00:00",
        "updated_at": "2026-07-29T00:00:00+00:00",
    }


def _settings_revision_row() -> dict[str, Any]:
    return {
        "id": 5,
        "created_at": "2026-07-29T00:00:00+00:00",
        "change_note": "update",
        "changed_by": "tester",
        "changed_keys_json": json.dumps(["mode"]),
        "previous_values_json": json.dumps({"mode": "old"}),
        "values_json": json.dumps({"mode": "new"}),
        "metadata_json": json.dumps({"source": "test"}),
        "fingerprint": "fp-settings",
    }


def _event_row(event_id: int = 1, *, previous_hash: str = "0" * 64) -> dict[str, Any]:
    return {
        "id": event_id,
        "timestamp_utc": "2026-07-29T00:00:00+00:00",
        "run_id": "run-1",
        "iteration": 3,
        "category": "audit",
        "level": "INFO",
        "event": "TEST_EVENT",
        "title": "Test event",
        "message": "ok",
        "payload_json": json.dumps({"task_id": "task-aa"}),
        "previous_hash": previous_hash,
        "record_hash": "1" * 64,
    }


def _research_task_row(
    task_id: str = "task-aa",
    *,
    status: str = "READY",
    phase: str = "WAITING",
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "name": "A-share Research",
        "market": "A 股",
        "data_path": "/data",
        "data_start": "2010-01-04",
        "data_end": "2026-07-29",
        "snapshot_hash": "snap",
        "status": status,
        "run_id": "run-1",
        "phase": phase,
        "iteration": 3,
        "stop_requested": 0,
        "last_error": None,
        "protocol_json": json.dumps({"exploration_start": "2020-01-01"}),
        "protocol_hash": "proto",
        "protocol_revision": 1,
        "notes": "test",
        "created_at": "2026-07-29T00:00:00+00:00",
        "updated_at": "2026-07-29T00:00:00+00:00",
    }


def _iteration_row(
    *,
    status: str = "COMPLETED",
    candidate_id: str | None = "F_pg",
) -> dict[str, Any]:
    return {
        "id": 9,
        "run_id": "run-1",
        "iteration": 3,
        "candidate_id": candidate_id,
        "status": status,
        "proposal_json": json.dumps({"name": "Factor"}),
        "metrics_json": json.dumps({"long_only_sharpe_ratio": 1.2}),
        "decision": "ADD",
        "error": None,
        "started_at": "2026-07-29T00:00:00+00:00",
        "finished_at": "2026-07-29T00:01:00+00:00",
    }


def _llm_artifact_row() -> dict[str, Any]:
    return {
        "id": 3,
        "task_id": "task-aa",
        "run_id": "run-1",
        "iteration": 3,
        "candidate_id": "F_pg",
        "role": "risk_officer",
        "stage": "REVIEW",
        "status": "COMPLETED",
        "artifact_json": json.dumps({"risk": "ok"}),
        "usage_json": json.dumps({"total_tokens": 123}),
        "prompt_hash": "prompt",
        "response_hash": "response",
        "error": None,
        "created_at": "2026-07-29T00:00:00+00:00",
    }


class FakeCursor:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses
        self.executed: list[tuple[str, tuple[Any, ...]]] = []
        self.rowcount = 1

    def execute(self, sql: str, parameters: tuple[Any, ...] = ()) -> FakeCursor:
        self.executed.append((sql, parameters))
        return self

    def fetchone(self) -> Any | None:
        return self.responses.pop(0) if self.responses else None

    def fetchall(self) -> list[Any]:
        response = self.responses.pop(0) if self.responses else []
        return list(response)


class FakeConnection:
    def __init__(self, cursor: FakeCursor) -> None:
        self._cursor = cursor
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def cursor(self) -> FakeCursor:
        return self._cursor

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1

    def close(self) -> None:
        self.closed = True


def test_postgres_system_job_store_enqueues_with_postgres_sql() -> None:
    cursor = FakeCursor([_job_row()])
    connection = FakeConnection(cursor)
    store = PostgresSystemJobStore(connection_factory=lambda: connection)

    job = store.enqueue_system_job(
        job_id="job-pg",
        queue="system",
        job_type="factor_knowledge_map_sync",
        payload={"x": 1},
        progress_total=1,
    )

    assert job["job_id"] == "job-pg"
    assert job["payload"] == {"x": 1}
    assert any("INSERT INTO system_jobs" in sql and "%s" in sql for sql, _ in cursor.executed)
    assert any("INSERT INTO system_job_logs" in sql and "%s" in sql for sql, _ in cursor.executed)
    assert connection.commits == 2
    assert connection.rollbacks == 0


def test_postgres_system_job_store_updates_and_logs_status_change() -> None:
    cursor = FakeCursor([
        {"status": "QUEUED"},
        _job_row(status="RUNNING"),
    ])
    connection = FakeConnection(cursor)
    store = PostgresSystemJobStore(connection_factory=lambda: connection)

    job = store.update_system_job("job-pg", status="RUNNING", lease_owner="worker-1")

    assert job["status"] == "RUNNING"
    assert any(
        "UPDATE system_jobs SET status=%s, lease_owner=%s" in sql
        for sql, _ in cursor.executed
    )
    assert any(
        parameters[3] == "STATUS_CHANGED"
        for _, parameters in cursor.executed
        if len(parameters) >= 4
    )


def test_postgres_system_job_store_claims_with_skip_locked() -> None:
    claimed = _job_row(status="RUNNING")
    claimed["lease_owner"] = "worker-1"
    cursor = FakeCursor([
        [_job_row()],
        {"capacity": 1},
        {"active": 0},
        claimed,
    ])
    connection = FakeConnection(cursor)
    store = PostgresSystemJobStore(connection_factory=lambda: connection)

    job = store.claim_system_job(queue="system", worker_id="worker-1")

    assert job is not None
    assert job["lease_owner"] == "worker-1"
    assert any("FOR UPDATE SKIP LOCKED" in sql for sql, _ in cursor.executed)
    assert any("attempts=attempts + 1" in sql for sql, _ in cursor.executed)


def test_postgres_store_upserts_materialized_snapshot() -> None:
    cursor = FakeCursor(
        [
            {
                "key": "strategy_bus",
                "payload_json": json.dumps({"count": 3}),
                "fingerprint": "fp",
                "source": "job:sync",
                "status": "READY",
                "expires_at": None,
                "updated_at": "2026-07-29T00:00:00+00:00",
            }
        ]
    )
    connection = FakeConnection(cursor)
    store = PostgresSystemJobStore(connection_factory=lambda: connection)

    snapshot = store.upsert_materialized_snapshot(
        "strategy_bus",
        {"count": 3},
        fingerprint="fp",
        source="job:sync",
    )

    assert snapshot["payload"] == {"count": 3}
    assert snapshot["cache_state"]["status"] == "NO_TTL"
    assert any("INSERT INTO materialized_snapshots" in sql for sql, _ in cursor.executed)
    assert any("ON CONFLICT(key) DO UPDATE" in sql for sql, _ in cursor.executed)


def test_postgres_store_upserts_strategy_experiment_object() -> None:
    row = {
        "experiment_id": "EXP_pg",
        "stage": "COMBINATION_CANDIDATE",
        "object_type": "factor_combination",
        "source_system": "TEST",
        "source_id": "candidate-1",
        "title": "Candidate 1",
        "status": "RESEARCH_LEADER",
        "market": "CN_A",
        "protocol_json": json.dumps({"primary": "long_only"}),
        "metrics_json": json.dumps({"portfolio_sharpe_ratio": 1.1}),
        "evidence_json": json.dumps({"factor_ids": ["F_1"]}),
        "tags_json": json.dumps(["AUTO"]),
        "created_at": "2026-07-29T00:00:00+00:00",
        "updated_at": "2026-07-29T00:00:00+00:00",
    }
    cursor = FakeCursor([row])
    connection = FakeConnection(cursor)
    store = PostgresSystemJobStore(connection_factory=lambda: connection)

    record = store.upsert_strategy_experiment_object(
        experiment_id="EXP_pg",
        stage="COMBINATION_CANDIDATE",
        object_type="factor_combination",
        source_system="TEST",
        source_id="candidate-1",
        title="Candidate 1",
        status="RESEARCH_LEADER",
        market="CN_A",
        protocol={"primary": "long_only"},
        metrics={"portfolio_sharpe_ratio": 1.1},
        evidence={"factor_ids": ["F_1"]},
        tags=["AUTO"],
    )

    assert record["experiment_id"] == "EXP_pg"
    assert record["protocol"] == {"primary": "long_only"}
    assert record["metrics"] == {"portfolio_sharpe_ratio": 1.1}
    assert record["evidence"] == {"factor_ids": ["F_1"]}
    assert record["tags"] == ["AUTO"]
    assert any("INSERT INTO strategy_experiment_objects" in sql for sql, _ in cursor.executed)


def test_postgres_store_upserts_strategy_experiment_edge() -> None:
    cursor = FakeCursor([])
    connection = FakeConnection(cursor)
    store = PostgresSystemJobStore(connection_factory=lambda: connection)

    store.upsert_strategy_experiment_edge(
        "EXP_factor",
        "EXP_combo",
        "CONTRIBUTES_TO",
        evidence={"weight": 0.4},
    )

    assert any("INSERT INTO strategy_experiment_edges" in sql for sql, _ in cursor.executed)
    assert cursor.executed[0][1][:3] == ("EXP_factor", "EXP_combo", "CONTRIBUTES_TO")


def test_postgres_store_creates_formal_strategy_version() -> None:
    cursor = FakeCursor([
        {"version": 0},
        _formal_strategy_row(),
    ])
    connection = FakeConnection(cursor)
    store = PostgresSystemJobStore(connection_factory=lambda: connection)

    strategy = store.create_formal_strategy_version(
        strategy_uid="STR_pg",
        source_experiment_id="EXP_pg",
        name="PG Strategy",
        market="CN_A",
        lifecycle="RESEARCH",
        signal_policy={"factor_ids": ["F_1"]},
        rebalance_policy={"frequency": "WEEKLY"},
        execution_policy={"fill": "NEXT_OPEN"},
        risk_policy={"max_positions": 50},
        cost_policy={"commission_bps": 3},
        monitoring_policy={"max_drawdown": -0.12},
        evidence={"public_validation_passed": False},
    )

    assert strategy["strategy_uid"] == "STR_pg"
    assert strategy["version"] == 1
    assert strategy["signal_policy"] == {"factor_ids": ["F_1"]}
    assert strategy["execution_policy"] == {"fill": "NEXT_OPEN"}
    assert any("INSERT INTO formal_strategy_versions" in sql for sql, _ in cursor.executed)


def test_postgres_store_lists_formal_strategy_versions() -> None:
    cursor = FakeCursor([[_formal_strategy_row(), _formal_strategy_row(version=2)]])
    connection = FakeConnection(cursor)
    store = PostgresSystemJobStore(connection_factory=lambda: connection)

    strategies = store.formal_strategy_versions(lifecycle="RESEARCH", limit=10)

    assert [item["version"] for item in strategies] == [1, 2]
    assert all(item["lifecycle"] == "RESEARCH" for item in strategies)
    assert any(
        "SELECT * FROM formal_strategy_versions WHERE lifecycle=%s" in sql
        for sql, _ in cursor.executed
    )


def test_postgres_store_updates_formal_strategy_lifecycle() -> None:
    cursor = FakeCursor([_formal_strategy_row(lifecycle="FROZEN")])
    connection = FakeConnection(cursor)
    store = PostgresSystemJobStore(connection_factory=lambda: connection)

    strategy = store.update_formal_strategy_lifecycle(
        "STR_pg",
        1,
        lifecycle="FROZEN",
        evidence={"public_validation_passed": True},
    )

    assert strategy["lifecycle"] == "FROZEN"
    assert strategy["evidence"] == {"public_validation_passed": True}
    assert any("UPDATE formal_strategy_versions" in sql for sql, _ in cursor.executed)


def test_postgres_store_upserts_factor_knowledge_and_edges() -> None:
    cursor = FakeCursor([
        {"exists": 1},
        {"exists": 1},
        _factor_knowledge_row(),
        [_factor_edge_row()],
    ])
    connection = FakeConnection(cursor)
    store = PostgresSystemJobStore(connection_factory=lambda: connection)

    knowledge = store.upsert_factor_knowledge(
        factor_id="F_pg",
        canonical_mechanism="TURNOVER_LIQUIDITY",
        mechanism_summary="Volume stability proxy",
        tags=["AUTOALPHA", "TURNOVER_LIQUIDITY"],
        review={"behavior_cluster_id": "B1"},
        falsification={"near_duplicate": False},
        related_factors=[
            {
                "factor_id": "F_peer",
                "relation": "RELATED",
                "confidence": 1.7,
                "rationale": "similar behavior",
            }
        ],
    )

    assert knowledge["factor_id"] == "F_pg"
    assert knowledge["tags"] == ["AUTOALPHA", "TURNOVER_LIQUIDITY"]
    assert knowledge["review"] == {"behavior_cluster_id": "B1"}
    assert knowledge["falsification"] == {"near_duplicate": False}
    assert knowledge["edges"][0]["target_factor_id"] == "F_peer"
    edge_insert = [
        item
        for item in cursor.executed
        if "INSERT INTO factor_knowledge_edges" in item[0]
    ][0]
    assert edge_insert[1][3] == 1.0


def test_postgres_store_factor_knowledge_missing_factor_raises() -> None:
    cursor = FakeCursor([None])
    connection = FakeConnection(cursor)
    store = PostgresSystemJobStore(connection_factory=lambda: connection)

    try:
        store.upsert_factor_knowledge(
            factor_id="F_missing",
            canonical_mechanism="TURNOVER_LIQUIDITY",
            mechanism_summary="missing",
            tags=[],
            review={},
            falsification={},
            related_factors=[],
        )
    except KeyError as error:
        assert "F_missing" in str(error)
    else:
        raise AssertionError("expected missing factor to raise")


def test_postgres_store_reads_factor_knowledge_catalog() -> None:
    cursor = FakeCursor([[_factor_knowledge_catalog_row()]])
    connection = FakeConnection(cursor)
    store = PostgresSystemJobStore(connection_factory=lambda: connection)

    catalog = store.factor_knowledge_catalog(task_id="task-aa", limit=25)

    assert catalog[0]["factor_id"] == "F_pg"
    assert catalog[0]["name"] == "Volume Stability"
    assert catalog[0]["source_task_id"] == "task-aa"
    assert catalog[0]["review"] == {"behavior_cluster_id": "B1"}
    assert any("WHERE pool.source_task_id=%s" in sql for sql, _ in cursor.executed)


def test_postgres_store_upserts_factor_pool_and_initial_lifecycle() -> None:
    cursor = FakeCursor([])
    connection = FakeConnection(cursor)
    store = PostgresSystemJobStore(connection_factory=lambda: connection)

    store.upsert_factor_pool(
        factor_id="F_pg",
        source_task_id="task-aa",
        source_iteration=7,
        proposal={"name": "Volume Stability", "family": "Liquidity"},
        metrics={"long_only_sharpe_ratio": 1.23},
        status="ELIGIBLE",
        status_reason="passed",
    )

    assert any("INSERT INTO factor_pool" in sql for sql, _ in cursor.executed)
    lifecycle_insert = [
        item
        for item in cursor.executed
        if "INSERT INTO factor_lifecycle_events" in item[0]
    ][0]
    assert lifecycle_insert[1][1] == "QUALIFIED"


def test_postgres_store_reads_factor_pool_records() -> None:
    cursor = FakeCursor([
        [_factor_pool_row(source_iteration=9), _factor_pool_row("F_pg2", source_iteration=8)],
        {"count": 2},
        _factor_pool_row(),
    ])
    connection = FakeConnection(cursor)
    store = PostgresSystemJobStore(connection_factory=lambda: connection)

    pool = store.factor_pool(limit=10)
    count = store.factor_pool_count()
    record = store.factor_pool_record("F_pg")

    assert [item["factor_id"] for item in pool] == ["F_pg", "F_pg2"]
    assert pool[0]["proposal"]["name"] == "Volume Stability"
    assert pool[0]["metrics"]["long_only_sharpe_ratio"] == 1.23
    assert count == 2
    assert record is not None
    assert record["source_task_id"] == "task-aa"
    assert any("SELECT COUNT(*) AS count FROM factor_pool" in sql for sql, _ in cursor.executed)


def test_postgres_store_merges_factor_pool_metrics() -> None:
    cursor = FakeCursor([
        {"metrics_json": json.dumps({"long_only_sharpe_ratio": 1.23})},
    ])
    connection = FakeConnection(cursor)
    store = PostgresSystemJobStore(connection_factory=lambda: connection)

    merged = store.merge_factor_pool_metrics(
        "F_pg",
        {"homogeneity_cluster_id": "B001"},
    )

    assert merged == {"long_only_sharpe_ratio": 1.23, "homogeneity_cluster_id": "B001"}
    update_sql, update_params = cursor.executed[-1]
    assert "UPDATE factor_pool SET metrics_json=%s" in update_sql
    assert json.loads(update_params[0])["homogeneity_cluster_id"] == "B001"
    assert update_params[2] == "F_pg"


def test_postgres_store_saves_and_reads_settings() -> None:
    cursor = FakeCursor([[{"key": "mode", "value": "production"}]])
    connection = FakeConnection(cursor)
    store = PostgresSystemJobStore(connection_factory=lambda: connection)

    store.save_settings({"mode": "production"})
    settings = store.settings()

    assert settings == {"mode": "production"}
    assert any("INSERT INTO settings" in sql for sql, _ in cursor.executed)


def test_postgres_store_saves_settings_revision_without_secrets() -> None:
    cursor = FakeCursor([
        [{"key": "mode", "value": "old"}],
        {"id": 5},
        _settings_revision_row(),
    ])
    connection = FakeConnection(cursor)
    store = PostgresSystemJobStore(connection_factory=lambda: connection)

    revision = store.save_settings_revision(
        {"mode": "new"},
        change_note="update",
        changed_by="tester",
        metadata={"source": "test"},
    )

    assert revision is not None
    assert revision["changed_keys"] == ["mode"]
    assert revision["previous_values"] == {"mode": "old"}
    assert revision["values"] == {"mode": "new"}
    assert any("INSERT INTO settings_revisions" in sql for sql, _ in cursor.executed)


def test_postgres_store_settings_revision_rejects_secrets() -> None:
    cursor = FakeCursor([])
    connection = FakeConnection(cursor)
    store = PostgresSystemJobStore(connection_factory=lambda: connection)

    try:
        store.save_settings_revision({"api_key": "secret"}, change_note="bad")
    except ValueError as error:
        assert "Secrets cannot be stored" in str(error)
    else:
        raise AssertionError("expected secret setting revision to raise")


def test_postgres_store_appends_and_reads_events_with_hash_chain() -> None:
    previous_hash = "a" * 64
    cursor = FakeCursor([
        {"record_hash": previous_hash},
        {"id": 7},
        [_event_row(7, previous_hash=previous_hash)],
    ])
    connection = FakeConnection(cursor)
    store = PostgresSystemJobStore(connection_factory=lambda: connection)

    event = store.append_event(
        "audit",
        "TEST_EVENT",
        "Test event",
        "ok",
        run_id="run-1",
        iteration=3,
        payload={"task_id": "task-aa"},
    )
    events = store.events(after_id=0, task_id="task-aa", run_id="run-1")

    assert event["id"] == 7
    assert event["previous_hash"] == previous_hash
    assert len(event["record_hash"]) == 64
    assert events[0]["payload"] == {"task_id": "task-aa"}
    assert any("RETURNING id" in sql and "INSERT INTO events" in sql for sql, _ in cursor.executed)
    assert any("payload_json::jsonb ->> 'task_id'" in sql for sql, _ in cursor.executed)


def test_postgres_store_creates_and_reads_research_task() -> None:
    cursor = FakeCursor([
        _research_task_row(),
        [_research_task_row(), _research_task_row("task-b", status="DRAFT", phase="CONFIGURE")],
        _research_task_row(),
    ])
    connection = FakeConnection(cursor)
    store = PostgresSystemJobStore(connection_factory=lambda: connection)

    created = store.create_research_task(
        task_id="task-aa",
        name="A-share Research",
        market="A 股",
        data_path="/data",
        data_start="2010-01-04",
        data_end="2026-07-29",
        snapshot_hash="snap",
        status="READY",
        run_id="run-1",
        protocol={"exploration_start": "2020-01-01"},
        protocol_hash="proto",
        notes="test",
    )
    tasks = store.research_tasks()
    loaded = store.research_task("task-aa")

    assert created["phase"] == "WAITING"
    assert created["protocol"] == {"exploration_start": "2020-01-01"}
    assert [item["task_id"] for item in tasks] == ["task-aa", "task-b"]
    assert loaded is not None
    assert loaded["task_id"] == "task-aa"
    assert any("INSERT INTO research_tasks" in sql for sql, _ in cursor.executed)


def test_postgres_store_research_task_stats_and_state_update() -> None:
    cursor = FakeCursor([
        {"count": 5},
        {"count": 12},
        _research_task_row(status="RUNNING", phase="MEMORY"),
        _research_task_row(status="RUNNING", phase="MEMORY"),
    ])
    connection = FakeConnection(cursor)
    store = PostgresSystemJobStore(connection_factory=lambda: connection)

    stats = store.research_task_stats("task-aa", "run-1")
    state = store.update_research_task_state(
        "task-aa",
        state="RUNNING",
        phase="MEMORY",
        iteration=4,
    )

    assert stats == {"factor_count": 5, "iteration_count": 12}
    assert state["state"] == "RUNNING"
    assert state["phase"] == "MEMORY"
    assert any("UPDATE research_tasks SET status=%s" in sql for sql, _ in cursor.executed)


def test_postgres_store_research_task_update_rejects_unknown_field() -> None:
    store = PostgresSystemJobStore(connection_factory=lambda: FakeConnection(FakeCursor([])))

    try:
        store.update_research_task("task-aa", unknown=True)
    except ValueError as error:
        assert "Unknown research-task fields" in str(error)
    else:
        raise AssertionError("expected unknown research task field to raise")


def test_postgres_store_records_iteration_lifecycle() -> None:
    cursor = FakeCursor([
        _iteration_row(),
        [_iteration_row()],
        [{"status": "COMPLETED", "count": 1}, {"status": "RUNNING", "count": 1}],
    ])
    connection = FakeConnection(cursor)
    store = PostgresSystemJobStore(connection_factory=lambda: connection)

    store.begin_iteration("run-1", 3)
    store.stage_iteration_candidate(
        "run-1",
        3,
        candidate_id="F_pg",
        proposal={"name": "Factor"},
    )
    store.finish_iteration(
        "run-1",
        3,
        status="COMPLETED",
        metrics={"long_only_sharpe_ratio": 1.2},
        decision="ADD",
    )
    record = store.iteration_record("run-1", 3)
    history = store.iteration_history(run_id="run-1")
    stats = store.iteration_stats(run_id="run-1")

    assert record is not None
    assert record["proposal"] == {"name": "Factor"}
    assert record["metrics"] == {"long_only_sharpe_ratio": 1.2}
    assert history[0]["candidate_id"] == "F_pg"
    assert stats["completed"] == 1
    assert stats["running"] == 1
    assert any("INSERT INTO iterations" in sql for sql, _ in cursor.executed)
    assert any("UPDATE iterations SET candidate_id=%s" in sql for sql, _ in cursor.executed)


def test_postgres_store_metric_history_and_candidate_exists() -> None:
    cursor = FakeCursor([
        [_iteration_row()],
        {"exists": 1},
    ])
    connection = FakeConnection(cursor)
    store = PostgresSystemJobStore(connection_factory=lambda: connection)

    metrics = store.metric_history(run_id="run-1")
    exists = store.candidate_exists("F_pg")

    assert metrics[0]["candidate_id"] == "F_pg"
    assert metrics[0]["long_only_sharpe_ratio"] == 1.2
    assert exists is True


def test_postgres_store_records_and_summarizes_llm_artifacts() -> None:
    cursor = FakeCursor([
        {"id": 3},
        [_llm_artifact_row()],
        [
            {
                "role": "risk_officer",
                "status": "COMPLETED",
                "count": 2,
                "total_tokens": 246,
                "latest_at": "2026-07-29T00:00:00+00:00",
            }
        ],
    ])
    connection = FakeConnection(cursor)
    store = PostgresSystemJobStore(connection_factory=lambda: connection)

    artifact = store.record_llm_role_artifact(
        task_id="task-aa",
        run_id="run-1",
        iteration=3,
        candidate_id="F_pg",
        role="risk_officer",
        stage="REVIEW",
        status="COMPLETED",
        artifact={"risk": "ok"},
        usage={"total_tokens": 123},
        prompt_hash="prompt",
        response_hash="response",
    )
    artifacts = store.llm_role_artifacts(task_id="task-aa", run_id="run-1")
    summary = store.llm_role_summary(task_id="task-aa")

    assert artifact["id"] == 3
    assert artifacts[0]["artifact"] == {"risk": "ok"}
    assert artifacts[0]["usage"] == {"total_tokens": 123}
    assert summary["roles"]["risk_officer"]["completed"] == 2
    assert summary["roles"]["risk_officer"]["total_tokens"] == 246
    assert any("INSERT INTO llm_role_artifacts" in sql for sql, _ in cursor.executed)
    assert any("usage_json::jsonb" in sql for sql, _ in cursor.executed)
