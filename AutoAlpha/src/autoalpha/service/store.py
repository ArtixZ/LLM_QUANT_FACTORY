from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from autoalpha.registry.lifecycle import ALLOWED_TRANSITIONS, FactorState
from autoalpha.service.system_job_sql import system_job_sql

LogCategory = Literal["audit", "action", "research", "delivery"]


class ServiceStore:
    """SQLite state, memory, metrics, and a tamper-evident unified event stream."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._system_job_sql = system_job_sql("sqlite")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self.connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS service_state (
                    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                    state TEXT NOT NULL,
                    phase TEXT NOT NULL DEFAULT 'CONFIGURE',
                    run_id TEXT,
                    iteration INTEGER NOT NULL DEFAULT 0,
                    stop_requested INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL,
                    last_error TEXT
                );
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS settings_revisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    change_note TEXT NOT NULL,
                    changed_by TEXT NOT NULL,
                    changed_keys_json TEXT NOT NULL,
                    previous_values_json TEXT NOT NULL,
                    values_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    fingerprint TEXT NOT NULL UNIQUE
                );
                CREATE TABLE IF NOT EXISTS entity_favorites (
                    entity_type TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    label TEXT NOT NULL DEFAULT '',
                    context_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(entity_type, entity_id)
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp_utc TEXT NOT NULL,
                    run_id TEXT,
                    iteration INTEGER,
                    category TEXT NOT NULL,
                    level TEXT NOT NULL,
                    event TEXT NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    previous_hash TEXT NOT NULL,
                    record_hash TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS iterations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    iteration INTEGER NOT NULL,
                    candidate_id TEXT,
                    status TEXT NOT NULL,
                    proposal_json TEXT,
                    metrics_json TEXT,
                    decision TEXT,
                    error TEXT,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    UNIQUE(run_id, iteration)
                );
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    iteration INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS factor_pool (
                    factor_id TEXT PRIMARY KEY,
                    source_task_id TEXT NOT NULL DEFAULT 'legacy-ashare',
                    source_iteration INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    family TEXT NOT NULL,
                    proposal_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    status_reason TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS research_tasks (
                    task_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    market TEXT NOT NULL,
                    data_path TEXT NOT NULL,
                    data_start TEXT,
                    data_end TEXT,
                    snapshot_hash TEXT,
                    status TEXT NOT NULL,
                    run_id TEXT,
                    phase TEXT NOT NULL DEFAULT 'WAITING',
                    iteration INTEGER NOT NULL DEFAULT 0,
                    stop_requested INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    protocol_json TEXT NOT NULL DEFAULT '{}',
                    protocol_hash TEXT,
                    protocol_revision INTEGER NOT NULL DEFAULT 1,
                    notes TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS portfolio_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    iteration INTEGER NOT NULL,
                    action TEXT NOT NULL,
                    candidate_id TEXT,
                    removed_factor_id TEXT,
                    accepted INTEGER NOT NULL,
                    reason TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS portfolio_members (
                    version_id INTEGER NOT NULL REFERENCES portfolio_versions(id),
                    factor_id TEXT NOT NULL REFERENCES factor_pool(factor_id),
                    weight REAL NOT NULL,
                    PRIMARY KEY(version_id, factor_id)
                );
                CREATE TABLE IF NOT EXISTS research_generations (
                    generation_id TEXT PRIMARY KEY,
                    protocol_version TEXT NOT NULL,
                    status TEXT NOT NULL,
                    candidate_attempts INTEGER NOT NULL DEFAULT 0,
                    holdout_attempts INTEGER NOT NULL DEFAULT 0,
                    maximum_candidates INTEGER NOT NULL,
                    maximum_holdout_attempts INTEGER NOT NULL,
                    started_at TEXT NOT NULL,
                    sealed_at TEXT
                );
                CREATE TABLE IF NOT EXISTS generation_experiments (
                    candidate_hash TEXT PRIMARY KEY,
                    generation_id TEXT NOT NULL REFERENCES research_generations(generation_id),
                    factor_id TEXT NOT NULL,
                    family TEXT NOT NULL,
                    iteration INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    public_verdict TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS blind_evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_hash TEXT NOT NULL UNIQUE,
                    generation_id TEXT NOT NULL REFERENCES research_generations(generation_id),
                    iteration INTEGER NOT NULL,
                    holdout_verdict TEXT NOT NULL,
                    holdout_passed INTEGER NOT NULL,
                    holdout_evidence_hash TEXT NOT NULL,
                    capital_verdict TEXT,
                    capital_passed INTEGER,
                    capital_evidence_hash TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS direction_campaigns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    generation_id TEXT NOT NULL REFERENCES research_generations(generation_id),
                    direction TEXT NOT NULL,
                    title TEXT NOT NULL,
                    objective TEXT NOT NULL,
                    status TEXT NOT NULL,
                    diagnostic_score REAL NOT NULL,
                    rationale_json TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    baseline_json TEXT NOT NULL,
                    maximum_attempts INTEGER NOT NULL,
                    attempts_used INTEGER NOT NULL DEFAULT 0,
                    successful_attempts INTEGER NOT NULL DEFAULT 0,
                    started_iteration INTEGER NOT NULL,
                    last_iteration INTEGER,
                    closed_iteration INTEGER,
                    closure_reason TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS direction_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    campaign_id INTEGER NOT NULL REFERENCES direction_campaigns(id),
                    run_id TEXT,
                    iteration INTEGER NOT NULL,
                    candidate_id TEXT,
                    status TEXT NOT NULL,
                    outcome TEXT,
                    improved INTEGER,
                    objective_resolved INTEGER,
                    baseline_json TEXT NOT NULL,
                    diagnostics_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(campaign_id, iteration),
                    UNIQUE(run_id, iteration)
                );
                CREATE TABLE IF NOT EXISTS manual_backtests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    status TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    metrics_json TEXT,
                    artifact_path TEXT,
                    result_hash TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    favorite INTEGER NOT NULL DEFAULT 0,
                    title TEXT,
                    notes TEXT NOT NULL DEFAULT '',
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    updated_at TEXT
                );
                CREATE TABLE IF NOT EXISTS manual_research_exposures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    backtest_id INTEGER NOT NULL REFERENCES manual_backtests(id),
                    generation_id TEXT NOT NULL,
                    factor_id TEXT NOT NULL REFERENCES factor_pool(factor_id),
                    period_start TEXT NOT NULL,
                    period_end TEXT NOT NULL,
                    visibility_scope TEXT NOT NULL,
                    contaminated INTEGER NOT NULL,
                    evidence_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(backtest_id, factor_id)
                );
                CREATE TABLE IF NOT EXISTS factor_lifecycle_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    factor_id TEXT NOT NULL REFERENCES factor_pool(factor_id),
                    previous_state TEXT,
                    state TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS llm_role_artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    iteration INTEGER NOT NULL,
                    candidate_id TEXT,
                    role TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL,
                    artifact_json TEXT NOT NULL,
                    usage_json TEXT NOT NULL,
                    prompt_hash TEXT,
                    response_hash TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE(run_id, iteration, role)
                );
                CREATE TABLE IF NOT EXISTS factor_knowledge (
                    factor_id TEXT PRIMARY KEY REFERENCES factor_pool(factor_id),
                    canonical_mechanism TEXT NOT NULL,
                    mechanism_summary TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    review_json TEXT NOT NULL,
                    falsification_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS factor_knowledge_edges (
                    source_factor_id TEXT NOT NULL REFERENCES factor_pool(factor_id),
                    target_factor_id TEXT NOT NULL REFERENCES factor_pool(factor_id),
                    relation TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    rationale TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(source_factor_id, target_factor_id, relation)
                );
                CREATE TABLE IF NOT EXISTS paper_portfolios (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    initial_cash_usd REAL NOT NULL,
                    cash_usd REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_rebalanced_date TEXT
                );
                CREATE TABLE IF NOT EXISTS paper_positions (
                    portfolio_id INTEGER NOT NULL REFERENCES paper_portfolios(id) ON DELETE CASCADE,
                    symbol TEXT NOT NULL,
                    security_name TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    average_cost_usd REAL NOT NULL,
                    acquired_trade_date TEXT,
                    last_trade_date TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(portfolio_id, symbol)
                );
                CREATE TABLE IF NOT EXISTS paper_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    portfolio_id INTEGER NOT NULL REFERENCES paper_portfolios(id) ON DELETE CASCADE,
                    trade_date TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    security_name TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    price_usd REAL NOT NULL,
                    notional_usd REAL NOT NULL,
                    fees_usd REAL NOT NULL,
                    reason TEXT NOT NULL,
                    execution_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS paper_nav (
                    portfolio_id INTEGER NOT NULL REFERENCES paper_portfolios(id) ON DELETE CASCADE,
                    trade_date TEXT NOT NULL,
                    nav_usd REAL NOT NULL,
                    cash_usd REAL NOT NULL,
                    market_value_usd REAL NOT NULL,
                    gross_exposure REAL NOT NULL,
                    daily_return REAL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(portfolio_id, trade_date)
                );
                CREATE TABLE IF NOT EXISTS strategy_experiment_objects (
                    experiment_id TEXT PRIMARY KEY,
                    stage TEXT NOT NULL,
                    object_type TEXT NOT NULL,
                    source_system TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    status TEXT NOT NULL,
                    market TEXT,
                    protocol_json TEXT NOT NULL DEFAULT '{}',
                    metrics_json TEXT NOT NULL DEFAULT '{}',
                    evidence_json TEXT NOT NULL DEFAULT '{}',
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(source_system, source_id, stage)
                );
                CREATE TABLE IF NOT EXISTS strategy_experiment_edges (
                    source_experiment_id TEXT NOT NULL
                        REFERENCES strategy_experiment_objects(experiment_id)
                        ON DELETE CASCADE,
                    target_experiment_id TEXT NOT NULL
                        REFERENCES strategy_experiment_objects(experiment_id)
                        ON DELETE CASCADE,
                    relation TEXT NOT NULL,
                    evidence_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(source_experiment_id, target_experiment_id, relation)
                );
                CREATE TABLE IF NOT EXISTS formal_strategy_versions (
                    strategy_uid TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    source_experiment_id TEXT
                        REFERENCES strategy_experiment_objects(experiment_id),
                    name TEXT NOT NULL,
                    market TEXT NOT NULL,
                    lifecycle TEXT NOT NULL,
                    signal_policy_json TEXT NOT NULL,
                    rebalance_policy_json TEXT NOT NULL,
                    execution_policy_json TEXT NOT NULL,
                    risk_policy_json TEXT NOT NULL,
                    cost_policy_json TEXT NOT NULL,
                    monitoring_policy_json TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    specification_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(strategy_uid, version)
                );
                CREATE TABLE IF NOT EXISTS system_jobs (
                    job_id TEXT PRIMARY KEY,
                    queue TEXT NOT NULL,
                    job_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    priority INTEGER NOT NULL DEFAULT 100,
                    resource_group TEXT NOT NULL DEFAULT 'default',
                    max_workers INTEGER NOT NULL DEFAULT 1,
                    progress_current INTEGER NOT NULL DEFAULT 0,
                    progress_total INTEGER NOT NULL DEFAULT 0,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    checkpoint_json TEXT NOT NULL DEFAULT '{}',
                    result_json TEXT NOT NULL DEFAULT '{}',
                    error TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL DEFAULT 3,
                    lease_owner TEXT,
                    lease_expires_at TEXT,
                    heartbeat_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT
                );
                CREATE TABLE IF NOT EXISTS system_job_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL REFERENCES system_jobs(job_id) ON DELETE CASCADE,
                    timestamp_utc TEXT NOT NULL,
                    level TEXT NOT NULL,
                    event TEXT NOT NULL,
                    message TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS materialized_snapshots (
                    key TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'unknown',
                    status TEXT NOT NULL DEFAULT 'READY',
                    expires_at TEXT,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_factor_pool_status
                ON factor_pool(status, source_iteration);
                CREATE INDEX IF NOT EXISTS idx_entity_favorites_updated
                ON entity_favorites(entity_type, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_research_tasks_status
                ON research_tasks(status, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_portfolio_versions_accepted
                ON portfolio_versions(accepted, id);
                CREATE INDEX IF NOT EXISTS idx_generation_experiments_generation
                ON generation_experiments(generation_id, iteration);
                CREATE INDEX IF NOT EXISTS idx_direction_campaigns_generation
                ON direction_campaigns(generation_id, id);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_direction_campaigns_one_active
                ON direction_campaigns(generation_id) WHERE status='ACTIVE';
                CREATE INDEX IF NOT EXISTS idx_direction_attempts_campaign
                ON direction_attempts(campaign_id, iteration);
                CREATE INDEX IF NOT EXISTS idx_manual_backtests_created
                ON manual_backtests(id DESC);
                CREATE INDEX IF NOT EXISTS idx_manual_exposures_generation
                ON manual_research_exposures(generation_id, contaminated, factor_id);
                CREATE INDEX IF NOT EXISTS idx_factor_lifecycle_factor
                ON factor_lifecycle_events(factor_id, id DESC);
                CREATE INDEX IF NOT EXISTS idx_llm_role_artifacts_task
                ON llm_role_artifacts(task_id, id DESC);
                CREATE INDEX IF NOT EXISTS idx_llm_role_artifacts_candidate
                ON llm_role_artifacts(candidate_id, id DESC);
                CREATE INDEX IF NOT EXISTS idx_factor_knowledge_mechanism
                ON factor_knowledge(canonical_mechanism, factor_id);
                CREATE INDEX IF NOT EXISTS idx_paper_portfolios_status
                ON paper_portfolios(status, id DESC);
                CREATE INDEX IF NOT EXISTS idx_paper_trades_portfolio
                ON paper_trades(portfolio_id, id DESC);
                CREATE INDEX IF NOT EXISTS idx_paper_nav_portfolio
                ON paper_nav(portfolio_id, trade_date DESC);
                CREATE INDEX IF NOT EXISTS idx_strategy_experiment_stage
                ON strategy_experiment_objects(stage, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_strategy_experiment_source
                ON strategy_experiment_objects(source_system, source_id);
                CREATE INDEX IF NOT EXISTS idx_strategy_experiment_edges_target
                ON strategy_experiment_edges(target_experiment_id, relation);
                CREATE INDEX IF NOT EXISTS idx_formal_strategy_lifecycle
                ON formal_strategy_versions(lifecycle, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_system_jobs_queue
                ON system_jobs(queue, status, priority, created_at);
                CREATE INDEX IF NOT EXISTS idx_system_job_logs_job
                ON system_job_logs(job_id, id DESC);
                CREATE TRIGGER IF NOT EXISTS prevent_manual_exposure_update
                BEFORE UPDATE ON manual_research_exposures
                BEGIN
                    SELECT RAISE(ABORT, 'manual research exposure ledger is append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS prevent_manual_exposure_delete
                BEFORE DELETE ON manual_research_exposures
                BEGIN
                    SELECT RAISE(ABORT, 'manual research exposure ledger is append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS prevent_lifecycle_event_update
                BEFORE UPDATE ON factor_lifecycle_events
                BEGIN
                    SELECT RAISE(ABORT, 'factor lifecycle ledger is append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS prevent_lifecycle_event_delete
                BEFORE DELETE ON factor_lifecycle_events
                BEGIN
                    SELECT RAISE(ABORT, 'factor lifecycle ledger is append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS prevent_llm_role_artifact_update
                BEFORE UPDATE ON llm_role_artifacts
                BEGIN
                    SELECT RAISE(ABORT, 'LLM role artifacts are append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS prevent_llm_role_artifact_delete
                BEFORE DELETE ON llm_role_artifacts
                BEGIN
                    SELECT RAISE(ABORT, 'LLM role artifacts are append-only');
                END;
                """
            )
            connection.execute(
                """INSERT INTO factor_lifecycle_events
                (factor_id, previous_state, state, actor, reason, created_at)
                SELECT fp.factor_id, NULL,
                    CASE fp.status
                        WHEN 'ACTIVE' THEN 'SHADOW'
                        WHEN 'ELIGIBLE' THEN 'QUALIFIED'
                        ELSE 'RESEARCH'
                    END,
                    'SYSTEM_MIGRATION', 'Initialized from factor-pool research status', ?
                FROM factor_pool fp
                WHERE NOT EXISTS (
                    SELECT 1 FROM factor_lifecycle_events event
                    WHERE event.factor_id=fp.factor_id
                )""",
                (_now(),),
            )
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(service_state)").fetchall()
            }
            if "phase" not in columns:
                connection.execute(
                    "ALTER TABLE service_state ADD COLUMN phase TEXT NOT NULL DEFAULT 'CONFIGURE'"
                )
            manual_columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(manual_backtests)").fetchall()
            }
            manual_migrations = {
                "favorite": "INTEGER NOT NULL DEFAULT 0",
                "title": "TEXT",
                "notes": "TEXT NOT NULL DEFAULT ''",
                "tags_json": "TEXT NOT NULL DEFAULT '[]'",
                "updated_at": "TEXT",
            }
            for name, declaration in manual_migrations.items():
                if name not in manual_columns:
                    connection.execute(
                        f"ALTER TABLE manual_backtests ADD COLUMN {name} {declaration}"  # noqa: S608
                    )
            factor_columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(factor_pool)").fetchall()
            }
            if "source_task_id" not in factor_columns:
                connection.execute(
                    "ALTER TABLE factor_pool ADD COLUMN source_task_id "
                    "TEXT NOT NULL DEFAULT 'legacy-ashare'"
                )
            task_columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(research_tasks)").fetchall()
            }
            task_migrations = {
                "phase": "TEXT NOT NULL DEFAULT 'WAITING'",
                "iteration": "INTEGER NOT NULL DEFAULT 0",
                "stop_requested": "INTEGER NOT NULL DEFAULT 0",
                "last_error": "TEXT",
                "protocol_json": "TEXT NOT NULL DEFAULT '{}'",
                "protocol_hash": "TEXT",
                "protocol_revision": "INTEGER NOT NULL DEFAULT 1",
            }
            for name, declaration in task_migrations.items():
                if name not in task_columns:
                    connection.execute(
                        f"ALTER TABLE research_tasks ADD COLUMN {name} {declaration}"  # noqa: S608
                    )
            job_columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(system_jobs)").fetchall()
            }
            job_migrations = {
                "lease_owner": "TEXT",
                "lease_expires_at": "TEXT",
                "heartbeat_at": "TEXT",
            }
            for name, declaration in job_migrations.items():
                if name not in job_columns:
                    connection.execute(
                        f"ALTER TABLE system_jobs ADD COLUMN {name} {declaration}"  # noqa: S608
                    )
            snapshot_columns = {
                str(row["name"])
                for row in connection.execute(
                    "PRAGMA table_info(materialized_snapshots)"
                ).fetchall()
            }
            snapshot_migrations = {
                "source": "TEXT NOT NULL DEFAULT 'unknown'",
                "status": "TEXT NOT NULL DEFAULT 'READY'",
                "expires_at": "TEXT",
            }
            for name, declaration in snapshot_migrations.items():
                if name not in snapshot_columns:
                    connection.execute(
                        f"ALTER TABLE materialized_snapshots ADD COLUMN {name} {declaration}"  # noqa: S608
                    )
            paper_trade_columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(paper_trades)").fetchall()
            }
            if "execution_json" not in paper_trade_columns:
                connection.execute(
                    "ALTER TABLE paper_trades ADD COLUMN execution_json "
                    "TEXT NOT NULL DEFAULT '{}'"
                )
            paper_position_columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(paper_positions)").fetchall()
            }
            paper_position_migrations = {
                "acquired_trade_date": "TEXT",
                "last_trade_date": "TEXT",
            }
            for name, declaration in paper_position_migrations.items():
                if name not in paper_position_columns:
                    connection.execute(
                        f"ALTER TABLE paper_positions ADD COLUMN {name} {declaration}"  # noqa: S608
                    )
            direction_table = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='direction_attempts'"
            ).fetchone()
            if direction_table and "iteration INTEGER NOT NULL UNIQUE" in str(
                direction_table["sql"]
            ):
                connection.executescript(
                    """
                    ALTER TABLE direction_attempts RENAME TO direction_attempts_legacy;
                    CREATE TABLE direction_attempts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        campaign_id INTEGER NOT NULL REFERENCES direction_campaigns(id),
                        run_id TEXT,
                        iteration INTEGER NOT NULL,
                        candidate_id TEXT,
                        status TEXT NOT NULL,
                        outcome TEXT,
                        improved INTEGER,
                        objective_resolved INTEGER,
                        baseline_json TEXT NOT NULL,
                        diagnostics_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        UNIQUE(campaign_id, iteration),
                        UNIQUE(run_id, iteration)
                    );
                    INSERT INTO direction_attempts
                    (id, campaign_id, run_id, iteration, candidate_id, status, outcome,
                     improved, objective_resolved, baseline_json, diagnostics_json,
                     created_at, updated_at)
                    SELECT id, campaign_id, NULL, iteration, candidate_id, status, outcome,
                     improved, objective_resolved, baseline_json, diagnostics_json,
                     created_at, updated_at FROM direction_attempts_legacy;
                    DROP TABLE direction_attempts_legacy;
                    CREATE INDEX IF NOT EXISTS idx_direction_attempts_campaign
                    ON direction_attempts(campaign_id, iteration);
                    """
                )
            now = _now()
            connection.execute(
                """INSERT OR IGNORE INTO service_state
                (singleton, state, iteration, stop_requested, updated_at)
                VALUES (1, 'WAITING_CONFIGURATION', 0, 0, ?)""",
                (now,),
            )

    def state(self) -> dict[str, Any]:
        with self.connection() as connection:
            row = connection.execute("SELECT * FROM service_state WHERE singleton=1").fetchone()
        assert row is not None
        return dict(row)

    def update_state(self, **values: Any) -> dict[str, Any]:
        allowed = {"state", "phase", "run_id", "iteration", "stop_requested", "last_error"}
        invalid = set(values) - allowed
        if invalid:
            raise ValueError(f"Unknown state fields: {sorted(invalid)}")
        values["updated_at"] = _now()
        assignments = ", ".join(f"{key}=?" for key in values)
        with self.connection() as connection:
            connection.execute(
                f"UPDATE service_state SET {assignments} WHERE singleton=1",  # noqa: S608
                tuple(values.values()),
            )
        return self.state()

    def settings(self) -> dict[str, str]:
        with self.connection() as connection:
            rows = connection.execute("SELECT key, value FROM settings").fetchall()
        return {str(row["key"]): str(row["value"]) for row in rows}

    def save_settings(self, values: dict[str, str]) -> None:
        now = _now()
        with self.connection() as connection:
            connection.executemany(
                """INSERT INTO settings(key, value, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                value=excluded.value, updated_at=excluded.updated_at""",
                [(key, value, now) for key, value in values.items()],
            )

    def set_favorite(
        self,
        entity_type: str,
        entity_id: str,
        *,
        favorite: bool,
        label: str = "",
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        clean_type = entity_type.strip().lower()
        clean_id = entity_id.strip()
        clean_label = label.strip()
        if not clean_type or len(clean_type) > 40:
            raise ValueError("entity_type must contain 1 to 40 characters")
        if not clean_id or len(clean_id) > 200:
            raise ValueError("entity_id must contain 1 to 200 characters")
        if len(clean_label) > 160:
            raise ValueError("label must contain at most 160 characters")
        context_json = _canonical(context or {})
        if len(context_json.encode("utf-8")) > 32_768:
            raise ValueError("favorite context must contain at most 32 KiB")
        if not favorite:
            with self.connection() as connection:
                connection.execute(
                    "DELETE FROM entity_favorites WHERE entity_type=? AND entity_id=?",
                    (clean_type, clean_id),
                )
            return None
        now = _now()
        with self.connection() as connection:
            connection.execute(
                """INSERT INTO entity_favorites
                (entity_type, entity_id, label, context_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(entity_type, entity_id) DO UPDATE SET
                    label=excluded.label,
                    context_json=excluded.context_json,
                    updated_at=excluded.updated_at""",
                (clean_type, clean_id, clean_label, context_json, now, now),
            )
        return self.favorite(clean_type, clean_id)

    def favorite(self, entity_type: str, entity_id: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM entity_favorites WHERE entity_type=? AND entity_id=?",
                (entity_type.strip().lower(), entity_id.strip()),
            ).fetchone()
        return self._favorite_record(row) if row else None

    def favorites(
        self, *, entity_type: str | None = None, limit: int = 1000
    ) -> list[dict[str, Any]]:
        parameters: list[Any] = []
        where = ""
        if entity_type:
            where = "WHERE entity_type=?"
            parameters.append(entity_type.strip().lower())
        parameters.append(min(max(limit, 1), 5000))
        with self.connection() as connection:
            rows = connection.execute(
                f"SELECT * FROM entity_favorites {where} "  # noqa: S608
                "ORDER BY updated_at DESC LIMIT ?",
                parameters,
            ).fetchall()
        return [self._favorite_record(row) for row in rows]

    def favorite_ids(self, entity_type: str) -> set[str]:
        return {item["entity_id"] for item in self.favorites(entity_type=entity_type, limit=5000)}

    @staticmethod
    def _favorite_record(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["context"] = json.loads(item.pop("context_json"))
        return item

    def save_settings_revision(
        self,
        values: dict[str, str],
        *,
        change_note: str,
        changed_by: str = "local-operator",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Atomically persist changed settings and an immutable, secret-free snapshot."""
        if not values:
            return None
        secret_keys = {"api_key", "openai_api_key", "tushare_token"}
        forbidden = secret_keys.intersection(key.casefold() for key in values)
        if forbidden:
            raise ValueError(f"Secrets cannot be stored in settings revisions: {sorted(forbidden)}")
        now = _now()
        with self.connection() as connection:
            rows = connection.execute("SELECT key, value FROM settings").fetchall()
            previous = {str(row["key"]): str(row["value"]) for row in rows}
            changed = {
                str(key): str(value)
                for key, value in values.items()
                if previous.get(str(key)) != str(value)
            }
            if not changed:
                return None
            current = {**previous, **changed}
            fingerprint = hashlib.sha256(
                _canonical(
                    {
                        "created_at": now,
                        "changed_keys": sorted(changed),
                        "values": current,
                        "metadata": metadata or {},
                    }
                ).encode()
            ).hexdigest()
            connection.executemany(
                """INSERT INTO settings(key, value, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                value=excluded.value, updated_at=excluded.updated_at""",
                [(key, value, now) for key, value in changed.items()],
            )
            cursor = connection.execute(
                """INSERT INTO settings_revisions
                (created_at, change_note, changed_by, changed_keys_json,
                 previous_values_json, values_json, metadata_json, fingerprint)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    now,
                    change_note.strip() or "更新全局设置",
                    changed_by.strip() or "local-operator",
                    _canonical(sorted(changed)),
                    _canonical(previous),
                    _canonical(current),
                    _canonical(metadata or {}),
                    fingerprint,
                ),
            )
            revision_id = int(cursor.lastrowid)
        return self.settings_revision(revision_id)

    def settings_revisions(self, *, limit: int = 30) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM settings_revisions ORDER BY id DESC LIMIT ?",
                (min(max(limit, 1), 200),),
            ).fetchall()
        return [self._settings_revision_record(row) for row in rows]

    def settings_revision(self, revision_id: int) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM settings_revisions WHERE id=?", (revision_id,)
            ).fetchone()
        return self._settings_revision_record(row) if row else None

    @staticmethod
    def _settings_revision_record(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["changed_keys"] = json.loads(item.pop("changed_keys_json"))
        item["previous_values"] = json.loads(item.pop("previous_values_json"))
        item["values"] = json.loads(item.pop("values_json"))
        item["metadata"] = json.loads(item.pop("metadata_json"))
        return item

    def create_research_task(
        self,
        *,
        task_id: str,
        name: str,
        market: str,
        data_path: str,
        data_start: str | None,
        data_end: str | None,
        snapshot_hash: str | None,
        status: str = "DRAFT",
        run_id: str | None = None,
        protocol: dict[str, Any] | None = None,
        protocol_hash: str | None = None,
        protocol_revision: int = 1,
        notes: str = "",
    ) -> dict[str, Any]:
        now = _now()
        with self.connection() as connection:
            connection.execute(
                """INSERT INTO research_tasks
                (task_id, name, market, data_path, data_start, data_end, snapshot_hash,
                 status, run_id, phase, protocol_json, protocol_hash, protocol_revision,
                 notes, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    task_id,
                    name,
                    market,
                    data_path,
                    data_start,
                    data_end,
                    snapshot_hash,
                    status,
                    run_id,
                    "WAITING" if status == "READY" else "CONFIGURE",
                    _canonical(protocol or {}),
                    protocol_hash,
                    protocol_revision,
                    notes,
                    now,
                    now,
                ),
            )
        task = self.research_task(task_id)
        assert task is not None
        return task

    def research_tasks(self) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM research_tasks ORDER BY updated_at DESC, created_at DESC"
            ).fetchall()
        return [self._research_task_record(row) for row in rows]

    def research_task(self, task_id: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM research_tasks WHERE task_id=?", (task_id,)
            ).fetchone()
        return self._research_task_record(row) if row is not None else None

    def research_task_stats(self, task_id: str, run_id: str | None) -> dict[str, int]:
        with self.connection() as connection:
            factor_count = connection.execute(
                "SELECT COUNT(*) FROM factor_pool WHERE source_task_id=?", (task_id,)
            ).fetchone()[0]
            iteration_count = (
                connection.execute(
                    "SELECT COUNT(*) FROM iterations WHERE run_id=?", (run_id,)
                ).fetchone()[0]
                if run_id
                else 0
            )
        return {"factor_count": int(factor_count), "iteration_count": int(iteration_count)}

    def research_task_state(self, task_id: str) -> dict[str, Any]:
        task = self.research_task(task_id)
        if task is None:
            raise KeyError(f"Research task not found: {task_id}")
        return {
            "state": task["status"],
            "phase": task["phase"],
            "run_id": task["run_id"],
            "iteration": int(task["iteration"]),
            "stop_requested": int(task["stop_requested"]),
            "updated_at": task["updated_at"],
            "last_error": task["last_error"],
        }

    def update_research_task_state(self, task_id: str, **values: Any) -> dict[str, Any]:
        allowed = {"state", "phase", "run_id", "iteration", "stop_requested", "last_error"}
        invalid = set(values) - allowed
        if invalid:
            raise ValueError(f"Unknown research-task state fields: {sorted(invalid)}")
        task_values = {
            ("status" if key == "state" else key): value for key, value in values.items()
        }
        self.update_research_task(task_id, **task_values)
        return self.research_task_state(task_id)

    def update_research_task(self, task_id: str, **values: Any) -> dict[str, Any]:
        allowed = {
            "name",
            "market",
            "data_path",
            "data_start",
            "data_end",
            "snapshot_hash",
            "status",
            "run_id",
            "phase",
            "iteration",
            "stop_requested",
            "last_error",
            "protocol_json",
            "protocol_hash",
            "protocol_revision",
            "notes",
        }
        invalid = set(values) - allowed
        if invalid:
            raise ValueError(f"Unknown research-task fields: {sorted(invalid)}")
        if not values:
            task = self.research_task(task_id)
            if task is None:
                raise KeyError(f"Research task not found: {task_id}")
            return task
        values["updated_at"] = _now()
        assignments = ", ".join(f"{key}=?" for key in values)
        with self.connection() as connection:
            cursor = connection.execute(
                f"UPDATE research_tasks SET {assignments} WHERE task_id=?",  # noqa: S608
                (*values.values(), task_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Research task not found: {task_id}")
        task = self.research_task(task_id)
        assert task is not None
        return task

    @staticmethod
    def _research_task_record(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["protocol"] = json.loads(item.pop("protocol_json"))
        return item

    def append_event(
        self,
        category: LogCategory,
        event: str,
        title: str,
        message: str,
        *,
        level: str = "INFO",
        run_id: str | None = None,
        iteration: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body_payload = payload or {}
        timestamp = _now()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            previous = connection.execute(
                "SELECT record_hash FROM events ORDER BY id DESC LIMIT 1"
            ).fetchone()
            previous_hash = str(previous["record_hash"]) if previous else "0" * 64
            body = {
                "timestamp_utc": timestamp,
                "run_id": run_id,
                "iteration": iteration,
                "category": category,
                "level": level,
                "event": event,
                "title": title,
                "message": message,
                "payload": body_payload,
                "previous_hash": previous_hash,
            }
            record_hash = _hash(body)
            cursor = connection.execute(
                """INSERT INTO events
                (timestamp_utc, run_id, iteration, category, level, event, title, message,
                 payload_json, previous_hash, record_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    timestamp,
                    run_id,
                    iteration,
                    category,
                    level,
                    event,
                    title,
                    message,
                    _canonical(body_payload),
                    previous_hash,
                    record_hash,
                ),
            )
            event_id = int(cursor.lastrowid)
        return {"id": event_id, **body, "record_hash": record_hash}

    def events(
        self,
        *,
        after_id: int = 0,
        category: str | None = None,
        run_id: str | None = None,
        task_id: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        clauses = ["id > ?"]
        parameters: list[Any] = [after_id]
        if category and category != "all":
            clauses.append("category = ?")
            parameters.append(category)
        if task_id and run_id:
            clauses.append("(run_id = ? OR json_extract(payload_json, '$.task_id') = ?)")
            parameters.extend((run_id, task_id))
        elif task_id:
            clauses.append("json_extract(payload_json, '$.task_id') = ?")
            parameters.append(task_id)
        elif run_id:
            clauses.append("run_id = ?")
            parameters.append(run_id)
        parameters.append(min(max(limit, 1), 1000))
        with self.connection() as connection:
            rows = connection.execute(
                f"SELECT * FROM events WHERE {' AND '.join(clauses)} ORDER BY id DESC LIMIT ?",  # noqa: S608
                parameters,
            ).fetchall()
        result = []
        for row in reversed(rows):
            item = dict(row)
            item["payload"] = json.loads(item.pop("payload_json"))
            result.append(item)
        return result

    def verify_events(self) -> int:
        previous_hash = "0" * 64
        with self.connection() as connection:
            rows = connection.execute("SELECT * FROM events ORDER BY id").fetchall()
        for row in rows:
            payload = json.loads(row["payload_json"])
            body = {
                "timestamp_utc": row["timestamp_utc"],
                "run_id": row["run_id"],
                "iteration": row["iteration"],
                "category": row["category"],
                "level": row["level"],
                "event": row["event"],
                "title": row["title"],
                "message": row["message"],
                "payload": payload,
                "previous_hash": row["previous_hash"],
            }
            if row["previous_hash"] != previous_hash or row["record_hash"] != _hash(body):
                raise RuntimeError(f"Event hash chain failed at id={row['id']}")
            previous_hash = str(row["record_hash"])
        return len(rows)

    def begin_iteration(self, run_id: str, iteration: int) -> None:
        with self.connection() as connection:
            connection.execute(
                """INSERT INTO iterations(run_id, iteration, status, started_at)
                VALUES (?, ?, 'RUNNING', ?)""",
                (run_id, iteration, _now()),
            )

    def stage_iteration_candidate(
        self,
        run_id: str,
        iteration: int,
        *,
        candidate_id: str,
        proposal: dict[str, Any],
    ) -> None:
        with self.connection() as connection:
            cursor = connection.execute(
                """UPDATE iterations SET candidate_id=?, proposal_json=?
                WHERE run_id=? AND iteration=? AND status='RUNNING'""",
                (candidate_id, _canonical(proposal), run_id, iteration),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Running iteration not found: {run_id}/{iteration}")

    def finish_iteration(
        self,
        run_id: str,
        iteration: int,
        *,
        status: str,
        candidate_id: str | None = None,
        proposal: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        decision: str | None = None,
        error: str | None = None,
    ) -> None:
        with self.connection() as connection:
            connection.execute(
                """UPDATE iterations SET status=?,
                candidate_id=COALESCE(?, candidate_id),
                proposal_json=COALESCE(?, proposal_json),
                metrics_json=COALESCE(?, metrics_json),
                decision=COALESCE(?, decision), error=?, finished_at=?
                WHERE run_id=? AND iteration=?""",
                (
                    status,
                    candidate_id,
                    _canonical(proposal) if proposal else None,
                    _canonical(metrics) if metrics else None,
                    decision,
                    error,
                    _now(),
                    run_id,
                    iteration,
                ),
            )

    def iteration_record(self, run_id: str, iteration: int) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM iterations WHERE run_id=? AND iteration=?",
                (run_id, iteration),
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        for key in ("proposal_json", "metrics_json"):
            result[key.removesuffix("_json")] = (
                json.loads(result[key]) if result[key] is not None else None
            )
            result.pop(key)
        return result

    def iteration_stats(self, *, run_id: str | None = None) -> dict[str, Any]:
        where = "WHERE run_id=?" if run_id else ""
        parameters: tuple[Any, ...] = (run_id,) if run_id else ()
        with self.connection() as connection:
            rows = connection.execute(
                f"SELECT status, COUNT(*) AS count FROM iterations {where} GROUP BY status",  # noqa: S608
                parameters,
            ).fetchall()
        counts = {str(row["status"]): int(row["count"]) for row in rows}
        total = sum(counts.values())
        completed = counts.get("COMPLETED", 0)
        return {
            "total": total,
            "completed": completed,
            "failed": counts.get("FAILED", 0),
            "running": counts.get("RUNNING", 0),
            "success_rate": completed / total if total else 0.0,
        }

    def iteration_history(
        self, limit: int = 100, *, run_id: str | None = None
    ) -> list[dict[str, Any]]:
        where = "WHERE run_id=?" if run_id else ""
        parameters: list[Any] = [run_id] if run_id else []
        parameters.append(min(max(limit, 1), 500))
        with self.connection() as connection:
            rows = connection.execute(
                f"SELECT * FROM iterations {where} ORDER BY id DESC LIMIT ?",  # noqa: S608
                parameters,
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["proposal"] = (
                json.loads(item.pop("proposal_json")) if item["proposal_json"] else None
            )
            item["metrics"] = json.loads(item.pop("metrics_json")) if item["metrics_json"] else None
            result.append(item)
        return result

    def restore_failed_iteration_evidence(
        self,
        run_id: str,
        iteration: int,
        *,
        candidate_id: str,
        proposal: dict[str, Any],
    ) -> bool:
        with self.connection() as connection:
            cursor = connection.execute(
                """UPDATE iterations SET candidate_id=?, proposal_json=?
                WHERE run_id=? AND iteration=? AND status='FAILED'
                AND candidate_id IS NULL AND proposal_json IS NULL""",
                (candidate_id, _canonical(proposal), run_id, iteration),
            )
        return cursor.rowcount == 1

    def reconcile_orphaned_iterations(self) -> list[dict[str, Any]]:
        error = "ServiceRestartInterrupted: iteration was active when the service restarted"
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT run_id, iteration FROM iterations WHERE status='RUNNING' ORDER BY id"
            ).fetchall()
            connection.execute(
                """UPDATE iterations SET status='FAILED', error=?, finished_at=?
                WHERE status='RUNNING'""",
                (error, _now()),
            )
        return [{**dict(row), "error": error} for row in rows]

    def reconcile_orphaned_generation_experiments(self) -> list[dict[str, Any]]:
        now = _now()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """SELECT experiment.candidate_hash, experiment.generation_id,
                experiment.iteration
                FROM generation_experiments AS experiment
                JOIN iterations AS iteration
                  ON iteration.iteration=experiment.iteration
                WHERE experiment.status='RESERVED'
                  AND iteration.status!='RUNNING'
                ORDER BY experiment.created_at"""
            ).fetchall()
            connection.executemany(
                """UPDATE generation_experiments
                SET status='CRASHED', public_verdict='SERVICE_RESTART_INTERRUPTED',
                    updated_at=?
                WHERE candidate_hash=? AND status='RESERVED'""",
                [(now, row["candidate_hash"]) for row in rows],
            )
        return [dict(row) for row in rows]

    def metric_history(
        self, limit: int = 500, *, run_id: str | None = None
    ) -> list[dict[str, Any]]:
        where = "WHERE metrics_json IS NOT NULL"
        parameters: list[Any] = []
        if run_id:
            where += " AND run_id=?"
            parameters.append(run_id)
        parameters.append(min(max(limit, 1), 5000))
        with self.connection() as connection:
            rows = connection.execute(
                f"""SELECT iteration, candidate_id, decision, metrics_json, finished_at
                FROM iterations {where} ORDER BY id DESC LIMIT ?""",  # noqa: S608
                parameters,
            ).fetchall()
        result = []
        for row in reversed(rows):
            result.append(
                {
                    "iteration": row["iteration"],
                    "candidate_id": row["candidate_id"],
                    "decision": row["decision"],
                    "finished_at": row["finished_at"],
                    **json.loads(row["metrics_json"]),
                }
            )
        return result

    def supplement_iteration_metrics(
        self, candidate_id: str, metrics: dict[str, Any]
    ) -> dict[str, Any]:
        """Add newly derived fields without changing the original candidate or decision."""
        with self.connection() as connection:
            row = connection.execute(
                "SELECT metrics_json FROM iterations WHERE candidate_id=?", (candidate_id,)
            ).fetchone()
            if row is None or row["metrics_json"] is None:
                raise KeyError(f"Candidate metrics not found: {candidate_id}")
            merged = {**json.loads(row["metrics_json"]), **metrics}
            connection.execute(
                "UPDATE iterations SET metrics_json=? WHERE candidate_id=?",
                (_canonical(merged), candidate_id),
            )
        return merged

    def candidate_exists(self, candidate_id: str) -> bool:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT 1 FROM iterations WHERE candidate_id=? LIMIT 1", (candidate_id,)
            ).fetchone()
        return row is not None

    def remember(self, run_id: str, iteration: int, kind: str, content: str) -> None:
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        with self.connection() as connection:
            connection.execute(
                """INSERT OR IGNORE INTO memories
                (run_id, iteration, kind, content, content_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (run_id, iteration, kind, content, content_hash, _now()),
            )

    def recent_memories(
        self, limit: int = 20, *, run_id: str | None = None
    ) -> list[dict[str, Any]]:
        where = ""
        parameters: list[Any] = []
        if run_id:
            where = "WHERE run_id=?"
            parameters.append(run_id)
        parameters.append(min(max(limit, 1), 10_000))
        with self.connection() as connection:
            rows = connection.execute(
                f"SELECT * FROM memories {where} ORDER BY id DESC LIMIT ?",  # noqa: S608
                parameters,
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def record_llm_role_artifact(
        self,
        *,
        task_id: str,
        run_id: str,
        iteration: int,
        candidate_id: str | None,
        role: str,
        stage: str,
        status: str,
        artifact: dict[str, Any],
        usage: dict[str, int],
        prompt_hash: str | None,
        response_hash: str | None,
        error: str | None = None,
    ) -> dict[str, Any]:
        now = _now()
        with self.connection() as connection:
            cursor = connection.execute(
                """INSERT INTO llm_role_artifacts
                (task_id, run_id, iteration, candidate_id, role, stage, status,
                 artifact_json, usage_json, prompt_hash, response_hash, error, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    task_id,
                    run_id,
                    iteration,
                    candidate_id,
                    role,
                    stage,
                    status,
                    _canonical(artifact),
                    _canonical(usage),
                    prompt_hash,
                    response_hash,
                    error,
                    now,
                ),
            )
            artifact_id = int(cursor.lastrowid)
        return {
            "id": artifact_id,
            "task_id": task_id,
            "run_id": run_id,
            "iteration": iteration,
            "candidate_id": candidate_id,
            "role": role,
            "stage": stage,
            "status": status,
            "artifact": artifact,
            "usage": usage,
            "prompt_hash": prompt_hash,
            "response_hash": response_hash,
            "error": error,
            "created_at": now,
        }

    def llm_role_artifacts(
        self,
        *,
        task_id: str | None = None,
        run_id: str | None = None,
        candidate_id: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        clauses = []
        parameters: list[Any] = []
        for name, value in (
            ("task_id", task_id),
            ("run_id", run_id),
            ("candidate_id", candidate_id),
        ):
            if value:
                clauses.append(f"{name}=?")
                parameters.append(value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        parameters.append(min(max(limit, 1), 2000))
        with self.connection() as connection:
            rows = connection.execute(
                f"SELECT * FROM llm_role_artifacts {where} ORDER BY id DESC LIMIT ?",  # noqa: S608
                parameters,
            ).fetchall()
        return [self._llm_role_artifact_record(row) for row in rows]

    def llm_role_summary(self, *, task_id: str | None = None) -> dict[str, Any]:
        where = "WHERE task_id=?" if task_id else ""
        parameters: tuple[Any, ...] = (task_id,) if task_id else ()
        with self.connection() as connection:
            rows = connection.execute(
                f"""SELECT role, status, COUNT(*) AS count,
                COALESCE(SUM(json_extract(usage_json, '$.total_tokens')), 0) AS total_tokens,
                MAX(created_at) AS latest_at
                FROM llm_role_artifacts {where}
                GROUP BY role, status ORDER BY role, status""",  # noqa: S608
                parameters,
            ).fetchall()
        roles: dict[str, dict[str, Any]] = {}
        for row in rows:
            role = roles.setdefault(
                str(row["role"]),
                {"completed": 0, "failed": 0, "total_tokens": 0, "latest_at": None},
            )
            status = str(row["status"])
            if status == "COMPLETED":
                role["completed"] += int(row["count"])
            else:
                role["failed"] += int(row["count"])
            role["total_tokens"] += int(row["total_tokens"] or 0)
            role["latest_at"] = max(
                filter(None, (role["latest_at"], row["latest_at"])),
                default=None,
            )
        artifact_count = sum(v["completed"] + v["failed"] for v in roles.values())
        return {"roles": roles, "artifact_count": artifact_count}

    @staticmethod
    def _llm_role_artifact_record(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["artifact"] = json.loads(item.pop("artifact_json"))
        item["usage"] = json.loads(item.pop("usage_json"))
        return item

    def upsert_factor_knowledge(
        self,
        *,
        factor_id: str,
        canonical_mechanism: str,
        mechanism_summary: str,
        tags: list[str],
        review: dict[str, Any],
        falsification: dict[str, Any],
        related_factors: list[dict[str, Any]],
    ) -> dict[str, Any]:
        now = _now()
        with self.connection() as connection:
            if (
                connection.execute(
                    "SELECT 1 FROM factor_pool WHERE factor_id=?", (factor_id,)
                ).fetchone()
                is None
            ):
                raise KeyError(f"Factor not found: {factor_id}")
            connection.execute(
                """INSERT INTO factor_knowledge
                (factor_id, canonical_mechanism, mechanism_summary, tags_json,
                 review_json, falsification_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(factor_id) DO UPDATE SET
                canonical_mechanism=excluded.canonical_mechanism,
                mechanism_summary=excluded.mechanism_summary,
                tags_json=excluded.tags_json,
                review_json=excluded.review_json,
                falsification_json=excluded.falsification_json,
                updated_at=excluded.updated_at""",
                (
                    factor_id,
                    canonical_mechanism,
                    mechanism_summary,
                    _canonical(tags),
                    _canonical(review),
                    _canonical(falsification),
                    now,
                ),
            )
            for relation in related_factors[:20]:
                target = str(relation.get("factor_id", ""))
                if not target or target == factor_id:
                    continue
                if (
                    connection.execute(
                        "SELECT 1 FROM factor_pool WHERE factor_id=?", (target,)
                    ).fetchone()
                    is None
                ):
                    continue
                confidence = _bounded_confidence(relation.get("confidence", 0.0))
                connection.execute(
                    """INSERT INTO factor_knowledge_edges
                    (source_factor_id, target_factor_id, relation, confidence,
                     rationale, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(source_factor_id, target_factor_id, relation) DO UPDATE SET
                    confidence=excluded.confidence,
                    rationale=excluded.rationale""",
                    (
                        factor_id,
                        target,
                        str(relation.get("relation", "RELATED"))[:80],
                        confidence,
                        str(relation.get("rationale", ""))[:2000],
                        now,
                    ),
                )
        knowledge = self.factor_knowledge(factor_id)
        assert knowledge is not None
        return knowledge

    def factor_knowledge(self, factor_id: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM factor_knowledge WHERE factor_id=?", (factor_id,)
            ).fetchone()
            edges = connection.execute(
                """SELECT edge.*, pool.name AS target_name
                FROM factor_knowledge_edges AS edge
                JOIN factor_pool AS pool ON pool.factor_id=edge.target_factor_id
                WHERE edge.source_factor_id=?
                ORDER BY edge.confidence DESC, edge.target_factor_id""",
                (factor_id,),
            ).fetchall()
        if row is None:
            return None
        item = dict(row)
        item["tags"] = json.loads(item.pop("tags_json"))
        item["review"] = json.loads(item.pop("review_json"))
        item["falsification"] = json.loads(item.pop("falsification_json"))
        item["edges"] = [dict(edge) for edge in edges]
        return item

    def factor_knowledge_catalog(
        self, *, task_id: str | None = None, limit: int = 500
    ) -> list[dict[str, Any]]:
        where = "WHERE pool.source_task_id=?" if task_id else ""
        parameters: list[Any] = [task_id] if task_id else []
        parameters.append(min(max(limit, 1), 5000))
        with self.connection() as connection:
            rows = connection.execute(
                f"""SELECT knowledge.*, pool.name, pool.family, pool.source_task_id,
                pool.source_iteration
                FROM factor_knowledge AS knowledge
                JOIN factor_pool AS pool ON pool.factor_id=knowledge.factor_id
                {where} ORDER BY knowledge.updated_at DESC LIMIT ?""",  # noqa: S608
                parameters,
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["tags"] = json.loads(item.pop("tags_json"))
            item["review"] = json.loads(item.pop("review_json"))
            item["falsification"] = json.loads(item.pop("falsification_json"))
            result.append(item)
        return result

    def upsert_factor_pool(
        self,
        *,
        factor_id: str,
        source_iteration: int,
        proposal: dict[str, Any],
        metrics: dict[str, Any],
        status: str,
        status_reason: str,
        source_task_id: str = "legacy-ashare",
    ) -> None:
        now = _now()
        with self.connection() as connection:
            connection.execute(
                """INSERT INTO factor_pool
                (factor_id, source_task_id, source_iteration, name, family, proposal_json,
                 metrics_json,
                 status, status_reason, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(factor_id) DO UPDATE SET
                source_task_id=excluded.source_task_id,
                source_iteration=excluded.source_iteration,
                name=excluded.name,
                family=excluded.family,
                proposal_json=excluded.proposal_json,
                metrics_json=excluded.metrics_json,
                status=excluded.status,
                status_reason=excluded.status_reason,
                updated_at=excluded.updated_at""",
                (
                    factor_id,
                    source_task_id,
                    source_iteration,
                    str(proposal.get("name", factor_id)),
                    str(proposal.get("family", "unknown")),
                    _canonical(proposal),
                    _canonical(metrics),
                    status,
                    status_reason,
                    now,
                    now,
                ),
            )
            initial_state = (
                "SHADOW"
                if status == "ACTIVE"
                else "QUALIFIED"
                if status == "ELIGIBLE"
                else "RESEARCH"
            )
            connection.execute(
                """INSERT INTO factor_lifecycle_events
                (factor_id, previous_state, state, actor, reason, created_at)
                SELECT ?, NULL, ?, 'AUTO_RESEARCH', ?, ?
                WHERE NOT EXISTS (
                    SELECT 1 FROM factor_lifecycle_events WHERE factor_id=?
                )""",
                (factor_id, initial_state, status_reason, now, factor_id),
            )

    def factor_pool(self, *, limit: int = 500) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """SELECT * FROM factor_pool
                ORDER BY source_iteration DESC LIMIT ?""",
                (min(max(limit, 1), 5000),),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["proposal"] = json.loads(item.pop("proposal_json"))
            item["metrics"] = json.loads(item.pop("metrics_json"))
            result.append(item)
        return result

    def factor_pool_count(self) -> int:
        with self.connection() as connection:
            row = connection.execute("SELECT COUNT(*) FROM factor_pool").fetchone()
        return int(row[0])

    def factor_pool_record(self, factor_id: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM factor_pool WHERE factor_id=?", (factor_id,)
            ).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["proposal"] = json.loads(item.pop("proposal_json"))
        item["metrics"] = json.loads(item.pop("metrics_json"))
        return item

    def merge_factor_pool_metrics(self, factor_id: str, metrics: dict[str, Any]) -> dict[str, Any]:
        """Add derived library metadata without replacing frozen evaluation evidence."""
        if not metrics:
            record = self.factor_pool_record(factor_id)
            if record is None:
                raise KeyError(f"Factor not found: {factor_id}")
            return record["metrics"]
        now = _now()
        with self.connection() as connection:
            row = connection.execute(
                "SELECT metrics_json FROM factor_pool WHERE factor_id=?", (factor_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Factor not found: {factor_id}")
            current = json.loads(row["metrics_json"] or "{}")
            merged = {**current, **metrics}
            connection.execute(
                "UPDATE factor_pool SET metrics_json=?, updated_at=? WHERE factor_id=?",
                (_canonical(merged), now, factor_id),
            )
        return merged

    def factor_research_diagnostics(self) -> dict[str, dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """SELECT candidate_id, metrics_json, decision, iteration
                FROM iterations WHERE candidate_id IS NOT NULL AND metrics_json IS NOT NULL
                ORDER BY id"""
            ).fetchall()
        result: dict[str, dict[str, Any]] = {}
        for row in rows:
            result[str(row["candidate_id"])] = {
                "metrics": json.loads(row["metrics_json"]),
                "decision": row["decision"],
                "iteration": row["iteration"],
            }
        return result

    def update_factor_status(self, factor_id: str, status: str, reason: str) -> None:
        with self.connection() as connection:
            connection.execute(
                """UPDATE factor_pool SET status=?, status_reason=?, updated_at=?
                WHERE factor_id=?""",
                (status, reason, _now(), factor_id),
            )

    def apply_factor_reevaluations(self, updates: list[dict[str, Any]]) -> int:
        """Atomically replace public metrics and research status for a frozen factor batch."""
        if not updates:
            raise ValueError("At least one factor reevaluation is required")
        factor_ids = [str(update["factor_id"]) for update in updates]
        if len(factor_ids) != len(set(factor_ids)):
            raise ValueError("Factor reevaluation batch contains duplicate ids")
        allowed_statuses = {"ELIGIBLE", "SCREENED_OUT"}
        invalid = {str(update["status"]) for update in updates} - allowed_statuses
        if invalid:
            raise ValueError(f"Invalid reevaluation statuses: {sorted(invalid)}")
        now = _now()
        with self.connection() as connection:
            placeholders = ",".join("?" for _ in factor_ids)
            existing = {
                str(row["factor_id"])
                for row in connection.execute(
                    f"SELECT factor_id FROM factor_pool WHERE factor_id IN ({placeholders})",  # noqa: S608
                    tuple(factor_ids),
                ).fetchall()
            }
            missing = sorted(set(factor_ids) - existing)
            if missing:
                raise KeyError(f"Unknown factors in reevaluation batch: {missing}")
            connection.executemany(
                """UPDATE factor_pool
                SET metrics_json=?, status=?, status_reason=?, updated_at=?
                WHERE factor_id=?""",
                [
                    (
                        _canonical(update["metrics"]),
                        str(update["status"]),
                        str(update["status_reason"]),
                        now,
                        str(update["factor_id"]),
                    )
                    for update in updates
                ],
            )
        return len(updates)

    def create_manual_backtest(self, request: dict[str, Any]) -> int:
        now = _now()
        with self.connection() as connection:
            cursor = connection.execute(
                """INSERT INTO manual_backtests(status, request_json, created_at, updated_at)
                VALUES ('RUNNING', ?, ?, ?)""",
                (_canonical(request), now, now),
            )
            return int(cursor.lastrowid)

    def record_manual_research_exposures(
        self,
        *,
        backtest_id: int,
        generation_id: str,
        factor_ids: list[str],
        period_start: str,
        period_end: str,
        holdout_start: str,
        holdout_end: str,
    ) -> list[dict[str, Any]]:
        overlaps = period_start <= holdout_end and period_end >= holdout_start
        if overlaps and period_start < holdout_start:
            scope = "MIXED_PUBLIC_AND_HOLDOUT"
        elif overlaps:
            scope = "HOLDOUT_EXPOSED"
        else:
            scope = "PUBLIC_RESEARCH_ONLY"
        now = _now()
        records = []
        with self.connection() as connection:
            for factor_id in factor_ids:
                evidence = {
                    "backtest_id": backtest_id,
                    "generation_id": generation_id,
                    "factor_id": factor_id,
                    "period_start": period_start,
                    "period_end": period_end,
                    "holdout_start": holdout_start,
                    "holdout_end": holdout_end,
                    "visibility_scope": scope,
                    "contaminated": overlaps,
                }
                evidence_hash = hashlib.sha256(_canonical(evidence).encode()).hexdigest()
                connection.execute(
                    """INSERT OR IGNORE INTO manual_research_exposures
                    (backtest_id, generation_id, factor_id, period_start, period_end,
                     visibility_scope, contaminated, evidence_hash, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        backtest_id,
                        generation_id,
                        factor_id,
                        period_start,
                        period_end,
                        scope,
                        int(overlaps),
                        evidence_hash,
                        now,
                    ),
                )
                records.append({**evidence, "evidence_hash": evidence_hash, "created_at": now})
        return records

    def contamination_ledger(
        self, *, generation_id: str | None = None, limit: int = 500
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM manual_research_exposures"
        parameters: tuple[Any, ...]
        if generation_id:
            query += " WHERE generation_id=?"
            parameters = (generation_id, min(max(limit, 1), 5000))
        else:
            parameters = (min(max(limit, 1), 5000),)
        query += " ORDER BY id DESC LIMIT ?"
        with self.connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        result = [dict(row) for row in rows]
        for item in result:
            item["contaminated"] = bool(item["contaminated"])
        return result

    def contaminated_factor_ids(self, generation_id: str | None = None) -> set[str]:
        # A new generation name cannot make an already viewed holdout unseen.
        del generation_id
        with self.connection() as connection:
            rows = connection.execute(
                """SELECT DISTINCT factor_id FROM manual_research_exposures
                WHERE contaminated=1"""
            ).fetchall()
        return {str(row["factor_id"]) for row in rows}

    def portfolio_contamination(
        self, generation_id: str, factor_ids: list[str]
    ) -> list[dict[str, Any]]:
        del generation_id
        if not factor_ids:
            return []
        placeholders = ",".join("?" for _ in factor_ids)
        with self.connection() as connection:
            rows = connection.execute(
                f"""SELECT * FROM manual_research_exposures
                WHERE contaminated=1
                AND factor_id IN ({placeholders}) ORDER BY id DESC""",  # noqa: S608
                tuple(factor_ids),
            ).fetchall()
        return [dict(row) for row in rows]

    def factor_lifecycle_states(self) -> dict[str, dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """SELECT event.* FROM factor_lifecycle_events event
                JOIN (
                    SELECT factor_id, MAX(id) AS latest_id
                    FROM factor_lifecycle_events GROUP BY factor_id
                ) latest ON latest.latest_id=event.id"""
            ).fetchall()
        return {str(row["factor_id"]): dict(row) for row in rows}

    def factor_lifecycle_history(self, factor_id: str) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM factor_lifecycle_events WHERE factor_id=? ORDER BY id",
                (factor_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def transition_factor_lifecycle(
        self, factor_id: str, target: str, *, actor: str, reason: str
    ) -> dict[str, Any]:
        if not actor.strip() or not reason.strip():
            raise ValueError("Lifecycle actor and reason are required")
        history = self.factor_lifecycle_history(factor_id)
        if not history:
            raise KeyError(f"Unknown factor lifecycle: {factor_id}")
        current = FactorState(history[-1]["state"])
        requested = FactorState(target)
        if requested not in ALLOWED_TRANSITIONS[current]:
            raise ValueError(f"Invalid factor transition: {current} -> {requested}")
        now = _now()
        with self.connection() as connection:
            cursor = connection.execute(
                """INSERT INTO factor_lifecycle_events
                (factor_id, previous_state, state, actor, reason, created_at)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (factor_id, current.value, requested.value, actor.strip(), reason.strip(), now),
            )
            event_id = int(cursor.lastrowid)
        return {
            "id": event_id,
            "factor_id": factor_id,
            "previous_state": current.value,
            "state": requested.value,
            "actor": actor.strip(),
            "reason": reason.strip(),
            "created_at": now,
        }

    def complete_manual_backtest(
        self,
        backtest_id: int,
        *,
        metrics: dict[str, Any],
        artifact_path: str,
        result_hash: str,
    ) -> None:
        with self.connection() as connection:
            cursor = connection.execute(
                """UPDATE manual_backtests
                SET status='COMPLETED', metrics_json=?, artifact_path=?, result_hash=?,
                    error=NULL, completed_at=?, updated_at=?
                WHERE id=? AND status='RUNNING'""",
                (
                    _canonical(metrics),
                    artifact_path,
                    result_hash,
                    _now(),
                    _now(),
                    backtest_id,
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Running manual backtest not found: {backtest_id}")

    def fail_manual_backtest(self, backtest_id: int, error: str) -> None:
        with self.connection() as connection:
            cursor = connection.execute(
                """UPDATE manual_backtests
                SET status='FAILED', error=?, completed_at=?, updated_at=?
                WHERE id=? AND status='RUNNING'""",
                (error, _now(), _now(), backtest_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Running manual backtest not found: {backtest_id}")

    def update_manual_backtest_metadata(
        self,
        backtest_id: int,
        *,
        favorite: bool,
        title: str | None,
        notes: str,
        tags: list[str],
    ) -> dict[str, Any]:
        with self.connection() as connection:
            cursor = connection.execute(
                """UPDATE manual_backtests
                SET favorite=?, title=?, notes=?, tags_json=?, updated_at=? WHERE id=?""",
                (
                    int(favorite),
                    title.strip() if title and title.strip() else None,
                    notes.strip(),
                    _canonical(tags),
                    _now(),
                    backtest_id,
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Manual backtest not found: {backtest_id}")
        record = self.manual_backtest(backtest_id)
        assert record is not None
        return record

    def manual_backtests(
        self, *, limit: int = 50, favorite_only: bool = False
    ) -> list[dict[str, Any]]:
        where = " WHERE favorite=1" if favorite_only else ""
        with self.connection() as connection:
            rows = connection.execute(
                f"SELECT * FROM manual_backtests{where} ORDER BY id DESC LIMIT ?",  # noqa: S608
                (min(max(limit, 1), 500),),
            ).fetchall()
        return [self._manual_backtest_record(row) for row in rows]

    def manual_backtest(self, backtest_id: int) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM manual_backtests WHERE id=?", (backtest_id,)
            ).fetchone()
        return self._manual_backtest_record(row) if row is not None else None

    @staticmethod
    def _manual_backtest_record(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["request"] = json.loads(item.pop("request_json"))
        metrics_json = item.pop("metrics_json")
        item["metrics"] = json.loads(metrics_json) if metrics_json else None
        item["favorite"] = bool(item.get("favorite", 0))
        tags_json = item.pop("tags_json", "[]")
        item["tags"] = json.loads(tags_json) if tags_json else []
        return item

    def create_paper_portfolio(
        self, *, name: str, config: dict[str, Any], initial_cash_usd: float
    ) -> dict[str, Any]:
        now = _now()
        with self.connection() as connection:
            cursor = connection.execute(
                """INSERT INTO paper_portfolios
                (name, status, config_json, initial_cash_usd, cash_usd, created_at, updated_at)
                VALUES (?, 'ACTIVE', ?, ?, ?, ?, ?)""",
                (name.strip(), _canonical(config), initial_cash_usd, initial_cash_usd, now, now),
            )
            portfolio_id = int(cursor.lastrowid)
        record = self.paper_portfolio(portfolio_id)
        assert record is not None
        return record

    def paper_portfolios(self, *, limit: int = 100) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """SELECT portfolio.*, nav.trade_date, nav.nav_usd, nav.market_value_usd,
                          nav.gross_exposure, nav.daily_return
                FROM paper_portfolios portfolio
                LEFT JOIN paper_nav nav ON nav.portfolio_id=portfolio.id
                    AND nav.trade_date=(
                        SELECT max(trade_date) FROM paper_nav WHERE portfolio_id=portfolio.id
                    )
                ORDER BY portfolio.id DESC LIMIT ?""",
                (min(max(limit, 1), 500),),
            ).fetchall()
        return [self._paper_portfolio_record(row) for row in rows]

    def paper_portfolio(self, portfolio_id: int) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                """SELECT portfolio.*, nav.trade_date, nav.nav_usd, nav.market_value_usd,
                          nav.gross_exposure, nav.daily_return
                FROM paper_portfolios portfolio
                LEFT JOIN paper_nav nav ON nav.portfolio_id=portfolio.id
                    AND nav.trade_date=(
                        SELECT max(trade_date) FROM paper_nav WHERE portfolio_id=portfolio.id
                    )
                WHERE portfolio.id=?""",
                (portfolio_id,),
            ).fetchone()
            if row is None:
                return None
            positions = connection.execute(
                "SELECT * FROM paper_positions WHERE portfolio_id=? ORDER BY symbol",
                (portfolio_id,),
            ).fetchall()
            trades = connection.execute(
                "SELECT * FROM paper_trades WHERE portfolio_id=? ORDER BY id DESC LIMIT 500",
                (portfolio_id,),
            ).fetchall()
            nav = connection.execute(
                "SELECT * FROM paper_nav WHERE portfolio_id=? ORDER BY trade_date", (portfolio_id,)
            ).fetchall()
        item = self._paper_portfolio_record(row)
        item["positions"] = [dict(position) for position in positions]
        item["trades"] = [self._paper_trade_record(trade) for trade in trades]
        item["nav_history"] = [dict(value) for value in nav]
        return item

    def update_paper_portfolio_status(self, portfolio_id: int, status: str) -> dict[str, Any]:
        if status not in {"ACTIVE", "PAUSED", "CLOSED"}:
            raise ValueError(f"Unsupported paper portfolio status: {status}")
        with self.connection() as connection:
            cursor = connection.execute(
                "UPDATE paper_portfolios SET status=?, updated_at=? WHERE id=?",
                (status, _now(), portfolio_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Paper portfolio not found: {portfolio_id}")
        record = self.paper_portfolio(portfolio_id)
        assert record is not None
        return record

    def delete_paper_portfolio(self, portfolio_id: int) -> None:
        with self.connection() as connection:
            cursor = connection.execute("DELETE FROM paper_portfolios WHERE id=?", (portfolio_id,))
            if cursor.rowcount != 1:
                raise KeyError(f"Paper portfolio not found: {portfolio_id}")

    def apply_paper_portfolio_update(
        self,
        *,
        portfolio_id: int,
        cash_usd: float,
        positions: list[dict[str, Any]],
        trades: list[dict[str, Any]],
        nav: dict[str, Any],
        rebalanced: bool,
    ) -> None:
        now = _now()
        with self.connection() as connection:
            connection.execute("DELETE FROM paper_positions WHERE portfolio_id=?", (portfolio_id,))
            connection.executemany(
                """INSERT INTO paper_positions
                (portfolio_id, symbol, security_name, quantity, average_cost_usd,
                 acquired_trade_date, last_trade_date, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        portfolio_id,
                        position["symbol"],
                        position["security_name"],
                        position["quantity"],
                        position["average_cost_usd"],
                        position.get("acquired_trade_date"),
                        position.get("last_trade_date"),
                        now,
                    )
                    for position in positions
                    if int(position["quantity"]) > 0
                ],
            )
            connection.executemany(
                """INSERT INTO paper_trades
                (portfolio_id, trade_date, symbol, security_name, side, quantity, price_usd,
                 notional_usd, fees_usd, reason, execution_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        portfolio_id,
                        trade["trade_date"],
                        trade["symbol"],
                        trade["security_name"],
                        trade["side"],
                        trade["quantity"],
                        trade["price_usd"],
                        trade["notional_usd"],
                        trade["fees_usd"],
                        trade["reason"],
                        _canonical(trade.get("execution") or {}),
                        now,
                    )
                    for trade in trades
                ],
            )
            previous = connection.execute(
                """SELECT nav_usd FROM paper_nav WHERE portfolio_id=?
                ORDER BY trade_date DESC LIMIT 1""",
                (portfolio_id,),
            ).fetchone()
            daily_return = (
                nav["nav_usd"] / float(previous["nav_usd"]) - 1.0
                if previous and float(previous["nav_usd"])
                else None
            )
            connection.execute(
                """INSERT INTO paper_nav
                (portfolio_id, trade_date, nav_usd, cash_usd, market_value_usd, gross_exposure,
                 daily_return, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(portfolio_id, trade_date) DO UPDATE SET
                nav_usd=excluded.nav_usd, cash_usd=excluded.cash_usd,
                market_value_usd=excluded.market_value_usd, gross_exposure=excluded.gross_exposure,
                daily_return=excluded.daily_return, updated_at=excluded.updated_at""",
                (
                    portfolio_id,
                    nav["trade_date"],
                    nav["nav_usd"],
                    cash_usd,
                    nav["market_value_usd"],
                    nav["gross_exposure"],
                    daily_return,
                    now,
                ),
            )
            connection.execute(
                """UPDATE paper_portfolios SET cash_usd=?, updated_at=?,
                last_rebalanced_date=CASE WHEN ? THEN ? ELSE last_rebalanced_date END WHERE id=?""",
                (cash_usd, now, int(rebalanced), nav["trade_date"], portfolio_id),
            )

    @staticmethod
    def _paper_portfolio_record(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["config"] = json.loads(item.pop("config_json"))
        return item

    @staticmethod
    def _paper_trade_record(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        execution_json = item.pop("execution_json", "{}") or "{}"
        item["execution"] = json.loads(execution_json)
        return item

    def upsert_strategy_experiment_object(
        self,
        *,
        experiment_id: str,
        stage: str,
        object_type: str,
        source_system: str,
        source_id: str,
        title: str,
        status: str,
        market: str | None = None,
        protocol: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        evidence: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        now = _now()
        with self.connection() as connection:
            connection.execute(
                """INSERT INTO strategy_experiment_objects
                (experiment_id, stage, object_type, source_system, source_id, title, status,
                 market, protocol_json, metrics_json, evidence_json, tags_json,
                 created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(experiment_id) DO UPDATE SET
                stage=excluded.stage,
                object_type=excluded.object_type,
                source_system=excluded.source_system,
                source_id=excluded.source_id,
                title=excluded.title,
                status=excluded.status,
                market=excluded.market,
                protocol_json=excluded.protocol_json,
                metrics_json=excluded.metrics_json,
                evidence_json=excluded.evidence_json,
                tags_json=excluded.tags_json,
                updated_at=excluded.updated_at""",
                (
                    experiment_id,
                    stage,
                    object_type,
                    source_system,
                    source_id,
                    title,
                    status,
                    market,
                    _canonical(protocol or {}),
                    _canonical(metrics or {}),
                    _canonical(evidence or {}),
                    _canonical(tags or []),
                    now,
                    now,
                ),
            )
        record = self.strategy_experiment_object(experiment_id)
        assert record is not None
        return record

    def upsert_strategy_experiment_edge(
        self,
        source_experiment_id: str,
        target_experiment_id: str,
        relation: str,
        *,
        evidence: dict[str, Any] | None = None,
    ) -> None:
        now = _now()
        with self.connection() as connection:
            connection.execute(
                """INSERT INTO strategy_experiment_edges
                (source_experiment_id, target_experiment_id, relation, evidence_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(source_experiment_id, target_experiment_id, relation)
                DO UPDATE SET evidence_json=excluded.evidence_json""",
                (
                    source_experiment_id,
                    target_experiment_id,
                    relation,
                    _canonical(evidence or {}),
                    now,
                ),
            )

    def strategy_experiment_object(self, experiment_id: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM strategy_experiment_objects WHERE experiment_id=?",
                (experiment_id,),
            ).fetchone()
        return self._strategy_experiment_record(row) if row is not None else None

    def strategy_experiment_objects(
        self, *, stage: str | None = None, limit: int = 1000
    ) -> list[dict[str, Any]]:
        where = "WHERE stage=?" if stage else ""
        parameters: tuple[Any, ...] = (
            (stage, min(max(limit, 1), 10_000)) if stage else (min(max(limit, 1), 10_000),)
        )
        with self.connection() as connection:
            rows = connection.execute(
                f"""SELECT * FROM strategy_experiment_objects {where}
                ORDER BY updated_at DESC LIMIT ?""",  # noqa: S608
                parameters,
            ).fetchall()
        return [self._strategy_experiment_record(row) for row in rows]

    def strategy_experiment_edges(
        self, *, experiment_id: str | None = None, limit: int = 5000
    ) -> list[dict[str, Any]]:
        where = (
            "WHERE source_experiment_id=? OR target_experiment_id=?" if experiment_id else ""
        )
        parameters: tuple[Any, ...] = (
            (experiment_id, experiment_id, min(max(limit, 1), 20_000))
            if experiment_id
            else (min(max(limit, 1), 20_000),)
        )
        with self.connection() as connection:
            rows = connection.execute(
                f"""SELECT * FROM strategy_experiment_edges {where}
                ORDER BY created_at DESC LIMIT ?""",  # noqa: S608
                parameters,
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["evidence"] = json.loads(item.pop("evidence_json"))
            result.append(item)
        return result

    def strategy_experiment_summary(self) -> dict[str, Any]:
        with self.connection() as connection:
            stage_rows = connection.execute(
                """SELECT stage, COUNT(*) AS count FROM strategy_experiment_objects
                GROUP BY stage ORDER BY stage"""
            ).fetchall()
            status_rows = connection.execute(
                """SELECT status, COUNT(*) AS count FROM strategy_experiment_objects
                GROUP BY status ORDER BY count DESC"""
            ).fetchall()
            edge_count = connection.execute(
                "SELECT COUNT(*) FROM strategy_experiment_edges"
            ).fetchone()
        return {
            "by_stage": {str(row["stage"]): int(row["count"]) for row in stage_rows},
            "by_status": {str(row["status"]): int(row["count"]) for row in status_rows},
            "edge_count": int(edge_count[0] if edge_count else 0),
        }

    @staticmethod
    def _strategy_experiment_record(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["protocol"] = json.loads(item.pop("protocol_json"))
        item["metrics"] = json.loads(item.pop("metrics_json"))
        item["evidence"] = json.loads(item.pop("evidence_json"))
        item["tags"] = json.loads(item.pop("tags_json"))
        return item

    def create_formal_strategy_version(
        self,
        *,
        strategy_uid: str,
        source_experiment_id: str | None,
        name: str,
        market: str,
        lifecycle: str,
        signal_policy: dict[str, Any],
        rebalance_policy: dict[str, Any],
        execution_policy: dict[str, Any],
        risk_policy: dict[str, Any],
        cost_policy: dict[str, Any],
        monitoring_policy: dict[str, Any],
        evidence: dict[str, Any],
    ) -> dict[str, Any]:
        now = _now()
        specification = {
            "strategy_uid": strategy_uid,
            "name": name.strip(),
            "market": market,
            "lifecycle": lifecycle,
            "source_experiment_id": source_experiment_id,
            "signal_policy": signal_policy,
            "rebalance_policy": rebalance_policy,
            "execution_policy": execution_policy,
            "risk_policy": risk_policy,
            "cost_policy": cost_policy,
            "monitoring_policy": monitoring_policy,
            "evidence": evidence,
        }
        specification_hash = hashlib.sha256(_canonical(specification).encode()).hexdigest()
        with self.connection() as connection:
            row = connection.execute(
                """SELECT COALESCE(MAX(version), 0)
                FROM formal_strategy_versions WHERE strategy_uid=?""",
                (strategy_uid,),
            ).fetchone()
            version = int(row[0]) + 1
            connection.execute(
                """INSERT INTO formal_strategy_versions
                (strategy_uid, version, source_experiment_id, name, market, lifecycle,
                 signal_policy_json, rebalance_policy_json, execution_policy_json,
                 risk_policy_json, cost_policy_json, monitoring_policy_json, evidence_json,
                 specification_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    strategy_uid,
                    version,
                    source_experiment_id,
                    name.strip(),
                    market,
                    lifecycle,
                    _canonical(signal_policy),
                    _canonical(rebalance_policy),
                    _canonical(execution_policy),
                    _canonical(risk_policy),
                    _canonical(cost_policy),
                    _canonical(monitoring_policy),
                    _canonical(evidence),
                    specification_hash,
                    now,
                ),
            )
        return self.formal_strategy_version(strategy_uid, version)

    def formal_strategy_versions(
        self, *, lifecycle: str | None = None, limit: int = 500
    ) -> list[dict[str, Any]]:
        where = "WHERE lifecycle=?" if lifecycle else ""
        parameters: tuple[Any, ...] = (
            (lifecycle, min(max(limit, 1), 5000))
            if lifecycle
            else (min(max(limit, 1), 5000),)
        )
        with self.connection() as connection:
            rows = connection.execute(
                f"""SELECT * FROM formal_strategy_versions {where}
                ORDER BY created_at DESC LIMIT ?""",  # noqa: S608
                parameters,
            ).fetchall()
        return [self._formal_strategy_record(row) for row in rows]

    def formal_strategy_version(self, strategy_uid: str, version: int) -> dict[str, Any]:
        with self.connection() as connection:
            row = connection.execute(
                """SELECT * FROM formal_strategy_versions
                WHERE strategy_uid=? AND version=?""",
                (strategy_uid, version),
            ).fetchone()
        if row is None:
            raise KeyError(f"Formal strategy version not found: {strategy_uid} v{version}")
        return self._formal_strategy_record(row)

    def update_formal_strategy_lifecycle(
        self,
        strategy_uid: str,
        version: int,
        *,
        lifecycle: str,
        evidence: dict[str, Any],
    ) -> dict[str, Any]:
        with self.connection() as connection:
            cursor = connection.execute(
                """UPDATE formal_strategy_versions
                SET lifecycle=?, evidence_json=?
                WHERE strategy_uid=? AND version=?""",
                (lifecycle, _canonical(evidence), strategy_uid, int(version)),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Formal strategy version not found: {strategy_uid} v{version}")
        return self.formal_strategy_version(strategy_uid, version)

    @staticmethod
    def _formal_strategy_record(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        for key in (
            "signal_policy",
            "rebalance_policy",
            "execution_policy",
            "risk_policy",
            "cost_policy",
            "monitoring_policy",
            "evidence",
        ):
            item[key] = json.loads(item.pop(f"{key}_json"))
        return item

    def enqueue_system_job(
        self,
        *,
        job_id: str,
        queue: str,
        job_type: str,
        payload: dict[str, Any],
        priority: int = 100,
        resource_group: str = "default",
        max_workers: int = 1,
        progress_total: int = 0,
        max_attempts: int = 3,
    ) -> dict[str, Any]:
        now = _now()
        with self.connection() as connection:
            connection.execute(
                self._system_job_sql.enqueue_job_sql(),
                (
                    job_id,
                    queue,
                    job_type,
                    int(priority),
                    resource_group,
                    max(1, int(max_workers)),
                    max(0, int(progress_total)),
                    _canonical(payload),
                    max(1, int(max_attempts)),
                    now,
                    now,
                ),
            )
            self._insert_system_job_log(
                connection,
                job_id,
                level="INFO",
                event="ENQUEUED",
                message=f"{job_type} queued on {queue}",
                payload={
                    "queue": queue,
                    "job_type": job_type,
                    "priority": int(priority),
                    "resource_group": resource_group,
                    "max_workers": max(1, int(max_workers)),
                },
            )
        return self.system_job(job_id)

    def update_system_job(self, job_id: str, **values: Any) -> dict[str, Any]:
        allowed = {
            "status",
            "progress_current",
            "progress_total",
            "checkpoint",
            "result",
            "error",
            "attempts",
            "started_at",
            "finished_at",
            "lease_owner",
            "lease_expires_at",
            "heartbeat_at",
        }
        invalid = set(values) - allowed
        if invalid:
            raise ValueError(f"Unknown system job fields: {sorted(invalid)}")
        encoded: dict[str, Any] = {}
        for key, value in values.items():
            if key in {"checkpoint", "result"}:
                encoded[f"{key}_json"] = _canonical(value)
            else:
                encoded[key] = value
        encoded["updated_at"] = _now()
        with self.connection() as connection:
            previous = connection.execute(
                self._system_job_sql.select_job_by_id_sql(projection="status"), (job_id,)
            ).fetchone()
            if previous is None:
                raise KeyError(f"System job not found: {job_id}")
            cursor = connection.execute(
                self._system_job_sql.update_job_sql(list(encoded)),
                (*encoded.values(), job_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"System job not found: {job_id}")
            if "status" in values and str(previous["status"]) != str(values["status"]):
                self._insert_system_job_log(
                    connection,
                    job_id,
                    level="ERROR" if str(values["status"]) == "FAILED" else "INFO",
                    event="STATUS_CHANGED",
                    message=f"{previous['status']} -> {values['status']}",
                    payload={
                        "from_status": str(previous["status"]),
                        "to_status": str(values["status"]),
                        "error": values.get("error"),
                    },
                )
        return self.system_job(job_id)

    def claim_system_job(
        self,
        *,
        queue: str,
        worker_id: str,
        lease_seconds: int = 300,
        resource_group: str | None = None,
        max_queue_running: int | None = None,
        max_global_running: int | None = None,
    ) -> dict[str, Any] | None:
        now = _now()
        lease_expires_at = (
            datetime.now(UTC) + timedelta(seconds=max(30, int(lease_seconds)))
        ).isoformat()
        clauses = [
            "queue=?",
            "(status='QUEUED' OR (status='RUNNING' AND lease_expires_at IS NOT NULL "
            "AND lease_expires_at < ?))",
            "attempts < max_attempts",
        ]
        parameters: list[Any] = [queue, now]
        if resource_group:
            clauses.append("resource_group=?")
            parameters.append(resource_group)
        where = " AND ".join(clauses)
        claimed_job_id: str | None = None
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                self._system_job_sql.claim_candidates_sql(where),
                tuple(parameters),
            ).fetchall()
            for row in rows:
                job_id = str(row["job_id"])
                group = str(row["resource_group"])
                global_active = connection.execute(
                    self._system_job_sql.active_global_sql(),
                    (job_id, now),
                ).fetchone()
                if max_global_running is not None and int(
                    global_active[0] if global_active else 0
                ) >= max(1, int(max_global_running)):
                    self._insert_system_job_log(
                        connection,
                        job_id,
                        level="DEBUG",
                        event="CLAIM_DEFERRED_GLOBAL_CAPACITY",
                        message="Global running job capacity is exhausted",
                        payload={
                            "worker_id": worker_id,
                            "max_global_running": max(1, int(max_global_running)),
                            "active_global": int(global_active[0] if global_active else 0),
                        },
                    )
                    continue
                queue_active = connection.execute(
                    self._system_job_sql.active_queue_sql(),
                    (queue, job_id, now),
                ).fetchone()
                if max_queue_running is not None and int(
                    queue_active[0] if queue_active else 0
                ) >= max(1, int(max_queue_running)):
                    self._insert_system_job_log(
                        connection,
                        job_id,
                        level="DEBUG",
                        event="CLAIM_DEFERRED_QUEUE_CAPACITY",
                        message="Queue running job capacity is exhausted",
                        payload={
                            "worker_id": worker_id,
                            "queue": queue,
                            "max_queue_running": max(1, int(max_queue_running)),
                            "active_queue": int(queue_active[0] if queue_active else 0),
                        },
                    )
                    continue
                capacity_row = connection.execute(
                    self._system_job_sql.resource_capacity_sql(),
                    (queue, group),
                ).fetchone()
                capacity = max(
                    1,
                    int(
                        capacity_row[0]
                        if capacity_row and capacity_row[0] is not None
                        else row["max_workers"]
                    ),
                )
                active = connection.execute(
                    self._system_job_sql.active_resource_sql(),
                    (queue, group, job_id, now),
                ).fetchone()
                if int(active[0] if active else 0) >= capacity:
                    self._insert_system_job_log(
                        connection,
                        job_id,
                        level="DEBUG",
                        event="CLAIM_DEFERRED_RESOURCE_CAPACITY",
                        message="Resource group running job capacity is exhausted",
                        payload={
                            "worker_id": worker_id,
                            "queue": queue,
                            "resource_group": group,
                            "capacity": capacity,
                            "active_resource": int(active[0] if active else 0),
                        },
                    )
                    continue
                connection.execute(
                    self._system_job_sql.claim_update_sql(),
                    (worker_id, lease_expires_at, now, now, now, job_id),
                )
                self._insert_system_job_log(
                    connection,
                    job_id,
                    level="INFO",
                    event="CLAIMED",
                    message=f"Lease acquired by {worker_id}",
                    payload={
                        "worker_id": worker_id,
                        "lease_expires_at": lease_expires_at,
                        "attempt": int(row["attempts"]) + 1,
                    },
                )
                claimed_job_id = job_id
                break
        return self.system_job(claimed_job_id) if claimed_job_id else None

    def heartbeat_system_job(
        self,
        job_id: str,
        *,
        worker_id: str,
        lease_seconds: int = 300,
        progress_current: int | None = None,
        checkpoint: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = _now()
        lease_expires_at = (
            datetime.now(UTC) + timedelta(seconds=max(30, int(lease_seconds)))
        ).isoformat()
        values: dict[str, Any] = {
            "heartbeat_at": now,
            "lease_expires_at": lease_expires_at,
        }
        if progress_current is not None:
            values["progress_current"] = max(0, int(progress_current))
        if checkpoint is not None:
            values["checkpoint"] = checkpoint
        with self.connection() as connection:
            row = connection.execute(
                "SELECT lease_owner, status FROM system_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"System job not found: {job_id}")
            if row["status"] != "RUNNING" or row["lease_owner"] != worker_id:
                raise RuntimeError("System job lease is not held by this worker")
        updated = self.update_system_job(job_id, **values)
        self.append_system_job_log(
            job_id,
            level="DEBUG",
            event="HEARTBEAT",
            message=f"Heartbeat from {worker_id}",
            payload={
                "worker_id": worker_id,
                "lease_expires_at": lease_expires_at,
                "progress_current": progress_current,
                "checkpoint_keys": sorted((checkpoint or {}).keys()),
            },
        )
        return updated

    def command_system_job(
        self,
        job_id: str,
        *,
        command: str,
        actor: str = "local-operator",
        reason: str = "",
    ) -> dict[str, Any]:
        command_key = command.strip().lower()
        if command_key not in {"cancel", "pause", "resume"}:
            raise ValueError(f"Unsupported system job command: {command}")
        now = _now()
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM system_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"System job not found: {job_id}")
            status = str(row["status"])
            terminal = {"COMPLETED", "FAILED", "CANCELLED", "BLOCKED_UNSUPPORTED"}
            if status in terminal:
                raise RuntimeError(f"System job is terminal: {status}")

            if command_key == "cancel":
                if status in {"QUEUED", "PAUSED"}:
                    next_status = "CANCELLED"
                    lease_owner = None
                    lease_expires_at = None
                    heartbeat_at = None
                    finished_at = now
                elif status in {"RUNNING", "PAUSE_REQUESTED"}:
                    next_status = "CANCEL_REQUESTED"
                    lease_owner = row["lease_owner"]
                    lease_expires_at = row["lease_expires_at"]
                    heartbeat_at = row["heartbeat_at"]
                    finished_at = None
                else:
                    raise RuntimeError(f"System job cannot be cancelled from {status}")
            elif command_key == "pause":
                if status == "QUEUED":
                    next_status = "PAUSED"
                    lease_owner = None
                    lease_expires_at = None
                    heartbeat_at = None
                    finished_at = None
                elif status == "RUNNING":
                    next_status = "PAUSE_REQUESTED"
                    lease_owner = row["lease_owner"]
                    lease_expires_at = row["lease_expires_at"]
                    heartbeat_at = row["heartbeat_at"]
                    finished_at = None
                else:
                    raise RuntimeError(f"System job cannot be paused from {status}")
            else:
                if status != "PAUSED":
                    raise RuntimeError(f"System job cannot be resumed from {status}")
                next_status = "QUEUED"
                lease_owner = None
                lease_expires_at = None
                heartbeat_at = None
                finished_at = None

            error = reason.strip() if reason.strip() else f"{command_key} by {actor}"
            connection.execute(
                """UPDATE system_jobs
                SET status=?, error=?, lease_owner=?, lease_expires_at=?,
                    heartbeat_at=?, finished_at=?, updated_at=?
                WHERE job_id=?""",
                (
                    next_status,
                    error,
                    lease_owner,
                    lease_expires_at,
                    heartbeat_at,
                    finished_at,
                    now,
                    job_id,
                ),
            )
            self._insert_system_job_log(
                connection,
                job_id,
                level="WARNING" if command_key == "cancel" else "INFO",
                event=f"COMMAND_{command_key.upper()}",
                message=f"{actor} requested {command_key}: {status} -> {next_status}",
                payload={
                    "actor": actor,
                    "command": command_key,
                    "reason": reason,
                    "from_status": status,
                    "to_status": next_status,
                },
            )
        return self.system_job(job_id)

    def recover_expired_system_jobs(self, *, queue: str | None = None) -> int:
        now = _now()
        clauses = ["status='RUNNING'", "lease_expires_at IS NOT NULL", "lease_expires_at < ?"]
        parameters: list[Any] = [now]
        if queue:
            clauses.append("queue=?")
            parameters.append(queue)
        where = " AND ".join(clauses)
        with self.connection() as connection:
            expired = connection.execute(
                self._system_job_sql.recover_expired_select_sql(where),
                tuple(parameters),
            ).fetchall()
            cursor = connection.execute(
                self._system_job_sql.recover_expired_update_sql(where),
                (now, *parameters),
            )
            for row in expired:
                exhausted = int(row["attempts"]) >= int(row["max_attempts"])
                self._insert_system_job_log(
                    connection,
                    str(row["job_id"]),
                    level="ERROR" if exhausted else "WARNING",
                    event="LEASE_EXPIRED_RECOVERED",
                    message="Lease expired; job failed"
                    if exhausted
                    else "Lease expired; job returned to queue",
                    payload={
                        "attempts": int(row["attempts"]),
                        "max_attempts": int(row["max_attempts"]),
                        "recovered_status": "FAILED" if exhausted else "QUEUED",
                    },
                )
            return int(cursor.rowcount)

    def upsert_materialized_snapshot(
        self,
        key: str,
        payload: dict[str, Any],
        *,
        fingerprint: str | None = None,
        ttl_seconds: int | None = None,
        source: str = "unknown",
        status: str = "READY",
    ) -> dict[str, Any]:
        now = _now()
        expires_at = (
            (datetime.now(UTC) + timedelta(seconds=max(1, int(ttl_seconds)))).isoformat()
            if ttl_seconds is not None
            else None
        )
        encoded = _canonical(payload)
        digest = fingerprint or hashlib.sha256(encoded.encode()).hexdigest()
        with self.connection() as connection:
            connection.execute(
                """INSERT INTO materialized_snapshots
                (key, payload_json, fingerprint, source, status, expires_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                payload_json=excluded.payload_json,
                fingerprint=excluded.fingerprint,
                source=excluded.source,
                status=excluded.status,
                expires_at=excluded.expires_at,
                updated_at=excluded.updated_at""",
                (key, encoded, digest, source, status, expires_at, now),
            )
        snapshot = self.materialized_snapshot(key)
        assert snapshot is not None
        return snapshot

    def materialized_snapshot(self, key: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM materialized_snapshots WHERE key=?", (key,)
            ).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["payload"] = json.loads(item.pop("payload_json"))
        item["cache_state"] = _materialized_cache_state(item)
        return item

    def materialized_snapshot_summary(self) -> dict[str, Any]:
        with self.connection() as connection:
            rows = connection.execute(
                """SELECT key, source, status, expires_at, updated_at
                FROM materialized_snapshots ORDER BY key"""
            ).fetchall()
        snapshots = []
        states: dict[str, int] = {}
        for row in rows:
            item = dict(row)
            cache_state = _materialized_cache_state(item)
            state = str(cache_state["status"])
            states[state] = states.get(state, 0) + 1
            snapshots.append(
                {
                    "key": item["key"],
                    "source": item.get("source") or "unknown",
                    "status": item.get("status") or "READY",
                    "updated_at": item.get("updated_at"),
                    "expires_at": item.get("expires_at"),
                    "cache_state": cache_state,
                }
            )
        stale = [
            item
            for item in snapshots
            if item["cache_state"].get("status") == "STALE"
        ]
        no_ttl = [
            item
            for item in snapshots
            if item["cache_state"].get("status") == "NO_TTL"
        ]
        return {
            "protocol": "AUTOALPHA_MATERIALIZED_SNAPSHOT_SUMMARY_V1",
            "total": len(snapshots),
            "states": states,
            "stale_count": len(stale),
            "no_ttl_count": len(no_ttl),
            "stale_keys": [str(item["key"]) for item in stale],
            "no_ttl_keys": [str(item["key"]) for item in no_ttl],
            "snapshots": snapshots,
        }

    def append_system_job_log(
        self,
        job_id: str,
        *,
        level: str,
        event: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self.connection() as connection:
            exists = connection.execute(
                self._system_job_sql.select_job_by_id_sql(projection="1"), (job_id,)
            ).fetchone()
            if exists is None:
                raise KeyError(f"System job not found: {job_id}")
            log_id = self._insert_system_job_log(
                connection,
                job_id,
                level=level,
                event=event,
                message=message,
                payload=payload or {},
            )
        return self.system_job_log(log_id)

    def system_job_log(self, log_id: int) -> dict[str, Any]:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM system_job_logs WHERE id=?", (int(log_id),)
            ).fetchone()
        if row is None:
            raise KeyError(f"System job log not found: {log_id}")
        return self._system_job_log_record(row)

    def system_job_logs(self, job_id: str, *, limit: int = 200) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """SELECT * FROM system_job_logs WHERE job_id=?
                ORDER BY id DESC LIMIT ?""",
                (job_id, min(max(int(limit), 1), 2000)),
            ).fetchall()
        return [self._system_job_log_record(row) for row in rows]

    def system_job_logs_for_jobs(
        self, job_ids: list[str], *, limit_per_job: int = 3
    ) -> dict[str, list[dict[str, Any]]]:
        if not job_ids:
            return {}
        result: dict[str, list[dict[str, Any]]] = {job_id: [] for job_id in job_ids}
        with self.connection() as connection:
            rows = connection.execute(
                self._system_job_sql.logs_for_jobs_sql(len(job_ids)),
                tuple(job_ids),
            ).fetchall()
        for row in rows:
            job_id = str(row["job_id"])
            if len(result.setdefault(job_id, [])) < max(1, int(limit_per_job)):
                result[job_id].append(self._system_job_log_record(row))
        return result

    def system_job(self, job_id: str) -> dict[str, Any]:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM system_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"System job not found: {job_id}")
        return self._system_job_record(row)

    def system_jobs(
        self, *, queue: str | None = None, status: str | None = None, limit: int = 500
    ) -> list[dict[str, Any]]:
        clauses = []
        parameters: list[Any] = []
        if queue:
            clauses.append(f"queue={self._system_job_sql.placeholder(len(parameters) + 1)}")
            parameters.append(queue)
        if status:
            clauses.append(f"status={self._system_job_sql.placeholder(len(parameters) + 1)}")
            parameters.append(status)
        parameters.append(min(max(limit, 1), 5000))
        with self.connection() as connection:
            rows = connection.execute(
                self._system_job_sql.list_jobs_sql(
                    queue=bool(queue),
                    status=bool(status),
                    limit_index=len(parameters),
                ),
                tuple(parameters),
            ).fetchall()
        return [self._system_job_record(row) for row in rows]

    def system_job_summary(self) -> dict[str, Any]:
        now = _now()
        with self.connection() as connection:
            rows = connection.execute(
                """SELECT queue, status, COUNT(*) AS count FROM system_jobs
                GROUP BY queue, status ORDER BY queue, status"""
            ).fetchall()
            resource_rows = connection.execute(
                """SELECT queue, resource_group, status, COUNT(*) AS count,
                          MIN(max_workers) AS capacity
                FROM system_jobs
                GROUP BY queue, resource_group, status
                ORDER BY queue, resource_group, status"""
            ).fetchall()
            expired_rows = connection.execute(
                """SELECT queue, resource_group, COUNT(*) AS count FROM system_jobs
                WHERE status='RUNNING' AND lease_expires_at IS NOT NULL
                AND lease_expires_at < ?
                GROUP BY queue, resource_group
                ORDER BY queue, resource_group""",
                (now,),
            ).fetchall()
        queues: dict[str, dict[str, int]] = {}
        for row in rows:
            queue = queues.setdefault(str(row["queue"]), {})
            queue[str(row["status"])] = int(row["count"])
        resources: dict[str, dict[str, dict[str, Any]]] = {}
        for row in resource_rows:
            queue = resources.setdefault(str(row["queue"]), {})
            group = queue.setdefault(
                str(row["resource_group"]),
                {"statuses": {}, "capacity": int(row["capacity"] or 1)},
            )
            group["statuses"][str(row["status"])] = int(row["count"])
            group["capacity"] = min(int(group["capacity"]), int(row["capacity"] or 1))
        resource_utilization = []
        for queue_name, groups in sorted(resources.items()):
            for resource_group, detail in sorted(groups.items()):
                statuses = dict(detail.get("statuses") or {})
                running = int(statuses.get("RUNNING", 0))
                queued = int(statuses.get("QUEUED", 0))
                capacity = max(1, int(detail.get("capacity") or 1))
                utilization = running / capacity
                resource_utilization.append(
                    {
                        "queue": queue_name,
                        "resource_group": resource_group,
                        "capacity": capacity,
                        "running": running,
                        "queued": queued,
                        "paused": int(statuses.get("PAUSED", 0)),
                        "failed": int(statuses.get("FAILED", 0))
                        + int(statuses.get("BLOCKED_UNSUPPORTED", 0)),
                        "terminal": int(statuses.get("COMPLETED", 0))
                        + int(statuses.get("CANCELLED", 0)),
                        "utilization": utilization,
                        "saturated": running >= capacity and queued > 0,
                        "idle_capacity": max(0, capacity - running),
                    }
                )
        expired_running = [
            {
                "queue": str(row["queue"]),
                "resource_group": str(row["resource_group"]),
                "count": int(row["count"]),
            }
            for row in expired_rows
        ]
        return {
            "queues": queues,
            "resources": resources,
            "resource_utilization": resource_utilization,
            "saturated_resources": [
                item for item in resource_utilization if item["saturated"]
            ],
            "expired_running": expired_running,
            "expired_running_count": sum(item["count"] for item in expired_running),
            "total": sum(sum(item.values()) for item in queues.values()),
        }

    @staticmethod
    def _system_job_record(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["payload"] = json.loads(item.pop("payload_json"))
        item["checkpoint"] = json.loads(item.pop("checkpoint_json"))
        item["result"] = json.loads(item.pop("result_json"))
        return item

    @staticmethod
    def _system_job_log_record(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["payload"] = json.loads(item.pop("payload_json"))
        return item

    @staticmethod
    def _insert_system_job_log(
        connection: sqlite3.Connection,
        job_id: str,
        *,
        level: str,
        event: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> int:
        cursor = connection.execute(
            system_job_sql("sqlite").insert_log_sql(),
            (
                job_id,
                _now(),
                level.upper(),
                event,
                message,
                _canonical(payload or {}),
            ),
        )
        return int(cursor.lastrowid)

    def record_portfolio_decision(
        self,
        *,
        run_id: str,
        iteration: int,
        action: str,
        candidate_id: str | None,
        removed_factor_id: str | None,
        accepted: bool,
        reason: str,
        metrics: dict[str, Any],
        members: list[tuple[str, float]],
    ) -> int:
        with self.connection() as connection:
            cursor = connection.execute(
                """INSERT INTO portfolio_versions
                (run_id, iteration, action, candidate_id, removed_factor_id, accepted,
                 reason, metrics_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    iteration,
                    action,
                    candidate_id,
                    removed_factor_id,
                    int(accepted),
                    reason,
                    _canonical(metrics),
                    _now(),
                ),
            )
            version_id = int(cursor.lastrowid)
            connection.executemany(
                """INSERT INTO portfolio_members(version_id, factor_id, weight)
                VALUES (?, ?, ?)""",
                [(version_id, factor_id, weight) for factor_id, weight in members],
            )
        return version_id

    def portfolio_history(
        self, *, limit: int = 100, run_id: str | None = None
    ) -> list[dict[str, Any]]:
        where = "WHERE run_id=?" if run_id else ""
        parameters: list[Any] = [run_id] if run_id else []
        parameters.append(min(max(limit, 1), 1000))
        with self.connection() as connection:
            versions = connection.execute(
                f"SELECT * FROM portfolio_versions {where} ORDER BY id DESC LIMIT ?",  # noqa: S608
                parameters,
            ).fetchall()
            result = []
            for version in versions:
                item = dict(version)
                members = connection.execute(
                    """SELECT pm.factor_id, pm.weight, fp.name, fp.family,
                    fp.source_iteration FROM portfolio_members pm
                    JOIN factor_pool fp ON fp.factor_id=pm.factor_id
                    WHERE pm.version_id=? ORDER BY pm.weight DESC, pm.factor_id""",
                    (item["id"],),
                ).fetchall()
                item["accepted"] = bool(item["accepted"])
                item["metrics"] = json.loads(item.pop("metrics_json"))
                item["members"] = [dict(member) for member in members]
                result.append(item)
        return result

    def active_portfolio(self, *, run_id: str | None = None) -> dict[str, Any] | None:
        where = "WHERE accepted=1"
        parameters: tuple[Any, ...] = ()
        if run_id:
            where += " AND run_id=?"
            parameters = (run_id,)
        with self.connection() as connection:
            row = connection.execute(
                f"""SELECT * FROM portfolio_versions
                {where} ORDER BY id DESC LIMIT 1""",  # noqa: S608
                parameters,
            ).fetchone()
        if row is None:
            return None
        version_id = int(row["id"])
        with self.connection() as connection:
            members = connection.execute(
                """SELECT pm.factor_id, pm.weight, fp.name, fp.family,
                fp.source_iteration FROM portfolio_members pm
                JOIN factor_pool fp ON fp.factor_id=pm.factor_id
                WHERE pm.version_id=? ORDER BY pm.weight DESC, pm.factor_id""",
                (version_id,),
            ).fetchall()
        item = dict(row)
        item["accepted"] = True
        item["metrics"] = json.loads(item.pop("metrics_json"))
        item["members"] = [dict(member) for member in members]
        return item

    def ensure_generation(
        self,
        *,
        generation_id: str,
        protocol_version: str,
        maximum_candidates: int,
        maximum_holdout_attempts: int,
    ) -> dict[str, Any]:
        with self.connection() as connection:
            connection.execute(
                """INSERT OR IGNORE INTO research_generations
                (generation_id, protocol_version, status, candidate_attempts, holdout_attempts,
                 maximum_candidates, maximum_holdout_attempts, started_at)
                VALUES (?, ?, 'ACTIVE', 0, 0, ?, ?, ?)""",
                (
                    generation_id,
                    protocol_version,
                    maximum_candidates,
                    maximum_holdout_attempts,
                    _now(),
                ),
            )
            row = connection.execute(
                "SELECT * FROM research_generations WHERE generation_id=?", (generation_id,)
            ).fetchone()
        assert row is not None
        result = dict(row)
        if result["protocol_version"] != protocol_version:
            raise RuntimeError("Generation protocol version is immutable")
        return result

    def generation_state(self, generation_id: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM research_generations WHERE generation_id=?", (generation_id,)
            ).fetchone()
        return dict(row) if row else None

    def ensure_running_generation(
        self,
        *,
        base_generation_id: str,
        protocol_version: str,
        maximum_candidates: int,
        maximum_holdout_attempts: int,
    ) -> dict[str, Any]:
        with self.connection() as connection:
            task_scope = (
                base_generation_id.split("--", 1)[1] if "--" in base_generation_id else None
            )
            scope_clause = ""
            scope_parameters: tuple[Any, ...] = ()
            if task_scope:
                scope_clause = "AND (generation_id LIKE ? OR generation_id LIKE ?)"
                scope_parameters = (f"%--{task_scope}", f"%--{task_scope}-public-%")
            else:
                scope_clause = "AND generation_id NOT LIKE '%--task-%'"
            connection.execute(
                f"""UPDATE research_generations SET status='SEALED', sealed_at=?
                WHERE status='ACTIVE' {scope_clause}
                AND NOT (generation_id=? OR generation_id LIKE ?)""",  # noqa: S608
                (
                    _now(),
                    *scope_parameters,
                    base_generation_id,
                    f"{base_generation_id}-public-%",
                ),
            )
            rows = connection.execute(
                """SELECT * FROM research_generations
                WHERE generation_id=? OR generation_id LIKE ?
                ORDER BY started_at DESC""",
                (base_generation_id, f"{base_generation_id}-public-%"),
            ).fetchall()
        current = dict(rows[0]) if rows else None
        if current is None:
            return self.ensure_generation(
                generation_id=base_generation_id,
                protocol_version=protocol_version,
                maximum_candidates=maximum_candidates,
                maximum_holdout_attempts=maximum_holdout_attempts,
            )
        root = next(
            (dict(row) for row in rows if row["generation_id"] == base_generation_id),
            None,
        )
        total_candidate_attempts = sum(int(row["candidate_attempts"]) for row in rows)
        total_holdout_attempts = sum(int(row["holdout_attempts"]) for row in rows)
        if (
            current["status"] == "SEALED"
            and root is not None
            and total_candidate_attempts < int(root["maximum_candidates"])
            and total_holdout_attempts < int(root["maximum_holdout_attempts"])
        ):
            recovered_candidate_limit = int(current["candidate_attempts"]) + (
                int(root["maximum_candidates"]) - total_candidate_attempts
            )
            recovered_holdout_limit = int(current["holdout_attempts"]) + (
                int(root["maximum_holdout_attempts"]) - total_holdout_attempts
            )
            with self.connection() as connection:
                connection.execute(
                    """UPDATE research_generations
                    SET status='ACTIVE', sealed_at=NULL, maximum_candidates=?,
                        maximum_holdout_attempts=?
                    WHERE generation_id=? AND status='SEALED'""",
                    (
                        recovered_candidate_limit,
                        recovered_holdout_limit,
                        current["generation_id"],
                    ),
                )
                recovered = connection.execute(
                    "SELECT * FROM research_generations WHERE generation_id=?",
                    (current["generation_id"],),
                ).fetchone()
            assert recovered is not None
            return {**dict(recovered), "recovered_cross_scope_seal": True}
        if (
            current["status"] == "ACTIVE"
            and current["candidate_attempts"] < current["maximum_candidates"]
        ):
            return current

        suffix = len(rows) + 1
        next_id = f"{base_generation_id}-public-{suffix:03d}"
        with self.connection() as connection:
            connection.execute(
                """UPDATE research_generations SET status='SEALED', sealed_at=?
                WHERE generation_id=? AND status='ACTIVE'""",
                (_now(), current["generation_id"]),
            )
        return self.ensure_generation(
            generation_id=next_id,
            protocol_version=protocol_version,
            maximum_candidates=maximum_candidates,
            maximum_holdout_attempts=0,
        )

    def latest_generation(self, base_generation_id: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                """SELECT * FROM research_generations
                WHERE generation_id=? OR generation_id LIKE ?
                ORDER BY started_at DESC LIMIT 1""",
                (base_generation_id, f"{base_generation_id}-public-%"),
            ).fetchone()
        return dict(row) if row else None

    def start_direction_campaign(
        self,
        *,
        generation_id: str,
        direction: str,
        title: str,
        objective: str,
        diagnostic_score: float,
        rationale: list[str],
        evidence: dict[str, Any],
        baseline: dict[str, Any],
        maximum_attempts: int,
        started_iteration: int,
    ) -> dict[str, Any]:
        if maximum_attempts <= 0:
            raise ValueError("Direction campaign maximum_attempts must be positive")
        now = _now()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            active = connection.execute(
                """SELECT id FROM direction_campaigns
                WHERE generation_id=? AND status='ACTIVE'""",
                (generation_id,),
            ).fetchone()
            if active is not None:
                raise RuntimeError("An active direction campaign already exists")
            cursor = connection.execute(
                """INSERT INTO direction_campaigns
                (generation_id, direction, title, objective, status, diagnostic_score,
                 rationale_json, evidence_json, baseline_json, maximum_attempts,
                 started_iteration, created_at, updated_at)
                VALUES (?, ?, ?, ?, 'ACTIVE', ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    generation_id,
                    direction,
                    title,
                    objective,
                    diagnostic_score,
                    _canonical(rationale),
                    _canonical(evidence),
                    _canonical(baseline),
                    maximum_attempts,
                    started_iteration,
                    now,
                    now,
                ),
            )
            campaign_id = int(cursor.lastrowid)
        campaign = self.direction_campaign(campaign_id)
        assert campaign is not None
        return campaign

    def direction_campaign(self, campaign_id: int) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM direction_campaigns WHERE id=?", (campaign_id,)
            ).fetchone()
            return self._direction_campaign_record(connection, row) if row else None

    def active_direction_campaign(self, generation_id: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                """SELECT * FROM direction_campaigns
                WHERE generation_id=? AND status='ACTIVE' ORDER BY id DESC LIMIT 1""",
                (generation_id,),
            ).fetchone()
            return self._direction_campaign_record(connection, row) if row else None

    def direction_campaign_history(
        self, generation_id: str, *, limit: int = 20
    ) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """SELECT * FROM direction_campaigns
                WHERE generation_id=? ORDER BY id DESC LIMIT ?""",
                (generation_id, min(max(limit, 1), 500)),
            ).fetchall()
            return [self._direction_campaign_record(connection, row) for row in rows]

    def reserve_direction_attempt(
        self,
        *,
        campaign_id: int,
        iteration: int,
        baseline: dict[str, Any],
        run_id: str | None = None,
    ) -> dict[str, Any]:
        now = _now()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            campaign = connection.execute(
                "SELECT * FROM direction_campaigns WHERE id=?", (campaign_id,)
            ).fetchone()
            if campaign is None or campaign["status"] != "ACTIVE":
                raise RuntimeError("Direction campaign is not active")
            if int(campaign["attempts_used"]) >= int(campaign["maximum_attempts"]):
                raise PermissionError("Direction campaign attempt budget is exhausted")
            cursor = connection.execute(
                """INSERT INTO direction_attempts
                (campaign_id, run_id, iteration, status, baseline_json, diagnostics_json,
                 created_at, updated_at)
                VALUES (?, ?, ?, 'RESERVED', ?, '{}', ?, ?)""",
                (campaign_id, run_id, iteration, _canonical(baseline), now, now),
            )
            connection.execute(
                """UPDATE direction_campaigns
                SET attempts_used=attempts_used+1, last_iteration=?, updated_at=?
                WHERE id=?""",
                (iteration, now, campaign_id),
            )
            attempt_id = int(cursor.lastrowid)
        attempt = self.direction_attempt(iteration, run_id=run_id)
        assert attempt is not None and int(attempt["id"]) == attempt_id
        return attempt

    def direction_attempt(
        self, iteration: int, *, run_id: str | None = None
    ) -> dict[str, Any] | None:
        where = "iteration=?"
        parameters: tuple[Any, ...] = (iteration,)
        if run_id:
            where += " AND run_id=?"
            parameters = (iteration, run_id)
        with self.connection() as connection:
            row = connection.execute(
                f"SELECT * FROM direction_attempts WHERE {where} ORDER BY id DESC LIMIT 1",  # noqa: S608
                parameters,
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["baseline"] = json.loads(result.pop("baseline_json"))
        result["diagnostics"] = json.loads(result.pop("diagnostics_json"))
        result["improved"] = None if result["improved"] is None else bool(result["improved"])
        result["objective_resolved"] = (
            None if result["objective_resolved"] is None else bool(result["objective_resolved"])
        )
        return result

    def complete_direction_attempt(
        self,
        *,
        iteration: int,
        candidate_id: str | None,
        outcome: str,
        improved: bool,
        objective_resolved: bool,
        diagnostics: dict[str, Any],
        early_stop_consecutive_misses: int,
        run_id: str | None = None,
        attempt_id: int | None = None,
    ) -> dict[str, Any]:
        now = _now()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if attempt_id is not None:
                where = "id=?"
                parameters: tuple[Any, ...] = (attempt_id,)
            else:
                where = "iteration=?"
                parameters = (iteration,)
                if run_id:
                    where += " AND run_id=?"
                    parameters = (iteration, run_id)
            attempt = connection.execute(
                f"SELECT * FROM direction_attempts WHERE {where} ORDER BY id DESC LIMIT 1",  # noqa: S608
                parameters,
            ).fetchone()
            if attempt is None:
                raise KeyError(f"Direction attempt not found for iteration {iteration}")
            if attempt["status"] != "RESERVED":
                campaign = connection.execute(
                    "SELECT * FROM direction_campaigns WHERE id=?",
                    (attempt["campaign_id"],),
                ).fetchone()
                assert campaign is not None
                return self._direction_campaign_record(connection, campaign)
            connection.execute(
                """UPDATE direction_attempts
                SET candidate_id=?, status='COMPLETED', outcome=?, improved=?,
                    objective_resolved=?, diagnostics_json=?, updated_at=?
                WHERE id=?""",
                (
                    candidate_id,
                    outcome,
                    int(improved),
                    int(objective_resolved),
                    _canonical(diagnostics),
                    now,
                    attempt["id"],
                ),
            )
            campaign = connection.execute(
                "SELECT * FROM direction_campaigns WHERE id=?", (attempt["campaign_id"],)
            ).fetchone()
            assert campaign is not None
            recent = connection.execute(
                """SELECT improved FROM direction_attempts
                WHERE campaign_id=? AND status='COMPLETED' ORDER BY id DESC""",
                (campaign["id"],),
            ).fetchall()
            consecutive_misses = 0
            for row in recent:
                if bool(row["improved"]):
                    break
                consecutive_misses += 1
            status = "ACTIVE"
            closure_reason = None
            if objective_resolved:
                status = "COMPLETED"
                closure_reason = "PUBLIC_OBJECTIVE_RESOLVED"
            elif int(campaign["attempts_used"]) >= int(campaign["maximum_attempts"]):
                status = "EXHAUSTED"
                closure_reason = "MAXIMUM_ATTEMPTS_REACHED"
            elif consecutive_misses >= early_stop_consecutive_misses:
                status = "EARLY_STOPPED"
                closure_reason = "CONSECUTIVE_DIRECTION_MISSES"
            connection.execute(
                """UPDATE direction_campaigns
                SET status=?, successful_attempts=successful_attempts+?,
                    closed_iteration=?, closure_reason=?, updated_at=?
                WHERE id=?""",
                (
                    status,
                    int(improved),
                    iteration if status != "ACTIVE" else None,
                    closure_reason,
                    now,
                    campaign["id"],
                ),
            )
            updated = connection.execute(
                "SELECT * FROM direction_campaigns WHERE id=?", (campaign["id"],)
            ).fetchone()
            assert updated is not None
            return self._direction_campaign_record(connection, updated)

    def cancel_direction_attempt(
        self,
        *,
        attempt_id: int,
        outcome: str,
        diagnostics: dict[str, Any],
    ) -> dict[str, Any]:
        """Cancel an operationally interrupted attempt without charging research budget."""
        now = _now()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            attempt = connection.execute(
                "SELECT * FROM direction_attempts WHERE id=?", (attempt_id,)
            ).fetchone()
            if attempt is None:
                raise KeyError(f"Direction attempt not found: {attempt_id}")
            campaign = connection.execute(
                "SELECT * FROM direction_campaigns WHERE id=?",
                (attempt["campaign_id"],),
            ).fetchone()
            assert campaign is not None
            if attempt["status"] != "RESERVED":
                return self._direction_campaign_record(connection, campaign)
            connection.execute(
                """UPDATE direction_attempts
                SET status='CANCELLED_OPERATIONAL', outcome=?, improved=0,
                    objective_resolved=0, diagnostics_json=?, updated_at=?
                WHERE id=?""",
                (outcome, _canonical(diagnostics), now, attempt_id),
            )
            connection.execute(
                """UPDATE direction_campaigns
                SET attempts_used=MAX(0, attempts_used-1), updated_at=?
                WHERE id=?""",
                (now, campaign["id"]),
            )
            updated = connection.execute(
                "SELECT * FROM direction_campaigns WHERE id=?", (campaign["id"],)
            ).fetchone()
            assert updated is not None
            return self._direction_campaign_record(connection, updated)

    def reconcile_orphaned_direction_attempts(
        self, *, early_stop_consecutive_misses: int
    ) -> list[dict[str, Any]]:
        del early_stop_consecutive_misses
        with self.connection() as connection:
            rows = connection.execute(
                """SELECT attempt.id, attempt.run_id, attempt.iteration,
                       iteration.candidate_id, iteration.status AS iteration_status
                FROM direction_attempts AS attempt
                LEFT JOIN iterations AS iteration
                  ON iteration.run_id=attempt.run_id
                 AND iteration.iteration=attempt.iteration
                WHERE attempt.status='RESERVED'
                  AND (iteration.status IS NULL OR iteration.status!='RUNNING')
                ORDER BY attempt.id"""
            ).fetchall()
        reconciled = []
        for row in rows:
            campaign = self.cancel_direction_attempt(
                attempt_id=int(row["id"]),
                outcome="SERVICE_RESTART_INTERRUPTED",
                diagnostics={
                    "reconciled": True,
                    "run_id": row["run_id"],
                    "iteration_status": row["iteration_status"],
                    "candidate_id": row["candidate_id"],
                    "research_budget_charged": False,
                },
            )
            reconciled.append(
                {
                    "attempt_id": int(row["id"]),
                    "run_id": row["run_id"],
                    "iteration": int(row["iteration"]),
                    "campaign_id": campaign["id"],
                    "budget_refunded": True,
                }
            )
        return reconciled

    def reconcile_exhausted_direction_campaigns(self) -> list[dict[str, Any]]:
        """Close impossible ACTIVE campaigns left behind by an older service version."""
        now = _now()
        with self.connection() as connection:
            rows = connection.execute(
                """SELECT id, generation_id, direction, last_iteration
                FROM direction_campaigns
                WHERE status='ACTIVE' AND attempts_used>=maximum_attempts"""
            ).fetchall()
            for row in rows:
                connection.execute(
                    """UPDATE direction_campaigns
                    SET status='EXHAUSTED', closed_iteration=last_iteration,
                        closure_reason='RECOVERED_EXHAUSTED_CAMPAIGN', updated_at=?
                    WHERE id=?""",
                    (now, row["id"]),
                )
        return [dict(row) for row in rows]

    def _direction_campaign_record(
        self, connection: sqlite3.Connection, row: sqlite3.Row
    ) -> dict[str, Any]:
        item = dict(row)
        item["rationale"] = json.loads(item.pop("rationale_json"))
        item["evidence"] = json.loads(item.pop("evidence_json"))
        item["baseline"] = json.loads(item.pop("baseline_json"))
        attempts = connection.execute(
            "SELECT * FROM direction_attempts WHERE campaign_id=? ORDER BY id",
            (item["id"],),
        ).fetchall()
        item["attempts"] = []
        for attempt in attempts:
            attempt_item = dict(attempt)
            attempt_item["baseline"] = json.loads(attempt_item.pop("baseline_json"))
            attempt_item["diagnostics"] = json.loads(attempt_item.pop("diagnostics_json"))
            attempt_item["improved"] = (
                None if attempt_item["improved"] is None else bool(attempt_item["improved"])
            )
            attempt_item["objective_resolved"] = (
                None
                if attempt_item["objective_resolved"] is None
                else bool(attempt_item["objective_resolved"])
            )
            item["attempts"].append(attempt_item)
        return item

    def reserve_generation_experiment(
        self,
        *,
        candidate_hash: str,
        generation_id: str,
        factor_id: str,
        family: str,
        iteration: int,
        maximum_family_candidates: int,
    ) -> dict[str, Any]:
        now = _now()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            generation = connection.execute(
                "SELECT * FROM research_generations WHERE generation_id=?", (generation_id,)
            ).fetchone()
            if generation is None:
                raise KeyError(f"Research generation not found: {generation_id}")
            if generation["status"] != "ACTIVE":
                raise PermissionError(f"Research generation is {generation['status']}")
            if generation["candidate_attempts"] >= generation["maximum_candidates"]:
                raise PermissionError("Research generation candidate budget exhausted")
            family_count = connection.execute(
                """SELECT COUNT(*) AS count FROM generation_experiments
                WHERE generation_id=? AND family=?""",
                (generation_id, family),
            ).fetchone()["count"]
            if int(family_count) >= maximum_family_candidates:
                raise PermissionError(f"Factor family candidate budget exhausted: {family}")
            connection.execute(
                """INSERT INTO generation_experiments
                (candidate_hash, generation_id, factor_id, family, iteration, status,
                 created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 'RESERVED', ?, ?)""",
                (candidate_hash, generation_id, factor_id, family, iteration, now, now),
            )
            connection.execute(
                """UPDATE research_generations SET candidate_attempts=candidate_attempts+1
                WHERE generation_id=?""",
                (generation_id,),
            )
            updated = connection.execute(
                "SELECT * FROM research_generations WHERE generation_id=?", (generation_id,)
            ).fetchone()
        return dict(updated)

    def close_generation_experiment(
        self, candidate_hash: str, *, status: str, public_verdict: str
    ) -> None:
        with self.connection() as connection:
            cursor = connection.execute(
                """UPDATE generation_experiments
                SET status=?, public_verdict=?, updated_at=?
                WHERE candidate_hash=? AND status='RESERVED'""",
                (status, public_verdict, _now(), candidate_hash),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Reserved generation experiment not found: {candidate_hash}")

    def record_blind_evaluation(
        self,
        *,
        candidate_hash: str,
        generation_id: str,
        iteration: int,
        holdout_verdict: str,
        holdout_passed: bool,
        holdout_evidence_hash: str,
        capital_verdict: str | None = None,
        capital_passed: bool | None = None,
        capital_evidence_hash: str | None = None,
    ) -> dict[str, Any]:
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            generation = connection.execute(
                "SELECT * FROM research_generations WHERE generation_id=?", (generation_id,)
            ).fetchone()
            if generation is None or generation["status"] != "ACTIVE":
                raise PermissionError("No active research generation for blind evaluation")
            if generation["holdout_attempts"] >= generation["maximum_holdout_attempts"]:
                raise PermissionError("Research generation holdout budget exhausted")
            connection.execute(
                """INSERT INTO blind_evaluations
                (candidate_hash, generation_id, iteration, holdout_verdict, holdout_passed,
                 holdout_evidence_hash, capital_verdict, capital_passed, capital_evidence_hash,
                 created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    candidate_hash,
                    generation_id,
                    iteration,
                    holdout_verdict,
                    int(holdout_passed),
                    holdout_evidence_hash,
                    capital_verdict,
                    int(capital_passed) if capital_passed is not None else None,
                    capital_evidence_hash,
                    _now(),
                ),
            )
            connection.execute(
                """UPDATE research_generations SET holdout_attempts=holdout_attempts+1
                WHERE generation_id=?""",
                (generation_id,),
            )
            row = connection.execute(
                "SELECT * FROM blind_evaluations WHERE candidate_hash=?", (candidate_hash,)
            ).fetchone()
        assert row is not None
        result = dict(row)
        result["holdout_passed"] = bool(result["holdout_passed"])
        if result["capital_passed"] is not None:
            result["capital_passed"] = bool(result["capital_passed"])
        return result

    def update_blind_capital(
        self,
        candidate_hash: str,
        *,
        capital_verdict: str,
        capital_passed: bool,
        capital_evidence_hash: str,
    ) -> None:
        with self.connection() as connection:
            cursor = connection.execute(
                """UPDATE blind_evaluations
                SET capital_verdict=?, capital_passed=?, capital_evidence_hash=?
                WHERE candidate_hash=? AND capital_verdict IS NULL""",
                (
                    capital_verdict,
                    int(capital_passed),
                    capital_evidence_hash,
                    candidate_hash,
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Pending blind capital evaluation not found: {candidate_hash}")

    def blind_evaluations(
        self, *, limit: int = 100, generation_id: str | None = None
    ) -> list[dict[str, Any]]:
        where = "WHERE generation_id=?" if generation_id else ""
        parameters: list[Any] = [generation_id] if generation_id else []
        parameters.append(min(max(limit, 1), 1000))
        with self.connection() as connection:
            rows = connection.execute(
                f"SELECT * FROM blind_evaluations {where} ORDER BY id DESC LIMIT ?",  # noqa: S608
                parameters,
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["holdout_passed"] = bool(item["holdout_passed"])
            if item["capital_passed"] is not None:
                item["capital_passed"] = bool(item["capital_passed"])
            result.append(item)
        return result

    def bootstrap_factor_pool(self) -> int:
        """Import completed legacy iterations without changing their original evidence."""
        imported = 0
        for record in reversed(self.iteration_history(limit=500)):
            if (
                record["status"] != "COMPLETED"
                or not record.get("candidate_id")
                or not record.get("proposal")
                or not record.get("metrics")
            ):
                continue
            if self.factor_pool_record(record["candidate_id"]) is not None:
                continue
            metrics = record["metrics"]
            eligible = (
                float(metrics.get("long_only_sharpe_ratio", float("-inf"))) > 0
                and float(metrics.get("long_only_simple_annual_return", float("-inf"))) > 0
                and float(metrics.get("long_only_coverage", 0)) >= 0.80
                and float(metrics.get("long_only_cost_stress_net_ir", float("-inf"))) > 0
            )
            self.upsert_factor_pool(
                factor_id=record["candidate_id"],
                source_iteration=int(record["iteration"]),
                proposal=record["proposal"],
                metrics=metrics,
                status="ELIGIBLE" if eligible else "SCREENED_OUT",
                status_reason="legacy A-share long-only deterministic screen",
            )
            imported += 1
        return imported


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _materialized_cache_state(snapshot: dict[str, Any]) -> dict[str, Any]:
    updated_at = _parse_datetime(snapshot.get("updated_at"))
    expires_at = _parse_datetime(snapshot.get("expires_at"))
    now = datetime.now(UTC)
    age_seconds = int((now - updated_at).total_seconds()) if updated_at else None
    if expires_at is None:
        status = "NO_TTL"
        stale = False
    else:
        stale = expires_at <= now
        status = "STALE" if stale else "FRESH"
    return {
        "status": status,
        "stale": stale,
        "age_seconds": age_seconds,
        "expires_at": snapshot.get("expires_at"),
        "source": snapshot.get("source") or "unknown",
        "snapshot_status": snapshot.get("status") or "READY",
    }


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _bounded_confidence(value: Any) -> float:
    if isinstance(value, str):
        label = value.strip().upper()
        if label in {"LOW", "MEDIUM", "HIGH"}:
            return {"LOW": 0.30, "MEDIUM": 0.60, "HIGH": 0.90}[label]
    try:
        return min(max(float(value), 0.0), 1.0)
    except (TypeError, ValueError):
        return 0.0


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()
