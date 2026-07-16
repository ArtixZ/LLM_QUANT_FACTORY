from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from autoalpha.registry.lifecycle import ALLOWED_TRANSITIONS, FactorState

LogCategory = Literal["audit", "action", "research", "delivery"]


class ServiceStore:
    """SQLite state, memory, metrics, and a tamper-evident unified event stream."""

    def __init__(self, path: Path) -> None:
        self.path = path
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
                    initial_cash_cny REAL NOT NULL,
                    cash_cny REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_rebalanced_date TEXT
                );
                CREATE TABLE IF NOT EXISTS paper_positions (
                    portfolio_id INTEGER NOT NULL REFERENCES paper_portfolios(id) ON DELETE CASCADE,
                    symbol TEXT NOT NULL,
                    security_name TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    average_cost_cny REAL NOT NULL,
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
                    price_cny REAL NOT NULL,
                    notional_cny REAL NOT NULL,
                    fees_cny REAL NOT NULL,
                    reason TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS paper_nav (
                    portfolio_id INTEGER NOT NULL REFERENCES paper_portfolios(id) ON DELETE CASCADE,
                    trade_date TEXT NOT NULL,
                    nav_cny REAL NOT NULL,
                    cash_cny REAL NOT NULL,
                    market_value_cny REAL NOT NULL,
                    gross_exposure REAL NOT NULL,
                    daily_return REAL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(portfolio_id, trade_date)
                );
                CREATE INDEX IF NOT EXISTS idx_factor_pool_status
                ON factor_pool(status, source_iteration);
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
            if connection.execute(
                "SELECT 1 FROM factor_pool WHERE factor_id=?", (factor_id,)
            ).fetchone() is None:
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
                if connection.execute(
                    "SELECT 1 FROM factor_pool WHERE factor_id=?", (target,)
                ).fetchone() is None:
                    continue
                confidence = min(max(float(relation.get("confidence", 0.0)), 0.0), 1.0)
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
        self, *, name: str, config: dict[str, Any], initial_cash_cny: float
    ) -> dict[str, Any]:
        now = _now()
        with self.connection() as connection:
            cursor = connection.execute(
                """INSERT INTO paper_portfolios
                (name, status, config_json, initial_cash_cny, cash_cny, created_at, updated_at)
                VALUES (?, 'ACTIVE', ?, ?, ?, ?, ?)""",
                (name.strip(), _canonical(config), initial_cash_cny, initial_cash_cny, now, now),
            )
            portfolio_id = int(cursor.lastrowid)
        record = self.paper_portfolio(portfolio_id)
        assert record is not None
        return record

    def paper_portfolios(self, *, limit: int = 100) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """SELECT portfolio.*, nav.trade_date, nav.nav_cny, nav.market_value_cny,
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
                """SELECT portfolio.*, nav.trade_date, nav.nav_cny, nav.market_value_cny,
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
        item["trades"] = [dict(trade) for trade in trades]
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
        cash_cny: float,
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
                (portfolio_id, symbol, security_name, quantity, average_cost_cny, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)""",
                [
                    (
                        portfolio_id,
                        position["symbol"],
                        position["security_name"],
                        position["quantity"],
                        position["average_cost_cny"],
                        now,
                    )
                    for position in positions
                    if int(position["quantity"]) > 0
                ],
            )
            connection.executemany(
                """INSERT INTO paper_trades
                (portfolio_id, trade_date, symbol, security_name, side, quantity, price_cny,
                 notional_cny, fees_cny, reason, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        portfolio_id,
                        trade["trade_date"],
                        trade["symbol"],
                        trade["security_name"],
                        trade["side"],
                        trade["quantity"],
                        trade["price_cny"],
                        trade["notional_cny"],
                        trade["fees_cny"],
                        trade["reason"],
                        now,
                    )
                    for trade in trades
                ],
            )
            previous = connection.execute(
                """SELECT nav_cny FROM paper_nav WHERE portfolio_id=?
                ORDER BY trade_date DESC LIMIT 1""",
                (portfolio_id,),
            ).fetchone()
            daily_return = (
                nav["nav_cny"] / float(previous["nav_cny"]) - 1.0
                if previous and float(previous["nav_cny"])
                else None
            )
            connection.execute(
                """INSERT INTO paper_nav
                (portfolio_id, trade_date, nav_cny, cash_cny, market_value_cny, gross_exposure,
                 daily_return, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(portfolio_id, trade_date) DO UPDATE SET
                nav_cny=excluded.nav_cny, cash_cny=excluded.cash_cny,
                market_value_cny=excluded.market_value_cny, gross_exposure=excluded.gross_exposure,
                daily_return=excluded.daily_return, updated_at=excluded.updated_at""",
                (
                    portfolio_id,
                    nav["trade_date"],
                    nav["nav_cny"],
                    cash_cny,
                    nav["market_value_cny"],
                    nav["gross_exposure"],
                    daily_return,
                    now,
                ),
            )
            connection.execute(
                """UPDATE paper_portfolios SET cash_cny=?, updated_at=?,
                last_rebalanced_date=CASE WHEN ? THEN ? ELSE last_rebalanced_date END WHERE id=?""",
                (cash_cny, now, int(rebalanced), nav["trade_date"], portfolio_id),
            )

    @staticmethod
    def _paper_portfolio_record(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["config"] = json.loads(item.pop("config_json"))
        return item

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
    ) -> dict[str, Any]:
        now = _now()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            where = "iteration=?"
            parameters: tuple[Any, ...] = (iteration,)
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

    def reconcile_orphaned_direction_attempts(
        self, *, early_stop_consecutive_misses: int
    ) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """SELECT attempt.iteration, iteration.candidate_id
                FROM direction_attempts AS attempt
                LEFT JOIN iterations AS iteration ON iteration.iteration=attempt.iteration
                WHERE attempt.status='RESERVED'
                  AND (iteration.status IS NULL OR iteration.status!='RUNNING')
                ORDER BY attempt.id"""
            ).fetchall()
        reconciled = []
        for row in rows:
            campaign = self.complete_direction_attempt(
                iteration=int(row["iteration"]),
                candidate_id=row["candidate_id"],
                outcome="SERVICE_RESTART_INTERRUPTED",
                improved=False,
                objective_resolved=False,
                diagnostics={"reconciled": True},
                early_stop_consecutive_misses=early_stop_consecutive_misses,
            )
            reconciled.append({"iteration": int(row["iteration"]), "campaign_id": campaign["id"]})
        return reconciled

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
                float(metrics.get("sharpe_ratio", float("-inf"))) > 0
                and float(metrics.get("simple_annual_return", float("-inf"))) > 0
                and float(metrics.get("coverage", 0)) >= 0.80
                and float(metrics.get("cost_stress_net_ir", float("-inf"))) > 0
            )
            self.upsert_factor_pool(
                factor_id=record["candidate_id"],
                source_iteration=int(record["iteration"]),
                proposal=record["proposal"],
                metrics=metrics,
                status="ELIGIBLE" if eligible else "SCREENED_OUT",
                status_reason="legacy deterministic screen",
            )
            imported += 1
        return imported


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()
