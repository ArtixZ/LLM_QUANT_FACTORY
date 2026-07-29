from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from typing import Any

from autoalpha.service.autocombine_intelligence import (
    enrich_factor_record,
    factor_snapshot_homogeneity_summary,
)
from autoalpha.service.store import ServiceStore


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class AutoCombineStore:
    """Persistent AutoCombine state sharing the AutoAlpha SQLite database."""

    def __init__(self, base: ServiceStore) -> None:
        self.base = base
        self._initialize()

    def _initialize(self) -> None:
        with self.base.connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS combine_tasks (
                    task_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    market TEXT NOT NULL,
                    data_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    protocol_json TEXT NOT NULL,
                    scope_json TEXT NOT NULL,
                    construction_json TEXT NOT NULL,
                    objective_json TEXT NOT NULL,
                    budget_json TEXT NOT NULL,
                    factor_snapshot_json TEXT NOT NULL,
                    snapshot_hash TEXT NOT NULL,
                    iteration INTEGER NOT NULL DEFAULT 0,
                    best_experiment_id INTEGER,
                    qualified_experiment_id INTEGER,
                    production_candidate_experiment_id INTEGER,
                    qualification_status TEXT NOT NULL DEFAULT 'NO_EXPERIMENT',
                    strategy_cluster_id TEXT,
                    nearest_task_id TEXT,
                    nearest_active_return_correlation REAL,
                    blind_verdict TEXT,
                    blind_evidence_hash TEXT,
                    stop_requested INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    notes TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS combine_experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL REFERENCES combine_tasks(task_id),
                    iteration INTEGER NOT NULL,
                    candidate_hash TEXT NOT NULL,
                    action TEXT NOT NULL,
                    proposal_source TEXT NOT NULL,
                    factor_ids_json TEXT NOT NULL,
                    weights_json TEXT NOT NULL,
                    rationale TEXT NOT NULL,
                    hypothesis TEXT NOT NULL,
                    metrics_json TEXT,
                    score REAL,
                    gate_distance REAL,
                    qualification TEXT NOT NULL DEFAULT 'EVALUATED',
                    gate_status TEXT NOT NULL,
                    failed_gates_json TEXT NOT NULL,
                    prompt_hash TEXT,
                    response_hash TEXT,
                    return_artifact_path TEXT,
                    return_artifact_hash TEXT,
                    duration_seconds REAL,
                    created_at TEXT NOT NULL,
                    UNIQUE(task_id, candidate_hash)
                );
                CREATE INDEX IF NOT EXISTS idx_combine_experiments_task
                ON combine_experiments(task_id, id DESC);
                CREATE TABLE IF NOT EXISTS combine_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL REFERENCES combine_tasks(task_id),
                    iteration INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    content TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS combine_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT,
                    category TEXT NOT NULL,
                    level TEXT NOT NULL,
                    event TEXT NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_combine_events_task
                ON combine_events(task_id, id DESC);
                CREATE TABLE IF NOT EXISTS strategy_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    source_task_id TEXT NOT NULL REFERENCES combine_tasks(task_id),
                    source_experiment_id INTEGER NOT NULL REFERENCES combine_experiments(id),
                    name TEXT NOT NULL,
                    market TEXT NOT NULL,
                    lifecycle TEXT NOT NULL,
                    specification_json TEXT NOT NULL,
                    evidence_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(strategy_id, version)
                );
                """
            )
            task_columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(combine_tasks)").fetchall()
            }
            if "blind_verdict" not in task_columns:
                connection.execute("ALTER TABLE combine_tasks ADD COLUMN blind_verdict TEXT")
            if "blind_evidence_hash" not in task_columns:
                connection.execute("ALTER TABLE combine_tasks ADD COLUMN blind_evidence_hash TEXT")
            task_migrations = {
                "qualified_experiment_id": "INTEGER",
                "production_candidate_experiment_id": "INTEGER",
                "qualification_status": "TEXT NOT NULL DEFAULT 'NO_EXPERIMENT'",
                "strategy_cluster_id": "TEXT",
                "nearest_task_id": "TEXT",
                "nearest_active_return_correlation": "REAL",
            }
            for name, declaration in task_migrations.items():
                if name not in task_columns:
                    connection.execute(
                        f"ALTER TABLE combine_tasks ADD COLUMN {name} {declaration}"  # noqa: S608
                    )
            experiment_columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(combine_experiments)").fetchall()
            }
            experiment_migrations = {
                "gate_distance": "REAL",
                "qualification": "TEXT NOT NULL DEFAULT 'EVALUATED'",
                "return_artifact_path": "TEXT",
                "return_artifact_hash": "TEXT",
            }
            for name, declaration in experiment_migrations.items():
                if name not in experiment_columns:
                    connection.execute(
                        f"ALTER TABLE combine_experiments ADD COLUMN {name} {declaration}"  # noqa: S608
                    )
            connection.execute(
                """UPDATE combine_tasks
                SET qualified_experiment_id=best_experiment_id
                WHERE qualified_experiment_id IS NULL AND best_experiment_id IN (
                    SELECT id FROM combine_experiments WHERE gate_status='PASSED'
                )"""
            )
            connection.execute(
                """UPDATE combine_tasks
                SET production_candidate_experiment_id=qualified_experiment_id
                WHERE production_candidate_experiment_id IS NULL
                  AND blind_verdict='BLIND_GENERALIZATION_PASSED'"""
            )
            connection.execute(
                """UPDATE combine_tasks SET qualification_status=CASE
                    WHEN production_candidate_experiment_id IS NOT NULL THEN 'PRODUCTION_CANDIDATE'
                    WHEN qualified_experiment_id IS NOT NULL THEN 'QUALIFIED_CHAMPION'
                    WHEN best_experiment_id IS NOT NULL THEN 'RESEARCH_LEADER_ONLY'
                    ELSE 'NO_EXPERIMENT' END"""
            )

    def create_task(self, record: dict[str, Any]) -> dict[str, Any]:
        now = _now()
        with self.base.connection() as connection:
            connection.execute(
                """INSERT INTO combine_tasks
                (task_id, name, market, data_path, status, phase, protocol_json, scope_json,
                 construction_json, objective_json, budget_json, factor_snapshot_json,
                 snapshot_hash, notes, created_at, updated_at)
                VALUES (?, ?, ?, ?, 'READY', 'WAITING', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record["task_id"],
                    record["name"],
                    record["market"],
                    record["data_path"],
                    _json(record["protocol"]),
                    _json(record["scope"]),
                    _json(record["construction"]),
                    _json(record["objective"]),
                    _json(record["budget"]),
                    _json(record["factor_snapshot"]),
                    record["snapshot_hash"],
                    record.get("notes", ""),
                    now,
                    now,
                ),
            )
        task = self.task(record["task_id"])
        assert task is not None
        return task

    def tasks(self) -> list[dict[str, Any]]:
        with self.base.connection() as connection:
            rows = connection.execute(
                """SELECT task.*, COUNT(exp.id) AS experiment_count
                FROM combine_tasks task
                LEFT JOIN combine_experiments exp ON exp.task_id=task.task_id
                GROUP BY task.task_id ORDER BY task.updated_at DESC"""
            ).fetchall()
        return [self._task(row) for row in rows]

    def task(self, task_id: str) -> dict[str, Any] | None:
        with self.base.connection() as connection:
            row = connection.execute(
                """SELECT task.*, COUNT(exp.id) AS experiment_count
                FROM combine_tasks task
                LEFT JOIN combine_experiments exp ON exp.task_id=task.task_id
                WHERE task.task_id=? GROUP BY task.task_id""",
                (task_id,),
            ).fetchone()
        return self._task(row) if row is not None else None

    @staticmethod
    def _task(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        for key in ("protocol", "scope", "construction", "objective", "budget", "factor_snapshot"):
            item[key] = json.loads(item.pop(f"{key}_json"))
        item["factor_snapshot"] = [
            record
            if record.get("mechanism_fingerprint") and record.get("search_cluster_id")
            else enrich_factor_record(record)
            for record in item["factor_snapshot"]
        ]
        item["stop_requested"] = bool(item["stop_requested"])
        item["factor_count"] = len(item["factor_snapshot"])
        item["homogeneity_summary"] = factor_snapshot_homogeneity_summary(
            item["factor_snapshot"]
        )
        return item

    def update_task(self, task_id: str, **values: Any) -> dict[str, Any]:
        allowed = {
            "status",
            "phase",
            "iteration",
            "best_experiment_id",
            "stop_requested",
            "last_error",
            "notes",
            "blind_verdict",
            "blind_evidence_hash",
            "qualified_experiment_id",
            "production_candidate_experiment_id",
            "qualification_status",
            "strategy_cluster_id",
            "nearest_task_id",
            "nearest_active_return_correlation",
        }
        invalid = set(values) - allowed
        if invalid:
            raise ValueError(f"Unknown AutoCombine task fields: {sorted(invalid)}")
        values["updated_at"] = _now()
        assignments = ", ".join(f"{key}=?" for key in values)
        with self.base.connection() as connection:
            cursor = connection.execute(
                f"UPDATE combine_tasks SET {assignments} WHERE task_id=?",  # noqa: S608
                (*values.values(), task_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"AutoCombine task not found: {task_id}")
        task = self.task(task_id)
        assert task is not None
        return task

    def candidate_exists(self, task_id: str, candidate_hash: str) -> bool:
        with self.base.connection() as connection:
            row = connection.execute(
                "SELECT 1 FROM combine_experiments WHERE task_id=? AND candidate_hash=?",
                (task_id, candidate_hash),
            ).fetchone()
        return row is not None

    def record_experiment(self, task_id: str, record: dict[str, Any]) -> dict[str, Any]:
        with self.base.connection() as connection:
            cursor = connection.execute(
                """INSERT INTO combine_experiments
                (task_id, iteration, candidate_hash, action, proposal_source,
                 factor_ids_json, weights_json, rationale, hypothesis, metrics_json,
                 score, gate_distance, qualification, gate_status, failed_gates_json,
                 prompt_hash, response_hash, return_artifact_path, return_artifact_hash,
                 duration_seconds, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    task_id,
                    record["iteration"],
                    record["candidate_hash"],
                    record["action"],
                    record["proposal_source"],
                    _json(record["factor_ids"]),
                    _json(record["weights"]),
                    record.get("rationale", ""),
                    record.get("hypothesis", ""),
                    _json(record["metrics"]) if record.get("metrics") is not None else None,
                    record.get("score"),
                    record.get("gate_distance"),
                    record.get("qualification", "EVALUATED"),
                    record["gate_status"],
                    _json(record.get("failed_gates", [])),
                    record.get("prompt_hash"),
                    record.get("response_hash"),
                    record.get("return_artifact_path"),
                    record.get("return_artifact_hash"),
                    record.get("duration_seconds"),
                    _now(),
                ),
            )
            experiment_id = int(cursor.lastrowid)
        experiment = self.experiment(experiment_id)
        assert experiment is not None
        return experiment

    def experiments(self, task_id: str, *, limit: int = 500) -> list[dict[str, Any]]:
        with self.base.connection() as connection:
            rows = connection.execute(
                """SELECT * FROM combine_experiments WHERE task_id=?
                ORDER BY id DESC LIMIT ?""",
                (task_id, min(max(limit, 1), 5000)),
            ).fetchall()
        return [self._experiment(row) for row in rows]

    def experiment(self, experiment_id: int) -> dict[str, Any] | None:
        with self.base.connection() as connection:
            row = connection.execute(
                "SELECT * FROM combine_experiments WHERE id=?", (experiment_id,)
            ).fetchone()
        return self._experiment(row) if row is not None else None

    def update_experiment(self, experiment_id: int, **values: Any) -> dict[str, Any]:
        allowed = {
            "metrics",
            "gate_distance",
            "qualification",
            "gate_status",
            "failed_gates",
            "return_artifact_path",
            "return_artifact_hash",
        }
        invalid = set(values) - allowed
        if invalid:
            raise ValueError(f"Unknown AutoCombine experiment fields: {sorted(invalid)}")
        encoded: dict[str, Any] = {}
        for key, value in values.items():
            encoded[f"{key}_json" if key in {"metrics", "failed_gates"} else key] = (
                _json(value) if key in {"metrics", "failed_gates"} else value
            )
        assignments = ", ".join(f"{key}=?" for key in encoded)
        with self.base.connection() as connection:
            cursor = connection.execute(
                f"UPDATE combine_experiments SET {assignments} WHERE id=?",  # noqa: S608
                (*encoded.values(), experiment_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"AutoCombine experiment not found: {experiment_id}")
        experiment = self.experiment(experiment_id)
        assert experiment is not None
        return experiment

    @staticmethod
    def _experiment(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["factor_ids"] = json.loads(item.pop("factor_ids_json"))
        item["weights"] = json.loads(item.pop("weights_json"))
        metrics = item.pop("metrics_json")
        item["metrics"] = json.loads(metrics) if metrics else None
        item["failed_gates"] = json.loads(item.pop("failed_gates_json"))
        return item

    def remember(
        self, task_id: str, iteration: int, kind: str, content: str, payload: dict[str, Any]
    ) -> None:
        with self.base.connection() as connection:
            connection.execute(
                """INSERT INTO combine_memories
                (task_id, iteration, kind, content, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (task_id, iteration, kind, content, _json(payload), _now()),
            )

    def memories(self, task_id: str, *, limit: int = 30) -> list[dict[str, Any]]:
        with self.base.connection() as connection:
            rows = connection.execute(
                """SELECT * FROM combine_memories WHERE task_id=?
                ORDER BY id DESC LIMIT ?""",
                (task_id, min(max(limit, 1), 500)),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item.pop("payload_json"))
            result.append(item)
        return result

    def event(
        self,
        task_id: str | None,
        category: str,
        event: str,
        title: str,
        message: str,
        *,
        level: str = "INFO",
        payload: dict[str, Any] | None = None,
    ) -> None:
        with self.base.connection() as connection:
            connection.execute(
                """INSERT INTO combine_events
                (task_id, category, level, event, title, message, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (task_id, category, level, event, title, message, _json(payload or {}), _now()),
            )

    def events(self, task_id: str | None = None, *, limit: int = 200) -> list[dict[str, Any]]:
        where = "WHERE task_id=?" if task_id else ""
        parameters: tuple[Any, ...] = (task_id, limit) if task_id else (limit,)
        with self.base.connection() as connection:
            rows = connection.execute(
                f"SELECT * FROM combine_events {where} ORDER BY id DESC LIMIT ?",  # noqa: S608
                parameters,
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item.pop("payload_json"))
            result.append(item)
        return result

    def promote_strategy(self, task_id: str, experiment_id: int, name: str) -> dict[str, Any]:
        task = self.task(task_id)
        experiment = self.experiment(experiment_id)
        if task is None or experiment is None or experiment["task_id"] != task_id:
            raise KeyError("AutoCombine task or experiment not found")
        if experiment["gate_status"] != "PASSED":
            raise ValueError("Only a gate-passing experiment can enter the strategy registry")
        if task.get("production_candidate_experiment_id") != experiment_id:
            raise ValueError("Strategy promotion requires the task production candidate")
        if task.get("blind_verdict") != "BLIND_GENERALIZATION_PASSED":
            raise ValueError("Strategy promotion requires a passing isolated holdout verdict")
        strategy_id = f"S_{hashlib.sha256(task_id.encode()).hexdigest()[:12]}"
        with self.base.connection() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(version), 0) FROM strategy_versions WHERE strategy_id=?",
                (strategy_id,),
            ).fetchone()
            version = int(row[0]) + 1
            specification = {
                "strategy_id": strategy_id,
                "version": version,
                "name": name.strip(),
                "market": task["market"],
                "factor_snapshot_hash": task["snapshot_hash"],
                "factor_ids": experiment["factor_ids"],
                "factor_weights": experiment["weights"],
                "strategy_cluster_id": task.get("strategy_cluster_id"),
                "nearest_task_id": task.get("nearest_task_id"),
                "nearest_active_return_correlation": task.get("nearest_active_return_correlation"),
                "signal_composition": "weighted_cross_sectional_zscore",
                "portfolio_mode": "long_only",
                "protocol": task["protocol"],
                "construction": task["construction"],
                "evaluation": experiment["metrics"],
                "execution": {
                    "signal_time": "END_OF_DAY_AFTER_CLOSE",
                    "execution_lag_sessions": 1,
                    "rebalance": "WEEKLY_FIRST_SESSION",
                    "engine": "A_SHARE_LONG_ONLY_WEEKLY_VECTOR_PROXY_V1",
                },
            }
            evidence_hash = hashlib.sha256(_json(specification).encode()).hexdigest()
            cursor = connection.execute(
                """INSERT INTO strategy_versions
                (strategy_id, version, source_task_id, source_experiment_id, name,
                 market, lifecycle, specification_json, evidence_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?, 'QUALIFIED', ?, ?, ?)""",
                (
                    strategy_id,
                    version,
                    task_id,
                    experiment_id,
                    name.strip(),
                    task["market"],
                    _json(specification),
                    evidence_hash,
                    _now(),
                ),
            )
            row_id = int(cursor.lastrowid)
        return self.strategy_version(row_id)

    def strategies(self) -> list[dict[str, Any]]:
        with self.base.connection() as connection:
            rows = connection.execute("SELECT * FROM strategy_versions ORDER BY id DESC").fetchall()
        return [self._strategy(row) for row in rows]

    def strategy_version(self, row_id: int) -> dict[str, Any]:
        with self.base.connection() as connection:
            row = connection.execute(
                "SELECT * FROM strategy_versions WHERE id=?", (row_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Strategy version not found: {row_id}")
        return self._strategy(row)

    @staticmethod
    def _strategy(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["specification"] = json.loads(item.pop("specification_json"))
        return item
