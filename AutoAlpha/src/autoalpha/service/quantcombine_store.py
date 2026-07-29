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


class QuantCombineStore:
    """Persistent state for deterministic and statistical portfolio searches."""

    def __init__(self, base: ServiceStore) -> None:
        self.base = base
        self._initialize()

    def _initialize(self) -> None:
        with self.base.connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS quant_combine_tasks (
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
                    engine_json TEXT NOT NULL,
                    budget_json TEXT NOT NULL,
                    factor_snapshot_json TEXT NOT NULL,
                    snapshot_hash TEXT NOT NULL,
                    iteration INTEGER NOT NULL DEFAULT 0,
                    evaluation_count INTEGER NOT NULL DEFAULT 0,
                    best_candidate_id INTEGER,
                    qualified_candidate_id INTEGER,
                    production_candidate_id INTEGER,
                    qualification_status TEXT NOT NULL DEFAULT 'NO_CANDIDATE',
                    blind_verdict TEXT,
                    blind_evidence_hash TEXT,
                    stop_requested INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    notes TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS quant_factor_screen (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL REFERENCES quant_combine_tasks(task_id),
                    factor_id TEXT NOT NULL,
                    cluster_id TEXT,
                    cluster_leader INTEGER NOT NULL DEFAULT 0,
                    stability_score REAL NOT NULL,
                    metrics_json TEXT NOT NULL,
                    exclusion_reason TEXT,
                    return_artifact_path TEXT,
                    return_artifact_hash TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE(task_id, factor_id)
                );
                CREATE INDEX IF NOT EXISTS idx_quant_factor_screen_task
                ON quant_factor_screen(task_id, stability_score DESC);
                CREATE TABLE IF NOT EXISTS quant_combine_candidates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL REFERENCES quant_combine_tasks(task_id),
                    iteration INTEGER NOT NULL,
                    stage TEXT NOT NULL,
                    algorithm TEXT NOT NULL,
                    action TEXT NOT NULL,
                    candidate_hash TEXT NOT NULL,
                    parent_ids_json TEXT NOT NULL,
                    factor_ids_json TEXT NOT NULL,
                    weights_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    objectives_json TEXT NOT NULL,
                    score REAL NOT NULL,
                    gate_distance REAL NOT NULL,
                    gate_status TEXT NOT NULL,
                    failed_gates_json TEXT NOT NULL,
                    pareto_rank INTEGER,
                    crowding_distance REAL,
                    qualification TEXT NOT NULL DEFAULT 'EVALUATED',
                    return_artifact_path TEXT,
                    return_artifact_hash TEXT,
                    duration_seconds REAL,
                    created_at TEXT NOT NULL,
                    UNIQUE(task_id, candidate_hash)
                );
                CREATE INDEX IF NOT EXISTS idx_quant_candidates_task
                ON quant_combine_candidates(task_id, id DESC);
                CREATE TABLE IF NOT EXISTS quant_combine_events (
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
                CREATE INDEX IF NOT EXISTS idx_quant_events_task
                ON quant_combine_events(task_id, id DESC);
                CREATE TABLE IF NOT EXISTS quant_strategy_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    source_task_id TEXT NOT NULL REFERENCES quant_combine_tasks(task_id),
                    source_candidate_id INTEGER NOT NULL REFERENCES quant_combine_candidates(id),
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

    def create_task(self, record: dict[str, Any]) -> dict[str, Any]:
        now = _now()
        with self.base.connection() as connection:
            connection.execute(
                """INSERT INTO quant_combine_tasks
                (task_id, name, market, data_path, status, phase, protocol_json, scope_json,
                 construction_json, objective_json, engine_json, budget_json,
                 factor_snapshot_json, snapshot_hash, notes, created_at, updated_at)
                VALUES (?, ?, ?, ?, 'READY', 'WAITING', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record["task_id"],
                    record["name"],
                    record["market"],
                    record["data_path"],
                    _json(record["protocol"]),
                    _json(record["scope"]),
                    _json(record["construction"]),
                    _json(record["objective"]),
                    _json(record["engine"]),
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
                """SELECT task.*, COUNT(candidate.id) AS candidate_count
                FROM quant_combine_tasks task
                LEFT JOIN quant_combine_candidates candidate ON candidate.task_id=task.task_id
                GROUP BY task.task_id ORDER BY task.updated_at DESC"""
            ).fetchall()
        return [self._task(row) for row in rows]

    def task(self, task_id: str) -> dict[str, Any] | None:
        with self.base.connection() as connection:
            row = connection.execute(
                """SELECT task.*, COUNT(candidate.id) AS candidate_count
                FROM quant_combine_tasks task
                LEFT JOIN quant_combine_candidates candidate ON candidate.task_id=task.task_id
                WHERE task.task_id=? GROUP BY task.task_id""",
                (task_id,),
            ).fetchone()
        return self._task(row) if row is not None else None

    @staticmethod
    def _task(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        for key in (
            "protocol",
            "scope",
            "construction",
            "objective",
            "engine",
            "budget",
            "factor_snapshot",
        ):
            item[key] = json.loads(item.pop(f"{key}_json"))
        item["factor_snapshot"] = [
            value
            if value.get("mechanism_fingerprint") and value.get("search_cluster_id")
            else enrich_factor_record(value)
            for value in item["factor_snapshot"]
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
            "evaluation_count",
            "best_candidate_id",
            "qualified_candidate_id",
            "production_candidate_id",
            "qualification_status",
            "blind_verdict",
            "blind_evidence_hash",
            "stop_requested",
            "last_error",
            "notes",
        }
        invalid = set(values) - allowed
        if invalid:
            raise ValueError(f"Unknown QuantCombine task fields: {sorted(invalid)}")
        values["updated_at"] = _now()
        assignments = ", ".join(f"{key}=?" for key in values)
        with self.base.connection() as connection:
            cursor = connection.execute(
                f"UPDATE quant_combine_tasks SET {assignments} WHERE task_id=?",  # noqa: S608
                (*values.values(), task_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"QuantCombine task not found: {task_id}")
        task = self.task(task_id)
        assert task is not None
        return task

    def upsert_factor_screen(self, task_id: str, record: dict[str, Any]) -> None:
        with self.base.connection() as connection:
            connection.execute(
                """INSERT INTO quant_factor_screen
                (task_id, factor_id, cluster_id, cluster_leader, stability_score, metrics_json,
                 exclusion_reason, return_artifact_path, return_artifact_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id, factor_id) DO UPDATE SET
                  cluster_id=excluded.cluster_id,
                  cluster_leader=excluded.cluster_leader,
                  stability_score=excluded.stability_score,
                  metrics_json=excluded.metrics_json,
                  exclusion_reason=excluded.exclusion_reason,
                  return_artifact_path=excluded.return_artifact_path,
                  return_artifact_hash=excluded.return_artifact_hash""",
                (
                    task_id,
                    record["factor_id"],
                    record.get("cluster_id"),
                    int(bool(record.get("cluster_leader"))),
                    record["stability_score"],
                    _json(record.get("metrics", {})),
                    record.get("exclusion_reason"),
                    record.get("return_artifact_path"),
                    record.get("return_artifact_hash"),
                    _now(),
                ),
            )

    def factor_screen(self, task_id: str) -> list[dict[str, Any]]:
        with self.base.connection() as connection:
            rows = connection.execute(
                """SELECT * FROM quant_factor_screen WHERE task_id=?
                ORDER BY stability_score DESC, factor_id""",
                (task_id,),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["metrics"] = json.loads(item.pop("metrics_json"))
            item["cluster_leader"] = bool(item["cluster_leader"])
            result.append(item)
        return result

    def candidate_exists(self, task_id: str, candidate_hash: str) -> bool:
        with self.base.connection() as connection:
            row = connection.execute(
                "SELECT 1 FROM quant_combine_candidates WHERE task_id=? AND candidate_hash=?",
                (task_id, candidate_hash),
            ).fetchone()
        return row is not None

    def candidate_by_hash(self, task_id: str, candidate_hash: str) -> dict[str, Any] | None:
        with self.base.connection() as connection:
            row = connection.execute(
                """SELECT * FROM quant_combine_candidates
                WHERE task_id=? AND candidate_hash=?""",
                (task_id, candidate_hash),
            ).fetchone()
        return self._candidate(row) if row is not None else None

    def record_candidate(self, task_id: str, record: dict[str, Any]) -> dict[str, Any]:
        with self.base.connection() as connection:
            cursor = connection.execute(
                """INSERT INTO quant_combine_candidates
                (task_id, iteration, stage, algorithm, action, candidate_hash,
                 parent_ids_json, factor_ids_json, weights_json, metrics_json, objectives_json,
                 score, gate_distance, gate_status, failed_gates_json, pareto_rank,
                 crowding_distance, qualification, return_artifact_path,
                 return_artifact_hash, duration_seconds, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    task_id,
                    record["iteration"],
                    record["stage"],
                    record["algorithm"],
                    record["action"],
                    record["candidate_hash"],
                    _json(record.get("parent_ids", [])),
                    _json(record["factor_ids"]),
                    _json(record["weights"]),
                    _json(record["metrics"]),
                    _json(record["objectives"]),
                    record["score"],
                    record["gate_distance"],
                    record["gate_status"],
                    _json(record.get("failed_gates", [])),
                    record.get("pareto_rank"),
                    record.get("crowding_distance"),
                    record.get("qualification", "EVALUATED"),
                    record.get("return_artifact_path"),
                    record.get("return_artifact_hash"),
                    record.get("duration_seconds"),
                    _now(),
                ),
            )
            candidate_id = int(cursor.lastrowid)
        candidate = self.candidate(candidate_id)
        assert candidate is not None
        return candidate

    def candidates(self, task_id: str, *, limit: int = 2000) -> list[dict[str, Any]]:
        with self.base.connection() as connection:
            rows = connection.execute(
                """SELECT * FROM quant_combine_candidates WHERE task_id=?
                ORDER BY id DESC LIMIT ?""",
                (task_id, min(max(1, limit), 10000)),
            ).fetchall()
        return [self._candidate(row) for row in rows]

    def candidate(self, candidate_id: int) -> dict[str, Any] | None:
        with self.base.connection() as connection:
            row = connection.execute(
                "SELECT * FROM quant_combine_candidates WHERE id=?", (candidate_id,)
            ).fetchone()
        return self._candidate(row) if row is not None else None

    def best_candidate_references(
        self, *, exclude_task_id: str | None = None, limit: int = 25
    ) -> list[dict[str, Any]]:
        parameters: list[Any] = []
        where = ["task.best_candidate_id IS NOT NULL", "candidate.return_artifact_path IS NOT NULL"]
        if exclude_task_id:
            where.append("task.task_id != ?")
            parameters.append(exclude_task_id)
        parameters.append(min(max(1, int(limit)), 200))
        with self.base.connection() as connection:
            rows = connection.execute(
                f"""SELECT candidate.*, task.name AS task_name, task.status AS task_status,
                    task.updated_at AS task_updated_at
                FROM quant_combine_tasks task
                JOIN quant_combine_candidates candidate ON candidate.id=task.best_candidate_id
                WHERE {' AND '.join(where)}
                ORDER BY datetime(task.updated_at) DESC LIMIT ?""",  # noqa: S608
                tuple(parameters),
            ).fetchall()
        return [self._candidate(row) for row in rows]

    @staticmethod
    def _candidate(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        for key in ("parent_ids", "factor_ids", "weights", "metrics", "objectives", "failed_gates"):
            item[key] = json.loads(item.pop(f"{key}_json"))
        return item

    def update_candidate(self, candidate_id: int, **values: Any) -> dict[str, Any]:
        allowed = {
            "metrics",
            "score",
            "gate_distance",
            "gate_status",
            "failed_gates",
            "pareto_rank",
            "crowding_distance",
            "qualification",
        }
        invalid = set(values) - allowed
        if invalid:
            raise ValueError(f"Unknown QuantCombine candidate fields: {sorted(invalid)}")
        encoded: dict[str, Any] = {}
        for key, value in values.items():
            target = f"{key}_json" if key in {"metrics", "failed_gates"} else key
            encoded[target] = _json(value) if key in {"metrics", "failed_gates"} else value
        assignments = ", ".join(f"{key}=?" for key in encoded)
        with self.base.connection() as connection:
            cursor = connection.execute(
                f"UPDATE quant_combine_candidates SET {assignments} WHERE id=?",  # noqa: S608
                (*encoded.values(), candidate_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"QuantCombine candidate not found: {candidate_id}")
        candidate = self.candidate(candidate_id)
        assert candidate is not None
        return candidate

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
                """INSERT INTO quant_combine_events
                (task_id, category, level, event, title, message, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (task_id, category, level, event, title, message, _json(payload or {}), _now()),
            )

    def events(self, task_id: str | None = None, *, limit: int = 300) -> list[dict[str, Any]]:
        where = "WHERE task_id=?" if task_id else ""
        parameters: tuple[Any, ...] = (task_id, limit) if task_id else (limit,)
        with self.base.connection() as connection:
            rows = connection.execute(
                f"SELECT * FROM quant_combine_events {where} ORDER BY id DESC LIMIT ?",  # noqa: S608
                parameters,
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item.pop("payload_json"))
            result.append(item)
        return result

    def promote_strategy(self, task_id: str, candidate_id: int, name: str) -> dict[str, Any]:
        task = self.task(task_id)
        candidate = self.candidate(candidate_id)
        if task is None or candidate is None or candidate["task_id"] != task_id:
            raise KeyError("QuantCombine task or candidate not found")
        if task.get("production_candidate_id") != candidate_id:
            raise ValueError("Only the isolated-blind-test production candidate can be promoted")
        strategy_id = f"QS_{hashlib.sha256(task_id.encode()).hexdigest()[:12]}"
        with self.base.connection() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(version), 0) FROM quant_strategy_versions WHERE strategy_id=?",
                (strategy_id,),
            ).fetchone()
            version = int(row[0]) + 1
            specification = {
                "strategy_id": strategy_id,
                "version": version,
                "name": name.strip(),
                "market": task["market"],
                "source": "QUANTCOMBINE",
                "factor_snapshot_hash": task["snapshot_hash"],
                "factor_ids": candidate["factor_ids"],
                "factor_weights": candidate["weights"],
                "portfolio_mode": "long_only",
                "protocol": task["protocol"],
                "construction": task["construction"],
                "engine": task["engine"],
                "evaluation": candidate["metrics"],
                "execution": {
                    "signal_time": "END_OF_DAY_AFTER_CLOSE",
                    "execution_lag_sessions": 1,
                    "rebalance": "WEEKLY_FIRST_SESSION",
                    "engine": "A_SHARE_LONG_ONLY_WEEKLY_VECTOR_PROXY_V1",
                },
            }
            evidence_hash = hashlib.sha256(_json(specification).encode()).hexdigest()
            cursor = connection.execute(
                """INSERT INTO quant_strategy_versions
                (strategy_id, version, source_task_id, source_candidate_id, name, market,
                 lifecycle, specification_json, evidence_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?, 'QUALIFIED', ?, ?, ?)""",
                (
                    strategy_id,
                    version,
                    task_id,
                    candidate_id,
                    name.strip(),
                    task["market"],
                    _json(specification),
                    evidence_hash,
                    _now(),
                ),
            )
            row_id = int(cursor.lastrowid)
        return self.strategy(row_id)

    def strategies(self) -> list[dict[str, Any]]:
        with self.base.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM quant_strategy_versions ORDER BY id DESC"
            ).fetchall()
        return [self._strategy(row) for row in rows]

    def strategy(self, row_id: int) -> dict[str, Any]:
        with self.base.connection() as connection:
            row = connection.execute(
                "SELECT * FROM quant_strategy_versions WHERE id=?", (row_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Quant strategy version not found: {row_id}")
        return self._strategy(row)

    @staticmethod
    def _strategy(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["specification"] = json.loads(item.pop("specification_json"))
        return item
