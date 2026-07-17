from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class BatchBacktestStore:
    """Durable job and factor checkpoints for the isolated batch service."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=60)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA busy_timeout=60000")
        return connection

    def _initialize(self) -> None:
        with self.connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS batch_jobs (
                    job_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    source_database TEXT NOT NULL,
                    factor_snapshot_hash TEXT NOT NULL,
                    data_fingerprint TEXT,
                    config_json TEXT NOT NULL,
                    factor_count INTEGER NOT NULL,
                    completed_count INTEGER NOT NULL DEFAULT 0,
                    failed_count INTEGER NOT NULL DEFAULT 0,
                    started_at TEXT,
                    finished_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_error TEXT
                );
                CREATE TABLE IF NOT EXISTS factor_snapshots (
                    job_id TEXT NOT NULL REFERENCES batch_jobs(job_id),
                    factor_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    family TEXT NOT NULL,
                    source_task_id TEXT,
                    source_iteration INTEGER,
                    source_status TEXT,
                    proposal_json TEXT NOT NULL,
                    PRIMARY KEY(job_id, factor_id)
                );
                CREATE TABLE IF NOT EXISTS factor_results (
                    job_id TEXT NOT NULL REFERENCES batch_jobs(job_id),
                    factor_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    elapsed_seconds REAL,
                    metrics_json TEXT,
                    monte_carlo_json TEXT,
                    curve_path TEXT,
                    monte_carlo_path TEXT,
                    started_at TEXT,
                    finished_at TEXT,
                    error TEXT,
                    PRIMARY KEY(job_id, factor_id)
                );
                CREATE TABLE IF NOT EXISTS window_results (
                    job_id TEXT NOT NULL,
                    factor_id TEXT NOT NULL,
                    window_id TEXT NOT NULL,
                    period_start TEXT NOT NULL,
                    period_end TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    PRIMARY KEY(job_id, factor_id, window_id)
                );
                CREATE TABLE IF NOT EXISTS robustness_results (
                    job_id TEXT NOT NULL,
                    factor_id TEXT NOT NULL,
                    test_type TEXT NOT NULL,
                    variant TEXT NOT NULL,
                    metrics_json TEXT,
                    error TEXT,
                    PRIMARY KEY(job_id, factor_id, test_type, variant)
                );
                CREATE TABLE IF NOT EXISTS batch_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    level TEXT NOT NULL,
                    event TEXT NOT NULL,
                    message TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_batch_jobs_created
                ON batch_jobs(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_factor_results_status
                ON factor_results(job_id, status, finished_at);
                CREATE INDEX IF NOT EXISTS idx_batch_events_job
                ON batch_events(job_id, id DESC);
                """
            )

    def recover_interrupted(self) -> int:
        now = _now()
        with self.connection() as connection:
            jobs = connection.execute(
                "SELECT job_id FROM batch_jobs WHERE status IN ('RUNNING', 'PAUSING')"
            ).fetchall()
            connection.execute(
                """UPDATE batch_jobs SET status='PAUSED', phase='INTERRUPTED', updated_at=?,
                last_error='Batch service restarted; resume from factor checkpoint.'
                WHERE status IN ('RUNNING', 'PAUSING')""",
                (now,),
            )
            connection.execute(
                """DELETE FROM factor_results WHERE status='RUNNING' AND job_id IN
                (SELECT job_id FROM batch_jobs WHERE status='PAUSED')"""
            )
        return len(jobs)

    def create_job(
        self,
        *,
        name: str,
        config: dict[str, Any],
        source_database: Path,
    ) -> dict[str, Any]:
        records = _read_factor_library(source_database)
        if not records:
            raise RuntimeError(f"No factors found in source database: {source_database}")
        snapshot_hash = _hash_value(
            [
                {
                    "factor_id": item["factor_id"],
                    "proposal": json.loads(item["proposal_json"]),
                }
                for item in records
            ]
        )
        job_id = f"batch-{uuid.uuid4().hex[:12]}"
        now = _now()
        with self.connection() as connection:
            connection.execute(
                """INSERT INTO batch_jobs
                (job_id, name, status, phase, source_database, factor_snapshot_hash,
                 config_json, factor_count, created_at, updated_at)
                VALUES (?, ?, 'READY', 'WAITING', ?, ?, ?, ?, ?, ?)""",
                (
                    job_id,
                    name.strip(),
                    str(source_database),
                    snapshot_hash,
                    _json(config),
                    len(records),
                    now,
                    now,
                ),
            )
            connection.executemany(
                """INSERT INTO factor_snapshots
                (job_id, factor_id, name, family, source_task_id, source_iteration,
                 source_status, proposal_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        job_id,
                        item["factor_id"],
                        item["name"],
                        item["family"],
                        item["source_task_id"],
                        item["source_iteration"],
                        item["status"],
                        item["proposal_json"],
                    )
                    for item in records
                ],
            )
        self.append_event(
            job_id,
            "INFO",
            "JOB_CREATED",
            f"Frozen {len(records)} factor definitions for batch evaluation.",
            {"factor_snapshot_hash": snapshot_hash, "source_database": str(source_database)},
        )
        return self.job(job_id)

    def jobs(self) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM batch_jobs ORDER BY created_at DESC"
            ).fetchall()
        return [_job_record(row) for row in rows]

    def job(self, job_id: str) -> dict[str, Any]:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM batch_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Batch job not found: {job_id}")
        return _job_record(row)

    def update_job(self, job_id: str, **values: Any) -> dict[str, Any]:
        allowed = {
            "status",
            "phase",
            "data_fingerprint",
            "completed_count",
            "failed_count",
            "started_at",
            "finished_at",
            "last_error",
        }
        unknown = set(values) - allowed
        if unknown:
            raise KeyError(f"Unknown batch job columns: {sorted(unknown)}")
        values["updated_at"] = _now()
        assignments = ", ".join(f"{name}=?" for name in values)
        with self.connection() as connection:
            connection.execute(
                f"UPDATE batch_jobs SET {assignments} WHERE job_id=?",  # noqa: S608
                (*values.values(), job_id),
            )
        return self.job(job_id)

    def pending_factors(self, job_id: str) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """SELECT snapshot.* FROM factor_snapshots AS snapshot
                LEFT JOIN factor_results AS result
                  ON result.job_id=snapshot.job_id AND result.factor_id=snapshot.factor_id
                WHERE snapshot.job_id=? AND
                  (result.factor_id IS NULL OR result.status='RUNNING')
                ORDER BY snapshot.source_iteration, snapshot.factor_id""",
                (job_id,),
            ).fetchall()
        return [_factor_snapshot(row) for row in rows]

    def mark_factor_running(self, job_id: str, factor_id: str) -> None:
        now = _now()
        with self.connection() as connection:
            connection.execute(
                """INSERT INTO factor_results
                (job_id, factor_id, status, started_at)
                VALUES (?, ?, 'RUNNING', ?)
                ON CONFLICT(job_id, factor_id) DO UPDATE SET
                  status='RUNNING', started_at=excluded.started_at, finished_at=NULL, error=NULL""",
                (job_id, factor_id, now),
            )

    def complete_factor(
        self,
        job_id: str,
        factor_id: str,
        *,
        elapsed_seconds: float,
        metrics: dict[str, Any],
        monte_carlo: dict[str, Any],
        curve_path: str,
        monte_carlo_path: str,
        windows: list[dict[str, Any]],
        robustness: list[dict[str, Any]],
    ) -> None:
        now = _now()
        with self.connection() as connection:
            connection.execute(
                """UPDATE factor_results SET status='SUCCESS', elapsed_seconds=?,
                metrics_json=?, monte_carlo_json=?, curve_path=?, monte_carlo_path=?,
                finished_at=?, error=NULL WHERE job_id=? AND factor_id=?""",
                (
                    elapsed_seconds,
                    _json(metrics),
                    _json(monte_carlo),
                    curve_path,
                    monte_carlo_path,
                    now,
                    job_id,
                    factor_id,
                ),
            )
            connection.execute(
                "DELETE FROM window_results WHERE job_id=? AND factor_id=?",
                (job_id, factor_id),
            )
            connection.executemany(
                """INSERT INTO window_results
                (job_id, factor_id, window_id, period_start, period_end, metrics_json)
                VALUES (?, ?, ?, ?, ?, ?)""",
                [
                    (
                        job_id,
                        factor_id,
                        item["window_id"],
                        item["period_start"],
                        item["period_end"],
                        _json(item["metrics"]),
                    )
                    for item in windows
                ],
            )
            connection.execute(
                "DELETE FROM robustness_results WHERE job_id=? AND factor_id=?",
                (job_id, factor_id),
            )
            connection.executemany(
                """INSERT INTO robustness_results
                (job_id, factor_id, test_type, variant, metrics_json, error)
                VALUES (?, ?, ?, ?, ?, ?)""",
                [
                    (
                        job_id,
                        factor_id,
                        item["test_type"],
                        item["variant"],
                        _json(item["metrics"]) if item.get("metrics") is not None else None,
                        item.get("error"),
                    )
                    for item in robustness
                ],
            )
            self._refresh_counts(connection, job_id)

    def fail_factor(
        self, job_id: str, factor_id: str, *, elapsed_seconds: float, error: str
    ) -> None:
        with self.connection() as connection:
            connection.execute(
                """UPDATE factor_results SET status='FAILED', elapsed_seconds=?,
                finished_at=?, error=? WHERE job_id=? AND factor_id=?""",
                (elapsed_seconds, _now(), error, job_id, factor_id),
            )
            self._refresh_counts(connection, job_id)

    def _refresh_counts(self, connection: sqlite3.Connection, job_id: str) -> None:
        row = connection.execute(
            """SELECT COUNT(*) AS completed,
            SUM(CASE WHEN status='FAILED' THEN 1 ELSE 0 END) AS failed
            FROM factor_results WHERE job_id=? AND status IN ('SUCCESS', 'FAILED')""",
            (job_id,),
        ).fetchone()
        connection.execute(
            """UPDATE batch_jobs SET completed_count=?, failed_count=?, updated_at=?
            WHERE job_id=?""",
            (int(row["completed"]), int(row["failed"] or 0), _now(), job_id),
        )

    def successful_metrics(self, job_id: str) -> dict[str, dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """SELECT factor_id, metrics_json FROM factor_results
                WHERE job_id=? AND status='SUCCESS'""",
                (job_id,),
            ).fetchall()
        return {str(row["factor_id"]): json.loads(row["metrics_json"]) for row in rows}

    def update_adjusted_metrics(
        self, job_id: str, metrics_by_factor: dict[str, dict[str, Any]]
    ) -> None:
        with self.connection() as connection:
            connection.executemany(
                """UPDATE factor_results SET metrics_json=?
                WHERE job_id=? AND factor_id=? AND status='SUCCESS'""",
                [
                    (_json(metrics), job_id, factor_id)
                    for factor_id, metrics in metrics_by_factor.items()
                ],
            )

    def results(self, job_id: str, *, limit: int = 5000) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """SELECT snapshot.factor_id, snapshot.name, snapshot.family,
                snapshot.source_task_id, snapshot.source_iteration, snapshot.source_status,
                result.status, result.elapsed_seconds, result.metrics_json,
                result.monte_carlo_json, result.error, result.finished_at
                FROM factor_snapshots AS snapshot
                LEFT JOIN factor_results AS result
                  ON result.job_id=snapshot.job_id AND result.factor_id=snapshot.factor_id
                WHERE snapshot.job_id=?
                ORDER BY snapshot.source_iteration DESC LIMIT ?""",
                (job_id, min(max(1, limit), 10_000)),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["status"] = item["status"] or "PENDING"
            item["metrics"] = json.loads(item.pop("metrics_json")) if item["metrics_json"] else None
            item["monte_carlo"] = (
                json.loads(item.pop("monte_carlo_json")) if item["monte_carlo_json"] else None
            )
            result.append(item)
        return result

    def factor_detail(self, job_id: str, factor_id: str) -> dict[str, Any]:
        with self.connection() as connection:
            row = connection.execute(
                """SELECT snapshot.*, result.status, result.elapsed_seconds,
                result.metrics_json, result.monte_carlo_json, result.curve_path,
                result.monte_carlo_path, result.error, result.finished_at
                FROM factor_snapshots AS snapshot
                LEFT JOIN factor_results AS result
                  ON result.job_id=snapshot.job_id AND result.factor_id=snapshot.factor_id
                WHERE snapshot.job_id=? AND snapshot.factor_id=?""",
                (job_id, factor_id),
            ).fetchone()
            if row is None:
                raise KeyError(f"Factor not found in job: {factor_id}")
            windows = connection.execute(
                """SELECT * FROM window_results WHERE job_id=? AND factor_id=?
                ORDER BY period_start""",
                (job_id, factor_id),
            ).fetchall()
            robustness = connection.execute(
                """SELECT * FROM robustness_results WHERE job_id=? AND factor_id=?
                ORDER BY test_type, variant""",
                (job_id, factor_id),
            ).fetchall()
        item = dict(row)
        item["proposal"] = json.loads(item.pop("proposal_json"))
        item["status"] = item["status"] or "PENDING"
        item["metrics"] = json.loads(item.pop("metrics_json")) if item["metrics_json"] else None
        item["monte_carlo"] = (
            json.loads(item.pop("monte_carlo_json")) if item["monte_carlo_json"] else None
        )
        item["windows"] = [
            {**dict(window), "metrics": json.loads(window["metrics_json"])} for window in windows
        ]
        item["robustness"] = [
            {
                **dict(entry),
                "metrics": json.loads(entry["metrics_json"]) if entry["metrics_json"] else None,
            }
            for entry in robustness
        ]
        return item

    def append_event(
        self,
        job_id: str,
        level: str,
        event: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        with self.connection() as connection:
            connection.execute(
                """INSERT INTO batch_events
                (job_id, level, event, message, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (job_id, level, event, message, _json(payload or {}), _now()),
            )

    def events(self, job_id: str, *, limit: int = 100) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """SELECT * FROM batch_events WHERE job_id=?
                ORDER BY id DESC LIMIT ?""",
                (job_id, min(max(1, limit), 1000)),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item.pop("payload_json"))
            result.append(item)
        return result


def _read_factor_library(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """SELECT factor_id, name, family, source_task_id, source_iteration,
            status, proposal_json FROM factor_pool ORDER BY source_iteration, factor_id"""
        ).fetchall()
    finally:
        connection.close()
    return [dict(row) for row in rows]


def _job_record(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    item["config"] = json.loads(item.pop("config_json"))
    return item


def _factor_snapshot(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    item["proposal"] = json.loads(item.pop("proposal_json"))
    return item


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash_value(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat()
