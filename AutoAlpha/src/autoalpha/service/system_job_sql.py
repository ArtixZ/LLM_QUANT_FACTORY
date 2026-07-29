from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

SystemJobSqlBackend = Literal["sqlite", "postgresql"]


@dataclass(frozen=True)
class SystemJobSql:
    """Small SQL dialect boundary for the Job Center hot path."""

    backend: SystemJobSqlBackend = "sqlite"

    def placeholder(self, index: int) -> str:
        del index
        if self.backend == "postgresql":
            return "%s"
        return "?"

    def placeholders(self, count: int, *, start: int = 1) -> str:
        if count <= 0:
            raise ValueError("placeholder count must be positive")
        return ", ".join(self.placeholder(start + offset) for offset in range(count))

    def assignments(self, columns: list[str], *, start: int = 1) -> str:
        if not columns:
            raise ValueError("at least one assignment column is required")
        return ", ".join(
            f"{column}={self.placeholder(start + offset)}"
            for offset, column in enumerate(columns)
        )

    def enqueue_job_sql(self) -> str:
        return f"""INSERT INTO system_jobs
                (job_id, queue, job_type, status, priority, resource_group, max_workers,
                 progress_total, payload_json, max_attempts, created_at, updated_at)
                VALUES ({self.placeholders(3)}, 'QUEUED', {self.placeholders(8, start=4)})"""

    def update_job_sql(self, columns: list[str]) -> str:
        assignment_sql = self.assignments(columns)
        job_id_placeholder = self.placeholder(len(columns) + 1)
        return f"UPDATE system_jobs SET {assignment_sql} WHERE job_id={job_id_placeholder}"

    def select_job_by_id_sql(self, *, projection: str = "*") -> str:
        return f"SELECT {projection} FROM system_jobs WHERE job_id={self.placeholder(1)}"

    def list_jobs_sql(
        self,
        *,
        queue: bool = False,
        status: bool = False,
        limit_index: int | None = None,
    ) -> str:
        clauses: list[str] = []
        next_index = 1
        if queue:
            clauses.append(f"queue={self.placeholder(next_index)}")
            next_index += 1
        if status:
            clauses.append(f"status={self.placeholder(next_index)}")
            next_index += 1
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        placeholder_index = limit_index or next_index
        return (
            f"SELECT * FROM system_jobs {where} "
            f"ORDER BY priority ASC, created_at DESC LIMIT {self.placeholder(placeholder_index)}"
        )

    def claim_candidates_sql(self, where: str) -> str:
        lock_clause = " FOR UPDATE SKIP LOCKED" if self.backend == "postgresql" else ""
        return f"""SELECT * FROM system_jobs WHERE {where}
                ORDER BY priority ASC, created_at ASC LIMIT 50{lock_clause}"""

    def resource_capacity_sql(self) -> str:
        return f"""SELECT MIN(max_workers) FROM system_jobs
                    WHERE queue={self.placeholder(1)} AND resource_group={self.placeholder(2)}
                      AND status IN ('QUEUED', 'RUNNING')"""

    def active_resource_sql(self) -> str:
        return f"""SELECT COUNT(*) FROM system_jobs
                    WHERE queue={self.placeholder(1)}
                      AND resource_group={self.placeholder(2)}
                      AND status='RUNNING'
                      AND job_id<>{self.placeholder(3)}
                      AND (lease_expires_at IS NULL OR lease_expires_at >= {self.placeholder(4)})"""

    def active_queue_sql(self) -> str:
        return f"""SELECT COUNT(*) FROM system_jobs
                    WHERE queue={self.placeholder(1)}
                      AND status='RUNNING'
                      AND job_id<>{self.placeholder(2)}
                      AND (lease_expires_at IS NULL OR lease_expires_at >= {self.placeholder(3)})"""

    def active_global_sql(self) -> str:
        return f"""SELECT COUNT(*) FROM system_jobs
                    WHERE status='RUNNING'
                      AND job_id<>{self.placeholder(1)}
                      AND (lease_expires_at IS NULL OR lease_expires_at >= {self.placeholder(2)})"""

    def claim_update_sql(self) -> str:
        return f"""UPDATE system_jobs
                    SET status='RUNNING',
                        attempts=attempts + 1,
                        lease_owner={self.placeholder(1)},
                        lease_expires_at={self.placeholder(2)},
                        heartbeat_at={self.placeholder(3)},
                        started_at=COALESCE(started_at, {self.placeholder(4)}),
                        updated_at={self.placeholder(5)}
                    WHERE job_id={self.placeholder(6)}"""

    def recover_expired_select_sql(self, where: str) -> str:
        return f"SELECT job_id, attempts, max_attempts FROM system_jobs WHERE {where}"

    def recover_expired_update_sql(self, where: str) -> str:
        return f"""UPDATE system_jobs
                SET status=CASE WHEN attempts >= max_attempts THEN 'FAILED' ELSE 'QUEUED' END,
                    error=CASE
                        WHEN attempts >= max_attempts THEN 'lease expired after maximum attempts'
                        ELSE error
                    END,
                    lease_owner=NULL,
                    lease_expires_at=NULL,
                    updated_at={self.placeholder(1)}
                WHERE {where}"""

    def logs_for_jobs_sql(self, job_count: int) -> str:
        return f"""SELECT * FROM system_job_logs
                WHERE job_id IN ({self.placeholders(job_count)})
                ORDER BY job_id ASC, id DESC"""

    def insert_log_sql(self) -> str:
        return f"""INSERT INTO system_job_logs
            (job_id, timestamp_utc, level, event, message, payload_json)
            VALUES ({self.placeholders(6)})"""


def system_job_sql(backend: SystemJobSqlBackend = "sqlite") -> SystemJobSql:
    return SystemJobSql(backend=backend)


def normalize_system_job_where(
    dialect: SystemJobSql,
    clauses: list[str],
    parameters: list[Any],
    *,
    extra_limit: int | None = None,
) -> tuple[str, tuple[Any, ...]]:
    del dialect
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    values = list(parameters)
    if extra_limit is not None:
        values.append(extra_limit)
    return where, tuple(values)
