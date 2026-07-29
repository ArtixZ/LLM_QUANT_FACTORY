from __future__ import annotations

import pytest

from autoalpha.service.system_job_sql import system_job_sql


def test_sqlite_system_job_sql_uses_qmark_placeholders() -> None:
    dialect = system_job_sql("sqlite")

    assert dialect.placeholder(1) == "?"
    assert dialect.placeholders(3) == "?, ?, ?"
    assert "VALUES (?, ?, ?, 'QUEUED', ?, ?, ?, ?, ?, ?, ?, ?)" in (
        dialect.enqueue_job_sql()
    )
    assert dialect.update_job_sql(["status", "updated_at"]) == (
        "UPDATE system_jobs SET status=?, updated_at=? WHERE job_id=?"
    )
    assert "job_id IN (?, ?)" in dialect.logs_for_jobs_sql(2)
    assert "ORDER BY priority ASC, created_at ASC LIMIT 50" in (
        dialect.claim_candidates_sql("queue=?")
    )
    assert "queue=? AND resource_group=?" in dialect.resource_capacity_sql()
    assert "job_id<>?" in dialect.active_resource_sql()
    assert "WHERE queue=?" in dialect.active_queue_sql()
    assert "WHERE status='RUNNING'" in dialect.active_global_sql()
    assert "lease_owner=?" in dialect.claim_update_sql()


def test_postgres_system_job_sql_uses_psycopg_placeholders() -> None:
    dialect = system_job_sql("postgresql")

    assert dialect.placeholder(1) == "%s"
    assert dialect.placeholders(3) == "%s, %s, %s"
    assert "VALUES (%s, %s, %s, 'QUEUED', %s, %s, %s, %s, %s, %s, %s, %s)" in (
        dialect.enqueue_job_sql()
    )
    assert dialect.update_job_sql(["status", "updated_at"]) == (
        "UPDATE system_jobs SET status=%s, updated_at=%s WHERE job_id=%s"
    )
    assert "job_id IN (%s, %s, %s)" in dialect.logs_for_jobs_sql(3)
    assert "FOR UPDATE SKIP LOCKED" in dialect.claim_candidates_sql("queue=%s")
    assert "queue=%s AND resource_group=%s" in dialect.resource_capacity_sql()
    assert "job_id<>%s" in dialect.active_resource_sql()
    assert "WHERE queue=%s" in dialect.active_queue_sql()
    assert "job_id<>%s" in dialect.active_global_sql()
    assert "lease_owner=%s" in dialect.claim_update_sql()


def test_system_job_sql_rejects_empty_dynamic_lists() -> None:
    dialect = system_job_sql("sqlite")

    with pytest.raises(ValueError, match="placeholder count"):
        dialect.placeholders(0)
    with pytest.raises(ValueError, match="assignment column"):
        dialect.assignments([])
