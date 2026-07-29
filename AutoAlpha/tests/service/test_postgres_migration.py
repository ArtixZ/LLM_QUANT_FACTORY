from __future__ import annotations

from autoalpha.service.postgres_migration import (
    generate_postgres_schema,
    postgres_type,
    sqlite_catalog,
)
from autoalpha.service.store import ServiceStore


def test_sqlite_catalog_generates_postgres_schema_for_service_store(tmp_path) -> None:
    store = ServiceStore(tmp_path / "autoalpha.sqlite3")
    store.enqueue_system_job(
        job_id="job-migrate",
        queue="system",
        job_type="factor_knowledge_map_sync",
        payload={"hello": "world"},
    )

    tables = sqlite_catalog(store.path)
    schema = generate_postgres_schema(tables)
    table_names = {table.name for table in tables}
    system_jobs = next(table for table in tables if table.name == "system_jobs")

    assert {"system_jobs", "system_job_logs", "factor_pool"} <= table_names
    assert system_jobs.primary_key_columns == ("job_id",)
    assert '"system_jobs"' in schema
    assert '"payload_json" TEXT NOT NULL' in schema
    assert 'PRIMARY KEY ("job_id")' in schema
    assert "CREATE TABLE IF NOT EXISTS" in schema


def test_postgres_type_translation_keeps_runtime_payloads_portable() -> None:
    assert postgres_type("INTEGER") == "BIGINT"
    assert postgres_type("REAL") == "DOUBLE PRECISION"
    assert postgres_type("TEXT") == "TEXT"
    assert postgres_type("") == "TEXT"
