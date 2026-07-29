from __future__ import annotations

from autoalpha.service.database_backend import database_runtime_config


def test_sqlite_is_default_database_backend(monkeypatch) -> None:
    monkeypatch.delenv("AUTOALPHA_DATABASE_BACKEND", raising=False)
    monkeypatch.delenv("AUTOALPHA_DATABASE_URL", raising=False)

    config = database_runtime_config()

    assert config.backend == "sqlite"
    assert config.blockers == ()
    assert config.to_dict()["url"] is None
    assert config.migration_stage == "SQLITE_ACTIVE_POSTGRES_MIGRATOR_AVAILABLE"
    assert config.to_dict()["adapter_capabilities"]["system_jobs"] == "sqlite_only"
    assert config.to_dict()["adapter_capabilities"]["strategy_experiments"] == "sqlite_only"
    assert config.to_dict()["adapter_capabilities"]["formal_strategy_versions"] == (
        "sqlite_only"
    )
    assert config.to_dict()["adapter_capabilities"]["factor_knowledge"] == "sqlite_only"
    assert config.to_dict()["adapter_capabilities"]["settings"] == "sqlite_only"
    assert config.to_dict()["adapter_capabilities"]["events"] == "sqlite_only"
    assert config.to_dict()["adapter_capabilities"]["research_tasks"] == "sqlite_only"
    assert config.to_dict()["adapter_capabilities"]["iterations"] == "sqlite_only"
    assert config.to_dict()["adapter_capabilities"]["llm_role_artifacts"] == "sqlite_only"
    assert "migrate-sqlite-to-postgres.py" in config.to_dict()["migration_utility"][
        "schema_command"
    ]


def test_postgres_backend_reports_masked_url_and_adapter_blocker(monkeypatch) -> None:
    monkeypatch.setenv("AUTOALPHA_DATABASE_BACKEND", "postgresql")
    monkeypatch.setenv(
        "AUTOALPHA_DATABASE_URL",
        "postgresql://user:secret@localhost:5432/autoalpha",
    )

    config = database_runtime_config()
    payload = config.to_dict()

    assert config.backend == "postgresql"
    assert payload["url"] == "postgresql://***@localhost:5432/autoalpha"
    assert "Full PostgreSQL ServiceStore adapter migration is not enabled yet" in payload[
        "blockers"
    ]
    assert config.migration_stage == (
        "POSTGRES_JOB_CENTER_ADAPTER_AVAILABLE_SERVICE_STORE_PENDING"
    )
    assert payload["adapter_capabilities"]["system_jobs"] == "postgresql_available"
    assert payload["adapter_capabilities"]["materialized_snapshots"] == (
        "postgresql_available"
    )
    assert payload["adapter_capabilities"]["strategy_experiments"] == (
        "postgresql_available"
    )
    assert payload["adapter_capabilities"]["formal_strategy_versions"] == (
        "postgresql_available"
    )
    assert payload["adapter_capabilities"]["factor_knowledge"] == "postgresql_available"
    assert payload["adapter_capabilities"]["factor_pool"] == "postgresql_available"
    assert payload["adapter_capabilities"]["settings"] == "postgresql_available"
    assert payload["adapter_capabilities"]["events"] == "postgresql_available"
    assert payload["adapter_capabilities"]["research_tasks"] == "postgresql_available"
    assert payload["adapter_capabilities"]["iterations"] == "postgresql_available"
    assert payload["adapter_capabilities"]["llm_role_artifacts"] == "postgresql_available"


def test_postgres_backend_requires_postgres_url(monkeypatch) -> None:
    monkeypatch.setenv("AUTOALPHA_DATABASE_BACKEND", "postgresql")
    monkeypatch.setenv("AUTOALPHA_DATABASE_URL", "sqlite:///runtime.sqlite3")

    payload = database_runtime_config().to_dict()

    assert "AUTOALPHA_DATABASE_URL must use postgresql:// or postgres://" in payload["blockers"]
