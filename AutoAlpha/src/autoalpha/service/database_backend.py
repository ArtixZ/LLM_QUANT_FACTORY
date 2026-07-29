from __future__ import annotations

import importlib.util
import os
from dataclasses import asdict, dataclass
from typing import Any, Literal
from urllib.parse import urlparse

DatabaseBackend = Literal["sqlite", "postgresql"]


@dataclass(frozen=True)
class DatabaseRuntimeConfig:
    backend: DatabaseBackend
    url: str | None
    url_configured: bool
    driver_available: bool
    job_center_adapter_available: bool
    migration_stage: str
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["url"] = _mask_database_url(self.url) if self.url else None
        payload["blockers"] = list(self.blockers)
        payload["migration_utility"] = {
            "schema_command": (
                "uv run python scripts/migrate-sqlite-to-postgres.py "
                "--schema-only --sqlite runtime-full-llm/autoalpha.sqlite3"
            ),
            "copy_command": (
                "AUTOALPHA_DATABASE_URL=postgresql://... "
                "uv run python scripts/migrate-sqlite-to-postgres.py "
                "--sqlite runtime-full-llm/autoalpha.sqlite3 --truncate"
            ),
        }
        payload["adapter_capabilities"] = {
            "system_jobs": (
                "postgresql_available"
                if self.job_center_adapter_available
                else "sqlite_only"
            ),
            "materialized_snapshots": (
                "postgresql_available"
                if self.job_center_adapter_available
                else "sqlite_only"
            ),
            "strategy_experiments": (
                "postgresql_available"
                if self.job_center_adapter_available
                else "sqlite_only"
            ),
            "formal_strategy_versions": (
                "postgresql_available"
                if self.job_center_adapter_available
                else "sqlite_only"
            ),
            "factor_knowledge": (
                "postgresql_available"
                if self.job_center_adapter_available
                else "sqlite_only"
            ),
            "factor_pool": (
                "postgresql_available"
                if self.job_center_adapter_available
                else "sqlite_only"
            ),
            "settings": (
                "postgresql_available"
                if self.job_center_adapter_available
                else "sqlite_only"
            ),
            "events": (
                "postgresql_available"
                if self.job_center_adapter_available
                else "sqlite_only"
            ),
            "research_tasks": (
                "postgresql_available"
                if self.job_center_adapter_available
                else "sqlite_only"
            ),
            "iterations": (
                "postgresql_available"
                if self.job_center_adapter_available
                else "sqlite_only"
            ),
            "llm_role_artifacts": (
                "postgresql_available"
                if self.job_center_adapter_available
                else "sqlite_only"
            ),
            "service_store": "sqlite_active",
        }
        return payload


def database_runtime_config() -> DatabaseRuntimeConfig:
    raw_backend = os.getenv("AUTOALPHA_DATABASE_BACKEND", "sqlite").strip().casefold()
    backend: DatabaseBackend = (
        "postgresql" if raw_backend in {"postgres", "postgresql"} else "sqlite"
    )
    url = os.getenv("AUTOALPHA_DATABASE_URL")
    driver_available = _postgres_driver_available() if backend == "postgresql" else True
    job_center_adapter_available = backend == "postgresql" and driver_available
    blockers: list[str] = []
    if backend == "postgresql":
        if not url:
            blockers.append("AUTOALPHA_DATABASE_URL is required for PostgreSQL backend")
        elif not _is_postgres_url(url):
            blockers.append("AUTOALPHA_DATABASE_URL must use postgresql:// or postgres://")
        if not driver_available:
            blockers.append("psycopg driver is not installed")
        blockers.append("Full PostgreSQL ServiceStore adapter migration is not enabled yet")
    return DatabaseRuntimeConfig(
        backend=backend,
        url=url,
        url_configured=bool(url),
        driver_available=driver_available,
        job_center_adapter_available=job_center_adapter_available,
        migration_stage=(
            "SQLITE_ACTIVE_POSTGRES_MIGRATOR_AVAILABLE"
            if backend == "sqlite"
            else "POSTGRES_JOB_CENTER_ADAPTER_AVAILABLE_SERVICE_STORE_PENDING"
        ),
        blockers=tuple(blockers),
    )


def _postgres_driver_available() -> bool:
    return importlib.util.find_spec("psycopg") is not None


def _is_postgres_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"postgresql", "postgres"} and bool(parsed.netloc)


def _mask_database_url(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.netloc:
        return "***"
    host = parsed.hostname or "unknown"
    port = f":{parsed.port}" if parsed.port else ""
    database = parsed.path or ""
    return f"{parsed.scheme}://***@{host}{port}{database}"
