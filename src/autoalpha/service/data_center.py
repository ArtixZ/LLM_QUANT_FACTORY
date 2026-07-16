from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from autoalpha.data.execution_basis import inspect_execution_data_basis
from autoalpha.data.workspace import inspect_data_workspace


def build_data_center_snapshot(
    settings: Mapping[str, str],
    *,
    sync_status: Mapping[str, Any],
    token_configured: bool,
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return operational data status without ever returning credential material."""
    data_path = Path(settings.get("data_path", "")).expanduser()
    market_data_root = Path(settings.get("market_data_root", "")).expanduser()
    workspace: dict[str, Any] | None = None
    execution_basis: dict[str, Any] | None = None
    workspace_error: str | None = None
    try:
        report = inspect_data_workspace(data_path)
        workspace = report.to_dict()
        execution_basis = inspect_execution_data_basis(Path(report.panel_path)).to_dict()
    except (FileNotFoundError, RuntimeError, TypeError, ValueError, OSError) as error:
        workspace_error = f"{type(error).__name__}: {error}"

    return {
        "workspace": workspace,
        "workspace_error": workspace_error,
        "execution_basis": execution_basis,
        "downloader": inspect_downloader(market_data_root),
        "credentials": {"tushare_token_configured": token_configured},
        "schedule": {
            "enabled": settings.get("data_auto_update_enabled", "false").casefold() == "true",
            "hour": _integer(settings.get("data_update_hour"), 18),
            "last_sync_date": settings.get("last_data_sync_date"),
        },
        "sync": dict(sync_status),
        "recent_events": [
            event
            for event in events
            if str(event.get("event", "")).startswith("MARKET_DATA_SYNC")
            or str(event.get("event", "")) == "DATA_CENTER_SETTINGS_UPDATED"
        ][:20],
    }


def inspect_downloader(root: Path) -> dict[str, Any]:
    root = root.expanduser()
    cli = root / "sync_cli.py"
    downloads = root / "data" / "downloads"
    tasks = []
    if downloads.is_dir():
        for task in sorted(downloads.glob("a_daily_*_csv-parquet"), key=_mtime, reverse=True):
            parquet = task / "parquet"
            if not parquet.is_dir():
                continue
            adjustment = (
                "qfq"
                if "_qfq_" in task.name
                else "none"
                if "_none_" in task.name
                else "unknown"
            )
            tasks.append(
                {
                    "name": task.name,
                    "adjustment": adjustment,
                    "parquet_files": sum(1 for _ in parquet.glob("*.parquet")),
                    "updated_at": datetime.fromtimestamp(task.stat().st_mtime)
                    .astimezone()
                    .isoformat(),
                }
            )
        cross = downloads / "a_daily_cross_sectional_raw_adj"
        market = cross / "market_parquet"
        factors = cross / "adj_factor_parquet"
        if market.is_dir() and factors.is_dir():
            tasks.append(
                {
                    "name": cross.name,
                    "adjustment": "raw_plus_adj_factor",
                    "parquet_files": sum(1 for _ in market.glob("*.parquet"))
                    + sum(1 for _ in factors.glob("*.parquet")),
                    "updated_at": datetime.fromtimestamp(cross.stat().st_mtime)
                    .astimezone()
                    .isoformat(),
                }
            )
    return {
        "root_path": str(root.resolve()) if root.exists() else str(root),
        "root_exists": root.is_dir(),
        "sync_cli_available": cli.is_file(),
        "python_available": (root / ".venv" / "bin" / "python").is_file(),
        "download_tasks": tasks[:12],
        "qfq_available": any(task["adjustment"] == "qfq" for task in tasks),
        "raw_available": any(task["adjustment"] == "none" for task in tasks),
        "cross_sectional_available": any(
            task["adjustment"] == "raw_plus_adj_factor" for task in tasks
        ),
    }


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _integer(value: Any, fallback: int) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return fallback


def read_json(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None
