from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from autoalpha.data.execution_basis import inspect_execution_data_basis
from autoalpha.data.tushare_catalog import DEFAULT_PRODUCT_IDS, data_product_catalog
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

    downloader = inspect_downloader(market_data_root)
    selected_products = _selected_products(settings.get("data_product_ids"))
    products = inspect_data_products(
        market_data_root,
        selected_products=selected_products,
        panel_columns=set(workspace.get("columns", [])) if workspace else set(),
        research_factor_fields=set(workspace.get("factor_fields", [])) if workspace else set(),
        panel_first_date=str(workspace.get("first_trade_date") or "") if workspace else "",
        panel_last_date=str(workspace.get("last_trade_date") or "") if workspace else "",
    )
    return {
        "workspace": workspace,
        "workspace_error": workspace_error,
        "execution_basis": execution_basis,
        "downloader": downloader,
        "data_products": products,
        "credentials": {"tushare_token_configured": token_configured},
        "schedule": {
            "enabled": settings.get("data_auto_update_enabled", "false").casefold() == "true",
            "hour": _integer(settings.get("data_update_hour"), 18),
            "last_sync_date": settings.get("last_data_sync_date"),
            "selected_product_ids": selected_products,
        },
        "sync": dict(sync_status),
        "recent_events": [
            event
            for event in events
            if str(event.get("event", "")).startswith("MARKET_DATA_SYNC")
            or str(event.get("event", "")) == "DATA_CENTER_SETTINGS_UPDATED"
        ][:20],
    }


def inspect_data_products(
    root: Path,
    *,
    selected_products: list[str],
    panel_columns: set[str],
    research_factor_fields: set[str] | None = None,
    panel_first_date: str = "",
    panel_last_date: str = "",
) -> dict[str, Any]:
    feature_root = root / "data" / "downloads" / "a_share_feature_store"
    state_root = root / "data" / "state"
    products = []
    for product in data_product_catalog():
        dataset_id = str(product["dataset_id"])
        selected = dataset_id in selected_products
        mandatory = dataset_id == "core_market"
        download_selectable = not mandatory and str(product["sync_strategy"]) != "CATALOG"
        if mandatory:
            selection_lock_reason = "核心行情由主数据流水线强制维护"
        elif not download_selectable:
            selection_lock_reason = "该接口尚未实现可恢复的下载契约"
        else:
            selection_lock_reason = None
        if dataset_id == "core_market":
            source = root / "data" / "downloads" / "a_daily_cross_sectional_raw_adj"
            state = read_json(str(state_root / "a_daily_cross_sectional_raw_adj.json")) or {}
            manifest = read_json(str(source / "market_legacy_source.json")) or {}
            files = sum(1 for _ in (source / "market_parquet").glob("*.parquet"))
            completed = len(state.get("completed_dates", []))
            failed = len(state.get("failed_dates", {}))
            first_date = manifest.get("first_trade_date") or state.get("migration_start_date")
            last_date = state.get("target_date")
        else:
            source = feature_root / dataset_id
            manifest = read_json(str(source / "_manifest.json")) or {}
            state = read_json(str(state_root / f"a_feature_{dataset_id}.json")) or {}
            files = sum(1 for _ in source.glob("*.parquet")) if source.is_dir() else 0
            completed = len(state.get("completed_dates", []))
            failed = len(state.get("failed_dates", {}))
            first_date = manifest.get("first_date")
            last_date = manifest.get("last_date") or state.get("target_date")
        panel_fields = [str(value) for value in product.get("panel_fields", [])]
        panel_available = [value for value in panel_fields if value in panel_columns]
        research_fields = research_factor_fields or set()
        research_contract_ready = bool(panel_fields) and set(panel_fields).issubset(
            research_fields
        )
        coverage_ready = bool(
            first_date
            and last_date
            and (not panel_first_date or str(first_date) <= panel_first_date.replace("-", ""))
            and (not panel_last_date or str(last_date) >= panel_last_date.replace("-", ""))
        )
        if failed:
            storage_state = "PARTIAL"
        elif completed or files:
            storage_state = "READY"
        elif product["integration_state"] == "CATALOG" and not download_selectable:
            storage_state = "CATALOG"
        else:
            storage_state = "NOT_DOWNLOADED"
        if product["integration_state"] == "LIVE":
            research_state = (
                "RESEARCH_READY"
                if coverage_ready or research_contract_ready
                else "WAITING_FOR_COVERAGE"
            )
        elif product["integration_state"] == "RAW_READY":
            research_state = "RAW_DATA_ONLY"
        elif download_selectable:
            research_state = "RAW_DOWNLOAD_ONLY_REQUIRES_PIT_INTEGRATION"
        else:
            research_state = "CATALOG_ONLY"
        products.append(
            {
                **product,
                "selected": selected,
                "mandatory": mandatory,
                "download_selectable": download_selectable,
                "selection_lock_reason": selection_lock_reason,
                "research_state": research_state,
                "storage_state": storage_state,
                "storage_root": str(source),
                "files": files,
                "completed": completed,
                "failed": failed,
                "first_date": first_date,
                "last_date": last_date,
                "panel_available_fields": panel_available,
                "panel_ready": (
                    bool(panel_fields)
                    and set(panel_fields).issubset(panel_columns)
                    and (coverage_ready or research_contract_ready)
                ),
                "research_contract_ready": research_contract_ready,
            }
        )
    categories: dict[str, dict[str, int]] = {}
    for product in products:
        summary = categories.setdefault(
            str(product["category"]), {"total": 0, "selected": 0, "ready": 0, "panel_ready": 0}
        )
        summary["total"] += 1
        summary["selected"] += int(bool(product["selected"]))
        summary["ready"] += int(product["storage_state"] == "READY")
        summary["panel_ready"] += int(bool(product["panel_ready"]))
    return {
        "products": products,
        "categories": categories,
        "selected_count": sum(bool(product["selected"]) for product in products),
        "ready_count": sum(product["storage_state"] == "READY" for product in products),
        "panel_field_count": len(research_factor_fields or set()),
        "staged_field_count": len(
            {field for product in products for field in product["panel_available_fields"]}
        ),
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
                "qfq" if "_qfq_" in task.name else "none" if "_none_" in task.name else "unknown"
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


def _selected_products(value: str | None) -> list[str]:
    if not value:
        return list(DEFAULT_PRODUCT_IDS)
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return list(DEFAULT_PRODUCT_IDS)
    if not isinstance(parsed, list):
        return list(DEFAULT_PRODUCT_IDS)
    return [str(item) for item in parsed]


def read_json(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None
