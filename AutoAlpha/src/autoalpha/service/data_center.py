from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from autoalpha.data.execution_basis import inspect_execution_data_basis
from autoalpha.data.product_catalog import DEFAULT_PRODUCT_IDS, data_product_catalog
from autoalpha.data.workspace import inspect_data_workspace


def build_data_center_snapshot(
    settings: Mapping[str, str],
    *,
    sync_status: Mapping[str, Any],
    gateway_ready: bool,
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
        "capability_matrix": build_data_capability_matrix(
            workspace=workspace,
            execution_basis=execution_basis,
            workspace_error=workspace_error,
        ),
        "downloader": downloader,
        "data_products": products,
        "credentials": {"ibkr_gateway_ready": gateway_ready},
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


def build_data_capability_matrix(
    *,
    workspace: Mapping[str, Any] | None,
    execution_basis: Mapping[str, Any] | None,
    workspace_error: str | None = None,
) -> dict[str, Any]:
    """Describe which platform modules may use the current data basis."""
    workspace_blockers = _workspace_blockers(workspace, workspace_error)
    research_ready = (
        bool(workspace and workspace.get("price_research_ready")) and not workspace_error
    )
    proxy_ready = bool(execution_basis and execution_basis.get("capital_ledger_proxy_ready"))
    strict_pit_ready = bool(
        workspace
        and workspace.get("institutional_pit_ready")
        and execution_basis
        and execution_basis.get("capital_ledger_ready")
    )
    proxy_blockers = _basis_messages(execution_basis, "proxy_blockers")
    strict_blockers = _basis_messages(execution_basis, "blockers")
    if workspace_error:
        proxy_blockers = [workspace_error, *proxy_blockers]
        strict_blockers = [workspace_error, *strict_blockers]
    if not workspace or not workspace.get("institutional_pit_ready"):
        strict_blockers = [
            "missing point-in-time institutional market-state workspace",
            *strict_blockers,
        ]
    rows = [
        _capability_row(
            module_id="auto_research",
            label="自动因子研究",
            level="RESEARCH_READY" if research_ready else "BLOCKED",
            allowed=research_ready,
            data_mode="FORWARD_ADJUSTED_RESEARCH_PANEL",
            summary=(
                "可使用 ADJUSTED_LAST 研究价格与成交活动字段构造截面因子。"
                if research_ready
                else "研究面板缺少必要价格或活动字段。"
            ),
            blockers=workspace_blockers if not research_ready else [],
        ),
        _capability_row(
            module_id="screener",
            label="收盘后选股器",
            level="RESEARCH_READY" if research_ready else "BLOCKED",
            allowed=research_ready,
            data_mode="EOD_SIGNAL_ONLY",
            summary=(
                "可在最新收盘截面生成候选股票；不代表次日一定可成交。"
                if research_ready
                else "选股器需要完整研究面板。"
            ),
            blockers=workspace_blockers if not research_ready else [],
        ),
        _capability_row(
            module_id="manual_backtest_proxy",
            label="手动回测 · 非 PIT 代理",
            level="PROXY_BACKTEST_READY" if proxy_ready else "BLOCKED",
            allowed=proxy_ready,
            data_mode="NON_PIT_PROXY_NEXT_OPEN_LEDGER",
            summary=(
                "可使用 TRADES 开盘价、整数股、费用和开盘资格代理做研究级现金账本。"
                if proxy_ready
                else "缺少未复权执行价格或开盘可交易代理字段。"
            ),
            blockers=proxy_blockers if not proxy_ready else [],
            caveats=["non-PIT proxy is research and paper trading only"],
        ),
        _capability_row(
            module_id="batch_backtest_proxy",
            label="批量回测 · 非 PIT 代理",
            level="PROXY_BACKTEST_READY" if proxy_ready else "BLOCKED",
            allowed=proxy_ready,
            data_mode="NON_PIT_PROXY_VECTOR_OR_EVENT_LEDGER",
            summary=(
                "可批量评估因子和组合，但不能宣称真实生产现金账本。"
                if proxy_ready
                else "批量现金代理回测需要完整执行代理字段。"
            ),
            blockers=proxy_blockers if not proxy_ready else [],
            caveats=["vector engine results must be reconciled against event ledger"],
        ),
        _capability_row(
            module_id="paper_trading",
            label="模拟交易",
            level="PROXY_PAPER_READY" if proxy_ready else "BLOCKED",
            allowed=proxy_ready,
            data_mode="NEXT_SESSION_OPEN_PROXY",
            summary=(
                "可做次日开盘代理成交、整数股、费用和交割单级模拟组合。"
                if proxy_ready
                else "模拟交易需要开盘执行代理与可买卖状态。"
            ),
            blockers=proxy_blockers if not proxy_ready else [],
            caveats=["strict production promotion still requires PIT market state"],
        ),
        _capability_row(
            module_id="strict_capital_ledger",
            label="真实现金账本 / 生产候选",
            level="STRICT_PIT_READY" if strict_pit_ready else "PRODUCTION_BLOCKED",
            allowed=strict_pit_ready,
            data_mode="STRICT_POINT_IN_TIME_CAPITAL_LEDGER",
            summary=(
                "PIT 上市退市、停牌、开盘资格、行业和基准成员状态已满足。"
                if strict_pit_ready
                else "不能宣称生产可交易；需要补齐 PIT 市场状态和版本化基础数据。"
            ),
            blockers=strict_blockers if not strict_pit_ready else [],
            required_fields=[
                "listing_date",
                "delisting_date",
                "is_halted",
                "can_buy_open",
                "can_sell_open",
                "sector_code",
                "index_membership",
                "free_float_market_cap",
            ],
        ),
    ]
    levels = {str(row["level"]): 0 for row in rows}
    for row in rows:
        levels[str(row["level"])] += 1
    return {
        "protocol": "AUTOALPHA_DATA_CAPABILITY_MATRIX_V1",
        "production_policy": (
            "research and paper trading may use NON_PIT_PROXY; production requires STRICT_PIT"
        ),
        "summary": {
            "research_ready": research_ready,
            "non_pit_proxy_ready": proxy_ready,
            "strict_pit_ready": strict_pit_ready,
            "production_allowed": strict_pit_ready,
            "levels": levels,
        },
        "rows": rows,
    }


def _capability_row(
    *,
    module_id: str,
    label: str,
    level: str,
    allowed: bool,
    data_mode: str,
    summary: str,
    blockers: Sequence[str] | None = None,
    caveats: Sequence[str] | None = None,
    required_fields: Sequence[str] | None = None,
) -> dict[str, Any]:
    return {
        "module_id": module_id,
        "label": label,
        "level": level,
        "allowed": allowed,
        "data_mode": data_mode,
        "summary": summary,
        "blockers": list(dict.fromkeys(str(item) for item in blockers or [] if item)),
        "caveats": list(dict.fromkeys(str(item) for item in caveats or [] if item)),
        "required_fields": list(required_fields or []),
    }


def _workspace_blockers(
    workspace: Mapping[str, Any] | None,
    workspace_error: str | None,
) -> list[str]:
    if workspace_error:
        return [workspace_error]
    if not workspace:
        return ["data workspace is unavailable"]
    blockers = [str(item) for item in workspace.get("blockers", [])]
    warnings = [str(item) for item in workspace.get("warnings", [])]
    if not workspace.get("price_research_ready"):
        return list(dict.fromkeys([*blockers, *warnings, "price research panel is not ready"]))
    return list(dict.fromkeys([*blockers, *warnings]))


def _basis_messages(
    execution_basis: Mapping[str, Any] | None,
    key: str,
) -> list[str]:
    if not execution_basis:
        return ["execution basis is unavailable"]
    return [str(item) for item in execution_basis.get(key, [])]


def inspect_data_products(
    root: Path,
    *,
    selected_products: list[str],
    panel_columns: set[str],
    research_factor_fields: set[str] | None = None,
    panel_first_date: str = "",
    panel_last_date: str = "",
) -> dict[str, Any]:
    downloads = root / "downloads"
    slice_files = list(downloads.glob("*.parquet")) if downloads.is_dir() else []
    products = []
    for product in data_product_catalog():
        dataset_id = str(product["dataset_id"])
        selected = dataset_id in selected_products
        integrated = str(product["integration_state"]) == "INTEGRATED"
        mandatory = dataset_id in {"core_market", "execution_market", "tradability"}
        download_selectable = integrated and not mandatory
        if mandatory:
            selection_lock_reason = "The canonical IBKR panel maintains this product together"
        elif not download_selectable:
            selection_lock_reason = "No point-in-time ingestion contract is implemented"
        else:
            selection_lock_reason = None
        source = downloads
        files = len(slice_files) if integrated else 0
        completed = files
        failed = 0
        first_date = panel_first_date or None
        last_date = panel_last_date or None
        panel_fields = [str(value) for value in product.get("panel_fields", [])]
        panel_available = [value for value in panel_fields if value in panel_columns]
        research_fields = research_factor_fields or set()
        research_contract_ready = bool(panel_fields) and set(panel_fields).issubset(
            research_fields
        )
        coverage_ready = bool(
            first_date
            and last_date
            and (not panel_first_date or str(first_date) <= panel_first_date)
            and (not panel_last_date or str(last_date) >= panel_last_date)
        )
        if files:
            storage_state = "READY"
        elif not integrated:
            storage_state = "CATALOG"
        else:
            storage_state = "NOT_DOWNLOADED"
        if integrated:
            research_state = (
                "RESEARCH_READY"
                if coverage_ready and set(panel_fields).issubset(panel_columns)
                else "WAITING_FOR_COVERAGE"
            )
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
    downloads = root / "downloads"
    tasks = []
    if downloads.is_dir():
        slices = sorted(downloads.glob("*.parquet"), key=_mtime, reverse=True)
        if slices:
            tasks.append(
                {
                    "name": "IBKR per-symbol daily slices",
                    "adjustment": "adjusted_research_plus_split_adjusted_execution",
                    "parquet_files": len(slices),
                    "updated_at": datetime.fromtimestamp(slices[0].stat().st_mtime)
                    .astimezone()
                    .isoformat(),
                }
            )
    return {
        "root_path": str(root.resolve()) if root.exists() else str(root),
        "root_exists": root.is_dir(),
        "source": "interactive_brokers_gateway",
        "sync_managed_by_service": True,
        "download_tasks": tasks[:12],
        "research_adjusted_available": bool(tasks),
        "execution_split_adjusted_available": bool(tasks),
        "panel_available": (root / "processed" / "daily_panel").is_dir(),
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
