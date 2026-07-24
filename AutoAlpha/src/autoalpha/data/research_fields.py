from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

from autoalpha.data.tushare_catalog import PRODUCT_BY_ID
from autoalpha.dsl.expression import Expression
from autoalpha.dsl.semantics import FieldDefinition

BASE_RESEARCH_FIELDS = ("open", "high", "low", "close", "adj_close", "amount", "vol")
PRICE_FIELDS = frozenset({"open", "high", "low", "close", "adj_close"})
SHARE_FIELDS = frozenset(
    {
        "vol",
        "total_share",
        "float_share",
        "free_share",
        "buy_sm_vol",
        "sell_sm_vol",
        "buy_md_vol",
        "sell_md_vol",
        "buy_lg_vol",
        "sell_lg_vol",
        "buy_elg_vol",
        "sell_elg_vol",
        "net_mf_vol",
    }
)
CURRENCY_FIELDS = frozenset(
    {
        "amount",
        "total_mv",
        "circ_mv",
        "buy_sm_amount",
        "sell_sm_amount",
        "buy_md_amount",
        "sell_md_amount",
        "buy_lg_amount",
        "sell_lg_amount",
        "buy_elg_amount",
        "sell_elg_amount",
        "net_mf_amount",
    }
)
EXTENDED_RESEARCH_FIELDS = tuple(
    field
    for dataset_id in ("daily_basic", "moneyflow")
    for field in PRODUCT_BY_ID[dataset_id].panel_fields
)
AUTO_RESEARCH_PRODUCTS = ("daily_basic", "moneyflow")
MAXIMUM_FEATURE_END_LAG_DAYS = 7


@dataclass(frozen=True)
class ResearchFieldProfile:
    name: str
    label: str
    description: str
    source_product: str
    unit: str
    availability: str = "END_OF_DAY_NEXT_SESSION"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


_FIELD_DETAILS: dict[str, tuple[str, str, str]] = {
    "open": (
        "Open price",
        "Adjusted session open observed by end of day; signals may trade next session.",
        "core_market",
    ),
    "high": (
        "High price",
        "Adjusted session high observed by end of day; signals may trade next session.",
        "core_market",
    ),
    "low": (
        "Low price",
        "Adjusted session low observed by end of day; signals may trade next session.",
        "core_market",
    ),
    "close": (
        "Close price",
        "Unadjusted session close under the workspace price basis.",
        "core_market",
    ),
    "adj_close": (
        "Adjusted close",
        "Point-in-time event-adjusted close for return and path research.",
        "core_market",
    ),
    "amount": (
        "Trading amount",
        "Session trading amount in CNY; a liquidity and activity observation.",
        "core_market",
    ),
    "vol": (
        "Trading volume",
        "Session traded shares; a liquidity and participation observation.",
        "core_market",
    ),
    "turnover_rate": ("Turnover rate", "Turnover as a percentage of total shares.", "daily_basic"),
    "turnover_rate_f": (
        "Free-float turnover",
        "Turnover as a percentage of free-float shares.",
        "daily_basic",
    ),
    "volume_ratio": (
        "Volume ratio",
        "Current activity relative to its recent reference activity.",
        "daily_basic",
    ),
    "pe": ("PE", "Price-to-earnings ratio based on reported earnings.", "daily_basic"),
    "pe_ttm": (
        "PE TTM",
        "Price-to-earnings ratio based on trailing-twelve-month earnings.",
        "daily_basic",
    ),
    "pb": ("PB", "Price-to-book ratio.", "daily_basic"),
    "ps": ("PS", "Price-to-sales ratio based on reported sales.", "daily_basic"),
    "ps_ttm": (
        "PS TTM",
        "Price-to-sales ratio based on trailing-twelve-month sales.",
        "daily_basic",
    ),
    "dv_ratio": (
        "Dividend yield",
        "Dividend yield based on the source's reported-period convention.",
        "daily_basic",
    ),
    "dv_ttm": ("Dividend yield TTM", "Trailing-twelve-month dividend yield.", "daily_basic"),
    "total_share": ("Total shares", "Total shares outstanding in the source unit.", "daily_basic"),
    "float_share": ("Float shares", "Tradable float shares in the source unit.", "daily_basic"),
    "free_share": ("Free-float shares", "Free-float shares in the source unit.", "daily_basic"),
    "total_mv": (
        "Total market value",
        "Total market capitalization in the source currency unit.",
        "daily_basic",
    ),
    "circ_mv": (
        "Float market value",
        "Tradable-float market capitalization in the source currency unit.",
        "daily_basic",
    ),
    "buy_sm_vol": (
        "Small-order buy volume",
        "Buy-side volume classified as small orders.",
        "moneyflow",
    ),
    "buy_sm_amount": (
        "Small-order buy amount",
        "Buy-side amount classified as small orders.",
        "moneyflow",
    ),
    "sell_sm_vol": (
        "Small-order sell volume",
        "Sell-side volume classified as small orders.",
        "moneyflow",
    ),
    "sell_sm_amount": (
        "Small-order sell amount",
        "Sell-side amount classified as small orders.",
        "moneyflow",
    ),
    "buy_md_vol": (
        "Medium-order buy volume",
        "Buy-side volume classified as medium orders.",
        "moneyflow",
    ),
    "buy_md_amount": (
        "Medium-order buy amount",
        "Buy-side amount classified as medium orders.",
        "moneyflow",
    ),
    "sell_md_vol": (
        "Medium-order sell volume",
        "Sell-side volume classified as medium orders.",
        "moneyflow",
    ),
    "sell_md_amount": (
        "Medium-order sell amount",
        "Sell-side amount classified as medium orders.",
        "moneyflow",
    ),
    "buy_lg_vol": (
        "Large-order buy volume",
        "Buy-side volume classified as large orders.",
        "moneyflow",
    ),
    "buy_lg_amount": (
        "Large-order buy amount",
        "Buy-side amount classified as large orders.",
        "moneyflow",
    ),
    "sell_lg_vol": (
        "Large-order sell volume",
        "Sell-side volume classified as large orders.",
        "moneyflow",
    ),
    "sell_lg_amount": (
        "Large-order sell amount",
        "Sell-side amount classified as large orders.",
        "moneyflow",
    ),
    "buy_elg_vol": (
        "Extra-large buy volume",
        "Buy-side volume classified as extra-large orders.",
        "moneyflow",
    ),
    "buy_elg_amount": (
        "Extra-large buy amount",
        "Buy-side amount classified as extra-large orders.",
        "moneyflow",
    ),
    "sell_elg_vol": (
        "Extra-large sell volume",
        "Sell-side volume classified as extra-large orders.",
        "moneyflow",
    ),
    "sell_elg_amount": (
        "Extra-large sell amount",
        "Sell-side amount classified as extra-large orders.",
        "moneyflow",
    ),
    "net_mf_vol": (
        "Net money-flow volume",
        "Net classified order-flow volume after the session close.",
        "moneyflow",
    ),
    "net_mf_amount": (
        "Net money-flow amount",
        "Net classified order-flow amount after the session close.",
        "moneyflow",
    ),
}


def research_field_profiles() -> dict[str, ResearchFieldProfile]:
    return {
        name: ResearchFieldProfile(
            name=name,
            label=label,
            description=description,
            source_product=product,
            unit=_unit(name, amount_unit="cny", volume_unit="shares"),
        )
        for name, (label, description, product) in _FIELD_DETAILS.items()
    }


def expression_fields(expression: Expression | Mapping[str, Any] | None) -> set[str]:
    if expression is None:
        return set()
    if isinstance(expression, Expression):
        names = {str(expression.parameter("name"))} if expression.operator == "field" else set()
        for argument in expression.arguments:
            names.update(expression_fields(argument))
        return names
    operator = str(expression.get("operator", ""))
    parameters = expression.get("parameters", {})
    names = (
        {str(parameters.get("name"))}
        if operator == "field" and isinstance(parameters, Mapping) and parameters.get("name")
        else set()
    )
    arguments = expression.get("arguments", [])
    if isinstance(arguments, list):
        for argument in arguments:
            if isinstance(argument, Mapping):
                names.update(expression_fields(argument))
    return names


def build_research_data_capabilities(
    workspace: Any,
    *,
    required_start: date | str | None = None,
    required_end: date | str | None = None,
) -> dict[str, Any]:
    """Build a bounded LLM data contract without unlocking incomplete fields."""
    metadata = _read_metadata(getattr(workspace, "metadata_path", None))
    sources = metadata.get("sources", {}) if isinstance(metadata.get("sources"), dict) else {}
    coverage = (
        sources.get("feature_coverage", {})
        if isinstance(sources.get("feature_coverage"), dict)
        else {}
    )
    panel_columns = set(getattr(workspace, "columns", ()))
    effective_start = required_start or getattr(workspace, "first_trade_date", None)
    effective_end = required_end or getattr(workspace, "last_trade_date", None)
    eligible, coverage_gates = research_eligible_fields(
        panel_columns,
        metadata,
        required_start=effective_start,
        required_end=effective_end,
        declared_fields=getattr(workspace, "factor_fields", ()),
    )
    profiles = research_field_profiles()
    field_catalog = []
    for name in (*BASE_RESEARCH_FIELDS, *EXTENDED_RESEARCH_FIELDS):
        profile = profiles[name].to_dict()
        if name in eligible:
            status = "RESEARCH_ELIGIBLE"
        elif name in panel_columns:
            status = "STAGED_COVERAGE_INCOMPLETE"
        else:
            status = "DISCOVERABLE_NOT_LOADED"
        field_catalog.append(
            {**profile, "status": status, "allowed_in_expression": name in eligible}
        )

    products = []
    for product in PRODUCT_BY_ID.values():
        item_coverage = coverage.get(product.dataset_id, {})
        if not isinstance(item_coverage, dict):
            item_coverage = {}
        product_fields = (
            set(BASE_RESEARCH_FIELDS)
            if product.dataset_id == "core_market"
            else set(product.panel_fields)
        )
        eligible_fields = sorted(product_fields & eligible)
        staged_fields = sorted((product_fields & panel_columns) - eligible)
        if product.dataset_id == "core_market":
            research_state = "RESEARCH_ELIGIBLE"
        elif product.dataset_id == "stk_limit":
            research_state = (
                "EXECUTION_CONSTRAINT_AVAILABLE"
                if item_coverage
                else "EXECUTION_CONSTRAINT_PLANNED"
            )
        elif eligible_fields:
            research_state = "RESEARCH_ELIGIBLE"
        elif staged_fields or item_coverage:
            research_state = "STAGED_COVERAGE_INCOMPLETE"
        elif product.integration_state == "CATALOG":
            research_state = "DISCOVERABLE_REQUIRES_PIT_INTEGRATION"
        else:
            research_state = "DOWNLOADABLE_REQUIRES_INTEGRATION"
        products.append(
            {
                "dataset_id": product.dataset_id,
                "label": product.label,
                "category": product.category,
                "feature_family": product.feature_family,
                "availability": product.availability,
                "pit_policy": product.pit_policy,
                "research_state": research_state,
                "eligible_fields": eligible_fields,
                "staged_fields": staged_fields,
                "coverage": {
                    key: item_coverage.get(key) for key in ("first_date", "last_date", "files")
                },
                "coverage_gate": coverage_gates.get(product.dataset_id),
            }
        )

    eligible_extended = sorted(eligible & set(EXTENDED_RESEARCH_FIELDS))
    staged_extended = sorted((panel_columns & set(EXTENDED_RESEARCH_FIELDS)) - eligible)
    return {
        "contract_version": "research-data-v2",
        "eligible_fields": sorted(eligible),
        "eligible_extended_fields": eligible_extended,
        "staged_extended_fields": staged_extended,
        "field_catalog": field_catalog,
        "data_products": products,
        "signal_timing": "END_OF_DAY_INFORMATION_MAY_TRADE_NEXT_SESSION_OPEN",
        "policy": {
            "expression_allowlist": "eligible_fields_only",
            "staged_fields_forbidden": True,
            "catalog_only_products_forbidden": True,
            "automatic_unlock": (
                "panel fields present plus bounded date coverage; latest feature lag <= 7 days"
            ),
            "point_in_time_required_before_fundamental_or_event_research": True,
        },
    }


def research_eligible_fields(
    columns: Iterable[str],
    metadata: Mapping[str, Any],
    *,
    required_start: date | str | None,
    required_end: date | str | None,
    declared_fields: Iterable[str] = (),
) -> tuple[set[str], dict[str, dict[str, Any]]]:
    """Resolve one auditable field allowlist for UI, prompts, semantics, and loading."""
    available = set(columns)
    eligible = set(BASE_RESEARCH_FIELDS) & available
    eligible.update(set(declared_fields) & available & set(EXTENDED_RESEARCH_FIELDS))
    sources = metadata.get("sources", {}) if isinstance(metadata.get("sources"), Mapping) else {}
    coverage = (
        sources.get("feature_coverage", {})
        if isinstance(sources.get("feature_coverage"), Mapping)
        else {}
    )
    start = _optional_date(required_start)
    end = _optional_date(required_end)
    gates: dict[str, dict[str, Any]] = {}
    for product_id in AUTO_RESEARCH_PRODUCTS:
        product = PRODUCT_BY_ID[product_id]
        product_fields = set(product.panel_fields)
        item = coverage.get(product_id, {})
        item = item if isinstance(item, Mapping) else {}
        first = _optional_date(item.get("first_date"))
        last = _optional_date(item.get("last_date"))
        missing_fields = sorted(product_fields - available)
        start_covered = bool(start and first and first <= start)
        end_lag_days = (end - last).days if end and last else None
        end_covered = bool(
            end_lag_days is not None and end_lag_days <= MAXIMUM_FEATURE_END_LAG_DAYS
        )
        files = int(item.get("files", 0) or 0)
        ready = bool(
            start is not None
            and end is not None
            and files > 0
            and not missing_fields
            and start_covered
            and end_covered
        )
        if ready:
            eligible.update(product_fields)
        gates[product_id] = {
            "passed": ready,
            "required_start": start.isoformat() if start else None,
            "required_end": end.isoformat() if end else None,
            "observed_start": first.isoformat() if first else None,
            "observed_end": last.isoformat() if last else None,
            "end_lag_days": end_lag_days,
            "maximum_end_lag_days": MAXIMUM_FEATURE_END_LAG_DAYS,
            "files": files,
            "missing_panel_fields": missing_fields,
        }
    return eligible, gates


def products_for_fields(fields: Iterable[str]) -> list[str]:
    profiles = research_field_profiles()
    return sorted({profiles[name].source_product for name in fields if name in profiles})


def available_factor_fields(columns: Iterable[str]) -> list[str]:
    available = set(columns)
    return [
        field for field in (*BASE_RESEARCH_FIELDS, *EXTENDED_RESEARCH_FIELDS) if field in available
    ]


def field_definitions(
    columns: Iterable[str],
    *,
    amount_unit: str = "currency",
    volume_unit: str = "shares",
    include_open: bool = True,
) -> list[FieldDefinition]:
    available = set(columns)
    names = available_factor_fields(available)
    if include_open and "open" in available:
        names.insert(0, "open")
    return [
        FieldDefinition(name, _unit(name, amount_unit=amount_unit, volume_unit=volume_unit))
        for name in names
    ]


def _unit(name: str, *, amount_unit: str, volume_unit: str) -> str:
    if name in PRICE_FIELDS:
        return "price"
    if name == "amount":
        return amount_unit
    if name == "vol":
        return volume_unit
    if name in SHARE_FIELDS:
        return "shares_source_unit"
    if name in CURRENCY_FIELDS:
        return "currency_source_unit"
    return "dimensionless"


def _read_metadata(raw_path: Any) -> dict[str, Any]:
    if not raw_path:
        return {}
    path = Path(str(raw_path))
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _optional_date(value: Any) -> date | None:
    if isinstance(value, date):
        return value
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip().replace("/", "-")
    if len(text) == 8 and text.isdigit():
        text = f"{text[:4]}-{text[4:6]}-{text[6:]}"
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None
