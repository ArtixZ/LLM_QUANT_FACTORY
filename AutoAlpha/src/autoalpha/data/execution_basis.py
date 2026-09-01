from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow.dataset as ds

# US equities have no daily price limits and no ST designation, so the
# point-in-time state a cash ledger needs is listing lifecycle plus halt and
# side-specific open eligibility.
CAPITAL_EXECUTION_FIELDS = frozenset(
    {
        "listing_date",
        "delisting_date",
        "is_halted",
        "can_buy_open",
        "can_sell_open",
    }
)

# IBKR does not serve truly unadjusted bars; TRADES is split-adjusted, which is
# the most raw execution basis available and is accepted here as such.
EXECUTION_PRICE_BASES = frozenset({"unadjusted", "raw", "split_adjusted"})
CURRENCY_UNITS = frozenset({"usd", "dollar"})


@dataclass(frozen=True)
class ExecutionDataBasis:
    price_adjustment: str
    execution_price_adjustment: str
    volume_unit: str
    amount_unit: str
    capital_ledger_ready: bool
    capital_ledger_proxy_ready: bool
    blockers: tuple[str, ...]
    proxy_blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def require_capital_ledger(self) -> None:
        if not self.capital_ledger_ready:
            raise RuntimeError("Capital ledger data basis is invalid: " + "; ".join(self.blockers))

    def require_capital_ledger_proxy(self) -> None:
        if not self.capital_ledger_proxy_ready:
            raise RuntimeError(
                "Non-PIT capital ledger proxy data basis is invalid: "
                + "; ".join(self.proxy_blockers)
            )


def inspect_execution_data_basis(panel_path: Path) -> ExecutionDataBasis:
    metadata_path = panel_path / "_metadata.json"
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        value = json.loads(metadata_path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            metadata = value

    source = str(metadata.get("source", "")).casefold()
    price_adjustment = str(metadata.get("price_adjustment", "unknown")).casefold()
    if price_adjustment == "unknown" and "adjusted" in source:
        price_adjustment = "split_and_dividend_adjusted"
    volume_unit = str(metadata.get("volume_unit", "unknown")).casefold()
    amount_unit = str(metadata.get("amount_unit", "unknown")).casefold()
    execution_price_adjustment = str(
        metadata.get("execution_price_adjustment", price_adjustment)
    ).casefold()

    blockers = []
    if execution_price_adjustment not in EXECUTION_PRICE_BASES:
        blockers.append(
            "cash execution requires split-adjusted or unadjusted OHLC, got "
            f"{execution_price_adjustment}"
        )
    if volume_unit != "shares":
        blockers.append(f"cash execution requires volume in shares, got {volume_unit}")
    if amount_unit not in CURRENCY_UNITS:
        blockers.append(f"cash execution requires dollar volume in USD, got {amount_unit}")
    declared_ready = metadata.get("capital_ledger_ready")
    if declared_ready is False:
        blockers.append("panel metadata explicitly blocks capital-ledger use")
    columns: set[str] = set()
    files = sorted(panel_path.rglob("*.parquet"))
    if files:
        columns = set(ds.dataset(files, format="parquet").schema.names)
        missing = sorted(CAPITAL_EXECUTION_FIELDS - columns)
        if missing:
            blockers.append(f"cash execution requires point-in-time market state: {missing}")
    proxy_blockers = []
    if execution_price_adjustment not in EXECUTION_PRICE_BASES:
        proxy_blockers.append(
            "non-PIT proxy execution requires split-adjusted or unadjusted OHLC, got "
            f"{execution_price_adjustment}"
        )
    if volume_unit != "shares":
        proxy_blockers.append(
            f"non-PIT proxy execution requires volume in shares, got {volume_unit}"
        )
    if amount_unit not in CURRENCY_UNITS:
        proxy_blockers.append(
            f"non-PIT proxy execution requires dollar volume in USD, got {amount_unit}"
        )
    if columns:
        required_proxy_prices = (
            {"raw_open", "raw_close"}
            if price_adjustment not in EXECUTION_PRICE_BASES
            else {"open", "close"}
        )
        missing_proxy_prices = sorted(required_proxy_prices - columns)
        if missing_proxy_prices:
            proxy_blockers.append(
                "non-PIT proxy execution requires raw execution prices: "
                f"{missing_proxy_prices}"
            )
    if metadata.get("capital_ledger_proxy_ready") is False:
        proxy_blockers.append("panel metadata explicitly blocks non-PIT capital-ledger proxy use")
    return ExecutionDataBasis(
        price_adjustment=price_adjustment,
        execution_price_adjustment=execution_price_adjustment,
        volume_unit=volume_unit,
        amount_unit=amount_unit,
        capital_ledger_ready=not blockers,
        capital_ledger_proxy_ready=not proxy_blockers,
        blockers=tuple(dict.fromkeys(blockers)),
        proxy_blockers=tuple(dict.fromkeys(proxy_blockers)),
    )


def expression_research_basis_blockers(
    expression: dict[str, Any],
    basis: ExecutionDataBasis,
) -> tuple[str, ...]:
    """Reject adjusted price-level/activity mixtures whose economic units cannot align."""
    fields = _expression_fields(expression)
    price_fields = {"open", "high", "low", "close", "adj_close"}
    activity_fields = {"vol", "amount"}
    blockers = []
    if (
        basis.price_adjustment not in {"unadjusted", "raw"}
        and fields & price_fields
        and fields & activity_fields
    ):
        blockers.append(
            "adjusted price levels cannot be mixed with raw volume/amount in one factor"
        )
    return tuple(blockers)


def _expression_fields(expression: Any) -> set[str]:
    if not isinstance(expression, dict):
        return set()
    fields = set()
    if expression.get("operator") == "field":
        name = expression.get("parameters", {}).get("name")
        if isinstance(name, str):
            fields.add(name)
    for argument in expression.get("arguments", []):
        fields.update(_expression_fields(argument))
    return fields
