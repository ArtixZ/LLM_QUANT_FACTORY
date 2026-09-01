from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

US_EQUITY_CURRENCY = "USD"
US_PRIMARY_EXCHANGES = frozenset({"NASDAQ", "NYSE", "AMEX", "ARCA", "BATS", "IEX", "NYSENAT"})
_SYMBOL_PATTERN = re.compile(r"^[A-Z][A-Z0-9]{0,5}( [A-Z])?$")


class ContractResolutionError(RuntimeError):
    """A symbol could not be resolved to exactly one tradable US equity contract."""


@dataclass(frozen=True)
class USEquity:
    """A resolved US equity contract, decoupled from the broker client library."""

    symbol: str
    con_id: int
    primary_exchange: str
    currency: str = US_EQUITY_CURRENCY
    local_symbol: str = ""
    trading_class: str = ""
    routing_exchange: str = "SMART"

    def __post_init__(self) -> None:
        if not self.symbol.strip():
            raise ValueError("Contract symbol is required")
        if self.con_id <= 0:
            raise ValueError(f"{self.symbol} has an invalid contract id: {self.con_id}")
        if self.currency != US_EQUITY_CURRENCY:
            raise ValueError(
                f"{self.symbol} is denominated in {self.currency}; this platform trades US equities"
            )

    @property
    def panel_symbol(self) -> str:
        """Stable research identifier used as the panel's symbol key."""
        return self.symbol.replace(" ", ".")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_symbol(symbol: str) -> str:
    """Convert common vendor spellings into IBKR's symbol convention.

    IBKR separates share classes with a space (``BRK B``) where most data vendors
    and humans write a dot or hyphen (``BRK.B``, ``BRK-B``).
    """
    cleaned = symbol.strip().upper()
    if not cleaned:
        raise ValueError("Symbol cannot be empty")
    cleaned = re.sub(r"[.\-_/]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not _SYMBOL_PATTERN.match(cleaned):
        raise ValueError(f"{symbol!r} is not a valid US equity symbol")
    return cleaned


def panel_symbol(symbol: str) -> str:
    """Research-side spelling of a symbol: dots rather than IBKR's spaces."""
    return normalize_symbol(symbol).replace(" ", ".")


def select_primary_listing(candidates: list[dict[str, Any]], symbol: str) -> dict[str, Any]:
    """Pick the single US primary listing from an ambiguous contract search.

    IBKR returns one row per listing venue for a symbol. We keep USD common stock
    on a recognised US primary exchange, then require the result to be unique so
    an ambiguous ticker fails loudly instead of silently binding the wrong company.
    """
    usable = [
        candidate
        for candidate in candidates
        if str(candidate.get("currency", "")).upper() == US_EQUITY_CURRENCY
        and str(candidate.get("primaryExchange", "")).upper() in US_PRIMARY_EXCHANGES
        and int(candidate.get("conId") or 0) > 0
    ]
    if not usable:
        raise ContractResolutionError(
            f"{symbol} did not resolve to a US-listed USD equity contract"
        )
    unique_ids = {int(candidate["conId"]) for candidate in usable}
    if len(unique_ids) > 1:
        venues = sorted(
            f"{candidate.get('primaryExchange')}:{candidate.get('conId')}" for candidate in usable
        )
        raise ContractResolutionError(
            f"{symbol} is ambiguous across {len(unique_ids)} contracts ({', '.join(venues)}); "
            "pass an explicit primary exchange"
        )
    return usable[0]
