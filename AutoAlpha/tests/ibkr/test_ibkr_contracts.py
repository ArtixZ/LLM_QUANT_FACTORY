from __future__ import annotations

import pytest

from autoalpha.ibkr.contracts import (
    ContractResolutionError,
    USEquity,
    normalize_symbol,
    panel_symbol,
    select_primary_listing,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("aapl", "AAPL"),
        (" MSFT ", "MSFT"),
        ("BRK.B", "BRK B"),
        ("BRK-B", "BRK B"),
        ("brk_b", "BRK B"),
    ],
)
def test_normalize_symbol_matches_ibkr_convention(raw: str, expected: str) -> None:
    assert normalize_symbol(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "TOOLONGSYM", "12ABC", "A/B/C"])
def test_normalize_symbol_rejects_invalid_tickers(raw: str) -> None:
    with pytest.raises(ValueError):
        normalize_symbol(raw)


def test_panel_symbol_uses_dot_for_share_class() -> None:
    assert panel_symbol("BRK B") == "BRK.B"
    assert panel_symbol("AAPL") == "AAPL"


def test_us_equity_panel_symbol_round_trips() -> None:
    equity = USEquity(symbol="BRK B", con_id=1, primary_exchange="NYSE")
    assert equity.panel_symbol == "BRK.B"


def test_us_equity_rejects_non_usd() -> None:
    with pytest.raises(ValueError, match="US equities"):
        USEquity(symbol="RY", con_id=5, primary_exchange="NYSE", currency="CAD")


def test_us_equity_rejects_missing_contract_id() -> None:
    with pytest.raises(ValueError, match="invalid contract id"):
        USEquity(symbol="AAPL", con_id=0, primary_exchange="NASDAQ")


def test_select_primary_listing_picks_the_us_usd_row() -> None:
    candidates = [
        {"conId": 1, "currency": "EUR", "primaryExchange": "IBIS"},
        {"conId": 2, "currency": "USD", "primaryExchange": "NASDAQ"},
    ]
    assert select_primary_listing(candidates, "AAPL")["conId"] == 2


def test_select_primary_listing_rejects_ambiguity() -> None:
    candidates = [
        {"conId": 2, "currency": "USD", "primaryExchange": "NASDAQ"},
        {"conId": 3, "currency": "USD", "primaryExchange": "NYSE"},
    ]
    with pytest.raises(ContractResolutionError, match="ambiguous"):
        select_primary_listing(candidates, "XYZ")


def test_select_primary_listing_rejects_when_nothing_qualifies() -> None:
    candidates = [{"conId": 9, "currency": "GBP", "primaryExchange": "LSE"}]
    with pytest.raises(ContractResolutionError, match="US-listed"):
        select_primary_listing(candidates, "VOD")
