from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import pytest

from autoalpha.backtest.capital import CapitalBacktestSpec, _market_frame


def _spec() -> CapitalBacktestSpec:
    return CapitalBacktestSpec(start=date(2026, 1, 1), end=date(2026, 1, 31))


def _panel_frame(**overrides: Any) -> pd.DataFrame:
    data: dict[str, Any] = {
        "trade_date": pd.to_datetime(["2026-01-05", "2026-01-05"]),
        "symbol": ["AAPL", "MSFT"],
        "open": [10.0, 20.0],
        "close": [11.0, 21.0],
        "vol": [1000.0, 2000.0],
        "is_valid_ohlc": [True, True],
        "is_tradable_observation": [True, True],
        "can_buy_open": [True, False],
        "can_sell_open": [np.nan, True],
    }
    data.update(overrides)
    return pd.DataFrame(data)


def test_market_frame_applies_the_panels_tradability_flags() -> None:
    market = _market_frame(_panel_frame(), _spec())

    assert list(market["can_buy_open"]) == [True, False]
    # A missing flag observation is not permission to trade.
    assert list(market["can_sell_open"]) == [False, True]


def test_market_frame_blocks_both_sides_on_invalid_bars() -> None:
    market = _market_frame(
        _panel_frame(is_valid_ohlc=[True, False]), _spec()
    )

    msft = market[market["symbol"] == "MSFT"].iloc[0]
    assert not msft["can_buy_open"]
    assert not msft["can_sell_open"]
    assert np.isnan(msft["open"]) and np.isnan(msft["close"])


def test_market_frame_fails_loudly_when_a_tradability_flag_is_missing() -> None:
    """_load_panel loads both flags; dropping one must not silently widen eligibility."""
    frame = _panel_frame().drop(columns=["can_sell_open"])

    with pytest.raises(KeyError):
        _market_frame(frame, _spec())
