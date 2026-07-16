from __future__ import annotations

import pandas as pd
import pytest

from autoalpha.backtest.costs import ChinaAExecutionCosts
from autoalpha.backtest.ledger import LedgerBacktester, LedgerConfig


def _market() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=6)
    rows = []
    for index, date in enumerate(dates):
        rows.extend(
            [
                {
                    "date": date,
                    "symbol": "A",
                    "open": 10.0 + index,
                    "close": 10.5 + index,
                    "volume": 10_000_000,
                    "can_buy_open": True,
                    "can_sell_open": index != 2,
                },
                {
                    "date": date,
                    "symbol": "B",
                    "open": 20.0,
                    "close": 20.0,
                    "volume": 10_000_000,
                    "can_buy_open": True,
                    "can_sell_open": True,
                },
            ]
        )
    return pd.DataFrame(rows)


def _signal() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=6)
    return pd.DataFrame(
        {
            "A": [2.0, -2.0, -2.0, -2.0, -2.0, -2.0],
            "B": [-2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        },
        index=dates,
        dtype=float,
    )


def _backtester(horizon: int = 1) -> LedgerBacktester:
    return LedgerBacktester(
        LedgerConfig(
            horizon=horizon,
            initial_cash=1_000_000,
            top_fraction=0.5,
            lot_size=100,
            max_volume_participation=1.0,
        ),
        ChinaAExecutionCosts(
            commission_bps_each_side=0,
            stamp_duty_bps_sell=0,
            transfer_fee_bps_each_side=0,
            minimum_commission_cny=0,
        ),
    )


def test_signal_executes_at_next_open_and_marks_real_daily_path() -> None:
    result = _backtester().run(_signal(), _market())
    first_trade = result.trades.iloc[0]

    assert first_trade["signal_date"] == pd.Timestamp("2024-01-02")
    assert first_trade["date"] == pd.Timestamp("2024-01-03")
    assert first_trade["symbol"] == "A"
    assert first_trade["price"] == 11.0
    assert {
        "commission",
        "transfer_fee",
        "stamp_duty",
        "fees",
        "net_cash_flow",
        "cash_after",
    }.issubset(result.trades.columns)
    assert result.nav.loc["2024-01-03"] > result.nav.loc["2024-01-02"]


def test_unsellable_position_remains_valued_and_retries_next_day() -> None:
    result = _backtester().run(_signal(), _market())
    trades = result.trades

    blocked_day_sales = trades[
        (trades["date"] == pd.Timestamp("2024-01-04"))
        & (trades["symbol"] == "A")
        & (trades["side"] == "SELL")
    ]
    retry_sales = trades[
        (trades["date"] == pd.Timestamp("2024-01-05"))
        & (trades["symbol"] == "A")
        & (trades["side"] == "SELL")
    ]

    assert blocked_day_sales.empty
    assert not retry_sales.empty
    assert result.nav.loc["2024-01-04"] > 0


def test_horizon_creates_independent_overlapping_sleeves() -> None:
    result = _backtester(horizon=2).run(_signal(), _market())

    assert set(result.trades["sleeve"]) == {0, 1}
    assert result.nav.iloc[0] == pytest.approx(1_000_000)


def test_max_positions_caps_fractional_selection() -> None:
    backtester = LedgerBacktester(
        LedgerConfig(
            horizon=1,
            initial_cash=1_000_000,
            top_fraction=1.0,
            max_positions=1,
            max_volume_participation=1.0,
        ),
        ChinaAExecutionCosts(
            commission_bps_each_side=0,
            stamp_duty_bps_sell=0,
            transfer_fee_bps_each_side=0,
            minimum_commission_cny=0,
        ),
    )

    result = backtester.run(_signal(), _market())

    bought_per_day = (
        result.trades[result.trades["side"] == "BUY"].groupby("date")["symbol"].nunique()
    )
    assert bought_per_day.max() == 1


def test_weekly_schedule_rebalances_on_first_actual_session_only() -> None:
    result = LedgerBacktester(
        LedgerConfig(
            horizon=5,
            initial_cash=1_000_000,
            top_fraction=0.5,
            max_volume_participation=1.0,
            rebalance_schedule="WEEKLY_FIRST_SESSION",
        ),
        ChinaAExecutionCosts(
            commission_bps_each_side=0,
            stamp_duty_bps_sell=0,
            transfer_fee_bps_each_side=0,
            minimum_commission_cny=0,
        ),
    ).run(_signal(), _market())

    assert set(result.trades["date"]) == {pd.Timestamp("2024-01-08")}
    assert set(result.trades["sleeve"]) == {0}
    assert result.trades.iloc[0]["signal_date"] == pd.Timestamp("2024-01-05")


def test_modeled_slippage_records_reference_and_execution_prices() -> None:
    backtester = LedgerBacktester(
        LedgerConfig(
            horizon=1,
            initial_cash=1_000_000,
            top_fraction=0.5,
            max_volume_participation=1.0,
            slippage_bps_each_side=10.0,
        ),
        ChinaAExecutionCosts(
            commission_bps_each_side=0,
            stamp_duty_bps_sell=0,
            transfer_fee_bps_each_side=0,
            minimum_commission_cny=0,
        ),
    )

    first_trade = backtester.run(_signal(), _market()).trades.iloc[0]

    assert first_trade["reference_price"] == pytest.approx(11.0)
    assert first_trade["price"] == pytest.approx(11.011)
    assert first_trade["slippage_bps"] == pytest.approx(10.0)
