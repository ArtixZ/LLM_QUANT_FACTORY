from __future__ import annotations

import numpy as np
import pandas as pd

from autoalpha.backtest.costs import USEquityExecutionCosts
from autoalpha.backtest.ledger import LedgerBacktester, LedgerConfig
from autoalpha.backtest.target_book import (
    rebalance_mask,
    select_target_positions,
    select_target_symbols,
)
from autoalpha.backtest.us_vector import USVectorBacktester, USVectorConfig


def test_numpy_and_series_target_selection_are_identical_and_stable() -> None:
    row = pd.Series([2.0, 2.0, np.nan, 1.0], index=["B", "A", "D", "C"])

    positions = select_target_positions(
        row.to_numpy(), selection_fraction=0.5, maximum_positions=2
    )
    symbols = select_target_symbols(
        row, selection_fraction=0.5, maximum_positions=2
    )

    assert positions.tolist() == [0, 1]
    assert symbols == ("B", "A")


def test_target_selection_does_not_apply_execution_eligibility() -> None:
    signal = np.array([10.0, 5.0, 1.0])
    selected = select_target_positions(
        signal, selection_fraction=0.34, maximum_positions=1
    )

    assert selected.tolist() == [0]


def test_weekly_and_biweekly_masks_use_actual_first_sessions() -> None:
    dates = pd.DatetimeIndex(
        ["2024-01-02", "2024-01-05", "2024-01-08", "2024-01-15", "2024-01-22"]
    )

    weekly = rebalance_mask(dates, "WEEKLY_FIRST_SESSION")
    biweekly = rebalance_mask(dates, "BIWEEKLY_FIRST_SESSION")

    assert weekly.tolist() == [False, False, True, True, True]
    assert biweekly.tolist() == [False, False, True, False, True]


def test_vector_and_event_engines_align_in_fractional_ideal_case() -> None:
    dates = pd.bdate_range("2024-01-02", periods=80)
    symbols = ["A", "B", "C", "D"]
    price = pd.DataFrame(
        {
            symbol: 10.0 * (1.0 + (index + 1) * 0.0004) ** np.arange(len(dates))
            for index, symbol in enumerate(symbols)
        },
        index=dates,
    )
    signal = pd.DataFrame(
        {
            "A": np.where(np.arange(len(dates)) < 40, 4.0, 1.0),
            "B": np.where(np.arange(len(dates)) < 40, 3.0, 2.0),
            "C": np.where(np.arange(len(dates)) < 40, 2.0, 3.0),
            "D": np.where(np.arange(len(dates)) < 40, 1.0, 4.0),
        },
        index=dates,
    )
    tradable = pd.DataFrame(True, index=dates, columns=symbols)
    vector = USVectorBacktester(
        USVectorConfig(
            initial_cash_usd=100_000_000.0,
            gross_exposure=0.90,
            selection_fraction=0.50,
            maximum_positions=2,
            commission_bps_each_side=0.0,
            sec_fee_bps_sell=0.0,
            slippage_bps_each_side=0.0,
        )
    ).run(signal, price, price, tradable, tradable, start=dates[0], end=dates[-1])
    market = (
        price.rename_axis(index="date", columns="symbol")
        .stack(future_stack=True)
        .rename("open")
        .reset_index()
    )
    market["close"] = market["open"]
    market["volume"] = 1_000_000_000.0
    market["can_buy_open"] = True
    market["can_sell_open"] = True
    event = LedgerBacktester(
        LedgerConfig(
            horizon=5,
            initial_cash=100_000_000.0,
            top_fraction=0.50,
            max_positions=2,
            lot_size=1,
            max_volume_participation=1.0,
            investment_buffer=0.10,
            rebalance_schedule="WEEKLY_FIRST_SESSION",
        ),
        USEquityExecutionCosts(
            commission_per_share=0.0,
            sec_fee_per_million_usd_sell=0.0,
            finra_taf_per_share_sell=0.0,
        ),
    ).run(signal, market)

    aligned_event = event.daily_return.shift(-1).reindex(vector.path.index)
    paired = pd.concat([vector.path["net"], aligned_event], axis=1).dropna()

    assert paired.corr().iloc[0, 1] > 0.999999
    assert float((paired.iloc[:, 0] - paired.iloc[:, 1]).abs().max()) < 1e-7
    audited_trade_dates = event.trades.loc[
        event.trades["date"] <= vector.path.index.max(), "date"
    ].nunique()
    assert vector.metrics["rebalance_count"] == audited_trade_dates
