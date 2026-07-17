from __future__ import annotations

import numpy as np
import pandas as pd

from autoalpha.backtest.ashare_vector import AshareVectorBacktester, AshareVectorConfig


def _panels() -> tuple[pd.DataFrame, ...]:
    dates = pd.bdate_range("2024-01-02", periods=90)
    columns = ["A", "B", "C"]
    signal = pd.DataFrame(
        np.tile([3.0, 2.0, 1.0], (len(dates), 1)), index=dates, columns=columns
    )
    adjusted_open = pd.DataFrame(
        {
            "A": 10.0 * np.cumprod(np.full(len(dates), 1.002)),
            "B": 10.0 * np.cumprod(np.full(len(dates), 1.001)),
            "C": 10.0 * np.cumprod(np.full(len(dates), 0.999)),
        },
        index=dates,
    )
    raw_open = adjusted_open.copy()
    tradable = pd.DataFrame(True, index=dates, columns=columns)
    return signal, adjusted_open, raw_open, tradable.copy(), tradable.copy()


def test_weekly_long_only_proxy_uses_prior_close_signal_and_position_cap() -> None:
    panels = _panels()
    result = AshareVectorBacktester(
        AshareVectorConfig(
            gross_exposure=0.90,
            selection_fraction=0.50,
            maximum_positions=1,
            commission_bps_each_side=0.0,
            transfer_fee_bps_each_side=0.0,
            stamp_duty_bps_sell=0.0,
            minimum_commission_cny=0.0,
            slippage_bps_each_side=0.0,
        )
    ).run(*panels, start="2024-01-02", end="2024-05-31")

    assert result.metrics["portfolio_mode"] == "long_only"
    assert result.path["position_count"].max() == 1
    assert result.path["gross"].sum() > 0
    assert 16 <= result.metrics["rebalance_count"] <= 20


def test_open_buy_constraint_blocks_untradeable_top_name() -> None:
    signal, adjusted_open, raw_open, can_buy, can_sell = _panels()
    can_buy["A"] = False
    result = AshareVectorBacktester(
        AshareVectorConfig(
            selection_fraction=0.50,
            maximum_positions=1,
            commission_bps_each_side=0.0,
            transfer_fee_bps_each_side=0.0,
            stamp_duty_bps_sell=0.0,
            minimum_commission_cny=0.0,
            slippage_bps_each_side=0.0,
        )
    ).run(
        signal,
        adjusted_open,
        raw_open,
        can_buy,
        can_sell,
        start="2024-01-02",
        end="2024-05-31",
    )

    assert result.path["gross"].mean() < 0.0015


def test_realistic_costs_reduce_equity_and_are_reported_in_cny() -> None:
    panels = _panels()
    free = AshareVectorBacktester(
        AshareVectorConfig(
            maximum_positions=1,
            commission_bps_each_side=0.0,
            transfer_fee_bps_each_side=0.0,
            stamp_duty_bps_sell=0.0,
            minimum_commission_cny=0.0,
            slippage_bps_each_side=0.0,
        )
    ).run(*panels, start="2024-01-02", end="2024-05-31")
    costed = AshareVectorBacktester(AshareVectorConfig(maximum_positions=1)).run(
        *panels, start="2024-01-02", end="2024-05-31"
    )

    assert costed.equity.iloc[-1] < free.equity.iloc[-1]
    assert costed.metrics["total_transaction_cost_cny"] > 0


def test_bankruptcy_is_a_terminal_screening_outcome_not_an_engine_error() -> None:
    signal, adjusted_open, raw_open, can_buy, can_sell = _panels()
    adjusted_open.loc[adjusted_open.index[10], "A"] = 0.000001
    result = AshareVectorBacktester(
        AshareVectorConfig(
            gross_exposure=1.0,
            selection_fraction=0.50,
            maximum_positions=1,
            commission_bps_each_side=0.0,
            transfer_fee_bps_each_side=0.0,
            stamp_duty_bps_sell=0.0,
            minimum_commission_cny=0.0,
            slippage_bps_each_side=20_000.0,
        )
    ).run(
        signal,
        adjusted_open,
        raw_open,
        can_buy,
        can_sell,
        start="2024-01-02",
        end="2024-05-31",
    )

    assert result.metrics["bankrupt"] is True
    assert result.metrics["bankruptcy_date"]
    assert result.metrics["total_return"] <= -0.999999
    terminal = result.path.index.get_loc(pd.Timestamp(result.metrics["bankruptcy_date"]))
    assert (result.path.iloc[terminal + 1 :]["net"] == 0.0).all()
