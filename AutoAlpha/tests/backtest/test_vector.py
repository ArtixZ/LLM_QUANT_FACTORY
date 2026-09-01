import numpy as np
import pandas as pd
import pytest

from autoalpha.backtest.timing import entry_aligned_open_return
from autoalpha.backtest.vector import (
    VectorBacktestConfig,
    VectorBacktester,
    reconcile_vector_paths,
)


def _panels(periods: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.bdate_range("2024-01-02", periods=periods)
    columns = [f"S{number}" for number in range(6)]
    signal = pd.DataFrame(
        np.vstack([np.roll(np.arange(6, dtype=float), shift) for shift in range(periods)]),
        index=index,
        columns=columns,
    )
    open_prices = pd.DataFrame(
        {
            column: 10.0 * np.cumprod(1.0 + 0.001 * (number + 1) + np.arange(periods) / 10000)
            for number, column in enumerate(columns)
        },
        index=index,
    )
    return signal, open_prices


def _legacy_reference(
    signal: pd.DataFrame, open_prices: pd.DataFrame, config: VectorBacktestConfig
) -> pd.DataFrame:
    ranks = signal.rank(axis=1, pct=True)
    ordinal_long = signal.rank(axis=1, ascending=False, method="first")
    ordinal_short = signal.rank(axis=1, ascending=True, method="first")
    positions = (
        (ranks >= 1.0 - config.selection_fraction)
        & (ordinal_long <= config.maximum_positions_per_side)
    ).astype(float) - (
        (ranks <= config.selection_fraction) & (ordinal_short <= config.maximum_positions_per_side)
    ).astype(float)
    gross = positions.abs().sum(axis=1).replace(0, np.nan)
    target = positions.div(gross, axis=0).fillna(0.0) * config.gross_exposure
    held = target.rolling(config.holding_period_days, min_periods=1).mean().shift(1)
    gross_return = (held * entry_aligned_open_return(open_prices)).sum(axis=1, min_count=1).dropna()
    turnover = held.diff().abs().sum(axis=1).mul(0.5).reindex(gross_return.index).fillna(0)
    one_way_bps = config.commission_bps_each_side + config.sec_fee_bps_sell / 2
    return pd.DataFrame(
        {
            "gross": gross_return,
            "net": gross_return - turnover * one_way_bps / 10_000,
            "stressed": gross_return
            - turnover * one_way_bps * config.cost_stress_multiplier / 10_000,
            "turnover": turnover,
        }
    ).dropna()


def test_legacy_mode_reconciles_exactly_with_existing_manual_formula() -> None:
    signal, open_prices = _panels()
    config = VectorBacktestConfig(
        holding_period_days=3,
        gross_exposure=0.5,
        selection_fraction=0.34,
        maximum_positions_per_side=2,
        cost_model="legacy_half_turnover",
    )
    candidate = VectorBacktester(config).run(signal, open_prices).path
    reference = _legacy_reference(signal, open_prices, config)

    reconciliation = reconcile_vector_paths(reference, candidate)

    assert reconciliation.passed
    assert max(reconciliation.maximum_absolute_difference.values()) == pytest.approx(0.0)


def test_side_aware_costs_charge_each_traded_side() -> None:
    signal, open_prices = _panels()
    common = dict(
        holding_period_days=1,
        selection_fraction=0.34,
        maximum_positions_per_side=1,
    )
    legacy = VectorBacktester(
        VectorBacktestConfig(**common, cost_model="legacy_half_turnover")
    ).run(signal, open_prices)
    corrected = VectorBacktester(VectorBacktestConfig(**common, cost_model="side_aware")).run(
        signal, open_prices
    )

    assert corrected.path["transaction_cost"].sum() > legacy.path["transaction_cost"].sum()
    # Buys pay commission only; sells additionally pay the SEC Section 31 fee.
    expected = (
        corrected.path["buy_turnover"] * 0.5 + corrected.path["sell_turnover"] * 0.778
    ) / 10_000
    pd.testing.assert_series_equal(corrected.path["transaction_cost"], expected, check_names=False)


def test_future_price_change_does_not_alter_earlier_signal_returns() -> None:
    signal, open_prices = _panels()
    config = VectorBacktestConfig(
        holding_period_days=1,
        selection_fraction=0.34,
        maximum_positions_per_side=1,
        path_index="signal_session",
    )
    original = VectorBacktester(config).run(signal, open_prices).path
    changed_prices = open_prices.copy()
    changed_prices.iloc[8:] *= 3.0
    changed = VectorBacktester(config).run(signal, changed_prices).path

    cutoff = signal.index[5]
    pd.testing.assert_series_equal(original.loc[:cutoff, "gross"], changed.loc[:cutoff, "gross"])


def test_end_date_excludes_returns_exiting_after_requested_window() -> None:
    signal, open_prices = _panels()
    end = signal.index[7]
    config = VectorBacktestConfig(
        selection_fraction=0.34,
        maximum_positions_per_side=1,
        path_index="entry_session",
    )

    result = VectorBacktester(config).run(signal, open_prices, end=end)

    assert result.path.index.max() == signal.index[6]


def test_precomputed_entry_returns_reproduce_default_vector_path() -> None:
    signal, open_prices = _panels()
    config = VectorBacktestConfig(
        holding_period_days=3,
        selection_fraction=0.34,
        maximum_positions_per_side=2,
    )
    backtester = VectorBacktester(config)

    default = backtester.run(signal, open_prices)
    cached = backtester.run(
        signal,
        open_prices,
        precomputed_entry_returns=entry_aligned_open_return(open_prices),
    )

    pd.testing.assert_frame_equal(default.path, cached.path)
