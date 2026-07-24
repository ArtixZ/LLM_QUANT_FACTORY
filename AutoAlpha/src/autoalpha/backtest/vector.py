from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from autoalpha.backtest.timing import (
    EOD_NEXT_OPEN_RETURN_CONVENTION,
    entry_aligned_open_return,
    next_open_return_for_eod_signal,
)

CostModel = Literal["legacy_half_turnover", "side_aware"]
PathIndex = Literal["entry_session", "signal_session"]
SelectionMethod = Literal["percentile", "percentile_with_ordinal_cap"]


@dataclass(frozen=True)
class VectorBacktestConfig:
    """Configuration for a causal cross-sectional vector backtest.

    Signals are assumed to become available after the close. They are executed at
    the next session open and earn the following open-to-open return.
    """

    holding_period_days: int = 5
    gross_exposure: float = 1.0
    selection_fraction: float = 0.10
    maximum_positions_per_side: int | None = None
    long_only: bool = False
    commission_bps_each_side: float = 1.5
    stamp_duty_bps_sell: float = 5.0
    transfer_fee_bps_each_side: float = 0.1
    slippage_bps_each_side: float = 0.0
    cost_stress_multiplier: float = 2.0
    cost_model: CostModel = "side_aware"
    path_index: PathIndex = "entry_session"
    selection_method: SelectionMethod = "percentile_with_ordinal_cap"
    initial_cash_cny: float = 1_000_000.0
    trading_days_per_year: int = 245

    def __post_init__(self) -> None:
        if self.holding_period_days <= 0:
            raise ValueError("holding_period_days must be positive")
        if not 0 < self.gross_exposure <= 2:
            raise ValueError("gross_exposure must be in (0, 2]")
        if not 0 < self.selection_fraction <= 0.5:
            raise ValueError("selection_fraction must be in (0, 0.5]")
        if self.maximum_positions_per_side is not None and self.maximum_positions_per_side <= 0:
            raise ValueError("maximum_positions_per_side must be positive when provided")
        if (
            min(
                self.commission_bps_each_side,
                self.stamp_duty_bps_sell,
                self.transfer_fee_bps_each_side,
                self.slippage_bps_each_side,
            )
            < 0
        ):
            raise ValueError("execution costs cannot be negative")
        if self.cost_stress_multiplier < 1:
            raise ValueError("cost_stress_multiplier must be at least one")
        if self.initial_cash_cny <= 0 or self.trading_days_per_year <= 0:
            raise ValueError("capital and annualization settings must be positive")
        if self.selection_method == "percentile" and self.maximum_positions_per_side is not None:
            raise ValueError("percentile selection does not support a hard position cap")


@dataclass(frozen=True)
class VectorBacktestResult:
    positions: pd.DataFrame
    target_weights: pd.DataFrame
    formed_weights: pd.DataFrame
    held_weights: pd.DataFrame
    path: pd.DataFrame
    equity: pd.Series
    drawdown: pd.Series
    metrics: dict[str, float | int | str]


@dataclass(frozen=True)
class VectorReconciliation:
    passed: bool
    observations: int
    tolerance: float
    maximum_absolute_difference: dict[str, float]
    metric_difference: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "observations": self.observations,
            "tolerance": self.tolerance,
            "maximum_absolute_difference": self.maximum_absolute_difference,
            "metric_difference": self.metric_difference,
        }


class VectorBacktester:
    """Matrix-based research backtester with explicit timing and fee semantics."""

    def __init__(self, config: VectorBacktestConfig) -> None:
        self.config = config

    def run(
        self,
        signal: pd.DataFrame,
        open_prices: pd.DataFrame,
        *,
        start: object | None = None,
        end: object | None = None,
        precomputed_entry_returns: pd.DataFrame | None = None,
    ) -> VectorBacktestResult:
        signal, open_prices = _aligned_panels(signal, open_prices)
        positions = select_positions(
            signal,
            selection_fraction=self.config.selection_fraction,
            maximum_positions_per_side=self.config.maximum_positions_per_side,
            long_only=self.config.long_only,
            method=self.config.selection_method,
        )
        gross = positions.abs().sum(axis=1).replace(0, np.nan)
        target_weights = positions.div(gross, axis=0).fillna(0.0) * self.config.gross_exposure
        formed_weights = target_weights.rolling(
            self.config.holding_period_days, min_periods=1
        ).mean()

        if self.config.path_index == "entry_session":
            held_weights = formed_weights.shift(1)
            realized_return = (
                precomputed_entry_returns.reindex(
                    index=open_prices.index, columns=open_prices.columns
                )
                if precomputed_entry_returns is not None
                else entry_aligned_open_return(open_prices)
            )
            if end is not None:
                exit_session = pd.Series(realized_return.index, index=realized_return.index).shift(
                    -1
                )
                realized_return = realized_return.where(exit_session.le(pd.Timestamp(end)), axis=0)
        else:
            held_weights = formed_weights
            realized_return = next_open_return_for_eod_signal(open_prices)
            if end is not None:
                exit_session = pd.Series(realized_return.index, index=realized_return.index).shift(
                    -2
                )
                realized_return = realized_return.where(exit_session.le(pd.Timestamp(end)), axis=0)

        gross_return = (held_weights * realized_return).sum(axis=1, min_count=1).dropna()
        if self.config.cost_model == "legacy_half_turnover":
            # Historical AutoAlpha behavior, retained only for exact result reconciliation.
            weight_change = held_weights.diff().reindex(gross_return.index).fillna(0.0)
        else:
            funded_weights = held_weights.fillna(0.0)
            weight_change = funded_weights.diff()
            weight_change.iloc[0] = funded_weights.iloc[0]
            weight_change = weight_change.reindex(gross_return.index).fillna(0.0)
        turnover = weight_change.abs().sum(axis=1).mul(0.5)
        buy_turnover = weight_change.clip(lower=0).sum(axis=1)
        sell_turnover = weight_change.clip(upper=0).abs().sum(axis=1)
        base_cost = self._transaction_cost(turnover, buy_turnover, sell_turnover)
        stressed_cost = self._transaction_cost(
            turnover,
            buy_turnover,
            sell_turnover,
            multiplier=self.config.cost_stress_multiplier,
        )
        path = pd.DataFrame(
            {
                "gross": gross_return,
                "net": gross_return - base_cost,
                "stressed": gross_return - stressed_cost,
                "turnover": turnover,
                "buy_turnover": buy_turnover,
                "sell_turnover": sell_turnover,
                "transaction_cost": base_cost,
            }
        ).dropna()
        if start is not None:
            path = path.loc[path.index >= pd.Timestamp(start)]
        if end is not None:
            path = path.loc[path.index <= pd.Timestamp(end)]

        net = path["net"]
        equity = self.config.initial_cash_cny * (1.0 + net).cumprod()
        drawdown = equity.div(equity.cummax().clip(lower=self.config.initial_cash_cny)).sub(1.0)
        metrics = _metrics(path, equity, drawdown, self.config)
        return VectorBacktestResult(
            positions=positions,
            target_weights=target_weights,
            formed_weights=formed_weights,
            held_weights=held_weights,
            path=path,
            equity=equity,
            drawdown=drawdown,
            metrics=metrics,
        )

    def _transaction_cost(
        self,
        turnover: pd.Series,
        buy_turnover: pd.Series,
        sell_turnover: pd.Series,
        *,
        multiplier: float = 1.0,
    ) -> pd.Series:
        if self.config.cost_model == "legacy_half_turnover":
            one_way_bps = (
                self.config.commission_bps_each_side
                + self.config.transfer_fee_bps_each_side
                + self.config.stamp_duty_bps_sell / 2
                + self.config.slippage_bps_each_side
            )
            return turnover * one_way_bps * multiplier / 10_000
        both_side_bps = (
            self.config.commission_bps_each_side
            + self.config.transfer_fee_bps_each_side
            + self.config.slippage_bps_each_side
        )
        return (
            (
                buy_turnover * both_side_bps
                + sell_turnover * (both_side_bps + self.config.stamp_duty_bps_sell)
            )
            * multiplier
            / 10_000
        )


def select_positions(
    signal: pd.DataFrame,
    *,
    selection_fraction: float,
    maximum_positions_per_side: int | None,
    long_only: bool,
    method: SelectionMethod,
) -> pd.DataFrame:
    ranks = signal.rank(axis=1, pct=True)
    if method == "percentile":
        long_positions = (ranks >= 1.0 - selection_fraction).astype(float)
        short_positions = (ranks <= selection_fraction).astype(float)
    else:
        if maximum_positions_per_side is None:
            raise ValueError("ordinal-cap selection requires maximum_positions_per_side")
        ordinal_long = signal.rank(axis=1, ascending=False, method="first")
        ordinal_short = signal.rank(axis=1, ascending=True, method="first")
        long_positions = (
            (ranks >= 1.0 - selection_fraction) & (ordinal_long <= maximum_positions_per_side)
        ).astype(float)
        short_positions = (
            (ranks <= selection_fraction) & (ordinal_short <= maximum_positions_per_side)
        ).astype(float)
    if long_only:
        return long_positions
    return long_positions - short_positions


def reconcile_vector_paths(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    tolerance: float = 1e-12,
) -> VectorReconciliation:
    columns = ["gross", "net", "stressed", "turnover"]
    missing = [column for column in columns if column not in reference or column not in candidate]
    if missing:
        raise ValueError(f"Reconciliation paths are missing columns: {missing}")
    common = reference.index.intersection(candidate.index)
    if common.empty:
        raise ValueError("Reconciliation paths have no common dates")
    maximum = {
        column: float((reference.loc[common, column] - candidate.loc[common, column]).abs().max())
        for column in columns
    }
    reference_net = reference.loc[common, "net"]
    candidate_net = candidate.loc[common, "net"]
    metric_difference = {
        "simple_annual_return": float((candidate_net.mean() - reference_net.mean()) * 245),
        "sharpe_ratio": float(_annualized_ratio(candidate_net) - _annualized_ratio(reference_net)),
        "total_return": float((1.0 + candidate_net).prod() - (1.0 + reference_net).prod()),
    }
    return VectorReconciliation(
        passed=len(common) == len(reference) == len(candidate)
        and all(value <= tolerance for value in maximum.values()),
        observations=len(common),
        tolerance=tolerance,
        maximum_absolute_difference=maximum,
        metric_difference=metric_difference,
    )


def _aligned_panels(
    signal: pd.DataFrame, open_prices: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not isinstance(signal.index, pd.DatetimeIndex) or not isinstance(
        open_prices.index, pd.DatetimeIndex
    ):
        raise TypeError("signal and open_prices must use DatetimeIndex")
    if signal.index.has_duplicates or open_prices.index.has_duplicates:
        raise ValueError("signal and open_prices indexes must be unique")
    columns = signal.columns.intersection(open_prices.columns, sort=False)
    common_dates = signal.index.intersection(open_prices.index)
    insufficient_observations = (
        len(signal.index) < 3 or len(open_prices.index) < 3 or len(common_dates) < 3
    )
    if insufficient_observations or len(columns) == 0:
        raise ValueError("signal and open_prices do not have enough aligned observations")
    return (
        signal.sort_index().reindex(columns=columns).astype(float),
        open_prices.sort_index().reindex(columns=columns).astype(float),
    )


def _metrics(
    path: pd.DataFrame,
    equity: pd.Series,
    drawdown: pd.Series,
    config: VectorBacktestConfig,
) -> dict[str, float | int | str]:
    if path.empty:
        return {
            "observations": 0,
            "cost_model": config.cost_model,
            "return_convention": EOD_NEXT_OPEN_RETURN_CONVENTION,
        }
    net = path["net"]
    total_growth = float((1.0 + net).prod())
    return {
        "simple_annual_return": float(net.mean() * config.trading_days_per_year),
        "compound_annual_return": float(
            total_growth ** (config.trading_days_per_year / len(net)) - 1.0
        ),
        "total_return": float(total_growth - 1.0),
        "final_equity_cny": float(equity.iloc[-1]),
        "sharpe_ratio": _annualized_ratio(net, config.trading_days_per_year),
        "annual_volatility": float(net.std(ddof=1) * math.sqrt(config.trading_days_per_year)),
        "max_drawdown": float(drawdown.min()),
        "annual_turnover": float(path["turnover"].mean() * config.trading_days_per_year),
        "total_transaction_cost_return": float(path["transaction_cost"].sum()),
        "observations": len(path),
        "backtest_start": path.index.min().date().isoformat(),
        "backtest_end": path.index.max().date().isoformat(),
        "cost_model": config.cost_model,
        "path_index": config.path_index,
        "signal_availability": "END_OF_DAY_AFTER_CLOSE",
        "execution_lag_sessions": 1,
        "return_convention": EOD_NEXT_OPEN_RETURN_CONVENTION,
    }


def _annualized_ratio(values: pd.Series, trading_days: int = 245) -> float:
    standard_deviation = float(values.std(ddof=1))
    return (
        float(values.mean() / standard_deviation * math.sqrt(trading_days))
        if standard_deviation
        else 0.0
    )
