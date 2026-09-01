from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from autoalpha.backtest.target_book import rebalance_mask, select_target_positions

RebalanceSchedule = Literal[
    "WEEKLY_FIRST_SESSION", "BIWEEKLY_FIRST_SESSION", "MONTHLY_FIRST_SESSION"
]
US_PROXY_RETURN_CONVENTION = "EOD_T__OPEN_T1_TO_OPEN_T2_TOTAL_RETURN_PROXY"


@dataclass(frozen=True)
class USVectorConfig:
    """Long-only US equity execution proxy using total-return prices.

    This engine works in weight space and never sees share counts, so the
    per-share commission schedule modelled exactly in
    :class:`~autoalpha.backtest.costs.USEquityExecutionCosts` is approximated
    here as basis points. The SEC Section 31 fee is naturally proportional to
    notional and converts exactly ($27.80 per $1M is 0.278 bps); the FINRA TAF
    is per-share and small enough to fold into the commission approximation.
    Use the event ledger, not this engine, for exact fee accounting.
    """

    initial_cash_usd: float = 1_000_000.0
    gross_exposure: float = 0.90
    selection_fraction: float = 0.10
    maximum_positions: int = 30
    rebalance_schedule: RebalanceSchedule = "WEEKLY_FIRST_SESSION"
    commission_bps_each_side: float = 0.5
    sec_fee_bps_sell: float = 0.278
    slippage_bps_each_side: float = 5.0
    cost_stress_multiplier: float = 2.0
    trading_days_per_year: int = 252

    def __post_init__(self) -> None:
        if self.initial_cash_usd <= 0 or not 0 < self.gross_exposure <= 1:
            raise ValueError("initial cash and gross exposure are invalid")
        if not 0 < self.selection_fraction <= 0.5 or self.maximum_positions <= 0:
            raise ValueError("selection settings are invalid")
        if self.cost_stress_multiplier < 1:
            raise ValueError("cost stress multiplier must be at least one")
        if any(
            value < 0
            for value in (
                self.commission_bps_each_side,
                self.sec_fee_bps_sell,
                self.slippage_bps_each_side,
            )
        ):
            raise ValueError("execution cost rates must be non-negative")


@dataclass(frozen=True)
class USVectorResult:
    path: pd.DataFrame
    equity: pd.Series
    drawdown: pd.Series
    metrics: dict[str, float | int | str | bool]


class USVectorBacktester:
    """Weekly long-only vector ledger with side-specific open eligibility.

    Signals are formed after the prior session close. Trades occur at the next
    scheduled open. Adjusted open-to-open returns preserve corporate-action total
    returns while tradability flags constrain whether target changes can execute.
    US equities have no daily price limits, so those flags reflect halts and
    absent prints rather than limit-up/limit-down state.
    """

    def __init__(self, config: USVectorConfig) -> None:
        self.config = config

    def run(
        self,
        signal: pd.DataFrame,
        adjusted_open: pd.DataFrame,
        raw_open: pd.DataFrame,
        can_buy_open: pd.DataFrame,
        can_sell_open: pd.DataFrame,
        *,
        start: object,
        end: object,
    ) -> USVectorResult:
        panels = _align_panels(signal, adjusted_open, raw_open, can_buy_open, can_sell_open)
        signal, adjusted_open, raw_open, can_buy_open, can_sell_open = panels
        entry_return = adjusted_open.pct_change(fill_method=None).shift(-1)
        exit_session = pd.Series(entry_return.index, index=entry_return.index).shift(-1)
        active = (
            (entry_return.index >= pd.Timestamp(start))
            & (entry_return.index <= pd.Timestamp(end))
            & exit_session.le(pd.Timestamp(end)).fillna(False).to_numpy()
        )
        active_positions = np.flatnonzero(active)
        if active_positions.size < 60:
            raise ValueError("US vector backtest requires at least 60 entry sessions")

        dates = entry_return.index
        schedule = rebalance_mask(dates, self.config.rebalance_schedule, active)
        signal_values = signal.to_numpy(dtype=float, copy=False)
        return_values = entry_return.to_numpy(dtype=float, copy=False)
        buy_values = can_buy_open.fillna(False).to_numpy(dtype=bool, copy=False)
        sell_values = can_sell_open.fillna(False).to_numpy(dtype=bool, copy=False)
        weights = np.zeros(signal.shape[1], dtype=float)
        equity = self.config.initial_cash_usd
        rows: list[dict[str, float]] = []
        bankrupt = False
        bankruptcy_date: str | None = None

        for position in active_positions:
            if bankrupt:
                rows.append(
                    {
                        "gross": 0.0,
                        "net": 0.0,
                        "stressed": 0.0,
                        "turnover": 0.0,
                        "buy_turnover": 0.0,
                        "sell_turnover": 0.0,
                        "transaction_cost": 0.0,
                        "transaction_cost_usd": 0.0,
                        "gross_exposure": 0.0,
                        "position_count": 0.0,
                        "rebalance": 0.0,
                    }
                )
                continue
            transaction_cost = 0.0
            buy_turnover = 0.0
            sell_turnover = 0.0
            if schedule[position] and position > 0:
                desired = self._desired_weights(signal_values[position - 1])
                weights, buy_turnover, sell_turnover, transaction_cost = self._execute_target(
                    weights,
                    desired,
                    buy_values[position],
                    sell_values[position],
                    dates[position],
                    equity,
                )

            asset_returns = np.nan_to_num(return_values[position], nan=0.0, posinf=0.0, neginf=0.0)
            gross_return = float(np.dot(weights, asset_returns))
            net_return = gross_return - transaction_cost
            stressed_return = gross_return - transaction_cost * self.config.cost_stress_multiplier
            growth = 1.0 + net_return
            if not math.isfinite(growth) or growth <= 0:
                # Bankruptcy is a valid, terminal strategy outcome. Keep a tiny
                # positive ledger balance so downstream annualization remains
                # finite, then hold cash for the rest of the evaluation window.
                net_return = -1.0 + 1e-12
                stressed_return = max(-1.0 + 1e-12, stressed_return)
                growth = 1e-12
                bankrupt = True
                bankruptcy_date = dates[position].date().isoformat()
            rows.append(
                {
                    "gross": gross_return,
                    "net": net_return,
                    "stressed": stressed_return,
                    "turnover": (buy_turnover + sell_turnover) * 0.5,
                    "buy_turnover": buy_turnover,
                    "sell_turnover": sell_turnover,
                    "transaction_cost": transaction_cost,
                    "transaction_cost_usd": transaction_cost * equity,
                    "gross_exposure": float(weights.sum()),
                    "position_count": float(np.count_nonzero(weights > 1e-12)),
                    "rebalance": float(schedule[position]),
                }
            )
            equity *= growth
            if bankrupt:
                weights.fill(0.0)
            else:
                weights = weights * (1.0 + asset_returns) / growth
                weights[~np.isfinite(weights) | (weights < 0)] = 0.0

        path = pd.DataFrame(rows, index=dates[active_positions])
        equity_path = self.config.initial_cash_usd * (1.0 + path["net"]).cumprod()
        drawdown = equity_path.div(
            equity_path.cummax().clip(lower=self.config.initial_cash_usd)
        ).sub(1)
        return USVectorResult(
            path=path,
            equity=equity_path,
            drawdown=drawdown,
            metrics={
                **_metrics(path, equity_path, drawdown, self.config),
                "bankrupt": bankrupt,
                "bankruptcy_date": bankruptcy_date or "",
            },
        )

    def _desired_weights(self, signal: np.ndarray) -> np.ndarray:
        candidates = select_target_positions(
            signal,
            selection_fraction=self.config.selection_fraction,
            maximum_positions=self.config.maximum_positions,
        )
        desired = np.zeros_like(signal, dtype=float)
        if candidates.size == 0:
            return desired
        desired[candidates] = self.config.gross_exposure / candidates.size
        return desired

    def _execute_target(
        self,
        current: np.ndarray,
        desired: np.ndarray,
        can_buy: np.ndarray,
        can_sell: np.ndarray,
        trade_date: pd.Timestamp,
        equity: float,
    ) -> tuple[np.ndarray, float, float, float]:
        target = current.copy()
        sell_need = np.clip(current - desired, 0.0, None)
        sells = np.where(can_sell, sell_need, 0.0)
        target -= sells

        buy_need = np.where(can_buy, np.clip(desired - target, 0.0, None), 0.0)
        available = max(0.0, self.config.gross_exposure - float(target.sum()))
        requested = float(buy_need.sum())
        buys = buy_need * min(1.0, available / requested) if requested > 0 else buy_need
        target += buys
        cost = self._cost_rate(buys, sells, trade_date, equity)
        return target, float(buys.sum()), float(sells.sum()), cost

    def _cost_rate(
        self, buys: np.ndarray, sells: np.ndarray, trade_date: pd.Timestamp, equity: float
    ) -> float:
        """Turnover cost as a fraction of equity. Sells additionally pay the SEC fee."""

        def side_cost(changes: np.ndarray, extra_bps: float) -> float:
            active = changes[changes > 1e-12]
            if active.size == 0:
                return 0.0
            rate = (
                self.config.commission_bps_each_side
                + self.config.slippage_bps_each_side
                + extra_bps
            ) / 10_000.0
            return float(active.sum() * rate)

        return side_cost(buys, 0.0) + side_cost(sells, self.config.sec_fee_bps_sell)


def _align_panels(*panels: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    if any(not isinstance(panel.index, pd.DatetimeIndex) for panel in panels):
        raise TypeError("all US vector panels must use a DatetimeIndex")
    index = panels[0].index
    columns = panels[0].columns
    for panel in panels[1:]:
        index = index.intersection(panel.index)
        columns = columns.intersection(panel.columns, sort=False)
    if len(index) < 3 or len(columns) == 0:
        raise ValueError("US vector panels do not have enough aligned observations")
    return tuple(panel.sort_index().reindex(index=index, columns=columns) for panel in panels)


def _metrics(
    path: pd.DataFrame,
    equity: pd.Series,
    drawdown: pd.Series,
    config: USVectorConfig,
) -> dict[str, float | int | str | bool]:
    net = path["net"]
    total_growth = float((1.0 + net).prod())
    volatility = float(net.std(ddof=1))
    return {
        "simple_annual_return": float(net.mean() * config.trading_days_per_year),
        "compound_annual_return": float(
            total_growth ** (config.trading_days_per_year / len(net)) - 1.0
        ),
        "total_return": total_growth - 1.0,
        "final_equity_usd": float(equity.iloc[-1]),
        "sharpe_ratio": (
            float(net.mean() / volatility * math.sqrt(config.trading_days_per_year))
            if volatility
            else 0.0
        ),
        "annual_volatility": volatility * math.sqrt(config.trading_days_per_year),
        "max_drawdown": float(drawdown.min()),
        "annual_turnover": float(path["turnover"].mean() * config.trading_days_per_year),
        "total_transaction_cost_usd": float(path["transaction_cost_usd"].sum()),
        "average_gross_exposure": float(path["gross_exposure"].mean()),
        "average_positions": float(path["position_count"].mean()),
        "rebalance_count": int(path["rebalance"].sum()),
        "observations": len(path),
        "backtest_start": path.index.min().date().isoformat(),
        "backtest_end": path.index.max().date().isoformat(),
        "portfolio_mode": "long_only",
        "rebalance_schedule": config.rebalance_schedule,
        "execution_lag_sessions": 1,
        "signal_availability": "END_OF_DAY_AFTER_CLOSE",
        "return_convention": US_PROXY_RETURN_CONVENTION,
        "execution_price_basis": "TRADEABILITY_CONSTRAINED_ADJUSTED_OPEN_TOTAL_RETURN",
        "production_eligible": False,
    }
