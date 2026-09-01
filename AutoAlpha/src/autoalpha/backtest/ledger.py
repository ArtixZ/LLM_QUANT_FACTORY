from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from autoalpha.backtest.costs import USEquityExecutionCosts
from autoalpha.backtest.target_book import is_rebalance_session, select_target_symbols

REQUIRED_MARKET_COLUMNS = frozenset(
    {"date", "symbol", "open", "close", "volume", "can_buy_open", "can_sell_open"}
)
RebalanceSchedule = Literal[
    "DAILY_ROLLING",
    "WEEKLY_FIRST_SESSION",
    "MONTHLY_FIRST_SESSION",
]


@dataclass(frozen=True)
class LedgerConfig:
    horizon: int
    initial_cash: float = 100_000_000.0
    top_fraction: float = 0.10
    max_positions: int | None = None
    lot_size: int = 1
    max_volume_participation: float = 0.05
    investment_buffer: float = 0.002
    trading_days_per_year: int = 252
    rebalance_schedule: RebalanceSchedule = "DAILY_ROLLING"
    slippage_bps_each_side: float = 0.0

    def __post_init__(self) -> None:
        if self.horizon <= 0 or self.initial_cash <= 0 or self.lot_size <= 0:
            raise ValueError("horizon, initial_cash, and lot_size must be positive")
        if not 0 < self.top_fraction <= 1:
            raise ValueError("top_fraction must be in (0, 1]")
        if self.max_positions is not None and self.max_positions <= 0:
            raise ValueError("max_positions must be positive when provided")
        if not 0 < self.max_volume_participation <= 1:
            raise ValueError("max_volume_participation must be in (0, 1]")
        if not 0 <= self.investment_buffer < 1:
            raise ValueError("investment_buffer must be in [0, 1)")
        if self.rebalance_schedule not in {
            "DAILY_ROLLING",
            "WEEKLY_FIRST_SESSION",
            "MONTHLY_FIRST_SESSION",
        }:
            raise ValueError("unsupported rebalance_schedule")
        if not 0 <= self.slippage_bps_each_side < 10_000:
            raise ValueError("slippage_bps_each_side must be in [0, 10000)")


@dataclass
class _Sleeve:
    cash: float
    positions: dict[str, int] = field(default_factory=dict)
    pending_target: tuple[str, ...] | None = None
    signal_date: pd.Timestamp | None = None


@dataclass(frozen=True)
class LedgerResult:
    nav: pd.Series
    daily_return: pd.Series
    gross_exposure: pd.Series
    trades: pd.DataFrame
    final_positions: pd.DataFrame
    annual_return: float
    sharpe: float
    max_drawdown: float
    total_fees: float


class LedgerBacktester:
    def __init__(
        self,
        config: LedgerConfig,
        costs: USEquityExecutionCosts | None = None,
    ) -> None:
        self.config = config
        self.costs = costs or USEquityExecutionCosts()

    def run(self, signal: pd.DataFrame, market: pd.DataFrame) -> LedgerResult:
        _validate_inputs(signal, market)
        panels = _market_panels(market)
        dates = panels["close"].index
        symbols = panels["close"].columns
        previous_close = panels["valuation_close"].shift(1)
        signal = signal.reindex(index=dates, columns=symbols)
        sleeve_count = (
            self.config.horizon if self.config.rebalance_schedule == "DAILY_ROLLING" else 1
        )
        sleeves = [
            _Sleeve(cash=self.config.initial_cash / sleeve_count)
            for _ in range(sleeve_count)
        ]
        trade_records: list[dict[str, Any]] = []
        nav_values: list[float] = []
        gross_exposure_values: list[float] = []

        for date_index, current_date in enumerate(dates):
            open_prices = panels["open"].loc[current_date]
            close_prices = panels["valuation_close"].loc[current_date]
            can_buy = panels["can_buy_open"].loc[current_date]
            can_sell = panels["can_sell_open"].loc[current_date]
            volumes = panels["volume"].loc[current_date]

            if date_index > 0 and is_rebalance_session(
                dates,
                date_index,
                self.config.rebalance_schedule,
            ):
                signal_date = dates[date_index - 1]
                sleeve_index = (
                    (date_index - 1) % self.config.horizon
                    if self.config.rebalance_schedule == "DAILY_ROLLING"
                    else 0
                )
                sleeve = sleeves[sleeve_index]
                sleeve.pending_target = select_target_symbols(
                    signal.loc[signal_date],
                    selection_fraction=self.config.top_fraction,
                    maximum_positions=self.config.max_positions,
                )
                sleeve.signal_date = signal_date

            for sleeve_id, sleeve in enumerate(sleeves):
                if sleeve.pending_target is not None:
                    complete = self._rebalance(
                        sleeve=sleeve,
                        sleeve_id=sleeve_id,
                        trade_date=current_date,
                        open_prices=open_prices,
                        previous_close=previous_close.loc[current_date],
                        volumes=volumes,
                        can_buy=can_buy,
                        can_sell=can_sell,
                        trade_records=trade_records,
                    )
                    if complete:
                        sleeve.pending_target = None
            nav_value = sum(_sleeve_value(sleeve, close_prices) for sleeve in sleeves)
            invested_value = sum(_positions_value(sleeve, close_prices) for sleeve in sleeves)
            nav_values.append(nav_value)
            gross_exposure_values.append(invested_value / nav_value if nav_value > 0 else 0.0)

        nav = pd.Series(nav_values, index=dates, name="nav", dtype=float)
        daily_return = nav.pct_change(fill_method=None).fillna(0.0).rename("daily_return")
        gross_exposure = pd.Series(
            gross_exposure_values, index=dates, name="gross_exposure", dtype=float
        )
        trades = pd.DataFrame.from_records(trade_records, columns=_trade_columns())
        final_positions = _positions_frame(sleeves)
        annual_return, sharpe, max_drawdown = _performance_metrics(
            nav, daily_return, self.config.trading_days_per_year
        )
        total_fees = float(trades["fees"].sum()) if not trades.empty else 0.0
        return LedgerResult(
            nav=nav,
            daily_return=daily_return,
            gross_exposure=gross_exposure,
            trades=trades,
            final_positions=final_positions,
            annual_return=annual_return,
            sharpe=sharpe,
            max_drawdown=max_drawdown,
            total_fees=total_fees,
        )

    def _rebalance(
        self,
        *,
        sleeve: _Sleeve,
        sleeve_id: int,
        trade_date: pd.Timestamp,
        open_prices: pd.Series,
        previous_close: pd.Series,
        volumes: pd.Series,
        can_buy: pd.Series,
        can_sell: pd.Series,
        trade_records: list[dict[str, Any]],
    ) -> bool:
        targets = sleeve.pending_target or ()
        open_nav = _sleeve_value(sleeve, open_prices, fallback=previous_close)
        target_notional = (
            open_nav * (1.0 - self.config.investment_buffer) / len(targets) if targets else 0.0
        )
        desired = {
            symbol: _round_lot(target_notional / float(open_prices[symbol]), self.config.lot_size)
            for symbol in targets
            if _positive_finite(open_prices.get(symbol))
        }

        incomplete = False
        held_symbols = sorted(set(sleeve.positions) | set(desired))
        for symbol in held_symbols:
            current = sleeve.positions.get(symbol, 0)
            excess = current - desired.get(symbol, 0)
            if excess < self.config.lot_size:
                continue
            price = open_prices.get(symbol)
            if not _positive_finite(price) or not bool(can_sell.get(symbol, False)):
                incomplete = True
                continue
            quantity = self._volume_limited_quantity(excess, volumes.get(symbol))
            if quantity <= 0:
                incomplete = True
                continue
            self._execute(
                sleeve,
                sleeve_id,
                trade_date,
                symbol,
                "SELL",
                quantity,
                float(price),
                trade_records,
            )
            if quantity < excess:
                incomplete = True

        for symbol in targets:
            current = sleeve.positions.get(symbol, 0)
            deficit = desired.get(symbol, 0) - current
            if deficit < self.config.lot_size:
                continue
            price = open_prices.get(symbol)
            if not _positive_finite(price) or not bool(can_buy.get(symbol, False)):
                incomplete = True
                continue
            quantity = self._volume_limited_quantity(deficit, volumes.get(symbol))
            execution_price = self._execution_price(float(price), "BUY")
            affordable = self.costs.affordable_shares(sleeve.cash, execution_price, trade_date)
            quantity = min(quantity, affordable)
            if quantity <= 0:
                if any(
                    shares >= self.config.lot_size
                    for held_symbol, shares in sleeve.positions.items()
                    if held_symbol not in targets
                ):
                    incomplete = True
                continue
            self._execute(
                sleeve,
                sleeve_id,
                trade_date,
                symbol,
                "BUY",
                quantity,
                float(price),
                trade_records,
            )
            if quantity < deficit:
                incomplete = True
        return not incomplete

    def _volume_limited_quantity(self, desired: int, volume: Any) -> int:
        if not _positive_finite(volume):
            return 0
        limit = _round_lot(
            float(volume) * self.config.max_volume_participation,
            self.config.lot_size,
        )
        return min(desired, limit)

    def _execute(
        self,
        sleeve: _Sleeve,
        sleeve_id: int,
        trade_date: pd.Timestamp,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        records: list[dict[str, Any]],
    ) -> None:
        reference_price = price
        price = self._execution_price(reference_price, side)
        notional = quantity * price
        breakdown = self.costs.fee_breakdown(  # type: ignore[arg-type]
            side,
            notional,
            quantity,
            trade_date,
        )
        fees = sum(breakdown.values())
        if side == "SELL":
            quantity = min(quantity, sleeve.positions.get(symbol, 0))
            notional = quantity * price
            breakdown = self.costs.fee_breakdown("SELL", notional, quantity, trade_date)
            fees = sum(breakdown.values())
            sleeve.positions[symbol] = sleeve.positions.get(symbol, 0) - quantity
            if sleeve.positions[symbol] == 0:
                del sleeve.positions[symbol]
            sleeve.cash += notional - fees
        else:
            total = notional + fees
            if total > sleeve.cash + 1e-7:
                raise RuntimeError("Buy execution would make sleeve cash negative")
            sleeve.positions[symbol] = sleeve.positions.get(symbol, 0) + quantity
            sleeve.cash -= total
        records.append(
            {
                "date": trade_date,
                "signal_date": sleeve.signal_date,
                "sleeve": sleeve_id,
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "reference_price": reference_price,
                "slippage_bps": self.config.slippage_bps_each_side,
                "notional": notional,
                **breakdown,
                "fees": fees,
                "net_cash_flow": notional - fees if side == "SELL" else -(notional + fees),
                "cash_after": sleeve.cash,
            }
        )

    def _execution_price(self, reference_price: float, side: str) -> float:
        direction = 1.0 if side == "BUY" else -1.0
        return reference_price * (
            1.0 + direction * self.config.slippage_bps_each_side / 10_000.0
        )


def _validate_inputs(signal: pd.DataFrame, market: pd.DataFrame) -> None:
    if not isinstance(signal, pd.DataFrame) or not isinstance(signal.index, pd.DatetimeIndex):
        raise TypeError("signal must be a DataFrame with a DatetimeIndex")
    missing = REQUIRED_MARKET_COLUMNS - set(market.columns)
    if missing:
        raise ValueError(f"market is missing required columns: {sorted(missing)}")
    if market.duplicated(["date", "symbol"]).any():
        raise ValueError("market contains duplicate (date, symbol) rows")
    if (market[["open", "close"]].dropna() <= 0).any().any():
        raise ValueError("market prices must be positive")
    if (market["volume"].dropna() < 0).any():
        raise ValueError("market volume must be non-negative")


def _market_panels(market: pd.DataFrame) -> dict[str, pd.DataFrame]:
    data = market.copy()
    data["date"] = pd.to_datetime(data["date"])
    data["symbol"] = data["symbol"].astype(str)
    panels = {
        column: data.pivot(index="date", columns="symbol", values=column).sort_index()
        for column in ("open", "close", "volume", "can_buy_open", "can_sell_open")
    }
    panels["valuation_close"] = panels["close"].ffill()
    return panels


def _sleeve_value(
    sleeve: _Sleeve,
    prices: pd.Series,
    *,
    fallback: pd.Series | None = None,
) -> float:
    value = sleeve.cash
    for symbol, shares in sleeve.positions.items():
        price = prices.get(symbol)
        if not _positive_finite(price) and fallback is not None:
            price = fallback.get(symbol)
        if not _positive_finite(price):
            raise ValueError(f"No valuation price for held symbol {symbol!r}")
        value += shares * float(price)
    return float(value)


def _positions_value(sleeve: _Sleeve, prices: pd.Series) -> float:
    return float(
        sum(
            shares * float(prices[symbol])
            for symbol, shares in sleeve.positions.items()
            if _positive_finite(prices.get(symbol))
        )
    )


def _round_lot(shares: float, lot_size: int) -> int:
    if not np.isfinite(shares) or shares <= 0:
        return 0
    return int(math.floor(shares / lot_size) * lot_size)


def _positive_finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(value) and float(value) > 0)
    except (TypeError, ValueError):
        return False


def _trade_columns() -> list[str]:
    return [
        "date",
        "signal_date",
        "sleeve",
        "symbol",
        "side",
        "quantity",
        "price",
        "reference_price",
        "slippage_bps",
        "notional",
        "commission",
        "sec_fee",
        "finra_taf",
        "fees",
        "net_cash_flow",
        "cash_after",
    ]


def _positions_frame(sleeves: list[_Sleeve]) -> pd.DataFrame:
    rows = [
        {"sleeve": sleeve_id, "symbol": symbol, "shares": shares}
        for sleeve_id, sleeve in enumerate(sleeves)
        for symbol, shares in sorted(sleeve.positions.items())
    ]
    return pd.DataFrame.from_records(rows, columns=["sleeve", "symbol", "shares"])


def _performance_metrics(
    nav: pd.Series,
    daily_return: pd.Series,
    trading_days: int,
) -> tuple[float, float, float]:
    periods = max(len(nav) - 1, 1)
    annual_return = float((nav.iloc[-1] / nav.iloc[0]) ** (trading_days / periods) - 1.0)
    volatility = float(daily_return.iloc[1:].std())
    sharpe = (
        float(daily_return.iloc[1:].mean() / volatility * np.sqrt(trading_days))
        if volatility > 0
        else float("nan")
    )
    drawdown = nav / nav.cummax() - 1.0
    return annual_return, sharpe, float(drawdown.min())
