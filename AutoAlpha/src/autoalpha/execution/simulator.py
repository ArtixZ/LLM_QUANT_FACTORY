from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

from autoalpha.backtest.costs import Side, USEquityExecutionCosts


class ExecutionStyle(StrEnum):
    OPEN = "OPEN"
    CLOSE = "CLOSE"
    TWAP = "TWAP"
    VWAP = "VWAP"
    POV = "POV"


@dataclass(frozen=True)
class Order:
    order_id: str
    symbol: str
    side: Side
    quantity: int
    decision_price: float
    style: ExecutionStyle = ExecutionStyle.VWAP
    maximum_participation: float = 0.10
    lot_size: int = 1


@dataclass(frozen=True)
class MarketImpactModel:
    half_spread_bps: float = 2.0
    square_root_coefficient: float = 0.50

    def impact_bps(self, quantity: int, adv_shares: float, daily_volatility: float) -> float:
        if quantity <= 0 or adv_shares <= 0:
            return 0.0
        participation = quantity / adv_shares
        return float(
            self.square_root_coefficient
            * max(0.0, daily_volatility)
            * math.sqrt(participation)
            * 10_000
        )


@dataclass(frozen=True)
class ExecutionReport:
    order: Order
    fills: pd.DataFrame
    filled_quantity: int
    unfilled_quantity: int
    explicit_fees: float
    spread_cost: float
    impact_cost: float
    opportunity_cost: float
    gross_notional: float

    @property
    def total_cost(self) -> float:
        return self.explicit_fees + self.spread_cost + self.impact_cost + self.opportunity_cost


class ExecutionSimulator:
    def __init__(
        self,
        impact: MarketImpactModel | None = None,
        fees: USEquityExecutionCosts | None = None,
    ) -> None:
        self.impact = impact or MarketImpactModel()
        self.fees = fees or USEquityExecutionCosts()

    def execute(
        self,
        order: Order,
        market_slices: pd.DataFrame,
        *,
        adv_shares: float,
        daily_volatility: float,
        alpha_decay_bps: float = 0.0,
    ) -> ExecutionReport:
        _validate_order_and_market(order, market_slices)
        weights = _schedule_weights(order.style, market_slices)
        remaining = order.quantity
        records: list[dict[str, object]] = []
        spread_cost = 0.0
        impact_cost = 0.0
        explicit_fees = 0.0
        gross_notional = 0.0

        for position, (timestamp, row) in enumerate(market_slices.iterrows()):
            if remaining < order.lot_size or not bool(row["can_trade"]):
                continue
            if order.style is ExecutionStyle.POV:
                desired = float(row["volume"]) * order.maximum_participation
            else:
                if weights[position] <= 0:
                    continue
                desired = max(order.lot_size, order.quantity * weights[position])
            slice_limit = float(row["volume"]) * order.maximum_participation
            quantity = _round_lot(min(remaining, desired, slice_limit), order.lot_size)
            if quantity <= 0:
                continue
            market_price = float(row["price"])
            impact_bps = self.impact.impact_bps(quantity, adv_shares, daily_volatility)
            total_move_bps = self.impact.half_spread_bps + impact_bps
            direction = 1.0 if order.side == "BUY" else -1.0
            fill_price = market_price * (1 + direction * total_move_bps / 10_000)
            notional = quantity * fill_price
            fees = self.fees.fees(order.side, notional, quantity)
            slice_spread = quantity * market_price * self.impact.half_spread_bps / 10_000
            slice_impact = quantity * market_price * impact_bps / 10_000
            records.append(
                {
                    "order_id": order.order_id,
                    "timestamp": timestamp,
                    "symbol": order.symbol,
                    "side": order.side,
                    "quantity": quantity,
                    "market_price": market_price,
                    "fill_price": fill_price,
                    "notional": notional,
                    "fees": fees,
                    "spread_cost": slice_spread,
                    "impact_cost": slice_impact,
                    "participation": quantity / float(row["volume"]),
                }
            )
            remaining -= quantity
            explicit_fees += fees
            spread_cost += slice_spread
            impact_cost += slice_impact
            gross_notional += notional

        opportunity_cost = remaining * order.decision_price * max(0.0, alpha_decay_bps) / 10_000
        fills = pd.DataFrame.from_records(records, columns=_fill_columns())
        return ExecutionReport(
            order=order,
            fills=fills,
            filled_quantity=order.quantity - remaining,
            unfilled_quantity=remaining,
            explicit_fees=float(explicit_fees),
            spread_cost=float(spread_cost),
            impact_cost=float(impact_cost),
            opportunity_cost=float(opportunity_cost),
            gross_notional=float(gross_notional),
        )


def _schedule_weights(style: ExecutionStyle, market: pd.DataFrame) -> np.ndarray:
    count = len(market)
    if style is ExecutionStyle.OPEN:
        return np.array([1.0, *([0.0] * (count - 1))])
    if style is ExecutionStyle.CLOSE:
        return np.array([*([0.0] * (count - 1)), 1.0])
    if style is ExecutionStyle.VWAP:
        volume = market["volume"].clip(lower=0).to_numpy(dtype=float)
        return volume / volume.sum() if volume.sum() else np.repeat(1 / count, count)
    return np.repeat(1 / count, count)


def _round_lot(quantity: float, lot_size: int) -> int:
    return int(max(0, math.floor(quantity / lot_size) * lot_size))


def _validate_order_and_market(order: Order, market: pd.DataFrame) -> None:
    if order.quantity <= 0 or order.quantity % order.lot_size:
        raise ValueError("Order quantity must be a positive integer-lot quantity")
    if not 0 < order.maximum_participation <= 1:
        raise ValueError("maximum_participation must be in (0, 1]")
    missing = {"price", "volume", "can_trade"} - set(market.columns)
    if missing or market.empty:
        raise ValueError(f"Invalid market slices; missing={sorted(missing)}")
    if (market[["price", "volume"]] < 0).any().any():
        raise ValueError("Market slice price and volume cannot be negative")


def _fill_columns() -> list[str]:
    return [
        "order_id",
        "timestamp",
        "symbol",
        "side",
        "quantity",
        "market_price",
        "fill_price",
        "notional",
        "fees",
        "spread_cost",
        "impact_cost",
        "participation",
    ]
