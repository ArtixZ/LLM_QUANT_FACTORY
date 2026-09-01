from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from autoalpha.ibkr.client import IBKRGateway, OrderTransmissionBlocked
from autoalpha.ibkr.contracts import USEquity, panel_symbol

logger = logging.getLogger(__name__)

Action = Literal["BUY", "SELL"]
OrderType = Literal["MKT", "LMT", "MOO"]

# The research protocol forms signals after the close and executes at the next
# open, which maps exactly onto a market-on-open order (MKT with an OPG
# time-in-force). MKT is the fallback for intraday manual submission.
MARKET_ON_OPEN_TIF = "OPG"


class OrderPlanError(ValueError):
    """A target book could not be turned into a valid order plan."""


@dataclass(frozen=True)
class PlannedOrder:
    """A single intended trade, broker-agnostic and inspectable before submission."""

    symbol: str
    action: Action
    quantity: int
    order_type: OrderType = "MOO"
    limit_price: float | None = None
    reference_price: float | None = None

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise OrderPlanError(f"{self.symbol}: order quantity must be positive")
        if self.action not in {"BUY", "SELL"}:
            raise OrderPlanError(f"{self.symbol}: unknown action {self.action!r}")
        if self.order_type == "LMT" and not self.limit_price:
            raise OrderPlanError(f"{self.symbol}: a limit order requires a limit price")

    @property
    def notional(self) -> float:
        price = self.limit_price or self.reference_price or 0.0
        return float(self.quantity) * float(price)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "action": self.action,
            "quantity": self.quantity,
            "order_type": self.order_type,
            "limit_price": self.limit_price,
            "reference_price": self.reference_price,
            "notional": round(self.notional, 2),
        }


def plan_orders(
    target_shares: Mapping[str, float],
    current_shares: Mapping[str, float],
    *,
    reference_prices: Mapping[str, float] | None = None,
    order_type: OrderType = "MOO",
    minimum_shares: int = 1,
) -> list[PlannedOrder]:
    """Diff a target book against current holdings into whole-share orders.

    US equities trade in single shares, so unlike the A-share original there is
    no round-lot quantisation; ``minimum_shares`` exists only to suppress
    one-share noise trades that cost more in commission than they correct.
    """
    if minimum_shares < 1:
        raise OrderPlanError("minimum_shares must be at least one")
    prices = dict(reference_prices or {})
    symbols = sorted({*target_shares, *current_shares})
    orders: list[PlannedOrder] = []
    for symbol in symbols:
        key = panel_symbol(symbol)
        target = float(target_shares.get(symbol, 0.0))
        current = float(current_shares.get(symbol, 0.0))
        delta = target - current
        quantity = int(abs(round(delta)))
        if quantity < minimum_shares:
            continue
        orders.append(
            PlannedOrder(
                symbol=key,
                action="BUY" if delta > 0 else "SELL",
                quantity=quantity,
                order_type=order_type,
                reference_price=prices.get(symbol) or prices.get(key),
            )
        )
    return orders


def build_ib_order(planned: PlannedOrder, *, account: str, transmit: bool = False) -> Any:
    """Construct an ``ib_async`` order object. ``transmit`` stays False by default."""
    from ib_async import LimitOrder, MarketOrder

    if planned.order_type == "LMT":
        order = LimitOrder(planned.action, planned.quantity, float(planned.limit_price or 0.0))
    else:
        order = MarketOrder(planned.action, planned.quantity)
        if planned.order_type == "MOO":
            order.tif = MARKET_ON_OPEN_TIF
    order.account = account
    order.transmit = transmit
    return order


def preview_plan(
    gateway: IBKRGateway,
    plan: list[PlannedOrder],
    contracts: Mapping[str, USEquity],
) -> list[dict[str, Any]]:
    """Run every planned order through IBKR's whatIf margin check.

    Nothing is transmitted: each order is copied with ``whatIf=True`` and
    ``transmit=False`` before it reaches the gateway.
    """
    previews: list[dict[str, Any]] = []
    for planned in plan:
        equity = contracts.get(planned.symbol)
        if equity is None:
            previews.append(
                {**planned.to_dict(), "error": f"no resolved contract for {planned.symbol}"}
            )
            continue
        order = build_ib_order(planned, account=gateway.account, transmit=False)
        try:
            preview = gateway.preview_order(equity, order)
        except Exception as error:  # noqa: BLE001 - surfaced per order, never fatal
            previews.append({**planned.to_dict(), "error": str(error)})
            continue
        # The plan owns the order's identity; the broker reply contributes only
        # its margin and commission estimate.
        previews.append({**preview, **planned.to_dict()})
    return previews


def submit_plan(
    gateway: IBKRGateway,
    plan: list[PlannedOrder],
    contracts: Mapping[str, USEquity],
    *,
    confirm: bool = False,
) -> list[Any]:
    """Transmit a reviewed plan. Requires a writable session and explicit confirmation.

    Two independent gates stand in front of live order flow: the session must
    have been built with ``GatewaySettings.writable()``, and the caller must pass
    ``confirm=True``. Neither defaults to permissive.
    """
    if not confirm:
        raise OrderTransmissionBlocked(
            "submit_plan requires confirm=True; review preview_plan output first"
        )
    if gateway.settings.readonly:
        raise OrderTransmissionBlocked(
            "Gateway session is read-only; rebuild it with GatewaySettings.writable()"
        )
    trades: list[Any] = []
    for planned in plan:
        equity = contracts[planned.symbol]
        order = build_ib_order(planned, account=gateway.account, transmit=True)
        logger.info(
            "transmitting %s %s x%s (%s)",
            planned.action,
            planned.symbol,
            planned.quantity,
            planned.order_type,
        )
        trades.append(gateway.transmit_order(equity, order))
    return trades
