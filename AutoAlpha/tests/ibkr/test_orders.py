from __future__ import annotations

import pytest

from autoalpha.ibkr.client import OrderTransmissionBlocked
from autoalpha.ibkr.contracts import USEquity
from autoalpha.ibkr.orders import (
    OrderPlanError,
    PlannedOrder,
    build_ib_order,
    plan_orders,
    preview_plan,
    submit_plan,
)
from autoalpha.ibkr.settings import GatewaySettings


class RecordingGateway:
    """Captures preview/transmit calls without touching a broker."""

    def __init__(self, *, readonly: bool = True) -> None:
        self.settings = GatewaySettings(readonly=readonly)
        self.account = "DU000000"
        self.previewed: list[object] = []
        self.transmitted: list[object] = []

    def preview_order(self, equity: USEquity, order: object) -> dict[str, object]:
        self.previewed.append(order)
        return {"commission": 1.0, "initial_margin_after": 100.0, "warning": ""}

    def transmit_order(self, equity: USEquity, order: object) -> str:
        self.transmitted.append(order)
        return f"trade:{equity.symbol}"


@pytest.fixture
def contracts() -> dict[str, USEquity]:
    return {
        "AAPL": USEquity(symbol="AAPL", con_id=265598, primary_exchange="NASDAQ"),
        "MSFT": USEquity(symbol="MSFT", con_id=272093, primary_exchange="NASDAQ"),
    }


def test_plan_orders_diffs_target_against_current() -> None:
    plan = plan_orders({"AAPL": 100, "MSFT": 50}, {"AAPL": 30})
    assert [(o.symbol, o.action, o.quantity) for o in plan] == [
        ("AAPL", "BUY", 70),
        ("MSFT", "BUY", 50),
    ]


def test_plan_orders_emits_sells_for_reduced_and_exited_names() -> None:
    plan = plan_orders({"AAPL": 10}, {"AAPL": 40, "MSFT": 25})
    assert [(o.symbol, o.action, o.quantity) for o in plan] == [
        ("AAPL", "SELL", 30),
        ("MSFT", "SELL", 25),
    ]


def test_plan_orders_uses_single_share_granularity() -> None:
    """US equities have no round-lot constraint, unlike the A-share original."""
    plan = plan_orders({"AAPL": 7}, {})
    assert plan[0].quantity == 7


def test_plan_orders_suppresses_sub_threshold_noise() -> None:
    assert plan_orders({"AAPL": 100.4}, {"AAPL": 100}) == []
    plan = plan_orders({"AAPL": 105}, {"AAPL": 100}, minimum_shares=10)
    assert plan == []


def test_plan_orders_skips_unchanged_positions() -> None:
    assert plan_orders({"AAPL": 100}, {"AAPL": 100}) == []


def test_plan_orders_normalizes_share_class_symbols() -> None:
    plan = plan_orders({"BRK-B": 5}, {})
    assert plan[0].symbol == "BRK.B"


def test_plan_orders_rejects_bad_minimum() -> None:
    with pytest.raises(OrderPlanError, match="minimum_shares"):
        plan_orders({"AAPL": 5}, {}, minimum_shares=0)


def test_planned_order_requires_positive_quantity() -> None:
    with pytest.raises(OrderPlanError, match="positive"):
        PlannedOrder(symbol="AAPL", action="BUY", quantity=0)


def test_limit_order_requires_a_price() -> None:
    with pytest.raises(OrderPlanError, match="limit price"):
        PlannedOrder(symbol="AAPL", action="BUY", quantity=5, order_type="LMT")


def test_notional_uses_reference_price() -> None:
    order = PlannedOrder(symbol="AAPL", action="BUY", quantity=10, reference_price=200.0)
    assert order.notional == pytest.approx(2_000.0)


def test_market_on_open_maps_to_opg_time_in_force() -> None:
    order = build_ib_order(
        PlannedOrder(symbol="AAPL", action="BUY", quantity=5),
        account="DU1",
        transmit=False,
        order_reference="quantfactory-20260903-01",
    )
    assert order.orderType == "MKT"
    assert order.tif == "OPG"
    assert order.transmit is False
    assert order.account == "DU1"
    assert order.orderRef == "quantfactory-20260903-01"


def test_build_ib_order_defaults_to_not_transmitting() -> None:
    order = build_ib_order(PlannedOrder(symbol="AAPL", action="SELL", quantity=1), account="DU1")
    assert order.transmit is False


def test_preview_plan_never_transmits(contracts: dict[str, USEquity]) -> None:
    gateway = RecordingGateway()
    plan = plan_orders({"AAPL": 10}, {})
    previews = preview_plan(gateway, plan, contracts)
    assert len(previews) == 1
    assert previews[0]["commission"] == 1.0
    assert gateway.transmitted == []
    # preview_plan hands the gateway an un-flagged order; IBKRGateway.preview_order
    # is what copies it into a whatIf probe, so nothing here is transmittable.
    assert all(order.transmit is False for order in gateway.previewed)


def test_preview_plan_reports_missing_contracts(contracts: dict[str, USEquity]) -> None:
    gateway = RecordingGateway()
    plan = [PlannedOrder(symbol="TSLA", action="BUY", quantity=1)]
    previews = preview_plan(gateway, plan, contracts)
    assert "no resolved contract" in previews[0]["error"]


def test_submit_plan_requires_explicit_confirmation(contracts: dict[str, USEquity]) -> None:
    gateway = RecordingGateway(readonly=False)
    plan = plan_orders({"AAPL": 10}, {})
    with pytest.raises(OrderTransmissionBlocked, match="confirm=True"):
        submit_plan(gateway, plan, contracts)
    assert gateway.transmitted == []


def test_submit_plan_refuses_a_readonly_session(contracts: dict[str, USEquity]) -> None:
    gateway = RecordingGateway(readonly=True)
    plan = plan_orders({"AAPL": 10}, {})
    with pytest.raises(OrderTransmissionBlocked, match="read-only"):
        submit_plan(gateway, plan, contracts, confirm=True)
    assert gateway.transmitted == []


def test_submit_plan_transmits_when_both_gates_are_open(
    contracts: dict[str, USEquity]
) -> None:
    gateway = RecordingGateway(readonly=False)
    plan = plan_orders({"AAPL": 10}, {})
    trades = submit_plan(
        gateway,
        plan,
        contracts,
        confirm=True,
        order_reference_prefix="quantfactory-20260903",
    )
    assert trades == ["trade:AAPL"]
    assert all(order.transmit is True for order in gateway.transmitted)
    assert gateway.transmitted[0].orderRef == "quantfactory-20260903-01"


def test_submit_plan_requires_stable_order_reference(
    contracts: dict[str, USEquity],
) -> None:
    gateway = RecordingGateway(readonly=False)
    plan = plan_orders({"AAPL": 10}, {})

    with pytest.raises(OrderTransmissionBlocked, match="order-reference"):
        submit_plan(gateway, plan, contracts, confirm=True)

    assert gateway.transmitted == []


def test_submit_plan_validates_all_contracts_before_transmitting(
    contracts: dict[str, USEquity],
) -> None:
    gateway = RecordingGateway(readonly=False)
    plan = plan_orders({"AAPL": 10, "MSFT": 20}, {})

    with pytest.raises(OrderTransmissionBlocked, match="MSFT"):
        submit_plan(
            gateway,
            plan,
            {"AAPL": contracts["AAPL"]},
            confirm=True,
            order_reference_prefix="quantfactory-20260903",
        )

    assert gateway.transmitted == []
