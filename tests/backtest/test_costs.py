from __future__ import annotations

import pytest

from autoalpha.backtest.costs import ChinaAExecutionCosts


def test_sell_includes_stamp_duty_and_minimum_commission() -> None:
    costs = ChinaAExecutionCosts(
        commission_bps_each_side=1.5,
        stamp_duty_bps_sell=5.0,
        transfer_fee_bps_each_side=0.1,
        minimum_commission_cny=5.0,
    )

    assert costs.fees("BUY", 1_000.0) == pytest.approx(5.01)
    assert costs.fees("SELL", 1_000.0) == pytest.approx(5.51)
    assert costs.fees("SELL", 1_000_000.0) > costs.fees("BUY", 1_000_000.0)
    breakdown = costs.fee_breakdown("SELL", 1_000.0)
    assert breakdown == pytest.approx({"commission": 5.0, "transfer_fee": 0.01, "stamp_duty": 0.5})
    assert sum(breakdown.values()) == pytest.approx(costs.fees("SELL", 1_000.0))


def test_affordable_notional_reserves_fees() -> None:
    costs = ChinaAExecutionCosts()
    notional = costs.affordable_notional(10_000.0)

    assert notional + costs.fees("BUY", notional) <= 10_000.0 + 1e-8
    assert notional < 10_000.0


def test_historical_fee_schedule_uses_trade_date_cutovers() -> None:
    costs = ChinaAExecutionCosts(
        commission_bps_each_side=0,
        stamp_duty_bps_sell=5.0,
        transfer_fee_bps_each_side=0.1,
        minimum_commission_cny=0,
        use_historical_fee_schedule=True,
    )

    before_transfer_cut = costs.fee_breakdown("SELL", 1_000_000, "2022-04-28")
    before_stamp_cut = costs.fee_breakdown("SELL", 1_000_000, "2023-08-27")
    current = costs.fee_breakdown("SELL", 1_000_000, "2023-08-28")

    assert before_transfer_cut["transfer_fee"] == pytest.approx(20.0)
    assert before_transfer_cut["stamp_duty"] == pytest.approx(1_000.0)
    assert before_stamp_cut["transfer_fee"] == pytest.approx(10.0)
    assert before_stamp_cut["stamp_duty"] == pytest.approx(1_000.0)
    assert current["transfer_fee"] == pytest.approx(10.0)
    assert current["stamp_duty"] == pytest.approx(500.0)
