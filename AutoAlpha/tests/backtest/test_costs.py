from __future__ import annotations

from datetime import date

import pytest

from autoalpha.backtest.costs import USEquityExecutionCosts


def test_commission_is_charged_per_share() -> None:
    costs = USEquityExecutionCosts()
    breakdown = costs.fee_breakdown("BUY", notional=10_000.0, shares=1_000)
    assert breakdown["commission"] == pytest.approx(3.5)


def test_commission_respects_the_minimum() -> None:
    costs = USEquityExecutionCosts()
    breakdown = costs.fee_breakdown("BUY", notional=5_000.0, shares=10)
    assert breakdown["commission"] == pytest.approx(0.35)


def test_commission_is_capped_at_one_percent_of_notional() -> None:
    """A large share count in a penny stock is capped, not charged per share."""
    costs = USEquityExecutionCosts()
    breakdown = costs.fee_breakdown("BUY", notional=100.0, shares=10_000)
    assert breakdown["commission"] == pytest.approx(1.0)


def test_buys_pay_no_regulatory_fees() -> None:
    costs = USEquityExecutionCosts()
    breakdown = costs.fee_breakdown("BUY", notional=100_000.0, shares=1_000)
    assert breakdown["sec_fee"] == 0.0
    assert breakdown["finra_taf"] == 0.0


def test_sells_pay_sec_and_taf() -> None:
    costs = USEquityExecutionCosts()
    breakdown = costs.fee_breakdown("SELL", notional=1_000_000.0, shares=10_000)
    assert breakdown["sec_fee"] == pytest.approx(27.80)
    assert breakdown["finra_taf"] == pytest.approx(1.66)


def test_taf_is_capped_per_trade() -> None:
    costs = USEquityExecutionCosts()
    breakdown = costs.fee_breakdown("SELL", notional=1_000_000.0, shares=1_000_000)
    assert breakdown["finra_taf"] == pytest.approx(8.30)


def test_zero_quantity_is_free() -> None:
    costs = USEquityExecutionCosts()
    assert costs.fees("BUY", 0.0, 0) == 0.0
    assert costs.fees("SELL", 1_000.0, 0) == 0.0


def test_unknown_side_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown side"):
        USEquityExecutionCosts().fee_breakdown("HOLD", 100.0, 1)  # type: ignore[arg-type]


def test_negative_shares_are_rejected() -> None:
    with pytest.raises(ValueError, match="negative"):
        USEquityExecutionCosts().fee_breakdown("BUY", 100.0, -1)


def test_negative_parameters_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        USEquityExecutionCosts(commission_per_share=-1.0)


def test_commission_cap_fraction_is_validated() -> None:
    with pytest.raises(ValueError, match="maximum_commission_fraction"):
        USEquityExecutionCosts(maximum_commission_fraction=0.0)


def test_historical_sec_rate_applies_by_trade_date() -> None:
    costs = USEquityExecutionCosts(
        sec_fee_per_million_usd_sell=27.80,
        historical_sec_rates=((date(2020, 1, 1), 22.10), (date(2024, 5, 22), 27.80)),
    )
    early = costs.fee_breakdown("SELL", 1_000_000.0, 100, date(2021, 6, 1))
    late = costs.fee_breakdown("SELL", 1_000_000.0, 100, date(2025, 6, 1))
    assert early["sec_fee"] == pytest.approx(22.10)
    assert late["sec_fee"] == pytest.approx(27.80)


def test_historical_rate_falls_back_before_the_first_entry() -> None:
    costs = USEquityExecutionCosts(historical_sec_rates=((date(2024, 5, 22), 27.80),))
    breakdown = costs.fee_breakdown("SELL", 1_000_000.0, 100, date(2000, 1, 1))
    assert breakdown["sec_fee"] == pytest.approx(costs.sec_fee_per_million_usd_sell)


def test_affordable_shares_leaves_room_for_commission() -> None:
    costs = USEquityExecutionCosts()
    shares = costs.affordable_shares(cash=1_000.0, price=100.0)
    assert shares == 9
    notional = shares * 100.0
    assert notional + costs.fees("BUY", notional, shares) <= 1_000.0


def test_affordable_shares_uses_single_share_granularity() -> None:
    """US equities have no round-lot constraint."""
    assert USEquityExecutionCosts().affordable_shares(cash=350.0, price=100.0) == 3


def test_affordable_shares_handles_unaffordable_cash() -> None:
    costs = USEquityExecutionCosts()
    assert costs.affordable_shares(cash=0.10, price=100.0) == 0
    assert costs.affordable_shares(cash=1_000.0, price=0.0) == 0
