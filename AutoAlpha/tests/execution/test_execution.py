import pandas as pd

from autoalpha.execution.capacity import CapacityAnalyzer
from autoalpha.execution.simulator import (
    ExecutionSimulator,
    ExecutionStyle,
    MarketImpactModel,
    Order,
)
from autoalpha.execution.tca import transaction_cost_analysis


def _market() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "price": [10.0, 10.1, 10.2, 10.3],
            "volume": [1_000, 2_000, 2_000, 1_000],
            "can_trade": [True, True, False, True],
        },
        index=pd.date_range("2024-01-02 09:30", periods=4, freq="30min"),
    )


def test_execution_records_partial_fills_costs_and_unfilled() -> None:
    report = ExecutionSimulator(MarketImpactModel(half_spread_bps=2)).execute(
        Order("O1", "600000.SH", "BUY", 1_000, 10.0, ExecutionStyle.POV, 0.10),
        _market(),
        adv_shares=100_000,
        daily_volatility=0.02,
        alpha_decay_bps=5,
    )
    assert report.filled_quantity == 400
    assert report.unfilled_quantity == 600
    assert report.explicit_fees > 0
    assert report.spread_cost > 0
    assert report.impact_cost > 0
    assert report.opportunity_cost == 3.0
    assert (report.fills["participation"] <= 0.10).all()


def test_open_and_close_execution_choose_expected_slice() -> None:
    simulator = ExecutionSimulator()
    cases = [
        (ExecutionStyle.OPEN, _market().index[0]),
        (ExecutionStyle.CLOSE, _market().index[-1]),
    ]
    for style, expected in cases:
        report = simulator.execute(
            Order(f"O-{style}", "A", "SELL", 100, 10.0, style, 1.0),
            _market(),
            adv_shares=100_000,
            daily_volatility=0.01,
        )
        assert report.fills.iloc[0]["timestamp"] == expected


def test_capacity_declines_with_capital_and_returns_recommendation() -> None:
    returns = pd.Series([0.001, -0.0004, 0.0012, -0.0002] * 80)
    report = CapacityAnalyzer().analyze(
        returns,
        annual_turnover=10,
        aggregate_adv_cny=100_000_000,
        daily_volatility=0.02,
        capital_grid_cny=(10_000_000, 100_000_000, 1_000_000_000),
        minimum_net_ir=0.0,
    )
    assert report.points[0].net_ir > report.points[-1].net_ir
    assert report.recommended_capacity_cny > 0


def test_tca_reconciles_fill_records() -> None:
    report = ExecutionSimulator().execute(
        Order("O2", "A", "BUY", 200, 10.0, ExecutionStyle.TWAP, 1.0),
        _market(),
        adv_shares=100_000,
        daily_volatility=0.01,
    )
    tca = transaction_cost_analysis(report.fills, decision_price=10.0, close_price=10.3, side="BUY")
    assert tca["filled_quantity"] == 200
    assert tca["implementation_shortfall_bps"] > 0
    assert tca["fees"] == report.explicit_fees
