from datetime import date

import pandas as pd
import pytest

from autoalpha.backtest.presets import (
    A_SHARE_NON_PIT_PROXY_WEEKLY_V1,
    A_SHARE_REALISTIC_WEEKLY_V1,
    MANUAL_BACKTEST_PRESETS,
)
from autoalpha.service.manual_backtest import (
    ManualBacktestSpec,
    _select_positions,
    _trade_statement_rows,
)


def test_advanced_manual_backtest_settings_validate_a_share_execution() -> None:
    spec = ManualBacktestSpec(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 12, 31),
        initial_cash_cny=1_000_000,
        gross_exposure=0.8,
        holding_period_days=5,
        selection_fraction=0.15,
        maximum_positions=20,
        lot_size=100,
        maximum_volume_participation=0.03,
        opening_limit_threshold=0.095,
        cost_stress_multiplier=3.0,
    )
    assert spec.maximum_positions == 20
    assert spec.cost_stress_multiplier == 3.0

    with pytest.raises(ValueError, match="selection_fraction"):
        ManualBacktestSpec(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
            initial_cash_cny=1_000_000,
            gross_exposure=0.8,
            holding_period_days=5,
            selection_fraction=0.8,
        )


def test_position_selection_respects_fraction_and_hard_position_cap() -> None:
    signal = pd.DataFrame(
        [range(10), range(10, 0, -1)],
        index=pd.date_range("2024-01-01", periods=2),
        columns=[f"S{index}" for index in range(10)],
    )
    long_only = _select_positions(
        signal, selection_fraction=0.50, maximum_positions=2, long_only=True
    )
    market_neutral = _select_positions(
        signal, selection_fraction=0.50, maximum_positions=2, long_only=False
    )

    assert (long_only.gt(0).sum(axis=1) == 2).all()
    assert (market_neutral.gt(0).sum(axis=1) == 2).all()
    assert (market_neutral.lt(0).sum(axis=1) == 2).all()


def test_trade_statement_serializes_fee_breakdown_and_security_name() -> None:
    trades = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-03"),
                "signal_date": pd.Timestamp("2024-01-02"),
                "sleeve": 0,
                "symbol": "000001.SZ",
                "side": "BUY",
                "quantity": 100,
                "price": 10.0,
                "notional": 1000.0,
                "commission": 5.0,
                "transfer_fee": 0.01,
                "stamp_duty": 0.0,
                "fees": 5.01,
                "net_cash_flow": -1005.01,
                "cash_after": 998994.99,
            }
        ]
    )
    market = pd.DataFrame({"ts_code": ["000001.SZ"], "name": ["平安银行"]})

    statement = _trade_statement_rows(trades, market)

    assert statement[0]["trade_id"] == 1
    assert statement[0]["security_name"] == "平安银行"
    assert statement[0]["commission_cny"] == 5.0
    assert statement[0]["net_cash_flow_cny"] == -1005.01


def test_realistic_a_share_preset_is_complete_and_tamper_evident() -> None:
    settings = MANUAL_BACKTEST_PRESETS[A_SHARE_REALISTIC_WEEKLY_V1]["settings"]
    spec = ManualBacktestSpec(
        start_date=date(2020, 1, 2),
        end_date=date(2024, 12, 31),
        initial_cash_cny=1_000_000,
        backtest_preset=A_SHARE_REALISTIC_WEEKLY_V1,
        **settings,
    )

    assert spec.backtest_engine == "EVENT_LEDGER"
    assert spec.rebalance_schedule == "WEEKLY_FIRST_SESSION"
    assert spec.use_historical_fee_schedule is True
    assert spec.slippage_bps_each_side == 5.0

    with pytest.raises(ValueError, match="was modified"):
        ManualBacktestSpec(
            start_date=date(2020, 1, 2),
            end_date=date(2024, 12, 31),
            initial_cash_cny=1_000_000,
            backtest_preset=A_SHARE_REALISTIC_WEEKLY_V1,
            **{**settings, "gross_exposure": 1.0},
        )


def test_non_pit_proxy_preset_is_explicitly_research_only() -> None:
    settings = MANUAL_BACKTEST_PRESETS[A_SHARE_NON_PIT_PROXY_WEEKLY_V1]["settings"]
    spec = ManualBacktestSpec(
        start_date=date(2020, 1, 2),
        end_date=date(2024, 12, 31),
        initial_cash_cny=1_000_000,
        backtest_preset=A_SHARE_NON_PIT_PROXY_WEEKLY_V1,
        **settings,
    )

    assert spec.backtest_engine == "EVENT_LEDGER"
    assert spec.execution_data_mode == "NON_PIT_PROXY"
