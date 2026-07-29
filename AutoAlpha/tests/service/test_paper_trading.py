from __future__ import annotations

from datetime import date

import pandas as pd

from autoalpha.service.paper_trading import (
    PAPER_EXECUTION_PROTOCOL,
    PaperTradingEngine,
    _open_trade_permissions,
    _trade,
)
from autoalpha.service.store import ServiceStore


def test_next_open_market_state_uses_open_proxy_permissions(tmp_path) -> None:
    data_root = tmp_path / "panel"
    partition = data_root / "trade_year=2026"
    partition.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "trade_date": "2026-07-20",
                "ts_code": "000001.SZ",
                "raw_open": 10.0,
                "raw_pre_close": 9.9,
                "is_valid_ohlc": True,
                "is_tradable_observation": True,
                "can_buy_open_proxy": False,
                "can_sell_open_proxy": True,
            },
            {
                "trade_date": "2026-07-20",
                "ts_code": "000002.SZ",
                "raw_open": 20.0,
                "raw_pre_close": 20.1,
                "is_valid_ohlc": True,
                "is_tradable_observation": True,
                "can_buy_open_proxy": True,
                "can_sell_open_proxy": False,
            },
        ]
    ).to_parquet(partition / "data.parquet", index=False)
    engine = PaperTradingEngine(ServiceStore(tmp_path / "service.sqlite3"), data_root)

    state, trade_date = engine._next_open_market_state(["000001.SZ", "000002.SZ"], "2026-07-17")

    assert trade_date == "2026-07-20"
    assert state["000001.SZ"] == {
        "raw_open": 10.0,
        "can_buy_open": False,
        "can_sell_open": True,
    }
    assert state["000002.SZ"] == {
        "raw_open": 20.0,
        "can_buy_open": True,
        "can_sell_open": False,
    }


def test_open_trade_permissions_falls_back_to_open_limit_threshold() -> None:
    limit_up = pd.Series(
        {
            "raw_open": 11.0,
            "raw_pre_close": 10.0,
            "is_valid_ohlc": True,
            "is_tradable_observation": True,
        }
    )
    limit_down = pd.Series(
        {
            "raw_open": 9.0,
            "raw_pre_close": 10.0,
            "is_valid_ohlc": True,
            "is_tradable_observation": True,
        }
    )
    halted = pd.Series(
        {
            "raw_open": 10.0,
            "raw_pre_close": 10.0,
            "is_valid_ohlc": True,
            "is_tradable_observation": False,
        }
    )

    assert _open_trade_permissions(limit_up) == (False, True)
    assert _open_trade_permissions(limit_down) == (True, False)
    assert _open_trade_permissions(halted) == (False, False)


def test_paper_trade_execution_details_are_persisted(tmp_path) -> None:
    store = ServiceStore(tmp_path / "service.sqlite3")
    portfolio = store.create_paper_portfolio(
        name="execution-ledger",
        config={
            "execution_assumption": "NEXT_SESSION_RAW_OPEN_PROXY_FILL_V1",
            "signal_time": "END_OF_DAY_AFTER_CLOSE",
            "execution_time": "NEXT_SESSION_OPEN",
        },
        initial_cash_cny=1_000_000,
    )
    store.apply_paper_portfolio_update(
        portfolio_id=portfolio["id"],
        cash_cny=900_000,
        positions=[
            {
                "symbol": "000001.SZ",
                "security_name": "平安银行",
                "quantity": 10_000,
                "average_cost_cny": 10.0,
                "acquired_trade_date": "2026-07-20",
                "last_trade_date": "2026-07-20",
            }
        ],
        trades=[
            {
                "trade_date": "2026-07-20",
                "symbol": "000001.SZ",
                "security_name": "平安银行",
                "side": "BUY",
                "quantity": 10_000,
                "price_cny": 10.005,
                "notional_cny": 100_050,
                "fees_cny": 25.0,
                "reason": "INITIAL_ALLOCATION",
                "execution": {
                    "signal_date": "2026-07-17",
                    "execution_time": "NEXT_SESSION_OPEN",
                    "execution_assumption": "NEXT_SESSION_RAW_OPEN_PROXY_FILL_V1",
                    "reference_price_cny": 10.0,
                    "price_basis": "RAW_OPEN",
                    "slippage_bps_each_side": 5.0,
                },
            }
        ],
        nav={
            "trade_date": "2026-07-20",
            "nav_cny": 1_000_000,
            "market_value_cny": 100_000,
            "gross_exposure": 0.1,
        },
        rebalanced=True,
    )

    detail = store.paper_portfolio(portfolio["id"])
    assert detail is not None
    trade = detail["trades"][0]

    assert trade["execution"]["signal_date"] == "2026-07-17"
    assert trade["execution"]["execution_time"] == "NEXT_SESSION_OPEN"
    assert trade["execution"]["price_basis"] == "RAW_OPEN"
    assert detail["positions"][0]["acquired_trade_date"] == "2026-07-20"
    assert detail["positions"][0]["last_trade_date"] == "2026-07-20"


def test_paper_execution_protocol_documents_next_open_proxy_fill() -> None:
    assert PAPER_EXECUTION_PROTOCOL["protocol"] == "A_SHARE_PAPER_NEXT_OPEN_PROXY_EXECUTION_V2"
    assert PAPER_EXECUTION_PROTOCOL["signal_time"] == "END_OF_DAY_AFTER_CLOSE"
    assert PAPER_EXECUTION_PROTOCOL["execution_time"] == "NEXT_SESSION_OPEN"
    assert PAPER_EXECUTION_PROTOCOL["execution_lag_sessions"] == 1
    assert PAPER_EXECUTION_PROTOCOL["price_basis"] == "RAW_OPEN"
    assert PAPER_EXECUTION_PROTOCOL["mark_to_market_price_basis"] == "RAW_CLOSE"
    assert PAPER_EXECUTION_PROTOCOL["t_plus_one_sell_lock"] is True
    assert PAPER_EXECUTION_PROTOCOL["blocked_order_policy"] == "SKIP_ORDER_AND_AUDIT"
    assert PAPER_EXECUTION_PROTOCOL["production_caveat"] == "NON_PIT_PROXY_RESEARCH_AND_PAPER_ONLY"


def test_paper_rebalance_blocks_same_day_t_plus_one_sell(tmp_path, monkeypatch) -> None:
    store = ServiceStore(tmp_path / "service.sqlite3")
    portfolio = store.create_paper_portfolio(
        name="t-plus-one",
        config={
            "factor_ids": ["F_keep"],
            "weights": [1.0],
            "selection_count": 1,
            "gross_exposure": 0.9,
            "slippage_bps_each_side": 0.0,
            "lot_size": 100,
            "execution_assumption": "NEXT_SESSION_RAW_OPEN_PROXY_FILL_V1",
            "signal_time": "END_OF_DAY_AFTER_CLOSE",
            "execution_time": "NEXT_SESSION_OPEN",
        },
        initial_cash_cny=1_000_000,
    )
    store.apply_paper_portfolio_update(
        portfolio_id=portfolio["id"],
        cash_cny=900_000,
        positions=[
            {
                "symbol": "000001.SZ",
                "security_name": "平安银行",
                "quantity": 10_000,
                "average_cost_cny": 10.0,
                "acquired_trade_date": "2026-07-20",
                "last_trade_date": "2026-07-20",
            }
        ],
        trades=[
            _trade(
                "2026-07-20",
                "000001.SZ",
                "平安银行",
                "BUY",
                10_000,
                10.0,
                25.0,
                "INITIAL_ALLOCATION",
            )
        ],
        nav={
            "trade_date": "2026-07-20",
            "nav_cny": 1_000_000,
            "market_value_cny": 100_000,
            "gross_exposure": 0.1,
        },
        rebalanced=True,
    )
    engine = PaperTradingEngine(store, tmp_path / "panel")
    monkeypatch.setattr(engine, "_factor_records", lambda factor_ids: [{"factor_id": "F_keep"}])
    monkeypatch.setattr(
        "autoalpha.service.paper_trading.factor_from_pool_record",
        lambda record: record,
    )

    class FakeScreener:
        def __init__(self, data_path) -> None:
            self.data_path = data_path

        def screen(self, factors, weights, spec):  # noqa: ANN001, ANN201
            return {
                "as_of_date": "2026-07-17",
                "rows": [
                    {
                        "ts_code": "000002.SZ",
                        "name": "万科A",
                        "rank": 1,
                    }
                ],
            }

    monkeypatch.setattr("autoalpha.service.paper_trading.CrossSectionalScreener", FakeScreener)
    monkeypatch.setattr(
        engine,
        "_next_open_market_state",
        lambda symbols, signal_date: (
            {
                "000001.SZ": {
                    "raw_open": 10.0,
                    "can_buy_open": True,
                    "can_sell_open": True,
                },
                "000002.SZ": {
                    "raw_open": 20.0,
                    "can_buy_open": True,
                    "can_sell_open": True,
                },
            },
            "2026-07-20",
        ),
    )

    result = engine.rebalance(portfolio["id"], date.fromisoformat("2026-07-17"))

    assert any(position["symbol"] == "000001.SZ" for position in result["positions"])
    assert all(trade["side"] != "SELL" for trade in result["trades"])
    events = store.events(limit=5)
    assert events[0]["event"] == "PAPER_TRADING_OPEN_CONSTRAINTS_APPLIED"
    assert events[0]["payload"]["blocked_orders"][0]["reason"] == "T_PLUS_ONE_LOCKED"


def test_paper_rebalance_trade_carries_full_execution_protocol(tmp_path, monkeypatch) -> None:
    store = ServiceStore(tmp_path / "service.sqlite3")
    portfolio = store.create_paper_portfolio(
        name="protocol-ledger",
        config={
            "factor_ids": ["F_keep"],
            "weights": [1.0],
            "selection_count": 1,
            "gross_exposure": 0.9,
            "slippage_bps_each_side": 5.0,
            "lot_size": PAPER_EXECUTION_PROTOCOL["lot_size"],
            "execution_protocol": PAPER_EXECUTION_PROTOCOL,
            "execution_assumption": PAPER_EXECUTION_PROTOCOL["execution_assumption"],
            "signal_time": PAPER_EXECUTION_PROTOCOL["signal_time"],
            "execution_time": PAPER_EXECUTION_PROTOCOL["execution_time"],
            "execution_lag_sessions": PAPER_EXECUTION_PROTOCOL["execution_lag_sessions"],
        },
        initial_cash_cny=1_000_000,
    )
    engine = PaperTradingEngine(store, tmp_path / "panel")
    monkeypatch.setattr(engine, "_factor_records", lambda factor_ids: [{"factor_id": "F_keep"}])
    monkeypatch.setattr(
        "autoalpha.service.paper_trading.factor_from_pool_record",
        lambda record: record,
    )

    class FakeScreener:
        def __init__(self, data_path) -> None:
            self.data_path = data_path

        def screen(self, factors, weights, spec):  # noqa: ANN001, ANN201
            return {
                "as_of_date": "2026-07-17",
                "rows": [{"ts_code": "000002.SZ", "name": "万科A", "rank": 1}],
            }

    monkeypatch.setattr("autoalpha.service.paper_trading.CrossSectionalScreener", FakeScreener)
    monkeypatch.setattr(
        engine,
        "_next_open_market_state",
        lambda symbols, signal_date: (
            {
                "000002.SZ": {
                    "raw_open": 20.0,
                    "can_buy_open": True,
                    "can_sell_open": True,
                },
            },
            "2026-07-20",
        ),
    )

    result = engine.rebalance(portfolio["id"], date.fromisoformat("2026-07-17"))
    trade = result["trades"][0]

    assert result["config"]["execution_protocol"]["protocol"] == (
        "A_SHARE_PAPER_NEXT_OPEN_PROXY_EXECUTION_V2"
    )
    assert trade["execution"]["execution_protocol"]["price_basis"] == "RAW_OPEN"
    assert trade["execution"]["execution_protocol"]["mark_to_market_price_basis"] == "RAW_CLOSE"
    assert trade["execution"]["execution_lag_sessions"] == 1
    assert trade["execution"]["t_plus_one_sell_lock"] is True
