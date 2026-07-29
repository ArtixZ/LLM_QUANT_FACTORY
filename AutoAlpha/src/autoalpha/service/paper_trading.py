from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from autoalpha.backtest.costs import ChinaAExecutionCosts
from autoalpha.service.multifactor import factor_from_pool_record
from autoalpha.service.screener import CrossSectionalScreener, ScreenerSpec
from autoalpha.service.store import ServiceStore

PAPER_EXECUTION_PROTOCOL = {
    "protocol": "A_SHARE_PAPER_NEXT_OPEN_PROXY_EXECUTION_V2",
    "execution_assumption": "NEXT_SESSION_RAW_OPEN_PROXY_FILL_V1",
    "signal_time": "END_OF_DAY_AFTER_CLOSE",
    "execution_time": "NEXT_SESSION_OPEN",
    "execution_lag_sessions": 1,
    "price_basis": "RAW_OPEN",
    "mark_to_market_price_basis": "RAW_CLOSE",
    "portfolio_mode": "LONG_ONLY_CASH",
    "lot_size": 100,
    "t_plus_one_sell_lock": True,
    "blocked_order_policy": "SKIP_ORDER_AND_AUDIT",
    "tradability_fields": [
        "raw_open",
        "raw_pre_close",
        "is_valid_ohlc",
        "is_tradable_observation",
        "can_buy_open_proxy",
        "can_sell_open_proxy",
    ],
    "fee_model": "CHINA_A_HISTORICAL_FEE_SCHEDULE_WITH_CONFIGURED_SLIPPAGE",
    "production_caveat": "NON_PIT_PROXY_RESEARCH_AND_PAPER_ONLY",
}


@dataclass(frozen=True)
class PaperStrategySpec:
    name: str
    factor_ids: list[str]
    weights: list[float]
    initial_cash_cny: float
    selection_count: int
    gross_exposure: float
    slippage_bps_each_side: float
    as_of_date: date
    market: str = "CN_A"
    data_path: str = ""
    source_task_ids: tuple[str, ...] = ()


class PaperTradingEngine:
    """Persistent A-share paper portfolios using next-session open proxy fills."""

    def __init__(self, store: ServiceStore, data_path: Path) -> None:
        self.store = store
        self.data_path = data_path

    def create(self, spec: PaperStrategySpec) -> dict[str, Any]:
        if not spec.name.strip():
            raise ValueError("Strategy name is required")
        records = self._factor_records(spec.factor_ids)
        config = {
            "factor_ids": spec.factor_ids,
            "weights": spec.weights,
            "selection_count": spec.selection_count,
            "gross_exposure": spec.gross_exposure,
            "slippage_bps_each_side": spec.slippage_bps_each_side,
            "lot_size": PAPER_EXECUTION_PROTOCOL["lot_size"],
            "execution_protocol": PAPER_EXECUTION_PROTOCOL,
            "execution_assumption": PAPER_EXECUTION_PROTOCOL["execution_assumption"],
            "signal_time": PAPER_EXECUTION_PROTOCOL["signal_time"],
            "execution_time": PAPER_EXECUTION_PROTOCOL["execution_time"],
            "execution_lag_sessions": PAPER_EXECUTION_PROTOCOL["execution_lag_sessions"],
            "rebalance_schedule": "MANUAL_OR_STRATEGY_WEEKLY_FIRST_SESSION",
            "market": spec.market,
            "data_path": spec.data_path or str(self.data_path),
            "source_task_ids": list(spec.source_task_ids),
        }
        portfolio = self.store.create_paper_portfolio(
            name=spec.name, config=config, initial_cash_cny=spec.initial_cash_cny
        )
        try:
            self._rebalance(portfolio, records, spec.as_of_date, reason="INITIAL_ALLOCATION")
        except Exception:
            self.store.update_paper_portfolio_status(int(portfolio["id"]), "CLOSED")
            raise
        result = self.store.paper_portfolio(int(portfolio["id"]))
        assert result is not None
        return result

    def rebalance(self, portfolio_id: int, as_of_date: date) -> dict[str, Any]:
        portfolio = self._require_portfolio(portfolio_id)
        if portfolio["status"] != "ACTIVE":
            raise ValueError("Only ACTIVE paper portfolios can be rebalanced")
        records = self._factor_records(list(portfolio["config"]["factor_ids"]))
        self._rebalance(portfolio, records, as_of_date, reason="MANUAL_REBALANCE")
        result = self.store.paper_portfolio(portfolio_id)
        assert result is not None
        return result

    def mark_all(self) -> list[dict[str, Any]]:
        results = []
        for portfolio in self.store.paper_portfolios(limit=500):
            if portfolio["status"] != "ACTIVE":
                continue
            configured_path = Path(portfolio["config"].get("data_path") or self.data_path)
            engine = (
                self
                if configured_path == self.data_path
                else type(self)(self.store, configured_path)
            )
            results.append(engine.mark(int(portfolio["id"])))
        return results

    def mark(self, portfolio_id: int) -> dict[str, Any]:
        portfolio = self._require_portfolio(portfolio_id)
        positions = portfolio.get("positions", [])
        if not positions:
            return portfolio
        prices, trade_date = self._prices([item["symbol"] for item in positions], None)
        cash = float(portfolio["cash_cny"])
        market_value = sum(
            int(position["quantity"]) * float(prices.get(position["symbol"], 0.0))
            for position in positions
        )
        nav = cash + market_value
        self.store.apply_paper_portfolio_update(
            portfolio_id=portfolio_id,
            cash_cny=cash,
            positions=[
                {
                    "symbol": item["symbol"],
                    "security_name": item["security_name"],
                    "quantity": item["quantity"],
                    "average_cost_cny": item["average_cost_cny"],
                    "acquired_trade_date": item.get("acquired_trade_date"),
                    "last_trade_date": item.get("last_trade_date"),
                }
                for item in positions
            ],
            trades=[],
            nav={
                "trade_date": trade_date,
                "nav_cny": nav,
                "market_value_cny": market_value,
                "gross_exposure": market_value / nav if nav else 0.0,
            },
            rebalanced=False,
        )
        result = self.store.paper_portfolio(portfolio_id)
        assert result is not None
        return result

    def _rebalance(
        self,
        portfolio: dict[str, Any],
        records: list[dict[str, Any]],
        as_of_date: date,
        *,
        reason: str,
    ) -> None:
        config = portfolio["config"]
        screen = CrossSectionalScreener(self.data_path).screen(
            [factor_from_pool_record(record) for record in records],
            list(config["weights"]),
            ScreenerSpec(as_of_date=as_of_date, selection_count=int(config["selection_count"])),
        )
        targets = {row["ts_code"]: row for row in screen["rows"]}
        if not targets:
            raise ValueError("No target securities selected by the EOD screener")
        signal_date = str(screen["as_of_date"])
        previous_positions = {item["symbol"]: dict(item) for item in portfolio.get("positions", [])}
        market_state, trade_date = self._next_open_market_state(
            [*previous_positions, *targets], signal_date
        )
        prices = {
            symbol: state["raw_open"]
            for symbol, state in market_state.items()
            if state.get("raw_open", 0.0) > 0
        }
        buyable_targets = {
            symbol: row
            for symbol, row in targets.items()
            if prices.get(symbol) and market_state.get(symbol, {}).get("can_buy_open")
        }
        if not buyable_targets and not previous_positions:
            raise ValueError("No target securities have usable next-session raw open prices")
        cash = float(portfolio["cash_cny"])
        market_value = sum(
            int(item["quantity"]) * float(prices.get(symbol, 0.0))
            for symbol, item in previous_positions.items()
        )
        nav_before = cash + market_value
        lot_size = int(config.get("lot_size", 100))
        allocation_count = max(1, len(buyable_targets) or len(targets))
        target_value = nav_before * float(config["gross_exposure"]) / allocation_count
        desired = {
            symbol: _round_lot(target_value / price, lot_size)
            for symbol, price in prices.items()
            if symbol in targets and price > 0
        }
        costs = ChinaAExecutionCosts(
            commission_bps_each_side=2.5,
            stamp_duty_bps_sell=5.0,
            transfer_fee_bps_each_side=0.1,
            minimum_commission_cny=5.0,
            use_historical_fee_schedule=True,
        )
        slippage = float(config["slippage_bps_each_side"])
        positions = {symbol: dict(item) for symbol, item in previous_positions.items()}
        trades = []
        blocked_orders = []
        for symbol in sorted(set(positions) | set(desired)):
            current = int(positions.get(symbol, {}).get("quantity", 0))
            target = desired.get(symbol, 0)
            if current <= target:
                continue
            quantity = current - target
            reference = prices.get(symbol)
            if not reference:
                continue
            if not market_state.get(symbol, {}).get("can_sell_open", False):
                blocked_orders.append(
                    _blocked_order(trade_date, symbol, "SELL", quantity, "OPEN_SELL_BLOCKED")
                )
                continue
            acquired_trade_date = positions[symbol].get("acquired_trade_date")
            last_trade_date = positions[symbol].get("last_trade_date")
            if (
                acquired_trade_date
                and str(acquired_trade_date) >= str(trade_date)
                or last_trade_date
                and str(last_trade_date) >= str(trade_date)
            ):
                blocked_orders.append(
                    _blocked_order(trade_date, symbol, "SELL", quantity, "T_PLUS_ONE_LOCKED")
                )
                continue
            price = reference * (1.0 - slippage / 10_000.0)
            notional = quantity * price
            fees = costs.fees("SELL", notional, trade_date)
            cash += notional - fees
            positions[symbol]["quantity"] = target
            positions[symbol]["last_trade_date"] = trade_date
            trades.append(
                _trade(
                    trade_date,
                    symbol,
                    positions[symbol]["security_name"],
                    "SELL",
                    quantity,
                    price,
                    fees,
                    reason,
                    execution={
                        "signal_date": signal_date,
                        "execution_time": config.get("execution_time"),
                        "execution_assumption": config.get("execution_assumption"),
                        "execution_protocol": config.get("execution_protocol"),
                        "execution_lag_sessions": config.get("execution_lag_sessions"),
                        "reference_price_cny": reference,
                        "price_basis": "RAW_OPEN",
                        "slippage_bps_each_side": slippage,
                        "t_plus_one_sell_lock": True,
                    },
                )
            )
        for symbol in sorted(desired, key=lambda value: targets[value]["rank"]):
            current = int(positions.get(symbol, {}).get("quantity", 0))
            target = desired[symbol]
            if target <= current:
                continue
            if not market_state.get(symbol, {}).get("can_buy_open", False):
                blocked_orders.append(
                    _blocked_order(
                        trade_date,
                        symbol,
                        "BUY",
                        target - current,
                        "OPEN_BUY_BLOCKED",
                    )
                )
                continue
            reference = prices[symbol]
            price = reference * (1.0 + slippage / 10_000.0)
            affordable = _round_lot(costs.affordable_notional(cash, trade_date) / price, lot_size)
            quantity = min(target - current, affordable)
            if quantity <= 0:
                continue
            notional = quantity * price
            fees = costs.fees("BUY", notional, trade_date)
            cash -= notional + fees
            old_quantity = current
            old_cost = float(positions.get(symbol, {}).get("average_cost_cny", 0.0))
            average_cost = (old_quantity * old_cost + quantity * price) / (old_quantity + quantity)
            positions[symbol] = {
                "symbol": symbol,
                "security_name": targets[symbol]["name"],
                "quantity": old_quantity + quantity,
                "average_cost_cny": average_cost,
                "acquired_trade_date": positions.get(symbol, {}).get("acquired_trade_date")
                or trade_date,
                "last_trade_date": trade_date,
            }
            trades.append(
                _trade(
                    trade_date,
                    symbol,
                    targets[symbol]["name"],
                    "BUY",
                    quantity,
                    price,
                    fees,
                    reason,
                    execution={
                        "signal_date": signal_date,
                        "execution_time": config.get("execution_time"),
                        "execution_assumption": config.get("execution_assumption"),
                        "execution_protocol": config.get("execution_protocol"),
                        "execution_lag_sessions": config.get("execution_lag_sessions"),
                        "reference_price_cny": reference,
                        "price_basis": "RAW_OPEN",
                        "slippage_bps_each_side": slippage,
                        "t_plus_one_sell_lock": True,
                    },
                )
            )
        positions = {key: value for key, value in positions.items() if int(value["quantity"]) > 0}
        market_value = sum(
            int(item["quantity"]) * prices[symbol] for symbol, item in positions.items()
        )
        nav = cash + market_value
        self.store.apply_paper_portfolio_update(
            portfolio_id=int(portfolio["id"]),
            cash_cny=cash,
            positions=list(positions.values()),
            trades=trades,
            nav={
                "trade_date": trade_date,
                "nav_cny": nav,
                "market_value_cny": market_value,
                "gross_exposure": market_value / nav if nav else 0.0,
            },
            rebalanced=True,
        )
        if blocked_orders:
            self.store.append_event(
                "audit",
                "PAPER_TRADING_OPEN_CONSTRAINTS_APPLIED",
                "模拟交易开盘约束已应用",
                f"{len(blocked_orders)} 笔目标订单因涨跌停、停牌或代理约束未成交。",
                payload={
                    "portfolio_id": portfolio["id"],
                    "signal_date": signal_date,
                    "trade_date": trade_date,
                    "blocked_orders": blocked_orders,
                    "execution_assumption": config.get("execution_assumption"),
                    "execution_time": config.get("execution_time"),
                    "execution_protocol": config.get("execution_protocol"),
                },
                level="WARN",
            )

    def _factor_records(self, factor_ids: list[str]) -> list[dict[str, Any]]:
        records = []
        for factor_id in factor_ids:
            record = self.store.factor_pool_record(factor_id)
            if record is None:
                raise KeyError(f"Factor not found: {factor_id}")
            records.append(record)
        return records

    def _require_portfolio(self, portfolio_id: int) -> dict[str, Any]:
        portfolio = self.store.paper_portfolio(portfolio_id)
        if portfolio is None:
            raise KeyError(f"Paper portfolio not found: {portfolio_id}")
        return portfolio

    def _prices(self, symbols: list[str], as_of_date: str | None) -> tuple[dict[str, float], str]:
        screener = CrossSectionalScreener(self.data_path)
        requested = (
            date.fromisoformat(as_of_date)
            if as_of_date
            else date.fromisoformat(screener.workspace.last_trade_date)
        )
        _, snapshot, resolved = screener._load_snapshot(requested)
        selected = snapshot.reindex(symbols)
        prices = {
            str(symbol): float(value)
            for symbol, value in selected["raw_close"].items()
            if pd.notna(value) and float(value) > 0
        }
        return prices, resolved.date().isoformat()

    def _next_open_market_state(
        self, symbols: list[str], signal_as_of_date: str
    ) -> tuple[dict[str, dict[str, Any]], str]:
        unique_symbols = sorted({str(symbol) for symbol in symbols if str(symbol)})
        if not unique_symbols:
            return {}, signal_as_of_date
        signal_date = pd.Timestamp(signal_as_of_date)
        end_date = signal_date + pd.Timedelta(days=14)
        frames = []
        columns = [
            "trade_date",
            "ts_code",
            "raw_open",
            "raw_pre_close",
            "is_valid_ohlc",
            "is_tradable_observation",
            "can_buy_open_proxy",
            "can_sell_open_proxy",
        ]
        for year in range(signal_date.year, end_date.year + 1):
            for path in sorted((self.data_path / f"trade_year={year}").glob("*.parquet")):
                available = set(pq.read_schema(path).names)
                selected_columns = [column for column in columns if column in available]
                frames.append(
                    pd.read_parquet(path, columns=selected_columns)
                )
        if not frames:
            raise FileNotFoundError(
                f"No panel partitions found after {signal_as_of_date} for open execution"
            )
        data = pd.concat(frames, ignore_index=True)
        data["trade_date"] = pd.to_datetime(data["trade_date"])
        data = data[
            (data["trade_date"] > signal_date)
            & (data["trade_date"] <= end_date)
            & (data["ts_code"].isin(unique_symbols))
        ]
        if data.empty:
            raise ValueError(f"No next-session raw open prices found after {signal_as_of_date}")
        execution_date = data["trade_date"].min()
        selected = data[data["trade_date"] == execution_date].set_index("ts_code")
        market_state: dict[str, dict[str, Any]] = {}
        for symbol, row in selected.iterrows():
            raw_open = row.get("raw_open")
            if pd.isna(raw_open) or float(raw_open) <= 0:
                continue
            can_buy, can_sell = _open_trade_permissions(row)
            market_state[str(symbol)] = {
                "raw_open": float(raw_open),
                "can_buy_open": can_buy,
                "can_sell_open": can_sell,
            }
        if not market_state:
            raise ValueError(f"No usable raw open prices on {execution_date.date().isoformat()}")
        return market_state, execution_date.date().isoformat()


def _round_lot(quantity: float, lot_size: int) -> int:
    return max(0, int(math.floor(quantity / lot_size)) * lot_size)


def _open_trade_permissions(
    row: pd.Series, opening_limit_threshold: float = 0.0995
) -> tuple[bool, bool]:
    valid = True
    if "is_valid_ohlc" in row.index:
        valid = valid and bool(row.get("is_valid_ohlc"))
    if "is_tradable_observation" in row.index:
        valid = valid and bool(row.get("is_tradable_observation"))
    raw_open = row.get("raw_open")
    if pd.isna(raw_open) or float(raw_open) <= 0:
        valid = False
    if "can_buy_open_proxy" in row.index and "can_sell_open_proxy" in row.index:
        return (
            valid and bool(row.get("can_buy_open_proxy")),
            valid and bool(row.get("can_sell_open_proxy")),
        )
    raw_pre_close = row.get("raw_pre_close")
    if pd.isna(raw_pre_close) or float(raw_pre_close) <= 0:
        return valid, valid
    open_move = float(raw_open) / float(raw_pre_close) - 1.0
    return (
        valid and open_move < opening_limit_threshold,
        valid and open_move > -opening_limit_threshold,
    )


def _blocked_order(
    trade_date: str, symbol: str, side: str, quantity: int, reason: str
) -> dict[str, Any]:
    return {
        "trade_date": trade_date,
        "symbol": symbol,
        "side": side,
        "quantity": int(quantity),
        "reason": reason,
    }


def _trade(
    trade_date: str,
    symbol: str,
    security_name: str,
    side: str,
    quantity: int,
    price: float,
    fees: float,
    reason: str,
    execution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "trade_date": trade_date,
        "symbol": symbol,
        "security_name": security_name,
        "side": side,
        "quantity": quantity,
        "price_cny": price,
        "notional_cny": quantity * price,
        "fees_cny": fees,
        "reason": reason,
        "execution": execution or {},
    }
