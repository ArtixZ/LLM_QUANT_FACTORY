from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from autoalpha.backtest.costs import ChinaAExecutionCosts
from autoalpha.service.multifactor import factor_from_pool_record
from autoalpha.service.screener import CrossSectionalScreener, ScreenerSpec
from autoalpha.service.store import ServiceStore


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
    """Persistent end-of-day A-share paper portfolios with explicit close-fill assumptions."""

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
            "lot_size": 100,
            "execution_assumption": "RAW_CLOSE_EOD_MODEL_FILL_V1",
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
        targets = {row["ts_code"]: row for row in screen["rows"] if row["raw_close"]}
        if not targets:
            raise ValueError("No target securities have usable raw close prices")
        previous_positions = {item["symbol"]: dict(item) for item in portfolio.get("positions", [])}
        prices, trade_date = self._prices([*previous_positions, *targets], screen["as_of_date"])
        prices.update({symbol: float(row["raw_close"]) for symbol, row in targets.items()})
        cash = float(portfolio["cash_cny"])
        market_value = sum(
            int(item["quantity"]) * float(prices.get(symbol, 0.0))
            for symbol, item in previous_positions.items()
        )
        nav_before = cash + market_value
        lot_size = int(config.get("lot_size", 100))
        target_value = nav_before * float(config["gross_exposure"]) / len(targets)
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
        for symbol in sorted(set(positions) | set(desired)):
            current = int(positions.get(symbol, {}).get("quantity", 0))
            target = desired.get(symbol, 0)
            if current <= target:
                continue
            quantity = current - target
            reference = prices.get(symbol)
            if not reference:
                continue
            price = reference * (1.0 - slippage / 10_000.0)
            notional = quantity * price
            fees = costs.fees("SELL", notional, trade_date)
            cash += notional - fees
            positions[symbol]["quantity"] = target
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
                )
            )
        for symbol in sorted(desired, key=lambda value: targets[value]["rank"]):
            current = int(positions.get(symbol, {}).get("quantity", 0))
            target = desired[symbol]
            if target <= current:
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


def _round_lot(quantity: float, lot_size: int) -> int:
    return max(0, int(math.floor(quantity / lot_size)) * lot_size)


def _trade(
    trade_date: str,
    symbol: str,
    security_name: str,
    side: str,
    quantity: int,
    price: float,
    fees: float,
    reason: str,
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
    }
