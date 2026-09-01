from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from autoalpha.backtest.capital import (
    CapitalBacktestSpec,
    run_capital_backtest,
    write_capital_backtest_artifacts,
)
from autoalpha.backtest.costs import USEquityExecutionCosts
from autoalpha.data.current_panel import inspect_current_panel
from autoalpha.dsl.compiler import FactorCompiler
from autoalpha.dsl.expression import (
    Expression,
    FactorDefinition,
    constant,
    operation,
)
from autoalpha.dsl.semantics import FieldDefinition, SemanticValidator
from autoalpha.research.evaluation import cross_sectional_ic
from autoalpha.research.incremental import annual_robustness
from autoalpha.service.store import ServiceStore

TRADING_DAYS = 245


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a persisted multi-factor portfolio version"
    )
    parser.add_argument("--database", type=Path, default=Path("runtime/autoalpha.sqlite3"))
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--panel", type=Path, default=Path("../data/processed/daily_panel"))
    parser.add_argument("--start", type=date.fromisoformat, default=date(2010, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 7, 15))
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    parser.add_argument("--target-exposure", type=float, default=0.50)
    parser.add_argument("--holding-period-days", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    store = ServiceStore(args.database)
    version = _portfolio_version(store, args.version)
    members = _portfolio_members(store, version)
    composite = _composite_factor(args.version, members)
    costs = USEquityExecutionCosts(
        commission_bps_each_side=1.5,
        stamp_duty_bps_sell=5.0,
        transfer_fee_bps_each_side=0.1,
        minimum_commission_usd=5.0,
    )
    research = _run_research_backtest(
        members,
        args.panel,
        args.start,
        args.end,
        costs,
        args.holding_period_days,
    )
    spec = CapitalBacktestSpec(
        start=args.start,
        end=args.end,
        initial_cash=args.initial_cash,
        target_gross_exposure=args.target_exposure,
        top_fraction=0.10,
        max_positions=30,
        max_volume_participation=0.05,
        holding_period_days=args.holding_period_days,
    )
    capital = run_capital_backtest(composite, args.panel, spec, costs=costs)
    capital_paths = write_capital_backtest_artifacts(capital, args.output / "capital")
    args.output.mkdir(parents=True, exist_ok=True)
    research["daily_returns"].to_csv(
        args.output / "research_daily_returns.csv", header=True, float_format="%.10f"
    )
    research["yearly_returns"].rename("return").to_csv(
        args.output / "research_yearly_returns.csv", header=True, float_format="%.10f"
    )
    summary = {
        "protocol": "autoalpha_portfolio_version_replay_v1",
        "selection_warning": (
            "The portfolio was selected using later experiment results. The full-period "
            "2010-2026 replay is selection-biased and is not untouched out-of-sample evidence."
        ),
        "implementation_warning": (
            "The 50% exposure and Top-30 limits are rebalance targets, not guaranteed hard "
            "caps. Unsellable or volume-limited residual positions can temporarily raise "
            "realized exposure and position count above those targets."
        ),
        "source_portfolio": {
            key: version[key]
            for key in (
                "id",
                "run_id",
                "iteration",
                "action",
                "candidate_id",
                "accepted",
                "reason",
                "created_at",
            )
        },
        "source_validation_metrics": version["metrics"],
        "members": [
            {
                "factor_id": item["factor"].factor_id,
                "source_iteration": item["source_iteration"],
                "name": item["factor"].name,
                "family": item["factor"].family,
                "expected_direction": item["factor"].expected_direction,
                "weight": item["weight"],
                "expression": item["factor"].expression.to_dict(),
            }
            for item in members
        ],
        "composite_factor": {
            "factor_id": composite.factor_id,
            "name": composite.name,
            "expression": composite.expression.to_dict(),
        },
        "requested_window": {"start": args.start.isoformat(), "end": args.end.isoformat()},
        "data_readiness": inspect_current_panel(args.panel).to_dict(),
        "research_long_short": {
            "metrics": research["metrics"],
            "annual_returns": research["yearly_returns"].to_dict(),
            "daily_returns_csv": str(args.output / "research_daily_returns.csv"),
            "yearly_returns_csv": str(args.output / "research_yearly_returns.csv"),
        },
        "capital_long_only": {
            "spec": asdict(spec),
            "metrics": capital.metrics,
            "annual_returns": capital.annual_returns,
            "artifacts": capital_paths,
        },
    }
    summary_path = args.output / "run_summary.json"
    summary_path.write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2, sort_keys=True))


def _portfolio_version(store: ServiceStore, version_id: int) -> dict[str, Any]:
    version = next(
        (item for item in store.portfolio_history(limit=1000) if item["id"] == version_id),
        None,
    )
    if version is None:
        raise ValueError(f"Portfolio version not found: {version_id}")
    if not version["accepted"]:
        raise ValueError(f"Portfolio version was not accepted: {version_id}")
    return version


def _portfolio_members(store: ServiceStore, version: dict[str, Any]) -> list[dict[str, Any]]:
    members = []
    for member in version["members"]:
        record = store.factor_pool_record(str(member["factor_id"]))
        if record is None:
            raise RuntimeError(f"Factor pool record is missing: {member['factor_id']}")
        proposal = record["proposal"]
        members.append(
            {
                "factor": FactorDefinition(
                    name=str(proposal["name"]),
                    family=str(proposal["family"]),
                    hypothesis=str(proposal["hypothesis"]),
                    expression=Expression.from_dict(proposal["expression"]),
                    expected_direction=int(proposal["expected_direction"]),
                ),
                "weight": float(member["weight"]),
                "source_iteration": int(member["source_iteration"]),
            }
        )
    total_weight = sum(item["weight"] for item in members)
    if not members or total_weight <= 0:
        raise ValueError("Portfolio must contain positive-weight factors")
    for item in members:
        item["weight"] /= total_weight
    return members


def _composite_factor(version_id: int, members: list[dict[str, Any]]) -> FactorDefinition:
    weighted = []
    for item in members:
        factor = item["factor"]
        directed = (
            operation("negate", factor.expression)
            if factor.expected_direction == -1
            else factor.expression
        )
        weighted.append(operation("multiply", constant(item["weight"]), directed))
    expression = weighted[0]
    for component in weighted[1:]:
        expression = operation("add", expression, component)
    return FactorDefinition(
        name=f"PortfolioVersion{version_id}_Composite",
        family="Persisted Multi-Factor Portfolio",
        hypothesis="Replay the exact direction-adjusted weights of the persisted portfolio.",
        expression=expression,
        expected_direction=1,
    )


def _run_research_backtest(
    members: list[dict[str, Any]],
    panel_path: Path,
    start: date,
    end: date,
    costs: USEquityExecutionCosts,
    holding_period_days: int,
) -> dict[str, Any]:
    if holding_period_days <= 0:
        raise ValueError("holding_period_days must be positive")
    fields = _load_fields(panel_path, start, end)
    validator = SemanticValidator(
        [
            FieldDefinition("close", "price"),
            FieldDefinition("adj_close", "price"),
            FieldDefinition("amount", "currency"),
            FieldDefinition("vol", "shares"),
        ],
        maximum_nodes=30,
        maximum_lookback=252,
    )
    compiler = FactorCompiler(validator)
    signals = []
    for item in members:
        factor = item["factor"]
        raw = compiler.evaluate(factor.expression, fields) * factor.expected_direction
        raw = raw.loc[(raw.index >= pd.Timestamp(start)) & (raw.index <= pd.Timestamp(end))]
        scale = raw.std(axis=1).replace(0, np.nan)
        signals.append(raw.sub(raw.mean(axis=1), axis=0).div(scale, axis=0) * item["weight"])
    composite = sum(signals[1:], signals[0].copy())
    next_return = fields["adj_close"].pct_change(fill_method=None).shift(-1)
    next_return = next_return.reindex(composite.index)
    ranks = composite.rank(axis=1, pct=True)
    positions = (ranks >= 0.9).astype(float) - (ranks <= 0.1).astype(float)
    gross = positions.abs().sum(axis=1).replace(0, np.nan)
    target_weights = positions.div(gross, axis=0).fillna(0.0)
    weights = target_weights.rolling(holding_period_days, min_periods=1).mean()
    gross_return = (weights * next_return).sum(axis=1, min_count=1).dropna()
    turnover = weights.diff().abs().sum(axis=1).mul(0.5).reindex(gross_return.index).fillna(0)
    one_way_bps = (
        costs.commission_bps_each_side
        + costs.transfer_fee_bps_each_side
        + costs.stamp_duty_bps_sell / 2
    )
    net = gross_return - turnover * one_way_bps / 10_000
    stressed = gross_return - turnover * one_way_bps * 2 / 10_000
    wealth = (1 + net).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    rank_ic = cross_sectional_ic(composite, next_return, minimum_names=30)
    yearly = net.groupby(net.index.year).apply(lambda values: float((1 + values).prod() - 1))
    robustness = annual_robustness(net)
    signal_correlation = (
        signals[0].corrwith(signals[1], axis=1).dropna()
        if len(signals) == 2
        else pd.Series(dtype=float)
    )
    selected_amount = fields["amount"].where(positions.ne(0)).median(axis=1) * 1000
    metrics = {
        "start_date": net.index.min().date().isoformat(),
        "end_date": net.index.max().date().isoformat(),
        "trading_days": len(net),
        "holding_period_days": holding_period_days,
        "sharpe_ratio": _annualized_ir(net),
        "simple_annual_return": float(net.mean() * TRADING_DAYS),
        "compound_annual_return": float(wealth.iloc[-1] ** (TRADING_DAYS / len(net)) - 1),
        "max_drawdown": float(drawdown.min()),
        "cost_stress_net_ir": _annualized_ir(stressed),
        "annual_turnover": float(turnover.mean() * TRADING_DAYS),
        "coverage": float(composite.notna().sum().sum() / composite.size),
        "capacity_usd": float(selected_amount.median() * 0.05 * 20),
        "rank_ic_mean": float(rank_ic.mean()),
        "rank_ic_ir": _annualized_ir(rank_ic),
        "positive_year_ratio": robustness.positive_year_ratio,
        "worst_year_return": robustness.worst_year_return,
        "annual_return_dispersion": robustness.annual_return_dispersion,
        "factor_signal_correlation_median": (
            float(signal_correlation.median()) if not signal_correlation.empty else 0.0
        ),
    }
    return {"metrics": metrics, "daily_returns": net, "yearly_returns": yearly}


def _load_fields(panel_path: Path, start: date, end: date) -> dict[str, pd.DataFrame]:
    warmup = pd.Timestamp(start) - pd.Timedelta(days=400)
    columns = [
        "trade_date",
        "symbol",
        "close",
        "adj_close",
        "amount",
        "vol",
        "is_valid_ohlc",
        "is_tradable_observation",
    ]
    frames = []
    for year in range(warmup.year, end.year + 1):
        for path in sorted((panel_path / f"trade_year={year}").glob("*.parquet")):
            frames.append(pd.read_parquet(path, columns=columns))
    if not frames:
        raise FileNotFoundError(f"No panel partitions found under {panel_path}")
    data = pd.concat(frames, ignore_index=True)
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    data = data[(data["trade_date"] >= warmup) & (data["trade_date"] <= pd.Timestamp(end))]
    valid = data["is_valid_ohlc"].fillna(False) & data["is_tradable_observation"].fillna(False)
    data.loc[~valid, ["close", "adj_close", "amount", "vol"]] = np.nan
    return {
        name: data.pivot(index="trade_date", columns="symbol", values=name).sort_index()
        for name in ("close", "adj_close", "amount", "vol")
    }


def _annualized_ir(values: pd.Series) -> float:
    clean = values.dropna()
    volatility = float(clean.std(ddof=1))
    return float(clean.mean() / volatility * math.sqrt(TRADING_DAYS))


def _json_ready(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


if __name__ == "__main__":
    main()
