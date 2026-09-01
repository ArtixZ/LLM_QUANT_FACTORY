from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from autoalpha.backtest.costs import USEquityExecutionCosts
from autoalpha.backtest.ledger import LedgerBacktester, LedgerConfig, LedgerResult
from autoalpha.data.execution_basis import inspect_execution_data_basis
from autoalpha.data.research_fields import field_definitions
from autoalpha.dsl.compiler import FactorCompiler
from autoalpha.dsl.expression import Expression, FactorDefinition
from autoalpha.dsl.semantics import SemanticValidator

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class CapitalBacktestSpec:
    start: date
    end: date
    initial_cash: float = 1_000_000.0
    target_gross_exposure: float = 0.50
    top_fraction: float = 0.10
    max_positions: int = 30
    lot_size: int = 1
    max_volume_participation: float = 0.05
    trading_days_per_year: int = 252
    holding_period_days: int = 1

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError("Backtest start cannot be after end")
        if not 0 < self.target_gross_exposure <= 1:
            raise ValueError("target_gross_exposure must be in (0, 1]")
        if self.max_positions <= 0:
            raise ValueError("max_positions must be positive")
        if self.holding_period_days <= 0:
            raise ValueError("holding_period_days must be positive")


@dataclass(frozen=True)
class CapitalBacktestReport:
    spec: CapitalBacktestSpec
    factor: FactorDefinition
    ledger: LedgerResult
    metrics: dict[str, Any]
    annual_returns: dict[str, float]


def run_capital_backtest(
    factor: FactorDefinition,
    panel_path: Path,
    spec: CapitalBacktestSpec,
    *,
    costs: USEquityExecutionCosts | None = None,
) -> CapitalBacktestReport:
    inspect_execution_data_basis(panel_path).require_capital_ledger()
    data = _load_panel(panel_path, spec)
    signal = _compile_signal(factor, data)
    market = _market_frame(data, spec)
    ledger = LedgerBacktester(
        LedgerConfig(
            horizon=spec.holding_period_days,
            initial_cash=spec.initial_cash,
            top_fraction=spec.top_fraction,
            max_positions=spec.max_positions,
            lot_size=spec.lot_size,
            max_volume_participation=spec.max_volume_participation,
            investment_buffer=1.0 - spec.target_gross_exposure,
            trading_days_per_year=spec.trading_days_per_year,
        ),
        costs,
    ).run(signal, market)
    metrics, annual_returns = _account_metrics(ledger, spec)
    return CapitalBacktestReport(spec, factor, ledger, metrics, annual_returns)


def write_capital_backtest_artifacts(
    report: CapitalBacktestReport, output_dir: Path
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    curve_path = output_dir / "equity_curve.csv"
    metrics_path = output_dir / "backtest_report.json"
    chart_path = output_dir / "pnl_curve.png"

    nav = report.ledger.nav
    curve = pd.DataFrame(
        {
            "nav_usd": nav,
            "pnl_usd": nav - report.spec.initial_cash,
            "cumulative_return": nav / report.spec.initial_cash - 1.0,
            "drawdown": nav / nav.cummax() - 1.0,
            "daily_return": report.ledger.daily_return,
            "gross_exposure": report.ledger.gross_exposure,
        }
    )
    curve.index.name = "trade_date"
    curve.to_csv(curve_path, float_format="%.10f")
    payload = {
        "protocol": "a_share_capital_ledger_v1",
        "selection_warning": (
            "The factor was selected using prior experiment results; this full-period replay "
            "is selection-biased and is not an untouched out-of-sample estimate."
        ),
        "spec": _json_ready(asdict(report.spec)),
        "factor": {
            "factor_id": report.factor.factor_id,
            "name": report.factor.name,
            "family": report.factor.family,
            "hypothesis": report.factor.hypothesis,
            "expected_direction": report.factor.expected_direction,
            "expression": report.factor.expression.to_dict(),
        },
        "metrics": report.metrics,
        "annual_returns": report.annual_returns,
    }
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    _render_chart(report, chart_path)
    return {
        "equity_curve_csv": str(curve_path),
        "backtest_report_json": str(metrics_path),
        "pnl_curve_png": str(chart_path),
    }


def factor_from_iteration(record: dict[str, Any]) -> FactorDefinition:
    proposal = record.get("proposal")
    if not proposal:
        raise ValueError("Iteration does not contain a persisted proposal")
    return FactorDefinition(
        name=str(proposal["name"]),
        family=str(proposal["family"]),
        hypothesis=str(proposal["hypothesis"]),
        expression=Expression.from_dict(proposal["expression"]),
        expected_direction=int(proposal["expected_direction"]),
    )


def _load_panel(panel_path: Path, spec: CapitalBacktestSpec) -> pd.DataFrame:
    warmup = pd.Timestamp(spec.start) - pd.Timedelta(days=400)
    columns = [
        "trade_date",
        "symbol",
        "open",
        "close",
        "adj_close",
        "vol",
        "amount",
        "is_valid_ohlc",
        "is_tradable_observation",
        # Side-specific open eligibility is point-in-time panel state; without
        # loading it the market frame would silently fall back to bar validity.
        "can_buy_open",
        "can_sell_open",
    ]
    frames = []
    for year in range(warmup.year, spec.end.year + 1):
        for path in sorted((panel_path / f"trade_year={year}").glob("*.parquet")):
            frames.append(pd.read_parquet(path, columns=columns))
    if not frames:
        raise FileNotFoundError(f"No panel partitions found under {panel_path}")
    data = pd.concat(frames, ignore_index=True)
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    return data[
        (data["trade_date"] >= warmup) & (data["trade_date"] <= pd.Timestamp(spec.end))
    ].copy()


def _compile_signal(factor: FactorDefinition, data: pd.DataFrame) -> pd.DataFrame:
    fields = {}
    valid = data["is_valid_ohlc"].fillna(False) & data["is_tradable_observation"].fillna(False)
    required = _expression_fields(factor.expression)
    for name in required:
        values = data[["trade_date", "symbol", name]].copy()
        values.loc[~valid, name] = np.nan
        fields[name] = values.pivot(index="trade_date", columns="symbol", values=name).sort_index()
    validator = SemanticValidator(
        field_definitions(data.columns, include_open=False),
        # Composite portfolios contain several independently validated factor trees.
        maximum_nodes=160,
        maximum_lookback=252,
    )
    return FactorCompiler(validator).evaluate(factor.expression, fields) * factor.expected_direction


def _market_frame(data: pd.DataFrame, spec: CapitalBacktestSpec) -> pd.DataFrame:
    market = data[
        (data["trade_date"] >= pd.Timestamp(spec.start))
        & (data["trade_date"] <= pd.Timestamp(spec.end))
    ].copy()
    valid = market["is_valid_ohlc"].fillna(False) & market["is_tradable_observation"].fillna(False)
    # US equities have no daily price limits, so open eligibility comes from the
    # panel's point-in-time tradability flags rather than an open-move threshold.
    # _load_panel always loads both flags; a missing column should raise here
    # rather than silently fall back to bar validity.
    for side in ("can_buy_open", "can_sell_open"):
        market[side] = valid & market[side].fillna(False)
    market.loc[~market["is_valid_ohlc"].fillna(False), ["open", "close"]] = np.nan
    return market.rename(columns={"trade_date": "date", "vol": "volume"})[
        ["date", "symbol", "open", "close", "volume", "can_buy_open", "can_sell_open"]
    ]


def _account_metrics(
    ledger: LedgerResult, spec: CapitalBacktestSpec
) -> tuple[dict[str, Any], dict[str, float]]:
    nav = ledger.nav
    returns = ledger.daily_return.iloc[1:]
    downside = returns[returns < 0]
    downside_deviation = float(np.sqrt(np.mean(np.square(downside)))) if len(downside) else 0.0
    sortino = (
        float(returns.mean() / downside_deviation * math.sqrt(spec.trading_days_per_year))
        if downside_deviation > 0
        else float("nan")
    )
    years = max((nav.index[-1] - nav.index[0]).days / 365.2425, 1 / 365.2425)
    total_notional = float(ledger.trades["notional"].sum()) if not ledger.trades.empty else 0.0
    gains = float(returns[returns > 0].sum())
    losses = abs(float(returns[returns < 0].sum()))
    drawdown = nav / nav.cummax() - 1.0
    annual_returns = {
        str(int(year)): float(values.iloc[-1] / values.iloc[0] - 1.0)
        for year, values in nav.groupby(nav.index.year)
        if len(values) > 1
    }
    metrics = {
        "start_date": nav.index[0].date().isoformat(),
        "end_date": nav.index[-1].date().isoformat(),
        "trading_days": len(nav),
        "initial_cash_usd": spec.initial_cash,
        "final_nav_usd": float(nav.iloc[-1]),
        "total_pnl_usd": float(nav.iloc[-1] - spec.initial_cash),
        "total_return": float(nav.iloc[-1] / spec.initial_cash - 1.0),
        "peak_nav_usd": float(nav.max()),
        "peak_nav_date": nav.idxmax().date().isoformat(),
        "ending_drawdown": float(drawdown.iloc[-1]),
        "simple_annual_return": float(returns.mean() * spec.trading_days_per_year),
        "compound_annual_return": ledger.annual_return,
        "annualized_volatility": float(returns.std(ddof=1) * math.sqrt(spec.trading_days_per_year)),
        "sharpe_ratio": ledger.sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": ledger.annual_return / abs(ledger.max_drawdown)
        if ledger.max_drawdown < 0
        else float("nan"),
        "max_drawdown": ledger.max_drawdown,
        "max_drawdown_duration_trading_days": _maximum_drawdown_duration(drawdown),
        "daily_win_rate": float((returns > 0).mean()),
        "profit_factor_daily": gains / losses if losses > 0 else float("nan"),
        "one_day_var_95": float(-returns.quantile(0.05)),
        "one_day_cvar_95": float(-returns[returns <= returns.quantile(0.05)].mean()),
        "target_gross_exposure": spec.target_gross_exposure,
        "average_gross_exposure": float(ledger.gross_exposure.mean()),
        "maximum_gross_exposure": float(ledger.gross_exposure.max()),
        "annualized_one_way_turnover": 0.5 * total_notional / float(nav.mean()) / years,
        "total_trade_notional_usd": total_notional,
        "total_fees_usd": ledger.total_fees,
        "trade_count": len(ledger.trades),
        "unique_traded_symbols": int(ledger.trades["symbol"].nunique())
        if not ledger.trades.empty
        else 0,
        "final_position_count": int(ledger.final_positions["symbol"].nunique())
        if not ledger.final_positions.empty
        else 0,
        "subperiods": {
            "2020_2021": _period_metrics(
                nav, pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31"), spec
            ),
            "selection_validation": _period_metrics(
                nav, pd.Timestamp("2021-12-24"), pd.Timestamp("2024-11-12"), spec
            ),
            "post_selection_observation": _period_metrics(
                nav, pd.Timestamp("2024-12-04"), nav.index[-1], spec
            ),
        },
    }
    return metrics, annual_returns


def _maximum_drawdown_duration(drawdown: pd.Series) -> int:
    longest = current = 0
    for value in drawdown:
        current = current + 1 if value < 0 else 0
        longest = max(longest, current)
    return longest


def _period_metrics(
    nav: pd.Series, start: pd.Timestamp, end: pd.Timestamp, spec: CapitalBacktestSpec
) -> dict[str, Any]:
    values = nav[(nav.index >= start) & (nav.index <= end)]
    if len(values) < 2:
        return {}
    returns = values.pct_change(fill_method=None).dropna()
    periods = len(returns)
    total_return = float(values.iloc[-1] / values.iloc[0] - 1.0)
    volatility = float(returns.std(ddof=1))
    drawdown = values / values.cummax() - 1.0
    return {
        "start_date": values.index[0].date().isoformat(),
        "end_date": values.index[-1].date().isoformat(),
        "trading_days": len(values),
        "total_return": total_return,
        "compound_annual_return": float(
            (values.iloc[-1] / values.iloc[0]) ** (spec.trading_days_per_year / periods) - 1.0
        ),
        "sharpe_ratio": float(returns.mean() / volatility * math.sqrt(spec.trading_days_per_year))
        if volatility > 0
        else float("nan"),
        "max_drawdown": float(drawdown.min()),
    }


def _expression_fields(expression: Expression) -> set[str]:
    names = {str(expression.parameter("name"))} if expression.operator == "field" else set()
    for argument in expression.arguments:
        names.update(_expression_fields(argument))
    return names


def _render_chart(report: CapitalBacktestReport, path: Path) -> None:
    nav = report.ledger.nav
    pnl = nav - report.spec.initial_cash
    drawdown = nav / nav.cummax() - 1.0
    figure, (axis_nav, axis_drawdown) = plt.subplots(
        2, 1, figsize=(12, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    axis_nav.plot(nav.index, nav, color="#245eea", linewidth=1.8, label="Portfolio value")
    axis_nav.axhline(report.spec.initial_cash, color="#68758a", linewidth=1, linestyle="--")
    axis_nav.fill_between(
        nav.index, report.spec.initial_cash, nav, where=pnl >= 0, color="#09845b", alpha=0.12
    )
    axis_nav.fill_between(
        nav.index, report.spec.initial_cash, nav, where=pnl < 0, color="#c33832", alpha=0.12
    )
    axis_nav.set_title(f"A-share Capital Backtest | {report.factor.name} | 50% Target Exposure")
    axis_nav.set_ylabel("Portfolio value (USD)")
    axis_nav.grid(alpha=0.2)
    axis_nav.legend(loc="upper left")
    axis_drawdown.fill_between(drawdown.index, drawdown, 0, color="#c33832", alpha=0.35)
    axis_drawdown.plot(drawdown.index, drawdown, color="#c33832", linewidth=1)
    axis_drawdown.set_ylabel("Drawdown")
    axis_drawdown.set_xlabel("Trade date")
    axis_drawdown.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def _json_ready(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value
