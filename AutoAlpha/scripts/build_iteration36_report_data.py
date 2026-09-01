from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from autoalpha.backtest.capital import (
    CapitalBacktestSpec,
    factor_from_iteration,
    run_capital_backtest,
)
from autoalpha.backtest.costs import USEquityExecutionCosts
from autoalpha.dsl.expression import FactorDefinition, field, operation
from autoalpha.research.evaluation import cross_sectional_ic
from autoalpha.service.evaluator import PriceVolumeEvaluator
from autoalpha.service.store import ServiceStore

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATABASE = ROOT / "runtime/autoalpha.sqlite3"
PANEL = ROOT.parent / "data/processed/daily_panel"
CONFIG = ROOT / "config/research.toml"
CAPITAL_DIR = ROOT / "runtime/backtests/iteration_36_current_best_2020_2026_1m_50pct"
OUTPUT = ROOT / "output/pdf/iteration_36_factor_research"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    store = ServiceStore(DATABASE)
    state = store.state()
    record = store.iteration_record(str(state["run_id"]), 36)
    if record is None:
        raise RuntimeError("Iteration 36 is missing from the current run")
    factor = factor_from_iteration(record)
    original_metrics = record["metrics"]

    evaluator = PriceVolumeEvaluator(PANEL, CONFIG)
    fields = evaluator._load_fields()  # noqa: SLF001 - frozen research-report extraction
    signal = evaluator.compiler.evaluate(factor.expression, fields) * factor.expected_direction
    validation_start = pd.Timestamp(evaluator.config.splits.validation.start)
    validation_end = pd.Timestamp(evaluator.config.splits.validation.end)
    signal = signal.loc[validation_start:validation_end]
    forward_return = (
        fields["adj_close"].pct_change(fill_method=None).shift(-1).reindex(signal.index)
    )

    sensitivity = _parameter_sensitivity(evaluator)
    yearly_ic, deciles, exposures = _signal_diagnostics(signal, forward_return, fields)
    capital = _capital_diagnostics(factor)
    curve = _load_curve()
    temporal = _temporal_diagnostics(curve)
    comparison = _iteration20_comparison(curve)
    experiment_snapshot = _experiment_snapshot()

    snapshot = {
        "report_as_of": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "database": str(DATABASE),
        "panel": str(PANEL),
        "service_state_at_extraction": state,
        "experiment_snapshot": experiment_snapshot,
        "iteration": 36,
        "candidate_id": record["candidate_id"],
        "factor": {
            "factor_id": factor.factor_id,
            "name": factor.name,
            "family": factor.family,
            "hypothesis": factor.hypothesis,
            "expected_direction": factor.expected_direction,
            "expression": factor.expression.to_dict(),
        },
        "exploratory_evaluation": original_metrics,
        "parameter_sensitivity": sensitivity,
        "yearly_rank_ic": yearly_ic,
        "decile_forward_returns": deciles,
        "cross_sectional_exposures": exposures,
        "capital_scenarios": capital,
        "capital_temporal_diagnostics": temporal,
        "iteration_20_comparison": comparison,
        "known_limitations": [
            "The factor was selected after repeated LLM-generated experiments.",
            "The panel is not institutionally point-in-time ready.",
            "Historical industry, free-float market cap, and eligibility histories "
            "are unavailable.",
            "The capital ledger includes explicit fees but not spread or nonlinear market impact.",
            "The full-period replay includes dates before factor discovery and is "
            "selection-biased.",
        ],
    }
    (OUTPUT / "research_snapshot.json").write_text(
        json.dumps(_json_ready(snapshot), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    pd.DataFrame(sensitivity).to_csv(OUTPUT / "parameter_sensitivity.csv", index=False)
    pd.DataFrame(yearly_ic).to_csv(OUTPUT / "yearly_rank_ic.csv", index=False)
    pd.DataFrame(deciles).to_csv(OUTPUT / "decile_forward_returns.csv", index=False)
    _plot_performance(curve, temporal)
    _plot_calendar(curve)
    _plot_diagnostics(sensitivity, deciles, exposures, yearly_ic)
    print(json.dumps(_json_ready(snapshot), ensure_ascii=False, indent=2))


def _parameter_sensitivity(evaluator: PriceVolumeEvaluator) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for window in (10, 20, 40, 60):
        for threshold in (3, 5, 7):
            expression = operation(
                "cs_zscore",
                operation(
                    "winsorize_mad",
                    operation("rolling_std", field("amount"), window=window),
                    threshold=threshold,
                ),
            )
            candidate = FactorDefinition(
                name=f"Dollar_Volume_Stability_{window}d_MAD{threshold}",
                family="Dollar Volume Volatility",
                hypothesis="Parameter-neighborhood diagnostic for iteration 36.",
                expression=expression,
                expected_direction=-1,
            )
            result = evaluator.evaluate(candidate)
            rows.append(
                {
                    "window": window,
                    "mad_threshold": threshold,
                    "sharpe_ratio": result.metrics["sharpe_ratio"],
                    "simple_annual_return": result.metrics["simple_annual_return"],
                    "rank_ic_mean": result.metrics["rank_ic_mean"],
                    "rank_ic_ir": result.metrics["rank_ic_ir"],
                    "annual_turnover": result.metrics["annual_turnover"],
                    "max_drawdown": result.metrics["incremental_max_drawdown"],
                }
            )
    return rows


def _signal_diagnostics(
    signal: pd.DataFrame,
    forward_return: pd.DataFrame,
    fields: dict[str, pd.DataFrame],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    rank_ic = cross_sectional_ic(signal, forward_return, minimum_names=30)
    yearly_ic = []
    for year, values in rank_ic.groupby(rank_ic.index.year):
        yearly_ic.append(
            {
                "year": int(year),
                "mean_rank_ic": float(values.mean()),
                "rank_ic_ir": _annualized_ir(values),
                "positive_day_ratio": float((values > 0).mean()),
                "observations": int(len(values)),
            }
        )

    percentile = signal.rank(axis=1, pct=True, method="average")
    deciles = []
    for decile in range(1, 11):
        low = (decile - 1) / 10
        high = decile / 10
        mask = (percentile > low) & (percentile <= high)
        daily = forward_return.where(mask).mean(axis=1).dropna()
        deciles.append(
            {
                "decile": decile,
                "simple_annual_return": float(daily.mean() * 245),
                "daily_mean_return": float(daily.mean()),
                "observations": int(len(daily)),
            }
        )

    aligned = {name: frame.reindex(signal.index) for name, frame in fields.items()}
    return_volatility = aligned["adj_close"].pct_change(fill_method=None).rolling(20).std()
    exposure_frames = {
        "log_amount": np.log1p(aligned["amount"].clip(lower=0)),
        "log_volume": np.log1p(aligned["vol"].clip(lower=0)),
        "log_price": np.log(aligned["adj_close"].where(aligned["adj_close"] > 0)),
        "return_volatility_20d": return_volatility,
    }
    exposures = {}
    for name, exposure in exposure_frames.items():
        daily = signal.corrwith(exposure, axis=1, method="spearman").dropna()
        exposures[name] = float(daily.mean())
    return yearly_ic, deciles, exposures


def _capital_diagnostics(factor: FactorDefinition) -> dict[str, dict[str, Any]]:
    spec = CapitalBacktestSpec(
        start=pd.Timestamp("2020-01-01").date(),
        end=pd.Timestamp("2026-07-14").date(),
        initial_cash=1_000_000,
        target_gross_exposure=0.50,
        top_fraction=0.10,
        max_positions=30,
        max_volume_participation=0.05,
    )
    zero_cost = USEquityExecutionCosts(0, 0, 0, 0)
    double_cost = USEquityExecutionCosts(3.0, 10.0, 0.2, 10.0)
    delayed_factor = FactorDefinition(
        name=f"{factor.name}_extra_delay_1d",
        family=factor.family,
        hypothesis=factor.hypothesis,
        expression=operation("delay", factor.expression, periods=1),
        expected_direction=factor.expected_direction,
    )
    scenarios = {
        "zero_explicit_cost": run_capital_backtest(factor, PANEL, spec, costs=zero_cost),
        "double_explicit_cost": run_capital_backtest(factor, PANEL, spec, costs=double_cost),
        "extra_signal_delay_1d": run_capital_backtest(
            delayed_factor, PANEL, spec, costs=USEquityExecutionCosts()
        ),
    }
    return {
        name: {
            key: report.metrics[key]
            for key in (
                "final_nav_usd",
                "total_return",
                "compound_annual_return",
                "sharpe_ratio",
                "max_drawdown",
                "total_fees_usd",
                "annualized_one_way_turnover",
                "average_gross_exposure",
            )
        }
        for name, report in scenarios.items()
    }


def _load_curve() -> pd.DataFrame:
    curve = pd.read_csv(CAPITAL_DIR / "equity_curve.csv", parse_dates=["trade_date"])
    return curve.set_index("trade_date").sort_index()


def _temporal_diagnostics(curve: pd.DataFrame) -> dict[str, Any]:
    returns = curve["daily_return"].iloc[1:]
    rolling_mean = returns.rolling(245, min_periods=120).mean()
    rolling_std = returns.rolling(245, min_periods=120).std(ddof=1)
    rolling_sharpe = (
        rolling_mean.div(rolling_std).mul(math.sqrt(245)).replace([np.inf, -np.inf], np.nan)
    )
    monthly = (1 + returns).groupby(returns.index.to_period("M")).prod() - 1
    episodes = _drawdown_episodes(curve["nav_usd"])
    return {
        "rolling_245d_sharpe_latest": float(rolling_sharpe.dropna().iloc[-1]),
        "rolling_245d_sharpe_min": float(rolling_sharpe.min()),
        "rolling_245d_sharpe_max": float(rolling_sharpe.max()),
        "positive_month_ratio": float((monthly > 0).mean()),
        "best_month": {"month": str(monthly.idxmax()), "return": float(monthly.max())},
        "worst_month": {"month": str(monthly.idxmin()), "return": float(monthly.min())},
        "top_drawdown_episodes": episodes[:5],
    }


def _drawdown_episodes(nav: pd.Series) -> list[dict[str, Any]]:
    running_peak = nav.cummax()
    drawdown = nav / running_peak - 1
    episodes = []
    in_drawdown = False
    start = trough = None
    trough_value = 0.0
    for date, value in drawdown.items():
        if value < 0 and not in_drawdown:
            in_drawdown = True
            start = date
            trough = date
            trough_value = float(value)
        elif value < trough_value and in_drawdown:
            trough = date
            trough_value = float(value)
        elif value >= 0 and in_drawdown:
            episodes.append(
                {
                    "start": start.date().isoformat(),
                    "trough": trough.date().isoformat(),
                    "recovery": date.date().isoformat(),
                    "max_drawdown": trough_value,
                    "duration_trading_days": int(len(nav.loc[start:date]) - 1),
                }
            )
            in_drawdown = False
            trough_value = 0.0
    if in_drawdown:
        episodes.append(
            {
                "start": start.date().isoformat(),
                "trough": trough.date().isoformat(),
                "recovery": None,
                "max_drawdown": trough_value,
                "duration_trading_days": int(len(nav.loc[start:]) - 1),
            }
        )
    return sorted(episodes, key=lambda item: item["max_drawdown"])


def _iteration20_comparison(curve36: pd.DataFrame) -> dict[str, Any]:
    path = ROOT / "runtime/backtests/best_factor_2020_2026_1m_50pct/equity_curve.csv"
    curve20 = pd.read_csv(path, parse_dates=["trade_date"]).set_index("trade_date")
    paired = pd.concat(
        [
            curve20["daily_return"].rename("iteration20"),
            curve36["daily_return"].rename("iteration36"),
        ],
        axis=1,
        join="inner",
    ).iloc[1:]
    return {
        "daily_return_correlation": float(paired.corr().iloc[0, 1]),
        "iteration20_total_return": float(curve20["cumulative_return"].iloc[-1]),
        "iteration36_total_return": float(curve36["cumulative_return"].iloc[-1]),
        "iteration20_sharpe": 0.8799111507673891,
        "iteration36_sharpe": 1.0860430139023607,
    }


def _experiment_snapshot() -> dict[str, Any]:
    with sqlite3.connect(DATABASE) as connection:
        completed, failed, maximum = connection.execute(
            """SELECT
            SUM(status='COMPLETED'), SUM(status='FAILED'), MAX(iteration)
            FROM iterations"""
        ).fetchone()
        leaders = connection.execute(
            """SELECT iteration, candidate_id,
            json_extract(proposal_json, '$.name'),
            json_extract(metrics_json, '$.sharpe_ratio')
            FROM iterations WHERE status='COMPLETED'
            ORDER BY CAST(json_extract(metrics_json, '$.sharpe_ratio') AS REAL) DESC LIMIT 3"""
        ).fetchall()
    return {
        "completed_iterations": int(completed or 0),
        "failed_iterations": int(failed or 0),
        "maximum_iteration_seen": int(maximum or 0),
        "top_three_by_exploratory_sharpe": [
            {"iteration": row[0], "candidate_id": row[1], "name": row[2], "sharpe": row[3]}
            for row in leaders
        ],
    }


def _plot_performance(curve: pd.DataFrame, temporal: dict[str, Any]) -> None:
    returns = curve["daily_return"].iloc[1:]
    rolling = (
        returns.rolling(245, min_periods=120).mean()
        / returns.rolling(245, min_periods=120).std(ddof=1)
        * math.sqrt(245)
    )
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 9), sharex=True, height_ratios=[2.2, 1, 1])
    axes[0].plot(curve.index, curve["nav_usd"] / 1_000_000, color="#175CD3", lw=1.8)
    axes[0].axhline(1, color="#667085", lw=0.8, ls="--")
    axes[0].set_ylabel("NAV (USD mn)")
    axes[0].set_title("Iteration 36 capital replay: performance and stability")
    axes[1].fill_between(curve.index, curve["drawdown"] * 100, 0, color="#D92D20", alpha=0.28)
    axes[1].plot(curve.index, curve["drawdown"] * 100, color="#D92D20", lw=0.8)
    axes[1].set_ylabel("Drawdown (%)")
    axes[2].plot(rolling.index, rolling, color="#067647", lw=1.2)
    axes[2].axhline(0, color="#667085", lw=0.8)
    axes[2].axhline(1, color="#F79009", lw=0.8, ls="--")
    axes[2].set_ylabel("Rolling Sharpe")
    axes[2].set_xlabel("Trade date")
    for axis in axes:
        axis.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(OUTPUT / "performance_and_stability.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_calendar(curve: pd.DataFrame) -> None:
    returns = curve["daily_return"].iloc[1:]
    monthly = (1 + returns).groupby(returns.index.to_period("M")).prod() - 1
    table = monthly.rename("return").to_frame()
    table["year"] = table.index.year
    table["month"] = table.index.month
    heatmap = table.pivot(index="year", columns="month", values="return").reindex(
        columns=range(1, 13)
    )
    annual = (1 + returns).groupby(returns.index.year).prod() - 1
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3), width_ratios=[1, 2.2])
    colors = ["#D92D20" if value < 0 else "#067647" for value in annual]
    ax1.bar([str(year) for year in annual.index], annual * 100, color=colors)
    ax1.axhline(0, color="#667085", lw=0.8)
    ax1.set_title("Calendar-year returns")
    ax1.set_ylabel("Return (%)")
    ax1.tick_params(axis="x", rotation=45)
    image = ax2.imshow(heatmap.to_numpy() * 100, cmap="RdYlGn", aspect="auto", vmin=-8, vmax=8)
    ax2.set_title("Monthly returns (%)")
    ax2.set_xticks(range(12), [str(month) for month in range(1, 13)])
    ax2.set_yticks(range(len(heatmap.index)), [str(year) for year in heatmap.index])
    ax2.set_xlabel("Month")
    for row in range(len(heatmap.index)):
        for column in range(12):
            value = heatmap.iloc[row, column]
            if pd.notna(value):
                ax2.text(column, row, f"{value * 100:.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax2, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(OUTPUT / "calendar_returns.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_diagnostics(
    sensitivity: list[dict[str, Any]],
    deciles: list[dict[str, Any]],
    exposures: dict[str, float],
    yearly_ic: list[dict[str, Any]],
) -> None:
    sensitivity_frame = pd.DataFrame(sensitivity)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8))
    for threshold, values in sensitivity_frame.groupby("mad_threshold"):
        axes[0, 0].plot(
            values["window"], values["sharpe_ratio"], marker="o", label=f"MAD {threshold}"
        )
    axes[0, 0].set_title("Validation Sharpe parameter neighborhood")
    axes[0, 0].set_xlabel("Rolling window")
    axes[0, 0].set_ylabel("Sharpe")
    axes[0, 0].legend(frameon=False)

    decile_frame = pd.DataFrame(deciles)
    colors = [
        "#D92D20" if value < 0 else "#175CD3" for value in decile_frame["simple_annual_return"]
    ]
    axes[0, 1].bar(decile_frame["decile"], decile_frame["simple_annual_return"] * 100, color=colors)
    axes[0, 1].set_title("Forward return by signed-signal decile")
    axes[0, 1].set_xlabel("Decile (10 = preferred)")
    axes[0, 1].set_ylabel("Simple annual return (%)")

    exposure_labels = [name.replace("_", " ") for name in exposures]
    exposure_values = list(exposures.values())
    exposure_colors = ["#D92D20" if value < 0 else "#067647" for value in exposure_values]
    axes[1, 0].barh(exposure_labels, exposure_values, color=exposure_colors)
    axes[1, 0].axvline(0, color="#667085", lw=0.8)
    axes[1, 0].set_title("Mean daily cross-sectional Spearman exposure")
    axes[1, 0].set_xlabel("Correlation")

    ic_frame = pd.DataFrame(yearly_ic)
    ic_colors = ["#D92D20" if value < 0 else "#067647" for value in ic_frame["mean_rank_ic"]]
    axes[1, 1].bar(ic_frame["year"].astype(str), ic_frame["mean_rank_ic"] * 100, color=ic_colors)
    axes[1, 1].axhline(0, color="#667085", lw=0.8)
    axes[1, 1].set_title("Rank IC by year")
    axes[1, 1].set_ylabel("Mean Rank IC (%)")
    for axis in axes.flat:
        axis.grid(alpha=0.16)
    fig.tight_layout()
    fig.savefig(OUTPUT / "factor_diagnostics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _annualized_ir(values: pd.Series) -> float:
    deviation = float(values.std(ddof=1))
    return float(values.mean() / deviation * math.sqrt(245)) if deviation > 0 else float("nan")


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (pd.Timestamp, pd.Period)):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


if __name__ == "__main__":
    main()
