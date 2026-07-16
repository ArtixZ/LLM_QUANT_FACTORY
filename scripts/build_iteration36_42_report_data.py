from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from autoalpha.backtest.capital import (
    CapitalBacktestSpec,
    factor_from_iteration,
    run_capital_backtest,
    write_capital_backtest_artifacts,
)
from autoalpha.dsl.expression import FactorDefinition, operation
from autoalpha.research.evaluation import cross_sectional_ic
from autoalpha.research.statistics import hac_mean_inference
from autoalpha.service.evaluator import PriceVolumeEvaluator
from autoalpha.service.multifactor import (
    _portfolio_action_gate_failures,
    portfolio_utility,
)
from autoalpha.service.store import ServiceStore

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATABASE = ROOT / "runtime/autoalpha.sqlite3"
DATA = ROOT.parent / "data"
CONFIG = ROOT / "config/research.toml"
OUTPUT = ROOT / "output/pdf/iteration_36_42_multifactor_research"
TRADING_DAYS = 245


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    store = ServiceStore(DATABASE)
    state = store.state()
    run_id = str(state["run_id"])
    records = {iteration: store.iteration_record(run_id, iteration) for iteration in (36, 42)}
    if any(record is None for record in records.values()):
        raise RuntimeError("Iteration 36 or 42 is missing from the current run")
    factor36 = factor_from_iteration(records[36])
    factor42 = factor_from_iteration(records[42])
    composite_factor = _composite_factor(factor36, factor42)
    evaluator = PriceVolumeEvaluator(DATA, CONFIG)

    evaluation36 = evaluator.evaluate_portfolio([factor36])
    evaluation42 = evaluator.evaluate_portfolio([factor42])
    combination = evaluator.evaluate_portfolio(
        [factor36, factor42], benchmark_factors=[factor36]
    )
    path36 = evaluator._portfolio_path([factor36])  # noqa: SLF001
    path42 = evaluator._portfolio_path([factor42])  # noqa: SLF001
    path_combo = evaluator._portfolio_path([factor36, factor42])  # noqa: SLF001

    returns = pd.concat(
        {
            "Iteration 36": path36["net"],
            "Iteration 42": path42["net"],
            "36+42 Equal Composite": path_combo["net"],
        },
        axis=1,
        join="inner",
    ).dropna()
    yearly = _yearly_returns(returns)
    monthly = _monthly_returns(returns["36+42 Equal Composite"])
    drawdowns = (1 + returns).cumprod().div((1 + returns).cumprod().cummax()) - 1
    rolling_sharpe = _rolling_sharpe(returns, window=120)

    signal36 = evaluator._factor_signal(factor36)  # noqa: SLF001
    signal42 = evaluator._factor_signal(factor42)  # noqa: SLF001
    composite_signal = (signal36 + signal42) / 2
    forward_return = (
        evaluator._load_fields()["adj_close"]  # noqa: SLF001
        .pct_change(fill_method=None)
        .shift(-1)
        .reindex(composite_signal.index)
    )
    composite_rank_ic = cross_sectional_ic(
        composite_signal,
        forward_return,
        minimum_names=evaluator.config.minimum_cross_section,
    )
    signal_correlation = signal36.corrwith(signal42, axis=1).dropna()
    return_correlation = float(returns["Iteration 36"].corr(returns["Iteration 42"]))
    inference = hac_mean_inference(returns["36+42 Equal Composite"].to_numpy(), lags=5)
    bootstrap = _moving_block_bootstrap(returns["36+42 Equal Composite"])
    capital = _capital_replay(composite_factor)

    comparison = _comparison_table(
        evaluation36.metrics,
        evaluation42.metrics,
        combination.metrics,
    )
    gate_failures = _portfolio_action_gate_failures(
        combination.metrics, evaluator.config
    )
    snapshot = {
        "report_as_of": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "database": str(DATABASE),
        "data_workspace": evaluator.workspace.to_dict(),
        "service_state_at_extraction": state,
        "research_window": {
            "start": returns.index.min().date().isoformat(),
            "end": returns.index.max().date().isoformat(),
            "observations": len(returns),
            "protocol_split": "validation",
        },
        "factors": {
            "iteration_36": _factor_snapshot(factor36, records[36]),
            "iteration_42": _factor_snapshot(factor42, records[42]),
        },
        "combination": {
            "method": "equal mean of cross-sectional z-scored factor signals",
            "weights": {factor36.factor_id: 0.5, factor42.factor_id: 0.5},
            "metrics": combination.metrics,
            "utility_change_vs_iteration_36": (
                portfolio_utility(combination.metrics)
                - portfolio_utility(evaluation36.metrics)
            ),
            "factor_signal_correlation_median": float(signal_correlation.median()),
            "factor_signal_correlation_mean": float(signal_correlation.mean()),
            "strategy_return_correlation": return_correlation,
            "composite_rank_ic_mean": float(composite_rank_ic.mean()),
            "composite_rank_ic_ir": _annualized_ir(composite_rank_ic),
            "net_return_hac_t_stat": inference.t_stat,
            "net_return_hac_p_value": _two_sided_p(inference.t_stat),
            "moving_block_bootstrap": bootstrap,
            "diversification_gate_failures": gate_failures,
            "diversification_gate_passed": not gate_failures,
        },
        "extended_capital_replay": capital["snapshot"],
        "comparison": comparison,
        "yearly_returns": yearly.reset_index().to_dict("records"),
        "monthly_returns": monthly.reset_index().to_dict("records"),
        "limitations": [
            "Both factors and their combination were evaluated on the same validation split.",
            "The combination result is not an untouched holdout result.",
            "Both factors belong to the trading-activity stability theme.",
            "The local panel is not institutionally point-in-time ready.",
            "The portfolio is a dollar-neutral top/bottom-decile research portfolio, "
            "not a long-only capital ledger.",
            "The evaluator models explicit fees but not a full nonlinear spread and "
            "market-impact surface.",
            "The reported bootstrap interval does not correct for factor-selection multiplicity.",
        ],
    }
    (OUTPUT / "research_snapshot.json").write_text(
        json.dumps(_json_ready(snapshot), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    comparison.to_csv(OUTPUT / "metric_comparison.csv", index=False)
    yearly.to_csv(OUTPUT / "yearly_returns.csv")
    monthly.to_csv(OUTPUT / "monthly_returns.csv", index=False)
    pd.DataFrame(
        {
            "trade_date": returns.index,
            "factor_signal_correlation": signal_correlation.reindex(returns.index),
            "composite_rank_ic": composite_rank_ic.reindex(returns.index),
        }
    ).to_csv(OUTPUT / "daily_diagnostics.csv", index=False)
    _plot_performance(returns, drawdowns, rolling_sharpe)
    _plot_calendar_and_attribution(yearly, comparison, signal_correlation)
    _plot_monthly_heatmap(monthly)
    _plot_capital_comparison(capital["iteration_36_curve"], capital["combination_curve"])
    print(json.dumps(_json_ready(snapshot), ensure_ascii=False, indent=2))


def _factor_snapshot(factor, record: dict[str, Any]) -> dict[str, Any]:
    return {
        "iteration": record["iteration"],
        "factor_id": factor.factor_id,
        "name": factor.name,
        "family": factor.family,
        "hypothesis": factor.hypothesis,
        "expected_direction": factor.expected_direction,
        "expression": factor.expression.to_dict(),
        "original_metrics": record["metrics"],
    }


def _composite_factor(
    factor36: FactorDefinition, factor42: FactorDefinition
) -> FactorDefinition:
    expression36 = (
        operation("negate", factor36.expression)
        if factor36.expected_direction == -1
        else factor36.expression
    )
    expression42 = (
        operation("negate", factor42.expression)
        if factor42.expected_direction == -1
        else factor42.expression
    )
    return FactorDefinition(
        name="Iteration36_42_EqualComposite",
        family="Trading Activity Stability Composite",
        hypothesis=(
            "Combining dollar-volume volatility stability and share-volume coefficient-of-"
            "variation stability improves risk-adjusted stock selection."
        ),
        expression=operation("add", expression36, expression42),
        expected_direction=1,
    )


def _capital_replay(composite_factor: FactorDefinition) -> dict[str, Any]:
    spec = CapitalBacktestSpec(
        start=pd.Timestamp("2020-01-01").date(),
        end=pd.Timestamp("2026-07-14").date(),
        initial_cash=1_000_000,
        target_gross_exposure=0.50,
        top_fraction=0.10,
        max_positions=30,
        max_volume_participation=0.05,
    )
    panel = DATA / "processed/daily_panel"
    report = run_capital_backtest(composite_factor, panel, spec)
    artifacts = write_capital_backtest_artifacts(report, OUTPUT / "capital_replay")
    combination_curve = pd.DataFrame(
        {
            "nav_cny": report.ledger.nav,
            "daily_return": report.ledger.daily_return,
        }
    )
    iteration36_path = (
        ROOT / "runtime/backtests/iteration_36_current_best_2020_2026_1m_50pct"
    )
    iteration36_curve = pd.read_csv(
        iteration36_path / "equity_curve.csv", parse_dates=["trade_date"]
    ).set_index("trade_date")
    iteration36_report = json.loads(
        (iteration36_path / "backtest_report.json").read_text(encoding="utf-8")
    )
    return {
        "snapshot": {
            "protocol": "a_share_capital_ledger_v1",
            "comparability": (
                "Same dates, initial cash, target exposure, selection fraction, position cap, "
                "lot size, fee model and opening tradability rules as the prior Iteration 36 "
                "report."
            ),
            "selection_warning": (
                "The full-period replay contains dates before factor discovery and is not an "
                "untouched out-of-sample estimate."
            ),
            "combination_factor": {
                "factor_id": composite_factor.factor_id,
                "expression": composite_factor.expression.to_dict(),
            },
            "spec": {
                "start": spec.start.isoformat(),
                "end": spec.end.isoformat(),
                "actual_start": report.metrics["start_date"],
                "actual_end": report.metrics["end_date"],
                "initial_cash_cny": spec.initial_cash,
                "target_gross_exposure": spec.target_gross_exposure,
                "top_fraction": spec.top_fraction,
                "max_positions": spec.max_positions,
            },
            "iteration_36_metrics": iteration36_report["metrics"],
            "combination_metrics": report.metrics,
            "combination_annual_returns": report.annual_returns,
            "artifacts": artifacts,
        },
        "iteration_36_curve": iteration36_curve,
        "combination_curve": combination_curve,
    }


def _comparison_table(
    metrics36: dict[str, Any],
    metrics42: dict[str, Any],
    combo: dict[str, Any],
) -> pd.DataFrame:
    rows = [
        ("Sharpe", "portfolio_sharpe_ratio"),
        ("Simple annual return", "portfolio_simple_annual_return"),
        ("Compound annual return", "portfolio_compound_annual_return"),
        ("Maximum drawdown", "portfolio_max_drawdown"),
        ("Cost-stress net IR", "portfolio_cost_stress_net_ir"),
        ("Annual turnover", "portfolio_annual_turnover"),
        ("Capacity CNY", "portfolio_capacity_cny"),
        ("Annual return dispersion", "portfolio_annual_return_dispersion"),
    ]
    return pd.DataFrame(
        [
            {
                "metric": label,
                "iteration_36": metrics36[key],
                "iteration_42": metrics42[key],
                "combination_36_42": combo[key],
            }
            for label, key in rows
        ]
    )


def _yearly_returns(returns: pd.DataFrame) -> pd.DataFrame:
    return (1 + returns).groupby(returns.index.year).prod() - 1


def _monthly_returns(returns: pd.Series) -> pd.DataFrame:
    monthly = (1 + returns).groupby(returns.index.to_period("M")).prod() - 1
    return pd.DataFrame(
        {
            "year": monthly.index.year,
            "month": monthly.index.month,
            "return": monthly.to_numpy(),
        }
    )


def _rolling_sharpe(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    mean = returns.rolling(window, min_periods=60).mean()
    std = returns.rolling(window, min_periods=60).std(ddof=1).replace(0, np.nan)
    return mean.div(std).mul(math.sqrt(TRADING_DAYS))


def _moving_block_bootstrap(returns: pd.Series) -> dict[str, float]:
    values = returns.to_numpy()
    rng = np.random.default_rng(20260715)
    block = 20
    samples = 2_000
    annual_returns = np.empty(samples)
    sharpes = np.empty(samples)
    starts = np.arange(0, len(values) - block + 1)
    blocks_needed = math.ceil(len(values) / block)
    for sample in range(samples):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        path = np.concatenate([values[start : start + block] for start in chosen])[: len(values)]
        annual_returns[sample] = path.mean() * TRADING_DAYS
        sharpes[sample] = path.mean() / path.std(ddof=1) * math.sqrt(TRADING_DAYS)
    return {
        "block_length": block,
        "samples": samples,
        "annual_return_ci_95_low": float(np.quantile(annual_returns, 0.025)),
        "annual_return_ci_95_high": float(np.quantile(annual_returns, 0.975)),
        "sharpe_ci_95_low": float(np.quantile(sharpes, 0.025)),
        "sharpe_ci_95_high": float(np.quantile(sharpes, 0.975)),
    }


def _plot_performance(
    returns: pd.DataFrame,
    drawdowns: pd.DataFrame,
    rolling_sharpe: pd.DataFrame,
) -> None:
    colors = ["#175CD3", "#B54708", "#067647"]
    wealth = (1 + returns).cumprod()
    figure, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for column, color in zip(returns.columns, colors, strict=True):
        axes[0].plot(wealth.index, wealth[column], label=column, color=color, linewidth=2)
        axes[1].plot(
            drawdowns.index,
            drawdowns[column] * 100,
            label=column,
            color=color,
            linewidth=1.5,
        )
        axes[2].plot(
            rolling_sharpe.index,
            rolling_sharpe[column],
            label=column,
            color=color,
            linewidth=1.5,
        )
    axes[0].set_ylabel("Wealth index")
    axes[0].legend(frameon=False, ncol=3, fontsize=9)
    axes[1].set_ylabel("Drawdown (%)")
    axes[2].set_ylabel("Rolling 120d Sharpe")
    axes[2].axhline(0, color="#98A2B3", linewidth=0.8)
    for axis in axes:
        axis.grid(axis="y", color="#E4E7EC", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Validation Performance: Iteration 36, Iteration 42 and Equal Composite")
    figure.tight_layout()
    figure.savefig(OUTPUT / "performance_overview.png", dpi=190, bbox_inches="tight")
    plt.close(figure)


def _plot_calendar_and_attribution(
    yearly: pd.DataFrame,
    comparison: pd.DataFrame,
    signal_correlation: pd.Series,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    yearly.mul(100).plot.bar(ax=axes[0], color=["#175CD3", "#B54708", "#067647"])
    axes[0].set_title("Calendar return")
    axes[0].set_ylabel("Return (%)")
    axes[0].set_xlabel("")
    axes[0].legend(frameon=False, fontsize=7)

    selected = comparison.set_index("metric").loc[
        ["Sharpe", "Cost-stress net IR", "Annual turnover"]
    ]
    normalized = selected.div(selected["iteration_36"], axis=0)
    normalized.T.plot.bar(ax=axes[1], color=["#175CD3", "#067647", "#B54708"])
    axes[1].set_title("Metric ratio vs Iteration 36")
    axes[1].axhline(1, color="#98A2B3", linewidth=0.8)
    axes[1].set_xlabel("")
    axes[1].legend(frameon=False, fontsize=7)

    axes[2].hist(signal_correlation, bins=30, color="#175CD3", alpha=0.85)
    axes[2].axvline(signal_correlation.median(), color="#B42318", linewidth=1.5)
    axes[2].set_title("Daily cross-sectional signal correlation")
    axes[2].set_xlabel("Spearman/Pearson cross-sectional correlation")
    axes[2].set_ylabel("Trading days")
    for axis in axes:
        axis.grid(axis="y", color="#E4E7EC", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(OUTPUT / "calendar_and_attribution.png", dpi=190, bbox_inches="tight")
    plt.close(figure)


def _plot_monthly_heatmap(monthly: pd.DataFrame) -> None:
    matrix = monthly.pivot(index="year", columns="month", values="return").reindex(
        columns=range(1, 13)
    )
    values = matrix.to_numpy() * 100
    limit = max(abs(np.nanmin(values)), abs(np.nanmax(values)))
    figure, axis = plt.subplots(figsize=(11, 2.8))
    image = axis.imshow(values, cmap="RdYlGn", aspect="auto", vmin=-limit, vmax=limit)
    axis.set_xticks(range(12), [str(month) for month in range(1, 13)])
    axis.set_yticks(range(len(matrix.index)), [str(year) for year in matrix.index])
    axis.set_xlabel("Month")
    axis.set_ylabel("Year")
    axis.set_title("36+42 Composite Monthly Return Heatmap (%)")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            if np.isfinite(values[row, column]):
                axis.text(
                    column,
                    row,
                    f"{values[row, column]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#101828",
                )
    figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    figure.tight_layout()
    figure.savefig(OUTPUT / "monthly_heatmap.png", dpi=190, bbox_inches="tight")
    plt.close(figure)


def _plot_capital_comparison(
    iteration36: pd.DataFrame, combination: pd.DataFrame
) -> None:
    curves = pd.concat(
        {
            "Iteration 36": iteration36["nav_cny"],
            "36+42 Composite": combination["nav_cny"],
        },
        axis=1,
        join="inner",
    ).dropna()
    drawdowns = curves.div(curves.cummax()) - 1
    returns = curves.pct_change(fill_method=None).dropna()
    rolling = _rolling_sharpe(returns, window=245)
    figure, axes = plt.subplots(3, 1, figsize=(12, 8.5), sharex=True)
    colors = {"Iteration 36": "#175CD3", "36+42 Composite": "#067647"}
    for name in curves:
        axes[0].plot(curves.index, curves[name], color=colors[name], label=name, linewidth=1.8)
        axes[1].plot(
            drawdowns.index,
            drawdowns[name] * 100,
            color=colors[name],
            label=name,
            linewidth=1.3,
        )
        axes[2].plot(
            rolling.index,
            rolling[name],
            color=colors[name],
            label=name,
            linewidth=1.3,
        )
    axes[0].axhline(1_000_000, color="#98A2B3", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("Portfolio NAV (CNY)")
    axes[0].legend(frameon=False)
    axes[1].set_ylabel("Drawdown (%)")
    axes[2].set_ylabel("Rolling 245d Sharpe")
    axes[2].axhline(0, color="#98A2B3", linewidth=0.8)
    for axis in axes:
        axis.grid(axis="y", color="#E4E7EC", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Comparable Capital Replay: CNY 1m, 50% Target Exposure")
    figure.tight_layout()
    figure.savefig(OUTPUT / "capital_replay_comparison.png", dpi=190, bbox_inches="tight")
    plt.close(figure)


def _annualized_ir(values: pd.Series) -> float:
    return float(values.mean() / values.std(ddof=1) * math.sqrt(TRADING_DAYS))


def _two_sided_p(t_stat: float) -> float:
    return float(math.erfc(abs(t_stat) / math.sqrt(2)))


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return [_json_ready(item) for item in value.to_dict("records")]
    if isinstance(value, pd.Series):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value


if __name__ == "__main__":
    main()
