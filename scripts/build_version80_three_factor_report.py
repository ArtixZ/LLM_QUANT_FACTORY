from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from autoalpha.data.current_panel import inspect_current_panel
from autoalpha.dsl.expression import Expression, FactorDefinition
from autoalpha.research.evaluation import cross_sectional_ic
from autoalpha.service.evaluator import PriceVolumeEvaluator, _walk_forward_dates
from autoalpha.service.multifactor import _absolute_portfolio_gate_failures
from autoalpha.service.store import ServiceStore

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATABASE = ROOT / "runtime/autoalpha.sqlite3"
DATA = ROOT.parent / "data"
PANEL = DATA / "processed/daily_panel"
CONFIG = ROOT / "config/research.toml"
OUTPUT = ROOT / "output/pdf/version80_three_factor_research"
VERSION = 80
TRADING_DAYS = 245

DISPLAY_NAMES = {
    "ForceIndex_Reversal_20d": "Force Reversal",
    "Dollar_Volume_Stability_20d": "Dollar Volume Stability",
    "Volume_Coefficient_Variation_20d": "Volume CV Stability",
}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    store = ServiceStore(DATABASE)
    state = store.state()
    version = _portfolio_version(store, VERSION)
    members = _portfolio_members(store, version)
    factors = [item["factor"] for item in members]
    weights = [item["weight"] for item in members]
    generation = store.generation_state("institutional_v3_walkforward_20260715")
    trials = int(generation["candidate_attempts"]) if generation else 1

    evaluator = PriceVolumeEvaluator(DATA, CONFIG)
    evaluator.set_trial_count(trials)
    factor_evaluations = {
        factor.factor_id: evaluator.evaluate(factor) for factor in factors
    }
    portfolio = evaluator.evaluate_portfolio(factors, weights=weights)
    portfolio_failures = _absolute_portfolio_gate_failures(
        portfolio.metrics, evaluator.config
    )

    paths = {
        DISPLAY_NAMES[factor.name]: _public_path(evaluator, [factor], [1.0])
        for factor in factors
    }
    paths["VERSION 80 Composite"] = _public_path(evaluator, factors, weights)
    returns = pd.concat(
        {name: path["net"] for name, path in paths.items()}, axis=1, join="inner"
    ).dropna()
    stressed = paths["VERSION 80 Composite"]["stressed"].reindex(returns.index)
    turnover = paths["VERSION 80 Composite"]["turnover"].reindex(returns.index)

    signals = {factor.factor_id: evaluator._factor_signal(factor) for factor in factors}  # noqa: SLF001
    composite_signal = signals[factors[0].factor_id] * weights[0]
    for factor, weight in zip(factors[1:], weights[1:], strict=True):
        composite_signal = composite_signal + signals[factor.factor_id] * weight
    public_dates = _walk_forward_dates(composite_signal.index, evaluator.config)
    composite_signal = composite_signal.loc[public_dates]
    fields = evaluator._load_fields()  # noqa: SLF001

    yearly = _yearly_returns(returns)
    monthly = _monthly_returns(returns["VERSION 80 Composite"])
    ic_decay = _ic_decay(factors, signals, composite_signal, fields["adj_close"])
    correlation_matrix, correlation_series = _signal_correlations(
        factors, signals, public_dates
    )
    leave_one_out = _leave_one_out(evaluator, factors, weights, portfolio.metrics)
    bootstrap = _moving_block_bootstrap(returns["VERSION 80 Composite"])
    migration = _migration_verdicts()
    readiness = inspect_current_panel(PANEL).to_dict()

    factor_rows = []
    for item in members:
        factor = item["factor"]
        metrics = factor_evaluations[factor.factor_id].metrics
        factor_rows.append(
            {
                "factor_id": factor.factor_id,
                "iteration": item["source_iteration"],
                "name": factor.name,
                "display_name": DISPLAY_NAMES[factor.name],
                "family": factor.family,
                "weight": item["weight"],
                "expected_direction": factor.expected_direction,
                "hypothesis": factor.hypothesis,
                "expression": factor.expression.to_dict(),
                "metrics": metrics,
            }
        )

    snapshot = {
        "report_as_of": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "report_scope": "public adaptive research evidence through 2024 only",
        "service_state_at_extraction": state,
        "generation_at_extraction": generation,
        "source_portfolio": version,
        "data_workspace": evaluator.workspace.to_dict(),
        "data_readiness": readiness,
        "factors": factor_rows,
        "public_v3_portfolio": {
            "weights": {
                factor.factor_id: weight
                for factor, weight in zip(factors, weights, strict=True)
            },
            "metrics": portfolio.metrics,
            "absolute_gate_failures": portfolio_failures,
            "bootstrap": bootstrap,
            "stressed_simple_annual_return": float(stressed.mean() * TRADING_DAYS),
            "stressed_sharpe": _annualized_sharpe(stressed),
            "average_daily_one_way_turnover": float(turnover.mean()),
        },
        "leave_one_out": leave_one_out.to_dict("records"),
        "ic_decay": ic_decay.to_dict("records"),
        "signal_correlation_matrix": correlation_matrix.to_dict(),
        "migration_smoke": migration,
        "governance": {
            "formal_v3_blind_record_exists": False,
            "hidden_metrics_in_report": False,
            "migration_smoke_is_admissible_for_promotion": False,
            "production_status": "RESEARCH_ONLY_DATA_BLOCKED",
        },
    }

    (OUTPUT / "research_snapshot.json").write_text(
        json.dumps(_json_ready(snapshot), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _factor_metric_table(factor_rows, portfolio.metrics).to_csv(
        OUTPUT / "public_metric_comparison.csv", index=False
    )
    _fold_table(factor_rows, portfolio.metrics).to_csv(
        OUTPUT / "walk_forward_folds.csv", index=False
    )
    yearly.to_csv(OUTPUT / "yearly_returns.csv")
    monthly.to_csv(OUTPUT / "monthly_returns.csv", index=False)
    ic_decay.to_csv(OUTPUT / "ic_decay.csv", index=False)
    correlation_matrix.to_csv(OUTPUT / "signal_correlation_matrix.csv")
    correlation_series.to_csv(OUTPUT / "daily_signal_correlations.csv")
    leave_one_out.to_csv(OUTPUT / "leave_one_out.csv", index=False)
    pd.DataFrame(
        {
            "trade_date": returns.index,
            **{name: returns[name].to_numpy() for name in returns.columns},
        }
    ).to_csv(OUTPUT / "public_daily_returns.csv", index=False)

    _plot_performance(returns)
    _plot_walk_forward(factor_rows, portfolio.metrics)
    _plot_correlations(correlation_matrix, correlation_series)
    _plot_ic_decay(ic_decay)
    _plot_leave_one_out(leave_one_out)
    _plot_monthly_heatmap(monthly)
    _plot_protocol()
    print(json.dumps(_json_ready(snapshot), ensure_ascii=False, indent=2))


def _portfolio_version(store: ServiceStore, version_id: int) -> dict[str, Any]:
    version = next(
        (item for item in store.portfolio_history(limit=1000) if item["id"] == version_id),
        None,
    )
    if version is None or not version["accepted"]:
        raise ValueError(f"Accepted portfolio version not found: {version_id}")
    return version


def _portfolio_members(
    store: ServiceStore, version: dict[str, Any]
) -> list[dict[str, Any]]:
    members = []
    for member in version["members"]:
        record = store.factor_pool_record(str(member["factor_id"]))
        if record is None:
            raise RuntimeError(f"Missing factor pool record: {member['factor_id']}")
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
    total = sum(item["weight"] for item in members)
    for item in members:
        item["weight"] /= total
    return members


def _public_path(
    evaluator: PriceVolumeEvaluator,
    factors: list[FactorDefinition],
    weights: list[float],
) -> pd.DataFrame:
    path = evaluator._portfolio_path(factors, weights)  # noqa: SLF001
    dates = _walk_forward_dates(path.index, evaluator.config)
    selected = path.loc[dates].copy()
    selected.attrs.update(path.attrs)
    return selected


def _yearly_returns(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.groupby(returns.index.year).apply(
        lambda values: (1 + values).prod() - 1,
        include_groups=False,
    )


def _monthly_returns(values: pd.Series) -> pd.DataFrame:
    monthly = values.groupby(values.index.to_period("M")).apply(
        lambda period: float((1 + period).prod() - 1)
    )
    return pd.DataFrame(
        {
            "year": monthly.index.year,
            "month": monthly.index.month,
            "return": monthly.to_numpy(),
        }
    )


def _ic_decay(
    factors: list[FactorDefinition],
    signals: dict[str, pd.DataFrame],
    composite_signal: pd.DataFrame,
    close: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    named_signals = {
        DISPLAY_NAMES[factor.name]: signals[factor.factor_id].reindex(composite_signal.index)
        for factor in factors
    }
    named_signals["VERSION 80 Composite"] = composite_signal
    for horizon in (1, 5, 10, 20):
        forward = close.shift(-horizon).div(close).sub(1).reindex(composite_signal.index)
        for name, signal in named_signals.items():
            rank_ic = cross_sectional_ic(signal, forward, minimum_names=30)
            rows.append(
                {
                    "horizon_days": horizon,
                    "series": name,
                    "rank_ic_mean": float(rank_ic.mean()),
                    "rank_ic_ir": _annualized_sharpe(rank_ic),
                    "positive_fraction": float((rank_ic > 0).mean()),
                    "observations": len(rank_ic),
                }
            )
    return pd.DataFrame(rows)


def _signal_correlations(
    factors: list[FactorDefinition],
    signals: dict[str, pd.DataFrame],
    dates: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = [DISPLAY_NAMES[factor.name] for factor in factors]
    matrix = pd.DataFrame(np.eye(len(factors)), index=labels, columns=labels)
    series: dict[str, pd.Series] = {}
    for left_index, left in enumerate(factors):
        for right_index in range(left_index + 1, len(factors)):
            right = factors[right_index]
            daily = (
                signals[left.factor_id]
                .reindex(dates)
                .corrwith(signals[right.factor_id].reindex(dates), axis=1)
                .dropna()
            )
            left_label = DISPLAY_NAMES[left.name]
            right_label = DISPLAY_NAMES[right.name]
            matrix.loc[left_label, right_label] = float(daily.median())
            matrix.loc[right_label, left_label] = float(daily.median())
            series[f"{left_label} / {right_label}"] = daily
    return matrix, pd.DataFrame(series)


def _leave_one_out(
    evaluator: PriceVolumeEvaluator,
    factors: list[FactorDefinition],
    weights: list[float],
    full_metrics: dict[str, Any],
) -> pd.DataFrame:
    rows = [
        {
            "portfolio": "Full VERSION 80",
            "removed": "None",
            **_portfolio_metric_row(full_metrics),
            "sharpe_delta_vs_full": 0.0,
            "annual_return_delta_vs_full": 0.0,
        }
    ]
    for removed_index, removed in enumerate(factors):
        kept_factors = [factor for index, factor in enumerate(factors) if index != removed_index]
        kept_weights = [weight for index, weight in enumerate(weights) if index != removed_index]
        total = sum(kept_weights)
        kept_weights = [weight / total for weight in kept_weights]
        evaluation = evaluator.evaluate_portfolio(kept_factors, weights=kept_weights)
        metric_row = _portfolio_metric_row(evaluation.metrics)
        rows.append(
            {
                "portfolio": " + ".join(DISPLAY_NAMES[factor.name] for factor in kept_factors),
                "removed": DISPLAY_NAMES[removed.name],
                **metric_row,
                "sharpe_delta_vs_full": (
                    metric_row["sharpe"] - full_metrics["portfolio_sharpe_ratio"]
                ),
                "annual_return_delta_vs_full": (
                    metric_row["simple_annual_return"]
                    - full_metrics["portfolio_simple_annual_return"]
                ),
            }
        )
    return pd.DataFrame(rows)


def _portfolio_metric_row(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "sharpe": float(metrics["portfolio_sharpe_ratio"]),
        "simple_annual_return": float(metrics["portfolio_simple_annual_return"]),
        "max_drawdown": float(metrics["portfolio_max_drawdown"]),
        "annual_turnover": float(metrics["portfolio_annual_turnover"]),
        "worst_fold_sharpe": float(metrics["portfolio_walk_forward_worst_sharpe"]),
        "median_fold_sharpe": float(metrics["portfolio_walk_forward_median_sharpe"]),
    }


def _moving_block_bootstrap(values: pd.Series) -> dict[str, float]:
    rng = np.random.default_rng(20260716)
    array = values.dropna().to_numpy()
    block = 20
    samples = 2000
    annual_returns = []
    sharpes = []
    for _ in range(samples):
        starts = rng.integers(0, len(array) - block + 1, math.ceil(len(array) / block))
        sample = np.concatenate([array[start : start + block] for start in starts])[: len(array)]
        annual_returns.append(float(sample.mean() * TRADING_DAYS))
        std = sample.std(ddof=1)
        sharpes.append(float(sample.mean() / std * math.sqrt(TRADING_DAYS)) if std else 0.0)
    return {
        "block_days": block,
        "samples": samples,
        "simple_annual_return_ci_2_5": float(np.quantile(annual_returns, 0.025)),
        "simple_annual_return_ci_97_5": float(np.quantile(annual_returns, 0.975)),
        "sharpe_ci_2_5": float(np.quantile(sharpes, 0.025)),
        "sharpe_ci_97_5": float(np.quantile(sharpes, 0.975)),
    }


def _migration_verdicts() -> dict[str, Any]:
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    row = connection.execute(
        "SELECT payload_json FROM events WHERE event='EVALUATION_PROTOCOL_V3_DEPLOYED' "
        "ORDER BY id DESC LIMIT 1"
    ).fetchone()
    connection.close()
    if row is None:
        return {"available": False}
    payload = json.loads(row["payload_json"])
    return {"available": True, **payload.get("version80_migration_smoke", {})}


def _factor_metric_table(
    factor_rows: list[dict[str, Any]], portfolio_metrics: dict[str, Any]
) -> pd.DataFrame:
    rows = []
    for factor in factor_rows:
        metrics = factor["metrics"]
        rows.append(
            {
                "series": factor["display_name"],
                "weight": factor["weight"],
                "sharpe": metrics["sharpe_ratio"],
                "simple_annual_return": metrics["simple_annual_return"],
                "max_drawdown": metrics["incremental_max_drawdown"],
                "annual_turnover": metrics["annual_turnover"],
                "rank_ic_mean": metrics["rank_ic_mean"],
                "rank_ic_ir": metrics["rank_ic_ir"],
                "worst_fold_sharpe": metrics["walk_forward_worst_sharpe"],
                "parameter_worst_sharpe": metrics["parameter_stability_worst_sharpe"],
                "gate_failures": ", ".join(metrics["exploratory_gate_failures"]),
            }
        )
    rows.append(
        {
            "series": "VERSION 80 Composite",
            "weight": 1.0,
            "sharpe": portfolio_metrics["portfolio_sharpe_ratio"],
            "simple_annual_return": portfolio_metrics["portfolio_simple_annual_return"],
            "max_drawdown": portfolio_metrics["portfolio_max_drawdown"],
            "annual_turnover": portfolio_metrics["portfolio_annual_turnover"],
            "rank_ic_mean": np.nan,
            "rank_ic_ir": np.nan,
            "worst_fold_sharpe": portfolio_metrics["portfolio_walk_forward_worst_sharpe"],
            "parameter_worst_sharpe": np.nan,
            "gate_failures": "",
        }
    )
    return pd.DataFrame(rows)


def _fold_table(
    factor_rows: list[dict[str, Any]], portfolio_metrics: dict[str, Any]
) -> pd.DataFrame:
    rows = []
    for factor in factor_rows:
        for fold in factor["metrics"]["walk_forward_folds"]:
            rows.append({"series": factor["display_name"], **fold})
    for fold in portfolio_metrics["portfolio_walk_forward_folds"]:
        rows.append({"series": "VERSION 80 Composite", **fold})
    return pd.DataFrame(rows)


def _plot_performance(returns: pd.DataFrame) -> None:
    wealth = (1 + returns).cumprod()
    composite = wealth["VERSION 80 Composite"]
    drawdown = composite.div(composite.cummax()).sub(1)
    rolling = returns["VERSION 80 Composite"].rolling(120)
    rolling_sharpe = rolling.mean().div(rolling.std()).mul(math.sqrt(TRADING_DAYS))
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 9.2), sharex=True)
    colors = ["#175CD3", "#067647", "#B54708", "#101828"]
    for column, color in zip(wealth.columns, colors, strict=True):
        axes[0].plot(wealth.index, wealth[column], label=column, color=color, linewidth=1.5)
    axes[0].set_ylabel("Research NAV")
    axes[0].legend(ncol=2, frameon=False, fontsize=9)
    axes[0].set_title("Public Walk-Forward Research Performance (2015-2024)")
    axes[1].fill_between(drawdown.index, drawdown, 0, color="#B42318", alpha=0.28)
    axes[1].plot(drawdown.index, drawdown, color="#B42318", linewidth=1.0)
    axes[1].set_ylabel("Composite drawdown")
    axes[2].plot(rolling_sharpe.index, rolling_sharpe, color="#175CD3", linewidth=1.2)
    axes[2].axhline(0, color="#98A2B3", linewidth=0.8)
    axes[2].set_ylabel("120d rolling Sharpe")
    axes[2].set_xlabel("Trade date")
    _finish_figure(fig, OUTPUT / "public_performance.png")


def _plot_walk_forward(
    factor_rows: list[dict[str, Any]], portfolio_metrics: dict[str, Any]
) -> None:
    folds = _fold_table(factor_rows, portfolio_metrics)
    composite = folds[folds["series"] == "VERSION 80 Composite"].copy()
    validation_years = composite["validation_start"].str[:4]
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.4), sharex=True)
    colors = ["#067647" if value >= 0 else "#B42318" for value in composite["annual_return"]]
    axes[0].bar(validation_years, composite["annual_return"] * 100, color=colors)
    axes[0].axhline(0, color="#667085", linewidth=0.8)
    axes[0].set_ylabel("Annual return (%)")
    axes[0].set_title("VERSION 80 Annual Walk-Forward Folds")
    axes[1].plot(
        validation_years,
        composite["sharpe"],
        marker="o",
        color="#175CD3",
        linewidth=1.6,
    )
    axes[1].axhline(0, color="#667085", linewidth=0.8)
    axes[1].set_ylabel("Fold Sharpe")
    axes[1].set_xlabel("Validation year")
    _finish_figure(fig, OUTPUT / "walk_forward_folds.png")


def _plot_correlations(matrix: pd.DataFrame, daily: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [0.8, 1.5]})
    image = axes[0].imshow(matrix.to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r")
    axes[0].set_xticks(range(len(matrix.columns)), matrix.columns, rotation=30, ha="right")
    axes[0].set_yticks(range(len(matrix.index)), matrix.index)
    for row in range(len(matrix.index)):
        for column in range(len(matrix.columns)):
            axes[0].text(column, row, f"{matrix.iloc[row, column]:.3f}", ha="center", va="center")
    axes[0].set_title("Median Cross-Sectional Signal Correlation")
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
    rolling = daily.rolling(120).median()
    for column in rolling.columns:
        axes[1].plot(rolling.index, rolling[column], label=column, linewidth=1.15)
    axes[1].axhline(0, color="#98A2B3", linewidth=0.8)
    axes[1].set_title("120d Rolling Median Correlation")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].set_ylim(-0.2, 0.8)
    _finish_figure(fig, OUTPUT / "signal_correlations.png")


def _plot_ic_decay(ic_decay: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    for name, group in ic_decay.groupby("series"):
        ax.plot(group["horizon_days"], group["rank_ic_mean"], marker="o", label=name)
    ax.axhline(0, color="#98A2B3", linewidth=0.8)
    ax.set_xticks([1, 5, 10, 20])
    ax.set_xlabel("Forward horizon (trading days)")
    ax.set_ylabel("Mean Rank IC")
    ax.set_title("Public Rank IC Decay")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    _finish_figure(fig, OUTPUT / "ic_decay.png")


def _plot_leave_one_out(values: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.7))
    labels = ["Full", "No Force", "No DollarVol", "No VolumeCV"]
    axes[0].bar(labels, values["sharpe"], color="#175CD3")
    axes[0].set_title("Sharpe")
    axes[1].bar(labels, values["simple_annual_return"] * 100, color="#067647")
    axes[1].set_title("Simple annual return (%)")
    axes[2].bar(labels, values["max_drawdown"] * 100, color="#B42318")
    axes[2].set_title("Maximum drawdown (%)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=28)
        ax.axhline(0, color="#98A2B3", linewidth=0.8)
    fig.suptitle("Leave-One-Factor-Out Attribution")
    _finish_figure(fig, OUTPUT / "leave_one_out.png")


def _plot_monthly_heatmap(monthly: pd.DataFrame) -> None:
    pivot = monthly.pivot(index="year", columns="month", values="return").reindex(
        columns=range(1, 13)
    )
    fig, ax = plt.subplots(figsize=(11.3, 5.4))
    image = ax.imshow(pivot.to_numpy() * 100, aspect="auto", cmap="RdYlGn", vmin=-6, vmax=6)
    ax.set_xticks(
        range(12),
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    )
    ax.set_yticks(range(len(pivot.index)), pivot.index.astype(str))
    ax.set_title("VERSION 80 Public Monthly Return Heatmap (%)")
    for row in range(len(pivot.index)):
        for column in range(12):
            value = pivot.iloc[row, column]
            if pd.notna(value):
                ax.text(column, row, f"{value * 100:.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    _finish_figure(fig, OUTPUT / "monthly_heatmap.png")


def _plot_protocol() -> None:
    stages = [
        ("Exploration", "2010-2017", "#EAF2FF"),
        ("Public WF", "5Y -> 1Y\n2015-2024", "#E8F5EE"),
        ("Frozen", "Candidate + weights", "#FFF4E5"),
        ("Blind", "Categorical only", "#F2EAFE"),
        ("Capital", "Execution gates", "#FDECEC"),
    ]
    fig, ax = plt.subplots(figsize=(11.5, 2.6))
    ax.set_xlim(0, len(stages) * 2.2)
    ax.set_ylim(0, 2)
    ax.axis("off")
    for index, (title, subtitle, color) in enumerate(stages):
        x = index * 2.2 + 0.15
        rectangle = plt.Rectangle(
            (x, 0.55),
            1.75,
            0.9,
            facecolor=color,
            edgecolor="#98A2B3",
            linewidth=1,
        )
        ax.add_patch(rectangle)
        ax.text(x + 0.875, 1.12, title, ha="center", va="center", weight="bold", fontsize=10)
        ax.text(x + 0.875, 0.78, subtitle, ha="center", va="center", fontsize=8, color="#475467")
        if index < len(stages) - 1:
            ax.annotate(
                "",
                xy=(x + 2.1, 1),
                xytext=(x + 1.78, 1),
                arrowprops={"arrowstyle": "->", "color": "#667085"},
            )
    ax.set_title("Institutional Walk-Forward v3 Evidence Boundary", pad=4)
    _finish_figure(fig, OUTPUT / "protocol_flow.png")


def _finish_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=210, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _annualized_sharpe(values: pd.Series) -> float:
    clean = values.dropna()
    std = clean.std(ddof=1)
    return float(clean.mean() / std * math.sqrt(TRADING_DAYS)) if std else 0.0


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp, pd.Period)):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    main()
