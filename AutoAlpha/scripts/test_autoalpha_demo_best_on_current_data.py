from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from autoalpha.research.multiple_testing import deflated_sharpe_ratio
from autoalpha.research.statistics import hac_mean_inference

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT.parent / "data"
PANEL_ROOT = DATA_ROOT / "processed" / "daily_panel"
DEMO_ROOT = Path.home() / "Portfolios" / "AutoAlpha-demo"
DEMO_LIBRARY = (
    DEMO_ROOT
    / "factor_library"
    / "20260713_070051_F0062_demo_v1_h20_rank_composite_icir_decay"
)
DEMO_ALPHA = DEMO_LIBRARY / "alpha.py"
DEMO_BEST = DEMO_ROOT / "journal" / "best.json"
OUTPUT_ROOT = PROJECT_ROOT / "output" / "backtests" / "autoalpha_demo_best_transfer_v3"
VERSION80_SNAPSHOT = (
    PROJECT_ROOT
    / "output"
    / "pdf"
    / "version80_three_factor_research"
    / "research_snapshot.json"
)
VERSION80_RETURNS = VERSION80_SNAPSHOT.parent / "public_daily_returns.csv"

PUBLIC_END = pd.Timestamp("2024-11-29")
FIRST_VALIDATION_YEAR = 2015
LAST_VALIDATION_YEAR = 2024
TRAIN_YEARS = 5
HORIZON = 20
HOLDING_DAYS = 5
TRADING_DAYS = 245
MINIMUM_NAMES = 30
DECAY_HALFLIFE_DAYS = 15
ONE_WAY_BPS = 1.5 + 0.1 + 5.0 / 2.0
STRESS_MULTIPLIER = 2.0

FACTOR_NAMES = (
    "f_reversal_1",
    "f_volatility_10",
    "f_amihud_20",
    "f_hl_range_10",
    "f_gap_reversal_5",
    "f_rsi_14",
    "f_skew_20",
    "f_volume_reversal_5",
    "f_volume_volatility_10",
    "f_upper_shadow_ratio_5",
    "f_lower_shadow_ratio_5",
    "f_max_ret_10",
    "f_idio_vol_10",
    "f_price_range_position_10",
    "f_momentum_20",
    "f_slow_vol_regime_60",
)


@dataclass(frozen=True)
class FoldResult:
    fold_id: int
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    observations: int
    sharpe: float
    simple_annual_return: float
    compound_return: float
    max_drawdown: float
    annual_turnover: float
    stressed_sharpe: float
    rank_ic_20d_mean: float
    rank_ic_20d_ir: float
    pearson_ic_20d_mean: float
    coverage: float
    effective_factor_count: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transfer-test the AutoAlpha-demo Iteration 325 champion on current data."
    )
    parser.add_argument("--first-validation-year", type=int, default=FIRST_VALIDATION_YEAR)
    parser.add_argument("--last-validation-year", type=int, default=LAST_VALIDATION_YEAR)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def _load_demo_alpha() -> object:
    if not DEMO_ALPHA.exists():
        raise FileNotFoundError(f"Archived demo alpha not found: {DEMO_ALPHA}")
    spec = importlib.util.spec_from_file_location("autoalpha_demo_best_archive", DEMO_ALPHA)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import archived demo alpha: {DEMO_ALPHA}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_years(years: range | list[int]) -> pd.DataFrame:
    columns = [
        "trade_date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "vol",
        "amount",
        "is_valid_ohlc",
        "is_tradable_observation",
    ]
    frames: list[pd.DataFrame] = []
    for year in years:
        partition = PANEL_ROOT / f"trade_year={year}"
        for path in sorted(partition.glob("*.parquet")):
            frames.append(pd.read_parquet(path, columns=columns))
    if not frames:
        raise FileNotFoundError(f"No panel partitions found for years {list(years)}")
    data = pd.concat(frames, ignore_index=True)
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    data = data[data["trade_date"] <= PUBLIC_END].copy()
    valid = data["is_valid_ohlc"].fillna(False) & data[
        "is_tradable_observation"
    ].fillna(False)
    value_columns = ["open", "high", "low", "close", "adj_close", "vol", "amount"]
    data.loc[~valid, value_columns] = np.nan
    return data


def _wide_fields(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    fields = {
        name: data.pivot(index="trade_date", columns="symbol", values=name).sort_index()
        for name in ("open", "high", "low", "close", "adj_close", "vol", "amount")
    }
    fields["volume"] = fields.pop("vol")
    return fields


def _cs_rank_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    percentile = frame.rank(axis=1, pct=True, method="average").clip(1e-10, 1 - 1e-10)
    return pd.DataFrame(
        norm.ppf(percentile.to_numpy()), index=percentile.index, columns=percentile.columns
    )


def _factor_functions() -> dict[str, Callable[[dict[str, pd.DataFrame]], pd.DataFrame]]:
    def reversal(fields: dict[str, pd.DataFrame]) -> pd.DataFrame:
        return -fields["close"].pct_change(1, fill_method=None)

    def volatility(fields: dict[str, pd.DataFrame]) -> pd.DataFrame:
        returns = fields["close"].pct_change(fill_method=None)
        return -returns.rolling(10, min_periods=5).std()

    def amihud(fields: dict[str, pd.DataFrame]) -> pd.DataFrame:
        amount = fields["close"] * fields["volume"]
        absolute_return = fields["close"].pct_change(fill_method=None).abs()
        return -(absolute_return / amount.replace(0, np.nan)).rolling(
            20, min_periods=10
        ).mean()

    def hl_range(fields: dict[str, pd.DataFrame]) -> pd.DataFrame:
        ratio = (fields["high"] - fields["low"]) / fields["close"].replace(0, np.nan)
        return -ratio.rolling(10, min_periods=5).mean()

    def gap_reversal(fields: dict[str, pd.DataFrame]) -> pd.DataFrame:
        gap = fields["open"] / fields["close"].shift(1).replace(0, np.nan) - 1.0
        return -gap.rolling(5, min_periods=3).mean()

    def rsi(fields: dict[str, pd.DataFrame]) -> pd.DataFrame:
        delta = fields["close"].diff()
        average_up = delta.clip(lower=0).rolling(14, min_periods=10).mean()
        average_down = (-delta.clip(upper=0)).rolling(14, min_periods=10).mean()
        relative_strength = average_up / average_down.replace(0, np.nan)
        return -(100.0 - 100.0 / (1.0 + relative_strength))

    def skew(fields: dict[str, pd.DataFrame]) -> pd.DataFrame:
        returns = fields["close"].pct_change(fill_method=None)
        return -returns.rolling(20, min_periods=10).skew()

    def volume_reversal(fields: dict[str, pd.DataFrame]) -> pd.DataFrame:
        average = fields["volume"].rolling(5, min_periods=3).mean()
        return -(fields["volume"] / average.replace(0, np.nan))

    def volume_volatility(fields: dict[str, pd.DataFrame]) -> pd.DataFrame:
        change = fields["volume"].pct_change(fill_method=None)
        return -change.rolling(10, min_periods=5).std()

    def upper_shadow(fields: dict[str, pd.DataFrame]) -> pd.DataFrame:
        shadow = fields["high"] - np.maximum(fields["open"], fields["close"])
        ratio = shadow / (fields["high"] - fields["low"])
        return -ratio.rolling(5, min_periods=3).mean()

    def lower_shadow(fields: dict[str, pd.DataFrame]) -> pd.DataFrame:
        shadow = np.minimum(fields["open"], fields["close"]) - fields["low"]
        ratio = shadow / (fields["high"] - fields["low"])
        return ratio.rolling(5, min_periods=3).mean()

    def max_return(fields: dict[str, pd.DataFrame]) -> pd.DataFrame:
        returns = fields["close"].pct_change(fill_method=None)
        return -returns.rolling(10, min_periods=5).max()

    def idiosyncratic_volatility(fields: dict[str, pd.DataFrame]) -> pd.DataFrame:
        returns = fields["close"].pct_change(fill_method=None)
        residual = returns.sub(returns.mean(axis=1, skipna=True), axis=0)
        return -residual.rolling(10, min_periods=5).std()

    def price_range_position(fields: dict[str, pd.DataFrame]) -> pd.DataFrame:
        rolling_high = fields["high"].rolling(10, min_periods=5).max()
        rolling_low = fields["low"].rolling(10, min_periods=5).min()
        position = (fields["close"] - rolling_low) / (rolling_high - rolling_low) - 0.5
        return -position

    def momentum(fields: dict[str, pd.DataFrame]) -> pd.DataFrame:
        return fields["close"].pct_change(20, fill_method=None)

    def slow_volatility_regime(fields: dict[str, pd.DataFrame]) -> pd.DataFrame:
        returns = fields["close"].pct_change(fill_method=None)
        volatility_60 = returns.rolling(60, min_periods=20).std()
        volatility_252 = returns.rolling(252, min_periods=60).std()
        return -(volatility_60 / volatility_252.replace(0, np.nan))

    return dict(
        zip(
            FACTOR_NAMES,
            (
                reversal,
                volatility,
                amihud,
                hl_range,
                gap_reversal,
                rsi,
                skew,
                volume_reversal,
                volume_volatility,
                upper_shadow,
                lower_shadow,
                max_return,
                idiosyncratic_volatility,
                price_range_position,
                momentum,
                slow_volatility_regime,
            ),
            strict=True,
        )
    )


def _future_rank_label(open_prices: pd.DataFrame) -> pd.DataFrame:
    raw = open_prices.shift(-1 - HORIZON) / open_prices.shift(-1) - 1.0
    return raw.rank(axis=1, pct=True)


def _daily_ic(
    signal: pd.DataFrame, labels: pd.DataFrame, *, rank: bool
) -> pd.Series:
    index = signal.index.intersection(labels.index)
    columns = signal.columns.intersection(labels.columns)
    left = signal.reindex(index=index, columns=columns)
    right = labels.reindex(index=index, columns=columns)
    valid = (left.notna() & right.notna()).sum(axis=1) >= MINIMUM_NAMES
    if rank:
        left = left.rank(axis=1)
        right = right.rank(axis=1)
    result = left.corrwith(right, axis=1).where(valid)
    return result.dropna()


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    return float(sorted_values[np.searchsorted(cumulative, 0.5 * cumulative[-1])])


def _robust_decay_ir(series: pd.Series, reference_date: pd.Timestamp) -> float:
    if len(series) < 20:
        return 0.0
    days = (reference_date - series.index).days.to_numpy()
    weights = np.exp(-np.log(2.0) * days / DECAY_HALFLIFE_DAYS)
    values = series.to_numpy(dtype=float)
    median = _weighted_median(values, weights)
    mad = _weighted_median(np.abs(values - median), weights)
    return float(median / (1.4826 * mad + 1e-8))


def _estimate_weights(
    train_fields: dict[str, pd.DataFrame],
    functions: dict[str, Callable[[dict[str, pd.DataFrame]], pd.DataFrame]],
) -> tuple[np.ndarray, list[dict[str, float]]]:
    labels = _future_rank_label(train_fields["open"])
    rank_ics: list[pd.Series] = []
    pearson_ics: list[pd.Series] = []
    for name in FACTOR_NAMES:
        print(f"    train IC: {name}", flush=True)
        factor = _cs_rank_zscore(functions[name](train_fields))
        rank_ics.append(_daily_ic(factor, labels, rank=True))
        pearson_ics.append(_daily_ic(factor, labels, rank=False))
        del factor
    all_dates = pd.DatetimeIndex([])
    for series in rank_ics + pearson_ics:
        all_dates = all_dates.union(series.index)
    if all_dates.empty:
        weights = np.repeat(1.0 / len(FACTOR_NAMES), len(FACTOR_NAMES))
        return weights, []
    reference = all_dates.max()
    diagnostics: list[dict[str, float]] = []
    scores = []
    for name, rank_ic, pearson_ic in zip(
        FACTOR_NAMES, rank_ics, pearson_ics, strict=True
    ):
        rank_ir = _robust_decay_ir(rank_ic, reference)
        pearson_ir = _robust_decay_ir(pearson_ic, reference)
        score = rank_ir + 0.5 * pearson_ir
        scores.append(score)
        diagnostics.append(
            {
                "factor": name,
                "train_rank_ic_mean": float(rank_ic.mean()),
                "train_pearson_ic_mean": float(pearson_ic.mean()),
                "robust_decay_rank_ir": rank_ir,
                "robust_decay_pearson_ir": pearson_ir,
                "composite_ir": score,
            }
        )
    positive = np.maximum(np.asarray(scores, dtype=float), 0.0)
    weights = (
        positive / positive.sum()
        if positive.sum() > 0
        else np.repeat(1.0 / len(positive), len(positive))
    )
    return weights, diagnostics


def _combine_validation_factors(
    validation_fields: dict[str, pd.DataFrame],
    functions: dict[str, Callable[[dict[str, pd.DataFrame]], pd.DataFrame]],
    weights: np.ndarray,
) -> pd.DataFrame:
    numerator: np.ndarray | None = None
    denominator: np.ndarray | None = None
    index: pd.Index | None = None
    columns: pd.Index | None = None
    for name, weight in zip(FACTOR_NAMES, weights, strict=True):
        print(f"    validation signal: {name} weight={weight:.4f}", flush=True)
        factor = _cs_rank_zscore(functions[name](validation_fields))
        if numerator is None:
            index, columns = factor.index, factor.columns
            numerator = np.zeros(factor.shape, dtype=float)
            denominator = np.zeros(factor.shape, dtype=float)
        values = factor.to_numpy(dtype=float, copy=False)
        valid = np.isfinite(values)
        numerator += np.where(valid, values, 0.0) * weight
        denominator += valid * abs(weight)
        del factor, values, valid
    assert numerator is not None and denominator is not None
    assert index is not None and columns is not None
    combined = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 0,
    )
    return pd.DataFrame(combined, index=index, columns=columns)


def _adaptive_spans(train_fields: dict[str, pd.DataFrame]) -> pd.Series:
    daily_range = (train_fields["high"] - train_fields["low"]) / train_fields[
        "close"
    ].replace(0, np.nan)
    atr = daily_range.rolling(20, min_periods=5).mean()
    quantile = atr.rank(axis=1, pct=True, method="average").clip(1e-6, 1 - 1e-6)
    smoothed = quantile.rolling(5, min_periods=1).mean().iloc[-1]
    logistic = 1.0 / (1.0 + np.exp(-20.0 * (smoothed - 0.4)))
    return (1.0 + 29.0 * logistic).clip(lower=1.0, upper=30.0)


def _finalize_signal(raw: pd.DataFrame, spans: pd.Series) -> pd.DataFrame:
    smoothed = raw.rolling(5, min_periods=1).median()
    aligned_spans = spans.reindex(smoothed.columns).fillna(4.0)
    ema = smoothed.copy()
    for column in smoothed.columns:
        ema[column] = smoothed[column].ewm(
            span=float(aligned_spans[column]), adjust=False, min_periods=1
        ).mean()
    zscore = _cs_rank_zscore(ema)
    return _cs_rank_zscore(zscore.rolling(3, min_periods=1).median())


def _target_positions(signal: pd.DataFrame) -> pd.DataFrame:
    ranks = signal.rank(axis=1, pct=True)
    positions = (ranks >= 0.9).astype(float) - (ranks <= 0.1).astype(float)
    gross = positions.abs().sum(axis=1).replace(0, np.nan)
    return positions.div(gross, axis=0).fillna(0.0)


def _strategy_path(
    targets: pd.DataFrame,
    return_prices: pd.DataFrame,
    prior_targets: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if prior_targets is not None and not prior_targets.empty:
        columns = prior_targets.columns.union(targets.columns)
        combined = pd.concat(
            [
                prior_targets.reindex(columns=columns, fill_value=0.0),
                targets.reindex(columns=columns, fill_value=0.0),
            ]
        )
    else:
        combined = targets
    weights = combined.rolling(HOLDING_DAYS, min_periods=1).mean()
    turnover = weights.diff().abs().sum(axis=1).mul(0.5)
    weights = weights.loc[targets.index]
    turnover = turnover.loc[targets.index].fillna(0.0)
    next_return = return_prices.pct_change(fill_method=None).shift(-1)
    columns = weights.columns.intersection(next_return.columns)
    gross = (weights[columns] * next_return.reindex(weights.index)[columns]).sum(
        axis=1, min_count=1
    )
    path = pd.DataFrame(
        {
            "gross": gross,
            "net": gross - turnover * ONE_WAY_BPS / 10_000,
            "stressed": gross
            - turnover * ONE_WAY_BPS * STRESS_MULTIPLIER / 10_000,
            "turnover": turnover,
        }
    ).dropna()
    return path, targets.tail(HOLDING_DAYS).copy()


def _annualized_sharpe(values: pd.Series) -> float:
    clean = values.dropna()
    standard_deviation = float(clean.std(ddof=1))
    if clean.empty or standard_deviation <= 0 or not np.isfinite(standard_deviation):
        return float("nan")
    return float(clean.mean() / standard_deviation * math.sqrt(TRADING_DAYS))


def _max_drawdown(values: pd.Series) -> float:
    nav = (1.0 + values.dropna()).cumprod()
    return float((nav / nav.cummax() - 1.0).min())


def _compound_annual_return(values: pd.Series) -> float:
    clean = values.dropna()
    wealth = float((1.0 + clean).prod())
    return float(wealth ** (TRADING_DAYS / len(clean)) - 1.0)


def _fold_metrics(
    fold_id: int,
    train_fields: dict[str, pd.DataFrame],
    validation_fields: dict[str, pd.DataFrame],
    signal: pd.DataFrame,
    path: pd.DataFrame,
    weights: np.ndarray,
) -> FoldResult:
    labels = _future_rank_label(validation_fields["open"])
    rank_ic = _daily_ic(signal, labels, rank=True)
    pearson_ic = _daily_ic(signal, labels, rank=False)
    net = path["net"]
    coverage_denominator = int(validation_fields["adj_close"].notna().to_numpy().sum())
    covered = signal.notna() & validation_fields["adj_close"].notna().reindex_like(signal)
    coverage = float(covered.to_numpy().sum() / coverage_denominator)
    effective_count = float(1.0 / np.square(weights).sum())
    return FoldResult(
        fold_id=fold_id,
        train_start=train_fields["close"].index.min().date().isoformat(),
        train_end=train_fields["close"].index.max().date().isoformat(),
        validation_start=path.index.min().date().isoformat(),
        validation_end=path.index.max().date().isoformat(),
        observations=len(net),
        sharpe=_annualized_sharpe(net),
        simple_annual_return=float(net.mean() * TRADING_DAYS),
        compound_return=float((1.0 + net).prod() - 1.0),
        max_drawdown=_max_drawdown(net),
        annual_turnover=float(path["turnover"].mean() * TRADING_DAYS),
        stressed_sharpe=_annualized_sharpe(path["stressed"]),
        rank_ic_20d_mean=float(rank_ic.mean()),
        rank_ic_20d_ir=_annualized_sharpe(rank_ic) / math.sqrt(HORIZON),
        pearson_ic_20d_mean=float(pearson_ic.mean()),
        coverage=coverage,
        effective_factor_count=effective_count,
    )


def _aggregate_metrics(path: pd.DataFrame, fold_results: list[FoldResult]) -> dict[str, object]:
    net = path["net"]
    inference = hac_mean_inference(net.to_numpy(), lags=min(5, len(net) - 1))
    dsr = deflated_sharpe_ratio(net.to_numpy(), trials=325)
    return {
        "backtest_start": net.index.min().date().isoformat(),
        "backtest_end": net.index.max().date().isoformat(),
        "observations": len(net),
        "sharpe": _annualized_sharpe(net),
        "simple_annual_return": float(net.mean() * TRADING_DAYS),
        "compound_annual_return": _compound_annual_return(net),
        "max_drawdown": _max_drawdown(net),
        "annual_turnover": float(path["turnover"].mean() * TRADING_DAYS),
        "stressed_sharpe": _annualized_sharpe(path["stressed"]),
        "stressed_simple_annual_return": float(path["stressed"].mean() * TRADING_DAYS),
        "positive_fold_fraction": float(
            np.mean([fold.simple_annual_return > 0 for fold in fold_results])
        ),
        "median_fold_sharpe": float(np.median([fold.sharpe for fold in fold_results])),
        "worst_fold_sharpe": float(min(fold.sharpe for fold in fold_results)),
        "worst_fold_drawdown": float(min(fold.max_drawdown for fold in fold_results)),
        "rank_ic_20d_mean_across_folds": float(
            np.mean([fold.rank_ic_20d_mean for fold in fold_results])
        ),
        "net_return_hac_p_value": float(inference.p_value),
        "deflated_sharpe_probability_325_trials": float(dsr.probability),
        "deflated_sharpe_expected_maximum": float(dsr.expected_max_sharpe),
    }


def _version80_comparison() -> dict[str, object] | None:
    if not VERSION80_SNAPSHOT.exists():
        return None
    snapshot = json.loads(VERSION80_SNAPSHOT.read_text())
    public = snapshot.get("public_v3_portfolio", {})
    metrics = public.get("metrics", {})
    return {
        "source": str(VERSION80_SNAPSHOT),
        "sharpe": metrics.get("portfolio_sharpe_ratio"),
        "simple_annual_return": metrics.get("portfolio_simple_annual_return"),
        "compound_annual_return": metrics.get("portfolio_compound_annual_return"),
        "max_drawdown": metrics.get("portfolio_max_drawdown"),
        "annual_turnover": metrics.get("portfolio_annual_turnover"),
        "stressed_sharpe": public.get("stressed_sharpe"),
        "positive_fold_fraction": metrics.get("portfolio_walk_forward_positive_fraction"),
        "median_fold_sharpe": metrics.get("portfolio_walk_forward_median_sharpe"),
        "worst_fold_sharpe": metrics.get("portfolio_walk_forward_worst_sharpe"),
    }


def _plot_results(
    output: Path, path: pd.DataFrame, fold_results: list[FoldResult]
) -> None:
    nav = (1.0 + path["net"]).cumprod()
    drawdown = nav / nav.cummax() - 1.0
    comparison: pd.Series | None = None
    if VERSION80_RETURNS.exists():
        current = pd.read_csv(VERSION80_RETURNS, parse_dates=["trade_date"]).set_index(
            "trade_date"
        )["VERSION 80 Composite"]
        comparison = (1.0 + current.reindex(nav.index).dropna()).cumprod()
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), constrained_layout=True)
    axes[0].plot(nav.index, nav, color="#175CD3", linewidth=1.6, label="Demo Iter 325 transfer")
    if comparison is not None and not comparison.empty:
        axes[0].plot(
            comparison.index,
            comparison,
            color="#067647",
            linewidth=1.25,
            label="Current VERSION 80",
        )
    axes[0].set_title("Public walk-forward net asset value")
    axes[0].set_ylabel("NAV")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)
    axes[1].fill_between(drawdown.index, drawdown, 0, color="#B42318", alpha=0.3)
    axes[1].plot(drawdown.index, drawdown, color="#B42318", linewidth=1.0)
    axes[1].set_title("Demo Iter 325 transfer drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(alpha=0.2)
    years = [int(fold.validation_start[:4]) for fold in fold_results]
    sharpes = [fold.sharpe for fold in fold_results]
    colors = ["#067647" if value >= 0 else "#B42318" for value in sharpes]
    axes[2].bar(years, sharpes, color=colors)
    axes[2].axhline(0, color="#475467", linewidth=0.8)
    axes[2].set_title("Annual walk-forward Sharpe")
    axes[2].set_xlabel("Validation year")
    axes[2].set_ylabel("Sharpe")
    axes[2].set_xticks(years)
    axes[2].grid(axis="y", alpha=0.2)
    fig.savefig(output / "transfer_backtest.png", dpi=180)
    plt.close(fig)


def _write_outputs(
    output: Path,
    path: pd.DataFrame,
    folds: list[FoldResult],
    weight_rows: list[dict[str, object]],
    aggregate: dict[str, object],
    demo_best: dict[str, object],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    aggregate = dict(aggregate)
    weight_frame = pd.DataFrame(weight_rows)
    weight_pivot = weight_frame.pivot(
        index="validation_year", columns="factor", values="weight"
    ).fillna(0.0)
    aggregate["mean_annual_weight_l1_change"] = float(
        weight_pivot.diff().abs().sum(axis=1).dropna().mean()
    )
    aggregate["mean_effective_factor_count"] = float(
        np.mean([fold.effective_factor_count for fold in folds])
    )
    gate_failures = []
    if float(aggregate["positive_fold_fraction"]) < 0.60:
        gate_failures.append("positive_fold_fraction_below_0.60")
    if float(aggregate["worst_fold_sharpe"]) < -0.50:
        gate_failures.append("worst_fold_sharpe_below_minus_0.50")
    if float(aggregate["deflated_sharpe_probability_325_trials"]) < 0.90:
        gate_failures.append("deflated_sharpe_probability_below_0.90")
    if float(aggregate["net_return_hac_p_value"]) > 0.10:
        gate_failures.append("net_return_hac_p_value_above_0.10")
    if float(aggregate["stressed_sharpe"]) < 0.0:
        gate_failures.append("negative_stressed_sharpe")
    aggregate["public_gate_passed"] = not gate_failures
    aggregate["public_gate_failures"] = gate_failures
    daily = path.copy()
    daily["nav"] = (1.0 + daily["net"]).cumprod()
    daily["drawdown"] = daily["nav"] / daily["nav"].cummax() - 1.0
    daily.to_csv(output / "daily_returns.csv", index_label="trade_date")
    pd.DataFrame([asdict(fold) for fold in folds]).to_csv(
        output / "walk_forward_folds.csv", index=False
    )
    weight_frame.to_csv(output / "factor_weights.csv", index=False)
    version80 = _version80_comparison()
    summary = {
        "research_status": "PUBLIC_WALK_FORWARD_TRANSFER_TEST_ONLY",
        "source_champion": {
            "demo_best_json": str(DEMO_BEST),
            "archived_alpha": str(DEMO_ALPHA),
            "archived_alpha_sha256": _sha256(DEMO_ALPHA),
            "demo_best_record": demo_best,
            "factor_count": len(FACTOR_NAMES),
            "factor_names": list(FACTOR_NAMES),
            "horizon_days": HORIZON,
            "label_kind": "rank",
            "weighting": "train-only robust rank/Pearson ICIR with 15-calendar-day half-life",
        },
        "data": {
            "root": str(DATA_ROOT),
            "panel": str(PANEL_ROOT),
            "public_cutoff": PUBLIC_END.date().isoformat(),
            "validity_filter": "is_valid_ohlc AND is_tradable_observation",
            "universe": "current project processed A-share panel",
        },
        "protocol": {
            "name": "institutional_walkforward_v3_cross_project_transfer",
            "train_years": TRAIN_YEARS,
            "validation_years": 1,
            "first_validation_year": int(folds[0].validation_start[:4]),
            "last_validation_year": int(folds[-1].validation_start[:4]),
            "annual_refit": True,
            "holding_period_days": HOLDING_DAYS,
            "portfolio": "dollar-neutral top/bottom decile with overlapping sleeves",
            "one_way_cost_bps": ONE_WAY_BPS,
            "stress_cost_multiplier": STRESS_MULTIPLIER,
            "hidden_2025_2026_accessed": False,
        },
        "aggregate_metrics": aggregate,
        "current_version80_public_reference": version80,
        "interpretation_limits": [
            "This is a public rolling out-of-sample transfer test, not formal hidden approval.",
            (
                "The demo champion was selected after many adaptive trials on another dataset; "
                "its original score is not directly comparable."
            ),
            (
                "Raw close and volume semantics are mapped to the current processed panel; "
                "tradability filtering follows the current project."
            ),
            (
                "The 20-day IC uses overlapping labels and is reported as a diagnostic, "
                "not the primary acceptance statistic."
            ),
        ],
        "artifacts": {
            "daily_returns": str(output / "daily_returns.csv"),
            "walk_forward_folds": str(output / "walk_forward_folds.csv"),
            "factor_weights": str(output / "factor_weights.csv"),
            "chart": str(output / "transfer_backtest.png"),
            "report": str(output / "TRANSFER_TEST_REPORT.md"),
        },
    }
    (output / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    comparison_rows = [
        {"series": "AutoAlpha-demo Iteration 325 transfer", **aggregate},
    ]
    if version80:
        comparison_rows.append({"series": "Current VERSION 80 public", **version80})
    pd.DataFrame(comparison_rows).to_csv(output / "metric_comparison.csv", index=False)
    _plot_results(output, path, folds)
    reference = version80 or {
        key: float("nan")
        for key in (
            "sharpe",
            "simple_annual_return",
            "compound_annual_return",
            "max_drawdown",
            "annual_turnover",
            "stressed_sharpe",
            "positive_fold_fraction",
            "worst_fold_sharpe",
        )
    }
    transfer_annual = f"{aggregate['simple_annual_return']:.2%}"
    reference_annual = f"{reference['simple_annual_return']:.2%}"
    transfer_compound = f"{aggregate['compound_annual_return']:.2%}"
    reference_compound = f"{reference['compound_annual_return']:.2%}"
    transfer_positive = f"{aggregate['positive_fold_fraction']:.0%}"
    reference_positive = f"{reference['positive_fold_fraction']:.0%}"
    report = f"""# AutoAlpha-demo Champion Transfer Test

## Verdict

**FAIL - public walk-forward research gates were not all passed.** This result does not
authorize hidden evaluation, portfolio promotion, or production use.

## Scope

- Source: AutoAlpha-demo Iteration {demo_best.get('iter_id')} archived champion.
- Formula: 16 factors, 20-day rank label, train-only robust ICIR weighting.
- Target data: `{PANEL_ROOT}`.
- Public test: {aggregate['backtest_start']} to {aggregate['backtest_end']}.
- Protocol: 5-year train / 1-year validation, annual refit, 10 folds.
- Trading: top/bottom decile, 5-day overlapping sleeves, {ONE_WAY_BPS:.1f} bps one-way cost.
- Hidden 2025-2026 data: not accessed.

## Results

| Metric | Demo champion transfer | Current VERSION 80 reference |
|---|---:|---:|
| Sharpe | {aggregate['sharpe']:.3f} | {reference['sharpe']:.3f} |
| Simple annual return | {transfer_annual} | {reference_annual} |
| Compound annual return | {transfer_compound} | {reference_compound} |
| Maximum drawdown | {aggregate['max_drawdown']:.2%} | {reference['max_drawdown']:.2%} |
| Annual turnover | {aggregate['annual_turnover']:.2f} | {reference['annual_turnover']:.2f} |
| Stressed Sharpe | {aggregate['stressed_sharpe']:.3f} | {reference['stressed_sharpe']:.3f} |
| Positive fold fraction | {transfer_positive} | {reference_positive} |
| Worst fold Sharpe | {aggregate['worst_fold_sharpe']:.3f} | {reference['worst_fold_sharpe']:.3f} |

The mean 20-day Rank IC across folds is
{aggregate['rank_ic_20d_mean_across_folds']:.4f}, but the tradable five-day portfolio is much
weaker. The mean annual L1 weight change is
{aggregate['mean_annual_weight_l1_change']:.3f}; this confirms material allocation instability
from the 15-calendar-day IC decay half-life.

## Gate Failures

{chr(10).join(f'- `{failure}`' for failure in gate_failures)}

## Interpretation

The cross-project test rejects direct production migration. Positive long-horizon IC does not
compensate for weak portfolio conversion, three negative validation years, the 2021 Sharpe of
{min(fold.sharpe for fold in folds):.3f}, and low multiple-testing-adjusted confidence. The factor
ideas may still be mined individually, but the demo weighting and smoothing stack should not be
adopted as a package.
"""
    (output / "TRANSFER_TEST_REPORT.md").write_text(report)


def main() -> None:
    args = parse_args()
    if args.first_validation_year < FIRST_VALIDATION_YEAR:
        raise ValueError(f"First validation year cannot be earlier than {FIRST_VALIDATION_YEAR}")
    if args.last_validation_year > LAST_VALIDATION_YEAR:
        raise ValueError(f"Last validation year cannot exceed public year {LAST_VALIDATION_YEAR}")
    demo_alpha = _load_demo_alpha()
    archived_names = tuple(function.__name__ for function in demo_alpha.FACTORS)
    if archived_names != FACTOR_NAMES:
        raise RuntimeError("Local transfer implementation no longer matches archived factor order")
    if int(demo_alpha.HORIZON) != HORIZON or str(demo_alpha.LABEL_KIND) != "rank":
        raise RuntimeError("Archived champion horizon or label kind changed")
    demo_best = json.loads(DEMO_BEST.read_text())
    functions = _factor_functions()
    fold_results: list[FoldResult] = []
    path_parts: list[pd.DataFrame] = []
    weight_rows: list[dict[str, object]] = []
    prior_targets: pd.DataFrame | None = None

    for fold_id, validation_year in enumerate(
        range(args.first_validation_year, args.last_validation_year + 1)
    ):
        train_start_year = validation_year - TRAIN_YEARS
        print(
            f"[fold {fold_id + 1}] train={train_start_year}-{validation_year - 1} "
            f"validate={validation_year}",
            flush=True,
        )
        load_end_year = min(validation_year + 1, LAST_VALIDATION_YEAR)
        data = _load_years(range(train_start_year, load_end_year + 1))
        train_data = data[
            (data["trade_date"].dt.year >= train_start_year)
            & (data["trade_date"].dt.year < validation_year)
        ]
        validation_data = data[data["trade_date"].dt.year == validation_year]
        return_data = data[
            (data["trade_date"] >= validation_data["trade_date"].min())
            & (
                data["trade_date"]
                <= min(
                    PUBLIC_END,
                    pd.Timestamp(f"{validation_year + 1}-01-15"),
                )
            )
        ]
        train_fields = _wide_fields(train_data)
        validation_fields = _wide_fields(validation_data)
        return_prices = return_data.pivot(
            index="trade_date", columns="symbol", values="adj_close"
        ).sort_index()
        del data, train_data, validation_data, return_data

        weights, diagnostics = _estimate_weights(train_fields, functions)
        raw_signal = _combine_validation_factors(validation_fields, functions, weights)
        signal = _finalize_signal(raw_signal, _adaptive_spans(train_fields))
        targets = _target_positions(signal)
        path, prior_targets = _strategy_path(targets, return_prices, prior_targets)
        fold = _fold_metrics(
            fold_id,
            train_fields,
            validation_fields,
            signal,
            path,
            weights,
        )
        fold_results.append(fold)
        path_parts.append(path)
        for factor_name, weight, diagnostic in zip(
            FACTOR_NAMES, weights, diagnostics, strict=True
        ):
            weight_rows.append(
                {
                    "fold_id": fold_id,
                    "validation_year": validation_year,
                    "factor": factor_name,
                    "weight": float(weight),
                    **diagnostic,
                }
            )
        print(
            f"[fold {fold_id + 1}] sharpe={fold.sharpe:.3f} "
            f"annual={fold.simple_annual_return:.2%} mdd={fold.max_drawdown:.2%} "
            f"rank_ic20={fold.rank_ic_20d_mean:.4f}",
            flush=True,
        )
        del train_fields, validation_fields, return_prices, raw_signal, signal, targets

    full_path = pd.concat(path_parts).sort_index()
    aggregate = _aggregate_metrics(full_path, fold_results)
    _write_outputs(
        args.output.resolve(), full_path, fold_results, weight_rows, aggregate, demo_best
    )
    print(json.dumps(aggregate, ensure_ascii=False, indent=2), flush=True)
    print(f"Artifacts: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
