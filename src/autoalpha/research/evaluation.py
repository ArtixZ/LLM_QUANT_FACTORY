from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from autoalpha.research.splits import PurgedWalkForward
from autoalpha.research.statistics import MeanInference, hac_mean_inference

CorrelationMethod = Literal["rank", "pearson"]


@dataclass(frozen=True)
class FoldEvaluation:
    fold_id: int
    start: str
    end: str
    observations: int
    rank_ic_mean: float
    rank_ic_hac_t: float
    rank_ic_hit_rate: float
    pearson_ic_mean: float


@dataclass(frozen=True)
class WalkForwardReport:
    folds: tuple[FoldEvaluation, ...]
    pooled_rank_ic: MeanInference
    pooled_pearson_ic: MeanInference
    positive_fold_fraction: float
    worst_fold_rank_ic: float


@dataclass(frozen=True)
class RobustnessResult:
    segment_metrics: dict[str, float]
    positive_fraction: float
    worst_segment: float
    dispersion: float


@dataclass(frozen=True)
class InvestmentDiagnostics:
    coverage: float
    quantile_returns: tuple[float, ...]
    monotonicity: float
    top_bottom_spread: float
    ic_decay: dict[int, float]


def investment_diagnostics(
    signal: pd.DataFrame,
    forward_returns: dict[int, pd.DataFrame],
    *,
    quantiles: int = 10,
    minimum_names: int = 30,
) -> InvestmentDiagnostics:
    if not forward_returns or quantiles < 2:
        raise ValueError("Forward returns and at least two quantiles are required")
    first_horizon = min(forward_returns)
    target = forward_returns[first_horizon]
    quantile_observations: list[list[float]] = [[] for _ in range(quantiles)]
    valid_points = 0
    total_points = signal.size
    for date in signal.index.intersection(target.index):
        pair = pd.concat(
            [signal.loc[date].rename("signal"), target.loc[date].rename("return")], axis=1
        ).dropna()
        valid_points += len(pair)
        if len(pair) < max(minimum_names, quantiles):
            continue
        buckets = pd.qcut(pair["signal"].rank(method="first"), quantiles, labels=False)
        for bucket in range(quantiles):
            bucket_return = pair.loc[buckets == bucket, "return"].mean()
            quantile_observations[bucket].append(float(bucket_return))
    quantile_returns = tuple(
        float(np.mean(values)) if values else float("nan") for values in quantile_observations
    )
    finite = np.asarray(quantile_returns, dtype=float)
    valid_quantiles = np.isfinite(finite)
    monotonicity = (
        float(
            stats.spearmanr(
                np.arange(quantiles)[valid_quantiles], finite[valid_quantiles]
            ).statistic
        )
        if valid_quantiles.sum() >= 2
        else float("nan")
    )
    decay = {
        horizon: float(
            cross_sectional_ic(
                signal,
                labels,
                method="rank",
                minimum_names=minimum_names,
            ).mean()
        )
        for horizon, labels in sorted(forward_returns.items())
    }
    return InvestmentDiagnostics(
        coverage=valid_points / total_points if total_points else 0.0,
        quantile_returns=quantile_returns,
        monotonicity=monotonicity,
        top_bottom_spread=float(finite[-1] - finite[0]),
        ic_decay=decay,
    )


def parameter_neighborhood_stability(metrics: Mapping[str, float]) -> RobustnessResult:
    finite = np.asarray([value for value in metrics.values() if np.isfinite(value)])
    if finite.size == 0:
        raise ValueError("No finite parameter-neighborhood metrics")
    return RobustnessResult(
        segment_metrics=dict(metrics),
        positive_fraction=float((finite > 0).mean()),
        worst_segment=float(finite.min()),
        dispersion=float(finite.std()),
    )


def cross_sectional_ic(
    signal: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    method: CorrelationMethod = "rank",
    minimum_names: int = 30,
) -> pd.Series:
    if method not in {"rank", "pearson"}:
        raise ValueError(f"Unknown correlation method: {method}")
    dates = signal.index.intersection(labels.index)
    symbols = signal.columns.intersection(labels.columns)
    x = signal.reindex(index=dates, columns=symbols).astype(float)
    y = labels.reindex(index=dates, columns=symbols).astype(float)
    valid = np.isfinite(x.to_numpy()) & np.isfinite(y.to_numpy())
    if method == "rank":
        x = x.where(valid).rank(axis=1, method="average")
        y = y.where(valid).rank(axis=1, method="average")
    correlations, counts = _rowwise_pairwise_correlation(x.to_numpy(), y.to_numpy())
    accepted = (counts >= minimum_names) & np.isfinite(correlations)
    return pd.Series(
        correlations[accepted],
        index=dates[accepted],
        name=f"{method}_ic",
        dtype=float,
    ).sort_index()


def _rowwise_pairwise_correlation(
    left: np.ndarray, right: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(left) & np.isfinite(right)
    counts = valid.sum(axis=1)
    left_values = np.where(valid, left, 0.0)
    right_values = np.where(valid, right, 0.0)
    safe_counts = np.maximum(counts, 1)
    left_sum = left_values.sum(axis=1)
    right_sum = right_values.sum(axis=1)
    covariance = (left_values * right_values).sum(axis=1) - (left_sum * right_sum / safe_counts)
    left_variance = (left_values * left_values).sum(axis=1) - left_sum**2 / safe_counts
    right_variance = (right_values * right_values).sum(axis=1) - right_sum**2 / safe_counts
    denominator = np.sqrt(np.maximum(left_variance * right_variance, 0.0))
    correlations = np.divide(
        covariance,
        denominator,
        out=np.full(left.shape[0], np.nan, dtype=float),
        where=denominator > 0,
    )
    return correlations, counts


class WalkForwardEvaluator:
    def __init__(
        self,
        splitter: PurgedWalkForward,
        *,
        minimum_names: int = 30,
        hac_lags: int | None = None,
    ) -> None:
        self.splitter = splitter
        self.minimum_names = minimum_names
        self.hac_lags = hac_lags if hac_lags is not None else splitter.label_horizon - 1

    def evaluate(self, signal: pd.DataFrame, labels: pd.DataFrame) -> WalkForwardReport:
        common_dates = signal.index.intersection(labels.index)
        rank_all: list[pd.Series] = []
        pearson_all: list[pd.Series] = []
        fold_reports: list[FoldEvaluation] = []
        for fold in self.splitter.split(common_dates):
            validation_dates = fold.validation_dates
            rank_ic = cross_sectional_ic(
                signal.loc[validation_dates],
                labels.loc[validation_dates],
                method="rank",
                minimum_names=self.minimum_names,
            )
            pearson_ic = cross_sectional_ic(
                signal.loc[validation_dates],
                labels.loc[validation_dates],
                method="pearson",
                minimum_names=self.minimum_names,
            )
            if len(rank_ic) < 2 or len(pearson_ic) < 2:
                continue
            inference = hac_mean_inference(
                rank_ic.to_numpy(), lags=min(self.hac_lags, len(rank_ic) - 1)
            )
            fold_reports.append(
                FoldEvaluation(
                    fold_id=fold.fold_id,
                    start=validation_dates.min().date().isoformat(),
                    end=validation_dates.max().date().isoformat(),
                    observations=len(rank_ic),
                    rank_ic_mean=float(rank_ic.mean()),
                    rank_ic_hac_t=inference.t_stat,
                    rank_ic_hit_rate=float((rank_ic > 0).mean()),
                    pearson_ic_mean=float(pearson_ic.mean()),
                )
            )
            rank_all.append(rank_ic)
            pearson_all.append(pearson_ic)
        if not fold_reports:
            raise ValueError("No evaluable walk-forward folds")
        pooled_rank = pd.concat(rank_all).sort_index()
        pooled_pearson = pd.concat(pearson_all).sort_index()
        rank_inference = hac_mean_inference(
            pooled_rank.to_numpy(), lags=min(self.hac_lags, len(pooled_rank) - 1)
        )
        pearson_inference = hac_mean_inference(
            pooled_pearson.to_numpy(), lags=min(self.hac_lags, len(pooled_pearson) - 1)
        )
        means = np.array([fold.rank_ic_mean for fold in fold_reports])
        return WalkForwardReport(
            folds=tuple(fold_reports),
            pooled_rank_ic=rank_inference,
            pooled_pearson_ic=pearson_inference,
            positive_fold_fraction=float((means > 0).mean()),
            worst_fold_rank_ic=float(means.min()),
        )


def robustness_by_segment(
    signal: pd.DataFrame,
    labels: pd.DataFrame,
    segments: Mapping[str, pd.Series | np.ndarray],
    *,
    minimum_names: int = 30,
) -> RobustnessResult:
    metrics: dict[str, float] = {}
    for name, mask in segments.items():
        mask_series = (
            pd.Series(mask, index=signal.index) if not isinstance(mask, pd.Series) else mask
        )
        selected_dates = signal.index.intersection(mask_series.index[mask_series.astype(bool)])
        ic = cross_sectional_ic(
            signal.loc[selected_dates],
            labels.loc[selected_dates],
            method="rank",
            minimum_names=minimum_names,
        )
        metrics[name] = float(ic.mean()) if len(ic) else float("nan")
    finite = np.array([value for value in metrics.values() if np.isfinite(value)])
    if finite.size == 0:
        raise ValueError("No segment produced a finite robustness metric")
    return RobustnessResult(
        segment_metrics=metrics,
        positive_fraction=float((finite > 0).mean()),
        worst_segment=float(finite.min()),
        dispersion=float(finite.std()),
    )
