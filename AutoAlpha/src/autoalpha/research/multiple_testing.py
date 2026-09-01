from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class DeflatedSharpeResult:
    observed_sharpe: float
    expected_max_sharpe: float
    probability: float


def deflated_sharpe_ratio(
    returns: np.ndarray | list[float],
    *,
    trials: int,
    periods_per_year: int = 252,
) -> DeflatedSharpeResult:
    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 3 or trials <= 0:
        raise ValueError("At least three returns and one trial are required")
    standard_deviation = values.std(ddof=1)
    if standard_deviation <= 0:
        expected_daily = _expected_maximum_sharpe(trials) / math.sqrt(values.size)
        return DeflatedSharpeResult(
            observed_sharpe=0.0,
            expected_max_sharpe=expected_daily * math.sqrt(periods_per_year),
            probability=0.0,
        )
    daily_sharpe = float(values.mean() / standard_deviation)
    annualized = daily_sharpe * math.sqrt(periods_per_year)
    expected_daily = _expected_maximum_sharpe(trials) / math.sqrt(values.size)
    skewness = float(stats.skew(values, bias=False))
    kurtosis = float(stats.kurtosis(values, fisher=False, bias=False))
    denominator = math.sqrt(
        max(1e-12, 1.0 - skewness * daily_sharpe + (kurtosis - 1.0) * daily_sharpe**2 / 4.0)
    )
    test_statistic = (daily_sharpe - expected_daily) * math.sqrt(values.size - 1) / denominator
    return DeflatedSharpeResult(
        observed_sharpe=annualized,
        expected_max_sharpe=expected_daily * math.sqrt(periods_per_year),
        probability=float(stats.norm.cdf(test_statistic)),
    )


def probability_of_backtest_overfitting(
    performance: np.ndarray,
    *,
    maximum_splits: int = 10_000,
) -> float:
    matrix = np.asarray(performance, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 4 or matrix.shape[1] < 2:
        raise ValueError("performance must have at least four periods and two candidates")
    periods = matrix.shape[0]
    if periods % 2:
        raise ValueError("The number of periods must be even")
    combinations = itertools.combinations(range(periods), periods // 2)
    logits: list[float] = []
    for index, in_sample_tuple in enumerate(combinations):
        if index >= maximum_splits:
            break
        in_sample = np.array(in_sample_tuple)
        out_sample = np.setdiff1d(np.arange(periods), in_sample)
        best_candidate = int(np.nanargmax(matrix[in_sample].mean(axis=0)))
        out_scores = matrix[out_sample].mean(axis=0)
        rank = stats.rankdata(out_scores, method="average")[best_candidate]
        relative_rank = rank / (matrix.shape[1] + 1.0)
        logits.append(math.log(relative_rank / (1.0 - relative_rank)))
    if not logits:
        raise ValueError("No CSCV splits were generated")
    return float((np.asarray(logits) <= 0).mean())


def _expected_maximum_sharpe(trials: int) -> float:
    if trials == 1:
        return 0.0
    euler_gamma = 0.5772156649015329
    first = stats.norm.ppf(1.0 - 1.0 / trials)
    second = stats.norm.ppf(1.0 - 1.0 / (trials * math.e))
    return float((1.0 - euler_gamma) * first + euler_gamma * second)
