from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class MeanInference:
    mean: float
    standard_error: float
    t_stat: float
    p_value: float
    confidence_interval: tuple[float, float]
    observations: int
    lags: int


def hac_mean_inference(
    values: np.ndarray | list[float],
    *,
    lags: int,
    confidence: float = 0.95,
) -> MeanInference:
    sample = np.asarray(values, dtype=float)
    sample = sample[np.isfinite(sample)]
    if sample.size < 2:
        raise ValueError("At least two finite observations are required")
    if lags < 0 or lags >= sample.size:
        raise ValueError("lags must be in [0, observations)")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")

    observations = sample.size
    mean = float(sample.mean())
    centered = sample - mean
    long_run_variance = float(np.dot(centered, centered) / observations)
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        autocovariance = float(np.dot(centered[lag:], centered[:-lag]) / observations)
        long_run_variance += 2.0 * weight * autocovariance
    long_run_variance = max(long_run_variance, 0.0)
    standard_error = float(np.sqrt(long_run_variance / observations))
    if standard_error > 0:
        t_stat = mean / standard_error
    elif mean == 0:
        t_stat = 0.0
    else:
        t_stat = float(np.copysign(np.inf, mean))
    p_value = float(2.0 * stats.norm.sf(abs(t_stat)))
    critical = float(stats.norm.ppf(0.5 + confidence / 2.0))
    interval = (mean - critical * standard_error, mean + critical * standard_error)
    return MeanInference(
        mean=mean,
        standard_error=standard_error,
        t_stat=float(t_stat),
        p_value=p_value,
        confidence_interval=(float(interval[0]), float(interval[1])),
        observations=int(observations),
        lags=lags,
    )


def benjamini_hochberg(p_values: np.ndarray | list[float], *, alpha: float = 0.05) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("p_values must be a non-empty one-dimensional array")
    if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("p_values must be finite values in [0, 1]")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

    order = np.argsort(values)
    ordered = values[order]
    thresholds = alpha * np.arange(1, values.size + 1) / values.size
    passing = np.flatnonzero(ordered <= thresholds)
    rejected = np.zeros(values.size, dtype=bool)
    if passing.size:
        rejected[order[: passing[-1] + 1]] = True
    return rejected


def stationary_block_bootstrap_mean(
    values: np.ndarray | list[float],
    *,
    block_size: int,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if data.size < 2 or block_size <= 0 or block_size > data.size:
        raise ValueError("Invalid data or block_size")
    if samples <= 0 or not 0 < confidence < 1:
        raise ValueError("samples and confidence are invalid")

    rng = np.random.default_rng(seed)
    block_count = int(np.ceil(data.size / block_size))
    starts = rng.integers(0, data.size, size=(samples, block_count))
    offsets = np.arange(block_size)
    indices = (starts[..., None] + offsets) % data.size
    bootstrap = data[indices.reshape(samples, -1)[:, : data.size]].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(bootstrap, [tail, 1.0 - tail])
    return float(low), float(high)
