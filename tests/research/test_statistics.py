from __future__ import annotations

import numpy as np

from autoalpha.research.statistics import (
    benjamini_hochberg,
    hac_mean_inference,
    stationary_block_bootstrap_mean,
)


def test_hac_standard_error_reflects_positive_serial_correlation() -> None:
    rng = np.random.default_rng(7)
    innovations = rng.normal(size=2000)
    values = np.empty_like(innovations)
    values[0] = innovations[0]
    for index in range(1, len(values)):
        values[index] = 0.8 * values[index - 1] + innovations[index]

    iid = hac_mean_inference(values, lags=0)
    adjusted = hac_mean_inference(values, lags=10)

    assert adjusted.standard_error > iid.standard_error
    assert adjusted.observations == 2000


def test_benjamini_hochberg_controls_family() -> None:
    rejected = benjamini_hochberg([0.001, 0.01, 0.04, 0.20], alpha=0.05)

    assert rejected.tolist() == [True, True, False, False]


def test_block_bootstrap_is_reproducible() -> None:
    values = np.linspace(-0.1, 0.2, 100)

    first = stationary_block_bootstrap_mean(values, block_size=5, samples=500, seed=42)
    second = stationary_block_bootstrap_mean(values, block_size=5, samples=500, seed=42)

    assert first == second
    assert first[0] < values.mean() < first[1]
