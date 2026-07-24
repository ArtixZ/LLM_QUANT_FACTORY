from __future__ import annotations

import numpy as np
import pandas as pd

from autoalpha.research.evaluation import (
    WalkForwardEvaluator,
    cross_sectional_ic,
    investment_diagnostics,
    parameter_neighborhood_stability,
    robustness_by_segment,
)
from autoalpha.research.splits import PurgedWalkForward


def _panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(12)
    dates = pd.bdate_range("2020-01-01", periods=140)
    symbols = [f"S{index:03d}" for index in range(60)]
    signal = pd.DataFrame(rng.normal(size=(len(dates), len(symbols))), index=dates, columns=symbols)
    labels = signal * 0.05 + pd.DataFrame(
        rng.normal(scale=0.95, size=signal.shape), index=dates, columns=symbols
    )
    signal.iloc[0, 0] = np.nan
    labels.iloc[0, 1] = np.nan
    return signal, labels


def test_cross_sectional_ic_uses_only_pairwise_valid_names() -> None:
    signal, labels = _panels()

    rank_ic = cross_sectional_ic(signal, labels, minimum_names=30)

    assert len(rank_ic) == len(signal)
    assert rank_ic.mean() > 0


def test_walk_forward_and_robustness_produce_evidence() -> None:
    signal, labels = _panels()
    evaluator = WalkForwardEvaluator(
        PurgedWalkForward(50, 20, 20, label_horizon=5, embargo_size=5),
        minimum_names=30,
    )

    report = evaluator.evaluate(signal, labels)
    robustness = robustness_by_segment(
        signal,
        labels,
        {
            "first_half": pd.Series(np.arange(len(signal)) < 70, index=signal.index),
            "second_half": pd.Series(np.arange(len(signal)) >= 70, index=signal.index),
        },
        minimum_names=30,
    )

    assert len(report.folds) >= 3
    assert report.pooled_rank_ic.mean > 0
    assert robustness.positive_fraction == 1.0


def test_investment_diagnostics_reports_monotonic_spread_and_decay() -> None:
    rng = np.random.default_rng(13)
    dates = pd.date_range("2022-01-01", periods=20)
    symbols = [f"S{i:02d}" for i in range(40)]
    signal = pd.DataFrame(rng.normal(size=(20, 40)), index=dates, columns=symbols)
    near = signal * 0.02 + pd.DataFrame(
        rng.normal(scale=0.002, size=(20, 40)), index=dates, columns=symbols
    )
    far = signal * 0.005 + pd.DataFrame(
        rng.normal(scale=0.01, size=(20, 40)), index=dates, columns=symbols
    )

    diagnostics = investment_diagnostics(signal, {1: near, 5: far}, quantiles=5)

    assert diagnostics.monotonicity > 0.8
    assert diagnostics.top_bottom_spread > 0
    assert diagnostics.ic_decay[1] > diagnostics.ic_decay[5]
    stability = parameter_neighborhood_stability({"w4": 0.02, "w5": 0.03, "w6": 0.01})
    assert stability.positive_fraction == 1.0
