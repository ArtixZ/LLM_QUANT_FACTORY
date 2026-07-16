from __future__ import annotations

import numpy as np

from autoalpha.research.multiple_testing import (
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
)


def test_deflated_sharpe_penalizes_more_trials() -> None:
    rng = np.random.default_rng(3)
    returns = rng.normal(0.001, 0.01, 500)

    few_trials = deflated_sharpe_ratio(returns, trials=2)
    many_trials = deflated_sharpe_ratio(returns, trials=1000)

    assert many_trials.expected_max_sharpe > few_trials.expected_max_sharpe
    assert many_trials.probability < few_trials.probability


def test_pbo_is_high_when_in_sample_winners_reverse_out_of_sample() -> None:
    performance = np.array(
        [
            [4.0, 1.0],
            [4.0, 1.0],
            [1.0, 4.0],
            [1.0, 4.0],
            [4.0, 1.0],
            [1.0, 4.0],
        ]
    )

    pbo = probability_of_backtest_overfitting(performance)

    assert 0 <= pbo <= 1
