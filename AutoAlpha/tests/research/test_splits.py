from __future__ import annotations

import pandas as pd

from autoalpha.research.splits import PurgedWalkForward


def test_purged_walk_forward_has_no_label_overlap() -> None:
    dates = pd.bdate_range("2020-01-01", periods=80)
    splitter = PurgedWalkForward(
        train_size=30,
        validation_size=10,
        step_size=10,
        label_horizon=5,
        embargo_size=3,
    )

    folds = list(splitter.split(dates))

    assert len(folds) == 4
    assert all(len(fold.purge_dates) == 5 for fold in folds)
    assert all(len(fold.embargo_dates) in {0, 3} for fold in folds)
    for fold in folds:
        assert fold.train_dates.max() < fold.purge_dates.min()
        assert fold.purge_dates.max() < fold.validation_dates.min()


def test_expanding_window_keeps_original_train_start() -> None:
    dates = pd.bdate_range("2020-01-01", periods=70)
    folds = list(PurgedWalkForward(20, 10, 10, label_horizon=3, expanding=True).split(dates))

    assert len(folds[0].train_dates) == 20
    assert len(folds[1].train_dates) == 30
    assert folds[0].train_dates.min() == folds[1].train_dates.min()
