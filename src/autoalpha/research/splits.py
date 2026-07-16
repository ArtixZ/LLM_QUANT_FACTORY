from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: int
    train_dates: pd.DatetimeIndex
    validation_dates: pd.DatetimeIndex
    purge_dates: pd.DatetimeIndex
    embargo_dates: pd.DatetimeIndex

    def __post_init__(self) -> None:
        if len(self.train_dates) == 0 or len(self.validation_dates) == 0:
            raise ValueError("A fold requires non-empty train and validation dates")
        if self.train_dates.max() >= self.validation_dates.min():
            raise ValueError("Train dates must precede validation dates")
        if len(self.train_dates.intersection(self.validation_dates)):
            raise ValueError("Train and validation dates overlap")


@dataclass(frozen=True)
class PurgedWalkForward:
    train_size: int
    validation_size: int
    step_size: int
    label_horizon: int
    embargo_size: int = 0
    expanding: bool = False

    def __post_init__(self) -> None:
        values = (
            self.train_size,
            self.validation_size,
            self.step_size,
            self.label_horizon,
        )
        if any(value <= 0 for value in values) or self.embargo_size < 0:
            raise ValueError("Split sizes and label_horizon must be positive; embargo may be zero")
        if self.train_size <= self.label_horizon:
            raise ValueError("train_size must exceed label_horizon")

    def split(self, dates: pd.DatetimeIndex) -> Iterator[WalkForwardFold]:
        calendar = pd.DatetimeIndex(dates).drop_duplicates().sort_values()
        first_validation = self.train_size + self.label_horizon
        fold_id = 0
        validation_start = first_validation
        while validation_start + self.validation_size <= len(calendar):
            train_end = validation_start - self.label_horizon
            train_start = 0 if self.expanding else train_end - self.train_size
            if train_start < 0:
                validation_start += self.step_size
                continue
            validation_end = validation_start + self.validation_size
            purge = calendar[train_end:validation_start]
            embargo_end = min(len(calendar), validation_end + self.embargo_size)
            yield WalkForwardFold(
                fold_id=fold_id,
                train_dates=calendar[train_start:train_end],
                validation_dates=calendar[validation_start:validation_end],
                purge_dates=purge,
                embargo_dates=calendar[validation_end:embargo_end],
            )
            fold_id += 1
            validation_start += self.step_size
