from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pandas as pd

RebalanceSchedule = Literal[
    "DAILY_ROLLING",
    "WEEKLY_FIRST_SESSION",
    "BIWEEKLY_FIRST_SESSION",
    "MONTHLY_FIRST_SESSION",
]


def select_target_positions(
    signal: np.ndarray,
    *,
    selection_fraction: float,
    maximum_positions: int | None,
) -> np.ndarray:
    """Select a deterministic target set without applying execution eligibility.

    Tradability belongs to execution, not portfolio construction. Keeping it out
    of this function lets vector and event engines consume the same target intent.
    Ties are resolved by the stable input-column position.
    """
    values = np.asarray(signal, dtype=float)
    candidates = np.flatnonzero(np.isfinite(values))
    if candidates.size == 0:
        return candidates
    count = max(1, int(math.ceil(candidates.size * selection_fraction)))
    if maximum_positions is not None:
        count = min(count, maximum_positions)
    order = np.lexsort((candidates, -values[candidates]))
    return candidates[order[:count]]


def select_target_symbols(
    signal: pd.Series,
    *,
    selection_fraction: float,
    maximum_positions: int | None,
) -> tuple[str, ...]:
    positions = select_target_positions(
        signal.to_numpy(dtype=float, copy=False),
        selection_fraction=selection_fraction,
        maximum_positions=maximum_positions,
    )
    return tuple(str(signal.index[position]) for position in positions)


def is_rebalance_session(
    dates: pd.DatetimeIndex,
    date_index: int,
    schedule: RebalanceSchedule,
) -> bool:
    if date_index <= 0:
        return False
    if schedule == "DAILY_ROLLING":
        return True
    current = pd.Timestamp(dates[date_index])
    previous = pd.Timestamp(dates[date_index - 1])
    if schedule in {"WEEKLY_FIRST_SESSION", "BIWEEKLY_FIRST_SESSION"}:
        changed = current.isocalendar()[:2] != previous.isocalendar()[:2]
        if not changed or schedule == "WEEKLY_FIRST_SESSION":
            return bool(changed)
        first_sessions = [
            position
            for position in range(1, date_index + 1)
            if pd.Timestamp(dates[position]).isocalendar()[:2]
            != pd.Timestamp(dates[position - 1]).isocalendar()[:2]
        ]
        return first_sessions[-1] == date_index and (len(first_sessions) - 1) % 2 == 0
    return (current.year, current.month) != (previous.year, previous.month)


def rebalance_mask(
    dates: pd.DatetimeIndex,
    schedule: RebalanceSchedule,
    active: np.ndarray | None = None,
) -> np.ndarray:
    mask = np.zeros(len(dates), dtype=bool)
    if schedule == "DAILY_ROLLING":
        mask[1:] = True
    else:
        for index in range(1, len(dates)):
            mask[index] = is_rebalance_session(dates, index, schedule)
    if active is not None:
        mask &= np.asarray(active, dtype=bool)
    return mask
