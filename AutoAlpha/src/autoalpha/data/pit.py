from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from autoalpha.data.contracts import TableContract


@dataclass(frozen=True)
class PointInTimeFrame:
    frame: pd.DataFrame
    contract: TableContract

    def __post_init__(self) -> None:
        self.contract.validate(self.frame).raise_for_errors()

    def as_of(
        self,
        known_at: pd.Timestamp | str,
        *,
        event_at_or_before: pd.Timestamp | str | None = None,
    ) -> pd.DataFrame:
        cutoff = _utc_timestamp(known_at)
        knowledge = pd.to_datetime(self.frame[self.contract.knowledge_time], utc=True)
        visible = self.frame.loc[knowledge <= cutoff].copy()
        if event_at_or_before is not None:
            event_cutoff = pd.Timestamp(event_at_or_before)
            event = pd.to_datetime(visible[self.contract.event_time])
            visible = visible.loc[event <= event_cutoff]
        if visible.empty:
            return visible.reset_index(drop=True)
        visible["__known_at"] = pd.to_datetime(visible[self.contract.knowledge_time], utc=True)
        visible = visible.sort_values([*self.contract.as_of_key, "__known_at"])
        visible = visible.drop_duplicates(list(self.contract.as_of_key), keep="last")
        return visible.drop(columns="__known_at").reset_index(drop=True)


def _utc_timestamp(value: pd.Timestamp | str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")
