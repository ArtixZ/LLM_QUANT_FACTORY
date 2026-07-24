from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from autoalpha.data.pit import PointInTimeFrame


@dataclass(frozen=True)
class SecurityUniverse:
    memberships: PointInTimeFrame
    symbol_column: str = "symbol"
    valid_from_column: str = "valid_from"
    valid_to_column: str = "valid_to"
    eligible_column: str = "is_eligible"

    def as_of(
        self,
        trading_date: pd.Timestamp | str,
        *,
        known_at: pd.Timestamp | str,
    ) -> pd.Index:
        visible = self.memberships.as_of(known_at)
        date = pd.Timestamp(trading_date).normalize()
        valid_from = pd.to_datetime(visible[self.valid_from_column]).dt.normalize()
        valid_to = pd.to_datetime(visible[self.valid_to_column], errors="coerce").dt.normalize()
        active = (valid_from <= date) & (valid_to.isna() | (valid_to >= date))
        if self.eligible_column in visible:
            active &= visible[self.eligible_column].astype(bool)
        symbols = visible.loc[active, self.symbol_column].astype(str).sort_values().unique()
        return pd.Index(symbols, name=self.symbol_column)
