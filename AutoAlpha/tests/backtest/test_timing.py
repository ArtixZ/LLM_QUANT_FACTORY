from __future__ import annotations

import pandas as pd
import pytest

from autoalpha.backtest.timing import (
    entry_aligned_open_return,
    next_open_return_for_eod_signal,
)


def test_eod_signal_return_starts_at_the_following_session_open() -> None:
    dates = pd.bdate_range("2024-01-02", periods=4)
    opens = pd.DataFrame({"A": [10.0, 11.0, 13.2, 12.0]}, index=dates)

    signal_aligned = next_open_return_for_eod_signal(opens)
    entry_aligned = entry_aligned_open_return(opens)

    assert signal_aligned.loc[dates[0], "A"] == pytest.approx(13.2 / 11.0 - 1.0)
    assert entry_aligned.loc[dates[1], "A"] == pytest.approx(13.2 / 11.0 - 1.0)
    assert pd.isna(signal_aligned.loc[dates[-2], "A"])
