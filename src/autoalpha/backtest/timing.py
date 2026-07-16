from __future__ import annotations

import pandas as pd

EOD_NEXT_OPEN_RETURN_CONVENTION = "EOD_T__OPEN_T1_TO_OPEN_T2"


def next_open_return_for_eod_signal(open_prices: pd.DataFrame) -> pd.DataFrame:
    """Return earned after an EOD signal is executed at the next session open."""
    return open_prices.shift(-2).div(open_prices.shift(-1)).sub(1.0)


def entry_aligned_open_return(open_prices: pd.DataFrame) -> pd.DataFrame:
    """Open-to-next-open return indexed by the entry session."""
    return open_prices.pct_change(fill_method=None).shift(-1)
