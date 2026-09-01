from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

import pandas as pd

from autoalpha.ibkr.client import (
    EXECUTION_PRICE_BASIS,
    TOTAL_RETURN_PRICE_BASIS,
    IBKRGateway,
)
from autoalpha.ibkr.contracts import USEquity

logger = logging.getLogger(__name__)

RESEARCH_PRICE_COLUMNS = ("open", "high", "low", "close")
EXECUTION_PRICE_COLUMNS = ("raw_open", "raw_high", "raw_low", "raw_close")
PANEL_COLUMNS = (
    "symbol",
    "trade_date",
    *RESEARCH_PRICE_COLUMNS,
    "adj_close",
    *EXECUTION_PRICE_COLUMNS,
    "raw_pre_close",
    "vol",
    "amount",
    "bar_count",
    "listing_date",
    "delisting_date",
    "is_valid_ohlc",
    "is_tradable_observation",
    "can_buy_open",
    "can_sell_open",
    "is_halted",
)
MAXIMUM_DURATION_YEARS = 30

# IBKR computes ADJUSTED_LAST relative to the present, so it is only served for
# requests anchored at "now" (an explicit endDateTime silently returns nothing).
# Both series are therefore pulled as one long window ending today, and the
# caller's end date is applied as a client-side trim.
ADJUSTMENT_ANCHOR = "AS_OF_DOWNLOAD_DATE"


class HistoryDownloadError(RuntimeError):
    """A symbol's history could not be assembled into a usable panel slice."""


@dataclass(frozen=True)
class SymbolHistory:
    symbol: str
    con_id: int
    frame: pd.DataFrame
    requests: int
    first_date: date | None
    last_date: date | None

    @property
    def rows(self) -> int:
        return len(self.frame)


def duration_for_range(start: date, *, today: date | None = None) -> str:
    """Smallest IBKR duration string covering ``start`` through today."""
    anchor = today or date.today()
    if start > anchor:
        raise ValueError(f"start {start} is in the future relative to {anchor}")
    days = (anchor - start).days
    if days <= 360:
        return f"{max(days, 1)} D"
    years = math.ceil((days + 1) / 365)
    return f"{min(years, MAXIMUM_DURATION_YEARS)} Y"


def download_symbol_history(
    gateway: IBKRGateway,
    equity: USEquity,
    *,
    start: date,
    end: date,
    today: date | None = None,
) -> SymbolHistory:
    """Assemble one symbol's daily panel slice from both IBKR price bases.

    ``ADJUSTED_LAST`` provides the split- and dividend-adjusted prices that
    factors are computed on; ``TRADES`` provides the split-adjusted prices and
    share volume the execution ledger fills against. They are joined on session
    date, so a gap in either series surfaces as a missing row rather than a
    silently shifted price.

    Both series are requested as a single window ending at the download moment
    because IBKR will not serve adjusted bars against an explicit end date. The
    adjustment basis is therefore as-of today, not point-in-time.
    """
    if start > end:
        raise ValueError(f"{equity.symbol}: start {start} is after end {end}")
    duration = duration_for_range(start, today=today)
    adjusted = gateway.daily_bars(equity, end=None, duration=duration,
                                  what_to_show=TOTAL_RETURN_PRICE_BASIS)
    traded = gateway.daily_bars(equity, end=None, duration=duration,
                                what_to_show=EXECUTION_PRICE_BASIS)
    frame = _merge_price_bases(equity, adjusted, traded)
    frame = _trim(frame, start=start, end=end)
    if frame.empty:
        raise HistoryDownloadError(
            f"{equity.symbol}: no sessions between {start} and {end}"
        )
    return SymbolHistory(
        symbol=equity.panel_symbol,
        con_id=equity.con_id,
        frame=frame,
        requests=2,
        first_date=frame["trade_date"].min().date(),
        last_date=frame["trade_date"].max().date(),
    )


def _trim(frame: pd.DataFrame, *, start: date, end: date) -> pd.DataFrame:
    sessions = frame["trade_date"].dt.date
    mask = (sessions >= start) & (sessions <= end)
    return frame.loc[mask].sort_values("trade_date").reset_index(drop=True)


def _merge_price_bases(
    equity: USEquity,
    adjusted: pd.DataFrame,
    traded: pd.DataFrame,
) -> pd.DataFrame:
    if adjusted.empty or traded.empty:
        missing = TOTAL_RETURN_PRICE_BASIS if adjusted.empty else EXECUTION_PRICE_BASIS
        raise HistoryDownloadError(f"{equity.symbol}: IBKR returned no bars for {missing}")
    research = adjusted[["date", "open", "high", "low", "close"]].copy()
    research["adj_close"] = adjusted["close"].to_numpy()
    execution = traded[["date", "open", "high", "low", "close", "volume", "average"]].copy()
    execution = execution.rename(
        columns={
            "open": "raw_open",
            "high": "raw_high",
            "low": "raw_low",
            "close": "raw_close",
            "volume": "vol",
        }
    )
    execution["bar_count"] = traded["bar_count"].to_numpy()
    frame = research.merge(execution, on="date", how="inner", validate="one_to_one")
    if frame.empty:
        raise HistoryDownloadError(f"{equity.symbol}: adjusted and traded bar dates do not overlap")
    frame = frame.rename(columns={"date": "trade_date"})
    frame["symbol"] = equity.panel_symbol
    # IBKR reports a per-bar VWAP; dollar volume is the honest product of the two.
    frame["amount"] = frame["vol"] * frame["average"]
    frame["raw_pre_close"] = frame["raw_close"].shift(1)
    frame = _attach_tradability(frame)
    frame["listing_date"] = frame["trade_date"].min()
    frame["delisting_date"] = pd.NaT
    return frame.reindex(columns=list(PANEL_COLUMNS)).reset_index(drop=True)


def _attach_tradability(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive US execution eligibility flags.

    US equities have no daily price limits, so unlike the A-share original there
    is no limit-up/limit-down gate: what remains is bar validity and whether the
    session actually traded. Intraday LULD halts are invisible in daily bars, so
    ``is_halted`` only captures whole sessions with no prints.
    """
    prices = frame[["raw_open", "raw_high", "raw_low", "raw_close"]]
    positive = (prices > 0).all(axis=1)
    ordered = (
        (frame["raw_high"] >= frame["raw_low"])
        & (frame["raw_high"] >= frame["raw_open"])
        & (frame["raw_high"] >= frame["raw_close"])
        & (frame["raw_low"] <= frame["raw_open"])
        & (frame["raw_low"] <= frame["raw_close"])
    )
    frame["is_valid_ohlc"] = (positive & ordered).fillna(False).astype(bool)
    traded = frame["vol"].fillna(0.0) > 0
    frame["is_tradable_observation"] = (frame["is_valid_ohlc"] & traded).astype(bool)
    frame["is_halted"] = ~frame["is_tradable_observation"]
    frame["can_buy_open"] = frame["is_tradable_observation"]
    frame["can_sell_open"] = frame["is_tradable_observation"]
    return frame


def download_universe_history(
    gateway: IBKRGateway,
    equities: list[USEquity],
    *,
    start: date,
    end: date,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> tuple[list[SymbolHistory], dict[str, str]]:
    """Download every symbol, collecting per-symbol failures instead of aborting."""
    histories: list[SymbolHistory] = []
    failures: dict[str, str] = {}
    total = len(equities)
    for index, equity in enumerate(equities, start=1):
        if on_progress is not None:
            on_progress(index, total, equity.symbol)
        try:
            histories.append(download_symbol_history(gateway, equity, start=start, end=end))
        except (HistoryDownloadError, ValueError) as error:
            failures[equity.panel_symbol] = str(error)
            logger.warning("history download failed for %s: %s", equity.symbol, error)
    return histories, failures
