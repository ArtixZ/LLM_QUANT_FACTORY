from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from autoalpha.ibkr.client import EXECUTION_PRICE_BASIS, TOTAL_RETURN_PRICE_BASIS
from autoalpha.ibkr.contracts import USEquity
from autoalpha.ibkr.history import (
    PANEL_COLUMNS,
    HistoryDownloadError,
    download_symbol_history,
    download_universe_history,
    duration_for_range,
)

TODAY = date(2026, 8, 9)
WINDOW = {"start": date(2026, 8, 3), "end": date(2026, 8, 7), "today": TODAY}


def test_duration_for_range_uses_days_inside_a_year() -> None:
    assert duration_for_range(date(2026, 8, 1), today=TODAY) == "8 D"


def test_duration_for_range_rounds_up_to_whole_years() -> None:
    assert duration_for_range(date(2020, 1, 1), today=TODAY) == "7 Y"


def test_duration_for_range_is_capped() -> None:
    assert duration_for_range(date(1900, 1, 1), today=TODAY) == "30 Y"


def test_duration_for_range_rejects_future_start() -> None:
    with pytest.raises(ValueError, match="future"):
        duration_for_range(date(2027, 1, 1), today=TODAY)


def test_download_symbol_history_joins_both_price_bases(gateway, equity: USEquity) -> None:
    history = download_symbol_history(gateway, equity, **WINDOW)
    assert history.rows == 5
    assert history.requests == 2
    assert list(history.frame.columns) == list(PANEL_COLUMNS)
    assert history.symbol == "AAPL"
    # Adjusted close drives research prices; TRADES close drives execution prices.
    assert history.frame["close"].iloc[0] == pytest.approx(99.0)
    assert history.frame["raw_close"].iloc[0] == pytest.approx(100.0)
    assert history.frame["adj_close"].iloc[0] == pytest.approx(99.0)


def test_download_symbol_history_anchors_both_requests_at_the_present(
    gateway, equity: USEquity
) -> None:
    download_symbol_history(gateway, equity, **WINDOW)
    assert [call["what"] for call in gateway.calls] == [
        TOTAL_RETURN_PRICE_BASIS,
        EXECUTION_PRICE_BASIS,
    ]
    # ADJUSTED_LAST is only served without an explicit end date.
    assert all(call["end"] is None for call in gateway.calls)


def test_download_symbol_history_trims_to_the_requested_window(
    gateway, equity: USEquity
) -> None:
    history = download_symbol_history(
        gateway, equity, start=date(2026, 8, 5), end=date(2026, 8, 6), today=TODAY
    )
    assert history.rows == 2
    assert history.first_date == date(2026, 8, 5)
    assert history.last_date == date(2026, 8, 6)


def test_dollar_volume_is_volume_times_vwap(gateway, equity: USEquity) -> None:
    row = download_symbol_history(gateway, equity, **WINDOW).frame.iloc[0]
    assert row["amount"] == pytest.approx(row["vol"] * 100.0)


def test_zero_volume_session_is_marked_halted(
    make_bars, make_gateway, equity: USEquity, sessions: list[str]
) -> None:
    traded = make_bars(sessions, close_start=100.0)
    traded.loc[2, "volume"] = 0.0
    gateway = make_gateway(make_bars(sessions, close_start=99.0), traded)
    frame = download_symbol_history(gateway, equity, **WINDOW).frame
    assert bool(frame.loc[2, "is_halted"]) is True
    assert bool(frame.loc[2, "can_buy_open"]) is False
    assert bool(frame.loc[2, "can_sell_open"]) is False
    assert bool(frame.loc[1, "can_buy_open"]) is True


def test_inconsistent_ohlc_is_rejected_as_untradable(
    make_bars, make_gateway, equity: USEquity, sessions: list[str]
) -> None:
    traded = make_bars(sessions, close_start=100.0)
    traded.loc[1, "high"] = 1.0  # high below low/open/close
    gateway = make_gateway(make_bars(sessions, close_start=99.0), traded)
    frame = download_symbol_history(gateway, equity, **WINDOW).frame
    assert bool(frame.loc[1, "is_valid_ohlc"]) is False
    assert bool(frame.loc[1, "is_tradable_observation"]) is False


def test_missing_execution_series_raises(
    make_bars, make_gateway, equity: USEquity, sessions: list[str]
) -> None:
    gateway = make_gateway(make_bars(sessions), pd.DataFrame())
    with pytest.raises(HistoryDownloadError, match="TRADES"):
        download_symbol_history(gateway, equity, **WINDOW)


def test_missing_adjusted_series_raises(
    make_bars, make_gateway, equity: USEquity, sessions: list[str]
) -> None:
    gateway = make_gateway(pd.DataFrame(), make_bars(sessions))
    with pytest.raises(HistoryDownloadError, match="ADJUSTED_LAST"):
        download_symbol_history(gateway, equity, **WINDOW)


def test_non_overlapping_dates_raise(make_bars, make_gateway, equity: USEquity) -> None:
    gateway = make_gateway(
        make_bars(["2026-08-03", "2026-08-04"]),
        make_bars(["2026-07-01", "2026-07-02"]),
    )
    with pytest.raises(HistoryDownloadError, match="do not overlap"):
        download_symbol_history(
            gateway, equity, start=date(2026, 1, 1), end=date(2026, 8, 7), today=TODAY
        )


def test_start_after_end_is_rejected(gateway, equity: USEquity) -> None:
    with pytest.raises(ValueError, match="is after end"):
        download_symbol_history(
            gateway, equity, start=date(2026, 8, 8), end=date(2026, 8, 1), today=TODAY
        )


def test_universe_download_collects_failures_without_aborting(
    make_bars, make_gateway, equity: USEquity, sessions: list[str]
) -> None:
    gateway = make_gateway(
        make_bars(sessions, close_start=99.0),
        make_bars(sessions, close_start=100.0),
        empty_symbols=["MSFT"],
    )
    missing = USEquity(symbol="MSFT", con_id=272093, primary_exchange="NASDAQ")
    histories, failures = download_universe_history(
        gateway, [equity, missing], start=date(2026, 8, 3), end=date(2026, 8, 7)
    )
    assert [item.symbol for item in histories] == ["AAPL"]
    assert "MSFT" in failures


def test_existing_slice_symbols_recovers_share_class_dots(tmp_path) -> None:
    from autoalpha.data.ibkr_sync import existing_slice_symbols

    downloads = tmp_path / "downloads"
    downloads.mkdir()
    (downloads / "AAPL.parquet").touch()
    (downloads / "BRK_B.parquet").touch()
    (downloads / "_download_manifest.json").touch()

    assert existing_slice_symbols(tmp_path) == {"AAPL", "BRK.B"}


def test_existing_slice_symbols_is_empty_without_downloads(tmp_path) -> None:
    from autoalpha.data.ibkr_sync import existing_slice_symbols

    assert existing_slice_symbols(tmp_path) == set()


def test_prune_slices_removes_only_symbols_outside_the_universe(tmp_path) -> None:
    from autoalpha.data.ibkr_sync import existing_slice_symbols, prune_slices

    downloads = tmp_path / "downloads"
    downloads.mkdir()
    for name in ("AAPL", "MSFT", "ORPHAN"):
        (downloads / f"{name}.parquet").touch()

    removed = prune_slices(tmp_path, keep=["AAPL", "MSFT"])

    assert removed == ["ORPHAN"]
    assert existing_slice_symbols(tmp_path) == {"AAPL", "MSFT"}
