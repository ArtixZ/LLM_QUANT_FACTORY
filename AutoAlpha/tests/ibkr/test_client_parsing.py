from __future__ import annotations

import math

import pandas as pd
import pytest

from autoalpha.ibkr.client import (
    UNSET_DOUBLE,
    _as_float,
    _as_optional_float,
    _bars_to_frame,
    _format_end_datetime,
)


@pytest.mark.parametrize("value", [UNSET_DOUBLE, UNSET_DOUBLE * 2, float("nan"), None, ""])
def test_unset_broker_fields_become_none(value: object) -> None:
    """IBKR sends Double.MAX_VALUE for absent fields; it must not read as a number."""
    assert _as_optional_float(value) is None


@pytest.mark.parametrize(("value", "expected"), [("12.5", 12.5), (0, 0.0), (-3.25, -3.25)])
def test_real_values_parse(value: object, expected: float) -> None:
    assert _as_optional_float(value) == pytest.approx(expected)


def test_as_float_defaults_unset_to_zero() -> None:
    assert _as_float(UNSET_DOUBLE) == 0.0
    assert _as_float("7.5") == pytest.approx(7.5)


def test_unset_is_distinguishable_from_genuine_zero() -> None:
    assert _as_optional_float(0.0) == 0.0
    assert _as_optional_float(UNSET_DOUBLE) is None


def test_format_end_datetime_blank_for_present() -> None:
    assert _format_end_datetime(None) == ""
    assert _format_end_datetime("") == ""


def test_format_end_datetime_treats_a_date_as_end_of_session() -> None:
    from datetime import date, datetime

    assert _format_end_datetime(date(2026, 8, 7)) == "20260807 23:59:59"
    assert _format_end_datetime(datetime(2026, 8, 7, 16, 0, 0)) == "20260807 16:00:00"


def test_bars_to_frame_is_empty_but_typed_without_bars() -> None:
    frame = _bars_to_frame([])
    assert frame.empty
    assert "close" in frame.columns


def test_bars_to_frame_normalizes_and_sorts() -> None:
    class Bar:
        def __init__(self, day: str, close: float) -> None:
            self.date = day
            self.open = close - 1
            self.high = close + 1
            self.low = close - 2
            self.close = close
            self.volume = 1_000
            self.average = close
            self.barCount = 10

    frame = _bars_to_frame([Bar("2026-08-05", 11.0), Bar("2026-08-04", 10.0)])
    assert list(frame["close"]) == [10.0, 11.0]
    assert frame["date"].iloc[0] == pd.Timestamp("2026-08-04")
    assert not math.isnan(frame["average"].iloc[0])
