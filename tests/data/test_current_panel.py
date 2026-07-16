import pandas as pd
import pytest

from autoalpha.data.current_panel import inspect_current_panel


def test_panel_inspection_does_not_invent_missing_pit_fields(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "trade_date": [pd.Timestamp("2024-01-02")],
            "open": [10.0],
            "high": [10.5],
            "low": [9.8],
            "close": [10.2],
            "adj_close": [10.2],
            "vol": [1_000.0],
            "amount": [10_000.0],
            "is_valid_ohlc": [True],
            "is_tradable_observation": [True],
        }
    )
    frame.to_parquet(tmp_path / "data.parquet")
    report = inspect_current_panel(tmp_path)
    assert report.price_research_ready
    assert not report.institutional_pit_ready
    assert any("knowledge" in blocker for blocker in report.blockers)
    with pytest.raises(RuntimeError, match="not institutionally"):
        report.require_institutional_pit()
