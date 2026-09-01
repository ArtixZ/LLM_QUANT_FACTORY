from types import SimpleNamespace

import pandas as pd

from autoalpha.service.screener import CrossSectionalScreener


def test_snapshot_load_deduplicates_base_and_factor_fields(tmp_path) -> None:
    partition = tmp_path / "trade_year=2026"
    partition.mkdir()
    pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-07-17", "2026-07-17"]),
            "symbol": ["000001.SZ", "600000.SH"],
            "name": ["平安银行", "浦发银行"],
            "open": [10.0, 11.0],
            "close": [10.2, 10.9],
            "raw_close": [10.2, 10.9],
            "is_valid_ohlc": [True, True],
            "is_tradable_observation": [True, True],
        }
    ).to_parquet(partition / "data_0.parquet")

    screener = object.__new__(CrossSectionalScreener)
    screener.workspace = SimpleNamespace(factor_fields=("open", "close"))
    screener.panel_path = tmp_path

    fields, snapshot, resolved = screener._load_snapshot(pd.Timestamp("2026-07-17").date())

    assert list(fields) == ["open", "close"]
    assert fields["open"].columns.is_unique
    assert snapshot.index.is_unique
    assert resolved == pd.Timestamp("2026-07-17")
