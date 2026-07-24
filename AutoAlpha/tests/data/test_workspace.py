from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from autoalpha.data.workspace import inspect_data_workspace


def _workspace(tmp_path: Path, *, quality_passed: bool = True) -> Path:
    root = tmp_path / "data"
    panel = root / "processed/daily_panel/trade_year=2024"
    panel.mkdir(parents=True)
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
    frame.to_parquet(panel / "data.parquet")
    source = root / "source"
    source.mkdir()
    catalog = root / "catalog"
    catalog.mkdir()
    (catalog / "daily_catalog.csv").write_text("ts_code\n000001.SZ\n", encoding="utf-8")
    (catalog / "data_quality.json").write_text(
        json.dumps(
            {
                "source": "source",
                "passed": quality_passed,
                "summary": {
                    "rows": 1,
                    "symbols_with_rows": 1,
                    "first_trade_date": "20240102",
                    "last_trade_date": "20240102",
                },
            }
        ),
        encoding="utf-8",
    )
    (panel.parent / "_metadata.json").write_text(
        json.dumps({"rows": 1, "symbols": 1}), encoding="utf-8"
    )
    return root


def test_workspace_root_resolves_panel_catalog_source_and_fingerprint(tmp_path: Path) -> None:
    root = _workspace(tmp_path)

    report = inspect_data_workspace(root)

    assert report.price_research_ready
    assert not report.institutional_pit_ready
    assert report.quality_passed is True
    assert report.source_integrity_passed
    assert report.panel_path.endswith("processed/daily_panel")
    assert report.source_path and report.source_path.endswith("source")
    assert report.catalog_path and report.catalog_path.endswith("daily_catalog.csv")
    assert len(report.fingerprint) == 64


def test_direct_panel_path_recovers_workspace_root(tmp_path: Path) -> None:
    root = _workspace(tmp_path)

    report = inspect_data_workspace(root / "processed/daily_panel")

    assert report.root_path == str(root.resolve())
    assert report.quality_passed is True


def test_failed_source_quality_blocks_price_research(tmp_path: Path) -> None:
    root = _workspace(tmp_path, quality_passed=False)

    report = inspect_data_workspace(root)

    assert not report.price_research_ready
    assert not report.source_integrity_passed
    with pytest.raises(RuntimeError, match="quality report failed"):
        report.require_price_research()


def test_source_and_panel_row_mismatch_blocks_research(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    path = root / "catalog/data_quality.json"
    quality = json.loads(path.read_text(encoding="utf-8"))
    quality["summary"]["rows"] = 2
    path.write_text(json.dumps(quality), encoding="utf-8")

    report = inspect_data_workspace(root)

    assert not report.source_integrity_passed
    assert any("quality rows mismatch" in blocker for blocker in report.blockers)
