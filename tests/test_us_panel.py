from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pytest

from multifactor_us.data import (
    AMOUNT_UNIT,
    CURRENCY,
    EXECUTION_PRICE_ADJUSTMENT,
    MARKET,
    PRICE_ADJUSTMENT,
    VOLUME_UNIT,
    PanelBuildError,
    audit_dataset,
    build_panel,
    write_catalog,
)

SLICE_ROWS = [
    # symbol, trade_date, adjusted OHLC, adj_close, raw OHLC, raw_pre_close, vol, amount
    ("AAPL", "2025-01-02", 99.0, 101.0, 98.0, 100.0, 100.0, 199.0, 201.0, 198.0, 200.0, 199.0),
    ("AAPL", "2025-01-03", 100.0, 102.0, 99.0, 101.0, 101.0, 200.0, 202.0, 199.0, 201.0, 200.0),
    ("AAPL", "2026-01-05", 102.0, 104.0, 101.0, 103.0, 103.0, 202.0, 204.0, 201.0, 203.0, 201.0),
    ("MSFT", "2025-01-02", 49.0, 51.0, 48.0, 50.0, 50.0, 99.0, 101.0, 98.0, 100.0, 99.0),
    ("MSFT", "2025-01-03", 50.0, 52.0, 49.0, 51.0, 51.0, 100.0, 102.0, 99.0, 101.0, 100.0),
]


def _write_slices(root: Path, *, corrupt: bool = False, drop_column: bool = False) -> Path:
    downloads = root / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    try:
        connection.execute(
            """
            CREATE TABLE slice (
                symbol VARCHAR, trade_date DATE,
                open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, adj_close DOUBLE,
                raw_open DOUBLE, raw_high DOUBLE, raw_low DOUBLE, raw_close DOUBLE,
                raw_pre_close DOUBLE, vol DOUBLE, amount DOUBLE,
                listing_date DATE, delisting_date DATE,
                is_valid_ohlc BOOLEAN, is_tradable_observation BOOLEAN,
                can_buy_open BOOLEAN, can_sell_open BOOLEAN, is_halted BOOLEAN
            )
            """
        )
        for row in SLICE_ROWS:
            symbol, trade_date = row[0], row[1]
            prices = row[2:]
            connection.execute(
                """
                INSERT INTO slice VALUES (
                    ?, CAST(? AS DATE), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    1000.0, 100000.0, CAST('2025-01-02' AS DATE), NULL,
                    TRUE, TRUE, TRUE, TRUE, FALSE
                )
                """,
                [symbol, trade_date, *prices],
            )
        if corrupt:
            connection.execute(
                """
                INSERT INTO slice VALUES (
                    'AAPL', CAST('2025-01-02' AS DATE), 1,1,1,1,1,1,1,1,1,1,
                    1.0, 1.0, CAST('2025-01-02' AS DATE), NULL, TRUE, TRUE, TRUE, TRUE, FALSE
                )
                """
            )
        columns = "* EXCLUDE (adj_close)" if drop_column else "*"
        connection.execute(
            f"COPY (SELECT {columns} FROM slice) TO '{downloads / 'slices.parquet'}' (FORMAT PARQUET)"
        )
    finally:
        connection.close()
    return downloads


def test_audit_passes_on_well_formed_slices(tmp_path: Path) -> None:
    source = _write_slices(tmp_path)
    report = audit_dataset(source, tmp_path / "quality.json")
    assert report["passed"] is True
    assert report["market"] == MARKET
    assert report["summary"]["rows"] == 5
    assert report["summary"]["symbols_with_rows"] == 2
    assert report["failures"] == {"null_keys": 0, "duplicate_keys": 0, "null_close": 0}
    assert json.loads((tmp_path / "quality.json").read_text())["passed"] is True


def test_audit_detects_duplicate_keys(tmp_path: Path) -> None:
    source = _write_slices(tmp_path, corrupt=True)
    report = audit_dataset(source, tmp_path / "quality.json")
    assert report["passed"] is False
    assert report["failures"]["duplicate_keys"] == 1


def test_audit_rejects_a_missing_contract_column(tmp_path: Path) -> None:
    source = _write_slices(tmp_path, drop_column=True)
    with pytest.raises(PanelBuildError, match="adj_close"):
        audit_dataset(source, None)


def test_audit_requires_source_slices(tmp_path: Path) -> None:
    (tmp_path / "downloads").mkdir()
    with pytest.raises(FileNotFoundError, match="No parquet slices"):
        audit_dataset(tmp_path / "downloads", None)


def test_build_panel_partitions_by_trade_year(tmp_path: Path) -> None:
    source = _write_slices(tmp_path)
    output = tmp_path / "processed" / "daily_panel"
    metadata = build_panel(source, output)
    assert metadata["rows"] == 5
    assert metadata["symbols"] == 2
    assert metadata["year_partitions"] == 2
    assert sorted(p.name for p in output.glob("trade_year=*")) == [
        "trade_year=2025",
        "trade_year=2026",
    ]


def test_panel_metadata_declares_us_units(tmp_path: Path) -> None:
    source = _write_slices(tmp_path)
    output = tmp_path / "panel"
    build_panel(source, output)
    metadata = json.loads((output / "_metadata.json").read_text())
    assert metadata["market"] == MARKET
    assert metadata["currency"] == CURRENCY
    assert metadata["volume_unit"] == VOLUME_UNIT
    assert metadata["amount_unit"] == AMOUNT_UNIT
    assert metadata["price_adjustment"] == PRICE_ADJUSTMENT
    assert metadata["execution_price_adjustment"] == EXECUTION_PRICE_ADJUSTMENT
    assert metadata["institutional_pit_ready"] is False
    assert metadata["capital_ledger_ready"] is False
    assert metadata["capital_ledger_proxy_ready"] is True
    assert any("point-in-time" in caveat for caveat in metadata["caveats"])


def test_build_panel_computes_returns_per_symbol(tmp_path: Path) -> None:
    source = _write_slices(tmp_path)
    output = tmp_path / "panel"
    build_panel(source, output)
    connection = duckdb.connect()
    try:
        rows = connection.execute(
            f"""
            SELECT symbol, trade_date, ret_1d, history_observations
            FROM read_parquet('{output / "**/*.parquet"}', hive_partitioning = true)
            WHERE symbol = 'AAPL' ORDER BY trade_date
            """
        ).fetchall()
    finally:
        connection.close()
    assert rows[0][2] is None  # first observation has no prior close
    assert rows[1][2] == pytest.approx(101.0 / 100.0 - 1)
    assert [row[3] for row in rows] == [1, 2, 3]


def test_build_panel_refuses_to_clobber_without_overwrite(tmp_path: Path) -> None:
    source = _write_slices(tmp_path)
    output = tmp_path / "panel"
    build_panel(source, output)
    with pytest.raises(FileExistsError, match="--overwrite"):
        build_panel(source, output)
    metadata = build_panel(source, output, overwrite=True)
    assert metadata["rows"] == 5


def test_write_catalog_records_one_row_per_symbol(tmp_path: Path) -> None:
    source = _write_slices(tmp_path)
    destination = tmp_path / "catalog" / "daily_catalog.csv"
    assert write_catalog(source, destination) == {"symbols": 2}
    lines = destination.read_text().strip().splitlines()
    assert lines[0].startswith("symbol,rows,first_trade_date,last_trade_date")
    assert len(lines) == 3


def _write_stale_slice(root: Path) -> None:
    """Add a symbol whose history stops before the rest of the panel."""
    connection = duckdb.connect()
    try:
        connection.execute(
            """
            COPY (
                SELECT 'STALE' AS symbol, CAST('2025-01-02' AS DATE) AS trade_date,
                    99.0 AS open, 101.0 AS high, 98.0 AS low, 100.0 AS close,
                    100.0 AS adj_close, 199.0 AS raw_open, 201.0 AS raw_high,
                    198.0 AS raw_low, 200.0 AS raw_close, 199.0 AS raw_pre_close,
                    1000.0 AS vol, 100000.0 AS amount,
                    CAST('2025-01-02' AS DATE) AS listing_date,
                    CAST(NULL AS DATE) AS delisting_date,
                    TRUE AS is_valid_ohlc, TRUE AS is_tradable_observation,
                    TRUE AS can_buy_open, TRUE AS can_sell_open, FALSE AS is_halted
            ) TO '%s' (FORMAT PARQUET)
            """
            % (root / "downloads" / "stale.parquet")
        )
    finally:
        connection.close()


def test_audit_flags_symbols_left_behind_by_a_partial_sync(tmp_path: Path) -> None:
    """Slices refresh independently; a stale symbol becomes NaN in a cross-section."""
    source = _write_slices(tmp_path)
    _write_stale_slice(tmp_path)
    report = audit_dataset(source, None)

    assert report["passed"] is True  # staleness is a warning, not a contract failure
    assert "STALE" in report["stale_symbols"]
    assert report["stale_symbols"]["STALE"].startswith("2025-01-02")


def test_audit_flags_the_ragged_history_in_the_base_fixture(tmp_path: Path) -> None:
    """MSFT stops at 2025-01-03 while AAPL runs to 2026-01-05."""
    source = _write_slices(tmp_path)
    report = audit_dataset(source, None)

    assert report["warnings"]["stale_symbols"] == 1
    assert report["stale_symbols"]["MSFT"].startswith("2025-01-03")
    assert "AAPL" not in report["stale_symbols"]
