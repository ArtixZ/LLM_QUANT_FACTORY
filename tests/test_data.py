from __future__ import annotations

import json
from pathlib import Path

import duckdb

from multifactor_ashare.data import (
    audit_dataset,
    build_cross_sectional_panel,
    build_hybrid_panel,
    build_panel,
)


def _source_fixture(root: Path) -> Path:
    source = root / "source"
    (source / "csv").mkdir(parents=True)
    (source / "parquet").mkdir()
    rows = [
        ("000001.SZ", "Alpha", 20240102, 10.0, 10.5, 9.8, 10.2, 10.0, 0.2, 2.0, 100.0, 1000.0),
        ("000001.SZ", "Alpha", 20240103, 10.2, 10.8, 10.1, 10.5, 10.2, 0.3, 2.9412, 120.0, 1200.0),
        ("600001.SH", "Beta", 20240102, 20.0, 20.2, 19.8, 20.0, 20.0, 0.0, 0.0, 80.0, 800.0),
    ]
    connection = duckdb.connect()
    try:
        connection.execute(
            """
            CREATE TABLE prices (
                ts_code VARCHAR, name VARCHAR, trade_date BIGINT, open DOUBLE, high DOUBLE,
                low DOUBLE, close DOUBLE, pre_close DOUBLE, change DOUBLE, pct_chg DOUBLE,
                vol DOUBLE, amount DOUBLE
            )
            """
        )
        connection.executemany(
            "INSERT INTO prices VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
        )
        for ts_code in ("000001.SZ", "600001.SH"):
            stem = ts_code.replace(".", "_")
            connection.execute(
                f"COPY (SELECT * FROM prices WHERE ts_code = ?) TO '{source / 'parquet' / f'{stem}.parquet'}' "
                "(FORMAT PARQUET)",
                [ts_code],
            )
            (source / "csv" / f"{stem}.csv").write_text("fixture\n", encoding="utf-8")
    finally:
        connection.close()
    return source


def _cross_sectional_fixture(root: Path) -> Path:
    source = root / "cross"
    market_dir = source / "market_parquet"
    factor_dir = source / "adj_factor_parquet"
    market_dir.mkdir(parents=True)
    factor_dir.mkdir()
    connection = duckdb.connect()
    try:
        connection.execute(
            """
            CREATE TABLE market (
                ts_code VARCHAR, trade_date BIGINT, open DOUBLE, high DOUBLE, low DOUBLE,
                close DOUBLE, pre_close DOUBLE, change DOUBLE, pct_chg DOUBLE, vol DOUBLE, amount DOUBLE
            )
            """
        )
        connection.executemany(
            "INSERT INTO market VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                ("000001.SZ", 20240102, 10.0, 10.2, 9.8, 10.0, 10.0, 0.0, 0.0, 100.0, 1000.0),
                # A two-for-one split: raw price halves while its factor doubles.
                ("000001.SZ", 20240103, 5.0, 5.1, 4.9, 5.0, 5.0, 0.0, 0.0, 120.0, 1200.0),
                ("000001.SZ", 20240104, 5.5, 5.6, 5.4, 5.5, 5.0, 0.5, 10.0, 130.0, 1300.0),
            ],
        )
        connection.execute(
            "CREATE TABLE factors (ts_code VARCHAR, trade_date BIGINT, adj_factor DOUBLE)"
        )
        connection.executemany(
            "INSERT INTO factors VALUES (?, ?, ?)",
            [
                ("000001.SZ", 20240102, 1.0),
                ("000001.SZ", 20240103, 2.0),
                ("000001.SZ", 20240104, 2.0),
            ],
        )
        # Each daily source file intentionally contains a single date.
        for date in (20240102, 20240103, 20240104):
            connection.execute(
                f"COPY (SELECT * FROM market WHERE trade_date = {date}) TO "
                f"'{market_dir / f'{date}.parquet'}' (FORMAT PARQUET)"
            )
        for date in (20240102, 20240103, 20240104):
            connection.execute(
                f"COPY (SELECT * FROM factors WHERE trade_date = {date}) TO "
                f"'{factor_dir / f'{date}.parquet'}' (FORMAT PARQUET)"
            )
        connection.execute(
            f"COPY (SELECT '000001.SZ' AS ts_code, 'Alpha' AS name) TO "
            f"'{source / 'stock_basic.parquet'}' (FORMAT PARQUET)"
        )
    finally:
        connection.close()
    return source


def test_audit_and_build(tmp_path: Path) -> None:
    source = _source_fixture(tmp_path)
    catalog = tmp_path / "catalog.csv"
    report_path = tmp_path / "quality.json"
    output = tmp_path / "panel"

    report = audit_dataset(source, catalog, report_path)
    metadata = build_panel(source, output)

    assert report["passed"] is True
    assert report["summary"]["rows"] == 3
    assert report["files"]["catalog_rows"] == 2
    assert json.loads(report_path.read_text(encoding="utf-8"))["passed"] is True
    assert "parquet/000001_SZ.parquet" in catalog.read_text(encoding="utf-8")
    assert metadata["rows"] == 3
    assert metadata["year_partitions"] == 1

    connection = duckdb.connect()
    try:
        result = connection.execute(
            f"""
            SELECT ret_1d, close_to_close_ret, history_observations, is_tradable_observation
            FROM read_parquet('{output / "**/*.parquet"}', hive_partitioning = true)
            WHERE ts_code = '000001.SZ' AND trade_date = DATE '2024-01-03'
            """
        ).fetchone()
    finally:
        connection.close()
    assert result[0] == result[1]
    assert result[2:] == (2, True)


def test_build_refuses_existing_output(tmp_path: Path) -> None:
    source = _source_fixture(tmp_path)
    output = tmp_path / "panel"
    output.mkdir()

    try:
        build_panel(source, output)
    except FileExistsError as error:
        assert "--overwrite" in str(error)
    else:
        raise AssertionError("Expected existing output to be rejected")


def test_build_hybrid_panel_normalizes_execution_units(tmp_path: Path) -> None:
    research = _source_fixture(tmp_path / "research")
    execution = _source_fixture(tmp_path / "execution")
    output = tmp_path / "workspace/processed/daily_panel"
    catalog = tmp_path / "workspace/catalog/daily_catalog.csv"
    report = tmp_path / "workspace/catalog/data_quality.json"

    metadata = build_hybrid_panel(
        research,
        execution,
        output,
        catalog_path=catalog,
        report_path=report,
    )

    assert metadata["execution_price_adjustment"] == "unadjusted"
    assert metadata["capital_ledger_proxy_ready"] is True
    assert metadata["volume_unit"] == "shares"
    assert metadata["amount_unit"] == "cny"
    connection = duckdb.connect()
    try:
        row = connection.execute(
            f"""
            SELECT open, raw_open, vol, amount, can_buy_open_proxy, can_sell_open_proxy
            FROM read_parquet('{output / "**/*.parquet"}', hive_partitioning = true)
            WHERE ts_code = '000001.SZ' AND trade_date = DATE '2024-01-02'
            """
        ).fetchone()
    finally:
        connection.close()
    assert row == (10.0, 10.0, 10_000.0, 1_000_000.0, True, True)
    quality = json.loads(report.read_text(encoding="utf-8"))
    assert quality["passed"] is True
    assert quality["dataset_kind"].endswith("NON_PIT_PROXY")


def test_cross_sectional_panel_preserves_adjusted_returns_without_future_anchor(tmp_path: Path) -> None:
    source = _cross_sectional_fixture(tmp_path)
    output = tmp_path / "workspace/processed/daily_panel"
    catalog = tmp_path / "workspace/catalog/daily_catalog.csv"
    report = tmp_path / "workspace/catalog/data_quality.json"

    metadata = build_cross_sectional_panel(
        source, output, catalog_path=catalog, report_path=report
    )

    assert metadata["price_adjustment"] == "event_adjusted_pit"
    connection = duckdb.connect()
    try:
        rows = connection.execute(
            f"""
            SELECT CAST(trade_date AS VARCHAR), raw_close, adj_factor, close, round(ret_1d, 8)
            FROM read_parquet('{output / "**/*.parquet"}', hive_partitioning = true)
            ORDER BY trade_date
            """
        ).fetchall()
    finally:
        connection.close()
    # raw*factor is [10, 10, 11]: the corporate action has zero return and the
    # next session has the same +10% return as a forward-adjusted series.
    assert rows == [
        ("2024-01-02", 10.0, 1.0, 10.0, 0.0),
        ("2024-01-03", 5.0, 2.0, 10.0, 0.0),
        ("2024-01-04", 5.5, 2.0, 11.0, 0.1),
    ]
