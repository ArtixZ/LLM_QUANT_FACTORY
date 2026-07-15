from __future__ import annotations

import json
from pathlib import Path

import duckdb

from multifactor_ashare.data import audit_dataset, build_panel


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
