from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

DEFAULT_SOURCE = Path("data/mainboard_non_st_qfq_20100101_20260715")
DEFAULT_OUTPUT = Path("data/processed/daily_panel")
DEFAULT_CATALOG = Path("data/catalog/daily_catalog.csv")
DEFAULT_REPORT = Path("data/catalog/data_quality.json")
DEFAULT_MARKET_DATA_ROOT = Path.home() / "MarketData/Ashare/data/downloads"
REQUIRED_COLUMNS = (
    "ts_code",
    "name",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "change",
    "pct_chg",
    "vol",
    "amount",
)


def _parquet_glob(source: Path) -> str:
    parquet_dir = source.resolve() / "parquet"
    if not parquet_dir.is_dir():
        raise FileNotFoundError(f"Parquet source directory not found: {parquet_dir}")
    return str(parquet_dir / "*.parquet")


def _sql_string(value: str | Path) -> str:
    return str(value).replace("'", "''")


def _query_one(connection: duckdb.DuckDBPyConnection, sql: str) -> dict[str, Any]:
    cursor = connection.execute(sql)
    row = cursor.fetchone()
    return dict(zip((item[0] for item in cursor.description), row, strict=True))


def _read_schema(connection: duckdb.DuckDBPyConnection, parquet_glob: str) -> list[str]:
    rows = connection.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{_sql_string(parquet_glob)}', union_by_name = true)"
    ).fetchall()
    return [row[0] for row in rows]


def _raw_relation(parquet_glob: str, *, filename: bool = False) -> str:
    filename_option = ", filename = true" if filename else ""
    return f"read_parquet('{_sql_string(parquet_glob)}', union_by_name = true{filename_option})"


def _write_catalog(
    connection: duckdb.DuckDBPyConnection,
    source: Path,
    parquet_glob: str,
    destination: Path,
) -> dict[str, int]:
    parquet_files = {path.stem: path for path in (source / "parquet").glob("*.parquet")}
    csv_files = {path.stem: path for path in (source / "csv").glob("*.csv")}

    observed = connection.execute(
        f"""
        SELECT
            regexp_extract(filename, '([^/]+)\\.parquet$', 1) AS stem,
            min(CAST(trade_date AS VARCHAR)) AS first_trade_date,
            max(CAST(trade_date AS VARCHAR)) AS last_trade_date,
            count(*) AS rows,
            arg_max(ts_code, trade_date) AS ts_code,
            arg_max(name, trade_date) AS name
        FROM {_raw_relation(parquet_glob, filename=True)}
        GROUP BY filename
        ORDER BY stem
        """
    ).fetchall()
    by_stem = {row[0]: row[1:] for row in observed}

    destination.parent.mkdir(parents=True, exist_ok=True)
    all_stems = sorted(parquet_files.keys() | csv_files.keys())
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "ts_code",
                "name",
                "first_trade_date",
                "last_trade_date",
                "rows",
                "csv_path",
                "parquet_path",
                "has_csv",
                "has_parquet",
            ]
        )
        for stem in all_stems:
            first_date, last_date, rows, ts_code, name = by_stem.get(
                stem, (None, None, 0, None, None)
            )
            writer.writerow(
                [
                    ts_code or stem.replace("_", "."),
                    name or "",
                    first_date or "",
                    last_date or "",
                    rows,
                    str(csv_files[stem].relative_to(source)) if stem in csv_files else "",
                    (str(parquet_files[stem].relative_to(source)) if stem in parquet_files else ""),
                    stem in csv_files,
                    stem in parquet_files,
                ]
            )
    return {
        "catalog_rows": len(all_stems),
        "csv_files": len(csv_files),
        "parquet_files": len(parquet_files),
        "zero_row_parquet_files": len(parquet_files) - len(by_stem),
    }


def audit_dataset(source: Path, catalog_path: Path, report_path: Path) -> dict[str, Any]:
    source = source.resolve()
    parquet_glob = _parquet_glob(source)
    connection = duckdb.connect()
    try:
        columns = _read_schema(connection, parquet_glob)
        missing_columns = sorted(set(REQUIRED_COLUMNS) - set(columns))
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

        relation = _raw_relation(parquet_glob)
        summary = _query_one(
            connection,
            f"""
            SELECT
                count(*) AS rows,
                count(DISTINCT ts_code) AS symbols_with_rows,
                min(CAST(trade_date AS VARCHAR)) AS first_trade_date,
                max(CAST(trade_date AS VARCHAR)) AS last_trade_date,
                count(*) FILTER (WHERE ts_code IS NULL OR trade_date IS NULL) AS null_keys,
                count(*) FILTER (
                    WHERE open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL
                       OR pre_close IS NULL OR vol IS NULL OR amount IS NULL
                ) AS null_market_values,
                count(*) FILTER (
                    WHERE high < greatest(open, close, low)
                       OR low > least(open, close, high)
                       OR open <= 0 OR high <= 0 OR low <= 0 OR close <= 0
                ) AS invalid_ohlc_rows,
                count(*) FILTER (WHERE vol < 0 OR amount < 0) AS negative_activity_rows,
                count(*) FILTER (
                    WHERE pre_close <> 0
                      AND abs((close / pre_close - 1) - pct_chg / 100.0) > 0.0005
                ) AS return_mismatch_rows
            FROM {relation}
            """,
        )
        duplicate_keys = connection.execute(
            f"""
            SELECT count(*)
            FROM (
                SELECT ts_code, trade_date
                FROM {relation}
                GROUP BY ts_code, trade_date
                HAVING count(*) > 1
            )
            """
        ).fetchone()[0]
        catalog = _write_catalog(connection, source, parquet_glob, catalog_path)
    finally:
        connection.close()

    checks = {
        "duplicate_keys": duplicate_keys,
        **{
            key: summary[key]
            for key in (
                "null_keys",
                "null_market_values",
                "invalid_ohlc_rows",
                "negative_activity_rows",
                "return_mismatch_rows",
            )
        },
    }
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": os.path.relpath(source, Path.cwd()),
        "schema": columns,
        "summary": {
            key: summary[key]
            for key in ("rows", "symbols_with_rows", "first_trade_date", "last_trade_date")
        },
        "files": catalog,
        "checks": checks,
        "passed": all(value == 0 for value in checks.values()),
        "known_limitations": [
            "No point-in-time ST status or historical index/universe membership.",
            "Missing rows cannot distinguish suspension, pre-listing, and post-delisting periods.",
            "The snapshot includes names containing the delisting marker 退.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def build_panel(source: Path, output: Path, *, overwrite: bool = False) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    parquet_glob = _parquet_glob(source)
    if output.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {output}; pass --overwrite to replace it"
            )

    staging = output.with_name(f".{output.name}.staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.parent.mkdir(parents=True, exist_ok=True)

    relation = _raw_relation(parquet_glob)
    trade_date = "strptime(CAST(trade_date AS VARCHAR), '%Y%m%d')::DATE"
    connection = duckdb.connect()
    try:
        connection.execute(
            f"""
            COPY (
                WITH typed AS (
                    SELECT
                        CAST(ts_code AS VARCHAR) AS ts_code,
                        CAST(name AS VARCHAR) AS name,
                        {trade_date} AS trade_date,
                        CAST(open AS DOUBLE) AS open,
                        CAST(high AS DOUBLE) AS high,
                        CAST(low AS DOUBLE) AS low,
                        CAST(close AS DOUBLE) AS close,
                        CAST(close AS DOUBLE) AS adj_close,
                        CAST(pre_close AS DOUBLE) AS pre_close,
                        CAST(change AS DOUBLE) AS change,
                        CAST(pct_chg AS DOUBLE) AS pct_chg,
                        CAST(vol AS DOUBLE) AS vol,
                        CAST(amount AS DOUBLE) AS amount
                    FROM {relation}
                ),
                enriched AS (
                    SELECT
                        *,
                        year(trade_date) AS trade_year,
                        close / nullif(pre_close, 0) - 1 AS ret_1d,
                        close / nullif(
                            lag(close) OVER (PARTITION BY ts_code ORDER BY trade_date), 0
                        ) - 1 AS close_to_close_ret,
                        date_diff(
                            'day',
                            lag(trade_date) OVER (PARTITION BY ts_code ORDER BY trade_date),
                            trade_date
                        ) AS observation_gap_days,
                        row_number() OVER (
                            PARTITION BY ts_code ORDER BY trade_date
                        ) AS history_observations,
                        (
                            open > 0 AND high > 0 AND low > 0 AND close > 0
                            AND high >= greatest(open, close, low)
                            AND low <= least(open, close, high)
                        ) AS is_valid_ohlc,
                        (coalesce(vol, 0) > 0 AND coalesce(amount, 0) > 0) AS has_activity
                    FROM typed
                )
                SELECT
                    *,
                    (is_valid_ohlc AND has_activity) AS is_tradable_observation
                FROM enriched
                ORDER BY trade_year, trade_date, ts_code
            ) TO '{_sql_string(staging)}' (
                FORMAT PARQUET,
                PARTITION_BY (trade_year),
                COMPRESSION ZSTD,
                ROW_GROUP_SIZE 100000
            )
            """
        )
        stats = _query_one(
            connection,
            f"""
            SELECT
                count(*) AS rows,
                count(DISTINCT ts_code) AS symbols,
                min(trade_date) AS first_trade_date,
                max(trade_date) AS last_trade_date,
                count(DISTINCT trade_year) AS year_partitions
            FROM read_parquet('{_sql_string(staging / "**/*.parquet")}', hive_partitioning = true)
            """,
        )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        connection.close()

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": os.path.relpath(source, output.parent),
        "format": "parquet",
        "partitioning": ["trade_year"],
        "price_adjustment": "forward_adjusted",
        "volume_unit": "board_lot_100_shares",
        "amount_unit": "thousand_cny",
        "capital_ledger_ready": False,
        **{
            key: (value.isoformat() if hasattr(value, "isoformat") else value)
            for key, value in stats.items()
        },
    }
    (staging / "_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    backup = output.with_name(f".{output.name}.backup")
    if backup.exists():
        shutil.rmtree(backup)
    if output.exists():
        output.rename(backup)
    try:
        staging.rename(output)
    except Exception:
        if backup.exists():
            backup.rename(output)
        raise
    shutil.rmtree(backup, ignore_errors=True)
    return metadata


def discover_download_source(download_root: Path, adjustment: str) -> Path:
    suffix = f"_{adjustment}_csv-parquet"
    candidates = [
        path
        for path in download_root.expanduser().resolve().glob("a_daily_*")
        if path.name.endswith(suffix) and (path / "parquet").is_dir()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No A-share {adjustment} download task found under {download_root}"
        )
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))


def _cross_sectional_glob(source: Path, directory: str) -> str:
    parquet_dir = source.resolve() / directory
    if not parquet_dir.is_dir():
        raise FileNotFoundError(f"Cross-sectional parquet directory not found: {parquet_dir}")
    return str(parquet_dir / "*.parquet")


def _cross_sectional_market_relation(source: Path) -> tuple[str, list[str]]:
    globs = [_cross_sectional_glob(source, "market_parquet")]
    manifest_path = source / "market_legacy_source.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
        legacy_root = Path(str(manifest.get("parquet_root", ""))).expanduser()
        if legacy_root.is_dir():
            globs.insert(0, str(legacy_root / "*.parquet"))
    sources = ", ".join(f"'{_sql_string(path)}'" for path in globs)
    return f"read_parquet([{sources}], union_by_name = true)", globs


def build_cross_sectional_panel(
    source: Path,
    output: Path,
    *,
    catalog_path: Path,
    report_path: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build a PIT-safe research panel from raw daily bars and daily adj factors.

    ``raw_price * adj_factor`` is an event-adjusted series whose return between
    two observations needs no future anchor.  It intentionally replaces the
    mutable ``pro_bar(adj='qfq')`` level series used by the legacy pipeline.
    """
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    market_relation, market_sources = _cross_sectional_market_relation(source)
    factor_glob = _cross_sectional_glob(source, "adj_factor_parquet")
    master_path = source / "stock_basic.parquet"
    if not master_path.is_file():
        raise FileNotFoundError(f"Cross-sectional stock master not found: {master_path}")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}; pass --overwrite to replace it")

    staging = output.with_name(f".{output.name}.staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.parent.mkdir(parents=True, exist_ok=True)
    trade_date = "strptime(CAST(trade_date AS VARCHAR), '%Y%m%d')::DATE"
    connection = duckdb.connect()
    try:
        connection.execute(
            f"""
            COPY (
                WITH stock_master AS (
                    SELECT CAST(ts_code AS VARCHAR) AS ts_code, any_value(CAST(name AS VARCHAR)) AS name
                    FROM read_parquet('{_sql_string(master_path)}')
                    GROUP BY ts_code
                ),
                market AS (
                    SELECT
                        CAST(d.ts_code AS VARCHAR) AS ts_code,
                        coalesce(m.name, CAST(d.ts_code AS VARCHAR)) AS name,
                        {trade_date} AS trade_date,
                        CAST(d.open AS DOUBLE) AS raw_open,
                        CAST(d.high AS DOUBLE) AS raw_high,
                        CAST(d.low AS DOUBLE) AS raw_low,
                        CAST(d.close AS DOUBLE) AS raw_close,
                        CAST(d.pre_close AS DOUBLE) AS raw_pre_close,
                        CAST(d.vol AS DOUBLE) * 100.0 AS vol,
                        CAST(d.amount AS DOUBLE) * 1000.0 AS amount
                    FROM {market_relation} AS d
                    LEFT JOIN stock_master AS m USING (ts_code)
                ),
                factors AS (
                    SELECT
                        CAST(ts_code AS VARCHAR) AS ts_code,
                        {trade_date} AS trade_date,
                        CAST(adj_factor AS DOUBLE) AS adj_factor
                    FROM read_parquet('{_sql_string(factor_glob)}', union_by_name = true)
                ),
                adjusted AS (
                    SELECT
                        market.*,
                        factors.adj_factor,
                        raw_open * adj_factor AS open,
                        raw_high * adj_factor AS high,
                        raw_low * adj_factor AS low,
                        raw_close * adj_factor AS close,
                        coalesce(
                            lag(raw_close * adj_factor) OVER (
                                PARTITION BY market.ts_code ORDER BY market.trade_date
                            ),
                            raw_pre_close * adj_factor
                        ) AS pre_close
                    FROM market
                    INNER JOIN factors USING (ts_code, trade_date)
                    WHERE NOT (
                        market.ts_code LIKE '%.BJ'
                        OR split_part(market.ts_code, '.', 1) LIKE '4%'
                        OR split_part(market.ts_code, '.', 1) LIKE '8%'
                        OR split_part(market.ts_code, '.', 1) LIKE '300%'
                        OR split_part(market.ts_code, '.', 1) LIKE '301%'
                        OR split_part(market.ts_code, '.', 1) LIKE '688%'
                        OR split_part(market.ts_code, '.', 1) LIKE '689%'
                        OR upper(market.name) LIKE '%ST%'
                    )
                ),
                enriched AS (
                    SELECT
                        *,
                        close AS adj_close,
                        close - pre_close AS change,
                        100.0 * (close / nullif(pre_close, 0) - 1) AS pct_chg,
                        year(trade_date) AS trade_year,
                        close / nullif(pre_close, 0) - 1 AS ret_1d,
                        close / nullif(
                            lag(close) OVER (PARTITION BY ts_code ORDER BY trade_date), 0
                        ) - 1 AS close_to_close_ret,
                        date_diff(
                            'day',
                            lag(trade_date) OVER (PARTITION BY ts_code ORDER BY trade_date),
                            trade_date
                        ) AS observation_gap_days,
                        row_number() OVER (PARTITION BY ts_code ORDER BY trade_date) AS history_observations,
                        (
                            open > 0 AND high > 0 AND low > 0 AND close > 0
                            AND high >= greatest(open, close, low)
                            AND low <= least(open, close, high)
                            AND raw_open > 0 AND raw_high > 0 AND raw_low > 0 AND raw_close > 0
                            AND raw_high >= greatest(raw_open, raw_close, raw_low)
                            AND raw_low <= least(raw_open, raw_close, raw_high)
                        ) AS is_valid_ohlc,
                        (coalesce(vol, 0) > 0 AND coalesce(amount, 0) > 0) AS has_activity
                    FROM adjusted
                )
                SELECT
                    *,
                    (is_valid_ohlc AND has_activity) AS is_tradable_observation,
                    (is_valid_ohlc AND has_activity AND raw_open < raw_pre_close * 1.095) AS can_buy_open_proxy,
                    (is_valid_ohlc AND has_activity AND raw_open > raw_pre_close * 0.905) AS can_sell_open_proxy
                FROM enriched
                ORDER BY trade_year, trade_date, ts_code
            ) TO '{_sql_string(staging)}' (
                FORMAT PARQUET,
                PARTITION_BY (trade_year),
                COMPRESSION ZSTD,
                ROW_GROUP_SIZE 100000
            )
            """
        )
        panel_relation = (
            f"read_parquet('{_sql_string(staging / '**/*.parquet')}', hive_partitioning = true)"
        )
        stats = _query_one(
            connection,
            f"""
            SELECT
                count(*) AS rows,
                count(DISTINCT ts_code) AS symbols,
                min(trade_date) AS first_trade_date,
                max(trade_date) AS last_trade_date,
                count(DISTINCT trade_year) AS year_partitions,
                count(*) FILTER (WHERE NOT is_valid_ohlc) AS invalid_ohlc_rows,
                count(*) FILTER (WHERE NOT has_activity) AS missing_activity_rows
            FROM {panel_relation}
            """,
        )
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        connection.execute(
            f"""
            COPY (
                SELECT ts_code, arg_max(name, trade_date) AS name,
                       min(trade_date) AS first_trade_date, max(trade_date) AS last_trade_date,
                       count(*) AS rows
                FROM {panel_relation}
                GROUP BY ts_code
                ORDER BY ts_code
            ) TO '{_sql_string(catalog_path)}' (HEADER, DELIMITER ',')
            """
        )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        connection.close()

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": os.path.relpath(source, output.parent),
        "sources": {
            "market_unadjusted_cross_section": str(source / "market_parquet"),
            "market_legacy_bridge": market_sources[0] if len(market_sources) > 1 else None,
            "adj_factor_cross_section": str(source / "adj_factor_parquet"),
            "stock_master": str(master_path),
        },
        "format": "parquet",
        "partitioning": ["trade_year"],
        "price_adjustment": "event_adjusted_pit",
        "research_price_formula": "raw_ohlc * adj_factor_same_session",
        "execution_price_adjustment": "unadjusted",
        "execution_price_fields": ["raw_open", "raw_high", "raw_low", "raw_close"],
        "volume_unit": "shares",
        "amount_unit": "cny",
        "execution_state_mode": "NON_PIT_PROXY",
        "capital_ledger_ready": False,
        "capital_ledger_proxy_ready": True,
        "universe_mode": "CURRENT_NAME_MAINBOARD_NON_ST_PROXY",
        **{
            key: (value.isoformat() if hasattr(value, "isoformat") else value)
            for key, value in stats.items()
            if key not in {"invalid_ohlc_rows", "missing_activity_rows"}
        },
    }
    (staging / "_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    backup = output.with_name(f".{output.name}.backup")
    if backup.exists():
        shutil.rmtree(backup)
    if output.exists():
        output.rename(backup)
    try:
        staging.rename(output)
    except Exception:
        if backup.exists():
            backup.rename(output)
        raise
    shutil.rmtree(backup, ignore_errors=True)

    checks = {
        "invalid_ohlc_rows": int(stats["invalid_ohlc_rows"]),
        "missing_activity_rows": int(stats["missing_activity_rows"]),
    }
    report = {
        "generated_at_utc": metadata["generated_at_utc"],
        "dataset_kind": "PIT_EVENT_ADJUSTED_RESEARCH_UNADJUSTED_EXECUTION_NON_PIT_PROXY",
        "source": metadata["source"],
        "sources": metadata["sources"],
        "summary": {
            "rows": metadata["rows"],
            "symbols_with_rows": metadata["symbols"],
            "first_trade_date": metadata["first_trade_date"],
            "last_trade_date": metadata["last_trade_date"],
        },
        "checks": checks,
        "passed": all(value == 0 for value in checks.values()),
        "known_limitations": [
            "No point-in-time ST, listing, delisting, suspension, or board membership state.",
            "Opening eligibility is a 9.5% main-board non-ST proxy derived from raw prices.",
            "The universe excludes boards and ST names using current identifiers and names.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def build_hybrid_panel(
    research_source: Path,
    execution_source: Path,
    output: Path,
    *,
    catalog_path: Path,
    report_path: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build a qfq research + unadjusted execution panel without PIT market state."""
    research_source = research_source.expanduser().resolve()
    execution_source = execution_source.expanduser().resolve()
    output = output.expanduser().resolve()
    research_glob = _parquet_glob(research_source)
    execution_glob = _parquet_glob(execution_source)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}; pass --overwrite to replace it")

    staging = output.with_name(f".{output.name}.staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.parent.mkdir(parents=True, exist_ok=True)
    trade_date = "strptime(CAST(trade_date AS VARCHAR), '%Y%m%d')::DATE"
    connection = duckdb.connect()
    try:
        connection.execute(
            f"""
            COPY (
                WITH research AS (
                    SELECT
                        CAST(ts_code AS VARCHAR) AS ts_code,
                        CAST(name AS VARCHAR) AS name,
                        {trade_date} AS trade_date,
                        CAST(open AS DOUBLE) AS open,
                        CAST(high AS DOUBLE) AS high,
                        CAST(low AS DOUBLE) AS low,
                        CAST(close AS DOUBLE) AS close,
                        CAST(pre_close AS DOUBLE) AS pre_close,
                        CAST(change AS DOUBLE) AS change,
                        CAST(pct_chg AS DOUBLE) AS pct_chg
                    FROM {_raw_relation(research_glob)}
                ),
                execution AS (
                    SELECT
                        CAST(ts_code AS VARCHAR) AS ts_code,
                        {trade_date} AS trade_date,
                        CAST(open AS DOUBLE) AS raw_open,
                        CAST(high AS DOUBLE) AS raw_high,
                        CAST(low AS DOUBLE) AS raw_low,
                        CAST(close AS DOUBLE) AS raw_close,
                        CAST(pre_close AS DOUBLE) AS raw_pre_close,
                        CAST(vol AS DOUBLE) * 100.0 AS vol,
                        CAST(amount AS DOUBLE) * 1000.0 AS amount
                    FROM {_raw_relation(execution_glob)}
                ),
                joined AS (
                    SELECT research.*, execution.* EXCLUDE (ts_code, trade_date)
                    FROM research
                    INNER JOIN execution USING (ts_code, trade_date)
                    WHERE NOT (
                        research.ts_code LIKE '%.BJ'
                        OR split_part(research.ts_code, '.', 1) LIKE '4%'
                        OR split_part(research.ts_code, '.', 1) LIKE '8%'
                        OR split_part(research.ts_code, '.', 1) LIKE '300%'
                        OR split_part(research.ts_code, '.', 1) LIKE '301%'
                        OR split_part(research.ts_code, '.', 1) LIKE '688%'
                        OR split_part(research.ts_code, '.', 1) LIKE '689%'
                        OR upper(research.name) LIKE '%ST%'
                    )
                ),
                enriched AS (
                    SELECT
                        *,
                        close AS adj_close,
                        year(trade_date) AS trade_year,
                        close / nullif(pre_close, 0) - 1 AS ret_1d,
                        close / nullif(
                            lag(close) OVER (PARTITION BY ts_code ORDER BY trade_date), 0
                        ) - 1 AS close_to_close_ret,
                        date_diff(
                            'day',
                            lag(trade_date) OVER (PARTITION BY ts_code ORDER BY trade_date),
                            trade_date
                        ) AS observation_gap_days,
                        row_number() OVER (
                            PARTITION BY ts_code ORDER BY trade_date
                        ) AS history_observations,
                        (
                            open > 0 AND high > 0 AND low > 0 AND close > 0
                            AND high >= greatest(open, close, low)
                            AND low <= least(open, close, high)
                            AND raw_open > 0 AND raw_high > 0 AND raw_low > 0 AND raw_close > 0
                            AND raw_high >= greatest(raw_open, raw_close, raw_low)
                            AND raw_low <= least(raw_open, raw_close, raw_high)
                        ) AS is_valid_ohlc,
                        (coalesce(vol, 0) > 0 AND coalesce(amount, 0) > 0) AS has_activity
                    FROM joined
                )
                SELECT
                    *,
                    (is_valid_ohlc AND has_activity) AS is_tradable_observation,
                    (
                        is_valid_ohlc AND has_activity
                        AND raw_open < raw_pre_close * 1.095
                    ) AS can_buy_open_proxy,
                    (
                        is_valid_ohlc AND has_activity
                        AND raw_open > raw_pre_close * 0.905
                    ) AS can_sell_open_proxy
                FROM enriched
                ORDER BY trade_year, trade_date, ts_code
            ) TO '{_sql_string(staging)}' (
                FORMAT PARQUET,
                PARTITION_BY (trade_year),
                COMPRESSION ZSTD,
                ROW_GROUP_SIZE 100000
            )
            """
        )
        panel_relation = (
            f"read_parquet('{_sql_string(staging / '**/*.parquet')}', "
            "hive_partitioning = true)"
        )
        stats = _query_one(
            connection,
            f"""
            SELECT
                count(*) AS rows,
                count(DISTINCT ts_code) AS symbols,
                min(trade_date) AS first_trade_date,
                max(trade_date) AS last_trade_date,
                count(DISTINCT trade_year) AS year_partitions,
                count(*) FILTER (WHERE NOT is_valid_ohlc) AS invalid_ohlc_rows,
                count(*) FILTER (WHERE NOT has_activity) AS missing_activity_rows
            FROM {panel_relation}
            """,
        )
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        connection.execute(
            f"""
            COPY (
                SELECT
                    ts_code,
                    arg_max(name, trade_date) AS name,
                    min(trade_date) AS first_trade_date,
                    max(trade_date) AS last_trade_date,
                    count(*) AS rows
                FROM {panel_relation}
                GROUP BY ts_code
                ORDER BY ts_code
            ) TO '{_sql_string(catalog_path)}' (HEADER, DELIMITER ',')
            """
        )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        connection.close()

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": os.path.relpath(research_source, output.parent),
        "sources": {
            "research_qfq": str(research_source),
            "execution_unadjusted": str(execution_source),
        },
        "format": "parquet",
        "partitioning": ["trade_year"],
        "price_adjustment": "forward_adjusted",
        "execution_price_adjustment": "unadjusted",
        "execution_price_fields": ["raw_open", "raw_high", "raw_low", "raw_close"],
        "volume_unit": "shares",
        "amount_unit": "cny",
        "execution_state_mode": "NON_PIT_PROXY",
        "capital_ledger_ready": False,
        "capital_ledger_proxy_ready": True,
        "universe_mode": "CURRENT_NAME_MAINBOARD_NON_ST_PROXY",
        **{
            key: (value.isoformat() if hasattr(value, "isoformat") else value)
            for key, value in stats.items()
            if key not in {"invalid_ohlc_rows", "missing_activity_rows"}
        },
    }
    (staging / "_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    backup = output.with_name(f".{output.name}.backup")
    if backup.exists():
        shutil.rmtree(backup)
    if output.exists():
        output.rename(backup)
    try:
        staging.rename(output)
    except Exception:
        if backup.exists():
            backup.rename(output)
        raise
    shutil.rmtree(backup, ignore_errors=True)

    checks = {
        "invalid_ohlc_rows": int(stats["invalid_ohlc_rows"]),
        "missing_activity_rows": int(stats["missing_activity_rows"]),
    }
    report = {
        "generated_at_utc": metadata["generated_at_utc"],
        "dataset_kind": "QFQ_RESEARCH_UNADJUSTED_EXECUTION_NON_PIT_PROXY",
        "source": metadata["source"],
        "sources": metadata["sources"],
        "summary": {
            "rows": metadata["rows"],
            "symbols_with_rows": metadata["symbols"],
            "first_trade_date": metadata["first_trade_date"],
            "last_trade_date": metadata["last_trade_date"],
        },
        "checks": checks,
        "passed": all(value == 0 for value in checks.values()),
        "known_limitations": [
            "No point-in-time ST, listing, delisting, suspension, or board membership state.",
            "Opening eligibility is a 9.5% main-board non-ST proxy derived from raw prices.",
            "The universe excludes boards and ST names using current identifiers and names.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def _path(value: str) -> Path:
    return Path(value).expanduser()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A-share daily data engineering pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--source", type=_path, default=DEFAULT_SOURCE)

    audit = subparsers.add_parser("audit", help="validate source data and rebuild its catalog")
    add_common(audit)
    audit.add_argument("--catalog", type=_path, default=DEFAULT_CATALOG)
    audit.add_argument("--report", type=_path, default=DEFAULT_REPORT)

    build = subparsers.add_parser("build", help="build the canonical research panel")
    add_common(build)
    build.add_argument("--output", type=_path, default=DEFAULT_OUTPUT)
    build.add_argument("--overwrite", action="store_true")

    all_command = subparsers.add_parser("all", help="run audit and panel build")
    add_common(all_command)
    all_command.add_argument("--catalog", type=_path, default=DEFAULT_CATALOG)
    all_command.add_argument("--report", type=_path, default=DEFAULT_REPORT)
    all_command.add_argument("--output", type=_path, default=DEFAULT_OUTPUT)
    all_command.add_argument("--overwrite", action="store_true")

    hybrid = subparsers.add_parser(
        "hybrid", help="build a qfq research + unadjusted execution NON_PIT_PROXY panel"
    )
    hybrid.add_argument("--download-root", type=_path, default=DEFAULT_MARKET_DATA_ROOT)
    hybrid.add_argument("--research-source", type=_path)
    hybrid.add_argument("--execution-source", type=_path)
    hybrid.add_argument("--catalog", type=_path, default=DEFAULT_CATALOG)
    hybrid.add_argument("--report", type=_path, default=DEFAULT_REPORT)
    hybrid.add_argument("--output", type=_path, default=DEFAULT_OUTPUT)
    hybrid.add_argument("--overwrite", action="store_true")

    cross = subparsers.add_parser(
        "cross-sectional", help="build PIT event-adjusted research panel from daily raw/factor slices"
    )
    cross.add_argument("--source", type=_path, required=True)
    cross.add_argument("--catalog", type=_path, default=DEFAULT_CATALOG)
    cross.add_argument("--report", type=_path, default=DEFAULT_REPORT)
    cross.add_argument("--output", type=_path, default=DEFAULT_OUTPUT)
    cross.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command in {"audit", "all"}:
        report = audit_dataset(args.source, args.catalog, args.report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.command in {"build", "all"}:
        metadata = build_panel(args.source, args.output, overwrite=args.overwrite)
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
    if args.command == "hybrid":
        research_source = args.research_source or discover_download_source(
            args.download_root, "qfq"
        )
        execution_source = args.execution_source or discover_download_source(
            args.download_root, "none"
        )
        metadata = build_hybrid_panel(
            research_source,
            execution_source,
            args.output,
            catalog_path=args.catalog,
            report_path=args.report,
            overwrite=args.overwrite,
        )
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
    if args.command == "cross-sectional":
        metadata = build_cross_sectional_panel(
            args.source,
            args.output,
            catalog_path=args.catalog,
            report_path=args.report,
            overwrite=args.overwrite,
        )
        print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
