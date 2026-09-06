"""Audit immutable IBKR daily slices and build the standard US equity panel.

The download layer (``autoalpha.data.ibkr_sync``) writes one immutable Parquet
slice per symbol. This module never talks to a broker: it validates those
slices against the panel contract, then builds the year-partitioned research
panel the platform reads, together with a catalog and a data-quality report.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

DEFAULT_MARKET_DATA_ROOT = Path.home() / "MarketData" / "US"
DEFAULT_SOURCE = DEFAULT_MARKET_DATA_ROOT / "downloads"
DEFAULT_OUTPUT = DEFAULT_MARKET_DATA_ROOT / "processed" / "daily_panel"
DEFAULT_CATALOG = DEFAULT_MARKET_DATA_ROOT / "catalog" / "daily_catalog.csv"
DEFAULT_REPORT = DEFAULT_MARKET_DATA_ROOT / "catalog" / "data_quality.json"

MARKET = "US_EQUITY"
CURRENCY = "USD"
# ADJUSTED_LAST is split- and dividend-adjusted; TRADES is split-adjusted only.
PRICE_ADJUSTMENT = "split_and_dividend_adjusted"
EXECUTION_PRICE_ADJUSTMENT = "split_adjusted"
ADJUSTMENT_ANCHOR = "AS_OF_DOWNLOAD_DATE"
VOLUME_UNIT = "shares"
AMOUNT_UNIT = "usd"

REQUIRED_COLUMNS = (
    "symbol",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "raw_open",
    "raw_high",
    "raw_low",
    "raw_close",
    "raw_pre_close",
    "vol",
    "amount",
    "listing_date",
    "delisting_date",
    "is_valid_ohlc",
    "is_tradable_observation",
    "can_buy_open",
    "can_sell_open",
    "is_halted",
)


class PanelBuildError(RuntimeError):
    """The source slices could not be turned into a valid research panel."""


def _sql_string(value: str | Path) -> str:
    return str(value).replace("'", "''")


def _parquet_glob(source: Path) -> str:
    resolved = source.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Source slice directory not found: {resolved}")
    if not any(resolved.glob("*.parquet")):
        raise FileNotFoundError(f"No parquet slices found under {resolved}")
    return str(resolved / "*.parquet")


def _raw_relation(parquet_glob: str, *, filename: bool = False) -> str:
    filename_option = ", filename = true" if filename else ""
    return f"read_parquet('{_sql_string(parquet_glob)}', union_by_name = true{filename_option})"


def _query_one(connection: duckdb.DuckDBPyConnection, sql: str) -> dict[str, Any]:
    cursor = connection.execute(sql)
    row = cursor.fetchone()
    if row is None:
        return {}
    return dict(zip((item[0] for item in cursor.description), row, strict=True))


def _read_schema(connection: duckdb.DuckDBPyConnection, parquet_glob: str) -> list[str]:
    rows = connection.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{_sql_string(parquet_glob)}', union_by_name = true)"
    ).fetchall()
    return [row[0] for row in rows]


def audit_dataset(
    source: Path = DEFAULT_SOURCE,
    report_path: Path | None = DEFAULT_REPORT,
) -> dict[str, Any]:
    """Validate raw slices against the panel contract before anything is built."""
    parquet_glob = _parquet_glob(source)
    connection = duckdb.connect()
    try:
        columns = _read_schema(connection, parquet_glob)
        missing = sorted(set(REQUIRED_COLUMNS) - set(columns))
        if missing:
            raise PanelBuildError(f"Missing required columns: {', '.join(missing)}")
        relation = _raw_relation(parquet_glob)
        summary = _query_one(
            connection,
            f"""
            SELECT
                count(*) AS rows,
                count(DISTINCT symbol) AS symbols_with_rows,
                min(trade_date) AS first_trade_date,
                max(trade_date) AS last_trade_date,
                count(*) FILTER (WHERE symbol IS NULL OR trade_date IS NULL) AS null_keys,
                count(*) FILTER (WHERE NOT is_valid_ohlc) AS invalid_ohlc,
                count(*) FILTER (WHERE NOT is_tradable_observation) AS untradable,
                count(*) FILTER (WHERE coalesce(vol, 0) <= 0) AS zero_volume,
                count(*) FILTER (WHERE coalesce(amount, 0) <= 0) AS zero_amount,
                count(*) FILTER (WHERE close IS NULL OR raw_close IS NULL) AS null_close
            FROM {relation}
            """,
        )
        # Slices are downloaded per symbol and can be refreshed independently, so
        # a symbol left out of the latest sync stays behind. Unioned into a panel
        # it yields NaN at recent dates, which silently poisons a cross-section
        # rather than failing, so staleness is reported per symbol here.
        stale_rows = connection.execute(
            f"""
            WITH per_symbol AS (
                SELECT symbol, max(trade_date) AS last_trade_date
                FROM {relation} GROUP BY symbol
            )
            SELECT symbol, CAST(last_trade_date AS VARCHAR)
            FROM per_symbol
            WHERE last_trade_date < (SELECT max(last_trade_date) FROM per_symbol)
            ORDER BY last_trade_date, symbol
            """
        ).fetchall()
        duplicates = connection.execute(
            f"""
            SELECT count(*) FROM (
                SELECT symbol, trade_date
                FROM {relation}
                GROUP BY symbol, trade_date
                HAVING count(*) > 1
            )
            """
        ).fetchone()
        duplicate_keys = int(duplicates[0]) if duplicates else 0
    finally:
        connection.close()

    failures = {
        "null_keys": int(summary.get("null_keys") or 0),
        "duplicate_keys": duplicate_keys,
        "null_close": int(summary.get("null_close") or 0),
    }
    stale = {str(symbol): str(last) for symbol, last in stale_rows}
    warnings = {
        "invalid_ohlc": int(summary.get("invalid_ohlc") or 0),
        "untradable": int(summary.get("untradable") or 0),
        "zero_volume": int(summary.get("zero_volume") or 0),
        "zero_amount": int(summary.get("zero_amount") or 0),
        "stale_symbols": len(stale),
    }
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "market": MARKET,
        "source": str(source.resolve()),
        "schema": columns,
        "summary": {
            key: _jsonable(summary.get(key))
            for key in ("rows", "symbols_with_rows", "first_trade_date", "last_trade_date")
        },
        "failures": failures,
        "warnings": warnings,
        "stale_symbols": stale,
        "passed": not any(failures.values()),
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def write_catalog(
    source: Path = DEFAULT_SOURCE,
    destination: Path = DEFAULT_CATALOG,
) -> dict[str, int]:
    """Record one row per source slice so the panel's provenance is inspectable."""
    parquet_glob = _parquet_glob(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    try:
        connection.execute(
            f"""
            COPY (
                SELECT
                    symbol,
                    count(*) AS rows,
                    min(trade_date) AS first_trade_date,
                    max(trade_date) AS last_trade_date,
                    count(*) FILTER (WHERE NOT is_tradable_observation) AS untradable_sessions,
                    any_value(filename) AS source_file
                FROM {_raw_relation(parquet_glob, filename=True)}
                GROUP BY symbol
                ORDER BY symbol
            ) TO '{_sql_string(destination)}' (FORMAT CSV, HEADER)
            """
        )
        counted = _query_one(
            connection,
            f"SELECT count(DISTINCT symbol) AS symbols FROM {_raw_relation(parquet_glob)}",
        )
    finally:
        connection.close()
    return {"symbols": int(counted.get("symbols") or 0)}


def build_panel(
    source: Path = DEFAULT_SOURCE,
    output: Path = DEFAULT_OUTPUT,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build the year-partitioned research panel from audited source slices."""
    source = source.resolve()
    output = output.resolve()
    parquet_glob = _parquet_glob(source)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}; pass --overwrite to replace it")

    staging = output.with_name(f".{output.name}.staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.parent.mkdir(parents=True, exist_ok=True)

    relation = _raw_relation(parquet_glob)
    connection = duckdb.connect()
    try:
        connection.execute(
            f"""
            COPY (
                WITH typed AS (
                    SELECT
                        CAST(symbol AS VARCHAR) AS symbol,
                        CAST(trade_date AS DATE) AS trade_date,
                        CAST(open AS DOUBLE) AS open,
                        CAST(high AS DOUBLE) AS high,
                        CAST(low AS DOUBLE) AS low,
                        CAST(close AS DOUBLE) AS close,
                        CAST(adj_close AS DOUBLE) AS adj_close,
                        CAST(raw_open AS DOUBLE) AS raw_open,
                        CAST(raw_high AS DOUBLE) AS raw_high,
                        CAST(raw_low AS DOUBLE) AS raw_low,
                        CAST(raw_close AS DOUBLE) AS raw_close,
                        CAST(raw_pre_close AS DOUBLE) AS raw_pre_close,
                        CAST(vol AS DOUBLE) AS vol,
                        CAST(amount AS DOUBLE) AS amount,
                        CAST(listing_date AS DATE) AS listing_date,
                        CAST(delisting_date AS DATE) AS delisting_date,
                        CAST(is_valid_ohlc AS BOOLEAN) AS is_valid_ohlc,
                        CAST(is_tradable_observation AS BOOLEAN) AS is_tradable_observation,
                        CAST(can_buy_open AS BOOLEAN) AS can_buy_open,
                        CAST(can_sell_open AS BOOLEAN) AS can_sell_open,
                        CAST(is_halted AS BOOLEAN) AS is_halted
                    FROM {relation}
                )
                SELECT
                    *,
                    year(trade_date) AS trade_year,
                    close / nullif(
                        lag(close) OVER (PARTITION BY symbol ORDER BY trade_date), 0
                    ) - 1 AS ret_1d,
                    raw_close / nullif(raw_pre_close, 0) - 1 AS raw_ret_1d,
                    date_diff(
                        'day',
                        lag(trade_date) OVER (PARTITION BY symbol ORDER BY trade_date),
                        trade_date
                    ) AS observation_gap_days,
                    row_number() OVER (
                        PARTITION BY symbol ORDER BY trade_date
                    ) AS history_observations
                FROM typed
                ORDER BY year(trade_date), trade_date, symbol
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
                count(DISTINCT symbol) AS symbols,
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
        "market": MARKET,
        "currency": CURRENCY,
        "format": "parquet",
        "partitioning": ["trade_year"],
        "price_adjustment": PRICE_ADJUSTMENT,
        "execution_price_adjustment": EXECUTION_PRICE_ADJUSTMENT,
        "adjustment_anchor": ADJUSTMENT_ANCHOR,
        "volume_unit": VOLUME_UNIT,
        "amount_unit": AMOUNT_UNIT,
        "institutional_pit_ready": False,
        "capital_ledger_ready": False,
        "capital_ledger_proxy_ready": True,
        "rows": _jsonable(stats.get("rows")),
        "symbols": _jsonable(stats.get("symbols")),
        "first_trade_date": _jsonable(stats.get("first_trade_date")),
        "last_trade_date": _jsonable(stats.get("last_trade_date")),
        "year_partitions": _jsonable(stats.get("year_partitions")),
        "caveats": [
            "Adjustment factors are as of the download date, not point-in-time.",
            "Execution prices are split-adjusted; IBKR does not serve truly unadjusted bars.",
            "Intraday LULD halts are invisible in daily bars; only no-print sessions are flagged.",
            "The universe is the requested symbol list and carries no delisting history.",
        ],
    }
    (staging / "_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    if output.exists():
        shutil.rmtree(output)
    staging.rename(output)
    return metadata


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, int | float | str | bool):
        return value
    return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mf-us", description="Audit IBKR slices and build the US equity research panel"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="Validate raw slices against the panel contract")
    audit.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    audit.add_argument("--report", type=Path, default=DEFAULT_REPORT)

    catalog = subparsers.add_parser("catalog", help="Write the per-symbol source catalog")
    catalog.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    catalog.add_argument("--destination", type=Path, default=DEFAULT_CATALOG)

    panel = subparsers.add_parser("panel", help="Build the year-partitioned research panel")
    panel.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    panel.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    panel.add_argument("--overwrite", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "audit":
        report = audit_dataset(args.source, args.report)
        print(json.dumps(report, indent=2))
        return 0 if report["passed"] else 1
    if args.command == "catalog":
        print(json.dumps(write_catalog(args.source, args.destination), indent=2))
        return 0
    metadata = build_panel(args.source, args.output, overwrite=args.overwrite)
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
