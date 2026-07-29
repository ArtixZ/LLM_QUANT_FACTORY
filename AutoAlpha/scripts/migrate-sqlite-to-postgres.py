#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from autoalpha.service.postgres_migration import (
    generate_postgres_schema,
    migrate_sqlite_to_postgres,
    sqlite_catalog,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate an AutoAlpha SQLite runtime database to PostgreSQL."
    )
    parser.add_argument(
        "--sqlite",
        default="runtime-full-llm/autoalpha.sqlite3",
        help="Path to the source SQLite database.",
    )
    parser.add_argument(
        "--postgres-url",
        default=os.getenv("AUTOALPHA_DATABASE_URL", ""),
        help="Target PostgreSQL DSN. Defaults to AUTOALPHA_DATABASE_URL.",
    )
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="Print generated PostgreSQL DDL without copying data.",
    )
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="Truncate target tables before copying rows.",
    )
    parser.add_argument("--batch-size", type=int, default=1000)
    args = parser.parse_args()

    sqlite_path = Path(args.sqlite).expanduser().resolve()
    if args.schema_only:
        print(generate_postgres_schema(sqlite_catalog(sqlite_path)))
        return
    if not args.postgres_url:
        raise SystemExit("--postgres-url or AUTOALPHA_DATABASE_URL is required")
    result = migrate_sqlite_to_postgres(
        sqlite_path=sqlite_path,
        postgres_url=args.postgres_url,
        truncate=args.truncate,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
