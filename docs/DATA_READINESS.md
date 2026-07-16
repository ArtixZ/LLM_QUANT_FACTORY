# Current Data Readiness

Assessment target: `../data/processed/daily_panel`, inspected on 2026-07-15.

The online service may be configured with `../data` directly. `DataWorkspaceReport` resolves the
processed panel, source directory, catalog, quality report, and panel metadata; it binds their
metadata fingerprint to every iteration and delivery artifact.

## Available

The partitioned panel contains 17 annual parquet files, 9,482,111 rows, and 3,192 symbols.
It provides raw and adjusted OHLC prices, volume, amount, returns, observation history, OHLC
validity, activity, and a generic tradable-observation flag. This is sufficient for provisional
price/volume factor development when signals are delayed to the next trading session.

## Production blockers

- No source `knowledge_time`, ingestion batch, or revision history.
- No point-in-time listing, delisting, ST, name, board, or eligibility history.
- No explicit suspension state/reason or side-specific open-time limit tradability.
- No historical industry classification, index membership, or free-float capitalization.
- No source lineage proving when each record became visible to the research system.

The platform therefore does not label the current panel as institutionally point-in-time ready.
`autoalpha inspect-data ../data/processed/daily_panel` reports these blockers and strict workflows
must call `require_institutional_pit()` before production admission. Missing fields must arrive from
versioned source tables; they must not be synthesized from present-day state or filled with defaults.

## Required next ingestion

Ingest security master revisions, exchange trading status and price limits, index/industry history,
point-in-time shares and free float, and vendor ingestion timestamps. Preserve raw source batches,
then produce standard and PIT feature tables through `TableContract` and immutable snapshots.
