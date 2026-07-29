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

## Platform Capability Levels

The service exposes the same module-level decision through `/ready` and `/api/data-center` under
`AUTOALPHA_DATA_CAPABILITY_MATRIX_V1`. Operators should treat these labels as policy, not as
decorative UI state:

| Level | Allowed use | Production meaning |
|---|---|---|
| `RESEARCH_READY` | AutoAlpha factor research and close-of-day screening | Research only; no execution claim |
| `PROXY_BACKTEST_READY` | Manual and batch A-share long-only proxy backtests | Non-PIT evidence only; reconcile vector and event engines |
| `PROXY_PAPER_READY` | Paper trading with next-session open proxy, T+1 and fees | Operational rehearsal only; still blocked from production |
| `PRODUCTION_BLOCKED` | Strict capital ledger and production candidate promotion | Missing PIT market state or source lineage |
| `STRICT_PIT_READY` | Strict capital ledger and production promotion gates | Requires versioned PIT state, eligibility, limits and classifications |

The current local data basis is expected to be `RESEARCH_READY` plus non-PIT proxy levels where raw
execution prices are available. It must remain `PRODUCTION_BLOCKED` until the blockers below are
removed from versioned source tables.

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
