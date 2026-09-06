# Current Data Readiness

Assessment target: `$AUTOALPHA_DATA_PATH`, normally `~/MarketData/US`.

`DataWorkspaceReport` resolves the year-partitioned panel, per-symbol IBKR download slices, source
catalog, quality report, and panel metadata. Their combined fingerprint is bound to research and
delivery evidence.

## Available

The integrated downloader requests two daily series per resolved US equity:

- `ADJUSTED_LAST` for split- and dividend-adjusted factor research;
- `TRADES` for split-adjusted execution-proxy OHLC, share volume, and derived USD amount.

The panel also records OHLC validity, traded-session status, and side-specific open eligibility.
This is enough for price/volume research and non-PIT next-open proxy backtests. The exact row,
symbol, and date counts depend on the operator's configured universe and licensed IBKR history.

## Platform capability levels

The service exposes the same module-level decision through `/ready` and `/api/data-center` under
`AUTOALPHA_DATA_CAPABILITY_MATRIX_V1`.

| Level | Allowed use | Production meaning |
|---|---|---|
| `RESEARCH_READY` | Factor research and close-of-day screening | Research only |
| `PROXY_BACKTEST_READY` | Manual and batch US-equity long-only proxy backtests | Non-PIT evidence |
| `PROXY_PAPER_READY` | Paper portfolios using next-session split-adjusted open proxies | Operational rehearsal only |
| `PRODUCTION_BLOCKED` | Strict capital ledger and production-candidate promotion | Missing PIT state or lineage |
| `STRICT_PIT_READY` | Strict capital ledger and production gates | Requires versioned PIT eligibility and classifications |

IBKR slices explicitly set `institutional_pit_ready=false` and `capital_ledger_ready=false`, while
allowing `capital_ledger_proxy_ready=true` when execution prices and units pass validation.

## Production blockers

- Built-in universes contain current members and therefore carry survivorship bias.
- Adjustment factors are anchored to the download date rather than historical knowledge time.
- No source `knowledge_time`, ingestion revision history, or immutable vendor publication timestamp.
- Downloaded `listing_date` is the first observed bar, not authoritative listing history;
  delisting history is absent.
- Daily bars cannot reconstruct intraday LULD halts or exact open-auction eligibility.
- Historical sector, benchmark membership, and point-in-time free-float capitalization are absent.

Strict workflows must call `require_institutional_pit()` and fail closed. These fields must come
from versioned source tables; current state must never be backfilled as if it were historical.

## Required next ingestion

Add an effective-dated US security master, exchange halt/LULD records, historical sector and index
membership, point-in-time shares/free float, and source-visible timestamps. Preserve immutable raw
batches and keep the existing research/proxy labels until those contracts pass.
