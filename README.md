# A-share Multi-Factor Research

Research workspace for point-in-time A-share cross-sectional factor strategies. Raw market
data and generated datasets are intentionally excluded from Git.

## Dataset layout

```text
data/
  mainboard_non_st_qfq_20100101_20260715/  # immutable source snapshot
  catalog/                                  # generated catalog and quality report
  processed/daily_panel/                    # canonical year-partitioned Parquet panel
```

The source snapshot contains forward-adjusted daily OHLCV observations. It is a sparse
observation table, not a point-in-time security master: a missing row cannot by itself tell
whether a stock was suspended, not yet listed, or already delisted. Daily ST status and
historical universe membership are also unavailable. Research code must not treat the folder
name `non_st` as a historical universe definition.

## Setup and pipeline

```bash
uv sync
uv run mf-data all
uv run pytest
```

Useful individual commands:

```bash
uv run mf-data audit
uv run mf-data build
uv run mf-data build --overwrite
```

`audit` writes `data/catalog/daily_catalog.csv` with paths relative to the source snapshot and
`data/catalog/data_quality.json` with structural and row-level checks. `build` writes a canonical
panel partitioned by `trade_year`; it refuses to replace an existing output unless `--overwrite`
is supplied.

## Canonical panel semantics

- `trade_date` is a proper date and `(trade_date, ts_code)` is the intended primary key.
- Prices are the supplied forward-adjusted prices; `close` is also exposed as `adj_close`.
- `ret_1d` is `close / pre_close - 1`, while `close_to_close_ret` uses the prior available
  observation. Their distinction matters after missing trading days.
- `history_observations` counts available observations, not exchange-listed calendar days.
- `is_tradable_observation` only confirms valid OHLC and positive volume/amount on an existing
  row. It does not reconstruct suspension or price-limit eligibility.
- No forward return label is materialized here. Labels should be built later against an explicit
  market calendar and execution convention to avoid accidental look-ahead.
