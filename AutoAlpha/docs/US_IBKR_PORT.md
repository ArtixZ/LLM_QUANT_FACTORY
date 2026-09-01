# US equity port and IBKR gateway integration

This platform originally targeted China A-shares with Tushare as its data
vendor. It now targets US equities with an Interactive Brokers gateway as both
the data source and the execution venue. This document records what changed,
the broker-API behaviour that shaped the design, and what is still outstanding.

## Running it

Start an IB Gateway (paper mode listens on 4002, live on 4001), then:

```bash
cd AutoAlpha && uv sync --python 3.12
```

Download a universe and build the research panel:

```bash
uv run python -c "from datetime import date; from pathlib import Path; from autoalpha.data.ibkr_sync import sync_universe; from autoalpha.ibkr import GatewaySettings; from autoalpha.data.universe_catalog import resolve_universe; print(sync_universe(resolve_universe('MEGA_CAP_LIQUID_V1')[1], start=date(2015,1,1), end=date.today(), root=Path.home()/'MarketData'/'US', settings=GatewaySettings(port=4002, readonly=True)).to_dict())"
```

```bash
uv run mf-us audit && uv run mf-us panel --overwrite
```

Connection settings come from `IBKR_HOST`, `IBKR_PORT`, `IBKR_CLIENT_ID`,
`IBKR_ACCOUNT`, `IBKR_READONLY`, and `IBKR_REQUIRE_PAPER_ACCOUNT`. Defaults are
the paper port, read-only, and a refusal to attach to a non-paper account.

For the scheduled weekday job that runs this against the paper account
and reports to Telegram, see [DAILY_OPERATIONS.md](DAILY_OPERATIONS.md).

## Broker-API behaviour that shaped the design

Four IBKR behaviours are non-obvious and each is load-bearing:

1. **`ADJUSTED_LAST` is only served anchored at the present.** Passing an
   explicit `endDateTime` returns an empty series rather than an error. Both
   price series are therefore requested as one long window ending now, and the
   caller's end date is applied as a client-side trim. A consequence worth
   stating plainly: adjustment factors are as of the download date, so the
   research panel is **not** point-in-time in its corporate-action treatment.

2. **A what-if order must set `transmit=True`.** `whatIf=True` is what stops the
   order from reaching a venue; `transmit=False` makes TWS reject the request
   (error 321) and then never reply, which presents as a hang. `preview_order`
   sets both flags correctly and bounds the call with a timeout.

3. **IBKR sends `Double.MAX_VALUE` for absent numeric fields.** Read naively it
   becomes `1.797e308` and poisons any sum. `_as_optional_float` maps the
   sentinel to `None` so "not computed" stays distinguishable from "zero".

4. **Daily bars for a 20-year window arrive in a single request.** Chunking
   backwards year by year is unnecessary and burns the 60-requests-per-10-minutes
   pacing budget; two requests per symbol is enough.

Observed on the paper gateway: `whatIf` returns margin but leaves commission
unpopulated for market, market-on-open, and limit orders alike. The platform's
own `USEquityExecutionCosts` is therefore the commission source of truth, and
the broker reply contributes only margin.

## What changed in the research core

| Area | A-share original | US replacement |
|---|---|---|
| Data source | Tushare parquet dumps | IBKR `reqHistoricalData` |
| Panel key | `ts_code` | `symbol` |
| Research prices | qfq forward-adjusted | `ADJUSTED_LAST` (split + dividend) |
| Execution prices | unadjusted | `TRADES` (split-adjusted; IBKR serves nothing rawer) |
| Currency | CNY | USD |
| Commission | 1.5–2.5 bps/side, ¥5 floor | $0.0035/share, $0.35 floor, 1% notional cap |
| Sell-side levies | stamp duty 5 bps, transfer fee 0.1 bps | SEC §31 $27.80/M, FINRA TAF $0.000166/share capped $8.30 |
| Lot size | 100 | 1 |
| Price limits | ±9.5% open gate | none; eligibility is bar validity and volume |
| Settlement lock | T+1 sell lock | none |
| Trading days/year | 245 | 252 |
| Tradability fields | `is_st`, `limit_up`, `limit_down`, `is_suspended` | `is_halted`, `can_buy_open`, `can_sell_open` |

The vector engine works in weight space and never sees share counts, so it
approximates the per-share commission as basis points and says so in its
docstring. The event ledger applies the exact per-share schedule. Use the ledger
for fee-sensitive conclusions.

## Order path and its gates

`plan_orders` diffs a target book against live positions into whole-share
orders, defaulting to market-on-open (`MKT` with `tif=OPG`), which matches the
research protocol's "signal after close, execute at next open" convention.

`preview_plan` runs every order through `whatIfOrder` and routes nothing.
`submit_plan` requires **two** independent gates: a session built with
`GatewaySettings.writable()`, and an explicit `confirm=True`. Neither defaults
to permissive, and `require_paper_account` additionally refuses to attach to a
live account unless deliberately disabled.

## Panel staleness

Slices are downloaded per symbol and refresh independently, so a symbol left out
of the latest sync stays behind. Unioned into a panel it produces `NaN` at
recent dates, which silently poisons a cross-section instead of failing. The
audit therefore reports `stale_symbols` per symbol, and callers should exclude
them before forming a signal:

```bash
uv run mf-us audit | python3 -c "import json,sys; print(json.load(sys.stdin)['stale_symbols'])"
```

This is a warning rather than a contract failure: a partially refreshed panel is
still valid for research over its overlapping window.

## Honest limitations

- **Adjustment is not point-in-time.** See quirk 1 above.
- **The universe is current membership only**, so any backtest over the built-in
  universes carries survivorship bias. Dated index membership is what would fix
  this; nothing here approximates it.
- **Intraday halts are invisible.** Daily bars only reveal whole sessions with no
  prints. LULD halts within a session are not modelled.
- **No fundamentals, sector codes, or free float.** IBKR is a broker feed.
  `inspect_current_panel` correctly reports `institutional_pit_ready=False` for
  IBKR-built panels because of exactly these gaps; that is the intended signal,
  not a bug to suppress.
- **Slippage is a fixed conservative proxy**, not an intraday impact model.

## Still outstanding

- The HTTP layer still uses `tushare_token` field names in request/response
  payloads and settings templates. `DataSyncWorker.token_configured()` and
  `set_token()` are documented shims mapping onto gateway readiness until those
  ~20 call sites in `service/app.py` are renamed.
- Service-layer UI strings and several docs remain in Chinese and describe
  A-share workflows.
- `README.md` / `README_EN.md` still describe the A-share platform.
