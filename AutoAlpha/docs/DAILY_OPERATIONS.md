# Daily paper operations

A scheduled weekday job refreshes US market data from the IBKR gateway,
rebuilds the research panel, forms a 12-1 momentum target book, previews the
resulting orders against IBKR's margin engine, and pushes a digest to Telegram.

## Moving parts

| Piece | Path |
|---|---|
| Scheduler | `~/Library/LaunchAgents/com.quantfactory.daily.plist` |
| Wrapper | `AutoAlpha/scripts/run_daily_us.sh` |
| Runner | `AutoAlpha/scripts/daily_run.py` |
| Logic | `autoalpha/operations/daily.py` |
| Notifications | `$QUANTFACTORY_NOTIFY_SCRIPT`, source `quantfactory` |
| Run logs | `AutoAlpha/logs/scheduler/run-<date>.log` |
| Report artifacts | `AutoAlpha/logs/daily-<date>.json` |

The job runs Mon–Fri at **13:35 PT (16:35 ET)** — 35 minutes after the close so
IBKR has settled the session's daily bars. Stagger it against any other job
on the host that uses the gateway, so the two never contend for it.

## Running it by hand

```bash
cd /path/to/LLM_QUANT_FACTORY/AutoAlpha && uv run python scripts/daily_run.py --dry-run
```

`--dry-run` prints the digest without notifying. `--skip-sync` reuses the panel
on disk, which turns a ~20 minute run into a few seconds while iterating.

## Notifications

The wrapper sources whatever files `QUANTFACTORY_ENV_FILES` lists (colon-
separated), so a host-local dispatcher can be configured there:

```
NOTIFY_SOURCE_QUANTFACTORY=on      # on | errors | off
```

Set it to `errors` to keep failures but drop the daily digest, if your
dispatcher supports it. Leaving `QUANTFACTORY_NOTIFY_SCRIPT` unset skips
notification entirely; the run still writes its logs and artifacts. Watchtower gets a recurring ping under the slug
`quantfactory-daily` on a 26h cadence, so a weekday run that never happens
raises a dead-man's-switch alert.

Severity maps to what actually went wrong:

| Severity | Meaning |
|---|---|
| `ok` | Clean run: audit passed, no stale symbols, every order previewed |
| `warn` | Stale symbols excluded, or an order failed its what-if preview |
| `error` | Audit failed, a symbol failed to sync, or the run raised |

## Pacing: why a full run takes ~20 minutes

IBKR allows no more than 60 historical-data requests in any rolling ten-minute
window. Each symbol costs two requests (`ADJUSTED_LAST` + `TRADES`), so a
57-symbol universe needs 114 requests and the pacer deliberately sleeps through
the limit. This is expected, not a hang. `--skip-sync` bypasses it entirely
when you only want to re-form the book.

## Universe and staleness

Slices refresh independently, so a symbol dropped from the configured universe
would sit at a stale date and silently become `NaN` in every later
cross-section. The sync therefore refreshes **the configured universe plus
everything already on disk**, which makes the panel self-healing: staleness
cannot accumulate on its own.

To deliberately shrink the universe, prune the orphans first:

```python
from pathlib import Path
from autoalpha.data.ibkr_sync import prune_slices
prune_slices(Path.home() / "MarketData" / "US", keep=["AAPL", "MSFT"])
```

The audit reports any symbol left behind under `stale_symbols`, and the daily
run excludes those from the signal rather than trusting a `NaN`.

## Enabling order submission

**The scheduled job does not place orders.** It stops at the what-if preview.

To let it trade, edit the last command in `scripts/run_daily_us.sh`:

```bash
# from
"$PYTHON" -u scripts/daily_run.py >> "$RUN_LOG" 2>&1
# to
"$PYTHON" -u scripts/daily_run.py --submit --confirm >> "$RUN_LOG" 2>&1
```

Both flags are required, and `--submit --confirm` is also what flips the
gateway session out of read-only. Before enabling it, be aware of what the
current configuration would send each weekday:

- a **market-on-open** order per name, which fills at whatever the opening
  auction prints — there is no limit price protecting it
- a full rebalance to the top 5 momentum names at **95% gross exposure**, so a
  change in the top 5 sells one entire position and buys another
- no per-order size cap beyond the target book, and no daily loss limit

`GatewaySettings.require_paper_account` still refuses to attach to a non-paper
account, so an accidental switch to port 4001 fails closed rather than trading
your live account.

## Known limitations carried from the port

The research caveats in [US_IBKR_PORT.md](US_IBKR_PORT.md) all apply here —
in particular the universe is current-membership only (survivorship bias) and
adjustment factors are as-of download date rather than point-in-time. The daily
digest is an operational report, not evidence that the strategy works.
