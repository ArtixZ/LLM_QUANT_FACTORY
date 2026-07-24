# Factor Behavior Clustering

## Purpose

The factor library keeps two independent taxonomies:

- `cluster_id`: expression structure and declared mechanism similarity.
- `behavior_cluster_id`: observed signal and long-only return similarity under one frozen
  A-share evaluation protocol.

Neither taxonomy changes factor admission, lifecycle state, or historical metrics.

## Frozen protocol

- Evaluation period: 2015-01-01 through 2024-12-31.
- Portfolio: A-share long-only, 90% target gross exposure, maximum 30 positions.
- Rebalance: first actual trading session of each week.
- Timing: signal after close on T, execution at the next scheduled open.
- Execution: raw-open buy/sell eligibility proxy; adjusted open-to-open total-return path.
- Costs: side-aware commission, transfer fee, sell stamp duty, minimum commission and slippage.
- Scope: research proxy, not a point-in-time production claim.

## Engine alignment

Vector and event engines share `autoalpha.backtest.target_book` for:

- deterministic cross-sectional ranking and tie handling;
- target count and maximum position handling;
- daily, weekly, biweekly and monthly schedule boundaries.

Tradability is deliberately excluded from target construction. Both engines first form the same
target intent and then apply side-specific execution constraints. The event ledger remains the
gold standard for cash, integer lots, volume participation, pending orders and exact fees.

## Fingerprint and clustering

Each factor stores a resumable artifact containing:

- the daily net return path;
- deterministic random projections of weekly cross-sectional percentile ranks.

Daily returns are residualized by subtracting the cross-factor daily median. Pair similarity is:

```text
0.65 * positive(signal fingerprint correlation)
+ 0.35 * positive(residual return correlation)
```

Average-linkage hierarchical clustering uses a default similarity boundary of `0.74`. Cluster IDs
are stable hashes of sorted membership, not display-order numbers.

## Redundancy labels

- `NEAR_DUPLICATE`: signal correlation >= 0.95 and residual return correlation >= 0.90.
- `SUBSTITUTE`: composite similarity >= 0.82.
- `RELATED`: same behavior cluster but below the substitute boundary.
- `DISTINCT`: singleton behavior cluster.
- `PENDING`: no completed behavior artifact in the latest snapshot.

## Operations

Run or resume the full library:

```bash
nice -n 10 .venv/bin/python scripts/recompute_factor_behavior_clusters.py
```

Progress is written atomically to:

```text
runtime-full-llm/factor-behavior/progress.json
```

The completed, versioned snapshot consumed by `/api/factors` is:

```text
runtime-full-llm/factor-behavior/latest.json
```

Immutable snapshots are retained under `runtime-full-llm/factor-behavior/snapshots/`. The
snapshot ID binds the protocol, data fingerprint, evaluation period, cluster threshold and ordered
factor universe, so an incremental run never silently rewrites the evidence identity.
