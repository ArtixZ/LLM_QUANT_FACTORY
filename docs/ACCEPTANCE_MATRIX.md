# Institutional Acceptance Matrix

| Goal | Delivered evidence |
|---|---|
| P0 | Isolated execution, AST policy, frozen protocol, purged splits, HAC/bootstrap, budgets, hidden holdout, daily cohort ledger, explicit A-share fees |
| G1 | Versioned contracts, revision-aware as-of reads, dynamic universes, immutable parquet snapshots, real-panel readiness gate |
| G2 | Canonical typed DSL, temporal semantics, stable IDs, neutralization, division safety, common-subexpression cache |
| G3 | Paired control/treatment portfolio increments, net-return HAC/bootstrap, walk-forward, robustness segments, DSR, PBO, FDR, IC diagnostics, sequential hard gates, Pareto ranking |
| G4 | Neutralization, conditional IC, incremental R2/IR, correlation clustering, immutable factor cards, parent lineage, lifecycle state machine, rejection cooldown |
| G5 | Shrunk factor covariance, specific risk, active risk, constrained optimizer, linearized turnover, liquidity/frozen holdings, attribution, stress scenarios, deterministic fallback |
| G6 | Orders, integer lots, partial fills, OPEN/CLOSE/TWAP/VWAP/POV, fees, spread, square-root impact, opportunity cost, capacity curves, TCA |
| G7 | Six role capability domains, controlled feedback, model/prompt/context/token/tool audit, budgets, pause/resume, anomaly and robustness stopping |
| G8 | Content-addressed artifacts, atomic idempotent tasks, paper/live comparison, drift alerts, human-approved phased releases, rollback, CI, container, runbook |

## Automated acceptance

Run:

```bash
uv sync --frozen --all-groups
uv run ruff check .
uv run pytest --cov=autoalpha --cov-report=term-missing
uv run autoalpha inspect-data ../data/processed/daily_panel
```

The end-to-end integration test uses an explicitly timestamped synthetic PIT panel. The local real
panel is allowed for provisional delayed price/volume research, but the production gate remains
closed until the blockers in `DATA_READINESS.md` are ingested from versioned sources.
