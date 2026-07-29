# AutoAlpha PostgreSQL Migration

AutoAlpha still runs on SQLite by default. PostgreSQL support is being introduced
in two explicit phases so operators do not mistake a data copy for a live store
cutover.

## Phase 1: Copy Runtime Data

Generate PostgreSQL DDL from the current SQLite runtime database:

```bash
uv run python scripts/migrate-sqlite-to-postgres.py \
  --schema-only \
  --sqlite runtime-full-llm/autoalpha.sqlite3
```

Copy rows into a target PostgreSQL database:

```bash
AUTOALPHA_DATABASE_URL='postgresql://user:password@host:5432/autoalpha' \
uv run python scripts/migrate-sqlite-to-postgres.py \
  --sqlite runtime-full-llm/autoalpha.sqlite3 \
  --truncate
```

The migration creates matching tables and copies explicit IDs/natural keys. It
does not yet reconstruct every SQLite foreign key, trigger, index, or sequence.

## Phase 2: Job Center Adapter

Set the intended backend:

```bash
export AUTOALPHA_DATABASE_BACKEND=postgresql
export AUTOALPHA_DATABASE_URL='postgresql://user:password@host:5432/autoalpha'
```

The control-plane hot path now has a PostgreSQL adapter boundary for:

- `system_jobs`: enqueue, update, claim with `FOR UPDATE SKIP LOCKED`, lease
  recovery, and structured logs
- `materialized_snapshots`: Strategy Bus, factor knowledge map, and gate
  diagnostics cache state
- `strategy_experiment_objects` / `strategy_experiment_edges`: the experiment
  lineage model from factor candidate to combination candidate
- `formal_strategy_versions`: the versioned strategy-library object that carries
  signal, rebalance, execution, risk, cost, monitoring, evidence, and lifecycle
- `factor_knowledge` / `factor_knowledge_edges`: the research-map layer used by
  homogeneity backfills, mechanism reviews, falsification notes, and related
  factor links
- `factor_pool`: factor candidates, source task lineage, public metrics, and
  initial lifecycle seeding
- `settings` / `settings_revisions`: runtime configuration and secret-free
  revision history
- `events`: tamper-evident audit/action/research/delivery log chain
- `research_tasks`: task definitions, protocol revision, visible data range,
  run state, phase, iteration, and stop/error flags
- `iterations`: candidate lifecycle records, proposal/metric/decision history,
  metric charts, and restart reconciliation inputs
- `llm_role_artifacts`: structured LLM researcher, risk, audit, data, portfolio,
  and trader role outputs

These are the first slices because they are most sensitive to SQLite writer
locks and repeated full-cache rebuilds during large batch jobs.

Until the complete PostgreSQL `ServiceStore` adapter is enabled, `/ready` will
still report a degraded state:

```text
POSTGRES_JOB_CENTER_ADAPTER_AVAILABLE_SERVICE_STORE_PENDING
```

That degraded state is intentional: it means the runtime can inspect/copy data
and the control-plane adapter exists for jobs, materialized snapshots, strategy
experiment lineage, formal strategy versions, factor pool records, factor
knowledge records, settings, audit events, and research task state, but the
production app is still using the SQLite `ServiceStore` path until the remaining
adapters are complete and wired.

## Phase 3: Full Store Cutover

The next migration step is to add tested PostgreSQL adapters for:

- factor-library ranking state and reevaluation batches
- actual runtime wiring and end-to-end PostgreSQL smoke tests

Only after those adapters are live should `AUTOALPHA_DATABASE_BACKEND=postgresql`
be treated as a full service-store cutover.
