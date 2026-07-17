# AutoCombine V1

AutoCombine is the portfolio-research layer above AutoAlpha. AutoAlpha owns factor discovery,
factor lineage, and canonical single-factor evaluation. AutoCombine consumes an immutable factor
snapshot and searches for a bounded static factor subset and non-negative weights.

## Run

```bash
AUTOALPHA_RUNTIME=runtime-full-llm uv run autocombine-service
```

The service listens on `127.0.0.1:8888` by default. Override it with
`AUTOCOMBINE_HOST` and `AUTOCOMBINE_PORT`. It shares AutoAlpha's runtime database and provider
credential reference, but stores its task, experiment, memory, event, and strategy state in
dedicated tables.

AutoAlpha and AutoCombine use `runtime-full-llm` as their shared default runtime. Keep
`AUTOALPHA_RUNTIME` identical when overriding it. The health and bootstrap endpoints expose the
resolved runtime path, task count, factor count, and factor-registry fingerprint.

## Research Contract

- A task freezes the data protocol, factor definitions, source lineage, and factor universe.
- The task-creation UI defaults to `DRAWDOWN_FIRST`; operators can switch the objective preset
  before creating a task without changing prior experiment history.
- Holdout-contaminated factors remain available for exploratory combination research. Their
  snapshot and event records carry an explicit contamination marker and cannot be represented as
  fresh blind evidence.
- V1 supports static A-share long-only composite signals with non-negative factor weights.
- Factor direction is inherited from AutoAlpha and cannot be flipped inside AutoCombine.
- The LLM proposes an existing factor subset and a mechanism hypothesis. It cannot create code,
  reference a factor outside the snapshot, or inspect hidden-period metrics.
- A deterministic constrained optimizer evaluates rounded weight neighborhoods.
- Every persisted experiment records the subset, final weights, public metrics, failed gates,
  proposal source, hashes, and duration.
- Provider failures or invalid JSON contracts are audited and fall back to deterministic search.
- Service restarts recover running tasks as paused checkpoints; a completed task is immutable.

## Default Gates

The default profile is `ROBUST_ACTIVE_LONG_ONLY`. Feasibility requires minimum coverage and
positive walk-forward fraction, acceptable worst-fold Sharpe and drawdown, bounded turnover and
factor correlation, and non-negative cost-stressed IR. Passing portfolios are ranked with a
robust public score led by active IR, active annual return, worst-fold Sharpe, and drawdown.

Hidden dates are visible to the human operator, but hidden exact metrics are not included in the
LLM context or public experiment payload. A public gate-passing candidate is submitted once to
the A-share long-only blind boundary; only its categorical verdict and evidence hash are stored.
Only a blind-passing candidate can be registered as a `QUALIFIED` StrategySpec. Paper and
production promotion remain separate lifecycle decisions.

## Main APIs

- `POST /api/autocombine/quick-task` on AutoAlpha creates a manual factor snapshot from the factor
  library selection and asks AutoCombine to start it immediately.

```text
GET    /api/bootstrap
POST   /api/tasks
GET    /api/tasks/{task_id}
POST   /api/tasks/{task_id}/start
POST   /api/tasks/{task_id}/stop
POST   /api/tasks/{task_id}/promote
GET    /api/strategies
GET    /api/health
```

The UI provides a task registry, phase flow, immutable factor scope, public metric trajectory,
Pareto frontier, experiment ledger, continuous memory, audit events, and strategy registry.
