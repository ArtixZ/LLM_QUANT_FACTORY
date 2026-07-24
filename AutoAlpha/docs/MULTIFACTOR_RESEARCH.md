# Multi-Factor Continuous Research Protocol

## Scope

The online service has two independent loops. The inner loop asks the configured LLM for one typed,
falsifiable factor expression. The outer loop is deterministic and decides whether that candidate
changes the active portfolio. LLM output can never directly set portfolio weights, bypass gates, or
approve paper or production deployment.

## Persistent state

- `factor_pool` stores the canonical proposal, single-factor evidence, source iteration and status.
- `portfolio_versions` stores every evaluated action, including rejected `HOLD` decisions.
- `portfolio_members` stores the exact equal weights belonging to each version.
- The active portfolio is always the most recent accepted version. A later rejected action cannot
  overwrite it.
- Existing completed iterations are imported idempotently during the first upgraded iteration.

Factor states are `SCREENED_OUT`, `ELIGIBLE`, and `ACTIVE`. The initial screen requires positive
cost-adjusted Sharpe and annual return, positive stressed IR, and at least 80% signal coverage. This
screen only controls entry to combination research; it is not a production approval.

## Combination construction

Each active factor is evaluated with its registered direction. On every date, the factor panel is
cross-sectionally standardized. The composite signal is the equal-weight mean of those standardized
panels. The deterministic backtester then forms dollar-neutral top/bottom decile positions and
applies the configured A-share transaction costs.

Equal weighting is deliberate for the first production protocol: it limits parameter search,
provides stable attribution and prevents the LLM from fitting weights. The maximum active factor
count is a service setting and defaults to five.

## Action search

For every eligible candidate the engine evaluates:

1. `ADD` when the portfolio is below the factor-count limit;
2. `REPLACE` for each active member;
3. `REMOVE` for each active member when at least two factors are active;
4. `HOLD` when no proposed action passes all gates.

The accepted action must improve deterministic portfolio utility and pass coverage, capacity,
turnover, annual stability and maximum factor-correlation limits. Portfolio value has two valid
paths:

- return accretion: positive incremental net IR and annual return;
- diversification upgrade: Sharpe improves by at least 0.25, maximum drawdown improves by at least
  one percentage point, annual return deterioration is no worse than two percentage points, and
  stressed IR does not deteriorate.

This dual path prevents the engine from rejecting a materially safer portfolio solely because it
gives up a small amount of raw return. It also prevents a high standalone IC from compensating for
poor portfolio value.

## Evidence and audit

Every iteration emits factor-pool and portfolio-action events into the existing hash-chained event
stream. The immutable research artifact contains the single-factor metrics, portfolio metrics,
action, active factor IDs, removed factor ID, correlations, failed gates and portfolio version ID.
The dashboard exposes the active composition, weights, absolute portfolio metrics and recent action
history.

## Production boundary

This protocol remains research-only while `institutional_pit_ready` is false. Multi-factor success
does not relax point-in-time data, hidden holdout, paper-trading, human risk approval or production
release requirements. The current local panel therefore produces
`MULTIFACTOR_*_RESEARCH_ONLY_DATA_BLOCKED` decisions only.
