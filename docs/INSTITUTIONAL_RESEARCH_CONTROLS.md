# Institutional Research Controls

This document describes the four production-oriented controls implemented outside the
point-in-time data program.

## Product templates and capital ledger

Manual research must select an explicit product template:

- `MARKET_NEUTRAL_RESEARCH`: vector research only; no short-borrow simulation.
- `LONG_ONLY_CAPITAL`: next-open A-share cash ledger with integer lots, fees, volume limits,
  approximate open-limit checks, retained unsellable positions, and overlapping holding sleeves.
- `UNIVERSE_INDEX_ENHANCED_PROXY`: long-only capital ledger plus an equal-weight-universe
  benchmark and constrained risk-model optimization diagnostic.
- `UNIVERSE_HEDGED_PROXY`: capital-ledger long book with an equal-weight market-return hedge
  proxy.

Proxy benchmark templates are explicitly ineligible for production until official index
membership, historical weights, and execution-grade market-state data are available.

The manual backtest workbench persists every execution assumption: selection fraction, maximum
positions per side, holding period, gross exposure, lot size, ADV participation, opening-limit
threshold, commission, stamp duty, transfer fee, minimum commission, and cost-stress multiplier.
Each artifact contains a configuration hash. Completed runs can be named, tagged, annotated,
favorited, filtered, compared side by side, and loaded back into the form without changing the
automated research state.

Capital-ledger templates also emit a simulated A-share trade statement. Every row records signal
and execution dates, security code and name, side, quantity, price, notional, commission, transfer
fee, stamp duty, total fees, net cash flow, sleeve, and post-trade sleeve cash. Statements are
stored as separate CSV artifacts with SHA-256 integrity checks and can be filtered and paginated
through the control-plane API. They are simulation records, not broker-issued contract notes.

## Manual research contamination ledger

Every manual backtest records generation, factor, requested period, visibility scope, timestamp,
and a SHA-256 evidence hash in `manual_research_exposures`. A request overlapping the configured
hidden test period is marked contaminated before evaluation starts.

The automated worker checks the ledger before blind evaluation. Any same-generation portfolio
containing an exposed factor receives `MANUAL_HOLDOUT_CONTAMINATION` and cannot access the hidden
evaluator. Opening a new research generation is required; the holdout budget is not consumed by
the blocked request.

## Risk model and index enhancement

The risk model estimates cross-sectional price/volume beta, volatility, momentum, liquidity, and
size-proxy returns with covariance shrinkage and specific risk. The optimizer supports long-only
and active-weight bounds, turnover, ADV participation, frozen untradable positions, named risk
exposure bounds, and an annualized tracking-error constraint. Infeasible cases return a
deterministic no-trade fallback with constraint diagnostics.

The current equal-weight universe is a research proxy, not a formal index benchmark.

## Factor analytics and lifecycle

The factor library keeps every candidate and adds:

- expression-structure and mechanism clusters, with cluster leaders and nearest peers;
- historical marginal portfolio contribution from ADD/REPLACE diagnostics;
- generation-specific hidden-period contamination status;
- a persisted lifecycle: `RESEARCH -> QUALIFIED -> SHADOW -> PAPER -> PRODUCTION`, with watch,
  suspension, decay, rejection, and retirement paths.

Lifecycle transitions are append-only audit events and are available through the factor API.
Automatic factor-pool status and deployment lifecycle remain separate concepts.
