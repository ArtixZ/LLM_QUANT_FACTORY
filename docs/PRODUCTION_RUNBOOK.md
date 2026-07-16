# AutoAlpha Production Runbook

## Ownership and service levels

- Research owner: validates factor evidence, factor card, and failed-factor history.
- Data owner: validates point-in-time snapshot contracts and source freshness before calculation.
- Risk owner: approves risk limits, paper-trading completion, capital allocation, and releases.
- Operations owner: runs daily tasks and responds to alerts. Daily portfolio output is due before
  the configured execution window; a missed deadline suspends new orders.

## Daily sequence

1. Verify source checksums and create an immutable point-in-time snapshot.
2. Resolve approved factor artifacts and compile DSL expressions.
3. Run factor, risk, optimizer, and execution-plan tasks through `IdempotentPipeline`.
4. Compare target orders with current holdings and reject duplicate task keys or order IDs.
5. Publish outputs to `ArtifactRegistry`; only complete, checksum-valid outputs are visible.
6. Record paper, shadow, or production fills and run TCA and drift monitoring.

## Research generation sequence

1. Bind the immutable protocol, data fingerprint, candidate budget, and factor-family budget.
2. Evaluate exploration diagnostics and annual 5Y-to-1Y public walk-forward folds.
3. Apply HAC, FDR, Deflated Sharpe, PBO, parameter-neighborhood, turnover, and portfolio gates.
4. Freeze the exact factor IDs and weights before requesting one holdout access.
5. Return only a categorical holdout verdict and evidence hash to the research process.
6. After holdout passage, run the capital ledger with lot, fee, volume, exposure, and residual-position constraints.
7. Promote only to paper research; production still requires PIT data readiness and human risk approval.

Public walk-forward evidence is adaptive after repeated model feedback. It must never be described as untouched
out-of-sample evidence. A failed holdout candidate cannot be tuned against hidden details or evaluated twice.

## Failure handling

- Data contract, checksum, missingness, or freshness failure: stop before factor calculation.
- Optimization infeasible: use the deterministic no-trade result and open a risk review.
- Order task retry: reuse the same task key and order IDs; never generate a second order batch.
- Partial fill or blocked sell: retain and mark the position, retry under the next approved plan.
- Artifact integrity failure: quarantine the artifact and rebuild from its immutable source IDs.

## Alerts and degradation

- Data completeness breach: suspend new signals and preserve existing positions for valuation.
- IC or execution deterioration: deweight and open a factor review.
- Exposure drift: hold allocation and request risk review.
- Material shadow-versus-live PnL divergence: suspend and roll back to the prior approved artifact.

## Release and rollback

Production promotion requires a factor approval, completed paper observation, risk approval ID,
owner, monitoring thresholds, and an initial allocation fraction. Increase capital only through a
new release event. `ReleaseRegistry.rollback` restores the previous immutable artifact and records
the approver and approval ID in the hash-chain audit log.

## Recovery and retirement

Rebuild any task from data snapshot ID, source artifact IDs, config fingerprint, code revision, and
random seed. A suspended factor may return only through paper trading. A retired factor cannot be
reactivated in place; create a child factor card and retain the parent and retirement reason.
