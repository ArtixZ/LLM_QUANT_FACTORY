# ADR 0001: Institutional Core Is the Product Boundary

Status: Accepted, amended 2026-07-15

## Decision

The installable `src/autoalpha` package is the only product boundary. Research is expressed through
typed factors or capability-limited candidate code and executed by audited services. Evaluation
returns a structured evidence matrix and sequential gate decision; no scalar score API exists.

The earlier root-level single-file workflow is removed. Production code, documentation, CI, and
release artifacts must not import or recreate those entry points.

## Consequences

- Data, evaluation, portfolio, execution, governance, and operations have explicit ownership and
  testable contracts.
- A candidate cannot compensate for a failed data, statistical, risk, capacity, holdout, or paper
  gate with strength elsewhere.
- Batch prioritization uses Pareto dominance only after equivalent hard-gate eligibility.
- Historical artifacts from unsupported protocols require full re-evaluation before registration.
- The current local panel remains blocked from production despite platform readiness.
