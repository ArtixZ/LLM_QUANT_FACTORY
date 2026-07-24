# AutoAlpha Implementation Status

This file tracks production delivery against `INSTITUTIONAL_IMPROVEMENT_GOALS.md`.

| Area | Status | Current milestone |
|---|---|---|
| Engineering baseline | Complete | Package, versioned config, tests, dependency lock |
| P0 trustworthy baseline | Complete | Isolation, protocol, statistics, ledger, budget, holdout |
| G1 point-in-time data | Platform complete, data blocked | Contracts, revisions, universe, snapshots and strict readiness gate |
| G2 factor DSL | Complete | Typed expressions, semantic validation, canonical compiler |
| G3 evaluation | Complete | Paired portfolio increment, robustness, DSR/PBO/FDR, hard gates, Pareto ranking |
| G4 factor lifecycle | Complete | Immutable cards, novelty, neutralization and lifecycle events |
| G5 portfolio and risk | Complete | Shrunk risk model, attribution and constrained optimizer |
| G6 execution and capacity | Complete | Orders, fills, impact, capacity and TCA |
| G7 LLM governance | Complete | Role capabilities, controlled feedback, audit and stopping |
| G8 production operations | Complete | Artifacts, idempotent tasks, paper monitoring, release and rollback |
| Online multi-factor loop | Complete | Persistent pool, deterministic add/drop/replace, immutable champion versions and dashboard |

## Supported protocol

`institutional_v2` is the only supported research generation. The production package has no scalar
research utility, parent-score comparison, or legacy single-file runner. Candidate retention is
decided by a complete evidence matrix and sequential hard gates; IC remains diagnostic.

## Acceptance

- Automated tests cover unit, adversarial, accounting, optimization, evaluation, orchestration, and
  integration behavior.
- The integration workflow replays PIT snapshot -> DSL -> diagnostics -> factor card -> constrained
  portfolio -> execution -> immutable artifact without duplicate work or orders.
- `ruff` and the full test suite run under the frozen `uv.lock`; CI and container definitions are
  versioned.
- Platform capability is implemented, but the current 9,482,111-row local panel is not approved for
  production because source-visible timestamps and historical master tables are absent. See
  `docs/DATA_READINESS.md`.

## Production controls

- Full configuration and named-data-checksum fingerprinting.
- Purged and embargoed walk-forward splits, HAC/block bootstrap, DSR/PBO and FDR.
- Daily position ledger, T+1 execution, explicit A-share fees, lots, limits and volume participation.
- AST capability policy, isolated runtime, resource limits, append-only budget and hash-chain audit.
- Model-agnostic Researcher/Reviewer/Executor orchestration returning institutional evidence.
- Candidate-bound, expiring, one-time holdout approval with filtered public metrics.
- Persistent factor pool and equal-risk composite loop with return-accretion and diversification gates.
