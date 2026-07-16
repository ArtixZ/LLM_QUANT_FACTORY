# Evaluation Protocol Migration V2

## Status

`institutional_v2` is the only supported evaluation generation. The former scalar-score workflow,
its scripts, utility function, journals, snapshots, and score leaderboard are removed rather than
kept as compatibility APIs.

## Contract changes

- Executors return a complete `CandidateEvidence`, never a scalar score.
- `EvaluationMatrix` separates portfolio value, statistical reliability, risk/tradability, and IC
  diagnostics.
- `InstitutionalAdmission` executes sequential hard gates and stops on the first failure.
- The experiment ledger records admission decisions and research retention, not parent/child score
  deltas.
- IC has no admission weight. A weak IC candidate may proceed when paired portfolio increment is
  reliable; a high IC candidate fails when economic, risk, capacity, or tradability evidence fails.
- Candidates that pass the same gate level may be Pareto-ranked; Pareto rank cannot promote a gate
  failure.

## Migration rule

Historical scalar-score artifacts are evidence from a different and unsupported protocol. They may
be retained in cold storage for audit, but cannot be loaded into the production factor registry,
compared with `institutional_v2`, or used to seed production approval. Re-evaluation requires a new
candidate ID, current PIT snapshot, current control portfolio, and a complete V2 evidence matrix.

Any future change to data timing, splits, control, costs, thresholds, or holdout rules creates a new
research generation and protocol fingerprint.
