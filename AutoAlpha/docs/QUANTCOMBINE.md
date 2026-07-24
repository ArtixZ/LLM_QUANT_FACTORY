# QuantCombine

QuantCombine is the non-LLM portfolio-search service connected to the AutoAlpha factor registry.
It runs independently on port `8889` and shares the same immutable factor definitions, data
workspace, evaluation protocol and SQLite audit database as AutoAlpha and AutoCombine.

## Design goals

- Produce reproducible factor selection and weights without model-provider credentials.
- Treat factor selection, weight estimation and qualification as separate stages.
- Prefer independent return sources over differently named versions of the same signal.
- Preserve every screening, search, gate and delivery decision as structured evidence.
- Keep public research, isolated holdout evaluation and production promotion separated.

## Pipeline

1. Freeze the factor snapshot, expressions, protocol, data path and engine configuration.
2. Re-evaluate every visible factor with the common long-only A-share execution proxy.
3. Cluster factors using validation-return correlation and semantic fingerprints.
4. Run sequential forward floating selection (SFFS).
5. Run NSGA-II subset mutation and crossover when enabled.
6. Update factor-inclusion utilities and run adaptive posterior sampling when enabled.
7. Generate constrained weights from equal weight, inverse risk, shrunk maximum Sharpe,
   minimum variance, maximum diversification, CVaR and deterministic Dirichlet candidates.
8. Rank candidates by hard gates first and the configured objective second.
9. Maintain a Pareto frontier across return, Sharpe, drawdown, worst fold, turnover,
   correlation, effective factor bets and effective mechanisms.
10. Run leave-one-out marginal contribution diagnostics for the leading candidate.
11. Submit a public-gate-passing candidate once to the isolated holdout evaluator.
12. Promote only a passing production candidate to the versioned QuantCombine strategy registry.

## Engine modes

- `DETERMINISTIC`: stability screening, clustering and SFFS only.
- `EVOLUTIONARY`: deterministic stages plus NSGA-II.
- `BAYESIAN`: deterministic stages plus adaptive factor-inclusion sampling.
- `ENSEMBLE`: all stages; the production default.

## Persistence

The service uses dedicated tables in `runtime-full-llm/autoalpha.sqlite3`:

- `quant_combine_tasks`
- `quant_factor_screen`
- `quant_combine_candidates`
- `quant_combine_events`
- `quant_strategy_versions`

Daily net and active-return artifacts are written beneath
`runtime-full-llm/artifacts/quantcombine/` with SHA-256 evidence hashes.

## Run

```bash
cd /Users/jiangjingzhe/Portfolios/MultiFactorAshare/AutoAlpha
AUTOALPHA_RUNTIME="$PWD/runtime-full-llm" QUANTCOMBINE_PORT=8889 \
  .venv/bin/quantcombine-service
```
