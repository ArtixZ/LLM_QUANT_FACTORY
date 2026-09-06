# AutoAlpha Research Platform

AutoAlpha is the research-platform component of the
[LLM_QUANT_FACTORY monorepo](../README_EN.md). It provides auditable LLM-assisted factor discovery,
factor knowledge management, deterministic portfolio search, long-only backtesting, strategy
versioning and local research control planes.

The project is source-available under the PolyForm Noncommercial License 1.0.0. Commercial use
requires separate prior written permission. Market data, API credentials, runtime databases,
private model conversations and hidden-test results are not included.

> Current local US-equity validation uses a current-membership, non-PIT execution proxy. It is suitable for research
> diagnostics, not for production trading approval or investment advice.

## Quick start

Requires Python 3.12+ and `uv`.

```bash
uv sync --frozen --all-groups
uv run pytest -q

export AUTOALPHA_DATA_PATH="$PWD/../data"
export AUTOALPHA_SERVICE_TOKEN="replace-with-a-strong-local-token"
./start-services.sh --no-resume
```

Services:

| Service | Command | Default URL | Responsibility |
|---|---|---|---|
| AutoAlpha | `autoalpha-service` | http://127.0.0.1:8788 | Research tasks, factors, backtest, screener, paper portfolios, data and jobs |
| AutoCombine | `autocombine-service` | http://127.0.0.1:8888 | LLM-assisted constrained factor combination research |
| QuantCombine | `quantcombine-service` | http://127.0.0.1:8889 | Deterministic statistical combination optimization |

Stop the services:

```bash
./stop-services.sh
```

`start-services.sh` is idempotent. It resumes no task by default. To restore an explicit local task:

```bash
AUTOALPHA_RESUME_TASK_ID="task-your-id" \
AUTOCOMBINE_RESUME_TASK_ID="combine-your-id" \
./start-services.sh
```

See [`.env.example`](.env.example) for common configuration variables.

## Package map

```text
src/autoalpha/
  agents/       structured research roles and permission boundaries
  data/         PIT contracts, snapshots, universes and research-field catalog
  dsl/          typed factor expressions, timing semantics and compilation
  research/     splits, statistics, evidence matrices, hard gates and ranking
  portfolio/    risk, constraints, optimization and attribution
  execution/    A-share fills, costs, impact, capacity and TCA
  governance/   audit, blind evaluation, release and rollback
  operations/   artifacts, idempotent jobs and monitoring
  service/      persistent research loops, stores, APIs and web applications
  backtest/     vector, event-ledger and capital-account implementations
config/         versioned research protocol and threshold configuration
docs/           controls, migration notes and runbooks
tests/          unit, integration, accounting and adversarial tests
```

## Decision boundary

LLMs may:

- propose a mechanism hypothesis;
- generate an expression within the supported field and operator grammar;
- design falsification checks;
- summarize public evidence;
- advise on factor/portfolio interactions.

LLMs may not:

- read precise hidden-test metrics;
- change deterministic weights or gates without a versioned configuration change;
- approve paper or production promotion;
- infer missing PIT market state;
- place live orders;
- convert a failed hard gate into a pass.

`ResearchOrchestrator` produces a complete evidence matrix and one of:

- `REJECTED`
- `RESEARCH`
- `APPROVED_FOR_PAPER`
- `APPROVED_FOR_PRODUCTION`

Production approval additionally requires human risk authorization and a PIT-capable execution
dataset.

## Research protocol

The default protocol is US-equity long-only first:

- end-of-day signal availability;
- execution no earlier than the next session open;
- weekly capital allocation with explicit costs;
- rolling public out-of-sample folds;
- isolated hidden evaluation with categorical feedback;
- DSR, PBO and FDR for repeated research;
- parameter-neighborhood, regime, turnover, capacity and drawdown checks;
- marginal portfolio contribution and strategy-independence gates.

Rank IC and long-short returns remain diagnostic. They are not primary promotion metrics.

Detailed definitions:

- [Evaluation constitution](evaluation.md)
- [Institutional research controls](docs/INSTITUTIONAL_RESEARCH_CONTROLS.md)
- [Multi-factor research](docs/MULTIFACTOR_RESEARCH.md)
- [AutoCombine](docs/AUTOCOMBINE.md)
- [QuantCombine](docs/QUANTCOMBINE.md)
- [Vector backtest engine](docs/VECTOR_BACKTEST_ENGINE.md)
- [Data readiness](docs/DATA_READINESS.md)
- [Production runbook](docs/PRODUCTION_RUNBOOK.md)

## Data

The service accepts either a canonical panel path or the monorepo `data/` workspace. A workspace
inspection binds:

- data fingerprint and date range;
- row and security counts;
- available field catalog;
- price, volume and amount semantics;
- quality report status;
- PIT and execution blockers.

The repository does not distribute market data. Build the canonical panel from your licensed source
with the outer package:

```bash
cd ..
uv run mf-us audit
uv run mf-us catalog
uv run mf-us panel
```

Current non-PIT data must fail closed for production when historical listing, delisting, ST,
suspension, limit state, point-in-time classification or unadjusted execution prices are missing.

## Credentials and persistence

Use the OS keychain or environment variables:

```bash
export OPENAI_BASE_URL="https://api.openai.com/v1"
export AUTOALPHA_MODEL="your-model"
export AUTOALPHA_API_KEY="your-api-key"
export TUSHARE_TOKEN="your-token"
```

`AUTOALPHA_SERVICE_TOKEN` protects control-plane APIs. Non-loopback deployment also requires TLS,
network access control and reviewed filesystem mounts.

All services share `AUTOALPHA_RUNTIME` and persist:

- task state and checkpoints;
- factor and strategy registries;
- metrics, memories and role artifacts;
- audit/action/research/delivery events;
- immutable backtest and strategy artifacts.

`runtime/`, `runtime-*/`, SQLite, logs and artifacts are ignored by Git.

## Development

```bash
uv run ruff check .
uv run pytest -q
```

For a specific service:

```bash
AUTOALPHA_RUNTIME="$PWD/runtime-full-llm" AUTOALPHA_PORT=8788 \
  .venv/bin/autoalpha-service

AUTOALPHA_RUNTIME="$PWD/runtime-full-llm" AUTOCOMBINE_PORT=8888 \
  .venv/bin/autocombine-service

AUTOALPHA_RUNTIME="$PWD/runtime-full-llm" QUANTCOMBINE_PORT=8889 \
  .venv/bin/quantcombine-service
```

## Contributing and security

Use the monorepo [contribution guide](../CONTRIBUTING.md) and
[security policy](../SECURITY.md). Agents and second-stage developers should also read the
[repository handoff guide](../AGENTS.md). Large changes to metric semantics, data timing, database
contracts or promotion rules should begin with an issue and an explicit evidence contract.

## License

Copyright 2026 Jiang Jingzhe.

Licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE). Commercial use is prohibited
unless separately authorized in writing by the copyright holder. This is a source-available,
noncommercial license, not an OSI-approved open-source license. Nothing in this project constitutes
investment advice or a guarantee of performance.
