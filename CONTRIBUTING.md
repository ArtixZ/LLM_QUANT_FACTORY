# Contributing to AutoAlpha

Thank you for helping improve auditable quantitative research. Contributions are welcome across
data engineering, factor knowledge management, statistical evaluation, multi-agent collaboration,
backtesting, execution modeling, documentation and user experience.

## Before you start

- Search existing issues and discussions.
- Open an issue before a large architectural change or a change to research semantics.
- Never attach licensed market data, API keys, private prompts, hidden-test results or runtime
  databases to an issue or pull request.
- Treat every historical performance claim as research evidence, not as a product promise.

By participating, you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Development setup

```bash
git clone https://github.com/khakhasshi/MultiFactorAshare.git
cd MultiFactorAshare

uv sync --frozen --all-groups
uv run pytest -q

cd AutoAlpha
uv sync --frozen --all-groups
uv run pytest -q
```

No market data is required for most unit tests. Integration tests that need a panel should use a
small synthetic fixture or a contributor-owned local dataset outside Git.

## Change expectations

### Code

- Follow the existing package boundaries and prefer small, explicit changes.
- Add or update tests in proportion to the behavioral risk.
- Keep factor expressions deterministic and validate timing, units and field availability.
- Do not add a second metric definition when an existing canonical implementation can be reused.
- Preserve backward-compatible evidence and database migrations where practical.

### Research protocols

A pull request that changes evaluation, search or promotion logic should document:

- the exact protocol and return convention;
- whether data are PIT, non-PIT or synthetic;
- exploration, public validation and hidden-test boundaries;
- costs, turnover, capacity and execution assumptions;
- the number of candidates or repeated trials;
- old/new behavior on at least one deterministic fixture;
- remaining limitations and failure cases.

Do not tune a change against hidden-test metrics. A result that fails a hard gate must stay failed.

### User interface

- Keep A-share long-only metrics primary; long-short and IC metrics are diagnostics.
- Show signal time, execution time, rebalance schedule, cost model and data limitations near results.
- Preserve the shared top navigation and keyboard-accessible form labels.
- Include a screenshot for visible changes, but remove credentials and private data.

## Validation

Run before opening a pull request:

```bash
uv run ruff check src tests scripts
uv run ruff format --check scripts/check_public_release.py scripts/export_public_research_snapshot.py
uv run pytest -q

cd AutoAlpha
uv run ruff check .
uv run pytest -q

cd ..
uv run python scripts/check_public_release.py
uv build --out-dir /tmp/multifactor-ashare-dist

cd AutoAlpha
uv build --out-dir /tmp/autoalpha-dist
```

## Pull requests

Keep each pull request focused. The description should explain the problem, design decision,
verification commands, research-semantic impact and any migration or operational risk.

By submitting a contribution, you certify that you have the right to provide it and agree that it
will be distributed under the
[PolyForm Noncommercial License 1.0.0](LICENSE). The contribution does not receive or grant
commercial-use rights through this repository.
