# Source-Available Release Checklist

This checklist separates repository preparation from settings that must be completed on GitHub.
The project is source-available for noncommercial use and is not OSI open source.

## Completed in the repository

- [x] PolyForm Noncommercial 1.0.0 license and matching notices
- [x] Public README with architecture, screenshots, boundaries and contact
- [x] Contributor, security and conduct policies
- [x] CI workflow for both Python packages
- [x] Issue and pull-request templates
- [x] Sanitized research snapshot with an export script
- [x] Raw data, runtime databases, logs, artifacts and `.env` files ignored
- [x] Portable default paths and no fixed private task auto-resume
- [x] Release-hygiene script for links, samples, secrets and tracked artifacts
- [x] Citation metadata and roadmap

## Verify immediately before making the repository public

- [ ] Run `uv run python scripts/check_public_release.py`
- [ ] Run both complete test suites from a clean clone
- [ ] Review the full Git history with a secret scanner
- [ ] Revoke any credential that may ever have appeared outside the current tracked tree
- [ ] Confirm all screenshots are intentionally public
- [ ] Confirm every bundled research record is approved for public release
- [ ] Confirm external market-data links and licenses independently
- [ ] Create a signed release tag from the reviewed commit

## GitHub repository settings

- [ ] Add the repository description, topics and social preview
- [ ] Enable branch protection for `main`
- [ ] Require the CI workflow before merge
- [ ] Enable Dependabot and secret scanning
- [ ] Enable private vulnerability reporting
- [ ] Configure Discussions for research design and Q&A
- [ ] Add issue labels for `factor-knowledge`, `multi-agent`, `homogeneity`, `data`, `backtest`,
      `governance` and `good-first-issue`

## Publication boundary

Making the source public does not publish or license:

- raw or derived market datasets not present in Git;
- local runtime databases and private research history;
- API credentials or provider accounts;
- hidden-test results and private LLM conversations;
- production trading approval.
