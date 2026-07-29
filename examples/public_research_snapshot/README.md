# Public Research Snapshot

This directory is a small, sanitized sample of AutoAlpha research records. It exists to document
the public schemas and make code review easier; it is not a performance benchmark.

## Contents

| File | Description |
|---|---|
| `factors.jsonl` | Twelve diverse factor definitions, expressions, lineage and selected long-only diagnostics |
| `combinations.jsonl` | Three research-leading QuantCombine candidates, weights, metrics and failed gates |
| `strategy_spec.json` | One versioned `RESEARCH` strategy specification with signal, rebalance, execution, risk and monitoring policies |
| `audit_events.jsonl` | A short sanitized event stream with a public-snapshot hash chain |
| `manifest.json` | Record counts, safety declarations and SHA-256 checksums |

The records intentionally include rejected candidates and negative evidence. A
`RESEARCH_LEADER` is the best item seen inside one search task; it is not a qualified or
production-ready strategy.

## Excluded

- raw prices, security-level returns and holdings;
- licensed market datasets;
- API keys and provider credentials;
- local filesystem paths;
- private prompts and full LLM conversations;
- precise hidden-test results;
- production order instructions.

The audit hashes authenticate this sanitized sample only. They are recomputed after redaction and do
not claim to reproduce hashes from the private source database.

## Regeneration

From the repository root:

```bash
uv run python scripts/export_public_research_snapshot.py \
  --database AutoAlpha/runtime-full-llm/autoalpha.sqlite3 \
  --output examples/public_research_snapshot
```

Review the resulting diff before publication. Do not export from a database that contains records
you are not authorized to disclose.
