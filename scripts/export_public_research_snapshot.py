#!/usr/bin/env python3
"""Export a small, sanitized AutoAlpha research snapshot for public examples."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

FACTOR_IDS = (
    "F_37498e3de03ddcdb",
    "F_db8cdbe8aad8cedd",
    "F_0f2e2d24ab679a2f",
    "F_22b2fe45aa839aa2",
    "F_aa546c0c967ec911",
    "F_9e2b139455a9a05c",
    "F_a823aa37927028c6",
    "F_69ff5b84ecfe4d4d",
    "F_76e31bf6d745a558",
    "F_2c97dc313efbc63e",
    "F_18047c61d894195f",
    "F_17e9eb5ab781755b",
)

QUANT_TASK_IDS = (
    "qcombine-82bae04b2fee",
    "qcombine-85eca5364c6b",
    "qcombine-5d9ff0fe3459",
)

FACTOR_METRICS = (
    "evaluation_protocol",
    "canonical_mechanism",
    "long_only_backtest_start",
    "long_only_backtest_end",
    "long_only_sharpe_ratio",
    "long_only_simple_annual_return",
    "long_only_max_drawdown",
    "long_only_annual_turnover",
    "long_only_coverage",
    "long_only_walk_forward_fold_count",
    "long_only_walk_forward_median_sharpe",
    "long_only_walk_forward_worst_sharpe",
    "long_only_positive_year_ratio",
    "long_only_cost_stress_net_ir",
    "rank_ic_mean",
    "rank_ic_ir",
    "behavior_cluster_id",
    "behavior_cluster_role",
    "behavior_cluster_size",
    "production_promotion_gate_passed",
    "production_promotion_gate_failures",
)

PORTFOLIO_METRICS = (
    "portfolio_evaluation_protocol",
    "portfolio_execution_protocol",
    "portfolio_mode",
    "portfolio_backtest_start",
    "portfolio_backtest_end",
    "portfolio_sharpe_ratio",
    "portfolio_simple_annual_return",
    "portfolio_max_drawdown",
    "portfolio_annual_turnover",
    "portfolio_walk_forward_fold_count",
    "portfolio_walk_forward_median_sharpe",
    "portfolio_walk_forward_worst_sharpe",
    "portfolio_positive_year_ratio",
    "portfolio_max_factor_correlation",
    "portfolio_effective_factor_bets",
    "portfolio_effective_mechanisms",
    "portfolio_marginal_positive_fraction",
    "portfolio_max_strategy_active_correlation",
    "portfolio_capacity_cny",
    "portfolio_production_eligible",
    "portfolio_production_blockers",
)

LOCAL_PATH_PATTERN = re.compile(r"/" r"Users/[^/\s]+(?:/[^\s\"']+)?")
TASK_FINGERPRINT_PATTERN = re.compile(r"-task-[a-f0-9]+")
SECRET_PATTERN = re.compile(r"(?i)(api[_ -]?key|token|password|secret)\s*[:=]\s*([^\s,;]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("AutoAlpha/runtime-full-llm/autoalpha.sqlite3"),
        help="Source AutoAlpha SQLite database.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/public_research_snapshot"),
        help="Output directory.",
    )
    return parser.parse_args()


def load_json(value: str | None) -> dict[str, Any] | list[Any]:
    if not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, (dict, list)):
        raise TypeError("Expected a JSON object or array")
    return parsed


def selected(source: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: source[key] for key in keys if key in source}


def sanitize_text(value: str) -> str:
    value = LOCAL_PATH_PATTERN.sub("<local-path>", value)
    value = TASK_FINGERPRINT_PATTERN.sub("-public-demo", value)
    return SECRET_PATTERN.sub(lambda match: f"{match.group(1)}=<redacted>", value)


def sanitize(value: Any) -> Any:
    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, dict):
        return {
            key: sanitize(item)
            for key, item in value.items()
            if key
            not in {
                "api_key",
                "credential",
                "private_prompt",
                "return_artifact_path",
                "source_experiment_id",
                "task_id",
                "token",
            }
        }
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def export_factors(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    placeholders = ",".join("?" for _ in FACTOR_IDS)
    rows = connection.execute(
        f"""
        SELECT factor_id, name, family, proposal_json, metrics_json, status
        FROM factor_pool
        WHERE factor_id IN ({placeholders})
        """,
        FACTOR_IDS,
    ).fetchall()
    by_id = {row["factor_id"]: row for row in rows}
    missing = sorted(set(FACTOR_IDS) - set(by_id))
    if missing:
        raise RuntimeError(f"Missing selected factors: {missing}")

    result: list[dict[str, Any]] = []
    for factor_id in FACTOR_IDS:
        row = by_id[factor_id]
        proposal = load_json(row["proposal_json"])
        metrics = load_json(row["metrics_json"])
        assert isinstance(proposal, dict)
        assert isinstance(metrics, dict)
        result.append(
            sanitize(
                {
                    "factor_id": factor_id,
                    "name": row["name"],
                    "family": row["family"],
                    "status": row["status"],
                    "canonical_mechanism": metrics.get(
                        "canonical_mechanism",
                        proposal.get("canonical_mechanism", "UNCLASSIFIED"),
                    ),
                    "hypothesis": proposal.get("hypothesis", ""),
                    "expected_direction": proposal.get("expected_direction"),
                    "expression": proposal.get("expression"),
                    "data_lineage": proposal.get("data_lineage", {}),
                    "metrics": selected(metrics, FACTOR_METRICS),
                }
            )
        )
    return result


def export_combinations(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for public_index, task_id in enumerate(QUANT_TASK_IDS, start=1):
        row = connection.execute(
            """
            SELECT
                t.name,
                t.protocol_json,
                t.best_candidate_id,
                c.iteration,
                c.stage,
                c.algorithm,
                c.factor_ids_json,
                c.weights_json,
                c.metrics_json,
                c.score,
                c.gate_status,
                c.failed_gates_json,
                c.qualification
            FROM quant_combine_tasks AS t
            JOIN quant_combine_candidates AS c ON c.id = t.best_candidate_id
            WHERE t.task_id = ?
            """,
            (task_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"Task has no best candidate: {task_id}")
        metrics = load_json(row["metrics_json"])
        assert isinstance(metrics, dict)
        result.append(
            sanitize(
                {
                    "candidate_id": f"DEMO_COMBINATION_{public_index:03d}",
                    "source_task_name": row["name"],
                    "iteration": row["iteration"],
                    "stage": row["stage"],
                    "algorithm": row["algorithm"],
                    "factor_ids": load_json(row["factor_ids_json"]),
                    "weights": load_json(row["weights_json"]),
                    "score": row["score"],
                    "gate_status": row["gate_status"],
                    "qualification": row["qualification"],
                    "failed_gates": load_json(row["failed_gates_json"]),
                    "protocol": load_json(row["protocol_json"]),
                    "metrics": selected(metrics, PORTFOLIO_METRICS),
                }
            )
        )
    return result


def export_strategy(connection: sqlite3.Connection) -> dict[str, Any]:
    row = connection.execute(
        """
        SELECT *
        FROM formal_strategy_versions
        WHERE lifecycle = 'RESEARCH'
        ORDER BY created_at DESC
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        raise RuntimeError("No RESEARCH strategy version is available")
    return sanitize(
        {
            "strategy_uid": "STR_PUBLIC_DEMO",
            "version": 1,
            "name": row["name"],
            "market": row["market"],
            "lifecycle": row["lifecycle"],
            "signal_policy": load_json(row["signal_policy_json"]),
            "rebalance_policy": load_json(row["rebalance_policy_json"]),
            "execution_policy": load_json(row["execution_policy_json"]),
            "risk_policy": load_json(row["risk_policy_json"]),
            "cost_policy": load_json(row["cost_policy_json"]),
            "monitoring_policy": load_json(row["monitoring_policy_json"]),
            "evidence": load_json(row["evidence_json"]),
            "source_specification_hash": row["specification_hash"],
            "created_at": row["created_at"],
        }
    )


def export_events(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT timestamp_utc, category, level, event, title, message
        FROM events
        WHERE category IN ('audit', 'action', 'research', 'delivery')
        ORDER BY id DESC
        LIMIT 20
        """
    ).fetchall()
    rows = list(reversed(rows))
    previous_hash = "0" * 64
    result: list[dict[str, Any]] = []
    for row in rows:
        public_record = sanitize(
            {
                "timestamp_utc": row["timestamp_utc"],
                "category": row["category"],
                "level": row["level"],
                "event": row["event"],
                "title": row["title"],
                "message": row["message"],
                "previous_hash": previous_hash,
            }
        )
        record_hash = hashlib.sha256(canonical_json(public_record).encode()).hexdigest()
        public_record["record_hash"] = record_hash
        previous_hash = record_hash
        result.append(public_record)
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    database = args.database.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not database.is_file():
        raise SystemExit(f"Database not found: {database}")
    output.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        factors = export_factors(connection)
        combinations = export_combinations(connection)
        strategy = export_strategy(connection)
        events = export_events(connection)
    finally:
        connection.close()

    write_jsonl(output / "factors.jsonl", factors)
    write_jsonl(output / "combinations.jsonl", combinations)
    write_json(output / "strategy_spec.json", strategy)
    write_jsonl(output / "audit_events.jsonl", events)

    data_files = (
        "factors.jsonl",
        "combinations.jsonl",
        "strategy_spec.json",
        "audit_events.jsonl",
    )
    manifest = {
        "schema_version": "autoalpha-public-research-snapshot-v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source": "sanitized local AutoAlpha research database",
        "research_only": True,
        "contains_market_data": False,
        "contains_hidden_test_metrics": False,
        "contains_credentials": False,
        "audit_chain_note": (
            "Event hashes were recomputed after sanitization and authenticate only this public sample."
        ),
        "record_counts": {
            "factors": len(factors),
            "combinations": len(combinations),
            "strategies": 1,
            "audit_events": len(events),
        },
        "files": {
            name: {
                "bytes": (output / name).stat().st_size,
                "sha256": sha256(output / name),
            }
            for name in data_files
        },
    }
    write_json(output / "manifest.json", manifest)
    print(
        f"Exported {len(factors)} factors, {len(combinations)} combinations, "
        f"1 strategy and {len(events)} events to {output}"
    )


if __name__ == "__main__":
    main()
