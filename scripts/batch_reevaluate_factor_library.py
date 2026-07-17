from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from autoalpha.config import ResearchConfig
from autoalpha.data.execution_basis import expression_research_basis_blockers
from autoalpha.data.workspace import inspect_data_workspace
from autoalpha.research.batch_reevaluation import apply_batch_multiple_testing
from autoalpha.service.canonical_evaluation import (
    CANONICAL_LIBRARY_PROTOCOL,
    canonical_library_config,
    evaluate_canonical_library_factor,
)
from autoalpha.service.evaluator import PriceVolumeEvaluator
from autoalpha.service.multifactor import _candidate_screen_failures, factor_from_pool_record
from autoalpha.service.store import ServiceStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PROCESS_EVALUATOR: PriceVolumeEvaluator | None = None

# Keep native kernels single-threaded; the factor-level pool owns CPU parallelism.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reevaluate the complete persistent factor library under the current public protocol. "
            "The script never reads the configured hidden test period."
        )
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=PROJECT_ROOT / "runtime/autoalpha.sqlite3",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config/research.toml",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        help="Defaults to the data_path stored in the service database.",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--factor-id",
        action="append",
        default=[],
        help="Restrict the run for diagnostics; repeat for multiple ids.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Discard a matching checkpoint and recompute selected factors.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Atomically write the completed current-protocol metrics back to factor_pool.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Factor evaluation workers; defaults to every logical CPU.",
    )
    parser.add_argument(
        "--backend",
        choices=("process", "thread"),
        default="process",
        help="Process mode bypasses the GIL; thread mode shares one address space.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    database = args.database.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    store = ServiceStore(database)
    state = store.state()
    if state["state"] not in {"STOPPED", "WAITING_CONFIGURATION"}:
        raise RuntimeError(
            f"Automated research must be stopped before batch reevaluation; state={state['state']}"
        )
    data_path = (
        args.data_path.expanduser().resolve()
        if args.data_path
        else Path(store.settings()["data_path"]).expanduser().resolve()
    )
    workspace = inspect_data_workspace(data_path)
    config = canonical_library_config(
        ResearchConfig.from_toml(config_path),
        data_start=datetime.fromisoformat(workspace.first_trade_date).date(),
    )
    output_root = PROJECT_ROOT / "output/reevaluation" / config.governance.protocol_version
    checkpoint_path = (args.checkpoint or output_root / "checkpoint.json").resolve()
    report_path = (args.report or output_root / "factor_library_report.json").resolve()

    all_records = sorted(store.factor_pool(limit=5000), key=lambda item: item["source_iteration"])
    requested = set(args.factor_id)
    records = [
        record for record in all_records if not requested or record["factor_id"] in requested
    ]
    missing_requested = sorted(requested - {str(record["factor_id"]) for record in records})
    if missing_requested:
        raise KeyError(f"Unknown requested factor ids: {missing_requested}")
    if not records:
        raise RuntimeError("No factors were selected for reevaluation")

    evaluator = PriceVolumeEvaluator(data_path, config=config)
    evaluator.set_trial_count(len(all_records))
    metadata = {
        "protocol": config.governance.protocol_version,
        "canonical_protocol": CANONICAL_LIBRARY_PROTOCOL,
        "generation": config.generation,
        "data_path": str(data_path),
        "data_fingerprint": evaluator.workspace.fingerprint,
        "config_sha256": _file_hash(config_path),
        "factor_ids": [str(record["factor_id"]) for record in records],
        "candidate_family_size": len(all_records),
        "public_validation_end": config.splits.validation.end.isoformat(),
        "public_validation_start": config.splits.validation.start.isoformat(),
        "recent_evaluation_start": "2020-01-01",
        "recent_evaluation_end": "2024-12-31",
        "hidden_period_accessed": False,
        "return_convention": "EOD_T__OPEN_T1_TO_OPEN_T2",
    }
    checkpoint = _load_checkpoint(checkpoint_path, metadata, fresh=args.fresh)
    results: dict[str, dict[str, Any]] = checkpoint["results"]

    if args.apply:
        store.append_event(
            "audit",
            "FACTOR_LIBRARY_REEVALUATION_STARTED",
            "因子库批量重评开始",
            f"按 {config.governance.protocol_version} 重评 {len(records)} 个因子。",
            run_id=state.get("run_id"),
            iteration=state.get("iteration"),
            payload={**metadata, "workers": args.workers, "backend": args.backend},
        )

    pending = [
        record
        for record in records
        if not (
            results.get(str(record["factor_id"]))
            and results[str(record["factor_id"])].get("metrics") is not None
        )
    ]
    completed = len(records) - len(pending)
    if completed:
        print(f"resume {completed}/{len(records)} completed factors", flush=True)
    shared_fields = evaluator._load_fields()
    local = threading.local()

    def evaluate_record(record: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        thread_evaluator = getattr(local, "evaluator", None)
        if thread_evaluator is None:
            thread_evaluator = PriceVolumeEvaluator(data_path, config=config)
            thread_evaluator.set_trial_count(len(all_records))
            thread_evaluator._fields = shared_fields
            local.evaluator = thread_evaluator
        return _evaluate_record(thread_evaluator, record)

    if args.backend == "process":
        global _PROCESS_EVALUATOR
        _PROCESS_EVALUATOR = evaluator
        pool = ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=get_context("fork"),
        )
        submit = _process_evaluate_record
    else:
        pool = ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="factor-reeval")
        submit = evaluate_record
    with pool:
        futures = {pool.submit(submit, record): record for record in pending}
        for future in as_completed(futures):
            factor_id, result = future.result()
            results[factor_id] = result
            completed += 1
            outcome = "ok" if result["error"] is None else "failed"
            print(
                f"[{completed:03d}/{len(records):03d}] {outcome} {factor_id} "
                f"{result['elapsed_seconds']:.2f}s",
                flush=True,
            )
            _write_json_atomic(checkpoint_path, {"metadata": metadata, "results": results})

    errors = {
        factor_id: item["error"]
        for factor_id, item in results.items()
        if factor_id in metadata["factor_ids"] and item.get("error")
    }
    if errors:
        raise RuntimeError(
            f"{len(errors)} factor evaluations failed; checkpoint retained at {checkpoint_path}"
        )

    raw_metrics = {
        str(record["factor_id"]): results[str(record["factor_id"])]["metrics"] for record in records
    }
    adjusted, family_pbo = apply_batch_multiple_testing(
        raw_metrics,
        alpha=config.evaluation.maximum_net_return_p_value,
    )
    contaminated = store.contaminated_factor_ids()
    batch_id = datetime.now(UTC).strftime("reeval-%Y%m%dT%H%M%SZ")
    updates = []
    report_factors = []
    for record in records:
        factor_id = str(record["factor_id"])
        metrics = adjusted[factor_id]
        data_basis_blockers = expression_research_basis_blockers(
            record["proposal"]["expression"], evaluator.execution_basis
        )
        metrics.update(
            {
                "reevaluation_batch_id": batch_id,
                "reevaluated_at": datetime.now(UTC).isoformat(),
                "previous_evaluation_protocol": record["metrics"].get("evaluation_protocol"),
                "holdout_contaminated": factor_id in contaminated,
                "hidden_period_accessed": False,
                "data_basis_compatible": not data_basis_blockers,
                "data_basis_blockers": list(data_basis_blockers),
            }
        )
        failures = _candidate_screen_failures(metrics, config)
        status = "ELIGIBLE" if not failures else "SCREENED_OUT"
        suffix = "; current holdout contaminated" if factor_id in contaminated else ""
        reason = (
            f"{config.governance.protocol_version} batch screen passed{suffix}"
            if not failures
            else f"{config.governance.protocol_version}: {', '.join(failures)}{suffix}"
        )
        updates.append(
            {
                "factor_id": factor_id,
                "metrics": metrics,
                "status": status,
                "status_reason": reason,
            }
        )
        report_factors.append(
            {
                "factor_id": factor_id,
                "name": record["name"],
                "family": record["family"],
                "source_iteration": record["source_iteration"],
                "previous_status": record["status"],
                "new_status": status,
                "gate_failures": failures,
                "holdout_contaminated": factor_id in contaminated,
                "metrics": metrics,
            }
        )

    report = {
        "batch_id": batch_id,
        "created_at": datetime.now(UTC).isoformat(),
        "metadata": metadata,
        "summary": {
            "selected_factors": len(records),
            "eligible": sum(item["new_status"] == "ELIGIBLE" for item in report_factors),
            "promotion_eligible": sum(
                item["new_status"] == "ELIGIBLE" and not item["holdout_contaminated"]
                for item in report_factors
            ),
            "screened_out": sum(item["new_status"] == "SCREENED_OUT" for item in report_factors),
            "holdout_contaminated": sum(
                bool(item["holdout_contaminated"]) for item in report_factors
            ),
            "family_probability_backtest_overfitting": family_pbo,
            "hidden_period_accessed": False,
            "workers": args.workers,
            "backend": args.backend,
        },
        "factors": report_factors,
    }
    report_hash = _value_hash(report)
    report["report_sha256"] = report_hash
    _write_json_atomic(report_path, report)

    backup_path = None
    if args.apply:
        backup_path = _backup_database(database, batch_id)
        applied = store.apply_factor_reevaluations(updates)
        store.append_event(
            "delivery",
            "FACTOR_LIBRARY_REEVALUATION_COMPLETED",
            "因子库批量重评完成",
            f"已原子更新 {applied} 个因子的当前协议指标。",
            run_id=state.get("run_id"),
            iteration=state.get("iteration"),
            payload={
                **report["summary"],
                "batch_id": batch_id,
                "report_path": str(report_path),
                "report_sha256": report_hash,
                "database_backup": str(backup_path),
            },
        )

    print(
        json.dumps(
            {
                **report["summary"],
                "applied": bool(args.apply),
                "report": str(report_path),
                "checkpoint": str(checkpoint_path),
                "backup": str(backup_path) if backup_path else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _load_checkpoint(path: Path, metadata: dict[str, Any], *, fresh: bool) -> dict[str, Any]:
    if fresh or not path.exists():
        return {"metadata": metadata, "results": {}}
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("metadata") != metadata:
        raise RuntimeError(
            f"Checkpoint protocol/data/factor set does not match this run: {path}; use --fresh"
        )
    return value


def _process_evaluate_record(record: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if _PROCESS_EVALUATOR is None:
        raise RuntimeError("Process evaluator was not initialized before worker fork")
    return _evaluate_record(_PROCESS_EVALUATOR, record)


def _evaluate_record(
    evaluator: PriceVolumeEvaluator,
    record: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    started = datetime.now(UTC)
    try:
        metrics = evaluate_canonical_library_factor(
            evaluator,
            factor_from_pool_record(record),
            trials=evaluator.trial_count,
            source_task_metrics=record.get("metrics"),
        )
        result = {
            "metrics": _json_ready(metrics),
            "error": None,
            "elapsed_seconds": (datetime.now(UTC) - started).total_seconds(),
        }
    except Exception as error:  # noqa: BLE001
        result = {
            "metrics": None,
            "error": f"{type(error).__name__}: {error}",
            "elapsed_seconds": (datetime.now(UTC) - started).total_seconds(),
        }
    finally:
        evaluator._signal_cache.clear()
    return str(record["factor_id"]), result


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_ready(value), ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _backup_database(database: Path, batch_id: str) -> Path:
    backup = database.parent / "backups" / f"autoalpha-before-{batch_id}.sqlite3"
    backup.parent.mkdir(parents=True, exist_ok=True)
    if backup.exists():
        raise FileExistsError(f"Refusing to overwrite database backup: {backup}")
    source = sqlite3.connect(database)
    target = sqlite3.connect(backup)
    try:
        source.backup(target)
    finally:
        target.close()
        source.close()
    return backup


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _value_hash(value: Any) -> str:
    encoded = json.dumps(
        _json_ready(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


if __name__ == "__main__":
    main()
