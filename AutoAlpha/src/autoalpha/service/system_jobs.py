from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from autoalpha.config import ResearchConfig
from autoalpha.data.research_fields import expression_fields
from autoalpha.data.workspace import inspect_data_workspace
from autoalpha.service.autocombine import DEFAULT_CONSTRUCTION, OBJECTIVE_PRESETS
from autoalpha.service.autocombine_store import AutoCombineStore
from autoalpha.service.direction import classify_mechanism
from autoalpha.service.factor_behavior import load_behavior_snapshot
from autoalpha.service.factor_homogeneity import build_homogeneity_report
from autoalpha.service.gate_feedback import (
    append_gate_feedback_notes,
    apply_gate_feedback,
    gate_feedback_policy,
)
from autoalpha.service.mechanism import normalize_mechanism
from autoalpha.service.quantcombine import (
    DEFAULT_BUDGET as QUANT_DEFAULT_BUDGET,
)
from autoalpha.service.quantcombine import (
    DEFAULT_ENGINE as QUANT_DEFAULT_ENGINE,
)
from autoalpha.service.quantcombine import (
    create_quant_task_record,
)
from autoalpha.service.quantcombine_store import QuantCombineStore
from autoalpha.service.research_protocol import (
    default_task_protocol,
    normalize_task_protocol,
    panel_validation_fold_capacity,
)
from autoalpha.service.store import ServiceStore
from autoalpha.service.strategy_bus import (
    advance_formal_strategy_lifecycle,
    build_strategy_bus_snapshot,
    create_formal_strategy_from_experiment,
    factor_knowledge_map,
    strategy_lifecycle_readiness,
    strategy_promotion_candidates,
    strategy_public_validation_gap,
)

SUPPORTED_SYSTEM_JOB_TYPES = {
    "factor_library_refresh",
    "factor_homogeneity_backfill",
    "factor_knowledge_map_sync",
    "gate_feedback_policy_sync",
    "gate_funnel_diagnostics",
    "quantcombine_repair_task_seed",
    "strategy_library_seed",
    "strategy_public_validation_freeze",
    "strategy_bus_sync",
    "market_data_sync",
}

SNAPSHOT_TTLS = {
    "factor_library": 900,
    "factor_homogeneity_backfill": 3600,
    "factor_knowledge_map": 1800,
    "gate_funnel_diagnostics": 900,
    "gate_feedback_policy": 900,
    "strategy_bus": 300,
}


class SystemJobRunner:
    """Execute small control-plane jobs through the unified system_jobs queue."""

    def __init__(
        self,
        store: ServiceStore,
        *,
        autocombine_store: AutoCombineStore,
        quantcombine_store: QuantCombineStore,
        runtime_root: Path,
        market_data_sync_runner: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        factor_library_builder: Callable[[], dict[str, Any]] | None = None,
        worker_id: str = "autoalpha-system-job-runner",
    ) -> None:
        self.store = store
        self.autocombine_store = autocombine_store
        self.quantcombine_store = quantcombine_store
        self.runtime_root = runtime_root
        self.market_data_sync_runner = market_data_sync_runner
        self.factor_library_builder = factor_library_builder
        self.worker_id = worker_id

    def run_next(
        self,
        *,
        queue: str = "system",
        lease_seconds: int = 900,
        max_queue_running: int | None = None,
        max_global_running: int | None = None,
    ) -> dict[str, Any]:
        self.store.recover_expired_system_jobs(queue=queue)
        job = self.store.claim_system_job(
            queue=queue,
            worker_id=self.worker_id,
            lease_seconds=lease_seconds,
            max_queue_running=max_queue_running,
            max_global_running=max_global_running,
        )
        if job is None:
            return {"claimed": False, "supported_job_types": sorted(SUPPORTED_SYSTEM_JOB_TYPES)}
        try:
            if job["job_type"] not in SUPPORTED_SYSTEM_JOB_TYPES:
                self.store.update_system_job(
                    job["job_id"],
                    status="BLOCKED_UNSUPPORTED",
                    lease_owner=None,
                    lease_expires_at=None,
                    heartbeat_at=None,
                    error=f"Unsupported system job type: {job['job_type']}",
                )
                return {"claimed": False, "unsupported_job": job}
            result = self.run_claimed(job)
            return {"claimed": True, "job": result}
        except Exception as error:
            self.store.update_system_job(
                job["job_id"],
                status="FAILED",
                error=f"{type(error).__name__}: {error}",
                result={"failed_at": _now(), "job_type": job["job_type"]},
                finished_at=_now(),
            )
            raise

    def run_claimed(self, job: dict[str, Any]) -> dict[str, Any]:
        job_type = str(job["job_type"])
        if job_type == "factor_library_refresh":
            result = self.materialize_factor_library(job)
        elif job_type == "factor_homogeneity_backfill":
            result = self.backfill_factor_homogeneity(job)
        elif job_type == "factor_knowledge_map_sync":
            result = self.materialize_factor_knowledge_map(job)
        elif job_type == "gate_funnel_diagnostics":
            result = self.materialize_gate_funnel(job)
        elif job_type == "gate_feedback_policy_sync":
            result = self.materialize_gate_feedback_policy(job)
        elif job_type == "quantcombine_repair_task_seed":
            result = self.seed_quantcombine_repair_task(job)
        elif job_type == "strategy_library_seed":
            result = self.seed_strategy_library(job)
        elif job_type == "strategy_public_validation_freeze":
            result = self.freeze_public_validation_ready_strategies(job)
        elif job_type == "strategy_bus_sync":
            result = self.materialize_strategy_bus(job)
        elif job_type == "market_data_sync":
            if self.market_data_sync_runner is None:
                raise RuntimeError("Market data sync runner is not configured")
            result = self.market_data_sync_runner(job)
        else:
            raise ValueError(f"Unsupported system job type: {job_type}")
        latest = self.store.system_job(str(job["job_id"]))
        if latest["status"] == "CANCEL_REQUESTED":
            return self.store.update_system_job(
                job["job_id"],
                status="CANCELLED",
                result={
                    "cancelled_at": _now(),
                    "job_type": job_type,
                    "partial_result": result,
                },
                lease_owner=None,
                lease_expires_at=None,
                heartbeat_at=_now(),
                finished_at=_now(),
            )
        if latest["status"] == "PAUSE_REQUESTED":
            return self.store.update_system_job(
                job["job_id"],
                status="PAUSED",
                checkpoint={
                    "paused_at": _now(),
                    "job_type": job_type,
                    "partial_result": result,
                },
                lease_owner=None,
                lease_expires_at=None,
                heartbeat_at=None,
            )
        return self.store.update_system_job(
            job["job_id"],
            status="COMPLETED",
            progress_current=job.get("progress_total") or result.get("processed_count", 1),
            result=result,
            lease_owner=None,
            lease_expires_at=None,
            heartbeat_at=_now(),
            finished_at=_now(),
        )

    def backfill_factor_homogeneity(self, job: dict[str, Any]) -> dict[str, Any]:
        payload = job.get("payload") or {}
        source_task_id = payload.get("source_task_id")
        limit = int(payload.get("limit") or 5000)
        pool_records = self.store.factor_pool(limit=limit)
        behavior = load_behavior_snapshot(self.runtime_root / "factor-behavior")
        behavior_factors = behavior.get("factors") or {}
        report = build_homogeneity_report(
            pool_records,
            behavior,
            source_task_id=str(source_task_id) if source_task_id else None,
        )
        cluster_members = _cluster_members(behavior_factors)
        processed = 0
        for record in pool_records:
            if source_task_id and str(record.get("source_task_id")) != str(source_task_id):
                continue
            factor_id = str(record["factor_id"])
            evidence = behavior_factors.get(factor_id, {})
            mechanism = _canonical_mechanism(record)
            cluster_id = evidence.get("behavior_cluster_id") or (
                (record.get("metrics") or {}).get("online_behavior_cluster_id")
            )
            redundancy = evidence.get("behavior_redundancy") or (
                (record.get("metrics") or {}).get("online_behavior_redundancy")
            )
            tags = sorted(
                {
                    "AUTOALPHA",
                    mechanism,
                    f"BEHAVIOR_CLUSTER:{cluster_id}" if cluster_id else "BEHAVIOR_PENDING",
                    f"BEHAVIOR_ROLE:{evidence.get('behavior_cluster_role', 'PENDING')}",
                    f"REDUNDANCY:{redundancy or 'PENDING'}",
                }
            )
            related = _related_factors(factor_id, evidence, cluster_members)
            review = {
                "protocol": "MATERIALIZED_FACTOR_HOMOGENEITY_BACKFILL_V1",
                "canonical_mechanism": mechanism,
                "raw_family": record.get("family"),
                "raw_canonical_mechanism": (record.get("proposal") or {}).get(
                    "canonical_mechanism"
                ),
                "behavior_cluster_id": cluster_id,
                "behavior_cluster_size": evidence.get("behavior_cluster_size"),
                "behavior_cluster_role": evidence.get("behavior_cluster_role"),
                "behavior_redundancy": redundancy,
                "nearest_factor_id": evidence.get("behavior_nearest_factor_id"),
                "nearest_similarity": evidence.get("behavior_nearest_similarity"),
                "expression_signature": _expression_signature(
                    (record.get("proposal") or {}).get("expression")
                ),
                "parameter_family": _parameter_family(
                    (record.get("proposal") or {}).get("expression")
                ),
            }
            self.store.upsert_factor_knowledge(
                factor_id=factor_id,
                canonical_mechanism=mechanism,
                mechanism_summary=_mechanism_summary(record, mechanism, evidence),
                tags=tags,
                review=review,
                falsification={
                    "non_pit_caveat": True,
                    "crowded_cluster": _as_int(evidence.get("behavior_cluster_size")) >= 8,
                    "near_duplicate": (
                        _as_float(evidence.get("behavior_nearest_similarity")) >= 0.92
                    ),
                    "requires_event_engine_confirmation": True,
                },
                related_factors=related,
            )
            self.store.merge_factor_pool_metrics(
                factor_id,
                _homogeneity_metric_patch(review, evidence, related),
            )
            processed += 1
            if processed % 100 == 0:
                self.store.heartbeat_system_job(
                    job["job_id"],
                    worker_id=self.worker_id,
                    progress_current=processed,
                    checkpoint={"last_factor_id": factor_id},
                )
        snapshot = {
            "protocol": "MATERIALIZED_FACTOR_HOMOGENEITY_BACKFILL_V1",
            "created_at": _now(),
            "source_task_id": source_task_id,
            "processed_count": processed,
            "behavior_status": behavior.get("status"),
            "behavior_snapshot_id": behavior.get("snapshot_id"),
            "report": report,
        }
        self.store.upsert_materialized_snapshot(
            "factor_homogeneity_backfill",
            snapshot,
            ttl_seconds=SNAPSHOT_TTLS["factor_homogeneity_backfill"],
            source=f"job:{job['job_id']}",
        )
        return snapshot

    def materialize_factor_library(self, job: dict[str, Any]) -> dict[str, Any]:
        if self.factor_library_builder is None:
            raise RuntimeError("Factor library builder is not configured")
        snapshot = {
            **self.factor_library_builder(),
            "api_payload_protocol": "MATERIALIZED_FACTOR_LIBRARY_API_V1",
            "materialized": False,
            "protocol": "MATERIALIZED_FACTOR_LIBRARY_API_V1",
            "created_at": _now(),
        }
        self.store.upsert_materialized_snapshot(
            "factor_library",
            snapshot,
            ttl_seconds=SNAPSHOT_TTLS["factor_library"],
            source=f"job:{job['job_id']}",
        )
        return {
            "protocol": "MATERIALIZED_FACTOR_LIBRARY_REFRESH_V1",
            "factor_count": (snapshot.get("summary") or {}).get("factor_count"),
            "processed_count": len(snapshot.get("factors") or []),
            "created_at": snapshot["created_at"],
        }

    def materialize_factor_knowledge_map(self, job: dict[str, Any]) -> dict[str, Any]:
        snapshot = factor_knowledge_map(
            self.store,
            behavior_snapshot=load_behavior_snapshot(self.runtime_root / "factor-behavior"),
        )
        snapshot = {
            **snapshot,
            "protocol": "MATERIALIZED_FACTOR_KNOWLEDGE_MAP_V1",
            "created_at": _now(),
            "processed_count": snapshot.get("factor_count", 0),
        }
        self.store.upsert_materialized_snapshot(
            "factor_knowledge_map",
            snapshot,
            ttl_seconds=SNAPSHOT_TTLS["factor_knowledge_map"],
            source=f"job:{job['job_id']}",
        )
        return snapshot

    def materialize_gate_funnel(self, job: dict[str, Any]) -> dict[str, Any]:
        snapshot = build_gate_funnel_diagnostics(
            self.autocombine_store,
            self.quantcombine_store,
        )
        self.store.upsert_materialized_snapshot(
            "gate_funnel_diagnostics",
            snapshot,
            ttl_seconds=SNAPSHOT_TTLS["gate_funnel_diagnostics"],
            source=f"job:{job['job_id']}",
        )
        return snapshot

    def materialize_gate_feedback_policy(self, job: dict[str, Any]) -> dict[str, Any]:
        gate_snapshot = build_gate_funnel_diagnostics(
            self.autocombine_store,
            self.quantcombine_store,
        )
        self.store.upsert_materialized_snapshot(
            "gate_funnel_diagnostics",
            gate_snapshot,
            ttl_seconds=SNAPSHOT_TTLS["gate_funnel_diagnostics"],
            source=f"job:{job['job_id']}",
        )
        policy = gate_feedback_policy(self.store)
        policy = {
            **policy,
            "created_at": _now(),
            "processed_count": len(policy.get("action_ids") or []),
        }
        self.store.upsert_materialized_snapshot(
            "gate_feedback_policy",
            policy,
            ttl_seconds=SNAPSHOT_TTLS["gate_feedback_policy"],
            source=f"job:{job['job_id']}",
        )
        return policy

    def seed_quantcombine_repair_task(self, job: dict[str, Any]) -> dict[str, Any]:
        payload = job.get("payload") or {}
        feedback = self.materialize_gate_feedback_policy(job)
        settings = self.store.settings()
        data_path = Path(
            payload.get("data_path")
            or settings.get("data_path")
            or self.runtime_root.parent / "data"
        ).expanduser()
        protocol = payload.get("protocol")
        if not isinstance(protocol, dict):
            workspace = inspect_data_workspace(data_path)
            protocol = default_task_protocol(
                workspace.first_trade_date,
                workspace.last_trade_date,
                ResearchConfig.from_toml(
                    Path(
                        payload.get("config_path")
                        or self.runtime_root.parent / "config/research.toml"
                    )
                ),
            )
        protocol, protocol_guard = _guard_repair_protocol_against_panel(protocol, data_path)
        profile = str(feedback.get("profile_override") or "DRAWDOWN_FIRST")
        objective_preset = OBJECTIVE_PRESETS.get(profile, OBJECTIVE_PRESETS["DRAWDOWN_FIRST"])
        objective = {
            key: value
            for key, value in objective_preset.items()
            if key not in {"label", "description"}
        }
        objective = apply_gate_feedback(
            objective,
            feedback["adjustments"]["objective"],
        )
        construction = apply_gate_feedback(
            {
                **DEFAULT_CONSTRUCTION,
                "min_factors": int(payload.get("min_factors") or 2),
                "max_factors": int(payload.get("max_factors") or 5),
            },
            feedback["adjustments"]["construction"],
        )
        engine = apply_gate_feedback(
            dict(QUANT_DEFAULT_ENGINE),
            feedback["adjustments"]["engine"],
        )
        budget = apply_gate_feedback(
            dict(QUANT_DEFAULT_BUDGET),
            feedback["adjustments"]["budget"],
        )
        scope = {
            "mode": payload.get("scope_mode") or "SMART",
            "factor_ids": list(payload.get("factor_ids") or []),
            "required_factor_ids": list(payload.get("required_factor_ids") or []),
            "excluded_factor_ids": list(payload.get("excluded_factor_ids") or []),
            "source_task_ids": list(payload.get("source_task_ids") or []),
            "statuses": list(payload.get("statuses") or ["ELIGIBLE", "SCREENED_OUT"]),
            "families": list(payload.get("families") or []),
        }
        auto_start_task = bool(payload.get("auto_start_task", True))
        record = create_quant_task_record(
            self.store,
            name=str(payload.get("name") or "Gate feedback repair · QuantCombine"),
            market=str(payload.get("market") or "CN_A"),
            data_path=str(data_path),
            protocol=protocol,
            scope=scope,
            construction=construction,
            objective=objective,
            engine=engine,
            budget=budget,
            notes=append_gate_feedback_notes(
                _append_protocol_guard_note(
                    _append_quant_autostart_note(
                        str(
                            payload.get("notes")
                            or "Seeded by AutoAlpha gate feedback repair job."
                        ),
                        auto_start_task,
                    ),
                    protocol_guard,
                ),
                feedback,
            ),
        )
        if int(record["budget"]["maximum_evaluations"]) <= len(record["factor_snapshot"]):
            record["budget"]["maximum_evaluations"] = len(record["factor_snapshot"]) + max(
                10,
                int(record["construction"]["candidate_pool_limit"]),
            )
        task = self.quantcombine_store.create_task(record)
        self.quantcombine_store.event(
            task["task_id"],
            "action",
            "QUANT_REPAIR_TASK_CREATED",
            "门禁反馈修复型 QuantCombine 任务已创建",
            f"{task['factor_count']} 个因子 · profile={task['objective']['profile']}",
            payload={
                "feedback_protocol": feedback["protocol"],
                "feedback_action_ids": feedback.get("action_ids") or [],
                "snapshot_hash": task["snapshot_hash"],
            },
        )
        return {
            "protocol": "AUTOALPHA_QUANTCOMBINE_REPAIR_TASK_SEED_V1",
            "created_at": _now(),
            "processed_count": 1,
            "task_id": task["task_id"],
            "task_url": f"http://127.0.0.1:8889/tasks/{task['task_id']}",
            "auto_start_requested": auto_start_task,
            "factor_count": task["factor_count"],
            "snapshot_hash": task["snapshot_hash"],
            "feedback_policy": {
                "protocol": feedback["protocol"],
                "source_fingerprint": feedback.get("source_fingerprint"),
                "action_ids": feedback.get("action_ids") or [],
                "profile_override": feedback.get("profile_override"),
            },
            "construction": task["construction"],
            "objective": task["objective"],
            "engine": task["engine"],
            "budget": task["budget"],
            "task_protocol": task["protocol"],
            "protocol_guard": protocol_guard,
        }

    def materialize_strategy_bus(self, job: dict[str, Any]) -> dict[str, Any]:
        snapshot = build_strategy_bus_snapshot(
            self.store,
            autocombine_store=self.autocombine_store,
            quantcombine_store=self.quantcombine_store,
            behavior_snapshot=load_behavior_snapshot(self.runtime_root / "factor-behavior"),
            sync=True,
        )
        self.store.upsert_materialized_snapshot(
            "strategy_bus",
            snapshot,
            ttl_seconds=SNAPSHOT_TTLS["strategy_bus"],
            source=f"job:{job['job_id']}",
        )
        return {
            "protocol": "MATERIALIZED_STRATEGY_BUS_SYNC_V1",
            "created_at": _now(),
            "summary": snapshot["summary"],
            "processed_count": sum(snapshot["summary"].get("by_stage", {}).values()),
        }

    def seed_strategy_library(self, job: dict[str, Any]) -> dict[str, Any]:
        payload = job.get("payload") or {}
        limit = max(1, min(int(payload.get("limit") or 10), 100))
        allowed_classes = {
            str(value)
            for value in payload.get(
                "candidate_classes",
                ["QUALIFIED", "RESEARCH_LEADER"],
            )
        }
        bus_snapshot = build_strategy_bus_snapshot(
            self.store,
            autocombine_store=self.autocombine_store,
            quantcombine_store=self.quantcombine_store,
            behavior_snapshot=load_behavior_snapshot(self.runtime_root / "factor-behavior"),
            sync=True,
        )
        candidates = [
            item
            for item in strategy_promotion_candidates(self.store, limit=limit * 5)
            if str(item.get("candidate_class")) in allowed_classes
        ][:limit]
        created: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []
        for index, candidate in enumerate(candidates, start=1):
            experiment_id = str(candidate["experiment_id"])
            try:
                strategy = create_formal_strategy_from_experiment(
                    self.store,
                    experiment_id,
                    name=str(candidate.get("title") or experiment_id),
                    lifecycle="RESEARCH",
                )
                created.append(
                    {
                        "experiment_id": experiment_id,
                        "strategy_uid": strategy["strategy_uid"],
                        "version": strategy["version"],
                        "candidate_class": candidate.get("candidate_class"),
                    }
                )
            except (KeyError, RuntimeError, ValueError) as error:
                failed.append(
                    {
                        "experiment_id": experiment_id,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
            self.store.heartbeat_system_job(
                job["job_id"],
                worker_id=self.worker_id,
                progress_current=index,
                checkpoint={
                    "last_experiment_id": experiment_id,
                    "created_count": len(created),
                    "failed_count": len(failed),
                },
            )
        refreshed = build_strategy_bus_snapshot(
            self.store,
            autocombine_store=self.autocombine_store,
            quantcombine_store=self.quantcombine_store,
            behavior_snapshot=load_behavior_snapshot(self.runtime_root / "factor-behavior"),
            sync=True,
        )
        self.store.upsert_materialized_snapshot(
            "strategy_bus",
            refreshed,
            ttl_seconds=SNAPSHOT_TTLS["strategy_bus"],
            source=f"job:{job['job_id']}",
        )
        return {
            "protocol": "AUTOALPHA_STRATEGY_LIBRARY_SEED_V1",
            "created_at": _now(),
            "processed_count": len(candidates),
            "created_count": len(created),
            "failed_count": len(failed),
            "created": created,
            "failed": failed,
            "candidate_classes": sorted(allowed_classes),
            "initial_bus_summary": bus_snapshot["summary"],
            "refreshed_bus_summary": refreshed["summary"],
        }

    def freeze_public_validation_ready_strategies(self, job: dict[str, Any]) -> dict[str, Any]:
        payload = job.get("payload") or {}
        limit = max(1, min(int(payload.get("limit") or 100), 1000))
        strategies = [
            item
            for item in self.store.formal_strategy_versions(limit=5000)
            if str(item.get("lifecycle")) == "RESEARCH"
        ][:limit]
        frozen: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []
        for index, strategy in enumerate(strategies, start=1):
            strategy_uid = str(strategy["strategy_uid"])
            version = int(strategy["version"])
            try:
                readiness = strategy_lifecycle_readiness(self.store, strategy_uid, version)
                if readiness.get("next_lifecycle") != "FROZEN" or not readiness.get("ready"):
                    skipped.append(
                        {
                            "strategy_uid": strategy_uid,
                            "version": version,
                            "missing_evidence": readiness.get("missing_evidence") or [],
                            "public_validation_gap": strategy_public_validation_gap(
                                self.store, strategy
                            ),
                        }
                    )
                    continue
                updated = advance_formal_strategy_lifecycle(self.store, strategy_uid, version)
                frozen.append(
                    {
                        "strategy_uid": strategy_uid,
                        "version": version,
                        "lifecycle": updated["lifecycle"],
                        "source_experiment_id": updated.get("source_experiment_id"),
                    }
                )
            except (KeyError, RuntimeError, ValueError) as error:
                failed.append(
                    {
                        "strategy_uid": strategy_uid,
                        "version": version,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
            self.store.heartbeat_system_job(
                job["job_id"],
                worker_id=self.worker_id,
                progress_current=index,
                checkpoint={
                    "last_strategy_uid": strategy_uid,
                    "frozen_count": len(frozen),
                    "skipped_count": len(skipped),
                    "failed_count": len(failed),
                },
            )
        refreshed = build_strategy_bus_snapshot(
            self.store,
            autocombine_store=self.autocombine_store,
            quantcombine_store=self.quantcombine_store,
            behavior_snapshot=load_behavior_snapshot(self.runtime_root / "factor-behavior"),
            sync=True,
        )
        self.store.upsert_materialized_snapshot(
            "strategy_bus",
            refreshed,
            ttl_seconds=SNAPSHOT_TTLS["strategy_bus"],
            source=f"job:{job['job_id']}",
        )
        return {
            "protocol": "AUTOALPHA_STRATEGY_PUBLIC_VALIDATION_FREEZE_V1",
            "created_at": _now(),
            "processed_count": len(strategies),
            "frozen_count": len(frozen),
            "skipped_count": len(skipped),
            "failed_count": len(failed),
            "frozen": frozen,
            "skipped": skipped,
            "failed": failed,
            "refreshed_bus_summary": refreshed["summary"],
        }


def build_gate_funnel_diagnostics(
    autocombine_store: AutoCombineStore,
    quantcombine_store: QuantCombineStore,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for task in autocombine_store.tasks():
        for experiment in autocombine_store.experiments(str(task["task_id"]), limit=5000):
            rows.append(
                {
                    "system": "AUTOCOMBINE",
                    "task_id": task["task_id"],
                    "candidate_id": experiment["id"],
                    "gate_status": experiment.get("gate_status"),
                    "qualification": experiment.get("qualification"),
                    "failed_gates": experiment.get("failed_gates") or [],
                    "metrics": experiment.get("metrics") or {},
                }
            )
    for task in quantcombine_store.tasks():
        for candidate in quantcombine_store.candidates(str(task["task_id"]), limit=10000):
            rows.append(
                {
                    "system": "QUANTCOMBINE",
                    "task_id": task["task_id"],
                    "candidate_id": candidate["id"],
                    "gate_status": candidate.get("gate_status"),
                    "qualification": candidate.get("qualification"),
                    "failed_gates": candidate.get("failed_gates") or [],
                    "metrics": candidate.get("metrics") or {},
                }
            )
    by_system: dict[str, dict[str, Any]] = defaultdict(_funnel_bucket)
    failure_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    root_cause_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    for row in rows:
        system_bucket = by_system[str(row["system"])]
        status = str(row.get("gate_status") or "UNKNOWN")
        failures = [str(value) for value in row.get("failed_gates") or []]
        status_counts[status] += 1
        system_bucket["total"] += 1
        system_bucket["status_counts"][status] += 1
        if status == "PASSED":
            system_bucket["passed"] += 1
            continue
        if not failures:
            failures = ["NO_EXPLICIT_GATE_FAILURE_RECORDED"]
        for failure in failures:
            category = _gate_category(failure)
            root_cause = _gate_root_cause(failure, category)
            failure_counts[failure] += 1
            category_counts[category] += 1
            root_cause_counts[root_cause] += 1
            system_bucket["failure_counts"][failure] += 1
            system_bucket["category_counts"][category] += 1
            system_bucket["root_cause_counts"][root_cause] += 1
    total = len(rows)
    passed = status_counts.get("PASSED", 0)
    rejected = total - passed
    root_causes = _top_counter(root_cause_counts, 20)
    return {
        "protocol": "AUTOALPHA_GATE_FUNNEL_DIAGNOSTICS_V2",
        "created_at": _now(),
        "total_candidates": total,
        "passed_candidates": passed,
        "rejected_candidates": rejected,
        "pass_rate": round(passed / total, 6) if total else 0.0,
        "status_counts": dict(status_counts),
        "by_system": {key: _freeze_bucket(value) for key, value in sorted(by_system.items())},
        "top_failed_gates": _top_counter(failure_counts, 25),
        "failure_categories": _top_counter(category_counts, 20),
        "root_causes": root_causes,
        "operator_actions": _gate_operator_actions(
            total,
            passed,
            category_counts,
            root_cause_counts,
        ),
        "diagnosis": _gate_diagnosis(total, passed, category_counts, root_cause_counts),
    }


def _canonical_mechanism(record: dict[str, Any]) -> str:
    proposal = record.get("proposal") or {}
    fields = expression_fields(proposal.get("expression"))
    mechanism = normalize_mechanism(
        proposal.get("canonical_mechanism"),
        default=classify_mechanism(
            fields=fields,
            family=str(record.get("family", "")),
            name=str(record.get("name", "")),
            hypothesis=str(proposal.get("hypothesis", "")),
        ),
    )
    return mechanism


def _mechanism_summary(
    record: dict[str, Any], mechanism: str, evidence: dict[str, Any]
) -> str:
    cluster = evidence.get("behavior_cluster_id") or "PENDING_CLUSTER"
    role = evidence.get("behavior_cluster_role") or "PENDING"
    nearest = evidence.get("behavior_nearest_factor_id") or "none"
    similarity = evidence.get("behavior_nearest_similarity")
    return (
        f"{record.get('name')} belongs to {mechanism}; behavior cluster={cluster}, "
        f"role={role}, nearest={nearest}, similarity={similarity}."
    )


def _related_factors(
    factor_id: str,
    evidence: dict[str, Any],
    cluster_members: dict[str, list[str]],
) -> list[dict[str, Any]]:
    related: list[dict[str, Any]] = []
    nearest = evidence.get("behavior_nearest_factor_id")
    if nearest and nearest != factor_id:
        related.append(
            {
                "factor_id": nearest,
                "relation": "NEAREST_BEHAVIOR_PEER",
                "confidence": _as_float(evidence.get("behavior_nearest_similarity")),
                "rationale": "Highest combined signal and residual-return behavior similarity.",
            }
        )
    cluster_id = evidence.get("behavior_cluster_id")
    for peer in cluster_members.get(str(cluster_id), []):
        if peer in (factor_id, nearest):
            continue
        related.append(
            {
                "factor_id": peer,
                "relation": "SAME_BEHAVIOR_CLUSTER",
                "confidence": 0.75,
                "rationale": f"Shares behavior cluster {cluster_id}.",
            }
        )
        if len(related) >= 10:
            break
    return related


def _cluster_members(behavior_factors: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)
    for factor_id, evidence in behavior_factors.items():
        cluster_id = evidence.get("behavior_cluster_id")
        if cluster_id:
            result[str(cluster_id)].append(str(factor_id))
    return result


def _homogeneity_metric_patch(
    review: dict[str, Any],
    evidence: dict[str, Any],
    related: list[dict[str, Any]],
) -> dict[str, Any]:
    cluster_size = _as_int(review.get("behavior_cluster_size"))
    nearest_similarity = _as_float(review.get("nearest_similarity"))
    cluster_id = review.get("behavior_cluster_id")
    redundancy = review.get("behavior_redundancy") or "PENDING"
    return {
        "homogeneity_protocol": review["protocol"],
        "canonical_mechanism": review["canonical_mechanism"],
        "expression_signature": review["expression_signature"],
        "parameter_family": review["parameter_family"],
        "homogeneity_cluster_id": cluster_id,
        "homogeneity_cluster_size": cluster_size,
        "homogeneity_nearest_factor_id": review.get("nearest_factor_id"),
        "homogeneity_nearest_similarity": nearest_similarity,
        "homogeneity_redundancy_label": redundancy,
        "homogeneity_crowded_cluster": cluster_size >= 8,
        "homogeneity_gate_passed": not (
            cluster_size >= 8
            and (
                nearest_similarity >= 0.82
                or str(redundancy).upper() in {"SUBSTITUTE", "NEAR_DUPLICATE"}
            )
        ),
        "behavior_cluster_id": cluster_id,
        "behavior_cluster_size": cluster_size,
        "behavior_cluster_role": review.get("behavior_cluster_role"),
        "behavior_redundancy": redundancy,
        "behavior_nearest_factor_id": review.get("nearest_factor_id"),
        "behavior_nearest_similarity": nearest_similarity,
        "behavior_signal_correlation": evidence.get("behavior_signal_correlation"),
        "behavior_return_correlation": evidence.get("behavior_return_correlation"),
        "homogeneity_related_factor_count": len(related),
    }


def _guard_repair_protocol_against_panel(
    protocol: dict[str, Any],
    data_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized = normalize_task_protocol(protocol)
    try:
        workspace = inspect_data_workspace(data_path)
        capacity = panel_validation_fold_capacity(normalized, Path(workspace.panel_path))
    except (FileNotFoundError, RuntimeError, TypeError, ValueError, OSError) as error:
        return normalized, {
            "protocol": "QUANT_REPAIR_PROTOCOL_CAPACITY_GUARD_V1",
            "status": "SKIPPED",
            "reason": f"{type(error).__name__}: {error}",
            "requested_minimum_folds": int(normalized["minimum_folds"]),
            "applied_minimum_folds": int(normalized["minimum_folds"]),
        }
    return _clamp_repair_protocol_to_capacity(normalized, capacity)


def _clamp_repair_protocol_to_capacity(
    protocol: dict[str, Any],
    capacity: dict[str, Any],
    *,
    fold_safety_buffer_days: int = 5,
) -> tuple[dict[str, Any], dict[str, Any]]:
    guarded = dict(protocol)
    requested = int(guarded["minimum_folds"])
    minimum_observations = int(capacity.get("minimum_observations_per_fold") or 60)
    observations_by_year = capacity.get("observations_by_year") or {}
    safe_years = [
        int(year)
        for year, count in observations_by_year.items()
        if int(count) >= minimum_observations + int(fold_safety_buffer_days)
    ]
    maximum_folds = int(capacity.get("maximum_folds") or 0)
    safe_maximum_folds = len(safe_years)
    effective_maximum = safe_maximum_folds or maximum_folds
    if effective_maximum > 0:
        guarded["minimum_folds"] = max(1, min(requested, effective_maximum))
    applied = int(guarded["minimum_folds"])
    return guarded, {
        "protocol": "QUANT_REPAIR_PROTOCOL_CAPACITY_GUARD_V1",
        "status": "ADJUSTED" if applied != requested else "UNCHANGED",
        "requested_minimum_folds": requested,
        "applied_minimum_folds": applied,
        "maximum_folds": maximum_folds,
        "safe_maximum_folds": safe_maximum_folds,
        "fold_safety_buffer_days": int(fold_safety_buffer_days),
        "evaluable_years": list(capacity.get("evaluable_years") or []),
        "safe_evaluable_years": safe_years,
        "observations_by_year": dict(observations_by_year),
    }


def _append_protocol_guard_note(notes: str, guard: dict[str, Any]) -> str:
    if guard.get("status") != "ADJUSTED":
        return notes
    return (
        f"{notes}\n"
        "protocol-capacity-guard:"
        f" minimum_folds {guard['requested_minimum_folds']} ->"
        f" {guard['applied_minimum_folds']}"
        f" using {guard['safe_maximum_folds']} execution-safe public folds."
    )


def _append_quant_autostart_note(notes: str, enabled: bool) -> str:
    if not enabled or "[quantcombine-autostart:AUTOALPHA_REPAIR_V1]" in notes:
        return notes
    return f"{notes}\n[quantcombine-autostart:AUTOALPHA_REPAIR_V1]"


def _expression_signature(expression: dict[str, Any] | None) -> str:
    return hashlib.sha256(
        json.dumps(_expression_shape(expression), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def _expression_shape(expression: dict[str, Any] | None) -> Any:
    if not isinstance(expression, dict):
        return None
    parameters = (
        expression.get("parameters") if isinstance(expression.get("parameters"), dict) else {}
    )
    retained = {
        key: value
        for key, value in parameters.items()
        if key in {"name", "field", "window", "period", "periods", "lookback"}
    }
    return [
        expression.get("operator"),
        retained,
        [_expression_shape(child) for child in expression.get("arguments", [])],
    ]


def _parameter_family(expression: dict[str, Any] | None) -> str:
    values: list[str] = []

    def visit(node: dict[str, Any] | None) -> None:
        if not isinstance(node, dict):
            return
        parameters = node.get("parameters") if isinstance(node.get("parameters"), dict) else {}
        for key in ("window", "period", "periods", "lookback"):
            if key in parameters:
                values.append(f"{key}={parameters[key]}")
        for child in node.get("arguments", []):
            visit(child)

    visit(expression)
    return "|".join(values) if values else "NO_EXPLICIT_LOOKBACK"


def _funnel_bucket() -> dict[str, Any]:
    return {
        "total": 0,
        "passed": 0,
        "status_counts": Counter(),
        "failure_counts": Counter(),
        "category_counts": Counter(),
        "root_cause_counts": Counter(),
    }


def _freeze_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    total = int(bucket["total"])
    passed = int(bucket["passed"])
    return {
        "total": total,
        "passed": passed,
        "rejected": total - passed,
        "pass_rate": round(passed / total, 6) if total else 0.0,
        "status_counts": dict(bucket["status_counts"]),
        "top_failed_gates": _top_counter(bucket["failure_counts"], 12),
        "failure_categories": _top_counter(bucket["category_counts"], 12),
        "root_causes": _top_counter(bucket["root_cause_counts"], 12),
    }


def _gate_category(failure: str) -> str:
    normalized = failure.lower()
    if "dsr" in normalized or "deflated" in normalized or "multiple" in normalized:
        return "MULTIPLE_TESTING_OR_DSR"
    if "fold" in normalized or "sample" in normalized or "coverage" in normalized:
        return "SAMPLE_OR_COVERAGE"
    if "drawdown" in normalized or "risk" in normalized:
        return "DRAWDOWN_OR_RISK"
    if "correlation" in normalized or "corr" in normalized:
        return "CORRELATION_OR_INDEPENDENCE"
    if "turnover" in normalized:
        return "TURNOVER"
    if "sharpe" in normalized or "annual" in normalized or "return" in normalized:
        return "RETURN_OR_SHARPE"
    if "capacity" in normalized or "liquidity" in normalized:
        return "CAPACITY_OR_LIQUIDITY"
    return "OTHER"


def _gate_root_cause(failure: str, category: str) -> str:
    normalized = failure.lower()
    if category == "SAMPLE_OR_COVERAGE":
        return "VALIDATION_DATA_OR_FOLD_CAPACITY"
    if category == "MULTIPLE_TESTING_OR_DSR":
        return "TRIAL_BUDGET_AND_OVERFITTING_PENALTY"
    if category == "CORRELATION_OR_INDEPENDENCE":
        return "FACTOR_INDEPENDENCE_INSUFFICIENT"
    if category == "DRAWDOWN_OR_RISK":
        return "RISK_CONSTRAINT_BREACH"
    if category in {"TURNOVER", "CAPACITY_OR_LIQUIDITY"}:
        return "TRADABILITY_OR_CAPACITY_CONSTRAINT"
    if category == "RETURN_OR_SHARPE":
        return "SEARCH_ALPHA_STRENGTH_INSUFFICIENT"
    if "marginal" in normalized or "incremental" in normalized:
        return "MARGINAL_CONTRIBUTION_INSUFFICIENT"
    if "no_explicit" in normalized:
        return "MISSING_GATE_TELEMETRY"
    return "UNCATEGORIZED_GATE_FAILURE"


def _gate_operator_actions(
    total: int,
    passed: int,
    categories: Counter[str],
    root_causes: Counter[str],
) -> list[dict[str, Any]]:
    if not total:
        return [
            {
                "priority": "P0",
                "action": "RUN_COMBINATION_EXPERIMENTS",
                "reason": "No combination candidates are available for gate diagnosis.",
            }
        ]
    if passed:
        return [
            {
                "priority": "P0",
                "action": "PROMOTE_GATE_PASSING_CANDIDATES_TO_STRATEGY_LIBRARY",
                "reason": "At least one combination candidate passed public gates.",
            }
        ]
    actions = []
    if root_causes.get("VALIDATION_DATA_OR_FOLD_CAPACITY"):
        actions.append(
            {
                "priority": "P0",
                "action": "REPAIR_WALK_FORWARD_CAPACITY_OR_COVERAGE",
                "reason": "Validation folds, sample size, or factor coverage dominate rejections.",
                "evidence_count": int(root_causes["VALIDATION_DATA_OR_FOLD_CAPACITY"]),
            }
        )
    if root_causes.get("TRIAL_BUDGET_AND_OVERFITTING_PENALTY"):
        actions.append(
            {
                "priority": "P0",
                "action": "REDUCE_EFFECTIVE_TRIAL_COUNT_AND_DEDUPLICATE_SEARCH_SPACE",
                "reason": "DSR or multiple-testing penalties indicate too many similar attempts.",
                "evidence_count": int(root_causes["TRIAL_BUDGET_AND_OVERFITTING_PENALTY"]),
            }
        )
    if root_causes.get("FACTOR_INDEPENDENCE_INSUFFICIENT"):
        actions.append(
            {
                "priority": "P1",
                "action": "FORCE_CROSS_MECHANISM_AND_CROSS_CLUSTER_SELECTION",
                "reason": "Correlation or independence gates reject too many combinations.",
                "evidence_count": int(root_causes["FACTOR_INDEPENDENCE_INSUFFICIENT"]),
            }
        )
    if root_causes.get("RISK_CONSTRAINT_BREACH"):
        actions.append(
            {
                "priority": "P1",
                "action": "SWITCH_OBJECTIVE_TO_DRAWDOWN_AND_TAIL_RISK_FIRST",
                "reason": "Risk constraints are binding before production admission.",
                "evidence_count": int(root_causes["RISK_CONSTRAINT_BREACH"]),
            }
        )
    if root_causes.get("SEARCH_ALPHA_STRENGTH_INSUFFICIENT") and not actions:
        actions.append(
            {
                "priority": "P1",
                "action": "EXPAND_OR_RESEED_FACTOR_MECHANISMS",
                "reason": "Return and Sharpe gates dominate after other constraints.",
                "evidence_count": int(root_causes["SEARCH_ALPHA_STRENGTH_INSUFFICIENT"]),
            }
        )
    if not actions:
        top_category = categories.most_common(1)[0][0] if categories else "OTHER"
        actions.append(
            {
                "priority": "P2",
                "action": "INSPECT_UNCATEGORIZED_GATE_FAILURES",
                "reason": f"Dominant failure category is {top_category}; telemetry is incomplete.",
            }
        )
    return actions[:5]


def _gate_diagnosis(
    total: int,
    passed: int,
    categories: Counter[str],
    root_causes: Counter[str],
) -> list[dict[str, Any]]:
    if not total:
        return [{"class": "NO_CANDIDATES", "message": "No combination candidates found."}]
    messages: list[dict[str, Any]] = []
    if passed == 0:
        messages.append(
            {
                "class": "NO_GATE_PASSING_COMBINATIONS",
                "message": "Search produced no production-gate passing combinations.",
            }
        )
    for category, count in categories.most_common(5):
        messages.append(
            {
                "class": category,
                "count": int(count),
                "share_of_rejections": round(count / max(1, total - passed), 6),
            }
        )
    for root_cause, count in root_causes.most_common(5):
        messages.append(
            {
                "class": root_cause,
                "count": int(count),
                "diagnostic_type": "ROOT_CAUSE",
                "share_of_rejections": round(count / max(1, total - passed), 6),
            }
        )
    return messages


def _top_counter(counter: Counter[str], limit: int) -> list[dict[str, Any]]:
    return [{"key": key, "count": int(count)} for key, count in counter.most_common(limit)]


def _as_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _as_int(value: Any) -> int:
    return int(_as_float(value))


def _now() -> str:
    return datetime.now(UTC).isoformat()
