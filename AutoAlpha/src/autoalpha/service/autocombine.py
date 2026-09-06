from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from autoalpha.config import ResearchConfig
from autoalpha.service.autocombine_intelligence import (
    enrich_factor_record,
    load_return_artifact,
    mechanism_independence_metrics,
    public_metric_bands,
    return_independence,
    write_return_artifact,
)
from autoalpha.service.autocombine_store import AutoCombineStore
from autoalpha.service.blind_evaluator import BlindEvaluationBoundary
from autoalpha.service.evaluator import PriceVolumeEvaluator
from autoalpha.service.multifactor import factor_from_pool_record
from autoalpha.service.openai_client import CompatibleChatClient, ModelInvocationError
from autoalpha.service.research_protocol import task_research_config
from autoalpha.service.store import ServiceStore
from autoalpha.service.worker import SecretVault

DEFAULT_CONSTRUCTION = {
    "min_factors": 2,
    "max_factors": 5,
    "minimum_weight": 0.05,
    "maximum_weight": 0.50,
    "weight_step": 0.05,
    "candidate_pool_limit": 30,
    "allow_negative_weights": False,
    "maximum_same_family": 2,
    "maximum_same_semantic_cluster": 1,
    "maximum_same_parameter_family": 1,
}
DEFAULT_OBJECTIVE = {
    "profile": "ROBUST_ACTIVE_LONG_ONLY",
    "preset_version": 1,
    "minimum_coverage": 0.80,
    "minimum_positive_fold_fraction": 0.50,
    "minimum_worst_fold_sharpe": -0.50,
    "maximum_drawdown": 0.30,
    "maximum_annual_turnover": 40.0,
    "maximum_factor_correlation": 0.75,
    "minimum_effective_factor_bets": 1.35,
    "minimum_effective_mechanisms": 1.40,
    "maximum_mechanism_weight": 0.75,
    "maximum_strategy_active_correlation": 0.75,
    "minimum_marginal_positive_fraction": 0.60,
    "minimum_deflated_sharpe_probability": 0.50,
    "maximum_duplicate_semantic_factors": 0,
    "minimum_cost_stress_ir": 0.0,
    "minimum_simple_annual_return": 0.0,
}
OBJECTIVE_PRESETS: dict[str, dict[str, Any]] = {
    "ROBUST_ACTIVE_LONG_ONLY": {
        **DEFAULT_OBJECTIVE,
        "label": "稳健均衡",
        "description": "优先样本外稳定性，兼顾主动 IR、收益、回撤、换手与相关性。",
    },
    "DRAWDOWN_FIRST": {
        **DEFAULT_OBJECTIVE,
        "profile": "DRAWDOWN_FIRST",
        "label": "降低回撤优先",
        "description": "先压低样本外最大回撤，再比较最差折、夏普与收益。",
        "maximum_drawdown": 0.18,
        "minimum_simple_annual_return": 0.03,
        "minimum_positive_fold_fraction": 0.60,
    },
    "PORTFOLIO_SHARPE_FIRST": {
        **DEFAULT_OBJECTIVE,
        "profile": "PORTFOLIO_SHARPE_FIRST",
        "label": "组合夏普优先",
        "description": "最大化纯多组合样本外夏普，同时保留回撤与最差折硬门禁。",
        "maximum_drawdown": 0.25,
        "minimum_positive_fold_fraction": 0.60,
    },
    "ABSOLUTE_LONG_ONLY": {
        **DEFAULT_OBJECTIVE,
        "profile": "ABSOLUTE_LONG_ONLY",
        "label": "年化收益优先",
        "description": "在风险预算内优先提高成本后纯多年化收益。",
        "minimum_simple_annual_return": 0.05,
        "maximum_drawdown": 0.25,
    },
    "LOW_TURNOVER": {
        **DEFAULT_OBJECTIVE,
        "profile": "LOW_TURNOVER",
        "label": "低换手容量优先",
        "description": "优先低换手、成本压力与容量友好，适合较大资金规模。",
        "maximum_annual_turnover": 15.0,
        "minimum_simple_annual_return": 0.02,
    },
    "DIVERSIFICATION_FIRST": {
        **DEFAULT_OBJECTIVE,
        "profile": "DIVERSIFICATION_FIRST",
        "label": "分散化优先",
        "description": "优先降低因子相关性和单一机制集中度，再比较稳健收益。",
        "maximum_factor_correlation": 0.60,
        "minimum_simple_annual_return": 0.02,
    },
}
DEFAULT_BUDGET = {
    "maximum_experiments": 60,
    "maximum_llm_proposals": 20,
    "maximum_runtime_minutes": 180,
    "maximum_holdout_submissions": 1,
    "weight_evaluations_per_subset": 12,
    "maximum_subset_revisits": 2,
    "maximum_same_direction_attempts": 3,
    "iteration_interval_seconds": 0.5,
}


def canonical_hash(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode()).hexdigest()


def build_factor_snapshot(
    store: ServiceStore,
    scope: dict[str, Any],
    construction: dict[str, Any],
) -> list[dict[str, Any]]:
    mode = str(scope.get("mode", "SMART")).upper()
    explicit = {str(value) for value in scope.get("factor_ids", [])}
    required = {str(value) for value in scope.get("required_factor_ids", [])}
    if mode == "MANUAL" and not explicit:
        raise ValueError("手动选择模式至少需要一个因子")
    if mode == "HYBRID" and not required:
        raise ValueError("混合模式至少需要一个必选因子")
    excluded = {str(value) for value in scope.get("excluded_factor_ids", [])}
    sources = {str(value) for value in scope.get("source_task_ids", [])}
    statuses = {str(value) for value in scope.get("statuses", ["ELIGIBLE", "SCREENED_OUT"])}
    families = {str(value).casefold() for value in scope.get("families", [])}
    contaminated = store.contaminated_factor_ids()
    knowledge_lookup = {
        str(item["factor_id"]): item for item in store.factor_knowledge_catalog(limit=5000)
    }
    records: list[dict[str, Any]] = []
    available_ids: set[str] = set()
    for record in store.factor_pool(limit=5000):
        factor_id = str(record["factor_id"])
        available_ids.add(factor_id)
        if factor_id in excluded:
            continue
        selected_directly = factor_id in explicit or factor_id in required
        if mode == "MANUAL" and factor_id not in explicit:
            continue
        if (
            mode != "MANUAL"
            and sources
            and not selected_directly
            and str(record.get("source_task_id")) not in sources
        ):
            continue
        if (
            mode != "MANUAL"
            and statuses
            and not selected_directly
            and str(record.get("status")) not in statuses
        ):
            continue
        if (
            mode != "MANUAL"
            and families
            and not selected_directly
            and str(record.get("family", "")).casefold() not in families
        ):
            continue
        proposal = record.get("proposal") or {}
        if not isinstance(proposal.get("expression"), dict):
            continue
        try:
            factor_from_pool_record(record)
        except (KeyError, TypeError, ValueError):
            continue
        knowledge = knowledge_lookup.get(factor_id) or {}
        review = knowledge.get("review") or {}
        metrics = record.get("metrics") or {}
        behavior_cluster_id = (
            review.get("behavior_cluster_id")
            or metrics.get("homogeneity_cluster_id")
            or metrics.get("online_behavior_cluster_id")
        )
        canonical_mechanism = knowledge.get("canonical_mechanism")
        enriched_proposal = {
            **proposal,
            **({"canonical_mechanism": canonical_mechanism} if canonical_mechanism else {}),
        }
        enriched = enrich_factor_record(
            {
                "factor_id": factor_id,
                "name": record["name"],
                "family": record["family"],
                "status": record["status"],
                "source_task_id": record.get("source_task_id"),
                "source_iteration": record.get("source_iteration"),
                "proposal": enriched_proposal,
                "metrics": metrics,
                "prefilter_score": _prefilter_score(metrics),
                "required": factor_id in required,
                "holdout_contaminated": factor_id in contaminated,
                "behavior_cluster_id": behavior_cluster_id,
                "parameter_family": review.get("parameter_family"),
                "expression_signature": review.get("expression_signature"),
            }
        )
        enriched["parameter_family"] = enriched.get("parameter_family") or _parameter_family(
            proposal.get("expression")
        )
        records.append(enriched)
    requested = explicit | required
    unknown = requested - available_ids
    if unknown:
        raise ValueError(f"因子不存在：{', '.join(sorted(unknown))}")
    missing = requested - {item["factor_id"] for item in records}
    if missing:
        raise ValueError(f"因子表达式无效，无法用于组合研究：{', '.join(sorted(missing))}")
    records.sort(key=lambda item: item["prefilter_score"], reverse=True)
    limit = int(construction.get("candidate_pool_limit", 30))
    if mode == "MANUAL":
        return records
    required_records = [item for item in records if item["required"]]
    optional_records = [item for item in records if not item["required"]]
    effective_limit = max(len(required_records), max(5, min(limit, 100)))
    return [*required_records, *optional_records[: max(0, effective_limit - len(required_records))]]


def _metric(metrics: dict[str, Any], *names: str, default: float = 0.0) -> float:
    for name in names:
        value = metrics.get(name)
        if isinstance(value, int | float) and math.isfinite(float(value)):
            return float(value)
    return default


def _prefilter_score(metrics: dict[str, Any]) -> float:
    sharpe = _metric(
        metrics,
        "recent_long_only_sharpe_ratio",
        "long_only_sharpe_ratio",
        "sharpe_ratio",
    )
    annual = _metric(
        metrics,
        "recent_long_only_simple_annual_return",
        "long_only_simple_annual_return",
        "simple_annual_return",
    )
    worst = _metric(
        metrics,
        "recent_long_only_walk_forward_worst_sharpe",
        "long_only_walk_forward_worst_sharpe",
        "walk_forward_worst_sharpe",
        default=-1.0,
    )
    drawdown = _metric(
        metrics,
        "recent_long_only_max_drawdown",
        "long_only_max_drawdown",
        "max_drawdown",
        default=-1.0,
    )
    turnover = _metric(
        metrics,
        "recent_long_only_annual_turnover",
        "long_only_annual_turnover",
        "annual_turnover",
        default=100.0,
    )
    return sharpe + 2.0 * annual + 0.45 * worst + 0.5 * drawdown - 0.005 * turnover


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


def merge_defaults(value: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    return {**defaults, **value}


@dataclass(frozen=True)
class CombineProposal:
    action: str
    factor_ids: tuple[str, ...]
    rationale: str
    hypothesis: str
    source: str
    prompt_hash: str | None = None
    response_hash: str | None = None


class AutoCombineWorker:
    def __init__(
        self,
        task_id: str,
        store: ServiceStore,
        combine_store: AutoCombineStore,
        vault: SecretVault,
        *,
        config_path: Path,
    ) -> None:
        self.task_id = task_id
        self.store = store
        self.combine_store = combine_store
        self.vault = vault
        self.config_path = config_path
        self.artifact_root = self.store.path.parent / "artifacts" / "autocombine"
        self._task: asyncio.Task[None] | None = None
        self._evaluator: PriceVolumeEvaluator | None = None

    @property
    def alive(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> dict[str, Any]:
        task = self._require_task()
        if self.alive:
            return task
        if task["status"] == "COMPLETED":
            raise RuntimeError("已完成任务请复制为新任务后继续搜索")
        self.combine_store.update_task(
            self.task_id,
            status="RUNNING",
            phase="PREFLIGHT" if task["iteration"] == 0 else "SEARCHING",
            stop_requested=0,
            last_error=None,
        )
        self.combine_store.event(
            self.task_id,
            "action",
            "COMBINE_STARTED",
            "组合研究已启动",
            "冻结因子快照将按实验预算持续搜索，直到完成或收到停止请求。",
        )
        self._task = asyncio.create_task(self._loop(), name=f"autocombine-{self.task_id}")
        return self._require_task()

    async def stop(self) -> dict[str, Any]:
        self._require_task()
        if not self.alive:
            return self.combine_store.update_task(
                self.task_id, status="PAUSED", phase="PAUSED", stop_requested=0
            )
        self.combine_store.update_task(
            self.task_id, status="STOPPING", phase="CHECKPOINT", stop_requested=1
        )
        self.combine_store.event(
            self.task_id,
            "action",
            "COMBINE_STOP_REQUESTED",
            "停止请求已登记",
            "当前组合实验完成并落库后暂停。",
            level="WARN",
        )
        return self._require_task()

    async def shutdown(self) -> None:
        if self.alive:
            self.combine_store.update_task(self.task_id, stop_requested=1)
            await self._task

    async def _loop(self) -> None:
        started = time.monotonic()
        try:
            task = self._require_task()
            if len(task["factor_snapshot"]) < int(task["construction"]["min_factors"]):
                raise RuntimeError("冻结因子范围不足以满足最小因子数")
            evaluator = await asyncio.to_thread(self._build_evaluator, task)
            self.combine_store.update_task(self.task_id, phase="SCREENING")
            while True:
                task = self._require_task()
                budget = task["budget"]
                elapsed_minutes = (time.monotonic() - started) / 60
                if task["stop_requested"]:
                    self.combine_store.update_task(
                        self.task_id, status="PAUSED", phase="PAUSED", stop_requested=0
                    )
                    return
                if int(task["iteration"]) >= int(budget["maximum_experiments"]):
                    await self._complete_task("实验预算已用完")
                    return
                if elapsed_minutes >= float(budget["maximum_runtime_minutes"]):
                    await self._complete_task("运行时长预算已用完")
                    return
                iteration = int(task["iteration"]) + 1
                self.combine_store.update_task(self.task_id, phase="SEARCHING", iteration=iteration)
                proposal = await self._propose(task, iteration)
                if proposal is None:
                    await self._complete_task("冻结搜索空间内已无新候选")
                    return
                try:
                    evaluated = await asyncio.to_thread(
                        self._evaluate_proposal, evaluator, task, proposal, iteration
                    )
                except _CandidateEvaluationRejected as error:
                    evaluated = _rejected_experiment_record(task, proposal, iteration, error)
                experiment = self.combine_store.record_experiment(self.task_id, evaluated)
                if experiment["qualification"] != "CANDIDATE_EVALUATION_REJECTED":
                    self._update_best(task, experiment)
                    self._write_memory(experiment)
                self.combine_store.event(
                    self.task_id,
                    "research",
                    "COMBINE_EXPERIMENT_COMPLETED",
                    f"实验 #{iteration} · {experiment['gate_status']}",
                    _experiment_summary(experiment),
                    level="INFO" if experiment["gate_status"] == "PASSED" else "WARN",
                    payload={
                        "experiment_id": experiment["id"],
                        "factor_ids": experiment["factor_ids"],
                        "weights": experiment["weights"],
                        "score": experiment["score"],
                        "gate_distance": experiment.get("gate_distance"),
                        "qualification": experiment.get("qualification"),
                        "return_artifact_hash": experiment.get("return_artifact_hash"),
                    },
                )
                await asyncio.sleep(float(budget["iteration_interval_seconds"]))
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self.combine_store.update_task(
                self.task_id,
                status="PAUSED_FAILURE",
                phase="FAILED",
                stop_requested=0,
                last_error=f"{type(error).__name__}: {error}",
            )
            self.combine_store.event(
                self.task_id,
                "audit",
                "COMBINE_FAILED",
                "组合研究异常暂停",
                f"{type(error).__name__}: {error}",
                level="ERROR",
            )

    async def _complete_task(self, reason: str) -> None:
        task = self._require_task()
        qualified = (
            self.combine_store.experiment(int(task["qualified_experiment_id"]))
            if task.get("qualified_experiment_id")
            else None
        )
        status = "EXHAUSTED"
        if qualified and qualified["gate_status"] == "PASSED":
            self.combine_store.update_task(self.task_id, phase="BLIND_REVIEW")
            records = {item["factor_id"]: item for item in task["factor_snapshot"]}
            factors = [factor_from_pool_record(records[value]) for value in qualified["factor_ids"]]
            evaluator = self._evaluator or await asyncio.to_thread(self._build_evaluator, task)
            verdict = await asyncio.to_thread(
                BlindEvaluationBoundary(evaluator.panel_path, evaluator.config).evaluate_holdout,
                factors,
                tuple(float(value) for value in qualified["weights"]),
            )
            self.combine_store.update_task(
                self.task_id,
                blind_verdict=verdict.verdict,
                blind_evidence_hash=verdict.evidence_hash,
                production_candidate_experiment_id=(qualified["id"] if verdict.passed else None),
                qualification_status=(
                    "PRODUCTION_CANDIDATE" if verdict.passed else "BLIND_REJECTED"
                ),
            )
            self.combine_store.event(
                self.task_id,
                "audit",
                "COMBINE_BLIND_VERDICT",
                "隔离盲测已完成",
                verdict.verdict,
                level="INFO" if verdict.passed else "WARN",
                payload={"verdict": verdict.verdict, "evidence_hash": verdict.evidence_hash},
            )
            status = "COMPLETED" if verdict.passed else "BLIND_REJECTED"
        self.combine_store.update_task(
            self.task_id, status=status, phase="DELIVERY", stop_requested=0
        )
        await asyncio.to_thread(refresh_task_strategy_clusters, self.combine_store)
        self.combine_store.event(
            self.task_id,
            "delivery",
            "COMBINE_COMPLETED",
            "组合搜索已封存",
            reason,
            payload={"best_experiment_id": task.get("best_experiment_id")},
        )

    def _build_evaluator(self, task: dict[str, Any]) -> PriceVolumeEvaluator:
        base = ResearchConfig.from_toml(self.config_path)
        config = task_research_config(base, task["protocol"], task_id=self.task_id)
        evaluator = PriceVolumeEvaluator(Path(task["data_path"]), config=config)
        evaluator.set_trial_count(max(1, int(task["budget"]["maximum_experiments"])))
        self._evaluator = evaluator
        factors = [factor_from_pool_record(record) for record in task["factor_snapshot"]]
        evaluator.prime_factor_signals(factors)
        return evaluator

    async def _propose(self, task: dict[str, Any], iteration: int) -> CombineProposal | None:
        experiments = self.combine_store.experiments(self.task_id, limit=1000)
        proposal = self._deterministic_proposal(task, experiments, iteration)
        llm_count = sum(item["proposal_source"] == "LLM" for item in experiments)
        should_use_llm = iteration > 3 and iteration % 2 == 0
        if (
            should_use_llm
            and self.vault.configured()
            and llm_count < int(task["budget"]["maximum_llm_proposals"])
        ):
            try:
                llm = await self._llm_proposal(task, experiments, iteration)
                if llm and not self._proposal_exists(llm, task):
                    return llm
            except (
                ModelInvocationError,
                RuntimeError,
                TypeError,
                ValueError,
                _CandidateEvaluationRejected,
            ) as error:
                self.combine_store.event(
                    self.task_id,
                    "audit",
                    "COMBINE_LLM_FALLBACK",
                    "LLM 提议不可用，已使用确定性搜索",
                    f"{type(error).__name__}: {error}",
                    level="WARN",
                )
        return proposal

    def _deterministic_proposal(
        self,
        task: dict[str, Any],
        experiments: list[dict[str, Any]],
        iteration: int,
    ) -> CombineProposal | None:
        pool = task["factor_snapshot"]
        construction = task["construction"]
        min_count = int(construction["min_factors"])
        max_count = int(construction["max_factors"])
        ranked = [item["factor_id"] for item in pool]
        required = {str(value) for value in task["scope"].get("required_factor_ids", [])}
        mechanism = {item["factor_id"]: str(item["mechanism"]) for item in pool}
        search_cluster = {item["factor_id"]: str(item["search_cluster_id"]) for item in pool}
        parameter_family = {
            item["factor_id"]: str(item.get("parameter_family") or "NO_EXPLICIT_LOOKBACK")
            for item in pool
        }
        passing = [item for item in experiments if item["gate_status"] == "PASSED"]
        incumbent = max(
            passing or experiments,
            key=lambda item: item.get("score") or -999,
            default=None,
        )

        candidates: list[tuple[str, ...]] = []
        if not experiments:
            candidates.extend(
                self._diversified_seeds(ranked, mechanism, max(min_count, len(required)), required)
            )
        else:
            current = list(incumbent["factor_ids"]) if incumbent else ranked[:min_count]
            unused = [factor_id for factor_id in ranked if factor_id not in current]
            if len(current) < max_count:
                for factor_id in unused:
                    candidates.append(tuple([*current, factor_id]))
            for replacement in unused:
                for index in range(len(current) - 1, -1, -1):
                    if current[index] in required:
                        continue
                    candidate = current.copy()
                    candidate[index] = replacement
                    candidates.append(tuple(candidate))
            if len(current) > min_count:
                for index in range(len(current)):
                    if current[index] in required:
                        continue
                    candidates.append(tuple(value for i, value in enumerate(current) if i != index))
            candidates.extend(
                self._diversified_seeds(ranked, mechanism, max(min_count, len(required)), required)
            )

        recent_limit = int(task["budget"].get("maximum_same_direction_attempts", 3))
        recent_directions = [
            self._dominant_mechanism(item["factor_ids"], mechanism)
            for item in experiments[:recent_limit]
        ]
        for factor_ids in candidates:
            factor_ids = tuple(dict.fromkeys(factor_ids))
            if not min_count <= len(factor_ids) <= max_count:
                continue
            if not required <= set(factor_ids):
                continue
            if self._family_limit_violation(
                factor_ids,
                mechanism,
                search_cluster,
                construction,
                parameter_family=parameter_family,
            ):
                continue
            dominant = self._dominant_mechanism(factor_ids, mechanism)
            if (
                recent_directions
                and len(recent_directions) >= recent_limit
                and all(value == dominant for value in recent_directions)
            ):
                continue
            proposal = CombineProposal(
                action="SEED" if not experiments else "ADD_REPLACE",
                factor_ids=factor_ids,
                rationale="确定性分散搜索：优先保留高质量因子并限制同机制集中。",
                hypothesis="互补机制有望改善纯多样本外收益、回撤与成本后的稳定性。",
                source="DETERMINISTIC",
            )
            if not self._proposal_exists(proposal, task):
                return proposal
        return None

    @staticmethod
    def _diversified_seeds(
        ranked: list[str], family: dict[str, str], count: int, required: set[str] | None = None
    ) -> list[tuple[str, ...]]:
        required = required or set()
        seeds: list[tuple[str, ...]] = []
        for offset in range(min(12, len(ranked))):
            chosen = [factor_id for factor_id in ranked if factor_id in required]
            seen = {family[factor_id] for factor_id in chosen}
            for factor_id in [*ranked[offset:], *ranked[:offset]]:
                if factor_id in chosen:
                    continue
                if family[factor_id] in seen and len(seen) < count:
                    continue
                chosen.append(factor_id)
                seen.add(family[factor_id])
                if len(chosen) == count:
                    seeds.append(tuple(chosen))
                    break
        return seeds

    @staticmethod
    def _family_limit_ok(
        factor_ids: tuple[str, ...],
        mechanism: dict[str, str],
        search_cluster: dict[str, str],
        construction: dict[str, Any],
        *,
        parameter_family: dict[str, str] | None = None,
    ) -> bool:
        return (
            AutoCombineWorker._family_limit_violation(
                factor_ids,
                mechanism,
                search_cluster,
                construction,
                parameter_family=parameter_family,
            )
            is None
        )

    @staticmethod
    def _family_limit_violation(
        factor_ids: tuple[str, ...],
        mechanism: dict[str, str],
        search_cluster: dict[str, str],
        construction: dict[str, Any],
        *,
        parameter_family: dict[str, str] | None = None,
    ) -> dict[str, Any] | None:
        maximum = int(construction.get("maximum_same_family", 2))
        maximum_cluster = int(construction.get("maximum_same_semantic_cluster", 1))
        maximum_parameter_family = int(construction.get("maximum_same_parameter_family", 1))
        dimensions: list[tuple[str, dict[str, str], int]] = [
            ("mechanism", mechanism, maximum),
            ("search_cluster", search_cluster, maximum_cluster),
        ]
        if parameter_family is None:
            parameter_family = {}
        filtered_parameter_family = {
            factor_id: label
            for factor_id, label in parameter_family.items()
            if label != "NO_EXPLICIT_LOOKBACK"
        }
        dimensions.append(("parameter_family", filtered_parameter_family, maximum_parameter_family))
        for dimension, labels, maximum_allowed in dimensions:
            label_counts = Counter(
                labels.get(factor_id, factor_id)
                for factor_id in factor_ids
                if factor_id in labels
            )
            crowded = {
                label: count
                for label, count in label_counts.items()
                if count > max(0, maximum_allowed)
            }
            if crowded:
                return {
                    "reason": "HOMOGENEITY_DIVERSIFICATION_CONSTRAINT",
                    "dimension": dimension,
                    "maximum_allowed": maximum_allowed,
                    "crowded_labels": crowded,
                    "factor_ids": list(factor_ids),
                }
        return None

    @staticmethod
    def _dominant_mechanism(
        factor_ids: list[str] | tuple[str, ...], mechanism: dict[str, str]
    ) -> str:
        counts: dict[str, int] = {}
        for factor_id in factor_ids:
            value = mechanism[factor_id]
            counts[value] = counts.get(value, 0) + 1
        return max(counts, key=lambda value: (counts[value], value))

    async def _llm_proposal(
        self,
        task: dict[str, Any],
        experiments: list[dict[str, Any]],
        iteration: int,
    ) -> CombineProposal | None:
        settings = self.store.settings()
        client = CompatibleChatClient(
            base_url=settings["base_url"],
            api_key=self.vault.get(),
            model=settings["model"],
            timeout=30,
            temperature=min(float(settings.get("temperature", "0.4")), 0.6),
            max_tokens=3000,
            transport_retries=0,
        )
        compact_pool = [
            {
                "factor_id": item["factor_id"],
                "name": item["name"],
                "reported_family": item["family"],
                "mechanism": item["mechanism"],
                "semantic_cluster_id": item["semantic_cluster_id"],
                "behavior_cluster_id": item.get("behavior_cluster_id"),
                "search_cluster_id": item["search_cluster_id"],
                "expression_summary": item["expression_summary"],
                "expression_fields": item["expression_fields"],
                "expression_windows": item["expression_windows"],
                "hypothesis": item["proposal"].get("hypothesis", ""),
                "prefilter_score": round(item["prefilter_score"], 4),
                "holdout_contaminated": bool(item.get("holdout_contaminated")),
            }
            for item in task["factor_snapshot"]
        ]
        history = [
            {
                "iteration": item["iteration"],
                "factor_ids": item["factor_ids"],
                "weights": item["weights"],
                "gate_status": item["gate_status"],
                "failed_gates": item["failed_gates"],
                "public_metric_bands": public_metric_bands(item.get("metrics") or {}),
                "mechanism_weights": (item.get("metrics") or {}).get(
                    "portfolio_mechanism_weights", {}
                ),
                "redundant_factor_ids": (item.get("metrics") or {}).get(
                    "portfolio_redundant_factor_ids", []
                ),
            }
            for item in experiments[:12]
        ]
        analysis = await client.analyze(
            role="AUTOCOMBINE_PORTFOLIO_ARCHITECT",
            system_prompt=(
                "You are the portfolio architecture role in an institutional US-equity research "
                "system. Select only existing factor_ids. Propose one static long-only composite "
                "signal subset. Do not invent factors, inspect hidden periods, flip signs, or tune "
                "decimal weights. Treat differently named factors with the same mechanism, fields, "
                "window and behavior/search cluster as redundant. Favor genuinely independent "
                "mechanisms and robust public out-of-sample evidence. Exact public metrics are "
                "intentionally hidden to reduce adaptive validation overfit. Return JSON with "
                "action, factor_ids, rationale, hypothesis, risk."
            ),
            context={
                "iteration": iteration,
                "constraints": task["construction"],
                "objective": task["objective"],
                "factor_snapshot": compact_pool,
                "recent_public_experiments": history,
                "instruction": "Use between min_factors and max_factors unique factor ids.",
                "required_factor_ids": task["scope"].get("required_factor_ids", []),
            },
            required_keys={"action", "factor_ids", "rationale", "hypothesis", "risk"},
        )
        factor_ids = tuple(str(value) for value in analysis.artifact["factor_ids"])
        valid_ids = {item["factor_id"] for item in task["factor_snapshot"]}
        minimum = int(task["construction"]["min_factors"])
        maximum = int(task["construction"]["max_factors"])
        if len(set(factor_ids)) != len(factor_ids) or not minimum <= len(factor_ids) <= maximum:
            raise ValueError("LLM combination violates factor-count constraints")
        if not set(factor_ids) <= valid_ids:
            raise ValueError("LLM combination references factors outside the frozen snapshot")
        required = {str(value) for value in task["scope"].get("required_factor_ids", [])}
        if not required <= set(factor_ids):
            raise ValueError("LLM combination omitted required factors")
        records = {item["factor_id"]: item for item in task["factor_snapshot"]}
        mechanism = {factor_id: str(item["mechanism"]) for factor_id, item in records.items()}
        clusters = {
            factor_id: str(item["search_cluster_id"]) for factor_id, item in records.items()
        }
        parameter_family = {
            factor_id: str(item.get("parameter_family") or "NO_EXPLICIT_LOOKBACK")
            for factor_id, item in records.items()
        }
        violation = self._family_limit_violation(
            factor_ids,
            mechanism,
            clusters,
            task["construction"],
            parameter_family=parameter_family,
        )
        if violation is not None:
            raise _CandidateEvaluationRejected(
                f"HOMOGENEITY_DIVERSIFICATION_CONSTRAINT:{violation['dimension']}"
            )
        return CombineProposal(
            action=str(analysis.artifact["action"]).upper()[:32],
            factor_ids=factor_ids,
            rationale=str(analysis.artifact["rationale"])[:3000],
            hypothesis=str(analysis.artifact["hypothesis"])[:3000],
            source="LLM",
            prompt_hash=analysis.prompt_hash,
            response_hash=analysis.response_hash,
        )

    def _proposal_exists(self, proposal: CombineProposal, task: dict[str, Any]) -> bool:
        proposed = frozenset(proposal.factor_ids)
        count = sum(
            frozenset(experiment["factor_ids"]) == proposed
            for experiment in self.combine_store.experiments(self.task_id, limit=5000)
        )
        return count >= int(task["budget"].get("maximum_subset_revisits", 2))

    def _evaluate_proposal(
        self,
        evaluator: PriceVolumeEvaluator,
        task: dict[str, Any],
        proposal: CombineProposal,
        iteration: int,
    ) -> dict[str, Any]:
        started = time.monotonic()
        records = {item["factor_id"]: item for item in task["factor_snapshot"]}
        mechanism = {factor_id: str(item["mechanism"]) for factor_id, item in records.items()}
        search_cluster = {
            factor_id: str(item["search_cluster_id"]) for factor_id, item in records.items()
        }
        parameter_family = {
            factor_id: str(item.get("parameter_family") or "NO_EXPLICIT_LOOKBACK")
            for factor_id, item in records.items()
        }
        homogeneity_violation = self._family_limit_violation(
            proposal.factor_ids,
            mechanism,
            search_cluster,
            task["construction"],
            parameter_family=parameter_family,
        )
        if homogeneity_violation is not None:
            self.combine_store.event(
                self.task_id,
                "audit",
                "COMBINE_HOMOGENEITY_CANDIDATE_REJECTED",
                "同质候选已跳过",
                (
                    f"{homogeneity_violation['dimension']} 超过上限 "
                    f"{homogeneity_violation['maximum_allowed']}"
                ),
                level="WARN",
                payload=homogeneity_violation,
            )
            raise _CandidateEvaluationRejected(
                f"HOMOGENEITY_DIVERSIFICATION_CONSTRAINT:{homogeneity_violation['dimension']}"
            )
        factors = [factor_from_pool_record(records[factor_id]) for factor_id in proposal.factor_ids]
        best: (
            tuple[
                tuple[float, ...],
                tuple[float, ...],
                Any,
                dict[str, Any],
                list[str],
                float,
            ]
            | None
        ) = None
        evaluated_count = 0
        rejected_reasons: list[str] = []
        for weights in self._weight_candidates(proposal.factor_ids, task, iteration):
            candidate_hash = _candidate_hash(proposal.factor_ids, weights)
            if self.combine_store.candidate_exists(self.task_id, candidate_hash):
                continue
            try:
                evaluation = evaluator.evaluate_portfolio(factors, weights=weights)
            except (ValueError, FloatingPointError, ArithmeticError) as error:
                reason = _recoverable_evaluation_failure_reason(error)
                if reason is None:
                    raise
                rejected_reasons.append(reason)
                self.combine_store.event(
                    self.task_id,
                    "audit",
                    "COMBINE_WEIGHT_CANDIDATE_REJECTED",
                    "权重候选评估失败，已作为候选级问题跳过",
                    f"{reason}: {error}",
                    level="WARN",
                    payload={
                        "factor_ids": list(proposal.factor_ids),
                        "weights": list(weights),
                        "failure_class": reason,
                    },
                )
                continue
            evaluated_count += 1
            metrics = dict(evaluation.metrics)
            metrics.update(
                mechanism_independence_metrics(records, list(proposal.factor_ids), list(weights))
            )
            active_returns = evaluation.net_returns - evaluator._market_benchmark_returns(
                evaluation.net_returns.index
            )
            metrics.update(self._strategy_independence_metrics(active_returns))
            failures = _gate_failures(metrics, task["objective"])
            distance = _gate_distance(metrics, task["objective"])
            score = _portfolio_score(metrics, failures, task["objective"])
            rank = (
                float(not failures),
                -float(len(failures)),
                -distance,
                score,
            )
            if best is None or rank > best[0]:
                best = (rank, weights, evaluation, metrics, failures, score)
        if best is None:
            if rejected_reasons:
                raise _CandidateEvaluationRejected(
                    f"All weight candidates rejected: {sorted(set(rejected_reasons))}"
                )
            raise RuntimeError("该组合的全部权重候选均已评估")
        _, weights, evaluation, metrics, failures, score = best
        if not failures:
            metrics.update(
                self._leave_one_out_metrics(
                    evaluator,
                    records,
                    list(proposal.factor_ids),
                    list(weights),
                    metrics,
                    task,
                )
            )
            failures = _gate_failures(metrics, task["objective"])
        distance = _gate_distance(metrics, task["objective"])
        candidate_hash = _candidate_hash(proposal.factor_ids, weights)
        active_returns = evaluation.net_returns - evaluator._market_benchmark_returns(
            evaluation.net_returns.index
        )
        artifact_path, artifact_hash = write_return_artifact(
            self.artifact_root,
            task_id=self.task_id,
            candidate_hash=candidate_hash,
            net_returns=evaluation.net_returns,
            active_returns=active_returns,
        )
        metrics.update(
            {
                "autocombine_objective_profile": task["objective"]["profile"],
                "autocombine_weight_evaluations": evaluated_count,
                "autocombine_snapshot_hash": task["snapshot_hash"],
                "autocombine_hidden_metrics_exposed": False,
                "autocombine_gate_distance": distance,
                "autocombine_return_artifact_path": artifact_path,
                "autocombine_return_artifact_hash": artifact_hash,
            }
        )
        return {
            "iteration": iteration,
            "candidate_hash": candidate_hash,
            "action": proposal.action,
            "proposal_source": proposal.source,
            "factor_ids": list(proposal.factor_ids),
            "weights": list(weights),
            "rationale": proposal.rationale,
            "hypothesis": proposal.hypothesis,
            "metrics": metrics,
            "score": score,
            "gate_distance": distance,
            "qualification": "QUALIFIED" if not failures else "EVALUATED",
            "gate_status": "PASSED" if not failures else "REJECTED",
            "failed_gates": failures,
            "prompt_hash": proposal.prompt_hash,
            "response_hash": proposal.response_hash,
            "return_artifact_path": artifact_path,
            "return_artifact_hash": artifact_hash,
            "duration_seconds": time.monotonic() - started,
        }

    def _weight_candidates(
        self, factor_ids: tuple[str, ...], task: dict[str, Any], iteration: int
    ) -> list[tuple[float, ...]]:
        construction = task["construction"]
        n = len(factor_ids)
        minimum = float(construction["minimum_weight"])
        maximum = float(construction["maximum_weight"])
        step = float(construction["weight_step"])
        limit = int(task["budget"]["weight_evaluations_per_subset"])
        records = {item["factor_id"]: item for item in task["factor_snapshot"]}
        scores = np.array(
            [max(-4.0, min(4.0, records[factor_id]["prefilter_score"])) for factor_id in factor_ids]
        )
        raw: list[np.ndarray] = [np.full(n, 1 / n)]
        softmax = np.exp(scores - scores.max())
        raw.append(softmax / softmax.sum())
        inverse_risk = np.array(
            [
                1.0
                / max(
                    0.05,
                    abs(
                        _metric(
                            records[factor_id].get("metrics") or {},
                            "recent_long_only_max_drawdown",
                            "long_only_max_drawdown",
                            default=-0.5,
                        )
                    ),
                )
                for factor_id in factor_ids
            ]
        )
        raw.append(inverse_risk / inverse_risk.sum())
        for index in range(n):
            overweight = np.full(n, (1.0 - maximum) / max(1, n - 1))
            overweight[index] = maximum
            raw.append(overweight)
            underweight = np.full(n, (1.0 - minimum) / max(1, n - 1))
            underweight[index] = minimum
            raw.append(underweight)
        seed = int(
            canonical_hash(
                {"task": self.task_id, "factors": sorted(factor_ids), "iteration": iteration}
            )[:16],
            16,
        )
        generator = np.random.default_rng(seed)
        for _ in range(max(0, limit - len(raw)) * 3):
            raw.append(generator.dirichlet(np.ones(n) * 1.5))
        candidates: list[tuple[float, ...]] = []
        for values in raw:
            clipped = np.clip(values, minimum, maximum)
            clipped = clipped / clipped.sum()
            rounded = np.round(clipped / step) * step
            rounded = _repair_weight_sum(rounded, minimum, maximum, step)
            candidate = tuple(round(float(value), 6) for value in rounded)
            if candidate not in candidates:
                candidates.append(candidate)
            if len(candidates) >= limit:
                break
        return candidates

    def _strategy_independence_metrics(self, active_returns: Any) -> dict[str, Any]:
        maximum = 0.0
        nearest_strategy_id: str | None = None
        comparisons = 0
        for strategy in self.combine_store.strategies():
            evaluation = strategy.get("specification", {}).get("evaluation", {})
            path = evaluation.get("autocombine_return_artifact_path")
            if not path or not Path(path).is_file():
                continue
            try:
                reference = load_return_artifact(path)["active_return"]
                comparison = return_independence(active_returns, reference)
            except (OSError, TypeError, ValueError):
                continue
            comparisons += 1
            correlation = abs(float(comparison["pearson"]))
            if correlation > maximum:
                maximum = correlation
                nearest_strategy_id = str(strategy["strategy_id"])
        return {
            "portfolio_max_strategy_active_correlation": maximum,
            "portfolio_nearest_strategy_id": nearest_strategy_id,
            "portfolio_strategy_independence_comparisons": comparisons,
        }

    def _leave_one_out_metrics(
        self,
        evaluator: PriceVolumeEvaluator,
        records: dict[str, dict[str, Any]],
        factor_ids: list[str],
        weights: list[float],
        base_metrics: dict[str, Any],
        task: dict[str, Any],
    ) -> dict[str, Any]:
        base_score = _portfolio_score(base_metrics, [], task["objective"])
        diagnostics: list[dict[str, Any]] = []
        for index, factor_id in enumerate(factor_ids):
            remaining_ids = [
                value for position, value in enumerate(factor_ids) if position != index
            ]
            remaining_weights = np.array(
                [value for position, value in enumerate(weights) if position != index], dtype=float
            )
            remaining_weights = remaining_weights / remaining_weights.sum()
            evaluation = evaluator.evaluate_portfolio(
                [factor_from_pool_record(records[value]) for value in remaining_ids],
                weights=tuple(float(value) for value in remaining_weights),
            )
            reduced_metrics = dict(evaluation.metrics)
            reduced_metrics.update(
                mechanism_independence_metrics(records, remaining_ids, list(remaining_weights))
            )
            reduced_active = evaluation.net_returns - evaluator._market_benchmark_returns(
                evaluation.net_returns.index
            )
            reduced_metrics.update(self._strategy_independence_metrics(reduced_active))
            reduced_score = _portfolio_score(reduced_metrics, [], task["objective"])
            score_delta = base_score - reduced_score
            diagnostics.append(
                {
                    "factor_id": factor_id,
                    "objective_score_delta": score_delta,
                    "sharpe_delta": _metric(base_metrics, "portfolio_sharpe_ratio")
                    - _metric(reduced_metrics, "portfolio_sharpe_ratio"),
                    "annual_return_delta": _metric(base_metrics, "portfolio_simple_annual_return")
                    - _metric(reduced_metrics, "portfolio_simple_annual_return"),
                    "worst_fold_delta": _metric(base_metrics, "portfolio_walk_forward_worst_sharpe")
                    - _metric(reduced_metrics, "portfolio_walk_forward_worst_sharpe"),
                    "positive": score_delta > 0.01,
                }
            )
        redundant = [item["factor_id"] for item in diagnostics if not item["positive"]]
        positive_fraction = (
            sum(bool(item["positive"]) for item in diagnostics) / len(diagnostics)
            if diagnostics
            else 1.0
        )
        return {
            "portfolio_leave_one_out": diagnostics,
            "portfolio_marginal_positive_fraction": positive_fraction,
            "portfolio_redundant_factor_count": len(redundant),
            "portfolio_redundant_factor_ids": redundant,
        }

    def _update_best(self, task: dict[str, Any], experiment: dict[str, Any]) -> None:
        del task
        current_task = self._require_task()
        current_leader = (
            self.combine_store.experiment(int(current_task["best_experiment_id"]))
            if current_task.get("best_experiment_id")
            else None
        )
        leader_changed = current_leader is None or _experiment_selection_key(
            experiment
        ) > _experiment_selection_key(current_leader)
        current_qualified = (
            self.combine_store.experiment(int(current_task["qualified_experiment_id"]))
            if current_task.get("qualified_experiment_id")
            else None
        )
        qualified_changed = experiment["gate_status"] == "PASSED" and (
            current_qualified is None
            or _qualified_selection_key(experiment) > _qualified_selection_key(current_qualified)
        )
        updates: dict[str, Any] = {"phase": "ROBUSTNESS"}
        if leader_changed:
            if current_leader is not None and current_leader["id"] != experiment["id"]:
                self.combine_store.update_experiment(
                    int(current_leader["id"]),
                    qualification=(
                        "QUALIFIED" if current_leader["gate_status"] == "PASSED" else "EVALUATED"
                    ),
                )
            updates["best_experiment_id"] = experiment["id"]
            self.combine_store.update_experiment(
                int(experiment["id"]), qualification="RESEARCH_LEADER"
            )
        if qualified_changed:
            if current_qualified is not None and current_qualified["id"] != experiment["id"]:
                self.combine_store.update_experiment(
                    int(current_qualified["id"]), qualification="QUALIFIED"
                )
            updates["qualified_experiment_id"] = experiment["id"]
            updates["qualification_status"] = "QUALIFIED_CHAMPION"
            self.combine_store.update_experiment(
                int(experiment["id"]), qualification="QUALIFIED_CHAMPION"
            )
        elif current_qualified is None:
            updates["qualification_status"] = "RESEARCH_LEADER_ONLY"
        if leader_changed or qualified_changed:
            self.combine_store.update_task(
                self.task_id,
                **updates,
            )

    def _write_memory(self, experiment: dict[str, Any]) -> None:
        metrics = experiment.get("metrics") or {}
        content = _experiment_summary(experiment)
        self.combine_store.remember(
            self.task_id,
            int(experiment["iteration"]),
            "SUCCESS" if experiment["gate_status"] == "PASSED" else "FAILURE",
            content,
            {
                "factor_ids": experiment["factor_ids"],
                "weights": experiment["weights"],
                "failed_gates": experiment["failed_gates"],
                "public_metrics": _public_metric_summary(metrics),
            },
        )

    def _require_task(self) -> dict[str, Any]:
        task = self.combine_store.task(self.task_id)
        if task is None:
            raise KeyError(f"AutoCombine task not found: {self.task_id}")
        return task


class AutoCombineManager:
    def __init__(
        self,
        store: ServiceStore,
        combine_store: AutoCombineStore,
        vault: SecretVault,
        *,
        config_path: Path,
        maximum_concurrent_tasks: int = 2,
    ) -> None:
        self.store = store
        self.combine_store = combine_store
        self.vault = vault
        self.config_path = config_path
        self.maximum_concurrent_tasks = maximum_concurrent_tasks
        self._workers: dict[str, AutoCombineWorker] = {}

    def worker(self, task_id: str) -> AutoCombineWorker:
        if task_id not in self._workers:
            self._workers[task_id] = AutoCombineWorker(
                task_id,
                self.store,
                self.combine_store,
                self.vault,
                config_path=self.config_path,
            )
        return self._workers[task_id]

    async def start(self, task_id: str) -> dict[str, Any]:
        active = sum(worker.alive for worker in self._workers.values())
        worker = self.worker(task_id)
        if not worker.alive and active >= self.maximum_concurrent_tasks:
            raise RuntimeError("AutoCombine 并发任务已达到上限")
        return await worker.start()

    async def stop(self, task_id: str) -> dict[str, Any]:
        return await self.worker(task_id).stop()

    def alive(self, task_id: str) -> bool:
        return task_id in self._workers and self._workers[task_id].alive

    @property
    def active_count(self) -> int:
        return sum(worker.alive for worker in self._workers.values())

    async def shutdown(self) -> None:
        await asyncio.gather(
            *(worker.shutdown() for worker in self._workers.values()),
            return_exceptions=True,
        )


def create_task_record(
    store: ServiceStore,
    *,
    name: str,
    market: str,
    data_path: str,
    protocol: dict[str, Any],
    scope: dict[str, Any],
    construction: dict[str, Any],
    objective: dict[str, Any],
    budget: dict[str, Any],
    notes: str,
) -> dict[str, Any]:
    construction = merge_defaults(construction, DEFAULT_CONSTRUCTION)
    profile = str(objective.get("profile", DEFAULT_OBJECTIVE["profile"]))
    preset = OBJECTIVE_PRESETS.get(profile, DEFAULT_OBJECTIVE)
    objective = merge_defaults(objective, preset)
    budget = merge_defaults(budget, DEFAULT_BUDGET)
    snapshot = build_factor_snapshot(store, scope, construction)
    if len(snapshot) < int(construction["min_factors"]):
        raise ValueError("所选因子范围不足以满足最小因子数")
    snapshot_hash = canonical_hash(
        {
            "market": market,
            "data_path": str(Path(data_path).expanduser().resolve()),
            "protocol": protocol,
            "factor_ids": [item["factor_id"] for item in snapshot],
            "proposals": [item["proposal"] for item in snapshot],
        }
    )
    return {
        "task_id": f"combine-{uuid.uuid4().hex[:12]}",
        "name": name.strip(),
        "market": market,
        "data_path": str(Path(data_path).expanduser().resolve()),
        "protocol": protocol,
        "scope": scope,
        "construction": construction,
        "objective": objective,
        "budget": budget,
        "factor_snapshot": snapshot,
        "snapshot_hash": snapshot_hash,
        "notes": notes.strip(),
    }


def _repair_weight_sum(
    values: np.ndarray, minimum: float, maximum: float, step: float
) -> np.ndarray:
    result = values.copy()
    units = int(round((1.0 - result.sum()) / step))
    direction = 1 if units > 0 else -1
    for _ in range(abs(units)):
        candidates = [
            index
            for index, value in enumerate(result)
            if minimum - 1e-9 <= value + direction * step <= maximum + 1e-9
        ]
        if not candidates:
            break
        index = (
            min(candidates, key=lambda item: result[item])
            if direction > 0
            else max(candidates, key=lambda item: result[item])
        )
        result[index] += direction * step
    if not math.isclose(float(result.sum()), 1.0, abs_tol=1e-8):
        result = result / result.sum()
    return result


def _candidate_hash(factor_ids: tuple[str, ...], weights: tuple[float, ...]) -> str:
    ordered = sorted(zip(factor_ids, weights, strict=True))
    return canonical_hash([(factor_id, round(weight, 6)) for factor_id, weight in ordered])


class _CandidateEvaluationRejected(Exception):
    pass


def _rejected_experiment_record(
    task: dict[str, Any],
    proposal: CombineProposal,
    iteration: int,
    error: _CandidateEvaluationRejected,
) -> dict[str, Any]:
    message = str(error)
    rejected_hash = canonical_hash({"factors": proposal.factor_ids, "rejected": message})[:12]
    return {
        "iteration": iteration,
        "candidate_hash": f"{rejected_hash}-rejected",
        "action": proposal.action,
        "proposal_source": proposal.source,
        "factor_ids": list(proposal.factor_ids),
        "weights": [],
        "rationale": proposal.rationale,
        "hypothesis": proposal.hypothesis,
        "metrics": {
            "autocombine_objective_profile": task["objective"]["profile"],
            "autocombine_snapshot_hash": task["snapshot_hash"],
            "autocombine_candidate_level_failure": message,
            "autocombine_hidden_metrics_exposed": False,
        },
        "score": -1_000_000.0,
        "gate_distance": 1_000_000.0,
        "qualification": "CANDIDATE_EVALUATION_REJECTED",
        "gate_status": "REJECTED",
        "failed_gates": ["candidate_evaluation_rejected"],
        "prompt_hash": proposal.prompt_hash,
        "response_hash": proposal.response_hash,
        "return_artifact_path": None,
        "return_artifact_hash": None,
        "duration_seconds": 0.0,
    }


def _recoverable_evaluation_failure_reason(error: Exception) -> str | None:
    message = str(error).lower()
    if "database" in message or "locked" in message or "no parquet" in message:
        return None
    markers = {
        "non-finite": "NON_FINITE_METRICS",
        "not finite": "NON_FINITE_METRICS",
        "nan": "NON_FINITE_METRICS",
        "inf": "NON_FINITE_METRICS",
        "insufficient coverage": "INSUFFICIENT_COVERAGE",
        "coverage": "INSUFFICIENT_COVERAGE",
        "no dates fall inside": "NO_PUBLIC_VALIDATION_DATES",
        "walk-forward folds": "INSUFFICIENT_WALK_FORWARD_FOLDS",
        "no target securities": "EMPTY_PORTFOLIO_SELECTION",
        "a portfolio requires at least one factor": "EMPTY_PORTFOLIO_SELECTION",
    }
    for marker, reason in markers.items():
        if marker in message:
            return reason
    return None


def _experiment_selection_key(experiment: dict[str, Any]) -> tuple[float, ...]:
    failures = experiment.get("failed_gates") or []
    distance = float(experiment.get("gate_distance") or 0.0)
    return (
        float(experiment.get("gate_status") == "PASSED"),
        -float(len(failures)),
        -distance,
        float(experiment.get("score") or -1_000.0),
    )


def _qualified_selection_key(experiment: dict[str, Any]) -> tuple[float, ...]:
    metrics = experiment.get("metrics") or {}
    return (
        float(experiment.get("score") or -1_000.0),
        _metric(metrics, "portfolio_walk_forward_worst_sharpe", default=-100.0),
        _metric(metrics, "portfolio_active_information_ratio", default=-100.0),
        _metric(metrics, "portfolio_max_drawdown", default=-1.0),
    )


def _gate_failures(metrics: dict[str, Any], objective: dict[str, Any]) -> list[str]:
    checks = {
        "coverage": _metric(metrics, "portfolio_coverage") >= float(objective["minimum_coverage"]),
        "positive_folds": _metric(metrics, "portfolio_walk_forward_positive_fraction")
        >= float(objective["minimum_positive_fold_fraction"]),
        "worst_fold": _metric(metrics, "portfolio_walk_forward_worst_sharpe", default=-100)
        >= float(objective["minimum_worst_fold_sharpe"]),
        "drawdown": _metric(metrics, "portfolio_max_drawdown", default=-1)
        >= -float(objective["maximum_drawdown"]),
        "turnover": _metric(metrics, "portfolio_annual_turnover", default=1_000)
        <= float(objective["maximum_annual_turnover"]),
        "correlation": _metric(metrics, "portfolio_max_factor_correlation", default=1)
        <= float(objective["maximum_factor_correlation"]),
        "effective_factor_bets": _metric(
            metrics,
            "portfolio_effective_factor_bets",
            default=float(objective.get("minimum_effective_factor_bets", 1.0)),
        )
        >= float(objective.get("minimum_effective_factor_bets", 1.0)),
        "effective_mechanisms": _metric(
            metrics,
            "portfolio_effective_mechanisms",
            default=float(objective.get("minimum_effective_mechanisms", 1.0)),
        )
        >= float(objective.get("minimum_effective_mechanisms", 1.0)),
        "mechanism_concentration": _metric(
            metrics,
            "portfolio_maximum_mechanism_weight",
            default=float(objective.get("maximum_mechanism_weight", 1.0)),
        )
        <= float(objective.get("maximum_mechanism_weight", 1.0)),
        "semantic_duplicates": _metric(
            metrics, "portfolio_duplicate_semantic_factor_count", default=0.0
        )
        <= float(objective.get("maximum_duplicate_semantic_factors", 0)),
        "strategy_independence": _metric(
            metrics, "portfolio_max_strategy_active_correlation", default=0.0
        )
        <= float(objective.get("maximum_strategy_active_correlation", 1.0)),
        "deflated_sharpe": _metric(
            metrics,
            "portfolio_deflated_sharpe_probability",
            default=float(objective.get("minimum_deflated_sharpe_probability", 0.0)),
        )
        >= float(objective.get("minimum_deflated_sharpe_probability", 0.0)),
        "cost_stress": _metric(metrics, "portfolio_cost_stress_net_ir", default=-100)
        >= float(objective["minimum_cost_stress_ir"]),
        "annual_return": _metric(metrics, "portfolio_simple_annual_return")
        >= float(objective.get("minimum_simple_annual_return", 0.0)),
    }
    if "portfolio_marginal_positive_fraction" in metrics:
        checks["marginal_contribution"] = _metric(
            metrics, "portfolio_marginal_positive_fraction"
        ) >= float(objective.get("minimum_marginal_positive_fraction", 0.0))
    return [name for name, passed in checks.items() if not passed]


def _gate_distance(metrics: dict[str, Any], objective: dict[str, Any]) -> float:
    lower_bounds = {
        "portfolio_coverage": float(objective["minimum_coverage"]),
        "portfolio_walk_forward_positive_fraction": float(
            objective["minimum_positive_fold_fraction"]
        ),
        "portfolio_walk_forward_worst_sharpe": float(objective["minimum_worst_fold_sharpe"]),
        "portfolio_max_drawdown": -float(objective["maximum_drawdown"]),
        "portfolio_cost_stress_net_ir": float(objective["minimum_cost_stress_ir"]),
        "portfolio_simple_annual_return": float(objective.get("minimum_simple_annual_return", 0.0)),
        "portfolio_effective_factor_bets": float(
            objective.get("minimum_effective_factor_bets", 1.0)
        ),
        "portfolio_effective_mechanisms": float(objective.get("minimum_effective_mechanisms", 1.0)),
        "portfolio_deflated_sharpe_probability": float(
            objective.get("minimum_deflated_sharpe_probability", 0.0)
        ),
    }
    upper_bounds = {
        "portfolio_annual_turnover": float(objective["maximum_annual_turnover"]),
        "portfolio_max_factor_correlation": float(objective["maximum_factor_correlation"]),
        "portfolio_maximum_mechanism_weight": float(objective.get("maximum_mechanism_weight", 1.0)),
        "portfolio_duplicate_semantic_factor_count": float(
            objective.get("maximum_duplicate_semantic_factors", 0)
        ),
        "portfolio_max_strategy_active_correlation": float(
            objective.get("maximum_strategy_active_correlation", 1.0)
        ),
    }
    distance = 0.0
    for key, bound in lower_bounds.items():
        value = _metric(metrics, key, default=bound)
        scale = max(abs(bound), 0.1)
        distance += max(0.0, bound - value) / scale
    for key, bound in upper_bounds.items():
        value = _metric(metrics, key, default=bound)
        scale = max(abs(bound), 0.1)
        distance += max(0.0, value - bound) / scale
    if "portfolio_marginal_positive_fraction" in metrics:
        bound = float(objective.get("minimum_marginal_positive_fraction", 0.0))
        distance += max(
            0.0,
            bound - _metric(metrics, "portfolio_marginal_positive_fraction"),
        ) / max(bound, 0.1)
    return float(distance)


def _portfolio_score(
    metrics: dict[str, Any],
    failures: list[str],
    objective: dict[str, Any] | None = None,
) -> float:
    del failures
    objective = merge_defaults(objective or {}, DEFAULT_OBJECTIVE)
    active_ir = _metric(metrics, "portfolio_active_information_ratio")
    active_annual = _metric(metrics, "portfolio_active_simple_annual_return")
    sharpe = _metric(metrics, "portfolio_sharpe_ratio")
    annual = _metric(metrics, "portfolio_simple_annual_return")
    worst = _metric(metrics, "portfolio_walk_forward_worst_sharpe", default=-3)
    positive = _metric(metrics, "portfolio_walk_forward_positive_fraction")
    drawdown = _metric(metrics, "portfolio_max_drawdown", default=-1)
    turnover = _metric(metrics, "portfolio_annual_turnover", default=100)
    correlation = _metric(metrics, "portfolio_max_factor_correlation", default=1)
    effective_bets = _metric(metrics, "portfolio_effective_factor_bets", default=1)
    effective_mechanisms = _metric(metrics, "portfolio_effective_mechanisms", default=1)
    mechanism_weight = _metric(metrics, "portfolio_maximum_mechanism_weight", default=1)
    strategy_correlation = _metric(metrics, "portfolio_max_strategy_active_correlation", default=0)
    profile = str(objective.get("profile", "ROBUST_ACTIVE_LONG_ONLY"))
    components = {
        "ROBUST_ACTIVE_LONG_ONLY": 0.25 * active_ir
        + 1.50 * active_annual
        + 0.20 * sharpe
        + 0.75 * annual
        + 0.20 * worst
        + 0.25 * positive
        + 0.40 * drawdown
        - 0.004 * turnover
        - 0.10 * correlation,
        "DRAWDOWN_FIRST": 4.00 * drawdown
        + 0.45 * worst
        + 0.35 * sharpe
        + 1.20 * annual
        + 0.20 * positive
        - 0.003 * turnover,
        "PORTFOLIO_SHARPE_FIRST": 0.95 * sharpe
        + 0.35 * worst
        + 0.25 * positive
        + 0.80 * annual
        + 0.55 * drawdown
        - 0.003 * turnover,
        "ABSOLUTE_LONG_ONLY": 4.00 * annual
        + 0.40 * sharpe
        + 0.25 * worst
        + 0.60 * drawdown
        - 0.004 * turnover,
        "LOW_TURNOVER": -0.018 * turnover
        + 0.45 * sharpe
        + 1.50 * annual
        + 0.30 * worst
        + 0.45 * drawdown,
        "DIVERSIFICATION_FIRST": -0.90 * correlation
        - 0.75 * strategy_correlation
        - 0.50 * mechanism_weight
        + 0.20 * effective_bets
        + 0.20 * effective_mechanisms
        + 0.35 * sharpe
        + 1.20 * annual
        + 0.30 * worst
        + 0.50 * drawdown,
    }
    independence_adjustment = (
        0.08 * effective_bets
        + 0.08 * effective_mechanisms
        - 0.12 * mechanism_weight
        - 0.12 * strategy_correlation
    )
    return float(
        components.get(profile, components["ROBUST_ACTIVE_LONG_ONLY"]) + independence_adjustment
    )


def _public_metric_summary(metrics: dict[str, Any]) -> dict[str, float]:
    keys = (
        "portfolio_sharpe_ratio",
        "portfolio_simple_annual_return",
        "portfolio_active_information_ratio",
        "portfolio_active_simple_annual_return",
        "portfolio_max_drawdown",
        "portfolio_annual_turnover",
        "portfolio_max_factor_correlation",
        "portfolio_effective_factor_bets",
        "portfolio_effective_mechanisms",
        "portfolio_maximum_mechanism_weight",
        "portfolio_max_strategy_active_correlation",
        "portfolio_marginal_positive_fraction",
        "portfolio_walk_forward_positive_fraction",
        "portfolio_walk_forward_worst_sharpe",
        "portfolio_cost_stress_net_ir",
    )
    return {key: float(metrics[key]) for key in keys if isinstance(metrics.get(key), int | float)}


def _experiment_summary(experiment: dict[str, Any]) -> str:
    metrics = experiment.get("metrics") or {}
    return (
        f"{len(experiment['factor_ids'])} 因子 · 夏普 "
        f"{_metric(metrics, 'portfolio_sharpe_ratio'):.2f} · 年化 "
        f"{_metric(metrics, 'portfolio_simple_annual_return'):.2%} · 最差折 "
        f"{_metric(metrics, 'portfolio_walk_forward_worst_sharpe', default=-100):.2f} · "
        f"门禁 {len(experiment['failed_gates'])} 项未通过"
    )


def refresh_task_strategy_clusters(
    combine_store: AutoCombineStore, *, threshold: float = 0.75
) -> dict[str, dict[str, Any]]:
    leaders: dict[str, tuple[dict[str, Any], Any]] = {}
    for task in combine_store.tasks():
        if not task.get("best_experiment_id"):
            continue
        experiment = combine_store.experiment(int(task["best_experiment_id"]))
        if experiment is None:
            continue
        path = experiment.get("return_artifact_path") or (experiment.get("metrics") or {}).get(
            "autocombine_return_artifact_path"
        )
        if not path or not Path(path).is_file():
            continue
        try:
            active = load_return_artifact(path)["active_return"]
        except (OSError, TypeError, ValueError):
            continue
        leaders[str(task["task_id"])] = (task, active)
    parent = {task_id: task_id for task_id in leaders}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    comparisons: dict[str, list[tuple[str, float]]] = {task_id: [] for task_id in leaders}
    task_ids = sorted(leaders)
    for left_index, left_id in enumerate(task_ids):
        for right_id in task_ids[left_index + 1 :]:
            result = return_independence(leaders[left_id][1], leaders[right_id][1])
            correlation = abs(float(result["pearson"]))
            comparisons[left_id].append((right_id, correlation))
            comparisons[right_id].append((left_id, correlation))
            if correlation >= threshold:
                union(left_id, right_id)
    groups: dict[str, list[str]] = {}
    for task_id in task_ids:
        groups.setdefault(find(task_id), []).append(task_id)
    cluster_ids = {
        task_id: f"RC_{canonical_hash(sorted(members))[:10]}"
        for members in groups.values()
        for task_id in members
    }
    result: dict[str, dict[str, Any]] = {}
    for task_id in task_ids:
        nearest = max(comparisons[task_id], key=lambda item: item[1], default=(None, 0.0))
        payload = {
            "strategy_cluster_id": cluster_ids[task_id],
            "nearest_task_id": nearest[0],
            "nearest_active_return_correlation": float(nearest[1]),
        }
        task = leaders[task_id][0]
        if any(task.get(key) != value for key, value in payload.items()):
            combine_store.update_task(task_id, **payload)
        result[task_id] = payload
    return result
