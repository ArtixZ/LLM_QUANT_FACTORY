from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from autoalpha.config import ResearchConfig
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
}
DEFAULT_OBJECTIVE = {
    "profile": "ROBUST_ACTIVE_LONG_ONLY",
    "preset_version": 1,
    "minimum_coverage": 0.80,
    "minimum_positive_fold_fraction": 0.50,
    "minimum_worst_fold_sharpe": -0.50,
    "maximum_drawdown": 0.30,
    "maximum_annual_turnover": 40.0,
    "maximum_factor_correlation": 0.85,
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
    "weight_evaluations_per_subset": 6,
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
    records: list[dict[str, Any]] = []
    available_ids: set[str] = set()
    for record in store.factor_pool(limit=5000):
        factor_id = str(record["factor_id"])
        available_ids.add(factor_id)
        if factor_id in excluded or factor_id in contaminated:
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
        records.append(
            {
                "factor_id": factor_id,
                "name": record["name"],
                "family": record["family"],
                "status": record["status"],
                "source_task_id": record.get("source_task_id"),
                "source_iteration": record.get("source_iteration"),
                "proposal": proposal,
                "metrics": record.get("metrics") or {},
                "prefilter_score": _prefilter_score(record.get("metrics") or {}),
                "required": factor_id in required,
            }
        )
    requested = explicit | required
    unknown = requested - available_ids
    if unknown:
        raise ValueError(f"因子不存在：{', '.join(sorted(unknown))}")
    missing = requested - {item["factor_id"] for item in records}
    if missing:
        raise ValueError(f"因子不可用于组合研究（污染或表达式无效）：{', '.join(sorted(missing))}")
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
                evaluated = await asyncio.to_thread(
                    self._evaluate_proposal, evaluator, task, proposal, iteration
                )
                experiment = self.combine_store.record_experiment(self.task_id, evaluated)
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
        best = (
            self.combine_store.experiment(int(task["best_experiment_id"]))
            if task.get("best_experiment_id")
            else None
        )
        status = "EXHAUSTED"
        if best and best["gate_status"] == "PASSED":
            self.combine_store.update_task(self.task_id, phase="BLIND_REVIEW")
            records = {item["factor_id"]: item for item in task["factor_snapshot"]}
            factors = [factor_from_pool_record(records[value]) for value in best["factor_ids"]]
            evaluator = self._evaluator or await asyncio.to_thread(self._build_evaluator, task)
            verdict = await asyncio.to_thread(
                BlindEvaluationBoundary(evaluator.panel_path, evaluator.config).evaluate_holdout,
                factors,
                tuple(float(value) for value in best["weights"]),
            )
            self.combine_store.update_task(
                self.task_id,
                blind_verdict=verdict.verdict,
                blind_evidence_hash=verdict.evidence_hash,
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
            except (ModelInvocationError, RuntimeError, TypeError, ValueError) as error:
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
        family = {item["factor_id"]: str(item["family"]).casefold() for item in pool}
        passing = [item for item in experiments if item["gate_status"] == "PASSED"]
        incumbent = max(
            passing or experiments,
            key=lambda item: item.get("score") or -999,
            default=None,
        )

        candidates: list[tuple[str, ...]] = []
        if not experiments:
            candidates.extend(
                self._diversified_seeds(ranked, family, max(min_count, len(required)), required)
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
                self._diversified_seeds(ranked, family, max(min_count, len(required)), required)
            )

        for factor_ids in candidates:
            factor_ids = tuple(dict.fromkeys(factor_ids))
            if not min_count <= len(factor_ids) <= max_count:
                continue
            if not required <= set(factor_ids):
                continue
            if not self._family_limit_ok(factor_ids, family, construction):
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
        factor_ids: tuple[str, ...], family: dict[str, str], construction: dict[str, Any]
    ) -> bool:
        maximum = int(construction.get("maximum_same_family", 2))
        return all(
            sum(family[value] == family[factor_id] for value in factor_ids) <= maximum
            for factor_id in factor_ids
        )

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
                "family": item["family"],
                "hypothesis": item["proposal"].get("hypothesis", ""),
                "prefilter_score": round(item["prefilter_score"], 4),
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
                "public_summary": _public_metric_summary(item.get("metrics") or {}),
            }
            for item in experiments[:12]
        ]
        analysis = await client.analyze(
            role="AUTOCOMBINE_PORTFOLIO_ARCHITECT",
            system_prompt=(
                "You are the portfolio architecture role in an institutional A-share research "
                "system. Select only existing factor_ids. Propose one static long-only composite "
                "signal subset. Do not invent factors, inspect hidden periods, flip signs, or tune "
                "decimal weights. Favor mechanism complementarity and robust public out-of-sample "
                "evidence. Return JSON with action, factor_ids, rationale, hypothesis, risk."
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
        del task
        proposed = frozenset(proposal.factor_ids)
        return any(
            frozenset(experiment["factor_ids"]) == proposed
            for experiment in self.combine_store.experiments(self.task_id, limit=5000)
        )

    def _evaluate_proposal(
        self,
        evaluator: PriceVolumeEvaluator,
        task: dict[str, Any],
        proposal: CombineProposal,
        iteration: int,
    ) -> dict[str, Any]:
        started = time.monotonic()
        records = {item["factor_id"]: item for item in task["factor_snapshot"]}
        factors = [factor_from_pool_record(records[factor_id]) for factor_id in proposal.factor_ids]
        best: tuple[float, tuple[float, ...], dict[str, Any], list[str]] | None = None
        evaluated_count = 0
        for weights in self._weight_candidates(proposal.factor_ids, task):
            evaluation = evaluator.evaluate_portfolio(factors, weights=weights)
            evaluated_count += 1
            metrics = evaluation.metrics
            failures = _gate_failures(metrics, task["objective"])
            score = _portfolio_score(metrics, failures, task["objective"])
            if best is None or score > best[0]:
                best = (score, weights, metrics, failures)
        if best is None:
            raise RuntimeError("该组合的全部权重候选均已评估")
        score, weights, metrics, failures = best
        metrics = dict(metrics)
        metrics.update(
            {
                "autocombine_objective_profile": task["objective"]["profile"],
                "autocombine_weight_evaluations": evaluated_count,
                "autocombine_snapshot_hash": task["snapshot_hash"],
                "autocombine_hidden_metrics_exposed": False,
            }
        )
        return {
            "iteration": iteration,
            "candidate_hash": _candidate_hash(proposal.factor_ids, weights),
            "action": proposal.action,
            "proposal_source": proposal.source,
            "factor_ids": list(proposal.factor_ids),
            "weights": list(weights),
            "rationale": proposal.rationale,
            "hypothesis": proposal.hypothesis,
            "metrics": metrics,
            "score": score,
            "gate_status": "PASSED" if not failures else "REJECTED",
            "failed_gates": failures,
            "prompt_hash": proposal.prompt_hash,
            "response_hash": proposal.response_hash,
            "duration_seconds": time.monotonic() - started,
        }

    def _weight_candidates(
        self, factor_ids: tuple[str, ...], task: dict[str, Any]
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
        for index in range(n):
            overweight = np.full(n, (1.0 - maximum) / max(1, n - 1))
            overweight[index] = maximum
            raw.append(overweight)
            underweight = np.full(n, (1.0 - minimum) / max(1, n - 1))
            underweight[index] = minimum
            raw.append(underweight)
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

    def _update_best(self, task: dict[str, Any], experiment: dict[str, Any]) -> None:
        current = (
            self.combine_store.experiment(int(task["best_experiment_id"]))
            if task.get("best_experiment_id")
            else None
        )
        candidate_passed = experiment["gate_status"] == "PASSED"
        current_passed = current is not None and current["gate_status"] == "PASSED"
        should_update = (
            current is None
            or (candidate_passed and not current_passed)
            or (
                candidate_passed == current_passed
                and float(experiment["score"]) > float(current["score"])
            )
        )
        if should_update:
            self.combine_store.update_task(
                self.task_id, best_experiment_id=experiment["id"], phase="ROBUSTNESS"
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
        "cost_stress": _metric(metrics, "portfolio_cost_stress_net_ir", default=-100)
        >= float(objective["minimum_cost_stress_ir"]),
        "annual_return": _metric(metrics, "portfolio_simple_annual_return")
        >= float(objective.get("minimum_simple_annual_return", 0.0)),
    }
    return [name for name, passed in checks.items() if not passed]


def _portfolio_score(
    metrics: dict[str, Any],
    failures: list[str],
    objective: dict[str, Any] | None = None,
) -> float:
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
        + 0.35 * sharpe
        + 1.20 * annual
        + 0.30 * worst
        + 0.50 * drawdown,
    }
    return float(
        components.get(profile, components["ROBUST_ACTIVE_LONG_ONLY"]) - 0.35 * len(failures)
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
