from __future__ import annotations

import asyncio
import hashlib
import math
import time
import uuid
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from autoalpha.config import ResearchConfig
from autoalpha.service.autocombine import (
    DEFAULT_CONSTRUCTION,
    DEFAULT_OBJECTIVE,
    OBJECTIVE_PRESETS,
    _candidate_hash,
    _gate_distance,
    _gate_failures,
    _metric,
    _portfolio_score,
    _repair_weight_sum,
    build_factor_snapshot,
    canonical_hash,
    merge_defaults,
)
from autoalpha.service.autocombine_intelligence import (
    load_return_artifact,
    mechanism_independence_metrics,
    return_independence,
    write_return_artifact,
)
from autoalpha.service.autocombine_store import AutoCombineStore
from autoalpha.service.blind_evaluator import BlindEvaluationBoundary
from autoalpha.service.evaluator import PriceVolumeEvaluator
from autoalpha.service.multifactor import factor_from_pool_record
from autoalpha.service.quantcombine_store import QuantCombineStore
from autoalpha.service.research_protocol import task_research_config
from autoalpha.service.store import ServiceStore

DEFAULT_ENGINE = {
    "mode": "ENSEMBLE",
    "cluster_correlation_threshold": 0.78,
    "strategy_reference_limit": 25,
    "strategy_reference_minimum_observations": 120,
    "minimum_stability_score": -2.0,
    "sffs_beam_width": 3,
    "evolution_population": 12,
    "evolution_generations": 4,
    "adaptive_trials": 16,
    "covariance_shrinkage": 0.35,
    "weight_regularization": 0.08,
    "random_seed": 20260718,
}

DEFAULT_BUDGET = {
    "maximum_evaluations": 180,
    "maximum_runtime_minutes": 240,
    "weight_candidates_per_subset": 8,
    "iteration_interval_seconds": 0.0,
}


def create_quant_task_record(
    store: ServiceStore,
    *,
    name: str,
    market: str,
    data_path: str,
    protocol: dict[str, Any],
    scope: dict[str, Any],
    construction: dict[str, Any],
    objective: dict[str, Any],
    engine: dict[str, Any],
    budget: dict[str, Any],
    notes: str,
) -> dict[str, Any]:
    construction = merge_defaults(construction, DEFAULT_CONSTRUCTION)
    profile = str(objective.get("profile", DEFAULT_OBJECTIVE["profile"]))
    objective = merge_defaults(objective, OBJECTIVE_PRESETS.get(profile, DEFAULT_OBJECTIVE))
    engine = merge_defaults(engine, DEFAULT_ENGINE)
    budget = merge_defaults(budget, DEFAULT_BUDGET)
    snapshot = build_factor_snapshot(store, scope, construction)
    if len(snapshot) < int(construction["min_factors"]):
        raise ValueError("所选因子范围不足以满足最小因子数")
    resolved = str(Path(data_path).expanduser().resolve())
    snapshot_hash = canonical_hash(
        {
            "market": market,
            "data_path": resolved,
            "protocol": protocol,
            "factors": [(item["factor_id"], item["mechanism_fingerprint"]) for item in snapshot],
            "engine": engine,
        }
    )
    return {
        "task_id": f"qcombine-{uuid.uuid4().hex[:12]}",
        "name": name.strip(),
        "market": market,
        "data_path": resolved,
        "protocol": protocol,
        "scope": scope,
        "construction": construction,
        "objective": objective,
        "engine": engine,
        "budget": budget,
        "factor_snapshot": snapshot,
        "snapshot_hash": snapshot_hash,
        "notes": notes.strip(),
    }


def _standalone_stability_score(metrics: dict[str, Any]) -> float:
    sharpe = _metric(metrics, "portfolio_sharpe_ratio")
    annual = _metric(metrics, "portfolio_simple_annual_return")
    active_ir = _metric(metrics, "portfolio_active_information_ratio")
    worst = _metric(metrics, "portfolio_walk_forward_worst_sharpe", default=-3.0)
    positive = _metric(metrics, "portfolio_walk_forward_positive_fraction")
    drawdown = _metric(metrics, "portfolio_max_drawdown", default=-1.0)
    turnover = _metric(metrics, "portfolio_annual_turnover", default=100.0)
    dsr = _metric(metrics, "portfolio_deflated_sharpe_probability")
    return float(
        0.55 * math.tanh(sharpe / 2)
        + 1.25 * math.tanh(annual / 0.25)
        + 0.25 * math.tanh(active_ir / 2)
        + 0.25 * math.tanh(worst / 2)
        + 0.25 * positive
        + 0.70 * drawdown
        - 0.003 * turnover
        + 0.15 * dsr
    )


def _candidate_key(candidate: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(candidate.get("gate_status") == "PASSED"),
        -float(len(candidate.get("failed_gates") or [])),
        -float(candidate.get("gate_distance") or 0.0),
        float(candidate.get("score") or -1_000.0),
    )


def _complete_linkage_groups(
    ordered: list[str], is_similar: Callable[[str, str], bool]
) -> list[list[str]]:
    groups: list[list[str]] = []
    for factor_id in ordered:
        destination = next(
            (
                members
                for members in groups
                if all(is_similar(factor_id, value) for value in members)
            ),
            None,
        )
        if destination is None:
            groups.append([factor_id])
        else:
            destination.append(factor_id)
    return groups


def _stage_evaluation_limits(
    mode: str, screening_evaluations: int, maximum_evaluations: int
) -> tuple[int, int]:
    available = max(0, maximum_evaluations - screening_evaluations)
    if mode == "DETERMINISTIC":
        return maximum_evaluations, maximum_evaluations
    sffs_fraction = 0.40 if mode == "ENSEMBLE" else 0.45
    sffs_limit = screening_evaluations + round(available * sffs_fraction)
    if mode == "EVOLUTIONARY":
        return sffs_limit, maximum_evaluations
    if mode == "BAYESIAN":
        return sffs_limit, sffs_limit
    evolution_limit = screening_evaluations + round(available * 0.75)
    return sffs_limit, evolution_limit


def _objective_vector(metrics: dict[str, Any]) -> list[float]:
    return [
        _metric(metrics, "portfolio_sharpe_ratio", default=-100),
        _metric(metrics, "portfolio_simple_annual_return", default=-100),
        _metric(metrics, "portfolio_max_drawdown", default=-1),
        _metric(metrics, "portfolio_walk_forward_worst_sharpe", default=-100),
        -_metric(metrics, "portfolio_annual_turnover", default=1_000),
        -_metric(metrics, "portfolio_max_factor_correlation", default=1),
        -_metric(metrics, "portfolio_max_strategy_active_correlation", default=1),
        _metric(metrics, "portfolio_effective_factor_bets", default=1),
        _metric(metrics, "portfolio_effective_mechanisms", default=1),
    ]


def _dominates(left: list[float], right: list[float]) -> bool:
    return all(a >= b for a, b in zip(left, right, strict=True)) and any(
        a > b for a, b in zip(left, right, strict=True)
    )


def _bounded_simplex(values: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    result = np.clip(np.asarray(values, dtype=float), minimum, maximum)
    for _ in range(50):
        difference = 1.0 - float(result.sum())
        if abs(difference) <= 1e-12:
            break
        eligible = np.flatnonzero(
            result < maximum - 1e-12 if difference > 0 else result > minimum + 1e-12
        )
        if not len(eligible):
            break
        result[eligible] += difference / len(eligible)
        result = np.clip(result, minimum, maximum)
    if not math.isclose(float(result.sum()), 1.0, abs_tol=1e-8):
        raise ValueError("Weight bounds do not admit a unit-sum portfolio")
    return result


def _homogeneity_limit_ok(
    factor_ids: tuple[str, ...],
    records: dict[str, dict[str, Any]],
    construction: dict[str, Any],
) -> bool:
    return _homogeneity_limit_violation(factor_ids, records, construction) is None


def _homogeneity_limit_violation(
    factor_ids: tuple[str, ...],
    records: dict[str, dict[str, Any]],
    construction: dict[str, Any],
) -> dict[str, Any] | None:
    maximum_family = int(construction.get("maximum_same_family", 2))
    maximum_cluster = int(construction.get("maximum_same_semantic_cluster", 1))
    maximum_parameter_family = int(construction.get("maximum_same_parameter_family", 1))
    mechanisms = {
        factor_id: str(records[factor_id].get("mechanism") or records[factor_id].get("family"))
        for factor_id in factor_ids
    }
    search_clusters = {
        factor_id: str(
            records[factor_id].get("search_cluster_id")
            or records[factor_id].get("semantic_cluster_id")
            or factor_id
        )
        for factor_id in factor_ids
    }
    parameter_families = {
        factor_id: str(records[factor_id].get("parameter_family") or "NO_EXPLICIT_LOOKBACK")
        for factor_id in factor_ids
        if str(records[factor_id].get("parameter_family") or "NO_EXPLICIT_LOOKBACK")
        != "NO_EXPLICIT_LOOKBACK"
    }
    for dimension, labels, maximum in (
        ("mechanism", mechanisms, maximum_family),
        ("search_cluster", search_clusters, maximum_cluster),
        ("parameter_family", parameter_families, maximum_parameter_family),
    ):
        label_counts = Counter(labels.values())
        crowded = {
            label: count for label, count in label_counts.items() if count > max(0, maximum)
        }
        if crowded:
            return {
                "reason": "HOMOGENEITY_DIVERSIFICATION_CONSTRAINT",
                "dimension": dimension,
                "maximum_allowed": maximum,
                "crowded_labels": crowded,
                "factor_ids": list(factor_ids),
            }
    return None


def pareto_ranks(candidates: list[dict[str, Any]]) -> dict[int, tuple[int, float]]:
    if not candidates:
        return {}
    ids = [int(item["id"]) for item in candidates]
    raw_vectors = {int(item["id"]): list(item["objectives"]) for item in candidates}
    width = max((len(value) for value in raw_vectors.values()), default=0)
    vectors = {
        candidate_id: [*value, *([0.0] * (width - len(value)))]
        for candidate_id, value in raw_vectors.items()
    }
    dominates: dict[int, list[int]] = {candidate_id: [] for candidate_id in ids}
    dominated_count = {candidate_id: 0 for candidate_id in ids}
    fronts: list[list[int]] = [[]]
    for left in ids:
        for right in ids:
            if left == right:
                continue
            if _dominates(vectors[left], vectors[right]):
                dominates[left].append(right)
            elif _dominates(vectors[right], vectors[left]):
                dominated_count[left] += 1
        if dominated_count[left] == 0:
            fronts[0].append(left)
    rank = 0
    while fronts[rank]:
        next_front: list[int] = []
        for left in fronts[rank]:
            for right in dominates[left]:
                dominated_count[right] -= 1
                if dominated_count[right] == 0:
                    next_front.append(right)
        rank += 1
        fronts.append(next_front)
    result: dict[int, tuple[int, float]] = {}
    for rank, front in enumerate(fronts[:-1]):
        crowding = {candidate_id: 0.0 for candidate_id in front}
        if len(front) <= 2:
            crowding = {candidate_id: 1_000_000.0 for candidate_id in front}
        else:
            for dimension in range(len(next(iter(vectors.values())))):
                ordered = sorted(front, key=lambda value: vectors[value][dimension])
                crowding[ordered[0]] = crowding[ordered[-1]] = 1_000_000.0
                low = vectors[ordered[0]][dimension]
                high = vectors[ordered[-1]][dimension]
                scale = max(high - low, 1e-12)
                for position in range(1, len(ordered) - 1):
                    crowding[ordered[position]] += (
                        vectors[ordered[position + 1]][dimension]
                        - vectors[ordered[position - 1]][dimension]
                    ) / scale
        result.update({candidate_id: (rank, crowding[candidate_id]) for candidate_id in front})
    return result


class QuantCombineWorker:
    def __init__(
        self,
        task_id: str,
        store: ServiceStore,
        quant_store: QuantCombineStore,
        *,
        config_path: Path,
    ) -> None:
        self.task_id = task_id
        self.store = store
        self.quant_store = quant_store
        self.config_path = config_path
        self.artifact_root = self.store.path.parent / "artifacts"
        self._task: asyncio.Task[None] | None = None
        self._evaluator: PriceVolumeEvaluator | None = None
        self._auto_store = AutoCombineStore(store)
        self._standalone_returns: dict[str, pd.Series] = {}
        self._started = 0.0

    @property
    def alive(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> dict[str, Any]:
        task = self._require_task()
        if self.alive:
            return task
        if task["status"] in {"COMPLETED", "RESEARCH_COMPLETED", "BLIND_REJECTED"}:
            raise RuntimeError("已完成任务需复制为新任务后继续研究")
        self.quant_store.update_task(
            self.task_id,
            status="RUNNING",
            phase="PREFLIGHT",
            stop_requested=0,
            last_error=None,
        )
        self.quant_store.event(
            self.task_id,
            "action",
            "QUANT_SEARCH_STARTED",
            "统计组合研究已启动",
            "不调用 LLM；将依次执行稳定性筛选、聚类、SFFS、NSGA-II与自适应采样。",
        )
        self._task = asyncio.create_task(self._loop(), name=f"quantcombine-{self.task_id}")
        return self._require_task()

    async def stop(self) -> dict[str, Any]:
        self._require_task()
        if not self.alive:
            return self.quant_store.update_task(
                self.task_id, status="PAUSED", phase="PAUSED", stop_requested=0
            )
        self.quant_store.event(
            self.task_id,
            "action",
            "QUANT_STOP_REQUESTED",
            "停止请求已登记",
            "当前评价完成并写入数据库后暂停。",
            level="WARN",
        )
        return self.quant_store.update_task(
            self.task_id, status="STOPPING", phase="CHECKPOINT", stop_requested=1
        )

    async def shutdown(self) -> None:
        if self.alive:
            self.quant_store.update_task(self.task_id, stop_requested=1)
            await self._task

    async def _loop(self) -> None:
        self._started = time.monotonic()
        try:
            task = self._require_task()
            evaluator = await asyncio.to_thread(self._build_evaluator, task)
            await asyncio.to_thread(self._run_sync, evaluator)
            await self._complete()
        except asyncio.CancelledError:
            raise
        except _SearchPaused:
            self.quant_store.update_task(
                self.task_id, status="PAUSED", phase="PAUSED", stop_requested=0
            )
        except _BudgetExhausted:
            await self._complete(reason="评价预算或运行时间已用完", exhausted=True)
        except Exception as error:
            self.quant_store.update_task(
                self.task_id,
                status="PAUSED_FAILURE",
                phase="FAILED",
                stop_requested=0,
                last_error=f"{type(error).__name__}: {error}",
            )
            self.quant_store.event(
                self.task_id,
                "audit",
                "QUANT_SEARCH_FAILED",
                "统计组合研究异常暂停",
                f"{type(error).__name__}: {error}",
                level="ERROR",
            )

    def _build_evaluator(self, task: dict[str, Any]) -> PriceVolumeEvaluator:
        config = task_research_config(
            ResearchConfig.from_toml(self.config_path), task["protocol"], task_id=self.task_id
        )
        evaluator = PriceVolumeEvaluator(Path(task["data_path"]), config=config)
        evaluator.set_trial_count(max(1, int(task["budget"]["maximum_evaluations"])))
        self._evaluator = evaluator
        return evaluator

    def _run_sync(self, evaluator: PriceVolumeEvaluator) -> None:
        self._screen_factors(evaluator)
        self._cluster_factors()
        leaders = self._searchable_factor_ids()
        task = self._require_task()
        if len(leaders) < int(task["construction"]["min_factors"]):
            raise RuntimeError("聚类后独立因子不足以满足最小因子数")
        mode = str(task["engine"]["mode"])
        maximum_evaluations = int(task["budget"]["maximum_evaluations"])
        finalization_reserve = min(
            int(task["construction"]["max_factors"]),
            max(0, maximum_evaluations - len(task["factor_snapshot"])),
        )
        search_evaluation_limit = maximum_evaluations - finalization_reserve
        sffs_limit, evolution_limit = _stage_evaluation_limits(
            mode,
            len(task["factor_snapshot"]),
            search_evaluation_limit,
        )
        self.quant_store.update_task(self.task_id, phase="SFFS")
        self._run_sffs(evaluator, leaders, evaluation_limit=sffs_limit)
        if mode in {"ENSEMBLE", "EVOLUTIONARY"}:
            self.quant_store.update_task(self.task_id, phase="EVOLUTION")
            self._run_evolution(evaluator, leaders, evaluation_limit=evolution_limit)
        if mode in {"ENSEMBLE", "BAYESIAN"}:
            self.quant_store.update_task(self.task_id, phase="ADAPTIVE")
            self._run_adaptive(
                evaluator,
                leaders,
                evaluation_limit=search_evaluation_limit,
            )
        self._refresh_pareto()
        self._qualify_best(evaluator)

    def _screen_factors(self, evaluator: PriceVolumeEvaluator) -> None:
        task = self._require_task()
        existing = {
            item["factor_id"]: item for item in self.quant_store.factor_screen(self.task_id)
        }
        records = {item["factor_id"]: item for item in task["factor_snapshot"]}
        self.quant_store.update_task(self.task_id, phase="SCREENING")
        for position, record in enumerate(task["factor_snapshot"], start=1):
            self._check_budget()
            factor_id = record["factor_id"]
            if factor_id in existing:
                path = existing[factor_id].get("return_artifact_path")
                if path and Path(path).is_file():
                    self._standalone_returns[factor_id] = load_return_artifact(path)["net_return"]
                    continue
            started = time.monotonic()
            evaluation = evaluator.evaluate_portfolio([factor_from_pool_record(record)])
            self._increment_evaluations()
            score = _standalone_stability_score(evaluation.metrics)
            artifact_path, artifact_hash = write_return_artifact(
                self.artifact_root / "quantcombine-screen",
                task_id=self.task_id,
                candidate_hash=factor_id,
                net_returns=evaluation.net_returns,
                active_returns=evaluation.net_returns
                - evaluator._market_benchmark_returns(evaluation.net_returns.index),
            )
            self._standalone_returns[factor_id] = evaluation.net_returns
            self.quant_store.upsert_factor_screen(
                self.task_id,
                {
                    "factor_id": factor_id,
                    "stability_score": score,
                    "metrics": evaluation.metrics,
                    "return_artifact_path": artifact_path,
                    "return_artifact_hash": artifact_hash,
                },
            )
            self.quant_store.event(
                self.task_id,
                "research",
                "FACTOR_SCREENED",
                f"单因子筛选 {position}/{len(records)}",
                f"{record['name']} · 稳定性 {score:.3f} · {time.monotonic() - started:.1f}s",
                payload={"factor_id": factor_id, "stability_score": score},
            )

    def _cluster_factors(self) -> None:
        task = self._require_task()
        screen = self.quant_store.factor_screen(self.task_id)
        records = {item["factor_id"]: item for item in task["factor_snapshot"]}
        threshold = float(task["engine"]["cluster_correlation_threshold"])
        correlations: dict[tuple[str, str], float] = {}
        aligned = pd.concat(self._standalone_returns, axis=1).dropna(how="all")
        residual_returns = aligned.sub(aligned.mean(axis=1), axis=0)

        def pair_correlation(left_id: str, right_id: str) -> float:
            key = tuple(sorted((left_id, right_id)))
            if key not in correlations:
                returns = pd.concat(
                    [residual_returns[left_id], residual_returns[right_id]], axis=1
                ).dropna()
                correlations[key] = (
                    abs(float(returns.corr().iloc[0, 1])) if len(returns) >= 20 else 0.0
                )
            return correlations[key]

        ordered = sorted(screen, key=lambda item: float(item["stability_score"]), reverse=True)
        screen_by_id = {item["factor_id"]: item for item in ordered}

        def is_similar(left_id: str, right_id: str) -> bool:
            search_id = records[left_id].get("search_cluster_id")
            same_search_cluster = bool(
                search_id and search_id == records[right_id].get("search_cluster_id")
            )
            if same_search_cluster:
                return True
            semantic_id = records[left_id].get("semantic_cluster_id")
            same_semantic = bool(
                semantic_id and semantic_id == records[right_id].get("semantic_cluster_id")
            )
            return same_semantic or pair_correlation(left_id, right_id) >= threshold

        # Common-mode residuals remove the shared long-only market path. Complete
        # linkage then prevents local A~B~C chains from collapsing an entire pool.
        group_ids = _complete_linkage_groups([item["factor_id"] for item in ordered], is_similar)
        minimum_coverage = float(task["objective"]["minimum_coverage"])
        for member_ids in group_ids:
            members = [screen_by_id[factor_id] for factor_id in member_ids]
            exclusion_reasons: dict[str, str | None] = {}
            for item in members:
                factor_id = item["factor_id"]
                if float(item["stability_score"]) < float(
                    task["engine"]["minimum_stability_score"]
                ):
                    exclusion_reasons[factor_id] = "LOW_STABILITY"
                elif (
                    _metric(
                        item["metrics"],
                        "portfolio_coverage",
                        "long_only_coverage",
                        "coverage",
                    )
                    < minimum_coverage
                ):
                    exclusion_reasons[factor_id] = "LOW_COVERAGE"
                else:
                    exclusion_reasons[factor_id] = None
            leader = next(
                (item for item in members if exclusion_reasons[item["factor_id"]] is None),
                None,
            )
            cluster_id = (
                "QC_"
                + hashlib.sha256(
                    ",".join(sorted(item["factor_id"] for item in members)).encode()
                ).hexdigest()[:10]
            )
            for item in members:
                factor_id = item["factor_id"]
                base_exclusion = exclusion_reasons[factor_id]
                self.quant_store.upsert_factor_screen(
                    self.task_id,
                    {
                        **item,
                        "cluster_id": cluster_id,
                        "cluster_leader": bool(
                            leader is not None and factor_id == leader["factor_id"]
                        ),
                        "exclusion_reason": (
                            base_exclusion
                            if base_exclusion
                            else None
                            if leader is not None and factor_id == leader["factor_id"]
                            else "CORRELATED_CLUSTER_MEMBER"
                        ),
                    },
                )
        self.quant_store.update_task(self.task_id, phase="CLUSTERING")
        self.quant_store.event(
            self.task_id,
            "research",
            "FACTOR_CLUSTERS_BUILT",
            "收益与语义聚类完成",
            f"{len(screen)} 个因子归入 {len(group_ids)} 个独立候选簇。",
            payload={"factor_count": len(screen), "cluster_count": len(group_ids)},
        )

    def _searchable_factor_ids(self) -> list[str]:
        task = self._require_task()
        required = {item["factor_id"] for item in task["factor_snapshot"] if item.get("required")}
        screen = self.quant_store.factor_screen(self.task_id)
        leaders = [
            item["factor_id"]
            for item in screen
            if item["cluster_leader"] and not item.get("exclusion_reason")
        ]
        ordered = list(required) + [value for value in leaders if value not in required]
        return list(dict.fromkeys(ordered))

    def _run_sffs(
        self, evaluator: PriceVolumeEvaluator, pool: list[str], *, evaluation_limit: int
    ) -> None:
        if self._evaluation_limit_reached(evaluation_limit):
            return
        task = self._require_task()
        construction = task["construction"]
        minimum = int(construction["min_factors"])
        maximum = min(int(construction["max_factors"]), len(pool))
        required = [item["factor_id"] for item in task["factor_snapshot"] if item.get("required")]
        screen_score = {
            item["factor_id"]: float(item["stability_score"])
            for item in self.quant_store.factor_screen(self.task_id)
        }
        seed = list(required)
        for factor_id in sorted(
            pool, key=lambda value: screen_score.get(value, -999), reverse=True
        ):
            if factor_id not in seed:
                seed.append(factor_id)
            if len(seed) >= minimum:
                break
        try:
            seed_candidate = self._evaluate_subset(
                evaluator,
                tuple(sorted(seed)),
                stage="SFFS",
                algorithm="STABILITY_SEED",
                action="SEED",
                evaluation_limit=evaluation_limit,
            )
        except _DuplicateCandidate:
            seed_candidate = self.quant_store.candidate_by_hash(
                self.task_id, canonical_hash(sorted(seed))
            )
            if seed_candidate is None:
                raise
        except _CandidateEvaluationRejected:
            seed_candidate = None
        beam = [seed_candidate]
        if seed_candidate is None:
            beam = []
        width = int(task["engine"]["sffs_beam_width"])
        while beam and max(len(item["factor_ids"]) for item in beam) < maximum:
            expanded: list[dict[str, Any]] = []
            for parent_candidate in beam:
                parent_ids = tuple(parent_candidate["factor_ids"])
                for factor_id in pool:
                    if factor_id in parent_ids:
                        continue
                    if self._evaluation_limit_reached(evaluation_limit):
                        return
                    try:
                        expanded.append(
                            self._evaluate_subset(
                                evaluator,
                                tuple(sorted((*parent_ids, factor_id))),
                                stage="SFFS",
                                algorithm="SFFS_FORWARD",
                                action="ADD",
                                parent_ids=[int(parent_candidate["id"])],
                                evaluation_limit=evaluation_limit,
                            )
                        )
                    except _DuplicateCandidate:
                        continue
                    except _CandidateEvaluationRejected:
                        continue
            if not expanded:
                break
            beam = sorted(expanded, key=_candidate_key, reverse=True)[:width]
            current = beam[0]
            improved = True
            while improved and len(current["factor_ids"]) > minimum:
                improved = False
                removals: list[dict[str, Any]] = []
                for factor_id in current["factor_ids"]:
                    if factor_id in required:
                        continue
                    if self._evaluation_limit_reached(evaluation_limit):
                        return
                    subset = tuple(value for value in current["factor_ids"] if value != factor_id)
                    try:
                        removals.append(
                            self._evaluate_subset(
                                evaluator,
                                tuple(sorted(subset)),
                                stage="SFFS",
                                algorithm="SFFS_BACKWARD",
                                action="REMOVE",
                                parent_ids=[int(current["id"])],
                                evaluation_limit=evaluation_limit,
                            )
                        )
                    except _DuplicateCandidate:
                        continue
                    except _CandidateEvaluationRejected:
                        continue
                if removals:
                    best_removal = max(removals, key=_candidate_key)
                    if _candidate_key(best_removal) > _candidate_key(current):
                        current = best_removal
                        improved = True
            beam = sorted([*beam, current], key=_candidate_key, reverse=True)[:width]

    def _run_evolution(
        self, evaluator: PriceVolumeEvaluator, pool: list[str], *, evaluation_limit: int
    ) -> None:
        task = self._require_task()
        rng = np.random.default_rng(int(task["engine"]["random_seed"]))
        population_size = int(task["engine"]["evolution_population"])
        minimum = int(task["construction"]["min_factors"])
        maximum = min(int(task["construction"]["max_factors"]), len(pool))
        required = {item["factor_id"] for item in task["factor_snapshot"] if item.get("required")}
        for generation in range(int(task["engine"]["evolution_generations"])):
            archive = sorted(
                self.quant_store.candidates(self.task_id), key=_candidate_key, reverse=True
            )
            parents = archive[: max(2, population_size)]
            proposals: list[tuple[tuple[str, ...], list[int], str]] = []
            while len(proposals) < population_size * 3:
                if len(parents) >= 2 and rng.random() < 0.45:
                    chosen = rng.choice(len(parents), size=2, replace=False)
                    left, right = parents[int(chosen[0])], parents[int(chosen[1])]
                    union = list(dict.fromkeys([*left["factor_ids"], *right["factor_ids"]]))
                    size = int(rng.integers(minimum, min(maximum, len(union)) + 1))
                    subset = set(rng.choice(union, size=size, replace=False).tolist()) | required
                    parent_ids = [int(left["id"]), int(right["id"])]
                    action = "CROSSOVER"
                else:
                    parent = parents[int(rng.integers(0, len(parents)))]
                    subset = set(parent["factor_ids"])
                    action_choice = str(rng.choice(["ADD", "REMOVE", "REPLACE"]))
                    if action_choice == "ADD" and len(subset) < maximum:
                        available = [value for value in pool if value not in subset]
                        if available:
                            subset.add(str(rng.choice(available)))
                    elif action_choice == "REMOVE" and len(subset) > minimum:
                        removable = [value for value in subset if value not in required]
                        if removable:
                            subset.remove(str(rng.choice(removable)))
                    else:
                        removable = [value for value in subset if value not in required]
                        available = [value for value in pool if value not in subset]
                        if removable and available:
                            subset.remove(str(rng.choice(removable)))
                            subset.add(str(rng.choice(available)))
                    parent_ids = [int(parent["id"])]
                    action = f"MUTATE_{action_choice}"
                if minimum <= len(subset) <= maximum:
                    proposals.append((tuple(sorted(subset)), parent_ids, action))
                if len({item[0] for item in proposals}) >= population_size:
                    break
            evaluated = 0
            rejected = 0
            for subset, parent_ids, action in proposals:
                if evaluated >= population_size:
                    break
                if self._evaluation_limit_reached(evaluation_limit):
                    return
                try:
                    self._evaluate_subset(
                        evaluator,
                        subset,
                        stage="EVOLUTION",
                        algorithm="NSGA2",
                        action=action,
                        parent_ids=parent_ids,
                        evaluation_limit=evaluation_limit,
                    )
                    evaluated += 1
                except _DuplicateCandidate:
                    continue
                except _CandidateEvaluationRejected:
                    rejected += 1
                    continue
            self._refresh_pareto()
            self.quant_store.event(
                self.task_id,
                "research",
                "NSGA2_GENERATION_COMPLETED",
                f"NSGA-II 第 {generation + 1} 代",
                f"新增 {evaluated} 个可审计组合，跳过 {rejected} 个不可评估候选。",
                payload={
                    "generation": generation + 1,
                    "evaluated": evaluated,
                    "candidate_level_rejected": rejected,
                },
            )

    def _run_adaptive(
        self,
        evaluator: PriceVolumeEvaluator,
        pool: list[str],
        *,
        evaluation_limit: int,
    ) -> None:
        task = self._require_task()
        rng = np.random.default_rng(int(task["engine"]["random_seed"]) + 7919)
        minimum = int(task["construction"]["min_factors"])
        maximum = min(int(task["construction"]["max_factors"]), len(pool))
        required = {item["factor_id"] for item in task["factor_snapshot"] if item.get("required")}
        for trial in range(int(task["engine"]["adaptive_trials"])):
            if self._evaluation_limit_reached(evaluation_limit):
                return
            archive = self.quant_store.candidates(self.task_id)
            if not archive:
                return
            scores = np.array([float(item["score"]) for item in archive], dtype=float)
            centered = (scores - scores.mean()) / max(scores.std(), 1e-8)
            utility = {factor_id: 0.0 for factor_id in pool}
            counts = {factor_id: 1.0 for factor_id in pool}
            for candidate, score in zip(archive, centered, strict=True):
                for factor_id in candidate["factor_ids"]:
                    if factor_id in utility:
                        utility[factor_id] += float(score)
                        counts[factor_id] += 1
            logits = np.array([utility[value] / counts[value] for value in pool])
            probabilities = np.exp(logits - logits.max()) + 0.15
            probabilities = probabilities / probabilities.sum()
            size = int(rng.integers(minimum, maximum + 1))
            optional_size = max(0, size - len(required))
            optional_pool = [value for value in pool if value not in required]
            optional_probabilities = np.array(
                [probabilities[pool.index(value)] for value in optional_pool], dtype=float
            )
            optional_probabilities /= optional_probabilities.sum()
            selected = set(required)
            if optional_size:
                selected.update(
                    rng.choice(
                        optional_pool,
                        size=min(optional_size, len(optional_pool)),
                        replace=False,
                        p=optional_probabilities,
                    ).tolist()
                )
            try:
                self._evaluate_subset(
                    evaluator,
                    tuple(sorted(selected)),
                    stage="ADAPTIVE",
                    algorithm="BAYESIAN_INCLUSION",
                    action="POSTERIOR_SAMPLE",
                    parent_ids=[],
                    evaluation_limit=evaluation_limit,
                )
            except _DuplicateCandidate:
                continue
            except _CandidateEvaluationRejected:
                continue
            if trial % 4 == 3:
                self._refresh_pareto()

    def _evaluate_subset(
        self,
        evaluator: PriceVolumeEvaluator,
        factor_ids: tuple[str, ...],
        *,
        stage: str,
        algorithm: str,
        action: str,
        parent_ids: list[int] | None = None,
        evaluation_limit: int | None = None,
    ) -> dict[str, Any]:
        self._check_budget()
        existing = self.quant_store.candidate_by_hash(
            self.task_id, canonical_hash(sorted(factor_ids))
        )
        if existing is not None:
            raise _DuplicateCandidate
        task = self._require_task()
        records = {item["factor_id"]: item for item in task["factor_snapshot"]}
        homogeneity_violation = _homogeneity_limit_violation(
            factor_ids, records, task["construction"]
        )
        if homogeneity_violation is not None:
            self.quant_store.event(
                self.task_id,
                "audit",
                "QUANT_HOMOGENEITY_CANDIDATE_REJECTED",
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
        factors = [factor_from_pool_record(records[value]) for value in factor_ids]
        started = time.monotonic()
        best: (
            tuple[tuple[float, ...], tuple[float, ...], Any, dict[str, Any], list[str], float]
            | None
        ) = None
        rejected_reasons: list[str] = []
        for weights in self._weight_candidates(factor_ids):
            self._check_budget()
            if evaluation_limit is not None and self._evaluation_limit_reached(evaluation_limit):
                break
            try:
                evaluation = evaluator.evaluate_portfolio(factors, weights=weights)
            except (ValueError, FloatingPointError, ArithmeticError) as error:
                reason = _recoverable_evaluation_failure_reason(error)
                if reason is None:
                    raise
                rejected_reasons.append(reason)
                self.quant_store.event(
                    self.task_id,
                    "audit",
                    "QUANT_WEIGHT_CANDIDATE_REJECTED",
                    "权重候选评估失败，已作为候选级问题跳过",
                    f"{reason}: {error}",
                    level="WARN",
                    payload={
                        "factor_ids": list(factor_ids),
                        "weights": list(weights),
                        "failure_class": reason,
                    },
                )
                continue
            self._increment_evaluations()
            metrics = dict(evaluation.metrics)
            metrics.update(mechanism_independence_metrics(records, list(factor_ids), list(weights)))
            active = evaluation.net_returns - evaluator._market_benchmark_returns(
                evaluation.net_returns.index
            )
            metrics.update(self._strategy_independence_metrics(active))
            failures = _gate_failures(metrics, task["objective"])
            distance = _gate_distance(metrics, task["objective"])
            score = _portfolio_score(metrics, failures, task["objective"])
            rank = (float(not failures), -float(len(failures)), -distance, score)
            if best is None or rank > best[0]:
                best = (rank, weights, evaluation, metrics, failures, score)
        if best is None:
            if rejected_reasons:
                raise _CandidateEvaluationRejected(
                    f"All weight candidates rejected: {sorted(set(rejected_reasons))}"
                )
            raise _BudgetExhausted
        _, weights, evaluation, metrics, failures, score = best
        candidate_hash = canonical_hash(sorted(factor_ids))
        active = evaluation.net_returns - evaluator._market_benchmark_returns(
            evaluation.net_returns.index
        )
        artifact_path, artifact_hash = write_return_artifact(
            self.artifact_root / "quantcombine",
            task_id=self.task_id,
            candidate_hash=_candidate_hash(factor_ids, weights),
            net_returns=evaluation.net_returns,
            active_returns=active,
        )
        iteration = int(task["iteration"]) + 1
        objectives = _objective_vector(metrics)
        candidate = self.quant_store.record_candidate(
            self.task_id,
            {
                "iteration": iteration,
                "stage": stage,
                "algorithm": algorithm,
                "action": action,
                "candidate_hash": candidate_hash,
                "parent_ids": parent_ids or [],
                "factor_ids": list(factor_ids),
                "weights": list(weights),
                "metrics": metrics,
                "objectives": objectives,
                "score": score,
                "gate_distance": _gate_distance(metrics, task["objective"]),
                "gate_status": "PASSED" if not failures else "REJECTED",
                "failed_gates": failures,
                "return_artifact_path": artifact_path,
                "return_artifact_hash": artifact_hash,
                "duration_seconds": time.monotonic() - started,
            },
        )
        self.quant_store.update_task(self.task_id, iteration=iteration)
        self._update_leaders(candidate)
        self.quant_store.event(
            self.task_id,
            "research",
            "QUANT_CANDIDATE_EVALUATED",
            f"候选 #{iteration} · {algorithm}",
            f"{len(factor_ids)} 因子 · 得分 {score:.3f} · 门禁 {len(failures)} 项失败",
            level="INFO" if not failures else "WARN",
            payload={
                "candidate_id": candidate["id"],
                "factor_ids": list(factor_ids),
                "weights": list(weights),
                "failed_gates": failures,
            },
        )
        return candidate

    def _weight_candidates(self, factor_ids: tuple[str, ...]) -> list[tuple[float, ...]]:
        task = self._require_task()
        construction = task["construction"]
        minimum = float(construction["minimum_weight"])
        maximum = float(construction["maximum_weight"])
        step = float(construction["weight_step"])
        n = len(factor_ids)
        equal = np.full(n, 1 / n)
        aligned = pd.concat(
            [self._standalone_returns[value] for value in factor_ids], axis=1
        ).dropna()
        matrix = aligned.to_numpy(dtype=float)
        annual_mean = np.nanmean(matrix, axis=0) * 245 if len(matrix) else np.zeros(n)
        sample_cov = np.cov(matrix, rowvar=False) * 245 if len(matrix) > 2 else np.eye(n)
        sample_cov = np.atleast_2d(sample_cov)
        diagonal = np.diag(np.diag(sample_cov))
        shrinkage = float(task["engine"]["covariance_shrinkage"])
        covariance = (1 - shrinkage) * sample_cov + shrinkage * diagonal + np.eye(n) * 1e-8
        volatility = np.sqrt(np.maximum(np.diag(covariance), 1e-10))
        bounds = [(minimum, maximum)] * n
        constraints = {"type": "eq", "fun": lambda value: float(np.sum(value) - 1)}
        regularization = float(task["engine"]["weight_regularization"])

        def solve(objective: Any) -> np.ndarray | None:
            result = minimize(
                objective,
                equal,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 120, "ftol": 1e-9},
            )
            return np.asarray(result.x, dtype=float) if result.success else None

        raw: list[np.ndarray] = [equal, (1 / volatility) / np.sum(1 / volatility)]
        methods = [
            lambda weight: (
                -(weight @ annual_mean) / max(math.sqrt(float(weight @ covariance @ weight)), 1e-8)
                + regularization * float(np.sum((weight - equal) ** 2))
            ),
            lambda weight: (
                float(weight @ covariance @ weight)
                + regularization * float(np.sum((weight - equal) ** 2))
            ),
            lambda weight: (
                -float(weight @ volatility)
                / max(math.sqrt(float(weight @ covariance @ weight)), 1e-8)
                + regularization * float(np.sum((weight - equal) ** 2))
            ),
        ]
        if len(matrix):
            methods.append(
                lambda weight: (
                    float(np.mean(np.sort(-(matrix @ weight))[-max(1, int(len(matrix) * 0.10)) :]))
                    + regularization * float(np.sum((weight - equal) ** 2))
                )
            )
        for method in methods:
            result = solve(method)
            if result is not None:
                raw.append(result)
        scores = np.array(
            [
                next(
                    item["stability_score"]
                    for item in self.quant_store.factor_screen(self.task_id)
                    if item["factor_id"] == factor_id
                )
                for factor_id in factor_ids
            ]
        )
        softmax = np.exp(scores - scores.max())
        raw.append(softmax / softmax.sum())
        seed = int(canonical_hash({"task": self.task_id, "factors": factor_ids})[:16], 16)
        rng = np.random.default_rng(seed)
        limit = int(task["budget"]["weight_candidates_per_subset"])
        while len(raw) < limit * 2:
            raw.append(rng.dirichlet(np.ones(n) * 1.8))
        candidates: list[tuple[float, ...]] = []
        for values in raw:
            projected = _bounded_simplex(values, minimum, maximum)
            rounded = np.round(projected / step) * step
            rounded = _repair_weight_sum(rounded, minimum, maximum, step)
            if np.any(rounded < minimum - 1e-8) or np.any(rounded > maximum + 1e-8):
                continue
            candidate = tuple(round(float(value), 6) for value in rounded)
            if candidate not in candidates:
                candidates.append(candidate)
            if len(candidates) >= limit:
                break
        return candidates

    def _strategy_independence_metrics(self, active_returns: pd.Series) -> dict[str, Any]:
        maximum = 0.0
        nearest: str | None = None
        nearest_kind: str | None = None
        nearest_observations = 0
        nearest_spearman = 0.0
        comparisons_detail: list[dict[str, Any]] = []
        task = self._require_task()
        minimum_observations = int(
            task["engine"].get("strategy_reference_minimum_observations", 120)
        )
        references: list[tuple[str, str, str]] = []
        for candidate in self.quant_store.best_candidate_references(
            exclude_task_id=self.task_id,
            limit=int(task["engine"].get("strategy_reference_limit", 25)),
        ):
            path = candidate.get("return_artifact_path")
            if path:
                references.append((str(candidate["task_id"]), "QUANT_TASK_BEST", path))
        for strategy in self.quant_store.strategies():
            path = (
                strategy["specification"]
                .get("evaluation", {})
                .get("quantcombine_return_artifact_path")
            )
            if path:
                references.append((strategy["strategy_id"], "QUANT_STRATEGY", path))
        for strategy in self._auto_store.strategies():
            path = (
                strategy["specification"]
                .get("evaluation", {})
                .get("autocombine_return_artifact_path")
            )
            if path:
                references.append((strategy["strategy_id"], "AUTOCOMBINE_STRATEGY", path))
        comparisons = 0
        seen: set[tuple[str, str]] = set()
        for strategy_id, reference_kind, path in references:
            key = (strategy_id, path)
            if key in seen:
                continue
            seen.add(key)
            if not Path(path).is_file():
                continue
            comparison = return_independence(
                active_returns, load_return_artifact(path)["active_return"]
            )
            if int(comparison["observations"]) < minimum_observations:
                continue
            comparisons += 1
            correlation = abs(float(comparison["pearson"]))
            detail = {
                "reference_id": strategy_id,
                "reference_kind": reference_kind,
                "pearson": float(comparison["pearson"]),
                "spearman": float(comparison["spearman"]),
                "absolute_pearson": correlation,
                "observations": int(comparison["observations"]),
            }
            comparisons_detail.append(detail)
            if correlation > maximum:
                maximum, nearest = correlation, strategy_id
                nearest_kind = reference_kind
                nearest_observations = int(comparison["observations"])
                nearest_spearman = float(comparison["spearman"])
        comparisons_detail.sort(key=lambda item: item["absolute_pearson"], reverse=True)
        return {
            "portfolio_max_strategy_active_correlation": maximum,
            "portfolio_nearest_strategy_id": nearest,
            "portfolio_nearest_strategy_kind": nearest_kind,
            "portfolio_nearest_strategy_observations": nearest_observations,
            "portfolio_nearest_strategy_spearman": nearest_spearman,
            "portfolio_strategy_independence_comparisons": comparisons,
            "portfolio_strategy_correlation_top": comparisons_detail[:5],
            "portfolio_strategy_reference_scope": "QUANT_TASK_BEST_AND_STRATEGY_V1",
            "portfolio_strategy_reference_minimum_observations": minimum_observations,
        }

    def _update_leaders(self, candidate: dict[str, Any]) -> None:
        task = self._require_task()
        leader = (
            self.quant_store.candidate(int(task["best_candidate_id"]))
            if task.get("best_candidate_id")
            else None
        )
        qualified = (
            self.quant_store.candidate(int(task["qualified_candidate_id"]))
            if task.get("qualified_candidate_id")
            else None
        )
        updates: dict[str, Any] = {}
        if leader is None or _candidate_key(candidate) > _candidate_key(leader):
            if leader:
                self.quant_store.update_candidate(int(leader["id"]), qualification="EVALUATED")
            updates["best_candidate_id"] = candidate["id"]
            self.quant_store.update_candidate(int(candidate["id"]), qualification="RESEARCH_LEADER")
        if candidate["gate_status"] == "PASSED" and (
            qualified is None or _candidate_key(candidate) > _candidate_key(qualified)
        ):
            if qualified:
                self.quant_store.update_candidate(int(qualified["id"]), qualification="QUALIFIED")
            updates["qualified_candidate_id"] = candidate["id"]
            updates["qualification_status"] = "QUALIFIED_CHAMPION"
            self.quant_store.update_candidate(
                int(candidate["id"]), qualification="QUALIFIED_CHAMPION"
            )
        elif qualified is None:
            updates["qualification_status"] = "RESEARCH_LEADER_ONLY"
        if updates:
            self.quant_store.update_task(self.task_id, **updates)

    def _refresh_pareto(self) -> None:
        candidates = self.quant_store.candidates(self.task_id)
        for candidate_id, (rank, crowding) in pareto_ranks(candidates).items():
            self.quant_store.update_candidate(
                candidate_id, pareto_rank=rank, crowding_distance=crowding
            )

    def _qualify_best(self, evaluator: PriceVolumeEvaluator) -> None:
        task = self._require_task()
        candidate_id = task.get("qualified_candidate_id") or task.get("best_candidate_id")
        if not candidate_id:
            return
        candidate = self.quant_store.candidate(int(candidate_id))
        assert candidate is not None
        records = {item["factor_id"]: item for item in task["factor_snapshot"]}
        base_score = _portfolio_score(candidate["metrics"], [], task["objective"])
        diagnostics = []
        for index, factor_id in enumerate(candidate["factor_ids"]):
            self._check_budget()
            remaining_ids = [value for value in candidate["factor_ids"] if value != factor_id]
            if not remaining_ids:
                continue
            remaining_weights = np.delete(np.asarray(candidate["weights"], dtype=float), index)
            remaining_weights /= remaining_weights.sum()
            evaluation = evaluator.evaluate_portfolio(
                [factor_from_pool_record(records[value]) for value in remaining_ids],
                weights=tuple(float(value) for value in remaining_weights),
            )
            self._increment_evaluations()
            metrics = dict(evaluation.metrics)
            metrics.update(
                mechanism_independence_metrics(records, remaining_ids, list(remaining_weights))
            )
            score_delta = base_score - _portfolio_score(metrics, [], task["objective"])
            diagnostics.append(
                {
                    "factor_id": factor_id,
                    "objective_score_delta": score_delta,
                    "positive": score_delta > 0.01,
                }
            )
        positive_fraction = (
            sum(item["positive"] for item in diagnostics) / len(diagnostics) if diagnostics else 1.0
        )
        metrics = {
            **candidate["metrics"],
            "portfolio_leave_one_out": diagnostics,
            "portfolio_marginal_positive_fraction": positive_fraction,
            "portfolio_redundant_factor_count": sum(not item["positive"] for item in diagnostics),
            "portfolio_redundant_factor_ids": [
                item["factor_id"] for item in diagnostics if not item["positive"]
            ],
            "quantcombine_return_artifact_path": candidate["return_artifact_path"],
            "quantcombine_return_artifact_hash": candidate["return_artifact_hash"],
            "quantcombine_engine_mode": task["engine"]["mode"],
            "quantcombine_hidden_metrics_exposed": False,
        }
        failures = _gate_failures(metrics, task["objective"])
        updated = self.quant_store.update_candidate(
            int(candidate["id"]),
            metrics=metrics,
            score=_portfolio_score(metrics, failures, task["objective"]),
            gate_distance=_gate_distance(metrics, task["objective"]),
            gate_status="PASSED" if not failures else "REJECTED",
            failed_gates=failures,
            qualification="QUALIFIED_CHAMPION" if not failures else "RESEARCH_LEADER",
        )
        if not failures:
            self.quant_store.update_task(
                self.task_id,
                qualified_candidate_id=updated["id"],
                qualification_status="QUALIFIED_CHAMPION",
            )
        else:
            self.quant_store.update_task(
                self.task_id,
                qualified_candidate_id=None,
                qualification_status="RESEARCH_LEADER_ONLY",
            )

    async def _complete(
        self, reason: str = "全部算法阶段已完成", *, exhausted: bool = False
    ) -> None:
        task = self._require_task()
        qualified = (
            self.quant_store.candidate(int(task["qualified_candidate_id"]))
            if task.get("qualified_candidate_id")
            else None
        )
        status = "EXHAUSTED" if exhausted else "RESEARCH_COMPLETED"
        if qualified and qualified["gate_status"] == "PASSED":
            self.quant_store.update_task(self.task_id, phase="BLIND_REVIEW")
            records = {item["factor_id"]: item for item in task["factor_snapshot"]}
            evaluator = self._evaluator or await asyncio.to_thread(self._build_evaluator, task)
            verdict = await asyncio.to_thread(
                BlindEvaluationBoundary(evaluator.panel_path, evaluator.config).evaluate_holdout,
                [factor_from_pool_record(records[value]) for value in qualified["factor_ids"]],
                tuple(float(value) for value in qualified["weights"]),
            )
            self.quant_store.update_task(
                self.task_id,
                blind_verdict=verdict.verdict,
                blind_evidence_hash=verdict.evidence_hash,
                production_candidate_id=qualified["id"] if verdict.passed else None,
                qualification_status=(
                    "PRODUCTION_CANDIDATE" if verdict.passed else "BLIND_REJECTED"
                ),
            )
            status = "COMPLETED" if verdict.passed else "BLIND_REJECTED"
        self.quant_store.update_task(
            self.task_id, status=status, phase="DELIVERY", stop_requested=0
        )
        self.quant_store.event(
            self.task_id,
            "delivery",
            "QUANT_SEARCH_COMPLETED",
            "统计组合任务已封存",
            reason,
            payload={"status": status, "evaluation_count": task["evaluation_count"]},
        )

    def _check_budget(self) -> None:
        task = self._require_task()
        if task["stop_requested"]:
            raise _SearchPaused
        if int(task["evaluation_count"]) >= int(task["budget"]["maximum_evaluations"]):
            raise _BudgetExhausted
        elapsed = (time.monotonic() - self._started) / 60 if self._started > 0 else 0.0
        if elapsed >= float(task["budget"]["maximum_runtime_minutes"]):
            raise _BudgetExhausted

    def _evaluation_limit_reached(self, limit: int) -> bool:
        return int(self._require_task()["evaluation_count"]) >= limit

    def _increment_evaluations(self) -> None:
        task = self._require_task()
        self.quant_store.update_task(
            self.task_id, evaluation_count=int(task["evaluation_count"]) + 1
        )

    def _require_task(self) -> dict[str, Any]:
        task = self.quant_store.task(self.task_id)
        if task is None:
            raise KeyError(f"QuantCombine task not found: {self.task_id}")
        return task


class QuantCombineManager:
    def __init__(
        self,
        store: ServiceStore,
        quant_store: QuantCombineStore,
        *,
        config_path: Path,
        maximum_concurrent_tasks: int = 1,
    ) -> None:
        self.store = store
        self.quant_store = quant_store
        self.config_path = config_path
        self.maximum_concurrent_tasks = maximum_concurrent_tasks
        self._workers: dict[str, QuantCombineWorker] = {}

    def worker(self, task_id: str) -> QuantCombineWorker:
        if task_id not in self._workers:
            self._workers[task_id] = QuantCombineWorker(
                task_id, self.store, self.quant_store, config_path=self.config_path
            )
        return self._workers[task_id]

    async def start(self, task_id: str) -> dict[str, Any]:
        worker = self.worker(task_id)
        active = sum(item.alive for item in self._workers.values())
        if not worker.alive and active >= self.maximum_concurrent_tasks:
            raise RuntimeError("QuantCombine 并发任务已达到上限")
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
            *(worker.shutdown() for worker in self._workers.values()), return_exceptions=True
        )


class _SearchPaused(Exception):
    pass


class _BudgetExhausted(Exception):
    pass


class _CandidateEvaluationRejected(Exception):
    pass


class _DuplicateCandidate(Exception):
    pass


def _recoverable_evaluation_failure_reason(error: Exception) -> str | None:
    message = str(error).lower()
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
    if "database" in message or "locked" in message or "no parquet" in message:
        return None
    for marker, reason in markers.items():
        if marker in message:
            return reason
    return None
