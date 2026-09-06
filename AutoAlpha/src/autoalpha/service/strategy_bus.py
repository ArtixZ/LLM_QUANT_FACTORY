from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from autoalpha.operations.artifacts import ArtifactRegistry
from autoalpha.service.autocombine_store import AutoCombineStore
from autoalpha.service.external_jobs import external_job_id
from autoalpha.service.factor_library import build_factor_library
from autoalpha.service.mechanism import normalize_mechanism
from autoalpha.service.quantcombine_store import QuantCombineStore
from autoalpha.service.store import ServiceStore

ALLOWED_INITIAL_STRATEGY_LIFECYCLES = {"RESEARCH"}
STRATEGY_LIFECYCLE_TRANSITIONS = {
    "RESEARCH": {"FROZEN"},
    "FROZEN": {"HIDDEN_HOLDOUT"},
    "HIDDEN_HOLDOUT": {"SHADOW"},
    "SHADOW": {"PAPER"},
    "PAPER": {"PRODUCTION_CANDIDATE"},
    "PRODUCTION_CANDIDATE": set(),
}
STRATEGY_TRANSITION_REQUIRED_EVIDENCE = {
    "FROZEN": {"source_experiment_id", "factor_ids", "weights", "public_validation_passed"},
    "HIDDEN_HOLDOUT": {"frozen_specification_hash", "holdout_evaluation_requested"},
    "SHADOW": {"hidden_holdout_passed", "holdout_evaluation_id"},
    "PAPER": {"shadow_trading_days", "shadow_execution_passed"},
    "PRODUCTION_CANDIDATE": {"paper_trading_days", "paper_trading_passed", "risk_approval"},
}
STRATEGY_TRANSITION_LABELS = {
    "FROZEN": "冻结公开验证通过的研究版本",
    "HIDDEN_HOLDOUT": "提交隔离隐藏盲测",
    "SHADOW": "进入影子交易",
    "PAPER": "进入模拟交易",
    "PRODUCTION_CANDIDATE": "登记为生产候选",
}

LONG_ONLY_PRIMARY_METRICS = (
    "long_only_sharpe_ratio",
    "long_only_simple_annual_return",
    "long_only_max_drawdown",
    "long_only_walk_forward_worst_sharpe",
    "long_only_annual_turnover",
    "long_only_capacity_usd",
    "recent_long_only_sharpe_ratio",
    "recent_long_only_simple_annual_return",
    "recent_long_only_max_drawdown",
    "recent_long_only_walk_forward_worst_sharpe",
)


def stable_experiment_id(source_system: str, source_id: str, stage: str) -> str:
    digest = hashlib.sha256(f"{source_system}|{source_id}|{stage}".encode()).hexdigest()[:16]
    return f"EXP_{digest}"


def build_strategy_bus_snapshot(
    store: ServiceStore,
    *,
    autocombine_store: AutoCombineStore,
    quantcombine_store: QuantCombineStore,
    behavior_snapshot: dict[str, Any] | None = None,
    sync: bool = True,
) -> dict[str, Any]:
    if sync:
        sync_strategy_bus(
            store,
            autocombine_store,
            quantcombine_store,
            behavior_snapshot=behavior_snapshot,
        )
    return {
        "summary": store.strategy_experiment_summary(),
        "objects": store.strategy_experiment_objects(limit=5000),
        "edges": store.strategy_experiment_edges(limit=20_000),
        "favorites": {
            "experiments": sorted(store.favorite_ids("strategy_experiment")),
            "strategies": sorted(store.favorite_ids("strategy_version")),
        },
        "protocol": {
            "object_model": (
                "FACTOR_CANDIDATE -> FACTOR_CLUSTER -> COMBINATION_CANDIDATE -> "
                "STRATEGY_VERSION -> PAPER_PORTFOLIO -> PRODUCTION_CANDIDATE"
            ),
            "primary_metric_convention": "US_EQUITY_LONG_ONLY_WEEKLY_NON_PIT_PROXY",
            "diagnostic_metric_policy": "long_short_ic_is_diagnostic_only",
        },
    }


def sync_strategy_bus(
    store: ServiceStore,
    autocombine_store: AutoCombineStore,
    quantcombine_store: QuantCombineStore,
    *,
    behavior_snapshot: dict[str, Any] | None = None,
) -> None:
    factors = store.factor_pool(limit=5000)
    factor_library = build_factor_library(
        factors,
        lifecycle_states=store.factor_lifecycle_states(),
        contaminated_factor_ids=store.contaminated_factor_ids(),
        research_diagnostics=store.factor_research_diagnostics(),
    )
    _merge_raw_factor_evidence(
        factor_library["factors"],
        factors,
        behavior_snapshot=behavior_snapshot,
    )
    factor_nodes = _sync_factor_candidates(store, factor_library["factors"])
    _sync_factor_clusters(store, factor_library["factors"], factor_nodes)
    _sync_auto_combine(store, autocombine_store, factor_nodes)
    _sync_quant_combine(store, quantcombine_store, factor_nodes)
    _sync_formal_strategy_versions(store, factor_nodes)
    _sync_paper_portfolios(store, factor_nodes)


def factor_knowledge_map(
    store: ServiceStore, *, behavior_snapshot: dict[str, Any] | None = None
) -> dict[str, Any]:
    factors = store.factor_pool(limit=5000)
    library = build_factor_library(
        factors,
        lifecycle_states=store.factor_lifecycle_states(),
        contaminated_factor_ids=store.contaminated_factor_ids(),
        research_diagnostics=store.factor_research_diagnostics(),
    )
    _merge_raw_factor_evidence(
        library["factors"],
        factors,
        behavior_snapshot=behavior_snapshot,
    )
    by_cluster: dict[str, list[dict[str, Any]]] = {}
    stale_or_failed: dict[str, list[str]] = {}
    for factor in library["factors"]:
        cluster = str(
            factor.get("behavior_cluster_id")
            or factor.get("similarity_cluster_id")
            or factor.get("mechanism_type")
            or "UNCLUSTERED"
        )
        by_cluster.setdefault(cluster, []).append(factor)
        failures = list(factor.get("production_promotion_gate_failures") or [])
        if failures:
            stale_or_failed[str(factor["factor_id"])] = failures
    clusters = []
    for cluster_id, members in by_cluster.items():
        leader = max(members, key=_long_only_score)
        clusters.append(
            {
                "cluster_id": cluster_id,
                "size": len(members),
                "leader_factor_id": leader["factor_id"],
                "leader_name": leader["name"],
                "mechanisms": sorted(
                    {
                        normalize_mechanism(item.get("mechanism_type") or item["family"])
                        for item in members
                    }
                ),
                "average_long_only_score": _average(
                    _long_only_score(item) for item in members
                ),
                "top_factors": [
                    {
                        "factor_id": item["factor_id"],
                        "name": item["name"],
                        "score": _long_only_score(item),
                        "status": item.get("status"),
                    }
                    for item in sorted(
                        members,
                        key=_long_only_score,
                        reverse=True,
                    )[:8]
                ],
            }
        )
    homogeneity_groups = _knowledge_homogeneity_groups(library["factors"])
    mechanism_map = _knowledge_mechanism_map(library["factors"])
    parameter_families = _knowledge_parameter_families(library["factors"])
    annual_heatmap = _knowledge_annual_heatmap(library["factors"])
    return {
        "research_map_protocol": "FACTOR_KNOWLEDGE_RESEARCH_MAP_V2",
        "cluster_count": len(clusters),
        "factor_count": len(library["factors"]),
        "clusters": sorted(
            clusters, key=lambda item: item["average_long_only_score"], reverse=True
        ),
        "homogeneity_fold_groups": homogeneity_groups,
        "mechanism_map": mechanism_map,
        "parameter_families": parameter_families,
        "annual_heatmap": annual_heatmap,
        "failure_tags": stale_or_failed,
        "questions_answered": [
            "belongs_to_return_source",
            "near_duplicates",
            "failure_modes",
            "combination_fit",
            "parameter_or_expression_clone_risk",
        ],
        "primary_metric_policy": "long_only_first",
    }


def formal_strategy_library(store: ServiceStore) -> dict[str, Any]:
    strategies = store.formal_strategy_versions(limit=1000)
    favorite = store.favorite_ids("strategy_version")
    for item in strategies:
        key = f"{item['strategy_uid']}@{item['version']}"
        item["favorite"] = key in favorite
        item["lifecycle_readiness"] = strategy_lifecycle_readiness(
            store, item["strategy_uid"], item["version"]
        )
        item["production_evidence_summary"] = _formal_strategy_evidence_summary(item)
    promotion_candidates = strategy_promotion_candidates(store, limit=50)
    return {
        "strategies": strategies,
        "count": len(strategies),
        "promotion_candidates": promotion_candidates,
        "promotion_candidate_count": len(promotion_candidates),
        "readiness": {
            "formal_strategy_versions": len(strategies),
            "promotion_candidates": len(promotion_candidates),
            "status": (
                "STRATEGY_LIBRARY_EMPTY_WITH_CANDIDATES"
                if not strategies and promotion_candidates
                else "READY"
            ),
        },
        "required_policy_blocks": [
            "signal_policy",
            "rebalance_policy",
            "execution_policy",
            "risk_policy",
            "cost_policy",
            "monitoring_policy",
        ],
        "production_funnel": strategy_production_funnel(store),
    }


def strategy_experiment_lineage(
    store: ServiceStore,
    experiment_id: str,
    *,
    depth: int = 2,
) -> dict[str, Any]:
    center = store.strategy_experiment_object(experiment_id)
    if center is None:
        raise KeyError(f"Strategy experiment not found: {experiment_id}")
    max_depth = max(1, min(int(depth), 4))
    all_edges = store.strategy_experiment_edges(limit=20_000)
    edges_by_source: dict[str, list[dict[str, Any]]] = {}
    edges_by_target: dict[str, list[dict[str, Any]]] = {}
    for edge in all_edges:
        edges_by_source.setdefault(str(edge["source_experiment_id"]), []).append(edge)
        edges_by_target.setdefault(str(edge["target_experiment_id"]), []).append(edge)
    upstream_ids = _lineage_walk(
        experiment_id,
        edges_by_target,
        next_id_key="source_experiment_id",
        max_depth=max_depth,
    )
    downstream_ids = _lineage_walk(
        experiment_id,
        edges_by_source,
        next_id_key="target_experiment_id",
        max_depth=max_depth,
    )
    node_ids = {experiment_id, *upstream_ids, *downstream_ids}
    objects = {
        str(item["experiment_id"]): item
        for item in store.strategy_experiment_objects(limit=10_000)
        if str(item["experiment_id"]) in node_ids
    }
    missing = sorted(node_ids - set(objects))
    selected_edges = [
        edge
        for edge in all_edges
        if str(edge["source_experiment_id"]) in node_ids
        and str(edge["target_experiment_id"]) in node_ids
    ]
    formal_refs = _lineage_formal_strategy_refs(store, node_ids)
    center_node = _lineage_node(center)
    center_node["formal_strategy_refs"] = formal_refs.get(str(experiment_id), [])
    nodes = [_lineage_node(objects[node_id]) for node_id in sorted(objects)]
    for node in nodes:
        node["formal_strategy_refs"] = formal_refs.get(str(node["experiment_id"]), [])
    return {
        "protocol": "AUTOALPHA_STRATEGY_EXPERIMENT_LINEAGE_V1",
        "experiment_id": experiment_id,
        "depth": max_depth,
        "center": center_node,
        "nodes": nodes,
        "edges": selected_edges,
        "upstream_experiment_ids": sorted(upstream_ids),
        "downstream_experiment_ids": sorted(downstream_ids),
        "missing_experiment_ids": missing,
        "formal_strategy_refs": formal_refs,
        "evidence_summary": _lineage_evidence_summary(
            center,
            selected_edges,
            objects,
            formal_refs,
        ),
    }


def _lineage_walk(
    start_id: str,
    edges_by_id: dict[str, list[dict[str, Any]]],
    *,
    next_id_key: str,
    max_depth: int,
) -> set[str]:
    seen: set[str] = set()
    frontier = {start_id}
    for _ in range(max_depth):
        next_frontier: set[str] = set()
        for current in frontier:
            for edge in edges_by_id.get(current, []):
                next_id = str(edge[next_id_key])
                if next_id not in seen and next_id != start_id:
                    seen.add(next_id)
                    next_frontier.add(next_id)
        frontier = next_frontier
        if not frontier:
            break
    return seen


def _knowledge_homogeneity_groups(factors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for factor in factors:
        cluster_id = str(
            factor.get("behavior_cluster_id")
            or _factor_metric(factor, "homogeneity_cluster_id")
            or factor.get("cluster_id")
            or "UNCLUSTERED"
        )
        groups.setdefault(cluster_id, []).append(factor)
    folded = []
    for cluster_id, members in groups.items():
        leader = max(members, key=_long_only_score)
        nearest_links = [
            {
                "factor_id": item["factor_id"],
                "nearest_factor_id": item.get("behavior_nearest_factor_id")
                or _factor_metric(item, "homogeneity_nearest_factor_id"),
                "nearest_similarity": item.get("behavior_nearest_similarity")
                or _factor_metric(item, "homogeneity_nearest_similarity"),
            }
            for item in members
            if item.get("behavior_nearest_factor_id")
            or _factor_metric(item, "homogeneity_nearest_factor_id")
        ]
        folded.append(
            {
                "cluster_id": cluster_id,
                "size": len(members),
                "leader_factor_id": leader["factor_id"],
                "leader_name": leader["name"],
                "leader_score": _long_only_score(leader),
                "mechanisms": sorted(
                    {
                        normalize_mechanism(item.get("mechanism_type") or item["family"])
                        for item in members
                    }
                ),
                "parameter_family_count": len(
                    {_factor_parameter_family(item) for item in members}
                ),
                "redundancy_counts": dict(
                    Counter(
                        str(
                            item.get("behavior_redundancy")
                            or _factor_metric(item, "homogeneity_redundancy_label")
                            or "UNKNOWN"
                        )
                        for item in members
                    )
                ),
                "members": [
                    {
                        "factor_id": item["factor_id"],
                        "name": item["name"],
                        "score": _long_only_score(item),
                        "mechanism": normalize_mechanism(
                            item.get("mechanism_type") or item["family"]
                        ),
                        "parameter_family": _factor_parameter_family(item),
                        "status": item.get("status"),
                    }
                    for item in sorted(members, key=_long_only_score, reverse=True)[:12]
                ],
                "nearest_links": nearest_links[:12],
            }
        )
    return sorted(folded, key=lambda item: (item["size"], item["leader_score"]), reverse=True)


def _knowledge_mechanism_map(factors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_mechanism: dict[str, list[dict[str, Any]]] = {}
    for factor in factors:
        mechanism = normalize_mechanism(factor.get("mechanism_type") or factor["family"])
        by_mechanism.setdefault(mechanism, []).append(factor)
    items = []
    for mechanism, members in by_mechanism.items():
        leader = max(members, key=_long_only_score)
        weak_years = _weak_years(members)
        items.append(
            {
                "mechanism": mechanism,
                "factor_count": len(members),
                "leader_factor_id": leader["factor_id"],
                "leader_name": leader["name"],
                "leader_score": _long_only_score(leader),
                "average_long_only_score": _average(_long_only_score(item) for item in members),
                "behavior_cluster_count": len(
                    {
                        item.get("behavior_cluster_id") or item.get("cluster_id")
                        for item in members
                    }
                ),
                "weak_years": weak_years,
                "top_factors": [
                    {
                        "factor_id": item["factor_id"],
                        "name": item["name"],
                        "score": _long_only_score(item),
                    }
                    for item in sorted(members, key=_long_only_score, reverse=True)[:8]
                ],
            }
        )
    return sorted(items, key=lambda item: item["average_long_only_score"], reverse=True)


def _knowledge_parameter_families(factors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = {}
    for factor in factors:
        by_family.setdefault(_factor_parameter_family(factor), []).append(factor)
    families = []
    for family, members in by_family.items():
        leader = max(members, key=_long_only_score)
        families.append(
            {
                "parameter_family": family,
                "factor_count": len(members),
                "leader_factor_id": leader["factor_id"],
                "leader_name": leader["name"],
                "leader_score": _long_only_score(leader),
                "mechanisms": sorted(
                    {
                        normalize_mechanism(item.get("mechanism_type") or item["family"])
                        for item in members
                    }
                ),
                "behavior_clusters": sorted(
                    {
                        str(item.get("behavior_cluster_id") or item.get("cluster_id") or "UNKNOWN")
                        for item in members
                    }
                )[:12],
            }
        )
    return sorted(
        families,
        key=lambda item: (item["factor_count"], item["leader_score"]),
        reverse=True,
    )


def _knowledge_annual_heatmap(factors: list[dict[str, Any]]) -> dict[str, Any]:
    values_by_mechanism: dict[str, dict[str, list[float]]] = {}
    for factor in factors:
        annual_returns = _factor_annual_returns(factor)
        if not annual_returns:
            continue
        mechanism = normalize_mechanism(factor.get("mechanism_type") or factor["family"])
        bucket = values_by_mechanism.setdefault(mechanism, {})
        for year, value in annual_returns.items():
            bucket.setdefault(str(year), []).append(value)
    years = sorted({year for values in values_by_mechanism.values() for year in values})
    rows = []
    for mechanism, values in values_by_mechanism.items():
        annual_returns = {
            year: _average(values[year])
            for year in years
            if year in values and values[year]
        }
        rows.append(
            {
                "mechanism": mechanism,
                "factor_count": sum(len(items) for items in values.values()),
                "annual_returns": annual_returns,
                "weak_years": [
                    year
                    for year, value in annual_returns.items()
                    if value < 0
                ],
            }
        )
    return {
        "years": years,
        "rows": sorted(rows, key=lambda item: item["mechanism"]),
    }


def _factor_parameter_family(factor: dict[str, Any]) -> str:
    expression = factor.get("expression")
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


def _factor_metric(factor: dict[str, Any], key: str) -> Any:
    summary = factor.get("metric_summary") if isinstance(factor.get("metric_summary"), dict) else {}
    if key in summary:
        return summary[key]
    historical = (
        factor.get("historical_metric_summary")
        if isinstance(factor.get("historical_metric_summary"), dict)
        else {}
    )
    if key in historical:
        return historical[key]
    return factor.get(key)


def _factor_annual_returns(factor: dict[str, Any]) -> dict[str, float]:
    candidates = (
        factor.get("annual_returns"),
        factor.get("long_only_annual_returns"),
        _factor_metric(factor, "annual_returns"),
        _factor_metric(factor, "long_only_annual_returns"),
    )
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        cleaned = {}
        for year, value in candidate.items():
            try:
                cleaned[str(year)] = float(value)
            except (TypeError, ValueError):
                continue
        if cleaned:
            return cleaned
    return {}


def _weak_years(factors: list[dict[str, Any]]) -> list[str]:
    by_year: dict[str, list[float]] = {}
    for factor in factors:
        for year, value in _factor_annual_returns(factor).items():
            by_year.setdefault(year, []).append(value)
    return sorted(year for year, values in by_year.items() if _average(values) < 0)


def _lineage_node(item: dict[str, Any]) -> dict[str, Any]:
    metrics = item.get("metrics") or {}
    evidence = item.get("evidence") or {}
    return {
        "experiment_id": item["experiment_id"],
        "stage": item["stage"],
        "object_type": item["object_type"],
        "source_system": item["source_system"],
        "source_id": item["source_id"],
        "title": item["title"],
        "status": item["status"],
        "market": item.get("market"),
        "tags": item.get("tags") or [],
        "updated_at": item.get("updated_at"),
        "primary_metrics": {
            key: metrics.get(key)
            for key in (
                "long_only_sharpe_ratio",
                "recent_long_only_sharpe_ratio",
                "portfolio_sharpe_ratio",
                "portfolio_simple_annual_return",
                "portfolio_max_drawdown",
                "portfolio_walk_forward_worst_sharpe",
                "portfolio_annual_turnover",
                "average_long_only_score",
                "cluster_size",
            )
            if key in metrics
        },
        "evidence_keys": sorted(evidence),
        "factor_ids": evidence.get("factor_ids")
        or evidence.get("member_factor_ids")
        or ([evidence["factor_id"]] if evidence.get("factor_id") else []),
        "failed_gates": evidence.get("failed_gates") or [],
        "gate_status": evidence.get("gate_status"),
        "source_task_id": evidence.get("source_task_id"),
        "system_job_id": evidence.get("system_job_id"),
        "job_center_url": evidence.get("job_center_url"),
    }


def _lineage_evidence_summary(
    center: dict[str, Any],
    edges: list[dict[str, Any]],
    objects: dict[str, dict[str, Any]],
    formal_refs: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    stages = Counter(str(item.get("stage")) for item in objects.values())
    relations = Counter(str(edge.get("relation")) for edge in edges)
    center_evidence = center.get("evidence") or {}
    center_metrics = center.get("metrics") or {}
    center_refs = formal_refs.get(str(center.get("experiment_id")), [])
    lifecycle_counts = Counter(
        str(ref.get("lifecycle") or "UNKNOWN")
        for refs in formal_refs.values()
        for ref in refs
    )
    return {
        "node_count": len(objects),
        "edge_count": len(edges),
        "stages": dict(sorted(stages.items())),
        "relations": dict(sorted(relations.items())),
        "source_system": center.get("source_system"),
        "source_status": center.get("status"),
        "gate_status": center_evidence.get("gate_status"),
        "failed_gates": center_evidence.get("failed_gates") or [],
        "is_formal_strategy_source": center.get("stage") == "STRATEGY_VERSION"
        or bool(center_evidence.get("strategy_uid")),
        "has_formal_strategy_version": bool(center_refs),
        "formal_strategy_count": sum(len(refs) for refs in formal_refs.values()),
        "formal_strategy_lifecycles": dict(sorted(lifecycle_counts.items())),
        "center_formal_strategies": center_refs,
        "primary_metric_policy": "long_only_or_long_only_portfolio_first",
        "primary_metrics": {
            key: center_metrics.get(key)
            for key in (
                "portfolio_sharpe_ratio",
                "portfolio_simple_annual_return",
                "portfolio_max_drawdown",
                "portfolio_walk_forward_worst_sharpe",
                "long_only_sharpe_ratio",
                "recent_long_only_sharpe_ratio",
            )
            if key in center_metrics
        },
    }


def _lineage_formal_strategy_refs(
    store: ServiceStore, experiment_ids: set[str]
) -> dict[str, list[dict[str, Any]]]:
    refs: dict[str, list[dict[str, Any]]] = {}
    if not experiment_ids:
        return refs
    for strategy in store.formal_strategy_versions(limit=5000):
        source_experiment_id = strategy.get("source_experiment_id")
        if not source_experiment_id or str(source_experiment_id) not in experiment_ids:
            continue
        refs.setdefault(str(source_experiment_id), []).append(
            {
                "strategy_uid": strategy["strategy_uid"],
                "version": strategy["version"],
                "name": strategy["name"],
                "lifecycle": strategy["lifecycle"],
                "market": strategy["market"],
                "specification_hash": strategy["specification_hash"],
            }
        )
    return refs


def strategy_production_funnel(store: ServiceStore) -> dict[str, Any]:
    summary = store.strategy_experiment_summary()
    by_stage = summary.get("by_stage") or {}
    by_status = summary.get("by_status") or {}
    combination_candidates = store.strategy_experiment_objects(
        stage="COMBINATION_CANDIDATE", limit=10_000
    )
    formal_strategies = store.formal_strategy_versions(limit=5000)
    quant_tasks = QuantCombineStore(store).tasks()
    repair_tasks = [
        task
        for task in quant_tasks
        if "gate-feedback:GATE_FUNNEL_FEEDBACK_POLICY_V1" in str(task.get("notes") or "")
    ]
    lifecycle_counts = _count_by(formal_strategies, "lifecycle")
    public_passed = [
        item
        for item in combination_candidates
        if str(item.get("status") or "") in {"QUALIFIED", "QUALIFIED_CHAMPION"}
        or str((item.get("evidence") or {}).get("gate_status") or "") in {"PASSED", "QUALIFIED"}
    ]
    research_leaders = [
        item
        for item in combination_candidates
        if str(item.get("status") or "")
        in {"RESEARCH_LEADER", "RESEARCH_LEADER_ONLY", "QUALIFIED_CHAMPION"}
    ]
    failed_gate_counter: dict[str, int] = {}
    root_cause_counter: dict[str, int] = {}
    operator_hint_counter: dict[str, int] = {}
    for item in combination_candidates:
        gates = [str(gate) for gate in (item.get("evidence") or {}).get("failed_gates") or []]
        for gate in gates:
            failed_gate_counter[str(gate)] = failed_gate_counter.get(str(gate), 0) + 1
        if gates:
            root_causes = sorted(
                {
                    _public_gate_root_cause(gate, _public_gate_category(gate))
                    for gate in gates
                }
            )
            for root_cause in root_causes:
                root_cause_counter[root_cause] = root_cause_counter.get(root_cause, 0) + 1
            hint = _public_gate_operator_hint(root_causes)
            operator_hint_counter[hint] = operator_hint_counter.get(hint, 0) + 1
    stages = [
        _funnel_stage(
            "factor_candidates",
            "因子候选",
            int(by_stage.get("FACTOR_CANDIDATE", 0)),
            "已进入统一实验总线的单因子资产",
        ),
        _funnel_stage(
            "factor_clusters",
            "行为/机制簇",
            int(by_stage.get("FACTOR_CLUSTER", 0)),
            "用于去同质化和收益来源折叠的因子簇",
        ),
        _funnel_stage(
            "combination_candidates",
            "组合候选",
            len(combination_candidates),
            "AutoCombine 与 QuantCombine 形成的权重组合",
        ),
        _funnel_stage(
            "research_leaders",
            "研究领先",
            len(research_leaders),
            "可被策略库审阅的组合研究领先项",
        ),
        _funnel_stage(
            "public_passed",
            "公开验证通过",
            len(public_passed),
            "公开门禁已通过且可冻结的候选",
        ),
        _funnel_stage(
            "repair_tasks",
            "门禁修复任务",
            len(repair_tasks),
            "由门禁漏斗反馈自动生成的 QuantCombine 修复实验",
        ),
        _funnel_stage(
            "formal_research",
            "正式研究版本",
            int(lifecycle_counts.get("RESEARCH", 0)),
            "已绑定完整 StrategySpec 的正式版本",
        ),
        _funnel_stage(
            "frozen_or_holdout",
            "冻结/盲测",
            int(lifecycle_counts.get("FROZEN", 0))
            + int(lifecycle_counts.get("HIDDEN_HOLDOUT", 0)),
            "已冻结并等待或进入隐藏盲测",
        ),
        _funnel_stage(
            "shadow_or_paper",
            "影子/模拟",
            int(lifecycle_counts.get("SHADOW", 0)) + int(lifecycle_counts.get("PAPER", 0)),
            "正在执行交易可行性观察",
        ),
        _funnel_stage(
            "production_candidates",
            "生产候选",
            int(lifecycle_counts.get("PRODUCTION_CANDIDATE", 0)),
            "完成研究、盲测、交易观察与风控审批的候选",
        ),
    ]
    previous = None
    for stage in stages:
        stage["conversion_from_previous"] = (
            None if previous in (None, 0) else stage["count"] / previous
        )
        previous = stage["count"]
    bottlenecks = _strategy_funnel_bottlenecks(stages, by_status, failed_gate_counter)
    return {
        "protocol": "AUTOALPHA_STRATEGY_PRODUCTION_FUNNEL_V1",
        "stages": stages,
        "by_status": by_status,
        "formal_lifecycle": lifecycle_counts,
        "repair_tasks": [
            {
                "task_id": task["task_id"],
                "name": task["name"],
                "status": task["status"],
                "phase": task["phase"],
                "evaluation_count": task["evaluation_count"],
                "candidate_count": task.get("candidate_count", 0),
                "factor_count": task.get("factor_count", 0),
                "objective_profile": (task.get("objective") or {}).get("profile"),
                "maximum_drawdown": (task.get("objective") or {}).get("maximum_drawdown"),
                "maximum_factor_correlation": (task.get("objective") or {}).get(
                    "maximum_factor_correlation"
                ),
                "created_at": task.get("created_at"),
                "updated_at": task.get("updated_at"),
            }
            for task in repair_tasks[:10]
        ],
        "top_failed_gates": sorted(
            failed_gate_counter.items(), key=lambda item: item[1], reverse=True
        )[:12],
        "top_root_causes": sorted(
            root_cause_counter.items(), key=lambda item: item[1], reverse=True
        )[:12],
        "top_operator_hints": sorted(
            operator_hint_counter.items(), key=lambda item: item[1], reverse=True
        )[:12],
        "bottlenecks": bottlenecks,
        "primary_metric_policy": "long_only_first_strategy_conversion",
    }


def _funnel_stage(key: str, label: str, count: int, description: str) -> dict[str, Any]:
    return {
        "key": key,
        "label": label,
        "count": int(count),
        "description": description,
    }


def _strategy_funnel_bottlenecks(
    stages: list[dict[str, Any]],
    by_status: dict[str, int],
    failed_gate_counter: dict[str, int],
) -> list[dict[str, Any]]:
    lookup = {item["key"]: item for item in stages}
    bottlenecks: list[dict[str, Any]] = []
    if lookup["combination_candidates"]["count"] and not lookup["public_passed"]["count"]:
        bottlenecks.append(
            {
                "severity": "P0",
                "key": "public_gate_zero_pass",
                "title": "组合候选尚未穿透公开生产门禁",
                "detail": "组合优化已产生候选，但没有形成可冻结的公开验证通过版本。",
            }
        )
    if lookup["public_passed"]["count"] and not lookup["formal_research"]["count"]:
        bottlenecks.append(
            {
                "severity": "P0",
                "key": "strategy_library_not_materialized",
                "title": "公开通过候选尚未正式入库",
                "detail": "需要把候选绑定 StrategySpec，形成可审计策略版本。",
            }
        )
    if lookup["formal_research"]["count"] and not lookup["frozen_or_holdout"]["count"]:
        bottlenecks.append(
            {
                "severity": "P1",
                "key": "research_versions_not_frozen",
                "title": "正式研究版本尚未冻结进入盲测链路",
                "detail": "需要补齐公开验证通过证据、规格哈希和隐藏盲测申请。",
            }
        )
    if by_status.get("SCREENED_OUT", 0) > by_status.get("ACTIVE", 0) * 20:
        bottlenecks.append(
            {
                "severity": "P1",
                "key": "factor_screenout_dominates",
                "title": "因子筛出量远高于活跃资产",
                "detail": "应优先做同质折叠、机制预算和失败模式复用，而不是继续扩大搜索空间。",
            }
        )
    if failed_gate_counter:
        gate, count = max(failed_gate_counter.items(), key=lambda item: item[1])
        bottlenecks.append(
            {
                "severity": "P1",
                "key": "dominant_gate_failure",
                "title": f"主失败门禁：{gate}",
                "detail": (
                    f"{count} 个组合候选触发该失败项，应反馈给 "
                    "AutoCombine/QuantCombine 默认约束。"
                ),
            }
        )
    return bottlenecks[:6]


def _count_by(items: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        value = str(item.get(key) or "UNKNOWN")
        counts[value] = counts.get(value, 0) + 1
    return counts


def strategy_lifecycle_readiness(
    store: ServiceStore, strategy_uid: str, version: int
) -> dict[str, Any]:
    strategy = store.formal_strategy_version(strategy_uid, version)
    current = str(strategy["lifecycle"])
    next_lifecycles = sorted(STRATEGY_LIFECYCLE_TRANSITIONS.get(current, set()))
    if not next_lifecycles:
        return {
            "strategy_uid": strategy_uid,
            "version": version,
            "current_lifecycle": current,
            "next_lifecycle": None,
            "ready": False,
            "terminal": True,
            "required_evidence": [],
            "suggested_evidence": {},
            "missing_evidence": [],
            "external_evidence": [],
            "transition_label": "终态",
        }
    target = next_lifecycles[0]
    required = sorted(STRATEGY_TRANSITION_REQUIRED_EVIDENCE.get(target, set()))
    suggested = _suggest_transition_evidence(store, strategy, target)
    missing = [key for key in required if suggested.get(key) in (None, "", False)]
    external = sorted(set(required) - set(suggested))
    public_validation_gap = (
        strategy_public_validation_gap(store, strategy)
        if target == "FROZEN" and "public_validation_passed" in missing
        else None
    )
    return {
        "strategy_uid": strategy_uid,
        "version": version,
        "current_lifecycle": current,
        "next_lifecycle": target,
        "ready": not missing,
        "terminal": False,
        "required_evidence": required,
        "suggested_evidence": suggested,
        "missing_evidence": missing,
        "external_evidence": external,
        "public_validation_gap": public_validation_gap,
        "transition_label": STRATEGY_TRANSITION_LABELS.get(target, target),
        "lifecycle_order": [
            "RESEARCH",
            "FROZEN",
            "HIDDEN_HOLDOUT",
            "SHADOW",
            "PAPER",
            "PRODUCTION_CANDIDATE",
        ],
    }


def approve_formal_strategy_transition(
    store: ServiceStore,
    strategy_uid: str,
    version: int,
    *,
    approver: str,
    approval_type: str,
    notes: str = "",
    target_lifecycle: str | None = None,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    readiness = strategy_lifecycle_readiness(store, strategy_uid, version)
    target = target_lifecycle or readiness.get("next_lifecycle")
    if not target:
        raise ValueError("Strategy lifecycle is terminal and cannot be approved further")
    if target != readiness.get("next_lifecycle"):
        raise ValueError("Approval can only target the current next lifecycle")
    transition_evidence = {
        **(readiness.get("suggested_evidence") or {}),
        **(evidence or {}),
    }
    transition_evidence["human_approval"] = {
        "approval_type": approval_type,
        "approver": approver.strip(),
        "notes": notes.strip(),
        "approved_at": datetime.now(UTC).isoformat(),
        "target_lifecycle": target,
        "readiness_missing_before_approval": readiness.get("missing_evidence") or [],
    }
    return promote_formal_strategy_lifecycle(
        store,
        strategy_uid,
        version,
        target_lifecycle=str(target),
        evidence=transition_evidence,
    )


def strategy_execution_package(
    store: ServiceStore, strategy_uid: str, version: int
) -> dict[str, Any]:
    strategy = store.formal_strategy_version(strategy_uid, version)
    readiness = strategy_lifecycle_readiness(store, strategy_uid, version)
    signal_policy = strategy.get("signal_policy") or {}
    rebalance_policy = strategy.get("rebalance_policy") or {}
    execution_policy = strategy.get("execution_policy") or {}
    risk_policy = strategy.get("risk_policy") or {}
    cost_policy = strategy.get("cost_policy") or {}
    evidence = strategy.get("evidence") or {}
    production_ready = strategy.get("lifecycle") == "PRODUCTION_CANDIDATE"
    blockers = []
    if not production_ready:
        blockers.append(f"lifecycle_not_production_candidate:{strategy.get('lifecycle')}")
    if readiness.get("missing_evidence"):
        blockers.append(
            "next_transition_missing:" + ",".join(readiness.get("missing_evidence") or [])
        )
    if not evidence.get("strict_pit_market_state_verified"):
        blockers.append("strict_pit_market_state_not_verified")
    return {
        "strategy_uid": strategy_uid,
        "version": version,
        "name": strategy["name"],
        "market": strategy["market"],
        "lifecycle": strategy["lifecycle"],
        "production_ready": production_ready and not blockers,
        "production_blockers": blockers,
        "specification_hash": strategy["specification_hash"],
        "source_experiment_id": strategy.get("source_experiment_id"),
        "lifecycle_readiness": readiness,
        "signal_contract": {
            "factor_ids": signal_policy.get("factor_ids") or [],
            "weights": signal_policy.get("weights") or [],
            "score_method": signal_policy.get("score_method"),
            "signal_time": signal_policy.get("signal_time"),
            "ranking_side": "LONG_ONLY_TOP_RANK",
        },
        "rebalance_contract": {
            "schedule": rebalance_policy.get("schedule"),
            "holding_period_days": rebalance_policy.get("holding_period_days"),
            "selection_count": rebalance_policy.get("selection_count"),
            "sell_rule": "SELL_POSITIONS_NOT_IN_CURRENT_TARGET_AT_NEXT_APPROVED_OPEN",
            "buy_rule": "BUY_TARGETS_AT_NEXT_APPROVED_OPEN_AFTER_SELLS",
        },
        "execution_contract": {
            "execution_time": execution_policy.get("execution_time"),
            "execution_lag_sessions": execution_policy.get("execution_lag_sessions"),
            "engine": execution_policy.get("engine"),
            "product_template": execution_policy.get("product_template"),
            "price_basis": "RAW_OPEN_FOR_CASH_EXECUTION",
            "signal_basis": "FORWARD_ADJUSTED_RESEARCH_PANEL",
            "tradability": {
                "buy": "require_can_buy_open_or_proxy",
                "sell": "require_can_sell_open_or_proxy",
                "blocked_order_policy": "retain_position_or_skip_order_and_audit",
            },
            "lot_size": 1,
        },
        "paper_trading_contract": _strategy_paper_trading_contract(
            strategy=strategy,
            signal_policy=signal_policy,
            rebalance_policy=rebalance_policy,
            execution_policy=execution_policy,
            risk_policy=risk_policy,
            cost_policy=cost_policy,
        ),
        "risk_contract": risk_policy,
        "cost_contract": cost_policy,
        "monitoring_contract": strategy.get("monitoring_policy") or {},
        "trading_playbook": _strategy_trading_playbook(
            signal_policy=signal_policy,
            rebalance_policy=rebalance_policy,
            execution_policy=execution_policy,
            risk_policy=risk_policy,
            cost_policy=cost_policy,
            monitoring_policy=strategy.get("monitoring_policy") or {},
        ),
        "audit_contract": {
            "evidence_hash": strategy["specification_hash"],
            "promotion_trail": evidence.get("promotion_trail") or [],
            "primary_metric_convention": evidence.get("primary_metric_convention"),
            "strict_pit_market_state_verified": bool(
                evidence.get("strict_pit_market_state_verified")
            ),
        },
    }


def _strategy_paper_trading_contract(
    *,
    strategy: dict[str, Any],
    signal_policy: dict[str, Any],
    rebalance_policy: dict[str, Any],
    execution_policy: dict[str, Any],
    risk_policy: dict[str, Any],
    cost_policy: dict[str, Any],
) -> dict[str, Any]:
    factor_ids = list(signal_policy.get("factor_ids") or [])
    weights = list(signal_policy.get("weights") or [])
    maximum_positions = int(
        risk_policy.get("maximum_positions")
        or rebalance_policy.get("selection_count")
        or max(len(factor_ids), 1)
    )
    return {
        "protocol": "AUTOALPHA_STRATEGY_TO_PAPER_PORTFOLIO_SEED_V1",
        "compatible_engine": "PaperTradingEngine",
        "execution_protocol": "US_EQUITY_PAPER_NEXT_OPEN_PROXY_EXECUTION_V2",
        "proxy_only": True,
        "production_caveat": "NON_PIT_PROXY_RESEARCH_AND_PAPER_ONLY",
        "required_operator_inputs": ["initial_cash_usd", "as_of_date"],
        "paper_portfolio_seed": {
            "name": f"{strategy['name']} · PAPER",
            "factor_ids": factor_ids,
            "weights": weights,
            "selection_count": int(rebalance_policy.get("selection_count") or maximum_positions),
            "gross_exposure": float(risk_policy.get("gross_exposure") or 0.9),
            "slippage_bps_each_side": float(
                cost_policy.get("slippage_bps_each_side")
                or cost_policy.get("default_slippage_bps_each_side")
                or 5.0
            ),
            "market": strategy.get("market") or "US",
            "source_strategy_uid": strategy.get("strategy_uid"),
            "source_strategy_version": strategy.get("version"),
        },
        "timing": {
            "signal_time": signal_policy.get("signal_time") or "END_OF_DAY_AFTER_CLOSE",
            "execution_time": execution_policy.get("execution_time") or "NEXT_SESSION_OPEN",
            "execution_lag_sessions": execution_policy.get("execution_lag_sessions") or 1,
            "rebalance_schedule": rebalance_policy.get("schedule") or "WEEKLY_FIRST_SESSION",
        },
        "tradability": {
            "buy": "can_buy_open_or_proxy_required",
            "sell": "can_sell_open_or_proxy_required",
            "t_plus_one_sell_lock": False,
            "blocked_order_policy": "skip_or_retain_and_audit",
        },
    }


def _strategy_trading_playbook(
    *,
    signal_policy: dict[str, Any],
    rebalance_policy: dict[str, Any],
    execution_policy: dict[str, Any],
    risk_policy: dict[str, Any],
    cost_policy: dict[str, Any],
    monitoring_policy: dict[str, Any],
) -> dict[str, Any]:
    selection_count = int(rebalance_policy.get("selection_count") or 0)
    gross_exposure = float(risk_policy.get("gross_exposure") or 0.0)
    maximum_positions = int(risk_policy.get("maximum_positions") or selection_count or 0)
    return {
        "protocol": "US_EQUITY_LONG_ONLY_TRADING_PLAYBOOK_V1",
        "portfolio_mode": "LONG_ONLY_CASH_EQUITY",
        "signal_cutoff": signal_policy.get("signal_time") or "END_OF_DAY_AFTER_CLOSE",
        "rebalance_trigger": rebalance_policy.get("schedule") or "WEEKLY_FIRST_SESSION",
        "execution_window": execution_policy.get("execution_time") or "NEXT_SESSION_OPEN",
        "open_close_sequence": [
            "after_close_compute_factor_scores",
            "before_next_open_freeze_target_list",
            "next_open_sell_positions_not_in_target_if_tradable",
            "next_open_buy_targets_after_sells_with_cash_budget",
            "after_fill_write_trade_ledger_and_blocked_order_audit",
        ],
        "target_selection": {
            "rank_side": "LONG_ONLY_TOP_RANK",
            "selection_count": selection_count,
            "score_method": signal_policy.get("score_method"),
            "factor_ids": list(signal_policy.get("factor_ids") or []),
            "weights": list(signal_policy.get("weights") or []),
        },
        "capital_allocation": {
            "gross_exposure": gross_exposure,
            "maximum_positions": maximum_positions,
            "per_position_target_weight": (
                gross_exposure / maximum_positions if maximum_positions else None
            ),
            "cash_reserve": max(0.0, 1.0 - gross_exposure),
            "lot_size": 1,
            "weighting": "equal_position_value_after_factor_selection",
        },
        "blocked_order_policy": {
            "buy_blocked": "skip_unbuyable_target_and_keep_cash",
            "sell_blocked": "retain_position_until_next_tradable_rebalance",
            "settlement": "same-day sells are permitted; settled-cash rules depend on account type",
            "audit": "record_blocked_order_with_reason_and_market_state",
        },
        "cost_assumptions": {
            "commission_per_share": cost_policy.get("commission_per_share"),
            "minimum_commission_usd": cost_policy.get("minimum_commission_usd"),
            "sec_fee_per_million_usd_sell": cost_policy.get(
                "sec_fee_per_million_usd_sell"
            ),
            "finra_taf_per_share_sell": cost_policy.get("finra_taf_per_share_sell"),
            "slippage_model": cost_policy.get("slippage_model"),
        },
        "disable_conditions": [
            "strict_pit_market_state_missing_for_production",
            "paper_trading_gate_failed",
            "drawdown_exceeds_risk_policy",
            "turnover_or_cost_stress_exceeds_policy",
            "factor_correlation_or_mechanism_crowding_breaks_policy",
        ],
        "monitoring": {
            "review_frequency": monitoring_policy.get("review_frequency"),
            "decay_checks": list(monitoring_policy.get("decay_checks") or []),
            "paper_first": bool(monitoring_policy.get("paper_first")),
        },
    }


def strategy_release_dossier(
    store: ServiceStore, strategy_uid: str, version: int
) -> dict[str, Any]:
    """Build an operator-facing dossier for a formal strategy version."""
    strategy = store.formal_strategy_version(strategy_uid, version)
    package = strategy_execution_package(store, strategy_uid, version)
    source_experiment = None
    source_experiment_id = strategy.get("source_experiment_id")
    if source_experiment_id:
        source_experiment = store.strategy_experiment_object(str(source_experiment_id))
    factor_map = {str(item["factor_id"]): item for item in store.factor_pool(limit=10_000)}
    factor_ids = list((strategy.get("signal_policy") or {}).get("factor_ids") or [])
    weights = list((strategy.get("signal_policy") or {}).get("weights") or [])
    factors = []
    for index, factor_id in enumerate(factor_ids):
        factor = factor_map.get(str(factor_id)) or {}
        proposal = factor.get("proposal") or {}
        metrics = factor.get("metrics") or {}
        factors.append(
            {
                "factor_id": factor_id,
                "weight": weights[index] if index < len(weights) else None,
                "name": factor.get("name") or proposal.get("name") or factor_id,
                "family": factor.get("family") or proposal.get("family"),
                "source_task_id": factor.get("source_task_id"),
                "source_iteration": factor.get("source_iteration"),
                "canonical_mechanism": normalize_mechanism(
                    proposal.get("canonical_mechanism")
                    or factor.get("canonical_mechanism")
                    or factor.get("mechanism_type")
                ),
                "behavior_cluster_id": metrics.get("online_behavior_cluster_id")
                or factor.get("behavior_cluster_id"),
                "long_only": {
                    key: metrics.get(key)
                    for key in LONG_ONLY_PRIMARY_METRICS
                    if key in metrics
                },
            }
        )
    source_metrics = (source_experiment or {}).get("metrics") or {}
    source_evidence = (source_experiment or {}).get("evidence") or {}
    return {
        "dossier_protocol": "AUTOALPHA_FORMAL_STRATEGY_RELEASE_DOSSIER_V1",
        "strategy": {
            "strategy_uid": strategy_uid,
            "version": version,
            "name": strategy["name"],
            "market": strategy["market"],
            "lifecycle": strategy["lifecycle"],
            "specification_hash": strategy["specification_hash"],
        },
        "source": {
            "experiment_id": source_experiment_id,
            "system": (source_experiment or {}).get("source_system"),
            "source_id": (source_experiment or {}).get("source_id"),
            "status": (source_experiment or {}).get("status"),
            "gate_status": source_evidence.get("gate_status"),
            "failed_gates": source_evidence.get("failed_gates") or [],
            "system_job_id": source_evidence.get("system_job_id"),
            "job_center_url": source_evidence.get("job_center_url"),
            "metrics": {
                key: source_metrics.get(key)
                for key in (
                    "portfolio_sharpe_ratio",
                    "portfolio_simple_annual_return",
                    "portfolio_max_drawdown",
                    "portfolio_walk_forward_worst_sharpe",
                    "portfolio_annual_turnover",
                    "portfolio_max_factor_correlation",
                )
                if key in source_metrics
            },
        },
        "execution_package": package,
        "factors": factors,
        "readiness": package["lifecycle_readiness"],
        "production_blockers": package["production_blockers"],
        "audit": {
            "primary_metric_convention": package["audit_contract"].get(
                "primary_metric_convention"
            ),
            "promotion_trail": package["audit_contract"].get("promotion_trail") or [],
            "release_decision": (
                "EXPORT_ONLY_NOT_PRODUCTION_READY"
                if not package["production_ready"]
                else "READY_FOR_HUMAN_PRODUCTION_REVIEW"
            ),
        },
    }


def publish_strategy_release_dossier(
    store: ServiceStore,
    artifact_root: Path,
    strategy_uid: str,
    version: int,
) -> dict[str, Any]:
    dossier = strategy_release_dossier(store, strategy_uid, version)
    payload = (
        json.dumps(dossier, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    artifact = ArtifactRegistry(artifact_root).publish(
        "strategy-release-dossier",
        payload,
        owner="formal-strategy-library",
        source_ids=(f"{strategy_uid}@{version}",),
        metadata={
            "strategy_uid": strategy_uid,
            "version": version,
            "strategy_name": dossier["strategy"]["name"],
            "lifecycle": dossier["strategy"]["lifecycle"],
            "production_ready": dossier["execution_package"]["production_ready"],
            "release_decision": dossier["audit"]["release_decision"],
            "specification_hash": dossier["strategy"]["specification_hash"],
        },
    )
    return {
        "artifact": asdict(artifact),
        "dossier": dossier,
        "content_type": "application/json",
        "filename": f"{strategy_uid}-v{version}-release-dossier.json",
    }


def strategy_promotion_candidates(store: ServiceStore, *, limit: int = 50) -> list[dict[str, Any]]:
    existing_sources = {
        str(item.get("source_experiment_id"))
        for item in store.formal_strategy_versions(limit=5000)
        if item.get("source_experiment_id")
    }
    candidates = []
    for experiment in store.strategy_experiment_objects(
        stage="COMBINATION_CANDIDATE", limit=10_000
    ):
        if experiment["experiment_id"] in existing_sources:
            continue
        evidence = experiment.get("evidence") or {}
        factor_ids = list(evidence.get("factor_ids") or [])
        weights = list(evidence.get("weights") or [])
        if not factor_ids or len(factor_ids) != len(weights):
            continue
        status = str(experiment.get("status") or "")
        gate_status = str(evidence.get("gate_status") or "")
        candidate_class = (
            "QUALIFIED"
            if status in {"QUALIFIED", "QUALIFIED_CHAMPION", "PRODUCTION_CANDIDATE"}
            or gate_status in {"PASSED", "QUALIFIED"}
            else "RESEARCH_LEADER"
            if status in {"RESEARCH_LEADER", "RESEARCH_LEADER_ONLY"}
            else "OBSERVATION"
        )
        if candidate_class == "OBSERVATION":
            continue
        metrics = experiment.get("metrics") or {}
        public_validation_gap = (
            None
            if candidate_class == "QUALIFIED"
            else _public_validation_gap_from_experiment(
                experiment, str(experiment["experiment_id"])
            )
        )
        production_evidence_summary = _candidate_production_evidence_summary(
            candidate_class=candidate_class,
            metrics=metrics,
            evidence=evidence,
            public_validation_gap=public_validation_gap,
        )
        candidates.append(
            {
                "experiment_id": experiment["experiment_id"],
                "title": experiment["title"],
                "source_system": experiment["source_system"],
                "source_id": experiment["source_id"],
                "market": experiment["market"],
                "status": status,
                "candidate_class": candidate_class,
                "score": _strategy_candidate_score(metrics),
                "metrics": {
                    key: metrics.get(key)
                    for key in (
                        "portfolio_sharpe_ratio",
                        "portfolio_simple_annual_return",
                        "portfolio_max_drawdown",
                        "portfolio_walk_forward_worst_sharpe",
                        "portfolio_annual_turnover",
                        "portfolio_max_factor_correlation",
                        "long_only_sharpe_ratio",
                        "long_only_simple_annual_return",
                        "long_only_max_drawdown",
                    )
                    if key in metrics
                },
                "factor_ids": factor_ids,
                "weights": weights,
                "failed_gates": evidence.get("failed_gates") or [],
                "create_research_version_allowed": True,
                "freeze_ready_after_creation": candidate_class == "QUALIFIED",
                "public_validation_gap": public_validation_gap,
                "production_evidence_summary": production_evidence_summary,
                "operator_hint": (
                    "READY_TO_FREEZE_PUBLIC_VALIDATION"
                    if candidate_class == "QUALIFIED"
                    else public_validation_gap["operator_hint"]
                ),
                "next_action": (
                    "CREATE_RESEARCH_VERSION_AND_FREEZE"
                    if candidate_class == "QUALIFIED"
                    else "CREATE_RESEARCH_VERSION_FOR_REVIEW"
                ),
                "promotion_path": [
                    "CREATE_RESEARCH_VERSION",
                    "FREEZE_PUBLIC_VALIDATION",
                    "HIDDEN_HOLDOUT",
                    "SHADOW_TRADING",
                    "PAPER_TRADING",
                    "PRODUCTION_CANDIDATE",
                ],
            }
        )
    return sorted(candidates, key=lambda item: item["score"], reverse=True)[:limit]


def _candidate_production_evidence_summary(
    *,
    candidate_class: str,
    metrics: dict[str, Any],
    evidence: dict[str, Any],
    public_validation_gap: dict[str, Any] | None,
) -> dict[str, Any]:
    failed_gates = [str(item) for item in evidence.get("failed_gates") or []]
    has_required_public_metrics = all(
        _metric_is_number(metrics.get(key))
        for key in (
            "portfolio_sharpe_ratio",
            "portfolio_simple_annual_return",
            "portfolio_max_drawdown",
            "portfolio_walk_forward_worst_sharpe",
        )
    )
    create_allowed = bool(evidence.get("factor_ids")) and bool(evidence.get("weights"))
    freeze_ready = candidate_class == "QUALIFIED" and has_required_public_metrics
    missing = []
    if not create_allowed:
        missing.append("factor_ids_and_weights")
    if not has_required_public_metrics:
        missing.append("canonical_public_validation_metrics")
    if candidate_class != "QUALIFIED":
        missing.append("public_validation_gate_passed")
    if public_validation_gap:
        missing.extend(public_validation_gap.get("root_causes") or [])
    return {
        "protocol": "AUTOALPHA_STRATEGY_PRODUCTION_EVIDENCE_SUMMARY_V1",
        "evidence_state": (
            "READY_TO_CREATE_AND_FREEZE"
            if freeze_ready
            else "READY_TO_CREATE_RESEARCH_VERSION"
            if create_allowed
            else "BLOCKED_INCOMPLETE_SPEC"
        ),
        "create_research_version_allowed": create_allowed,
        "freeze_ready_after_creation": freeze_ready,
        "missing_or_blocking_evidence": list(dict.fromkeys(missing)),
        "public_gate": {
            "status": evidence.get("gate_status"),
            "failed_gates": failed_gates,
            "root_causes": (public_validation_gap or {}).get("root_causes") or [],
            "operator_hint": (
                "READY_TO_FREEZE_PUBLIC_VALIDATION"
                if freeze_ready
                else (public_validation_gap or {}).get(
                    "operator_hint",
                    "INSPECT_SOURCE_CANDIDATE_GATE_TELEMETRY",
                )
            ),
        },
        "metric_coverage": {
            "has_portfolio_sharpe_ratio": _metric_is_number(
                metrics.get("portfolio_sharpe_ratio")
            ),
            "has_portfolio_simple_annual_return": _metric_is_number(
                metrics.get("portfolio_simple_annual_return")
            ),
            "has_portfolio_max_drawdown": _metric_is_number(
                metrics.get("portfolio_max_drawdown")
            ),
            "has_portfolio_walk_forward_worst_sharpe": _metric_is_number(
                metrics.get("portfolio_walk_forward_worst_sharpe")
            ),
        },
        "next_required_steps": [
            "CREATE_RESEARCH_VERSION",
            *([] if freeze_ready else ["REVIEW_PUBLIC_VALIDATION_EVIDENCE"]),
            "FREEZE_PUBLIC_VALIDATION",
            "HIDDEN_HOLDOUT",
            "SHADOW_TRADING",
            "PAPER_TRADING",
            "PRODUCTION_CANDIDATE",
        ],
    }


def _formal_strategy_evidence_summary(strategy: dict[str, Any]) -> dict[str, Any]:
    readiness = strategy.get("lifecycle_readiness") or {}
    evidence = strategy.get("evidence") or {}
    lifecycle = str(strategy.get("lifecycle") or "UNKNOWN")
    missing = [str(item) for item in readiness.get("missing_evidence") or []]
    if not evidence.get("strict_pit_market_state_verified"):
        missing.append("strict_pit_market_state_verified")
    if lifecycle != "PRODUCTION_CANDIDATE":
        missing.append(f"lifecycle_not_terminal:{lifecycle}")
    return {
        "protocol": "AUTOALPHA_STRATEGY_PRODUCTION_EVIDENCE_SUMMARY_V1",
        "evidence_state": (
            "PRODUCTION_REVIEW_READY"
            if lifecycle == "PRODUCTION_CANDIDATE" and not missing
            else "NEXT_TRANSITION_READY"
            if readiness.get("ready")
            else "EVIDENCE_INCOMPLETE"
        ),
        "current_lifecycle": lifecycle,
        "next_lifecycle": readiness.get("next_lifecycle"),
        "next_transition_ready": bool(readiness.get("ready")),
        "missing_or_blocking_evidence": list(dict.fromkeys(missing)),
        "public_gate": readiness.get("public_validation_gap"),
        "next_required_steps": [
            step
            for step in readiness.get("lifecycle_order", [])
            if step not in {"RESEARCH"} and step != lifecycle
        ],
    }


def create_formal_strategy_from_experiment(
    store: ServiceStore,
    experiment_id: str,
    *,
    name: str | None = None,
    lifecycle: str = "RESEARCH",
) -> dict[str, Any]:
    if lifecycle not in ALLOWED_INITIAL_STRATEGY_LIFECYCLES:
        raise ValueError(
            "Formal strategy creation starts at RESEARCH. Promote through freeze, "
            "hidden holdout, shadow trading, paper trading, and production-candidate gates."
        )
    experiment = store.strategy_experiment_object(experiment_id)
    if experiment is None:
        raise KeyError(f"Strategy experiment not found: {experiment_id}")
    if experiment.get("stage") != "COMBINATION_CANDIDATE":
        raise ValueError("Formal strategies can only be created from combination candidates")
    evidence = experiment["evidence"]
    factor_ids = list(evidence.get("factor_ids") or [])
    weights = list(evidence.get("weights") or [])
    if not factor_ids:
        raise ValueError("Combination candidate has no factor_ids")
    if len(factor_ids) != len(weights):
        raise ValueError("Combination candidate factor_ids and weights length mismatch")
    for existing in store.formal_strategy_versions(limit=5000):
        if existing.get("source_experiment_id") == experiment_id:
            return existing
    strategy_uid = f"STR_{hashlib.sha256(experiment_id.encode()).hexdigest()[:12]}"
    return store.create_formal_strategy_version(
        strategy_uid=strategy_uid,
        source_experiment_id=experiment_id,
        name=name or experiment["title"],
        market=experiment.get("market") or "US",
        lifecycle=lifecycle,
        signal_policy={
            "factor_ids": factor_ids,
            "weights": weights,
            "score_method": "weighted_cross_sectional_zscore",
            "signal_time": "END_OF_DAY_AFTER_CLOSE",
        },
        rebalance_policy={
            "schedule": "WEEKLY_FIRST_SESSION",
            "holding_period_days": 5,
            "selection_count": evidence.get("maximum_positions", 30),
        },
        execution_policy={
            "execution_time": "NEXT_SESSION_OPEN",
            "execution_lag_sessions": 1,
            "product_template": "LONG_ONLY_CAPITAL",
            "engine": "EVENT_LEDGER_OR_VECTOR_PROXY",
        },
        risk_policy={
            "gross_exposure": evidence.get("target_gross_exposure", 0.9),
            "maximum_positions": evidence.get("maximum_positions", 30),
            "maximum_drawdown": _metric(experiment["metrics"], "portfolio_max_drawdown"),
            "capacity_usd": _metric(experiment["metrics"], "portfolio_capacity_usd"),
        },
        cost_policy={
            "commission_per_share": 0.0035,
            "minimum_commission_usd": 0.35,
            "maximum_commission_fraction": 0.01,
            "sec_fee_per_million_usd_sell": 20.60,
            "finra_taf_per_share_sell": 0.000195,
            "slippage_model": "CONFIGURABLE_BPS",
        },
        monitoring_policy={
            "decay_checks": ["rolling_sharpe", "drawdown", "turnover", "factor_correlation"],
            "review_frequency": "DAILY_AFTER_CLOSE",
            "paper_first": True,
        },
        evidence={
            **evidence,
            "source_experiment_id": experiment_id,
            "source_system": experiment["source_system"],
            "primary_metric_convention": "US_EQUITY_LONG_ONLY_WEEKLY_NON_PIT_PROXY",
        },
    )


def promote_formal_strategy_lifecycle(
    store: ServiceStore,
    strategy_uid: str,
    version: int,
    *,
    target_lifecycle: str,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    strategy = store.formal_strategy_version(strategy_uid, version)
    current = str(strategy["lifecycle"])
    allowed = STRATEGY_LIFECYCLE_TRANSITIONS.get(current, set())
    if target_lifecycle not in allowed:
        raise ValueError(f"Invalid strategy lifecycle transition: {current} -> {target_lifecycle}")
    validation = _validate_strategy_transition_evidence(
        store,
        strategy,
        target_lifecycle,
        evidence,
    )
    if validation["missing_evidence"]:
        raise ValueError(
            f"Missing promotion evidence for {target_lifecycle}: "
            f"{', '.join(validation['missing_evidence'])}"
        )
    if validation["blocking_reasons"]:
        raise ValueError(
            f"Invalid promotion evidence for {target_lifecycle}: "
            f"{', '.join(validation['blocking_reasons'])}"
        )
    updated_evidence = dict(strategy.get("evidence") or {})
    trail = list(updated_evidence.get("promotion_trail") or [])
    trail.append(
        {
            "from": current,
            "to": target_lifecycle,
            "evidence": evidence,
            "validation": validation,
        }
    )
    updated_evidence.update(evidence)
    updated_evidence["last_transition_validation"] = validation
    updated_evidence["promotion_trail"] = trail
    return store.update_formal_strategy_lifecycle(
        strategy_uid,
        version,
        lifecycle=target_lifecycle,
        evidence=updated_evidence,
    )


def advance_formal_strategy_lifecycle(
    store: ServiceStore, strategy_uid: str, version: int
) -> dict[str, Any]:
    readiness = strategy_lifecycle_readiness(store, strategy_uid, version)
    if readiness["terminal"]:
        raise ValueError("Strategy lifecycle is already terminal")
    if not readiness["ready"]:
        missing = ", ".join(readiness["missing_evidence"])
        raise ValueError(f"Strategy is not ready for {readiness['next_lifecycle']}: {missing}")
    return promote_formal_strategy_lifecycle(
        store,
        strategy_uid,
        version,
        target_lifecycle=str(readiness["next_lifecycle"]),
        evidence=dict(readiness["suggested_evidence"]),
    )


def _suggest_transition_evidence(
    store: ServiceStore, strategy: dict[str, Any], target_lifecycle: str
) -> dict[str, Any]:
    if target_lifecycle == "FROZEN":
        source_experiment_id = strategy.get("source_experiment_id")
        experiment = (
            store.strategy_experiment_object(str(source_experiment_id))
            if source_experiment_id
            else None
        )
        source_evidence = (experiment or {}).get("evidence") or {}
        source_metrics = (experiment or {}).get("metrics") or {}
        gate_status = str(source_evidence.get("gate_status") or "")
        status = str((experiment or {}).get("status") or "")
        public_validation_passed = gate_status in {"PASSED", "QUALIFIED"} or status in {
            "QUALIFIED",
            "QUALIFIED_CHAMPION",
            "PRODUCTION_CANDIDATE",
        }
        return {
            "source_experiment_id": source_experiment_id,
            "factor_ids": strategy.get("signal_policy", {}).get("factor_ids"),
            "weights": strategy.get("signal_policy", {}).get("weights"),
            "public_validation_passed": public_validation_passed,
            "public_validation_basis": {
                "source_status": status,
                "gate_status": gate_status,
                "failed_gates": source_evidence.get("failed_gates") or [],
                "primary_metric_convention": strategy.get("evidence", {}).get(
                    "primary_metric_convention"
                ),
                "portfolio_sharpe_ratio": source_metrics.get("portfolio_sharpe_ratio"),
                "portfolio_simple_annual_return": source_metrics.get(
                    "portfolio_simple_annual_return"
                ),
                "portfolio_max_drawdown": source_metrics.get("portfolio_max_drawdown"),
            },
        }
    if target_lifecycle == "HIDDEN_HOLDOUT":
        return {"frozen_specification_hash": strategy.get("specification_hash")}
    return {}


def _validate_strategy_transition_evidence(
    store: ServiceStore,
    strategy: dict[str, Any],
    target_lifecycle: str,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    required = sorted(STRATEGY_TRANSITION_REQUIRED_EVIDENCE.get(target_lifecycle, set()))
    missing = [key for key in required if not _evidence_truthy(evidence.get(key))]
    blocking: list[str] = []
    checks: dict[str, Any] = {
        "target_lifecycle": target_lifecycle,
        "required_evidence": required,
    }
    if target_lifecycle == "FROZEN":
        freeze_checks = _validate_public_validation_freeze(store, strategy, evidence)
        blocking.extend(freeze_checks.pop("blocking_reasons", []))
        checks.update(freeze_checks)
    elif target_lifecycle == "HIDDEN_HOLDOUT":
        expected_hash = strategy.get("specification_hash")
        supplied_hash = evidence.get("frozen_specification_hash")
        checks["specification_hash_matches"] = supplied_hash == expected_hash
        if supplied_hash and supplied_hash != expected_hash:
            blocking.append("frozen_specification_hash_mismatch")
        if evidence.get("holdout_evaluation_requested") is not True:
            blocking.append("holdout_evaluation_request_must_be_true")
    elif target_lifecycle == "SHADOW":
        if evidence.get("hidden_holdout_passed") is not True:
            blocking.append("hidden_holdout_must_pass_before_shadow")
        if not str(evidence.get("holdout_evaluation_id") or "").strip():
            blocking.append("holdout_evaluation_id_required")
    elif target_lifecycle == "PAPER":
        days = _positive_int(evidence.get("shadow_trading_days"))
        checks["shadow_trading_days"] = days
        if days is None:
            blocking.append("shadow_trading_days_must_be_positive")
        if evidence.get("shadow_execution_passed") is not True:
            blocking.append("shadow_execution_must_pass_before_paper")
    elif target_lifecycle == "PRODUCTION_CANDIDATE":
        days = _positive_int(evidence.get("paper_trading_days"))
        checks["paper_trading_days"] = days
        if days is None:
            blocking.append("paper_trading_days_must_be_positive")
        if evidence.get("paper_trading_passed") is not True:
            blocking.append("paper_trading_must_pass_before_production_candidate")
        if str(evidence.get("risk_approval") or "").upper() not in {
            "APPROVED",
            "RISK_APPROVED",
            "PASS",
            "PASSED",
        }:
            blocking.append("risk_approval_must_be_approved")
        if evidence.get("strict_pit_market_state_verified") is not True:
            blocking.append("strict_pit_market_state_must_be_verified")
    return {
        "protocol": "FORMAL_STRATEGY_LIFECYCLE_TRANSITION_VALIDATION_V1",
        **checks,
        "missing_evidence": missing,
        "blocking_reasons": sorted(set(blocking)),
        "validated_at": datetime.now(UTC).isoformat(),
    }


def _validate_public_validation_freeze(
    store: ServiceStore, strategy: dict[str, Any], evidence: dict[str, Any]
) -> dict[str, Any]:
    blocking: list[str] = []
    source_experiment_id = str(
        evidence.get("source_experiment_id") or strategy.get("source_experiment_id") or ""
    )
    experiment = (
        store.strategy_experiment_object(source_experiment_id)
        if source_experiment_id
        else None
    )
    source_evidence = (experiment or {}).get("evidence") or {}
    status = str((experiment or {}).get("status") or "")
    gate_status = str(source_evidence.get("gate_status") or "")
    source_passed = gate_status in {"PASSED", "QUALIFIED"} or status in {
        "QUALIFIED",
        "QUALIFIED_CHAMPION",
        "PRODUCTION_CANDIDATE",
    }
    if evidence.get("public_validation_passed") is not True:
        blocking.append("public_validation_passed_must_be_true")
    if not source_passed:
        blocking.append("source_experiment_public_validation_not_passed")
    strategy_factor_ids = list((strategy.get("signal_policy") or {}).get("factor_ids") or [])
    strategy_weights = list((strategy.get("signal_policy") or {}).get("weights") or [])
    evidence_factor_ids = list(evidence.get("factor_ids") or [])
    evidence_weights = list(evidence.get("weights") or [])
    if evidence_factor_ids and evidence_factor_ids != strategy_factor_ids:
        blocking.append("factor_ids_do_not_match_strategy_spec")
    if evidence_weights and evidence_weights != strategy_weights:
        blocking.append("weights_do_not_match_strategy_spec")
    return {
        "source_experiment_id": source_experiment_id or None,
        "source_experiment_status": status,
        "source_gate_status": gate_status,
        "source_public_validation_passed": source_passed,
        "source_failed_gates": source_evidence.get("failed_gates") or [],
        "blocking_reasons": blocking,
    }


def _evidence_truthy(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    return not (isinstance(value, list | tuple | set | dict) and not value)


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def strategy_public_validation_gap(
    store: ServiceStore, strategy: dict[str, Any]
) -> dict[str, Any]:
    experiment = None
    source_experiment_id = strategy.get("source_experiment_id")
    if source_experiment_id:
        experiment = store.strategy_experiment_object(str(source_experiment_id))
    return _public_validation_gap_from_experiment(experiment, source_experiment_id)


def _public_validation_gap_from_experiment(
    experiment: dict[str, Any] | None, source_experiment_id: str | None
) -> dict[str, Any]:
    evidence = (experiment or {}).get("evidence") or {}
    failures = [str(value) for value in evidence.get("failed_gates") or []]
    if not failures:
        failures = ["NO_EXPLICIT_GATE_FAILURE_RECORDED"]
    categories = sorted({_public_gate_category(failure) for failure in failures})
    root_causes = sorted(
        {
            _public_gate_root_cause(failure, _public_gate_category(failure))
            for failure in failures
        }
    )
    return {
        "source_experiment_id": source_experiment_id,
        "source_system": (experiment or {}).get("source_system"),
        "source_id": (experiment or {}).get("source_id"),
        "source_status": (experiment or {}).get("status"),
        "gate_status": evidence.get("gate_status"),
        "failed_gates": failures,
        "failure_categories": categories,
        "root_causes": root_causes,
        "operator_hint": _public_gate_operator_hint(root_causes),
    }


def _public_gate_category(failure: str) -> str:
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
    if "no_explicit" in normalized:
        return "MISSING_TELEMETRY"
    return "OTHER"


def _public_gate_root_cause(failure: str, category: str) -> str:
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
    if category == "MISSING_TELEMETRY":
        return "MISSING_GATE_TELEMETRY"
    return "UNCATEGORIZED_GATE_FAILURE"


def _public_gate_operator_hint(root_causes: list[str]) -> str:
    if "VALIDATION_DATA_OR_FOLD_CAPACITY" in root_causes:
        return "REPAIR_WALK_FORWARD_CAPACITY_OR_COVERAGE"
    if "TRIAL_BUDGET_AND_OVERFITTING_PENALTY" in root_causes:
        return "REDUCE_EFFECTIVE_TRIAL_COUNT_AND_DEDUPLICATE_SEARCH_SPACE"
    if "FACTOR_INDEPENDENCE_INSUFFICIENT" in root_causes:
        return "FORCE_CROSS_MECHANISM_AND_CROSS_CLUSTER_SELECTION"
    if "RISK_CONSTRAINT_BREACH" in root_causes:
        return "SWITCH_OBJECTIVE_TO_DRAWDOWN_AND_TAIL_RISK_FIRST"
    if "SEARCH_ALPHA_STRENGTH_INSUFFICIENT" in root_causes:
        return "EXPAND_OR_RESEED_FACTOR_MECHANISMS"
    return "INSPECT_SOURCE_CANDIDATE_GATE_TELEMETRY"


def _sync_factor_candidates(
    store: ServiceStore, factors: list[dict[str, Any]]
) -> dict[str, str]:
    result: dict[str, str] = {}
    for factor in factors:
        factor_id = str(factor["factor_id"])
        experiment_id = stable_experiment_id("AUTOALPHA", factor_id, "FACTOR_CANDIDATE")
        metrics = _long_only_metrics(factor.get("ranking_values") or factor.get("metrics") or {})
        store.upsert_strategy_experiment_object(
            experiment_id=experiment_id,
            stage="FACTOR_CANDIDATE",
            object_type="factor",
            source_system="AUTOALPHA",
            source_id=factor_id,
            title=str(factor.get("name") or factor_id),
            status=str(factor.get("status") or "UNKNOWN"),
            market=str(factor.get("source_market") or "US"),
            metrics=metrics,
            evidence={
                "factor_id": factor_id,
                "source_task_id": factor.get("source_task_id"),
                "source_iteration": factor.get("source_iteration"),
                "family": factor.get("family"),
                "mechanism_type": normalize_mechanism(factor.get("mechanism_type")),
                "behavior_cluster_id": factor.get("behavior_cluster_id"),
                "behavior_cluster_label": factor.get("behavior_cluster_label"),
                "behavior_nearest_factor_id": factor.get("behavior_nearest_factor_id"),
                "behavior_nearest_similarity": factor.get("behavior_nearest_similarity"),
            },
            tags=[str(value) for value in (factor.get("tags") or [])],
        )
        result[factor_id] = experiment_id
    return result


def _sync_factor_clusters(
    store: ServiceStore, factors: list[dict[str, Any]], factor_nodes: dict[str, str]
) -> None:
    clusters: dict[str, list[dict[str, Any]]] = {}
    for factor in factors:
        cluster_id = str(
            factor.get("behavior_cluster_id")
            or factor.get("similarity_cluster_id")
            or normalize_mechanism(factor.get("mechanism_type") or factor.get("family"))
            or "UNCLUSTERED"
        )
        clusters.setdefault(cluster_id, []).append(factor)
    for cluster_id, members in clusters.items():
        experiment_id = stable_experiment_id("AUTOALPHA", cluster_id, "FACTOR_CLUSTER")
        leader = max(members, key=_long_only_score)
        store.upsert_strategy_experiment_object(
            experiment_id=experiment_id,
            stage="FACTOR_CLUSTER",
            object_type="factor_cluster",
            source_system="AUTOALPHA",
            source_id=cluster_id,
            title=f"{cluster_id} · {leader['name']}",
            status="ACTIVE",
            market=str(leader.get("source_market") or "US"),
            metrics={
                "cluster_size": len(members),
                "average_long_only_score": _average(
                    _long_only_score(item) for item in members
                ),
            },
            evidence={
                "leader_factor_id": leader["factor_id"],
                "member_factor_ids": [item["factor_id"] for item in members],
            },
        )
        for factor in members:
            factor_node = factor_nodes.get(str(factor["factor_id"]))
            if factor_node:
                store.upsert_strategy_experiment_edge(
                    factor_node,
                    experiment_id,
                    "BELONGS_TO_CLUSTER",
                    evidence={"cluster_id": cluster_id},
                )


def _sync_auto_combine(
    store: ServiceStore, combine_store: AutoCombineStore, factor_nodes: dict[str, str]
) -> None:
    for task in combine_store.tasks():
        for experiment in combine_store.experiments(str(task["task_id"]), limit=2000):
            experiment_id = stable_experiment_id(
                "AUTOCOMBINE", str(experiment["id"]), "COMBINATION_CANDIDATE"
            )
            metrics = experiment.get("metrics") or {}
            store.upsert_strategy_experiment_object(
                experiment_id=experiment_id,
                stage="COMBINATION_CANDIDATE",
                object_type="factor_combination",
                source_system="AUTOCOMBINE",
                source_id=str(experiment["id"]),
                title=f"{task['name']} · #{experiment['iteration']}",
                status=str(experiment.get("qualification") or experiment.get("gate_status")),
                market=str(task["market"]),
                protocol=task.get("protocol") or {},
                metrics=metrics,
                evidence={
                    "task_id": task["task_id"],
                    "system_job_id": external_job_id("autocombine", str(task["task_id"])),
                    "job_center_url": (
                        "http://127.0.0.1:8788/jobs?queue=autocombine&job="
                        f"{external_job_id('autocombine', str(task['task_id']))}"
                    ),
                    "factor_ids": experiment["factor_ids"],
                    "weights": experiment["weights"],
                    "return_artifact_path": experiment.get("return_artifact_path"),
                    "gate_status": experiment.get("gate_status"),
                    "failed_gates": experiment.get("failed_gates"),
                    "maximum_positions": metrics.get("portfolio_maximum_positions"),
                    "target_gross_exposure": metrics.get("portfolio_target_gross_exposure"),
                },
            )
            _link_factors(store, factor_nodes, experiment_id, experiment["factor_ids"])
        for strategy in combine_store.strategies():
            if strategy["source_task_id"] != task["task_id"]:
                continue
            _sync_strategy_version(store, "AUTOCOMBINE", strategy)


def _sync_quant_combine(
    store: ServiceStore, quant_store: QuantCombineStore, factor_nodes: dict[str, str]
) -> None:
    for task in quant_store.tasks():
        for candidate in quant_store.candidates(str(task["task_id"]), limit=2000):
            experiment_id = stable_experiment_id(
                "QUANTCOMBINE", str(candidate["id"]), "COMBINATION_CANDIDATE"
            )
            metrics = candidate.get("metrics") or {}
            store.upsert_strategy_experiment_object(
                experiment_id=experiment_id,
                stage="COMBINATION_CANDIDATE",
                object_type="factor_combination",
                source_system="QUANTCOMBINE",
                source_id=str(candidate["id"]),
                title=f"{task['name']} · #{candidate['iteration']}",
                status=str(candidate.get("qualification") or candidate.get("gate_status")),
                market=str(task["market"]),
                protocol=task.get("protocol") or {},
                metrics=metrics,
                evidence={
                    "task_id": task["task_id"],
                    "system_job_id": external_job_id("quantcombine", str(task["task_id"])),
                    "job_center_url": (
                        "http://127.0.0.1:8788/jobs?queue=quantcombine&job="
                        f"{external_job_id('quantcombine', str(task['task_id']))}"
                    ),
                    "factor_ids": candidate["factor_ids"],
                    "weights": candidate["weights"],
                    "return_artifact_path": candidate.get("return_artifact_path"),
                    "gate_status": candidate.get("gate_status"),
                    "failed_gates": candidate.get("failed_gates"),
                    "maximum_positions": metrics.get("portfolio_maximum_positions"),
                    "target_gross_exposure": metrics.get("portfolio_target_gross_exposure"),
                },
            )
            _link_factors(store, factor_nodes, experiment_id, candidate["factor_ids"])
        for strategy in quant_store.strategies():
            if strategy["source_task_id"] != task["task_id"]:
                continue
            _sync_strategy_version(store, "QUANTCOMBINE", strategy)


def _sync_paper_portfolios(store: ServiceStore, factor_nodes: dict[str, str]) -> None:
    for portfolio in store.paper_portfolios(limit=1000):
        experiment_id = stable_experiment_id(
            "PAPER_TRADING", str(portfolio["id"]), "PAPER_PORTFOLIO"
        )
        config = portfolio.get("config") or {}
        store.upsert_strategy_experiment_object(
            experiment_id=experiment_id,
            stage="PAPER_PORTFOLIO",
            object_type="paper_portfolio",
            source_system="PAPER_TRADING",
            source_id=str(portfolio["id"]),
            title=str(portfolio["name"]),
            status=str(portfolio["status"]),
            market=str(config.get("market") or "US"),
            metrics={
                "nav_usd": portfolio.get("nav_usd"),
                "cash_usd": portfolio.get("cash_usd"),
                "market_value_usd": portfolio.get("market_value_usd"),
                "gross_exposure": portfolio.get("gross_exposure"),
                "daily_return": portfolio.get("daily_return"),
            },
            evidence={
                "portfolio_id": portfolio["id"],
                "factor_ids": config.get("factor_ids") or [],
                "weights": config.get("weights") or [],
                "last_rebalanced_date": portfolio.get("last_rebalanced_date"),
            },
        )
        _link_factors(store, factor_nodes, experiment_id, list(config.get("factor_ids") or []))


def _sync_strategy_version(
    store: ServiceStore, source_system: str, strategy: dict[str, Any]
) -> None:
    specification = strategy["specification"]
    source_id = f"{strategy['strategy_id']}@{strategy['version']}"
    experiment_id = stable_experiment_id(source_system, source_id, "STRATEGY_VERSION")
    store.upsert_strategy_experiment_object(
        experiment_id=experiment_id,
        stage="STRATEGY_VERSION",
        object_type="strategy_version",
        source_system=source_system,
        source_id=source_id,
        title=str(strategy["name"]),
        status=str(strategy["lifecycle"]),
        market=str(strategy["market"]),
        protocol=specification.get("protocol") or {},
        metrics=specification.get("evaluation") or {},
        evidence={
            "strategy_id": strategy["strategy_id"],
            "version": strategy["version"],
            "factor_ids": specification.get("factor_ids") or [],
            "weights": specification.get("factor_weights") or [],
            "execution": specification.get("execution") or {},
            "evidence_hash": strategy.get("evidence_hash"),
        },
    )


def _sync_formal_strategy_versions(store: ServiceStore, factor_nodes: dict[str, str]) -> None:
    for strategy in store.formal_strategy_versions(limit=5000):
        source_id = f"{strategy['strategy_uid']}@{strategy['version']}"
        experiment_id = stable_experiment_id(
            "FORMAL_STRATEGY_LIBRARY", source_id, "STRATEGY_VERSION"
        )
        signal_policy = strategy.get("signal_policy") or {}
        factor_ids = list(signal_policy.get("factor_ids") or [])
        store.upsert_strategy_experiment_object(
            experiment_id=experiment_id,
            stage="STRATEGY_VERSION",
            object_type="formal_strategy_version",
            source_system="FORMAL_STRATEGY_LIBRARY",
            source_id=source_id,
            title=str(strategy["name"]),
            status=str(strategy["lifecycle"]),
            market=str(strategy["market"]),
            metrics=strategy.get("risk_policy") or {},
            evidence={
                "strategy_uid": strategy["strategy_uid"],
                "version": strategy["version"],
                "source_experiment_id": strategy.get("source_experiment_id"),
                "specification_hash": strategy.get("specification_hash"),
                "factor_ids": factor_ids,
                "weights": signal_policy.get("weights") or [],
                "signal_policy": signal_policy,
                "rebalance_policy": strategy.get("rebalance_policy") or {},
                "execution_policy": strategy.get("execution_policy") or {},
                "lifecycle": strategy.get("lifecycle"),
            },
        )
        source_experiment_id = strategy.get("source_experiment_id")
        if source_experiment_id:
            store.upsert_strategy_experiment_edge(
                str(source_experiment_id),
                experiment_id,
                "PROMOTED_TO_FORMAL_STRATEGY",
                evidence={
                    "strategy_uid": strategy["strategy_uid"],
                    "version": strategy["version"],
                    "lifecycle": strategy.get("lifecycle"),
                    "specification_hash": strategy.get("specification_hash"),
                },
            )
        _link_factors(store, factor_nodes, experiment_id, factor_ids)


def _link_factors(
    store: ServiceStore,
    factor_nodes: dict[str, str],
    target_experiment_id: str,
    factor_ids: list[str],
) -> None:
    for factor_id in factor_ids:
        source = factor_nodes.get(str(factor_id))
        if source:
            store.upsert_strategy_experiment_edge(
                source,
                target_experiment_id,
                "USED_IN",
                evidence={"factor_id": factor_id},
            )


def _long_only_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    flattened = dict(metrics)
    scores = metrics.get("scores") if isinstance(metrics.get("scores"), dict) else {}
    ranking_values = (
        metrics.get("ranking_values") if isinstance(metrics.get("ranking_values"), dict) else {}
    )
    if "long_only_overall" in scores:
        flattened["long_only_overall"] = scores["long_only_overall"]
    if "recent_long_only_overall" in scores:
        flattened["recent_long_only_overall"] = scores["recent_long_only_overall"]
    if "long_only_overall" in ranking_values:
        flattened.setdefault("long_only_overall", ranking_values["long_only_overall"])
    if "recent_long_only_overall" in ranking_values:
        flattened.setdefault(
            "recent_long_only_overall", ranking_values["recent_long_only_overall"]
        )
    return {
        key: flattened[key]
        for key in (*LONG_ONLY_PRIMARY_METRICS, "long_only_overall", "recent_long_only_overall")
        if key in flattened and flattened[key] is not None
    }


def _long_only_score(factor: dict[str, Any]) -> float:
    scores = factor.get("scores") if isinstance(factor.get("scores"), dict) else {}
    ranking_values = (
        factor.get("ranking_values") if isinstance(factor.get("ranking_values"), dict) else {}
    )
    metrics = factor.get("metrics") if isinstance(factor.get("metrics"), dict) else {}
    candidates = (
        factor.get("long_only_overall_score"),
        scores.get("long_only_overall"),
        ranking_values.get("long_only_overall"),
        factor.get("long_only_overall"),
        scores.get("recent_long_only_overall"),
        ranking_values.get("recent_long_only_overall"),
        factor.get("recent_long_only_overall"),
        metrics.get("long_only_overall"),
        metrics.get("recent_long_only_overall"),
    )
    for value in candidates:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    metric_source = {**metrics, **ranking_values, **factor}
    sharpe = _metric(
        metric_source,
        "long_only_sharpe_ratio",
        "recent_long_only_sharpe_ratio",
        default=0.0,
    )
    annual = _metric(
        metric_source,
        "long_only_simple_annual_return",
        "recent_long_only_simple_annual_return",
        default=0.0,
    )
    drawdown = abs(
        _metric(
            metric_source,
            "long_only_max_drawdown",
            "recent_long_only_max_drawdown",
            default=0.0,
        )
    )
    return sharpe * 0.65 + annual * 1.5 - drawdown * 0.25


def _strategy_candidate_score(metrics: dict[str, Any]) -> float:
    sharpe = _metric(
        metrics,
        "portfolio_sharpe_ratio",
        "long_only_sharpe_ratio",
        "sharpe_ratio",
        default=0.0,
    )
    annual = _metric(
        metrics,
        "portfolio_simple_annual_return",
        "long_only_simple_annual_return",
        "simple_annual_return",
        default=0.0,
    )
    drawdown = abs(
        _metric(
            metrics,
            "portfolio_max_drawdown",
            "long_only_max_drawdown",
            "max_drawdown",
            default=0.0,
        )
    )
    worst = _metric(
        metrics,
        "portfolio_walk_forward_worst_sharpe",
        "long_only_walk_forward_worst_sharpe",
        default=0.0,
    )
    correlation = _metric(metrics, "portfolio_max_factor_correlation", default=0.0)
    return sharpe * 0.45 + annual * 1.5 + worst * 0.25 - drawdown * 0.8 - correlation * 0.2


def _metric_is_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def _merge_raw_factor_evidence(
    library_factors: list[dict[str, Any]],
    raw_factors: list[dict[str, Any]],
    *,
    behavior_snapshot: dict[str, Any] | None = None,
) -> None:
    raw_by_id = {str(item["factor_id"]): item for item in raw_factors}
    behavior_by_id = (behavior_snapshot or {}).get("factors") or {}
    for factor in library_factors:
        behavior = behavior_by_id.get(str(factor["factor_id"])) or {}
        raw = raw_by_id.get(str(factor["factor_id"])) or {}
        metrics = raw.get("metrics") or {}
        proposal = raw.get("proposal") or {}
        if not factor.get("behavior_cluster_id"):
            factor["behavior_cluster_id"] = behavior.get("behavior_cluster_id") or metrics.get(
                "online_behavior_cluster_id"
            )
        if not factor.get("behavior_cluster_label"):
            factor["behavior_cluster_label"] = behavior.get(
                "behavior_cluster_label"
            ) or metrics.get("online_behavior_cluster_label")
        if not factor.get("behavior_cluster_size"):
            factor["behavior_cluster_size"] = behavior.get("behavior_cluster_size") or metrics.get(
                "online_behavior_cluster_size"
            )
        if not factor.get("behavior_cluster_role"):
            factor["behavior_cluster_role"] = behavior.get("behavior_cluster_role") or metrics.get(
                "online_behavior_cluster_role"
            )
        if not factor.get("behavior_nearest_factor_id"):
            factor["behavior_nearest_factor_id"] = behavior.get(
                "behavior_nearest_factor_id"
            ) or metrics.get("online_behavior_nearest_factor_id")
        if not factor.get("behavior_nearest_similarity"):
            factor["behavior_nearest_similarity"] = behavior.get(
                "behavior_nearest_similarity"
            ) or metrics.get(
                "online_behavior_nearest_similarity"
            )
        if not factor.get("behavior_redundancy"):
            factor["behavior_redundancy"] = behavior.get("behavior_redundancy") or metrics.get(
                "online_behavior_redundancy"
            )
        if not factor.get("mechanism_type"):
            factor["mechanism_type"] = normalize_mechanism(
                proposal.get("canonical_mechanism")
                or proposal.get("family")
                or raw.get("family")
                or "OTHER"
            )
        else:
            factor["mechanism_type"] = normalize_mechanism(factor.get("mechanism_type"))
        for key in ("annual_returns", "long_only_annual_returns"):
            if key not in factor and isinstance(metrics.get(key), dict):
                factor[key] = metrics[key]


def _metric(metrics: dict[str, Any], *keys: str, default: float | None = None) -> float | None:
    for key in keys:
        value = metrics.get(key)
        if isinstance(value, int | float):
            return float(value)
    return default


def _average(values: Any) -> float:
    collected = [float(value) for value in values]
    return sum(collected) / len(collected) if collected else 0.0
