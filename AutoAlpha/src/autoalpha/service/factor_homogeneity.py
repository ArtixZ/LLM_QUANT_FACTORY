from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

from autoalpha.data.research_fields import expression_fields
from autoalpha.service.direction import classify_mechanism
from autoalpha.service.mechanism import normalize_mechanism

HOMOGENEITY_PROTOCOL = "A_SHARE_FACTOR_HOMOGENEITY_CONTROL_V1"
HOMOGENEITY_INTEGRITY_PROTOCOL = "AUTOALPHA_FACTOR_HOMOGENEITY_INTEGRITY_V1"
DEFAULT_CROWDED_CLUSTER_SIZE = 8
DEFAULT_SUBSTITUTE_SIMILARITY = 0.82
DEFAULT_NEAR_DUPLICATE_SIMILARITY = 0.92
REQUIRED_KNOWLEDGE_REVIEW_FIELDS: tuple[str, ...] = (
    "expression_signature",
    "parameter_family",
    "behavior_cluster_id",
)


def build_homogeneity_report(
    pool_records: list[dict[str, Any]],
    behavior_snapshot: dict[str, Any],
    *,
    source_task_id: str | None = None,
    crowded_cluster_size: int = DEFAULT_CROWDED_CLUSTER_SIZE,
    crowded_limit: int = 12,
) -> dict[str, Any]:
    """Summarize library crowding and propose novelty-seeking research targets."""
    behavior_factors = behavior_snapshot.get("factors", {}) if behavior_snapshot else {}
    task_pool = [
        record
        for record in pool_records
        if source_task_id is None or str(record.get("source_task_id")) == source_task_id
    ]
    global_records = {str(record.get("factor_id")): record for record in pool_records}
    task_records = {str(record.get("factor_id")): record for record in task_pool}
    cluster_members: dict[str, list[str]] = defaultdict(list)
    cluster_mechanisms: dict[str, Counter[str]] = defaultdict(Counter)
    mechanism_counts: Counter[str] = Counter()
    mechanism_clusters: dict[str, set[str]] = defaultdict(set)
    unevaluated = 0

    for record in task_pool:
        factor_id = str(record.get("factor_id"))
        mechanism = _record_mechanism(record)
        mechanism_counts[mechanism] += 1
        evidence = behavior_factors.get(factor_id, {})
        cluster_id = evidence.get("behavior_cluster_id")
        if not cluster_id:
            unevaluated += 1
            continue
        cluster = str(cluster_id)
        cluster_members[cluster].append(factor_id)
        cluster_mechanisms[cluster][mechanism] += 1
        mechanism_clusters[mechanism].add(cluster)

    crowded_clusters = []
    for cluster_id, members in cluster_members.items():
        if len(members) < crowded_cluster_size:
            continue
        leader_id = _cluster_leader(members, behavior_factors, task_records, global_records)
        crowded_clusters.append(
            {
                "cluster_id": cluster_id,
                "size": len(members),
                "leader_factor_id": leader_id,
                "dominant_mechanism": cluster_mechanisms[cluster_id].most_common(1)[0][0],
                "member_sample": members[:8],
                "action": "FOLD_TO_LEADER_AND_AVOID_PARAMETER_VARIANTS",
            }
        )
    crowded_clusters.sort(key=lambda item: (-int(item["size"]), item["cluster_id"]))

    mechanism_profile = []
    for mechanism, count in mechanism_counts.items():
        cluster_count = len(mechanism_clusters.get(mechanism, set()))
        crowding_ratio = count / max(1, cluster_count)
        mechanism_profile.append(
            {
                "mechanism": mechanism,
                "factor_count": count,
                "behavior_cluster_count": cluster_count,
                "crowding_ratio": round(float(crowding_ratio), 4),
                "novelty_priority": _novelty_priority(count, cluster_count),
            }
        )
    mechanism_profile.sort(
        key=lambda item: (
            -float(item["novelty_priority"]),
            int(item["factor_count"]),
            item["mechanism"],
        )
    )
    target_mechanisms = [
        item["mechanism"]
        for item in mechanism_profile
        if float(item["novelty_priority"]) > 0
    ][:6]
    avoid_mechanisms = [
        item["mechanism"]
        for item in sorted(
            mechanism_profile,
            key=lambda value: (
                -float(value["crowding_ratio"]),
                -int(value["factor_count"]),
            ),
        )
        if float(item["crowding_ratio"]) >= 4.0
        and int(item["factor_count"]) >= crowded_cluster_size
    ][:6]

    return {
        "protocol": HOMOGENEITY_PROTOCOL,
        "source_task_id": source_task_id,
        "factor_count": len(task_pool),
        "behavior_evaluated_count": len(task_pool) - unevaluated,
        "behavior_unevaluated_count": unevaluated,
        "cluster_count": len(cluster_members),
        "crowded_cluster_size": crowded_cluster_size,
        "crowded_cluster_count": len(crowded_clusters),
        "crowded_clusters": crowded_clusters[:crowded_limit],
        "mechanism_profile": mechanism_profile,
        "target_mechanisms": target_mechanisms,
        "avoid_mechanisms": avoid_mechanisms,
        "llm_directive": {
            "primary_rule": (
                "Propose a factor that opens a new behavior cluster or a sparse mechanism; "
                "do not submit parameter-only variants of crowded cluster leaders."
            ),
            "must_avoid_cluster_ids": [
                item["cluster_id"] for item in crowded_clusters[:crowded_limit]
            ],
            "preferred_mechanisms": target_mechanisms,
            "discouraged_mechanisms": avoid_mechanisms,
            "admission_hint": (
                "A candidate with high signal similarity to an existing crowded cluster will "
                "be retained only as observation unless it has exceptional long-only evidence."
            ),
        },
    }


def candidate_homogeneity_metrics(
    *,
    factor_id: str,
    candidate_mechanism: str,
    candidate_fields: set[str],
    pool_records: list[dict[str, Any]],
    behavior_snapshot: dict[str, Any],
    redundancy_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Turn online nearest-neighbor evidence into admission metrics."""
    del candidate_fields
    behavior_factors = behavior_snapshot.get("factors", {}) if behavior_snapshot else {}
    nearest_id = redundancy_metrics.get("library_signal_correlation_peer")
    nearest_similarity = _float(
        redundancy_metrics.get("library_signal_correlation_max"),
        default=0.0,
    )
    online_cluster_id = redundancy_metrics.get("online_behavior_cluster_id")
    online_cluster_size = int(_float(redundancy_metrics.get("online_behavior_cluster_size"), 1.0))
    redundancy_label = str(redundancy_metrics.get("online_behavior_redundancy", "UNASSESSED"))
    peer = behavior_factors.get(str(nearest_id), {}) if nearest_id else {}
    cluster_id = str(online_cluster_id or peer.get("behavior_cluster_id") or "")
    cluster_size = max(
        online_cluster_size,
        int(_float(peer.get("behavior_cluster_size"), 1.0)),
    )
    records = {str(record.get("factor_id")): record for record in pool_records}
    peer_record = records.get(str(nearest_id)) if nearest_id else None
    peer_mechanism = _record_mechanism(peer_record) if peer_record else None
    same_mechanism = bool(peer_mechanism and peer_mechanism == candidate_mechanism)
    crowded = cluster_size >= DEFAULT_CROWDED_CLUSTER_SIZE
    near_duplicate = nearest_similarity >= DEFAULT_NEAR_DUPLICATE_SIMILARITY
    substitute = (
        nearest_similarity >= DEFAULT_SUBSTITUTE_SIMILARITY
        or redundancy_label in {"SUBSTITUTE", "NEAR_DUPLICATE"}
    )
    gate_passed = not (
        near_duplicate
        or (crowded and substitute)
        or (crowded and same_mechanism and nearest_similarity >= 0.74)
    )
    novelty_score = max(
        0.0,
        min(
            1.0,
            1.0
            - max(0.0, nearest_similarity)
            + (0.18 if not same_mechanism else 0.0)
            + (0.12 if not crowded else -0.12),
        ),
    )
    return {
        "homogeneity_protocol": HOMOGENEITY_PROTOCOL,
        "homogeneity_gate_passed": gate_passed,
        "homogeneity_gate_failure": None if gate_passed else _failure_reason(
            near_duplicate=near_duplicate,
            crowded=crowded,
            substitute=substitute,
            same_mechanism=same_mechanism,
        ),
        "homogeneity_novelty_score": round(float(novelty_score), 6),
        "homogeneity_cluster_id": cluster_id or None,
        "homogeneity_cluster_size": cluster_size,
        "homogeneity_nearest_factor_id": nearest_id,
        "homogeneity_nearest_similarity": round(float(nearest_similarity), 6),
        "homogeneity_nearest_mechanism": peer_mechanism,
        "homogeneity_same_mechanism_as_nearest": same_mechanism,
        "homogeneity_crowded_cluster": crowded,
        "homogeneity_redundancy_label": redundancy_label,
        "homogeneity_policy": {
            "crowded_cluster_size": DEFAULT_CROWDED_CLUSTER_SIZE,
            "substitute_similarity": DEFAULT_SUBSTITUTE_SIMILARITY,
            "near_duplicate_similarity": DEFAULT_NEAR_DUPLICATE_SIMILARITY,
        },
    }


def factor_homogeneity_integrity(
    pool_records: list[dict[str, Any]],
    knowledge_records: list[dict[str, Any]],
    *,
    sample_limit: int = 25,
) -> dict[str, Any]:
    """Report whether stored factor knowledge is usable for library-wide homogeneity controls."""
    pool_ids = {str(record.get("factor_id")) for record in pool_records if record.get("factor_id")}
    knowledge_by_id = {
        str(record.get("factor_id")): record
        for record in knowledge_records
        if record.get("factor_id")
    }
    missing_knowledge = sorted(pool_ids - set(knowledge_by_id))
    field_counts = {
        "canonical_mechanism": 0,
        **{field: 0 for field in REQUIRED_KNOWLEDGE_REVIEW_FIELDS},
    }
    missing_fields_by_factor: dict[str, list[str]] = {}

    for factor_id in sorted(pool_ids):
        knowledge = knowledge_by_id.get(factor_id)
        if not knowledge:
            missing_fields_by_factor[factor_id] = [
                "knowledge_record",
                "canonical_mechanism",
                *REQUIRED_KNOWLEDGE_REVIEW_FIELDS,
            ]
            continue
        missing_fields: list[str] = []
        if _present(knowledge.get("canonical_mechanism")):
            field_counts["canonical_mechanism"] += 1
        else:
            missing_fields.append("canonical_mechanism")
        review = knowledge.get("review") or {}
        for field in REQUIRED_KNOWLEDGE_REVIEW_FIELDS:
            if _present(review.get(field)):
                field_counts[field] += 1
            else:
                missing_fields.append(field)
        if missing_fields:
            missing_fields_by_factor[factor_id] = missing_fields

    factor_count = len(pool_ids)
    field_coverage = {
        field: {
            "count": count,
            "ratio": round(count / factor_count, 6) if factor_count else 1.0,
        }
        for field, count in field_counts.items()
    }
    complete = not missing_knowledge and not missing_fields_by_factor
    return {
        "protocol": HOMOGENEITY_INTEGRITY_PROTOCOL,
        "factor_count": factor_count,
        "knowledge_count": len(knowledge_by_id),
        "complete": complete,
        "stale": not complete,
        "missing_knowledge_count": len(missing_knowledge),
        "missing_knowledge_factor_ids": missing_knowledge[:sample_limit],
        "missing_field_count": len(missing_fields_by_factor),
        "missing_field_factor_ids": list(missing_fields_by_factor)[:sample_limit],
        "missing_fields_by_factor": dict(
            list(missing_fields_by_factor.items())[:sample_limit]
        ),
        "field_coverage": field_coverage,
        "required_review_fields": list(REQUIRED_KNOWLEDGE_REVIEW_FIELDS),
        "recommended_job_type": None if complete else "factor_homogeneity_backfill",
    }


def _record_mechanism(record: dict[str, Any] | None) -> str:
    if not record:
        return "OTHER_INTERPRETABLE"
    proposal = record.get("proposal") or {}
    fields = expression_fields(proposal.get("expression"))
    return normalize_mechanism(
        proposal.get("canonical_mechanism"),
        default=classify_mechanism(
            fields=fields,
            family=str(record.get("family", "")),
            name=str(record.get("name", "")),
            hypothesis=str(proposal.get("hypothesis", "")),
        ),
    )


def _cluster_leader(
    members: list[str],
    behavior_factors: dict[str, dict[str, Any]],
    task_records: dict[str, dict[str, Any]],
    global_records: dict[str, dict[str, Any]],
) -> str | None:
    for factor_id in members:
        if behavior_factors.get(factor_id, {}).get("behavior_cluster_role") == "LEADER":
            return factor_id
    ranked = sorted(
        members,
        key=lambda factor_id: _leader_score(
            task_records.get(factor_id) or global_records.get(factor_id) or {}
        ),
        reverse=True,
    )
    return ranked[0] if ranked else None


def _leader_score(record: dict[str, Any]) -> tuple[float, float, float]:
    metrics = record.get("metrics") or {}
    return (
        _float(metrics.get("recent_long_only_sharpe_ratio"), -100.0),
        _float(metrics.get("long_only_sharpe_ratio"), -100.0),
        _float(metrics.get("recent_long_only_simple_annual_return"), -100.0),
    )


def _novelty_priority(factor_count: int, cluster_count: int) -> float:
    if factor_count <= 0:
        return 0.0
    if cluster_count <= 0:
        return 1.0
    if factor_count < 3:
        return 0.8
    crowding_ratio = factor_count / max(1, cluster_count)
    return max(0.0, min(1.0, 1.0 - (crowding_ratio / 5.0)))


def _failure_reason(
    *,
    near_duplicate: bool,
    crowded: bool,
    substitute: bool,
    same_mechanism: bool,
) -> str:
    if near_duplicate:
        return "NEAR_DUPLICATE_SIGNAL"
    if crowded and substitute:
        return "CROWDED_CLUSTER_SUBSTITUTE"
    if crowded and same_mechanism:
        return "CROWDED_SAME_MECHANISM"
    return "INSUFFICIENT_BEHAVIOR_NOVELTY"


def _float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, int | float) and math.isfinite(float(value)):
        return float(value)
    return default


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, float):
        return math.isfinite(value)
    return True
