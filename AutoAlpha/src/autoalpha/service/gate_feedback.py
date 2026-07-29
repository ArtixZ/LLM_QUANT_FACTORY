from __future__ import annotations

from typing import Any

from autoalpha.service.store import ServiceStore


def gate_feedback_policy(store: ServiceStore) -> dict[str, Any]:
    snapshot = store.materialized_snapshot("gate_funnel_diagnostics")
    payload = snapshot["payload"] if snapshot else {}
    actions = list(payload.get("operator_actions") or [])
    action_ids = {str(item.get("action")) for item in actions}
    root_cause_intensity = _root_cause_intensity(payload)
    adjustments = {
        "construction": {},
        "objective": {},
        "engine": {},
        "budget": {},
    }
    rationales: list[str] = []
    recommendations: list[dict[str, Any]] = []
    if "REPAIR_WALK_FORWARD_CAPACITY_OR_COVERAGE" in action_ids:
        adjustments["objective"].update(
            {
                "minimum_coverage": {"floor": 0.85},
                "minimum_positive_fold_fraction": {"floor": 0.60},
            }
        )
        rationale = "Gate funnel indicates validation folds, sample size, or coverage bind."
        rationales.append(rationale)
        recommendations.append(
            _recommendation(
                "REPAIR_WALK_FORWARD_CAPACITY_OR_COVERAGE",
                "VALIDATION_DATA_OR_FOLD_CAPACITY",
                rationale,
                root_cause_intensity,
                {
                    "minimum_coverage": 0.85,
                    "minimum_positive_fold_fraction": 0.60,
                },
            )
        )
    if "REDUCE_EFFECTIVE_TRIAL_COUNT_AND_DEDUPLICATE_SEARCH_SPACE" in action_ids:
        adjustments["construction"].update(
            {
                "candidate_pool_limit": {"ceiling": 40},
                "maximum_same_family": {"ceiling": 1},
                "maximum_same_semantic_cluster": {"ceiling": 1},
            }
        )
        adjustments["objective"].update(
            {
                "maximum_duplicate_semantic_factors": {"ceiling": 0},
                "minimum_deflated_sharpe_probability": {"floor": 0.55},
            }
        )
        adjustments["budget"].update(
            {
                "maximum_experiments": {"ceiling": 40},
                "maximum_evaluations": {"ceiling": 120},
            }
        )
        rationale = "Gate funnel indicates DSR or multiple-testing penalties are binding."
        rationales.append(rationale)
        recommendations.append(
            _recommendation(
                "REDUCE_EFFECTIVE_TRIAL_COUNT_AND_DEDUPLICATE_SEARCH_SPACE",
                "TRIAL_BUDGET_AND_OVERFITTING_PENALTY",
                rationale,
                root_cause_intensity,
                {
                    "candidate_pool_limit_ceiling": 40,
                    "maximum_same_family": 1,
                    "maximum_same_semantic_cluster": 1,
                    "maximum_evaluations_ceiling": 120,
                },
            )
        )
    if "FORCE_CROSS_MECHANISM_AND_CROSS_CLUSTER_SELECTION" in action_ids:
        adjustments["construction"].update(
            {
                "maximum_same_family": {"ceiling": 1},
                "maximum_same_semantic_cluster": {"ceiling": 1},
            }
        )
        adjustments["objective"].update(
            {
                "maximum_factor_correlation": {"ceiling": 0.60},
                "minimum_effective_mechanisms": {"floor": 2.0},
                "maximum_mechanism_weight": {"ceiling": 0.55},
            }
        )
        adjustments["engine"].update({"cluster_correlation_threshold": {"ceiling": 0.70}})
        rationale = "Gate funnel indicates factor independence constraints are binding."
        rationales.append(rationale)
        recommendations.append(
            _recommendation(
                "FORCE_CROSS_MECHANISM_AND_CROSS_CLUSTER_SELECTION",
                "FACTOR_INDEPENDENCE_INSUFFICIENT",
                rationale,
                root_cause_intensity,
                {
                    "maximum_factor_correlation": 0.60,
                    "minimum_effective_mechanisms": 2.0,
                    "maximum_mechanism_weight": 0.55,
                },
            )
        )
    if "SWITCH_OBJECTIVE_TO_DRAWDOWN_AND_TAIL_RISK_FIRST" in action_ids:
        adjustments["objective"].update(
            {
                "maximum_drawdown": {"ceiling": 0.18},
                "minimum_worst_fold_sharpe": {"floor": -0.20},
            }
        )
        rationale = "Gate funnel indicates risk constraints bind before production admission."
        rationales.append(rationale)
        recommendations.append(
            _recommendation(
                "SWITCH_OBJECTIVE_TO_DRAWDOWN_AND_TAIL_RISK_FIRST",
                "RISK_CONSTRAINT_BREACH",
                rationale,
                root_cause_intensity,
                {
                    "profile": "DRAWDOWN_FIRST",
                    "maximum_drawdown": 0.18,
                    "minimum_worst_fold_sharpe": -0.20,
                },
            )
        )
    profile_override = (
        "DIVERSIFICATION_FIRST"
        if "FORCE_CROSS_MECHANISM_AND_CROSS_CLUSTER_SELECTION" in action_ids
        else "DRAWDOWN_FIRST"
        if "SWITCH_OBJECTIVE_TO_DRAWDOWN_AND_TAIL_RISK_FIRST" in action_ids
        else None
    )
    return {
        "protocol": "GATE_FUNNEL_FEEDBACK_POLICY_V1",
        "source_protocol": payload.get("protocol"),
        "source_fingerprint": snapshot["fingerprint"] if snapshot else None,
        "source_updated_at": snapshot["updated_at"] if snapshot else None,
        "actions": actions,
        "action_ids": sorted(action_ids),
        "profile_override": profile_override,
        "recommended_profile_reason": _profile_reason(profile_override, root_cause_intensity),
        "root_cause_intensity": root_cause_intensity,
        "recommendations": sorted(
            recommendations, key=lambda item: item["evidence_count"], reverse=True
        ),
        "adjustments": adjustments,
        "rationales": rationales,
        "active": bool(actions),
    }


def _root_cause_intensity(payload: dict[str, Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in payload.get("root_causes") or []:
        if isinstance(item, dict):
            key = item.get("key")
            count = item.get("count")
        elif isinstance(item, list | tuple) and len(item) >= 2:
            key, count = item[0], item[1]
        else:
            continue
        if key is None:
            continue
        try:
            result[str(key)] = int(count)
        except (TypeError, ValueError):
            continue
    for action in payload.get("operator_actions") or []:
        if not isinstance(action, dict):
            continue
        root = _action_root_cause(str(action.get("action") or ""))
        if not root or root in result:
            continue
        try:
            result[root] = int(action.get("evidence_count") or 0)
        except (TypeError, ValueError):
            result[root] = 0
    return result


def _action_root_cause(action: str) -> str | None:
    return {
        "REPAIR_WALK_FORWARD_CAPACITY_OR_COVERAGE": "VALIDATION_DATA_OR_FOLD_CAPACITY",
        "REDUCE_EFFECTIVE_TRIAL_COUNT_AND_DEDUPLICATE_SEARCH_SPACE": (
            "TRIAL_BUDGET_AND_OVERFITTING_PENALTY"
        ),
        "FORCE_CROSS_MECHANISM_AND_CROSS_CLUSTER_SELECTION": (
            "FACTOR_INDEPENDENCE_INSUFFICIENT"
        ),
        "SWITCH_OBJECTIVE_TO_DRAWDOWN_AND_TAIL_RISK_FIRST": "RISK_CONSTRAINT_BREACH",
        "EXPAND_OR_RESEED_FACTOR_MECHANISMS": "SEARCH_ALPHA_STRENGTH_INSUFFICIENT",
    }.get(action)


def _recommendation(
    action: str,
    root_cause: str,
    rationale: str,
    root_cause_intensity: dict[str, int],
    suggested_values: dict[str, Any],
) -> dict[str, Any]:
    return {
        "action": action,
        "root_cause": root_cause,
        "evidence_count": int(root_cause_intensity.get(root_cause, 0)),
        "rationale": rationale,
        "suggested_values": suggested_values,
    }


def _profile_reason(profile: str | None, root_cause_intensity: dict[str, int]) -> str | None:
    if profile == "DIVERSIFICATION_FIRST":
        count = root_cause_intensity.get("FACTOR_INDEPENDENCE_INSUFFICIENT", 0)
        return f"因子独立性不足出现 {count} 次，优先跨机制和跨搜索簇组合。"
    if profile == "DRAWDOWN_FIRST":
        count = root_cause_intensity.get("RISK_CONSTRAINT_BREACH", 0)
        return f"风险约束触发 {count} 次，优先降低回撤和尾部风险。"
    return None


def apply_gate_feedback(
    values: dict[str, Any],
    rules: dict[str, Any],
    *,
    explicit_fields: set[str] | None = None,
) -> dict[str, Any]:
    explicit_fields = explicit_fields or set()
    result = dict(values)
    for key, rule in rules.items():
        if key in explicit_fields:
            continue
        if key not in result or result[key] is None:
            continue
        value = result[key]
        if not isinstance(value, int | float):
            continue
        if "floor" in rule:
            result[key] = max(value, rule["floor"])
        if "ceiling" in rule:
            result[key] = min(result[key], rule["ceiling"])
    return result


def append_gate_feedback_notes(notes: str, policy: dict[str, Any]) -> str:
    if not policy.get("active"):
        return notes.strip()
    action_text = ", ".join(policy.get("action_ids") or [])
    prefix = f"[gate-feedback:{policy.get('protocol')}] {action_text}".strip()
    body = notes.strip()
    return f"{body}\n{prefix}".strip() if body else prefix
