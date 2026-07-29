from __future__ import annotations

from autoalpha.service.gate_feedback import apply_gate_feedback, gate_feedback_policy
from autoalpha.service.store import ServiceStore


def test_gate_feedback_policy_translates_operator_actions_to_search_constraints(tmp_path) -> None:
    store = ServiceStore(tmp_path / "autoalpha.sqlite3")
    store.upsert_materialized_snapshot(
        "gate_funnel_diagnostics",
        {
            "protocol": "AUTOALPHA_GATE_FUNNEL_DIAGNOSTICS_V2",
            "operator_actions": [
                {"action": "REDUCE_EFFECTIVE_TRIAL_COUNT_AND_DEDUPLICATE_SEARCH_SPACE"},
                {"action": "FORCE_CROSS_MECHANISM_AND_CROSS_CLUSTER_SELECTION"},
                {"action": "SWITCH_OBJECTIVE_TO_DRAWDOWN_AND_TAIL_RISK_FIRST"},
            ],
            "root_causes": [
                {"key": "TRIAL_BUDGET_AND_OVERFITTING_PENALTY", "count": 11},
                {"key": "FACTOR_INDEPENDENCE_INSUFFICIENT", "count": 7},
                {"key": "RISK_CONSTRAINT_BREACH", "count": 5},
            ],
        },
    )

    policy = gate_feedback_policy(store)

    assert policy["active"] is True
    assert policy["profile_override"] == "DIVERSIFICATION_FIRST"
    assert policy["adjustments"]["construction"]["maximum_same_family"] == {"ceiling": 1}
    assert policy["adjustments"]["objective"]["maximum_drawdown"] == {"ceiling": 0.18}
    assert policy["adjustments"]["engine"]["cluster_correlation_threshold"] == {
        "ceiling": 0.70
    }
    assert policy["adjustments"]["budget"]["maximum_evaluations"] == {"ceiling": 120}
    assert policy["root_cause_intensity"]["TRIAL_BUDGET_AND_OVERFITTING_PENALTY"] == 11
    assert policy["recommended_profile_reason"] == (
        "因子独立性不足出现 7 次，优先跨机制和跨搜索簇组合。"
    )
    assert policy["recommendations"][0] == {
        "action": "REDUCE_EFFECTIVE_TRIAL_COUNT_AND_DEDUPLICATE_SEARCH_SPACE",
        "root_cause": "TRIAL_BUDGET_AND_OVERFITTING_PENALTY",
        "evidence_count": 11,
        "rationale": "Gate funnel indicates DSR or multiple-testing penalties are binding.",
        "suggested_values": {
            "candidate_pool_limit_ceiling": 40,
            "maximum_same_family": 1,
            "maximum_same_semantic_cluster": 1,
            "maximum_evaluations_ceiling": 120,
        },
    }


def test_apply_gate_feedback_respects_explicit_fields() -> None:
    values = {"maximum_drawdown": 0.30, "minimum_coverage": 0.80}
    rules = {
        "maximum_drawdown": {"ceiling": 0.18},
        "minimum_coverage": {"floor": 0.85},
    }

    adjusted = apply_gate_feedback(values, rules, explicit_fields={"maximum_drawdown"})

    assert adjusted["maximum_drawdown"] == 0.30
    assert adjusted["minimum_coverage"] == 0.85
