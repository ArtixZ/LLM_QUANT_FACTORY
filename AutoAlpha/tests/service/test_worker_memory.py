from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from autoalpha.config import ResearchConfig
from autoalpha.dsl.expression import FactorDefinition, field
from autoalpha.service.store import ServiceStore
from autoalpha.service.worker import (
    ContinuousResearchWorker,
    SecretVault,
    _expression_structure,
    _frozen_portfolio_hash,
    _is_public_feasibility_recovery,
    _memory_summary,
    _multiple_testing_adjustments,
    _select_research_memories,
)


def _expression(window: int) -> dict:
    return {
        "operator": "cs_zscore",
        "arguments": [
            {
                "operator": "rolling_mean",
                "arguments": [
                    {
                        "operator": "field",
                        "arguments": [],
                        "parameters": {"name": "amount"},
                    }
                ],
                "parameters": {"window": window},
            }
        ],
        "parameters": {},
    }


def test_expression_structure_detects_parameter_only_variants() -> None:
    assert _expression_structure(_expression(10)) == _expression_structure(_expression(20))


def test_memory_summary_preserves_best_portfolio_failure() -> None:
    metrics = {
        "long_only_sharpe_ratio": 0.8,
        "long_only_simple_annual_return": 0.08,
        "long_only_annual_turnover": 9.0,
        "long_only_coverage": 0.96,
        "sharpe_ratio": 1.2,
        "simple_annual_return": 0.1,
        "rank_ic_mean": 0.02,
        "annual_turnover": 18.0,
        "coverage": 0.95,
        "exploratory_gate_failures": ["incremental_drawdown"],
        "portfolio_action": "HOLD",
        "portfolio_action_accepted": False,
        "portfolio_action_gate_failures": ["portfolio_value"],
        "portfolio_active_factor_ids": ["F_active"],
        "portfolio_option_diagnostics": [
            {
                "action": "ADD",
                "weights": [0.9, 0.1],
                "utility_change": 0.04,
                "failed_gates": ["utility_improvement"],
                "metrics": {
                    "portfolio_sharpe_change": 0.12,
                    "portfolio_annual_return_change": 0.01,
                    "portfolio_annual_turnover": 21.0,
                    "portfolio_max_factor_correlation": 0.3,
                },
            }
        ],
    }

    memory = json.loads(
        _memory_summary(
            "candidate",
            "liquidity",
            metrics,
            "MULTIFACTOR_HOLD_RESEARCH_ONLY_DATA_BLOCKED",
            proposal={"expression": _expression(20)},
        )
    )

    assert memory["portfolio"]["failed_gates"] == ["portfolio_value"]
    assert memory["portfolio"]["best_option"]["weights"] == [0.9, 0.1]
    assert "rolling_mean" in memory["expression_signature"]
    assert memory["single_factor"]["sharpe"] == 0.8
    assert memory["single_factor"]["alpha_diagnostic"]["sharpe"] == 1.2


def test_memory_summary_prioritizes_accepted_option_over_rejected_utility() -> None:
    metrics = {
        "portfolio_action": "ADD",
        "portfolio_action_accepted": True,
        "portfolio_option_diagnostics": [
            {
                "action": "ADD",
                "accepted": True,
                "weights": [0.8, 0.2],
                "utility_change": 0.6,
                "failed_gates": [],
                "metrics": {},
            },
            {
                "action": "REPLACE",
                "accepted": False,
                "weights": [0.5, 0.5],
                "utility_change": 1.3,
                "failed_gates": ["portfolio_value"],
                "metrics": {},
            },
        ],
    }

    memory = json.loads(_memory_summary("candidate", "test", metrics, "UPDATED"))

    assert memory["portfolio"]["best_option"]["action"] == "ADD"
    assert memory["portfolio"]["best_option"]["weights"] == [0.8, 0.2]


def test_multiple_testing_and_frozen_portfolio_identity_are_deterministic() -> None:
    folds = [
        {
            "validation_start": f"{year}-01-02",
            "annual_return": 0.01 * (year - 2014),
        }
        for year in range(2015, 2025)
    ]
    current = {
        "long_only_net_return_hac_p_value": 0.001,
        "long_only_walk_forward_folds": folds,
    }
    history = [
        {
            "research_generation": "older-generation",
            "long_only_net_return_hac_p_value": 0.002,
            "long_only_walk_forward_folds": [
                {**fold, "annual_return": fold["annual_return"] * 0.5} for fold in folds
            ],
        }
    ]

    adjusted = _multiple_testing_adjustments(current, history, generation="g1", alpha=0.10)
    factor = FactorDefinition("factor", "test", "test", field("amount"))
    first = _frozen_portfolio_hash((factor,), (1.0,), generation="g1")
    second = _frozen_portfolio_hash((factor,), (1.0,), generation="g1")

    assert adjusted["multiple_testing_fdr_passed"] is True
    assert adjusted["multiple_testing_family_size"] == 2
    assert adjusted["multiple_testing_scope"] == "CUMULATIVE_ALL_GENERATIONS"
    assert adjusted["multiple_testing_primary_basis"] == "A_SHARE_LONG_ONLY"
    assert 0 <= adjusted["probability_backtest_overfitting"] <= 1
    assert first == second
    assert len(first) == 64


def test_incomplete_feasibility_recovery_stays_out_of_holdout() -> None:
    transition = SimpleNamespace(
        accepted=True,
        evaluation=SimpleNamespace(
            metrics={
                "portfolio_feasibility_recovery": True,
                "portfolio_proposed_absolute_failures": ["annual_dispersion"],
            }
        ),
    )
    fully_feasible = SimpleNamespace(
        accepted=True,
        evaluation=SimpleNamespace(
            metrics={
                "portfolio_feasibility_recovery": True,
                "portfolio_proposed_absolute_failures": [],
            }
        ),
    )

    assert _is_public_feasibility_recovery(transition) is True
    assert _is_public_feasibility_recovery(fully_feasible) is False


def test_memory_selection_keeps_old_success_and_removes_legacy_pseudo_failure() -> None:
    memories = []
    for identifier in range(1, 26):
        content = {
            "name": f"factor-{identifier}",
            "family": f"family-{identifier % 4}",
            "single_factor": {"exploratory_failures": ["incremental_drawdown"]},
            "portfolio": {
                "accepted": identifier == 1,
                "best_option": None,
            },
        }
        memories.append(
            {
                "id": identifier,
                "iteration": identifier,
                "kind": "evaluation",
                "content": json.dumps(content),
            }
        )

    selected = _select_research_memories(memories, limit=8, recent=4)
    selected_by_id = {int(item["id"]): item for item in selected}

    assert 1 in selected_by_id
    assert {22, 23, 24, 25}.issubset(selected_by_id)
    old_success = json.loads(selected_by_id[1]["content"])
    assert old_success["single_factor"]["exploratory_failures"] == []
    assert old_success["feedback_version"] == "actionable_memory_v3"


def test_memory_selection_withholds_metrics_from_stale_protocol() -> None:
    content = {
        "name": "legacy winner",
        "family": "volume",
        "single_factor": {"sharpe": 9.0, "annual_return": 1.5},
        "portfolio": {"accepted": True, "active_factor_ids": ["factor-1"]},
        "protocol": {"version": "legacy-close-fill"},
    }
    selected = _select_research_memories(
        [{"id": 1, "iteration": 1, "kind": "evaluation", "content": json.dumps(content)}],
        current_protocol="next-open-v1",
    )

    memory = json.loads(selected[0]["content"])
    assert memory["single_factor"] == {
        "stale_protocol": True,
        "exploratory_failures": ["STALE_PROTOCOL_REEVALUATION_REQUIRED"],
    }
    assert memory["portfolio"]["accepted"] is False
    assert memory["protocol"]["promotion_evidence_allowed"] is False
    assert "sharpe" not in memory["single_factor"]


def test_research_context_revalues_legacy_incumbent_under_current_protocol(
    tmp_path: Path,
) -> None:
    store = ServiceStore(tmp_path / "service.sqlite3")
    config = ResearchConfig.from_toml(Path("config/research.toml"))
    factor = FactorDefinition("legacy", "test", "test", field("amount"))
    proposal = {
        "name": factor.name,
        "family": factor.family,
        "hypothesis": factor.hypothesis,
        "expected_direction": factor.expected_direction,
        "expression": factor.expression.to_dict(),
    }
    store.upsert_factor_pool(
        factor_id=factor.factor_id,
        source_iteration=1,
        proposal=proposal,
        metrics={"sharpe_ratio": 4.0},
        status="ACTIVE",
        status_reason="legacy",
    )
    store.record_portfolio_decision(
        run_id="run-1",
        iteration=1,
        action="BOOTSTRAP",
        candidate_id=factor.factor_id,
        removed_factor_id=None,
        accepted=True,
        reason="legacy",
        metrics={
            "portfolio_sharpe_ratio": 4.37,
            "portfolio_evaluation_protocol": config.governance.protocol_version,
        },
        members=[(factor.factor_id, 1.0)],
    )
    evaluator = SimpleNamespace(
        config=config,
        workspace=SimpleNamespace(blockers=()),
        evaluate_portfolio=lambda factors, weights: SimpleNamespace(
            metrics={"portfolio_sharpe_ratio": 3.81}
        ),
    )
    worker = ContinuousResearchWorker(
        store,
        SecretVault(api_key="test"),
        config_path=Path("config/research.toml"),
        artifact_root=tmp_path / "artifacts",
    )

    context = worker._research_program_context(2, evaluator, generation_id=config.generation)

    active = context["active_portfolio"]
    assert active["metrics"] == {"portfolio_sharpe_ratio": 3.81}
    assert active["legacy_portfolio_withheld"] is False
    assert active["stored_protocol"] == config.governance.protocol_version
    assert active["metrics_source"] == "current_protocol_revaluation"


def test_worker_freezes_stability_campaign_before_model_proposal(tmp_path: Path) -> None:
    store = ServiceStore(tmp_path / "service.sqlite3")
    config = ResearchConfig.from_toml(Path("config/research.toml"))
    store.ensure_generation(
        generation_id=config.generation,
        protocol_version=config.governance.protocol_version,
        maximum_candidates=10,
        maximum_holdout_attempts=2,
    )
    worker = ContinuousResearchWorker(
        store,
        SecretVault(api_key="test"),
        config_path=Path("config/research.toml"),
        artifact_root=tmp_path / "artifacts",
    )
    program = {
        "active_portfolio": {
            "metrics": {
                "portfolio_annual_return_dispersion": 0.22,
                "portfolio_walk_forward_worst_sharpe": 0.2,
                "portfolio_coverage": 0.95,
                "portfolio_capacity_cny": 100_000_000.0,
                "portfolio_annual_turnover": 10.0,
                "portfolio_cost_stress_net_ir": 1.0,
                "portfolio_max_drawdown": -0.08,
                "portfolio_max_factor_correlation": 0.25,
            }
        }
    }

    context = worker._prepare_direction_campaign("run-1", 1, config.generation, program, config)

    assert context["direction"] == "RESTORE_STABILITY"
    assert context["attempt_number"] == 1
    assert context["maximum_attempts"] == 3
    assert context["extension_allowed"] is False
    campaign = store.active_direction_campaign(config.generation)
    assert campaign is not None and campaign["attempts_used"] == 1


def test_canonical_family_unifies_label_variants() -> None:
    from autoalpha.dsl.expression import canonical_family

    variants = ["Valuation", "valuation", "VALUATION"]
    assert {canonical_family(name) for name in variants} == {"valuation"}
    assert canonical_family("TurnoverLiquidity") == "turnover_liquidity"
    assert canonical_family("TURNOVER_LIQUIDITY") == "turnover_liquidity"
    assert canonical_family("Order Flow") == "order_flow"
    assert canonical_family("order-flow") == "order_flow"
    assert canonical_family("  ") == "unknown"


def test_pool_expression_signatures_deduplicate_and_cap() -> None:
    from autoalpha.service.worker import _pool_expression_signatures

    pool = [
        {"proposal": {"expression": _expression(5)}},
        {"proposal": {"expression": _expression(5)}},
        {"proposal": {"expression": _expression(10)}},
        {"proposal": {}},
        {"proposal": {"expression": _expression(20)}},
    ]

    signatures = _pool_expression_signatures(pool, limit=2)

    assert len(signatures) == 2
    assert signatures[0] != signatures[1]
    assert all("rolling_mean" in signature for signature in signatures)
