from __future__ import annotations

import asyncio
from typing import Any

from autoalpha.service.full_llm import (
    FACTOR_LIBRARIAN,
    FALSIFICATION_DESIGNER,
    FULL_LLM_ROLES,
    FullLLMResearchTeam,
    _data_contract_context,
    categorical_research_feedback,
    evaluate_falsification_plan,
    role_catalog,
    summarize_research_team_domains,
)
from autoalpha.service.openai_client import GeneratedAnalysis


class StubAnalysisClient:
    async def analyze(
        self,
        *,
        role: str,
        system_prompt: str,
        context: dict[str, Any],
        required_keys: frozenset[str] | set[str],
    ) -> GeneratedAnalysis:
        del system_prompt
        artifact = {key: [] for key in required_keys}
        if role == FACTOR_LIBRARIAN:
            artifact.update(
                {
                    "canonical_mechanism": "not-a-real-mechanism",
                    "mechanism_summary": "Volume and price interact.",
                    "tags": ["Price Volume", "price volume"],
                    "related_factors": [
                        {
                            "factor_id": "F_ALLOWED",
                            "relation": "RELATED",
                            "confidence": "low",
                        },
                        {"factor_id": "F_HIDDEN", "relation": "RELATED"},
                    ],
                    "distinguishing_features": [],
                }
            )
        return GeneratedAnalysis(
            role=role,
            artifact=artifact,
            usage={"total_tokens": 10},
            prompt_hash="prompt",
            response_hash="response",
        )


class BrokenAnalysisClient:
    async def analyze(self, **_: Any) -> GeneratedAnalysis:
        raise RuntimeError("provider-specific decoding error")


def test_role_catalog_has_six_advisory_roles() -> None:
    catalog = role_catalog()

    assert {item["role"] for item in catalog} == set(FULL_LLM_ROLES)
    assert all(item["decision_authority"] == "ADVISORY_ONLY" for item in catalog)
    assert all(item["domains"] for item in catalog)


def test_research_team_domain_matrix_summarizes_structured_artifacts() -> None:
    matrix = summarize_research_team_domains(
        [
            {
                "role": "FACTOR_LIBRARIAN",
                "stage": "PRE_EVALUATION",
                "status": "COMPLETED",
                "candidate_id": "F_1",
                "run_id": "run-one",
                "iteration": 1,
                "created_at": "2026-07-29T01:00:00Z",
                "artifact": {
                    "canonical_mechanism": "PRICE_REVERSAL",
                    "mechanism_summary": "Short horizon reversal",
                },
            },
            {
                "role": "TCA_PAPER_OBSERVER",
                "stage": "POST_EVALUATION",
                "status": "FAILED",
                "candidate_id": "F_2",
                "run_id": "run-two",
                "iteration": 2,
                "created_at": "2026-07-29T01:01:00Z",
                "artifact": {"advisory_available": False},
                "error": "provider unavailable",
            },
        ]
    )
    domains = {item["domain"]: item for item in matrix["domains"]}

    assert matrix["protocol"] == "LLM_RESEARCH_TEAM_DOMAIN_MATRIX_V1"
    assert domains["RESEARCHER"]["status"] == "ACTIVE"
    assert domains["RESEARCHER"]["latest_headline"] == "PRICE_REVERSAL"
    assert domains["DATA_OFFICER"]["covered_roles"] == ["FACTOR_LIBRARIAN"]
    assert domains["TRADER"]["status"] == "FAILED_OPEN"
    assert domains["TRADER"]["latest_headline"] == "provider unavailable"
    assert domains["PORTFOLIO_MANAGER"]["status"] == "WAITING_FOR_ARTIFACTS"


def test_feedback_firewall_returns_categories_without_exact_metrics() -> None:
    feedback = categorical_research_feedback(
        {
            "sharpe_ratio": 4.37,
            "simple_annual_return": 0.3096,
            "exploratory_gate_failures": ["cost_stress", "walk_forward_worst_sharpe"],
            "portfolio_action_gate_failures": ["incremental_net_ir"],
        },
        decision="HOLD",
        candidate_eligible=False,
        portfolio_action="HOLD",
        portfolio_accepted=False,
    )

    serialized = str(feedback)
    assert "4.37" not in serialized
    assert "0.3096" not in serialized
    assert feedback["evidence_bands"]["execution_efficiency"] == "NEEDS_IMPROVEMENT"
    assert feedback["feedback_policy"] == "CATEGORICAL_PUBLIC_ONLY_NO_EXACT_METRICS"


def test_librarian_normalizes_mechanism_and_filters_unknown_relations() -> None:
    team = FullLLMResearchTeam(StubAnalysisClient())  # type: ignore[arg-type]

    outcomes = asyncio.run(
        team.pre_evaluation(
            candidate={"name": "factor", "expression": {"operator": "field"}},
            library_context=[{"factor_id": "F_ALLOWED", "name": "existing", "proposal": {}}],
            data_context={"first_trade_date": "2020-01-01"},
        )
    )

    artifact = outcomes[FACTOR_LIBRARIAN].artifact
    assert artifact["canonical_mechanism"] == "OTHER_INTERPRETABLE"
    assert artifact["tags"] == ["price-volume"]
    assert [item["factor_id"] for item in artifact["related_factors"]] == ["F_ALLOWED"]
    assert artifact["related_factors"][0]["confidence"] == 0.30


def test_falsification_results_use_only_deterministic_gate_categories() -> None:
    results = evaluate_falsification_plan(
        {
            "tests": [
                {"test_type": "COST_STRESS"},
                {"test_type": "PARAMETER_NEIGHBORHOOD"},
                {"test_type": "DELAY_SENSITIVITY"},
            ]
        },
        {"exploratory_gate_failures": ["cost_stress"]},
    )

    assert results == [
        {
            "test_type": "COST_STRESS",
            "status": "FAILED",
            "failure_categories": ["cost_stress"],
        },
        {
            "test_type": "PARAMETER_NEIGHBORHOOD",
            "status": "PASSED",
            "failure_categories": [],
        },
        {
            "test_type": "DELAY_SENSITIVITY",
            "status": "NOT_EVALUATED",
            "failure_categories": [],
        },
    ]


def test_unexpected_role_failure_is_audited_and_iteration_can_continue() -> None:
    team = FullLLMResearchTeam(BrokenAnalysisClient())  # type: ignore[arg-type]

    outcomes = asyncio.run(team.pre_evaluation(candidate={}, library_context=[], data_context={}))

    assert len(outcomes) == 3
    assert all(outcome.status == "FAILED" for outcome in outcomes.values())
    assert all(outcome.artifact == {"advisory_available": False} for outcome in outcomes.values())


def test_conditional_role_selection_invokes_only_requested_advisers() -> None:
    team = FullLLMResearchTeam(StubAnalysisClient())  # type: ignore[arg-type]

    outcomes = asyncio.run(
        team.pre_evaluation(
            candidate={},
            library_context=[],
            data_context={},
            roles=(FALSIFICATION_DESIGNER,),
        )
    )

    assert list(outcomes) == [FALSIFICATION_DESIGNER]


def test_llm_roles_receive_field_semantics_and_product_inventory() -> None:
    context = _data_contract_context(
        {
            "available_factor_fields": ["close", "turnover_rate"],
            "field_catalog": [{"name": "turnover_rate", "status": "RESEARCH_ELIGIBLE"}],
            "data_products": [{"dataset_id": "daily_basic"}],
            "data_policy": {"staged_fields_forbidden": True},
            "signal_timing": "END_OF_DAY_INFORMATION_MAY_TRADE_NEXT_SESSION_OPEN",
            "extended_data_experiment": {"enabled": True},
            "secret": "must-not-pass",
        }
    )

    assert context["field_catalog"][0]["name"] == "turnover_rate"
    assert context["data_products"][0]["dataset_id"] == "daily_basic"
    assert "secret" not in context
