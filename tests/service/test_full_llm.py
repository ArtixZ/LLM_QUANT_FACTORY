from __future__ import annotations

import asyncio
from typing import Any

from autoalpha.service.full_llm import (
    FACTOR_LIBRARIAN,
    FULL_LLM_ROLES,
    FullLLMResearchTeam,
    categorical_research_feedback,
    evaluate_falsification_plan,
    role_catalog,
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
                        {"factor_id": "F_ALLOWED", "relation": "RELATED"},
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
            library_context=[
                {"factor_id": "F_ALLOWED", "name": "existing", "proposal": {}}
            ],
            data_context={"first_trade_date": "2020-01-01"},
        )
    )

    artifact = outcomes[FACTOR_LIBRARIAN].artifact
    assert artifact["canonical_mechanism"] == "OTHER_INTERPRETABLE"
    assert artifact["tags"] == ["price-volume"]
    assert [item["factor_id"] for item in artifact["related_factors"]] == ["F_ALLOWED"]


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

    outcomes = asyncio.run(
        team.pre_evaluation(candidate={}, library_context=[], data_context={})
    )

    assert len(outcomes) == 3
    assert all(outcome.status == "FAILED" for outcome in outcomes.values())
    assert all(outcome.artifact == {"advisory_available": False} for outcome in outcomes.values())
