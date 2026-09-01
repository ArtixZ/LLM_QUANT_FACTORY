from __future__ import annotations

from pathlib import Path

from autoalpha.agents.orchestrator import (
    ExecutionReport,
    ResearchContext,
    ResearchOrchestrator,
    ResearchProposal,
    ReviewDecision,
)
from autoalpha.dsl.expression import FactorDefinition, field, operation
from autoalpha.dsl.semantics import FieldDefinition, SemanticValidator
from autoalpha.governance.audit import HashChainAuditLog
from autoalpha.research.budget import ExperimentBudgetLedger
from autoalpha.research.gates import CandidateEvidence


def _evidence() -> CandidateEvidence:
    return CandidateEvidence(
        data_passed=True,
        semantic_passed=True,
        coverage=0.98,
        rank_ic_mean=0.01,
        rank_ic_hac_p_value=0.20,
        fdr_passed=True,
        positive_fold_fraction=0.8,
        worst_fold_rank_ic=-0.01,
        maximum_library_correlation=0.4,
        incremental_net_ir=0.3,
        annual_turnover=15.0,
        capacity_usd=50_000_000,
        net_excess_hac_p_value=0.03,
        deflated_sharpe_probability=0.95,
        probability_backtest_overfitting=0.1,
        worst_fold_net_ir=-0.1,
        worst_regime_net_ir=-0.1,
        positive_year_ratio=0.75,
        worst_year_incremental_return=-0.01,
        annual_return_dispersion=0.04,
        incremental_annual_return=0.02,
        incremental_max_drawdown=0.0,
        return_drawdown_efficiency_change=0.1,
        cost_stress_net_ir=0.1,
        break_even_cost_multiplier=2.0,
        maximum_style_exposure=0.03,
        maximum_industry_active_weight=0.02,
        stress_loss=-0.04,
        untradeable_fraction=0.01,
    )


class _DslResearcher:
    def propose(self, context: ResearchContext) -> ResearchProposal:
        factor = FactorDefinition(
            name="reversal_5",
            family="reversal",
            hypothesis="Recent winners should mean revert.",
            expression=operation(
                "cs_zscore",
                operation("negate", operation("returns", field("close"), periods=5)),
            ),
        )
        return ResearchProposal(
            candidate_id=factor.factor_id,
            family=factor.family,
            hypothesis=factor.hypothesis,
            change="Add a typed five-day reversal expression.",
            expected="Positive conditional IC.",
            factor=factor,
        )


class _Reviewer:
    def review(self, proposal: ResearchProposal, context: ResearchContext) -> ReviewDecision:
        return ReviewDecision(True, ("DSL semantics valid",))


class _Executor:
    def execute(self, proposal: ResearchProposal, context: ResearchContext) -> ExecutionReport:
        return ExecutionReport(_evidence(), {"run_id": "dsl-run"})


def test_orchestrator_accepts_typed_dsl_without_python_source(tmp_path: Path) -> None:
    orchestrator = ResearchOrchestrator(
        researcher=_DslResearcher(),
        reviewer=_Reviewer(),
        executor=_Executor(),
        budget=ExperimentBudgetLedger(
            tmp_path / "budget.jsonl",
            max_generation=2,
            max_family=2,
        ),
        audit_log=HashChainAuditLog(tmp_path / "audit.jsonl"),
        semantic_validator=SemanticValidator([FieldDefinition("close", "price")]),
    )

    result = orchestrator.run_one(ResearchContext("g1"))

    assert result.proposal.source is None
    assert result.proposal.factor is not None
    assert result.decision is not None and result.decision.status == "RETAINED"
    assert result.evaluation is not None
