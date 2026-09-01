from __future__ import annotations

from pathlib import Path

from autoalpha.agents.orchestrator import (
    ExecutionReport,
    ResearchContext,
    ResearchOrchestrator,
    ResearchProposal,
    ReviewDecision,
)
from autoalpha.governance.audit import HashChainAuditLog
from autoalpha.research.budget import ExperimentBudgetLedger
from autoalpha.research.gates import CandidateEvidence


def _evidence() -> CandidateEvidence:
    return CandidateEvidence(
        data_passed=True,
        semantic_passed=True,
        coverage=0.95,
        rank_ic_mean=-0.002,
        rank_ic_hac_p_value=0.90,
        fdr_passed=True,
        positive_fold_fraction=0.80,
        worst_fold_rank_ic=-0.05,
        maximum_library_correlation=0.50,
        incremental_net_ir=0.30,
        annual_turnover=20.0,
        capacity_usd=100_000_000,
        net_excess_hac_p_value=0.02,
        deflated_sharpe_probability=0.95,
        probability_backtest_overfitting=0.10,
        worst_fold_net_ir=-0.10,
        worst_regime_net_ir=-0.10,
        positive_year_ratio=0.75,
        worst_year_incremental_return=-0.01,
        annual_return_dispersion=0.05,
        incremental_annual_return=0.02,
        incremental_max_drawdown=0.01,
        return_drawdown_efficiency_change=0.05,
        cost_stress_net_ir=0.10,
        break_even_cost_multiplier=2.0,
        maximum_style_exposure=0.04,
        maximum_industry_active_weight=0.03,
        stress_loss=-0.05,
        untradeable_fraction=0.01,
    )


class _Researcher:
    def propose(self, context: ResearchContext) -> ResearchProposal:
        return ResearchProposal(
            candidate_id="candidate-1",
            family="reversal",
            hypothesis="Short-term reversal should add independent information.",
            change="Add a five-day reversal factor.",
            expected="Positive incremental IC.",
            source="def main(payload):\n    return payload\n",
        )


class _Reviewer:
    def review(self, proposal: ResearchProposal, context: ResearchContext) -> ReviewDecision:
        return ReviewDecision(approved=True, reasons=("time semantics valid",))


class _Executor:
    def execute(self, proposal: ResearchProposal, context: ResearchContext) -> ExecutionReport:
        return ExecutionReport(evidence=_evidence(), diagnostics={"run_id": "run-1"})


def test_orchestrator_enforces_review_budget_and_audit(tmp_path: Path) -> None:
    budget = ExperimentBudgetLedger(
        tmp_path / "experiments.jsonl",
        max_generation=10,
        max_family=5,
    )
    audit = HashChainAuditLog(tmp_path / "audit.jsonl")
    orchestrator = ResearchOrchestrator(
        researcher=_Researcher(),
        reviewer=_Reviewer(),
        executor=_Executor(),
        budget=budget,
        audit_log=audit,
    )

    result = orchestrator.run_one(ResearchContext(generation="g1"))

    assert result.decision is not None
    assert result.decision.status == "RETAINED"
    assert result.evaluation is not None
    assert result.evaluation.admission.decision == "RESEARCH"
    assert result.evaluation.prediction_diagnostics["rank_ic_mean"] < 0
    assert [record.event for record in audit.records()] == [
        "PROPOSAL_CREATED",
        "PROPOSAL_REVIEWED",
        "EXPERIMENT_DECIDED",
    ]
    audit.verify()
