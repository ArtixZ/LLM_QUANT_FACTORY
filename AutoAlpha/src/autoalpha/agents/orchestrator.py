from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Protocol

from autoalpha.dsl.expression import FactorDefinition
from autoalpha.dsl.semantics import SemanticValidator
from autoalpha.governance.audit import HashChainAuditLog
from autoalpha.research.budget import ExperimentBudgetLedger, ExperimentRecord
from autoalpha.research.evidence import EvaluationMatrix, build_evaluation_matrix
from autoalpha.research.gates import CandidateEvidence, InstitutionalAdmission
from autoalpha.security.guard import CandidatePolicy, validate_candidate_source


@dataclass(frozen=True)
class ResearchContext:
    generation: str
    factor_library_summary: tuple[dict[str, Any], ...] = ()
    allowed_feedback: dict[str, Any] | None = None


@dataclass(frozen=True)
class ResearchProposal:
    candidate_id: str
    family: str
    hypothesis: str
    change: str
    expected: str
    source: str | None = None
    factor: FactorDefinition | None = None

    def __post_init__(self) -> None:
        if (self.source is None) == (self.factor is None):
            raise ValueError("A proposal must contain exactly one of source or factor")

    @property
    def artifact_hash(self) -> str:
        if self.factor is not None:
            return self.factor.expression.expression_hash
        assert self.source is not None
        return hashlib.sha256(self.source.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ReviewDecision:
    approved: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ExecutionReport:
    evidence: CandidateEvidence
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class OrchestrationResult:
    proposal: ResearchProposal
    review: ReviewDecision
    reservation: ExperimentRecord | None
    decision: ExperimentRecord | None
    execution: ExecutionReport | None
    evaluation: EvaluationMatrix | None


class Researcher(Protocol):
    def propose(self, context: ResearchContext) -> ResearchProposal: ...


class Reviewer(Protocol):
    def review(self, proposal: ResearchProposal, context: ResearchContext) -> ReviewDecision: ...


class ExperimentExecutor(Protocol):
    def execute(self, proposal: ResearchProposal, context: ResearchContext) -> ExecutionReport: ...


class ResearchOrchestrator:
    def __init__(
        self,
        *,
        researcher: Researcher,
        reviewer: Reviewer,
        executor: ExperimentExecutor,
        budget: ExperimentBudgetLedger,
        audit_log: HashChainAuditLog,
        policy: CandidatePolicy | None = None,
        semantic_validator: SemanticValidator | None = None,
        admission: InstitutionalAdmission | None = None,
    ) -> None:
        self.researcher = researcher
        self.reviewer = reviewer
        self.executor = executor
        self.budget = budget
        self.audit_log = audit_log
        self.policy = policy or CandidatePolicy()
        self.semantic_validator = semantic_validator
        self.admission = admission or InstitutionalAdmission()

    def run_one(self, context: ResearchContext) -> OrchestrationResult:
        proposal = self.researcher.propose(context)
        self.audit_log.append(
            "PROPOSAL_CREATED",
            {
                "candidate_id": proposal.candidate_id,
                "family": proposal.family,
                "artifact_hash": proposal.artifact_hash,
                "hypothesis": proposal.hypothesis,
            },
        )
        if proposal.factor is not None:
            if self.semantic_validator is None:
                raise RuntimeError("A semantic validator is required for factor DSL proposals")
            self.semantic_validator.validate(proposal.factor.expression)
        else:
            assert proposal.source is not None
            validate_candidate_source(proposal.source, self.policy)
        review = self.reviewer.review(proposal, context)
        self.audit_log.append(
            "PROPOSAL_REVIEWED",
            {
                "candidate_id": proposal.candidate_id,
                "approved": review.approved,
                "reasons": list(review.reasons),
            },
        )
        if not review.approved:
            return OrchestrationResult(proposal, review, None, None, None, None)

        reservation = self.budget.reserve(
            proposal.candidate_id, context.generation, proposal.family
        )
        execution = self.executor.execute(proposal, context)
        evaluation = build_evaluation_matrix(
            proposal.candidate_id, execution.evidence, self.admission
        )
        admission_decision = evaluation.admission.decision
        decision = self.budget.close(
            reservation,
            admission_decision=admission_decision,
            reason=(evaluation.admission.failed_gate or "; ".join(review.reasons)),
        )
        self.audit_log.append(
            "EXPERIMENT_DECIDED",
            {
                "candidate_id": proposal.candidate_id,
                "status": decision.status,
                "admission_decision": admission_decision,
                "failed_gate": evaluation.admission.failed_gate,
                "artifact_hash": proposal.artifact_hash,
            },
        )
        return OrchestrationResult(proposal, review, reservation, decision, execution, evaluation)
