"""Leakage-aware research evaluation primitives."""

from autoalpha.research.evidence import EvaluationMatrix, build_evaluation_matrix
from autoalpha.research.gates import CandidateEvidence, InstitutionalAdmission
from autoalpha.research.incremental import (
    AnnualRobustness,
    PortfolioIncrement,
    annual_robustness,
    compare_portfolios,
)
from autoalpha.research.protocol import ProtocolManifest
from autoalpha.research.ranking import CandidateValue, pareto_rank
from autoalpha.research.splits import PurgedWalkForward, WalkForwardFold
from autoalpha.research.statistics import MeanInference, hac_mean_inference

__all__ = [
    "MeanInference",
    "CandidateEvidence",
    "CandidateValue",
    "AnnualRobustness",
    "EvaluationMatrix",
    "InstitutionalAdmission",
    "ProtocolManifest",
    "PortfolioIncrement",
    "PurgedWalkForward",
    "WalkForwardFold",
    "hac_mean_inference",
    "build_evaluation_matrix",
    "annual_robustness",
    "compare_portfolios",
    "pareto_rank",
]
