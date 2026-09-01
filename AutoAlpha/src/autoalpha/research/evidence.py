from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from autoalpha.research.gates import (
    AdmissionReport,
    CandidateEvidence,
    InstitutionalAdmission,
)


@dataclass(frozen=True)
class EvaluationMatrix:
    candidate_id: str
    admission: AdmissionReport
    portfolio_value: dict[str, float]
    statistical_reliability: dict[str, float | bool]
    risk_and_tradability: dict[str, float]
    prediction_diagnostics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_evaluation_matrix(
    candidate_id: str,
    evidence: CandidateEvidence,
    admission: InstitutionalAdmission | None = None,
) -> EvaluationMatrix:
    if not candidate_id.strip():
        raise ValueError("candidate_id is required")
    _require_finite_evidence(evidence)
    evaluator = admission or InstitutionalAdmission()
    return EvaluationMatrix(
        candidate_id=candidate_id,
        admission=evaluator.evaluate(evidence),
        portfolio_value={
            "incremental_net_ir": evidence.incremental_net_ir,
            "incremental_annual_return": evidence.incremental_annual_return,
            "incremental_max_drawdown": evidence.incremental_max_drawdown,
            "return_drawdown_efficiency_change": (evidence.return_drawdown_efficiency_change),
            "cost_stress_net_ir": evidence.cost_stress_net_ir,
            "worst_fold_net_ir": evidence.worst_fold_net_ir,
            "worst_regime_net_ir": evidence.worst_regime_net_ir,
            "positive_year_ratio": evidence.positive_year_ratio,
            "worst_year_incremental_return": evidence.worst_year_incremental_return,
            "annual_return_dispersion": evidence.annual_return_dispersion,
        },
        statistical_reliability={
            "net_excess_hac_p_value": evidence.net_excess_hac_p_value,
            "deflated_sharpe_probability": evidence.deflated_sharpe_probability,
            "probability_backtest_overfitting": evidence.probability_backtest_overfitting,
            "multiple_testing_passed": evidence.fdr_passed,
        },
        risk_and_tradability={
            "maximum_library_correlation": evidence.maximum_library_correlation,
            "maximum_style_exposure": evidence.maximum_style_exposure,
            "maximum_industry_active_weight": evidence.maximum_industry_active_weight,
            "stress_loss": evidence.stress_loss,
            "annual_turnover": evidence.annual_turnover,
            "capacity_usd": evidence.capacity_usd,
            "break_even_cost_multiplier": evidence.break_even_cost_multiplier,
            "untradeable_fraction": evidence.untradeable_fraction,
        },
        prediction_diagnostics={
            "rank_ic_mean": evidence.rank_ic_mean,
            "rank_ic_hac_p_value": evidence.rank_ic_hac_p_value,
            "worst_fold_rank_ic": evidence.worst_fold_rank_ic,
        },
    )


def _require_finite_evidence(evidence: CandidateEvidence) -> None:
    exempt = {"paper_net_ir"}
    invalid = [
        name
        for name, value in asdict(evidence).items()
        if name not in exempt
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
        and not np.isfinite(value)
    ]
    if invalid:
        raise ValueError(f"Candidate evidence contains non-finite values: {invalid}")
