from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

GateDecision = Literal[
    "REJECTED",
    "RESEARCH",
    "APPROVED_FOR_PAPER",
    "APPROVED_FOR_PRODUCTION",
]


@dataclass(frozen=True)
class CandidateEvidence:
    data_passed: bool
    semantic_passed: bool
    coverage: float
    rank_ic_mean: float
    rank_ic_hac_p_value: float
    fdr_passed: bool
    positive_fold_fraction: float
    worst_fold_rank_ic: float
    maximum_library_correlation: float
    incremental_net_ir: float
    annual_turnover: float
    capacity_usd: float
    net_excess_hac_p_value: float
    deflated_sharpe_probability: float
    probability_backtest_overfitting: float
    worst_fold_net_ir: float
    worst_regime_net_ir: float
    positive_year_ratio: float
    worst_year_incremental_return: float
    annual_return_dispersion: float
    incremental_annual_return: float
    incremental_max_drawdown: float
    return_drawdown_efficiency_change: float
    cost_stress_net_ir: float
    break_even_cost_multiplier: float
    maximum_style_exposure: float
    maximum_industry_active_weight: float
    stress_loss: float
    untradeable_fraction: float
    parameter_stability_positive_fraction: float = 1.0
    parameter_stability_worst_sharpe: float = 0.0
    holdout_passed: bool = False
    paper_days: int = 0
    paper_net_ir: float | None = None


@dataclass(frozen=True)
class GatePolicy:
    minimum_coverage: float = 0.80
    maximum_net_return_p_value: float = 0.10
    minimum_deflated_sharpe_probability: float = 0.90
    maximum_probability_backtest_overfitting: float = 0.40
    minimum_positive_fold_fraction: float = 0.60
    minimum_worst_fold_net_ir: float = -0.50
    minimum_worst_regime_net_ir: float = -0.50
    minimum_positive_year_ratio: float = 0.50
    minimum_worst_year_incremental_return: float = -0.03
    maximum_annual_return_dispersion: float = 0.15
    minimum_incremental_net_ir: float = 0.10
    minimum_incremental_annual_return: float = 0.005
    maximum_incremental_drawdown_deterioration: float = 0.02
    minimum_return_drawdown_efficiency_change: float = -0.10
    minimum_cost_stress_net_ir: float = 0.0
    maximum_library_correlation: float = 0.85
    redundancy_override_net_ir: float = 0.25
    maximum_style_exposure: float = 0.10
    maximum_industry_active_weight: float = 0.05
    maximum_stress_loss: float = 0.10
    maximum_annual_turnover: float = 30.0
    minimum_capacity_usd: float = 10_000_000.0
    minimum_break_even_cost_multiplier: float = 1.50
    maximum_untradeable_fraction: float = 0.05
    minimum_paper_days: int = 60
    minimum_paper_net_ir: float = 0.0
    minimum_parameter_positive_fraction: float = 0.67
    minimum_parameter_worst_sharpe: float = 0.0
    minimum_diversification_sharpe_improvement: float = 0.10
    minimum_diversification_drawdown_improvement: float = 0.01
    minimum_stability_dispersion_reduction: float = 0.01
    minimum_stability_worst_fold_sharpe_improvement: float = 0.10
    maximum_transition_annual_return_sacrifice: float = 0.02

    @classmethod
    def from_config(cls, config: Any) -> GatePolicy:
        return cls(**asdict(config))


@dataclass(frozen=True)
class GateResult:
    gate: str
    passed: bool
    reasons: tuple[str, ...]
    observations: tuple[str, ...] = ()


@dataclass(frozen=True)
class AdmissionReport:
    decision: GateDecision
    gates: tuple[GateResult, ...]

    @property
    def failed_gate(self) -> str | None:
        return next((gate.gate for gate in self.gates if not gate.passed), None)


class InstitutionalAdmission:
    """Sequential institutional gates; no weighted score can compensate a failure."""

    def __init__(self, policy: GatePolicy | None = None) -> None:
        self.policy = policy or GatePolicy()

    def evaluate(self, evidence: CandidateEvidence) -> AdmissionReport:
        checks: list[tuple[str, list[str], tuple[str, ...]]] = [
            (
                "DATA_AND_SEMANTICS",
                _failures(
                    (evidence.data_passed, "data contract failed"),
                    (evidence.semantic_passed, "factor semantics failed"),
                    (
                        evidence.coverage >= self.policy.minimum_coverage,
                        f"coverage {evidence.coverage:.1%} below minimum",
                    ),
                ),
                (),
            ),
            (
                "STATISTICAL_RELIABILITY",
                _failures(
                    (
                        evidence.net_excess_hac_p_value <= self.policy.maximum_net_return_p_value,
                        "cost-adjusted incremental return is not statistically reliable",
                    ),
                    (evidence.fdr_passed, "multiple-testing adjustment failed"),
                    (
                        evidence.deflated_sharpe_probability
                        >= self.policy.minimum_deflated_sharpe_probability,
                        "deflated Sharpe probability is below minimum",
                    ),
                    (
                        evidence.probability_backtest_overfitting
                        <= self.policy.maximum_probability_backtest_overfitting,
                        "probability of backtest overfitting is above maximum",
                    ),
                ),
                (
                    f"diagnostic rank IC={evidence.rank_ic_mean:.4f}",
                    f"diagnostic IC p-value={evidence.rank_ic_hac_p_value:.4f}",
                ),
            ),
            (
                "ROBUSTNESS",
                _failures(
                    (
                        evidence.positive_fold_fraction
                        >= self.policy.minimum_positive_fold_fraction,
                        "insufficient positive portfolio-increment folds",
                    ),
                    (
                        evidence.worst_fold_net_ir >= self.policy.minimum_worst_fold_net_ir,
                        "worst walk-forward fold net IR is too weak",
                    ),
                    (
                        evidence.worst_regime_net_ir >= self.policy.minimum_worst_regime_net_ir,
                        "worst market-regime net IR is too weak",
                    ),
                    (
                        evidence.positive_year_ratio >= self.policy.minimum_positive_year_ratio,
                        "positive incremental year ratio is below minimum",
                    ),
                    (
                        evidence.worst_year_incremental_return
                        >= self.policy.minimum_worst_year_incremental_return,
                        "worst annual portfolio increment is too weak",
                    ),
                    (
                        evidence.annual_return_dispersion
                        <= self.policy.maximum_annual_return_dispersion,
                        "annual portfolio increment is too unstable",
                    ),
                    (
                        evidence.parameter_stability_positive_fraction
                        >= self.policy.minimum_parameter_positive_fraction,
                        "parameter neighborhood has too few positive variants",
                    ),
                    (
                        evidence.parameter_stability_worst_sharpe
                        >= self.policy.minimum_parameter_worst_sharpe,
                        "parameter neighborhood contains a performance cliff",
                    ),
                ),
                (f"diagnostic worst-fold rank IC={evidence.worst_fold_rank_ic:.4f}",),
            ),
            (
                "PORTFOLIO_INCREMENT",
                _failures(
                    (
                        evidence.incremental_net_ir >= self.policy.minimum_incremental_net_ir,
                        "incremental net IR is too small",
                    ),
                    (
                        evidence.incremental_annual_return
                        >= self.policy.minimum_incremental_annual_return,
                        "incremental annual net return is too small",
                    ),
                    (
                        evidence.incremental_max_drawdown
                        >= -self.policy.maximum_incremental_drawdown_deterioration,
                        "maximum drawdown deterioration exceeds risk budget",
                    ),
                    (
                        evidence.return_drawdown_efficiency_change
                        >= self.policy.minimum_return_drawdown_efficiency_change,
                        "return-to-drawdown efficiency deteriorates too much",
                    ),
                    (
                        evidence.cost_stress_net_ir >= self.policy.minimum_cost_stress_net_ir,
                        "net IR is negative under stressed transaction costs",
                    ),
                ),
                (),
            ),
            (
                "INDEPENDENCE_AND_RISK",
                _failures(
                    (
                        abs(evidence.maximum_library_correlation)
                        <= self.policy.maximum_library_correlation
                        or evidence.incremental_net_ir >= self.policy.redundancy_override_net_ir,
                        "high library correlation has no material portfolio-value override",
                    ),
                    (
                        abs(evidence.maximum_style_exposure) <= self.policy.maximum_style_exposure,
                        "style exposure exceeds limit",
                    ),
                    (
                        abs(evidence.maximum_industry_active_weight)
                        <= self.policy.maximum_industry_active_weight,
                        "industry active weight exceeds limit",
                    ),
                    (
                        evidence.stress_loss >= -self.policy.maximum_stress_loss,
                        "stress loss exceeds risk budget",
                    ),
                ),
                (),
            ),
            (
                "TRADABILITY_AND_CAPACITY",
                _failures(
                    (
                        evidence.annual_turnover <= self.policy.maximum_annual_turnover,
                        "annual turnover exceeds limit",
                    ),
                    (
                        evidence.capacity_usd >= self.policy.minimum_capacity_usd,
                        "capacity is below minimum",
                    ),
                    (
                        evidence.break_even_cost_multiplier
                        >= self.policy.minimum_break_even_cost_multiplier,
                        "transaction-cost safety margin is too small",
                    ),
                    (
                        evidence.untradeable_fraction <= self.policy.maximum_untradeable_fraction,
                        "untradeable order fraction exceeds limit",
                    ),
                ),
                (),
            ),
        ]
        results: list[GateResult] = []
        for name, reasons, observations in checks:
            result = GateResult(name, not reasons, tuple(reasons), observations)
            results.append(result)
            if reasons:
                return AdmissionReport("REJECTED", tuple(results))

        if not evidence.holdout_passed:
            results.append(GateResult("HOLDOUT", False, ("holdout approval is required",)))
            return AdmissionReport("RESEARCH", tuple(results))
        results.append(GateResult("HOLDOUT", True, ()))

        paper_reasons = _failures(
            (
                evidence.paper_days >= self.policy.minimum_paper_days,
                "paper-trading observation period is too short",
            ),
            (
                evidence.paper_net_ir is not None
                and evidence.paper_net_ir >= self.policy.minimum_paper_net_ir,
                "paper-trading net IR is below minimum",
            ),
        )
        results.append(GateResult("PAPER_TRADING", not paper_reasons, tuple(paper_reasons)))
        decision: GateDecision = (
            "APPROVED_FOR_PRODUCTION" if not paper_reasons else "APPROVED_FOR_PAPER"
        )
        return AdmissionReport(decision, tuple(results))


def _failures(*checks: tuple[bool, str]) -> list[str]:
    return [reason for passed, reason in checks if not passed]


def evidence_from_mapping(values: dict[str, Any]) -> CandidateEvidence:
    return CandidateEvidence(**values)
