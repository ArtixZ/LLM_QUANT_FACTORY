from dataclasses import replace
from pathlib import Path

import pytest

from autoalpha.config import ResearchConfig
from autoalpha.research.evidence import build_evaluation_matrix
from autoalpha.research.gates import CandidateEvidence, GatePolicy, InstitutionalAdmission


def _evidence() -> CandidateEvidence:
    return CandidateEvidence(
        data_passed=True,
        semantic_passed=True,
        coverage=0.95,
        rank_ic_mean=0.0,
        rank_ic_hac_p_value=0.8,
        fdr_passed=True,
        positive_fold_fraction=0.8,
        worst_fold_rank_ic=-0.05,
        maximum_library_correlation=0.4,
        incremental_net_ir=0.3,
        annual_turnover=12,
        capacity_usd=100e6,
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


def test_evaluation_matrix_separates_portfolio_value_from_ic_diagnostics() -> None:
    matrix = build_evaluation_matrix("F1", _evidence())

    assert matrix.admission.decision == "RESEARCH"
    assert matrix.portfolio_value["incremental_net_ir"] == 0.3
    assert matrix.prediction_diagnostics["rank_ic_mean"] == 0.0
    assert "rank_ic_mean" not in matrix.portfolio_value


def test_evaluation_matrix_rejects_non_finite_evidence() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        build_evaluation_matrix("F1", replace(_evidence(), incremental_net_ir=float("nan")))


def test_versioned_config_builds_the_executable_gate_policy() -> None:
    config = ResearchConfig.from_toml(Path(__file__).parents[2] / "config" / "research.toml")
    policy = GatePolicy.from_config(config.evaluation)
    admission = InstitutionalAdmission(policy)

    assert admission.policy.minimum_incremental_net_ir == 0.10
