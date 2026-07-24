from __future__ import annotations

from dataclasses import replace

from autoalpha.research.gates import CandidateEvidence, InstitutionalAdmission


def _evidence() -> CandidateEvidence:
    return CandidateEvidence(
        data_passed=True,
        semantic_passed=True,
        coverage=0.95,
        rank_ic_mean=0.03,
        rank_ic_hac_p_value=0.01,
        fdr_passed=True,
        positive_fold_fraction=0.80,
        worst_fold_rank_ic=0.001,
        maximum_library_correlation=0.50,
        incremental_net_ir=0.30,
        incremental_annual_return=0.02,
        incremental_max_drawdown=0.01,
        return_drawdown_efficiency_change=0.05,
        cost_stress_net_ir=0.10,
        net_excess_hac_p_value=0.02,
        deflated_sharpe_probability=0.95,
        probability_backtest_overfitting=0.10,
        worst_fold_net_ir=-0.10,
        worst_regime_net_ir=-0.10,
        positive_year_ratio=0.75,
        worst_year_incremental_return=-0.01,
        annual_return_dispersion=0.05,
        maximum_style_exposure=0.04,
        maximum_industry_active_weight=0.03,
        stress_loss=-0.05,
        annual_turnover=20.0,
        capacity_cny=100_000_000,
        break_even_cost_multiplier=2.0,
        untradeable_fraction=0.01,
    )


def test_admission_stops_at_first_failed_gate() -> None:
    report = InstitutionalAdmission().evaluate(replace(_evidence(), coverage=0.50))

    assert report.decision == "REJECTED"
    assert report.failed_gate == "DATA_AND_SEMANTICS"
    assert len(report.gates) == 1


def test_candidate_progresses_from_research_to_paper_to_production() -> None:
    admission = InstitutionalAdmission()

    research = admission.evaluate(_evidence())
    paper = admission.evaluate(replace(_evidence(), holdout_passed=True))
    production = admission.evaluate(
        replace(_evidence(), holdout_passed=True, paper_days=80, paper_net_ir=0.4)
    )

    assert research.decision == "RESEARCH"
    assert paper.decision == "APPROVED_FOR_PAPER"
    assert production.decision == "APPROVED_FOR_PRODUCTION"


def test_low_ic_candidate_passes_when_portfolio_increment_is_reliable() -> None:
    evidence = replace(
        _evidence(),
        rank_ic_mean=-0.002,
        rank_ic_hac_p_value=0.90,
        worst_fold_rank_ic=-0.05,
    )

    report = InstitutionalAdmission().evaluate(evidence)

    assert report.decision == "RESEARCH"
    statistical_gate = next(gate for gate in report.gates if gate.gate == "STATISTICAL_RELIABILITY")
    assert any("rank IC=-0.0020" in item for item in statistical_gate.observations)


def test_high_ic_candidate_fails_without_net_portfolio_value() -> None:
    evidence = replace(
        _evidence(),
        rank_ic_mean=0.12,
        rank_ic_hac_p_value=0.0001,
        incremental_net_ir=-0.10,
        incremental_annual_return=-0.02,
        cost_stress_net_ir=-0.30,
    )

    report = InstitutionalAdmission().evaluate(evidence)

    assert report.decision == "REJECTED"
    assert report.failed_gate == "PORTFOLIO_INCREMENT"
