from __future__ import annotations

from pathlib import Path

import pandas as pd

from autoalpha.backtest.timing import EOD_NEXT_OPEN_RETURN_CONVENTION
from autoalpha.backtest.us_vector import US_PROXY_RETURN_CONVENTION
from autoalpha.config import ResearchConfig
from autoalpha.dsl.expression import FactorDefinition, field
from autoalpha.service.canonical_evaluation import CANONICAL_LIBRARY_PROTOCOL
from autoalpha.service.evaluator import PortfolioEvaluation
from autoalpha.service.multifactor import (
    MultiFactorResearchEngine,
    _candidate_screen_failures,
    _library_admission_failures,
    _portfolio_action_gate_failures,
)
from autoalpha.service.store import ServiceStore


class FakeEvaluator:
    protocol_version = ResearchConfig.from_toml(
        Path("config/research.toml")
    ).governance.protocol_version

    def evaluate_portfolio(
        self,
        factors,
        *,
        weights=None,
        benchmark_factors=None,
        benchmark_weights=None,
        bootstrap_samples=500,
    ):
        count = len(factors)
        metrics = {
            "portfolio_sharpe_ratio": 1.0 + 0.3 * count,
            "portfolio_simple_annual_return": 0.08 + 0.02 * count,
            "portfolio_compound_annual_return": 0.09 + 0.02 * count,
            "portfolio_max_drawdown": -0.08,
            "portfolio_cost_stress_net_ir": 0.8,
            "portfolio_annual_turnover": 12.0,
            "portfolio_coverage": 0.95,
            "portfolio_capacity_usd": 50_000_000.0,
            "portfolio_positive_year_ratio": 1.0,
            "portfolio_worst_year_return": 0.01,
            "portfolio_annual_return_dispersion": 0.04,
            "portfolio_factor_count": count,
            "portfolio_max_factor_correlation": 0.2 if count > 1 else 0.0,
            "portfolio_incremental_net_ir": 0.35,
            "portfolio_incremental_annual_return": 0.02,
            "portfolio_incremental_max_drawdown": 0.0,
            "portfolio_incremental_return_drawdown_efficiency": 0.2,
            "portfolio_incremental_cost_stress_net_ir": 0.3,
            "portfolio_sharpe_change": 0.3,
            "portfolio_annual_return_change": 0.02,
            "portfolio_max_drawdown_change": 0.01,
            "portfolio_cost_stress_net_ir_change": 0.2,
            "portfolio_annual_turnover_change": 0.0,
            "portfolio_backtest_start": "2022-01-01",
            "portfolio_backtest_end": "2024-01-01",
            "portfolio_backtest_observations": 490,
            "portfolio_weight_method": "equal_risk_cross_sectional_zscore",
            "portfolio_evaluation_protocol": self.protocol_version,
            "portfolio_return_convention": EOD_NEXT_OPEN_RETURN_CONVENTION,
            "portfolio_incremental_net_return_hac_p_value": 0.01,
            "portfolio_bootstrap_samples": bootstrap_samples,
            "portfolio_evaluation_stage": (
                "FULL_INFERENCE" if bootstrap_samples else "VECTOR_SCREEN"
            ),
        }
        return PortfolioEvaluation(metrics, pd.Series([0.0]), {})


class RejectingEvaluator(FakeEvaluator):
    def evaluate_portfolio(self, factors, **kwargs):
        result = super().evaluate_portfolio(factors, **kwargs)
        metrics = {
            **result.metrics,
            "portfolio_incremental_net_ir": -0.1,
            "portfolio_incremental_annual_return": -0.01,
            "portfolio_sharpe_change": 0.0,
            "portfolio_annual_return_change": -0.03,
            "portfolio_max_drawdown_change": 0.0,
            "portfolio_cost_stress_net_ir_change": -0.1,
        }
        return PortfolioEvaluation(metrics, result.net_returns, result.factor_correlations)


class InfeasibleBaselineEvaluator(FakeEvaluator):
    def evaluate_portfolio(self, factors, **kwargs):
        result = super().evaluate_portfolio(factors, **kwargs)
        return PortfolioEvaluation(
            {**result.metrics, "portfolio_coverage": 0.10},
            result.net_returns,
            result.factor_correlations,
        )


class CountingRejectingEvaluator(RejectingEvaluator):
    def __init__(self) -> None:
        self.bootstrap_samples: list[int] = []

    def evaluate_portfolio(self, factors, **kwargs):
        self.bootstrap_samples.append(int(kwargs.get("bootstrap_samples", 500)))
        return super().evaluate_portfolio(factors, **kwargs)


def _factor(name: str, field_name: str) -> FactorDefinition:
    return FactorDefinition(name, "test", "test hypothesis", field(field_name))


def _proposal(factor: FactorDefinition) -> dict:
    return {
        "name": factor.name,
        "family": factor.family,
        "hypothesis": factor.hypothesis,
        "expected_direction": factor.expected_direction,
        "expression": factor.expression.to_dict(),
    }


def test_initial_ineligible_candidate_is_recorded_as_hold_without_active_weights(
    tmp_path: Path,
) -> None:
    store = ServiceStore(tmp_path / "service.sqlite3")
    config = ResearchConfig.from_toml(Path("config/research.toml"))
    candidate = _factor("candidate", "amount")
    engine = MultiFactorResearchEngine(store, FakeEvaluator(), config, run_id="run-new")

    decision = engine.decide(candidate, candidate_eligible=False)
    version = engine.persist("run-new", 1, decision)

    assert decision.action == "HOLD"
    assert not decision.accepted
    assert decision.factors == ()
    assert decision.weights == ()
    assert store.portfolio_history(run_id="run-new")[0]["id"] == version


def test_strategy_gate_rejects_long_short_return_convention() -> None:
    config = ResearchConfig.from_toml(Path("config/research.toml"))
    metrics = FakeEvaluator().evaluate_portfolio([_factor("a", "amount")]).metrics
    metrics["portfolio_strategy_gate_basis"] = "US_EQUITY_LONG_ONLY_WEEKLY_NON_PIT_PROXY"

    assert "invalid_return_convention" in _portfolio_action_gate_failures(metrics, config)

    metrics["portfolio_return_convention"] = US_PROXY_RETURN_CONVENTION
    assert "invalid_return_convention" not in _portfolio_action_gate_failures(metrics, config)

    metrics["portfolio_maximum_observed_positions"] = 46
    assert "residual_positions" in _portfolio_action_gate_failures(metrics, config)


def test_initial_candidate_governance_veto_preserves_empty_incumbent(
    tmp_path: Path,
) -> None:
    store = ServiceStore(tmp_path / "service.sqlite3")
    config = ResearchConfig.from_toml(Path("config/research.toml"))
    candidate = _factor("candidate", "amount")
    engine = MultiFactorResearchEngine(store, FakeEvaluator(), config, run_id="run-new")
    proposed = engine.decide(candidate, candidate_eligible=True)

    vetoed = engine.governance_veto(
        proposed,
        gate="HOLDOUT",
        reason="HOLDOUT_FAILED",
    )
    version = engine.persist("run-new", 1, vetoed)

    assert proposed.accepted
    assert vetoed.action == "HOLD"
    assert not vetoed.accepted
    assert vetoed.factors == ()
    assert vetoed.weights == ()
    assert vetoed.failed_gates == ("HOLDOUT",)
    assert store.portfolio_history(run_id="run-new")[0]["id"] == version


def test_multifactor_loop_accepts_incremental_add_and_persists_searched_weights(
    tmp_path: Path,
) -> None:
    store = ServiceStore(tmp_path / "service.sqlite3")
    config = ResearchConfig.from_toml(Path("config/research.toml"))
    incumbent = _factor("incumbent", "adj_close")
    candidate = _factor("candidate", "amount")
    for iteration, factor in enumerate((incumbent, candidate), start=1):
        store.upsert_factor_pool(
            factor_id=factor.factor_id,
            source_iteration=iteration,
            proposal=_proposal(factor),
            metrics={"sharpe_ratio": 1.0},
            status="ELIGIBLE",
            status_reason="passed",
        )
    store.record_portfolio_decision(
        run_id="run-1",
        iteration=1,
        action="BOOTSTRAP",
        candidate_id=incumbent.factor_id,
        removed_factor_id=None,
        accepted=True,
        reason="initial",
        metrics=FakeEvaluator().evaluate_portfolio([incumbent]).metrics,
        members=[(incumbent.factor_id, 1.0)],
    )
    engine = MultiFactorResearchEngine(store, FakeEvaluator(), config)

    decision = engine.decide(candidate, candidate_eligible=True)
    version = engine.persist("run-1", 2, decision)

    assert decision.action == "ADD"
    assert decision.accepted
    assert decision.utility_change > 0
    champion = store.active_portfolio()
    assert champion is not None
    assert champion["id"] == version
    assert {member["weight"] for member in champion["members"]} == {0.05, 0.95}
    assert len(decision.option_diagnostics) == 10


def test_hold_preserves_best_rejected_option_and_failed_gates(tmp_path: Path) -> None:
    store = ServiceStore(tmp_path / "service.sqlite3")
    config = ResearchConfig.from_toml(Path("config/research.toml"))
    incumbent = _factor("incumbent", "adj_close")
    candidate = _factor("candidate", "amount")
    for iteration, factor in enumerate((incumbent, candidate), start=1):
        store.upsert_factor_pool(
            factor_id=factor.factor_id,
            source_iteration=iteration,
            proposal=_proposal(factor),
            metrics={"sharpe_ratio": 1.0},
            status="ELIGIBLE",
            status_reason="passed",
        )
    evaluator = RejectingEvaluator()
    store.record_portfolio_decision(
        run_id="run-1",
        iteration=1,
        action="BOOTSTRAP",
        candidate_id=incumbent.factor_id,
        removed_factor_id=None,
        accepted=True,
        reason="initial",
        metrics=evaluator.evaluate_portfolio([incumbent]).metrics,
        members=[(incumbent.factor_id, 1.0)],
    )

    decision = MultiFactorResearchEngine(store, evaluator, config).decide(
        candidate, candidate_eligible=True
    )

    assert decision.action == "HOLD"
    assert "portfolio_value" in decision.failed_gates
    assert decision.option_diagnostics
    assert "best=" in decision.reason


def test_rejected_weight_grid_stops_at_vector_screen(tmp_path: Path) -> None:
    store = ServiceStore(tmp_path / "service.sqlite3")
    config = ResearchConfig.from_toml(Path("config/research.toml"))
    incumbent = _factor("incumbent", "adj_close")
    candidate = _factor("candidate", "amount")
    for iteration, factor in enumerate((incumbent, candidate), start=1):
        store.upsert_factor_pool(
            factor_id=factor.factor_id,
            source_iteration=iteration,
            proposal=_proposal(factor),
            metrics={"sharpe_ratio": 1.0},
            status="ELIGIBLE",
            status_reason="passed",
        )
    evaluator = CountingRejectingEvaluator()
    store.record_portfolio_decision(
        run_id="run-1",
        iteration=1,
        action="BOOTSTRAP",
        candidate_id=incumbent.factor_id,
        removed_factor_id=None,
        accepted=True,
        reason="initial",
        metrics=evaluator.evaluate_portfolio([incumbent]).metrics,
        members=[(incumbent.factor_id, 1.0)],
    )
    evaluator.bootstrap_samples.clear()

    decision = MultiFactorResearchEngine(store, evaluator, config).decide(
        candidate, candidate_eligible=True
    )

    assert not decision.accepted
    assert evaluator.bootstrap_samples.count(500) == 1  # incumbent only
    assert evaluator.bootstrap_samples.count(0) == 10


def test_unknown_protocol_incumbent_is_not_reused(tmp_path: Path) -> None:
    store = ServiceStore(tmp_path / "service.sqlite3")
    config = ResearchConfig.from_toml(Path("config/research.toml"))
    incumbent = _factor("incumbent", "adj_close")
    store.upsert_factor_pool(
        factor_id=incumbent.factor_id,
        source_iteration=1,
        proposal=_proposal(incumbent),
        metrics={"sharpe_ratio": 4.0},
        status="ACTIVE",
        status_reason="legacy",
    )
    legacy_metrics = FakeEvaluator().evaluate_portfolio([incumbent]).metrics
    legacy_metrics.pop("portfolio_evaluation_protocol")
    store.record_portfolio_decision(
        run_id="run-1",
        iteration=1,
        action="BOOTSTRAP",
        candidate_id=incumbent.factor_id,
        removed_factor_id=None,
        accepted=True,
        reason="legacy unknown protocol",
        metrics=legacy_metrics,
        members=[(incumbent.factor_id, 1.0)],
    )

    engine = MultiFactorResearchEngine(store, FakeEvaluator(), config)

    assert engine.active_components() == []


def test_bootstrap_skips_factor_with_manually_exposed_holdout(tmp_path: Path) -> None:
    store = ServiceStore(tmp_path / "service.sqlite3")
    config = ResearchConfig.from_toml(Path("config/research.toml"))
    contaminated = _factor("contaminated", "vol")
    clean = _factor("clean", "amount")
    for iteration, factor, sharpe in (
        (1, contaminated, 10.0),
        (2, clean, 1.0),
    ):
        store.upsert_factor_pool(
            factor_id=factor.factor_id,
            source_iteration=iteration,
            proposal=_proposal(factor),
            metrics={
                "evaluation_protocol": CANONICAL_LIBRARY_PROTOCOL,
                "long_only_sharpe_ratio": sharpe,
                "long_only_simple_annual_return": 0.1,
                "long_only_max_drawdown": 0.0,
                "long_only_annual_turnover": 1.0,
                "long_only_annual_return_dispersion": 0.1,
            },
            status="ELIGIBLE",
            status_reason="passed",
        )
    backtest_id = store.create_manual_backtest({"factor_ids": [contaminated.factor_id]})
    store.record_manual_research_exposures(
        backtest_id=backtest_id,
        generation_id=config.generation,
        factor_ids=[contaminated.factor_id],
        period_start="2025-01-02",
        period_end="2025-12-31",
        holdout_start="2025-01-02",
        holdout_end="2026-07-14",
    )

    decision = MultiFactorResearchEngine(store, FakeEvaluator(), config).bootstrap_champion(
        "run-1", 3
    )

    assert decision is not None
    assert decision.candidate_id == clean.factor_id


def test_bootstrap_records_rehabilitation_without_accepting_infeasible_control(
    tmp_path: Path,
) -> None:
    store = ServiceStore(tmp_path / "service.sqlite3")
    config = ResearchConfig.from_toml(Path("config/research.toml"))
    factor = _factor("weak-control", "amount")
    store.upsert_factor_pool(
        factor_id=factor.factor_id,
        source_iteration=1,
        proposal=_proposal(factor),
        metrics={
            "evaluation_protocol": CANONICAL_LIBRARY_PROTOCOL,
            "long_only_sharpe_ratio": 1.0,
        },
        status="ELIGIBLE",
        status_reason="library only",
    )

    decision = MultiFactorResearchEngine(
        store,
        InfeasibleBaselineEvaluator(),
        config,
        run_id="run-1",
    ).bootstrap_champion("run-1", 2)

    assert decision is None
    history = store.portfolio_history(run_id="run-1")
    assert history[0]["action"] == "BASELINE_REHABILITATION"
    assert not history[0]["accepted"]
    assert history[0]["metrics"]["portfolio_control_state"] == "NEEDS_REHABILITATION"
    assert store.active_portfolio(run_id="run-1") is None


def test_candidate_registration_uses_task_metrics_for_promotion(tmp_path: Path) -> None:
    store = ServiceStore(tmp_path / "service.sqlite3")
    config = ResearchConfig.from_toml(Path("config/research.toml"))
    factor = _factor("namespace", "amount")
    canonical = {
        "evaluation_protocol": CANONICAL_LIBRARY_PROTOCOL,
        "long_only_return_convention": US_PROXY_RETURN_CONVENTION,
        "long_only_sharpe_ratio": 0.25,
        "long_only_active_information_ratio": 0.15,
        "long_only_simple_annual_return": 0.04,
        "long_only_coverage": 0.93,
        "long_only_annual_turnover": 20.0,
        "long_only_walk_forward_fold_count": config.walk_forward.minimum_folds,
    }
    task = {
        "evaluation_protocol": config.governance.protocol_version,
        "long_only_return_convention": US_PROXY_RETURN_CONVENTION,
        "long_only_sharpe_ratio": 1.0,
        "long_only_simple_annual_return": 0.10,
        "long_only_coverage": 0.95,
        "long_only_cost_stress_net_ir": 0.8,
        "long_only_annual_turnover": 10.0,
        "long_only_annual_return_dispersion": 0.05,
        "long_only_walk_forward_fold_count": config.walk_forward.minimum_folds,
        "long_only_walk_forward_positive_fraction": 0.8,
        "long_only_walk_forward_worst_sharpe": 0.2,
        "long_only_deflated_sharpe_probability": 0.99,
        "long_only_net_return_hac_p_value": 0.01,
        "parameter_stability_positive_fraction": 1.0,
        "parameter_stability_worst_sharpe": 0.2,
        "multiple_testing_fdr_passed": True,
        "probability_backtest_overfitting": 0.1,
    }

    admitted = MultiFactorResearchEngine(store, FakeEvaluator(), config).register_candidate(
        factor,
        1,
        _proposal(factor),
        canonical,
        task_metrics=task,
    )

    assert admitted
    stored = store.factor_pool_record(factor.factor_id)["metrics"]
    assert stored["production_promotion_gate_passed"]
    assert "stale_evaluation_protocol" not in stored["production_promotion_gate_failures"]
    assert stored["metric_namespaces"]["task"]["protocol"] == config.governance.protocol_version
    assert stored["task_research_metrics"]["evaluation_protocol"] == (
        config.governance.protocol_version
    )


def test_candidate_screen_rejects_extreme_turnover_before_portfolio_search() -> None:
    config = ResearchConfig.from_toml(Path("config/research.toml"))
    metrics = {
        "long_only_sharpe_ratio": 1.0,
        "long_only_simple_annual_return": 0.1,
        "long_only_coverage": 0.95,
        "long_only_cost_stress_net_ir": 0.8,
        "long_only_annual_turnover": 100.0,
        "long_only_annual_return_dispersion": 0.1,
    }

    assert "excessive_turnover" in _candidate_screen_failures(metrics, config)


def test_candidate_screen_uses_long_only_metrics_before_alpha_diagnostics() -> None:
    config = ResearchConfig.from_toml(Path("config/research.toml"))
    metrics = {
        "evaluation_protocol": config.governance.protocol_version,
        "long_only_return_convention": US_PROXY_RETURN_CONVENTION,
        "long_only_sharpe_ratio": -0.2,
        "long_only_simple_annual_return": -0.01,
        "long_only_coverage": 0.95,
        "long_only_cost_stress_net_ir": -0.1,
        "long_only_annual_turnover": 10.0,
        "long_only_annual_return_dispersion": 0.05,
        "long_only_walk_forward_fold_count": config.walk_forward.minimum_folds,
        "long_only_walk_forward_positive_fraction": 0.8,
        "long_only_walk_forward_worst_sharpe": 0.1,
        "long_only_deflated_sharpe_probability": 0.99,
        "long_only_net_return_hac_p_value": 0.01,
        "parameter_stability_positive_fraction": 1.0,
        "parameter_stability_worst_sharpe": 0.1,
        "multiple_testing_fdr_passed": True,
        "probability_backtest_overfitting": 0.1,
        "sharpe_ratio": 9.0,
        "simple_annual_return": 0.9,
    }

    failures = _candidate_screen_failures(metrics, config)

    assert "non_positive_sharpe" in failures
    assert "non_positive_annual_return" in failures
    assert "cost_stress" in failures


def test_library_admission_retains_weak_signal_without_relaxing_promotion() -> None:
    config = ResearchConfig.from_toml(Path("config/research.toml"))
    metrics = {
        "evaluation_protocol": CANONICAL_LIBRARY_PROTOCOL,
        "long_only_return_convention": US_PROXY_RETURN_CONVENTION,
        "long_only_sharpe_ratio": 0.25,
        "long_only_active_information_ratio": 0.15,
        "long_only_simple_annual_return": 0.04,
        "long_only_coverage": 0.93,
        "long_only_annual_turnover": 20.0,
        "long_only_walk_forward_fold_count": config.walk_forward.minimum_folds,
        "long_only_deflated_sharpe_probability": 0.05,
    }

    assert _library_admission_failures(metrics, config) == []
    assert "deflated_sharpe" in _candidate_screen_failures(metrics, config)


def test_library_admission_screens_behaviorally_redundant_weak_signal() -> None:
    config = ResearchConfig.from_toml(Path("config/research.toml"))
    metrics = {
        "evaluation_protocol": CANONICAL_LIBRARY_PROTOCOL,
        "long_only_return_convention": US_PROXY_RETURN_CONVENTION,
        "long_only_sharpe_ratio": 0.10,
        "long_only_active_information_ratio": 0.05,
        "long_only_coverage": 0.93,
        "long_only_annual_turnover": 20.0,
        "long_only_walk_forward_fold_count": config.walk_forward.minimum_folds,
        "library_signal_correlation_max": 0.95,
        "library_signal_correlation_peer": "F_existing",
    }

    assert "signal_redundancy" in _library_admission_failures(metrics, config)

    strong = {
        **metrics,
        "long_only_sharpe_ratio": config.evaluation.redundancy_override_net_ir,
    }
    assert "signal_redundancy" not in _library_admission_failures(strong, config)

    orthogonal = {**metrics, "library_signal_correlation_max": 0.30}
    assert "signal_redundancy" not in _library_admission_failures(orthogonal, config)


def test_stability_upgrade_can_repair_an_infeasible_incumbent() -> None:
    config = ResearchConfig.from_toml(Path("config/research.toml"))
    incumbent = {
        **FakeEvaluator().evaluate_portfolio([_factor("a", "amount")]).metrics,
        "portfolio_annual_return_dispersion": 0.225,
        "portfolio_walk_forward_worst_sharpe": 1.35,
        "portfolio_walk_forward_fold_count": 10,
        "portfolio_walk_forward_positive_fraction": 1.0,
        "portfolio_deflated_sharpe_probability": 0.99,
        "portfolio_net_return_hac_p_value": 0.01,
        "portfolio_incremental_net_return_hac_p_value": 0.01,
    }
    proposed = {
        **incumbent,
        "portfolio_annual_return_dispersion": 0.145,
        "portfolio_walk_forward_worst_sharpe": 2.29,
        "portfolio_sharpe_change": 0.08,
        "portfolio_annual_return_change": -0.01,
        "portfolio_max_drawdown_change": 0.015,
        "portfolio_cost_stress_net_ir_change": 0.1,
        "portfolio_incremental_net_ir": -0.05,
        "portfolio_incremental_annual_return": -0.01,
        "portfolio_incremental_max_drawdown": 0.015,
    }

    failures = _portfolio_action_gate_failures(proposed, config, benchmark_metrics=incumbent)

    assert failures == []
