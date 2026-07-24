from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from autoalpha.backtest.ashare_vector import ASHARE_PROXY_RETURN_CONVENTION
from autoalpha.backtest.timing import EOD_NEXT_OPEN_RETURN_CONVENTION
from autoalpha.config import ResearchConfig
from autoalpha.dsl.expression import Expression, FactorDefinition
from autoalpha.service.canonical_evaluation import CANONICAL_LIBRARY_PROTOCOL
from autoalpha.service.evaluator import PortfolioEvaluation, PriceVolumeEvaluator
from autoalpha.service.store import ServiceStore

ADD_WEIGHT_GRID = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.70)
REPLACE_WEIGHT_GRID = (0.20, 0.40, 0.50, 0.60, 0.70)
OPTION_EVALUATION_WORKERS = min(2, max(1, os.cpu_count() or 1))


@dataclass(frozen=True)
class PortfolioDecision:
    action: str
    accepted: bool
    reason: str
    candidate_id: str | None
    removed_factor_id: str | None
    factors: tuple[FactorDefinition, ...]
    weights: tuple[float, ...]
    evaluation: PortfolioEvaluation
    failed_gates: tuple[str, ...]
    utility_change: float
    option_diagnostics: tuple[dict[str, Any], ...] = ()


class MultiFactorResearchEngine:
    """Deterministic add/drop/replace loop over a persistent factor pool."""

    def __init__(
        self,
        store: ServiceStore,
        evaluator: PriceVolumeEvaluator,
        config: ResearchConfig,
        *,
        maximum_factors: int = 5,
        run_id: str | None = None,
        source_task_id: str = "legacy-ashare",
    ) -> None:
        if maximum_factors < 1:
            raise ValueError("maximum_factors must be positive")
        self.store = store
        self.evaluator = evaluator
        self.config = config
        self.maximum_factors = maximum_factors
        self.run_id = run_id
        self.source_task_id = source_task_id

    def sync_legacy_pool(self) -> int:
        return self.store.bootstrap_factor_pool()

    def active_factors(self) -> list[FactorDefinition]:
        return [factor for factor, _ in self.active_components()]

    def active_components(self) -> list[tuple[FactorDefinition, float]]:
        portfolio = self.store.active_portfolio(run_id=self.run_id)
        if not portfolio:
            return []
        stored_protocol = portfolio.get("metrics", {}).get("portfolio_evaluation_protocol")
        if stored_protocol != self.config.governance.protocol_version:
            return []
        result = []
        for member in portfolio["members"]:
            record = self.store.factor_pool_record(member["factor_id"])
            if record is None:
                raise RuntimeError(f"Active factor is missing from pool: {member['factor_id']}")
            result.append((factor_from_pool_record(record), float(member["weight"])))
        return result

    def bootstrap_champion(self, run_id: str, iteration: int) -> PortfolioDecision | None:
        if self.active_components():
            return None
        contaminated = self.store.contaminated_factor_ids()
        eligible = [
            record
            for record in self.store.factor_pool()
            if record["status"] == "ELIGIBLE"
            and record.get("source_task_id", "legacy-ashare") == self.source_task_id
            and record["factor_id"] not in contaminated
            and record.get("metrics", {}).get("evaluation_protocol")
            == CANONICAL_LIBRARY_PROTOCOL
        ]
        if not eligible:
            return None
        champion = max(eligible, key=lambda record: _single_factor_utility(record["metrics"]))
        factor = factor_from_pool_record(champion)
        evaluation = self.evaluator.evaluate_portfolio([factor])
        decision = PortfolioDecision(
            action="BOOTSTRAP",
            accepted=True,
            reason="Highest deterministic utility among the legacy eligible factor pool",
            candidate_id=factor.factor_id,
            removed_factor_id=None,
            factors=(factor,),
            weights=(1.0,),
            evaluation=evaluation,
            failed_gates=(),
            utility_change=0.0,
        )
        self.persist(run_id, iteration, decision)
        return decision

    def register_candidate(
        self,
        factor: FactorDefinition,
        iteration: int,
        proposal: dict[str, Any],
        metrics: dict[str, Any],
    ) -> bool:
        admission_failures = _library_admission_failures(metrics, self.config)
        promotion_failures = _candidate_screen_failures(metrics, self.config)
        admitted = not admission_failures
        metrics.update(
            {
                "library_admission_gate_passed": admitted,
                "library_admission_gate_failures": admission_failures,
                "production_promotion_gate_passed": not promotion_failures,
                "production_promotion_gate_failures": promotion_failures,
                "gate_model": (
                    "LIBRARY_ADMISSION_THEN_PORTFOLIO_CONTRIBUTION_THEN_PRODUCTION"
                ),
            }
        )
        self.store.upsert_factor_pool(
            factor_id=factor.factor_id,
            source_iteration=iteration,
            proposal=proposal,
            metrics=metrics,
            status="ELIGIBLE" if admitted else "SCREENED_OUT",
            status_reason=(
                "library admission passed; production promotion pending"
                if admitted and promotion_failures
                else "library and production gates passed"
                if admitted
                else ", ".join(admission_failures)
            ),
            source_task_id=self.source_task_id,
        )
        return admitted

    def decide(self, candidate: FactorDefinition, *, candidate_eligible: bool) -> PortfolioDecision:
        active_components = self.active_components()
        active = [factor for factor, _ in active_components]
        if not active:
            evaluation = self.evaluator.evaluate_portfolio([candidate])
            failures = tuple(_absolute_portfolio_gate_failures(evaluation.metrics, self.config))
            return PortfolioDecision(
                action="ADD" if candidate_eligible and not failures else "HOLD",
                accepted=candidate_eligible and not failures,
                reason=(
                    "Initial portfolio candidate passed" if not failures else ", ".join(failures)
                ),
                candidate_id=candidate.factor_id,
                removed_factor_id=None,
                factors=(candidate,) if candidate_eligible and not failures else (),
                weights=(1.0,) if candidate_eligible and not failures else (),
                evaluation=evaluation,
                failed_gates=failures,
                utility_change=0.0,
            )

        active_weights = _normalize_weights([weight for _, weight in active_components])
        current = self.evaluator.evaluate_portfolio(active, weights=active_weights)
        options: list[tuple[str, list[FactorDefinition], tuple[float, ...], str | None]] = []
        if candidate_eligible:
            if len(active) < self.maximum_factors:
                for candidate_weight in ADD_WEIGHT_GRID:
                    options.append(
                        (
                            "ADD",
                            [*active, candidate],
                            tuple(weight * (1.0 - candidate_weight) for weight in active_weights)
                            + (candidate_weight,),
                            None,
                        )
                    )
            for removed_index, removed in enumerate(active):
                replacement_factors = [
                    factor for index, factor in enumerate(active) if index != removed_index
                ]
                remaining_weights = [
                    weight for index, weight in enumerate(active_weights) if index != removed_index
                ]
                if not replacement_factors:
                    options.append(("REPLACE", [candidate], (1.0,), removed.factor_id))
                    continue
                normalized_remaining = _normalize_weights(remaining_weights)
                for candidate_weight in REPLACE_WEIGHT_GRID:
                    options.append(
                        (
                            "REPLACE",
                            [*replacement_factors, candidate],
                            tuple(
                                weight * (1.0 - candidate_weight) for weight in normalized_remaining
                            )
                            + (candidate_weight,),
                            removed.factor_id,
                        )
                    )
        if len(active) > 1:
            for removed_index, removed in enumerate(active):
                remaining_factors = [
                    factor for index, factor in enumerate(active) if index != removed_index
                ]
                remaining_weights = [
                    weight for index, weight in enumerate(active_weights) if index != removed_index
                ]
                options.append(
                    (
                        "REMOVE",
                        remaining_factors,
                        _normalize_weights(remaining_weights),
                        removed.factor_id,
                    )
                )

        prime_signals = getattr(self.evaluator, "prime_factor_signals", None)
        if callable(prime_signals):
            prime_signals([*active, candidate])
        incumbent_failures = _absolute_portfolio_gate_failures(current.metrics, self.config)

        def evaluate_option(
            option: tuple[str, list[FactorDefinition], tuple[float, ...], str | None],
        ) -> PortfolioDecision:
            action, factors, weights, removed_factor_id = option
            evaluation = self.evaluator.evaluate_portfolio(
                factors,
                weights=weights,
                benchmark_factors=active,
                benchmark_weights=active_weights,
            )
            proposed_failures = _absolute_portfolio_gate_failures(evaluation.metrics, self.config)
            feasibility_recovery = _is_feasibility_recovery(
                evaluation.metrics, current.metrics, self.config
            )
            evaluation.metrics.update(
                {
                    "portfolio_incumbent_absolute_failures": incumbent_failures,
                    "portfolio_proposed_absolute_failures": proposed_failures,
                    "portfolio_feasibility_recovery": feasibility_recovery,
                    "portfolio_absolute_violation_score": _absolute_violation_score(
                        evaluation.metrics, self.config
                    ),
                }
            )
            failures = tuple(
                _portfolio_action_gate_failures(
                    evaluation.metrics,
                    self.config,
                    benchmark_metrics=current.metrics,
                )
            )
            improvement = float(evaluation.metrics["portfolio_sharpe_change"])
            return PortfolioDecision(
                action=action,
                accepted=not failures,
                reason=(
                    "all portfolio action gates passed" if not failures else ", ".join(failures)
                ),
                candidate_id=candidate.factor_id if action in {"ADD", "REPLACE"} else None,
                removed_factor_id=removed_factor_id,
                factors=tuple(factors),
                weights=weights,
                evaluation=evaluation,
                failed_gates=failures,
                utility_change=improvement,
            )

        if len(options) > 1:
            with ThreadPoolExecutor(
                max_workers=min(OPTION_EVALUATION_WORKERS, len(options)),
                thread_name_prefix="portfolio-option",
            ) as executor:
                evaluated = list(executor.map(evaluate_option, options))
        else:
            evaluated = [evaluate_option(option) for option in options]
        passing = [option for option in evaluated if option.accepted]
        diagnostics = tuple(_option_diagnostic(option) for option in evaluated)
        if passing:
            winner = max(passing, key=_portfolio_selection_key)
            return PortfolioDecision(
                **{
                    **winner.__dict__,
                    "option_diagnostics": diagnostics,
                }
            )
        candidate_rejections = [
            option for option in evaluated if option.action in {"ADD", "REPLACE"}
        ]
        best_rejected = max(
            candidate_rejections or evaluated,
            key=lambda option: _rejected_option_key(option, self.config),
            default=None,
        )
        failed_gates = best_rejected.failed_gates if best_rejected else ()
        near_miss = (
            f"; best={best_rejected.action} utility_change={best_rejected.utility_change:.4f} "
            f"failed={','.join(best_rejected.failed_gates) or 'none'}"
            if best_rejected
            else ""
        )
        return PortfolioDecision(
            action="HOLD",
            accepted=False,
            reason=(
                "candidate failed deterministic screen"
                if not candidate_eligible
                else "no add/drop/replace action improved the active portfolio through all gates"
            )
            + near_miss,
            candidate_id=candidate.factor_id,
            removed_factor_id=None,
            factors=tuple(active),
            weights=active_weights,
            evaluation=current,
            failed_gates=failed_gates,
            utility_change=0.0,
            option_diagnostics=diagnostics,
        )

    def persist(self, run_id: str, iteration: int, decision: PortfolioDecision) -> int:
        members = [
            (factor.factor_id, weight)
            for factor, weight in zip(decision.factors, decision.weights, strict=True)
        ]
        version_id = self.store.record_portfolio_decision(
            run_id=run_id,
            iteration=iteration,
            action=decision.action,
            candidate_id=decision.candidate_id,
            removed_factor_id=decision.removed_factor_id,
            accepted=decision.accepted,
            reason=decision.reason,
            metrics={
                **decision.evaluation.metrics,
                "portfolio_action_gate_failures": list(decision.failed_gates),
                "portfolio_utility_change": decision.utility_change,
                "portfolio_selection_policy": (
                    "alpha_screen_then_ashare_strategy_hard_gates_then_lexicographic_robustness"
                ),
                "factor_correlations": decision.evaluation.factor_correlations,
                "portfolio_option_diagnostics": list(decision.option_diagnostics),
            },
            members=members,
        )
        if decision.accepted:
            active_ids = {factor.factor_id for factor in decision.factors}
            for record in self.store.factor_pool():
                if record.get("source_task_id", "legacy-ashare") != self.source_task_id:
                    continue
                if record["factor_id"] in active_ids:
                    self.store.update_factor_status(
                        record["factor_id"], "ACTIVE", f"active in portfolio version {version_id}"
                    )
                elif record["status"] == "ACTIVE":
                    self.store.update_factor_status(
                        record["factor_id"],
                        "ELIGIBLE",
                        f"removed in portfolio version {version_id}",
                    )
        return version_id

    def governance_veto(
        self,
        decision: PortfolioDecision,
        *,
        gate: str,
        reason: str,
    ) -> PortfolioDecision:
        active_components = self.active_components()
        active = [factor for factor, _ in active_components]
        active_weights = (
            _normalize_weights([weight for _, weight in active_components])
            if active_components
            else ()
        )
        current = (
            self.evaluator.evaluate_portfolio(active, weights=active_weights)
            if active
            else decision.evaluation
        )
        attempted = {
            "action": decision.action,
            "accepted": False,
            "candidate_id": decision.candidate_id,
            "removed_factor_id": decision.removed_factor_id,
            "factor_ids": [factor.factor_id for factor in decision.factors],
            "weights": list(decision.weights),
            "failed_gates": [gate],
            "utility_change": decision.utility_change,
            "metrics": decision.evaluation.metrics,
            "factor_correlations": decision.evaluation.factor_correlations,
            "governance_verdict": reason,
        }
        return PortfolioDecision(
            action="HOLD",
            accepted=False,
            reason=f"{gate}: {reason}",
            candidate_id=decision.candidate_id,
            removed_factor_id=None,
            factors=tuple(active),
            weights=active_weights,
            evaluation=current,
            failed_gates=(gate,),
            utility_change=0.0,
            option_diagnostics=(*decision.option_diagnostics, attempted),
        )


def factor_from_pool_record(record: dict[str, Any]) -> FactorDefinition:
    proposal = record["proposal"]
    return FactorDefinition(
        name=str(proposal["name"]),
        family=str(proposal["family"]),
        hypothesis=str(proposal["hypothesis"]),
        expression=Expression.from_dict(proposal["expression"]),
        expected_direction=int(proposal.get("expected_direction", 1)),
    )


def _portfolio_selection_key(option: PortfolioDecision) -> tuple[float, ...]:
    """Rank gate-passing options without allowing cross-objective compensation."""
    metrics = option.evaluation.metrics
    return (
        float(metrics.get("portfolio_walk_forward_worst_sharpe", -100.0)),
        float(metrics["portfolio_max_drawdown"]),
        float(metrics["portfolio_incremental_net_ir"]),
        float(metrics["portfolio_simple_annual_return"]),
        -float(metrics["portfolio_annual_turnover"]),
        -float(metrics["portfolio_max_factor_correlation"]),
    )


def _single_factor_utility(metrics: dict[str, Any]) -> float:
    return float(
        metrics.get("long_only_sharpe_ratio", -100.0)
        + 0.75 * metrics.get("long_only_simple_annual_return", 0.0)
        + 0.50 * metrics.get("long_only_max_drawdown", -1.0)
        - 0.002 * metrics.get("long_only_annual_turnover", 1_000.0)
        - 0.25 * metrics.get("long_only_annual_return_dispersion", 1.0)
    )


def _candidate_screen_failures(metrics: dict[str, Any], config: ResearchConfig) -> list[str]:
    policy = config.evaluation
    checks = {
        "data_basis_incompatible": bool(metrics.get("data_basis_compatible", True)),
        "stale_evaluation_protocol": (
            metrics.get("evaluation_protocol") == config.governance.protocol_version
        ),
        "invalid_return_convention": (
            metrics.get("long_only_return_convention") == ASHARE_PROXY_RETURN_CONVENTION
        ),
        "capital_survival": not bool(metrics.get("long_only_bankrupt", False)),
        "non_positive_sharpe": float(metrics.get("long_only_sharpe_ratio", -1)) > 0,
        "non_positive_annual_return": float(metrics.get("long_only_simple_annual_return", -1)) > 0,
        "insufficient_coverage": float(metrics.get("long_only_coverage", 0)) >= 0.80,
        "cost_stress": float(metrics.get("long_only_cost_stress_net_ir", -1)) > 0,
        "excessive_turnover": float(metrics.get("long_only_annual_turnover", float("inf")))
        <= 2.0 * policy.maximum_annual_turnover,
        "unstable_annual_returns": float(
            metrics.get("long_only_annual_return_dispersion", float("inf"))
        )
        <= 1.5 * policy.maximum_annual_return_dispersion,
        "insufficient_walk_forward_folds": int(metrics.get("long_only_walk_forward_fold_count", 0))
        >= config.walk_forward.minimum_folds,
        "unstable_walk_forward": float(metrics.get("long_only_walk_forward_positive_fraction", 0.0))
        >= policy.minimum_positive_fold_fraction,
        "weak_worst_fold": float(metrics.get("long_only_walk_forward_worst_sharpe", -100.0))
        >= policy.minimum_worst_fold_net_ir,
        "deflated_sharpe": float(metrics.get("long_only_deflated_sharpe_probability", 0.0))
        >= policy.minimum_deflated_sharpe_probability,
        "net_return_significance": float(metrics.get("long_only_net_return_hac_p_value", 1.0))
        <= policy.maximum_net_return_p_value,
        "parameter_instability": float(metrics.get("parameter_stability_positive_fraction", 0.0))
        >= policy.minimum_parameter_positive_fraction,
        "parameter_cliff": float(metrics.get("parameter_stability_worst_sharpe", -100.0))
        >= policy.minimum_parameter_worst_sharpe,
        "multiple_testing_fdr": bool(metrics.get("multiple_testing_fdr_passed", False)),
        "backtest_overfitting": float(metrics.get("probability_backtest_overfitting", 1.0))
        <= policy.maximum_probability_backtest_overfitting,
    }
    return [name for name, passed in checks.items() if not passed]


def _library_admission_failures(metrics: dict[str, Any], config: ResearchConfig) -> list[str]:
    """Retain bounded weak signals for portfolio tests without relaxing promotion gates."""
    long_sharpe = float(metrics.get("long_only_sharpe_ratio", -100.0))
    active_ir = float(metrics.get("long_only_active_information_ratio", long_sharpe))
    rank_ic = abs(float(metrics.get("rank_ic_mean", 0.0)))
    checks = {
        "data_basis_incompatible": bool(metrics.get("data_basis_compatible", True)),
        "stale_evaluation_protocol": (
            metrics.get("evaluation_protocol") == CANONICAL_LIBRARY_PROTOCOL
        ),
        "invalid_return_convention": (
            metrics.get("long_only_return_convention") == ASHARE_PROXY_RETURN_CONVENTION
        ),
        "capital_survival": not bool(metrics.get("long_only_bankrupt", False)),
        "insufficient_coverage": float(metrics.get("long_only_coverage", 0.0))
        >= config.evaluation.minimum_coverage,
        "no_economic_signal": max(long_sharpe, active_ir, rank_ic * 100.0) > 0.0,
        "unbounded_turnover": float(metrics.get("long_only_annual_turnover", float("inf")))
        <= 4.0 * config.evaluation.maximum_annual_turnover,
        "insufficient_walk_forward_folds": int(
            metrics.get("long_only_walk_forward_fold_count", 0)
        )
        >= config.walk_forward.minimum_folds,
        # Behavioral redundancy: near-duplicate signals stay out of the eligible
        # pool unless the candidate is strong enough to justify coexistence.
        "signal_redundancy": (
            float(metrics.get("library_signal_correlation_max", 0.0))
            <= config.evaluation.maximum_library_correlation
            or float(metrics.get("long_only_sharpe_ratio", -100.0))
            >= config.evaluation.redundancy_override_net_ir
        ),
    }
    return [name for name, passed in checks.items() if not passed]


def _normalize_weights(weights: list[float] | tuple[float, ...]) -> tuple[float, ...]:
    total = sum(weights)
    if not weights or total <= 0:
        raise ValueError("Portfolio weights must contain positive mass")
    return tuple(float(weight / total) for weight in weights)


def _option_diagnostic(option: PortfolioDecision) -> dict[str, Any]:
    return {
        "action": option.action,
        "accepted": option.accepted,
        "candidate_id": option.candidate_id,
        "removed_factor_id": option.removed_factor_id,
        "factor_ids": [factor.factor_id for factor in option.factors],
        "weights": list(option.weights),
        "failed_gates": list(option.failed_gates),
        "utility_change": option.utility_change,
        "gate_failure_count": len(option.failed_gates),
        "feasibility_recovery": bool(
            option.evaluation.metrics.get("portfolio_feasibility_recovery", False)
        ),
        "absolute_violation_score": option.evaluation.metrics.get(
            "portfolio_absolute_violation_score"
        ),
        "metrics": option.evaluation.metrics,
        "factor_correlations": option.evaluation.factor_correlations,
    }


def _absolute_portfolio_gate_failures(metrics: dict[str, Any], config: ResearchConfig) -> list[str]:
    policy = config.evaluation
    strategy_basis = metrics.get("portfolio_strategy_gate_basis")
    expected_return_convention = (
        ASHARE_PROXY_RETURN_CONVENTION
        if strategy_basis == "A_SHARE_LONG_ONLY_WEEKLY_NON_PIT_PROXY"
        else EOD_NEXT_OPEN_RETURN_CONVENTION
    )
    checks = {
        "stale_evaluation_protocol": (
            metrics.get("portfolio_evaluation_protocol") == config.governance.protocol_version
        ),
        "invalid_return_convention": (
            metrics.get("portfolio_return_convention") == expected_return_convention
        ),
        "strategy_execution_proxy": (
            strategy_basis == "A_SHARE_LONG_ONLY_WEEKLY_NON_PIT_PROXY"
            if strategy_basis is not None
            else True
        ),
        "coverage": metrics["portfolio_coverage"] >= policy.minimum_coverage,
        "cost_stress": metrics["portfolio_cost_stress_net_ir"] >= policy.minimum_cost_stress_net_ir,
        "turnover": metrics["portfolio_annual_turnover"] <= policy.maximum_annual_turnover,
        "capacity": metrics["portfolio_capacity_cny"] >= policy.minimum_capacity_cny,
        "residual_positions": int(
            metrics.get(
                "portfolio_maximum_observed_positions",
                config.strategy_evaluation.maximum_positions,
            )
        )
        <= math.ceil(
            config.strategy_evaluation.maximum_positions
            * config.portfolio.maximum_residual_position_multiplier
        ),
        "annual_dispersion": (
            metrics["portfolio_annual_return_dispersion"] <= policy.maximum_annual_return_dispersion
        ),
        "walk_forward_folds": int(
            metrics.get("portfolio_walk_forward_fold_count", config.walk_forward.minimum_folds)
        )
        >= config.walk_forward.minimum_folds,
        "walk_forward_positive_fraction": float(
            metrics.get(
                "portfolio_walk_forward_positive_fraction", policy.minimum_positive_fold_fraction
            )
        )
        >= policy.minimum_positive_fold_fraction,
        "walk_forward_worst_sharpe": float(
            metrics.get("portfolio_walk_forward_worst_sharpe", policy.minimum_worst_fold_net_ir)
        )
        >= policy.minimum_worst_fold_net_ir,
        "deflated_sharpe": float(
            metrics.get(
                "portfolio_deflated_sharpe_probability",
                policy.minimum_deflated_sharpe_probability,
            )
        )
        >= policy.minimum_deflated_sharpe_probability,
        "net_return_significance": float(
            metrics.get("portfolio_net_return_hac_p_value", policy.maximum_net_return_p_value)
        )
        <= policy.maximum_net_return_p_value,
    }
    return [name for name, passed in checks.items() if not passed]


def _portfolio_action_gate_failures(
    metrics: dict[str, Any],
    config: ResearchConfig,
    *,
    benchmark_metrics: dict[str, Any] | None = None,
) -> list[str]:
    policy = config.evaluation
    failures = (
        _progressive_absolute_gate_failures(metrics, benchmark_metrics, config)
        if benchmark_metrics is not None
        else _absolute_portfolio_gate_failures(metrics, config)
    )
    return_accretion = (
        metrics["portfolio_incremental_net_ir"] >= policy.minimum_incremental_net_ir
        and metrics["portfolio_incremental_annual_return"]
        >= policy.minimum_incremental_annual_return
    )
    diversification_upgrade = (
        metrics["portfolio_sharpe_change"] >= policy.minimum_diversification_sharpe_improvement
        and metrics["portfolio_max_drawdown_change"]
        >= policy.minimum_diversification_drawdown_improvement
        and metrics["portfolio_annual_return_change"]
        >= -policy.maximum_transition_annual_return_sacrifice
        and metrics["portfolio_cost_stress_net_ir_change"] >= 0.0
    )
    stability_upgrade = False
    if benchmark_metrics is not None:
        stability_upgrade = (
            benchmark_metrics["portfolio_annual_return_dispersion"]
            - metrics["portfolio_annual_return_dispersion"]
            >= policy.minimum_stability_dispersion_reduction
            and metrics["portfolio_walk_forward_worst_sharpe"]
            - benchmark_metrics["portfolio_walk_forward_worst_sharpe"]
            >= policy.minimum_stability_worst_fold_sharpe_improvement
            and metrics["portfolio_annual_return_change"]
            >= -policy.maximum_transition_annual_return_sacrifice
            and metrics["portfolio_max_drawdown_change"]
            >= -policy.maximum_incremental_drawdown_deterioration
            and metrics["portfolio_cost_stress_net_ir_change"] >= 0.0
        )
    feasibility_recovery = bool(
        benchmark_metrics is not None
        and _is_feasibility_recovery(metrics, benchmark_metrics, config)
    )
    checks = {
        "portfolio_value": (
            return_accretion or diversification_upgrade or stability_upgrade or feasibility_recovery
        ),
        "incremental_drawdown": (
            metrics["portfolio_incremental_max_drawdown"]
            >= -policy.maximum_incremental_drawdown_deterioration
        ),
        "factor_correlation": (
            metrics["portfolio_max_factor_correlation"] <= policy.maximum_library_correlation
        ),
    }
    failures.extend(name for name, passed in checks.items() if not passed)
    return failures


def _absolute_gate_measurements(
    metrics: dict[str, Any], config: ResearchConfig
) -> dict[str, tuple[float, float, str, float]]:
    policy = config.evaluation
    return {
        "coverage": (metrics["portfolio_coverage"], policy.minimum_coverage, "min", 0.01),
        "cost_stress": (
            metrics["portfolio_cost_stress_net_ir"],
            policy.minimum_cost_stress_net_ir,
            "min",
            0.05,
        ),
        "turnover": (
            metrics["portfolio_annual_turnover"],
            policy.maximum_annual_turnover,
            "max",
            1.0,
        ),
        "capacity": (
            metrics["portfolio_capacity_cny"],
            policy.minimum_capacity_cny,
            "min",
            0.05 * policy.minimum_capacity_cny,
        ),
        "residual_positions": (
            float(
                metrics.get(
                    "portfolio_maximum_observed_positions",
                    config.strategy_evaluation.maximum_positions,
                )
            ),
            float(
                math.ceil(
                    config.strategy_evaluation.maximum_positions
                    * config.portfolio.maximum_residual_position_multiplier
                )
            ),
            "max",
            1.0,
        ),
        "annual_dispersion": (
            metrics["portfolio_annual_return_dispersion"],
            policy.maximum_annual_return_dispersion,
            "max",
            policy.minimum_stability_dispersion_reduction,
        ),
        "walk_forward_folds": (
            float(metrics.get("portfolio_walk_forward_fold_count", 0)),
            float(config.walk_forward.minimum_folds),
            "min",
            1.0,
        ),
        "walk_forward_positive_fraction": (
            float(metrics.get("portfolio_walk_forward_positive_fraction", 0.0)),
            policy.minimum_positive_fold_fraction,
            "min",
            0.10,
        ),
        "walk_forward_worst_sharpe": (
            float(metrics.get("portfolio_walk_forward_worst_sharpe", -100.0)),
            policy.minimum_worst_fold_net_ir,
            "min",
            policy.minimum_stability_worst_fold_sharpe_improvement,
        ),
        "deflated_sharpe": (
            float(metrics.get("portfolio_deflated_sharpe_probability", 0.0)),
            policy.minimum_deflated_sharpe_probability,
            "min",
            0.05,
        ),
        "net_return_significance": (
            float(metrics.get("portfolio_net_return_hac_p_value", 1.0)),
            policy.maximum_net_return_p_value,
            "max",
            0.01,
        ),
    }


def _progressive_absolute_gate_failures(
    metrics: dict[str, Any],
    benchmark_metrics: dict[str, Any],
    config: ResearchConfig,
) -> list[str]:
    proposed = _absolute_gate_measurements(metrics, config)
    benchmark = _absolute_gate_measurements(benchmark_metrics, config)
    proposed_failures = set(_absolute_portfolio_gate_failures(metrics, config))
    benchmark_failures = set(_absolute_portfolio_gate_failures(benchmark_metrics, config))
    unresolved: list[str] = []
    for gate in proposed_failures:
        if gate not in benchmark_failures:
            unresolved.append(gate)
            continue
        value, _, direction, minimum_step = proposed[gate]
        benchmark_value = benchmark[gate][0]
        improvement = value - benchmark_value if direction == "min" else benchmark_value - value
        if improvement + 1e-12 < minimum_step:
            unresolved.append(gate)
    return sorted(unresolved)


def _is_feasibility_recovery(
    metrics: dict[str, Any],
    benchmark_metrics: dict[str, Any],
    config: ResearchConfig,
) -> bool:
    incumbent_failures = _absolute_portfolio_gate_failures(benchmark_metrics, config)
    if not incumbent_failures:
        return False
    if _progressive_absolute_gate_failures(metrics, benchmark_metrics, config):
        return False
    return (
        metrics["portfolio_annual_return_change"]
        >= -config.evaluation.maximum_transition_annual_return_sacrifice
        and metrics["portfolio_max_drawdown_change"]
        >= -config.evaluation.maximum_incremental_drawdown_deterioration
        and metrics["portfolio_cost_stress_net_ir_change"] >= 0.0
    )


def _absolute_violation_score(metrics: dict[str, Any], config: ResearchConfig) -> float:
    score = 0.0
    for value, threshold, direction, _ in _absolute_gate_measurements(metrics, config).values():
        scale = max(abs(threshold), 0.10)
        violation = threshold - value if direction == "min" else value - threshold
        score += max(0.0, violation) / scale
    return float(score)


def _rejected_option_key(option: PortfolioDecision, config: ResearchConfig) -> tuple[float, ...]:
    return (
        -float(len(option.failed_gates)),
        -_absolute_violation_score(option.evaluation.metrics, config),
        float(option.evaluation.metrics.get("portfolio_walk_forward_worst_sharpe", -100.0)),
        option.utility_change,
    )
