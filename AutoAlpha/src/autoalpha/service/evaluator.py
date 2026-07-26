from __future__ import annotations

import math
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from autoalpha.backtest.ashare_vector import (
    ASHARE_PROXY_RETURN_CONVENTION,
    AshareVectorBacktester,
    AshareVectorConfig,
)
from autoalpha.backtest.timing import (
    EOD_NEXT_OPEN_RETURN_CONVENTION,
    next_open_return_for_eod_signal,
)
from autoalpha.config import ResearchConfig
from autoalpha.data.execution_basis import (
    expression_research_basis_blockers,
    inspect_execution_data_basis,
)
from autoalpha.data.research_fields import (
    build_research_data_capabilities,
    expression_fields,
    field_definitions,
)
from autoalpha.data.workspace import DataWorkspaceReport, inspect_data_workspace
from autoalpha.dsl.compiler import FactorCompiler
from autoalpha.dsl.expression import Expression, FactorDefinition
from autoalpha.dsl.semantics import SemanticValidator
from autoalpha.research.evaluation import cross_sectional_ic
from autoalpha.research.incremental import annual_robustness, compare_portfolios
from autoalpha.research.multiple_testing import deflated_sharpe_ratio
from autoalpha.research.statistics import hac_mean_inference
from autoalpha.service.autocombine_intelligence import signal_independence_metrics
from autoalpha.service.research_protocol import research_evidence_tier

_SHARED_SIGNAL_CACHE: OrderedDict[
    tuple[str, str, str, str, int], pd.DataFrame
] = OrderedDict()
_SHARED_SIGNAL_CACHE_LOCK = Lock()
_SHARED_SIGNAL_CACHE_MAX = max(0, int(os.getenv("AUTOALPHA_SHARED_SIGNAL_CACHE_SIZE", "8")))


@dataclass(frozen=True)
class ExploratoryEvaluation:
    metrics: dict[str, Any]
    decision: str
    observations: int
    net_returns: pd.Series


@dataclass(frozen=True)
class PortfolioEvaluation:
    metrics: dict[str, Any]
    net_returns: pd.Series
    factor_correlations: dict[str, float]


@dataclass(frozen=True)
class CandidatePreflight:
    passed: bool
    failures: tuple[str, ...]
    warnings: tuple[str, ...]
    metrics: dict[str, Any]


class PriceVolumeEvaluator:
    """Real price/volume evaluation that cannot produce production approval."""

    def __init__(
        self,
        data_path: Path,
        config_path: Path | None = None,
        *,
        config: ResearchConfig | None = None,
    ) -> None:
        self.data_path = data_path
        self.workspace: DataWorkspaceReport = inspect_data_workspace(data_path)
        self.workspace.require_price_research()
        self.panel_path = Path(self.workspace.panel_path)
        if config is None and config_path is None:
            raise ValueError("Either config_path or config is required")
        self.config = config or ResearchConfig.from_toml(config_path)  # type: ignore[arg-type]
        self.data_capabilities = build_research_data_capabilities(
            self.workspace,
            required_start=self.config.splits.train.start,
            required_end=self.config.splits.validation.end,
        )
        self.factor_fields = tuple(self.data_capabilities["eligible_fields"])
        self.research_evidence_tier = research_evidence_tier(self.config)
        basis = inspect_execution_data_basis(self.panel_path)
        self.execution_basis = basis
        fields = field_definitions(
            self.factor_fields,
            amount_unit=basis.amount_unit,
            volume_unit=basis.volume_unit,
        )
        self.validator = SemanticValidator(fields, maximum_nodes=30, maximum_lookback=252)
        self.compiler = FactorCompiler(self.validator)
        self._fields: dict[str, pd.DataFrame] | None = None
        self._signal_cache: OrderedDict[str, pd.DataFrame] = OrderedDict()
        self._signal_cache_lock = Lock()
        self._preflight_cache: OrderedDict[str, CandidatePreflight] = OrderedDict()
        self._preflight_cache_lock = Lock()
        self._portfolio_path_cache: OrderedDict[
            tuple[tuple[str, float], ...], tuple[pd.DataFrame, pd.DataFrame]
        ] = OrderedDict()
        self._portfolio_path_cache_lock = Lock()
        self._field_load_lock = Lock()
        self.trial_count = 1

    def set_trial_count(self, value: int) -> None:
        self.trial_count = max(1, int(value))

    def preflight(self, factor: FactorDefinition) -> CandidatePreflight:
        """Cheap signal viability checks before inference and parameter perturbations."""
        cache = getattr(self, "_preflight_cache", None)
        lock = getattr(self, "_preflight_cache_lock", None)
        if cache is not None and lock is not None:
            with lock:
                cached = cache.get(factor.factor_id)
                if cached is not None:
                    cache.move_to_end(factor.factor_id)
                    return cached
        started = perf_counter()
        self.validator.validate(factor.expression)
        signal = self._factor_signal(factor)
        validation = signal.loc[
            (signal.index >= pd.Timestamp(self.config.splits.validation.start))
            & (signal.index <= pd.Timestamp(self.config.splits.validation.end))
        ]
        if validation.empty:
            raise ValueError("Candidate preflight has no public validation observations")
        stride = max(1, len(validation) // 512)
        sampled = validation.iloc[::stride]
        effective_names = sampled.notna().sum(axis=1)
        dispersion = sampled.std(axis=1, skipna=True)
        valid_dates = effective_names.ge(self.config.minimum_cross_section) & dispersion.gt(1e-12)
        unique_fraction = sampled.nunique(axis=1, dropna=True).div(
            effective_names.replace(0, np.nan)
        )
        ranked = sampled.rank(axis=1, pct=True)
        paired = ranked.notna() & ranked.shift(1).notna()
        changed = ranked.diff().abs().gt(1e-8) & paired
        paired_count = int(paired.to_numpy().sum())
        temporal_update_rate = (
            float(changed.to_numpy().sum() / paired_count) if paired_count else 0.0
        )
        coverage = self._dynamic_coverage(signal)
        median_names = float(effective_names.median()) if not effective_names.empty else 0.0
        median_unique_fraction = (
            float(unique_fraction.median()) if unique_fraction.notna().any() else 0.0
        )
        median_cross_sectional_std = (
            float(dispersion.median()) if dispersion.notna().any() else 0.0
        )
        failures: list[str] = []
        warnings: list[str] = []
        if int(valid_dates.sum()) < 20:
            failures.append("insufficient_cross_sectional_dates")
        if median_names < self.config.minimum_cross_section:
            failures.append("insufficient_effective_names")
        if coverage < max(0.25, self.config.evaluation.minimum_coverage * 0.5):
            failures.append("severe_coverage_shortfall")
        if median_unique_fraction < 0.005 or median_cross_sectional_std <= 1e-12:
            failures.append("cross_sectionally_degenerate")
        if int(valid_dates.sum()) < 60:
            warnings.append("sparse_prediction_diagnostics")
        if temporal_update_rate < 0.01:
            warnings.append("low_temporal_update_rate")
        if median_unique_fraction < 0.02:
            warnings.append("tie_heavy_cross_section")
        metrics = {
            "protocol": "SIGNAL_PREFLIGHT_V1",
            "factor_id": factor.factor_id,
            "factor_fields": sorted(expression_fields(factor.expression)),
            "sample_stride": stride,
            "sampled_dates": len(sampled),
            "valid_cross_sectional_dates": int(valid_dates.sum()),
            "median_effective_names": median_names,
            "median_unique_fraction": median_unique_fraction,
            "median_cross_sectional_std": median_cross_sectional_std,
            "temporal_update_rate": temporal_update_rate,
            "coverage": coverage,
            "elapsed_seconds": perf_counter() - started,
        }
        outcome = CandidatePreflight(
            passed=not failures,
            failures=tuple(failures),
            warnings=tuple(warnings),
            metrics=metrics,
        )
        if cache is not None and lock is not None:
            with lock:
                cache[factor.factor_id] = outcome
                cache.move_to_end(factor.factor_id)
                while len(cache) > 64:
                    cache.popitem(last=False)
        return outcome

    def evaluate(self, factor: FactorDefinition) -> ExploratoryEvaluation:
        evaluation_started = perf_counter()
        preflight = self.preflight(factor)
        if not preflight.passed:
            raise ValueError(
                "Candidate signal preflight failed: " + ", ".join(preflight.failures)
            )
        fields = self._load_fields(expression_fields(factor.expression))
        signal = self._factor_signal(factor)
        path = self._signal_path(signal)
        evaluation_dates = _walk_forward_dates(path.index, self.config)
        path = path.loc[evaluation_dates]
        signal = signal.reindex(path.index)
        next_return = next_open_return_for_eod_signal(fields["open"]).reindex(path.index)
        rank_ic = cross_sectional_ic(
            signal, next_return, minimum_names=self.config.minimum_cross_section
        )
        pearson_ic = cross_sectional_ic(
            signal,
            next_return,
            method="pearson",
            minimum_names=self.config.minimum_cross_section,
        )
        rank_ic_summary = _ic_diagnostics(rank_ic)
        pearson_ic_summary = _ic_diagnostics(pearson_ic)
        prediction_diagnostics_available = len(rank_ic) >= 60 and len(pearson_ic) >= 60

        net_return = path["net"]
        stressed = path["stressed"]
        turnover = path["turnover"]
        control = pd.Series(0.0, index=net_return.index)
        increment = compare_portfolios(
            control,
            net_return,
            stressed_treatment_net_returns=stressed,
            hac_lags=5,
            bootstrap_samples=500,
            seed=self.config.random_seed,
        )
        annual = annual_robustness(net_return)
        simple_annual_return = float(net_return.mean() * 245)
        compound_annual_return = _compound_annual_return(net_return)
        sharpe_ratio = _annualized_ir(net_return)
        folds = _walk_forward_metrics(path, self.config)
        inference = hac_mean_inference(net_return.to_numpy(), lags=min(5, len(net_return) - 1))
        dsr = deflated_sharpe_ratio(net_return.to_numpy(), trials=self.trial_count)
        _, long_only_all = self._portfolio_paths([factor], (1.0,))
        parameter_stability = self._parameter_stability(factor)
        exploration = self._period_summary(
            long_only_all,
            self.config.splits.train.start,
            self.config.splits.train.end,
        )
        long_only_path = long_only_all.reindex(evaluation_dates).dropna().copy()
        long_only_path.attrs.update(long_only_all.attrs)
        long_only_metrics = _standalone_long_only_metrics(
            long_only_path,
            self.config,
            trials=self.trial_count,
        )
        market_benchmark = self._market_benchmark_returns(long_only_path.index)
        active_metrics = _active_return_metrics(
            long_only_path,
            market_benchmark,
            self.config,
            prefix="long_only",
        )
        neutral_metrics = self._size_neutral_diagnostics(signal, next_return, evaluation_dates)
        coverage = self._dynamic_coverage(signal)
        data_basis_blockers = expression_research_basis_blockers(
            factor.expression.to_dict(), self.execution_basis
        )
        metrics = {
            "prediction_diagnostics_available": prediction_diagnostics_available,
            "prediction_diagnostics_reason": (
                None
                if prediction_diagnostics_available
                else "fewer than 60 valid cross-sectional IC dates; long-only evaluation retained"
            ),
            "rank_ic_observations": len(rank_ic),
            "rank_ic_mean": rank_ic_summary["mean"],
            "rank_ic_ir": rank_ic_summary["ir"],
            "rank_ic_hac_p_value": rank_ic_summary["p_value"],
            "pearson_ic_observations": len(pearson_ic),
            "pearson_ic_mean": pearson_ic_summary["mean"],
            "pearson_ic_ir": pearson_ic_summary["ir"],
            "pearson_ic_hac_p_value": pearson_ic_summary["p_value"],
            "incremental_net_ir": increment.incremental_net_ir,
            "incremental_annual_return": increment.incremental_annual_return,
            "simple_annual_return": simple_annual_return,
            "compound_annual_return": compound_annual_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": _max_drawdown(net_return),
            "incremental_max_drawdown": increment.incremental_max_drawdown,
            "return_drawdown_efficiency_change": increment.return_drawdown_efficiency_change,
            "cost_stress_net_ir": float(increment.cost_stress_net_ir or 0.0),
            "incremental_bootstrap_confidence_interval": (
                list(increment.bootstrap_confidence_interval)
                if increment.bootstrap_confidence_interval is not None
                else None
            ),
            "bootstrap_samples": 500,
            "annual_turnover": float(turnover.mean() * 245),
            "capacity_cny": float(path.attrs["capacity_cny"]),
            "coverage": coverage,
            "positive_year_ratio": annual.positive_year_ratio,
            "worst_year_incremental_return": annual.worst_year_return,
            "annual_return_dispersion": annual.annual_return_dispersion,
            "backtest_start": net_return.index.min().date().isoformat(),
            "backtest_end": net_return.index.max().date().isoformat(),
            "backtest_observations": len(net_return),
            "evaluation_protocol": self.config.governance.protocol_version,
            "research_generation": self.config.generation,
            "research_evidence_tier": self.research_evidence_tier,
            "task_production_promotion_allowed": (
                self.research_evidence_tier == "PRIMARY_DISCOVERY"
            ),
            "holding_period_days": self.config.portfolio.holding_period_days,
            "signal_availability": "END_OF_DAY_AFTER_CLOSE",
            "execution_lag_sessions": 1,
            "return_convention": EOD_NEXT_OPEN_RETURN_CONVENTION,
            "data_basis_compatible": not data_basis_blockers,
            "data_basis_blockers": list(data_basis_blockers),
            "signal_preflight": {
                **preflight.metrics,
                "passed": preflight.passed,
                "failures": list(preflight.failures),
                "warnings": list(preflight.warnings),
            },
            "walk_forward_folds": folds,
            "walk_forward_fold_count": len(folds),
            "walk_forward_positive_fraction": float(
                np.mean([fold["annual_return"] > 0 for fold in folds])
            ),
            "walk_forward_median_sharpe": float(np.median([fold["sharpe"] for fold in folds])),
            "walk_forward_worst_sharpe": float(min(fold["sharpe"] for fold in folds)),
            "walk_forward_worst_drawdown": float(min(fold["max_drawdown"] for fold in folds)),
            "net_return_hac_p_value": inference.p_value,
            "deflated_sharpe_probability": dsr.probability,
            "deflated_sharpe_expected_maximum": dsr.expected_max_sharpe,
            "multiple_testing_trials": self.trial_count,
            "parameter_stability_positive_fraction": parameter_stability["positive_fraction"],
            "parameter_stability_worst_sharpe": parameter_stability["worst_sharpe"],
            "parameter_stability_dispersion": parameter_stability["dispersion"],
            "parameter_neighborhood": parameter_stability["neighborhood"],
            "exploration_metrics": exploration,
            **long_only_metrics,
            **active_metrics,
            **neutral_metrics,
        }
        metrics["evaluation_elapsed_seconds"] = perf_counter() - evaluation_started
        gate_failures = _exploratory_gate_failures(metrics, self.config)
        metrics["exploratory_gate_passed"] = not gate_failures
        metrics["exploratory_gate_failures"] = gate_failures
        metrics["exploratory_gate_failure_count"] = len(gate_failures)
        numeric_metrics = [value for value in metrics.values() if isinstance(value, int | float)]
        if not all(np.isfinite(value) for value in numeric_metrics):
            raise ValueError("Evaluation produced non-finite metrics")
        return ExploratoryEvaluation(
            metrics=metrics,
            decision="RESEARCH_ONLY_DATA_BLOCKED",
            observations=len(net_return),
            net_returns=net_return,
        )

    def evaluate_portfolio(
        self,
        factors: list[FactorDefinition],
        *,
        weights: list[float] | tuple[float, ...] | None = None,
        benchmark_factors: list[FactorDefinition] | None = None,
        benchmark_weights: list[float] | tuple[float, ...] | None = None,
        bootstrap_samples: int = 500,
    ) -> PortfolioEvaluation:
        """Evaluate alpha diagnostics and the deployable A-share portfolio separately."""
        if not factors:
            raise ValueError("A portfolio requires at least one factor")
        if bootstrap_samples < 0:
            raise ValueError("bootstrap_samples must be non-negative")
        evaluation_started = perf_counter()
        evidence_tier = getattr(self, "research_evidence_tier", research_evidence_tier(self.config))
        normalized_weights = _normalize_weights(factors, weights)
        alpha_all, proposed_all = self._portfolio_paths(factors, normalized_weights)
        evaluation_dates = _walk_forward_dates(proposed_all.index, self.config)
        proposed = proposed_all.loc[evaluation_dates].copy()
        proposed.attrs.update(proposed_all.attrs)
        alpha_proposed = alpha_all.reindex(evaluation_dates).dropna().copy()
        alpha_proposed.attrs.update(alpha_all.attrs)
        benchmark = (
            self._portfolio_paths(
                benchmark_factors,
                _normalize_weights(benchmark_factors, benchmark_weights),
            )[1].reindex(proposed.index)
            if benchmark_factors
            else pd.DataFrame(
                {
                    "net": pd.Series(0.0, index=proposed.index),
                    "stressed": pd.Series(0.0, index=proposed.index),
                    "turnover": pd.Series(0.0, index=proposed.index),
                }
            )
        )
        increment = compare_portfolios(
            benchmark["net"],
            proposed["net"],
            stressed_treatment_net_returns=proposed["stressed"],
            hac_lags=5,
            bootstrap_samples=bootstrap_samples,
            seed=self.config.random_seed,
        )
        annual = annual_robustness(proposed["net"])
        folds = _walk_forward_metrics(proposed, self.config)
        alpha_annual = annual_robustness(alpha_proposed["net"])
        alpha_folds = _walk_forward_metrics(alpha_proposed, self.config)
        inference = hac_mean_inference(proposed["net"].to_numpy(), lags=min(5, len(proposed) - 1))
        dsr = deflated_sharpe_ratio(proposed["net"].to_numpy(), trials=self.trial_count)
        market_benchmark = self._market_benchmark_returns(proposed.index)
        active_metrics = _active_return_metrics(
            proposed,
            market_benchmark,
            self.config,
            prefix="portfolio",
        )
        correlations = self._factor_correlations(factors)
        maximum_correlation = max((abs(value) for value in correlations.values()), default=0.0)
        signal_independence = signal_independence_metrics(
            [factor.factor_id for factor in factors], correlations
        )
        proposed_sharpe = _annualized_ir(proposed["net"])
        proposed_annual_return = float(proposed["net"].mean() * 245)
        proposed_drawdown = _max_drawdown(proposed["net"])
        proposed_cost_ir = _annualized_ir(proposed["stressed"])
        proposed_turnover = float(proposed["turnover"].mean() * 245)
        if benchmark_factors:
            benchmark_sharpe = _annualized_ir(benchmark["net"])
            benchmark_annual_return = float(benchmark["net"].mean() * 245)
            benchmark_drawdown = _max_drawdown(benchmark["net"])
            benchmark_cost_ir = _annualized_ir(benchmark["stressed"])
            benchmark_turnover = float(benchmark["turnover"].mean() * 245)
        else:
            benchmark_sharpe = 0.0
            benchmark_annual_return = 0.0
            benchmark_drawdown = 0.0
            benchmark_cost_ir = 0.0
            benchmark_turnover = 0.0
        metrics = {
            "portfolio_sharpe_ratio": proposed_sharpe,
            "portfolio_simple_annual_return": proposed_annual_return,
            "portfolio_compound_annual_return": _compound_annual_return(proposed["net"]),
            "portfolio_max_drawdown": proposed_drawdown,
            "portfolio_cost_stress_net_ir": proposed_cost_ir,
            "portfolio_annual_turnover": proposed_turnover,
            "portfolio_coverage": float(proposed.attrs["coverage"]),
            "portfolio_capacity_cny": float(proposed.attrs["capacity_cny"]),
            "portfolio_positive_year_ratio": annual.positive_year_ratio,
            "portfolio_worst_year_return": annual.worst_year_return,
            "portfolio_annual_return_dispersion": annual.annual_return_dispersion,
            "portfolio_factor_count": len(factors),
            "portfolio_max_factor_correlation": maximum_correlation,
            **signal_independence,
            "portfolio_incremental_net_ir": increment.incremental_net_ir,
            "portfolio_incremental_annual_return": increment.incremental_annual_return,
            "portfolio_incremental_max_drawdown": increment.incremental_max_drawdown,
            "portfolio_incremental_return_drawdown_efficiency": (
                increment.return_drawdown_efficiency_change
            ),
            "portfolio_incremental_cost_stress_net_ir": float(increment.cost_stress_net_ir or 0.0),
            "portfolio_incremental_bootstrap_confidence_interval": (
                list(increment.bootstrap_confidence_interval)
                if increment.bootstrap_confidence_interval is not None
                else None
            ),
            "portfolio_bootstrap_samples": bootstrap_samples,
            "portfolio_evaluation_stage": (
                "FULL_INFERENCE" if bootstrap_samples > 0 else "VECTOR_SCREEN"
            ),
            "portfolio_sharpe_change": proposed_sharpe - benchmark_sharpe,
            "portfolio_annual_return_change": (proposed_annual_return - benchmark_annual_return),
            "portfolio_max_drawdown_change": proposed_drawdown - benchmark_drawdown,
            "portfolio_cost_stress_net_ir_change": proposed_cost_ir - benchmark_cost_ir,
            "portfolio_annual_turnover_change": proposed_turnover - benchmark_turnover,
            "portfolio_backtest_start": proposed.index.min().date().isoformat(),
            "portfolio_backtest_end": proposed.index.max().date().isoformat(),
            "portfolio_backtest_observations": len(proposed),
            "portfolio_weight_method": "weighted_cross_sectional_zscore",
            "portfolio_factor_weights": {
                factor.factor_id: weight
                for factor, weight in zip(factors, normalized_weights, strict=True)
            },
            "portfolio_evaluation_protocol": self.config.governance.protocol_version,
            "portfolio_research_evidence_tier": evidence_tier,
            "portfolio_task_production_promotion_allowed": (evidence_tier == "PRIMARY_DISCOVERY"),
            "portfolio_holding_period_days": self.config.portfolio.holding_period_days,
            "portfolio_signal_availability": "END_OF_DAY_AFTER_CLOSE",
            "portfolio_execution_lag_sessions": 1,
            "portfolio_return_convention": ASHARE_PROXY_RETURN_CONVENTION,
            "portfolio_strategy_gate_basis": "A_SHARE_LONG_ONLY_WEEKLY_NON_PIT_PROXY",
            "portfolio_strategy_scope": "PUBLIC_VALIDATION_EXECUTION_PROXY",
            "portfolio_mode": "long_only",
            "portfolio_execution_protocol": self.config.strategy_evaluation.engine_protocol,
            "portfolio_execution_data_mode": (self.config.strategy_evaluation.execution_data_mode),
            "portfolio_rebalance_schedule": (self.config.strategy_evaluation.rebalance_schedule),
            "portfolio_target_gross_exposure": (self.config.strategy_evaluation.gross_exposure),
            "portfolio_maximum_positions": (self.config.strategy_evaluation.maximum_positions),
            "portfolio_initial_cash_cny": (self.config.strategy_evaluation.initial_cash_cny),
            "portfolio_average_gross_exposure": float(
                proposed.attrs.get("average_gross_exposure", 0.0)
            ),
            "portfolio_average_positions": float(proposed.attrs.get("average_positions", 0.0)),
            "portfolio_maximum_observed_positions": int(
                proposed.attrs.get("maximum_observed_positions", 0)
            ),
            "portfolio_rebalance_count": int(proposed.attrs.get("rebalance_count", 0)),
            "portfolio_total_transaction_cost_cny": float(
                proposed.attrs.get("total_transaction_cost_cny", 0.0)
            ),
            "portfolio_production_eligible": False,
            "portfolio_production_blockers": [
                "non-PIT historical ST, listing, delisting and suspension state",
                "opening eligibility uses a price-limit proxy",
                "vector weights approximate board lots and cash",
            ],
            "portfolio_walk_forward_folds": folds,
            "portfolio_walk_forward_fold_count": len(folds),
            "portfolio_walk_forward_positive_fraction": float(
                np.mean([fold["annual_return"] > 0 for fold in folds])
            ),
            "portfolio_walk_forward_median_sharpe": float(
                np.median([fold["sharpe"] for fold in folds])
            ),
            "portfolio_walk_forward_worst_sharpe": float(min(fold["sharpe"] for fold in folds)),
            "portfolio_walk_forward_worst_drawdown": float(
                min(fold["max_drawdown"] for fold in folds)
            ),
            "portfolio_net_return_hac_p_value": inference.p_value,
            "portfolio_deflated_sharpe_probability": dsr.probability,
            "portfolio_multiple_testing_trials": self.trial_count,
            **active_metrics,
            "alpha_diagnostic_scope": "NON_INVESTABLE_LONG_SHORT",
            "alpha_diagnostic_sharpe_ratio": _annualized_ir(alpha_proposed["net"]),
            "alpha_diagnostic_simple_annual_return": float(alpha_proposed["net"].mean() * 245),
            "alpha_diagnostic_compound_annual_return": _compound_annual_return(
                alpha_proposed["net"]
            ),
            "alpha_diagnostic_max_drawdown": _max_drawdown(alpha_proposed["net"]),
            "alpha_diagnostic_cost_stress_net_ir": _annualized_ir(alpha_proposed["stressed"]),
            "alpha_diagnostic_annual_turnover": float(alpha_proposed["turnover"].mean() * 245),
            "alpha_diagnostic_positive_year_ratio": alpha_annual.positive_year_ratio,
            "alpha_diagnostic_annual_return_dispersion": (alpha_annual.annual_return_dispersion),
            "alpha_diagnostic_walk_forward_folds": alpha_folds,
            "alpha_diagnostic_walk_forward_worst_sharpe": float(
                min(fold["sharpe"] for fold in alpha_folds)
            ),
            "alpha_diagnostic_return_convention": EOD_NEXT_OPEN_RETURN_CONVENTION,
        }
        metrics["portfolio_evaluation_elapsed_seconds"] = perf_counter() - evaluation_started
        numeric = [value for value in metrics.values() if isinstance(value, int | float)]
        if not all(np.isfinite(value) for value in numeric):
            raise ValueError("Portfolio evaluation produced non-finite metrics")
        return PortfolioEvaluation(metrics, proposed["net"], correlations)

    def _portfolio_paths(
        self,
        factors: list[FactorDefinition],
        weights: tuple[float, ...],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        key = tuple(
            (factor.factor_id, round(weight, 12))
            for factor, weight in zip(factors, weights, strict=True)
        )
        with self._portfolio_path_cache_lock:
            cached = self._portfolio_path_cache.get(key)
            if cached is not None:
                self._portfolio_path_cache.move_to_end(key)
                return cached
        alpha_path = self._alpha_portfolio_path(factors, weights)
        strategy_path = self._strategy_portfolio_path(factors, weights)
        with self._portfolio_path_cache_lock:
            self._portfolio_path_cache[key] = (alpha_path, strategy_path)
            self._portfolio_path_cache.move_to_end(key)
            while len(self._portfolio_path_cache) > 256:
                self._portfolio_path_cache.popitem(last=False)
        return alpha_path, strategy_path

    def _factor_signal(self, factor: FactorDefinition) -> pd.DataFrame:
        local_key = f"{factor.factor_id}:{factor.expected_direction}"
        lock = getattr(self, "_signal_cache_lock", None)
        if lock is None:
            cached = self._signal_cache.get(local_key)
            if cached is None and factor.expected_direction == 1:
                cached = self._signal_cache.get(factor.factor_id)
            if cached is not None:
                return cached
        else:
            with lock:
                cached = self._signal_cache.get(local_key)
                if cached is None and factor.expected_direction == 1:
                    cached = self._signal_cache.get(factor.factor_id)
                if cached is not None:
                    if local_key in self._signal_cache:
                        self._signal_cache.move_to_end(local_key)
                    return cached
        shared_key = (
            _shared_signal_cache_key(self, factor)
            if _SHARED_SIGNAL_CACHE_MAX
            and hasattr(self, "workspace")
            and hasattr(self, "config")
            else None
        )
        if shared_key is not None:
            with _SHARED_SIGNAL_CACHE_LOCK:
                cached = _SHARED_SIGNAL_CACHE.get(shared_key)
                if cached is not None:
                    _SHARED_SIGNAL_CACHE.move_to_end(shared_key)
            if cached is not None:
                if lock is None:
                    self._signal_cache[local_key] = cached
                else:
                    with lock:
                        self._signal_cache[local_key] = cached
                        self._signal_cache.move_to_end(local_key)
                return cached
        fields = self._load_fields(expression_fields(factor.expression))
        raw = self.compiler.evaluate(factor.expression, fields) * factor.expected_direction
        start = pd.Timestamp(self.config.splits.train.start)
        end = pd.Timestamp(self.config.splits.validation.end)
        raw = raw.loc[(raw.index >= start) & (raw.index <= end)]
        standard_deviation = raw.std(axis=1).replace(0, np.nan)
        signal = raw.sub(raw.mean(axis=1), axis=0).div(standard_deviation, axis=0)
        if lock is None:
            self._signal_cache[local_key] = signal
        else:
            with lock:
                self._signal_cache[local_key] = signal
                self._signal_cache.move_to_end(local_key)
                while len(self._signal_cache) > 32:
                    self._signal_cache.popitem(last=False)
        if shared_key is not None:
            with _SHARED_SIGNAL_CACHE_LOCK:
                _SHARED_SIGNAL_CACHE[shared_key] = signal
                _SHARED_SIGNAL_CACHE.move_to_end(shared_key)
                while len(_SHARED_SIGNAL_CACHE) > _SHARED_SIGNAL_CACHE_MAX:
                    _SHARED_SIGNAL_CACHE.popitem(last=False)
        return signal

    def prime_factor_signals(self, factors: list[FactorDefinition]) -> None:
        """Populate mutable signal caches before concurrent portfolio scoring."""
        for factor in factors:
            self._factor_signal(factor)

    def library_signal_correlation(
        self,
        factor: FactorDefinition,
        references: list[FactorDefinition],
        *,
        sample_stride: int = 5,
        max_references: int = 20,
    ) -> dict[str, Any]:
        """Peak median cross-sectional signal correlation against library references.

        Deterministic behavioral-redundancy diagnostic: signals are compared on a
        sampled date grid so up to ``max_references`` library factors stay cheap
        relative to a full evaluation. Reference factors whose stored expressions
        no longer compile are skipped rather than failing the iteration.
        """
        candidate = self._factor_signal(factor).iloc[:: max(int(sample_stride), 1)]
        checked = 0
        peak = 0.0
        peer_id: str | None = None
        peer_name: str | None = None
        for reference in references:
            if checked >= max_references:
                break
            if reference.factor_id == factor.factor_id:
                continue
            try:
                signal = self._factor_signal(reference)
            except Exception:
                continue
            checked += 1
            aligned = signal.reindex(index=candidate.index, columns=candidate.columns)
            daily = candidate.corrwith(aligned, axis=1).dropna()
            if daily.empty:
                continue
            value = float(daily.median())
            if abs(value) > abs(peak):
                peak = value
                peer_id = reference.factor_id
                peer_name = reference.name
        return {
            "library_signal_correlation_max": abs(peak),
            "library_signal_correlation_signed": peak,
            "library_signal_correlation_peer": peer_id,
            "library_signal_correlation_peer_name": peer_name,
            "library_signal_reference_count": checked,
        }

    def _alpha_portfolio_path(
        self,
        factors: list[FactorDefinition],
        weights: list[float] | tuple[float, ...] | None = None,
    ) -> pd.DataFrame:
        normalized_weights = _normalize_weights(factors, weights)
        signals = [self._factor_signal(factor) for factor in factors]
        composite = signals[0] * normalized_weights[0]
        for signal, weight in zip(signals[1:], normalized_weights[1:], strict=True):
            composite = composite + signal * weight
        next_return = next_open_return_for_eod_signal(self._load_fields()["open"]).reindex(
            composite.index
        )
        ranks = composite.rank(axis=1, pct=True)
        positions = (ranks >= 0.9).astype(float) - (ranks <= 0.1).astype(float)
        gross = positions.abs().sum(axis=1).replace(0, np.nan)
        target_weights = positions.div(gross, axis=0).fillna(0.0)
        weights = target_weights.rolling(
            self.config.portfolio.holding_period_days, min_periods=1
        ).mean()
        gross_return = (weights * next_return).sum(axis=1, min_count=1).dropna()
        turnover = weights.diff().abs().sum(axis=1).mul(0.5).reindex(gross_return.index).fillna(0)
        one_way_bps = (
            self.config.costs.commission_bps_each_side
            + self.config.costs.transfer_fee_bps_each_side
            + self.config.costs.stamp_duty_bps_sell / 2
        )
        result = pd.DataFrame(
            {
                "net": gross_return - turnover * one_way_bps / 10_000,
                "stressed": gross_return - turnover * one_way_bps * 2 / 10_000,
                "turnover": turnover,
            }
        ).dropna()
        selected_amount = self._load_fields()["amount"].where(positions.ne(0)).median(axis=1)
        result.attrs["capacity_cny"] = float(
            selected_amount.median() * self.config.costs.max_adv_participation * 20
        )
        result.attrs["coverage"] = self._dynamic_coverage(composite)
        return result

    def _strategy_portfolio_path(
        self,
        factors: list[FactorDefinition],
        weights: list[float] | tuple[float, ...] | None = None,
    ) -> pd.DataFrame:
        strategy = self.config.strategy_evaluation
        if not strategy.enabled:
            raise RuntimeError("A-share strategy evaluation is disabled")
        self.execution_basis.require_capital_ledger_proxy()
        normalized_weights = _normalize_weights(factors, weights)
        signals = [self._factor_signal(factor) for factor in factors]
        composite = signals[0] * normalized_weights[0]
        for signal, weight in zip(signals[1:], normalized_weights[1:], strict=True):
            composite = composite + signal * weight
        fields = self._load_fields()
        missing = [
            name
            for name in ("raw_open", "can_buy_open_proxy", "can_sell_open_proxy")
            if name not in fields
        ]
        if missing:
            raise RuntimeError(f"A-share strategy proxy fields are missing: {missing}")
        result = AshareVectorBacktester(
            AshareVectorConfig(
                initial_cash_cny=strategy.initial_cash_cny,
                gross_exposure=strategy.gross_exposure,
                selection_fraction=strategy.selection_fraction,
                maximum_positions=strategy.maximum_positions,
                rebalance_schedule=strategy.rebalance_schedule,  # type: ignore[arg-type]
                commission_bps_each_side=strategy.commission_bps_each_side,
                stamp_duty_bps_sell=strategy.stamp_duty_bps_sell,
                transfer_fee_bps_each_side=strategy.transfer_fee_bps_each_side,
                minimum_commission_cny=strategy.minimum_commission_cny,
                slippage_bps_each_side=strategy.slippage_bps_each_side,
                use_historical_fee_schedule=strategy.use_historical_fee_schedule,
                cost_stress_multiplier=strategy.cost_stress_multiplier,
            )
        ).run(
            composite,
            fields["open"],
            fields["raw_open"],
            fields["can_buy_open_proxy"],
            fields["can_sell_open_proxy"],
            start=self.config.splits.train.start,
            end=self.config.splits.validation.end,
        )
        path = result.path.copy()
        ranks = composite.rank(axis=1, pct=True)
        ordinal = composite.rank(axis=1, ascending=False, method="first")
        selected = (ranks >= 1.0 - strategy.selection_fraction) & (
            ordinal <= strategy.maximum_positions
        )
        median_adv = float(fields["amount"].where(selected).median(axis=1).median())
        capacity = (
            median_adv
            * strategy.maximum_volume_participation
            * strategy.maximum_positions
            / strategy.gross_exposure
            if math.isfinite(median_adv)
            else 0.0
        )
        path.attrs.update(
            {
                "capacity_cny": capacity,
                "coverage": self._dynamic_coverage(composite),
                "average_gross_exposure": result.metrics["average_gross_exposure"],
                "average_positions": result.metrics["average_positions"],
                "maximum_observed_positions": int(path["position_count"].max()),
                "rebalance_count": result.metrics["rebalance_count"],
                "total_transaction_cost_cny": result.metrics["total_transaction_cost_cny"],
                "bankrupt": result.metrics["bankrupt"],
                "bankruptcy_date": result.metrics["bankruptcy_date"],
            }
        )
        return path

    def _signal_path(self, signal: pd.DataFrame) -> pd.DataFrame:
        next_return = next_open_return_for_eod_signal(self._load_fields()["open"]).reindex(
            signal.index
        )
        ranks = signal.rank(axis=1, pct=True)
        positions = (ranks >= 0.9).astype(float) - (ranks <= 0.1).astype(float)
        gross = positions.abs().sum(axis=1).replace(0, np.nan)
        target_weights = positions.div(gross, axis=0).fillna(0.0)
        weights = target_weights.rolling(
            self.config.portfolio.holding_period_days, min_periods=1
        ).mean()
        gross_return = (weights * next_return).sum(axis=1, min_count=1).dropna()
        turnover = weights.diff().abs().sum(axis=1).mul(0.5).reindex(gross_return.index).fillna(0)
        one_way_bps = (
            self.config.costs.commission_bps_each_side
            + self.config.costs.transfer_fee_bps_each_side
            + self.config.costs.stamp_duty_bps_sell / 2
        )
        result = pd.DataFrame(
            {
                "net": gross_return - turnover * one_way_bps / 10_000,
                "stressed": gross_return - turnover * one_way_bps * 2 / 10_000,
                "turnover": turnover,
            }
        ).dropna()
        selected_amount = self._load_fields()["amount"].where(positions.ne(0)).median(axis=1)
        result.attrs["capacity_cny"] = float(
            selected_amount.median() * self.config.costs.max_adv_participation * 20
        )
        result.attrs["coverage"] = self._dynamic_coverage(signal)
        return result

    def _dynamic_coverage(self, signal: pd.DataFrame) -> float:
        eligible = (
            self._load_fields()["adj_close"]
            .notna()
            .reindex(index=signal.index, columns=signal.columns, fill_value=False)
        )
        denominator = int(eligible.to_numpy().sum())
        if denominator == 0:
            return 0.0
        covered = signal.notna() & eligible
        return float(covered.to_numpy().sum() / denominator)

    def _market_benchmark_returns(self, index: pd.DatetimeIndex) -> pd.Series:
        next_return = next_open_return_for_eod_signal(self._load_fields()["open"])
        benchmark = next_return.mean(axis=1).reindex(index).fillna(0.0)
        return benchmark * self.config.strategy_evaluation.gross_exposure

    def _size_neutral_diagnostics(
        self,
        signal: pd.DataFrame,
        next_return: pd.DataFrame,
        evaluation_dates: pd.DatetimeIndex,
    ) -> dict[str, Any]:
        size_field = (
            "circ_mv"
            if "circ_mv" in self.factor_fields
            else "total_mv"
            if "total_mv" in self.factor_fields
            else None
        )
        if size_field is None:
            return {
                "size_neutral_diagnostic_available": False,
                "size_neutral_diagnostic_reason": "point-in-time capitalization field unavailable",
            }
        fields = self._load_fields({size_field})
        exposure = np.log(fields[size_field].where(fields[size_field] > 0))
        neutral = _cross_sectional_residual(signal, exposure).reindex(evaluation_dates)
        neutral_ic = cross_sectional_ic(
            neutral,
            next_return.reindex(evaluation_dates),
            minimum_names=self.config.minimum_cross_section,
        )
        neutral_path = self._signal_path(neutral).reindex(evaluation_dates).dropna()
        return {
            "size_neutral_diagnostic_available": True,
            "size_neutral_exposure_field": size_field,
            "size_neutral_rank_ic_mean": float(neutral_ic.mean()) if not neutral_ic.empty else 0.0,
            "size_neutral_alpha_sharpe_ratio": _annualized_ir(neutral_path["net"]),
            "industry_neutral_diagnostic_available": False,
            "industry_neutral_diagnostic_reason": "historical industry classification unavailable",
        }

    def _period_summary(self, path: pd.DataFrame, start: Any, end: Any) -> dict[str, Any]:
        selected = path.loc[(path.index >= pd.Timestamp(start)) & (path.index <= pd.Timestamp(end))]
        return {
            "start": selected.index.min().date().isoformat(),
            "end": selected.index.max().date().isoformat(),
            "observations": len(selected),
            "sharpe": _annualized_ir(selected["net"]),
            "simple_annual_return": float(selected["net"].mean() * 245),
            "max_drawdown": _max_drawdown(selected["net"]),
        }

    def _parameter_stability(self, factor: FactorDefinition) -> dict[str, Any]:
        neighborhood: dict[str, float] = {}
        for label, multiplier in (("lower", 0.8), ("base", 1.0), ("upper", 1.2)):
            expression = (
                factor.expression
                if multiplier == 1.0
                else Expression.from_dict(
                    _scale_expression_parameters(factor.expression.to_dict(), multiplier)
                )
            )
            variant = FactorDefinition(
                name=f"{factor.name}_{label}",
                family=factor.family,
                hypothesis=factor.hypothesis,
                expression=expression,
                expected_direction=factor.expected_direction,
            )
            try:
                self.validator.validate(expression)
                if multiplier == 1.0:
                    path = self._portfolio_paths([variant], (1.0,))[1]
                else:
                    # Perturbed variants are throwaway probes: compute the strategy
                    # path directly to skip the unused alpha path and keep them out
                    # of the shared portfolio-path LRU cache.
                    path = self._strategy_portfolio_path([variant], (1.0,))
                dates = _walk_forward_dates(path.index, self.config)
                neighborhood[label] = _annualized_ir(path.loc[dates, "net"])
            except (TypeError, ValueError):
                neighborhood[label] = -100.0
        values = np.asarray(list(neighborhood.values()), dtype=float)
        return {
            "neighborhood": neighborhood,
            "positive_fraction": float((values > 0).mean()),
            "worst_sharpe": float(values.min()),
            "dispersion": float(values.std()),
        }

    def _factor_correlations(self, factors: list[FactorDefinition]) -> dict[str, float]:
        correlations: dict[str, float] = {}
        for left_index, left in enumerate(factors):
            left_signal = self._factor_signal(left)
            for right in factors[left_index + 1 :]:
                daily = left_signal.corrwith(self._factor_signal(right), axis=1).dropna()
                value = float(daily.median()) if not daily.empty else 1.0
                correlations[f"{left.factor_id}:{right.factor_id}"] = value
        return correlations

    def _load_fields(self, required_fields: set[str] | None = None) -> dict[str, pd.DataFrame]:
        required = set(required_fields or set())
        required.update({"open", "adj_close", "amount"})
        if self._fields is not None and required.issubset(self._fields):
            return self._fields
        lock = getattr(self, "_field_load_lock", None)
        if lock is None:
            return self._fields or {}
        with lock:
            if self._fields is not None and required.issubset(self._fields):
                return self._fields
            loaded = self._fields or {}
            missing_factor_fields = sorted((required - set(loaded)) & set(self.factor_fields))
            initial_load = not loaded
            if not missing_factor_fields and not initial_load:
                return loaded
        start = self.config.splits.train.start
        end = self.config.splits.validation.end
        years = range(start.year - 1, end.year + 1)
        paths = [
            path
            for year in years
            for path in sorted((self.panel_path / f"trade_year={year}").glob("*.parquet"))
        ]
        if not paths:
            raise FileNotFoundError(f"No parquet partitions found under {self.data_path}")
        available_columns = set(pq.read_schema(paths[0]).names)
        factor_columns = [name for name in missing_factor_fields if name in available_columns]
        base_columns = list(
            dict.fromkeys(
                [
                    "trade_date",
                    "ts_code",
                    *factor_columns,
                    "is_valid_ohlc",
                    "is_tradable_observation",
                ]
            )
        )
        optional_columns = [
            name
            for name in (
                "raw_open",
                "raw_pre_close",
                "can_buy_open_proxy",
                "can_sell_open_proxy",
            )
            if initial_load and name in available_columns
        ]
        columns = [*base_columns, *optional_columns]
        frames = [pd.read_parquet(path, columns=columns) for path in paths]
        data = pd.concat(frames, ignore_index=True)
        data["trade_date"] = pd.to_datetime(data["trade_date"])
        warmup = pd.Timestamp(start) - pd.Timedelta(days=400)
        data = data[(data["trade_date"] >= warmup) & (data["trade_date"] <= pd.Timestamp(end))]
        valid = data["is_valid_ohlc"].fillna(False) & data["is_tradable_observation"].fillna(False)
        signal_columns = factor_columns
        data.loc[~valid, signal_columns] = np.nan
        fields = {
            name: data.pivot(index="trade_date", columns="ts_code", values=name).sort_index()
            for name in signal_columns
        }
        if initial_load and self.execution_basis.capital_ledger_proxy_ready:
            raw_open_name = "raw_open" if "raw_open" in data else "open"
            raw_open = data[raw_open_name].where(data["is_valid_ohlc"].fillna(False))
            data["_strategy_raw_open"] = raw_open
            if {"can_buy_open_proxy", "can_sell_open_proxy"}.issubset(data.columns):
                data["_strategy_can_buy"] = valid & data["can_buy_open_proxy"].fillna(False)
                data["_strategy_can_sell"] = valid & data["can_sell_open_proxy"].fillna(False)
            elif "raw_pre_close" in data:
                open_move = raw_open.div(data["raw_pre_close"]).sub(1.0)
                threshold = self.config.strategy_evaluation.opening_limit_threshold
                data["_strategy_can_buy"] = valid & open_move.lt(threshold)
                data["_strategy_can_sell"] = valid & open_move.gt(-threshold)
            else:
                data["_strategy_can_buy"] = valid
                data["_strategy_can_sell"] = valid
            for source, target in (
                ("_strategy_raw_open", "raw_open"),
                ("_strategy_can_buy", "can_buy_open_proxy"),
                ("_strategy_can_sell", "can_sell_open_proxy"),
            ):
                fields[target] = data.pivot(
                    index="trade_date", columns="ts_code", values=source
                ).sort_index()
        loaded.update(fields)
        self._fields = loaded
        return self._fields


def _shared_signal_cache_key(
    evaluator: PriceVolumeEvaluator,
    factor: FactorDefinition,
) -> tuple[str, str, str, str, int]:
    return (
        evaluator.workspace.fingerprint,
        evaluator.config.splits.train.start.isoformat(),
        evaluator.config.splits.validation.end.isoformat(),
        factor.factor_id,
        factor.expected_direction,
    )


def _standalone_long_only_metrics(
    path: pd.DataFrame,
    config: ResearchConfig,
    *,
    trials: int,
) -> dict[str, Any]:
    if path.empty:
        raise ValueError("Standalone long-only evaluation produced no observations")
    annual = annual_robustness(path["net"])
    folds = _walk_forward_metrics(path, config)
    inference = hac_mean_inference(path["net"].to_numpy(), lags=min(5, len(path) - 1))
    dsr = deflated_sharpe_ratio(path["net"].to_numpy(), trials=trials)
    return {
        "long_only_sharpe_ratio": _annualized_ir(path["net"]),
        "long_only_simple_annual_return": float(path["net"].mean() * 245),
        "long_only_compound_annual_return": _compound_annual_return(path["net"]),
        "long_only_max_drawdown": _max_drawdown(path["net"]),
        "long_only_cost_stress_net_ir": _annualized_ir(path["stressed"]),
        "long_only_annual_turnover": float(path["turnover"].mean() * 245),
        "long_only_coverage": float(path.attrs["coverage"]),
        "long_only_capacity_cny": float(path.attrs["capacity_cny"]),
        "long_only_positive_year_ratio": annual.positive_year_ratio,
        "long_only_worst_year_return": annual.worst_year_return,
        "long_only_annual_return_dispersion": annual.annual_return_dispersion,
        "long_only_walk_forward_folds": folds,
        "long_only_walk_forward_fold_count": len(folds),
        "long_only_walk_forward_positive_fraction": float(
            np.mean([fold["annual_return"] > 0 for fold in folds])
        ),
        "long_only_walk_forward_median_sharpe": float(
            np.median([fold["sharpe"] for fold in folds])
        ),
        "long_only_walk_forward_worst_sharpe": float(min(fold["sharpe"] for fold in folds)),
        "long_only_walk_forward_worst_drawdown": float(min(fold["max_drawdown"] for fold in folds)),
        "long_only_net_return_hac_p_value": inference.p_value,
        "long_only_deflated_sharpe_probability": dsr.probability,
        "long_only_backtest_start": path.index.min().date().isoformat(),
        "long_only_backtest_end": path.index.max().date().isoformat(),
        "long_only_backtest_observations": len(path),
        "long_only_average_gross_exposure": float(path.attrs.get("average_gross_exposure", 0.0)),
        "long_only_average_positions": float(path.attrs.get("average_positions", 0.0)),
        "long_only_rebalance_count": int(path.attrs.get("rebalance_count", 0)),
        "long_only_bankrupt": bool(path.attrs.get("bankrupt", False)),
        "long_only_bankruptcy_date": str(path.attrs.get("bankruptcy_date", "")),
        "long_only_mode": "long_only",
        "long_only_strategy_gate_basis": "A_SHARE_LONG_ONLY_WEEKLY_NON_PIT_PROXY",
        "long_only_return_convention": ASHARE_PROXY_RETURN_CONVENTION,
    }


def _active_return_metrics(
    path: pd.DataFrame,
    benchmark: pd.Series,
    config: ResearchConfig,
    *,
    prefix: str,
) -> dict[str, Any]:
    aligned = benchmark.reindex(path.index).fillna(0.0)
    active = path["net"] - aligned
    active_frame = pd.DataFrame({"net": active, "turnover": path["turnover"]})
    folds = _walk_forward_metrics(active_frame, config)
    tracking_error = float(active.std(ddof=1) * math.sqrt(245))
    benchmark_variance = float(aligned.var(ddof=1))
    beta = (
        float(path["net"].cov(aligned) / benchmark_variance) if benchmark_variance > 1e-15 else 0.0
    )
    return {
        f"{prefix}_benchmark_mode": "ELIGIBLE_UNIVERSE_EQUAL_WEIGHT_PROXY",
        f"{prefix}_benchmark_simple_annual_return": float(aligned.mean() * 245),
        f"{prefix}_active_information_ratio": _annualized_ir(active),
        f"{prefix}_active_simple_annual_return": float(active.mean() * 245),
        f"{prefix}_active_tracking_error": tracking_error,
        f"{prefix}_market_beta": beta,
        f"{prefix}_active_walk_forward_positive_fraction": float(
            np.mean([fold["annual_return"] > 0 for fold in folds])
        ),
        f"{prefix}_active_walk_forward_worst_sharpe": float(min(fold["sharpe"] for fold in folds)),
    }


def _cross_sectional_residual(signal: pd.DataFrame, exposure: pd.DataFrame) -> pd.DataFrame:
    aligned_exposure = exposure.reindex(index=signal.index, columns=signal.columns)
    valid = signal.notna() & aligned_exposure.notna()
    y = signal.where(valid)
    x = aligned_exposure.where(valid)
    y_centered = y.sub(y.mean(axis=1), axis=0)
    x_centered = x.sub(x.mean(axis=1), axis=0)
    denominator = x_centered.pow(2).sum(axis=1).replace(0.0, np.nan)
    beta = x_centered.mul(y_centered).sum(axis=1).div(denominator)
    return y_centered.sub(x_centered.mul(beta, axis=0)).where(valid)


def _ic_diagnostics(values: pd.Series) -> dict[str, float]:
    clean = values.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {"mean": 0.0, "ir": 0.0, "p_value": 1.0}
    mean = float(clean.mean())
    information_ratio = _annualized_ir(clean)
    if len(clean) < 2:
        return {"mean": mean, "ir": information_ratio, "p_value": 1.0}
    inference = hac_mean_inference(
        clean.to_numpy(),
        lags=min(5, len(clean) - 1),
    )
    return {
        "mean": mean,
        "ir": information_ratio,
        "p_value": _two_sided_p(inference.t_stat),
    }


def _annualized_ir(values: pd.Series) -> float:
    mean = float(values.mean())
    standard_deviation = float(values.std(ddof=1))
    if not math.isfinite(standard_deviation) or standard_deviation <= 1e-15:
        if not math.isfinite(mean) or abs(mean) <= 1e-15:
            return 0.0
        return math.copysign(100.0, mean)
    return float(mean / standard_deviation * math.sqrt(245))


def _normalize_weights(
    factors: list[FactorDefinition],
    weights: list[float] | tuple[float, ...] | None,
) -> tuple[float, ...]:
    if not factors:
        raise ValueError("A portfolio requires at least one factor")
    raw = tuple(weights) if weights is not None else (1.0,) * len(factors)
    if len(raw) != len(factors):
        raise ValueError("Portfolio weights must align with factors")
    if any(not np.isfinite(weight) or weight < 0 for weight in raw):
        raise ValueError("Portfolio weights must be finite and non-negative")
    total = sum(raw)
    if total <= 0:
        raise ValueError("Portfolio weights must contain positive mass")
    return tuple(float(weight / total) for weight in raw)


def _compound_annual_return(values: pd.Series) -> float:
    clean = values.dropna()
    wealth = float((1.0 + clean).prod())
    if wealth <= 0 or clean.empty:
        raise ValueError("Cannot annualize a non-positive wealth path")
    return float(wealth ** (245 / len(clean)) - 1.0)


def _max_drawdown(values: pd.Series) -> float:
    wealth = (1.0 + values.dropna()).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def _two_sided_p(t_stat: float) -> float:
    return float(math.erfc(abs(t_stat) / math.sqrt(2)))


def _walk_forward_dates(index: pd.DatetimeIndex, config: ResearchConfig) -> pd.DatetimeIndex:
    validation = config.splits.validation
    selected = index[
        (index >= pd.Timestamp(validation.start)) & (index <= pd.Timestamp(validation.end))
    ]
    if selected.empty:
        raise ValueError("No dates fall inside the public validation range")
    return selected


def _walk_forward_metrics(path: pd.DataFrame, config: ResearchConfig) -> list[dict[str, Any]]:
    protocol = config.walk_forward
    folds: list[dict[str, Any]] = []
    validation = config.splits.validation
    public = path[
        (path.index >= pd.Timestamp(validation.start))
        & (path.index <= pd.Timestamp(validation.end))
    ]
    for validation_year in range(validation.start.year, validation.end.year + 1):
        selected = public[public.index.year == validation_year]
        if len(selected) < 60:
            continue
        net = selected["net"]
        folds.append(
            {
                "fold_id": len(folds),
                "train_start_year": validation_year - protocol.train_years,
                "train_end_year": validation_year - 1,
                "validation_start": selected.index.min().date().isoformat(),
                "validation_end": selected.index.max().date().isoformat(),
                "observations": len(selected),
                "sharpe": _annualized_ir(net),
                "annual_return": float(net.mean() * 245),
                "compound_return": float((1 + net).prod() - 1),
                "max_drawdown": _max_drawdown(net),
                "annual_turnover": float(selected["turnover"].mean() * 245),
            }
        )
    if len(folds) < protocol.minimum_folds:
        raise ValueError(
            f"Only {len(folds)} walk-forward folds were evaluable; minimum={protocol.minimum_folds}"
        )
    return folds


def _scale_expression_parameters(expression: dict[str, Any], multiplier: float) -> dict[str, Any]:
    parameters = dict(expression.get("parameters", {}))
    for name in ("window", "periods"):
        value = parameters.get(name)
        if isinstance(value, int) and value > 1:
            parameters[name] = max(2, int(round(value * multiplier)))
    return {
        "operator": expression["operator"],
        "parameters": parameters,
        "arguments": [
            _scale_expression_parameters(argument, multiplier)
            for argument in expression.get("arguments", [])
        ],
    }


def _exploratory_gate_failures(metrics: dict[str, Any], config: ResearchConfig) -> list[str]:
    policy = config.evaluation
    checks = {
        "coverage": metrics["long_only_coverage"] >= policy.minimum_coverage,
        "long_only_net_ir": metrics["long_only_sharpe_ratio"] >= policy.minimum_incremental_net_ir,
        "long_only_annual_return": metrics["long_only_simple_annual_return"]
        >= policy.minimum_incremental_annual_return,
        "cost_stress": metrics["long_only_cost_stress_net_ir"] >= policy.minimum_cost_stress_net_ir,
        "positive_year_ratio": metrics["long_only_positive_year_ratio"]
        >= policy.minimum_positive_year_ratio,
        "worst_year": metrics["long_only_worst_year_return"]
        >= policy.minimum_worst_year_incremental_return,
        "annual_dispersion": metrics["long_only_annual_return_dispersion"]
        <= policy.maximum_annual_return_dispersion,
        "turnover": metrics["long_only_annual_turnover"] <= policy.maximum_annual_turnover,
        "capacity": metrics["long_only_capacity_cny"] >= policy.minimum_capacity_cny,
        "walk_forward_fold_count": metrics.get(
            "long_only_walk_forward_fold_count", config.walk_forward.minimum_folds
        )
        >= config.walk_forward.minimum_folds,
        "walk_forward_positive_fraction": metrics.get(
            "long_only_walk_forward_positive_fraction", policy.minimum_positive_fold_fraction
        )
        >= policy.minimum_positive_fold_fraction,
        "walk_forward_worst_sharpe": metrics.get(
            "long_only_walk_forward_worst_sharpe", policy.minimum_worst_fold_net_ir
        )
        >= policy.minimum_worst_fold_net_ir,
        "deflated_sharpe": metrics.get(
            "long_only_deflated_sharpe_probability",
            policy.minimum_deflated_sharpe_probability,
        )
        >= policy.minimum_deflated_sharpe_probability,
        "net_return_significance": metrics.get(
            "long_only_net_return_hac_p_value", policy.maximum_net_return_p_value
        )
        <= policy.maximum_net_return_p_value,
        "parameter_positive_fraction": metrics.get(
            "parameter_stability_positive_fraction", policy.minimum_parameter_positive_fraction
        )
        >= policy.minimum_parameter_positive_fraction,
        "parameter_worst_sharpe": metrics.get(
            "parameter_stability_worst_sharpe", policy.minimum_parameter_worst_sharpe
        )
        >= policy.minimum_parameter_worst_sharpe,
    }
    return [name for name, passed in checks.items() if not passed]
