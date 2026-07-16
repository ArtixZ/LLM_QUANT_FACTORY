from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from autoalpha.backtest.timing import (
    EOD_NEXT_OPEN_RETURN_CONVENTION,
    next_open_return_for_eod_signal,
)
from autoalpha.config import ResearchConfig
from autoalpha.data.execution_basis import (
    expression_research_basis_blockers,
    inspect_execution_data_basis,
)
from autoalpha.data.workspace import DataWorkspaceReport, inspect_data_workspace
from autoalpha.dsl.compiler import FactorCompiler
from autoalpha.dsl.expression import Expression, FactorDefinition
from autoalpha.dsl.semantics import FieldDefinition, SemanticValidator
from autoalpha.research.evaluation import cross_sectional_ic
from autoalpha.research.incremental import annual_robustness, compare_portfolios
from autoalpha.research.multiple_testing import deflated_sharpe_ratio
from autoalpha.research.statistics import hac_mean_inference


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
        basis = inspect_execution_data_basis(self.panel_path)
        self.execution_basis = basis
        fields = [
            FieldDefinition("open", "price"),
            FieldDefinition("close", "price"),
            FieldDefinition("adj_close", "price"),
            FieldDefinition("amount", basis.amount_unit),
            FieldDefinition("vol", basis.volume_unit),
        ]
        self.validator = SemanticValidator(fields, maximum_nodes=30, maximum_lookback=252)
        self.compiler = FactorCompiler(self.validator)
        self._fields: dict[str, pd.DataFrame] | None = None
        self._signal_cache: dict[str, pd.DataFrame] = {}
        self.trial_count = 1

    def set_trial_count(self, value: int) -> None:
        self.trial_count = max(1, int(value))

    def evaluate(self, factor: FactorDefinition) -> ExploratoryEvaluation:
        fields = self._load_fields()
        self.validator.validate(factor.expression)
        signal = self.compiler.evaluate(factor.expression, fields) * factor.expected_direction
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
        if len(rank_ic) < 60:
            raise ValueError("Insufficient valid dates for exploratory evaluation")
        rank_ic_hac = hac_mean_inference(rank_ic.to_numpy(), lags=5)
        pearson_ic_hac = hac_mean_inference(pearson_ic.to_numpy(), lags=5)

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
        parameter_stability = self._parameter_stability(factor)
        exploration = self._period_summary(
            self._signal_path(self._factor_signal(factor)),
            self.config.splits.train.start,
            self.config.splits.train.end,
        )
        coverage = self._dynamic_coverage(signal)
        data_basis_blockers = expression_research_basis_blockers(
            factor.expression.to_dict(), self.execution_basis
        )
        metrics = {
            "rank_ic_mean": float(rank_ic.mean()),
            "rank_ic_ir": _annualized_ir(rank_ic),
            "rank_ic_hac_p_value": _two_sided_p(rank_ic_hac.t_stat),
            "pearson_ic_mean": float(pearson_ic.mean()),
            "pearson_ic_ir": _annualized_ir(pearson_ic),
            "pearson_ic_hac_p_value": _two_sided_p(pearson_ic_hac.t_stat),
            "incremental_net_ir": increment.incremental_net_ir,
            "incremental_annual_return": increment.incremental_annual_return,
            "simple_annual_return": simple_annual_return,
            "compound_annual_return": compound_annual_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": _max_drawdown(net_return),
            "incremental_max_drawdown": increment.incremental_max_drawdown,
            "return_drawdown_efficiency_change": increment.return_drawdown_efficiency_change,
            "cost_stress_net_ir": float(increment.cost_stress_net_ir or 0.0),
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
            "holding_period_days": self.config.portfolio.holding_period_days,
            "signal_availability": "END_OF_DAY_AFTER_CLOSE",
            "execution_lag_sessions": 1,
            "return_convention": EOD_NEXT_OPEN_RETURN_CONVENTION,
            "data_basis_compatible": not data_basis_blockers,
            "data_basis_blockers": list(data_basis_blockers),
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
        }
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
    ) -> PortfolioEvaluation:
        """Evaluate a weighted factor composite against the currently active composite."""
        if not factors:
            raise ValueError("A portfolio requires at least one factor")
        normalized_weights = _normalize_weights(factors, weights)
        proposed_all = self._portfolio_path(factors, normalized_weights)
        evaluation_dates = _walk_forward_dates(proposed_all.index, self.config)
        proposed = proposed_all.loc[evaluation_dates].copy()
        proposed.attrs.update(proposed_all.attrs)
        benchmark = (
            self._portfolio_path(
                benchmark_factors,
                _normalize_weights(benchmark_factors, benchmark_weights),
            ).reindex(proposed.index)
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
            bootstrap_samples=500,
            seed=self.config.random_seed,
        )
        annual = annual_robustness(proposed["net"])
        folds = _walk_forward_metrics(proposed, self.config)
        inference = hac_mean_inference(proposed["net"].to_numpy(), lags=min(5, len(proposed) - 1))
        dsr = deflated_sharpe_ratio(proposed["net"].to_numpy(), trials=self.trial_count)
        correlations = self._factor_correlations(factors)
        maximum_correlation = max((abs(value) for value in correlations.values()), default=0.0)
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
            "portfolio_incremental_net_ir": increment.incremental_net_ir,
            "portfolio_incremental_annual_return": increment.incremental_annual_return,
            "portfolio_incremental_max_drawdown": increment.incremental_max_drawdown,
            "portfolio_incremental_return_drawdown_efficiency": (
                increment.return_drawdown_efficiency_change
            ),
            "portfolio_incremental_cost_stress_net_ir": float(increment.cost_stress_net_ir or 0.0),
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
            "portfolio_holding_period_days": self.config.portfolio.holding_period_days,
            "portfolio_signal_availability": "END_OF_DAY_AFTER_CLOSE",
            "portfolio_execution_lag_sessions": 1,
            "portfolio_return_convention": EOD_NEXT_OPEN_RETURN_CONVENTION,
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
        }
        numeric = [value for value in metrics.values() if isinstance(value, int | float)]
        if not all(np.isfinite(value) for value in numeric):
            raise ValueError("Portfolio evaluation produced non-finite metrics")
        return PortfolioEvaluation(metrics, proposed["net"], correlations)

    def _factor_signal(self, factor: FactorDefinition) -> pd.DataFrame:
        cached = self._signal_cache.get(factor.factor_id)
        if cached is not None:
            return cached
        fields = self._load_fields()
        raw = self.compiler.evaluate(factor.expression, fields) * factor.expected_direction
        start = pd.Timestamp(self.config.splits.train.start)
        end = pd.Timestamp(self.config.splits.validation.end)
        raw = raw.loc[(raw.index >= start) & (raw.index <= end)]
        standard_deviation = raw.std(axis=1).replace(0, np.nan)
        signal = raw.sub(raw.mean(axis=1), axis=0).div(standard_deviation, axis=0)
        self._signal_cache[factor.factor_id] = signal
        return signal

    def prime_factor_signals(self, factors: list[FactorDefinition]) -> None:
        """Populate mutable signal caches before concurrent portfolio scoring."""
        for factor in factors:
            self._factor_signal(factor)

    def _portfolio_path(
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
        selected_amount = self._load_fields()["amount"].where(positions.ne(0)).median(axis=1) * 1000
        result.attrs["capacity_cny"] = float(
            selected_amount.median() * self.config.costs.max_adv_participation * 20
        )
        result.attrs["coverage"] = self._dynamic_coverage(composite)
        return result

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
        selected_amount = self._load_fields()["amount"].where(positions.ne(0)).median(axis=1) * 1000
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
                raw = (
                    self.compiler.evaluate(expression, self._load_fields())
                    * variant.expected_direction
                )
                path = self._signal_path(raw)
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

    def _load_fields(self) -> dict[str, pd.DataFrame]:
        if self._fields is not None:
            return self._fields
        start = self.config.splits.train.start
        end = self.config.splits.validation.end
        years = range(start.year - 1, end.year + 1)
        frames = []
        columns = [
            "trade_date",
            "ts_code",
            "open",
            "close",
            "adj_close",
            "amount",
            "vol",
            "is_valid_ohlc",
            "is_tradable_observation",
        ]
        for year in years:
            for path in sorted((self.panel_path / f"trade_year={year}").glob("*.parquet")):
                frames.append(pd.read_parquet(path, columns=columns))
        if not frames:
            raise FileNotFoundError(f"No parquet partitions found under {self.data_path}")
        data = pd.concat(frames, ignore_index=True)
        data["trade_date"] = pd.to_datetime(data["trade_date"])
        warmup = pd.Timestamp(start) - pd.Timedelta(days=400)
        data = data[(data["trade_date"] >= warmup) & (data["trade_date"] <= pd.Timestamp(end))]
        valid = data["is_valid_ohlc"].fillna(False) & data["is_tradable_observation"].fillna(False)
        data.loc[~valid, ["open", "close", "adj_close", "amount", "vol"]] = np.nan
        fields = {
            name: data.pivot(index="trade_date", columns="ts_code", values=name).sort_index()
            for name in ("open", "close", "adj_close", "amount", "vol")
        }
        self._fields = fields
        return self._fields


def _annualized_ir(values: pd.Series) -> float:
    standard_deviation = float(values.std(ddof=1))
    return float(values.mean() / standard_deviation * math.sqrt(245))


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
        "coverage": metrics["coverage"] >= policy.minimum_coverage,
        "incremental_net_ir": metrics["incremental_net_ir"] >= policy.minimum_incremental_net_ir,
        "incremental_annual_return": metrics["incremental_annual_return"]
        >= policy.minimum_incremental_annual_return,
        "cost_stress": metrics["cost_stress_net_ir"] >= policy.minimum_cost_stress_net_ir,
        "positive_year_ratio": metrics["positive_year_ratio"] >= policy.minimum_positive_year_ratio,
        "worst_year": metrics["worst_year_incremental_return"]
        >= policy.minimum_worst_year_incremental_return,
        "annual_dispersion": metrics["annual_return_dispersion"]
        <= policy.maximum_annual_return_dispersion,
        "turnover": metrics["annual_turnover"] <= policy.maximum_annual_turnover,
        "capacity": metrics["capacity_cny"] >= policy.minimum_capacity_cny,
        "walk_forward_fold_count": metrics.get(
            "walk_forward_fold_count", config.walk_forward.minimum_folds
        )
        >= config.walk_forward.minimum_folds,
        "walk_forward_positive_fraction": metrics.get(
            "walk_forward_positive_fraction", policy.minimum_positive_fold_fraction
        )
        >= policy.minimum_positive_fold_fraction,
        "walk_forward_worst_sharpe": metrics.get(
            "walk_forward_worst_sharpe", policy.minimum_worst_fold_net_ir
        )
        >= policy.minimum_worst_fold_net_ir,
        "deflated_sharpe": metrics.get(
            "deflated_sharpe_probability", policy.minimum_deflated_sharpe_probability
        )
        >= policy.minimum_deflated_sharpe_probability,
        "net_return_significance": metrics.get(
            "net_return_hac_p_value", policy.maximum_net_return_p_value
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
