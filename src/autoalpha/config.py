from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"Invalid ISO date: {value!r}") from error


@dataclass(frozen=True)
class DateRange:
    start: date
    end: date

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError(f"Date range starts after it ends: {self.start} > {self.end}")

    @classmethod
    def from_dict(cls, value: dict[str, str]) -> DateRange:
        return cls(start=_parse_date(value["start"]), end=_parse_date(value["end"]))


@dataclass(frozen=True)
class SplitConfig:
    train: DateRange
    validation: DateRange
    test: DateRange
    embargo_days: int = 20

    def __post_init__(self) -> None:
        if self.embargo_days < 0:
            raise ValueError("embargo_days must be non-negative")
        if not (self.train.end < self.validation.start <= self.validation.end < self.test.start):
            raise ValueError("train, validation, and test ranges must be ordered and disjoint")


@dataclass(frozen=True)
class CostConfig:
    commission_bps_each_side: float = 1.5
    stamp_duty_bps_sell: float = 5.0
    transfer_fee_bps_each_side: float = 0.1
    minimum_commission_cny: float = 5.0
    max_adv_participation: float = 0.05

    def __post_init__(self) -> None:
        values = asdict(self)
        if any(value < 0 for value in values.values()):
            raise ValueError("Cost parameters must be non-negative")
        if not 0 < self.max_adv_participation <= 1:
            raise ValueError("max_adv_participation must be in (0, 1]")


@dataclass(frozen=True)
class ExperimentBudget:
    max_candidates_per_family: int = 100
    max_candidates_per_generation: int = 500
    maximum_runtime_seconds: int = 2880

    def __post_init__(self) -> None:
        if self.max_candidates_per_family <= 0 or self.max_candidates_per_generation <= 0:
            raise ValueError("Experiment limits must be positive")
        if self.maximum_runtime_seconds <= 0:
            raise ValueError("Runtime limit must be positive")


@dataclass(frozen=True)
class AdaptiveDirectionConfig:
    enabled: bool = True
    maximum_attempts_per_campaign: int = 3
    early_stop_consecutive_misses: int = 2
    recent_candidate_window: int = 24
    minimum_recent_candidates: int = 6
    cooldown_campaigns: int = 2
    minimum_turnover_reduction_fraction: float = 0.05
    minimum_coverage_improvement: float = 0.01
    minimum_correlation_reduction: float = 0.05

    def __post_init__(self) -> None:
        if self.maximum_attempts_per_campaign <= 0:
            raise ValueError("Direction campaign attempt budget must be positive")
        if not 1 <= self.early_stop_consecutive_misses <= self.maximum_attempts_per_campaign:
            raise ValueError("Direction early stop must fit inside the attempt budget")
        if self.recent_candidate_window <= 0 or self.minimum_recent_candidates <= 0:
            raise ValueError("Direction diagnostic windows must be positive")
        if self.minimum_recent_candidates > self.recent_candidate_window:
            raise ValueError("Direction minimum history cannot exceed its recent window")
        if self.cooldown_campaigns < 0:
            raise ValueError("Direction cooldown cannot be negative")
        fractions = (
            self.minimum_turnover_reduction_fraction,
            self.minimum_coverage_improvement,
            self.minimum_correlation_reduction,
        )
        if any(not 0 <= value <= 1 for value in fractions):
            raise ValueError("Direction improvement fractions must be in [0, 1]")


@dataclass(frozen=True)
class WalkForwardConfig:
    train_years: int = 5
    validation_years: int = 1
    first_validation_year: int = 2015
    last_validation_year: int = 2024
    minimum_folds: int = 6

    def __post_init__(self) -> None:
        if self.train_years <= 0 or self.validation_years <= 0 or self.minimum_folds <= 0:
            raise ValueError("Walk-forward sizes and minimum_folds must be positive")
        if self.first_validation_year > self.last_validation_year:
            raise ValueError("Walk-forward validation years are reversed")


@dataclass(frozen=True)
class GovernanceConfig:
    protocol_version: str = "institutional_walkforward_v6_next_open"
    maximum_holdout_evaluations_per_generation: int = 10
    minimum_public_gates_before_holdout: int = 1
    holdout_minimum_sharpe: float = 0.0
    holdout_minimum_annual_return: float = 0.0
    holdout_maximum_drawdown: float = 0.20

    def __post_init__(self) -> None:
        if not self.protocol_version.strip():
            raise ValueError("protocol_version is required")
        if self.maximum_holdout_evaluations_per_generation <= 0:
            raise ValueError("Holdout budget must be positive")
        if self.minimum_public_gates_before_holdout <= 0:
            raise ValueError("Public gate requirement must be positive")
        if self.holdout_maximum_drawdown < 0:
            raise ValueError("Holdout drawdown limit must be non-negative")


@dataclass(frozen=True)
class PortfolioConstructionConfig:
    holding_period_days: int = 5
    initial_cash_cny: float = 1_000_000.0
    target_gross_exposure: float = 0.50
    top_fraction: float = 0.10
    maximum_positions: int = 30
    minimum_capital_sharpe: float = 0.50
    maximum_capital_drawdown: float = 0.25
    maximum_realized_gross_exposure: float = 0.65
    maximum_residual_position_multiplier: float = 1.50

    def __post_init__(self) -> None:
        if self.holding_period_days <= 0 or self.initial_cash_cny <= 0:
            raise ValueError("Holding period and initial cash must be positive")
        if not 0 < self.target_gross_exposure <= 1 or not 0 < self.top_fraction <= 1:
            raise ValueError("Portfolio exposure and top_fraction must be in (0, 1]")
        if self.maximum_positions <= 0 or self.maximum_residual_position_multiplier < 1:
            raise ValueError("Portfolio position limits are invalid")
        if self.maximum_capital_drawdown < 0 or self.maximum_realized_gross_exposure <= 0:
            raise ValueError("Capital risk limits are invalid")


@dataclass(frozen=True)
class StrategyEvaluationConfig:
    enabled: bool = True
    engine_protocol: str = "A_SHARE_LONG_ONLY_WEEKLY_VECTOR_PROXY_V1"
    execution_data_mode: str = "NON_PIT_PROXY"
    initial_cash_cny: float = 1_000_000.0
    gross_exposure: float = 0.90
    selection_fraction: float = 0.10
    maximum_positions: int = 30
    rebalance_schedule: str = "WEEKLY_FIRST_SESSION"
    opening_limit_threshold: float = 0.095
    commission_bps_each_side: float = 2.5
    stamp_duty_bps_sell: float = 5.0
    transfer_fee_bps_each_side: float = 0.1
    minimum_commission_cny: float = 5.0
    slippage_bps_each_side: float = 5.0
    use_historical_fee_schedule: bool = True
    cost_stress_multiplier: float = 3.0
    maximum_volume_participation: float = 0.01

    def __post_init__(self) -> None:
        if not self.engine_protocol.strip() or self.execution_data_mode != "NON_PIT_PROXY":
            raise ValueError("Strategy evaluation protocol is invalid")
        if self.rebalance_schedule not in {
            "WEEKLY_FIRST_SESSION",
            "BIWEEKLY_FIRST_SESSION",
            "MONTHLY_FIRST_SESSION",
        }:
            raise ValueError("Strategy rebalance schedule is invalid")
        if not 0 < self.opening_limit_threshold <= 0.30:
            raise ValueError("Strategy opening limit threshold is invalid")
        if self.initial_cash_cny <= 0 or not 0 < self.gross_exposure <= 1:
            raise ValueError("Strategy capital and exposure are invalid")
        if not 0 < self.selection_fraction <= 0.5 or self.maximum_positions <= 0:
            raise ValueError("Strategy selection settings are invalid")
        if not 0 < self.maximum_volume_participation <= 1:
            raise ValueError("Strategy volume participation must be in (0, 1]")
        costs = (
            self.commission_bps_each_side,
            self.stamp_duty_bps_sell,
            self.transfer_fee_bps_each_side,
            self.minimum_commission_cny,
            self.slippage_bps_each_side,
        )
        if any(value < 0 for value in costs) or self.cost_stress_multiplier < 1:
            raise ValueError("Strategy execution costs are invalid")


@dataclass(frozen=True)
class EvaluationConfig:
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
    minimum_capacity_cny: float = 10_000_000.0
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

    def __post_init__(self) -> None:
        unit_interval = {
            "minimum_coverage": self.minimum_coverage,
            "maximum_net_return_p_value": self.maximum_net_return_p_value,
            "minimum_deflated_sharpe_probability": self.minimum_deflated_sharpe_probability,
            "maximum_probability_backtest_overfitting": (
                self.maximum_probability_backtest_overfitting
            ),
            "minimum_positive_fold_fraction": self.minimum_positive_fold_fraction,
            "minimum_positive_year_ratio": self.minimum_positive_year_ratio,
            "maximum_untradeable_fraction": self.maximum_untradeable_fraction,
            "maximum_library_correlation": self.maximum_library_correlation,
            "minimum_parameter_positive_fraction": self.minimum_parameter_positive_fraction,
        }
        invalid = [name for name, value in unit_interval.items() if not 0 <= value <= 1]
        if invalid:
            raise ValueError(f"Evaluation probabilities/fractions are invalid: {invalid}")
        if any(
            value < 0
            for value in (
                self.maximum_incremental_drawdown_deterioration,
                self.maximum_annual_return_dispersion,
                self.maximum_library_correlation,
                self.maximum_style_exposure,
                self.maximum_industry_active_weight,
                self.maximum_stress_loss,
                self.maximum_annual_turnover,
                self.minimum_capacity_cny,
                self.minimum_break_even_cost_multiplier,
                self.minimum_paper_days,
                self.minimum_diversification_sharpe_improvement,
                self.minimum_diversification_drawdown_improvement,
                self.minimum_stability_dispersion_reduction,
                self.minimum_stability_worst_fold_sharpe_improvement,
                self.maximum_transition_annual_return_sacrifice,
            )
        ):
            raise ValueError("Evaluation limits must be non-negative")


@dataclass(frozen=True)
class ResearchConfig:
    name: str
    generation: str
    splits: SplitConfig
    costs: CostConfig = field(default_factory=CostConfig)
    budget: ExperimentBudget = field(default_factory=ExperimentBudget)
    adaptive_direction: AdaptiveDirectionConfig = field(default_factory=AdaptiveDirectionConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    governance: GovernanceConfig = field(default_factory=GovernanceConfig)
    portfolio: PortfolioConstructionConfig = field(default_factory=PortfolioConstructionConfig)
    strategy_evaluation: StrategyEvaluationConfig = field(
        default_factory=StrategyEvaluationConfig
    )
    horizons: tuple[int, ...] = (1, 3, 5, 10, 20)
    minimum_cross_section: int = 30
    random_seed: int = 20260715

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.generation.strip():
            raise ValueError("name and generation are required")
        if not self.horizons or any(value <= 0 for value in self.horizons):
            raise ValueError("horizons must contain positive integers")
        if len(set(self.horizons)) != len(self.horizons):
            raise ValueError("horizons must be unique")
        if self.minimum_cross_section < 2:
            raise ValueError("minimum_cross_section must be at least 2")

    @classmethod
    def from_toml(cls, path: Path) -> ResearchConfig:
        with path.open("rb") as handle:
            raw = tomllib.load(handle)
        split_raw = raw["splits"]
        return cls(
            name=raw["research"]["name"],
            generation=raw["research"]["generation"],
            splits=SplitConfig(
                train=DateRange.from_dict(split_raw["train"]),
                validation=DateRange.from_dict(split_raw["validation"]),
                test=DateRange.from_dict(split_raw["test"]),
                embargo_days=int(split_raw.get("embargo_days", 20)),
            ),
            costs=CostConfig(**raw.get("costs", {})),
            budget=ExperimentBudget(**raw.get("budget", {})),
            adaptive_direction=AdaptiveDirectionConfig(**raw.get("adaptive_direction", {})),
            evaluation=EvaluationConfig(**raw.get("evaluation", {})),
            walk_forward=WalkForwardConfig(**raw.get("walk_forward", {})),
            governance=GovernanceConfig(**raw.get("governance", {})),
            portfolio=PortfolioConstructionConfig(**raw.get("portfolio", {})),
            strategy_evaluation=StrategyEvaluationConfig(
                **raw.get("strategy_evaluation", {})
            ),
            horizons=tuple(raw["research"].get("horizons", (1, 3, 5, 10, 20))),
            minimum_cross_section=int(raw["research"].get("minimum_cross_section", 30)),
            random_seed=int(raw["research"].get("random_seed", 20260715)),
        )

    def canonical_dict(self) -> dict[str, Any]:
        return _json_ready(asdict(self))

    def fingerprint(self, *, data_checksums: dict[str, str] | None = None) -> str:
        payload = {
            "research_config": self.canonical_dict(),
            "data_checksums": dict(sorted((data_checksums or {}).items())),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    return value
