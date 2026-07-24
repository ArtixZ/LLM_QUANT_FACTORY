from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass(frozen=True)
class PortfolioConstraints:
    long_only: bool = True
    fully_invested: bool = True
    maximum_weight: float = 0.05
    maximum_active_weight: float = 0.03
    maximum_turnover: float = 0.30
    exposure_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    maximum_adv_participation: float = 0.10
    maximum_tracking_error: float | None = None
    risk_annualization: int = 252


@dataclass(frozen=True)
class OptimizationResult:
    target_weights: pd.Series
    success: bool
    used_fallback: bool
    message: str
    objective_attribution: dict[str, float]
    constraint_slacks: dict[str, float]


class PortfolioOptimizer:
    def __init__(
        self,
        *,
        risk_aversion: float = 5.0,
        turnover_penalty: float = 0.002,
        concentration_penalty: float = 0.01,
    ) -> None:
        self.risk_aversion = risk_aversion
        self.turnover_penalty = turnover_penalty
        self.concentration_penalty = concentration_penalty

    def optimize(
        self,
        alpha: pd.Series,
        covariance: pd.DataFrame,
        current_weights: pd.Series,
        benchmark_weights: pd.Series,
        exposures: pd.DataFrame,
        constraints: PortfolioConstraints,
        *,
        portfolio_value: float = 1.0,
        adv_cny: pd.Series | None = None,
        tradable: pd.Series | None = None,
    ) -> OptimizationResult:
        assets = alpha.index
        required_indexes = (
            covariance.index,
            covariance.columns,
            current_weights.index,
            benchmark_weights.index,
        )
        for item in required_indexes:
            assets = assets.intersection(item)
        if assets.empty:
            raise ValueError("No common assets for portfolio optimization")
        alpha_vector = alpha.loc[assets].fillna(0.0).to_numpy(dtype=float)
        covariance_matrix = covariance.loc[assets, assets].fillna(0.0).to_numpy(dtype=float)
        current = current_weights.reindex(assets).fillna(0.0).to_numpy(dtype=float)
        benchmark = benchmark_weights.reindex(assets).fillna(0.0).to_numpy(dtype=float)
        exposure_matrix = exposures.reindex(assets).fillna(0.0)

        lower = (
            np.zeros(len(assets))
            if constraints.long_only
            else np.full(len(assets), -constraints.maximum_weight)
        )
        upper = np.minimum(
            constraints.maximum_weight,
            benchmark + constraints.maximum_active_weight,
        )
        if not constraints.long_only:
            lower = np.maximum(lower, benchmark - constraints.maximum_active_weight)
        if adv_cny is not None:
            trade_limit = (
                adv_cny.reindex(assets).fillna(0.0).to_numpy()
                * constraints.maximum_adv_participation
                / portfolio_value
            )
            lower = np.maximum(lower, current - trade_limit)
            upper = np.minimum(upper, current + trade_limit)
        if tradable is not None:
            frozen = ~tradable.reindex(assets).fillna(False).to_numpy(dtype=bool)
            lower[frozen] = current[frozen]
            upper[frozen] = current[frozen]
        if np.any(lower > upper):
            return self._fallback(
                assets, current, benchmark, alpha_vector, covariance_matrix, "inconsistent bounds"
            )

        asset_count = len(assets)

        def components(
            weights: np.ndarray, turnover: float | None = None
        ) -> tuple[float, float, float, float]:
            active = weights - benchmark
            expected_alpha = float(alpha_vector @ weights)
            risk = float(active @ covariance_matrix @ active)
            if turnover is None:
                turnover = float(np.abs(weights - current).sum())
            concentration = float(weights @ weights)
            return expected_alpha, risk, turnover, concentration

        def objective(decision: np.ndarray) -> float:
            weights = decision[:asset_count]
            turnover = float(decision[asset_count:].sum())
            expected_alpha, risk, turnover, concentration = components(weights, turnover)
            return (
                -expected_alpha
                + self.risk_aversion * risk
                + self.turnover_penalty * turnover
                + self.concentration_penalty * concentration
            )

        scipy_constraints: list[dict[str, object]] = []
        if constraints.fully_invested:
            scipy_constraints.append(
                {
                    "type": "eq",
                    "fun": lambda decision: decision[:asset_count].sum() - 1.0,
                }
            )
        scipy_constraints.append(
            {
                "type": "ineq",
                "fun": lambda decision: constraints.maximum_turnover - decision[asset_count:].sum(),
            }
        )
        if constraints.maximum_tracking_error is not None:
            maximum_daily_variance = (
                constraints.maximum_tracking_error**2 / constraints.risk_annualization
            )
            scipy_constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda decision, limit=maximum_daily_variance: (
                        limit
                        - float(
                            (decision[:asset_count] - benchmark)
                            @ covariance_matrix
                            @ (decision[:asset_count] - benchmark)
                        )
                    ),
                }
            )
        scipy_constraints.append(
            {
                "type": "ineq",
                "fun": lambda decision: decision[asset_count:] - decision[:asset_count] + current,
            }
        )
        scipy_constraints.append(
            {
                "type": "ineq",
                "fun": lambda decision: decision[asset_count:] + decision[:asset_count] - current,
            }
        )
        for name, (minimum, maximum) in constraints.exposure_bounds.items():
            if name not in exposure_matrix.columns:
                raise ValueError(f"Missing portfolio exposure: {name}")
            vector = exposure_matrix[name].to_numpy(dtype=float)
            benchmark_exposure = float(vector @ benchmark)
            scipy_constraints.extend(
                [
                    {
                        "type": "ineq",
                        "fun": lambda decision, v=vector, b=benchmark_exposure, m=minimum: (
                            float(v @ decision[:asset_count]) - b - m
                        ),
                    },
                    {
                        "type": "ineq",
                        "fun": lambda decision, v=vector, b=benchmark_exposure, m=maximum: (
                            b + m - float(v @ decision[:asset_count])
                        ),
                    },
                ]
            )

        initial = np.clip(current if current.sum() else benchmark, lower, upper)
        if constraints.fully_invested and initial.sum() > 0:
            initial = initial / initial.sum()
        initial_decision = np.concatenate([initial, np.abs(initial - current)])
        bounds = [*zip(lower, upper, strict=True), *([(0.0, None)] * asset_count)]
        solved = minimize(
            objective,
            initial_decision,
            method="SLSQP",
            bounds=bounds,
            constraints=scipy_constraints,
            options={"maxiter": 500, "ftol": 1e-10},
        )
        if not solved.success:
            return self._fallback(
                assets, current, benchmark, alpha_vector, covariance_matrix, solved.message
            )
        weights = solved.x[:asset_count]
        turnover = float(solved.x[asset_count:].sum())
        expected_alpha, risk, turnover, concentration = components(weights, turnover)
        slacks = {
            "turnover": constraints.maximum_turnover - turnover,
            "minimum_weight": float((weights - lower).min()),
            "maximum_weight": float((upper - weights).min()),
        }
        if constraints.maximum_tracking_error is not None:
            tracking_error = math.sqrt(max(0.0, risk) * constraints.risk_annualization)
            slacks["tracking_error"] = constraints.maximum_tracking_error - tracking_error
        for name, (minimum, maximum) in constraints.exposure_bounds.items():
            vector = exposure_matrix[name].to_numpy(dtype=float)
            active_exposure = float(vector @ (weights - benchmark))
            slacks[f"exposure:{name}:lower"] = active_exposure - minimum
            slacks[f"exposure:{name}:upper"] = maximum - active_exposure
        return OptimizationResult(
            target_weights=pd.Series(weights, index=assets),
            success=True,
            used_fallback=False,
            message=str(solved.message),
            objective_attribution={
                "expected_alpha": expected_alpha,
                "active_risk_penalty": -self.risk_aversion * risk,
                "turnover_penalty": -self.turnover_penalty * turnover,
                "concentration_penalty": -self.concentration_penalty * concentration,
            },
            constraint_slacks=slacks,
        )

    def _fallback(
        self,
        assets: pd.Index,
        current: np.ndarray,
        benchmark: np.ndarray,
        alpha: np.ndarray,
        covariance: np.ndarray,
        reason: str,
    ) -> OptimizationResult:
        weights = current.copy()
        if weights.sum() <= 0 and benchmark.sum() > 0:
            weights = benchmark / benchmark.sum()
        active = weights - benchmark
        return OptimizationResult(
            target_weights=pd.Series(weights, index=assets),
            success=False,
            used_fallback=True,
            message=f"Deterministic no-trade fallback: {reason}",
            objective_attribution={
                "expected_alpha": float(alpha @ weights),
                "active_risk_penalty": -self.risk_aversion * float(active @ covariance @ active),
                "turnover_penalty": 0.0,
                "concentration_penalty": -self.concentration_penalty * float(weights @ weights),
            },
            constraint_slacks={},
        )
