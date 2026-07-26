from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from autoalpha.research.statistics import (
    MeanInference,
    hac_mean_inference,
    stationary_block_bootstrap_mean,
)


@dataclass(frozen=True)
class PortfolioIncrement:
    observations: int
    incremental_returns: pd.Series
    incremental_net_ir: float
    incremental_annual_return: float
    incremental_max_drawdown: float
    control_max_drawdown: float
    treatment_max_drawdown: float
    control_annual_return: float
    treatment_annual_return: float
    return_drawdown_efficiency_change: float
    hac: MeanInference
    bootstrap_confidence_interval: tuple[float, float] | None
    cost_stress_net_ir: float | None


@dataclass(frozen=True)
class AnnualRobustness:
    years: int
    annual_returns: dict[int, float]
    positive_year_ratio: float
    worst_year_return: float
    annual_return_dispersion: float
    stability_score: float


def compare_portfolios(
    control_net_returns: pd.Series,
    treatment_net_returns: pd.Series,
    *,
    stressed_treatment_net_returns: pd.Series | None = None,
    periods_per_year: int = 245,
    hac_lags: int = 5,
    bootstrap_block_size: int = 10,
    bootstrap_samples: int = 2000,
    seed: int = 0,
) -> PortfolioIncrement:
    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be non-negative")
    paired = pd.concat(
        [control_net_returns.rename("control"), treatment_net_returns.rename("treatment")],
        axis=1,
        join="inner",
    ).dropna()
    if len(paired) < max(3, hac_lags + 1, bootstrap_block_size):
        raise ValueError("Insufficient paired observations for portfolio comparison")
    incremental = (paired["treatment"] - paired["control"]).rename("incremental_net_return")
    hac = hac_mean_inference(incremental.to_numpy(), lags=hac_lags)
    bootstrap = (
        stationary_block_bootstrap_mean(
            incremental.to_numpy(),
            block_size=bootstrap_block_size,
            samples=bootstrap_samples,
            seed=seed,
        )
        if bootstrap_samples > 0
        else None
    )
    control_drawdown = _maximum_drawdown(paired["control"])
    treatment_drawdown = _maximum_drawdown(paired["treatment"])
    control_annual = _annualized_return(paired["control"], periods_per_year)
    treatment_annual = _annualized_return(paired["treatment"], periods_per_year)
    control_efficiency = control_annual / max(abs(control_drawdown), 0.10)
    treatment_efficiency = treatment_annual / max(abs(treatment_drawdown), 0.10)
    stressed_ir = None
    if stressed_treatment_net_returns is not None:
        stress = pd.concat(
            [paired["control"], stressed_treatment_net_returns.rename("stressed")],
            axis=1,
            join="inner",
        ).dropna()
        stressed_ir = _annualized_ir(stress["stressed"] - stress["control"], periods_per_year)
    return PortfolioIncrement(
        observations=len(paired),
        incremental_returns=incremental,
        incremental_net_ir=_annualized_ir(incremental, periods_per_year),
        incremental_annual_return=float(incremental.mean() * periods_per_year),
        incremental_max_drawdown=treatment_drawdown - control_drawdown,
        control_max_drawdown=control_drawdown,
        treatment_max_drawdown=treatment_drawdown,
        control_annual_return=control_annual,
        treatment_annual_return=treatment_annual,
        return_drawdown_efficiency_change=treatment_efficiency - control_efficiency,
        hac=hac,
        bootstrap_confidence_interval=bootstrap,
        cost_stress_net_ir=stressed_ir,
    )


def annual_robustness(
    returns: pd.Series, *, minimum_observations_per_year: int = 60
) -> AnnualRobustness:
    clean = returns.dropna()
    annual_returns = {}
    for year, values in clean.groupby(clean.index.year):
        if len(values) >= minimum_observations_per_year:
            annual_returns[int(year)] = float((1.0 + values).prod() - 1.0)
    if not annual_returns:
        raise ValueError("No year has enough observations for annual robustness")
    values = np.asarray(list(annual_returns.values()), dtype=float)
    positive_ratio = float((values > 0).mean())
    dispersion = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    consistency = 2.0 * positive_ratio - 1.0
    risk_adjusted = float(values.mean() / (dispersion + 0.10))
    stability = float(np.clip(0.65 * consistency + 0.35 * risk_adjusted, -1.0, 1.0))
    return AnnualRobustness(
        years=len(annual_returns),
        annual_returns=annual_returns,
        positive_year_ratio=positive_ratio,
        worst_year_return=float(values.min()),
        annual_return_dispersion=dispersion,
        stability_score=stability,
    )


def _annualized_ir(returns: pd.Series, periods_per_year: int) -> float:
    standard_deviation = float(returns.std(ddof=1))
    if standard_deviation == 0:
        return float(np.inf * np.sign(returns.mean()))
    return float(returns.mean() / standard_deviation * np.sqrt(periods_per_year))


def _maximum_drawdown(returns: pd.Series) -> float:
    nav = (1.0 + returns).cumprod()
    return float((nav / nav.cummax() - 1.0).min())


def _annualized_return(returns: pd.Series, periods_per_year: int) -> float:
    if returns.empty:
        return float("nan")
    return float((1.0 + returns).prod() ** (periods_per_year / len(returns)) - 1.0)
