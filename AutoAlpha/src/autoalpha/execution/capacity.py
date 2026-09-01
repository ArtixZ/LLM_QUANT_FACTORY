from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from autoalpha.execution.simulator import MarketImpactModel


@dataclass(frozen=True)
class CapacityPoint:
    capital_usd: float
    average_participation: float
    annual_explicit_cost: float
    annual_impact_cost: float
    net_return: float
    net_ir: float


@dataclass(frozen=True)
class CapacityReport:
    points: tuple[CapacityPoint, ...]
    recommended_capacity_usd: float
    binding_reason: str


class CapacityAnalyzer:
    def __init__(self, impact: MarketImpactModel | None = None) -> None:
        self.impact = impact or MarketImpactModel()

    def analyze(
        self,
        gross_daily_returns: pd.Series,
        *,
        annual_turnover: float,
        aggregate_adv_usd: float,
        daily_volatility: float,
        capital_grid_usd: tuple[float, ...],
        explicit_cost_bps: float = 8.0,
        maximum_participation: float = 0.10,
        minimum_net_ir: float = 0.0,
    ) -> CapacityReport:
        if aggregate_adv_usd <= 0 or not capital_grid_usd:
            raise ValueError("Positive ADV and a non-empty capital grid are required")
        gross_return = float(gross_daily_returns.mean() * 252)
        annual_volatility = float(gross_daily_returns.std(ddof=1) * np.sqrt(252))
        points: list[CapacityPoint] = []
        recommended = 0.0
        reason = "minimum net IR"
        for capital in sorted(capital_grid_usd):
            daily_traded = capital * annual_turnover / 252
            participation = daily_traded / aggregate_adv_usd
            explicit = annual_turnover * explicit_cost_bps / 10_000
            impact_bps = self.impact.impact_bps(1, 1 / max(participation, 1e-12), daily_volatility)
            impact_cost = annual_turnover * impact_bps / 10_000
            net_return = gross_return - explicit - impact_cost
            net_ir = net_return / annual_volatility if annual_volatility > 0 else float("nan")
            points.append(
                CapacityPoint(
                    capital,
                    participation,
                    explicit,
                    impact_cost,
                    net_return,
                    net_ir,
                )
            )
            if participation <= maximum_participation and net_ir >= minimum_net_ir:
                recommended = capital
                reason = "largest grid point satisfying participation and net IR"
        return CapacityReport(tuple(points), recommended, reason)
