from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from autoalpha.portfolio.optimizer import PortfolioConstraints, PortfolioOptimizer
from autoalpha.portfolio.products import ProductTemplate
from autoalpha.portfolio.risk import RiskModel, price_volume_risk_exposures


def index_enhancement_diagnostic(
    signal: pd.DataFrame,
    adjusted_close: pd.DataFrame,
    amount: pd.DataFrame,
    template: ProductTemplate,
    *,
    portfolio_value: float,
) -> dict[str, Any]:
    """Optimize the latest investable cross-section against an equal-weight proxy."""
    returns = adjusted_close.pct_change(fill_method=None).tail(260)
    latest_signal = signal.reindex(returns.index).iloc[-1]
    adv = amount.reindex(returns.index).tail(20).mean()
    observations = returns.notna().sum()
    eligible = latest_signal.notna() & adv.gt(0) & observations.ge(100)
    candidates = adv[eligible].nlargest(120).index
    if len(candidates) < 30:
        raise ValueError("Index enhancement diagnostic requires at least 30 liquid assets")
    returns = returns.loc[:, candidates]
    prices = adjusted_close.reindex(index=returns.index, columns=candidates)
    amounts = amount.reindex(index=returns.index, columns=candidates)
    exposures = price_volume_risk_exposures(prices, amounts)
    estimate = RiskModel(shrinkage=0.50).fit(returns, exposures)
    assets = estimate.asset_covariance.index
    alpha = latest_signal.reindex(assets).fillna(0.0)
    alpha_scale = float(alpha.std(ddof=1))
    if alpha_scale:
        alpha = (alpha - alpha.mean()) / alpha_scale
    benchmark = pd.Series(1.0 / len(assets), index=assets)
    constraints = PortfolioConstraints(
        maximum_weight=max(template.maximum_weight, 1.5 / len(assets)),
        maximum_active_weight=template.maximum_active_weight,
        maximum_turnover=template.maximum_turnover,
        maximum_adv_participation=0.05,
        maximum_tracking_error=template.maximum_tracking_error,
        exposure_bounds={
            "beta": (-0.10, 0.10),
            "volatility": (-0.15, 0.15),
            "momentum": (-0.20, 0.20),
            "liquidity": (-0.20, 0.20),
            "size_proxy": (-0.20, 0.20),
        },
    )
    result = PortfolioOptimizer(risk_aversion=8.0, turnover_penalty=0.004).optimize(
        alpha,
        estimate.asset_covariance,
        benchmark,
        benchmark,
        estimate.latest_exposures,
        constraints,
        portfolio_value=portfolio_value,
        adv_usd=adv.reindex(assets),
        tradable=pd.Series(True, index=assets),
    )
    active = result.target_weights - benchmark.reindex(result.target_weights.index)
    tracking_error = estimate.portfolio_volatility(active)
    active_exposures = estimate.latest_exposures.T @ active
    holdings = result.target_weights[result.target_weights > 1e-6].sort_values(ascending=False)
    return {
        "protocol": "PRICE_VOLUME_RISK_MODEL_EQUAL_WEIGHT_PROXY_V1",
        "as_of_date": returns.index[-1].date().isoformat(),
        "benchmark": "ELIGIBLE_UNIVERSE_EQUAL_WEIGHT_PROXY",
        "benchmark_quality": "PROXY_NOT_PRODUCTION",
        "asset_count": len(assets),
        "success": result.success,
        "used_fallback": result.used_fallback,
        "message": result.message,
        "predicted_tracking_error": tracking_error,
        "tracking_error_limit": template.maximum_tracking_error,
        "predicted_active_variance_daily": float(active @ estimate.asset_covariance @ active),
        "turnover": float((result.target_weights - benchmark).abs().sum()),
        "active_exposures": {
            name: float(value) for name, value in active_exposures.items() if math.isfinite(value)
        },
        "constraint_slacks": result.constraint_slacks,
        "objective_attribution": result.objective_attribution,
        "target_holdings": [
            {"symbol": str(symbol), "weight": float(weight)}
            for symbol, weight in holdings.head(template.maximum_positions).items()
        ],
        "risk_model_factors": list(estimate.factor_covariance.columns),
        "specific_risk_median": float(np.sqrt(estimate.specific_variance.median() * 252)),
    }
