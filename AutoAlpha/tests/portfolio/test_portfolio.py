import numpy as np
import pandas as pd

from autoalpha.portfolio.optimizer import PortfolioConstraints, PortfolioOptimizer
from autoalpha.portfolio.products import product_template, product_template_catalog
from autoalpha.portfolio.risk import RiskModel, return_attribution, stress_portfolio


def test_risk_model_produces_symmetric_positive_covariance() -> None:
    rng = np.random.default_rng(7)
    dates = pd.date_range("2023-01-01", periods=40)
    assets = pd.Index(["A", "B", "C", "D", "E"])
    size = pd.DataFrame(rng.normal(size=(40, 5)), index=dates, columns=assets)
    beta = pd.DataFrame(rng.normal(size=(40, 5)), index=dates, columns=assets)
    returns = (
        0.01 * size
        + 0.02 * beta
        + pd.DataFrame(rng.normal(scale=0.01, size=(40, 5)), index=dates, columns=assets)
    )
    estimate = RiskModel(shrinkage=0.5).fit(returns, {"size": size, "beta": beta})
    assert np.allclose(estimate.asset_covariance, estimate.asset_covariance.T)
    assert np.linalg.eigvalsh(estimate.asset_covariance).min() >= -1e-12
    assert estimate.portfolio_volatility(pd.Series(0.2, index=assets)) > 0


def test_optimizer_respects_weight_turnover_exposure_and_adv() -> None:
    assets = pd.Index(["A", "B", "C", "D"])
    alpha = pd.Series([0.04, 0.03, -0.01, -0.02], index=assets)
    covariance = pd.DataFrame(np.eye(4) * 0.02, index=assets, columns=assets)
    current = pd.Series(0.25, index=assets)
    benchmark = pd.Series(0.25, index=assets)
    exposures = pd.DataFrame({"beta": [0.8, 1.0, 1.1, 1.2]}, index=assets)
    constraints = PortfolioConstraints(
        maximum_weight=0.40,
        maximum_active_weight=0.15,
        maximum_turnover=0.20,
        exposure_bounds={"beta": (-0.03, 0.03)},
        maximum_adv_participation=0.10,
    )
    result = PortfolioOptimizer(risk_aversion=1.0).optimize(
        alpha,
        covariance,
        current,
        benchmark,
        exposures,
        constraints,
        portfolio_value=1_000_000,
        adv_usd=pd.Series(500_000, index=assets),
    )
    assert result.success
    assert abs(result.target_weights.sum() - 1) < 1e-8
    assert abs(result.target_weights - current).sum() <= 0.20 + 1e-8
    active_beta = float(exposures["beta"] @ (result.target_weights - benchmark))
    assert abs(active_beta) <= 0.03 + 1e-8
    assert (abs(result.target_weights - current) <= 0.05 + 1e-8).all()


def test_optimizer_enforces_annualized_tracking_error() -> None:
    assets = pd.Index(["A", "B", "C", "D"])
    covariance = pd.DataFrame(np.eye(4) * 0.0004, index=assets, columns=assets)
    benchmark = pd.Series(0.25, index=assets)
    result = PortfolioOptimizer(risk_aversion=0.1).optimize(
        pd.Series([1.0, 0.5, -0.5, -1.0], index=assets),
        covariance,
        benchmark,
        benchmark,
        pd.DataFrame(index=assets),
        PortfolioConstraints(
            maximum_weight=0.45,
            maximum_active_weight=0.20,
            maximum_turnover=0.60,
            maximum_tracking_error=0.05,
        ),
    )
    active = result.target_weights - benchmark
    tracking_error = float(np.sqrt(active @ covariance @ active * 252))
    assert result.success
    assert tracking_error <= 0.05 + 1e-7
    assert result.constraint_slacks["tracking_error"] >= -1e-7


def test_product_templates_separate_research_and_capital_execution() -> None:
    templates = product_template_catalog()
    assert len(templates) == 4
    assert product_template("MARKET_NEUTRAL_RESEARCH").execution_mode == "research_vector"
    enhanced = product_template("UNIVERSE_INDEX_ENHANCED_PROXY")
    assert enhanced.execution_mode == "a_share_capital_ledger"
    assert enhanced.maximum_tracking_error == 0.08
    assert not enhanced.production_eligible


def test_optimizer_freezes_untradable_position() -> None:
    assets = pd.Index(["A", "B"])
    result = PortfolioOptimizer().optimize(
        pd.Series([1.0, 0.0], index=assets),
        pd.DataFrame(np.eye(2), index=assets, columns=assets),
        pd.Series([0.8, 0.2], index=assets),
        pd.Series([0.5, 0.5], index=assets),
        pd.DataFrame(index=assets),
        PortfolioConstraints(maximum_weight=0.6, maximum_active_weight=0.1),
        tradable=pd.Series([False, True], index=assets),
    )
    assert result.success
    assert result.target_weights["A"] == 0.8


def test_optimizer_falls_back_when_constraints_are_infeasible() -> None:
    assets = pd.Index(["A", "B"])
    result = PortfolioOptimizer().optimize(
        pd.Series([1.0, 0.0], index=assets),
        pd.DataFrame(np.eye(2), index=assets, columns=assets),
        pd.Series([0.5, 0.5], index=assets),
        pd.Series([0.5, 0.5], index=assets),
        pd.DataFrame({"beta": [1.0, 1.0]}, index=assets),
        PortfolioConstraints(
            maximum_weight=0.6,
            maximum_active_weight=0.1,
            exposure_bounds={"beta": (0.1, 0.2)},
        ),
    )
    assert result.used_fallback
    assert "fallback" in result.message


def test_return_attribution_reconciles() -> None:
    assets = pd.Index(["A", "B"])
    attribution = return_attribution(
        pd.Series([0.1, -0.1], index=assets),
        pd.Series([0.03, -0.01], index=assets),
        pd.DataFrame({"industry": [1.0, 0.0], "beta": [1.0, 1.0]}, index=assets),
        pd.Series({"industry": 0.02, "beta": 0.01}),
        transaction_cost=0.0005,
    )
    assert abs(attribution["gross"] - attribution["factor_total"] - attribution["specific"]) < 1e-12
    assert attribution["net"] == attribution["gross"] - 0.0005


def test_stress_scenarios_map_factor_shocks_to_portfolio_loss() -> None:
    assets = pd.Index(["A", "B"])
    result = stress_portfolio(
        pd.Series([0.2, 0.1], index=assets),
        pd.DataFrame({"beta": [1.2, 0.8], "liquidity": [0.2, 1.0]}, index=assets),
        {"liquidity_crisis": pd.Series({"beta": -0.03, "liquidity": -0.05})},
    )
    assert result["liquidity_crisis"] < 0
