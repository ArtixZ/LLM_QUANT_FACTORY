import numpy as np
import pandas as pd

from autoalpha.research.incremental import annual_robustness, compare_portfolios


def test_paired_portfolio_comparison_measures_net_increment_not_standalone_return() -> None:
    rng = np.random.default_rng(31)
    dates = pd.bdate_range("2022-01-01", periods=300)
    common = rng.normal(0.0003, 0.01, len(dates))
    control = pd.Series(common, index=dates)
    alpha = rng.normal(0.0002, 0.0008, len(dates))
    treatment = pd.Series(common + alpha, index=dates)
    stressed = pd.Series(common + alpha - 0.0001, index=dates)

    result = compare_portfolios(
        control,
        treatment,
        stressed_treatment_net_returns=stressed,
        hac_lags=5,
        bootstrap_block_size=10,
        bootstrap_samples=200,
        seed=7,
    )

    assert result.observations == 300
    assert result.incremental_net_ir > 0
    assert result.incremental_annual_return > 0
    assert result.hac.mean == result.incremental_returns.mean()
    assert result.cost_stress_net_ir is not None
    assert np.isfinite(result.return_drawdown_efficiency_change)


def test_portfolio_comparison_uses_only_common_non_missing_dates() -> None:
    dates = pd.bdate_range("2024-01-01", periods=20)
    control = pd.Series(np.linspace(-0.01, 0.01, 20), index=dates)
    treatment = control + pd.Series(np.linspace(-0.001, 0.002, 20), index=dates)
    treatment.iloc[:3] = np.nan

    result = compare_portfolios(
        control,
        treatment,
        hac_lags=2,
        bootstrap_block_size=5,
        bootstrap_samples=50,
    )

    assert result.observations == 17
    assert result.incremental_returns.index[0] == dates[3]


def test_annual_robustness_reports_raw_year_evidence() -> None:
    dates = pd.bdate_range("2021-01-01", "2023-12-31")
    returns = pd.Series(0.0002, index=dates)
    returns.loc[returns.index.year == 2022] = -0.0001

    result = annual_robustness(returns)

    assert result.years == 3
    assert result.positive_year_ratio == 2 / 3
    assert result.worst_year_return < 0
    assert result.annual_return_dispersion > 0
