from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RiskModelEstimate:
    asset_covariance: pd.DataFrame
    factor_covariance: pd.DataFrame
    specific_variance: pd.Series
    latest_exposures: pd.DataFrame
    factor_returns: pd.DataFrame

    def portfolio_volatility(self, active_weights: pd.Series) -> float:
        weights = active_weights.reindex(self.asset_covariance.index).fillna(0.0).to_numpy()
        variance = float(weights @ self.asset_covariance.to_numpy() @ weights)
        return float(np.sqrt(max(0.0, variance * 252)))


class RiskModel:
    def __init__(self, *, shrinkage: float = 0.25, minimum_specific_variance: float = 1e-8):
        if not 0 <= shrinkage <= 1:
            raise ValueError("shrinkage must be between zero and one")
        self.shrinkage = shrinkage
        self.minimum_specific_variance = minimum_specific_variance

    def fit(
        self,
        returns: pd.DataFrame,
        exposures: dict[str, pd.DataFrame],
    ) -> RiskModelEstimate:
        if not exposures:
            raise ValueError("At least one risk exposure is required")
        common_dates = returns.index
        common_assets = returns.columns
        for frame in exposures.values():
            common_dates = common_dates.intersection(frame.index)
            common_assets = common_assets.intersection(frame.columns)
        if len(common_dates) < 3 or len(common_assets) < 2:
            raise ValueError("Risk model requires at least three dates and two assets")

        factor_names = list(exposures)
        factor_returns: list[np.ndarray] = []
        residual_rows: list[pd.Series] = []
        for date in common_dates:
            y = returns.loc[date, common_assets].astype(float)
            x = pd.concat(
                [exposures[name].loc[date, common_assets].rename(name) for name in factor_names],
                axis=1,
            ).astype(float)
            valid = y.notna() & x.notna().all(axis=1)
            if valid.sum() <= len(factor_names):
                continue
            coefficients, *_ = np.linalg.lstsq(
                x.loc[valid].to_numpy(), y.loc[valid].to_numpy(), rcond=None
            )
            factor_returns.append(coefficients)
            residual = pd.Series(np.nan, index=common_assets, dtype=float)
            residual.loc[valid] = y.loc[valid] - x.loc[valid].to_numpy() @ coefficients
            residual_rows.append(residual.rename(date))
        if len(factor_returns) < 2:
            raise ValueError("Insufficient valid cross-sections for risk estimation")

        factor_return_frame = pd.DataFrame(factor_returns, columns=factor_names)
        sample_covariance = factor_return_frame.cov()
        diagonal = pd.DataFrame(
            np.diag(np.diag(sample_covariance)), index=factor_names, columns=factor_names
        )
        factor_covariance = (1 - self.shrinkage) * sample_covariance + self.shrinkage * diagonal
        residual_frame = pd.DataFrame(residual_rows)
        specific_variance = residual_frame.var(ddof=1).fillna(self.minimum_specific_variance)
        specific_variance = specific_variance.clip(lower=self.minimum_specific_variance)
        latest_date = common_dates[-1]
        latest = pd.concat(
            [exposures[name].loc[latest_date, common_assets].rename(name) for name in factor_names],
            axis=1,
        ).fillna(0.0)
        covariance_values = (latest @ factor_covariance @ latest.T).to_numpy(copy=True)
        covariance_values[np.diag_indices_from(covariance_values)] += specific_variance.to_numpy()
        covariance = pd.DataFrame(covariance_values, index=common_assets, columns=common_assets)
        return RiskModelEstimate(
            asset_covariance=covariance,
            factor_covariance=factor_covariance,
            specific_variance=specific_variance,
            latest_exposures=latest,
            factor_returns=factor_return_frame,
        )


def price_volume_risk_exposures(
    adjusted_close: pd.DataFrame,
    amount: pd.DataFrame,
    *,
    beta_window: int = 126,
    volatility_window: int = 60,
    momentum_window: int = 126,
    liquidity_window: int = 20,
) -> dict[str, pd.DataFrame]:
    """Build point-in-time price/volume risk proxies without fundamental data."""
    returns = adjusted_close.pct_change(fill_method=None)
    market = returns.mean(axis=1)
    market_variance = market.rolling(beta_window, min_periods=40).var().replace(0, np.nan)
    beta = returns.rolling(beta_window, min_periods=40).cov(market).div(market_variance, axis=0)
    volatility = returns.rolling(volatility_window, min_periods=20).std()
    momentum = adjusted_close.pct_change(momentum_window, fill_method=None)
    liquidity = np.log1p(amount.rolling(liquidity_window, min_periods=10).mean())
    size_proxy = np.log1p(amount.rolling(60, min_periods=20).mean())
    raw = {
        "beta": beta,
        "volatility": volatility,
        "momentum": momentum,
        "liquidity": liquidity,
        "size_proxy": size_proxy,
    }
    return {name: _cross_sectional_standardize(frame) for name, frame in raw.items()}


def _cross_sectional_standardize(values: pd.DataFrame) -> pd.DataFrame:
    centered = values.sub(values.mean(axis=1), axis=0)
    scale = centered.std(axis=1).replace(0, np.nan)
    return centered.div(scale, axis=0).clip(-5, 5)


def return_attribution(
    active_weights: pd.Series,
    asset_returns: pd.Series,
    exposures: pd.DataFrame,
    factor_returns: pd.Series,
    *,
    transaction_cost: float = 0.0,
) -> dict[str, float]:
    assets = active_weights.index.intersection(asset_returns.index).intersection(exposures.index)
    weights = active_weights.loc[assets]
    realized = float(weights @ asset_returns.loc[assets])
    factor_exposure = exposures.loc[assets].T @ weights
    factors = factor_exposure.index.intersection(factor_returns.index)
    factor_contributions = factor_exposure.loc[factors] * factor_returns.loc[factors]
    result = {f"factor:{name}": float(value) for name, value in factor_contributions.items()}
    explained = float(factor_contributions.sum())
    result.update(
        {
            "gross": realized,
            "factor_total": explained,
            "specific": realized - explained,
            "cost": -abs(transaction_cost),
            "net": realized - abs(transaction_cost),
        }
    )
    return result


def stress_portfolio(
    active_weights: pd.Series,
    exposures: pd.DataFrame,
    scenarios: dict[str, pd.Series],
) -> dict[str, float]:
    assets = active_weights.index.intersection(exposures.index)
    factor_exposure = exposures.loc[assets].T @ active_weights.loc[assets]
    return {
        name: float(factor_exposure.reindex(shock.index).fillna(0.0) @ shock)
        for name, shock in scenarios.items()
    }
