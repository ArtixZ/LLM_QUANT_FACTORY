from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def neutralize_cross_section(
    factor: pd.DataFrame,
    exposures: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Regress each date on supplied exposures and return residuals."""
    result = pd.DataFrame(index=factor.index, columns=factor.columns, dtype=float)
    for date in factor.index:
        y = factor.loc[date].astype(float)
        columns = [frame.loc[date].astype(float).rename(name) for name, frame in exposures.items()]
        x = pd.concat(columns, axis=1) if columns else pd.DataFrame(index=y.index)
        valid = y.notna() & x.notna().all(axis=1)
        if valid.sum() <= x.shape[1] + 1:
            continue
        design = np.column_stack([np.ones(valid.sum()), x.loc[valid].to_numpy()])
        coefficients, *_ = np.linalg.lstsq(design, y.loc[valid].to_numpy(), rcond=None)
        result.loc[date, valid] = y.loc[valid] - design @ coefficients
    return result


@dataclass(frozen=True)
class FactorNovelty:
    maximum_correlation: float
    nearest_factor: str | None
    conditional_ic: float
    incremental_r_squared: float
    incremental_portfolio_ir: float
    is_duplicate: bool


def cluster_factors(
    correlations: pd.DataFrame, *, threshold: float = 0.85
) -> tuple[tuple[str, ...], ...]:
    if not correlations.index.equals(correlations.columns):
        raise ValueError("Factor correlation matrix must have matching index and columns")
    remaining = set(map(str, correlations.index))
    clusters = []
    while remaining:
        seed = min(remaining)
        cluster = {seed}
        frontier = [seed]
        while frontier:
            current = frontier.pop()
            neighbors = {
                str(name)
                for name, value in correlations.loc[current].items()
                if name in remaining and name != current and abs(float(value)) >= threshold
            }
            new = neighbors - cluster
            cluster.update(new)
            frontier.extend(sorted(new))
        remaining -= cluster
        clusters.append(tuple(sorted(cluster)))
    return tuple(sorted(clusters))


def assess_novelty(
    candidate: pd.Series,
    library: pd.DataFrame,
    forward_return: pd.Series,
    *,
    candidate_portfolio_return: pd.Series | None = None,
    library_portfolio_return: pd.Series | None = None,
    duplicate_threshold: float = 0.85,
) -> FactorNovelty:
    aligned = pd.concat(
        [candidate.rename("candidate"), forward_return.rename("target"), library], axis=1
    ).dropna()
    library_names = list(library.columns)
    if aligned.empty:
        raise ValueError("No common valid observations for novelty assessment")

    correlations = aligned[library_names].corrwith(aligned["candidate"]) if library_names else None
    if correlations is None or correlations.empty:
        maximum_correlation, nearest = 0.0, None
    else:
        nearest = correlations.abs().idxmax()
        maximum_correlation = float(correlations.loc[nearest])

    baseline = _regression_r_squared(aligned["target"], aligned[library_names])
    expanded = _regression_r_squared(aligned["target"], aligned[library_names + ["candidate"]])
    residual = _regression_residual(aligned["candidate"], aligned[library_names])
    conditional_ic = float(residual.corr(aligned["target"], method="spearman"))
    incremental_ir = _incremental_ir(candidate_portfolio_return, library_portfolio_return)
    return FactorNovelty(
        maximum_correlation=maximum_correlation,
        nearest_factor=nearest,
        conditional_ic=conditional_ic,
        incremental_r_squared=max(0.0, expanded - baseline),
        incremental_portfolio_ir=incremental_ir,
        is_duplicate=abs(maximum_correlation) >= duplicate_threshold,
    )


def _regression_residual(y: pd.Series, x: pd.DataFrame) -> pd.Series:
    if x.empty:
        return y - y.mean()
    design = np.column_stack([np.ones(len(x)), x.to_numpy(dtype=float)])
    coefficients, *_ = np.linalg.lstsq(design, y.to_numpy(dtype=float), rcond=None)
    return pd.Series(y.to_numpy() - design @ coefficients, index=y.index)


def _regression_r_squared(y: pd.Series, x: pd.DataFrame) -> float:
    residual = _regression_residual(y, x)
    total = float(((y - y.mean()) ** 2).sum())
    return 0.0 if total == 0 else 1.0 - float((residual**2).sum()) / total


def _incremental_ir(candidate: pd.Series | None, baseline: pd.Series | None) -> float:
    if candidate is None:
        return float("nan")
    combined = candidate if baseline is None else candidate.add(baseline, fill_value=np.nan)
    baseline_ir = 0.0 if baseline is None else _annualized_ir(baseline.dropna())
    return _annualized_ir(combined.dropna()) - baseline_ir


def _annualized_ir(returns: pd.Series) -> float:
    standard_deviation = float(returns.std(ddof=1))
    return (
        0.0
        if len(returns) < 2 or standard_deviation == 0
        else float(np.sqrt(252) * returns.mean() / standard_deviation)
    )
