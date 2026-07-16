from __future__ import annotations

import copy
from typing import Any

import numpy as np

from autoalpha.research.multiple_testing import probability_of_backtest_overfitting
from autoalpha.research.statistics import benjamini_hochberg


def apply_batch_multiple_testing(
    metrics_by_factor: dict[str, dict[str, Any]],
    *,
    alpha: float,
) -> tuple[dict[str, dict[str, Any]], float]:
    """Apply order-independent FDR and CSCV/PBO to one frozen candidate family."""
    if not metrics_by_factor:
        raise ValueError("At least one successful factor evaluation is required")
    factor_ids = list(metrics_by_factor)
    p_values = [
        float(metrics_by_factor[factor_id]["net_return_hac_p_value"]) for factor_id in factor_ids
    ]
    rejected = benjamini_hochberg(p_values, alpha=alpha)
    pbo = _family_pbo([metrics_by_factor[factor_id] for factor_id in factor_ids])
    adjusted: dict[str, dict[str, Any]] = {}
    for index, factor_id in enumerate(factor_ids):
        metrics = copy.deepcopy(metrics_by_factor[factor_id])
        metrics.update(
            {
                "multiple_testing_family_size": len(factor_ids),
                "multiple_testing_scope": "FROZEN_FULL_LIBRARY_CURRENT_PROTOCOL",
                "multiple_testing_fdr_alpha": alpha,
                "multiple_testing_fdr_passed": bool(rejected[index]),
                "probability_backtest_overfitting": pbo,
            }
        )
        adjusted[factor_id] = metrics
    return adjusted, pbo


def _family_pbo(metrics: list[dict[str, Any]]) -> float:
    profiles = [_fold_profile(item.get("walk_forward_folds", [])) for item in metrics]
    profiles = [profile for profile in profiles if profile]
    if len(profiles) < 2:
        return 0.0
    common = sorted(set.intersection(*(set(profile) for profile in profiles)))
    if len(common) < 4:
        return 0.0
    if len(common) % 2:
        common = common[:-1]
    matrix = np.asarray([[profile[year] for profile in profiles] for year in common], dtype=float)
    return float(probability_of_backtest_overfitting(matrix))


def _fold_profile(folds: list[dict[str, Any]]) -> dict[int, float]:
    return {
        int(str(fold["validation_start"])[:4]): float(fold["annual_return"])
        for fold in folds
        if fold.get("validation_start") and isinstance(fold.get("annual_return"), int | float)
    }
