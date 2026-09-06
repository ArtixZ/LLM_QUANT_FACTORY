from __future__ import annotations

from autoalpha.research.batch_reevaluation import apply_batch_multiple_testing


def _metrics(p_value: float, annual_returns: list[float]) -> dict:
    return {
        "long_only_net_return_hac_p_value": p_value,
        "long_only_walk_forward_folds": [
            {
                "validation_start": f"{2015 + index}-01-01",
                "annual_return": value,
            }
            for index, value in enumerate(annual_returns)
        ],
    }


def test_batch_adjustments_are_order_independent_and_family_wide() -> None:
    source = {
        "F_1": _metrics(0.001, [0.3, 0.2, 0.1, 0.0]),
        "F_2": _metrics(0.2, [-0.1, 0.0, 0.1, 0.2]),
    }

    adjusted, pbo = apply_batch_multiple_testing(source, alpha=0.05)

    assert adjusted["F_1"]["multiple_testing_fdr_passed"] is True
    assert adjusted["F_2"]["multiple_testing_fdr_passed"] is False
    assert adjusted["F_1"]["multiple_testing_family_size"] == 2
    assert adjusted["F_1"]["probability_backtest_overfitting"] == pbo
    assert adjusted["F_1"]["multiple_testing_primary_basis"] == "US_EQUITY_LONG_ONLY"
    assert source["F_1"].get("multiple_testing_fdr_passed") is None
