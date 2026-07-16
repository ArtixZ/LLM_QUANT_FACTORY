from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from autoalpha.config import ResearchConfig
from autoalpha.dsl.expression import FactorDefinition, field
from autoalpha.service.evaluator import (
    _annualized_ir,
    _compound_annual_return,
    _exploratory_gate_failures,
    _normalize_weights,
)


def test_dashboard_return_metrics_use_explicit_annualization() -> None:
    returns = pd.Series([0.01, -0.005, 0.002])

    assert _annualized_ir(returns) == pytest.approx(returns.mean() / returns.std(ddof=1) * 245**0.5)
    assert _compound_annual_return(returns) == pytest.approx(
        (1.01 * 0.995 * 1.002) ** (245 / 3) - 1
    )


def test_exploratory_gates_do_not_treat_zero_control_drawdown_as_incremental() -> None:
    config = ResearchConfig.from_toml(Path("config/research.toml"))
    metrics = {
        "coverage": 0.95,
        "incremental_net_ir": 0.64,
        "incremental_annual_return": 0.07,
        "incremental_max_drawdown": -0.18,
        "return_drawdown_efficiency_change": 0.30,
        "cost_stress_net_ir": 0.45,
        "positive_year_ratio": 1.0,
        "worst_year_incremental_return": 0.002,
        "annual_return_dispersion": 0.05,
        "annual_turnover": 51.0,
        "capacity_cny": 160_000_000.0,
    }

    failures = _exploratory_gate_failures(metrics, config)

    assert failures == ["turnover"]


def test_portfolio_weights_are_normalized_and_must_align() -> None:
    factors = [
        FactorDefinition("a", "test", "test", field("amount")),
        FactorDefinition("b", "test", "test", field("vol")),
    ]

    assert _normalize_weights(factors, [3.0, 1.0]) == (0.75, 0.25)
    with pytest.raises(ValueError, match="align"):
        _normalize_weights(factors, [1.0])
