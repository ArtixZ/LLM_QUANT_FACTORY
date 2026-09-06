from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
from datetime import date
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from autoalpha.config import DateRange, ResearchConfig, SplitConfig, WalkForwardConfig
from autoalpha.data.execution_basis import ExecutionDataBasis
from autoalpha.dsl.expression import FactorDefinition, field
from autoalpha.service.evaluator import (
    PriceVolumeEvaluator,
    _annualized_ir,
    _compound_annual_return,
    _exploratory_gate_failures,
    _ic_diagnostics,
    _normalize_weights,
    _shared_signal_cache_key,
    _standalone_long_only_metrics,
)


def test_dashboard_return_metrics_use_explicit_annualization() -> None:
    returns = pd.Series([0.01, -0.005, 0.002])

    assert _annualized_ir(returns) == pytest.approx(returns.mean() / returns.std(ddof=1) * 252**0.5)
    assert _compound_annual_return(returns) == pytest.approx(
        (1.01 * 0.995 * 1.002) ** (252 / 3) - 1
    )


def test_sparse_ic_is_diagnostic_only_and_remains_finite() -> None:
    empty = _ic_diagnostics(pd.Series(dtype=float))
    sparse = _ic_diagnostics(pd.Series([0.01, 0.02, -0.01]))

    assert empty == {"mean": 0.0, "ir": 0.0, "p_value": 1.0}
    assert sparse["mean"] == pytest.approx(0.02 / 3)
    assert np.isfinite(list(sparse.values())).all()


def test_shared_signal_cache_isolated_by_direction_and_data_scope() -> None:
    config = ResearchConfig.from_toml(Path("config/research.toml"))
    evaluator = SimpleNamespace(
        workspace=SimpleNamespace(fingerprint="data-v1"),
        config=config,
    )
    positive = FactorDefinition("positive", "test", "positive", field("amount"), 1)
    negative = FactorDefinition("negative", "test", "negative", field("amount"), -1)

    positive_key = _shared_signal_cache_key(evaluator, positive)
    negative_key = _shared_signal_cache_key(evaluator, negative)

    assert positive.factor_id == negative.factor_id
    assert positive_key != negative_key
    assert positive_key[:3] == (
        "data-v1",
        config.splits.train.start.isoformat(),
        config.splits.validation.end.isoformat(),
    )


def test_signal_preflight_distinguishes_slow_state_from_degenerate_cross_section() -> None:
    base = ResearchConfig.from_toml(Path("config/research.toml"))
    config = replace(base, minimum_cross_section=3)
    dates = pd.bdate_range(config.splits.validation.start, periods=100)
    columns = ["A", "B", "C", "D"]
    slow = FactorDefinition("slow", "test", "slow state", field("amount"))
    degenerate = FactorDefinition("flat", "test", "flat state", field("close"))
    evaluator = object.__new__(PriceVolumeEvaluator)
    evaluator.config = config
    evaluator.validator = type("Validator", (), {"validate": lambda self, expression: None})()
    evaluator._signal_cache = OrderedDict(
        {
            slow.factor_id: pd.DataFrame(
                np.tile([1.0, 2.0, 3.0, 4.0], (len(dates), 1)),
                index=dates,
                columns=columns,
            ),
            degenerate.factor_id: pd.DataFrame(1.0, index=dates, columns=columns),
        }
    )
    evaluator._signal_cache_lock = Lock()
    evaluator._preflight_cache = OrderedDict()
    evaluator._preflight_cache_lock = Lock()
    evaluator._field_load_lock = None
    evaluator._fields = {
        "adj_close": pd.DataFrame(10.0, index=dates, columns=columns),
        "open": pd.DataFrame(10.0, index=dates, columns=columns),
        "amount": pd.DataFrame(1_000_000.0, index=dates, columns=columns),
    }

    slow_result = evaluator.preflight(slow)
    flat_result = evaluator.preflight(degenerate)

    assert slow_result.passed
    assert "low_temporal_update_rate" in slow_result.warnings
    assert not flat_result.passed
    assert "cross_sectionally_degenerate" in flat_result.failures


def test_exploratory_gates_do_not_treat_zero_control_drawdown_as_incremental() -> None:
    config = ResearchConfig.from_toml(Path("config/research.toml"))
    metrics = {
        "long_only_coverage": 0.95,
        "long_only_sharpe_ratio": 0.64,
        "long_only_simple_annual_return": 0.07,
        "long_only_cost_stress_net_ir": 0.45,
        "long_only_positive_year_ratio": 1.0,
        "long_only_worst_year_return": 0.002,
        "long_only_annual_return_dispersion": 0.05,
        "long_only_annual_turnover": 51.0,
        "long_only_capacity_usd": 160_000_000.0,
    }

    failures = _exploratory_gate_failures(metrics, config)

    assert failures == ["turnover"]


def test_exploratory_gates_ignore_strong_alpha_when_long_only_is_weak() -> None:
    config = ResearchConfig.from_toml(Path("config/research.toml"))
    metrics = {
        "sharpe_ratio": 8.0,
        "simple_annual_return": 0.80,
        "long_only_coverage": 0.95,
        "long_only_sharpe_ratio": -0.20,
        "long_only_simple_annual_return": -0.02,
        "long_only_cost_stress_net_ir": -0.10,
        "long_only_positive_year_ratio": 0.40,
        "long_only_worst_year_return": -0.10,
        "long_only_annual_return_dispersion": 0.05,
        "long_only_annual_turnover": 10.0,
        "long_only_capacity_usd": 160_000_000.0,
    }

    failures = _exploratory_gate_failures(metrics, config)

    assert "long_only_net_ir" in failures
    assert "long_only_annual_return" in failures
    assert "cost_stress" in failures


def test_portfolio_weights_are_normalized_and_must_align() -> None:
    factors = [
        FactorDefinition("a", "test", "test", field("amount")),
        FactorDefinition("b", "test", "test", field("vol")),
    ]

    assert _normalize_weights(factors, [3.0, 1.0]) == (0.75, 0.25)
    with pytest.raises(ValueError, match="align"):
        _normalize_weights(factors, [1.0])


def test_portfolio_evaluation_separates_alpha_diagnostic_from_us_strategy() -> None:
    base = ResearchConfig.from_toml(Path("config/research.toml"))
    config = replace(
        base,
        splits=SplitConfig(
            DateRange(date(2019, 1, 2), date(2019, 12, 31)),
            DateRange(date(2020, 1, 2), date(2021, 12, 31)),
            DateRange(date(2022, 1, 3), date(2022, 12, 30)),
        ),
        walk_forward=WalkForwardConfig(
            train_years=1,
            validation_years=1,
            first_validation_year=2020,
            last_validation_year=2021,
            minimum_folds=2,
        ),
        strategy_evaluation=replace(
            base.strategy_evaluation,
            maximum_positions=1,
            selection_fraction=0.25,
        ),
    )
    dates = pd.bdate_range("2019-01-02", "2021-12-31")
    columns = ["A", "B", "C", "D"]
    growth = np.array([1.002, 1.001, 0.999, 0.998])
    adjusted_open = pd.DataFrame(
        10.0 * np.cumprod(np.tile(growth, (len(dates), 1)), axis=0),
        index=dates,
        columns=columns,
    )
    signal = pd.DataFrame(
        np.tile([4.0, 3.0, 2.0, 1.0], (len(dates), 1)),
        index=dates,
        columns=columns,
    )
    tradable = pd.DataFrame(True, index=dates, columns=columns)
    evaluator = object.__new__(PriceVolumeEvaluator)
    evaluator.config = config
    evaluator.execution_basis = ExecutionDataBasis(
        "forward_adjusted", "raw", "shares", "cny", False, True, (), ()
    )
    evaluator.trial_count = 1
    evaluator._fields = {
        "open": adjusted_open,
        "close": adjusted_open,
        "adj_close": adjusted_open,
        "amount": pd.DataFrame(100_000_000.0, index=dates, columns=columns),
        "vol": pd.DataFrame(10_000_000.0, index=dates, columns=columns),
        "raw_open": adjusted_open,
        "can_buy_open_proxy": tradable,
        "can_sell_open_proxy": tradable,
    }
    factor = FactorDefinition("synthetic", "test", "test", field("amount"))
    evaluator._signal_cache = {factor.factor_id: signal}
    evaluator._portfolio_path_cache = OrderedDict()
    evaluator._portfolio_path_cache_lock = Lock()

    metrics = evaluator.evaluate_portfolio([factor]).metrics

    assert metrics["portfolio_strategy_gate_basis"] == (
        "US_EQUITY_LONG_ONLY_WEEKLY_NON_PIT_PROXY"
    )
    assert metrics["portfolio_mode"] == "long_only"
    assert metrics["portfolio_rebalance_schedule"] == "WEEKLY_FIRST_SESSION"
    assert metrics["portfolio_maximum_positions"] == 1
    assert metrics["alpha_diagnostic_scope"] == "NON_INVESTABLE_LONG_SHORT"
    assert metrics["portfolio_sharpe_ratio"] != metrics["alpha_diagnostic_sharpe_ratio"]
    assert metrics["portfolio_benchmark_mode"] == "ELIGIBLE_UNIVERSE_EQUAL_WEIGHT_PROXY"
    assert "portfolio_active_information_ratio" in metrics
    assert "portfolio_market_beta" in metrics

    long_only_all = evaluator._strategy_portfolio_path([factor], (1.0,))
    long_only = long_only_all.loc[long_only_all.index >= pd.Timestamp("2020-01-02")].copy()
    long_only.attrs.update(long_only_all.attrs)
    standalone = _standalone_long_only_metrics(long_only, config, trials=1)

    assert standalone["long_only_mode"] == "long_only"
    assert standalone["long_only_sharpe_ratio"] == pytest.approx(metrics["portfolio_sharpe_ratio"])
    assert standalone["long_only_simple_annual_return"] == pytest.approx(
        metrics["portfolio_simple_annual_return"]
    )
    assert standalone["long_only_walk_forward_fold_count"] == 2


def test_library_signal_correlation_flags_behavioral_duplicate() -> None:
    dates = pd.bdate_range("2021-01-04", periods=40)
    columns = ["A", "B", "C", "D"]
    candidate_signal = pd.DataFrame(
        np.tile([1.0, 2.0, 3.0, 4.0], (len(dates), 1)), index=dates, columns=columns
    )
    duplicate_signal = candidate_signal * 2.0
    weak_signal = pd.DataFrame(
        np.tile([2.0, 1.0, 4.0, 3.0], (len(dates), 1)), index=dates, columns=columns
    )

    candidate = FactorDefinition("candidate", "test", "test", field("amount"))
    duplicate = FactorDefinition("duplicate", "test", "test", field("close"))
    weak_overlap = FactorDefinition("weak_overlap", "test", "test", field("vol"))
    broken = FactorDefinition("broken", "test", "test", field("nonexistent"))

    evaluator = object.__new__(PriceVolumeEvaluator)
    evaluator._signal_cache = {
        candidate.factor_id: candidate_signal,
        duplicate.factor_id: duplicate_signal,
        weak_overlap.factor_id: weak_signal,
    }

    metrics = evaluator.library_signal_correlation(
        candidate, [broken, weak_overlap, duplicate, candidate]
    )

    assert metrics["library_signal_correlation_max"] == pytest.approx(1.0)
    assert metrics["library_signal_correlation_peer"] == duplicate.factor_id
    assert metrics["library_signal_correlation_peer_name"] == "duplicate"
    # broken reference is skipped without failing; candidate itself is excluded
    assert metrics["library_signal_reference_count"] == 2
