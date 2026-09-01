from __future__ import annotations

import gc
import hashlib
import math
import os
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from autoalpha.backtest.timing import entry_aligned_open_return
from autoalpha.backtest.vector import VectorBacktestConfig, VectorBacktester
from autoalpha.config import ResearchConfig
from autoalpha.data.execution_basis import inspect_execution_data_basis
from autoalpha.data.research_fields import field_definitions
from autoalpha.data.workspace import inspect_data_workspace
from autoalpha.dsl.compiler import FactorCompiler
from autoalpha.dsl.expression import Expression, FactorDefinition
from autoalpha.dsl.semantics import SemanticValidator
from autoalpha.research.evaluation import cross_sectional_ic
from autoalpha.research.multiple_testing import deflated_sharpe_ratio
from autoalpha.research.statistics import hac_mean_inference

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")


@dataclass(frozen=True)
class MassiveBatchConfig:
    data_path: Path
    config_path: Path
    start_date: date
    end_date: date
    workers: int = 4
    holding_period_days: int = 5
    gross_exposure: float = 1.0
    selection_fraction: float = 0.10
    maximum_positions_per_side: int = 30
    commission_bps_each_side: float = 0.5
    sec_fee_bps_sell: float = 0.278
    slippage_bps_each_side: float = 2.0
    cost_stress_multiplier: float = 2.0
    window_months: int = 36
    step_months: int = 12
    monte_carlo_samples: int = 10_000
    monte_carlo_block_days: int = 20
    parameter_multipliers: tuple[float, ...] = (0.5, 2.0)
    holding_period_tests: tuple[int, ...] = (1, 20)

    def __post_init__(self) -> None:
        if self.start_date >= self.end_date:
            raise ValueError("Batch start date must be before end date")
        if not 1 <= self.workers <= 8:
            raise ValueError("Batch workers must be between 1 and 8")
        if self.window_months < 12 or self.step_months < 3:
            raise ValueError("Window and step sizes are too small for massive robustness tests")
        if not 1_000 <= self.monte_carlo_samples <= 100_000:
            raise ValueError("Monte Carlo samples must be between 1,000 and 100,000")
        if not 5 <= self.monte_carlo_block_days <= 120:
            raise ValueError("Monte Carlo block size must be between 5 and 120 sessions")

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.__dict__,
            "data_path": str(self.data_path),
            "config_path": str(self.config_path),
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "parameter_multipliers": list(self.parameter_multipliers),
            "holding_period_tests": list(self.holding_period_tests),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> MassiveBatchConfig:
        return cls(
            data_path=Path(value["data_path"]),
            config_path=Path(value["config_path"]),
            start_date=date.fromisoformat(value["start_date"]),
            end_date=date.fromisoformat(value["end_date"]),
            workers=int(value.get("workers", 4)),
            holding_period_days=int(value.get("holding_period_days", 5)),
            gross_exposure=float(value.get("gross_exposure", 1.0)),
            selection_fraction=float(value.get("selection_fraction", 0.10)),
            maximum_positions_per_side=int(value.get("maximum_positions_per_side", 30)),
            commission_bps_each_side=float(value.get("commission_bps_each_side", 0.5)),
            sec_fee_bps_sell=float(value.get("sec_fee_bps_sell", 0.278)),
            slippage_bps_each_side=float(value.get("slippage_bps_each_side", 2.0)),
            cost_stress_multiplier=float(value.get("cost_stress_multiplier", 2.0)),
            window_months=int(value.get("window_months", 36)),
            step_months=int(value.get("step_months", 12)),
            monte_carlo_samples=int(value.get("monte_carlo_samples", 10_000)),
            monte_carlo_block_days=int(value.get("monte_carlo_block_days", 20)),
            parameter_multipliers=tuple(
                float(item) for item in value.get("parameter_multipliers", [0.5, 2.0])
            ),
            holding_period_tests=tuple(
                int(item) for item in value.get("holding_period_tests", [1, 20])
            ),
        )


@dataclass(frozen=True)
class BatchFactorOutcome:
    factor_id: str
    elapsed_seconds: float
    metrics: dict[str, Any]
    monte_carlo: dict[str, Any]
    curve_path: str
    monte_carlo_path: str
    windows: list[dict[str, Any]]
    robustness: list[dict[str, Any]]


class MassiveVectorBatchEngine:
    """Shared-panel, factor-parallel vector evaluation engine."""

    def __init__(
        self,
        config: MassiveBatchConfig,
        artifact_root: Path,
        *,
        factor_family_size: int,
    ) -> None:
        self.config = config
        self.artifact_root = artifact_root
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.factor_family_size = factor_family_size
        self.research_config = ResearchConfig.from_toml(config.config_path)
        self.workspace = inspect_data_workspace(config.data_path)
        self.workspace.require_price_research()
        self.execution_basis = inspect_execution_data_basis(Path(self.workspace.panel_path))
        self.validator_fields = field_definitions(
            self.workspace.factor_fields,
            amount_unit=self.execution_basis.amount_unit,
            volume_unit=self.execution_basis.volume_unit,
        )
        self.fields = self._load_fields()
        self.entry_returns = entry_aligned_open_return(self.fields["open"])
        self.windows = generate_step_windows(
            config.start_date,
            config.end_date,
            window_months=config.window_months,
            step_months=config.step_months,
        )

    def run_factor(self, job_id: str, record: dict[str, Any]) -> BatchFactorOutcome:
        started = time.monotonic()
        factor = factor_from_snapshot(record)
        validator = SemanticValidator(self.validator_fields, maximum_nodes=30, maximum_lookback=252)
        compiler = FactorCompiler(validator)
        signal = compiler.evaluate(factor.expression, self.fields) * factor.expected_direction
        base = self._run_vector(signal, holding_period=self.config.holding_period_days)
        if len(base.path) < 252:
            raise ValueError("Base vector path contains fewer than 252 trading observations")

        selected_signal = signal.shift(1).reindex(base.path.index)
        selected_return = self.entry_returns.reindex(base.path.index)
        rank_ic = cross_sectional_ic(
            selected_signal,
            selected_return,
            minimum_names=self.research_config.minimum_cross_section,
        )
        pearson_ic = cross_sectional_ic(
            selected_signal,
            selected_return,
            method="pearson",
            minimum_names=self.research_config.minimum_cross_section,
        )
        window_results = self._window_results(base.path)
        metrics = self._enriched_metrics(
            base.path,
            base.metrics,
            signal=selected_signal,
            rank_ic=rank_ic,
            pearson_ic=pearson_ic,
            window_results=window_results,
        )
        robustness = self._robustness_tests(factor, signal)
        metrics["robustness_pass_fraction"] = _robustness_pass_fraction(
            robustness, float(metrics["sharpe_ratio"])
        )
        seed = int(hashlib.sha256(f"{job_id}:{record['factor_id']}".encode()).hexdigest()[:8], 16)
        monte_carlo_samples, monte_carlo = moving_block_monte_carlo(
            base.path["net"].to_numpy(),
            samples=self.config.monte_carlo_samples,
            block_size=self.config.monte_carlo_block_days,
            seed=seed,
        )
        factor_root = self.artifact_root / record["factor_id"]
        factor_root.mkdir(parents=True, exist_ok=True)
        curve_path = factor_root / "daily_path.parquet"
        monte_carlo_path = factor_root / "monte_carlo.parquet"
        curve = base.path.copy()
        curve["equity"] = base.equity
        curve["drawdown"] = base.drawdown
        _write_parquet_atomic(curve, curve_path)
        _write_parquet_atomic(monte_carlo_samples, monte_carlo_path)
        del signal, selected_signal, selected_return, base
        gc.collect()
        return BatchFactorOutcome(
            factor_id=str(record["factor_id"]),
            elapsed_seconds=time.monotonic() - started,
            metrics=metrics,
            monte_carlo=monte_carlo,
            curve_path=str(curve_path),
            monte_carlo_path=str(monte_carlo_path),
            windows=window_results,
            robustness=robustness,
        )

    def _run_vector(self, signal: pd.DataFrame, *, holding_period: int):
        return VectorBacktester(
            VectorBacktestConfig(
                holding_period_days=holding_period,
                gross_exposure=self.config.gross_exposure,
                selection_fraction=self.config.selection_fraction,
                maximum_positions_per_side=self.config.maximum_positions_per_side,
                long_only=False,
                commission_bps_each_side=self.config.commission_bps_each_side,
                sec_fee_bps_sell=self.config.sec_fee_bps_sell,
                slippage_bps_each_side=self.config.slippage_bps_each_side,
                cost_stress_multiplier=self.config.cost_stress_multiplier,
                cost_model="side_aware",
                path_index="entry_session",
                initial_cash_usd=1_000_000.0,
            )
        ).run(
            signal,
            self.fields["open"],
            start=self.config.start_date,
            end=self.config.end_date,
            precomputed_entry_returns=self.entry_returns,
        )

    def _window_results(self, path: pd.DataFrame) -> list[dict[str, Any]]:
        result = []
        for window_id, start, end in self.windows:
            selected = path.loc[
                (path.index >= pd.Timestamp(start)) & (path.index <= pd.Timestamp(end))
            ]
            if len(selected) < 120:
                continue
            result.append(
                {
                    "window_id": window_id,
                    "period_start": selected.index.min().date().isoformat(),
                    "period_end": selected.index.max().date().isoformat(),
                    "metrics": summarize_path(selected),
                }
            )
        return result

    def _enriched_metrics(
        self,
        path: pd.DataFrame,
        vector_metrics: dict[str, Any],
        *,
        signal: pd.DataFrame,
        rank_ic: pd.Series,
        pearson_ic: pd.Series,
        window_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        metrics = dict(vector_metrics)
        net = path["net"]
        inference = hac_mean_inference(net.to_numpy(), lags=min(5, len(net) - 1))
        dsr = deflated_sharpe_ratio(net.to_numpy(), trials=self.factor_family_size)
        open_panel = self.fields["open"].reindex(index=signal.index, columns=signal.columns)
        eligible = open_panel.notna()
        denominator = int(eligible.to_numpy().sum())
        folds = [
            {
                "validation_start": item["period_start"],
                "validation_end": item["period_end"],
                "annual_return": item["metrics"]["simple_annual_return"],
                "sharpe": item["metrics"]["sharpe_ratio"],
                "max_drawdown": item["metrics"]["max_drawdown"],
            }
            for item in window_results
        ]
        metrics.update(
            {
                "sortino_ratio": _sortino(net),
                "calmar_ratio": _calmar(net),
                "daily_win_rate": float((net > 0).mean()),
                "cost_stress_sharpe": _annualized_ratio(path["stressed"]),
                "coverage": (
                    float((signal.notna() & eligible).to_numpy().sum() / denominator)
                    if denominator
                    else 0.0
                ),
                "rank_ic_mean": _safe_mean(rank_ic),
                "rank_ic_ir": _annualized_ratio(rank_ic) if len(rank_ic) > 1 else None,
                "pearson_ic_mean": _safe_mean(pearson_ic),
                "pearson_ic_ir": _annualized_ratio(pearson_ic) if len(pearson_ic) > 1 else None,
                "net_return_hac_p_value": float(inference.p_value),
                "deflated_sharpe_probability": float(dsr.probability),
                "multiple_testing_trials": self.factor_family_size,
                "annual_returns": {
                    str(year): float((1.0 + values).prod() - 1.0)
                    for year, values in net.groupby(net.index.year)
                },
                "walk_forward_folds": folds,
                "large_window_count": len(folds),
                "large_window_positive_fraction": (
                    float(np.mean([item["annual_return"] > 0 for item in folds])) if folds else 0.0
                ),
                "large_window_worst_sharpe": (
                    float(min(item["sharpe"] for item in folds)) if folds else -100.0
                ),
                "large_window_median_sharpe": (
                    float(np.median([item["sharpe"] for item in folds])) if folds else -100.0
                ),
                "engine_protocol": "AUTOALPHA_MASSIVE_VECTOR_V1",
                "hidden_test_claim": False,
                "result_scope": "HUMAN_VISIBLE_MASSIVE_DIAGNOSTIC",
            }
        )
        return _json_ready(metrics)

    def _robustness_tests(
        self, factor: FactorDefinition, base_signal: pd.DataFrame
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for holding_period in self.config.holding_period_tests:
            if holding_period == self.config.holding_period_days:
                continue
            try:
                outcome = self._run_vector(base_signal, holding_period=holding_period)
                metrics = summarize_path(outcome.path)
                error = None
            except Exception as exception:  # noqa: BLE001
                metrics = None
                error = f"{type(exception).__name__}: {exception}"
            results.append(
                {
                    "test_type": "HOLDING_PERIOD",
                    "variant": f"{holding_period}d",
                    "metrics": metrics,
                    "error": error,
                }
            )
        original = factor.expression.to_dict()
        for multiplier in self.config.parameter_multipliers:
            scaled = _scale_expression_parameters(original, multiplier)
            if scaled == original:
                results.append(
                    {
                        "test_type": "EXPRESSION_WINDOW",
                        "variant": f"x{multiplier:g}",
                        "metrics": None,
                        "error": "NO_TUNABLE_WINDOW",
                    }
                )
                continue
            try:
                validator = SemanticValidator(
                    self.validator_fields, maximum_nodes=30, maximum_lookback=252
                )
                expression = Expression.from_dict(scaled)
                scaled_signal = FactorCompiler(validator).evaluate(expression, self.fields)
                scaled_signal = scaled_signal * factor.expected_direction
                outcome = self._run_vector(
                    scaled_signal, holding_period=self.config.holding_period_days
                )
                metrics = summarize_path(outcome.path)
                error = None
                del scaled_signal, outcome
            except Exception as exception:  # noqa: BLE001
                metrics = None
                error = f"{type(exception).__name__}: {exception}"
            results.append(
                {
                    "test_type": "EXPRESSION_WINDOW",
                    "variant": f"x{multiplier:g}",
                    "metrics": metrics,
                    "error": error,
                }
            )
        return results

    def _load_fields(self) -> dict[str, pd.DataFrame]:
        panel_path = Path(self.workspace.panel_path)
        load_start = pd.Timestamp(self.config.start_date) - pd.Timedelta(days=800)
        load_end = pd.Timestamp(self.config.end_date) + pd.Timedelta(days=10)
        factor_columns = list(self.workspace.factor_fields)
        columns = list(
            dict.fromkeys(
                [
                    "trade_date",
                    "symbol",
                    "open",
                    *factor_columns,
                    "is_valid_ohlc",
                    "is_tradable_observation",
                ]
            )
        )
        frames = []
        for year in range(load_start.year, load_end.year + 1):
            for path in sorted((panel_path / f"trade_year={year}").glob("*.parquet")):
                frames.append(pd.read_parquet(path, columns=columns))
        if not frames:
            raise FileNotFoundError(f"No parquet partitions found under {panel_path}")
        data = pd.concat(frames, ignore_index=True)
        data["trade_date"] = pd.to_datetime(data["trade_date"])
        data = data[(data["trade_date"] >= load_start) & (data["trade_date"] <= load_end)]
        valid = data["is_valid_ohlc"].fillna(False) & data["is_tradable_observation"].fillna(False)
        value_columns = list(dict.fromkeys(["open", *factor_columns]))
        data.loc[~valid, value_columns] = np.nan
        fields = {
            name: data.pivot(index="trade_date", columns="symbol", values=name).sort_index()
            for name in value_columns
        }
        del data, frames
        gc.collect()
        return fields


def factor_from_snapshot(record: dict[str, Any]) -> FactorDefinition:
    proposal = record["proposal"]
    return FactorDefinition(
        name=str(proposal["name"]),
        family=str(proposal["family"]),
        hypothesis=str(proposal["hypothesis"]),
        expression=Expression.from_dict(proposal["expression"]),
        expected_direction=int(proposal.get("expected_direction", 1)),
    )


def generate_step_windows(
    start: date, end: date, *, window_months: int, step_months: int
) -> list[tuple[str, date, date]]:
    left = pd.Timestamp(start)
    boundary = pd.Timestamp(end)
    windows: list[tuple[str, date, date]] = []
    while left <= boundary:
        right = min(left + pd.DateOffset(months=window_months) - pd.Timedelta(days=1), boundary)
        if (right - left).days >= 365:
            windows.append(
                (
                    f"W{len(windows) + 1:02d}",
                    left.date(),
                    right.date(),
                )
            )
        if right >= boundary:
            break
        left += pd.DateOffset(months=step_months)
    final_start = boundary - pd.DateOffset(months=window_months) + pd.Timedelta(days=1)
    if windows and windows[-1][2] != end and final_start.date() > windows[-1][1]:
        windows.append((f"W{len(windows) + 1:02d}", final_start.date(), end))
    return windows


def summarize_path(path: pd.DataFrame) -> dict[str, Any]:
    if path.empty:
        raise ValueError("Cannot summarize an empty vector path")
    net = path["net"].dropna()
    wealth = (1.0 + net).cumprod()
    bankrupt = bool((net <= -1.0).any() or wealth.iloc[-1] <= 0)
    drawdown = (wealth / wealth.cummax() - 1.0).clip(lower=-1.0)
    compound_return = -1.0 if bankrupt else float(wealth.iloc[-1] ** (252 / len(net)) - 1.0)
    return {
        "simple_annual_return": float(net.mean() * 252),
        "compound_annual_return": compound_return,
        "total_return": -1.0 if bankrupt else float(wealth.iloc[-1] - 1.0),
        "sharpe_ratio": _annualized_ratio(net),
        "sortino_ratio": _sortino(net),
        "max_drawdown": float(drawdown.min()),
        "calmar_ratio": -1.0 if bankrupt else _calmar(net),
        "annual_volatility": float(net.std(ddof=1) * math.sqrt(252)),
        "annual_turnover": float(path["turnover"].mean() * 252),
        "cost_stress_sharpe": _annualized_ratio(path["stressed"]),
        "observations": len(net),
        "backtest_start": net.index.min().date().isoformat(),
        "backtest_end": net.index.max().date().isoformat(),
    }


def moving_block_monte_carlo(
    values: np.ndarray | list[float],
    *,
    samples: int,
    block_size: int,
    seed: int,
    chunk_size: int = 250,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if data.size < max(60, block_size):
        raise ValueError("Insufficient returns for moving-block Monte Carlo")
    rng = np.random.default_rng(seed)
    block_count = int(math.ceil(data.size / block_size))
    output = {
        "simple_annual_return": np.empty(samples, dtype=float),
        "sharpe_ratio": np.empty(samples, dtype=float),
        "total_return": np.empty(samples, dtype=float),
        "max_drawdown": np.empty(samples, dtype=float),
    }
    offsets = np.arange(block_size)
    cursor = 0
    while cursor < samples:
        size = min(chunk_size, samples - cursor)
        starts = rng.integers(0, data.size, size=(size, block_count))
        indices = (starts[..., None] + offsets) % data.size
        simulated = data[indices.reshape(size, -1)[:, : data.size]]
        means = simulated.mean(axis=1)
        standard_deviations = simulated.std(axis=1, ddof=1)
        wealth = np.cumprod(1.0 + simulated, axis=1)
        peaks = np.maximum.accumulate(wealth, axis=1)
        selected = slice(cursor, cursor + size)
        output["simple_annual_return"][selected] = means * 252
        output["sharpe_ratio"][selected] = np.divide(
            means * math.sqrt(252),
            standard_deviations,
            out=np.zeros(size),
            where=standard_deviations > 0,
        )
        output["total_return"][selected] = wealth[:, -1] - 1.0
        output["max_drawdown"][selected] = (wealth / peaks - 1.0).min(axis=1)
        cursor += size
    frame = pd.DataFrame(output)
    quantiles = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    summary = {
        "protocol": "CIRCULAR_MOVING_BLOCK_BOOTSTRAP_V1",
        "samples": samples,
        "block_size_sessions": block_size,
        "seed": seed,
        "quantiles": {
            column: {
                f"p{int(level * 100):02d}": float(frame[column].quantile(level))
                for level in quantiles
            }
            for column in frame.columns
        },
        "probability_positive_annual_return": float((frame["simple_annual_return"] > 0).mean()),
        "probability_positive_sharpe": float((frame["sharpe_ratio"] > 0).mean()),
        "probability_drawdown_below_minus_20pct": float((frame["max_drawdown"] < -0.20).mean()),
    }
    return frame, summary


def _scale_expression_parameters(expression: dict[str, Any], multiplier: float) -> dict[str, Any]:
    parameters = dict(expression.get("parameters", {}))
    for name in ("window", "periods"):
        value = parameters.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value > 1:
            parameters[name] = min(252, max(2, int(round(value * multiplier))))
    if "min_periods" in parameters and "window" in parameters:
        parameters["min_periods"] = min(int(parameters["min_periods"]), int(parameters["window"]))
    return {
        "operator": expression["operator"],
        "parameters": parameters,
        "arguments": [
            _scale_expression_parameters(argument, multiplier)
            for argument in expression.get("arguments", [])
        ],
    }


def _robustness_pass_fraction(results: list[dict[str, Any]], base_sharpe: float) -> float:
    valid = [item["metrics"] for item in results if item.get("metrics") is not None]
    if not valid:
        return 0.0
    floor = max(0.0, base_sharpe * 0.25)
    return float(np.mean([float(item["sharpe_ratio"]) >= floor for item in valid]))


def _annualized_ratio(values: pd.Series) -> float:
    standard_deviation = float(values.std(ddof=1))
    return float(values.mean() / standard_deviation * math.sqrt(252)) if standard_deviation else 0.0


def _sortino(values: pd.Series) -> float:
    downside = values[values < 0].std(ddof=1)
    return (
        float(values.mean() / downside * math.sqrt(252))
        if downside and np.isfinite(downside)
        else 0.0
    )


def _calmar(values: pd.Series) -> float:
    wealth = (1.0 + values).cumprod()
    drawdown = float((wealth / wealth.cummax() - 1.0).min())
    annual = float(wealth.iloc[-1] ** (252 / len(values)) - 1.0)
    return float(annual / abs(drawdown)) if drawdown else 0.0


def _safe_mean(values: pd.Series) -> float | None:
    return float(values.mean()) if not values.empty else None


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, compression="zstd")
    temporary.replace(path)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value
