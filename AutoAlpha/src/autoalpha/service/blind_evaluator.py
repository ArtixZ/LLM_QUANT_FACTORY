from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from autoalpha.backtest.ashare_vector import AshareVectorBacktester, AshareVectorConfig
from autoalpha.backtest.capital import CapitalBacktestSpec, run_capital_backtest
from autoalpha.backtest.costs import ChinaAExecutionCosts
from autoalpha.config import ResearchConfig
from autoalpha.data.execution_basis import inspect_execution_data_basis
from autoalpha.data.research_fields import field_definitions
from autoalpha.data.workspace import inspect_data_workspace
from autoalpha.dsl.compiler import FactorCompiler
from autoalpha.dsl.expression import FactorDefinition, constant, operation
from autoalpha.dsl.semantics import SemanticValidator


@dataclass(frozen=True)
class BlindVerdict:
    passed: bool
    verdict: str
    evidence_hash: str


class BlindEvaluationBoundary:
    """Trusted boundary that never returns exact holdout or capital metrics."""

    def __init__(self, panel_path: Path, config: ResearchConfig) -> None:
        self.panel_path = panel_path
        self.config = config
        try:
            self.workspace = inspect_data_workspace(panel_path)
            self.factor_fields = self.workspace.factor_fields
        except FileNotFoundError:
            # Unit-level trusted-boundary tests may inject fields without a disk panel.
            self.workspace = None
            self.factor_fields = ("close", "adj_close", "amount", "vol")
        basis = inspect_execution_data_basis(panel_path)
        self.validator = SemanticValidator(
            field_definitions(
                self.factor_fields,
                amount_unit=basis.amount_unit,
                volume_unit=basis.volume_unit,
            ),
            maximum_nodes=30,
            maximum_lookback=252,
        )
        self.compiler = FactorCompiler(self.validator)

    def evaluate_holdout(
        self,
        factors: list[FactorDefinition],
        weights: tuple[float, ...],
        *,
        benchmark_factors: list[FactorDefinition] | None = None,
        benchmark_weights: tuple[float, ...] | None = None,
    ) -> BlindVerdict:
        fields = self._load_holdout_fields()
        treatment = self._portfolio_path(factors, weights, fields)
        benchmark = (
            self._portfolio_path(benchmark_factors, benchmark_weights or (), fields)
            if benchmark_factors
            else None
        )
        metrics = _path_metrics(treatment)
        if benchmark is not None:
            control = _path_metrics(benchmark)
            metrics["sharpe_change"] = metrics["sharpe"] - control["sharpe"]
            metrics["annual_return_change"] = metrics["annual_return"] - control["annual_return"]
        else:
            metrics["sharpe_change"] = metrics["sharpe"]
            metrics["annual_return_change"] = metrics["annual_return"]

        policy = self.config.governance
        if metrics["cost_stress_sharpe"] < 0:
            verdict = "BLIND_COST_FAILURE"
        elif metrics["max_drawdown"] < -policy.holdout_maximum_drawdown:
            verdict = "BLIND_RISK_FAILURE"
        elif (
            metrics["sharpe"] < policy.holdout_minimum_sharpe
            or metrics["annual_return"] < policy.holdout_minimum_annual_return
            or metrics["sharpe_change"] < -0.25
            or metrics["annual_return_change"] < -0.03
        ):
            verdict = "BLIND_NO_GENERALIZATION"
        else:
            verdict = "BLIND_GENERALIZATION_PASSED"
        return BlindVerdict(
            passed=verdict == "BLIND_GENERALIZATION_PASSED",
            verdict=verdict,
            evidence_hash=_evidence_hash(metrics),
        )

    def evaluate_capital(
        self,
        factors: list[FactorDefinition],
        weights: tuple[float, ...],
    ) -> BlindVerdict:
        basis = inspect_execution_data_basis(self.panel_path)
        if not basis.capital_ledger_ready:
            return BlindVerdict(
                passed=False,
                verdict="CAPITAL_DATA_BASIS_FAILURE",
                evidence_hash=_evidence_hash(basis.to_dict()),
            )
        composite = _composite_factor(factors, weights)
        portfolio = self.config.portfolio
        split = self.config.splits
        spec = CapitalBacktestSpec(
            start=split.train.start,
            end=split.test.end,
            initial_cash=portfolio.initial_cash_cny,
            target_gross_exposure=portfolio.target_gross_exposure,
            top_fraction=portfolio.top_fraction,
            max_positions=portfolio.maximum_positions,
            max_volume_participation=self.config.costs.max_adv_participation,
            holding_period_days=portfolio.holding_period_days,
        )
        report = run_capital_backtest(
            composite,
            self.panel_path,
            spec,
            costs=ChinaAExecutionCosts(
                commission_bps_each_side=self.config.costs.commission_bps_each_side,
                stamp_duty_bps_sell=self.config.costs.stamp_duty_bps_sell,
                transfer_fee_bps_each_side=self.config.costs.transfer_fee_bps_each_side,
                minimum_commission_cny=self.config.costs.minimum_commission_cny,
            ),
        )
        metrics = report.metrics
        residual_limit = int(
            math.ceil(portfolio.maximum_positions * portfolio.maximum_residual_position_multiplier)
        )
        if float(metrics["sharpe_ratio"]) < portfolio.minimum_capital_sharpe:
            verdict = "CAPITAL_RISK_ADJUSTED_RETURN_FAILURE"
        elif float(metrics["max_drawdown"]) < -portfolio.maximum_capital_drawdown:
            verdict = "CAPITAL_DRAWDOWN_FAILURE"
        elif float(metrics["maximum_gross_exposure"]) > portfolio.maximum_realized_gross_exposure:
            verdict = "CAPITAL_EXPOSURE_FAILURE"
        elif int(metrics["final_position_count"]) > residual_limit:
            verdict = "CAPITAL_RESIDUAL_POSITION_FAILURE"
        else:
            verdict = "CAPITAL_SIMULATION_PASSED"
        return BlindVerdict(
            passed=verdict == "CAPITAL_SIMULATION_PASSED",
            verdict=verdict,
            evidence_hash=_evidence_hash(metrics),
        )

    def _load_holdout_fields(self) -> dict[str, pd.DataFrame]:
        split = self.config.splits.test
        warmup = pd.Timestamp(split.start) - pd.Timedelta(days=400)
        factor_columns = list(self.factor_fields)
        columns = list(dict.fromkeys([
            "trade_date",
            "ts_code",
            "open",
            "raw_open",
            "can_buy_open_proxy",
            "can_sell_open_proxy",
            *factor_columns,
            "is_valid_ohlc",
            "is_tradable_observation",
        ]))
        frames = []
        for year in range(warmup.year, split.end.year + 1):
            for path in sorted((self.panel_path / f"trade_year={year}").glob("*.parquet")):
                frames.append(pd.read_parquet(path, columns=columns))
        if not frames:
            raise FileNotFoundError(f"No holdout panel partitions under {self.panel_path}")
        data = pd.concat(frames, ignore_index=True)
        data["trade_date"] = pd.to_datetime(data["trade_date"])
        data = data[
            (data["trade_date"] >= warmup) & (data["trade_date"] <= pd.Timestamp(split.end))
        ]
        valid = data["is_valid_ohlc"].fillna(False) & data["is_tradable_observation"].fillna(False)
        value_columns = list(dict.fromkeys(["open", "raw_open", *factor_columns]))
        data.loc[~valid, value_columns] = np.nan
        result = {
            name: data.pivot(index="trade_date", columns="ts_code", values=name).sort_index()
            for name in value_columns
        }
        result["can_buy_open_proxy"] = data.pivot(
            index="trade_date", columns="ts_code", values="can_buy_open_proxy"
        ).sort_index()
        result["can_sell_open_proxy"] = data.pivot(
            index="trade_date", columns="ts_code", values="can_sell_open_proxy"
        ).sort_index()
        return result

    def _portfolio_path(
        self,
        factors: list[FactorDefinition],
        weights: tuple[float, ...],
        fields: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        normalized = _normalize_weights(weights, len(factors))
        signals = []
        for factor in factors:
            self.validator.validate(factor.expression)
            raw = self.compiler.evaluate(factor.expression, fields) * factor.expected_direction
            scale = raw.std(axis=1).replace(0, np.nan)
            signals.append(raw.sub(raw.mean(axis=1), axis=0).div(scale, axis=0))
        composite = signals[0] * normalized[0]
        for signal, weight in zip(signals[1:], normalized[1:], strict=True):
            composite = composite + signal * weight

        split = self.config.splits.test
        strategy = self.config.strategy_evaluation
        adjusted_open = fields["open"]
        raw_open = fields.get("raw_open", adjusted_open)
        can_buy = fields.get(
            "can_buy_open_proxy",
            pd.DataFrame(True, index=adjusted_open.index, columns=adjusted_open.columns),
        )
        can_sell = fields.get(
            "can_sell_open_proxy",
            pd.DataFrame(True, index=adjusted_open.index, columns=adjusted_open.columns),
        )
        result = AshareVectorBacktester(
            AshareVectorConfig(
                initial_cash_cny=strategy.initial_cash_cny,
                gross_exposure=strategy.gross_exposure,
                selection_fraction=strategy.selection_fraction,
                maximum_positions=strategy.maximum_positions,
                rebalance_schedule=strategy.rebalance_schedule,
                commission_bps_each_side=strategy.commission_bps_each_side,
                stamp_duty_bps_sell=strategy.stamp_duty_bps_sell,
                transfer_fee_bps_each_side=strategy.transfer_fee_bps_each_side,
                minimum_commission_cny=strategy.minimum_commission_cny,
                slippage_bps_each_side=strategy.slippage_bps_each_side,
                use_historical_fee_schedule=strategy.use_historical_fee_schedule,
                cost_stress_multiplier=strategy.cost_stress_multiplier,
            )
        ).run(
            composite,
            adjusted_open,
            raw_open,
            can_buy,
            can_sell,
            start=split.start,
            end=split.end,
        )
        return result.path[["net", "stressed", "turnover"]]


def _path_metrics(path: pd.DataFrame) -> dict[str, float]:
    net = path["net"]
    stressed = path["stressed"]
    wealth = (1 + net).cumprod()
    return {
        "sharpe": _annualized_ir(net),
        "annual_return": float(net.mean() * 245),
        "max_drawdown": float((wealth / wealth.cummax() - 1).min()),
        "cost_stress_sharpe": _annualized_ir(stressed),
        "annual_turnover": float(path["turnover"].mean() * 245),
        "observations": float(len(path)),
    }


def _composite_factor(
    factors: list[FactorDefinition], weights: tuple[float, ...]
) -> FactorDefinition:
    normalized = _normalize_weights(weights, len(factors))
    components = []
    for factor, weight in zip(factors, normalized, strict=True):
        directed = (
            operation("negate", factor.expression)
            if factor.expected_direction == -1
            else factor.expression
        )
        components.append(operation("multiply", constant(weight), directed))
    expression = components[0]
    for component in components[1:]:
        expression = operation("add", expression, component)
    return FactorDefinition(
        name="FrozenBlindPortfolioComposite",
        family="Governed Multi-Factor Portfolio",
        hypothesis="Frozen portfolio evaluated inside the blind boundary.",
        expression=expression,
        expected_direction=1,
    )


def _normalize_weights(weights: tuple[float, ...], count: int) -> tuple[float, ...]:
    if len(weights) != count or count == 0 or any(weight < 0 for weight in weights):
        raise ValueError("Blind portfolio weights are invalid")
    total = sum(weights)
    if total <= 0:
        raise ValueError("Blind portfolio weights have no positive mass")
    return tuple(float(weight / total) for weight in weights)


def _annualized_ir(values: pd.Series) -> float:
    clean = values.dropna()
    volatility = float(clean.std(ddof=1))
    return float(clean.mean() / volatility * math.sqrt(245)) if volatility > 0 else 0.0


def _evidence_hash(metrics: dict[str, Any]) -> str:
    payload = json.dumps(metrics, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()
