from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from autoalpha.backtest.ashare_vector import AshareVectorBacktester, AshareVectorConfig
from autoalpha.dsl.compiler import FactorCompiler
from autoalpha.dsl.expression import Expression, FactorDefinition
from autoalpha.dsl.semantics import SemanticValidator
from autoalpha.service.batch_engine import (
    MassiveBatchConfig,
    MassiveVectorBatchEngine,
    _scale_expression_parameters,
    summarize_path,
)


@dataclass(frozen=True)
class RealisticAshareBatchConfig(MassiveBatchConfig):
    gross_exposure: float = 0.90
    commission_bps_each_side: float = 2.5
    slippage_bps_each_side: float = 5.0
    protocol: str = "A_SHARE_LONG_ONLY_WEEKLY_VECTOR_PROXY_V1"
    initial_cash_cny: float = 1_000_000.0
    minimum_commission_cny: float = 5.0
    use_historical_fee_schedule: bool = True
    rebalance_schedule: str = "WEEKLY_FIRST_SESSION"
    execution_data_mode: str = "NON_PIT_PROXY"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RealisticAshareBatchConfig:
        base = MassiveBatchConfig.from_dict(value)
        return cls(
            **base.__dict__,
            protocol=str(value.get("protocol", "A_SHARE_LONG_ONLY_WEEKLY_VECTOR_PROXY_V1")),
            initial_cash_cny=float(value.get("initial_cash_cny", 1_000_000.0)),
            minimum_commission_cny=float(value.get("minimum_commission_cny", 5.0)),
            use_historical_fee_schedule=bool(value.get("use_historical_fee_schedule", True)),
            rebalance_schedule=str(value.get("rebalance_schedule", "WEEKLY_FIRST_SESSION")),
            execution_data_mode=str(value.get("execution_data_mode", "NON_PIT_PROXY")),
        )


class RealisticAshareBatchEngine(MassiveVectorBatchEngine):
    """Massive factor evaluator using the long-only weekly A-share vector proxy."""

    config: RealisticAshareBatchConfig

    def __init__(
        self,
        config: RealisticAshareBatchConfig,
        artifact_root: Path,
        *,
        factor_family_size: int,
    ) -> None:
        super().__init__(config, artifact_root, factor_family_size=factor_family_size)
        self.execution_basis.require_capital_ledger_proxy()

    def _run_vector(self, signal: pd.DataFrame, *, holding_period: int):
        del holding_period
        return self._run_protocol(signal)

    def _run_protocol(
        self,
        signal: pd.DataFrame,
        *,
        schedule: str | None = None,
        maximum_positions: int | None = None,
    ):
        return AshareVectorBacktester(
            AshareVectorConfig(
                initial_cash_cny=self.config.initial_cash_cny,
                gross_exposure=self.config.gross_exposure,
                selection_fraction=self.config.selection_fraction,
                maximum_positions=maximum_positions or self.config.maximum_positions_per_side,
                rebalance_schedule=schedule or self.config.rebalance_schedule,  # type: ignore[arg-type]
                commission_bps_each_side=self.config.commission_bps_each_side,
                stamp_duty_bps_sell=self.config.stamp_duty_bps_sell,
                transfer_fee_bps_each_side=self.config.transfer_fee_bps_each_side,
                minimum_commission_cny=self.config.minimum_commission_cny,
                slippage_bps_each_side=self.config.slippage_bps_each_side,
                use_historical_fee_schedule=self.config.use_historical_fee_schedule,
                cost_stress_multiplier=self.config.cost_stress_multiplier,
            )
        ).run(
            signal,
            self.fields["open"],
            self.fields["raw_open"],
            self.fields["can_buy_open_proxy"],
            self.fields["can_sell_open_proxy"],
            start=self.config.start_date,
            end=self.config.end_date,
        )

    def _enriched_metrics(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        metrics = super()._enriched_metrics(*args, **kwargs)
        metrics.update(
            {
                "engine_protocol": self.config.protocol,
                "portfolio_mode": "long_only",
                "rebalance_schedule": self.config.rebalance_schedule,
                "maximum_positions": self.config.maximum_positions_per_side,
                "target_gross_exposure": self.config.gross_exposure,
                "execution_data_mode": self.config.execution_data_mode,
                "production_eligible": False,
                "production_blockers": [
                    "non-PIT historical ST, listing, delisting and suspension state",
                    "opening eligibility uses a main-board price-limit proxy",
                    "vector weights approximate board lots and cash",
                    "short selling is disabled",
                ],
                "result_scope": "HUMAN_VISIBLE_ASHARE_EXECUTION_PROXY_DIAGNOSTIC",
            }
        )
        return metrics

    def _robustness_tests(
        self, factor: FactorDefinition, base_signal: pd.DataFrame
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        variants: list[tuple[str, str, dict[str, Any]]] = [
            ("REBALANCE_SCHEDULE", "BIWEEKLY", {"schedule": "BIWEEKLY_FIRST_SESSION"}),
            ("REBALANCE_SCHEDULE", "MONTHLY", {"schedule": "MONTHLY_FIRST_SESSION"}),
            ("POSITION_COUNT", "20", {"maximum_positions": 20}),
            ("POSITION_COUNT", "50", {"maximum_positions": 50}),
        ]
        for test_type, variant, settings in variants:
            try:
                outcome = self._run_protocol(base_signal, **settings)
                metrics = summarize_path(outcome.path)
                error = None
            except Exception as exception:  # noqa: BLE001
                metrics = None
                error = f"{type(exception).__name__}: {exception}"
            results.append(
                {
                    "test_type": test_type,
                    "variant": variant,
                    "metrics": metrics,
                    "error": error,
                }
            )

        stressed_path = self._run_protocol(base_signal).path.copy()
        stressed_path["net"] = stressed_path["stressed"]
        results.append(
            {
                "test_type": "EXECUTION_COST",
                "variant": f"x{self.config.cost_stress_multiplier:g}",
                "metrics": summarize_path(stressed_path),
                "error": None,
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
                expression = Expression.from_dict(scaled)
                validator = SemanticValidator(
                    self.validator_fields, maximum_nodes=30, maximum_lookback=252
                )
                scaled_signal = FactorCompiler(validator).evaluate(expression, self.fields)
                scaled_signal *= factor.expected_direction
                outcome = self._run_protocol(scaled_signal)
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
        del stressed_path
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
                    "ts_code",
                    "open",
                    *factor_columns,
                    "raw_open",
                    "is_valid_ohlc",
                    "is_tradable_observation",
                    "can_buy_open_proxy",
                    "can_sell_open_proxy",
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
        value_columns = list(dict.fromkeys(["open", *factor_columns, "raw_open"]))
        data.loc[~valid, value_columns] = np.nan
        fields = {
            name: data.pivot(index="trade_date", columns="ts_code", values=name).sort_index()
            for name in value_columns
        }
        for name in ("can_buy_open_proxy", "can_sell_open_proxy"):
            fields[name] = (
                data.pivot(index="trade_date", columns="ts_code", values=name)
                .sort_index()
                .fillna(False)
                .astype(bool)
            )
        del data, frames
        gc.collect()
        return fields
