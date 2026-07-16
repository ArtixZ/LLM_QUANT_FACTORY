from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from autoalpha.backtest.costs import ChinaAExecutionCosts
from autoalpha.backtest.ledger import LedgerBacktester, LedgerConfig, RebalanceSchedule
from autoalpha.backtest.presets import validate_preset_settings
from autoalpha.backtest.timing import EOD_NEXT_OPEN_RETURN_CONVENTION, entry_aligned_open_return
from autoalpha.backtest.vector import VectorBacktestConfig, VectorBacktester, select_positions
from autoalpha.config import ResearchConfig
from autoalpha.data.execution_basis import inspect_execution_data_basis
from autoalpha.data.workspace import inspect_data_workspace
from autoalpha.dsl.compiler import FactorCompiler
from autoalpha.dsl.expression import Expression, FactorDefinition
from autoalpha.dsl.semantics import FieldDefinition, SemanticValidator
from autoalpha.portfolio.products import product_template
from autoalpha.research.evaluation import cross_sectional_ic
from autoalpha.research.multiple_testing import deflated_sharpe_ratio
from autoalpha.research.statistics import hac_mean_inference
from autoalpha.service.index_enhancement import index_enhancement_diagnostic


@dataclass(frozen=True)
class ManualBacktestSpec:
    start_date: date
    end_date: date
    initial_cash_cny: float
    gross_exposure: float
    holding_period_days: int
    backtest_preset: str = "CUSTOM"
    backtest_engine: Literal["VECTOR", "EVENT_LEDGER"] = "VECTOR"
    execution_data_mode: Literal["STRICT_PIT", "NON_PIT_PROXY"] = "STRICT_PIT"
    rebalance_schedule: RebalanceSchedule = "DAILY_ROLLING"
    vector_cost_model: Literal["side_aware", "legacy_half_turnover"] = "side_aware"
    product_template: str = "MARKET_NEUTRAL_RESEARCH"
    selection_fraction: float = 0.10
    maximum_positions: int = 30
    lot_size: int = 100
    maximum_volume_participation: float = 0.05
    opening_limit_threshold: float = 0.095
    commission_bps_each_side: float = 1.5
    stamp_duty_bps_sell: float = 5.0
    transfer_fee_bps_each_side: float = 0.1
    minimum_commission_cny: float = 5.0
    slippage_bps_each_side: float = 0.0
    use_historical_fee_schedule: bool = False
    cost_stress_multiplier: float = 2.0

    def __post_init__(self) -> None:
        if not 0 < self.selection_fraction <= 0.50:
            raise ValueError("selection_fraction must be in (0, 0.5]")
        if self.maximum_positions <= 0 or self.lot_size <= 0:
            raise ValueError("maximum_positions and lot_size must be positive")
        if not 0 < self.maximum_volume_participation <= 1:
            raise ValueError("maximum_volume_participation must be in (0, 1]")
        if not 0 < self.opening_limit_threshold <= 0.30:
            raise ValueError("opening_limit_threshold must be in (0, 0.3]")
        if (
            min(
                self.commission_bps_each_side,
                self.stamp_duty_bps_sell,
                self.transfer_fee_bps_each_side,
                self.minimum_commission_cny,
                self.slippage_bps_each_side,
            )
            < 0
        ):
            raise ValueError("execution costs cannot be negative")
        if self.cost_stress_multiplier < 1:
            raise ValueError("cost_stress_multiplier must be at least one")
        if self.backtest_engine == "VECTOR" and self.rebalance_schedule != "DAILY_ROLLING":
            raise ValueError("Fixed-calendar rebalance schedules require the event ledger")
        if self.execution_data_mode == "NON_PIT_PROXY" and self.backtest_engine != "EVENT_LEDGER":
            raise ValueError("NON_PIT_PROXY execution requires the event ledger")
        validate_preset_settings(self.backtest_preset, asdict(self))


class ManualFactorBacktester:
    """Ad-hoc factor backtests isolated from automated research governance."""

    def __init__(self, data_path: Path, config_path: Path) -> None:
        self.data_path = data_path
        self.workspace = inspect_data_workspace(data_path)
        self.workspace.require_price_research()
        self.panel_path = Path(self.workspace.panel_path)
        self.config = ResearchConfig.from_toml(config_path)
        basis = inspect_execution_data_basis(self.panel_path)
        self.execution_basis = basis
        fields = [
            FieldDefinition("open", "price"),
            FieldDefinition("close", "price"),
            FieldDefinition("adj_close", "price"),
            FieldDefinition("amount", basis.amount_unit),
            FieldDefinition("vol", basis.volume_unit),
        ]
        self.validator = SemanticValidator(fields, maximum_nodes=30, maximum_lookback=252)
        self.compiler = FactorCompiler(self.validator)

    def run(
        self,
        factors: list[FactorDefinition],
        weights: list[float],
        spec: ManualBacktestSpec,
        *,
        multiple_testing_trials: int,
    ) -> dict[str, Any]:
        if not factors:
            raise ValueError("At least one factor is required")
        if len(factors) != len(weights):
            raise ValueError("Factor and weight counts differ")
        if not all(math.isfinite(weight) and weight > 0 for weight in weights):
            raise ValueError("Factor weights must be finite and positive")
        if spec.start_date >= spec.end_date:
            raise ValueError("Backtest start date must be before end date")
        available_start = date.fromisoformat(self.workspace.first_trade_date)
        available_end = date.fromisoformat(self.workspace.last_trade_date)
        if spec.start_date < available_start or spec.end_date > available_end:
            raise ValueError(
                f"Requested period must be within {available_start.isoformat()} and "
                f"{available_end.isoformat()}"
            )

        template = product_template(spec.product_template)
        use_ledger = spec.backtest_engine == "EVENT_LEDGER"
        if use_ledger and template.portfolio_mode != "long_only":
            raise ValueError("The event ledger supports long-only product templates")
        if use_ledger:
            execution_basis = inspect_execution_data_basis(self.panel_path)
            if spec.execution_data_mode == "STRICT_PIT":
                execution_basis.require_capital_ledger()
            else:
                execution_basis.require_capital_ledger_proxy()

        normalized_weights = np.asarray(weights, dtype=float)
        normalized_weights = normalized_weights / normalized_weights.sum()
        required_fields = {"open", "adj_close", "amount"}
        maximum_lookback = 1
        for factor in factors:
            semantics = self.validator.validate(factor.expression)
            maximum_lookback = max(maximum_lookback, semantics.lookback)
            required_fields.update(_expression_fields(factor.expression))
        fields, market_data = self._load_fields(required_fields, spec, maximum_lookback)

        signals = []
        for factor in factors:
            raw = self.compiler.evaluate(factor.expression, fields) * factor.expected_direction
            standard_deviation = raw.std(axis=1).replace(0, np.nan)
            signals.append(raw.sub(raw.mean(axis=1), axis=0).div(standard_deviation, axis=0))
        composite = signals[0] * normalized_weights[0]
        for signal, weight in zip(signals[1:], normalized_weights[1:], strict=True):
            composite = composite + signal * weight

        realized_return = entry_aligned_open_return(fields["open"])
        one_way_bps = (
            spec.commission_bps_each_side
            + spec.transfer_fee_bps_each_side
            + spec.stamp_duty_bps_sell / 2
            + spec.slippage_bps_each_side
        )
        vector_result = VectorBacktester(
            VectorBacktestConfig(
                holding_period_days=spec.holding_period_days,
                gross_exposure=spec.gross_exposure,
                selection_fraction=spec.selection_fraction,
                maximum_positions_per_side=spec.maximum_positions,
                long_only=template.portfolio_mode == "long_only",
                commission_bps_each_side=spec.commission_bps_each_side,
                stamp_duty_bps_sell=spec.stamp_duty_bps_sell,
                transfer_fee_bps_each_side=spec.transfer_fee_bps_each_side,
                slippage_bps_each_side=spec.slippage_bps_each_side,
                cost_stress_multiplier=spec.cost_stress_multiplier,
                cost_model=spec.vector_cost_model,
                path_index="entry_session",
                initial_cash_cny=spec.initial_cash_cny,
            )
        ).run(composite, fields["open"], start=spec.start_date, end=spec.end_date)
        positions = vector_result.positions
        path = vector_result.path.copy()
        ledger = None
        if use_ledger:
            ledger = LedgerBacktester(
                LedgerConfig(
                    horizon=spec.holding_period_days,
                    initial_cash=spec.initial_cash_cny,
                    top_fraction=spec.selection_fraction,
                    max_positions=spec.maximum_positions,
                    lot_size=spec.lot_size,
                    max_volume_participation=spec.maximum_volume_participation,
                    investment_buffer=1.0 - spec.gross_exposure,
                    trading_days_per_year=245,
                    rebalance_schedule=spec.rebalance_schedule,
                    slippage_bps_each_side=spec.slippage_bps_each_side,
                ),
                ChinaAExecutionCosts(
                    commission_bps_each_side=spec.commission_bps_each_side,
                    stamp_duty_bps_sell=spec.stamp_duty_bps_sell,
                    transfer_fee_bps_each_side=spec.transfer_fee_bps_each_side,
                    minimum_commission_cny=spec.minimum_commission_cny,
                    use_historical_fee_schedule=spec.use_historical_fee_schedule,
                ),
            ).run(composite, _ledger_market(market_data, spec))
            ledger_turnover = _ledger_turnover(ledger.trades, ledger.nav)
            ledger_returns = ledger.daily_return.reindex(path.index).fillna(0.0)
            path = pd.DataFrame(
                {
                    "gross": ledger_returns,
                    "net": ledger_returns,
                    "stressed": ledger_returns
                    - ledger_turnover.reindex(path.index).fillna(0.0)
                    * one_way_bps
                    * (spec.cost_stress_multiplier - 1.0)
                    / 10_000,
                    "turnover": ledger_turnover.reindex(path.index).fillna(0.0),
                },
                index=path.index,
            )
        if len(path) < 60:
            raise ValueError("Manual backtest requires at least 60 trading observations")

        selected_signal = composite.shift(1).reindex(path.index)
        selected_forward_return = realized_return.reindex(path.index)
        rank_ic = cross_sectional_ic(
            selected_signal,
            selected_forward_return,
            minimum_names=self.config.minimum_cross_section,
        )
        pearson_ic = cross_sectional_ic(
            selected_signal,
            selected_forward_return,
            method="pearson",
            minimum_names=self.config.minimum_cross_section,
        )
        benchmark_return = realized_return.reindex(path.index).mean(axis=1).fillna(0.0)
        if template.benchmark_mode == "cash":
            benchmark_return[:] = 0.0
        else:
            benchmark_return = benchmark_return * spec.gross_exposure
        net = path["net"]
        if template.hedge_benchmark:
            net = net - benchmark_return
        equity = spec.initial_cash_cny * (1.0 + net).cumprod()
        benchmark_equity = spec.initial_cash_cny * (1.0 + benchmark_return).cumprod()
        active_return = net - benchmark_return
        running_peak = equity.cummax().clip(lower=spec.initial_cash_cny)
        drawdown = equity / running_peak - 1.0
        dsr = deflated_sharpe_ratio(net.to_numpy(), trials=max(1, int(multiple_testing_trials)))
        inference = hac_mean_inference(net.to_numpy(), lags=min(5, len(net) - 1))
        selected_positions = positions.shift(1).reindex(path.index)
        selected_amount = fields["amount"].reindex(path.index).where(selected_positions.ne(0))
        capacity = float(
            selected_amount.median(axis=1).median() * 1000 * spec.maximum_volume_participation * 20
        )
        eligible = (
            fields["adj_close"].reindex(index=path.index, columns=selected_signal.columns).notna()
        )
        denominator = int(eligible.to_numpy().sum())
        coverage = float((selected_signal.notna() & eligible).to_numpy().sum() / denominator)
        factor_correlations = _factor_correlations(factors, signals, path.index)
        annual_returns = {
            str(year): float((1.0 + values).prod() - 1.0)
            for year, values in net.groupby(net.index.year)
        }
        tracking_error = float(active_return.std(ddof=1) * math.sqrt(245))
        metrics = {
            "simple_annual_return": float(net.mean() * 245),
            "compound_annual_return": _compound_annual_return(net),
            "total_return": float(equity.iloc[-1] / spec.initial_cash_cny - 1.0),
            "net_profit_cny": float(equity.iloc[-1] - spec.initial_cash_cny),
            "final_equity_cny": float(equity.iloc[-1]),
            "sharpe_ratio": _annualized_ratio(net),
            "sortino_ratio": _sortino_ratio(net),
            "annual_volatility": float(net.std(ddof=1) * math.sqrt(245)),
            "max_drawdown": float(drawdown.min()),
            "calmar_ratio": _calmar_ratio(net, drawdown),
            "daily_win_rate": float((net > 0).mean()),
            "annual_turnover": float(path["turnover"].mean() * 245),
            "cost_stress_sharpe": _annualized_ratio(path["stressed"]),
            "transaction_cost_cny": float(
                (
                    (path["gross"] - path["net"]) * equity.shift(1).fillna(spec.initial_cash_cny)
                ).sum()
            ),
            "coverage": coverage,
            "capacity_cny": capacity,
            "rank_ic_mean": _safe_mean(rank_ic),
            "rank_ic_ir": _annualized_ratio(rank_ic) if len(rank_ic) > 1 else None,
            "pearson_ic_mean": _safe_mean(pearson_ic),
            "pearson_ic_ir": _annualized_ratio(pearson_ic) if len(pearson_ic) > 1 else None,
            "net_return_hac_p_value": float(inference.p_value),
            "deflated_sharpe_probability": float(dsr.probability),
            "multiple_testing_trials": max(1, int(multiple_testing_trials)),
            "maximum_factor_correlation": max(
                (abs(value) for value in factor_correlations.values()), default=0.0
            ),
            "observations": len(path),
            "average_positions": float(selected_positions.ne(0).sum(axis=1).mean()),
            "backtest_start": path.index.min().date().isoformat(),
            "backtest_end": path.index.max().date().isoformat(),
            "holding_period_days": spec.holding_period_days,
            "signal_availability": "END_OF_DAY_AFTER_CLOSE",
            "execution_lag_sessions": 1,
            "return_convention": EOD_NEXT_OPEN_RETURN_CONVENTION,
            "gross_exposure": spec.gross_exposure,
            "selection_fraction": spec.selection_fraction,
            "maximum_positions_per_side": spec.maximum_positions,
            "portfolio_mode": template.portfolio_mode,
            "product_template": template.template_id,
            "product_template_name": template.name,
            "execution_mode": (
                "a_share_capital_ledger" if use_ledger else "research_vector"
            ),
            "backtest_engine": spec.backtest_engine,
            "backtest_preset": spec.backtest_preset,
            "execution_data_mode": spec.execution_data_mode,
            "rebalance_schedule": spec.rebalance_schedule,
            "production_eligible": (
                template.production_eligible
                and use_ledger
                and spec.execution_data_mode == "STRICT_PIT"
            ),
            "product_limitation": template.limitation,
            "benchmark_mode": template.benchmark_mode,
            "benchmark_simple_annual_return": float(benchmark_return.mean() * 245),
            "active_simple_annual_return": float(active_return.mean() * 245),
            "tracking_error": tracking_error,
            "tracking_error_limit": template.maximum_tracking_error,
            "tracking_error_gate_passed": (
                template.maximum_tracking_error is None
                or tracking_error <= template.maximum_tracking_error
            ),
            "information_ratio": (
                float(active_return.mean() * 245 / tracking_error) if tracking_error else 0.0
            ),
            "market_beta": _market_beta(net, benchmark_return),
            "vector_engine": "AUTOALPHA_VECTOR_V1",
            "vector_cost_model": spec.vector_cost_model,
        }
        if ledger is not None:
            metrics.update(
                {
                    "capital_ledger_protocol": "A_SHARE_NEXT_OPEN_INTEGER_LOT_V1",
                    "trade_count": len(ledger.trades),
                    "total_trade_notional_cny": (
                        float(ledger.trades["notional"].sum()) if not ledger.trades.empty else 0.0
                    ),
                    "total_fees_cny": ledger.total_fees,
                    "transaction_cost_cny": ledger.total_fees,
                    "average_gross_exposure": float(ledger.gross_exposure.mean()),
                    "final_position_count": len(ledger.final_positions),
                    "slippage_bps_each_side": spec.slippage_bps_each_side,
                    "historical_fee_schedule": spec.use_historical_fee_schedule,
                }
            )
        curve = [
            {
                "date": timestamp.date().isoformat(),
                "equity": round(float(equity.loc[timestamp]), 4),
                "drawdown": round(float(drawdown.loc[timestamp]), 8),
                "net_return": round(float(net.loc[timestamp]), 10),
                "benchmark_equity": round(float(benchmark_equity.loc[timestamp]), 4),
                "active_return": round(float(active_return.loc[timestamp]), 10),
            }
            for timestamp in path.index
        ]
        optimizer_diagnostic = None
        if template.maximum_tracking_error is not None:
            try:
                end = pd.Timestamp(spec.end_date)
                optimizer_diagnostic = index_enhancement_diagnostic(
                    composite.loc[:end],
                    fields["adj_close"].loc[:end],
                    fields["amount"].loc[:end],
                    template,
                    portfolio_value=spec.initial_cash_cny,
                )
            except Exception as error:
                optimizer_diagnostic = {
                    "success": False,
                    "used_fallback": True,
                    "message": f"{type(error).__name__}: {error}",
                }
        trade_rows = _trade_statement_rows(ledger.trades, market_data) if ledger is not None else []
        trade_statement = {
            "available": ledger is not None,
            "protocol": "SIMULATED_A_SHARE_TRADE_BLOTTER_V1",
            "disclaimer": "Simulated execution record; not a broker-issued contract note.",
            "row_count": len(trade_rows),
            "buy_count": sum(row["side"] == "BUY" for row in trade_rows),
            "sell_count": sum(row["side"] == "SELL" for row in trade_rows),
            "total_notional_cny": (
                float(ledger.trades["notional"].sum())
                if ledger is not None and not ledger.trades.empty
                else 0.0
            ),
            "total_fees_cny": ledger.total_fees if ledger is not None else 0.0,
        }
        execution_assumptions = _execution_assumptions(
            spec,
            portfolio_mode=template.portfolio_mode,
            use_ledger=use_ledger,
            price_adjustment=self.execution_basis.price_adjustment,
            execution_price_adjustment=self.execution_basis.execution_price_adjustment,
        )
        result = {
            "scope": "MANUAL_NON_GOVERNANCE",
            "warning": (
                "Manual results are human-visible ad-hoc research and never update the automated "
                "champion, LLM memory, direction campaigns, or holdout budgets."
            ),
            "product": template.to_dict(),
            "configuration": _spec_payload(spec),
            "factors": [
                {
                    "factor_id": factor.factor_id,
                    "name": factor.name,
                    "family": factor.family,
                    "weight": float(weight),
                }
                for factor, weight in zip(factors, normalized_weights, strict=True)
            ],
            "metrics": metrics,
            "equity_curve": curve,
            "annual_returns": annual_returns,
            "factor_correlations": factor_correlations,
            "execution_assumptions": execution_assumptions,
            "index_enhancement_diagnostic": optimizer_diagnostic,
            "trade_statement": trade_statement,
            "data_fingerprint": self.workspace.fingerprint,
        }
        result["configuration_hash"] = hashlib.sha256(
            json.dumps(result["configuration"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        result["result_hash"] = hashlib.sha256(
            json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        result["_trade_statement_rows"] = trade_rows
        return result

    def _load_fields(
        self,
        required_fields: set[str],
        spec: ManualBacktestSpec,
        maximum_lookback: int,
    ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
        warmup_days = max(400, maximum_lookback * 2 + spec.holding_period_days * 2)
        load_start = pd.Timestamp(spec.start_date) - pd.Timedelta(days=warmup_days)
        load_end = pd.Timestamp(spec.end_date) + pd.Timedelta(days=10)
        columns = list(
            dict.fromkeys(
                [
                    "trade_date",
                    "ts_code",
                    "name",
                    *sorted(required_fields),
                    "open",
                    "close",
                    "pre_close",
                    "vol",
                    "is_valid_ohlc",
                    "is_tradable_observation",
                    *(
                        [
                            "listing_date",
                            "delisting_date",
                            "is_st",
                            "is_suspended",
                            "limit_up",
                            "limit_down",
                            "can_buy_open",
                            "can_sell_open",
                        ]
                        if (
                            spec.backtest_engine == "EVENT_LEDGER"
                            and spec.execution_data_mode == "STRICT_PIT"
                        )
                        else []
                    ),
                    *(
                        [
                            "raw_open",
                            "raw_close",
                            "raw_pre_close",
                            "can_buy_open_proxy",
                            "can_sell_open_proxy",
                        ]
                        if (
                            spec.backtest_engine == "EVENT_LEDGER"
                            and spec.execution_data_mode == "NON_PIT_PROXY"
                        )
                        else []
                    ),
                ]
            )
        )
        frames = []
        for year in range(load_start.year, load_end.year + 1):
            for path in sorted((self.panel_path / f"trade_year={year}").glob("*.parquet")):
                frames.append(pd.read_parquet(path, columns=columns))
        if not frames:
            raise FileNotFoundError(f"No parquet partitions found under {self.panel_path}")
        data = pd.concat(frames, ignore_index=True)
        data["trade_date"] = pd.to_datetime(data["trade_date"])
        data = data[(data["trade_date"] >= load_start) & (data["trade_date"] <= load_end)]
        valid = data["is_valid_ohlc"].fillna(False) & data["is_tradable_observation"].fillna(False)
        data.loc[~valid, list(required_fields)] = np.nan
        fields = {
            name: data.pivot(index="trade_date", columns="ts_code", values=name).sort_index()
            for name in required_fields
        }
        return fields, data


def _ledger_market(data: pd.DataFrame, spec: ManualBacktestSpec) -> pd.DataFrame:
    selected = data[
        (data["trade_date"] >= pd.Timestamp(spec.start_date))
        & (data["trade_date"] <= pd.Timestamp(spec.end_date))
    ].copy()
    valid = selected["is_valid_ohlc"].fillna(False) & selected["is_tradable_observation"].fillna(
        False
    )
    if spec.execution_data_mode == "NON_PIT_PROXY":
        selected[["open", "close", "pre_close"]] = selected[
            ["raw_open", "raw_close", "raw_pre_close"]
        ].to_numpy()
        if {"can_buy_open_proxy", "can_sell_open_proxy"}.issubset(selected.columns):
            selected["can_buy_open"] = valid & selected["can_buy_open_proxy"].fillna(False)
            selected["can_sell_open"] = valid & selected["can_sell_open_proxy"].fillna(False)
        else:
            open_move = selected["open"] / selected["pre_close"] - 1.0
            selected["can_buy_open"] = valid & (open_move < spec.opening_limit_threshold)
            selected["can_sell_open"] = valid & (open_move > -spec.opening_limit_threshold)
    elif {"can_buy_open", "can_sell_open"}.issubset(selected.columns):
        selected["can_buy_open"] = valid & selected["can_buy_open"].fillna(False)
        selected["can_sell_open"] = valid & selected["can_sell_open"].fillna(False)
    else:
        open_move = selected["open"] / selected["pre_close"] - 1.0
        selected["can_buy_open"] = valid & (open_move < spec.opening_limit_threshold)
        selected["can_sell_open"] = valid & (open_move > -spec.opening_limit_threshold)
    selected.loc[~selected["is_valid_ohlc"].fillna(False), ["open", "close"]] = np.nan
    return selected.rename(columns={"trade_date": "date", "ts_code": "symbol", "vol": "volume"})[
        ["date", "symbol", "open", "close", "volume", "can_buy_open", "can_sell_open"]
    ]


def _execution_assumptions(
    spec: ManualBacktestSpec,
    *,
    portfolio_mode: str,
    use_ledger: bool,
    price_adjustment: str,
    execution_price_adjustment: str,
) -> dict[str, Any]:
    daily_rolling = spec.rebalance_schedule == "DAILY_ROLLING"
    if daily_rolling:
        exit_rule = (
            f"Each daily sleeve exits after {spec.holding_period_days} trading sessions; "
            "the aggregate portfolio is rebalanced every session"
        )
        rebalance_frequency = "EVERY_TRADING_SESSION"
        calendar_rule = "DAILY_ROLLING_SLEEVE"
        sleeve_fraction: float | None = 1.0 / spec.holding_period_days
    elif spec.rebalance_schedule == "WEEKLY_FIRST_SESSION":
        exit_rule = (
            "Hold until the next week's first trading session; rejected orders retry on each "
            "following session until completed or replaced by the next scheduled target"
        )
        rebalance_frequency = "FIRST_TRADING_SESSION_EACH_ISO_WEEK"
        calendar_rule = "HOLIDAY_AWARE_WEEKLY_FIRST_SESSION"
        sleeve_fraction = None
    else:
        exit_rule = (
            "Hold until the next month's first trading session; rejected orders retry on each "
            "following session until completed or replaced by the next scheduled target"
        )
        rebalance_frequency = "FIRST_TRADING_SESSION_EACH_MONTH"
        calendar_rule = "HOLIDAY_AWARE_MONTHLY_FIRST_SESSION"
        sleeve_fraction = None
    common = {
        "protocol": "AUTOALPHA_EXECUTION_ASSUMPTIONS_V2",
        "engine": spec.backtest_engine,
        "preset": spec.backtest_preset,
        "signal_available": "T session after close",
        "entry": "T+1 session official open (09:30 assumption)",
        "exit": exit_rule,
        "return_measurement": (
            "cash ledger marked at each session close after official-open executions"
            if use_ledger
            else "T+1 open to T+2 open for each signal-date return"
        ),
        "rebalance_frequency": rebalance_frequency,
        "rebalance_schedule": spec.rebalance_schedule,
        "calendar_rule": calendar_rule,
        "fixed_weekday": False,
        "holding_period_trading_sessions": spec.holding_period_days,
        "daily_new_sleeve_fraction": sleeve_fraction,
        "selection_fraction_each_side": spec.selection_fraction,
        "maximum_positions_each_side": spec.maximum_positions,
        "target_gross_exposure": spec.gross_exposure,
        "portfolio_mode": portfolio_mode,
        "price_adjustment": price_adjustment,
        "execution_price_adjustment": execution_price_adjustment,
        "execution_data_mode": spec.execution_data_mode,
        "slippage_bps_each_side": spec.slippage_bps_each_side,
        "historical_fee_schedule": spec.use_historical_fee_schedule,
    }
    if use_ledger:
        common.update(
            {
                "position_model": "integer-share rotating cash sleeves",
                "execution_price": (
                    "unadjusted official open"
                    if execution_price_adjustment in {"unadjusted", "raw"}
                    else f"{execution_price_adjustment} daily open panel"
                ),
                "constraints_modeled": [
                    "cash",
                    "board lot",
                    (
                        "point-in-time opening buy/sell eligibility"
                        if spec.execution_data_mode == "STRICT_PIT"
                        else "opening buy/sell eligibility proxy"
                    ),
                    "volume participation",
                    "minimum commission",
                    "T+1 sell sequencing",
                    "failed-order retry",
                    "fixed opening slippage",
                    "historical stamp-duty and transfer-fee schedule",
                ],
                "constraints_not_modeled": [
                    "intraday order-book queue",
                    "market impact beyond fixed opening slippage",
                    *(
                        [
                            "point-in-time ST, listing, delisting, and suspension state",
                            "exact board-specific daily price limits",
                        ]
                        if spec.execution_data_mode == "NON_PIT_PROXY"
                        else []
                    ),
                ],
            }
        )
    else:
        common.update(
            {
                "position_model": "equal-weight daily targets averaged across rolling sleeves",
                "execution_price": f"{price_adjustment} daily open panel",
                "cost_model": spec.vector_cost_model,
                "constraints_modeled": ["linear buy/sell fees", "target gross exposure"],
                "constraints_not_modeled": [
                    "integer lots and cash",
                    "opening auction slippage",
                    "suspension carry and forced delisting treatment",
                    "limit-up/limit-down fills",
                    "short borrow availability, borrow fees, and recalls",
                    "market impact and order-book capacity",
                    "minimum commission per order",
                ],
                "missing_price_treatment": (
                    "missing selected-name returns contribute zero without order-level carry"
                ),
            }
        )
    return common


def _ledger_turnover(trades: pd.DataFrame, nav: pd.Series) -> pd.Series:
    if trades.empty:
        return pd.Series(0.0, index=nav.index)
    notional = trades.groupby("date")["notional"].sum().reindex(nav.index).fillna(0.0)
    return notional.div(nav.shift(1).fillna(nav.iloc[0])).mul(0.5)


def _market_beta(portfolio_return: pd.Series, benchmark_return: pd.Series) -> float | None:
    variance = float(benchmark_return.var(ddof=1))
    if variance <= 0:
        return None
    return float(portfolio_return.cov(benchmark_return) / variance)


def _spec_payload(spec: ManualBacktestSpec) -> dict[str, Any]:
    payload = asdict(spec)
    payload["start_date"] = spec.start_date.isoformat()
    payload["end_date"] = spec.end_date.isoformat()
    return payload


def _trade_statement_rows(trades: pd.DataFrame, market_data: pd.DataFrame) -> list[dict[str, Any]]:
    if trades.empty:
        return []
    names = (
        market_data[["ts_code", "name"]]
        .dropna()
        .drop_duplicates("ts_code", keep="last")
        .set_index("ts_code")["name"]
        .astype(str)
        .to_dict()
    )
    rows = []
    for trade_id, trade in enumerate(trades.itertuples(index=False), start=1):
        rows.append(
            {
                "trade_id": trade_id,
                "trade_date": pd.Timestamp(trade.date).date().isoformat(),
                "signal_date": (
                    pd.Timestamp(trade.signal_date).date().isoformat()
                    if pd.notna(trade.signal_date)
                    else None
                ),
                "sleeve": int(trade.sleeve),
                "symbol": str(trade.symbol),
                "security_name": names.get(str(trade.symbol), ""),
                "side": str(trade.side),
                "quantity": int(trade.quantity),
                "price_cny": round(float(trade.price), 6),
                "notional_cny": round(float(trade.notional), 4),
                "commission_cny": round(float(trade.commission), 4),
                "transfer_fee_cny": round(float(trade.transfer_fee), 4),
                "stamp_duty_cny": round(float(trade.stamp_duty), 4),
                "total_fees_cny": round(float(trade.fees), 4),
                "net_cash_flow_cny": round(float(trade.net_cash_flow), 4),
                "sleeve_cash_after_cny": round(float(trade.cash_after), 4),
            }
        )
    return rows


def _expression_fields(expression: Expression) -> set[str]:
    names = {str(expression.parameter("name"))} if expression.operator == "field" else set()
    for argument in expression.arguments:
        names.update(_expression_fields(argument))
    return names


def _select_positions(
    composite: pd.DataFrame,
    *,
    selection_fraction: float,
    maximum_positions: int,
    long_only: bool,
) -> pd.DataFrame:
    return select_positions(
        composite,
        selection_fraction=selection_fraction,
        maximum_positions_per_side=maximum_positions,
        long_only=long_only,
        method="percentile_with_ordinal_cap",
    )


def _factor_correlations(
    factors: list[FactorDefinition], signals: list[pd.DataFrame], dates: pd.Index
) -> dict[str, float]:
    correlations = {}
    for left_index, left in enumerate(factors):
        for right_index in range(left_index + 1, len(factors)):
            daily = (
                signals[left_index]
                .reindex(dates)
                .corrwith(signals[right_index].reindex(dates), axis=1)
                .dropna()
            )
            correlations[f"{left.factor_id}:{factors[right_index].factor_id}"] = (
                float(daily.median()) if not daily.empty else 1.0
            )
    return correlations


def _annualized_ratio(values: pd.Series) -> float:
    standard_deviation = float(values.std(ddof=1))
    return float(values.mean() / standard_deviation * math.sqrt(245)) if standard_deviation else 0.0


def _compound_annual_return(values: pd.Series) -> float:
    total = float((1.0 + values).prod())
    return float(total ** (245 / len(values)) - 1.0)


def _sortino_ratio(values: pd.Series) -> float:
    downside = values[values < 0]
    deviation = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    return float(values.mean() / deviation * math.sqrt(245)) if deviation else 0.0


def _calmar_ratio(values: pd.Series, drawdown: pd.Series) -> float:
    maximum_drawdown = abs(float(drawdown.min()))
    return _compound_annual_return(values) / maximum_drawdown if maximum_drawdown else 0.0


def _safe_mean(values: pd.Series) -> float | None:
    return float(values.mean()) if len(values) else None
