from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from autoalpha.data.execution_basis import inspect_execution_data_basis
from autoalpha.data.research_fields import field_definitions
from autoalpha.data.workspace import inspect_data_workspace
from autoalpha.dsl.compiler import FactorCompiler
from autoalpha.dsl.expression import FactorDefinition
from autoalpha.dsl.semantics import SemanticValidator


@dataclass(frozen=True)
class ScreenerSpec:
    as_of_date: date
    selection_count: int = 30
    selection_side: Literal["TOP", "BOTTOM"] = "TOP"


class CrossSectionalScreener:
    """Fast, end-of-day factor score snapshot isolated from research governance."""

    def __init__(self, data_path: Path) -> None:
        self.workspace = inspect_data_workspace(data_path)
        self.workspace.require_price_research()
        self.panel_path = Path(self.workspace.panel_path)
        basis = inspect_execution_data_basis(self.panel_path)
        fields = field_definitions(
            self.workspace.factor_fields,
            amount_unit=basis.amount_unit,
            volume_unit=basis.volume_unit,
        )
        self.validator = SemanticValidator(fields, maximum_nodes=30, maximum_lookback=252)
        self.compiler = FactorCompiler(self.validator)

    def screen(
        self,
        factors: list[FactorDefinition],
        weights: list[float],
        spec: ScreenerSpec,
    ) -> dict[str, Any]:
        if not factors or len(factors) != len(weights):
            raise ValueError("Factors and weights must be non-empty and aligned")
        if spec.selection_count < 1:
            raise ValueError("selection_count must be positive")
        if any(not np.isfinite(weight) or weight <= 0 for weight in weights):
            raise ValueError("Factor weights must be finite and positive")

        fields, snapshot, resolved_date = self._load_snapshot(spec.as_of_date)
        signals: list[pd.Series] = []
        evaluated: list[dict[str, Any]] = []
        skipped: list[dict[str, str]] = []
        for factor, weight in zip(factors, weights, strict=True):
            try:
                self.validator.validate(factor.expression)
                raw = self.compiler.evaluate(factor.expression, fields) * factor.expected_direction
                value = raw.loc[resolved_date]
                standard_deviation = float(value.std())
                if not np.isfinite(standard_deviation) or standard_deviation == 0:
                    raise ValueError("cross-sectional standard deviation is zero")
                signal = (value - value.mean()) / standard_deviation
                signals.append(signal)
                evaluated.append(
                    {
                        "factor_id": factor.factor_id,
                        "name": factor.name,
                        "family": factor.family,
                        "weight": float(weight),
                        "coverage": int(signal.notna().sum()),
                    }
                )
            except (TypeError, ValueError, KeyError) as error:
                skipped.append(
                    {
                        "factor_id": factor.factor_id,
                        "name": factor.name,
                        "reason": f"{type(error).__name__}: {error}",
                    }
                )
        if not signals:
            raise ValueError("No selected factors produced a usable signal")

        normalized = np.asarray([item["weight"] for item in evaluated], dtype=float)
        normalized = normalized / normalized.sum()
        composite = signals[0] * normalized[0]
        for signal, weight in zip(signals[1:], normalized[1:], strict=True):
            composite = composite + signal * weight
        snapshot = snapshot.reindex(composite.index)
        candidates = snapshot.assign(
            composite_score=composite,
            score_percentile=composite.rank(pct=True),
        ).dropna(subset=["composite_score"])
        candidates = candidates.sort_values(
            "composite_score", ascending=spec.selection_side == "BOTTOM"
        )
        candidates = candidates.head(spec.selection_count).copy()
        rows = []
        for rank, (symbol, row) in enumerate(candidates.iterrows(), start=1):
            factor_scores = {
                item["factor_id"]: _finite_float(signal.get(symbol))
                for item, signal in zip(evaluated, signals, strict=True)
            }
            rows.append(
                {
                    "rank": rank,
                    "ts_code": str(symbol),
                    "name": str(row["name"]),
                    "composite_score": _finite_float(row["composite_score"]),
                    "score_percentile": _finite_float(row["score_percentile"]),
                    "research_close": _finite_float(row["close"]),
                    "raw_open": _finite_float(row.get("raw_open")),
                    "raw_close": _finite_float(row["raw_close"]),
                    "amount_cny": _finite_float(row["amount"]),
                    "volume_shares": _finite_float(row["vol"]),
                    "factor_scores": factor_scores,
                }
            )
        for item, weight in zip(evaluated, normalized, strict=True):
            item["normalized_weight"] = float(weight)
        return {
            "scope": "EOD_FACTOR_SCREENING_NON_GOVERNANCE",
            "requested_as_of_date": spec.as_of_date.isoformat(),
            "as_of_date": resolved_date.date().isoformat(),
            "selection_side": spec.selection_side,
            "selection_count": spec.selection_count,
            "universe_size": int(composite.notna().sum()),
            "evaluated_factors": evaluated,
            "skipped_factors": skipped,
            "rows": rows,
            "signal_availability": "END_OF_DAY_AFTER_CLOSE",
            "execution_note": "A selection list, not an execution order or a backtest result.",
            "data_fingerprint": self.workspace.fingerprint,
        }

    def _load_snapshot(
        self, requested_date: date
    ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.Timestamp]:
        requested = pd.Timestamp(requested_date)
        load_start = requested - pd.Timedelta(days=500)
        factor_fields = list(self.workspace.factor_fields)
        workspace_columns = set(getattr(self.workspace, "columns", ()))
        columns = list(
            dict.fromkeys(
                [
                    "trade_date",
                    "ts_code",
                    "name",
                    "open",
                    *factor_fields,
                    "raw_open",
                    "raw_close",
                    "is_valid_ohlc",
                    "is_tradable_observation",
                ]
            )
        )
        columns = [
            column for column in columns if not workspace_columns or column in workspace_columns
        ]
        frames = []
        for year in range(load_start.year, requested.year + 1):
            for path in sorted((self.panel_path / f"trade_year={year}").glob("*.parquet")):
                available_columns = set(pq.read_schema(path).names)
                read_columns = [column for column in columns if column in available_columns]
                frames.append(pd.read_parquet(path, columns=read_columns))
        if not frames:
            raise FileNotFoundError(
                f"No panel partitions found before {requested_date.isoformat()}"
            )
        data = pd.concat(frames, ignore_index=True)
        data["trade_date"] = pd.to_datetime(data["trade_date"])
        data = data[(data["trade_date"] >= load_start) & (data["trade_date"] <= requested)]
        available_dates = data.loc[data["trade_date"] <= requested, "trade_date"]
        if available_dates.empty:
            raise ValueError(
                f"No trading session is available on or before {requested_date.isoformat()}"
            )
        resolved_date = available_dates.max()
        valid = data["is_valid_ohlc"].fillna(False) & data["is_tradable_observation"].fillna(False)
        field_names = list(dict.fromkeys(["open", *factor_fields]))
        data.loc[~valid, field_names] = np.nan
        fields = {
            name: data.pivot(index="trade_date", columns="ts_code", values=name).sort_index()
            for name in field_names
        }
        snapshot = data[data["trade_date"] == resolved_date].set_index("ts_code")
        return fields, snapshot, resolved_date


def _finite_float(value: Any) -> float | None:
    parsed = float(value)
    return parsed if np.isfinite(parsed) else None
