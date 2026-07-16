from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from autoalpha.dsl.expression import Expression
from autoalpha.dsl.semantics import SemanticValidator


class FactorCompiler:
    def __init__(self, validator: SemanticValidator) -> None:
        self.validator = validator
        self.last_cache_entries = 0

    def evaluate(
        self,
        expression: Expression,
        fields: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        self.validator.validate(expression)
        if not fields:
            raise ValueError("At least one field panel is required")
        cache: dict[str, pd.DataFrame | float] = {}
        result = self._evaluate(expression.canonical(), fields, cache)
        self.last_cache_entries = len(cache)
        if not isinstance(result, pd.DataFrame):
            raise TypeError("Factor expression did not produce a panel")
        return result.replace([np.inf, -np.inf], np.nan)

    def _evaluate(
        self,
        expression: Expression,
        fields: dict[str, pd.DataFrame],
        cache: dict[str, pd.DataFrame | float],
    ) -> pd.DataFrame | float:
        cache_key = expression.expression_hash
        if cache_key in cache:
            return cache[cache_key]
        operator = expression.operator
        if operator == "field":
            name = str(expression.parameter("name"))
            if name not in fields:
                raise KeyError(f"Missing field panel: {name}")
            result = fields[name].astype(float).copy()
            cache[cache_key] = result
            return result
        if operator == "constant":
            result = float(expression.parameter("value"))
            cache[cache_key] = result
            return result
        arguments = [self._evaluate(argument, fields, cache) for argument in expression.arguments]
        if operator == "negate":
            result = -arguments[0]
            cache[cache_key] = result
            return result
        if operator == "absolute":
            result = abs(arguments[0])
            cache[cache_key] = result
            return result
        if operator == "add":
            result = arguments[0] + arguments[1]
            cache[cache_key] = result
            return result
        if operator == "subtract":
            result = arguments[0] - arguments[1]
            cache[cache_key] = result
            return result
        if operator == "multiply":
            result = arguments[0] * arguments[1]
            cache[cache_key] = result
            return result
        if operator == "divide":
            denominator = arguments[1]
            if isinstance(denominator, pd.DataFrame):
                denominator = denominator.replace(0, np.nan)
            elif denominator == 0:
                denominator = np.nan
            result = arguments[0] / denominator
            cache[cache_key] = result
            return result
        if operator == "neutralize":
            result = _neutralize_panel(
                _as_panel(arguments[0], operator), _as_panel(arguments[1], operator)
            )
            cache[cache_key] = result
            return result
        panel = _as_panel(arguments[0], operator)
        if operator == "delay":
            result = panel.shift(int(expression.parameter("periods")))
            cache[cache_key] = result
            return result
        if operator == "delta":
            result = panel.diff(int(expression.parameter("periods")))
            cache[cache_key] = result
            return result
        if operator == "returns":
            periods = int(expression.parameter("periods"))
            result = panel.pct_change(periods, fill_method=None)
            cache[cache_key] = result
            return result
        if operator.startswith("rolling_"):
            window = int(expression.parameter("window"))
            minimum = int(expression.parameter("min_periods", window))
            rolling = panel.rolling(window, min_periods=minimum)
            function = operator.removeprefix("rolling_")
            result = getattr(rolling, function)()
            cache[cache_key] = result
            return result
        if operator == "cs_rank":
            result = panel.rank(axis=1, pct=True)
            cache[cache_key] = result
            return result
        if operator == "cs_zscore":
            mean = panel.mean(axis=1)
            standard_deviation = panel.std(axis=1).replace(0, np.nan)
            result = panel.sub(mean, axis=0).div(standard_deviation, axis=0)
            cache[cache_key] = result
            return result
        if operator == "winsorize_mad":
            threshold = float(expression.parameter("threshold", 3.0))
            median = panel.median(axis=1)
            mad = panel.sub(median, axis=0).abs().median(axis=1) * 1.4826
            result = panel.clip(
                lower=median - threshold * mad,
                upper=median + threshold * mad,
                axis=0,
            )
            cache[cache_key] = result
            return result
        raise ValueError(f"No compiler implementation for operator: {operator}")


def _as_panel(value: Any, operator: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"{operator} requires a panel")
    return value


def _neutralize_panel(factor: pd.DataFrame, exposure: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=factor.index, columns=factor.columns, dtype=float)
    for date in factor.index.intersection(exposure.index):
        y = factor.loc[date].astype(float)
        x = exposure.loc[date].astype(float)
        valid = y.notna() & x.notna()
        if valid.sum() < 3:
            continue
        design = np.column_stack([np.ones(valid.sum()), x.loc[valid].to_numpy()])
        coefficients, *_ = np.linalg.lstsq(design, y.loc[valid].to_numpy(), rcond=None)
        result.loc[date, valid] = y.loc[valid] - design @ coefficients
    return result
