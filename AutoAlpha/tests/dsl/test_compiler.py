from __future__ import annotations

import numpy as np
import pandas as pd

from autoalpha.dsl.compiler import FactorCompiler
from autoalpha.dsl.expression import field, operation
from autoalpha.dsl.semantics import FieldDefinition, SemanticValidator


def test_compiler_evaluates_point_in_time_reversal_factor() -> None:
    dates = pd.bdate_range("2024-01-01", periods=8)
    close = pd.DataFrame(
        {"A": np.arange(10.0, 18.0), "B": np.arange(20.0, 12.0, -1)},
        index=dates,
    )
    expression = operation(
        "cs_zscore",
        operation("negate", operation("returns", field("close"), periods=3)),
    )
    compiler = FactorCompiler(SemanticValidator([FieldDefinition("close", "price")]))

    result = compiler.evaluate(expression, {"close": close})

    assert result.iloc[:3].isna().all().all()
    assert np.allclose(result.iloc[3:].mean(axis=1), 0.0)
    assert (result.loc[dates[3] :, "B"] > result.loc[dates[3] :, "A"]).all()


def test_compiler_protects_division_by_zero() -> None:
    dates = pd.bdate_range("2024-01-01", periods=2)
    close = pd.DataFrame({"A": [1.0, 2.0]}, index=dates)
    zero = pd.DataFrame({"A": [0.0, 0.0]}, index=dates)
    expression = operation("divide", field("close"), field("zero"))
    compiler = FactorCompiler(
        SemanticValidator([FieldDefinition("close", "price"), FieldDefinition("zero", "price")])
    )

    result = compiler.evaluate(expression, {"close": close, "zero": zero})

    assert result.isna().all().all()


def test_compiler_neutralizes_exposure_and_reuses_common_expression() -> None:
    dates = pd.date_range("2024-01-01", periods=2)
    columns = ["A", "B", "C", "D"]
    size = pd.DataFrame([[1, 2, 3, 4], [4, 3, 2, 1]], index=dates, columns=columns)
    noise = pd.DataFrame([[1, -1, 1, -1], [-1, 1, -1, 1]], index=dates, columns=columns)
    raw = size * 2 + noise
    validator = SemanticValidator(
        [FieldDefinition("raw", "dimensionless"), FieldDefinition("size", "dimensionless")]
    )
    compiler = FactorCompiler(validator)
    neutral = operation("neutralize", field("raw"), field("size"))
    expression = operation("add", neutral, neutral)

    result = compiler.evaluate(expression, {"raw": raw, "size": size})

    assert abs(result.loc[dates[0]].corr(size.loc[dates[0]])) < 1e-12
    assert compiler.last_cache_entries == 4
