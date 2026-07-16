from __future__ import annotations

import pytest

from autoalpha.dsl.expression import field, operation
from autoalpha.dsl.semantics import FieldDefinition, SemanticError, SemanticValidator


def _validator() -> SemanticValidator:
    return SemanticValidator(
        [
            FieldDefinition("close", "price"),
            FieldDefinition("open", "price"),
            FieldDefinition("volume", "shares"),
        ]
    )


def test_semantics_tracks_lookback_and_unit() -> None:
    expression = operation("cs_zscore", operation("returns", field("close"), periods=20))

    result = _validator().validate(expression)

    assert result.lookback == 20
    assert result.unit == "dimensionless"


def test_semantics_rejects_future_or_invalid_unit_operations() -> None:
    with pytest.raises(SemanticError, match="future-unsafe"):
        _validator().validate(operation("lead", field("close"), periods=1))
    with pytest.raises(SemanticError, match="equal units"):
        _validator().validate(operation("add", field("close"), field("volume")))
    with pytest.raises(SemanticError, match="positive integer"):
        _validator().validate(operation("delay", field("close"), periods=-1))


def test_semantics_enforces_lookback_budget() -> None:
    validator = SemanticValidator([FieldDefinition("close", "price")], maximum_lookback=60)

    with pytest.raises(SemanticError, match="lookback"):
        validator.validate(operation("rolling_mean", field("close"), window=120))
