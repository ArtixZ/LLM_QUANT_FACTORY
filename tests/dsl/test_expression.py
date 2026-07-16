from __future__ import annotations

import pytest

from autoalpha.dsl.expression import Expression, field, operation


def test_commutative_expressions_have_same_identity() -> None:
    left = operation("add", field("close"), field("open"))
    right = operation("add", field("open"), field("close"))

    assert left.expression_hash == right.expression_hash
    assert left.canonical_json == right.canonical_json


def test_expression_round_trip_is_canonical() -> None:
    raw = {
        "operator": "returns",
        "arguments": [{"operator": "field", "parameters": {"name": "close"}}],
        "parameters": {"periods": 5},
    }

    expression = Expression.from_dict(raw)

    assert Expression.from_dict(expression.to_dict()) == expression


def test_model_parameter_aliases_normalize_to_canonical_dsl() -> None:
    expression = Expression.from_dict(
        {
            "operator": "rolling_mean",
            "arguments": [
                {
                    "operator": "returns",
                    "arguments": [{"operator": "field", "parameters": {"field": "close"}}],
                    "parameters": {"period": 1},
                }
            ],
            "parameters": {"periods": 20},
        }
    )

    assert expression.to_dict()["parameters"] == {"window": 20}
    assert expression.arguments[0].to_dict()["parameters"] == {"periods": 1}
    assert expression.arguments[0].arguments[0] == field("close")


def test_conflicting_parameter_aliases_are_rejected() -> None:
    with pytest.raises(ValueError, match="Conflicting"):
        Expression.from_dict(
            {
                "operator": "field",
                "parameters": {"name": "close", "field": "vol"},
            }
        )
