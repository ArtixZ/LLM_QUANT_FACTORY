from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

COMMUTATIVE_OPERATORS = frozenset({"add", "multiply"})
ROLLING_OPERATORS = frozenset(
    {"rolling_mean", "rolling_sum", "rolling_std", "rolling_min", "rolling_max"}
)
PERIOD_OPERATORS = frozenset({"delay", "delta", "returns"})


@dataclass(frozen=True)
class Expression:
    operator: str
    arguments: tuple[Expression, ...] = ()
    parameters: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Expression:
        if "operator" not in raw:
            raise ValueError("Expression requires an operator")
        operator = str(raw["operator"])
        parameters = raw.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise TypeError("Expression parameters must be a mapping")
        parameters = _normalize_parameters(operator, parameters)
        arguments = raw.get("arguments", [])
        if not isinstance(arguments, list):
            raise TypeError("Expression arguments must be a list")
        return cls(
            operator=operator,
            arguments=tuple(cls.from_dict(argument) for argument in arguments),
            parameters=tuple(sorted((str(key), value) for key, value in parameters.items())),
        ).canonical()

    def parameter(self, name: str, default: Any = None) -> Any:
        return dict(self.parameters).get(name, default)

    def canonical(self) -> Expression:
        arguments = tuple(argument.canonical() for argument in self.arguments)
        if self.operator in COMMUTATIVE_OPERATORS:
            arguments = tuple(sorted(arguments, key=lambda item: item.canonical_json))
        parameters = tuple(sorted(self.parameters))
        return Expression(self.operator, arguments, parameters)

    def to_dict(self) -> dict[str, Any]:
        expression = self.canonical()
        return {
            "operator": expression.operator,
            "arguments": [argument.to_dict() for argument in expression.arguments],
            "parameters": dict(expression.parameters),
        }

    @property
    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    @property
    def expression_hash(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FactorDefinition:
    name: str
    family: str
    hypothesis: str
    expression: Expression
    expected_direction: int = 1

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.family.strip() or not self.hypothesis.strip():
            raise ValueError("Factor name, family, and hypothesis are required")
        if self.expected_direction not in {-1, 1}:
            raise ValueError("expected_direction must be -1 or 1")

    @property
    def factor_id(self) -> str:
        return f"F_{self.expression.expression_hash[:16]}"


def field(name: str) -> Expression:
    return Expression("field", parameters=(("name", name),))


def canonical_family(name: str) -> str:
    """Normalize free-form family labels to one snake_case vocabulary.

    LLM proposals label the same family inconsistently ("Valuation",
    "valuation", "TURNOVER_LIQUIDITY", "TurnoverLiquidity", "Order Flow"),
    which silently fragments family budgets, crowding statistics, and
    diversity selection. Aggregations and newly stored proposals should use
    this canonical form; historical records are normalized at read time only.
    """
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(name).strip())
    text = re.sub(r"[\s\-/]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_").lower() or "unknown"


def constant(value: float, *, unit: str = "dimensionless") -> Expression:
    return Expression("constant", parameters=(("unit", unit), ("value", float(value))))


def operation(operator: str, *arguments: Expression, **parameters: Any) -> Expression:
    return Expression(
        operator=operator,
        arguments=tuple(arguments),
        parameters=tuple(sorted(parameters.items())),
    ).canonical()


def _normalize_parameters(operator: str, raw: Mapping[str, Any]) -> dict[str, Any]:
    parameters = {str(key): value for key, value in raw.items()}
    if operator == "field":
        _move_alias(parameters, canonical="name", aliases=("field",))
    elif operator in PERIOD_OPERATORS:
        _move_alias(parameters, canonical="periods", aliases=("period", "window"))
    elif operator in ROLLING_OPERATORS:
        _move_alias(parameters, canonical="window", aliases=("period", "periods"))
    return parameters


def _move_alias(parameters: dict[str, Any], *, canonical: str, aliases: tuple[str, ...]) -> None:
    supplied = [(key, parameters[key]) for key in (canonical, *aliases) if key in parameters]
    if not supplied:
        return
    values = {json.dumps(value, sort_keys=True) for _, value in supplied}
    if len(values) > 1:
        names = ", ".join(key for key, _ in supplied)
        raise ValueError(f"Conflicting expression parameters: {names}")
    parameters[canonical] = supplied[0][1]
    for alias in aliases:
        parameters.pop(alias, None)
