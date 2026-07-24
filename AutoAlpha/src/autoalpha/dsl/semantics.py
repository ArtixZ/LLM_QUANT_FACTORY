from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from autoalpha.dsl.expression import Expression


class SemanticError(ValueError):
    """An expression is type-invalid, unbounded, or not point-in-time safe."""


@dataclass(frozen=True)
class FieldDefinition:
    name: str
    unit: str
    knowledge_lag: int = 0

    def __post_init__(self) -> None:
        if not self.name or not self.unit or self.knowledge_lag < 0:
            raise ValueError("Field name/unit are required and knowledge_lag cannot be negative")


@dataclass(frozen=True)
class SemanticResult:
    value_type: str
    unit: str
    lookback: int
    nodes: int


@dataclass(frozen=True)
class OperatorSpec:
    arity: int
    result: Callable[[Expression, tuple[SemanticResult, ...]], SemanticResult]


class SemanticValidator:
    def __init__(
        self,
        fields: list[FieldDefinition] | tuple[FieldDefinition, ...],
        *,
        maximum_lookback: int = 504,
        maximum_nodes: int = 200,
    ) -> None:
        self.fields = {definition.name: definition for definition in fields}
        self.maximum_lookback = maximum_lookback
        self.maximum_nodes = maximum_nodes
        self.operators = _operators()

    def validate(self, expression: Expression) -> SemanticResult:
        result = self._visit(expression.canonical())
        if result.lookback > self.maximum_lookback:
            raise SemanticError(
                f"Expression lookback {result.lookback} exceeds {self.maximum_lookback}"
            )
        if result.nodes > self.maximum_nodes:
            raise SemanticError(f"Expression nodes {result.nodes} exceeds {self.maximum_nodes}")
        if result.value_type != "panel":
            raise SemanticError("A factor root must produce a panel")
        return result

    def _visit(self, expression: Expression) -> SemanticResult:
        if expression.operator == "field":
            name = expression.parameter("name")
            if name not in self.fields:
                raise SemanticError(f"Unknown field: {name!r}")
            definition = self.fields[name]
            return SemanticResult("panel", definition.unit, definition.knowledge_lag, 1)
        if expression.operator == "constant":
            value = expression.parameter("value")
            if not isinstance(value, (int, float)):
                raise SemanticError("constant.value must be numeric")
            return SemanticResult("scalar", str(expression.parameter("unit")), 0, 1)
        spec = self.operators.get(expression.operator)
        if spec is None:
            raise SemanticError(f"Unknown or future-unsafe operator: {expression.operator!r}")
        if len(expression.arguments) != spec.arity:
            raise SemanticError(
                f"{expression.operator} expects {spec.arity} arguments, "
                f"received {len(expression.arguments)}"
            )
        arguments = tuple(self._visit(argument) for argument in expression.arguments)
        result = spec.result(expression, arguments)
        return SemanticResult(
            result.value_type,
            result.unit,
            result.lookback,
            1 + sum(argument.nodes for argument in arguments),
        )


def _operators() -> dict[str, OperatorSpec]:
    return {
        "negate": OperatorSpec(1, _unary_preserve),
        "absolute": OperatorSpec(1, _unary_preserve),
        "add": OperatorSpec(2, _add_subtract),
        "subtract": OperatorSpec(2, _add_subtract),
        "multiply": OperatorSpec(2, _multiply),
        "divide": OperatorSpec(2, _divide),
        "delay": OperatorSpec(1, _window_preserve),
        "delta": OperatorSpec(1, _window_preserve),
        "returns": OperatorSpec(1, _returns),
        "rolling_mean": OperatorSpec(1, _window_preserve),
        "rolling_sum": OperatorSpec(1, _window_preserve),
        "rolling_std": OperatorSpec(1, _rolling_std),
        "rolling_min": OperatorSpec(1, _window_preserve),
        "rolling_max": OperatorSpec(1, _window_preserve),
        "cs_rank": OperatorSpec(1, _cross_sectional),
        "cs_zscore": OperatorSpec(1, _cross_sectional),
        "winsorize_mad": OperatorSpec(1, _unary_preserve),
        "neutralize": OperatorSpec(2, _neutralize),
    }


def _require_panel(arguments: tuple[SemanticResult, ...]) -> None:
    if not any(argument.value_type == "panel" for argument in arguments):
        raise SemanticError("Operator must receive at least one panel")


def _unary_preserve(
    expression: Expression, arguments: tuple[SemanticResult, ...]
) -> SemanticResult:
    argument = arguments[0]
    if argument.value_type != "panel":
        raise SemanticError(f"{expression.operator} requires a panel")
    return SemanticResult("panel", argument.unit, argument.lookback, 0)


def _add_subtract(expression: Expression, arguments: tuple[SemanticResult, ...]) -> SemanticResult:
    _require_panel(arguments)
    if arguments[0].unit != arguments[1].unit:
        raise SemanticError(
            f"{expression.operator} requires equal units, got "
            f"{arguments[0].unit!r} and {arguments[1].unit!r}"
        )
    return SemanticResult(
        "panel", arguments[0].unit, max(argument.lookback for argument in arguments), 0
    )


def _multiply(expression: Expression, arguments: tuple[SemanticResult, ...]) -> SemanticResult:
    _require_panel(arguments)
    units = [argument.unit for argument in arguments if argument.unit != "dimensionless"]
    unit = "*".join(sorted(units)) if units else "dimensionless"
    return SemanticResult("panel", unit, max(argument.lookback for argument in arguments), 0)


def _divide(expression: Expression, arguments: tuple[SemanticResult, ...]) -> SemanticResult:
    _require_panel(arguments)
    numerator, denominator = arguments
    unit = (
        "dimensionless"
        if numerator.unit == denominator.unit
        else f"{numerator.unit}/{denominator.unit}"
    )
    return SemanticResult("panel", unit, max(argument.lookback for argument in arguments), 0)


def _positive_window(expression: Expression) -> int:
    window = expression.parameter("periods", expression.parameter("window"))
    if not isinstance(window, int) or isinstance(window, bool) or window <= 0:
        raise SemanticError(f"{expression.operator} requires a positive integer window/periods")
    return window


def _window_preserve(
    expression: Expression, arguments: tuple[SemanticResult, ...]
) -> SemanticResult:
    argument = arguments[0]
    window = _positive_window(expression)
    return SemanticResult("panel", argument.unit, argument.lookback + window, 0)


def _returns(expression: Expression, arguments: tuple[SemanticResult, ...]) -> SemanticResult:
    argument = arguments[0]
    window = _positive_window(expression)
    return SemanticResult("panel", "return", argument.lookback + window, 0)


def _rolling_std(expression: Expression, arguments: tuple[SemanticResult, ...]) -> SemanticResult:
    result = _window_preserve(expression, arguments)
    return SemanticResult("panel", result.unit, result.lookback, 0)


def _cross_sectional(
    expression: Expression, arguments: tuple[SemanticResult, ...]
) -> SemanticResult:
    argument = arguments[0]
    return SemanticResult("panel", "dimensionless", argument.lookback, 0)


def _neutralize(expression: Expression, arguments: tuple[SemanticResult, ...]) -> SemanticResult:
    factor, exposure = arguments
    if factor.value_type != "panel" or exposure.value_type != "panel":
        raise SemanticError("neutralize requires factor and exposure panels")
    return SemanticResult("panel", factor.unit, max(factor.lookback, exposure.lookback), 0)
