from autoalpha.dsl.semantics import FieldDefinition, SemanticValidator
from autoalpha.service.openai_client import ModelInvocationError
from autoalpha.service.worker import (
    _circuit_breaker_reason,
    _codex_task_baseline,
    _genesis_baseline,
)


def test_genesis_baseline_is_stable_and_semantically_valid() -> None:
    factor = _genesis_baseline()
    validator = SemanticValidator([FieldDefinition("adj_close", "price")])

    result = validator.validate(factor.expression)

    assert factor.name == "baseline_reversal_20_v1"
    assert factor.factor_id == "F_3652a1c344c02c63"
    assert result.lookback == 20
    assert result.unit == "dimensionless"


def test_codex_task_baseline_is_stable_and_semantically_valid() -> None:
    factor = _codex_task_baseline()
    validator = SemanticValidator([FieldDefinition("adj_close", "price")])

    result = validator.validate(factor.expression)

    assert factor.name == "Codex_ShortTerm_Reversal_5d_Baseline"
    assert factor.factor_id == "F_ec804c2b24abced1"
    assert result.lookback == 5
    assert result.unit == "dimensionless"


def test_circuit_breaker_distinguishes_contract_and_transport_failures() -> None:
    assert (
        _circuit_breaker_reason(
            ValueError("contract"), consecutive_failures=3, same_failure_count=3
        )
        == "同一确定性错误连续出现三次"
    )
    transport = ModelInvocationError("timeout", stage="transport", prompt_hash="0" * 64)
    assert _circuit_breaker_reason(transport, consecutive_failures=3, same_failure_count=3) is None
    assert (
        _circuit_breaker_reason(transport, consecutive_failures=5, same_failure_count=5)
        == "连续失败达到五次"
    )
