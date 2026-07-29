from autoalpha.dsl.semantics import FieldDefinition, SemanticValidator
from autoalpha.service.openai_client import ModelInvocationError
from autoalpha.service.worker import (
    _candidate_failure_profile,
    _candidate_level_failure_reason,
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


def test_candidate_level_failure_classification_is_conservative() -> None:
    assert (
        _candidate_level_failure_reason(ValueError("Evaluation produced non-finite metrics"))
        == "NON_FINITE_SINGLE_FACTOR_METRICS"
    )
    assert (
        _candidate_level_failure_reason(ValueError("insufficient valid dates for backtest"))
        == "INSUFFICIENT_VALID_DATES"
    )
    assert (
        _candidate_level_failure_reason(ValueError("Duplicate candidate expression: F_1"))
        == "DUPLICATE_CANDIDATE_EXPRESSION"
    )
    assert (
        _candidate_level_failure_reason(ValueError("Parameter-only duplicate of F_2"))
        == "PARAMETER_ONLY_DUPLICATE"
    )
    assert (
        _candidate_level_failure_reason(
            ValueError("Candidate signal preflight failed: all factor values are missing")
        )
        == "EMPTY_FACTOR_SIGNAL"
    )
    assert (
        _candidate_level_failure_reason(
            ValueError("EXPLORE_EXTENDED_DATA requires at least one eligible extended field")
        )
        == "EXTENDED_DATA_FIELD_MISMATCH"
    )
    assert (
        _candidate_level_failure_reason(
            ValueError(
                "Frozen mechanism campaign requires LIQUIDITY; proposal classified as MOMENTUM"
            )
        )
        == "MECHANISM_CAMPAIGN_MISMATCH"
    )
    transport = ModelInvocationError("timeout", stage="transport", prompt_hash="0" * 64)
    assert _candidate_level_failure_reason(transport) is None
    assert _candidate_level_failure_reason(OSError("database is locked")) is None


def test_candidate_failure_profile_keeps_bad_candidates_out_of_task_circuit_breaker() -> None:
    profile = _candidate_failure_profile(
        {
            "exploratory_gate_failures": ["coverage", "homogeneity_near_duplicate"],
            "portfolio_action_gate_failures": ["turnover", "portfolio_max_factor_correlation"],
        },
        candidate_eligible=False,
        portfolio_accepted=False,
    )

    assert profile["protocol"] == "CANDIDATE_FAILURE_PROFILE_V1"
    assert profile["severity"] == "HARD_FAIL"
    assert profile["task_circuit_breaker_eligible"] is False
    assert profile["operational_failure"] is False
    assert profile["categories"] == [
        "COVERAGE",
        "HOMOGENEITY",
        "INDEPENDENCE",
        "PORTFOLIO_GATE",
        "SINGLE_FACTOR_GATE",
        "TURNOVER",
    ]


def test_candidate_failure_profile_passes_clean_candidate() -> None:
    profile = _candidate_failure_profile(
        {},
        candidate_eligible=True,
        portfolio_accepted=True,
    )

    assert profile["severity"] == "PASS"
    assert profile["categories"] == []
    assert profile["failure_count"] == 0
