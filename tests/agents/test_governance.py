import pytest

from autoalpha.agents.governance import (
    Capability,
    ControlledFeedbackPolicy,
    ExperimentRunController,
    ModelInvocation,
    Role,
    RolePolicy,
    RunState,
    StopPolicy,
)
from autoalpha.governance.audit import HashChainAuditLog


def test_role_policy_keeps_researcher_out_of_holdout() -> None:
    policy = RolePolicy()
    policy.require(Role.RESEARCHER, Capability.PROPOSE_DSL)
    with pytest.raises(PermissionError):
        policy.require(Role.RESEARCHER, Capability.READ_HOLDOUT)
    policy.require(Role.HUMAN_APPROVER, Capability.APPROVE_PRODUCTION)


def test_controlled_feedback_hides_details_and_reduces_precision() -> None:
    policy = ControlledFeedbackPolicy(frozenset({"rank_ic", "coverage"}), decimal_places=2)
    feedback = policy.filter(
        {"rank_ic": 0.012345, "coverage": 0.9876, "hidden_test_sharpe": 3.14159}
    )
    assert feedback == {"coverage": 0.99, "rank_ic": 0.01}


def test_model_invocation_records_replay_metadata_without_raw_prompt(tmp_path) -> None:
    audit = HashChainAuditLog(tmp_path / "audit.jsonl")
    invocation = ModelInvocation.capture(
        invocation_id="I1",
        role=Role.RESEARCHER,
        provider="provider-a",
        model="model-v1",
        prompt="secret research prompt",
        context={"generation": "g1"},
        input_tokens=100,
        output_tokens=20,
        estimated_cost_usd=0.01,
        tool_calls=("factor_library",),
    )
    invocation.record(audit)
    payload = audit.records()[0].payload
    assert payload["prompt_hash"] != "secret research prompt"
    assert payload["input_tokens"] == 100
    assert payload["tool_calls"] == ["factor_library"]
    audit.verify()


def test_run_controller_stops_on_failure_and_human_controls_pause(tmp_path) -> None:
    audit = HashChainAuditLog(tmp_path / "audit.jsonl")
    controller = ExperimentRunController(
        audit, StopPolicy(maximum_consecutive_failures=2, maximum_anomalies=2)
    )
    roles = RolePolicy()
    with pytest.raises(PermissionError):
        controller.pause(Role.RESEARCHER, roles, "not allowed")
    controller.pause(Role.HUMAN_APPROVER, roles, "review")
    assert controller.state is RunState.PAUSED
    controller.resume(Role.HUMAN_APPROVER, roles, "continue")
    controller.observe(success=False)
    assert controller.observe(success=False) is RunState.STOPPED
    with pytest.raises(RuntimeError):
        controller.require_active()
