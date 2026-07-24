from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from autoalpha.governance.audit import HashChainAuditLog


class Role(StrEnum):
    RESEARCHER = "RESEARCHER"
    REVIEWER = "REVIEWER"
    EXECUTOR = "EXECUTOR"
    EVALUATOR = "EVALUATOR"
    RISK_GATE = "RISK_GATE"
    HUMAN_APPROVER = "HUMAN_APPROVER"


class Capability(StrEnum):
    PROPOSE_DSL = "PROPOSE_DSL"
    REVIEW_PROPOSAL = "REVIEW_PROPOSAL"
    EXECUTE_DSL = "EXECUTE_DSL"
    READ_VALIDATION = "READ_VALIDATION"
    READ_HOLDOUT = "READ_HOLDOUT"
    ASSESS_RISK = "ASSESS_RISK"
    APPROVE_HOLDOUT = "APPROVE_HOLDOUT"
    APPROVE_PRODUCTION = "APPROVE_PRODUCTION"
    CONTROL_RUN = "CONTROL_RUN"


DEFAULT_ROLE_CAPABILITIES: dict[Role, frozenset[Capability]] = {
    Role.RESEARCHER: frozenset({Capability.PROPOSE_DSL}),
    Role.REVIEWER: frozenset({Capability.REVIEW_PROPOSAL}),
    Role.EXECUTOR: frozenset({Capability.EXECUTE_DSL}),
    Role.EVALUATOR: frozenset({Capability.READ_VALIDATION, Capability.READ_HOLDOUT}),
    Role.RISK_GATE: frozenset({Capability.ASSESS_RISK}),
    Role.HUMAN_APPROVER: frozenset(
        {
            Capability.APPROVE_HOLDOUT,
            Capability.APPROVE_PRODUCTION,
            Capability.CONTROL_RUN,
        }
    ),
}


class RolePolicy:
    def __init__(self, grants: dict[Role, frozenset[Capability]] | None = None) -> None:
        self.grants = grants or DEFAULT_ROLE_CAPABILITIES

    def require(self, role: Role, capability: Capability) -> None:
        if capability not in self.grants.get(role, frozenset()):
            raise PermissionError(f"Role {role} cannot use capability {capability}")


@dataclass(frozen=True)
class ModelInvocation:
    invocation_id: str
    role: Role
    provider: str
    model: str
    prompt_hash: str
    context_hash: str
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    tool_calls: tuple[str, ...]

    @classmethod
    def capture(
        cls,
        *,
        invocation_id: str,
        role: Role,
        provider: str,
        model: str,
        prompt: str,
        context: Any,
        input_tokens: int,
        output_tokens: int,
        estimated_cost_usd: float,
        tool_calls: tuple[str, ...] = (),
    ) -> ModelInvocation:
        return cls(
            invocation_id,
            role,
            provider,
            model,
            _sha256(prompt),
            _sha256(_canonical_json(context)),
            input_tokens,
            output_tokens,
            estimated_cost_usd,
            tool_calls,
        )

    def record(self, audit: HashChainAuditLog) -> None:
        payload = asdict(self)
        payload["role"] = self.role.value
        payload["tool_calls"] = list(self.tool_calls)
        audit.append("MODEL_INVOCATION", payload)


@dataclass(frozen=True)
class ControlledFeedbackPolicy:
    allowed_metrics: frozenset[str]
    decimal_places: int = 3

    def filter(self, metrics: dict[str, Any]) -> dict[str, Any]:
        public: dict[str, Any] = {}
        for key in sorted(self.allowed_metrics & metrics.keys()):
            value = metrics[key]
            public[key] = (
                round(float(value), self.decimal_places)
                if isinstance(value, (int, float))
                else value
            )
        return public


class RunState(StrEnum):
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"


@dataclass(frozen=True)
class StopPolicy:
    maximum_consecutive_failures: int = 3
    maximum_anomalies: int = 1
    stop_on_robustness_failure: bool = True


class ExperimentRunController:
    def __init__(self, audit: HashChainAuditLog, policy: StopPolicy | None = None) -> None:
        self.audit = audit
        self.policy = policy or StopPolicy()
        self.state = RunState.ACTIVE
        self.consecutive_failures = 0
        self.anomalies = 0

    def require_active(self) -> None:
        if self.state is not RunState.ACTIVE:
            raise RuntimeError(f"Experiment run is {self.state}")

    def observe(
        self,
        *,
        success: bool,
        anomaly: bool = False,
        robustness_passed: bool = True,
    ) -> RunState:
        self.require_active()
        self.consecutive_failures = 0 if success else self.consecutive_failures + 1
        self.anomalies += int(anomaly)
        reasons = []
        if self.consecutive_failures >= self.policy.maximum_consecutive_failures:
            reasons.append("consecutive failures")
        if self.anomalies >= self.policy.maximum_anomalies:
            reasons.append("anomaly limit")
        if self.policy.stop_on_robustness_failure and not robustness_passed:
            reasons.append("robustness gate")
        if reasons:
            self.state = RunState.STOPPED
            self.audit.append("RUN_STOPPED", {"reasons": reasons})
        return self.state

    def pause(self, role: Role, role_policy: RolePolicy, reason: str) -> None:
        role_policy.require(role, Capability.CONTROL_RUN)
        self.require_active()
        self.state = RunState.PAUSED
        self.audit.append("RUN_PAUSED", {"actor": role.value, "reason": reason})

    def resume(self, role: Role, role_policy: RolePolicy, reason: str) -> None:
        role_policy.require(role, Capability.CONTROL_RUN)
        if self.state is not RunState.PAUSED:
            raise RuntimeError("Only a paused run can be resumed")
        self.state = RunState.ACTIVE
        self.audit.append("RUN_RESUMED", {"actor": role.value, "reason": reason})


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
