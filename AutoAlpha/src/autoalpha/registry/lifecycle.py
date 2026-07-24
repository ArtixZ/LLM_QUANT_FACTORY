from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path

from autoalpha.governance.audit import HashChainAuditLog


class FactorState(StrEnum):
    CANDIDATE = "CANDIDATE"
    RESEARCH = "RESEARCH"
    APPROVED = "APPROVED"
    QUALIFIED = "QUALIFIED"
    SHADOW = "SHADOW"
    PAPER = "PAPER"
    PRODUCTION = "PRODUCTION"
    WATCH = "WATCH"
    DECAYED = "DECAYED"
    SUSPENDED = "SUSPENDED"
    RETIRED = "RETIRED"
    REJECTED = "REJECTED"


ALLOWED_TRANSITIONS: dict[FactorState, frozenset[FactorState]] = {
    FactorState.CANDIDATE: frozenset({FactorState.RESEARCH, FactorState.REJECTED}),
    FactorState.RESEARCH: frozenset(
        {FactorState.QUALIFIED, FactorState.APPROVED, FactorState.REJECTED}
    ),
    FactorState.QUALIFIED: frozenset({FactorState.SHADOW, FactorState.REJECTED}),
    FactorState.APPROVED: frozenset({FactorState.SHADOW, FactorState.PAPER, FactorState.REJECTED}),
    FactorState.SHADOW: frozenset({FactorState.PAPER, FactorState.SUSPENDED, FactorState.REJECTED}),
    FactorState.PAPER: frozenset({FactorState.PRODUCTION, FactorState.SUSPENDED}),
    FactorState.PRODUCTION: frozenset({FactorState.WATCH, FactorState.SUSPENDED}),
    FactorState.WATCH: frozenset(
        {FactorState.PRODUCTION, FactorState.DECAYED, FactorState.SUSPENDED}
    ),
    FactorState.DECAYED: frozenset({FactorState.SUSPENDED, FactorState.RETIRED}),
    FactorState.SUSPENDED: frozenset({FactorState.PAPER, FactorState.RETIRED}),
    FactorState.REJECTED: frozenset({FactorState.RESEARCH, FactorState.RETIRED}),
    FactorState.RETIRED: frozenset(),
}


@dataclass(frozen=True)
class LifecycleEvent:
    factor_id: str
    previous_state: FactorState | None
    state: FactorState
    actor: str
    reason: str
    timestamp_utc: str


class LifecycleStore:
    def __init__(self, path: Path, *, rejection_cooldown_days: int = 30) -> None:
        self.audit = HashChainAuditLog(path)
        self.rejection_cooldown_days = rejection_cooldown_days

    def history(self, factor_id: str) -> tuple[LifecycleEvent, ...]:
        return tuple(
            LifecycleEvent(
                factor_id=record.payload["factor_id"],
                previous_state=(
                    FactorState(record.payload["previous_state"])
                    if record.payload["previous_state"] is not None
                    else None
                ),
                state=FactorState(record.payload["state"]),
                actor=record.payload["actor"],
                reason=record.payload["reason"],
                timestamp_utc=record.timestamp_utc,
            )
            for record in self.audit.records()
            if record.event == "factor_state" and record.payload["factor_id"] == factor_id
        )

    def state(self, factor_id: str) -> FactorState | None:
        history = self.history(factor_id)
        return history[-1].state if history else None

    def register(self, factor_id: str, *, actor: str, reason: str) -> LifecycleEvent:
        if self.state(factor_id) is not None:
            raise ValueError(f"Factor {factor_id} is already registered")
        return self._append(factor_id, None, FactorState.CANDIDATE, actor, reason)

    def transition(
        self,
        factor_id: str,
        target: FactorState,
        *,
        actor: str,
        reason: str,
        now: datetime | None = None,
    ) -> LifecycleEvent:
        current = self.state(factor_id)
        if current is None:
            raise KeyError(f"Unknown factor {factor_id}")
        if target not in ALLOWED_TRANSITIONS[current]:
            raise ValueError(f"Invalid factor transition: {current} -> {target}")
        if current is FactorState.REJECTED and target is FactorState.RESEARCH:
            rejected_at = datetime.fromisoformat(self.history(factor_id)[-1].timestamp_utc)
            effective_now = now or datetime.now(UTC)
            if effective_now < rejected_at + timedelta(days=self.rejection_cooldown_days):
                raise ValueError("Rejected factor is still in its research cooldown")
        return self._append(factor_id, current, target, actor, reason)

    def _append(
        self,
        factor_id: str,
        previous: FactorState | None,
        state: FactorState,
        actor: str,
        reason: str,
    ) -> LifecycleEvent:
        if not actor.strip() or not reason.strip():
            raise ValueError("Lifecycle actor and reason are required")
        record = self.audit.append(
            "factor_state",
            {
                "factor_id": factor_id,
                "previous_state": previous.value if previous else None,
                "state": state.value,
                "actor": actor,
                "reason": reason,
            },
        )
        return LifecycleEvent(factor_id, previous, state, actor, reason, record.timestamp_utc)
