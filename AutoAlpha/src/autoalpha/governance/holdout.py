from __future__ import annotations

import base64
import hashlib
import hmac
import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from autoalpha.governance.audit import HashChainAuditLog


class HoldoutPermissionError(PermissionError):
    """Holdout evaluation was not authorized or the approval was already consumed."""


@dataclass(frozen=True)
class HoldoutApproval:
    token_id: str
    candidate_hash: str
    generation: str
    expires_at_utc: str


class ApprovalAuthority:
    def __init__(self, secret: bytes) -> None:
        if len(secret) < 32:
            raise ValueError("Approval secret must contain at least 32 bytes")
        self._secret = secret

    def issue(
        self,
        candidate_hash: str,
        generation: str,
        *,
        ttl: timedelta = timedelta(hours=1),
    ) -> str:
        if ttl.total_seconds() <= 0:
            raise ValueError("Approval ttl must be positive")
        approval = HoldoutApproval(
            token_id=str(uuid.uuid4()),
            candidate_hash=candidate_hash,
            generation=generation,
            expires_at_utc=(datetime.now(UTC) + ttl).isoformat(),
        )
        payload = json.dumps(approval.__dict__, sort_keys=True, separators=(",", ":")).encode()
        signature = hmac.new(self._secret, payload, hashlib.sha256).digest()
        return f"{_encode(payload)}.{_encode(signature)}"

    def verify(self, token: str) -> HoldoutApproval:
        try:
            payload_part, signature_part = token.split(".", 1)
            payload = _decode(payload_part)
            signature = _decode(signature_part)
        except (ValueError, TypeError) as error:
            raise HoldoutPermissionError("Malformed holdout approval") from error
        expected = hmac.new(self._secret, payload, hashlib.sha256).digest()
        if not hmac.compare_digest(signature, expected):
            raise HoldoutPermissionError("Invalid holdout approval signature")
        approval = HoldoutApproval(**json.loads(payload))
        if datetime.fromisoformat(approval.expires_at_utc) <= datetime.now(UTC):
            raise HoldoutPermissionError("Holdout approval expired")
        return approval


class HoldoutJudge:
    """Consumes one-time approval and invokes a trusted evaluator inside the holdout boundary."""

    def __init__(self, authority: ApprovalAuthority, audit_log: HashChainAuditLog) -> None:
        self.authority = authority
        self.audit_log = audit_log

    def evaluate(
        self,
        *,
        token: str,
        candidate_hash: str,
        generation: str,
        trusted_evaluator: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        approval = self.authority.verify(token)
        if approval.candidate_hash != candidate_hash or approval.generation != generation:
            raise HoldoutPermissionError("Approval is bound to another candidate or generation")
        consumed = {
            record.payload.get("token_id")
            for record in self.audit_log.records()
            if record.event == "HOLDOUT_EVALUATED"
        }
        if approval.token_id in consumed:
            raise HoldoutPermissionError("Holdout approval was already consumed")
        result = trusted_evaluator()
        public_result = _public_metrics(result)
        self.audit_log.append(
            "HOLDOUT_EVALUATED",
            {
                "token_id": approval.token_id,
                "candidate_hash": candidate_hash,
                "generation": generation,
                "metrics": public_result,
            },
        )
        return public_result


def _public_metrics(result: dict[str, Any]) -> dict[str, Any]:
    payload = json.dumps(result, sort_keys=True, separators=(",", ":"), default=str).encode()
    public = {
        key: result[key]
        for key in ("decision", "verdict", "passed")
        if key in result
    }
    public["evidence_hash"] = hashlib.sha256(payload).hexdigest()
    return public


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode().rstrip("=")


def _decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
