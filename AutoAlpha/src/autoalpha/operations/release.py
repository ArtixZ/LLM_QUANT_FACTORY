from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from autoalpha.governance.audit import HashChainAuditLog


@dataclass(frozen=True)
class Release:
    strategy_id: str
    artifact_id: str
    approver: str
    approval_id: str
    allocation_fraction: float
    previous_artifact_id: str | None


class ReleaseRegistry:
    def __init__(self, path: Path) -> None:
        self.audit = HashChainAuditLog(path)

    def current(self, strategy_id: str) -> Release | None:
        records = [
            record
            for record in self.audit.records()
            if record.event in {"PRODUCTION_RELEASE", "PRODUCTION_ROLLBACK"}
            and record.payload["strategy_id"] == strategy_id
        ]
        return Release(**records[-1].payload) if records else None

    def promote(
        self,
        strategy_id: str,
        artifact_id: str,
        *,
        approver: str,
        approval_id: str,
        allocation_fraction: float,
    ) -> Release:
        if not approver.strip() or not approval_id.strip():
            raise PermissionError("Human approver and approval record are required")
        if not 0 < allocation_fraction <= 1:
            raise ValueError("allocation_fraction must be in (0, 1]")
        previous = self.current(strategy_id)
        release = Release(
            strategy_id,
            artifact_id,
            approver,
            approval_id,
            allocation_fraction,
            previous.artifact_id if previous else None,
        )
        self.audit.append("PRODUCTION_RELEASE", asdict_release(release))
        return release

    def rollback(self, strategy_id: str, *, approver: str, approval_id: str) -> Release:
        current = self.current(strategy_id)
        if current is None or current.previous_artifact_id is None:
            raise RuntimeError("No previous production artifact is available")
        release = Release(
            strategy_id,
            current.previous_artifact_id,
            approver,
            approval_id,
            current.allocation_fraction,
            current.artifact_id,
        )
        self.audit.append("PRODUCTION_ROLLBACK", asdict_release(release))
        return release


def asdict_release(release: Release) -> dict[str, object]:
    return {
        "strategy_id": release.strategy_id,
        "artifact_id": release.artifact_id,
        "approver": release.approver,
        "approval_id": release.approval_id,
        "allocation_fraction": release.allocation_fraction,
        "previous_artifact_id": release.previous_artifact_id,
    }
