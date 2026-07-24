from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AuditRecord:
    sequence: int
    timestamp_utc: str
    event: str
    payload: dict[str, Any]
    previous_hash: str
    record_hash: str


class HashChainAuditLog:
    def __init__(self, path: Path) -> None:
        self.path = path

    def records(self) -> list[AuditRecord]:
        if not self.path.exists():
            return []
        return [
            AuditRecord(**json.loads(line))
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def append(self, event: str, payload: dict[str, Any]) -> AuditRecord:
        records = self.records()
        previous_hash = records[-1].record_hash if records else "0" * 64
        body = {
            "sequence": len(records) + 1,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "event": event,
            "payload": payload,
            "previous_hash": previous_hash,
        }
        record_hash = _hash_body(body)
        record = AuditRecord(**body, record_hash=record_hash)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True) + "\n")
        return record

    def verify(self) -> None:
        previous_hash = "0" * 64
        for expected_sequence, record in enumerate(self.records(), start=1):
            if record.sequence != expected_sequence or record.previous_hash != previous_hash:
                raise RuntimeError("Audit log sequence or hash chain is broken")
            body = asdict(record)
            record_hash = body.pop("record_hash")
            if _hash_body(body) != record_hash:
                raise RuntimeError(f"Audit record {record.sequence} was modified")
            previous_hash = record.record_hash


def _hash_body(body: dict[str, Any]) -> str:
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
