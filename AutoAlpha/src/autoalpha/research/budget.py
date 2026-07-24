from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

ExperimentStatus = Literal["RESERVED", "RETAINED", "REJECTED", "CRASHED", "CANCELLED"]


class BudgetExceeded(PermissionError):
    """The research generation or factor family exhausted its experiment budget."""


@dataclass(frozen=True)
class ExperimentRecord:
    experiment_id: str
    generation: str
    family: str
    status: ExperimentStatus
    created_at_utc: str
    admission_decision: str | None = None
    reason: str = ""


class ExperimentBudgetLedger:
    def __init__(
        self,
        path: Path,
        *,
        max_generation: int,
        max_family: int,
    ) -> None:
        if max_generation <= 0 or max_family <= 0:
            raise ValueError("Invalid experiment budget")
        self.path = path
        self.max_generation = max_generation
        self.max_family = max_family

    def records(self) -> list[ExperimentRecord]:
        with self._lock():
            return self._read_records()

    def _read_records(self) -> list[ExperimentRecord]:
        if not self.path.exists():
            return []
        records = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(ExperimentRecord(**json.loads(line)))
        return records

    def reserve(self, experiment_id: str, generation: str, family: str) -> ExperimentRecord:
        with self._lock():
            records = self._read_records()
            if any(record.experiment_id == experiment_id for record in records):
                raise ValueError(f"Duplicate experiment_id: {experiment_id}")
            reservations = [record for record in records if record.status == "RESERVED"]
            generation_count = sum(record.generation == generation for record in reservations)
            family_count = sum(
                record.generation == generation and record.family == family
                for record in reservations
            )
            if generation_count >= self.max_generation:
                raise BudgetExceeded(f"Generation {generation!r} exhausted its budget")
            if family_count >= self.max_family:
                raise BudgetExceeded(f"Family {family!r} exhausted its budget")
            record = ExperimentRecord(
                experiment_id=experiment_id,
                generation=generation,
                family=family,
                status="RESERVED",
                created_at_utc=datetime.now(UTC).isoformat(),
            )
            self._append_unlocked(record)
            return record

    def close(
        self,
        reservation: ExperimentRecord,
        *,
        admission_decision: str,
        reason: str = "",
    ) -> ExperimentRecord:
        if reservation.status != "RESERVED":
            raise ValueError("Only a reservation can be closed")
        if admission_decision not in {
            "REJECTED",
            "RESEARCH",
            "APPROVED_FOR_PAPER",
            "APPROVED_FOR_PRODUCTION",
        }:
            raise ValueError(f"Unknown admission decision: {admission_decision}")
        status: ExperimentStatus = "REJECTED" if admission_decision == "REJECTED" else "RETAINED"
        with self._lock():
            decision_id = f"{reservation.experiment_id}:decision"
            if any(record.experiment_id == decision_id for record in self._read_records()):
                raise ValueError(f"Reservation already closed: {reservation.experiment_id}")
            record = ExperimentRecord(
                experiment_id=decision_id,
                generation=reservation.generation,
                family=reservation.family,
                status=status,
                created_at_utc=datetime.now(UTC).isoformat(),
                admission_decision=admission_decision,
                reason=reason,
            )
            self._append_unlocked(record)
            return record

    def _append_unlocked(self, record: ExperimentRecord) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    @contextmanager
    def _lock(self) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        with lock_path.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
