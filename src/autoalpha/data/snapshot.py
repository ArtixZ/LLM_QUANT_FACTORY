from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from autoalpha.data.contracts import TableContract
from autoalpha.research.protocol import sha256_file


@dataclass(frozen=True)
class SnapshotTable:
    name: str
    contract_version: str
    contract_fingerprint: str
    file: str
    sha256: str
    rows: int


@dataclass(frozen=True)
class DatasetSnapshot:
    snapshot_id: str
    created_at_utc: str
    source: str
    tables: tuple[SnapshotTable, ...]
    manifest_hash: str

    @classmethod
    def read(cls, path: Path) -> DatasetSnapshot:
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["tables"] = tuple(SnapshotTable(**table) for table in raw["tables"])
        return cls(**raw)


class SnapshotStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def write(
        self,
        snapshot_id: str,
        tables: Iterable[tuple[pd.DataFrame, TableContract]],
        *,
        source: str,
    ) -> DatasetSnapshot:
        target = self.root / snapshot_id
        if target.exists():
            raise FileExistsError(f"Snapshot already exists: {target}")
        staging = self.root / f".{snapshot_id}.{uuid.uuid4().hex}.staging"
        staging.mkdir(parents=True, exist_ok=False)
        table_records: list[SnapshotTable] = []
        try:
            for frame, contract in tables:
                contract.validate(frame).raise_for_errors()
                relative = f"{contract.name}.parquet"
                output = staging / relative
                frame.to_parquet(output, index=False, compression="zstd")
                table_records.append(
                    SnapshotTable(
                        name=contract.name,
                        contract_version=contract.version,
                        contract_fingerprint=contract.fingerprint,
                        file=relative,
                        sha256=sha256_file(output),
                        rows=len(frame),
                    )
                )
            manifest_body = {
                "snapshot_id": snapshot_id,
                "created_at_utc": datetime.now(UTC).isoformat(),
                "source": source,
                "tables": [asdict(table) for table in table_records],
            }
            manifest_hash = _manifest_hash(manifest_body)
            snapshot = DatasetSnapshot(
                snapshot_id=snapshot_id,
                created_at_utc=manifest_body["created_at_utc"],
                source=source,
                tables=tuple(table_records),
                manifest_hash=manifest_hash,
            )
            (staging / "snapshot.json").write_text(
                json.dumps(asdict(snapshot), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            self.root.mkdir(parents=True, exist_ok=True)
            staging.rename(target)
            return snapshot
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def verify(self, snapshot_id: str) -> DatasetSnapshot:
        directory = self.root / snapshot_id
        snapshot = DatasetSnapshot.read(directory / "snapshot.json")
        body = {
            "snapshot_id": snapshot.snapshot_id,
            "created_at_utc": snapshot.created_at_utc,
            "source": snapshot.source,
            "tables": [asdict(table) for table in snapshot.tables],
        }
        if _manifest_hash(body) != snapshot.manifest_hash:
            raise RuntimeError("Snapshot manifest was modified")
        for table in snapshot.tables:
            if sha256_file(directory / table.file) != table.sha256:
                raise RuntimeError(f"Snapshot table was modified: {table.name}")
        return snapshot


def _manifest_hash(body: dict) -> str:
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
