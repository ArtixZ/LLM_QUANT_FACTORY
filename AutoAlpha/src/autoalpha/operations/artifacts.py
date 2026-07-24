from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Artifact:
    artifact_id: str
    kind: str
    content_hash: str
    created_at_utc: str
    owner: str
    source_ids: tuple[str, ...]
    metadata: dict[str, Any]
    payload_path: str


class ArtifactRegistry:
    def __init__(self, root: Path) -> None:
        self.root = root

    def publish(
        self,
        kind: str,
        payload: bytes,
        *,
        owner: str,
        source_ids: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> Artifact:
        if not kind.strip() or not owner.strip():
            raise ValueError("Artifact kind and owner are required")
        content_hash = hashlib.sha256(payload).hexdigest()
        artifact_id = f"{kind}-{content_hash[:20]}"
        directory = self.root / kind / content_hash[:2] / content_hash
        payload_path = directory / "payload.bin"
        manifest_path = directory / "manifest.json"
        if manifest_path.exists():
            return self.get(artifact_id)
        directory.mkdir(parents=True, exist_ok=True)
        _atomic_write(payload_path, payload)
        artifact = Artifact(
            artifact_id=artifact_id,
            kind=kind,
            content_hash=content_hash,
            created_at_utc=datetime.now(UTC).isoformat(),
            owner=owner,
            source_ids=source_ids,
            metadata=metadata or {},
            payload_path=str(payload_path.relative_to(self.root)),
        )
        _atomic_write(
            manifest_path,
            (json.dumps(asdict(artifact), sort_keys=True, indent=2) + "\n").encode(),
        )
        return artifact

    def get(self, artifact_id: str) -> Artifact:
        kind, short_hash = artifact_id.rsplit("-", 1)
        matches = list((self.root / kind / short_hash[:2]).glob(f"{short_hash}*/manifest.json"))
        if len(matches) != 1:
            raise KeyError(f"Unknown or ambiguous artifact {artifact_id}")
        payload = json.loads(matches[0].read_text(encoding="utf-8"))
        payload["source_ids"] = tuple(payload["source_ids"])
        artifact = Artifact(**payload)
        data = (self.root / artifact.payload_path).read_bytes()
        if hashlib.sha256(data).hexdigest() != artifact.content_hash:
            raise RuntimeError(f"Artifact payload failed integrity verification: {artifact_id}")
        return artifact

    def read(self, artifact_id: str) -> bytes:
        artifact = self.get(artifact_id)
        return (self.root / artifact.payload_path).read_bytes()


def _atomic_write(path: Path, payload: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
