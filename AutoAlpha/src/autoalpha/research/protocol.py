from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from autoalpha.config import ResearchConfig


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ProtocolManifest:
    generation: str
    config_fingerprint: str
    config: dict[str, Any]
    files: dict[str, str]
    created_at_utc: str

    @classmethod
    def create(
        cls,
        config: ResearchConfig,
        files: Iterable[Path],
        *,
        root: Path,
        data_checksums: dict[str, str] | None = None,
        created_at: datetime | None = None,
    ) -> ProtocolManifest:
        root = root.resolve()
        checksums: dict[str, str] = {}
        for path in sorted((path.resolve() for path in files), key=str):
            try:
                relative = path.relative_to(root)
            except ValueError as error:
                raise ValueError(f"Manifest file is outside root: {path}") from error
            checksums[relative.as_posix()] = sha256_file(path)
        timestamp = created_at or datetime.now(UTC)
        return cls(
            generation=config.generation,
            config_fingerprint=config.fingerprint(data_checksums=data_checksums),
            config=config.canonical_dict(),
            files=checksums,
            created_at_utc=timestamp.astimezone(UTC).isoformat(),
        )

    @classmethod
    def read(cls, path: Path) -> ProtocolManifest:
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(self), sort_keys=True, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def verify(self, root: Path, config: ResearchConfig, data_checksums: dict[str, str]) -> None:
        root = root.resolve()
        if config.generation != self.generation:
            raise RuntimeError(
                f"Research generation changed: {self.generation!r} -> {config.generation!r}"
            )
        current_fingerprint = config.fingerprint(data_checksums=data_checksums)
        if current_fingerprint != self.config_fingerprint:
            raise RuntimeError("Research configuration or data checksums changed")
        mismatches = {
            name: (expected, sha256_file(root / name) if (root / name).is_file() else None)
            for name, expected in self.files.items()
            if not (root / name).is_file() or sha256_file(root / name) != expected
        }
        if mismatches:
            names = ", ".join(sorted(mismatches))
            raise RuntimeError(f"Frozen protocol files changed or disappeared: {names}")
