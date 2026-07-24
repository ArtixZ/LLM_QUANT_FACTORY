from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from autoalpha.config import ResearchConfig
from autoalpha.research.protocol import ProtocolManifest

CONFIG_PATH = Path(__file__).parents[2] / "config" / "research.toml"


def test_protocol_manifest_detects_config_data_and_file_changes(tmp_path: Path) -> None:
    config = ResearchConfig.from_toml(CONFIG_PATH)
    frozen = tmp_path / "evaluation.py"
    frozen.write_text("SCORE = 1\n", encoding="utf-8")
    manifest = ProtocolManifest.create(
        config,
        [frozen],
        root=tmp_path,
        data_checksums={"panel": "abc"},
        created_at=datetime(2026, 7, 15, tzinfo=UTC),
    )

    manifest.verify(tmp_path, config, {"panel": "abc"})
    with pytest.raises(RuntimeError, match="configuration or data"):
        manifest.verify(tmp_path, config, {"panel": "changed"})

    frozen.write_text("SCORE = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="protocol files"):
        manifest.verify(tmp_path, config, {"panel": "abc"})


def test_manifest_rejects_files_outside_root(tmp_path: Path) -> None:
    config = ResearchConfig.from_toml(CONFIG_PATH)
    outside = tmp_path.parent / "outside.py"
    outside.write_text("pass\n", encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="outside root"):
            ProtocolManifest.create(config, [outside], root=tmp_path)
    finally:
        outside.unlink()
