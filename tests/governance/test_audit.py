from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoalpha.governance.audit import HashChainAuditLog


def test_hash_chain_detects_modified_history(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    log = HashChainAuditLog(path)
    log.append("FIRST", {"value": 1})
    log.append("SECOND", {"value": 2})
    log.verify()

    records = path.read_text(encoding="utf-8").splitlines()
    first = json.loads(records[0])
    first["payload"]["value"] = 999
    records[0] = json.dumps(first)
    path.write_text("\n".join(records) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="modified"):
        log.verify()
