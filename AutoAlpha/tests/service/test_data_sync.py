from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from autoalpha.service import data_sync
from autoalpha.service.data_sync import DataSyncWorker
from autoalpha.service.store import ServiceStore


def _worker(tmp_path: Path) -> DataSyncWorker:
    store = ServiceStore(tmp_path / "service.sqlite3")
    return DataSyncWorker(store, project_root=tmp_path, is_busy=lambda: False)


def _sync_result(**overrides: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "download_returncode": 0,
        "universe": "DOW_30_V1",
        "download_summary": {},
        "panel_rebuilt": True,
        "panel_metadata": {"rows": 1},
        "panel_error": None,
        "stale_symbols": {},
    }
    result.update(overrides)
    return result


def test_rebuild_panel_surfaces_stale_symbols_from_the_audit(tmp_path, monkeypatch) -> None:
    """The audit passes with staleness as a warning; the worker must not drop it."""
    report = {"passed": True, "stale_symbols": {"GOOGL": "2026-08-07"}}
    metadata = {"rows": 10}

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        command = argv[argv.index("multifactor_us.data") + 1]
        stdout = json.dumps(report if command == "audit" else metadata)
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(data_sync.subprocess, "run", fake_run)

    result = _worker(tmp_path)._rebuild_panel(tmp_path, tmp_path / "panel")

    assert result["returncode"] == 0
    assert result["metadata"] == metadata
    assert result["stale_symbols"] == {"GOOGL": "2026-08-07"}


def test_completed_sync_with_stale_symbols_warns_and_names_them(tmp_path) -> None:
    worker = _worker(tmp_path)

    status = worker._apply_sync_result(
        trigger="manual",
        result=_sync_result(stale_symbols={"GOOGL": "2026-08-07", "XOM": "2026-08-07"}),
    )

    assert status["state"] == "COMPLETED"
    assert status["stale_symbols"] == {"GOOGL": "2026-08-07", "XOM": "2026-08-07"}
    event = worker.store.events()[-1]
    assert event["event"] == "MARKET_DATA_SYNC_COMPLETED"
    assert event["level"] == "WARN"
    assert "GOOGL" in event["message"]
    assert "XOM" in event["message"]


def test_clean_completed_sync_stays_info(tmp_path) -> None:
    worker = _worker(tmp_path)

    status = worker._apply_sync_result(trigger="manual", result=_sync_result())

    assert status["state"] == "COMPLETED"
    event = worker.store.events()[-1]
    assert event["level"] == "INFO"
    assert "stale" not in event["message"]
