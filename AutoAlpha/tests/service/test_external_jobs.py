from __future__ import annotations

from autoalpha.service.external_jobs import external_job_id, mirror_external_research_job
from autoalpha.service.store import ServiceStore


def test_external_research_task_mirrors_into_system_job_center(tmp_path) -> None:
    store = ServiceStore(tmp_path / "service.sqlite3")
    task = {
        "task_id": "combine-demo",
        "name": "Demo combine task",
        "market": "CN_A",
        "status": "READY",
        "phase": "WAITING",
        "factor_count": 50,
        "snapshot_hash": "abc123",
        "iteration": 0,
        "budget": {"maximum_experiments": 100},
    }

    created = mirror_external_research_job(
        store,
        system="autocombine",
        task=task,
        queue="autocombine",
        job_type="external_autocombine_research",
        progress_current=0,
        progress_total=100,
        event="COMBINE_TASK_CREATED",
        message="created",
    )

    assert created["job_id"] == external_job_id("autocombine", "combine-demo")
    assert created["queue"] == "autocombine"
    assert created["job_type"] == "external_autocombine_research"
    assert created["status"] == "EXTERNAL_READY"
    assert created["progress_total"] == 100
    assert created["checkpoint"]["external_task_id"] == "combine-demo"

    running = mirror_external_research_job(
        store,
        system="autocombine",
        task={**task, "status": "RUNNING", "phase": "SEARCHING", "iteration": 10},
        queue="autocombine",
        job_type="external_autocombine_research",
        progress_current=10,
        progress_total=100,
        event="COMBINE_TASK_STARTED",
        message="started",
    )

    assert running["status"] == "EXTERNAL_RUNNING"
    assert running["progress_current"] == 10
    logs = store.system_job_logs(running["job_id"])
    assert logs[0]["event"] == "COMBINE_TASK_STARTED"
    assert any(log["event"] == "ENQUEUED" for log in logs)
