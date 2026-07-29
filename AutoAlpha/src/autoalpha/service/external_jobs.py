from __future__ import annotations

from typing import Any

from autoalpha.service.store import ServiceStore

EXTERNAL_JOB_STATUS_MAP = {
    "READY": "EXTERNAL_READY",
    "WAITING": "EXTERNAL_READY",
    "QUEUED": "EXTERNAL_READY",
    "RUNNING": "EXTERNAL_RUNNING",
    "STOPPING": "EXTERNAL_STOP_REQUESTED",
    "PAUSING": "EXTERNAL_PAUSE_REQUESTED",
    "PAUSED": "EXTERNAL_PAUSED",
    "STOPPED": "EXTERNAL_PAUSED",
    "COMPLETED": "EXTERNAL_COMPLETED",
    "FAILED": "EXTERNAL_FAILED",
    "CANCELLED": "EXTERNAL_CANCELLED",
}


def external_job_id(system: str, task_id: str) -> str:
    normalized = system.strip().casefold().replace("_", "-")
    return f"external-{normalized}-{task_id}"


def mirror_external_research_job(
    store: ServiceStore,
    *,
    system: str,
    task: dict[str, Any],
    queue: str,
    job_type: str,
    progress_current: int,
    progress_total: int,
    event: str,
    message: str,
) -> dict[str, Any]:
    """Mirror an independently managed research task into the system job center."""

    task_id = str(task["task_id"])
    job_id = external_job_id(system, task_id)
    payload = {
        "protocol": "AUTOALPHA_EXTERNAL_RESEARCH_JOB_MIRROR_V1",
        "external_system": system,
        "external_task_id": task_id,
        "external_status": task.get("status"),
        "external_phase": task.get("phase"),
        "source": "external_research_task",
    }
    try:
        job = store.system_job(job_id)
    except KeyError:
        job = store.enqueue_system_job(
            job_id=job_id,
            queue=queue,
            job_type=job_type,
            payload=payload,
            priority=70,
            resource_group=f"external-{system.casefold()}",
            max_workers=1,
            progress_total=max(0, int(progress_total)),
            max_attempts=1,
        )

    external_status = str(task.get("status") or "READY")
    mirrored_status = EXTERNAL_JOB_STATUS_MAP.get(external_status, f"EXTERNAL_{external_status}")
    error = task.get("last_error")
    checkpoint = {
        **payload,
        "name": task.get("name"),
        "market": task.get("market"),
        "factor_count": task.get("factor_count"),
        "snapshot_hash": task.get("snapshot_hash"),
        "qualification_status": task.get("qualification_status"),
        "best_candidate_id": task.get("best_candidate_id")
        or task.get("best_experiment_id"),
        "updated_at": task.get("updated_at"),
    }
    result = checkpoint if mirrored_status in {"EXTERNAL_COMPLETED", "EXTERNAL_FAILED"} else None
    update: dict[str, Any] = {
        "status": mirrored_status,
        "progress_current": max(0, int(progress_current)),
        "progress_total": max(0, int(progress_total)),
        "checkpoint": checkpoint,
        "heartbeat_at": task.get("updated_at"),
        "error": str(error) if error else None,
    }
    if result is not None:
        update["result"] = result
        update["finished_at"] = task.get("updated_at")
    mirrored = store.update_system_job(job["job_id"], **update)
    store.append_system_job_log(
        job["job_id"],
        level="ERROR" if mirrored_status == "EXTERNAL_FAILED" else "INFO",
        event=event,
        message=message,
        payload={
            "external_system": system,
            "external_task_id": task_id,
            "external_status": external_status,
            "mirrored_status": mirrored_status,
            "phase": task.get("phase"),
            "progress_current": update["progress_current"],
            "progress_total": update["progress_total"],
        },
    )
    return mirrored
