from __future__ import annotations

import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from autoalpha.research.batch_reevaluation import apply_batch_multiple_testing
from autoalpha.service.batch_engine import (
    BatchFactorOutcome,
    MassiveBatchConfig,
    MassiveVectorBatchEngine,
)
from autoalpha.service.batch_store import BatchBacktestStore


class MassiveBatchRunner:
    """Own one isolated, resumable massive factor job at a time."""

    def __init__(
        self,
        store: BatchBacktestStore,
        artifact_root: Path,
        *,
        config_type: type[MassiveBatchConfig] = MassiveBatchConfig,
        engine_type: type[MassiveVectorBatchEngine] = MassiveVectorBatchEngine,
        worker_prefix: str = "massive-vector-factor",
    ) -> None:
        self.store = store
        self.artifact_root = artifact_root
        self.config_type = config_type
        self.engine_type = engine_type
        self.worker_prefix = worker_prefix
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._pause = threading.Event()
        self._thread: threading.Thread | None = None
        self._active_job_id: str | None = None
        self.status_callback = None

    @property
    def active_job_id(self) -> str | None:
        with self._lock:
            return self._active_job_id if self._thread and self._thread.is_alive() else None

    def start(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            if self._thread and self._thread.is_alive():
                raise RuntimeError(f"Batch job already running: {self._active_job_id}")
            job = self.store.job(job_id)
            if job["status"] == "COMPLETED":
                raise RuntimeError("Completed jobs are immutable")
            self._pause.clear()
            self._active_job_id = job_id
            self._thread = threading.Thread(
                target=self._run,
                args=(job_id,),
                name=f"massive-batch-{job_id}",
                daemon=True,
            )
            self._thread.start()
        return self.store.job(job_id)

    def pause(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            if self._active_job_id != job_id or not self._thread or not self._thread.is_alive():
                raise RuntimeError("This batch job is not currently running")
            self._pause.set()
        self.store.update_job(job_id, status="PAUSING", phase="DRAINING_WORKERS")
        self.store.append_event(
            job_id,
            "INFO",
            "PAUSE_REQUESTED",
            "Pause requested; active factor threads will checkpoint before stopping.",
        )
        self._notify(job_id, "PAUSE_REQUESTED")
        return self.store.job(job_id)

    def _run(self, job_id: str) -> None:
        try:
            job = self.store.job(job_id)
            config = self.config_type.from_dict(job["config"])
            started_at = job.get("started_at") or datetime.now(UTC).isoformat()
            self.store.update_job(
                job_id,
                status="RUNNING",
                phase="DATA_LOADING",
                started_at=started_at,
                finished_at=None,
                last_error=None,
            )
            self._notify(job_id, "JOB_RUNNING")
            self.store.append_event(
                job_id,
                "INFO",
                "DATA_LOADING",
                "Loading one shared read-only panel for all factor workers.",
                {"workers": config.workers, "data_path": str(config.data_path)},
            )
            engine = self.engine_type(
                config,
                self.artifact_root / job_id,
                factor_family_size=int(job["factor_count"]),
            )
            self.store.update_job(
                job_id,
                status="RUNNING",
                phase="FACTOR_EVALUATION",
                data_fingerprint=engine.workspace.fingerprint,
            )
            pending = self.store.pending_factors(job_id)
            self.store.append_event(
                job_id,
                "INFO",
                "FACTOR_EVALUATION_STARTED",
                f"Evaluating {len(pending)} pending factors with {config.workers} threads.",
                {
                    "monte_carlo_samples_per_factor": config.monte_carlo_samples,
                    "large_windows": len(engine.windows),
                },
            )
            self._evaluate_pending(job_id, engine, pending, config.workers)
            if self._pause.is_set():
                self.store.update_job(job_id, status="PAUSED", phase="CHECKPOINTED")
                self.store.append_event(
                    job_id,
                    "INFO",
                    "JOB_PAUSED",
                    "All active factor results were committed; the remaining queue is resumable.",
                )
                self._notify(job_id, "JOB_PAUSED")
                return

            self.store.update_job(job_id, status="RUNNING", phase="MULTIPLE_TESTING")
            successful = self.store.successful_metrics(job_id)
            if successful:
                adjusted, family_pbo = apply_batch_multiple_testing(successful, alpha=0.05)
                self.store.update_adjusted_metrics(job_id, adjusted)
            else:
                family_pbo = 1.0
            final = self.store.job(job_id)
            self.store.update_job(
                job_id,
                status="COMPLETED",
                phase="COMPLETED",
                finished_at=datetime.now(UTC).isoformat(),
            )
            self.store.append_event(
                job_id,
                "INFO",
                "JOB_COMPLETED",
                f"Completed {final['completed_count']} factors; {final['failed_count']} failed.",
                {
                    "family_probability_backtest_overfitting": family_pbo,
                    "successful_factors": len(successful),
                },
            )
            self._notify(job_id, "JOB_COMPLETED")
        except Exception as error:  # noqa: BLE001
            self.store.update_job(
                job_id,
                status="FAILED",
                phase="FAILED",
                finished_at=datetime.now(UTC).isoformat(),
                last_error=f"{type(error).__name__}: {error}",
            )
            self.store.append_event(
                job_id,
                "ERROR",
                "JOB_FAILED",
                f"{type(error).__name__}: {error}",
            )
            self._notify(job_id, "JOB_FAILED")
        finally:
            with self._lock:
                self._active_job_id = None

    def _notify(self, job_id: str, event: str) -> None:
        callback = self.status_callback
        if not callable(callback):
            return
        try:
            callback(self.store.job(job_id), event)
        except Exception:
            return

    def _evaluate_pending(
        self,
        job_id: str,
        engine: MassiveVectorBatchEngine,
        pending: list[dict[str, Any]],
        workers: int,
    ) -> None:
        records = iter(pending)
        active: dict[Future[BatchFactorOutcome], dict[str, Any]] = {}
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix=self.worker_prefix
        ) as pool:
            for _ in range(workers):
                if not self._submit_next(job_id, engine, records, pool, active):
                    break
            while active:
                done, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in done:
                    record = active.pop(future)
                    factor_id = str(record["factor_id"])
                    try:
                        outcome = future.result()
                        self.store.complete_factor(
                            job_id,
                            factor_id,
                            elapsed_seconds=outcome.elapsed_seconds,
                            metrics=outcome.metrics,
                            monte_carlo=outcome.monte_carlo,
                            curve_path=outcome.curve_path,
                            monte_carlo_path=outcome.monte_carlo_path,
                            windows=outcome.windows,
                            robustness=outcome.robustness,
                        )
                        self.store.append_event(
                            job_id,
                            "INFO",
                            "FACTOR_COMPLETED",
                            f"{record['name']} completed in {outcome.elapsed_seconds:.1f}s.",
                            {"factor_id": factor_id},
                        )
                        self._notify(job_id, "FACTOR_COMPLETED")
                    except Exception as error:  # noqa: BLE001
                        elapsed = 0.0
                        self.store.fail_factor(
                            job_id,
                            factor_id,
                            elapsed_seconds=elapsed,
                            error=f"{type(error).__name__}: {error}",
                        )
                        self.store.append_event(
                            job_id,
                            "ERROR",
                            "FACTOR_FAILED",
                            f"{record['name']}: {type(error).__name__}: {error}",
                            {"factor_id": factor_id},
                        )
                        self._notify(job_id, "FACTOR_FAILED")
                    if not self._pause.is_set():
                        self._submit_next(job_id, engine, records, pool, active)

    def _submit_next(
        self,
        job_id: str,
        engine: MassiveVectorBatchEngine,
        records: Any,
        pool: ThreadPoolExecutor,
        active: dict[Future[BatchFactorOutcome], dict[str, Any]],
    ) -> bool:
        try:
            record = next(records)
        except StopIteration:
            return False
        factor_id = str(record["factor_id"])
        self.store.mark_factor_running(job_id, factor_id)
        active[pool.submit(engine.run_factor, job_id, record)] = record
        return True
