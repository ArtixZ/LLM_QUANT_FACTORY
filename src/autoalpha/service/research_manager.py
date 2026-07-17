from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path
from typing import Any

from autoalpha.config import ResearchConfig
from autoalpha.data.execution_basis import inspect_execution_data_basis
from autoalpha.data.workspace import inspect_data_workspace
from autoalpha.service.research_protocol import (
    panel_validation_fold_capacity,
    protocol_blockers,
    protocol_data_blockers,
    task_research_config,
)
from autoalpha.service.store import ServiceStore
from autoalpha.service.worker import ContinuousResearchWorker, SecretVault


class ResearchTaskManager:
    """Own independent continuous workers while sharing the central research catalog."""

    def __init__(
        self,
        store: ServiceStore,
        vault: SecretVault,
        *,
        config_path: Path,
        artifact_root: Path,
        maximum_concurrent_iterations: int | None = None,
    ) -> None:
        self.store = store
        self.vault = vault
        self.config_path = config_path
        self.artifact_root = artifact_root
        configured = maximum_concurrent_iterations or int(
            os.getenv("AUTOALPHA_MAX_CONCURRENT_RESEARCH", "2")
        )
        self.maximum_concurrent_iterations = min(max(configured, 1), 8)
        self._iteration_slots = asyncio.Semaphore(self.maximum_concurrent_iterations)
        self._workers: dict[str, ContinuousResearchWorker] = {}
        self._control_lock = asyncio.Lock()

    @property
    def alive(self) -> bool:
        return any(worker.alive for worker in self._workers.values())

    def worker_alive(self, task_id: str) -> bool:
        worker = self._workers.get(task_id)
        return bool(worker and worker.alive)

    def active_task_ids(self) -> list[str]:
        return sorted(task_id for task_id, worker in self._workers.items() if worker.alive)

    def readiness(self, task_id: str) -> dict[str, Any]:
        task = self.store.research_task(task_id)
        if task is None:
            raise KeyError(f"Research task not found: {task_id}")
        config = ResearchConfig.from_toml(self.config_path)
        blockers: list[str] = []
        snapshot_changed = False
        if task["market"] != "CN_A":
            blockers.append("当前研究评估器仅支持 A 股；目标市场数据适配器尚未接入")
        if not task.get("snapshot_hash"):
            blockers.append("数据快照尚未建立")
        start = str(task.get("data_start") or "")
        end = str(task.get("data_end") or "")
        protocol = task.get("protocol") or {}
        if not start or not end:
            blockers.append("数据覆盖起止日尚未确定")
        elif not protocol:
            blockers.append("任务级研究切分尚未配置")
        else:
            blockers.extend(protocol_blockers(protocol, data_start=start, data_end=end))
        try:
            workspace = inspect_data_workspace(Path(task["data_path"]))
        except (FileNotFoundError, RuntimeError, TypeError, ValueError, OSError) as error:
            blockers.append(f"数据工作区不可用：{type(error).__name__}: {error}")
            workspace = None
        if workspace is not None and start and end:
            current_hash = hashlib.sha256(
                f"{task['market']}|{workspace.fingerprint}|{start}|{end}".encode()
            ).hexdigest()
            if current_hash != task.get("snapshot_hash"):
                snapshot_changed = True
        fold_capacity = None
        if workspace is not None and protocol:
            blockers.extend(
                protocol_data_blockers(protocol, Path(workspace.panel_path))
            )
            if config.strategy_evaluation.enabled:
                execution_basis = inspect_execution_data_basis(Path(workspace.panel_path))
                if not execution_basis.capital_ledger_proxy_ready:
                    blockers.append(
                        "A股策略晋级需要非PIT交易代理数据："
                        + "；".join(execution_basis.proxy_blockers)
                    )
            fold_capacity = panel_validation_fold_capacity(
                protocol, Path(workspace.panel_path)
            )
        task_config = None
        if not blockers:
            task_config = task_research_config(
                config, protocol, task_id=str(task["task_id"])
            )
        return {
            "runnable": not blockers,
            "blockers": blockers,
            "snapshot_changed": snapshot_changed,
            "maximum_concurrent_iterations": self.maximum_concurrent_iterations,
            "protocol": protocol,
            "protocol_hash": task.get("protocol_hash"),
            "protocol_revision": task.get("protocol_revision", 1),
            "walk_forward_capacity": fold_capacity,
            "public_range": (
                {
                    "start": task_config.splits.train.start.isoformat(),
                    "end": task_config.splits.validation.end.isoformat(),
                }
                if task_config
                else None
            ),
            "hidden_test_range": {
                "start": protocol.get("holdout_start"),
                "end": protocol.get("holdout_end"),
                "feedback": "categorical_only",
            },
        }

    async def start(self, task_id: str) -> dict[str, Any]:
        async with self._control_lock:
            readiness = self.readiness(task_id)
            if not readiness["runnable"]:
                raise RuntimeError("；".join(readiness["blockers"]))
            worker = self._worker(task_id)
            return await worker.start()

    async def stop(self, task_id: str) -> dict[str, Any]:
        async with self._control_lock:
            worker = self._workers.get(task_id)
            if worker is None:
                if task_id == "legacy-ashare":
                    return self.store.update_state(
                        state="STOPPED", phase="STOPPED", stop_requested=0
                    )
                return self.store.update_research_task_state(
                    task_id, state="STOPPED", phase="STOPPED", stop_requested=0
                )
            return await worker.stop()

    async def run_genesis_baseline(self, task_id: str) -> dict[str, Any]:
        readiness = self.readiness(task_id)
        if not readiness["runnable"]:
            raise RuntimeError("；".join(readiness["blockers"]))
        return await self._worker(task_id).run_genesis_baseline()

    async def run_codex_baseline(self, task_id: str) -> dict[str, Any]:
        readiness = self.readiness(task_id)
        if not readiness["runnable"]:
            raise RuntimeError("；".join(readiness["blockers"]))
        return await self._worker(task_id).run_codex_baseline()

    async def restore(self) -> None:
        if not self.vault.configured():
            return
        for task in self.store.research_tasks():
            if task["task_id"] == "legacy-ashare":
                state = self.store.state()
                should_restore = state["state"] in {"RUNNING", "RETRYING"}
            else:
                if task["status"] == "STOPPING":
                    self.store.update_research_task_state(
                        str(task["task_id"]),
                        state="STOPPED",
                        phase="STOPPED",
                        stop_requested=0,
                    )
                    continue
                should_restore = task["status"] in {"RUNNING", "RETRYING"}
            if not should_restore:
                continue
            readiness = self.readiness(str(task["task_id"]))
            if readiness["runnable"]:
                await self.start(str(task["task_id"]))
            elif task["task_id"] != "legacy-ashare":
                self.store.update_research_task_state(
                    str(task["task_id"]),
                    state="PAUSED_FAILURE",
                    phase="BLOCKED",
                    stop_requested=0,
                    last_error="；".join(readiness["blockers"]),
                )

    async def shutdown(self) -> None:
        workers = list(self._workers.values())
        await asyncio.gather(*(worker.shutdown() for worker in workers), return_exceptions=True)

    def _worker(self, task_id: str) -> ContinuousResearchWorker:
        worker = self._workers.get(task_id)
        if worker is not None:
            return worker
        if self.store.research_task(task_id) is None:
            raise KeyError(f"Research task not found: {task_id}")
        worker = ContinuousResearchWorker(
            self.store,
            self.vault,
            config_path=self.config_path,
            artifact_root=self.artifact_root / task_id,
            task_id=task_id,
            iteration_semaphore=self._iteration_slots,
        )
        self._workers[task_id] = worker
        return worker
