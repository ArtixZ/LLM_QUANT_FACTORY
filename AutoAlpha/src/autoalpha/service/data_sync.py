from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from autoalpha.data.tushare_catalog import DEFAULT_PRODUCT_IDS, resolve_products
from autoalpha.service.credentials import SystemCredentialStore
from autoalpha.service.store import ServiceStore


class DataSyncWorker:
    """Refresh raw/factor daily slices and rebuild only a fully covered research panel."""

    def __init__(
        self,
        store: ServiceStore,
        *,
        project_root: Path,
        is_busy: Callable[[], bool],
    ) -> None:
        self.store = store
        self.project_root = project_root
        self.is_busy = is_busy
        self.market_data_root = Path.home() / "MarketData" / "Ashare"
        self.token_store = SystemCredentialStore(
            service_name="com.autoalpha.tushare", account_name="market-data"
        )
        self._task: asyncio.Task[dict[str, Any]] | None = None
        self._scheduler: asyncio.Task[None] | None = None
        self._status: dict[str, Any] = {"state": "IDLE", "updated_at": None}

    @property
    def alive(self) -> bool:
        return self._task is not None and not self._task.done()

    def status(self) -> dict[str, Any]:
        settings = self.store.settings()
        market_data_root = Path(
            settings.get("market_data_root", str(self.market_data_root))
        ).expanduser()
        return {
            **self._status,
            "running": self.alive,
            "download_progress": _download_progress(market_data_root),
        }

    def token_configured(self) -> bool:
        return bool(os.getenv("TUSHARE_TOKEN") or self.token_store.get())

    def set_token(self, value: str) -> None:
        self.token_store.set(value)

    async def start(
        self,
        *,
        trigger: str,
        dataset_ids: list[str] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> dict[str, Any]:
        if self.alive:
            raise RuntimeError("A market-data sync is already running")
        if self.is_busy():
            raise RuntimeError("Stop research and manual backtests before refreshing market data")
        if not self.token_configured():
            raise RuntimeError("Tushare Token is not configured")
        selected = dataset_ids or _configured_product_ids(self.store.settings())
        resolve_products(selected)
        self._task = asyncio.create_task(
            self._run(trigger, selected, start_date=start_date, end_date=end_date),
            name="autoalpha-market-data-sync",
        )
        return self.status()

    async def start_scheduler(self) -> None:
        if self._scheduler is None or self._scheduler.done():
            self._scheduler = asyncio.create_task(
                self._schedule_loop(), name="autoalpha-market-data-scheduler"
            )

    async def shutdown(self) -> None:
        for task in (self._task, self._scheduler):
            if task and not task.done():
                task.cancel()
        await asyncio.gather(
            *(task for task in (self._task, self._scheduler) if task), return_exceptions=True
        )

    async def _schedule_loop(self) -> None:
        while True:
            try:
                await self._maybe_run_scheduled_sync()
            except Exception as error:
                self._status = {
                    "state": "SCHEDULER_ERROR",
                    "updated_at": _now(),
                    "message": f"{type(error).__name__}: {error}",
                }
            await asyncio.sleep(60)

    async def _maybe_run_scheduled_sync(self) -> None:
        settings = self.store.settings()
        if settings.get("data_auto_update_enabled", "false").casefold() != "true":
            return
        now = datetime.now(ZoneInfo("Asia/Shanghai"))
        scheduled_hour = int(settings.get("data_update_hour", "18"))
        completed_today = settings.get("last_data_sync_date") == now.date().isoformat()
        if now.hour < scheduled_hour or completed_today:
            return
        if self.alive or self.is_busy():
            return
        if not self.token_configured():
            self._status = {
                "state": "WAITING_TOKEN",
                "updated_at": _now(),
                "message": "Automatic data sync is waiting for a Tushare Token",
            }
            return
        await self.start(trigger="scheduled")

    async def _run(
        self,
        trigger: str,
        dataset_ids: list[str],
        *,
        start_date: date | None,
        end_date: date | None,
    ) -> dict[str, Any]:
        self._status = {
            "state": "RUNNING",
            "updated_at": _now(),
            "trigger": trigger,
            "dataset_ids": dataset_ids,
        }
        self.store.append_event(
            "action",
            "MARKET_DATA_SYNC_STARTED",
            "市场数据增量同步开始",
            "按已启用的数据产品拉取行情、估值、资金流和交易状态；覆盖检查后原子重建研究面板。",
            payload={"trigger": trigger, "mode": "NON_PIT_PROXY", "dataset_ids": dataset_ids},
        )
        try:
            result = await asyncio.to_thread(
                self._sync_blocking,
                dataset_ids,
                start_date=start_date,
                end_date=end_date,
            )
            sync_ok = result["download_returncode"] == 0
            panel_rebuilt = bool(result.get("panel_rebuilt"))
            status = (
                "COMPLETED"
                if sync_ok and panel_rebuilt
                else "MIGRATION_PENDING"
                if sync_ok
                else "DEGRADED"
            )
            self._status = {"state": status, "updated_at": _now(), **result}
            completed_date = datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()
            self.store.save_settings({"last_data_sync_date": completed_date})
            self.store.append_event(
                "delivery",
                "MARKET_DATA_SYNC_COMPLETED",
                "市场数据面板已更新" if panel_rebuilt else "市场原始层已更新",
                (
                    "原始行情与复权因子已通过覆盖检查，研究面板已原子替换。"
                    if panel_rebuilt
                    else result.get("migration_message", "下载器仍报告部分失败，未替换研究面板。")
                ),
                level="INFO" if sync_ok else "WARN",
                payload=self._status,
            )
            return self._status
        except Exception as error:
            self._status = {
                "state": "FAILED",
                "updated_at": _now(),
                "message": f"{type(error).__name__}: {error}",
            }
            self.store.append_event(
                "audit",
                "MARKET_DATA_SYNC_FAILED",
                "市场数据同步失败",
                self._status["message"],
                level="ERROR",
                payload={"trigger": trigger},
            )
            raise

    def _sync_blocking(
        self,
        dataset_ids: list[str],
        *,
        start_date: date | None,
        end_date: date | None,
    ) -> dict[str, Any]:
        token = os.getenv("TUSHARE_TOKEN") or self.token_store.get()
        if not token:
            raise RuntimeError("Tushare Token is not configured")
        settings = self.store.settings()
        market_data_root = (
            Path(settings.get("market_data_root", str(self.market_data_root)))
            .expanduser()
            .resolve()
        )
        cli = market_data_root / "sync_cli.py"
        if not cli.is_file():
            raise FileNotFoundError(f"Market-data downloader not found: {cli}")
        downloader_python = market_data_root / ".venv" / "bin" / "python"
        if not downloader_python.is_file():
            downloader_python = Path(sys.executable)
        env = {**os.environ, "TUSHARE_TOKEN": token}
        selected = resolve_products(dataset_ids)
        include_core = any(product.dataset_id == "core_market" for product in selected)
        feature_products = [
            product.dataset_id
            for product in selected
            if product.sync_strategy not in {"CORE_PIPELINE", "CATALOG"}
        ]
        download_summary: dict[str, Any] = {}
        stderr = ""
        core_returncode = 0
        if include_core:
            with tempfile.TemporaryFile(mode="w+") as out, tempfile.TemporaryFile(mode="w+") as err:
                download = subprocess.Popen(
                    [str(downloader_python), str(cli), "--adjustment", "both"],
                    cwd=market_data_root,
                    env=env,
                    text=True,
                    stdout=out,
                    stderr=err,
                )
                self._wait_for_process(download, market_data_root, state="RUNNING_CORE")
                out.seek(0)
                err.seek(0)
                stdout, stderr = out.read(), err.read()
            download_summary["core_market"] = _last_json_document(stdout)
            core_returncode = int(download.returncode or 0)

        feature_error = ""
        feature_returncode = 0
        if feature_products:
            feature_python = self.project_root / "AutoAlpha" / ".venv" / "bin" / "python"
            if not feature_python.is_file():
                feature_python = Path(sys.executable)
            feature_command = [
                str(feature_python),
                "-m",
                "autoalpha.data.tushare_feature_sync",
                "--root",
                str(market_data_root),
                "--datasets",
                ",".join(feature_products),
                "--start-date",
                (start_date.strftime("%Y%m%d") if start_date else _panel_start_date(settings)),
                "--end-date",
                (end_date.strftime("%Y%m%d") if end_date else "20991231"),
            ]
            with tempfile.TemporaryFile(mode="w+") as out, tempfile.TemporaryFile(mode="w+") as err:
                feature = subprocess.Popen(
                    feature_command,
                    cwd=self.project_root / "AutoAlpha",
                    env=env,
                    text=True,
                    stdout=out,
                    stderr=err,
                )
                self._wait_for_process(feature, market_data_root, state="RUNNING_FEATURES")
                out.seek(0)
                err.seek(0)
                feature_stdout, feature_error = out.read(), err.read()
            download_summary["features"] = _last_json_document(feature_stdout)
            feature_returncode = int(feature.returncode or 0)

        data_path = Path(settings["data_path"]).expanduser().resolve()
        source = market_data_root / "data" / "downloads" / "a_daily_cross_sectional_raw_adj"
        coverage = _cross_sectional_coverage(source, data_path)
        if include_core and not coverage["ready"]:
            # A normal incremental run can report old historical gaps as a
            # non-zero exit.  Those gaps are exactly what the resumable
            # bootstrap process is designed to repair, so never let that
            # status prevent it from starting.
            with tempfile.TemporaryFile(mode="w+") as out, tempfile.TemporaryFile(mode="w+") as err:
                migration = subprocess.Popen(
                    [str(downloader_python), str(cli), "--bootstrap-history"],
                    cwd=market_data_root,
                    env=env,
                    text=True,
                    stdout=out,
                    stderr=err,
                )
                while migration.poll() is None:
                    self._status = {
                        "state": "MIGRATING_HISTORY",
                        "updated_at": _now(),
                        "trigger": self._status.get("trigger"),
                        "coverage": _cross_sectional_coverage(source, data_path),
                        "download_progress": _download_progress(market_data_root),
                    }
                    time.sleep(1)
                out.seek(0)
                err.seek(0)
                migration_stdout, migration_stderr = out.read(), err.read()
            if migration.returncode != 0:
                return {
                    "download_returncode": 0,
                    "download_summary": {
                        "incremental": download_summary,
                        "migration": _last_json_document(migration_stdout),
                    },
                    "download_error": _trim_output(migration_stderr) or _trim_output(stderr),
                    "download_progress": _download_progress(market_data_root),
                    "panel_rebuilt": False,
                    "coverage": _cross_sectional_coverage(source, data_path),
                    "migration_message": (
                        "历史复权因子回填出现失败；已保留断点，下一次同步会自动续传。"
                    ),
                }
            coverage = _cross_sectional_coverage(source, data_path)
            download_summary["migration"] = _last_json_document(migration_stdout)
            if not coverage["ready"]:
                return {
                    "download_returncode": 0,
                    "download_summary": download_summary,
                    "download_error": _trim_output(stderr) or _trim_output(migration_stderr),
                    "download_progress": _download_progress(market_data_root),
                    "panel_rebuilt": False,
                    "coverage": coverage,
                    "migration_message": (
                        "历史回填尚未达到完整覆盖；已保留断点，下一次同步会自动续传。"
                    ),
                }
        project_python = self.project_root / ".venv" / "bin" / "python"
        if not project_python.is_file():
            project_python = Path(sys.executable)
        build = subprocess.run(
            [
                str(project_python),
                "-m",
                "multifactor_ashare.data",
                "cross-sectional",
                "--source",
                str(source),
                "--output",
                str(data_path),
                "--catalog",
                str(self.project_root / "data" / "catalog" / "daily_catalog.csv"),
                "--report",
                str(self.project_root / "data" / "catalog" / "data_quality.json"),
                "--feature-store",
                str(market_data_root / "data" / "downloads" / "a_share_feature_store"),
                "--overwrite",
            ],
            cwd=self.project_root,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        if build.returncode != 0:
            raise RuntimeError(_command_error("Cross-sectional panel rebuild", build))
        metadata = _last_json_document(build.stdout)
        return {
            "download_returncode": max(core_returncode, feature_returncode),
            "download_summary": download_summary,
            "download_error": _trim_output(feature_error) or _trim_output(stderr),
            "download_progress": _download_progress(market_data_root),
            "panel_metadata": metadata,
            "panel_rebuilt": True,
            "coverage": coverage,
            "dataset_ids": dataset_ids,
        }

    def _wait_for_process(
        self,
        process: subprocess.Popen[str],
        market_data_root: Path,
        *,
        state: str,
    ) -> None:
        while process.poll() is None:
            self._status = {
                "state": state,
                "updated_at": _now(),
                "trigger": self._status.get("trigger"),
                "dataset_ids": self._status.get("dataset_ids", []),
                "download_progress": _download_progress(market_data_root),
            }
            time.sleep(1)


def _download_progress(market_data_root: Path) -> dict[str, Any]:
    """Read durable downloader checkpoints without touching its process."""
    state_root = market_data_root / "data" / "state"
    log_root = market_data_root / "data" / "logs"
    tasks = []
    if state_root.is_dir():
        for state_path in sorted(state_root.glob("a_daily_*_csv-parquet.json")):
            state = _read_json_file(state_path)
            completed_values = state.get("completed", [])
            completed = len(completed_values) if isinstance(completed_values, list) else 0
            failed = len(state.get("failed", {})) if isinstance(state.get("failed"), dict) else 0
            total = completed + failed
            task_name = state_path.stem
            log_path = log_root / f"{task_name}.log"
            tasks.append(
                {
                    "task_key": task_name,
                    "adjustment": _adjustment_from_task(task_name),
                    "completed": completed,
                    "failed": failed,
                    "total": total,
                    "checkpoint_percent": round(completed / total * 100, 2) if total else 0.0,
                    "updated_at": _path_timestamp(state_path),
                    "last_message": _last_log_line(log_path),
                }
            )
        cross_state_path = state_root / "a_daily_cross_sectional_raw_adj.json"
        if cross_state_path.is_file():
            state = _read_json_file(cross_state_path)
            completed_values = state.get("completed_dates", [])
            completed = len(completed_values) if isinstance(completed_values, list) else 0
            failed = (
                len(state.get("failed_dates", {}))
                if isinstance(state.get("failed_dates"), dict)
                else 0
            )
            target_date = str(state.get("target_date", ""))
            expected_dates = state.get("expected_dates")
            total = (
                int(expected_dates)
                if isinstance(expected_dates, int) and expected_dates > 0
                else completed + failed
            )
            tasks.append(
                {
                    "task_key": cross_state_path.stem,
                    "adjustment": "raw_plus_adj_factor",
                    "completed": completed,
                    "failed": failed,
                    "total": total,
                    "checkpoint_percent": round(completed / total * 100, 2) if total else 0.0,
                    "updated_at": _path_timestamp(cross_state_path),
                    "last_message": _last_log_line(log_root / f"{cross_state_path.stem}.log"),
                    "target_date": target_date,
                }
            )
        for state_path in sorted(state_root.glob("a_feature_*.json")):
            state = _read_json_file(state_path)
            completed_values = state.get("completed_dates", [])
            completed = len(completed_values) if isinstance(completed_values, list) else 0
            failed_values = state.get("failed_dates", {})
            failed = len(failed_values) if isinstance(failed_values, dict) else 0
            expected = state.get("expected_dates")
            total = (
                int(expected) if isinstance(expected, int) and expected > 0 else completed + failed
            )
            dataset_id = str(state.get("dataset_id") or state_path.stem.removeprefix("a_feature_"))
            tasks.append(
                {
                    "task_key": state_path.stem,
                    "dataset_id": dataset_id,
                    "adjustment": "feature",
                    "completed": completed,
                    "failed": failed,
                    "total": total,
                    "checkpoint_percent": round(completed / total * 100, 2) if total else 0.0,
                    "updated_at": _path_timestamp(state_path),
                    "last_message": _last_log_line(log_root / f"{state_path.stem}.log"),
                    "target_date": state.get("target_date"),
                }
            )
    recent = max(tasks, key=lambda item: item["updated_at"] or "", default=None)
    return {"tasks": tasks, "active_checkpoint": recent}


def _read_json_file(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _adjustment_from_task(task_key: str) -> str:
    if "_qfq_" in task_key:
        return "qfq"
    if "_none_" in task_key:
        return "none"
    return "unknown"


def _cross_sectional_coverage(source: Path, panel_path: Path) -> dict[str, Any]:
    raw_root = source / "market_parquet"
    factor_root = source / "adj_factor_parquet"
    state = _read_json_file(source.parent.parent / "state" / "a_daily_cross_sectional_raw_adj.json")
    manifest = _read_json_file(source / "market_legacy_source.json")
    raw_dates = (
        sorted(path.stem for path in raw_root.glob("*.parquet")) if raw_root.is_dir() else []
    )
    factor_dates = (
        {path.stem for path in factor_root.glob("*.parquet")} if factor_root.is_dir() else set()
    )
    expected_dates = set(state.get("migration_expected_dates", []))
    metadata = _read_json_file(panel_path / "_metadata.json")
    required_first = str(metadata.get("first_trade_date", "")).replace("-", "")
    required_last = str(metadata.get("last_trade_date", "")).replace("-", "")
    legacy_first = str(manifest.get("first_trade_date", ""))
    legacy_last = str(manifest.get("last_trade_date", ""))
    actual_first = min([value for value in [legacy_first, *raw_dates] if value], default="")
    actual_last = max([value for value in [legacy_last, *raw_dates] if value], default="")
    missing_factors = expected_dates - factor_dates
    return {
        "ready": bool(
            expected_dates
            and not missing_factors
            and (not required_first or actual_first <= required_first)
            and (not required_last or actual_last >= required_last)
            and (source / "stock_basic.parquet").is_file()
        ),
        "available_dates": len(factor_dates),
        "expected_dates": len(expected_dates),
        "missing_factor_dates": len(missing_factors),
        "first_trade_date": actual_first or None,
        "last_trade_date": actual_last or None,
        "required_first_trade_date": required_first or None,
        "required_last_trade_date": required_last or None,
    }


def _path_timestamp(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, ZoneInfo("Asia/Shanghai")).isoformat()
    except OSError:
        return None


def _last_log_line(path: Path) -> str | None:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    return lines[-1] if lines else None


def _last_json_document(output: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for start in reversed([index for index, char in enumerate(output) if char == "{"]):
        try:
            value, end = decoder.raw_decode(output[start:])
        except json.JSONDecodeError:
            continue
        if output[start + end :].strip():
            continue
        return value if isinstance(value, dict) else None
    return None


def _trim_output(value: str, limit: int = 2_000) -> str:
    return value.strip()[-limit:]


def _command_error(label: str, result: subprocess.CompletedProcess[str]) -> str:
    output = _trim_output(result.stderr) or _trim_output(result.stdout)
    return f"{label} failed with exit code {result.returncode}: {output}"


def _configured_product_ids(settings: dict[str, str]) -> list[str]:
    raw = settings.get("data_product_ids")
    if not raw:
        return list(DEFAULT_PRODUCT_IDS)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return list(DEFAULT_PRODUCT_IDS)
    if not isinstance(value, list):
        return list(DEFAULT_PRODUCT_IDS)
    return [str(item) for item in value]


def _panel_start_date(settings: dict[str, str]) -> str:
    metadata = _read_json_file(Path(settings.get("data_path", "")) / "_metadata.json")
    value = str(metadata.get("first_trade_date", "2010-01-01"))
    return value.replace("-", "")


def _now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
