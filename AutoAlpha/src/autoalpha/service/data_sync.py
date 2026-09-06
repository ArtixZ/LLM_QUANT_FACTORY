"""Refresh US equity market data from an IBKR gateway and rebuild the research panel.

The worker downloads immutable per-symbol slices through
:mod:`autoalpha.data.ibkr_sync`, audits them, and only then atomically replaces
the research panel. A sync never mutates the panel in place: a failed or partial
download leaves the previous panel serving research untouched.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from autoalpha.data.ibkr_sync import (
    DOWNLOADS_DIRECTORY,
    existing_slice_symbols,
    sync_universe,
)
from autoalpha.data.product_catalog import resolve_products
from autoalpha.data.universe_catalog import resolve_universe
from autoalpha.ibkr.settings import GatewaySettings
from autoalpha.service.store import ServiceStore

logger = logging.getLogger(__name__)

DEFAULT_MARKET_DATA_ROOT = Path.home() / "MarketData" / "US"
DEFAULT_UNIVERSE = "MEGA_CAP_LIQUID_V1"
DEFAULT_PANEL_START = "2015-01-01"
SYNC_INTERVAL_SECONDS = 3_600


class DataSyncWorker:
    """Own the exclusive market-data refresh: download, audit, rebuild, publish."""

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
        self.market_data_root = DEFAULT_MARKET_DATA_ROOT
        self._task: asyncio.Task[dict[str, Any]] | None = None
        self._scheduler: asyncio.Task[None] | None = None
        self._status: dict[str, Any] = {"state": "IDLE", "updated_at": None}

    # ---- lifecycle ---------------------------------------------------------

    @property
    def alive(self) -> bool:
        return self._task is not None and not self._task.done()

    def status(self) -> dict[str, Any]:
        settings = self.store.settings()
        root = Path(
            settings.get("market_data_root", str(self.market_data_root))
        ).expanduser()
        return {
            **self._status,
            "running": self.alive,
            "market": "US_EQUITY",
            "source": "interactive_brokers_gateway",
            "market_data_root": str(root),
            "gateway_ready": self.gateway_ready(),
            "universe": settings.get("universe", DEFAULT_UNIVERSE),
            "last_data_sync_date": settings.get("last_data_sync_date"),
        }

    def gateway_ready(self) -> bool:
        """Whether an IBKR gateway is accepting API connections right now."""
        import socket

        settings = self._gateway_settings()
        try:
            with socket.create_connection((settings.host, settings.port), timeout=2.0):
                return True
        except OSError:
            return False

    def token_configured(self) -> bool:
        """Deprecated alias for :meth:`gateway_ready`.

        IBKR authenticates through the gateway itself, so there is no API token
        to configure. This exists only until the HTTP layer's ``tushare_token``
        vocabulary is renamed, and reports gateway reachability instead.
        """
        return self.gateway_ready()

    def set_token(self, value: str) -> None:
        """No-op retained for the legacy HTTP payload field.

        There is no market-data token for IBKR; access is granted by logging the
        gateway in. Callers are logged so a stale UI field is visible in
        operations rather than silently accepted as meaningful.
        """
        logger.warning(
            "ignoring market-data token: IBKR authenticates through the gateway session"
        )

    def _gateway_settings(self) -> GatewaySettings:
        settings = self.store.settings()
        base = GatewaySettings.from_environment()
        port = settings.get("ibkr_port")
        host = settings.get("ibkr_host")
        return GatewaySettings(
            host=str(host) if host else base.host,
            port=int(port) if port else base.port,
            client_id=base.client_id,
            account=base.account,
            readonly=True,
            require_paper_account=base.require_paper_account,
        )

    # ---- triggers ----------------------------------------------------------

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
        if not self.gateway_ready():
            raise RuntimeError("The IBKR gateway is not accepting API connections")
        self._task = asyncio.create_task(
            self._run(trigger, dataset_ids, start_date=start_date, end_date=end_date),
            name="autoalpha-market-data-sync",
        )
        return self.status()

    def run_system_job(self, job: dict[str, Any]) -> dict[str, Any]:
        payload = job.get("payload") or {}
        trigger = str(payload.get("trigger") or "system_job")
        if self.is_busy():
            raise RuntimeError("Stop research and manual backtests before refreshing market data")
        if not self.gateway_ready():
            raise RuntimeError("The IBKR gateway is not accepting API connections")
        dataset_ids = payload.get("dataset_ids")
        selected = list(dataset_ids) if isinstance(dataset_ids, list) else None
        self._status = {
            "state": "RUNNING",
            "updated_at": _now(),
            "trigger": trigger,
            "system_job_id": job.get("job_id"),
        }
        result = self._sync_blocking(
            selected,
            start_date=_parse_iso_date(payload.get("start_date")),
            end_date=_parse_iso_date(payload.get("end_date")),
        )
        return self._apply_sync_result(trigger=trigger, result=result)

    async def start_scheduler(self) -> None:
        if self._scheduler is None or self._scheduler.done():
            self._scheduler = asyncio.create_task(
                self._schedule_loop(), name="autoalpha-market-data-scheduler"
            )

    async def shutdown(self) -> None:
        for task in (self._scheduler, self._task):
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
        self._scheduler = None
        self._task = None

    async def _schedule_loop(self) -> None:
        while True:
            await asyncio.sleep(SYNC_INTERVAL_SECONDS)
            try:
                await self._maybe_run_scheduled_sync()
            except asyncio.CancelledError:
                raise
            except Exception as error:  # noqa: BLE001 - a scheduler must not die
                logger.warning("scheduled market-data sync failed: %s", error)

    async def _maybe_run_scheduled_sync(self) -> None:
        settings = self.store.settings()
        if str(settings.get("auto_data_sync", "")).casefold() not in {"1", "true", "on"}:
            return
        if self.alive or self.is_busy():
            return
        if settings.get("last_data_sync_date") == date.today().isoformat():
            return
        if not self.gateway_ready():
            self._status = {
                "state": "WAITING",
                "updated_at": _now(),
                "message": "Automatic data sync is waiting for the IBKR gateway",
            }
            return
        await self.start(trigger="scheduled")

    # ---- execution ---------------------------------------------------------

    async def _run(
        self,
        trigger: str,
        dataset_ids: list[str] | None,
        *,
        start_date: date | None,
        end_date: date | None,
    ) -> dict[str, Any]:
        self._status = {"state": "RUNNING", "updated_at": _now(), "trigger": trigger}
        self.store.append_event(
            "action",
            "MARKET_DATA_SYNC_STARTED",
            "US market data sync started",
            "Downloading IBKR daily bars, then auditing coverage before the panel is rebuilt.",
            payload={"trigger": trigger, "mode": "NON_PIT_PROXY"},
        )
        try:
            result = await asyncio.to_thread(
                self._sync_blocking, dataset_ids, start_date=start_date, end_date=end_date
            )
            return self._apply_sync_result(trigger=trigger, result=result)
        except Exception as error:
            self._status = {
                "state": "FAILED",
                "updated_at": _now(),
                "message": f"{type(error).__name__}: {error}",
            }
            self.store.append_event(
                "audit",
                "MARKET_DATA_SYNC_FAILED",
                "US market data sync failed",
                self._status["message"],
                level="ERROR",
                payload={"trigger": trigger},
            )
            raise

    def _sync_blocking(
        self,
        dataset_ids: list[str] | None,
        *,
        start_date: date | None,
        end_date: date | None,
    ) -> dict[str, Any]:
        settings = self.store.settings()
        root = Path(settings.get("market_data_root", str(self.market_data_root))).expanduser()
        universe_name, symbols = resolve_universe(
            str(settings.get("universe", DEFAULT_UNIVERSE))
        )
        selected_products = resolve_products(dataset_ids or [])
        unavailable = [
            product.dataset_id
            for product in selected_products
            if product.integration_state != "INTEGRATED"
        ]
        if unavailable:
            raise ValueError(
                "Data products do not have an integrated IBKR sync contract: "
                + ", ".join(unavailable)
            )
        # All integrated products are two views of the same per-symbol IBKR
        # download plus derived tradability flags. Product IDs select fields;
        # they are never ticker symbols.
        carried = existing_slice_symbols(root) - set(symbols)
        if carried:
            logger.info("carrying %d symbol(s) beyond %s", len(carried), universe_name)
            symbols = tuple(symbols) + tuple(sorted(carried))
        configured_start = _parse_iso_date(settings.get("panel_start_date"))
        start = start_date or configured_start or date.fromisoformat(DEFAULT_PANEL_START)
        end = end_date or date.today()

        sync = sync_universe(
            symbols,
            start=start,
            end=end,
            root=root,
            settings=self._gateway_settings(),
        )
        panel_path = Path(settings.get("data_path", str(root / "processed" / "daily_panel")))
        build = self._rebuild_panel(root, panel_path)
        return {
            "download_returncode": 0 if sync.written else 1,
            "universe": universe_name,
            "download_summary": sync.to_dict(),
            "panel_rebuilt": build["returncode"] == 0,
            "panel_metadata": build["metadata"],
            "panel_error": build["error"],
            "stale_symbols": build.get("stale_symbols") or {},
        }

    def _rebuild_panel(self, root: Path, panel_path: Path) -> dict[str, Any]:
        """Audit the downloaded slices, then atomically rebuild the panel."""
        python = self.project_root / ".venv" / "bin" / "python"
        interpreter = str(python) if python.is_file() else sys.executable
        source = root / DOWNLOADS_DIRECTORY
        audit = subprocess.run(
            [interpreter, "-m", "multifactor_us.data", "audit", "--source", str(source),
             "--report", str(root / "catalog" / "data_quality.json")],
            cwd=self.project_root, text=True, capture_output=True, check=False,
        )
        if audit.returncode != 0:
            return {"returncode": audit.returncode, "metadata": None,
                    "error": _trim_output(audit.stdout or audit.stderr)}
        # The audit passes with per-symbol staleness as a warning only, and its
        # report is otherwise written to disk and read by nobody. Surface the
        # stale map here so a partial sync is visible in the sync status and
        # events instead of only as NaN columns in later cross-sections.
        report = _last_json_document(audit.stdout) or {}
        stale_symbols = dict(report.get("stale_symbols") or {})
        build = subprocess.run(
            [interpreter, "-m", "multifactor_us.data", "panel", "--source", str(source),
             "--output", str(panel_path), "--overwrite"],
            cwd=self.project_root, text=True, capture_output=True, check=False,
        )
        if build.returncode != 0:
            return {"returncode": build.returncode, "metadata": None,
                    "error": _trim_output(build.stderr), "stale_symbols": stale_symbols}
        return {"returncode": 0, "metadata": _last_json_document(build.stdout),
                "error": None, "stale_symbols": stale_symbols}

    def _apply_sync_result(self, *, trigger: str, result: dict[str, Any]) -> dict[str, Any]:
        sync_ok = result["download_returncode"] == 0
        panel_rebuilt = bool(result.get("panel_rebuilt"))
        state = (
            "COMPLETED" if sync_ok and panel_rebuilt
            else "MIGRATION_PENDING" if sync_ok
            else "DEGRADED"
        )
        stale_symbols = result.get("stale_symbols") or {}
        self._status = {"state": state, "updated_at": _now(), "trigger": trigger, **result}
        self.store.save_settings({"last_data_sync_date": date.today().isoformat()})
        if panel_rebuilt and stale_symbols:
            message = (
                f"The panel was replaced, but {len(stale_symbols)} symbol(s) end before its "
                f"latest trade date and stay NaN in recent cross-sections: "
                f"{', '.join(sorted(stale_symbols))}. Sync again with the gateway up, or "
                "prune_slices to drop them deliberately."
            )
        elif panel_rebuilt:
            message = (
                "Downloaded slices passed the contract audit and the panel was "
                "replaced atomically."
            )
        else:
            message = result.get("panel_error") or "The panel was not replaced."
        self.store.append_event(
            "delivery",
            "MARKET_DATA_SYNC_COMPLETED",
            "Research panel updated" if panel_rebuilt else "Raw market slices updated",
            message,
            level="INFO" if sync_ok and panel_rebuilt and not stale_symbols else "WARN",
            payload=self._status,
        )
        return self._status


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _parse_iso_date(value: Any) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _trim_output(value: str, limit: int = 2_000) -> str:
    text = (value or "").strip()
    return text if len(text) <= limit else text[:limit] + "…"


def _last_json_document(output: str) -> dict[str, Any] | None:
    """Extract the final JSON object printed by a CLI subprocess."""
    depth = 0
    start = -1
    last: dict[str, Any] | None = None
    for index, character in enumerate(output or ""):
        if character == "{":
            if depth == 0:
                start = index
            depth += 1
        elif character == "}" and depth:
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    parsed = json.loads(output[start : index + 1])
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    last = parsed
    return last
