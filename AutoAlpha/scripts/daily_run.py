#!/usr/bin/env python
"""Daily US equity paper run: refresh data, form a target book, report to Telegram.

Usage:
  daily_run.py                    full run: sync, rebuild panel, report, notify
  daily_run.py --dry-run          compute and print, no notification
  daily_run.py --skip-sync        reuse the panel on disk (fast iteration)
  daily_run.py --submit           ALSO transmit the order plan (needs --confirm)

The default run never transmits an order. It refreshes market data, rebuilds
the research panel, previews the plan through IBKR's what-if margin check, and
reports. Submission is opt-in and requires two flags plus a writable session.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from autoalpha.ibkr.settings import GatewaySettings  # noqa: E402
from autoalpha.operations.daily import (  # noqa: E402
    DailyConfig,
    DailyReport,
    DataHealth,
    run_daily,
)
from autoalpha.service.data_sync import DataSyncWorker  # noqa: E402
from autoalpha.service.store import ServiceStore  # noqa: E402

# Notification and trade-bus dispatch are host-local integrations: point these
# at your own scripts. Unset means that hop is skipped, not an error.
NOTIFY_SCRIPT = Path(os.environ.get("QUANTFACTORY_NOTIFY_SCRIPT", ""))
TRADEBUS_PUBLISH = Path(os.environ.get("QUANTFACTORY_TRADEBUS_PUBLISH", ""))
NOTIFY_SOURCE = "quantfactory"
RUNTIME_DIR = Path.home() / "MarketData" / "US" / "runtime"
LOG_DIR = REPO_ROOT / "logs"
SUBMISSION_DIR = RUNTIME_DIR / "submissions"

logger = logging.getLogger("daily_run")


def notify(title: str, body: str, severity: str = "info") -> None:
    """Fan out through the host's notification dispatcher, if one is configured."""
    if not NOTIFY_SCRIPT.name or not NOTIFY_SCRIPT.exists():
        logger.warning("notify script not found at %s; skipping notification", NOTIFY_SCRIPT)
        return
    env = os.environ.copy()
    # Recurring 26h cadence so Watchtower alerts if a weekday run goes missing.
    env["NOTIFY_WATCHTOWER_SLUG"] = "quantfactory-daily"
    env["NOTIFY_WATCHTOWER_ONESHOT"] = "false"
    env["NOTIFY_WATCHTOWER_INTERVAL"] = "93600"
    env["NOTIFY_WATCHTOWER_GRACE"] = "1800"
    try:
        subprocess.run(
            [str(NOTIFY_SCRIPT), title, body, severity, NOTIFY_SOURCE],
            env=env, timeout=60, check=False, capture_output=True,
        )
    except Exception as error:  # noqa: BLE001 - notification must never break the run
        logger.warning("notification failed: %s", error)


def publish_expected_book(report: DailyReport) -> None:
    """Tell tradebus which symbols this strategy intends to hold.

    broker_watch.py flags any position it cannot account for. Without an
    intended book it treats every fill as an orphan and fires a daily error,
    which would train the alert to be ignored. Published at heartbeat severity
    so it reaches the ops dashboard without buzzing a phone.
    """
    if not TRADEBUS_PUBLISH.name or not TRADEBUS_PUBLISH.exists():
        return
    expected = {pick["symbol"]: pick["target_shares"] for pick in report.picks}
    payload = {
        "strategy": "quantfactory",
        "expected_positions": expected,
        "submitted": report.submitted,
        "as_of": report.as_of,
        # A stable slug: without it dispatch derives one from the title, which
        # carries the date and would mint a new Watchtower check every day.
        # one_shot because the daily digest already carries the dead-man switch.
        "watchtower_slug": "quantfactory-intended-book",
        "watchtower_oneshot": "true",
    }
    try:
        subprocess.run(
            [sys.executable, str(TRADEBUS_PUBLISH),
             "--source", NOTIFY_SOURCE, "--kind", "strategy.book",
             "--severity", "heartbeat",
             "--title", f"Intended book {report.as_of} ({len(expected)} names)",
             "--body", ", ".join(f"{s} {q:,}" for s, q in sorted(expected.items())),
             "--data", json.dumps(payload)],
            timeout=30, check=False, capture_output=True,
        )
    except Exception as error:  # noqa: BLE001 - bus plumbing must not break the run
        logger.warning("could not publish the intended book: %s", error)


def refresh_market_data(config: DailyConfig, *, skip: bool) -> DataHealth:
    """Sync slices and rebuild the panel, then read the audit for health."""
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    store = ServiceStore(RUNTIME_DIR / "daily.sqlite3")
    store.save_settings(
        {
            "market_data_root": str(config.market_data_root),
            "universe": config.universe,
            "panel_start_date": config.history_start.isoformat(),
            "data_path": str(config.panel_path),
        }
    )
    sync_written, sync_failures = 0, {}
    if not skip:
        worker = DataSyncWorker(store, project_root=REPO_ROOT.parent, is_busy=lambda: False)
        if not worker.gateway_ready():
            raise RuntimeError("The IBKR gateway is not accepting API connections")
        result = worker.run_system_job(
            {"job_id": f"daily-{date.today().isoformat()}", "payload": {"trigger": "daily_run"}}
        )
        if not result.get("panel_rebuilt"):
            raise RuntimeError(f"Panel rebuild failed: {result.get('panel_error')}")
        summary = result["download_summary"]
        sync_written = int(summary["written"])
        sync_failures = {**summary["contract_failures"], **summary["history_failures"]}

    report = json.loads(config.quality_report_path.read_text(encoding="utf-8"))
    return DataHealth(
        panel_last_date=str(report["summary"]["last_trade_date"])[:10],
        panel_rows=int(report["summary"]["rows"]),
        panel_symbols=int(report["summary"]["symbols_with_rows"]),
        audit_passed=bool(report["passed"]),
        stale_symbols=dict(report.get("stale_symbols", {})),
        sync_written=sync_written,
        sync_failures=sync_failures,
    )


def write_artifact(report: DailyReport) -> Path:
    """Persist the full report so a Telegram digest is never the only record."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"daily-{report.as_of}.json"
    path.write_text(json.dumps(report.to_dict(), indent=2, default=str) + "\n", encoding="utf-8")
    return path


def reserve_submission(submission_key: str) -> Path:
    """Create a durable one-attempt marker before any broker order is sent."""
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    path = SUBMISSION_DIR / f"{submission_key}.json"
    payload = json.dumps(
        {
            "submission_key": submission_key,
            "status": "RESERVED",
            "created_on": date.today().isoformat(),
        }
    ).encode()
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError(
            f"Submission {submission_key} was already attempted; inspect {path} before retrying"
        ) from error
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload + b"\n")
    return path


def update_submission(path: Path, *, status: str, detail: str = "") -> None:
    path.write_text(
        json.dumps(
            {
                "submission_key": path.stem,
                "status": status,
                "detail": detail,
                "updated_on": date.today().isoformat(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="daily_run", description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print only; do not notify")
    parser.add_argument("--skip-sync", action="store_true", help="reuse the panel on disk")
    parser.add_argument("--submit", action="store_true", help="transmit the plan (needs --confirm)")
    parser.add_argument("--confirm", action="store_true", help="second gate for --submit")
    parser.add_argument("--universe", default="MEGA_CAP_LIQUID_V1")
    parser.add_argument("--positions", type=int, default=5)
    parser.add_argument("--gross-exposure", type=float, default=0.95)
    parser.add_argument("--port", type=int, default=None, help="override IBKR port")
    parser.add_argument(
        "--managed-account",
        default=os.getenv("QUANTFACTORY_MANAGED_ACCOUNT", ""),
        help="paper account dedicated to this strategy; required for submission",
    )
    args = parser.parse_args(argv)
    if args.submit != args.confirm:
        parser.error("--submit and --confirm must be supplied together")
    if args.submit and not args.managed_account.strip():
        parser.error(
            "--managed-account or QUANTFACTORY_MANAGED_ACCOUNT is required for submission"
        )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = DailyConfig(
        universe=args.universe,
        position_count=args.positions,
        gross_exposure=args.gross_exposure,
    )
    base = GatewaySettings.from_environment()
    settings = GatewaySettings(
        host=base.host,
        port=args.port or base.port,
        client_id=base.client_id,
        account=base.account,
        # Transmission needs a writable session; previewing never does.
        readonly=not (args.submit and args.confirm),
        require_paper_account=base.require_paper_account,
    )

    submission_key = f"quantfactory-{date.today():%Y%m%d}"
    submission_marker: Path | None = None
    try:
        health = refresh_market_data(config, skip=args.skip_sync)
        if args.submit:
            submission_marker = reserve_submission(submission_key)
        report = run_daily(
            config,
            settings=settings,
            health=health,
            submit=args.submit,
            confirm_submit=args.confirm,
            managed_account=args.managed_account.strip() or None,
            submission_key=submission_key,
        )
    except Exception as error:  # noqa: BLE001 - a failed run must still be reported
        if submission_marker is not None:
            update_submission(
                submission_marker,
                status="FAILED_REVIEW_REQUIRED",
                detail=f"{type(error).__name__}: {error}",
            )
        logger.exception("daily run failed")
        if not args.dry_run:
            notify(
                f"Daily run FAILED {date.today().isoformat()}",
                f"{type(error).__name__}: {error}",
                "error",
            )
        return 1
    if submission_marker is not None:
        update_submission(
            submission_marker,
            status="SUBMITTED" if report.submitted else "NO_ORDERS_SUBMITTED",
        )

    body = report.telegram_body()
    artifact = write_artifact(report)
    publish_expected_book(report)
    print(body)
    print(f"\nartifact: {artifact}")

    if args.dry_run:
        print("(dry run: no notification sent)")
        return 0
    notify(report.title, body, report.severity)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
