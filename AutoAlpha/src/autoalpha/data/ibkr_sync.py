"""Download US equity history from an IBKR gateway into immutable Parquet slices.

One slice per symbol lands under ``<root>/downloads``. The slices are the raw,
auditable input to the panel builder in ``multifactor_us.data``; nothing here
writes the research panel directly.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from autoalpha.ibkr.client import IBKRGateway
from autoalpha.ibkr.contracts import USEquity
from autoalpha.ibkr.history import SymbolHistory, download_symbol_history
from autoalpha.ibkr.settings import GatewaySettings

logger = logging.getLogger(__name__)

DEFAULT_MARKET_DATA_ROOT = Path.home() / "MarketData" / "US"
DOWNLOADS_DIRECTORY = "downloads"
MANIFEST_FILENAME = "_download_manifest.json"


@dataclass(frozen=True)
class SyncResult:
    root: str
    start: str
    end: str
    requested: int
    resolved: int
    written: int
    rows: int
    contract_failures: dict[str, str] = field(default_factory=dict)
    history_failures: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def failed(self) -> int:
        return len(self.contract_failures) + len(self.history_failures)


def existing_slice_symbols(root: Path) -> set[str]:
    """Panel symbols already downloaded, recovered from slice filenames."""
    downloads = root.expanduser() / DOWNLOADS_DIRECTORY
    if not downloads.is_dir():
        return set()
    return {
        path.stem.replace("_", ".")
        for path in downloads.glob("*.parquet")
        if not path.name.startswith("_")
    }


def prune_slices(root: Path, keep: Iterable[str]) -> list[str]:
    """Delete slices outside ``keep``. Used to deliberately shrink a universe."""
    wanted = {symbol.replace(" ", ".") for symbol in keep}
    removed = []
    for symbol in sorted(existing_slice_symbols(root) - wanted):
        slice_path(root, symbol).unlink(missing_ok=True)
        removed.append(symbol)
    return removed


def slice_path(root: Path, symbol: str) -> Path:
    """Filesystem-safe slice location for a panel symbol (``BRK.B`` -> ``BRK_B``)."""
    safe = symbol.replace(".", "_").replace(" ", "_")
    return root / DOWNLOADS_DIRECTORY / f"{safe}.parquet"


def write_slice(root: Path, history: SymbolHistory) -> Path:
    """Persist one symbol's history, replacing any previous slice atomically."""
    target = slice_path(root, history.symbol)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_suffix(".parquet.staging")
    history.frame.to_parquet(staging, index=False, compression="zstd")
    staging.replace(target)
    return target


def sync_universe(
    symbols: Iterable[str],
    *,
    start: date,
    end: date,
    root: Path = DEFAULT_MARKET_DATA_ROOT,
    settings: GatewaySettings | None = None,
    gateway: IBKRGateway | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> SyncResult:
    """Resolve, download, and persist a universe of US equities.

    Per-symbol failures are collected rather than raised so one bad ticker does
    not discard an otherwise complete sync. The caller decides whether the
    failure rate is acceptable.
    """
    requested = [symbol for symbol in symbols if symbol.strip()]
    if not requested:
        raise ValueError("The requested universe is empty")
    root = root.expanduser()
    owns_session = gateway is None
    session = gateway or IBKRGateway(settings or GatewaySettings.from_environment())
    if owns_session:
        session.connect()
    try:
        equities, contract_failures = session.resolve_universe(requested)
        histories, history_failures = _download_all(
            session, equities, start=start, end=end, on_progress=on_progress
        )
        rows = 0
        written = 0
        for history in histories:
            write_slice(root, history)
            written += 1
            rows += history.rows
    finally:
        if owns_session:
            session.disconnect()

    result = SyncResult(
        root=str(root.resolve()),
        start=start.isoformat(),
        end=end.isoformat(),
        requested=len(requested),
        resolved=len(equities),
        written=written,
        rows=rows,
        contract_failures=contract_failures,
        history_failures=history_failures,
    )
    _write_manifest(root, result, equities)
    return result


def _download_all(
    session: IBKRGateway,
    equities: Sequence[USEquity],
    *,
    start: date,
    end: date,
    on_progress: Callable[[int, int, str], None] | None,
) -> tuple[list[SymbolHistory], dict[str, str]]:
    histories: list[SymbolHistory] = []
    failures: dict[str, str] = {}
    total = len(equities)
    for index, equity in enumerate(equities, start=1):
        if on_progress is not None:
            on_progress(index, total, equity.symbol)
        try:
            histories.append(download_symbol_history(session, equity, start=start, end=end))
        except Exception as error:  # noqa: BLE001 - one bad symbol must not end the sync
            failures[equity.panel_symbol] = str(error)
            logger.warning("history download failed for %s: %s", equity.symbol, error)
    return histories, failures


def _write_manifest(root: Path, result: SyncResult, equities: Sequence[USEquity]) -> None:
    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "market": "US_EQUITY",
        "source": "interactive_brokers_gateway",
        **result.to_dict(),
        "contracts": [equity.to_dict() for equity in equities],
    }
    target = root / DOWNLOADS_DIRECTORY / MANIFEST_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
