from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from autoalpha.data.current_panel import inspect_current_panel
from autoalpha.data.research_fields import research_eligible_fields


@dataclass(frozen=True)
class DataWorkspaceReport:
    configured_path: str
    root_path: str
    panel_path: str
    source_path: str | None
    catalog_path: str | None
    quality_report_path: str | None
    metadata_path: str | None
    quality_passed: bool | None
    source_integrity_passed: bool
    price_research_ready: bool
    institutional_pit_ready: bool
    files: int
    rows: int
    symbols: int | None
    first_trade_date: str | None
    last_trade_date: str | None
    columns: tuple[str, ...]
    factor_fields: tuple[str, ...]
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def require_price_research(self) -> None:
        if not self.price_research_ready:
            details = "; ".join((*self.blockers, *self.warnings))
            raise RuntimeError(f"Data workspace is not ready for price research: {details}")


def inspect_data_workspace(configured_path: Path) -> DataWorkspaceReport:
    configured = configured_path.expanduser().resolve()
    root, panel = _resolve_paths(configured)
    readiness = inspect_current_panel(panel)
    quality_path = root / "catalog/data_quality.json"
    metadata_path = panel / "_metadata.json"
    quality = _read_json(quality_path)
    metadata = _read_json(metadata_path)
    quality_passed = quality.get("passed") if quality else None
    if quality_passed is not None:
        quality_passed = bool(quality_passed)
    source_path = _resolve_source(root, quality, metadata)
    catalog_path = root / "catalog/daily_catalog.csv"
    warnings = list(readiness.warnings)
    blockers = list(readiness.blockers)
    integrity_failures: list[str] = []
    if quality_passed is False:
        integrity_failures.append("data quality report failed")
    elif quality_passed is None:
        warnings.append("data quality report is absent; source-level checks are unavailable")
    if not catalog_path.exists():
        warnings.append("daily source catalog is absent")
    summary = quality.get("summary", {})
    symbols = _integer(metadata.get("symbols")) or _integer(summary.get("symbols_with_rows"))
    first_date = metadata.get("first_trade_date") or summary.get("first_trade_date")
    last_date = metadata.get("last_trade_date") or summary.get("last_trade_date")
    _check_equal(
        integrity_failures,
        "quality rows",
        _integer(summary.get("rows")),
        readiness.rows,
    )
    _check_equal(
        integrity_failures,
        "panel metadata rows",
        _integer(metadata.get("rows")),
        readiness.rows,
    )
    _check_equal(
        integrity_failures,
        "quality and metadata symbols",
        _integer(summary.get("symbols_with_rows")),
        _integer(metadata.get("symbols")),
    )
    blockers = [*integrity_failures, *blockers]
    source_integrity_passed = not integrity_failures
    fingerprint_payload = {
        "panel": {
            "path": str(panel),
            "files": readiness.files,
            "rows": readiness.rows,
            "columns": readiness.columns,
        },
        "quality": quality,
        "metadata": metadata,
    }
    declared_feature_fields = metadata.get("research_feature_ready_fields", [])
    if not isinstance(declared_feature_fields, list):
        declared_feature_fields = []
    factor_fields, _ = research_eligible_fields(
        readiness.columns,
        metadata,
        required_start=str(first_date) if first_date else None,
        required_end=str(last_date) if last_date else None,
        declared_fields=declared_feature_fields,
    )
    return DataWorkspaceReport(
        configured_path=str(configured),
        root_path=str(root),
        panel_path=str(panel),
        source_path=str(source_path) if source_path else None,
        catalog_path=str(catalog_path) if catalog_path.exists() else None,
        quality_report_path=str(quality_path) if quality_path.exists() else None,
        metadata_path=str(metadata_path) if metadata_path.exists() else None,
        quality_passed=quality_passed,
        source_integrity_passed=source_integrity_passed,
        price_research_ready=readiness.price_research_ready and source_integrity_passed,
        institutional_pit_ready=(readiness.institutional_pit_ready and source_integrity_passed),
        files=readiness.files,
        rows=readiness.rows,
        symbols=symbols,
        first_trade_date=str(first_date) if first_date else None,
        last_trade_date=str(last_date) if last_date else None,
        columns=readiness.columns,
        factor_fields=tuple(sorted(factor_fields)),
        blockers=tuple(blockers),
        warnings=tuple(warnings),
        fingerprint=_sha256(fingerprint_payload),
    )


def _resolve_paths(configured: Path) -> tuple[Path, Path]:
    candidates = (
        configured,
        configured / "daily_panel",
        configured / "processed/daily_panel",
    )
    panel = next((path for path in candidates if _is_panel(path)), None)
    if panel is None:
        raise FileNotFoundError(
            f"Cannot resolve a partitioned daily panel from data workspace: {configured}"
        )
    if panel == configured / "processed/daily_panel":
        root = configured
    elif panel == configured / "daily_panel" and configured.name == "processed":
        root = configured.parent
    elif configured.name == "daily_panel" and configured.parent.name == "processed":
        root = configured.parent.parent
    else:
        root = configured
    return root, panel


def _is_panel(path: Path) -> bool:
    return path.is_dir() and bool(next(path.glob("trade_year=*/*.parquet"), None))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _resolve_source(root: Path, quality: dict[str, Any], metadata: dict[str, Any]) -> Path | None:
    declared = quality.get("source") or metadata.get("source")
    if declared:
        path = Path(str(declared))
        candidates = [path] if path.is_absolute() else [root / path, root.parent / path]
        for candidate in candidates:
            if candidate.resolve().exists():
                return candidate.resolve()
    candidates = [
        path for path in root.iterdir() if path.is_dir() and path.name.startswith("mainboard")
    ]
    return candidates[0] if len(candidates) == 1 else None


def _integer(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) else None


def _check_equal(
    failures: list[str], name: str, declared: int | None, observed: int | None
) -> None:
    if declared is not None and observed is not None and declared != observed:
        failures.append(f"{name} mismatch: declared={declared}, observed={observed}")


def _sha256(value: Any) -> str:
    canonical = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()
