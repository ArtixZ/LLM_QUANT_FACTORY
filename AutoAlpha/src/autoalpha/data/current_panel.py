from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pyarrow.dataset as ds
import pyarrow.parquet as pq


@dataclass(frozen=True)
class PanelReadinessReport:
    path: str
    files: int
    rows: int
    columns: tuple[str, ...]
    price_research_ready: bool
    institutional_pit_ready: bool
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def require_institutional_pit(self) -> None:
        if not self.institutional_pit_ready:
            raise RuntimeError(
                "Panel is not institutionally point-in-time ready: " + "; ".join(self.blockers)
            )


PRICE_RESEARCH_FIELDS = frozenset(
    {
        "symbol",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "vol",
        "amount",
        "is_valid_ohlc",
        "is_tradable_observation",
    }
)

# US equities need listing lifecycle, halt state, and side-specific open
# eligibility. There is no ST designation and no daily price limit, so those
# A-share groups are gone; classification and free float remain required for
# institutional readiness because risk models depend on them.
INSTITUTIONAL_FIELD_GROUPS: dict[str, frozenset[str]] = {
    "source knowledge and revision timestamps": frozenset(
        {"knowledge_time", "source_batch", "revision_id"}
    ),
    "listing and delisting history": frozenset({"listing_date", "delisting_date"}),
    "halt and open eligibility state": frozenset(
        {"is_halted", "can_buy_open", "can_sell_open"}
    ),
    "historical classification and benchmark membership": frozenset(
        {"sector_code", "index_membership"}
    ),
    "point-in-time free-float capitalization": frozenset({"free_float_market_cap"}),
}


def inspect_current_panel(path: Path) -> PanelReadinessReport:
    files = sorted(path.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {path}")
    # The workspace may also contain catalog JSON beside the partitioned Parquet files.
    dataset = ds.dataset(files, format="parquet")
    columns = tuple(dataset.schema.names)
    column_set = set(columns)
    rows = sum(pq.ParquetFile(file).metadata.num_rows for file in files)
    blockers = tuple(
        f"missing {name}: {sorted(required - column_set)}"
        for name, required in INSTITUTIONAL_FIELD_GROUPS.items()
        if not required <= column_set
    )
    price_missing = PRICE_RESEARCH_FIELDS - column_set
    warnings = []
    if price_missing:
        warnings.append(f"price research fields missing: {sorted(price_missing)}")
    if "trade_year" in column_set:
        warnings.append("trade_year is a storage partition, not a point-in-time source field")
    if "is_tradable_observation" in column_set and "can_buy_open" not in column_set:
        warnings.append(
            "is_tradable_observation cannot replace side-specific open-time tradability"
        )
    return PanelReadinessReport(
        path=str(path.resolve()),
        files=len(files),
        rows=rows,
        columns=columns,
        price_research_ready=not price_missing,
        institutional_pit_ready=not blockers and not price_missing,
        blockers=blockers,
        warnings=tuple(warnings),
    )
