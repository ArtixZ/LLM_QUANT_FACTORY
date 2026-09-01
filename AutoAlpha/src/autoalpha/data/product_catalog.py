"""Registry of US equity market-data products available through IBKR.

The registry deliberately separates discoverability from execution. An
``INTEGRATED`` product is downloaded and joined into the daily panel; a
``CATALOG`` product is inventory that IBKR can serve but whose publication-time
contract is not implemented, so it is never silently mixed into research data.

IBKR is a broker feed rather than a fundamentals vendor, so the catalog is far
smaller than a data-vendor equivalent: it carries prices, volume, and corporate
actions. Valuation ratios, sector classification, and index membership are not
available here and are recorded as absent rather than approximated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

IBKR_DOCUMENTATION = "https://www.interactivebrokers.com/campus/ibkr-api-page/twsapi-doc/"


@dataclass(frozen=True)
class MarketDataProduct:
    dataset_id: str
    api_name: str
    label: str
    category: str
    description: str
    grain: str
    cadence: str
    availability: str
    pit_policy: str
    feature_family: str
    sync_strategy: str
    date_parameter: str | None
    default_enabled: bool = False
    panel_fields: tuple[str, ...] = ()
    integration_state: str = "CATALOG"

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["documentation_url"] = IBKR_DOCUMENTATION
        return value


MARKET_DATA_PRODUCTS = (
    MarketDataProduct(
        dataset_id="core_market",
        api_name="reqHistoricalData:ADJUSTED_LAST",
        label="Daily bars, split- and dividend-adjusted",
        category="price",
        description=(
            "Total-return daily OHLC used for factor research. IBKR computes the "
            "adjustment relative to the download date, so it is not point-in-time."
        ),
        grain="symbol x session",
        cadence="daily",
        availability="regular trading hours",
        pit_policy="ADJUSTED_AS_OF_DOWNLOAD",
        feature_family="price",
        sync_strategy="FULL_WINDOW_ANCHORED_AT_PRESENT",
        date_parameter=None,
        default_enabled=True,
        panel_fields=("open", "high", "low", "close", "adj_close"),
        integration_state="INTEGRATED",
    ),
    MarketDataProduct(
        dataset_id="execution_market",
        api_name="reqHistoricalData:TRADES",
        label="Daily bars, split-adjusted, with share volume",
        category="price",
        description=(
            "Execution-basis OHLC plus share volume and per-bar VWAP. IBKR does "
            "not serve truly unadjusted bars, so splits are already applied."
        ),
        grain="symbol x session",
        cadence="daily",
        availability="regular trading hours",
        pit_policy="SPLIT_ADJUSTED",
        feature_family="price",
        sync_strategy="FULL_WINDOW_ANCHORED_AT_PRESENT",
        date_parameter=None,
        default_enabled=True,
        panel_fields=("raw_open", "raw_high", "raw_low", "raw_close", "vol", "amount"),
        integration_state="INTEGRATED",
    ),
    MarketDataProduct(
        dataset_id="tradability",
        api_name="derived",
        label="Session tradability flags",
        category="status",
        description=(
            "Bar validity and whether the session actually printed. US equities "
            "have no daily price limits; intraday LULD halts are not observable "
            "in daily bars."
        ),
        grain="symbol x session",
        cadence="daily",
        availability="derived from daily bars",
        pit_policy="DERIVED_FROM_BARS",
        feature_family="status",
        sync_strategy="DERIVED",
        date_parameter=None,
        default_enabled=True,
        panel_fields=(
            "is_valid_ohlc",
            "is_tradable_observation",
            "can_buy_open",
            "can_sell_open",
            "is_halted",
        ),
        integration_state="INTEGRATED",
    ),
    MarketDataProduct(
        dataset_id="fundamentals",
        api_name="reqFundamentalData",
        label="Company fundamentals",
        category="fundamental",
        description=(
            "Reported financials. Requires a separate IBKR subscription and has "
            "no point-in-time restatement history, so it is inventory only."
        ),
        grain="symbol x report",
        cadence="quarterly",
        availability="subscription required",
        pit_policy="NOT_POINT_IN_TIME",
        feature_family="fundamental",
        sync_strategy="UNIMPLEMENTED",
        date_parameter=None,
        panel_fields=(),
        integration_state="CATALOG",
    ),
)

PRODUCT_BY_ID = {product.dataset_id: product for product in MARKET_DATA_PRODUCTS}
DEFAULT_PRODUCT_IDS = tuple(
    product.dataset_id for product in MARKET_DATA_PRODUCTS if product.default_enabled
)


def data_product_catalog() -> list[dict[str, object]]:
    return [product.to_dict() for product in MARKET_DATA_PRODUCTS]


def resolve_products(
    dataset_ids: list[str] | tuple[str, ...],
) -> list[MarketDataProduct]:
    """Resolve dataset ids, rejecting unknown or non-downloadable products."""
    unknown = [item for item in dataset_ids if item not in PRODUCT_BY_ID]
    if unknown:
        known = ", ".join(sorted(PRODUCT_BY_ID))
        raise ValueError(f"Unknown data products: {', '.join(unknown)}; known: {known}")
    return [PRODUCT_BY_ID[item] for item in dataset_ids]
