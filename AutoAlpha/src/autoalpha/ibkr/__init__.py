"""Interactive Brokers gateway adapter: contracts, US daily history, and orders."""

from autoalpha.ibkr.client import (
    AccountSummary,
    GatewayNotConnectedError,
    GatewayTimeoutError,
    IBKRGateway,
    OrderTransmissionBlocked,
    Position,
)
from autoalpha.ibkr.contracts import (
    ContractResolutionError,
    USEquity,
    normalize_symbol,
    panel_symbol,
)
from autoalpha.ibkr.history import (
    HistoryDownloadError,
    SymbolHistory,
    download_symbol_history,
    download_universe_history,
)
from autoalpha.ibkr.orders import (
    OrderPlanError,
    PlannedOrder,
    plan_orders,
    preview_plan,
    submit_plan,
)
from autoalpha.ibkr.pacing import HistoricalPacer
from autoalpha.ibkr.settings import GatewaySettings, TradingModeError, is_paper_account

__all__ = [
    "AccountSummary",
    "ContractResolutionError",
    "GatewayNotConnectedError",
    "GatewayTimeoutError",
    "GatewaySettings",
    "HistoricalPacer",
    "HistoryDownloadError",
    "IBKRGateway",
    "OrderPlanError",
    "OrderTransmissionBlocked",
    "PlannedOrder",
    "Position",
    "SymbolHistory",
    "TradingModeError",
    "USEquity",
    "download_symbol_history",
    "download_universe_history",
    "is_paper_account",
    "normalize_symbol",
    "panel_symbol",
    "plan_orders",
    "preview_plan",
    "submit_plan",
]
