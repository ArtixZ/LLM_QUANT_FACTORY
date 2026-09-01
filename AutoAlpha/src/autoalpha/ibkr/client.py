from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from types import TracebackType
from typing import Any

import pandas as pd

from autoalpha.ibkr.contracts import (
    US_EQUITY_CURRENCY,
    ContractResolutionError,
    USEquity,
    normalize_symbol,
    select_primary_listing,
)
from autoalpha.ibkr.pacing import HistoricalPacer
from autoalpha.ibkr.settings import GatewaySettings

logger = logging.getLogger(__name__)

BAR_COLUMNS = ("date", "open", "high", "low", "close", "volume", "average", "bar_count")
# IBKR sends Java's Double.MAX_VALUE to mean "this field has no value" rather
# than omitting it, so it must never be read as a real number.
UNSET_DOUBLE = 1.7976931348623157e308
DAILY_BAR_SIZE = "1 day"
# TRADES is split-adjusted only; ADJUSTED_LAST additionally back-adjusts dividends.
EXECUTION_PRICE_BASIS = "TRADES"
TOTAL_RETURN_PRICE_BASIS = "ADJUSTED_LAST"


class GatewayNotConnectedError(RuntimeError):
    """An operation required a live gateway session."""


class OrderTransmissionBlocked(RuntimeError):
    """Order transmission was attempted on a read-only session."""


class GatewayTimeoutError(RuntimeError):
    """The gateway did not answer a request inside its timeout."""


@dataclass(frozen=True)
class AccountSummary:
    account: str
    is_paper: bool
    net_liquidation: float
    total_cash: float
    buying_power: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    currency: str = US_EQUITY_CURRENCY


@dataclass(frozen=True)
class Position:
    account: str
    symbol: str
    con_id: int
    quantity: float
    average_cost: float
    market_price: float
    market_value: float
    unrealized_pnl: float


class IBKRGateway:
    """Narrow, synchronous facade over an Interactive Brokers gateway session.

    The ``ib_async`` types stay behind this boundary so the research platform can
    be exercised against fakes, and so a future broker swap touches one module.
    """

    def __init__(
        self,
        settings: GatewaySettings | None = None,
        *,
        ib_factory: Callable[[], Any] | None = None,
        pacer: HistoricalPacer | None = None,
    ) -> None:
        self.settings = settings or GatewaySettings.from_environment()
        self._ib_factory = ib_factory or _default_ib_factory
        self._pacer = pacer or HistoricalPacer()
        self._ib: Any | None = None
        self._account: str | None = None

    # ---- session lifecycle -------------------------------------------------

    def connect(self) -> IBKRGateway:
        if self.is_connected:
            return self
        ib = self._ib_factory()
        ib.connect(
            self.settings.host,
            self.settings.port,
            clientId=self.settings.client_id,
            timeout=self.settings.connect_timeout_seconds,
            readonly=self.settings.readonly,
        )
        self._ib = ib
        self._account = self._resolve_account(ib.managedAccounts())
        logger.info(
            "connected to IBKR gateway %s:%s account=%s readonly=%s",
            self.settings.host,
            self.settings.port,
            self._account,
            self.settings.readonly,
        )
        return self

    def disconnect(self) -> None:
        if self._ib is not None and self._ib.isConnected():
            self._ib.disconnect()
        self._ib = None
        self._account = None

    def __enter__(self) -> IBKRGateway:
        return self.connect()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.disconnect()

    @property
    def is_connected(self) -> bool:
        return self._ib is not None and bool(self._ib.isConnected())

    @property
    def account(self) -> str:
        if self._account is None:
            raise GatewayNotConnectedError("Gateway session has no resolved account")
        return self._account

    def _require_ib(self) -> Any:
        if self._ib is None or not self._ib.isConnected():
            raise GatewayNotConnectedError("Gateway is not connected; call connect() first")
        return self._ib

    def _resolve_account(self, managed: Sequence[str]) -> str:
        accounts = [item for item in managed if item.strip()]
        if not accounts:
            raise GatewayNotConnectedError("Gateway returned no managed accounts")
        if self.settings.account:
            if self.settings.account not in accounts:
                raise GatewayNotConnectedError(
                    f"Account {self.settings.account} is not managed by this session: {accounts}"
                )
            chosen = self.settings.account
        elif len(accounts) > 1:
            raise GatewayNotConnectedError(
                f"Session manages several accounts {accounts}; set GatewaySettings.account"
            )
        else:
            chosen = accounts[0]
        self.settings.verify_account(chosen)
        return chosen

    # ---- reference data ----------------------------------------------------

    def resolve_equity(self, symbol: str, *, primary_exchange: str | None = None) -> USEquity:
        """Resolve a ticker to a single US equity contract."""
        ib = self._require_ib()
        normalized = normalize_symbol(symbol)
        stock = _make_stock(normalized, primary_exchange)
        details = ib.reqContractDetails(stock)
        if not details:
            raise ContractResolutionError(f"{symbol} returned no contract details")
        candidates = [_contract_as_dict(item.contract) for item in details]
        chosen = select_primary_listing(candidates, normalized)
        return USEquity(
            symbol=normalized,
            con_id=int(chosen["conId"]),
            primary_exchange=str(chosen.get("primaryExchange", "")).upper(),
            currency=str(chosen.get("currency", US_EQUITY_CURRENCY)).upper(),
            local_symbol=str(chosen.get("localSymbol", "")),
            trading_class=str(chosen.get("tradingClass", "")),
        )

    def resolve_universe(self, symbols: Iterable[str]) -> tuple[list[USEquity], dict[str, str]]:
        """Resolve many symbols, returning the successes and a symbol->reason failure map."""
        resolved: list[USEquity] = []
        failures: dict[str, str] = {}
        for symbol in symbols:
            try:
                resolved.append(self.resolve_equity(symbol))
            except (ContractResolutionError, ValueError) as error:
                failures[symbol] = str(error)
                logger.warning("could not resolve %s: %s", symbol, error)
        return resolved, failures

    # ---- market data -------------------------------------------------------

    def daily_bars(
        self,
        equity: USEquity,
        *,
        end: date | datetime | str | None = None,
        duration: str = "1 Y",
        what_to_show: str = TOTAL_RETURN_PRICE_BASIS,
        use_regular_trading_hours: bool = True,
        timeout_seconds: float = 120.0,
    ) -> pd.DataFrame:
        """Download daily bars for one window, respecting IBKR historical pacing.

        ``end=None`` anchors the window at the present, which is the only form
        ``ADJUSTED_LAST`` accepts; an explicit end date returns an empty series
        for that price basis rather than raising.
        """
        ib = self._require_ib()
        end_text = _format_end_datetime(end)
        if end_text and what_to_show == TOTAL_RETURN_PRICE_BASIS:
            raise ValueError(
                "ADJUSTED_LAST cannot be requested with an explicit end date; "
                "anchor the request at the present and trim client-side"
            )
        self._pacer.acquire(f"{equity.con_id}:{what_to_show}:{duration}:{end_text}")
        bars = ib.reqHistoricalData(
            _stock_from_equity(equity),
            endDateTime=end_text,
            durationStr=duration,
            barSizeSetting=DAILY_BAR_SIZE,
            whatToShow=what_to_show,
            useRTH=use_regular_trading_hours,
            formatDate=1,
            timeout=timeout_seconds,
        )
        return _bars_to_frame(bars)

    # ---- account state -----------------------------------------------------

    def account_summary(self) -> AccountSummary:
        ib = self._require_ib()
        values = {
            item.tag: item.value
            for item in ib.accountValues(self.account)
            if item.currency in {US_EQUITY_CURRENCY, ""}
        }
        return AccountSummary(
            account=self.account,
            is_paper=self.account.upper().startswith(("DU", "DF")),
            net_liquidation=_as_float(values.get("NetLiquidation")),
            total_cash=_as_float(values.get("TotalCashValue")),
            buying_power=_as_float(values.get("BuyingPower")),
            unrealized_pnl=_as_float(values.get("UnrealizedPnL")),
            realized_pnl=_as_float(values.get("RealizedPnL")),
        )

    def positions(self) -> list[Position]:
        ib = self._require_ib()
        records: list[Position] = []
        for item in ib.portfolio(self.account):
            contract = item.contract
            records.append(
                Position(
                    account=self.account,
                    symbol=str(contract.symbol),
                    con_id=int(contract.conId),
                    quantity=float(item.position),
                    average_cost=float(item.averageCost),
                    market_price=float(item.marketPrice),
                    market_value=float(item.marketValue),
                    unrealized_pnl=float(item.unrealizedPNL),
                )
            )
        return records

    # ---- order path --------------------------------------------------------

    def preview_order(
        self, equity: USEquity, order: Any, *, timeout_seconds: float = 30.0
    ) -> dict[str, Any]:
        """Run IBKR's ``whatIf`` margin check without routing the order.

        A rejected what-if request is answered with an error message and no
        OrderState, so the call is bounded by a timeout rather than left to wait
        forever on a reply that will not come.
        """
        ib = self._require_ib()
        probe = _copy_order_as_what_if(order)
        state = _with_timeout(
            ib.whatIfOrderAsync(_stock_from_equity(equity), probe),
            timeout_seconds,
            f"whatIf preview for {equity.symbol}",
        )
        return {
            "symbol": equity.panel_symbol,
            "action": getattr(order, "action", ""),
            "quantity": float(getattr(order, "totalQuantity", 0.0) or 0.0),
            "order_type": getattr(order, "orderType", ""),
            "commission": _as_optional_float(getattr(state, "commission", None)),
            "commission_currency": getattr(state, "commissionCurrency", "") or US_EQUITY_CURRENCY,
            "initial_margin_after": _as_optional_float(getattr(state, "initMarginAfter", None)),
            "maintenance_margin_after": _as_optional_float(
                getattr(state, "maintMarginAfter", None)
            ),
            "equity_with_loan_after": _as_optional_float(
                getattr(state, "equityWithLoanAfter", None)
            ),
            "warning": getattr(state, "warningText", "") or "",
        }

    def transmit_order(self, equity: USEquity, order: Any) -> Any:
        """Submit a live order. Blocked unless the session was opened writable."""
        if self.settings.readonly:
            raise OrderTransmissionBlocked(
                "This gateway session is read-only. Rebuild it with "
                "GatewaySettings.writable() to transmit orders."
            )
        ib = self._require_ib()
        self.settings.verify_account(self.account)
        return ib.placeOrder(_stock_from_equity(equity), order)


# ---- module helpers --------------------------------------------------------


def _default_ib_factory() -> Any:
    from ib_async import IB

    return IB()


def _with_timeout(coroutine: Any, timeout_seconds: float, description: str) -> Any:
    """Run an ib_async coroutine on its event loop under a wall-clock timeout."""
    import asyncio

    from ib_async import util

    try:
        return util.run(asyncio.wait_for(coroutine, timeout=timeout_seconds))
    except TimeoutError as error:
        raise GatewayTimeoutError(
            f"{description} did not complete within {timeout_seconds:.0f}s; "
            "check the gateway log for a rejected request"
        ) from error


def _make_stock(symbol: str, primary_exchange: str | None) -> Any:
    from ib_async import Stock

    if primary_exchange:
        return Stock(symbol, "SMART", US_EQUITY_CURRENCY, primaryExchange=primary_exchange)
    return Stock(symbol, "SMART", US_EQUITY_CURRENCY)


def _stock_from_equity(equity: USEquity) -> Any:
    from ib_async import Stock

    return Stock(
        equity.symbol,
        equity.routing_exchange,
        equity.currency,
        primaryExchange=equity.primary_exchange,
        conId=equity.con_id,
    )


def _copy_order_as_what_if(order: Any) -> Any:
    """Copy an order into a margin-preview probe.

    ``whatIf=True`` is what stops the order from ever reaching a venue: TWS
    computes margin and commission and returns an OrderState instead of routing
    anything. Counterintuitively ``transmit`` must also be True — TWS rejects a
    what-if request with ``transmit=False`` (error 321) and then never replies,
    which reads as a hang rather than a failure.
    """
    from copy import copy

    probe = copy(order)
    probe.whatIf = True
    probe.transmit = True
    return probe


def _contract_as_dict(contract: Any) -> dict[str, Any]:
    return {
        "conId": getattr(contract, "conId", 0),
        "symbol": getattr(contract, "symbol", ""),
        "currency": getattr(contract, "currency", ""),
        "primaryExchange": getattr(contract, "primaryExchange", ""),
        "localSymbol": getattr(contract, "localSymbol", ""),
        "tradingClass": getattr(contract, "tradingClass", ""),
    }


def _format_end_datetime(end: date | datetime | str | None) -> str:
    if end is None or end == "":
        return ""
    if isinstance(end, str):
        return end
    if isinstance(end, datetime):
        return end.strftime("%Y%m%d %H:%M:%S")
    # A bare date means "through this session's close"; IBKR reads 23:59:59 as EOD.
    return datetime(end.year, end.month, end.day, 23, 59, 59).strftime("%Y%m%d %H:%M:%S")


def _bars_to_frame(bars: Sequence[Any]) -> pd.DataFrame:
    if not bars:
        return pd.DataFrame(columns=list(BAR_COLUMNS))
    rows = [
        {
            "date": pd.Timestamp(bar.date),
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": float(bar.volume),
            "average": float(getattr(bar, "average", 0.0) or 0.0),
            "bar_count": int(getattr(bar, "barCount", 0) or 0),
        }
        for bar in bars
    ]
    frame = pd.DataFrame(rows, columns=list(BAR_COLUMNS))
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    return frame.sort_values("date").reset_index(drop=True)


def _as_float(value: Any) -> float:
    number = _as_optional_float(value)
    return 0.0 if number is None else number


def _as_optional_float(value: Any) -> float | None:
    """Parse a broker numeric field, mapping IBKR's unset sentinel to None.

    Returning None rather than 0.0 keeps "the gateway did not compute this"
    distinguishable from "this really is zero", which matters for commission
    estimates that callers may sum.
    """
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number >= UNSET_DOUBLE or number != number:
        return None
    return number
