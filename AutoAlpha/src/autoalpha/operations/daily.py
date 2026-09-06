"""Daily paper-trading operations: refresh data, form a target book, report.

The run is deliberately split into a pure reporting stage and an optional,
double-gated submission stage. Everything up to and including the what-if
preview is side-effect free at the broker, so the job can run unattended and
still be trusted; transmitting orders requires both a writable gateway session
and an explicit confirmation.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.dataset as ds

from autoalpha.backtest.costs import USEquityExecutionCosts
from autoalpha.data.universe_catalog import resolve_universe
from autoalpha.ibkr.client import (
    AccountSummary,
    IBKRGateway,
    OpenOrder,
    OrderTransmissionBlocked,
    Position,
)
from autoalpha.ibkr.contracts import panel_symbol
from autoalpha.ibkr.orders import PlannedOrder, plan_orders, preview_plan
from autoalpha.ibkr.settings import GatewaySettings

logger = logging.getLogger(__name__)

DEFAULT_MARKET_DATA_ROOT = Path.home() / "MarketData" / "US"
Severity = str  # "ok" | "info" | "warn" | "error"


@dataclass(frozen=True)
class DailyConfig:
    """Everything the daily run needs, so a scheduled job stays reproducible."""

    universe: str = "MEGA_CAP_LIQUID_V1"
    market_data_root: Path = DEFAULT_MARKET_DATA_ROOT
    history_start: date = date(2016, 1, 1)
    # 12-1 momentum: a 12-month lookback that skips the most recent month.
    lookback_sessions: int = 252
    skip_sessions: int = 21
    position_count: int = 5
    gross_exposure: float = 0.95
    minimum_shares: int = 1
    order_type: str = "MOO"

    def __post_init__(self) -> None:
        if self.position_count <= 0:
            raise ValueError("position_count must be positive")
        if not 0 < self.gross_exposure <= 1:
            raise ValueError("gross_exposure must be in (0, 1]")
        if self.lookback_sessions <= self.skip_sessions:
            raise ValueError("lookback_sessions must exceed skip_sessions")

    @property
    def panel_path(self) -> Path:
        return self.market_data_root / "processed" / "daily_panel"

    @property
    def quality_report_path(self) -> Path:
        return self.market_data_root / "catalog" / "data_quality.json"


@dataclass(frozen=True)
class DataHealth:
    panel_last_date: str
    panel_rows: int
    panel_symbols: int
    audit_passed: bool
    stale_symbols: dict[str, str] = field(default_factory=dict)
    sync_written: int = 0
    sync_failures: dict[str, str] = field(default_factory=dict)

    @property
    def is_healthy(self) -> bool:
        return self.audit_passed and not self.sync_failures and not self.stale_symbols


@dataclass(frozen=True)
class DailyReport:
    as_of: str
    account: str
    is_paper: bool
    net_liquidation: float
    total_cash: float
    unrealized_pnl: float
    realized_pnl: float
    positions: list[dict[str, Any]]
    health: DataHealth
    picks: list[dict[str, Any]]
    plan: list[dict[str, Any]]
    previews: list[dict[str, Any]]
    modeled_commission: float
    plan_notional: float
    submitted: bool
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["severity"] = self.severity
        return value

    @property
    def severity(self) -> Severity:
        if not self.health.audit_passed or self.health.sync_failures:
            return "error"
        if self.health.stale_symbols or any("error" in p for p in self.previews):
            return "warn"
        return "ok"

    @property
    def title(self) -> str:
        state = "submitted" if self.submitted else "preview only"
        return f"Daily run {self.as_of} ({state})"

    def telegram_body(self) -> str:
        """Plain-text digest. notify.sh escapes the body, so no markup here."""
        money = lambda v: f"${v:,.0f}"  # noqa: E731 - local formatting shorthand
        lines = [
            f"Account {self.account} ({'paper' if self.is_paper else 'LIVE'})",
            f"NAV {money(self.net_liquidation)} · cash {money(self.total_cash)}"
            f" · unrealized {money(self.unrealized_pnl)}",
            "",
            f"Data: panel through {self.health.panel_last_date}"
            f" · {self.health.panel_symbols} symbols · {self.health.panel_rows:,} rows",
        ]
        if self.health.stale_symbols:
            stale = ", ".join(sorted(self.health.stale_symbols))
            lines.append(f"  stale, excluded: {stale}")
        if self.health.sync_failures:
            lines.append(f"  sync failures: {', '.join(sorted(self.health.sync_failures))}")

        lines.append("")
        if self.positions:
            lines.append(f"Positions ({len(self.positions)}):")
            for p in self.positions:
                lines.append(
                    f"  {p['symbol']:<6} {p['quantity']:>7,.0f} sh"
                    f" · {money(p['market_value'])} · uPnL {money(p['unrealized_pnl'])}"
                )
        else:
            lines.append("Positions: flat")

        lines.append("")
        lines.append(f"Signal (12-1 momentum, top {len(self.picks)}):")
        for pick in self.picks:
            lines.append(
                f"  {pick['symbol']:<6} {pick['score']:>+7.1%} @ {money(pick['price'])}"
                f" -> {pick['target_shares']:,} sh"
            )

        lines.append("")
        if self.plan:
            lines.append(f"Orders planned ({len(self.plan)}, {self.plan[0]['order_type']}):")
            for order in self.plan:
                lines.append(
                    f"  {order['action']:<4} {order['symbol']:<6}"
                    f" {order['quantity']:>7,} sh · {money(order['notional'])}"
                )
            lines.append(
                f"  total {money(self.plan_notional)}"
                f" · modeled commission ${self.modeled_commission:,.2f}"
            )
            lines.append(
                "  SUBMITTED to the broker" if self.submitted
                else "  NOT submitted - preview only"
            )
        else:
            lines.append("Orders planned: none (book already on target)")

        for note in self.notes:
            lines.append(f"! {note}")
        return "\n".join(lines)


def load_panel(
    config: DailyConfig,
    *,
    exclude: set[str] | None = None,
    include: set[str] | None = None,
) -> pd.DataFrame:
    """Read the research panel, dropping symbols the audit flagged as stale."""
    files = sorted(config.panel_path.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No panel partitions under {config.panel_path}")
    frame = ds.dataset(files, format="parquet").to_table().to_pandas()
    if exclude:
        frame = frame[~frame["symbol"].isin(exclude)]
    if include is not None:
        frame = frame[frame["symbol"].isin(include)]
    if frame.empty:
        raise ValueError("No panel rows remain inside the configured strategy universe")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    return frame


def momentum_scores(panel: pd.DataFrame, config: DailyConfig) -> pd.Series:
    """Latest cross-section of the configured momentum signal."""
    close = panel.pivot(index="trade_date", columns="symbol", values="close").sort_index()
    if len(close) <= config.lookback_sessions:
        raise ValueError(
            f"Panel has {len(close)} sessions, need more than {config.lookback_sessions}"
        )
    scores = close.shift(config.skip_sessions) / close.shift(config.lookback_sessions) - 1.0
    return scores.iloc[-1].dropna()


def build_target_book(
    panel: pd.DataFrame,
    config: DailyConfig,
    net_liquidation: float,
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    """Equal-weight the top-scoring names into whole-share targets."""
    scores = momentum_scores(panel, config)
    price_field = "raw_close" if "raw_close" in panel.columns else "close"
    prices = panel.pivot(index="trade_date", columns="symbol", values=price_field).sort_index()
    last_price = prices.iloc[-1]
    picks = scores.nlargest(config.position_count)
    budget = net_liquidation * config.gross_exposure / max(len(picks), 1)

    targets: dict[str, int] = {}
    detail: list[dict[str, Any]] = []
    for symbol, score in picks.items():
        price = float(last_price.get(symbol, float("nan")))
        if not price > 0 or pd.isna(price):
            logger.warning("skipping %s: no usable price in the panel", symbol)
            continue
        shares = int(budget // price)
        targets[str(symbol)] = shares
        detail.append(
            {
                "symbol": str(symbol),
                "score": float(score),
                "price": price,
                "target_shares": shares,
            }
        )
    return targets, detail


def run_daily(
    config: DailyConfig,
    *,
    gateway: IBKRGateway | None = None,
    settings: GatewaySettings | None = None,
    health: DataHealth,
    submit: bool = False,
    confirm_submit: bool = False,
    managed_account: str | None = None,
    submission_key: str | None = None,
    as_of_date: date | None = None,
) -> DailyReport:
    """Produce the daily report, previewing (never transmitting) the order plan.

    ``submit`` is threaded through for completeness but requires
    ``confirm_submit`` as well as a writable session; the default run is
    read-only and safe to schedule unattended.
    """
    owns_session = gateway is None
    session = gateway or IBKRGateway(settings or GatewaySettings.from_environment())
    if owns_session:
        session.connect()
    notes: list[str] = []
    try:
        run_date = as_of_date or date.today()
        account: AccountSummary = session.account_summary()
        positions: list[Position] = session.positions()
        _, configured_symbols = resolve_universe(config.universe)
        managed_symbols = {panel_symbol(symbol) for symbol in configured_symbols}
        panel = load_panel(
            config,
            exclude=set(health.stale_symbols),
            include=managed_symbols,
        )
        targets, picks = build_target_book(panel, config, account.net_liquidation)

        normalized_positions = {_position_symbol(position): position for position in positions}
        unmanaged_positions = [
            position
            for symbol, position in normalized_positions.items()
            if symbol not in managed_symbols
        ]
        if unmanaged_positions:
            notes.append(
                "unmanaged account positions were excluded from the strategy plan: "
                + ", ".join(sorted(position.symbol for position in unmanaged_positions))
            )
        current = {
            symbol: position.quantity
            for symbol, position in normalized_positions.items()
            if symbol in managed_symbols
        }
        price_field = "raw_close" if "raw_close" in panel.columns else "close"
        close = panel.pivot(
            index="trade_date", columns="symbol", values=price_field
        ).sort_index()
        plan = plan_orders(
            targets,
            current,
            reference_prices=close.iloc[-1].to_dict(),
            order_type=config.order_type,  # type: ignore[arg-type]
            minimum_shares=config.minimum_shares,
        )
        contracts = {}
        if plan:
            resolved, failures = session.resolve_universe([o.symbol for o in plan])
            contracts = {equity.panel_symbol: equity for equity in resolved}
            for symbol, reason in failures.items():
                notes.append(f"contract unresolved for {symbol}: {reason}")
        previews = preview_plan(session, plan, contracts) if plan else []

        costs = USEquityExecutionCosts()
        commission = sum(costs.fees(o.action, o.notional, o.quantity) for o in plan)
        notional = sum(o.notional for o in plan)

        submitted = False
        if submit:
            if not confirm_submit:
                notes.append("submission requested without confirmation; nothing was sent")
            else:
                from autoalpha.ibkr.orders import submit_plan

                open_orders = session.open_orders()
                blockers = _submission_blockers(
                    health=health,
                    account=account,
                    managed_account=managed_account,
                    submission_key=submission_key,
                    run_date=run_date,
                    previews=previews,
                    unmanaged_positions=unmanaged_positions,
                    open_orders=open_orders,
                )
                if blockers:
                    raise OrderTransmissionBlocked(
                        "Order submission blocked: " + "; ".join(blockers)
                    )
                submit_plan(
                    session,
                    plan,
                    contracts,
                    confirm=True,
                    order_reference_prefix=str(submission_key),
                )
                submitted = True

        return DailyReport(
            as_of=run_date.isoformat(),
            account=account.account,
            is_paper=account.is_paper,
            net_liquidation=account.net_liquidation,
            total_cash=account.total_cash,
            unrealized_pnl=account.unrealized_pnl,
            realized_pnl=account.realized_pnl,
            positions=[_position_row(p) for p in positions],
            health=health,
            picks=picks,
            plan=[_plan_row(o) for o in plan],
            previews=previews,
            modeled_commission=commission,
            plan_notional=notional,
            submitted=submitted,
            notes=notes,
        )
    finally:
        if owns_session:
            session.disconnect()


def _position_row(position: Position) -> dict[str, Any]:
    return {
        "symbol": position.symbol,
        "quantity": position.quantity,
        "average_cost": position.average_cost,
        "market_price": position.market_price,
        "market_value": position.market_value,
        "unrealized_pnl": position.unrealized_pnl,
    }


def _plan_row(order: PlannedOrder) -> dict[str, Any]:
    return order.to_dict()


def _position_symbol(position: Position) -> str:
    try:
        return panel_symbol(position.symbol)
    except ValueError:
        return position.symbol.strip().upper()


def _submission_blockers(
    *,
    health: DataHealth,
    account: AccountSummary,
    managed_account: str | None,
    submission_key: str | None,
    run_date: date,
    previews: list[dict[str, Any]],
    unmanaged_positions: list[Position],
    open_orders: list[OpenOrder],
) -> list[str]:
    blockers: list[str] = []
    if not health.is_healthy:
        blockers.append("data audit, sync, and symbol freshness must all pass")
    if health.panel_last_date != run_date.isoformat():
        blockers.append(
            f"panel is through {health.panel_last_date}, expected {run_date.isoformat()}"
        )
    if not account.is_paper:
        blockers.append("automated strategy submission is restricted to paper accounts")
    if not managed_account or managed_account != account.account:
        blockers.append("managed account must explicitly match the connected account")
    if not submission_key:
        blockers.append("stable submission key is required")
    if unmanaged_positions:
        blockers.append("account contains positions outside the configured strategy universe")
    if open_orders:
        blockers.append("account has existing open orders")
    if any(preview.get("error") for preview in previews):
        blockers.append("one or more broker previews failed")
    if any(str(preview.get("warning") or "").strip() for preview in previews):
        blockers.append("one or more broker previews returned warnings")
    return blockers
