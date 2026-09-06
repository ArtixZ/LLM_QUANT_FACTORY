from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal

Side = Literal["BUY", "SELL"]

# Regulatory fees are charged on the sell side only. The SEC Section 31 rate is
# reset by the Commission periodically, so it is a plain parameter here rather
# than a hardcoded schedule; supply `historical_sec_rates` to backtest across a
# rate change instead of guessing one.
DEFAULT_SEC_FEE_PER_MILLION_USD = 20.60
DEFAULT_FINRA_TAF_PER_SHARE = 0.000195
DEFAULT_FINRA_TAF_MAXIMUM_USD = 9.79

# Interactive Brokers US equity tiered pricing.
DEFAULT_COMMISSION_PER_SHARE = 0.0035
DEFAULT_MINIMUM_COMMISSION_USD = 0.35
DEFAULT_MAXIMUM_COMMISSION_FRACTION = 0.01


@dataclass(frozen=True)
class USEquityExecutionCosts:
    """Explicit US equity fees; spread and impact are added by the execution layer.

    Unlike a bps-only model, US commissions are charged per share with a floor
    and a percent-of-notional ceiling, so share count is a required input: a
    5,000-share trade in a $2 stock and a 50-share trade in a $200 stock have
    the same notional and very different commissions.
    """

    commission_per_share: float = DEFAULT_COMMISSION_PER_SHARE
    minimum_commission_usd: float = DEFAULT_MINIMUM_COMMISSION_USD
    maximum_commission_fraction: float = DEFAULT_MAXIMUM_COMMISSION_FRACTION
    sec_fee_per_million_usd_sell: float = DEFAULT_SEC_FEE_PER_MILLION_USD
    finra_taf_per_share_sell: float = DEFAULT_FINRA_TAF_PER_SHARE
    finra_taf_maximum_usd_sell: float = DEFAULT_FINRA_TAF_MAXIMUM_USD
    historical_sec_rates: tuple[tuple[date, float], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        values = (
            self.commission_per_share,
            self.minimum_commission_usd,
            self.sec_fee_per_million_usd_sell,
            self.finra_taf_per_share_sell,
            self.finra_taf_maximum_usd_sell,
        )
        if any(value < 0 for value in values):
            raise ValueError("Execution costs must be non-negative")
        if not 0 < self.maximum_commission_fraction <= 1:
            raise ValueError("maximum_commission_fraction must be in (0, 1]")
        if any(rate < 0 for _, rate in self.historical_sec_rates):
            raise ValueError("Historical SEC rates must be non-negative")

    def fees(
        self,
        side: Side,
        notional: float,
        shares: float,
        trade_date: date | datetime | str | None = None,
    ) -> float:
        return float(sum(self.fee_breakdown(side, notional, shares, trade_date).values()))

    def fee_breakdown(
        self,
        side: Side,
        notional: float,
        shares: float,
        trade_date: date | datetime | str | None = None,
    ) -> dict[str, float]:
        if side not in {"BUY", "SELL"}:
            raise ValueError(f"Unknown side: {side!r}")
        if shares < 0:
            raise ValueError("Share count cannot be negative")
        if notional <= 0 or shares <= 0:
            return {"commission": 0.0, "sec_fee": 0.0, "finra_taf": 0.0}
        commission = min(
            max(self.minimum_commission_usd, shares * self.commission_per_share),
            notional * self.maximum_commission_fraction,
        )
        sec_fee = 0.0
        finra_taf = 0.0
        if side == "SELL":
            sec_rate = self._sec_rate(trade_date)
            sec_fee = notional * sec_rate / 1_000_000.0
            finra_taf = min(
                shares * self.finra_taf_per_share_sell, self.finra_taf_maximum_usd_sell
            )
        return {
            "commission": float(commission),
            "sec_fee": float(sec_fee),
            "finra_taf": float(finra_taf),
        }

    def affordable_shares(
        self,
        cash: float,
        price: float,
        trade_date: date | datetime | str | None = None,
    ) -> int:
        """Largest whole-share buy whose notional plus commission fits inside cash."""
        if price <= 0 or cash <= self.minimum_commission_usd:
            return 0
        shares = int(cash // price)
        while shares > 0:
            notional = shares * price
            if notional + self.fees("BUY", notional, shares, trade_date) <= cash:
                return shares
            shares -= 1
        return 0

    def _sec_rate(self, trade_date: date | datetime | str | None) -> float:
        effective = _coerce_date(trade_date)
        if effective is None or not self.historical_sec_rates:
            return self.sec_fee_per_million_usd_sell
        applicable = [
            rate for start, rate in sorted(self.historical_sec_rates) if start <= effective
        ]
        return applicable[-1] if applicable else self.sec_fee_per_million_usd_sell


def _coerce_date(value: date | datetime | str | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])
