from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

Side = Literal["BUY", "SELL"]


@dataclass(frozen=True)
class ChinaAExecutionCosts:
    """Explicit A-share fees; impact and spread are added by the execution layer later."""

    commission_bps_each_side: float = 1.5
    stamp_duty_bps_sell: float = 5.0
    transfer_fee_bps_each_side: float = 0.1
    minimum_commission_cny: float = 5.0
    use_historical_fee_schedule: bool = False

    def __post_init__(self) -> None:
        if any(
            value < 0
            for value in (
                self.commission_bps_each_side,
                self.stamp_duty_bps_sell,
                self.transfer_fee_bps_each_side,
                self.minimum_commission_cny,
            )
        ):
            raise ValueError("Execution costs must be non-negative")

    def fees(
        self,
        side: Side,
        notional: float,
        trade_date: date | datetime | str | None = None,
    ) -> float:
        return float(sum(self.fee_breakdown(side, notional, trade_date).values()))

    def fee_breakdown(
        self,
        side: Side,
        notional: float,
        trade_date: date | datetime | str | None = None,
    ) -> dict[str, float]:
        if side not in {"BUY", "SELL"}:
            raise ValueError(f"Unknown side: {side!r}")
        if notional <= 0:
            return {"commission": 0.0, "transfer_fee": 0.0, "stamp_duty": 0.0}
        commission = max(
            self.minimum_commission_cny,
            notional * self.commission_bps_each_side / 10_000.0,
        )
        effective_date = _coerce_date(trade_date)
        transfer_bps = self.transfer_fee_bps_each_side
        stamp_bps = self.stamp_duty_bps_sell
        if self.use_historical_fee_schedule and effective_date is not None:
            if effective_date < date(2022, 4, 29):
                transfer_bps *= 2.0
            if effective_date < date(2023, 8, 28):
                stamp_bps *= 2.0
        transfer = notional * transfer_bps / 10_000.0
        stamp = notional * stamp_bps / 10_000.0 if side == "SELL" else 0.0
        return {
            "commission": float(commission),
            "transfer_fee": float(transfer),
            "stamp_duty": float(stamp),
        }

    def affordable_notional(
        self,
        cash: float,
        trade_date: date | datetime | str | None = None,
    ) -> float:
        """Largest buy notional whose notional plus explicit fees does not exceed cash."""
        if cash <= self.minimum_commission_cny:
            return 0.0
        low, high = 0.0, float(cash)
        for _ in range(60):
            middle = (low + high) / 2.0
            if middle + self.fees("BUY", middle, trade_date) <= cash:
                low = middle
            else:
                high = middle
        return low


def _coerce_date(value: date | datetime | str | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])
