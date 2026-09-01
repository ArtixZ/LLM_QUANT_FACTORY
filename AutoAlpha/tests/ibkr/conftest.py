from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date

import pandas as pd
import pytest

from autoalpha.ibkr.client import EXECUTION_PRICE_BASIS, TOTAL_RETURN_PRICE_BASIS
from autoalpha.ibkr.contracts import USEquity

BarFactory = Callable[..., pd.DataFrame]
GatewayFactory = Callable[..., "FakeGateway"]


@dataclass
class FakeGateway:
    """Stands in for IBKRGateway wherever only ``daily_bars`` is exercised."""

    series: dict[str, pd.DataFrame] = field(default_factory=dict)
    empty_symbols: frozenset[str] = frozenset()
    calls: list[dict[str, object]] = field(default_factory=list)
    account: str = "DU000000"

    def daily_bars(
        self,
        equity: USEquity,
        *,
        end: date | None = None,
        duration: str = "1 Y",
        what_to_show: str = TOTAL_RETURN_PRICE_BASIS,
        **_: object,
    ) -> pd.DataFrame:
        self.calls.append(
            {"symbol": equity.symbol, "end": end, "duration": duration, "what": what_to_show}
        )
        if equity.symbol in self.empty_symbols:
            return pd.DataFrame()
        return self.series.get(what_to_show, pd.DataFrame()).copy()


@pytest.fixture
def make_bars() -> BarFactory:
    """Build a well-formed daily bar frame in the shape the client returns."""

    def factory(
        sessions: Sequence[str],
        *,
        close_start: float = 100.0,
        step: float = 1.0,
        volume: float = 1_000_000.0,
    ) -> pd.DataFrame:
        rows = []
        for index, session in enumerate(sessions):
            close = close_start + index * step
            rows.append(
                {
                    "date": pd.Timestamp(session),
                    "open": close - 0.5,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": volume,
                    "average": close,
                    "bar_count": 1_000,
                }
            )
        return pd.DataFrame(rows)

    return factory


@pytest.fixture
def make_gateway() -> GatewayFactory:
    def factory(
        adjusted: pd.DataFrame,
        traded: pd.DataFrame,
        *,
        empty_symbols: Sequence[str] = (),
    ) -> FakeGateway:
        return FakeGateway(
            series={
                TOTAL_RETURN_PRICE_BASIS: adjusted,
                EXECUTION_PRICE_BASIS: traded,
            },
            empty_symbols=frozenset(empty_symbols),
        )

    return factory


@pytest.fixture
def equity() -> USEquity:
    return USEquity(
        symbol="AAPL",
        con_id=265598,
        primary_exchange="NASDAQ",
        local_symbol="AAPL",
        trading_class="NMS",
    )


@pytest.fixture
def sessions() -> list[str]:
    return ["2026-08-03", "2026-08-04", "2026-08-05", "2026-08-06", "2026-08-07"]


@pytest.fixture
def gateway(
    make_bars: BarFactory, make_gateway: GatewayFactory, sessions: list[str]
) -> FakeGateway:
    return make_gateway(
        make_bars(sessions, close_start=99.0),
        make_bars(sessions, close_start=100.0),
    )
