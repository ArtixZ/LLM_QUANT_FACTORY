from __future__ import annotations

from typing import Any

US_EQUITY_REALISTIC_WEEKLY_V1 = "US_EQUITY_REALISTIC_WEEKLY_V1"
US_EQUITY_ADJUSTED_PROXY_WEEKLY_V1 = "US_EQUITY_ADJUSTED_PROXY_WEEKLY_V1"

_US_COST_SETTINGS: dict[str, Any] = {
    "commission_per_share": 0.0035,
    "minimum_commission_usd": 0.35,
    "maximum_commission_fraction": 0.01,
    "sec_fee_per_million_usd_sell": 20.60,
    "finra_taf_per_share_sell": 0.000195,
    "slippage_bps_each_side": 5.0,
    "cost_stress_multiplier": 3.0,
}

MANUAL_BACKTEST_PRESETS: dict[str, dict[str, Any]] = {
    US_EQUITY_REALISTIC_WEEKLY_V1: {
        "preset_id": US_EQUITY_REALISTIC_WEEKLY_V1,
        "name": "US equities · weekly",
        "description": (
            "Production pre-check on daily bars: rebalance at the first session open of "
            "each week, carry failed orders forward, and enforce real cash, tradability, "
            "volume participation, slippage, and the IBKR per-share fee schedule."
        ),
        "settings": {
            "backtest_engine": "EVENT_LEDGER",
            "execution_data_mode": "STRICT_PIT",
            "vector_cost_model": "side_aware",
            "product_template": "LONG_ONLY_CAPITAL",
            "rebalance_schedule": "WEEKLY_FIRST_SESSION",
            "gross_exposure": 0.90,
            "holding_period_days": 5,
            "selection_fraction": 0.10,
            "maximum_positions": 30,
            "lot_size": 1,
            "maximum_volume_participation": 0.01,
            **_US_COST_SETTINGS,
        },
        "requirements": [
            "split-adjusted execution OHLC",
            "volume in shares and dollar volume in USD",
            "point-in-time listing, delisting, halt, and execution flags",
        ],
        "limitations": [
            "daily bars cannot reproduce opening-auction queue priority",
            "slippage is a fixed conservative proxy rather than an intraday impact model",
            "intraday LULD halts are invisible; only no-print sessions are excluded",
            "IBKR does not serve truly unadjusted bars, so splits are already applied",
        ],
    },
    US_EQUITY_ADJUSTED_PROXY_WEEKLY_V1: {
        "preset_id": US_EQUITY_ADJUSTED_PROXY_WEEKLY_V1,
        "name": "US equities cash-ledger proxy · weekly (non-PIT)",
        "description": (
            "Dividend-adjusted prices drive research signals while split-adjusted opens "
            "drive cash fills. The universe is current membership only. Research use "
            "only; this cannot substitute for a point-in-time production pre-check."
        ),
        "settings": {
            "backtest_engine": "EVENT_LEDGER",
            "execution_data_mode": "NON_PIT_PROXY",
            "vector_cost_model": "side_aware",
            "product_template": "LONG_ONLY_CAPITAL",
            "rebalance_schedule": "WEEKLY_FIRST_SESSION",
            "gross_exposure": 0.90,
            "holding_period_days": 5,
            "selection_fraction": 0.10,
            "maximum_positions": 30,
            "lot_size": 1,
            "maximum_volume_participation": 0.01,
            **_US_COST_SETTINGS,
        },
        "requirements": [
            "ADJUSTED_LAST research OHLC plus TRADES execution OHLC",
            "volume in shares and dollar volume in USD",
            "explicit NON_PIT_PROXY panel metadata",
        ],
        "limitations": [
            "current-membership universe carries survivorship bias",
            "adjustment factors are as of the download date, not point-in-time",
            "tradability is inferred from bar validity and volume, not official halt feeds",
            "result is research-only and must not be promoted to production",
        ],
    },
}


def manual_backtest_preset_catalog() -> list[dict[str, Any]]:
    return list(MANUAL_BACKTEST_PRESETS.values())


def validate_preset_settings(preset_id: str, values: dict[str, Any]) -> None:
    if preset_id == "CUSTOM":
        return
    try:
        expected = MANUAL_BACKTEST_PRESETS[preset_id]["settings"]
    except KeyError as error:
        raise ValueError(f"Unknown manual backtest preset: {preset_id}") from error
    mismatches = [
        key
        for key, expected_value in expected.items()
        if values.get(key) != expected_value
    ]
    if mismatches:
        raise ValueError(
            f"Preset {preset_id} was modified; use CUSTOM or restore: {', '.join(mismatches)}"
        )
