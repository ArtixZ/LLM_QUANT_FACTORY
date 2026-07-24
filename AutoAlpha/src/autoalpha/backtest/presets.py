from __future__ import annotations

from typing import Any

A_SHARE_REALISTIC_WEEKLY_V1 = "A_SHARE_REALISTIC_WEEKLY_V1"
A_SHARE_NON_PIT_PROXY_WEEKLY_V1 = "A_SHARE_NON_PIT_PROXY_WEEKLY_V1"

MANUAL_BACKTEST_PRESETS: dict[str, dict[str, Any]] = {
    A_SHARE_REALISTIC_WEEKLY_V1: {
        "preset_id": A_SHARE_REALISTIC_WEEKLY_V1,
        "name": "A股真实交易 · 周频",
        "description": (
            "日频数据可支持的生产预检口径：周首个交易日开盘换仓，失败订单顺延，"
            "真实现金、整手、交易状态、成交量、滑点及历史费率约束。"
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
            "lot_size": 100,
            "maximum_volume_participation": 0.01,
            "opening_limit_threshold": 0.095,
            "commission_bps_each_side": 2.5,
            "stamp_duty_bps_sell": 5.0,
            "transfer_fee_bps_each_side": 0.1,
            "minimum_commission_cny": 5.0,
            "slippage_bps_each_side": 5.0,
            "use_historical_fee_schedule": True,
            "cost_stress_multiplier": 3.0,
        },
        "requirements": [
            "unadjusted OHLC",
            "volume in shares and amount in CNY",
            "point-in-time listing, ST, suspension, price-limit and execution flags",
        ],
        "limitations": [
            "daily bars cannot reproduce opening-auction queue priority",
            "slippage is a fixed conservative proxy rather than an intraday impact model",
            "corporate-action cash flows require validated unadjusted market data",
        ],
    },
    A_SHARE_NON_PIT_PROXY_WEEKLY_V1: {
        "preset_id": A_SHARE_NON_PIT_PROXY_WEEKLY_V1,
        "name": "A股现金账本代理 · 周频（非PIT）",
        "description": (
            "前复权价格用于因子研究、未复权开盘价用于现金成交；使用当前名称筛选与开盘"
            "涨跌幅推导的成交限制代理。仅供研究，不能替代点时点生产预检。"
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
            "lot_size": 100,
            "maximum_volume_participation": 0.01,
            "opening_limit_threshold": 0.095,
            "commission_bps_each_side": 2.5,
            "stamp_duty_bps_sell": 5.0,
            "transfer_fee_bps_each_side": 0.1,
            "minimum_commission_cny": 5.0,
            "slippage_bps_each_side": 5.0,
            "use_historical_fee_schedule": True,
            "cost_stress_multiplier": 3.0,
        },
        "requirements": [
            "qfq research OHLC plus unadjusted execution OHLC",
            "volume in shares and amount in CNY",
            "explicit NON_PIT_PROXY data metadata",
        ],
        "limitations": [
            "current-name universe filtering is not historical ST or delisting state",
            (
                "opening eligibility is inferred from raw open movement, not official "
                "point-in-time flags"
            ),
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
