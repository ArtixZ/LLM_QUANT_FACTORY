from __future__ import annotations

import numpy as np
import pandas as pd


def transaction_cost_analysis(
    fills: pd.DataFrame,
    *,
    decision_price: float,
    close_price: float,
    side: str,
) -> dict[str, float]:
    if fills.empty:
        return {
            "filled_quantity": 0.0,
            "average_fill_price": float("nan"),
            "implementation_shortfall_bps": float("nan"),
            "timing_bps": float("nan"),
            "fees": 0.0,
            "spread_cost": 0.0,
            "impact_cost": 0.0,
        }
    direction = 1.0 if side == "BUY" else -1.0
    quantity = fills["quantity"].sum()
    average_price = float(np.average(fills["fill_price"], weights=fills["quantity"]))
    shortfall = direction * (average_price / decision_price - 1) * 10_000
    timing = direction * (close_price / decision_price - 1) * 10_000
    return {
        "filled_quantity": float(quantity),
        "average_fill_price": average_price,
        "implementation_shortfall_bps": float(shortfall),
        "timing_bps": float(timing),
        "fees": float(fills["fees"].sum()),
        "spread_cost": float(fills["spread_cost"].sum()),
        "impact_cost": float(fills["impact_cost"].sum()),
    }
