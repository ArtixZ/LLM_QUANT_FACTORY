from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

AlertAction = Literal["REVIEW", "DEWEIGHT", "SUSPEND", "ROLLBACK"]


@dataclass(frozen=True)
class MonitorThresholds:
    maximum_missing_fraction: float = 0.05
    minimum_rolling_ic: float = -0.01
    maximum_exposure_zscore: float = 3.0
    maximum_shortfall_bps: float = 20.0
    maximum_pnl_deviation_zscore: float = 3.0


@dataclass(frozen=True)
class Alert:
    metric: str
    observed: float
    threshold: float
    action: AlertAction
    reason: str


class ProductionMonitor:
    def __init__(self, thresholds: MonitorThresholds | None = None) -> None:
        self.thresholds = thresholds or MonitorThresholds()

    def evaluate(
        self,
        *,
        missing_fraction: float,
        rolling_ic: float,
        exposure_zscore: float,
        shortfall_bps: float,
        pnl_deviation_zscore: float,
    ) -> tuple[Alert, ...]:
        checks = [
            (
                missing_fraction > self.thresholds.maximum_missing_fraction,
                Alert(
                    "data_missing",
                    missing_fraction,
                    self.thresholds.maximum_missing_fraction,
                    "SUSPEND",
                    "input data completeness breached",
                ),
            ),
            (
                rolling_ic < self.thresholds.minimum_rolling_ic,
                Alert(
                    "rolling_ic",
                    rolling_ic,
                    self.thresholds.minimum_rolling_ic,
                    "DEWEIGHT",
                    "predictive efficacy deteriorated",
                ),
            ),
            (
                abs(exposure_zscore) > self.thresholds.maximum_exposure_zscore,
                Alert(
                    "exposure_drift",
                    exposure_zscore,
                    self.thresholds.maximum_exposure_zscore,
                    "REVIEW",
                    "risk exposure moved outside its reference range",
                ),
            ),
            (
                shortfall_bps > self.thresholds.maximum_shortfall_bps,
                Alert(
                    "execution_shortfall",
                    shortfall_bps,
                    self.thresholds.maximum_shortfall_bps,
                    "DEWEIGHT",
                    "execution cost exceeded the approved model",
                ),
            ),
            (
                abs(pnl_deviation_zscore) > self.thresholds.maximum_pnl_deviation_zscore,
                Alert(
                    "pnl_deviation",
                    pnl_deviation_zscore,
                    self.thresholds.maximum_pnl_deviation_zscore,
                    "ROLLBACK",
                    "realized PnL diverged materially from shadow PnL",
                ),
            ),
        ]
        return tuple(alert for triggered, alert in checks if triggered)


class PaperTradingBook:
    def __init__(self, path: Path) -> None:
        self.path = path

    def append(
        self,
        *,
        date: str,
        research_return: float,
        paper_return: float,
        turnover: float,
        shortfall_bps: float,
    ) -> None:
        record = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "date": date,
            "research_return": research_return,
            "paper_return": paper_return,
            "turnover": turnover,
            "shortfall_bps": shortfall_bps,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def summary(self) -> dict[str, float]:
        if not self.path.exists():
            return {"days": 0.0}
        frame = pd.DataFrame(
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        difference = frame["paper_return"] - frame["research_return"]
        return {
            "days": float(len(frame)),
            "paper_total_return": float((1 + frame["paper_return"]).prod() - 1),
            "direction_agreement": float(
                np.mean(np.sign(frame["paper_return"]) == np.sign(frame["research_return"]))
            ),
            "mean_return_deviation": float(difference.mean()),
            "mean_shortfall_bps": float(frame["shortfall_bps"].mean()),
            "mean_turnover": float(frame["turnover"].mean()),
        }
