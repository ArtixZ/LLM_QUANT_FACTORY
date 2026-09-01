from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from autoalpha.backtest.capital import (
    CapitalBacktestSpec,
    factor_from_iteration,
    run_capital_backtest,
    write_capital_backtest_artifacts,
)
from autoalpha.backtest.costs import USEquityExecutionCosts
from autoalpha.service.store import ServiceStore

PRIMARY_SELECTION_METRICS = (
    "recent_long_only_sharpe_ratio",
    "long_only_sharpe_ratio",
    "sharpe_ratio",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay the best completed factor in a cash ledger"
    )
    parser.add_argument("--database", type=Path, default=Path("runtime/autoalpha.sqlite3"))
    parser.add_argument(
        "--iteration",
        type=int,
        help="Freeze a specific completed iteration instead of selecting the latest best",
    )
    parser.add_argument("--panel", type=Path, default=Path("../data/processed/daily_panel"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runtime/backtests/best_factor_2020_2026_1m_50pct"),
    )
    args = parser.parse_args()

    store = ServiceStore(args.database)
    completed = [metric for metric in store.metric_history() if _selection_metric(metric)[0]]
    if not completed:
        raise ValueError("No completed iterations have usable selection metrics")
    if args.iteration is None:
        best = max(completed, key=lambda item: _selection_metric(item)[1])
    else:
        best = next(
            (item for item in completed if int(item["iteration"]) == args.iteration),
            None,
        )
        if best is None:
            raise ValueError(f"Completed iteration not found: {args.iteration}")
    record = store.iteration_record(store.state()["run_id"], int(best["iteration"]))
    if record is None:
        raise RuntimeError("Selected iteration record is missing")
    factor = factor_from_iteration(record)
    report = run_capital_backtest(
        factor,
        args.panel,
        CapitalBacktestSpec(
            start=date(2020, 1, 1),
            end=date(2026, 7, 14),
            initial_cash=1_000_000.0,
            target_gross_exposure=0.50,
            top_fraction=0.10,
            max_positions=30,
            max_volume_participation=0.05,
        ),
        costs=USEquityExecutionCosts(
            commission_bps_each_side=1.5,
            stamp_duty_bps_sell=5.0,
            transfer_fee_bps_each_side=0.1,
            minimum_commission_usd=5.0,
        ),
    )
    paths = write_capital_backtest_artifacts(report, args.output)
    selection_metric, selection_value = _selection_metric(best)
    summary = {
        "selection_iteration": best["iteration"],
        "selection_metric": selection_metric,
        "selection_metric_value": selection_value,
        "factor_id": factor.factor_id,
        "factor_name": factor.name,
        "metrics": report.metrics,
        "annual_returns": report.annual_returns,
        "paths": paths,
    }
    (args.output / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _selection_metric(metrics: dict[str, object]) -> tuple[str | None, float]:
    for key in PRIMARY_SELECTION_METRICS:
        value = metrics.get(key)
        if value is None:
            continue
        return key, float(value)
    return None, float("-inf")


if __name__ == "__main__":
    main()
