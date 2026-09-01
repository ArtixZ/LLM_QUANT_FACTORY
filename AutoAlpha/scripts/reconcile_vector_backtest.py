from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from autoalpha.backtest.vector import (
    VectorBacktestConfig,
    VectorBacktester,
    reconcile_vector_paths,
)
from autoalpha.config import ResearchConfig
from autoalpha.service.evaluator import PriceVolumeEvaluator
from autoalpha.service.multifactor import factor_from_pool_record
from autoalpha.service.store import ServiceStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconcile AutoAlpha's vector engine against the existing evaluator."
    )
    parser.add_argument(
        "--database", type=Path, default=PROJECT_ROOT / "runtime/autoalpha.sqlite3"
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/research.toml")
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--factor-id")
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "output/backtests/vector_engine_v1"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = ServiceStore(args.database.expanduser().resolve())
    config = ResearchConfig.from_toml(args.config.expanduser().resolve())
    data_path = (
        args.data_path.expanduser().resolve()
        if args.data_path
        else Path(store.settings()["data_path"]).expanduser().resolve()
    )
    record = (
        store.factor_pool_record(args.factor_id)
        if args.factor_id
        else _best_public_factor(store.factor_pool(limit=5000))
    )
    if record is None:
        raise KeyError(f"Unknown factor id: {args.factor_id}")
    factor = factor_from_pool_record(record)
    evaluator = PriceVolumeEvaluator(data_path, args.config.expanduser().resolve())
    signal = evaluator._factor_signal(factor)
    fields = evaluator._load_fields()

    base = dict(
        holding_period_days=config.portfolio.holding_period_days,
        gross_exposure=1.0,
        selection_fraction=config.portfolio.top_fraction,
        maximum_positions_per_side=None,
        selection_method="percentile",
        long_only=False,
        commission_bps_each_side=config.costs.commission_bps_each_side,
        stamp_duty_bps_sell=config.costs.stamp_duty_bps_sell,
        transfer_fee_bps_each_side=config.costs.transfer_fee_bps_each_side,
        cost_stress_multiplier=2.0,
        path_index="signal_session",
        initial_cash_usd=config.portfolio.initial_cash_usd,
    )
    legacy_config = VectorBacktestConfig(**base, cost_model="legacy_half_turnover")
    existing_path = evaluator._signal_path(signal).copy()
    one_way_bps = (
        config.costs.commission_bps_each_side
        + config.costs.transfer_fee_bps_each_side
        + config.costs.stamp_duty_bps_sell / 2
    )
    existing_path["gross"] = (
        existing_path["net"] + existing_path["turnover"] * one_way_bps / 10_000
    )
    legacy_result = VectorBacktester(legacy_config).run(signal, fields["open"])
    reconciliation = reconcile_vector_paths(existing_path, legacy_result.path)

    public_start = pd.Timestamp(config.walk_forward.first_validation_year, 1, 1)
    public_end = pd.Timestamp(config.splits.validation.end)
    corrected_result = VectorBacktester(
        VectorBacktestConfig(**base, cost_model="side_aware")
    ).run(signal, fields["open"], start=public_start, end=public_end)
    legacy_public = VectorBacktester(legacy_config).run(
        signal, fields["open"], start=public_start, end=public_end
    )
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    comparison = pd.DataFrame(
        {
            "legacy_net": legacy_public.path["net"],
            "corrected_net": corrected_result.path["net"],
            "legacy_cost": legacy_public.path["transaction_cost"],
            "corrected_cost": corrected_result.path["transaction_cost"],
            "turnover": corrected_result.path["turnover"],
        }
    )
    comparison.to_csv(output / "daily_path_comparison.csv", index_label="date")
    report = {
        "protocol": "AUTOALPHA_VECTOR_RECONCILIATION_V1",
        "created_at": datetime.now(UTC).isoformat(),
        "factor": {
            "factor_id": record["factor_id"],
            "name": record["name"],
            "family": record["family"],
            "source_iteration": record["source_iteration"],
        },
        "data": {
            "path": str(data_path),
            "fingerprint": evaluator.workspace.fingerprint,
            "hidden_period_accessed": False,
            "comparison_start": corrected_result.metrics["backtest_start"],
            "comparison_end": corrected_result.metrics["backtest_end"],
        },
        "existing_engine_reconciliation": reconciliation.to_dict(),
        "legacy_compatible_metrics": legacy_public.metrics,
        "side_aware_metrics": corrected_result.metrics,
        "side_aware_impact": _metric_impact(legacy_public.metrics, corrected_result.metrics),
        "stored_current_protocol_metrics": {
            key: record["metrics"].get(key)
            for key in (
                "simple_annual_return",
                "sharpe_ratio",
                "max_drawdown",
                "annual_turnover",
                "evaluation_protocol",
            )
        },
        "notes": [
            "Existing-engine reconciliation uses the same signal-indexed return labels.",
            "Legacy mode preserves the historical half-turnover cost formula exactly.",
            "Side-aware mode charges commission and transfer fee on both traded sides and stamp "
            "duty on sells; minimum per-order commission requires the capital ledger.",
            "No configured hidden-test dates were evaluated.",
        ],
    }
    (output / "reconciliation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _best_public_factor(records: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [
        record
        for record in records
        if record["metrics"].get("exploratory_gate_passed")
        and record["metrics"].get("data_basis_compatible", True)
        and not record["metrics"].get("holdout_contaminated", False)
        and not record["metrics"].get("hidden_period_accessed", False)
    ]
    if not eligible:
        raise RuntimeError("No clean public-gate factor is available for reconciliation")
    return max(eligible, key=lambda item: float(item["metrics"].get("sharpe_ratio", -100.0)))


def _metric_impact(
    legacy: dict[str, float | int | str], corrected: dict[str, float | int | str]
) -> dict[str, float]:
    keys = (
        "simple_annual_return",
        "compound_annual_return",
        "total_return",
        "sharpe_ratio",
        "max_drawdown",
        "annual_turnover",
        "total_transaction_cost_return",
    )
    return {key: float(corrected[key]) - float(legacy[key]) for key in keys}


if __name__ == "__main__":
    main()
