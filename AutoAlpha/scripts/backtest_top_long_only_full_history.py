from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from autoalpha.config import DateRange, ResearchConfig, SplitConfig, WalkForwardConfig
from autoalpha.data.workspace import inspect_data_workspace
from autoalpha.service.evaluator import PriceVolumeEvaluator, _standalone_long_only_metrics
from autoalpha.service.multifactor import factor_from_pool_record
from autoalpha.service.store import ServiceStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PROCESS_EVALUATOR: PriceVolumeEvaluator | None = None
_MULTIPLE_TESTING_TRIALS = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recheck the current long-only Sharpe leaders over all available history."
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=PROJECT_ROOT / "runtime-full-llm/autoalpha.sqlite3",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config/research.toml",
    )
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top <= 0 or args.workers <= 0:
        raise ValueError("--top and --workers must be positive")
    database = args.database.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    store = ServiceStore(database)
    data_path = (
        args.data_path.expanduser().resolve()
        if args.data_path
        else Path(store.settings()["data_path"]).expanduser().resolve()
    )
    workspace = inspect_data_workspace(data_path)
    first = date.fromisoformat(workspace.first_trade_date)
    last = date.fromisoformat(workspace.last_trade_date)
    config = _full_history_config(ResearchConfig.from_toml(config_path), first, last)

    records = store.factor_pool(limit=5000)
    ranked = sorted(
        (
            record
            for record in records
            if _finite(record.get("metrics", {}).get("long_only_sharpe_ratio")) is not None
        ),
        key=lambda record: float(record["metrics"]["long_only_sharpe_ratio"]),
        reverse=True,
    )[: args.top]
    if len(ranked) < args.top:
        raise RuntimeError(f"Only {len(ranked)} factors have a long-only Sharpe score")
    for current_rank, record in enumerate(ranked, start=1):
        record["_current_rank"] = current_rank

    evaluator = PriceVolumeEvaluator(data_path, config=config)
    evaluator.set_trial_count(len(records))
    factors = [factor_from_pool_record(record) for record in ranked]
    print(
        f"Loading {workspace.first_trade_date}..{workspace.last_trade_date} panel and "
        f"priming {len(factors)} factor signals...",
        flush=True,
    )
    evaluator._load_fields()
    evaluator.prime_factor_signals(factors)

    global _PROCESS_EVALUATOR, _MULTIPLE_TESTING_TRIALS
    _PROCESS_EVALUATOR = evaluator
    _MULTIPLE_TESTING_TRIALS = len(records)
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=min(args.workers, len(ranked)),
        mp_context=get_context("fork"),
    ) as pool:
        futures = {pool.submit(_evaluate_record, record): record for record in ranked}
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(
                f"[{completed:02d}/{len(ranked):02d}] {result['factor_id']} "
                f"Sharpe={result['full_history']['long_only_sharpe_ratio']:.4f}",
                flush=True,
            )

    by_full_sharpe = sorted(
        results,
        key=lambda item: item["full_history"]["long_only_sharpe_ratio"],
        reverse=True,
    )
    for full_rank, item in enumerate(by_full_sharpe, start=1):
        item["full_history_rank_among_selected"] = full_rank
        item["rank_change"] = item["current_rank"] - full_rank

    report = {
        "status": "FULL_HISTORY_READ_ONLY_AUDIT",
        "created_at": datetime.now(UTC).isoformat(),
        "database": str(database),
        "data_path": str(data_path),
        "data_fingerprint": workspace.fingerprint,
        "available_period": {
            "start": workspace.first_trade_date,
            "end": workspace.last_trade_date,
            "symbols": workspace.symbols,
            "rows": workspace.rows,
        },
        "selection": {
            "metric": "long_only_sharpe_ratio",
            "top": args.top,
            "library_factor_count": len(records),
            "selected_before_full_history_recheck": True,
        },
        "execution_assumptions": {
            "protocol": config.strategy_evaluation.engine_protocol,
            "portfolio_mode": "long_only",
            "initial_cash_usd": config.strategy_evaluation.initial_cash_usd,
            "gross_exposure": config.strategy_evaluation.gross_exposure,
            "selection_fraction": config.strategy_evaluation.selection_fraction,
            "maximum_positions": config.strategy_evaluation.maximum_positions,
            "rebalance_schedule": config.strategy_evaluation.rebalance_schedule,
            "signal_availability": "END_OF_DAY_AFTER_CLOSE",
            "execution_lag_sessions": 1,
            "execution_data_mode": config.strategy_evaluation.execution_data_mode,
            "production_eligible": False,
        },
        "database_updated": False,
        "ranking_changed_among_selected": any(
            item["current_rank"] != item["full_history_rank_among_selected"]
            for item in by_full_sharpe
        ),
        "factors": by_full_sharpe,
    }
    output = args.output or (
        PROJECT_ROOT
        / "output/full-history-audits"
        / datetime.now(UTC).strftime("top-long-only-%Y%m%dT%H%M%SZ.json")
    )
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = output.with_suffix(".md")
    markdown.write_text(_markdown_report(report), encoding="utf-8")
    print(json.dumps({"report": str(output), "markdown": str(markdown)}, indent=2))


def _full_history_config(base: ResearchConfig, first: date, last: date) -> ResearchConfig:
    validation_start = first + timedelta(days=1)
    future = last + timedelta(days=1)
    return replace(
        base,
        splits=SplitConfig(
            train=DateRange(first, first),
            validation=DateRange(validation_start, last),
            test=DateRange(future, future),
            embargo_days=0,
        ),
        walk_forward=WalkForwardConfig(
            train_years=base.walk_forward.train_years,
            validation_years=1,
            first_validation_year=first.year,
            last_validation_year=last.year,
            minimum_folds=1,
        ),
        governance=replace(
            base.governance,
            protocol_version="full_history_read_only_long_only_audit_v1",
        ),
        generation="full_history_read_only_audit",
    )


def _evaluate_record(record: dict[str, Any]) -> dict[str, Any]:
    if _PROCESS_EVALUATOR is None:
        raise RuntimeError("Worker evaluator was not initialized")
    factor = factor_from_pool_record(record)
    path = _PROCESS_EVALUATOR._strategy_portfolio_path([factor], (1.0,))
    metrics = _standalone_long_only_metrics(
        path,
        _PROCESS_EVALUATOR.config,
        trials=_MULTIPLE_TESTING_TRIALS,
    )
    current = record["metrics"]
    current_sharpe = float(current["long_only_sharpe_ratio"])
    full_sharpe = float(metrics["long_only_sharpe_ratio"])
    return {
        "factor_id": str(record["factor_id"]),
        "name": str(record["name"]),
        "family": str(record["family"]),
        "source_task_id": str(record.get("source_task_id", "legacy-ashare")),
        "source_iteration": int(record["source_iteration"]),
        "current_rank": int(record["_current_rank"]),
        "expression": record["proposal"]["expression"],
        "expected_direction": int(record["proposal"].get("expected_direction", 1)),
        "current_leaderboard": {
            "long_only_sharpe_ratio": current_sharpe,
            "long_only_simple_annual_return": current.get("long_only_simple_annual_return"),
            "long_only_max_drawdown": current.get("long_only_max_drawdown"),
            "long_only_backtest_start": current.get("long_only_backtest_start"),
            "long_only_backtest_end": current.get("long_only_backtest_end"),
            "evaluation_protocol": current.get("evaluation_protocol"),
        },
        "full_history": metrics,
        "sharpe_change": full_sharpe - current_sharpe,
    }


def _markdown_report(report: dict[str, Any]) -> str:
    period = report["available_period"]
    lines = [
        "# AutoAlpha 纯多夏普前十全历史复核",
        "",
        f"- 数据：{period['start']} 至 {period['end']}，{period['symbols']} 只股票",
        "- 口径：A股纯多、周初调仓、收盘后信号、下一交易日开盘成交",
        "- 本报告只读，不覆盖因子库指标",
        "",
        "| 全历史名次 | 原名次 | 变化 | 因子 | 原夏普 | 全历史夏普 | 年化 | 回撤 | 换手 |",
        "|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for item in report["factors"]:
        full = item["full_history"]
        change = int(item["rank_change"])
        lines.append(
            f"| {item['full_history_rank_among_selected']} | {item['current_rank']} | "
            f"{change:+d} | {item['name']} | "
            f"{item['current_leaderboard']['long_only_sharpe_ratio']:.4f} | "
            f"{full['long_only_sharpe_ratio']:.4f} | "
            f"{full['long_only_simple_annual_return']:.2%} | "
            f"{full['long_only_max_drawdown']:.2%} | "
            f"{full['long_only_annual_turnover']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def _finite(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed and abs(parsed) != float("inf") else None


if __name__ == "__main__":
    main()
