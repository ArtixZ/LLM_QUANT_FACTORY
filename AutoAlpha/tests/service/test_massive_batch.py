from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path

import numpy as np

from autoalpha.service.batch_engine import (
    generate_step_windows,
    moving_block_monte_carlo,
)
from autoalpha.service.batch_store import BatchBacktestStore
from autoalpha.service.realistic_batch_engine import RealisticAshareBatchConfig


def test_large_step_windows_cover_latest_partial_window() -> None:
    windows = generate_step_windows(
        date(2020, 1, 1),
        date(2026, 7, 16),
        window_months=36,
        step_months=12,
    )

    assert windows[0] == ("W01", date(2020, 1, 1), date(2022, 12, 31))
    assert windows[-1] == ("W05", date(2024, 1, 1), date(2026, 7, 16))


def test_moving_block_monte_carlo_is_reproducible_and_persists_samples() -> None:
    values = np.sin(np.arange(600) / 17) * 0.004 + 0.0003

    first, first_summary = moving_block_monte_carlo(values, samples=1_000, block_size=20, seed=42)
    second, second_summary = moving_block_monte_carlo(values, samples=1_000, block_size=20, seed=42)

    assert first.equals(second)
    assert first_summary == second_summary
    assert len(first) == 1_000
    assert set(first) == {
        "simple_annual_return",
        "sharpe_ratio",
        "total_return",
        "max_drawdown",
    }


def test_realistic_batch_config_round_trips_execution_protocol(tmp_path: Path) -> None:
    config = RealisticAshareBatchConfig(
        data_path=tmp_path,
        config_path=tmp_path / "research.toml",
        start_date=date(2020, 1, 1),
        end_date=date(2026, 7, 16),
        gross_exposure=0.90,
        commission_bps_each_side=2.5,
        slippage_bps_each_side=5.0,
    )

    restored = RealisticAshareBatchConfig.from_dict(config.to_dict())

    assert restored.protocol == "A_SHARE_LONG_ONLY_WEEKLY_VECTOR_PROXY_V1"
    assert restored.rebalance_schedule == "WEEKLY_FIRST_SESSION"
    assert restored.gross_exposure == 0.90


def test_batch_store_freezes_factor_definitions_and_resumes_pending(tmp_path: Path) -> None:
    source = tmp_path / "source.sqlite3"
    with sqlite3.connect(source) as connection:
        connection.execute(
            """CREATE TABLE factor_pool (
            factor_id TEXT, name TEXT, family TEXT, source_task_id TEXT,
            source_iteration INTEGER, status TEXT, proposal_json TEXT)"""
        )
        for index in range(2):
            proposal = {
                "name": f"Factor {index}",
                "family": "test",
                "hypothesis": "test hypothesis",
                "expression": {
                    "operator": "returns",
                    "arguments": [{"operator": "field", "parameters": {"name": "adj_close"}}],
                    "parameters": {"periods": 5 + index},
                },
                "expected_direction": 1,
            }
            connection.execute(
                "INSERT INTO factor_pool VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    f"F_{index}",
                    proposal["name"],
                    "test",
                    "task-test",
                    index,
                    "SCREENED_OUT",
                    json.dumps(proposal),
                ),
            )
    store = BatchBacktestStore(tmp_path / "batch.sqlite3")
    job = store.create_job(name="test", config={"start_date": "2020-01-01"}, source_database=source)

    assert job["factor_count"] == 2
    pending = store.pending_factors(job["job_id"])
    assert [item["factor_id"] for item in pending] == ["F_0", "F_1"]

    store.mark_factor_running(job["job_id"], "F_0")
    store.complete_factor(
        job["job_id"],
        "F_0",
        elapsed_seconds=1.2,
        metrics={"sharpe_ratio": 1.0},
        monte_carlo={"samples": 1000},
        curve_path="curve.parquet",
        monte_carlo_path="mc.parquet",
        windows=[],
        robustness=[],
    )

    assert [item["factor_id"] for item in store.pending_factors(job["job_id"])] == ["F_1"]
    assert store.job(job["job_id"])["completed_count"] == 1
