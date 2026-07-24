from __future__ import annotations

import math
import os
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from autoalpha.data.workspace import inspect_data_workspace
from autoalpha.service.batch_engine import MassiveBatchConfig, MassiveVectorBatchEngine
from autoalpha.service.batch_runner import MassiveBatchRunner
from autoalpha.service.batch_store import BatchBacktestStore
from autoalpha.service.realistic_batch_engine import (
    RealisticAshareBatchConfig,
    RealisticAshareBatchEngine,
)

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path.cwd()
BATCH_MODE = os.getenv("AUTOALPHA_BATCH_MODE", "RESEARCH_LONG_SHORT").strip().upper()
REALISTIC_ASHARE = BATCH_MODE == "ASHARE_REALISTIC_LONG_ONLY"
DEFAULT_PORT = "8790" if REALISTIC_ASHARE else "8789"
BATCH_RUNTIME = Path(
    os.getenv(
        "AUTOALPHA_BATCH_RUNTIME",
        PROJECT_ROOT
        / ("runtime-realistic-ashare-batch" if REALISTIC_ASHARE else "runtime-massive-batch"),
    )
).expanduser()
SOURCE_DATABASE = Path(
    os.getenv("AUTOALPHA_SOURCE_DB", PROJECT_ROOT / "runtime-full-llm/autoalpha.sqlite3")
).expanduser()
DEFAULT_DATA_PATH = Path(
    os.getenv("AUTOALPHA_BATCH_DATA", "/Users/jiangjingzhe/Portfolios/MultiFactorAshare/data")
).expanduser()
CONFIG_PATH = Path(os.getenv("AUTOALPHA_CONFIG", PROJECT_ROOT / "config/research.toml"))

DATABASE_NAME = "realistic_ashare_batch.sqlite3" if REALISTIC_ASHARE else "massive_batch.sqlite3"
store = BatchBacktestStore(BATCH_RUNTIME / DATABASE_NAME)
runner = MassiveBatchRunner(
    store,
    BATCH_RUNTIME / "artifacts",
    config_type=RealisticAshareBatchConfig if REALISTIC_ASHARE else MassiveBatchConfig,
    engine_type=RealisticAshareBatchEngine if REALISTIC_ASHARE else MassiveVectorBatchEngine,
    worker_prefix="ashare-realistic-factor" if REALISTIC_ASHARE else "massive-vector-factor",
)


class BatchJobRequest(BaseModel):
    name: str = Field(
        default=(
            "2020-2026 A股仅多头真实交易代理大回测"
            if REALISTIC_ASHARE
            else "2020-2026 全因子大规模稳健性回测"
        ),
        min_length=2,
        max_length=100,
    )
    data_path: str = str(DEFAULT_DATA_PATH)
    start_date: date = date(2020, 1, 1)
    end_date: date | None = None
    workers: int = Field(default=4, ge=1, le=8)
    holding_period_days: int = Field(default=5, ge=1, le=60)
    gross_exposure: float = Field(default=0.90 if REALISTIC_ASHARE else 1.0, gt=0, le=2)
    selection_fraction: float = Field(default=0.10, gt=0, le=0.50)
    maximum_positions_per_side: int = Field(default=30, ge=5, le=500)
    window_months: int = Field(default=36, ge=12, le=84)
    step_months: int = Field(default=12, ge=3, le=36)
    monte_carlo_samples: int = Field(default=10_000, ge=1_000, le=100_000)
    monte_carlo_block_days: int = Field(default=20, ge=5, le=120)
    parameter_multipliers: list[float] = Field(default_factory=lambda: [0.5, 2.0])
    holding_period_tests: list[int] = Field(default_factory=lambda: [1, 20])

    @field_validator("name", "data_path")
    @classmethod
    def clean_text(cls, value: str) -> str:
        return value.strip()


@asynccontextmanager
async def lifespan(_: FastAPI):
    store.recover_interrupted()
    yield


app = FastAPI(
    title=(
        "AutoAlpha A-share Realistic Massive Backtest"
        if REALISTIC_ASHARE
        else "AutoAlpha Massive Batch Backtest"
    ),
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=PACKAGE_ROOT / "static"), name="static")


@app.get("/")
async def batch_page() -> FileResponse:
    return FileResponse(PACKAGE_ROOT / "static/batch_backtest.html")


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "active_job_id": runner.active_job_id}


@app.get("/api/bootstrap")
async def bootstrap() -> dict[str, Any]:
    try:
        workspace = inspect_data_workspace(DEFAULT_DATA_PATH)
        coverage = {
            "start": workspace.first_trade_date,
            "end": workspace.last_trade_date,
            "fingerprint": workspace.fingerprint,
        }
    except Exception as error:  # noqa: BLE001
        coverage = {"error": f"{type(error).__name__}: {error}"}
    return {
        "jobs": [_job_view(job) for job in store.jobs()],
        "active_job_id": runner.active_job_id,
        "source_database": str(SOURCE_DATABASE),
        "batch_database": str(store.path),
        "artifact_root": str(BATCH_RUNTIME / "artifacts"),
        "data_coverage": coverage,
        "service": {
            "mode": BATCH_MODE,
            "port": int(DEFAULT_PORT),
            "page_title": (
                "A股真实交易代理批量回测" if REALISTIC_ASHARE else "全因子批量回测"
            ),
            "leaderboard_title": (
                "A股仅多头稳健性排行榜" if REALISTIC_ASHARE else "因子稳健性排行榜"
            ),
            "protocol": (
                "A_SHARE_LONG_ONLY_WEEKLY_VECTOR_PROXY_V1"
                if REALISTIC_ASHARE
                else "AUTOALPHA_MASSIVE_VECTOR_V1"
            ),
        },
        "defaults": {
            "name": (
                "2020-2026 A股仅多头真实交易代理全因子大回测"
                if REALISTIC_ASHARE
                else "2020-2026 全因子大规模稳健性回测"
            ),
            "data_path": str(DEFAULT_DATA_PATH),
            "start_date": "2020-01-01",
            "end_date": coverage.get("end"),
            "workers": 4,
            "window_months": 36,
            "step_months": 12,
            "monte_carlo_samples": 10_000,
            "monte_carlo_block_days": 20,
        },
    }


@app.post("/api/jobs")
async def create_job(payload: BatchJobRequest) -> dict[str, Any]:
    try:
        data_path = Path(payload.data_path).expanduser().resolve()
        workspace = inspect_data_workspace(data_path)
        end_date = payload.end_date or date.fromisoformat(workspace.last_trade_date)
        config_type = RealisticAshareBatchConfig if REALISTIC_ASHARE else MassiveBatchConfig
        config = config_type(
            data_path=data_path,
            config_path=CONFIG_PATH.resolve(),
            start_date=payload.start_date,
            end_date=end_date,
            workers=payload.workers,
            holding_period_days=payload.holding_period_days,
            gross_exposure=payload.gross_exposure,
            selection_fraction=payload.selection_fraction,
            maximum_positions_per_side=payload.maximum_positions_per_side,
            window_months=payload.window_months,
            step_months=payload.step_months,
            monte_carlo_samples=payload.monte_carlo_samples,
            monte_carlo_block_days=payload.monte_carlo_block_days,
            parameter_multipliers=tuple(payload.parameter_multipliers),
            holding_period_tests=tuple(payload.holding_period_tests),
            **(
                {
                    "initial_cash_cny": 1_000_000.0,
                    "minimum_commission_cny": 5.0,
                    "use_historical_fee_schedule": True,
                    "rebalance_schedule": "WEEKLY_FIRST_SESSION",
                    "execution_data_mode": "NON_PIT_PROXY",
                }
                if REALISTIC_ASHARE
                else {}
            ),
        )
        available_start = date.fromisoformat(workspace.first_trade_date)
        available_end = date.fromisoformat(workspace.last_trade_date)
        if config.start_date < available_start or config.end_date > available_end:
            raise ValueError(
                f"Backtest range must stay within {available_start} and {available_end}"
            )
        job = store.create_job(
            name=payload.name,
            config=config.to_dict(),
            source_database=SOURCE_DATABASE.resolve(),
        )
    except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return _job_view(job)


@app.get("/api/jobs/{job_id}")
async def job_detail(job_id: str) -> dict[str, Any]:
    try:
        job = store.job(job_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return {
        "job": _job_view(job),
        "results": _rank_results(store.results(job_id)),
        "events": store.events(job_id, limit=120),
    }


@app.post("/api/jobs/{job_id}/start")
async def start_job(job_id: str) -> dict[str, Any]:
    try:
        return _job_view(runner.start(job_id))
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/api/jobs/{job_id}/pause")
async def pause_job(job_id: str) -> dict[str, Any]:
    try:
        return _job_view(runner.pause(job_id))
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.get("/api/jobs/{job_id}/factors/{factor_id}")
async def factor_detail(job_id: str, factor_id: str) -> dict[str, Any]:
    try:
        detail = store.factor_detail(job_id, factor_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    detail["curve"] = _curve_payload(detail.get("curve_path"))
    detail["monte_carlo_histogram"] = _monte_carlo_histogram(detail.get("monte_carlo_path"))
    return detail


def _job_view(job: dict[str, Any]) -> dict[str, Any]:
    item = dict(job)
    total = int(item["factor_count"])
    completed = int(item["completed_count"])
    item["progress"] = completed / total if total else 0.0
    item["active"] = runner.active_job_id == item["job_id"]
    if item.get("started_at") and completed:
        elapsed = max(
            0.0,
            (
                pd.Timestamp(item.get("finished_at") or pd.Timestamp.now(tz="UTC"))
                - pd.Timestamp(item["started_at"])
            ).total_seconds(),
        )
        rate = completed / elapsed if elapsed else 0.0
        item["factors_per_hour"] = rate * 3600
        item["eta_seconds"] = (total - completed) / rate if rate else None
    else:
        item["factors_per_hour"] = 0.0
        item["eta_seconds"] = None
    return item


def _rank_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def score(item: dict[str, Any]) -> tuple[int, float, float]:
        metrics = item.get("metrics") or {}
        return (
            int(item["status"] == "SUCCESS"),
            float(metrics.get("large_window_worst_sharpe", -100.0)),
            float(metrics.get("sharpe_ratio", -100.0)),
        )

    ranked = sorted(results, key=score, reverse=True)
    for index, item in enumerate(ranked, 1):
        item["rank"] = index if item["status"] == "SUCCESS" else None
    return ranked


def _curve_payload(path_value: str | None, maximum_points: int = 700) -> list[dict[str, Any]]:
    if not path_value or not Path(path_value).exists():
        return []
    frame = pd.read_parquet(path_value, columns=["equity", "drawdown", "net"])
    step = max(1, math.ceil(len(frame) / maximum_points))
    selected = frame.iloc[::step]
    if not selected.index.equals(frame.index) and not selected.index[-1:].equals(frame.index[-1:]):
        selected = pd.concat([selected, frame.iloc[-1:]])
    return [
        {
            "date": timestamp.date().isoformat(),
            "equity": float(row.equity),
            "drawdown": float(row.drawdown),
            "net_return": float(row.net),
        }
        for timestamp, row in selected.iterrows()
    ]


def _monte_carlo_histogram(path_value: str | None) -> dict[str, Any]:
    if not path_value or not Path(path_value).exists():
        return {}
    frame = pd.read_parquet(path_value, columns=["sharpe_ratio", "simple_annual_return"])
    result: dict[str, Any] = {}
    for column in frame.columns:
        values = frame[column].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        counts, edges = np.histogram(values, bins=36)
        result[column] = {"counts": counts.tolist(), "edges": edges.tolist()}
    return result


def main() -> None:
    uvicorn.run(
        "autoalpha.service.batch_app:app",
        host="127.0.0.1",
        port=int(os.getenv("AUTOALPHA_BATCH_PORT", DEFAULT_PORT)),
        reload=False,
    )


if __name__ == "__main__":
    main()
