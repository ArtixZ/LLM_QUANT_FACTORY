from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator, model_validator

from autoalpha.config import ResearchConfig
from autoalpha.data.workspace import inspect_data_workspace
from autoalpha.service.autocombine import DEFAULT_CONSTRUCTION, OBJECTIVE_PRESETS
from autoalpha.service.autocombine_intelligence import enrich_factor_record
from autoalpha.service.quantcombine import (
    DEFAULT_BUDGET,
    DEFAULT_ENGINE,
    QuantCombineManager,
    create_quant_task_record,
    pareto_ranks,
)
from autoalpha.service.quantcombine_store import QuantCombineStore
from autoalpha.service.research_protocol import (
    default_task_protocol,
    panel_validation_fold_capacity,
    protocol_blockers,
    protocol_data_blockers,
)
from autoalpha.service.store import ServiceStore

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_ROOT = Path(os.getenv("AUTOALPHA_RUNTIME", PROJECT_ROOT / "runtime-full-llm"))
CONFIG_PATH = Path(os.getenv("AUTOALPHA_CONFIG", PROJECT_ROOT / "config/research.toml"))


class QuantProtocol(BaseModel):
    exploration_start: str
    exploration_end: str
    validation_start: str
    validation_end: str
    holdout_start: str
    holdout_end: str
    minimum_folds: int = Field(default=3, ge=1, le=20)


class QuantScope(BaseModel):
    mode: Literal["SMART", "MANUAL", "HYBRID"] = "SMART"
    factor_ids: list[str] = Field(default_factory=list)
    required_factor_ids: list[str] = Field(default_factory=list)
    excluded_factor_ids: list[str] = Field(default_factory=list)
    source_task_ids: list[str] = Field(default_factory=list)
    statuses: list[str] = Field(default_factory=lambda: ["ELIGIBLE", "SCREENED_OUT"])
    families: list[str] = Field(default_factory=list)


class QuantConstruction(BaseModel):
    min_factors: int = Field(default=2, ge=1, le=12)
    max_factors: int = Field(default=5, ge=1, le=12)
    minimum_weight: float = Field(default=0.05, ge=0.01, le=0.50)
    maximum_weight: float = Field(default=0.50, ge=0.10, le=1.0)
    weight_step: float = Field(default=0.05, ge=0.01, le=0.25)
    candidate_pool_limit: int = Field(default=30, ge=5, le=100)
    allow_negative_weights: bool = False
    maximum_same_family: int = Field(default=2, ge=1, le=5)
    maximum_same_semantic_cluster: int = Field(default=1, ge=1, le=5)

    @model_validator(mode="after")
    def valid_constraints(self) -> QuantConstruction:
        if self.min_factors > self.max_factors:
            raise ValueError("min_factors must not exceed max_factors")
        if self.minimum_weight > self.maximum_weight:
            raise ValueError("minimum_weight must not exceed maximum_weight")
        if self.minimum_weight * self.max_factors > 1.0 + 1e-9:
            raise ValueError("minimum weights make the selected factor count infeasible")
        if self.maximum_weight * self.min_factors < 1.0 - 1e-9:
            raise ValueError("maximum weights make the selected factor count infeasible")
        if self.allow_negative_weights:
            raise ValueError("QuantCombine V1 supports long-only factor weights")
        return self


class QuantObjective(BaseModel):
    profile: Literal[
        "ROBUST_ACTIVE_LONG_ONLY",
        "DRAWDOWN_FIRST",
        "PORTFOLIO_SHARPE_FIRST",
        "ABSOLUTE_LONG_ONLY",
        "LOW_TURNOVER",
        "DIVERSIFICATION_FIRST",
    ] = "DRAWDOWN_FIRST"
    preset_version: int = 1
    minimum_coverage: float = Field(default=0.80, ge=0, le=1)
    minimum_positive_fold_fraction: float = Field(default=0.50, ge=0, le=1)
    minimum_worst_fold_sharpe: float = Field(default=-0.50, ge=-10, le=10)
    maximum_drawdown: float = Field(default=0.30, ge=0.01, le=1)
    maximum_annual_turnover: float = Field(default=40.0, ge=0.1, le=1000)
    maximum_factor_correlation: float = Field(default=0.75, ge=0, le=1)
    minimum_effective_factor_bets: float = Field(default=1.35, ge=1, le=12)
    minimum_effective_mechanisms: float = Field(default=1.40, ge=1, le=12)
    maximum_mechanism_weight: float = Field(default=0.75, ge=0.1, le=1)
    maximum_strategy_active_correlation: float = Field(default=0.75, ge=0, le=1)
    minimum_marginal_positive_fraction: float = Field(default=0.60, ge=0, le=1)
    minimum_deflated_sharpe_probability: float = Field(default=0.50, ge=0, le=1)
    maximum_duplicate_semantic_factors: int = Field(default=0, ge=0, le=12)
    minimum_cost_stress_ir: float = Field(default=0.0, ge=-10, le=10)
    minimum_simple_annual_return: float = Field(default=0.0, ge=-1, le=10)


class QuantEngine(BaseModel):
    mode: Literal["ENSEMBLE", "DETERMINISTIC", "EVOLUTIONARY", "BAYESIAN"] = "ENSEMBLE"
    cluster_correlation_threshold: float = Field(default=0.78, ge=0.3, le=0.99)
    minimum_stability_score: float = Field(default=-2.0, ge=-10, le=10)
    sffs_beam_width: int = Field(default=3, ge=1, le=10)
    evolution_population: int = Field(default=12, ge=4, le=100)
    evolution_generations: int = Field(default=4, ge=1, le=100)
    adaptive_trials: int = Field(default=16, ge=0, le=1000)
    covariance_shrinkage: float = Field(default=0.35, ge=0, le=1)
    weight_regularization: float = Field(default=0.08, ge=0, le=10)
    random_seed: int = Field(default=20260718, ge=0, le=2**31 - 1)


class QuantBudget(BaseModel):
    maximum_evaluations: int = Field(default=180, ge=10, le=100000)
    maximum_runtime_minutes: int = Field(default=240, ge=1, le=10080)
    weight_candidates_per_subset: int = Field(default=8, ge=1, le=50)
    iteration_interval_seconds: float = Field(default=0.0, ge=0, le=3600)


class QuantTaskRequest(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    market: Literal["CN_A"] = "CN_A"
    data_path: str
    protocol: QuantProtocol
    scope: QuantScope = Field(default_factory=QuantScope)
    construction: QuantConstruction = Field(default_factory=QuantConstruction)
    objective: QuantObjective = Field(default_factory=QuantObjective)
    engine: QuantEngine = Field(default_factory=QuantEngine)
    budget: QuantBudget = Field(default_factory=QuantBudget)
    notes: str = Field(default="", max_length=2000)

    @field_validator("name", "data_path", "notes")
    @classmethod
    def clean_text(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def valid_scope(self) -> QuantTaskRequest:
        if self.scope.mode == "MANUAL" and not self.scope.factor_ids:
            raise ValueError("manual scope requires selected factors")
        if self.scope.mode == "HYBRID" and not self.scope.required_factor_ids:
            raise ValueError("hybrid scope requires mandatory factors")
        if len(self.scope.required_factor_ids) > self.construction.max_factors:
            raise ValueError("required factors exceed max_factors")
        return self


class PromoteRequest(BaseModel):
    candidate_id: int
    name: str = Field(min_length=2, max_length=100)


base_store = ServiceStore(RUNTIME_ROOT / "autoalpha.sqlite3")
quant_store = QuantCombineStore(base_store)
manager = QuantCombineManager(
    base_store,
    quant_store,
    config_path=CONFIG_PATH,
    maximum_concurrent_tasks=int(os.getenv("QUANTCOMBINE_MAX_CONCURRENT", "1")),
)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    try:
        yield
    finally:
        await manager.shutdown()


app = FastAPI(
    title="QuantCombine Statistical Portfolio Research", version="1.0.0", lifespan=lifespan
)
app.mount("/static", StaticFiles(directory=PACKAGE_ROOT / "static"), name="static")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(PACKAGE_ROOT / "static/quantcombine.html")


@app.get("/tasks/{task_id}", include_in_schema=False)
async def task_page(task_id: str) -> FileResponse:
    del task_id
    return FileResponse(PACKAGE_ROOT / "static/quantcombine.html")


@app.get("/strategies", include_in_schema=False)
async def strategy_page() -> FileResponse:
    return FileResponse(PACKAGE_ROOT / "static/quantcombine.html")


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "QuantCombine",
        "version": "1.0.0",
        "port": int(os.getenv("QUANTCOMBINE_PORT", "8889")),
        "llm_required": False,
        "factor_count": len(base_store.factor_pool(limit=5000)),
        "task_count": len(quant_store.tasks()),
        "active_tasks": manager.active_count,
        "maximum_concurrent_tasks": manager.maximum_concurrent_tasks,
        "runtime_root": str(RUNTIME_ROOT.resolve()),
    }


@app.get("/api/bootstrap")
async def bootstrap() -> dict[str, Any]:
    settings = base_store.settings()
    data_path = str(Path(settings.get("data_path", PROJECT_ROOT.parent / "data")).expanduser())
    data_range: dict[str, Any] = {"start": None, "end": None, "fingerprint": None}
    protocol: dict[str, Any] | None = None
    try:
        workspace = inspect_data_workspace(Path(data_path))
        data_range = {
            "start": workspace.first_trade_date,
            "end": workspace.last_trade_date,
            "fingerprint": workspace.fingerprint,
        }
        protocol = default_task_protocol(
            workspace.first_trade_date,
            workspace.last_trade_date,
            ResearchConfig.from_toml(CONFIG_PATH),
        )
        capacity = panel_validation_fold_capacity(protocol, Path(workspace.panel_path))
        if int(capacity["maximum_folds"]) > 0:
            protocol["minimum_folds"] = min(
                int(protocol["minimum_folds"]), int(capacity["maximum_folds"])
            )
    except (FileNotFoundError, RuntimeError, TypeError, ValueError, OSError):
        pass
    records = base_store.factor_pool(limit=5000)
    contaminated = base_store.contaminated_factor_ids()
    factors = []
    for record in records:
        enriched = enrich_factor_record(record)
        metrics = record.get("metrics") or {}
        factors.append(
            {
                "factor_id": record["factor_id"],
                "name": record["name"],
                "family": record["family"],
                "mechanism": enriched["mechanism"],
                "semantic_cluster_id": enriched.get("semantic_cluster_id"),
                "status": record["status"],
                "source_task_id": record.get("source_task_id"),
                "score": metrics.get("long_only_overall"),
                "sharpe": metrics.get("long_only_sharpe_ratio"),
                "annual_return": metrics.get("long_only_simple_annual_return"),
                "max_drawdown": metrics.get("long_only_max_drawdown"),
                "holdout_contaminated": record["factor_id"] in contaminated,
            }
        )
    return {
        "tasks": [_task_view(task) for task in quant_store.tasks()],
        "strategies": quant_store.strategies(),
        "factors": factors,
        "research_tasks": [
            {
                "task_id": task["task_id"],
                "name": task["name"],
                "market": task["market"],
                "status": task["status"],
            }
            for task in base_store.research_tasks()
        ],
        "objective_presets": list(OBJECTIVE_PRESETS.values()),
        "defaults": {
            "data_path": data_path,
            "data_range": data_range,
            "protocol": protocol,
            "construction": DEFAULT_CONSTRUCTION,
            "objective": {
                key: value
                for key, value in OBJECTIVE_PRESETS["DRAWDOWN_FIRST"].items()
                if key not in {"label", "description"}
            },
            "engine": DEFAULT_ENGINE,
            "budget": DEFAULT_BUDGET,
        },
        "llm_required": False,
        "maximum_concurrent_tasks": manager.maximum_concurrent_tasks,
    }


@app.post("/api/tasks")
async def create_task(payload: QuantTaskRequest) -> dict[str, Any]:
    data_path = Path(payload.data_path).expanduser().resolve()
    try:
        workspace = inspect_data_workspace(data_path)
        protocol = payload.protocol.model_dump()
        blockers = protocol_blockers(
            protocol,
            data_start=workspace.first_trade_date,
            data_end=workspace.last_trade_date,
        )
        blockers.extend(protocol_data_blockers(protocol, Path(workspace.panel_path)))
        if blockers:
            raise ValueError("；".join(blockers))
        objective = payload.objective.model_dump()
        preset = OBJECTIVE_PRESETS[payload.objective.profile]
        for key, value in preset.items():
            if key in QuantObjective.model_fields and key not in payload.objective.model_fields_set:
                objective[key] = value
        record = create_quant_task_record(
            base_store,
            name=payload.name,
            market=payload.market,
            data_path=str(data_path),
            protocol=protocol,
            scope=payload.scope.model_dump(),
            construction=payload.construction.model_dump(),
            objective=objective,
            engine=payload.engine.model_dump(),
            budget=payload.budget.model_dump(),
            notes=payload.notes,
        )
        if int(record["budget"]["maximum_evaluations"]) <= len(record["factor_snapshot"]):
            raise ValueError("评价预算必须大于冻结因子数量，才能完成组合搜索")
        task = quant_store.create_task(record)
    except (FileNotFoundError, RuntimeError, TypeError, ValueError, OSError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    quant_store.event(
        task["task_id"],
        "action",
        "QUANT_TASK_CREATED",
        "QuantCombine 任务已创建",
        f"冻结 {task['factor_count']} 个因子 · 快照 {task['snapshot_hash'][:12]} · 不调用 LLM。",
        payload={
            "factor_count": task["factor_count"],
            "engine_mode": task["engine"]["mode"],
            "snapshot_hash": task["snapshot_hash"],
        },
    )
    return _task_view(task)


@app.get("/api/tasks/{task_id}")
async def task_detail(task_id: str) -> dict[str, Any]:
    task = quant_store.task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="QuantCombine task not found")
    candidates = quant_store.candidates(task_id)
    ranks = pareto_ranks(candidates)
    return {
        "task": _task_view(task),
        "factor_snapshot": task["factor_snapshot"],
        "factor_screen": quant_store.factor_screen(task_id),
        "candidates": candidates,
        "pareto_frontier": [candidate_id for candidate_id, value in ranks.items() if value[0] == 0],
        "events": quant_store.events(task_id),
        "best": (
            quant_store.candidate(int(task["best_candidate_id"]))
            if task.get("best_candidate_id")
            else None
        ),
        "qualified": (
            quant_store.candidate(int(task["qualified_candidate_id"]))
            if task.get("qualified_candidate_id")
            else None
        ),
        "production": (
            quant_store.candidate(int(task["production_candidate_id"]))
            if task.get("production_candidate_id")
            else None
        ),
        "worker_alive": manager.alive(task_id),
    }


@app.post("/api/tasks/{task_id}/start")
async def start_task(task_id: str) -> dict[str, Any]:
    if quant_store.task(task_id) is None:
        raise HTTPException(status_code=404, detail="QuantCombine task not found")
    try:
        return _task_view(await manager.start(task_id))
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/api/tasks/{task_id}/stop")
async def stop_task(task_id: str) -> dict[str, Any]:
    if quant_store.task(task_id) is None:
        raise HTTPException(status_code=404, detail="QuantCombine task not found")
    return _task_view(await manager.stop(task_id))


@app.post("/api/tasks/{task_id}/promote")
async def promote(task_id: str, payload: PromoteRequest) -> dict[str, Any]:
    try:
        strategy = quant_store.promote_strategy(task_id, payload.candidate_id, payload.name)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    quant_store.event(
        task_id,
        "delivery",
        "QUANT_STRATEGY_PROMOTED",
        "统计候选已进入策略库",
        f"{strategy['strategy_id']} · VERSION {strategy['version']}",
        payload={"strategy_id": strategy["strategy_id"], "version": strategy["version"]},
    )
    return strategy


@app.get("/api/strategies")
async def strategies() -> dict[str, Any]:
    return {"strategies": quant_store.strategies()}


def _task_view(task: dict[str, Any]) -> dict[str, Any]:
    item = dict(task)
    item.pop("factor_snapshot", None)
    item["worker_alive"] = manager.alive(task["task_id"])
    maximum = max(1, int(task["budget"]["maximum_evaluations"]))
    item["progress"] = min(1.0, int(task["evaluation_count"]) / maximum)
    return item


def main() -> None:
    uvicorn.run(
        "autoalpha.service.quantcombine_app:app",
        host="127.0.0.1",
        port=int(os.getenv("QUANTCOMBINE_PORT", "8889")),
        reload=False,
    )


if __name__ == "__main__":
    main()
