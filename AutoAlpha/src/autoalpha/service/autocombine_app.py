from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator, model_validator

from autoalpha.config import ResearchConfig
from autoalpha.data.workspace import inspect_data_workspace
from autoalpha.service.autocombine import (
    DEFAULT_BUDGET,
    DEFAULT_CONSTRUCTION,
    OBJECTIVE_PRESETS,
    AutoCombineManager,
    canonical_hash,
    create_task_record,
    refresh_task_strategy_clusters,
)
from autoalpha.service.autocombine_intelligence import enrich_factor_record
from autoalpha.service.autocombine_store import AutoCombineStore
from autoalpha.service.credentials import SystemCredentialStore
from autoalpha.service.research_protocol import (
    default_task_protocol,
    panel_validation_fold_capacity,
    protocol_blockers,
    protocol_data_blockers,
)
from autoalpha.service.store import ServiceStore
from autoalpha.service.worker import SecretVault

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_ROOT = Path(os.getenv("AUTOALPHA_RUNTIME", PROJECT_ROOT / "runtime-full-llm"))
CONFIG_PATH = Path(os.getenv("AUTOALPHA_CONFIG", PROJECT_ROOT / "config/research.toml"))


class CombineProtocol(BaseModel):
    exploration_start: str
    exploration_end: str
    validation_start: str
    validation_end: str
    holdout_start: str
    holdout_end: str
    minimum_folds: int = Field(default=1, ge=1, le=20)


class FactorScope(BaseModel):
    mode: Literal["SMART", "MANUAL", "HYBRID"] = "SMART"
    factor_ids: list[str] = Field(default_factory=list)
    required_factor_ids: list[str] = Field(default_factory=list)
    excluded_factor_ids: list[str] = Field(default_factory=list)
    source_task_ids: list[str] = Field(default_factory=list)
    statuses: list[str] = Field(default_factory=lambda: ["ELIGIBLE", "SCREENED_OUT"])
    families: list[str] = Field(default_factory=list)


class ConstructionSpec(BaseModel):
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
    def valid_constraints(self) -> ConstructionSpec:
        if self.min_factors > self.max_factors:
            raise ValueError("min_factors must not exceed max_factors")
        if self.minimum_weight > self.maximum_weight:
            raise ValueError("minimum_weight must not exceed maximum_weight")
        if self.minimum_weight * self.max_factors > 1.0 + 1e-9:
            raise ValueError("minimum weights make the selected factor count infeasible")
        if self.maximum_weight * self.min_factors < 1.0 - 1e-9:
            raise ValueError("maximum weights make the selected factor count infeasible")
        if self.allow_negative_weights:
            raise ValueError("V1 freezes factor direction and does not permit negative weights")
        return self


class ObjectiveSpec(BaseModel):
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


class BudgetSpec(BaseModel):
    maximum_experiments: int = Field(default=60, ge=1, le=5000)
    maximum_llm_proposals: int = Field(default=20, ge=0, le=1000)
    maximum_runtime_minutes: int = Field(default=180, ge=1, le=10080)
    maximum_holdout_submissions: int = Field(default=1, ge=1, le=5)
    weight_evaluations_per_subset: int = Field(default=12, ge=1, le=100)
    maximum_subset_revisits: int = Field(default=2, ge=1, le=10)
    maximum_same_direction_attempts: int = Field(default=3, ge=1, le=10)
    iteration_interval_seconds: float = Field(default=0.5, ge=0, le=3600)


class CombineTaskRequest(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    market: Literal["CN_A"] = "CN_A"
    data_path: str
    protocol: CombineProtocol
    scope: FactorScope = Field(default_factory=FactorScope)
    construction: ConstructionSpec = Field(default_factory=ConstructionSpec)
    objective: ObjectiveSpec = Field(default_factory=ObjectiveSpec)
    budget: BudgetSpec = Field(default_factory=BudgetSpec)
    notes: str = Field(default="", max_length=2000)

    @field_validator("name", "data_path", "notes")
    @classmethod
    def clean_text(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def valid_scope(self) -> CombineTaskRequest:
        selected = set(self.scope.factor_ids)
        required = set(self.scope.required_factor_ids)
        if self.scope.mode == "MANUAL" and not selected:
            raise ValueError("manual scope requires selected factors")
        if self.scope.mode == "HYBRID" and not required:
            raise ValueError("hybrid scope requires mandatory factors")
        if len(required) > self.construction.max_factors:
            raise ValueError("required factors exceed max_factors")
        return self


class PromoteRequest(BaseModel):
    experiment_id: int
    name: str = Field(min_length=2, max_length=100)


class FavoriteRequest(BaseModel):
    favorite: bool = True
    label: str = Field(default="", max_length=160)
    context: dict[str, Any] = Field(default_factory=dict)


base_store = ServiceStore(RUNTIME_ROOT / "autoalpha.sqlite3")
combine_store = AutoCombineStore(base_store)
vault = SecretVault(credential_store=SystemCredentialStore())
manager = AutoCombineManager(
    base_store,
    combine_store,
    vault,
    config_path=CONFIG_PATH,
    maximum_concurrent_tasks=max(
        1,
        int(
            os.getenv(
                "AUTOCOMBINE_CONCURRENCY",
                base_store.settings().get("autocombine_concurrency", "2"),
            )
        ),
    ),
)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    for task in combine_store.tasks():
        if task["status"] in {"RUNNING", "STOPPING"}:
            combine_store.update_task(
                task["task_id"],
                status="PAUSED",
                phase="RECOVERED",
                stop_requested=0,
                last_error="服务重启后已从持久化检查点恢复，可继续启动。",
            )
    await asyncio.to_thread(refresh_task_strategy_clusters, combine_store)
    yield
    await manager.shutdown()


app = FastAPI(title="AutoCombine Portfolio Research", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=PACKAGE_ROOT / "static"), name="static")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(PACKAGE_ROOT / "static/autocombine.html")


@app.get("/tasks/{task_id}", include_in_schema=False)
async def task_page(task_id: str) -> FileResponse:
    if combine_store.task(task_id) is None:
        raise HTTPException(status_code=404, detail="AutoCombine task not found")
    return FileResponse(PACKAGE_ROOT / "static/autocombine.html")


@app.get("/strategies", include_in_schema=False)
async def strategies_page() -> FileResponse:
    return FileResponse(PACKAGE_ROOT / "static/autocombine.html")


@app.get("/settings", include_in_schema=False)
async def settings_page() -> RedirectResponse:
    return RedirectResponse("http://127.0.0.1:8788/settings")


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "AutoCombine",
        "version": "1.0.0",
        "port": 8888,
        "provider_configured": vault.configured(),
        "runtime_root": str(RUNTIME_ROOT.resolve()),
        "research_task_count": len(base_store.research_tasks()),
        "factor_count": len(base_store.factor_pool(limit=5000)),
        "maximum_concurrent_tasks": manager.maximum_concurrent_tasks,
        "active_tasks": manager.active_count,
    }


@app.get("/api/bootstrap")
async def bootstrap() -> dict[str, Any]:
    settings = base_store.settings()
    data_path = str(Path(settings.get("data_path", PROJECT_ROOT.parent / "data")).expanduser())
    data_range: dict[str, Any] = {"start": None, "end": None, "fingerprint": None}
    default_protocol: dict[str, Any] | None = None
    try:
        workspace = inspect_data_workspace(Path(data_path))
        data_range = {
            "start": workspace.first_trade_date,
            "end": workspace.last_trade_date,
            "fingerprint": workspace.fingerprint,
        }
        default_protocol = default_task_protocol(
            workspace.first_trade_date,
            workspace.last_trade_date,
            ResearchConfig.from_toml(CONFIG_PATH),
        )
        capacity = panel_validation_fold_capacity(default_protocol, Path(workspace.panel_path))
        if int(capacity["maximum_folds"]) > 0:
            default_protocol["minimum_folds"] = min(
                int(default_protocol["minimum_folds"]), int(capacity["maximum_folds"])
            )
    except (FileNotFoundError, RuntimeError, TypeError, ValueError, OSError):
        pass
    records = base_store.factor_pool(limit=5000)
    favorite_factors = base_store.favorite_ids("factor")
    favorite_tasks = base_store.favorite_ids("combine_task")
    favorite_strategies = base_store.favorite_ids("strategy")
    contaminated_ids = base_store.contaminated_factor_ids()
    factors = [
        {
            "factor_id": record["factor_id"],
            "name": record["name"],
            "family": record["family"],
            "mechanism": enrich_factor_record(record)["mechanism"],
            "status": record["status"],
            "source_task_id": record.get("source_task_id"),
            "source_iteration": record.get("source_iteration"),
            "score": (record.get("metrics") or {}).get("long_only_overall"),
            "sharpe": (record.get("metrics") or {}).get("long_only_sharpe_ratio"),
            "annual_return": (record.get("metrics") or {}).get("long_only_simple_annual_return"),
            "max_drawdown": (record.get("metrics") or {}).get("long_only_max_drawdown"),
            "holdout_contaminated": record["factor_id"] in contaminated_ids,
            "favorite": record["factor_id"] in favorite_factors,
        }
        for record in records
    ]
    factor_counts: dict[str, dict[str, int]] = {}
    for record in records:
        task_id = str(record.get("source_task_id") or "legacy-ashare")
        counts = factor_counts.setdefault(task_id, {"total": 0, "eligible": 0, "evaluated": 0})
        counts["total"] += 1
        if record.get("status") in {"ELIGIBLE", "ACTIVE"}:
            counts["eligible"] += 1
        if record.get("metrics"):
            counts["evaluated"] += 1
    research_tasks = []
    for task in base_store.research_tasks():
        research_tasks.append(
            {
                "task_id": task["task_id"],
                "name": task["name"],
                "market": task["market"],
                "status": task["status"],
                "phase": task["phase"],
                "updated_at": task["updated_at"],
                "counts": factor_counts.get(
                    str(task["task_id"]), {"total": 0, "eligible": 0, "evaluated": 0}
                ),
            }
        )
    registry_fingerprint = canonical_hash(
        {
            "runtime": str(RUNTIME_ROOT.resolve()),
            "tasks": [(item["task_id"], item["updated_at"]) for item in research_tasks],
            "factors": [(item["factor_id"], item["status"]) for item in factors],
        }
    )
    construction_defaults = {
        **DEFAULT_CONSTRUCTION,
        "min_factors": int(
            settings.get("autocombine_default_min_factors", DEFAULT_CONSTRUCTION["min_factors"])
        ),
        "max_factors": int(
            settings.get("autocombine_default_max_factors", DEFAULT_CONSTRUCTION["max_factors"])
        ),
        "minimum_weight": float(
            settings.get(
                "autocombine_default_minimum_weight", DEFAULT_CONSTRUCTION["minimum_weight"]
            )
        ),
        "maximum_weight": float(
            settings.get(
                "autocombine_default_maximum_weight", DEFAULT_CONSTRUCTION["maximum_weight"]
            )
        ),
        "weight_step": float(
            settings.get("autocombine_default_weight_step", DEFAULT_CONSTRUCTION["weight_step"])
        ),
        "candidate_pool_limit": int(
            settings.get(
                "autocombine_default_pool_limit", DEFAULT_CONSTRUCTION["candidate_pool_limit"]
            )
        ),
    }
    objective_profile = settings.get("autocombine_default_objective", "DRAWDOWN_FIRST")
    objective_defaults = {
        key: value
        for key, value in OBJECTIVE_PRESETS.get(
            objective_profile, OBJECTIVE_PRESETS["DRAWDOWN_FIRST"]
        ).items()
        if key not in {"label", "description"}
    }
    budget_defaults = {
        **DEFAULT_BUDGET,
        "maximum_experiments": int(
            settings.get(
                "autocombine_default_maximum_experiments",
                DEFAULT_BUDGET["maximum_experiments"],
            )
        ),
        "maximum_llm_proposals": int(
            settings.get(
                "autocombine_default_llm_proposals", DEFAULT_BUDGET["maximum_llm_proposals"]
            )
        ),
        "iteration_interval_seconds": float(
            settings.get(
                "autocombine_default_iteration_interval_seconds",
                DEFAULT_BUDGET["iteration_interval_seconds"],
            )
        ),
    }
    return {
        "tasks": [
            {**_task_view(task), "favorite": task["task_id"] in favorite_tasks}
            for task in combine_store.tasks()
        ],
        "strategies": [
            {**strategy, "favorite": strategy["strategy_id"] in favorite_strategies}
            for strategy in combine_store.strategies()
        ],
        "factors": factors,
        "research_tasks": research_tasks,
        "factor_registry": {
            "runtime_root": str(RUNTIME_ROOT.resolve()),
            "fingerprint": registry_fingerprint,
            "task_count": len(research_tasks),
            "factor_count": len(factors),
        },
        "objective_presets": list(OBJECTIVE_PRESETS.values()),
        "defaults": {
            "data_path": data_path,
            "data_range": data_range,
            "protocol": default_protocol,
            "construction": construction_defaults,
            "objective": objective_defaults,
            "budget": budget_defaults,
        },
        "provider_configured": vault.configured(),
        "maximum_concurrent_tasks": manager.maximum_concurrent_tasks,
    }


@app.post("/api/tasks")
async def create_task(payload: CombineTaskRequest) -> dict[str, Any]:
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
        record = create_task_record(
            base_store,
            name=payload.name,
            market=payload.market,
            data_path=str(data_path),
            protocol=protocol,
            scope=payload.scope.model_dump(),
            construction=payload.construction.model_dump(),
            objective=payload.objective.model_dump(),
            budget=payload.budget.model_dump(),
            notes=payload.notes,
        )
        task = combine_store.create_task(record)
    except (FileNotFoundError, RuntimeError, TypeError, ValueError, OSError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    contaminated_count = sum(
        bool(item.get("holdout_contaminated")) for item in task["factor_snapshot"]
    )
    mechanism_counts: dict[str, int] = {}
    for item in task["factor_snapshot"]:
        mechanism = str(item.get("mechanism", "OTHER"))
        mechanism_counts[mechanism] = mechanism_counts.get(mechanism, 0) + 1
    combine_store.event(
        task["task_id"],
        "action",
        "COMBINE_TASK_CREATED",
        "AutoCombine 任务已创建",
        f"已冻结 {task['factor_count']} 个可见因子，快照 {task['snapshot_hash'][:12]}。"
        + (f" 其中 {contaminated_count} 个因子带隐藏期污染标记。" if contaminated_count else ""),
        payload={
            "snapshot_hash": task["snapshot_hash"],
            "factor_count": task["factor_count"],
            "mechanism_counts": mechanism_counts,
            "semantic_cluster_count": len(
                {item.get("semantic_cluster_id") for item in task["factor_snapshot"]}
            ),
        },
    )
    return _task_view(task)


@app.get("/api/favorites")
async def favorite_index(
    entity_type: Literal["combine_task", "strategy"] | None = None,
) -> dict[str, Any]:
    records = base_store.favorites(entity_type=entity_type, limit=5000)
    return {"favorites": records, "entity_type": entity_type, "count": len(records)}


@app.put("/api/favorites/{entity_type}/{entity_id}")
async def update_favorite(
    entity_type: Literal["combine_task", "strategy"],
    entity_id: str,
    payload: FavoriteRequest,
) -> dict[str, Any]:
    try:
        record = base_store.set_favorite(
            entity_type,
            entity_id,
            favorite=payload.favorite,
            label=payload.label,
            context=payload.context,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    base_store.append_event(
        "action",
        "FAVORITE_ADDED" if payload.favorite else "FAVORITE_REMOVED",
        "AutoCombine 收藏已更新",
        f"{entity_type}:{entity_id} 已{'加入' if payload.favorite else '移出'}收藏。",
        payload={
            "entity_type": entity_type,
            "entity_id": entity_id,
            "favorite": payload.favorite,
        },
    )
    return {"favorite": payload.favorite, "record": record}


@app.get("/api/tasks/{task_id}")
async def task_detail(task_id: str) -> dict[str, Any]:
    task = combine_store.task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="AutoCombine task not found")
    experiments = combine_store.experiments(task_id)
    task_view = _task_view(task)
    task_view["favorite"] = base_store.favorite("combine_task", task_id) is not None
    return {
        "task": task_view,
        "experiments": experiments,
        "frontier": _pareto_frontier(experiments),
        "memories": combine_store.memories(task_id),
        "events": combine_store.events(task_id),
        "best": (
            combine_store.experiment(int(task["best_experiment_id"]))
            if task.get("best_experiment_id")
            else None
        ),
        "research_leader": (
            combine_store.experiment(int(task["best_experiment_id"]))
            if task.get("best_experiment_id")
            else None
        ),
        "qualified_champion": (
            combine_store.experiment(int(task["qualified_experiment_id"]))
            if task.get("qualified_experiment_id")
            else None
        ),
        "production_candidate": (
            combine_store.experiment(int(task["production_candidate_experiment_id"]))
            if task.get("production_candidate_experiment_id")
            else None
        ),
        "worker_alive": manager.alive(task_id),
        "factor_snapshot": [
            {
                "factor_id": item["factor_id"],
                "name": item["name"],
                "family": item["family"],
                "mechanism": item.get("mechanism", "OTHER"),
                "semantic_cluster_id": item.get("semantic_cluster_id"),
                "mechanism_fingerprint": item.get("mechanism_fingerprint"),
                "expression_summary": item.get("expression_summary"),
                "expression_fields": item.get("expression_fields", []),
                "expression_windows": item.get("expression_windows", []),
                "status": item["status"],
                "source_task_id": item["source_task_id"],
                "prefilter_score": item["prefilter_score"],
                "required": item.get("required", False),
                "holdout_contaminated": item.get("holdout_contaminated", False),
            }
            for item in task["factor_snapshot"]
        ],
    }


@app.post("/api/tasks/{task_id}/start")
async def start_task(task_id: str) -> dict[str, Any]:
    if combine_store.task(task_id) is None:
        raise HTTPException(status_code=404, detail="AutoCombine task not found")
    try:
        task = await manager.start(task_id)
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return _task_view(task)


@app.post("/api/tasks/{task_id}/stop")
async def stop_task(task_id: str) -> dict[str, Any]:
    if combine_store.task(task_id) is None:
        raise HTTPException(status_code=404, detail="AutoCombine task not found")
    return _task_view(await manager.stop(task_id))


@app.post("/api/tasks/{task_id}/promote")
async def promote_task(task_id: str, payload: PromoteRequest) -> dict[str, Any]:
    try:
        strategy = combine_store.promote_strategy(task_id, payload.experiment_id, payload.name)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    combine_store.event(
        task_id,
        "delivery",
        "STRATEGY_PROMOTED",
        "候选已进入策略库",
        f"{strategy['strategy_id']} · VERSION {strategy['version']} · QUALIFIED",
        payload={"strategy_id": strategy["strategy_id"], "version": strategy["version"]},
    )
    return strategy


@app.get("/api/strategies")
async def strategy_index() -> dict[str, Any]:
    favorites = base_store.favorite_ids("strategy")
    return {
        "strategies": [
            {**strategy, "favorite": strategy["strategy_id"] in favorites}
            for strategy in combine_store.strategies()
        ]
    }


def _task_view(task: dict[str, Any]) -> dict[str, Any]:
    item = dict(task)
    item.pop("factor_snapshot", None)
    item["worker_alive"] = manager.alive(task["task_id"])
    maximum = max(1, int(task["budget"]["maximum_experiments"]))
    item["progress"] = min(1.0, int(task["iteration"]) / maximum)
    return item


def _pareto_frontier(experiments: list[dict[str, Any]]) -> list[int]:
    candidates = [item for item in experiments if item.get("metrics")]
    frontier: list[int] = []
    for candidate in candidates:
        metrics = candidate["metrics"]
        point = (
            float(metrics.get("portfolio_active_information_ratio", 0.0)),
            float(metrics.get("portfolio_simple_annual_return", 0.0)),
            float(metrics.get("portfolio_max_drawdown", -1.0)),
            -float(metrics.get("portfolio_annual_turnover", 1_000.0)),
            float(metrics.get("portfolio_effective_factor_bets", 1.0)),
            float(metrics.get("portfolio_effective_mechanisms", 1.0)),
            -float(metrics.get("portfolio_max_strategy_active_correlation", 0.0)),
        )
        dominated = False
        for other in candidates:
            if other["id"] == candidate["id"]:
                continue
            other_metrics = other["metrics"]
            comparison = (
                float(other_metrics.get("portfolio_active_information_ratio", 0.0)),
                float(other_metrics.get("portfolio_simple_annual_return", 0.0)),
                float(other_metrics.get("portfolio_max_drawdown", -1.0)),
                -float(other_metrics.get("portfolio_annual_turnover", 1_000.0)),
                float(other_metrics.get("portfolio_effective_factor_bets", 1.0)),
                float(other_metrics.get("portfolio_effective_mechanisms", 1.0)),
                -float(other_metrics.get("portfolio_max_strategy_active_correlation", 0.0)),
            )
            if all(a >= b for a, b in zip(comparison, point, strict=True)) and any(
                a > b for a, b in zip(comparison, point, strict=True)
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(int(candidate["id"]))
    return frontier


def main() -> None:
    uvicorn.run(
        "autoalpha.service.autocombine_app:app",
        host=os.getenv("AUTOCOMBINE_HOST", "127.0.0.1"),
        port=int(os.getenv("AUTOCOMBINE_PORT", "8888")),
        reload=False,
    )


if __name__ == "__main__":
    main()
