from __future__ import annotations

import asyncio
import csv
import hashlib
import hmac
import json
import os
import urllib.error
import urllib.request
import uuid
from collections import Counter
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from datetime import date
from pathlib import Path
from typing import Annotated, Any, Literal

import uvicorn
from fastapi import Cookie, Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator, model_validator

from autoalpha.backtest.presets import (
    A_SHARE_NON_PIT_PROXY_WEEKLY_V1,
    manual_backtest_preset_catalog,
    validate_preset_settings,
)
from autoalpha.config import ResearchConfig
from autoalpha.data.execution_basis import inspect_execution_data_basis
from autoalpha.data.research_fields import build_research_data_capabilities
from autoalpha.data.tushare_catalog import DEFAULT_PRODUCT_IDS, resolve_products
from autoalpha.data.workspace import inspect_data_workspace
from autoalpha.portfolio.products import product_template, product_template_catalog
from autoalpha.service.autocombine import (
    DEFAULT_BUDGET,
    DEFAULT_CONSTRUCTION,
    OBJECTIVE_PRESETS,
)
from autoalpha.service.autocombine import (
    create_task_record as create_combine_task_record,
)
from autoalpha.service.autocombine_store import AutoCombineStore
from autoalpha.service.canonical_evaluation import CANONICAL_LIBRARY_PROTOCOL
from autoalpha.service.credentials import SystemCredentialStore
from autoalpha.service.data_center import (
    build_data_capability_matrix,
    build_data_center_snapshot,
)
from autoalpha.service.data_sync import DataSyncWorker
from autoalpha.service.database_backend import database_runtime_config
from autoalpha.service.factor_behavior import load_behavior_snapshot
from autoalpha.service.factor_homogeneity import (
    build_homogeneity_report,
    factor_homogeneity_integrity,
)
from autoalpha.service.factor_library import build_factor_library
from autoalpha.service.full_llm import role_catalog, summarize_research_team_domains
from autoalpha.service.gate_feedback import gate_feedback_policy
from autoalpha.service.manual_backtest import ManualBacktestSpec, ManualFactorBacktester
from autoalpha.service.metric_convention import check_long_only_metric_convention
from autoalpha.service.multifactor import factor_from_pool_record
from autoalpha.service.paper_trading import PaperStrategySpec, PaperTradingEngine
from autoalpha.service.quantcombine_store import QuantCombineStore
from autoalpha.service.research_manager import ResearchTaskManager
from autoalpha.service.research_protocol import (
    CUSTOM_PROTOCOL_DESIGN,
    RECENT_FIVE_YEAR_BACKWARD,
    REGIME_COVERAGE_BACKWARD,
    default_task_protocol,
    normalize_task_protocol,
    panel_validation_fold_capacity,
    protocol_blockers,
    protocol_data_blockers,
    protocol_fingerprint,
    recent_five_year_task_protocol,
    regime_coverage_task_protocol,
    task_research_config,
)
from autoalpha.service.screener import CrossSectionalScreener, ScreenerSpec
from autoalpha.service.settings_center import (
    RESTART_KEYS,
    GlobalSettingsUpdate,
    GlobalSettingsValues,
    default_settings,
    runtime_snapshot,
    settings_catalog,
    validate_operational_paths,
)
from autoalpha.service.store import ServiceStore
from autoalpha.service.strategy_bus import (
    advance_formal_strategy_lifecycle,
    approve_formal_strategy_transition,
    build_strategy_bus_snapshot,
    create_formal_strategy_from_experiment,
    factor_knowledge_map,
    formal_strategy_library,
    promote_formal_strategy_lifecycle,
    publish_strategy_release_dossier,
    strategy_execution_package,
    strategy_experiment_lineage,
    strategy_lifecycle_readiness,
    strategy_production_funnel,
    strategy_promotion_candidates,
    strategy_release_dossier,
)
from autoalpha.service.system_jobs import (
    SUPPORTED_SYSTEM_JOB_TYPES,
    SystemJobRunner,
    build_gate_funnel_diagnostics,
)
from autoalpha.service.worker import SecretVault

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_ROOT = Path(os.getenv("AUTOALPHA_RUNTIME", PROJECT_ROOT / "runtime-full-llm"))
CONFIG_PATH = Path(os.getenv("AUTOALPHA_CONFIG", PROJECT_ROOT / "config/research.toml"))
SNAPSHOT_TTLS = {
    "factor_library": 900,
    "factor_knowledge_map": 1800,
    "strategy_bus": 300,
    "gate_funnel_diagnostics": 900,
    "gate_feedback_policy": 900,
    "factor_homogeneity_backfill": 3600,
}
CRITICAL_MATERIALIZED_SNAPSHOTS = {
    "factor_library": {
        "label": "因子库主榜",
        "refresh_job_type": "factor_library_refresh",
        "refresh_endpoint": "POST /api/factors/refresh",
        "resource_group": "sqlite-writer",
    },
    "factor_knowledge_map": {
        "label": "因子知识地图",
        "refresh_job_type": "factor_knowledge_map_sync",
        "refresh_endpoint": "POST /api/jobs",
        "resource_group": "sqlite-writer",
    },
    "strategy_bus": {
        "label": "策略实验总线",
        "refresh_job_type": "strategy_bus_sync",
        "refresh_endpoint": "POST /api/strategy-bus/sync",
        "resource_group": "strategy_bus",
    },
    "gate_funnel_diagnostics": {
        "label": "门禁漏斗诊断",
        "refresh_job_type": "gate_funnel_diagnostics",
        "refresh_endpoint": "POST /api/jobs",
        "resource_group": "sqlite-writer",
    },
    "gate_feedback_policy": {
        "label": "门禁反馈策略",
        "refresh_job_type": "gate_feedback_policy_sync",
        "refresh_endpoint": "POST /api/jobs",
        "resource_group": "sqlite-writer",
    },
    "factor_homogeneity_backfill": {
        "label": "因子同质化回填",
        "refresh_job_type": "factor_homogeneity_backfill",
        "refresh_endpoint": "POST /api/jobs",
        "resource_group": "sqlite-writer",
    },
}
TRADE_STATEMENT_FIELDS = (
    "trade_id",
    "trade_date",
    "signal_date",
    "sleeve",
    "symbol",
    "security_name",
    "side",
    "quantity",
    "price_cny",
    "notional_cny",
    "commission_cny",
    "transfer_fee_cny",
    "stamp_duty_cny",
    "total_fees_cny",
    "net_cash_flow_cny",
    "sleeve_cash_after_cny",
)


class SessionRequest(BaseModel):
    token: str


class SettingsRestoreRequest(BaseModel):
    change_note: str = Field(default="恢复历史设置版本", max_length=300)


class SettingsRequest(BaseModel):
    base_url: str
    model: str
    data_path: str
    iteration_interval_seconds: float = Field(default=5.0, ge=0.5, le=3600)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    maximum_active_factors: int = Field(default=5, ge=1, le=12)
    full_llm_enabled: bool = True
    api_key: str | None = None
    tushare_token: str | None = None
    market_data_root: str = "~/MarketData/Ashare"
    data_auto_update_enabled: bool = True
    data_update_hour: int = Field(default=18, ge=0, le=23)

    @field_validator("base_url", "model", "data_path")
    @classmethod
    def required_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value cannot be empty")
        return value.strip()


class DataCenterSettingsRequest(BaseModel):
    data_path: str
    market_data_root: str
    tushare_token: str | None = None
    data_auto_update_enabled: bool = True
    data_update_hour: int = Field(default=18, ge=0, le=23)
    data_product_ids: list[str] = Field(default_factory=lambda: list(DEFAULT_PRODUCT_IDS))

    @field_validator("data_path", "market_data_root")
    @classmethod
    def required_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value cannot be empty")
        return value.strip()

    @field_validator("data_product_ids")
    @classmethod
    def valid_products(cls, value: list[str]) -> list[str]:
        cleaned = list(dict.fromkeys(item.strip() for item in value if item.strip()))
        resolve_products(cleaned)
        if "core_market" not in cleaned:
            raise ValueError("core_market must remain enabled for the canonical research panel")
        return cleaned


class DataSyncRequest(BaseModel):
    dataset_ids: list[str] | None = None
    start_date: date | None = None
    end_date: date | None = None
    run_now: bool = False
    queue: str = Field(default="system", min_length=2, max_length=80)
    priority: int = Field(default=30, ge=0, le=10_000)

    @field_validator("dataset_ids")
    @classmethod
    def valid_datasets(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return value
        cleaned = list(dict.fromkeys(item.strip() for item in value if item.strip()))
        resolve_products(cleaned)
        return cleaned

    @model_validator(mode="after")
    def valid_range(self) -> DataSyncRequest:
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("start_date must not be after end_date")
        return self


class ManualLogRequest(BaseModel):
    category: Literal["audit", "action", "research", "delivery"]
    content: str = Field(min_length=1, max_length=4000)

    @field_validator("content")
    @classmethod
    def clean_content(cls, value: str) -> str:
        return value.strip()


class ResearchProtocolRequest(BaseModel):
    exploration_start: date
    exploration_end: date
    validation_start: date
    validation_end: date
    holdout_start: date
    holdout_end: date
    minimum_folds: int = Field(default=1, ge=1, le=20)
    design: Literal["CUSTOM", "RECENT_FIVE_YEAR_BACKWARD", "REGIME_COVERAGE_BACKWARD"] = (
        CUSTOM_PROTOCOL_DESIGN
    )
    anchor_date: date | None = None
    exploration_years: int | None = Field(default=None, ge=2, le=10)
    validation_years: int | None = Field(default=None, ge=1, le=5)
    holdout_months: int | None = Field(default=None, ge=3, le=24)
    embargo_days: int | None = Field(default=None, ge=0, le=90)


class ResearchProtocolPresetRequest(BaseModel):
    data_path: str
    data_start: date
    data_end: date
    design: Literal["CUSTOM", "RECENT_FIVE_YEAR_BACKWARD", "REGIME_COVERAGE_BACKWARD"] = (
        RECENT_FIVE_YEAR_BACKWARD
    )
    exploration_years: int = Field(default=5, ge=2, le=10)
    validation_years: int | None = Field(default=None, ge=1, le=5)
    holdout_months: int = Field(default=6, ge=3, le=24)
    embargo_days: int = Field(default=30, ge=0, le=90)

    @property
    def resolved_validation_years(self) -> int:
        if self.validation_years is not None:
            return self.validation_years
        return 3 if self.design == REGIME_COVERAGE_BACKWARD else 2

    @model_validator(mode="after")
    def validate_coverage(self) -> ResearchProtocolPresetRequest:
        if self.data_start > self.data_end:
            raise ValueError("data_start must not be after data_end")
        return self


class ResearchProtocolPreviewRequest(BaseModel):
    data_path: str
    protocol: ResearchProtocolRequest


class ResearchTaskRequest(BaseModel):
    name: str = Field(min_length=2, max_length=80)
    market: Literal["CN_A", "HK", "US"] = "CN_A"
    data_path: str
    data_start: date | None = None
    data_end: date | None = None
    protocol: ResearchProtocolRequest | None = None
    notes: str = Field(default="", max_length=1000)

    @field_validator("name", "data_path", "notes")
    @classmethod
    def clean_task_text(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def validate_task_range(self) -> ResearchTaskRequest:
        if self.data_start and self.data_end and self.data_start > self.data_end:
            raise ValueError("data_start must not be after data_end")
        return self


class ManualBacktestRequest(BaseModel):
    factor_ids: list[str] = Field(min_length=1, max_length=12)
    weights: list[float] | None = None
    start_date: date
    end_date: date
    initial_cash_cny: float = Field(default=1_000_000, ge=10_000, le=10_000_000_000)
    gross_exposure: float = Field(default=0.5, ge=0.05, le=1.0)
    holding_period_days: int = Field(default=5, ge=1, le=60)
    backtest_preset: Literal[
        "CUSTOM",
        "A_SHARE_REALISTIC_WEEKLY_V1",
        "A_SHARE_NON_PIT_PROXY_WEEKLY_V1",
    ] = "CUSTOM"
    backtest_engine: Literal["VECTOR", "EVENT_LEDGER"] = "VECTOR"
    execution_data_mode: Literal["STRICT_PIT", "NON_PIT_PROXY"] = "STRICT_PIT"
    rebalance_schedule: Literal[
        "DAILY_ROLLING",
        "WEEKLY_FIRST_SESSION",
        "MONTHLY_FIRST_SESSION",
    ] = "DAILY_ROLLING"
    vector_cost_model: Literal["side_aware", "legacy_half_turnover"] = "side_aware"
    product_template: Literal[
        "MARKET_NEUTRAL_RESEARCH",
        "LONG_ONLY_CAPITAL",
        "UNIVERSE_INDEX_ENHANCED_PROXY",
        "UNIVERSE_HEDGED_PROXY",
    ] = "MARKET_NEUTRAL_RESEARCH"
    selection_fraction: float = Field(default=0.10, gt=0, le=0.50)
    maximum_positions: int = Field(default=30, ge=1, le=300)
    lot_size: int = Field(default=100, ge=100, le=10_000)
    maximum_volume_participation: float = Field(default=0.05, gt=0, le=0.50)
    opening_limit_threshold: float = Field(default=0.095, ge=0.01, le=0.30)
    commission_bps_each_side: float = Field(default=1.5, ge=0, le=100)
    stamp_duty_bps_sell: float = Field(default=5.0, ge=0, le=100)
    transfer_fee_bps_each_side: float = Field(default=0.1, ge=0, le=100)
    minimum_commission_cny: float = Field(default=5.0, ge=0, le=1000)
    slippage_bps_each_side: float = Field(default=0.0, ge=0, le=500)
    use_historical_fee_schedule: bool = False
    cost_stress_multiplier: float = Field(default=2.0, ge=1, le=10)

    @field_validator("factor_ids")
    @classmethod
    def unique_factors(cls, value: list[str]) -> list[str]:
        cleaned = [factor_id.strip() for factor_id in value]
        if any(not factor_id for factor_id in cleaned):
            raise ValueError("factor ids cannot be empty")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("factor ids must be unique")
        return cleaned

    @field_validator("weights")
    @classmethod
    def positive_weights(cls, value: list[float] | None) -> list[float] | None:
        if value is not None and any(weight <= 0 for weight in value):
            raise ValueError("weights must be positive")
        return value

    @field_validator("lot_size")
    @classmethod
    def a_share_lot_size(cls, value: int) -> int:
        if value % 100:
            raise ValueError("lot_size must be a multiple of 100 shares")
        return value

    @model_validator(mode="after")
    def validate_engine_and_preset(self) -> ManualBacktestRequest:
        if self.backtest_engine == "VECTOR" and self.rebalance_schedule != "DAILY_ROLLING":
            raise ValueError("Fixed-calendar rebalance schedules require the event ledger")
        if self.execution_data_mode == "NON_PIT_PROXY" and self.backtest_engine != "EVENT_LEDGER":
            raise ValueError("NON_PIT_PROXY execution requires the event ledger")
        validate_preset_settings(self.backtest_preset, self.model_dump(mode="python"))
        return self


class FactorScreenRequest(BaseModel):
    factor_ids: list[str] = Field(min_length=1, max_length=12)
    weights: list[float] | None = None
    as_of_date: date
    selection_count: int = Field(default=30, ge=1, le=500)
    selection_side: Literal["TOP", "BOTTOM"] = "TOP"

    @field_validator("factor_ids")
    @classmethod
    def unique_factors(cls, value: list[str]) -> list[str]:
        cleaned = [factor_id.strip() for factor_id in value]
        if any(not factor_id for factor_id in cleaned):
            raise ValueError("factor ids cannot be empty")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("factor ids must be unique")
        return cleaned

    @field_validator("weights")
    @classmethod
    def positive_weights(cls, value: list[float] | None) -> list[float] | None:
        if value is not None and any(weight <= 0 for weight in value):
            raise ValueError("weights must be positive")
        return value


class QuickAutoCombineRequest(BaseModel):
    factor_ids: list[str] = Field(min_length=1, max_length=100)
    objective_profile: Literal[
        "ROBUST_ACTIVE_LONG_ONLY",
        "DRAWDOWN_FIRST",
        "PORTFOLIO_SHARPE_FIRST",
        "ABSOLUTE_LONG_ONLY",
        "LOW_TURNOVER",
        "DIVERSIFICATION_FIRST",
    ] = "DRAWDOWN_FIRST"
    maximum_factors: int = Field(default=5, ge=1, le=12)
    start_immediately: bool = True

    @field_validator("factor_ids")
    @classmethod
    def unique_factors(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(item.strip() for item in value if item.strip()))


class PaperPortfolioRequest(FactorScreenRequest):
    name: str = Field(min_length=1, max_length=80)
    initial_cash_cny: float = Field(default=1_000_000, ge=10_000, le=10_000_000_000)
    gross_exposure: float = Field(default=0.95, gt=0.05, le=1.0)
    slippage_bps_each_side: float = Field(default=5.0, ge=0, le=500)


class StrategyPaperPortfolioRequest(BaseModel):
    initial_cash_cny: float = Field(default=1_000_000, ge=10_000, le=10_000_000_000)
    as_of_date: date
    name: str | None = Field(default=None, max_length=80)
    gross_exposure: float | None = Field(default=None, gt=0.05, le=1.0)
    slippage_bps_each_side: float | None = Field(default=None, ge=0, le=500)
    selection_count: int | None = Field(default=None, ge=1, le=500)


class PaperPortfolioStatusRequest(BaseModel):
    status: Literal["ACTIVE", "PAUSED", "CLOSED"]


class FormalStrategyCreateRequest(BaseModel):
    experiment_id: str
    name: str | None = Field(default=None, max_length=120)
    lifecycle: Literal["RESEARCH"] = "RESEARCH"


class FormalStrategyPromotionRequest(BaseModel):
    target_lifecycle: Literal[
        "FROZEN",
        "HIDDEN_HOLDOUT",
        "SHADOW",
        "PAPER",
        "PRODUCTION_CANDIDATE",
    ]
    evidence: dict[str, Any] = Field(default_factory=dict)


class FormalStrategyApprovalRequest(BaseModel):
    target_lifecycle: Literal[
        "FROZEN",
        "HIDDEN_HOLDOUT",
        "SHADOW",
        "PAPER",
        "PRODUCTION_CANDIDATE",
    ] | None = None
    approval_type: Literal[
        "PUBLIC_VALIDATION_REVIEW",
        "HIDDEN_HOLDOUT_REVIEW",
        "SHADOW_EXECUTION_REVIEW",
        "PAPER_TRADING_REVIEW",
        "RISK_APPROVAL",
    ]
    approver: str = Field(default="local-operator", min_length=2, max_length=80)
    notes: str = Field(default="", max_length=1000)
    evidence: dict[str, Any] = Field(default_factory=dict)


class SystemJobRequest(BaseModel):
    queue: str = Field(min_length=2, max_length=80)
    job_type: str = Field(min_length=2, max_length=120)
    payload: dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(default=100, ge=0, le=10_000)
    resource_group: str = Field(default="default", min_length=1, max_length=80)
    max_workers: int = Field(default=1, ge=1, le=128)
    progress_total: int = Field(default=0, ge=0, le=10_000_000)
    max_attempts: int = Field(default=3, ge=1, le=20)


class StrategyBusSyncRequest(BaseModel):
    run_now: bool = False
    force_new: bool = False
    queue: str = Field(default="system", min_length=2, max_length=80)
    priority: int = Field(default=35, ge=0, le=10_000)


class FactorLibraryRefreshRequest(BaseModel):
    run_now: bool = False
    force_new: bool = False
    queue: str = Field(default="system", min_length=2, max_length=80)
    priority: int = Field(default=40, ge=0, le=10_000)


class FactorKnowledgeMapSyncRequest(BaseModel):
    run_now: bool = False
    force_new: bool = False
    queue: str = Field(default="system", min_length=2, max_length=80)
    priority: int = Field(default=45, ge=0, le=10_000)


class GateFeedbackRepairSeedRequest(BaseModel):
    data_path: str | None = Field(default=None, max_length=1000)
    protocol: dict[str, Any] | None = None
    name: str = Field(default="Gate feedback repair · QuantCombine", max_length=120)
    notes: str = Field(default="Seeded from strategy gate feedback policy.", max_length=2000)
    queue: str = Field(default="system", min_length=2, max_length=80)
    priority: int = Field(default=40, ge=0, le=10_000)
    run_immediately: bool = False
    auto_start_task: bool = True

    @field_validator("data_path", "name", "notes", "queue")
    @classmethod
    def clean_optional_text(cls, value: str | None) -> str | None:
        return value.strip() if isinstance(value, str) else value


class StrategyLibrarySeedJobRequest(BaseModel):
    limit: int = Field(default=20, ge=1, le=100)
    candidate_classes: list[str] = Field(
        default_factory=lambda: ["QUALIFIED", "RESEARCH_LEADER"],
        max_length=8,
    )
    queue: str = Field(default="system", min_length=2, max_length=80)
    priority: int = Field(default=45, ge=0, le=10_000)
    run_immediately: bool = False

    @field_validator("candidate_classes")
    @classmethod
    def clean_candidate_classes(cls, value: list[str]) -> list[str]:
        cleaned = sorted({item.strip() for item in value if item.strip()})
        if not cleaned:
            raise ValueError("candidate_classes cannot be empty")
        return cleaned


class StrategyFreezeReadyJobRequest(BaseModel):
    limit: int = Field(default=100, ge=1, le=1000)
    queue: str = Field(default="system", min_length=2, max_length=80)
    priority: int = Field(default=50, ge=0, le=10_000)
    run_immediately: bool = False


class SystemJobClaimRequest(BaseModel):
    queue: str = Field(min_length=2, max_length=80)
    worker_id: str = Field(min_length=2, max_length=120)
    lease_seconds: int = Field(default=300, ge=30, le=86_400)
    resource_group: str | None = Field(default=None, max_length=80)
    max_queue_running: int | None = Field(default=None, ge=1, le=256)
    max_global_running: int | None = Field(default=None, ge=1, le=512)


class SystemJobHeartbeatRequest(BaseModel):
    worker_id: str = Field(min_length=2, max_length=120)
    lease_seconds: int = Field(default=300, ge=30, le=86_400)
    progress_current: int | None = Field(default=None, ge=0, le=10_000_000)
    checkpoint: dict[str, Any] | None = None


class SystemJobCommandRequest(BaseModel):
    actor: str = Field(default="local-operator", min_length=2, max_length=80)
    reason: str = Field(default="", max_length=500)


class SystemJobRunNextRequest(BaseModel):
    queue: str = Field(default="system", min_length=2, max_length=80)
    lease_seconds: int = Field(default=900, ge=30, le=86_400)
    max_queue_running: int | None = Field(default=None, ge=1, le=256)
    max_global_running: int | None = Field(default=None, ge=1, le=512)


class ManualBacktestMetadataRequest(BaseModel):
    favorite: bool = True
    title: str | None = Field(default=None, max_length=100)
    notes: str = Field(default="", max_length=2000)
    tags: list[str] = Field(default_factory=list, max_length=8)

    @field_validator("tags")
    @classmethod
    def clean_tags(cls, value: list[str]) -> list[str]:
        cleaned = [tag.strip() for tag in value if tag.strip()]
        if any(len(tag) > 24 for tag in cleaned):
            raise ValueError("each tag must contain at most 24 characters")
        return list(dict.fromkeys(cleaned))


FavoriteEntityType = Literal[
    "factor",
    "research_task",
    "llm_artifact",
    "paper_portfolio",
    "screener_preset",
    "combine_task",
    "strategy",
    "strategy_experiment",
    "strategy_version",
    "system_job",
]


class FavoriteRequest(BaseModel):
    favorite: bool = True
    label: str = Field(default="", max_length=160)
    context: dict[str, Any] = Field(default_factory=dict)


class LifecycleTransitionRequest(BaseModel):
    target_state: Literal[
        "RESEARCH",
        "QUALIFIED",
        "SHADOW",
        "PAPER",
        "PRODUCTION",
        "WATCH",
        "DECAYED",
        "SUSPENDED",
        "RETIRED",
        "REJECTED",
    ]
    reason: str = Field(min_length=3, max_length=1000)


store = ServiceStore(RUNTIME_ROOT / "autoalpha.sqlite3")
combine_store = AutoCombineStore(store)
quant_store = QuantCombineStore(store)
vault = SecretVault(credential_store=SystemCredentialStore())
research_manager = ResearchTaskManager(
    store,
    vault,
    config_path=CONFIG_PATH,
    artifact_root=RUNTIME_ROOT / "artifacts",
    maximum_concurrent_iterations=int(
        os.getenv(
            "AUTOALPHA_MAX_CONCURRENT_RESEARCH",
            store.settings().get("research_concurrency", "2"),
        )
    ),
)
manual_backtest_lock = asyncio.Lock()
# Research and manual backtests only read the immutable panel and may overlap.
# A panel refresh remains exclusive through DataSyncWorker.is_busy.
data_sync_worker = DataSyncWorker(
    store,
    project_root=PROJECT_ROOT.parent,
    is_busy=lambda: research_manager.alive or manual_backtest_lock.locked(),
)
system_job_runner = SystemJobRunner(
    store,
    autocombine_store=combine_store,
    quantcombine_store=quant_store,
    runtime_root=RUNTIME_ROOT,
    market_data_sync_runner=data_sync_worker.run_system_job,
)
system_job_scheduler_task: asyncio.Task[None] | None = None


def _system_job_scheduler_status() -> dict[str, Any]:
    enabled = os.getenv("AUTOALPHA_SYSTEM_JOB_WORKER_ENABLED", "true").casefold() == "true"
    alive = system_job_scheduler_task is not None and not system_job_scheduler_task.done()
    status = "running" if alive else ("disabled" if not enabled else "not_started")
    failure: str | None = None
    if system_job_scheduler_task is not None and system_job_scheduler_task.done():
        with suppress(asyncio.CancelledError):
            error = system_job_scheduler_task.exception()
            if error is not None:
                status = "failed"
                failure = f"{type(error).__name__}: {error}"
    return {
        "enabled": enabled,
        "alive": alive,
        "status": status,
        "queue": "system",
        "poll_seconds": {"claimed": 2, "idle": 15, "after_error": 30},
        "supported_job_types": sorted(SUPPORTED_SYSTEM_JOB_TYPES),
        "failure": failure,
    }


def _ensure_legacy_research_task() -> None:
    if store.research_task("legacy-ashare") is not None:
        return
    settings = store.settings()
    data_path = Path(settings.get("data_path", PROJECT_ROOT.parent / "data")).expanduser()
    data_start: str | None = None
    data_end: str | None = None
    snapshot_hash: str | None = None
    try:
        workspace = inspect_data_workspace(data_path)
        data_start = workspace.first_trade_date
        data_end = workspace.last_trade_date
        snapshot_hash = workspace.fingerprint
    except (FileNotFoundError, RuntimeError, TypeError, ValueError, OSError):
        pass
    state = store.state()
    base_config = ResearchConfig.from_toml(CONFIG_PATH)
    protocol = default_task_protocol(
        data_start or base_config.splits.train.start.isoformat(),
        data_end or base_config.splits.test.end.isoformat(),
        base_config,
        preserve_base=True,
    )
    store.create_research_task(
        task_id="legacy-ashare",
        name="历史 A 股研究",
        market="CN_A",
        data_path=str(data_path),
        data_start=data_start,
        data_end=data_end,
        snapshot_hash=snapshot_hash,
        status=str(state["state"]),
        run_id=state.get("run_id"),
        protocol=protocol,
        protocol_hash=protocol_fingerprint(protocol),
        notes="服务升级前的单例研究记录已归档到此任务。",
    )


def _task_data_snapshot(payload: ResearchTaskRequest) -> dict[str, Any]:
    data_path = Path(payload.data_path).expanduser().resolve()
    try:
        workspace = inspect_data_workspace(data_path)
    except (FileNotFoundError, RuntimeError, TypeError, ValueError, OSError) as error:
        return {
            "data_path": str(data_path),
            "data_start": payload.data_start.isoformat() if payload.data_start else None,
            "data_end": payload.data_end.isoformat() if payload.data_end else None,
            "snapshot_hash": None,
            "panel_path": None,
            "status": "DATA_REQUIRED",
            "data_error": f"{type(error).__name__}: {error}",
        }
    first = date.fromisoformat(str(workspace.first_trade_date))
    last = date.fromisoformat(str(workspace.last_trade_date))
    requested_start = payload.data_start or first
    requested_end = payload.data_end or last
    if requested_start < first or requested_end > last:
        raise ValueError(
            f"Task range must stay within the panel coverage {first.isoformat()} to "
            f"{last.isoformat()}"
        )
    fingerprint = hashlib.sha256(
        f"{payload.market}|{workspace.fingerprint}|{requested_start}|{requested_end}".encode()
    ).hexdigest()
    return {
        "data_path": str(data_path),
        "data_start": requested_start.isoformat(),
        "data_end": requested_end.isoformat(),
        "snapshot_hash": fingerprint,
        "panel_path": workspace.panel_path,
        "status": "READY",
        "data_error": None,
    }


def _resolved_task_protocol(
    payload: ResearchTaskRequest,
    snapshot: dict[str, Any],
    *,
    preserve_base: bool = False,
) -> dict[str, Any]:
    if not snapshot.get("data_start") or not snapshot.get("data_end"):
        return {}
    base = ResearchConfig.from_toml(CONFIG_PATH)
    protocol = (
        normalize_task_protocol(payload.protocol.model_dump(mode="json"))
        if payload.protocol
        else default_task_protocol(
            str(snapshot["data_start"]),
            str(snapshot["data_end"]),
            base,
            preserve_base=preserve_base,
        )
    )
    blockers = protocol_blockers(
        protocol,
        data_start=str(snapshot["data_start"]),
        data_end=str(snapshot["data_end"]),
    )
    if snapshot.get("panel_path"):
        blockers.extend(protocol_data_blockers(protocol, Path(str(snapshot["panel_path"]))))
    if blockers:
        raise ValueError("；".join(blockers))
    return protocol


def _research_task_view(task: dict[str, Any]) -> dict[str, Any]:
    item = dict(task)
    if item["task_id"] == "legacy-ashare":
        state = store.state()
        item["status"] = state["state"]
        item["run_id"] = state.get("run_id")
        item["phase"] = state["phase"]
        item["iteration"] = state["iteration"]
        item["stop_requested"] = state["stop_requested"]
        item["last_error"] = state["last_error"]
    item.update(store.research_task_stats(item["task_id"], item.get("run_id")))
    item["data_ready"] = bool(item.get("snapshot_hash") and item.get("data_start"))
    item["worker_alive"] = research_manager.worker_alive(str(item["task_id"]))
    try:
        item["readiness"] = research_manager.readiness(str(item["task_id"]))
    except (KeyError, RuntimeError, ValueError, OSError) as error:
        item["readiness"] = {"runnable": False, "blockers": [str(error)]}
    return item


def _task_research_context(
    task_id: str,
) -> tuple[dict[str, Any], ResearchConfig, str, dict[str, Any] | None]:
    task = store.research_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Research task not found: {task_id}")
    base = ResearchConfig.from_toml(CONFIG_PATH)
    config = (
        task_research_config(base, task["protocol"], task_id=task_id)
        if task.get("protocol")
        else base
    )
    generation_base = config.generation
    if task_id != "legacy-ashare":
        generation_base = f"{generation_base}--{task_id}"
    generation = store.latest_generation(generation_base)
    generation_id = str(generation["generation_id"]) if generation else generation_base
    return task, config, generation_id, generation


def _factor_source_context(
    records: list[dict[str, Any]],
) -> tuple[Path, str, list[str]]:
    task_ids = sorted({str(record.get("source_task_id") or "legacy-ashare") for record in records})
    tasks = []
    for task_id in task_ids:
        task = store.research_task(task_id)
        if task is None:
            raise HTTPException(
                status_code=409,
                detail=f"Factor source task is unavailable: {task_id}",
            )
        tasks.append(task)
    contexts = {(str(task["market"]), str(task["data_path"])) for task in tasks}
    if len(contexts) != 1:
        raise HTTPException(
            status_code=422,
            detail="Selected factors span different markets or data workspaces",
        )
    market, data_path = next(iter(contexts))
    if market != "CN_A":
        raise HTTPException(
            status_code=422,
            detail=f"The current screener and backtest engines do not support market {market}",
        )
    return Path(data_path), market, task_ids


def _start_autocombine_task(task_id: str) -> tuple[bool, str | None]:
    base_url = os.getenv("AUTOCOMBINE_URL", "http://127.0.0.1:8888").rstrip("/")
    request = urllib.request.Request(
        f"{base_url}/api/tasks/{task_id}/start",
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5):  # noqa: S310
            return True, None
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return False, f"{type(error).__name__}: {error}"


def _autocombine_health() -> dict[str, Any] | None:
    base_url = os.getenv("AUTOCOMBINE_URL", "http://127.0.0.1:8888").rstrip("/")
    try:
        with urllib.request.urlopen(f"{base_url}/api/health", timeout=2) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
        return None


def _backfill_task_protocols() -> None:
    base = ResearchConfig.from_toml(CONFIG_PATH)
    for task in store.research_tasks():
        if task.get("protocol") or not task.get("data_start") or not task.get("data_end"):
            continue
        protocol = default_task_protocol(
            str(task["data_start"]),
            str(task["data_end"]),
            base,
            preserve_base=task["task_id"] == "legacy-ashare",
        )
        store.update_research_task(
            str(task["task_id"]),
            protocol_json=json.dumps(protocol, sort_keys=True, separators=(",", ":")),
            protocol_hash=protocol_fingerprint(protocol),
            protocol_revision=1,
        )


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    global system_job_scheduler_task
    settings = store.settings()
    managed_defaults = GlobalSettingsValues.model_validate(
        default_settings(PROJECT_ROOT)
    ).to_store()
    missing_defaults = {
        key: value for key, value in managed_defaults.items() if key not in settings
    }
    if missing_defaults:
        store.save_settings(missing_defaults)
    _ensure_legacy_research_task()
    _backfill_task_protocols()
    state = store.state()
    orphans = store.reconcile_orphaned_iterations()
    for orphan in orphans:
        store.remember(
            orphan["run_id"],
            orphan["iteration"],
            "failure",
            f"FAILED during service restart: {orphan['error']}",
        )
        store.append_event(
            "audit",
            "ORPHANED_ITERATION_RECONCILED",
            "中断轮次已结算",
            f"第 {orphan['iteration']} 轮在服务重启时仍处于运行态，现已标记失败。",
            run_id=orphan["run_id"],
            iteration=orphan["iteration"],
            level="WARN",
            payload={"error": orphan["error"]},
        )
    orphaned_experiments = store.reconcile_orphaned_generation_experiments()
    for experiment in orphaned_experiments:
        store.append_event(
            "audit",
            "ORPHANED_GENERATION_EXPERIMENT_RECONCILED",
            "中断候选实验已结算",
            f"第 {experiment['iteration']} 轮的世代候选在服务重启时仍处于预留态，现已标记崩溃。",
            iteration=experiment["iteration"],
            level="WARN",
            payload={
                "candidate_hash": experiment["candidate_hash"],
                "generation_id": experiment["generation_id"],
                "public_verdict": "SERVICE_RESTART_INTERRUPTED",
            },
        )
    direction_config = ResearchConfig.from_toml(CONFIG_PATH).adaptive_direction
    orphaned_direction_attempts = store.reconcile_orphaned_direction_attempts(
        early_stop_consecutive_misses=direction_config.early_stop_consecutive_misses
    )
    for attempt in orphaned_direction_attempts:
        store.append_event(
            "audit",
            "ORPHANED_DIRECTION_ATTEMPT_RECONCILED",
            "中断方向尝试已从科研预算剥离",
            "服务中断属于运维故障，保留审计记录但不消耗方向研究额度。",
            run_id=attempt.get("run_id"),
            iteration=attempt["iteration"],
            level="WARN",
            payload=attempt,
        )
    exhausted_campaigns = store.reconcile_exhausted_direction_campaigns()
    for campaign in exhausted_campaigns:
        store.append_event(
            "audit",
            "EXHAUSTED_DIRECTION_CAMPAIGN_RECONCILED",
            "异常活动战役已关闭",
            "战役额度已耗尽但状态仍为活动，启动恢复已将其安全关闭。",
            iteration=campaign.get("last_iteration"),
            level="WARN",
            payload=campaign,
        )
    if state["state"] == "STOPPING":
        store.update_state(state="STOPPED", phase="STOPPED", stop_requested=0)
        store.append_event(
            "audit",
            "STOP_INTENT_RECOVERED",
            "停止状态已恢复",
            "服务重启前已收到停止请求，本次启动不会恢复研究循环。",
            run_id=state.get("run_id"),
            iteration=state.get("iteration"),
            level="WARN",
        )
    elif state["state"] in {"RUNNING", "RETRYING"} and not vault.configured():
        store.update_state(
            state="WAITING_CONFIGURATION", phase="CONFIGURE", last_error="API key required"
        )
    await research_manager.restore()
    await data_sync_worker.start_scheduler()
    if os.getenv("AUTOALPHA_SYSTEM_JOB_WORKER_ENABLED", "true").casefold() == "true":
        system_job_scheduler_task = asyncio.create_task(
            _system_job_scheduler_loop(),
            name="autoalpha-system-job-scheduler",
        )
    yield
    if system_job_scheduler_task and not system_job_scheduler_task.done():
        system_job_scheduler_task.cancel()
        with suppress(asyncio.CancelledError):
            await system_job_scheduler_task
    await data_sync_worker.shutdown()
    await research_manager.shutdown()


app = FastAPI(title="AutoAlpha Control Plane", version="0.7.0-full-llm", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=PACKAGE_ROOT / "static"), name="static")


async def _system_job_scheduler_loop() -> None:
    while True:
        try:
            result = await asyncio.to_thread(system_job_runner.run_next, queue="system")
            await asyncio.sleep(2 if result.get("claimed") else 15)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            store.append_event(
                "audit",
                "SYSTEM_JOB_SCHEDULER_ERROR",
                "系统作业调度器异常",
                f"{type(error).__name__}: {error}",
                level="ERROR",
            )
            await asyncio.sleep(30)


def _runtime_database_health() -> dict[str, Any]:
    backend_config = database_runtime_config()
    database_path = (RUNTIME_ROOT / "autoalpha.sqlite3").resolve()
    required_tables = {
        "factor_pool",
        "research_tasks",
        "strategy_experiment_objects",
        "formal_strategy_versions",
        "system_jobs",
        "materialized_snapshots",
        "paper_portfolios",
    }
    with store.connection() as connection:
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        tables = {str(row["name"]) for row in rows}
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()
        user_version = connection.execute("PRAGMA user_version").fetchone()
        busy_timeout = connection.execute("PRAGMA busy_timeout").fetchone()
        synchronous = connection.execute("PRAGMA synchronous").fetchone()
        wal_autocheckpoint = connection.execute("PRAGMA wal_autocheckpoint").fetchone()
        locking_mode = connection.execute("PRAGMA locking_mode").fetchone()
    missing = sorted(required_tables - tables)
    wal_path = database_path.with_name(f"{database_path.name}-wal")
    shm_path = database_path.with_name(f"{database_path.name}-shm")
    snapshot_summary = store.materialized_snapshot_summary()
    job_summary = store.system_job_summary()
    return {
        "runtime_root": str(RUNTIME_ROOT.resolve()),
        "backend": backend_config.to_dict(),
        "database_path": str(database_path),
        "schema_ready": not missing,
        "missing_tables": missing,
        "table_count": len(tables),
        "journal_mode": str(journal_mode[0]) if journal_mode else "unknown",
        "busy_timeout_ms": int(busy_timeout[0]) if busy_timeout else 0,
        "synchronous": int(synchronous[0]) if synchronous else None,
        "wal_autocheckpoint_pages": (
            int(wal_autocheckpoint[0]) if wal_autocheckpoint else None
        ),
        "locking_mode": str(locking_mode[0]) if locking_mode else "unknown",
        "database_size_bytes": _file_size(database_path),
        "wal_size_bytes": _file_size(wal_path),
        "shm_size_bytes": _file_size(shm_path),
        "materialized_snapshots": {
            "total": snapshot_summary["total"],
            "states": snapshot_summary["states"],
            "stale_count": snapshot_summary["stale_count"],
            "no_ttl_count": snapshot_summary["no_ttl_count"],
            "stale_keys": snapshot_summary["stale_keys"],
            "no_ttl_keys": snapshot_summary["no_ttl_keys"],
            "snapshots": snapshot_summary["snapshots"],
        },
        "system_jobs": {
            "total": job_summary["total"],
            "expired_running_count": job_summary["expired_running_count"],
            "expired_running": job_summary["expired_running"],
        },
        "user_version": int(user_version[0]) if user_version else 0,
        "service_variant": os.getenv("AUTOALPHA_VARIANT", "FULL LLM"),
    }


def _materialized_snapshot_policy(summary: dict[str, Any] | None = None) -> dict[str, Any]:
    summary = summary or store.materialized_snapshot_summary()
    snapshot_by_key = {
        str(item["key"]): item for item in summary.get("snapshots", [])
    }
    rows = []
    missing_keys = []
    stale_keys = []
    for key, policy in CRITICAL_MATERIALIZED_SNAPSHOTS.items():
        snapshot = snapshot_by_key.get(key)
        cache_state = (snapshot or {}).get("cache_state") or {}
        state = "MISSING" if snapshot is None else str(cache_state.get("status") or "UNKNOWN")
        if state == "MISSING":
            missing_keys.append(key)
        elif state == "STALE":
            stale_keys.append(key)
        rows.append(
            {
                "key": key,
                "label": policy["label"],
                "state": state,
                "present": snapshot is not None,
                "updated_at": (snapshot or {}).get("updated_at"),
                "expires_at": (snapshot or {}).get("expires_at"),
                "source": (snapshot or {}).get("source"),
                "refresh_job_type": policy["refresh_job_type"],
                "refresh_endpoint": policy["refresh_endpoint"],
                "resource_group": policy["resource_group"],
                "read_policy": (
                    "GET endpoints must read cached snapshots or return cache-miss metadata"
                ),
            }
        )
    return {
        "protocol": "AUTOALPHA_MATERIALIZED_SNAPSHOT_POLICY_V1",
        "status": "READY" if not missing_keys and not stale_keys else "ATTENTION",
        "critical_count": len(rows),
        "missing_keys": missing_keys,
        "stale_keys": stale_keys,
        "rows": rows,
        "policy": (
            "large derived views are materialized; refresh through Job Center or explicit POST; "
            "GET requests remain read-only"
        ),
    }


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def _pid_file_status(name: str, port: int) -> dict[str, Any]:
    pid_path = RUNTIME_ROOT / "pids" / f"{name}-{port}.pid"
    raw_pid = ""
    pid: int | None = None
    alive = False
    try:
        raw_pid = pid_path.read_text().strip()
        pid = int(raw_pid)
        os.kill(pid, 0)
        alive = True
    except (FileNotFoundError, ValueError, ProcessLookupError, PermissionError, OSError):
        alive = False
    return {
        "name": name,
        "port": port,
        "pid_file": str(pid_path),
        "pid": pid,
        "pid_file_present": pid_path.exists(),
        "alive": alive,
        "status": "ALIVE" if alive else ("STALE_PID" if raw_pid else "MISSING_PID"),
    }


def _platform_doctor_snapshot() -> dict[str, Any]:
    expected_routes = {
        "/health",
        "/ready",
        "/api/platform/doctor",
        "/api/jobs",
        "/api/factors",
        "/api/factor-knowledge-map",
        "/api/factor-knowledge-map/sync",
        "/api/strategy-bus",
        "/api/strategy-bus/sync",
        "/api/strategy-library",
        "/api/manual-backtests",
        "/api/paper-portfolios",
    }
    route_paths = {getattr(route, "path", "") for route in app.routes}
    missing_routes = sorted(expected_routes - route_paths)
    database_health = _runtime_database_health()
    snapshot_policy = _materialized_snapshot_policy(
        database_health.get("materialized_snapshots") or None
    )
    scheduler = _system_job_scheduler_status()
    processes = [
        _pid_file_status("autoalpha", int(os.getenv("AUTOALPHA_PORT", "8788"))),
        _pid_file_status("autocombine", int(os.getenv("AUTOCOMBINE_PORT", "8888"))),
        _pid_file_status("quantcombine", int(os.getenv("QUANTCOMBINE_PORT", "8889"))),
    ]
    blockers = []
    if missing_routes:
        blockers.append("missing_expected_routes")
    if not database_health.get("schema_ready"):
        blockers.append("database_schema_not_ready")
    if snapshot_policy["status"] != "READY":
        blockers.append("materialized_snapshot_attention")
    if scheduler["enabled"] and scheduler["status"] == "failed":
        blockers.append("system_job_scheduler_failed")
    stale_or_missing_pids = [
        item["name"] for item in processes if item["status"] != "ALIVE"
    ]
    if stale_or_missing_pids:
        blockers.append("stale_or_missing_service_pid_files")
    return {
        "protocol": "AUTOALPHA_PLATFORM_DOCTOR_V1",
        "status": "OK" if not blockers else "ATTENTION",
        "blockers": blockers,
        "runtime_root": str(RUNTIME_ROOT.resolve()),
        "expected_routes": sorted(expected_routes),
        "missing_routes": missing_routes,
        "route_count": len(route_paths),
        "processes": processes,
        "stale_or_missing_pids": stale_or_missing_pids,
        "database": database_health,
        "snapshot_policy": snapshot_policy,
        "job_summary": store.system_job_summary(),
        "system_job_scheduler": scheduler,
        "recommendations": _platform_doctor_recommendations(
            missing_routes=missing_routes,
            snapshot_policy=snapshot_policy,
            stale_or_missing_pids=stale_or_missing_pids,
            scheduler=scheduler,
        ),
    }


def _platform_doctor_recommendations(
    *,
    missing_routes: list[str],
    snapshot_policy: dict[str, Any],
    stale_or_missing_pids: list[str],
    scheduler: dict[str, Any],
) -> list[dict[str, Any]]:
    recommendations = []
    if missing_routes:
        recommendations.append(
            {
                "action": "RESTART_AUTOALPHA_SERVICE",
                "reason": "当前进程缺少预期 API 路由，可能运行了旧版本代码。",
                "command": "./stop-services.sh && ./start-services.sh --no-resume",
            }
        )
    if stale_or_missing_pids:
        recommendations.append(
            {
                "action": "REFRESH_PID_FILES",
                "reason": "PID 文件缺失或失效，服务管理脚本无法可靠判断运行态。",
                "affected_services": stale_or_missing_pids,
            }
        )
    for key in snapshot_policy.get("missing_keys", []) + snapshot_policy.get("stale_keys", []):
        row = next(
            (item for item in snapshot_policy.get("rows", []) if item["key"] == key),
            None,
        )
        if row and row.get("refresh_job_type"):
            recommendations.append(
                {
                    "action": "QUEUE_MATERIALIZED_SNAPSHOT_REFRESH",
                    "snapshot_key": key,
                    "job_type": row["refresh_job_type"],
                    "endpoint": row["refresh_endpoint"],
                }
            )
    if scheduler.get("enabled") and scheduler.get("status") == "failed":
        recommendations.append(
            {
                "action": "RESTART_SYSTEM_JOB_SCHEDULER",
                "reason": scheduler.get("failure") or "scheduler task is not alive",
            }
        )
    return recommendations


@app.get("/health")
async def health() -> dict[str, Any]:
    try:
        state = store.state()
        job_summary = store.system_job_summary()
        runtime = _runtime_database_health()
    except Exception as error:
        return {"status": "error", "database": "unavailable", "error": str(error)}
    return {
        "status": "ok",
        "database": "ok",
        "runtime": runtime,
        "state": state.get("state"),
        "phase": state.get("phase"),
        "jobs": job_summary,
        "system_job_scheduler": _system_job_scheduler_status(),
        "service": "autoalpha",
    }


@app.get("/ready")
async def ready() -> dict[str, Any]:
    settings = store.settings()
    data_path = Path(settings.get("data_path", PROJECT_ROOT.parent / "data")).expanduser()
    blockers: list[str] = []
    workspace_summary: dict[str, Any] = {"data_path": str(data_path)}
    data_capability_matrix: dict[str, Any]
    try:
        workspace = inspect_data_workspace(data_path)
        execution_basis = inspect_execution_data_basis(Path(workspace.panel_path))
        workspace_summary.update(
            {
                "price_research_ready": workspace.price_research_ready,
                "institutional_pit_ready": workspace.institutional_pit_ready,
                "first_trade_date": workspace.first_trade_date,
                "last_trade_date": workspace.last_trade_date,
                "execution_basis": execution_basis.to_dict(),
            }
        )
        production_data = _production_data_readiness(
            institutional_pit_ready=workspace.institutional_pit_ready,
            execution_basis=execution_basis.to_dict(),
        )
        data_capability_matrix = build_data_capability_matrix(
            workspace=workspace.to_dict(),
            execution_basis=execution_basis.to_dict(),
        )
        if not workspace.price_research_ready:
            blockers.append("price_research_workspace_not_ready")
    except Exception as error:
        workspace_summary["error"] = str(error)
        production_data = {
            "production_trading_allowed": False,
            "strict_pit_capital_ledger_ready": False,
            "non_pit_proxy_allowed": False,
            "blockers": [f"data_workspace_unavailable: {error}"],
            "proxy_blockers": [f"data_workspace_unavailable: {error}"],
            "policy": "research_service_may_degrade_but_production_trading_is_blocked",
        }
        data_capability_matrix = build_data_capability_matrix(
            workspace=None,
            execution_basis=None,
            workspace_error=str(error),
        )
        blockers.append("data_workspace_unavailable")
    if research_manager.alive and not vault.configured():
        blockers.append("research_running_without_configured_llm_credentials")
    scheduler_status = _system_job_scheduler_status()
    if scheduler_status["enabled"] and scheduler_status["status"] == "failed":
        blockers.append("system_job_scheduler_failed")
    runtime_database = _runtime_database_health()
    snapshot_policy = _materialized_snapshot_policy(
        runtime_database.get("materialized_snapshots") or None
    )
    backend_blockers = list((runtime_database.get("backend") or {}).get("blockers") or [])
    if backend_blockers:
        blockers.extend(f"database_backend: {blocker}" for blocker in backend_blockers)
    if runtime_database["system_jobs"]["expired_running_count"]:
        blockers.append("system_jobs_have_expired_leases")
    metric_convention = check_long_only_metric_convention(PROJECT_ROOT)
    return {
        "status": "ready" if not blockers else "degraded",
        "blockers": blockers,
        "workspace": workspace_summary,
        "production_data": production_data,
        "data_capability_matrix": data_capability_matrix,
        "research_alive": research_manager.alive,
        "data_sync": data_sync_worker.status(),
        "runtime": runtime_database,
        "snapshot_policy": snapshot_policy,
        "system_job_scheduler": scheduler_status,
        "metric_convention": metric_convention,
    }


def _production_data_readiness(
    *, institutional_pit_ready: bool, execution_basis: dict[str, Any]
) -> dict[str, Any]:
    strict_blockers = []
    if not institutional_pit_ready:
        strict_blockers.append("institutional_pit_workspace_not_ready")
    if not bool(execution_basis.get("capital_ledger_ready")):
        strict_blockers.extend(
            str(blocker) for blocker in execution_basis.get("blockers", ())
        )
    proxy_blockers = [str(blocker) for blocker in execution_basis.get("proxy_blockers", ())]
    return {
        "production_trading_allowed": not strict_blockers,
        "strict_pit_capital_ledger_ready": (
            institutional_pit_ready and bool(execution_basis.get("capital_ledger_ready"))
        ),
        "non_pit_proxy_allowed": bool(execution_basis.get("capital_ledger_proxy_ready")),
        "blockers": list(dict.fromkeys(strict_blockers)),
        "proxy_blockers": list(dict.fromkeys(proxy_blockers)),
        "policy": (
            "strict_pit_required_for_production; "
            "non_pit_proxy_is_research_and_paper_trading_only"
        ),
    }


def _materialized_response_fields(cached: dict[str, Any]) -> dict[str, Any]:
    return {
        "materialized": True,
        "materialized_at": cached["updated_at"],
        "materialized_fingerprint": cached["fingerprint"],
        "materialized_source": cached.get("source") or "unknown",
        "materialized_status": cached.get("status") or "READY",
        "cache_state": cached.get("cache_state")
        or {
            "status": "UNKNOWN",
            "stale": False,
            "age_seconds": None,
            "expires_at": cached.get("expires_at"),
            "source": cached.get("source") or "unknown",
            "snapshot_status": cached.get("status") or "READY",
        },
    }


def _authorized(session: Annotated[str | None, Cookie(alias="autoalpha_session")] = None) -> None:
    required = os.getenv("AUTOALPHA_SERVICE_TOKEN")
    expected = _session_value(required) if required else "local"
    if required and (not session or not hmac.compare_digest(session, expected)):
        raise HTTPException(status_code=401, detail="Authentication required")


@app.get("/api/platform/doctor", dependencies=[Depends(_authorized)])
async def platform_doctor() -> dict[str, Any]:
    try:
        return _platform_doctor_snapshot()
    except Exception as error:
        return {
            "protocol": "AUTOALPHA_PLATFORM_DOCTOR_V1",
            "status": "ERROR",
            "blockers": ["doctor_snapshot_failed"],
            "error": f"{type(error).__name__}: {error}",
        }


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(PACKAGE_ROOT / "static/index.html")


@app.get("/research/{task_id}", include_in_schema=False)
async def research_workspace_page(task_id: str) -> FileResponse:
    if store.research_task(task_id) is None:
        raise HTTPException(status_code=404, detail="Research task not found")
    return FileResponse(PACKAGE_ROOT / "static/index.html")


@app.get("/factors", include_in_schema=False)
async def factors_page() -> FileResponse:
    return FileResponse(PACKAGE_ROOT / "static/factors.html")


@app.get("/backtest", include_in_schema=False)
async def backtest_page() -> FileResponse:
    return FileResponse(PACKAGE_ROOT / "static/backtest.html")


@app.get("/screener", include_in_schema=False)
async def screener_page() -> FileResponse:
    return FileResponse(PACKAGE_ROOT / "static/screener.html")


@app.get("/paper-trading", include_in_schema=False)
async def paper_trading_page() -> FileResponse:
    return FileResponse(PACKAGE_ROOT / "static/paper_trading.html")


@app.get("/strategies", include_in_schema=False)
async def formal_strategies_page() -> FileResponse:
    return FileResponse(PACKAGE_ROOT / "static/formal_strategies.html")


@app.get("/data", include_in_schema=False)
async def data_center_page() -> FileResponse:
    return FileResponse(PACKAGE_ROOT / "static/data_center.html")


@app.get("/jobs", include_in_schema=False)
async def jobs_page() -> FileResponse:
    return FileResponse(PACKAGE_ROOT / "static/jobs.html")


@app.get("/research-tasks", include_in_schema=False)
async def research_tasks_page() -> FileResponse:
    return FileResponse(PACKAGE_ROOT / "static/research_tasks.html")


@app.get("/research-tasks/{task_id}", include_in_schema=False)
async def research_task_detail_page(task_id: str) -> FileResponse:
    return FileResponse(PACKAGE_ROOT / "static/research_tasks.html")


@app.get("/llm-team", include_in_schema=False)
async def llm_team_page() -> FileResponse:
    return FileResponse(PACKAGE_ROOT / "static/llm_team.html")


@app.get("/settings", include_in_schema=False)
async def settings_page() -> FileResponse:
    return FileResponse(PACKAGE_ROOT / "static/settings.html")


@app.get("/guide", include_in_schema=False)
async def system_guide_page() -> FileResponse:
    return FileResponse(PACKAGE_ROOT / "static/system_guide.html")


@app.post("/api/session")
async def create_session(payload: SessionRequest, response: Response) -> dict[str, bool]:
    required = os.getenv("AUTOALPHA_SERVICE_TOKEN")
    if required and not hmac.compare_digest(payload.token, required):
        raise HTTPException(status_code=401, detail="Invalid service token")
    response.set_cookie(
        "autoalpha_session",
        _session_value(required) if required else "local",
        httponly=True,
        secure=os.getenv("AUTOALPHA_SECURE_COOKIE", "0") == "1",
        samesite="strict",
        max_age=86400,
    )
    return {"authenticated": True}


@app.get("/api/snapshot", dependencies=[Depends(_authorized)])
async def snapshot() -> dict[str, Any]:
    return _research_workspace_snapshot("legacy-ashare")


def _research_workspace_snapshot(task_id: str) -> dict[str, Any]:
    task, research_config, generation_id, generation = _task_research_context(task_id)
    task_view = _research_task_view(task)
    settings = store.settings()
    run_id = task_view.get("run_id")
    data_workspace: dict[str, Any] | None = None
    data_error: str | None = None
    if task.get("data_path"):
        try:
            workspace = inspect_data_workspace(Path(task["data_path"]))
            data_workspace = workspace.to_dict()
            data_workspace["research_data_contract"] = build_research_data_capabilities(workspace)
            data_workspace["execution_basis"] = inspect_execution_data_basis(
                Path(workspace.panel_path)
            ).to_dict()
        except Exception as error:
            data_error = f"{type(error).__name__}: {error}"
    campaign_history = store.direction_campaign_history(generation_id, limit=12)
    active_campaign = next((item for item in campaign_history if item["status"] == "ACTIVE"), None)
    portfolio = store.active_portfolio(run_id=run_id) if run_id else None
    if portfolio:
        portfolio_protocol = portfolio["metrics"].get("portfolio_evaluation_protocol")
        portfolio["protocol_stale"] = (
            portfolio_protocol != research_config.governance.protocol_version
        )
        portfolio["current_protocol"] = research_config.governance.protocol_version
    return {
        "state": (
            store.state() if task_id == "legacy-ashare" else store.research_task_state(task_id)
        ),
        "research_task": task_view,
        "settings": {
            **settings,
            "service_variant": os.getenv("AUTOALPHA_VARIANT", "STANDARD"),
            "api_key_configured": vault.configured(),
            "tushare_token_configured": data_sync_worker.token_configured(),
            "service_token_required": bool(os.getenv("AUTOALPHA_SERVICE_TOKEN")),
        },
        "metrics": store.metric_history(run_id=run_id) if run_id else [],
        "iteration_stats": store.iteration_stats(run_id=run_id)
        if run_id
        else {
            "total": 0,
            "completed": 0,
            "failed": 0,
            "running": 0,
            "success_rate": 0.0,
        },
        "events": store.events(run_id=run_id, task_id=task_id, limit=200),
        "iterations": store.iteration_history(limit=100, run_id=run_id) if run_id else [],
        "factor_pool": [
            factor
            for factor in store.factor_pool(limit=5000)
            if factor.get("source_task_id") == task_id
        ][:500],
        "portfolio": portfolio,
        "portfolio_history": (store.portfolio_history(limit=100, run_id=run_id) if run_id else []),
        "research_protocol": {
            "generation": research_config.generation,
            "version": research_config.governance.protocol_version,
            "exploration": {
                "start": research_config.splits.train.start.isoformat(),
                "end": research_config.splits.train.end.isoformat(),
            },
            "walk_forward": {
                "train_years": research_config.walk_forward.train_years,
                "validation_years": research_config.walk_forward.validation_years,
                "first_validation_year": (research_config.walk_forward.first_validation_year),
                "last_validation_year": research_config.walk_forward.last_validation_year,
                "minimum_folds": research_config.walk_forward.minimum_folds,
            },
            "holdout": {
                "start": research_config.splits.test.start.isoformat(),
                "end": research_config.splits.test.end.isoformat(),
                "feedback": "categorical verdict and evidence hash only",
            },
            "portfolio": {
                "holding_period_days": research_config.portfolio.holding_period_days,
                "target_gross_exposure": (research_config.strategy_evaluation.gross_exposure),
                "initial_cash_cny": (research_config.strategy_evaluation.initial_cash_cny),
                "maximum_positions": (research_config.strategy_evaluation.maximum_positions),
                "rebalance_schedule": (research_config.strategy_evaluation.rebalance_schedule),
                "execution_protocol": (research_config.strategy_evaluation.engine_protocol),
                "execution_data_mode": (research_config.strategy_evaluation.execution_data_mode),
            },
            "evaluation_layers": {
                "strategy_promotion": {
                    "portfolio_mode": "long_only",
                    "purpose": "primary_single_factor_and_portfolio_governance",
                    "primary": True,
                    "investable": False,
                    "protocol": research_config.strategy_evaluation.engine_protocol,
                    "limitations": list(
                        data_workspace.get("execution_basis", {}).get("proxy_blockers", [])
                        if data_workspace
                        else []
                    ),
                },
                "alpha_diagnostic": {
                    "portfolio_mode": "market_neutral_long_short",
                    "purpose": "secondary_information_diagnostic_only",
                    "primary": False,
                    "investable": False,
                },
            },
        },
        "research_generation": generation,
        "adaptive_direction": {
            "config": {
                "enabled": research_config.adaptive_direction.enabled,
                "maximum_attempts_per_campaign": (
                    research_config.adaptive_direction.maximum_attempts_per_campaign
                ),
                "early_stop_consecutive_misses": (
                    research_config.adaptive_direction.early_stop_consecutive_misses
                ),
                "recent_candidate_window": (
                    research_config.adaptive_direction.recent_candidate_window
                ),
                "cooldown_campaigns": research_config.adaptive_direction.cooldown_campaigns,
                "feedback_source": "PUBLIC_RESEARCH_ONLY",
            },
            "active": active_campaign,
            "latest": campaign_history[0] if campaign_history else None,
            "history": campaign_history,
        },
        "blind_evaluations": store.blind_evaluations(limit=100, generation_id=generation_id),
        "memories": store.recent_memories(limit=20, run_id=run_id) if run_id else [],
        "memory_count": (len(store.recent_memories(limit=10_000, run_id=run_id)) if run_id else 0),
        "worker_alive": research_manager.worker_alive(task_id),
        "any_worker_alive": research_manager.alive,
        "active_research_task_ids": research_manager.active_task_ids(),
        "research_tasks": store.research_tasks(),
        "data_sync": data_sync_worker.status(),
        "data_workspace": data_workspace,
        "data_error": data_error,
    }


@app.get(
    "/api/research-tasks/{task_id}/workspace",
    dependencies=[Depends(_authorized)],
)
async def research_task_workspace(task_id: str) -> dict[str, Any]:
    return _research_workspace_snapshot(task_id)


@app.get("/api/favorites", dependencies=[Depends(_authorized)])
async def favorite_index(entity_type: FavoriteEntityType | None = None) -> dict[str, Any]:
    records = store.favorites(entity_type=entity_type, limit=5000)
    return {
        "favorites": records,
        "entity_type": entity_type,
        "count": len(records),
    }


@app.put(
    "/api/favorites/{entity_type}/{entity_id}",
    dependencies=[Depends(_authorized)],
)
async def update_favorite(
    entity_type: FavoriteEntityType,
    entity_id: str,
    payload: FavoriteRequest,
) -> dict[str, Any]:
    try:
        record = store.set_favorite(
            entity_type,
            entity_id,
            favorite=payload.favorite,
            label=payload.label,
            context=payload.context,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    store.append_event(
        "action",
        "FAVORITE_ADDED" if payload.favorite else "FAVORITE_REMOVED",
        "收藏已更新",
        f"{entity_type}:{entity_id} 已{'加入' if payload.favorite else '移出'}收藏。",
        payload={
            "entity_type": entity_type,
            "entity_id": entity_id,
            "favorite": payload.favorite,
            "label": payload.label,
        },
    )
    return {
        "favorite": payload.favorite,
        "record": record,
    }


@app.get("/api/llm-team", dependencies=[Depends(_authorized)])
async def llm_team_snapshot(task_id: str | None = None) -> dict[str, Any]:
    if task_id and store.research_task(task_id) is None:
        raise HTTPException(status_code=404, detail="Research task not found")
    artifacts = store.llm_role_artifacts(task_id=task_id, limit=300)
    favorite_artifacts = store.favorite_ids("llm_artifact")
    for artifact in artifacts:
        artifact["favorite"] = str(artifact["id"]) in favorite_artifacts
    return {
        "variant": os.getenv("AUTOALPHA_VARIANT", "FULL LLM"),
        "enabled": store.settings().get("full_llm_enabled", "true") == "true",
        "task_id": task_id,
        "roles": role_catalog(),
        "domain_matrix": summarize_research_team_domains(artifacts),
        "summary": store.llm_role_summary(task_id=task_id),
        "artifacts": artifacts,
        "knowledge": store.factor_knowledge_catalog(task_id=task_id, limit=500),
        "tasks": [
            {
                "task_id": task["task_id"],
                "name": task["name"],
                "market": task["market"],
                "state": task.get("state"),
            }
            for task in store.research_tasks()
        ],
        "decision_authority": "ADVISORY_ONLY",
        "feedback_policy": "CATEGORICAL_PUBLIC_ONLY_NO_EXACT_METRICS",
    }


@app.get("/api/factors/{factor_id}/intelligence", dependencies=[Depends(_authorized)])
async def factor_intelligence(factor_id: str) -> dict[str, Any]:
    factor = store.factor_pool_record(factor_id)
    if factor is None:
        raise HTTPException(status_code=404, detail="Factor not found")
    return {
        "factor_id": factor_id,
        "knowledge": store.factor_knowledge(factor_id),
        "role_artifacts": store.llm_role_artifacts(candidate_id=factor_id, limit=100),
        "decision_authority": "ADVISORY_ONLY",
    }


@app.get("/api/strategy-bus", dependencies=[Depends(_authorized)])
async def strategy_bus_snapshot(sync: bool = False) -> dict[str, Any]:
    cached = store.materialized_snapshot("strategy_bus")
    if cached:
        snapshot = {
            **cached["payload"],
            **_materialized_response_fields(cached),
            "read_only": True,
        }
        if sync:
            snapshot["sync_ignored"] = True
            snapshot["sync_hint"] = "Use POST /api/strategy-bus/sync to refresh materialized state."
        return snapshot
    snapshot = build_strategy_bus_snapshot(
        store,
        autocombine_store=combine_store,
        quantcombine_store=quant_store,
        behavior_snapshot=load_behavior_snapshot(RUNTIME_ROOT / "factor-behavior"),
        sync=False,
    )
    snapshot["materialized"] = False
    snapshot["read_only"] = True
    if sync:
        snapshot["sync_ignored"] = True
        snapshot["sync_hint"] = "Use POST /api/strategy-bus/sync to refresh materialized state."
    return snapshot


@app.post("/api/strategy-bus/sync", dependencies=[Depends(_authorized)])
async def sync_strategy_bus_snapshot(
    request: StrategyBusSyncRequest | None = None,
) -> dict[str, Any]:
    request = request or StrategyBusSyncRequest()
    if not request.run_now:
        existing = None if request.force_new else _existing_system_job_by_type(
            "strategy_bus_sync",
            queue=request.queue,
            statuses=("QUEUED", "RUNNING"),
        )
        if existing is not None:
            return {
                "queued": True,
                "deduplicated": True,
                "job": existing,
                "message": "Strategy bus sync is already queued or running.",
            }
        job = store.enqueue_system_job(
            job_id=f"job-strategy-bus-{uuid.uuid4().hex[:12]}",
            queue=request.queue,
            job_type="strategy_bus_sync",
            payload={"source": "api.strategy_bus_sync"},
            priority=request.priority,
            resource_group="strategy_bus",
            max_workers=1,
            progress_total=1,
        )
        store.append_event(
            "audit",
            "STRATEGY_BUS_SYNC_QUEUED",
            "策略实验总线同步已进入 Job Center",
            f"{job['job_id']} · queue={job['queue']} · priority={job['priority']}",
            payload={"job_id": job["job_id"], "job_type": job["job_type"]},
        )
        return {
            "queued": True,
            "deduplicated": False,
            "job": job,
            "message": "Strategy bus sync queued in Job Center.",
        }
    snapshot = build_strategy_bus_snapshot(
        store,
        autocombine_store=combine_store,
        quantcombine_store=quant_store,
        behavior_snapshot=load_behavior_snapshot(RUNTIME_ROOT / "factor-behavior"),
        sync=True,
    )
    cached = store.upsert_materialized_snapshot(
        "strategy_bus",
        snapshot,
        ttl_seconds=SNAPSHOT_TTLS["strategy_bus"],
        source="api.run_now",
    )
    store.append_event(
        "audit",
        "STRATEGY_BUS_SYNCED",
        "策略实验总线已同步",
        "因子候选、组合候选、策略版本和模拟组合索引已刷新。",
        payload={"summary": snapshot["summary"]},
    )
    return {
        **snapshot,
        **_materialized_response_fields(cached),
        "run_now": True,
    }


def _existing_system_job_by_type(
    job_type: str,
    *,
    queue: str,
    statuses: tuple[str, ...],
) -> dict[str, Any] | None:
    for status in statuses:
        for job in store.system_jobs(queue=queue, status=status, limit=200):
            if str(job.get("job_type")) == job_type:
                return job
    return None


@app.get("/api/factor-knowledge-map", dependencies=[Depends(_authorized)])
async def factor_knowledge_map_snapshot(refresh: bool = False) -> dict[str, Any]:
    cached = store.materialized_snapshot("factor_knowledge_map")
    if cached and not refresh:
        return {
            **cached["payload"],
            **_materialized_response_fields(cached),
        }
    if refresh:
        return {
            "research_map_protocol": "FACTOR_KNOWLEDGE_RESEARCH_MAP_V2",
            "protocol": "MATERIALIZED_FACTOR_KNOWLEDGE_MAP_V1",
            "materialized": False,
            "read_only": True,
            "cache_status": "MISSING" if cached is None else cached["cache_state"]["status"],
            "refresh_ignored": True,
            "refresh_hint": (
                "Use POST /api/factor-knowledge-map/sync or enqueue "
                "factor_knowledge_map_sync in Job Center."
            ),
        }
    snapshot = factor_knowledge_map(
        store,
        behavior_snapshot=load_behavior_snapshot(RUNTIME_ROOT / "factor-behavior"),
    )
    snapshot["materialized"] = False
    snapshot["read_only"] = True
    return snapshot


@app.post("/api/factor-knowledge-map/sync", dependencies=[Depends(_authorized)])
async def sync_factor_knowledge_map_snapshot(
    request: FactorKnowledgeMapSyncRequest | None = None,
) -> dict[str, Any]:
    request = request or FactorKnowledgeMapSyncRequest()
    if not request.run_now:
        existing = None if request.force_new else _existing_system_job_by_type(
            "factor_knowledge_map_sync",
            queue=request.queue,
            statuses=("QUEUED", "RUNNING"),
        )
        if existing is not None:
            return {
                "queued": True,
                "deduplicated": True,
                "job": existing,
                "message": "Factor knowledge map sync is already queued or running.",
            }
        job = store.enqueue_system_job(
            job_id=f"job-factor-knowledge-{uuid.uuid4().hex[:12]}",
            queue=request.queue,
            job_type="factor_knowledge_map_sync",
            payload={"source": "api.factor_knowledge_map_sync"},
            priority=request.priority,
            resource_group="sqlite-writer",
            max_workers=1,
            progress_total=1,
        )
        store.append_event(
            "audit",
            "FACTOR_KNOWLEDGE_MAP_SYNC_QUEUED",
            "因子知识地图同步已进入 Job Center",
            f"{job['job_id']} · queue={job['queue']} · priority={job['priority']}",
            payload={"job_id": job["job_id"], "job_type": job["job_type"]},
        )
        return {
            "queued": True,
            "deduplicated": False,
            "job": job,
            "message": "Factor knowledge map sync queued in Job Center.",
        }
    snapshot = factor_knowledge_map(
        store,
        behavior_snapshot=load_behavior_snapshot(RUNTIME_ROOT / "factor-behavior"),
    )
    snapshot = {
        **snapshot,
        "protocol": "MATERIALIZED_FACTOR_KNOWLEDGE_MAP_V1",
        "processed_count": snapshot.get("factor_count", 0),
    }
    cached = store.upsert_materialized_snapshot(
        "factor_knowledge_map",
        snapshot,
        ttl_seconds=SNAPSHOT_TTLS["factor_knowledge_map"],
        source="api.run_now",
    )
    store.append_event(
        "audit",
        "FACTOR_KNOWLEDGE_MAP_SYNCED",
        "因子知识地图已同步",
        "行为簇、机制簇、参数家族和年度画像已刷新。",
        payload={"factor_count": snapshot.get("factor_count", 0)},
    )
    return {
        **cached["payload"],
        **_materialized_response_fields(cached),
        "run_now": True,
    }


@app.get("/api/strategy-library", dependencies=[Depends(_authorized)])
async def strategy_library_snapshot() -> dict[str, Any]:
    return formal_strategy_library(store)


@app.get("/api/strategy-production-funnel", dependencies=[Depends(_authorized)])
async def strategy_production_funnel_snapshot() -> dict[str, Any]:
    return strategy_production_funnel(store)


@app.get(
    "/api/strategy-experiments/{experiment_id}/lineage",
    dependencies=[Depends(_authorized)],
)
async def strategy_experiment_lineage_snapshot(
    experiment_id: str, depth: int = 2
) -> dict[str, Any]:
    try:
        return strategy_experiment_lineage(store, experiment_id, depth=depth)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get(
    "/api/strategy-library/{strategy_uid}/versions/{version}/readiness",
    dependencies=[Depends(_authorized)],
)
async def formal_strategy_readiness(strategy_uid: str, version: int) -> dict[str, Any]:
    try:
        return strategy_lifecycle_readiness(store, strategy_uid, version)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get(
    "/api/strategy-library/{strategy_uid}/versions/{version}/execution-package",
    dependencies=[Depends(_authorized)],
)
async def formal_strategy_execution_package(strategy_uid: str, version: int) -> dict[str, Any]:
    try:
        return strategy_execution_package(store, strategy_uid, version)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.post(
    "/api/strategy-library/{strategy_uid}/versions/{version}/paper-portfolio",
    dependencies=[Depends(_authorized)],
)
async def create_paper_portfolio_from_strategy(
    strategy_uid: str,
    version: int,
    payload: StrategyPaperPortfolioRequest,
) -> dict[str, Any]:
    if data_sync_worker.alive:
        raise HTTPException(status_code=409, detail="Wait for the market-data refresh to finish")
    try:
        package = strategy_execution_package(store, strategy_uid, version)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    contract = package.get("paper_trading_contract") or {}
    seed = contract.get("paper_portfolio_seed") or {}
    factor_ids = list(seed.get("factor_ids") or [])
    weights = list(seed.get("weights") or [])
    if not factor_ids or len(factor_ids) != len(weights):
        raise HTTPException(status_code=422, detail="Strategy paper seed is incomplete")
    records = []
    missing = []
    for factor_id in factor_ids:
        record = store.factor_pool_record(str(factor_id))
        if record is None:
            missing.append(str(factor_id))
        else:
            records.append(record)
    if missing:
        raise HTTPException(status_code=404, detail=f"Factors not found: {', '.join(missing)}")
    data_path, market, task_ids = _factor_source_context(records)
    try:
        result = await asyncio.to_thread(
            PaperTradingEngine(store, data_path).create,
            PaperStrategySpec(
                name=payload.name or str(seed.get("name") or package["name"]),
                factor_ids=[str(item) for item in factor_ids],
                weights=[float(item) for item in weights],
                initial_cash_cny=payload.initial_cash_cny,
                selection_count=int(payload.selection_count or seed.get("selection_count") or 30),
                gross_exposure=float(
                    payload.gross_exposure
                    if payload.gross_exposure is not None
                    else seed.get("gross_exposure", 0.9)
                ),
                slippage_bps_each_side=float(
                    payload.slippage_bps_each_side
                    if payload.slippage_bps_each_side is not None
                    else seed.get("slippage_bps_each_side", 5.0)
                ),
                as_of_date=payload.as_of_date,
                market=market,
                data_path=str(data_path),
                source_task_ids=tuple(task_ids),
            ),
        )
    except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    result["source_strategy"] = {
        "strategy_uid": strategy_uid,
        "version": version,
        "execution_package_protocol": contract.get("protocol"),
        "execution_protocol": contract.get("execution_protocol"),
    }
    _paper_event("PAPER_PORTFOLIO_CREATED_FROM_STRATEGY", "策略模拟组合已创建", result)
    return result


@app.get(
    "/api/strategy-library/{strategy_uid}/versions/{version}/release-dossier",
    dependencies=[Depends(_authorized)],
)
async def formal_strategy_release_dossier(strategy_uid: str, version: int) -> dict[str, Any]:
    try:
        return strategy_release_dossier(store, strategy_uid, version)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.post(
    "/api/strategy-library/{strategy_uid}/versions/{version}/release-dossier/export",
    dependencies=[Depends(_authorized)],
)
async def export_formal_strategy_release_dossier(
    strategy_uid: str, version: int
) -> dict[str, Any]:
    try:
        published = publish_strategy_release_dossier(
            store,
            RUNTIME_ROOT / "artifacts",
            strategy_uid,
            version,
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    artifact = published["artifact"]
    store.append_event(
        "delivery",
        "STRATEGY_RELEASE_DOSSIER_EXPORTED",
        "策略发布档案已导出",
        f"{strategy_uid} v{version} · {artifact['artifact_id']}",
        payload={
            "strategy_uid": strategy_uid,
            "version": version,
            "artifact_id": artifact["artifact_id"],
            "content_hash": artifact["content_hash"],
            "production_ready": artifact["metadata"].get("production_ready"),
            "release_decision": artifact["metadata"].get("release_decision"),
        },
    )
    return {
        **published,
        "download_url": f"/api/artifacts/{artifact['artifact_id']}/download",
    }


@app.get("/api/artifacts/{artifact_id}/download", dependencies=[Depends(_authorized)])
async def download_artifact(artifact_id: str) -> FileResponse:
    if not artifact_id.startswith("strategy-release-dossier-"):
        raise HTTPException(status_code=404, detail="Artifact download is not exposed")
    try:
        from autoalpha.operations.artifacts import ArtifactRegistry

        artifact = ArtifactRegistry(RUNTIME_ROOT / "artifacts").get(artifact_id)
    except (KeyError, RuntimeError) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    payload_path = (RUNTIME_ROOT / "artifacts" / artifact.payload_path).resolve()
    artifact_root = (RUNTIME_ROOT / "artifacts").resolve()
    if not payload_path.is_relative_to(artifact_root) or not payload_path.is_file():
        raise HTTPException(status_code=500, detail="Artifact payload is unavailable")
    return FileResponse(
        payload_path,
        media_type="application/json",
        filename=f"{artifact.artifact_id}.json",
    )


@app.get("/api/gate-funnel", dependencies=[Depends(_authorized)])
async def gate_funnel_snapshot(refresh: bool = False) -> dict[str, Any]:
    cached = store.materialized_snapshot("gate_funnel_diagnostics")
    if cached and not refresh:
        return {
            **cached["payload"],
            **_materialized_response_fields(cached),
        }
    snapshot = build_gate_funnel_diagnostics(combine_store, quant_store)
    if refresh:
        cached = store.upsert_materialized_snapshot(
            "gate_funnel_diagnostics",
            snapshot,
            ttl_seconds=SNAPSHOT_TTLS["gate_funnel_diagnostics"],
            source="api.refresh",
        )
        return {
            **cached["payload"],
            **_materialized_response_fields(cached),
        }
    return {**snapshot, "materialized": False}


@app.get("/api/gate-feedback", dependencies=[Depends(_authorized)])
async def gate_feedback_snapshot(refresh: bool = False) -> dict[str, Any]:
    cached = store.materialized_snapshot("gate_feedback_policy")
    if cached and not refresh:
        return {
            **cached["payload"],
            **_materialized_response_fields(cached),
        }
    if refresh:
        gate_snapshot = build_gate_funnel_diagnostics(combine_store, quant_store)
        store.upsert_materialized_snapshot(
            "gate_funnel_diagnostics",
            gate_snapshot,
            ttl_seconds=SNAPSHOT_TTLS["gate_funnel_diagnostics"],
            source="api.refresh",
        )
    policy = gate_feedback_policy(store)
    if refresh:
        cached = store.upsert_materialized_snapshot(
            "gate_feedback_policy",
            policy,
            ttl_seconds=SNAPSHOT_TTLS["gate_feedback_policy"],
            source="api.refresh",
        )
        return {
            **cached["payload"],
            **_materialized_response_fields(cached),
        }
    return {**policy, "materialized": False}


@app.post("/api/gate-feedback/seed-quant-repair", dependencies=[Depends(_authorized)])
async def seed_gate_feedback_quant_repair(
    payload: GateFeedbackRepairSeedRequest,
) -> dict[str, Any]:
    gate_snapshot = build_gate_funnel_diagnostics(combine_store, quant_store)
    store.upsert_materialized_snapshot("gate_funnel_diagnostics", gate_snapshot)
    policy = gate_feedback_policy(store)
    cached = store.upsert_materialized_snapshot("gate_feedback_policy", policy)
    policy_payload = cached["payload"]
    if not policy_payload.get("active"):
        return {
            "status": "SKIPPED",
            "reason": "NO_ACTIVE_GATE_FEEDBACK_ACTIONS",
            "policy": policy_payload,
        }
    source_fingerprint = _stable_gate_feedback_seed_fingerprint(policy_payload)
    job_id = f"job-gate-repair-{source_fingerprint[:16]}"
    existing = None
    with suppress(KeyError):
        existing = store.system_job(job_id)
    if existing:
        return {
            "status": "EXISTING",
            "job": existing,
            "policy": policy_payload,
            **_repair_seed_task_reference(existing),
        }
    job_payload: dict[str, Any] = {
        "name": payload.name,
        "notes": payload.notes,
        "feedback_source_fingerprint": source_fingerprint,
        "feedback_actions": policy_payload.get("action_ids") or [],
        "feedback_recommendations": policy_payload.get("recommendations") or [],
        "auto_start_task": payload.auto_start_task,
    }
    if payload.data_path:
        job_payload["data_path"] = payload.data_path
    if payload.protocol:
        job_payload["protocol"] = payload.protocol
    try:
        job = store.enqueue_system_job(
            job_id=job_id,
            queue=payload.queue,
            job_type="quantcombine_repair_task_seed",
            payload=job_payload,
            priority=payload.priority,
            resource_group="sqlite-writer",
            max_workers=1,
            progress_total=1,
            max_attempts=2,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    store.append_event(
        "action",
        "GATE_FEEDBACK_REPAIR_JOB_ENQUEUED",
        "门禁反馈修复任务已入队",
        f"{job_id} · QuantCombine repair seed · {source_fingerprint[:12]}",
        payload={
            "job_id": job_id,
            "source_fingerprint": source_fingerprint,
            "action_ids": policy_payload.get("action_ids") or [],
        },
    )
    result = None
    if payload.run_immediately:
        result = await asyncio.to_thread(system_job_runner.run_next, queue=payload.queue)
    return {
        "status": "ENQUEUED",
        "job": job,
        "policy": policy_payload,
        "run_result": result,
        **_repair_seed_task_reference(job, result),
    }


def _repair_seed_task_reference(
    job: dict[str, Any], run_result: dict[str, Any] | None = None
) -> dict[str, Any]:
    result = run_result.get("job", {}).get("result") if isinstance(run_result, dict) else None
    if not isinstance(result, dict):
        result = job.get("result") if isinstance(job.get("result"), dict) else {}
    task_id = result.get("task_id")
    if not task_id:
        return {}
    task_url = result.get("task_url") or f"http://127.0.0.1:8889/tasks/{task_id}"
    return {"repair_task_id": task_id, "repair_task_url": task_url}


def _stable_gate_feedback_seed_fingerprint(policy_payload: dict[str, Any]) -> str:
    stable_payload = {
        "protocol": policy_payload.get("protocol"),
        "action_ids": sorted(policy_payload.get("action_ids") or []),
        "profile_override": policy_payload.get("profile_override"),
        "root_cause_intensity": policy_payload.get("root_cause_intensity") or {},
        "recommendations": [
            {
                "action": item.get("action"),
                "root_cause": item.get("root_cause"),
                "suggested_values": item.get("suggested_values") or {},
            }
            for item in policy_payload.get("recommendations") or []
            if isinstance(item, dict)
        ],
    }
    return hashlib.sha256(
        json.dumps(stable_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


@app.post("/api/strategy-library/seed-candidates", dependencies=[Depends(_authorized)])
async def seed_strategy_library_candidates(
    payload: StrategyLibrarySeedJobRequest,
) -> dict[str, Any]:
    allowed = set(payload.candidate_classes)
    candidates = [
        item
        for item in strategy_promotion_candidates(store, limit=payload.limit * 5)
        if str(item.get("candidate_class")) in allowed
    ][: payload.limit]
    if not candidates:
        return {
            "status": "SKIPPED",
            "reason": "NO_PROMOTION_CANDIDATES",
            "candidate_classes": sorted(allowed),
        }
    fingerprint = _stable_strategy_seed_fingerprint(
        "strategy_library_seed", payload.model_dump(), candidates
    )
    job_id = f"job-strategy-seed-{fingerprint[:16]}"
    existing = _existing_system_job(job_id)
    if existing:
        return {"status": "EXISTING", "job": existing, "candidate_count": len(candidates)}
    job_payload = {
        "limit": payload.limit,
        "candidate_classes": sorted(allowed),
        "candidate_ids": [item["experiment_id"] for item in candidates],
    }
    job = store.enqueue_system_job(
        job_id=job_id,
        queue=payload.queue,
        job_type="strategy_library_seed",
        payload=job_payload,
        priority=payload.priority,
        resource_group="sqlite-writer",
        max_workers=1,
        progress_total=len(candidates),
        max_attempts=2,
    )
    store.append_event(
        "action",
        "STRATEGY_LIBRARY_SEED_JOB_ENQUEUED",
        "策略候选入库作业已入队",
        f"{job_id} · {len(candidates)} 个候选",
        payload={"job_id": job_id, "candidate_count": len(candidates)},
    )
    result = None
    if payload.run_immediately:
        result = await asyncio.to_thread(system_job_runner.run_next, queue=payload.queue)
    return {
        "status": "ENQUEUED",
        "job": job,
        "candidate_count": len(candidates),
        "run_result": result,
    }


@app.post("/api/strategy-library/freeze-ready", dependencies=[Depends(_authorized)])
async def freeze_public_validation_ready_strategies(
    payload: StrategyFreezeReadyJobRequest,
) -> dict[str, Any]:
    ready_strategies: list[dict[str, Any]] = []
    for strategy in store.formal_strategy_versions(limit=5000):
        if str(strategy.get("lifecycle")) != "RESEARCH":
            continue
        readiness = strategy_lifecycle_readiness(
            store, str(strategy["strategy_uid"]), int(strategy["version"])
        )
        if readiness.get("next_lifecycle") == "FROZEN" and readiness.get("ready"):
            ready_strategies.append(strategy)
        if len(ready_strategies) >= payload.limit:
            break
    if not ready_strategies:
        return {
            "status": "SKIPPED",
            "reason": "NO_PUBLIC_VALIDATION_READY_STRATEGIES",
            "ready_count": 0,
        }
    fingerprint = _stable_strategy_seed_fingerprint(
        "strategy_public_validation_freeze", payload.model_dump(), ready_strategies
    )
    job_id = f"job-strategy-freeze-{fingerprint[:16]}"
    existing = _existing_system_job(job_id)
    if existing:
        return {"status": "EXISTING", "job": existing, "ready_count": len(ready_strategies)}
    job = store.enqueue_system_job(
        job_id=job_id,
        queue=payload.queue,
        job_type="strategy_public_validation_freeze",
        payload={"limit": payload.limit},
        priority=payload.priority,
        resource_group="sqlite-writer",
        max_workers=1,
        progress_total=len(ready_strategies),
        max_attempts=2,
    )
    store.append_event(
        "action",
        "STRATEGY_FREEZE_READY_JOB_ENQUEUED",
        "可冻结策略推进作业已入队",
        f"{job_id} · {len(ready_strategies)} 个策略",
        payload={"job_id": job_id, "ready_count": len(ready_strategies)},
    )
    result = None
    if payload.run_immediately:
        result = await asyncio.to_thread(system_job_runner.run_next, queue=payload.queue)
    return {
        "status": "ENQUEUED",
        "job": job,
        "ready_count": len(ready_strategies),
        "run_result": result,
    }


def _existing_system_job(job_id: str) -> dict[str, Any] | None:
    with suppress(KeyError):
        return store.system_job(job_id)
    return None


def _stable_strategy_seed_fingerprint(
    action: str, payload: dict[str, Any], items: list[dict[str, Any]]
) -> str:
    stable_payload = {
        "action": action,
        "payload": {
            key: value
            for key, value in payload.items()
            if key not in {"run_immediately", "queue", "priority"}
        },
        "items": [
            item.get("experiment_id")
            or f"{item.get('strategy_uid')}@{item.get('version')}"
            for item in items
        ],
    }
    return hashlib.sha256(
        json.dumps(stable_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


@app.post("/api/strategy-library", dependencies=[Depends(_authorized)])
async def create_formal_strategy(payload: FormalStrategyCreateRequest) -> dict[str, Any]:
    try:
        strategy = create_formal_strategy_from_experiment(
            store,
            payload.experiment_id,
            name=payload.name,
            lifecycle=payload.lifecycle,
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    store.append_event(
        "delivery",
        "FORMAL_STRATEGY_VERSION_CREATED",
        "正式策略版本已创建",
        f"{strategy['strategy_uid']} VERSION {strategy['version']} 已进入策略库。",
        payload={
            "strategy_uid": strategy["strategy_uid"],
            "version": strategy["version"],
            "source_experiment_id": payload.experiment_id,
        },
    )
    return strategy


@app.post(
    "/api/strategy-library/{strategy_uid}/versions/{version}/promote",
    dependencies=[Depends(_authorized)],
)
async def promote_formal_strategy(
    strategy_uid: str,
    version: int,
    payload: FormalStrategyPromotionRequest,
) -> dict[str, Any]:
    try:
        strategy = promote_formal_strategy_lifecycle(
            store,
            strategy_uid,
            version,
            target_lifecycle=payload.target_lifecycle,
            evidence=payload.evidence,
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    store.append_event(
        "audit",
        "FORMAL_STRATEGY_LIFECYCLE_PROMOTED",
        "正式策略版本已晋级",
        f"{strategy_uid} VERSION {version} -> {payload.target_lifecycle}。",
        payload={
            "strategy_uid": strategy_uid,
            "version": version,
            "target_lifecycle": payload.target_lifecycle,
            "evidence_keys": sorted(payload.evidence),
        },
    )
    return strategy


@app.post(
    "/api/strategy-library/{strategy_uid}/versions/{version}/advance",
    dependencies=[Depends(_authorized)],
)
async def advance_formal_strategy(strategy_uid: str, version: int) -> dict[str, Any]:
    try:
        strategy = advance_formal_strategy_lifecycle(store, strategy_uid, version)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    store.append_event(
        "audit",
        "FORMAL_STRATEGY_LIFECYCLE_ADVANCED",
        "正式策略版本已按就绪证据晋级",
        f"{strategy_uid} VERSION {version} -> {strategy['lifecycle']}。",
        payload={
            "strategy_uid": strategy_uid,
            "version": version,
            "target_lifecycle": strategy["lifecycle"],
            "evidence_source": "strategy_lifecycle_readiness",
        },
    )
    return strategy


@app.post(
    "/api/strategy-library/{strategy_uid}/versions/{version}/approve",
    dependencies=[Depends(_authorized)],
)
async def approve_formal_strategy(
    strategy_uid: str,
    version: int,
    payload: FormalStrategyApprovalRequest,
) -> dict[str, Any]:
    try:
        strategy = approve_formal_strategy_transition(
            store,
            strategy_uid,
            version,
            approver=payload.approver,
            approval_type=payload.approval_type,
            notes=payload.notes,
            target_lifecycle=payload.target_lifecycle,
            evidence=payload.evidence,
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    approval = (strategy.get("evidence") or {}).get("human_approval") or {}
    store.append_event(
        "audit",
        "FORMAL_STRATEGY_TRANSITION_APPROVED",
        "正式策略晋级审批已记录",
        f"{strategy_uid} VERSION {version} -> {strategy['lifecycle']}。",
        payload={
            "strategy_uid": strategy_uid,
            "version": version,
            "target_lifecycle": strategy["lifecycle"],
            "approval_type": payload.approval_type,
            "approver": payload.approver,
            "approved_at": approval.get("approved_at"),
            "evidence_keys": sorted(payload.evidence),
        },
    )
    return strategy


@app.get("/api/jobs", dependencies=[Depends(_authorized)])
async def job_queue_snapshot(
    queue: str | None = None,
    status: str | None = None,
    limit: int = 500,
) -> dict[str, Any]:
    database_health = _runtime_database_health()
    snapshot_policy = _materialized_snapshot_policy(
        database_health.get("materialized_snapshots") or None
    )
    jobs = store.system_jobs(queue=queue, status=status, limit=limit)
    recent_logs = store.system_job_logs_for_jobs(
        [str(job["job_id"]) for job in jobs],
        limit_per_job=3,
    )
    return {
        "summary": store.system_job_summary(),
        "jobs": jobs,
        "recent_logs": recent_logs,
        "scheduler": _system_job_scheduler_status(),
        "database": database_health,
        "snapshot_policy": snapshot_policy,
        "resource_policy": {
            "database_mode": "WAL",
            "recommended_reader_mode": "read_only_snapshot",
            "claim_quota": "global_capacity_queue_capacity_and_resource_group_capacity",
            "single_writer_group": "use max_workers=1 for sqlite-writer jobs",
            "checkpoint_policy": "every_completed_chunk",
            "retry_policy": "bounded_attempts_with_structured_error",
            "expired_lease_policy": "recover_expired_system_jobs_before_claim",
            "job_log_policy": "structured_system_job_logs_by_job_id",
            "snapshot_policy": (
                "read materialized snapshots; refresh through Job Center or explicit POST"
            ),
            "supported_system_job_types": sorted(SUPPORTED_SYSTEM_JOB_TYPES),
        },
    }


@app.post("/api/jobs", dependencies=[Depends(_authorized)])
async def enqueue_system_job(payload: SystemJobRequest) -> dict[str, Any]:
    job_id = f"job-{uuid.uuid4().hex[:12]}"
    try:
        job = store.enqueue_system_job(
            job_id=job_id,
            queue=payload.queue,
            job_type=payload.job_type,
            payload=payload.payload,
            priority=payload.priority,
            resource_group=payload.resource_group,
            max_workers=payload.max_workers,
            progress_total=payload.progress_total,
            max_attempts=payload.max_attempts,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    store.append_event(
        "action",
        "SYSTEM_JOB_ENQUEUED",
        "系统作业已入队",
        f"{job_id} · {payload.job_type} 已进入 {payload.queue} 队列。",
        payload={"job_id": job_id, "queue": payload.queue, "job_type": payload.job_type},
    )
    return job


@app.get("/api/jobs/{job_id}/logs", dependencies=[Depends(_authorized)])
async def system_job_logs(job_id: str, limit: int = 200) -> dict[str, Any]:
    try:
        job = store.system_job(job_id)
        logs = store.system_job_logs(job_id, limit=limit)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return {"job": job, "logs": logs}


@app.post("/api/jobs/claim", dependencies=[Depends(_authorized)])
async def claim_system_job(payload: SystemJobClaimRequest) -> dict[str, Any]:
    recovered = store.recover_expired_system_jobs(queue=payload.queue)
    job = store.claim_system_job(
        queue=payload.queue,
        worker_id=payload.worker_id,
        lease_seconds=payload.lease_seconds,
        resource_group=payload.resource_group,
        max_queue_running=payload.max_queue_running,
        max_global_running=payload.max_global_running,
    )
    return {"job": job, "recovered_expired_jobs": recovered}


@app.post("/api/jobs/run-next", dependencies=[Depends(_authorized)])
async def run_next_system_job(payload: SystemJobRunNextRequest) -> dict[str, Any]:
    return await asyncio.to_thread(
        system_job_runner.run_next,
        queue=payload.queue,
        lease_seconds=payload.lease_seconds,
        max_queue_running=payload.max_queue_running,
        max_global_running=payload.max_global_running,
    )


@app.post("/api/jobs/{job_id}/heartbeat", dependencies=[Depends(_authorized)])
async def heartbeat_system_job(
    job_id: str, payload: SystemJobHeartbeatRequest
) -> dict[str, Any]:
    try:
        return store.heartbeat_system_job(
            job_id,
            worker_id=payload.worker_id,
            lease_seconds=payload.lease_seconds,
            progress_current=payload.progress_current,
            checkpoint=payload.checkpoint,
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/api/jobs/{job_id}/{command}", dependencies=[Depends(_authorized)])
async def command_system_job(
    job_id: str,
    command: Literal["cancel", "pause", "resume"],
    payload: SystemJobCommandRequest,
) -> dict[str, Any]:
    try:
        job = store.command_system_job(
            job_id,
            command=command,
            actor=payload.actor,
            reason=payload.reason,
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    store.append_event(
        "action",
        "SYSTEM_JOB_COMMAND",
        "系统作业控制命令已记录",
        f"{job_id} · {command} -> {job['status']}。",
        payload={
            "job_id": job_id,
            "command": command,
            "actor": payload.actor,
            "target_status": job["status"],
            "reason": payload.reason,
        },
    )
    return job


@app.post("/api/jobs/recover", dependencies=[Depends(_authorized)])
async def recover_system_jobs(queue: str | None = None) -> dict[str, Any]:
    recovered = store.recover_expired_system_jobs(queue=queue)
    return {"recovered_expired_jobs": recovered, "summary": store.system_job_summary()}


@app.get("/api/data-center", dependencies=[Depends(_authorized)])
async def data_center_snapshot() -> dict[str, Any]:
    return build_data_center_snapshot(
        store.settings(),
        sync_status=data_sync_worker.status(),
        token_configured=data_sync_worker.token_configured(),
        events=store.events(limit=300),
    )


@app.get("/api/research-tasks", dependencies=[Depends(_authorized)])
async def research_task_index() -> dict[str, Any]:
    tasks = [_research_task_view(task) for task in store.research_tasks()]
    favorite_tasks = store.favorite_ids("research_task")
    settings = store.settings()
    default_data_path = Path(
        settings.get("data_path", PROJECT_ROOT.parent / "data")
    ).expanduser().resolve()
    for task in tasks:
        task["favorite"] = task["task_id"] in favorite_tasks
    return {
        "tasks": tasks,
        "defaults": {"data_path": str(default_data_path)},
        "markets": [
            {"value": "CN_A", "label": "A 股", "enabled": True},
            {"value": "HK", "label": "港股", "enabled": True},
            {"value": "US", "label": "美股", "enabled": True},
        ],
        "summary": {
            "task_count": len(tasks),
            "running_count": sum(
                task["status"] in {"RUNNING", "RETRYING", "STOPPING"} for task in tasks
            ),
            "ready_count": sum(task["data_ready"] for task in tasks),
            "factor_count": sum(task["factor_count"] for task in tasks),
            "maximum_concurrent_iterations": (research_manager.maximum_concurrent_iterations),
        },
    }


@app.post("/api/research-protocol/preview", dependencies=[Depends(_authorized)])
async def preview_research_protocol(
    payload: ResearchProtocolPreviewRequest,
) -> dict[str, Any]:
    try:
        workspace = inspect_data_workspace(Path(payload.data_path).expanduser().resolve())
        protocol = normalize_task_protocol(payload.protocol.model_dump(mode="json"))
        blockers = protocol_blockers(
            protocol,
            data_start=str(workspace.first_trade_date),
            data_end=str(workspace.last_trade_date),
        )
        capacity = panel_validation_fold_capacity(protocol, Path(workspace.panel_path))
        blockers.extend(protocol_data_blockers(protocol, Path(workspace.panel_path)))
    except (FileNotFoundError, RuntimeError, TypeError, ValueError, OSError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {
        "valid": not blockers,
        "blockers": blockers,
        "walk_forward_capacity": capacity,
        "data_fingerprint": workspace.fingerprint,
    }


@app.post("/api/research-protocol/preset", dependencies=[Depends(_authorized)])
async def build_research_protocol_preset(
    payload: ResearchProtocolPresetRequest,
) -> dict[str, Any]:
    try:
        workspace = inspect_data_workspace(Path(payload.data_path).expanduser().resolve())
        coverage_start = date.fromisoformat(str(workspace.first_trade_date))
        coverage_end = date.fromisoformat(str(workspace.last_trade_date))
        if payload.data_start < coverage_start or payload.data_end > coverage_end:
            raise ValueError(
                "任务区间必须位于数据覆盖 "
                f"{coverage_start.isoformat()} 至 {coverage_end.isoformat()} 内"
            )
        if payload.design == RECENT_FIVE_YEAR_BACKWARD:
            protocol = recent_five_year_task_protocol(
                payload.data_start.isoformat(),
                payload.data_end.isoformat(),
                exploration_years=payload.exploration_years,
                validation_years=payload.resolved_validation_years,
                holdout_months=payload.holdout_months,
            )
        elif payload.design == REGIME_COVERAGE_BACKWARD:
            protocol = regime_coverage_task_protocol(
                payload.data_start.isoformat(),
                payload.data_end.isoformat(),
                validation_years=payload.resolved_validation_years,
                holdout_months=payload.holdout_months,
                embargo_days=payload.embargo_days,
            )
        else:
            protocol = default_task_protocol(
                payload.data_start.isoformat(),
                payload.data_end.isoformat(),
                ResearchConfig.from_toml(CONFIG_PATH),
            )
            protocol["design"] = CUSTOM_PROTOCOL_DESIGN
        panel_path = Path(workspace.panel_path)
        capacity = panel_validation_fold_capacity(protocol, panel_path)
        if int(capacity["maximum_folds"]) > 0:
            protocol["minimum_folds"] = min(
                int(protocol["minimum_folds"]), int(capacity["maximum_folds"])
            )
        blockers = protocol_blockers(
            protocol,
            data_start=payload.data_start.isoformat(),
            data_end=payload.data_end.isoformat(),
        )
        blockers.extend(protocol_data_blockers(protocol, panel_path))
    except (FileNotFoundError, RuntimeError, TypeError, ValueError, OSError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {
        "protocol": protocol,
        "valid": not blockers,
        "blockers": blockers,
        "walk_forward_capacity": capacity,
        "data_fingerprint": workspace.fingerprint,
    }


@app.post("/api/research-tasks", dependencies=[Depends(_authorized)])
async def create_research_task(payload: ResearchTaskRequest) -> dict[str, Any]:
    try:
        snapshot = _task_data_snapshot(payload)
        protocol = _resolved_task_protocol(payload, snapshot)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    task_id = f"task-{uuid.uuid4().hex[:12]}"
    task = store.create_research_task(
        task_id=task_id,
        name=payload.name,
        market=payload.market,
        data_path=snapshot["data_path"],
        data_start=snapshot["data_start"],
        data_end=snapshot["data_end"],
        snapshot_hash=snapshot["snapshot_hash"],
        status=snapshot["status"],
        protocol=protocol,
        protocol_hash=protocol_fingerprint(protocol) if protocol else None,
        notes=payload.notes,
    )
    store.append_event(
        "action",
        "RESEARCH_TASK_CREATED",
        "自动研究任务已建档",
        f"{payload.name} · {payload.market} · {snapshot['status']}",
        payload={
            "task_id": task_id,
            **snapshot,
            "protocol_hash": protocol_fingerprint(protocol) if protocol else None,
        },
    )
    return {**_research_task_view(task), "data_error": snapshot["data_error"]}


@app.get("/api/research-tasks/{task_id}", dependencies=[Depends(_authorized)])
async def research_task_detail(task_id: str) -> dict[str, Any]:
    task = store.research_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Research task not found")
    return _research_task_view(task)


@app.put("/api/research-tasks/{task_id}", dependencies=[Depends(_authorized)])
async def update_research_task(task_id: str, payload: ResearchTaskRequest) -> dict[str, Any]:
    current = store.research_task(task_id)
    if current is None:
        raise HTTPException(status_code=404, detail="Research task not found")
    if _research_task_view(current)["status"] in {"RUNNING", "RETRYING", "STOPPING"}:
        raise HTTPException(status_code=409, detail="Stop the research task before editing it")
    try:
        snapshot = _task_data_snapshot(payload)
        protocol = _resolved_task_protocol(
            payload, snapshot, preserve_base=task_id == "legacy-ashare" and payload.protocol is None
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    protocol_hash = protocol_fingerprint(protocol) if protocol else None
    protocol_changed = protocol_hash != current.get("protocol_hash")
    task = store.update_research_task(
        task_id,
        name=payload.name,
        market=payload.market,
        data_path=snapshot["data_path"],
        data_start=snapshot["data_start"],
        data_end=snapshot["data_end"],
        snapshot_hash=snapshot["snapshot_hash"],
        status=snapshot["status"] if task_id != "legacy-ashare" else current["status"],
        phase=("WAITING" if snapshot["status"] == "READY" else "CONFIGURE"),
        stop_requested=0,
        last_error=None,
        protocol_json=json.dumps(protocol, sort_keys=True, separators=(",", ":")),
        protocol_hash=protocol_hash,
        protocol_revision=int(current.get("protocol_revision") or 1) + int(protocol_changed),
        notes=payload.notes,
    )
    store.append_event(
        "action",
        "RESEARCH_TASK_UPDATED",
        "自动研究任务配置已更新",
        f"{payload.name} · {payload.market} · {snapshot['status']}",
        payload={
            "task_id": task_id,
            **snapshot,
            "protocol_hash": protocol_hash,
            "protocol_revision": task["protocol_revision"],
            "protocol_changed": protocol_changed,
        },
    )
    return {**_research_task_view(task), "data_error": snapshot["data_error"]}


@app.post("/api/research-tasks/{task_id}/start", dependencies=[Depends(_authorized)])
async def start_research_task(task_id: str) -> dict[str, Any]:
    if store.research_task(task_id) is None:
        raise HTTPException(status_code=404, detail="Research task not found")
    if data_sync_worker.alive:
        raise HTTPException(status_code=409, detail="Wait for the market-data refresh to finish")
    try:
        await research_manager.start(task_id)
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    task = store.research_task(task_id)
    assert task is not None
    return _research_task_view(task)


@app.post("/api/research-tasks/{task_id}/stop", dependencies=[Depends(_authorized)])
async def stop_research_task(task_id: str) -> dict[str, Any]:
    if store.research_task(task_id) is None:
        raise HTTPException(status_code=404, detail="Research task not found")
    await research_manager.stop(task_id)
    task = store.research_task(task_id)
    assert task is not None
    return _research_task_view(task)


@app.post("/api/research-tasks/{task_id}/baseline", dependencies=[Depends(_authorized)])
async def run_research_task_baseline(task_id: str) -> dict[str, Any]:
    if store.research_task(task_id) is None:
        raise HTTPException(status_code=404, detail="Research task not found")
    if data_sync_worker.alive:
        raise HTTPException(status_code=409, detail="Wait for the market-data refresh to finish")
    try:
        return await research_manager.run_codex_baseline(task_id)
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.get("/api/research-tasks/{task_id}/activity", dependencies=[Depends(_authorized)])
async def research_task_activity(task_id: str) -> dict[str, Any]:
    task = store.research_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Research task not found")
    view = _research_task_view(task)
    run_id = view.get("run_id")
    return {
        "task": view,
        "events": store.events(run_id=run_id, task_id=task_id, limit=80),
        "metrics": store.metric_history(limit=100, run_id=run_id) if run_id else [],
        "portfolio": store.active_portfolio(run_id=run_id) if run_id else None,
    }


@app.get("/api/factors", dependencies=[Depends(_authorized)])
async def factor_library(response: Response, refresh: bool = False) -> dict[str, Any]:
    response.headers["Cache-Control"] = "no-store, max-age=0"
    cached = store.materialized_snapshot("factor_library")
    cached_payload = cached["payload"] if cached else {}
    cache_has_integrity = (
        (cached_payload.get("knowledge_integrity") or {}).get("protocol")
        == "AUTOALPHA_FACTOR_KNOWLEDGE_INTEGRITY_V1"
    )
    if cached and not refresh and cache_has_integrity:
        return {
            **cached["payload"],
            **_materialized_response_fields(cached),
            "read_only": True,
        }
    if refresh:
        return {
            **_factor_library_cache_miss_payload(cached),
            "refresh_ignored": True,
            "refresh_hint": "Use POST /api/factors/refresh to rebuild materialized state.",
        }
    return _factor_library_cache_miss_payload(cached)


@app.post("/api/factors/refresh", dependencies=[Depends(_authorized)])
async def refresh_factor_library(
    response: Response,
    request: FactorLibraryRefreshRequest | None = None,
) -> dict[str, Any]:
    response.headers["Cache-Control"] = "no-store, max-age=0"
    request = request or FactorLibraryRefreshRequest()
    if not request.run_now:
        existing = None if request.force_new else _existing_system_job_by_type(
            "factor_library_refresh",
            queue=request.queue,
            statuses=("QUEUED", "RUNNING"),
        )
        if existing is not None:
            return {
                "queued": True,
                "deduplicated": True,
                "job": existing,
                "message": "Factor library refresh is already queued or running.",
            }
        job = store.enqueue_system_job(
            job_id=f"job-factor-library-{uuid.uuid4().hex[:12]}",
            queue=request.queue,
            job_type="factor_library_refresh",
            payload={"source": "api.factor_library_refresh"},
            priority=request.priority,
            resource_group="sqlite-writer",
            max_workers=1,
            progress_total=1,
        )
        store.append_event(
            "audit",
            "FACTOR_LIBRARY_REFRESH_QUEUED",
            "因子库主榜刷新已进入 Job Center",
            f"{job['job_id']} · queue={job['queue']} · priority={job['priority']}",
            payload={"job_id": job["job_id"], "job_type": job["job_type"]},
        )
        return {
            "queued": True,
            "deduplicated": False,
            "job": job,
            "message": "Factor library refresh queued in Job Center.",
        }
    library = {
        **_build_factor_library_payload(),
        "api_payload_protocol": "MATERIALIZED_FACTOR_LIBRARY_API_V1",
        "materialized": False,
    }
    cached = store.upsert_materialized_snapshot(
        "factor_library",
        library,
        ttl_seconds=SNAPSHOT_TTLS["factor_library"],
        source="api.refresh",
    )
    return {
        **cached["payload"],
        **_materialized_response_fields(cached),
        "read_only": False,
        "run_now": True,
    }


def _factor_library_cache_miss_payload(cached: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "api_payload_protocol": "MATERIALIZED_FACTOR_LIBRARY_API_V1",
        "materialized": False,
        "read_only": True,
        "cache_status": "STALE" if cached else "MISSING",
        "summary": {},
        "factors": [],
        "research_tasks": [],
        "data": {},
        "knowledge_integrity": (
            (cached.get("payload") or {}).get("knowledge_integrity") if cached else None
        ),
        "refresh_required": True,
        "refresh_hint": "Use POST /api/factors/refresh to rebuild materialized state.",
    }


def _build_factor_library_payload() -> dict[str, Any]:
    task_lookup = {task["task_id"]: _research_task_view(task) for task in store.research_tasks()}
    generation_ids: dict[str, str] = {}
    for task_id in task_lookup:
        _, task_config, generation_id, _ = _task_research_context(str(task_id))
        del task_config
        generation_ids[str(task_id)] = generation_id
    contamination = store.contaminated_factor_ids()
    pool_records = store.factor_pool(limit=5000)
    pool_lookup = {str(record["factor_id"]): record for record in pool_records}
    library = build_factor_library(
        pool_records,
        lifecycle_states=store.factor_lifecycle_states(),
        contaminated_factor_ids=contamination,
        research_diagnostics=store.factor_research_diagnostics(),
        current_protocol=CANONICAL_LIBRARY_PROTOCOL,
    )
    behavior = load_behavior_snapshot(RUNTIME_ROOT / "factor-behavior")
    behavior_factors = behavior.get("factors", {})
    for factor in library["factors"]:
        evidence = behavior_factors.get(factor["factor_id"], {})
        online = pool_lookup.get(factor["factor_id"], {}).get("metrics", {})
        factor.update(
            {
                "behavior_cluster_id": evidence.get("behavior_cluster_id")
                or online.get("online_behavior_cluster_id"),
                "behavior_cluster_label": evidence.get("behavior_cluster_label")
                or online.get("online_behavior_cluster_label"),
                "behavior_cluster_size": evidence.get("behavior_cluster_size")
                or online.get("online_behavior_cluster_size"),
                "behavior_cluster_role": evidence.get("behavior_cluster_role")
                or online.get("online_behavior_cluster_role"),
                "behavior_nearest_factor_id": evidence.get("behavior_nearest_factor_id")
                or online.get("online_behavior_nearest_factor_id"),
                "behavior_nearest_similarity": evidence.get("behavior_nearest_similarity")
                if evidence
                else online.get("online_behavior_nearest_similarity"),
                "behavior_signal_correlation": evidence.get("behavior_signal_correlation")
                if evidence
                else online.get("online_behavior_signal_correlation"),
                "behavior_return_correlation": evidence.get("behavior_return_correlation"),
                "behavior_redundancy": evidence.get("behavior_redundancy")
                or online.get("online_behavior_redundancy", "PENDING"),
                "behavior_cluster_method": evidence.get("behavior_cluster_method")
                or online.get("online_behavior_cluster_method"),
                "behavior_pending_full_recluster": bool(
                    not evidence and online.get("online_behavior_pending_full_recluster", False)
                ),
            }
        )
    behavior_progress = behavior.get("progress", {})
    library["behavior_clustering"] = {
        key: value
        for key, value in behavior.items()
        if key not in {"factors", "failures", "clusters", "progress"}
    }
    library["behavior_clustering"].update(
        {
            "status": behavior_progress.get("status", behavior.get("status", "NOT_STARTED")),
            "completed_count": behavior_progress.get(
                "completed_count", behavior.get("evaluated_count", 0)
            ),
            "total_count": behavior_progress.get("total_count", len(library["factors"])),
            "failed_count": behavior_progress.get(
                "failed_count", behavior.get("failed_count", 0)
            ),
            "current_factor_id": behavior_progress.get("current_factor_id"),
            "clusters": behavior.get("clusters", []),
        }
    )
    library["summary"]["behavior_cluster_count"] = int(behavior.get("cluster_count", 0))
    library["summary"]["behavior_evaluated_count"] = len(behavior_factors)
    homogeneity_snapshot = store.materialized_snapshot("factor_homogeneity_backfill")
    homogeneity_control = (
        homogeneity_snapshot["payload"].get("report") if homogeneity_snapshot else None
    )
    if not homogeneity_control:
        homogeneity_control = build_homogeneity_report(
            pool_records,
            behavior,
            source_task_id=None,
        )
    library["homogeneity_control"] = homogeneity_control
    gate_funnel_snapshot = store.materialized_snapshot("gate_funnel_diagnostics")
    library["gate_funnel"] = (
        {
            **gate_funnel_snapshot["payload"],
            "materialized": True,
            "materialized_at": gate_funnel_snapshot["updated_at"],
            "materialized_fingerprint": gate_funnel_snapshot["fingerprint"],
        }
        if gate_funnel_snapshot
        else {**build_gate_funnel_diagnostics(combine_store, quant_store), "materialized": False}
    )
    library["materialization"] = {
        "factor_homogeneity_backfill": (
            {
                "updated_at": homogeneity_snapshot["updated_at"],
                "fingerprint": homogeneity_snapshot["fingerprint"],
                "processed_count": homogeneity_snapshot["payload"].get("processed_count"),
                "behavior_snapshot_id": homogeneity_snapshot["payload"].get(
                    "behavior_snapshot_id"
                ),
            }
            if homogeneity_snapshot
            else None
        ),
        "gate_funnel_diagnostics": (
            {
                "updated_at": gate_funnel_snapshot["updated_at"],
                "fingerprint": gate_funnel_snapshot["fingerprint"],
                "total_candidates": gate_funnel_snapshot["payload"].get("total_candidates"),
                "passed_candidates": gate_funnel_snapshot["payload"].get("passed_candidates"),
            }
            if gate_funnel_snapshot
            else None
        ),
    }
    library["summary"]["crowded_behavior_cluster_count"] = int(
        homogeneity_control["crowded_cluster_count"]
    )
    library["summary"]["homogeneity_target_mechanisms"] = list(
        homogeneity_control["target_mechanisms"]
    )
    favorite_factors = store.favorite_ids("factor")
    knowledge_lookup = {
        item["factor_id"]: item for item in store.factor_knowledge_catalog(limit=5000)
    }
    missing_knowledge_factor_ids = sorted(set(pool_lookup) - set(knowledge_lookup))
    homogeneity_integrity = factor_homogeneity_integrity(
        pool_records,
        list(knowledge_lookup.values()),
    )
    library["knowledge_integrity"] = {
        "protocol": "AUTOALPHA_FACTOR_KNOWLEDGE_INTEGRITY_V1",
        "factor_count": len(pool_lookup),
        "knowledge_count": len(knowledge_lookup),
        "missing_count": len(missing_knowledge_factor_ids),
        "missing_factor_ids": missing_knowledge_factor_ids[:25],
        "complete": not missing_knowledge_factor_ids,
        "stale": bool(missing_knowledge_factor_ids),
        "recommended_job_type": (
            "factor_homogeneity_backfill" if missing_knowledge_factor_ids else None
        ),
        "homogeneity": homogeneity_integrity,
    }
    library["summary"]["factor_knowledge_count"] = len(knowledge_lookup)
    library["summary"]["factor_knowledge_missing_count"] = len(missing_knowledge_factor_ids)
    library["summary"]["factor_homogeneity_complete"] = homogeneity_integrity["complete"]
    library["summary"]["factor_homogeneity_missing_count"] = homogeneity_integrity[
        "missing_field_count"
    ]
    for factor in library["factors"]:
        source = task_lookup.get(factor["source_task_id"])
        knowledge = knowledge_lookup.get(factor["factor_id"])
        if knowledge:
            factor["canonical_mechanism"] = knowledge.get("canonical_mechanism")
            factor["knowledge_tags"] = knowledge.get("tags", [])
            factor["mechanism_summary"] = knowledge.get("mechanism_summary")
        else:
            factor["canonical_mechanism"] = None
            factor["knowledge_tags"] = []
            factor["mechanism_summary"] = None
        factor["source_task_name"] = source["name"] if source else factor["source_task_id"]
        factor["source_market"] = source["market"] if source else None
        factor["favorite"] = factor["factor_id"] in favorite_factors
    library["research_map"] = _factor_research_map(library["factors"])
    library["research_tasks"] = [
        {"task_id": task["task_id"], "name": task["name"], "market": task["market"]}
        for task in task_lookup.values()
    ]
    settings = store.settings()
    workspace = inspect_data_workspace(Path(settings["data_path"]))
    execution_basis = inspect_execution_data_basis(Path(workspace.panel_path))
    library["data"] = {
        "path": workspace.root_path,
        "first_trade_date": workspace.first_trade_date,
        "last_trade_date": workspace.last_trade_date,
        "fingerprint": workspace.fingerprint,
        "price_research_ready": workspace.price_research_ready,
        "institutional_pit_ready": workspace.institutional_pit_ready,
        "execution_basis": execution_basis.to_dict(),
    }
    library["governance"] = {
        "manual_backtests_update_champion": False,
        "manual_backtests_update_memory": False,
        "manual_backtests_consume_holdout_budget": False,
        "manual_holdout_exposure_blocks_all_generations_using_same_holdout": True,
        "generation_ids": generation_ids,
        "contaminated_factor_count": len(contamination),
    }
    library["autocombine_defaults"] = {
        "objective_profile": settings.get("autocombine_default_objective", "DRAWDOWN_FIRST"),
        "maximum_factors": int(settings.get("autocombine_default_max_factors", "5")),
    }
    return library


system_job_runner.factor_library_builder = _build_factor_library_payload


def _factor_research_map(factors: list[dict[str, Any]]) -> dict[str, Any]:
    mechanism_groups: dict[str, list[dict[str, Any]]] = {}
    behavior_groups: dict[str, list[dict[str, Any]]] = {}
    for factor in factors:
        mechanism = str(
            factor.get("canonical_mechanism")
            or factor.get("mechanism_type")
            or factor.get("category")
            or "OTHER"
        )
        mechanism_groups.setdefault(mechanism, []).append(factor)
        behavior_id = str(
            factor.get("behavior_cluster_id") or factor.get("cluster_id") or "PENDING"
        )
        behavior_groups.setdefault(behavior_id, []).append(factor)
    behavior_clusters = [
        _factor_cluster_profile(cluster_id, members, cluster_type="behavior")
        for cluster_id, members in behavior_groups.items()
    ]
    mechanism_clusters = [
        _factor_cluster_profile(cluster_id, members, cluster_type="mechanism")
        for cluster_id, members in mechanism_groups.items()
    ]
    annual_heatmap = _factor_annual_heatmap(mechanism_groups)
    crowded = [
        cluster
        for cluster in behavior_clusters
        if int(cluster["size"]) >= 8 or cluster["redundancy_counts"].get("NEAR_DUPLICATE", 0)
    ]
    near_duplicates = []
    for factor in factors:
        similarity = _finite_float(
            factor.get("behavior_nearest_similarity")
            or (factor.get("metric_summary") or {}).get("homogeneity_nearest_similarity")
        )
        if similarity is not None and similarity >= 0.92:
            near_duplicates.append(
                {
                    "factor_id": factor["factor_id"],
                    "name": factor["name"],
                    "nearest_factor_id": factor.get("behavior_nearest_factor_id")
                    or (factor.get("metric_summary") or {}).get("homogeneity_nearest_factor_id"),
                    "similarity": similarity,
                    "cluster_id": factor.get("behavior_cluster_id") or factor.get("cluster_id"),
                }
            )
    parameter_families = _factor_parameter_family_profiles(factors)
    return {
        "protocol": "AUTOALPHA_FACTOR_RESEARCH_MAP_V2",
        "research_map_protocol": "AUTOALPHA_FACTOR_RESEARCH_MAP_V2",
        "factor_count": len(factors),
        "mechanism_cluster_count": len(mechanism_clusters),
        "behavior_cluster_count": len(behavior_clusters),
        "crowded_cluster_count": len(crowded),
        "near_duplicate_count": len(near_duplicates),
        "parameter_family_count": len(parameter_families),
        "mechanism_clusters": sorted(
            mechanism_clusters, key=lambda item: item["average_score"], reverse=True
        )[:24],
        "mechanism_map": sorted(
            mechanism_clusters, key=lambda item: item["average_score"], reverse=True
        )[:24],
        "behavior_clusters": sorted(
            behavior_clusters, key=lambda item: (item["size"], item["average_score"]), reverse=True
        )[:32],
        "crowded_clusters": sorted(
            crowded, key=lambda item: (item["size"], item["near_duplicate_count"]), reverse=True
        )[:16],
        "homogeneity_fold_groups": sorted(
            behavior_clusters,
            key=lambda item: (item["size"], item["near_duplicate_count"], item["average_score"]),
            reverse=True,
        )[:32],
        "parameter_families": parameter_families[:24],
        "annual_heatmap": annual_heatmap,
        "near_duplicates": sorted(
            near_duplicates, key=lambda item: item["similarity"], reverse=True
        )[:24],
        "map_questions": [
            "which_return_source",
            "which_representative_factor",
            "which_clusters_are_crowded",
            "which_factors_are_near_duplicates",
            "which_sources_need_combination_testing",
        ],
    }


def _factor_parameter_family_profiles(factors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = {}
    for factor in factors:
        by_family.setdefault(_factor_parameter_family(factor), []).append(factor)
    profiles = []
    for family, members in by_family.items():
        leader = max(members, key=_factor_research_map_score)
        scores = [_factor_research_map_score(member) for member in members]
        profiles.append(
            {
                "parameter_family": family,
                "factor_count": len(members),
                "leader_factor_id": leader["factor_id"],
                "leader_name": leader["name"],
                "leader_score": _factor_research_map_score(leader),
                "average_score": sum(scores) / len(scores) if scores else 0.0,
                "mechanisms": sorted(
                    {
                        str(
                            member.get("canonical_mechanism")
                            or member.get("mechanism_type")
                            or member.get("category")
                            or "OTHER"
                        )
                        for member in members
                    }
                )[:8],
                "behavior_clusters": sorted(
                    {
                        str(
                            member.get("behavior_cluster_id")
                            or member.get("cluster_id")
                            or "PENDING"
                        )
                        for member in members
                    }
                )[:12],
            }
        )
    return sorted(
        profiles,
        key=lambda item: (item["factor_count"], item["average_score"]),
        reverse=True,
    )


def _factor_parameter_family(factor: dict[str, Any]) -> str:
    values: list[str] = []

    def visit(node: dict[str, Any] | None) -> None:
        if not isinstance(node, dict):
            return
        parameters = node.get("parameters") if isinstance(node.get("parameters"), dict) else {}
        for key in ("window", "period", "periods", "lookback"):
            if key in parameters:
                values.append(f"{key}={parameters[key]}")
        for child in node.get("arguments", []):
            visit(child)

    visit(factor.get("expression"))
    return "|".join(values) if values else "NO_EXPLICIT_LOOKBACK"


def _factor_cluster_profile(
    cluster_id: str,
    members: list[dict[str, Any]],
    *,
    cluster_type: str,
) -> dict[str, Any]:
    leader = max(members, key=_factor_research_map_score)
    redundancy_counts = Counter(
        str(member.get("behavior_redundancy") or "UNKNOWN") for member in members
    )
    lifecycle_counts = Counter(
        str(member.get("lifecycle_state") or "UNKNOWN") for member in members
    )
    mechanism_counts = Counter(
        str(
            member.get("canonical_mechanism")
            or member.get("mechanism_type")
            or member.get("category")
            or "OTHER"
        )
        for member in members
    )
    scores = [_factor_research_map_score(member) for member in members]
    annual_profile = _average_annual_profiles(
        [_factor_annual_profile(member) for member in members]
    )
    return {
        "cluster_id": cluster_id,
        "cluster_type": cluster_type,
        "size": len(members),
        "leader_factor_id": leader["factor_id"],
        "leader_name": leader["name"],
        "leader_score": _factor_research_map_score(leader),
        "average_score": sum(scores) / len(scores) if scores else 0.0,
        "mechanisms": [item for item, _ in mechanism_counts.most_common(5)],
        "redundancy_counts": dict(redundancy_counts),
        "lifecycle_counts": dict(lifecycle_counts),
        "near_duplicate_count": int(redundancy_counts.get("NEAR_DUPLICATE", 0)),
        "annual_profile": annual_profile,
        "weak_years": [
            year for year, value in annual_profile.items() if value is not None and value < 0.0
        ],
        "top_factors": [
            {
                "factor_id": item["factor_id"],
                "name": item["name"],
                "score": _factor_research_map_score(item),
                "behavior_redundancy": item.get("behavior_redundancy"),
                "source_task_id": item.get("source_task_id"),
            }
            for item in sorted(members, key=_factor_research_map_score, reverse=True)[:6]
        ],
    }


def _factor_annual_heatmap(
    mechanism_groups: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    rows = []
    year_set: set[int] = set()
    for mechanism, members in mechanism_groups.items():
        profile = _average_annual_profiles(
            [_factor_annual_profile(member) for member in members]
        )
        year_set.update(profile)
        if not profile:
            continue
        values = [value for value in profile.values() if value is not None]
        rows.append(
            {
                "mechanism": mechanism,
                "factor_count": len(members),
                "leader_factor_id": max(members, key=_factor_research_map_score)["factor_id"],
                "annual_returns": profile,
                "average_annual_return": sum(values) / len(values) if values else None,
                "weak_years": [
                    year for year, value in profile.items() if value is not None and value < 0.0
                ],
            }
        )
    return {
        "protocol": "AUTOALPHA_FACTOR_ANNUAL_HEATMAP_V1",
        "years": sorted(year_set),
        "rows": sorted(
            rows,
            key=lambda item: (
                item["average_annual_return"] is not None,
                item["average_annual_return"] or -100.0,
            ),
            reverse=True,
        )[:12],
    }


def _factor_annual_profile(factor: dict[str, Any]) -> dict[int, float]:
    metrics = _factor_metric_source_for_annual_profile(factor)
    folds = (
        metrics.get("long_only_walk_forward_folds")
        or metrics.get("recent_long_only_walk_forward_folds")
        or []
    )
    profile: dict[int, float] = {}
    for fold in folds:
        if not isinstance(fold, dict):
            continue
        validation_start = str(fold.get("validation_start") or "")
        annual_return = _finite_float(fold.get("annual_return"))
        if len(validation_start) < 4 or annual_return is None:
            continue
        try:
            year = int(validation_start[:4])
        except ValueError:
            continue
        profile[year] = annual_return
    return profile


def _factor_metric_source_for_annual_profile(factor: dict[str, Any]) -> dict[str, Any]:
    for key in ("metric_summary", "historical_metric_summary", "metrics"):
        metrics = factor.get(key)
        if not isinstance(metrics, dict):
            continue
        if metrics.get("long_only_walk_forward_folds") or metrics.get(
            "recent_long_only_walk_forward_folds"
        ):
            return metrics
    return {}


def _average_annual_profiles(profiles: list[dict[int, float]]) -> dict[int, float]:
    values_by_year: dict[int, list[float]] = {}
    for profile in profiles:
        for year, value in profile.items():
            values_by_year.setdefault(year, []).append(value)
    return {
        year: sum(values) / len(values)
        for year, values in sorted(values_by_year.items())
        if values
    }


def _factor_research_map_score(factor: dict[str, Any]) -> float:
    for key in ("recent_long_only_overall", "long_only_overall"):
        value = ((factor.get("ranking_values") or {}).get(key))
        parsed = _finite_float(value)
        if parsed is not None:
            return parsed
    return _finite_float((factor.get("scores") or {}).get("long_only_overall")) or 0.0


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed and abs(parsed) != float("inf") else None


@app.post("/api/autocombine/quick-task", dependencies=[Depends(_authorized)])
async def quick_autocombine_task(payload: QuickAutoCombineRequest) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for factor_id in payload.factor_ids:
        record = store.factor_pool_record(factor_id)
        if record is None:
            missing.append(factor_id)
        else:
            records.append(record)
    if missing:
        raise HTTPException(status_code=404, detail=f"Factors not found: {', '.join(missing)}")
    try:
        data_path, market, task_ids = _factor_source_context(records)
        workspace = inspect_data_workspace(data_path)
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
        blockers = protocol_blockers(
            protocol,
            data_start=workspace.first_trade_date,
            data_end=workspace.last_trade_date,
        )
        blockers.extend(protocol_data_blockers(protocol, Path(workspace.panel_path)))
        if blockers:
            raise ValueError("；".join(blockers))
        factor_count = len(records)
        maximum_factors = min(payload.maximum_factors, factor_count)
        minimum_factors = min(2, maximum_factors)
        construction = {
            **DEFAULT_CONSTRUCTION,
            "min_factors": minimum_factors,
            "max_factors": maximum_factors,
            "candidate_pool_limit": max(5, min(100, factor_count)),
        }
        objective = {
            key: value
            for key, value in OBJECTIVE_PRESETS[payload.objective_profile].items()
            if key not in {"label", "description"}
        }
        record = create_combine_task_record(
            store,
            name=f"因子库快速组合 · {factor_count} 因子",
            market=market,
            data_path=str(data_path),
            protocol=protocol,
            scope={
                "mode": "MANUAL",
                "factor_ids": payload.factor_ids,
                "required_factor_ids": [],
                "excluded_factor_ids": [],
                "source_task_ids": task_ids,
                "statuses": [],
                "families": [],
            },
            construction=construction,
            objective=objective,
            budget=DEFAULT_BUDGET,
            notes="由 AutoAlpha 因子库多选快速创建。",
        )
        task = combine_store.create_task(record)
    except (FileNotFoundError, RuntimeError, TypeError, ValueError, OSError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    contaminated_count = sum(
        bool(item.get("holdout_contaminated")) for item in task["factor_snapshot"]
    )
    combine_store.event(
        task["task_id"],
        "action",
        "QUICK_COMBINE_TASK_CREATED",
        "因子库快速组合已创建",
        f"已冻结 {factor_count} 个手动候选，其中 {contaminated_count} 个带污染标记。",
        level="WARN" if contaminated_count else "INFO",
        payload={
            "factor_ids": payload.factor_ids,
            "objective_profile": payload.objective_profile,
            "contaminated_count": contaminated_count,
        },
    )
    started = False
    start_error = None
    if payload.start_immediately:
        started, start_error = await asyncio.to_thread(
            _start_autocombine_task, str(task["task_id"])
        )
    return {
        "task_id": task["task_id"],
        "task_url": f"http://127.0.0.1:8888/tasks/{task['task_id']}",
        "factor_count": factor_count,
        "contaminated_count": contaminated_count,
        "started": started,
        "start_error": start_error,
    }


@app.post("/api/screener", dependencies=[Depends(_authorized)])
async def factor_screen(payload: FactorScreenRequest) -> dict[str, Any]:
    if data_sync_worker.alive:
        raise HTTPException(status_code=409, detail="Wait for the market-data refresh to finish")
    if payload.weights is not None and len(payload.weights) != len(payload.factor_ids):
        raise HTTPException(status_code=422, detail="weights must align with factor_ids")
    records = []
    missing = []
    for factor_id in payload.factor_ids:
        record = store.factor_pool_record(factor_id)
        if record is None:
            missing.append(factor_id)
        else:
            records.append(record)
    if missing:
        raise HTTPException(status_code=404, detail=f"Factors not found: {', '.join(missing)}")
    weights = payload.weights or [1.0] * len(records)
    data_path, market, task_ids = _factor_source_context(records)
    try:
        screen = await asyncio.to_thread(
            CrossSectionalScreener(data_path).screen,
            [factor_from_pool_record(record) for record in records],
            weights,
            ScreenerSpec(
                as_of_date=payload.as_of_date,
                selection_count=payload.selection_count,
                selection_side=payload.selection_side,
            ),
        )
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    store.append_event(
        "action",
        "FACTOR_SCREEN_COMPLETED",
        "因子选股快照已生成",
        f"{screen['as_of_date']} 使用 {len(screen['evaluated_factors'])} 个因子生成 "
        f"{len(screen['rows'])} 个候选。",
        payload={
            "scope": screen["scope"],
            "market": market,
            "data_path": str(data_path),
            "source_task_ids": task_ids,
            "as_of_date": screen["as_of_date"],
            "factor_ids": [item["factor_id"] for item in screen["evaluated_factors"]],
            "selection_side": screen["selection_side"],
            "selection_count": screen["selection_count"],
            "universe_size": screen["universe_size"],
            "data_fingerprint": screen["data_fingerprint"],
        },
    )
    return screen


@app.get("/api/paper-portfolios", dependencies=[Depends(_authorized)])
async def paper_portfolios() -> dict[str, Any]:
    portfolios = store.paper_portfolios(limit=200)
    favorite_portfolios = store.favorite_ids("paper_portfolio")
    for portfolio in portfolios:
        portfolio["favorite"] = str(portfolio["id"]) in favorite_portfolios
    return {"portfolios": portfolios}


@app.get("/api/paper-portfolios/{portfolio_id}", dependencies=[Depends(_authorized)])
async def paper_portfolio(portfolio_id: int) -> dict[str, Any]:
    record = store.paper_portfolio(portfolio_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Paper portfolio not found")
    record["favorite"] = store.favorite("paper_portfolio", str(portfolio_id)) is not None
    return record


@app.post("/api/paper-portfolios", dependencies=[Depends(_authorized)])
async def create_paper_portfolio(payload: PaperPortfolioRequest) -> dict[str, Any]:
    if data_sync_worker.alive:
        raise HTTPException(status_code=409, detail="Wait for the market-data refresh to finish")
    if payload.weights is not None and len(payload.weights) != len(payload.factor_ids):
        raise HTTPException(status_code=422, detail="weights must align with factor_ids")
    records = []
    missing = []
    for factor_id in payload.factor_ids:
        record = store.factor_pool_record(factor_id)
        if record is None:
            missing.append(factor_id)
        else:
            records.append(record)
    if missing:
        raise HTTPException(status_code=404, detail=f"Factors not found: {', '.join(missing)}")
    data_path, market, task_ids = _factor_source_context(records)
    weights = payload.weights or [1.0] * len(payload.factor_ids)
    try:
        result = await asyncio.to_thread(
            PaperTradingEngine(store, data_path).create,
            PaperStrategySpec(
                name=payload.name,
                factor_ids=payload.factor_ids,
                weights=weights,
                initial_cash_cny=payload.initial_cash_cny,
                selection_count=payload.selection_count,
                gross_exposure=payload.gross_exposure,
                slippage_bps_each_side=payload.slippage_bps_each_side,
                as_of_date=payload.as_of_date,
                market=market,
                data_path=str(data_path),
                source_task_ids=tuple(task_ids),
            ),
        )
    except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    _paper_event("PAPER_PORTFOLIO_CREATED", "模拟组合已创建", result)
    return result


@app.post("/api/paper-portfolios/{portfolio_id}/rebalance", dependencies=[Depends(_authorized)])
async def rebalance_paper_portfolio(
    portfolio_id: int, payload: FactorScreenRequest
) -> dict[str, Any]:
    portfolio = store.paper_portfolio(portfolio_id)
    if portfolio is None:
        raise HTTPException(status_code=404, detail="Paper portfolio not found")
    data_path = Path(portfolio["config"].get("data_path") or store.settings()["data_path"])
    try:
        result = await asyncio.to_thread(
            PaperTradingEngine(store, data_path).rebalance,
            portfolio_id,
            payload.as_of_date,
        )
    except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    _paper_event("PAPER_PORTFOLIO_REBALANCED", "模拟组合已再平衡", result)
    return result


@app.post("/api/paper-portfolios/mark", dependencies=[Depends(_authorized)])
async def mark_paper_portfolios() -> dict[str, Any]:
    results = await asyncio.to_thread(
        PaperTradingEngine(store, Path(store.settings()["data_path"])).mark_all
    )
    return {"updated": len(results), "portfolios": results}


@app.patch("/api/paper-portfolios/{portfolio_id}", dependencies=[Depends(_authorized)])
async def set_paper_portfolio_status(
    portfolio_id: int, payload: PaperPortfolioStatusRequest
) -> dict[str, Any]:
    try:
        result = store.update_paper_portfolio_status(portfolio_id, payload.status)
    except (KeyError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    _paper_event("PAPER_PORTFOLIO_STATUS_CHANGED", "模拟组合状态已更新", result)
    return result


@app.delete("/api/paper-portfolios/{portfolio_id}", dependencies=[Depends(_authorized)])
async def delete_paper_portfolio(portfolio_id: int) -> dict[str, bool]:
    try:
        store.delete_paper_portfolio(portfolio_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    store.set_favorite("paper_portfolio", str(portfolio_id), favorite=False)
    store.append_event(
        "action",
        "PAPER_PORTFOLIO_DELETED",
        "模拟组合已删除",
        f"模拟组合 #{portfolio_id} 已删除。",
        payload={"portfolio_id": portfolio_id},
    )
    return {"deleted": True}


@app.get("/api/product-templates", dependencies=[Depends(_authorized)])
async def product_templates() -> dict[str, Any]:
    return {"templates": product_template_catalog()}


@app.get("/api/backtest-presets", dependencies=[Depends(_authorized)])
async def backtest_presets() -> dict[str, Any]:
    return {
        "default": A_SHARE_NON_PIT_PROXY_WEEKLY_V1,
        "presets": manual_backtest_preset_catalog(),
    }


@app.get("/api/contamination-ledger", dependencies=[Depends(_authorized)])
async def contamination_ledger(limit: int = 500) -> dict[str, Any]:
    config = ResearchConfig.from_toml(CONFIG_PATH)
    generation = store.latest_generation(config.generation)
    generation_id = str(generation["generation_id"]) if generation else config.generation
    records = store.contamination_ledger(limit=limit)
    return {
        "generation_id": generation_id,
        "records": records,
        "contaminated_factor_ids": sorted(
            {str(item["factor_id"]) for item in records if item["contaminated"]}
        ),
    }


@app.get("/api/factors/{factor_id}/lifecycle", dependencies=[Depends(_authorized)])
async def factor_lifecycle(factor_id: str) -> dict[str, Any]:
    if store.factor_pool_record(factor_id) is None:
        raise HTTPException(status_code=404, detail="Factor not found")
    return {"factor_id": factor_id, "events": store.factor_lifecycle_history(factor_id)}


@app.post("/api/factors/{factor_id}/lifecycle", dependencies=[Depends(_authorized)])
async def transition_factor_lifecycle(
    factor_id: str, payload: LifecycleTransitionRequest
) -> dict[str, Any]:
    try:
        event = store.transition_factor_lifecycle(
            factor_id,
            payload.target_state,
            actor="HUMAN_CONTROL_PLANE",
            reason=payload.reason,
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    store.append_event(
        "audit",
        "FACTOR_LIFECYCLE_TRANSITION",
        "因子生命周期已更新",
        f"{factor_id}: {event['previous_state']} -> {event['state']}",
        payload=event,
    )
    return event


@app.get("/api/manual-backtests", dependencies=[Depends(_authorized)])
async def manual_backtest_history(limit: int = 30, favorite_only: bool = False) -> dict[str, Any]:
    return {
        "backtests": store.manual_backtests(limit=limit, favorite_only=favorite_only),
        "favorite_only": favorite_only,
    }


@app.get("/api/manual-backtests/{backtest_id}", dependencies=[Depends(_authorized)])
async def manual_backtest_result(backtest_id: int) -> dict[str, Any]:
    record = store.manual_backtest(backtest_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Manual backtest not found")
    if record["status"] != "COMPLETED" or not record.get("artifact_path"):
        return record
    artifact_root = (RUNTIME_ROOT / "artifacts" / "manual-backtests").resolve()
    artifact_path = Path(record["artifact_path"]).resolve()
    if not artifact_path.is_relative_to(artifact_root) or not artifact_path.is_file():
        raise HTTPException(status_code=500, detail="Manual backtest artifact is unavailable")
    result = json.loads(artifact_path.read_text(encoding="utf-8"))
    return {
        "id": backtest_id,
        **result,
        "metadata": {
            "favorite": record["favorite"],
            "title": record.get("title"),
            "notes": record.get("notes", ""),
            "tags": record.get("tags", []),
            "updated_at": record.get("updated_at"),
        },
    }


@app.get("/api/manual-backtests/{backtest_id}/trades", dependencies=[Depends(_authorized)])
async def manual_backtest_trades(
    backtest_id: int,
    limit: int = 100,
    offset: int = 0,
    side: Literal["ALL", "BUY", "SELL"] = "ALL",
    symbol: str = "",
    start_date: date | None = None,
    end_date: date | None = None,
) -> dict[str, Any]:
    record, result, trade_path = _manual_trade_artifact(backtest_id)
    if trade_path is None:
        return {
            "backtest_id": backtest_id,
            "available": False,
            "statement": result.get("trade_statement"),
            "rows": [],
            "total": 0,
        }
    limit = min(max(limit, 1), 500)
    offset = max(offset, 0)
    selected = []
    symbol_query = symbol.strip().casefold()
    with trade_path.open("r", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            if side != "ALL" and row["side"] != side:
                continue
            if symbol_query and symbol_query not in (
                f"{row['symbol']} {row['security_name']}".casefold()
            ):
                continue
            if start_date and row["trade_date"] < start_date.isoformat():
                continue
            if end_date and row["trade_date"] > end_date.isoformat():
                continue
            selected.append(_typed_trade_row(row))
    return {
        "backtest_id": backtest_id,
        "available": True,
        "statement": result["trade_statement"],
        "rows": selected[offset : offset + limit],
        "total": len(selected),
        "offset": offset,
        "limit": limit,
        "has_more": offset + limit < len(selected),
        "result_hash": record.get("result_hash"),
    }


@app.get(
    "/api/manual-backtests/{backtest_id}/trades.csv",
    dependencies=[Depends(_authorized)],
)
async def download_manual_backtest_trades(backtest_id: int) -> FileResponse:
    _, _, trade_path = _manual_trade_artifact(backtest_id)
    if trade_path is None:
        raise HTTPException(status_code=404, detail="Trade statement is unavailable")
    return FileResponse(
        trade_path,
        media_type="text/csv; charset=utf-8",
        filename=f"autoalpha-backtest-{backtest_id:06d}-trade-statement.csv",
    )


@app.patch("/api/manual-backtests/{backtest_id}", dependencies=[Depends(_authorized)])
async def update_manual_backtest_metadata(
    backtest_id: int, payload: ManualBacktestMetadataRequest
) -> dict[str, Any]:
    try:
        record = store.update_manual_backtest_metadata(
            backtest_id,
            favorite=payload.favorite,
            title=payload.title,
            notes=payload.notes,
            tags=payload.tags,
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    store.append_event(
        "action",
        "MANUAL_BACKTEST_METADATA_UPDATED",
        "人工回测收藏信息已更新",
        f"回测 #{backtest_id} {'已收藏' if payload.favorite else '已取消收藏'}。",
        payload={
            "backtest_id": backtest_id,
            "favorite": payload.favorite,
            "title": payload.title,
            "tags": payload.tags,
            "notes_hash": hashlib.sha256(payload.notes.encode()).hexdigest(),
        },
    )
    return record


@app.post("/api/manual-backtests", dependencies=[Depends(_authorized)])
async def run_manual_backtest(payload: ManualBacktestRequest) -> dict[str, Any]:
    if manual_backtest_lock.locked():
        raise HTTPException(status_code=409, detail="Another manual backtest is running")
    if payload.weights is not None and len(payload.weights) != len(payload.factor_ids):
        raise HTTPException(status_code=422, detail="weights must align with factor_ids")
    template = product_template(payload.product_template)
    if payload.backtest_engine == "EVENT_LEDGER" and template.portfolio_mode != "long_only":
        raise HTTPException(
            status_code=422,
            detail="The event ledger supports long-only product templates",
        )
    records = []
    missing = []
    for factor_id in payload.factor_ids:
        record = store.factor_pool_record(factor_id)
        if record is None:
            missing.append(factor_id)
        else:
            records.append(record)
    if missing:
        raise HTTPException(status_code=404, detail=f"Factors not found: {', '.join(missing)}")
    data_path, market, task_ids = _factor_source_context(records)
    if payload.backtest_engine == "EVENT_LEDGER":
        workspace = inspect_data_workspace(data_path)
        try:
            basis = inspect_execution_data_basis(Path(workspace.panel_path))
            if payload.execution_data_mode == "STRICT_PIT":
                basis.require_capital_ledger()
            else:
                basis.require_capital_ledger_proxy()
        except RuntimeError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
    weights = payload.weights or [1.0] * len(records)
    request = payload.model_dump(mode="json")
    request["weights"] = weights
    request["market"] = market
    request["data_path"] = str(data_path)
    request["source_task_ids"] = task_ids
    backtest_id = store.create_manual_backtest(request)
    exposures = []
    generation_ids: dict[str, str] = {}
    for task_id in task_ids:
        _, task_config, generation_id, _ = _task_research_context(task_id)
        generation_ids[task_id] = generation_id
        task_factor_ids = [
            str(record["factor_id"])
            for record in records
            if str(record.get("source_task_id") or "legacy-ashare") == task_id
        ]
        exposures.extend(
            store.record_manual_research_exposures(
                backtest_id=backtest_id,
                generation_id=generation_id,
                factor_ids=task_factor_ids,
                period_start=payload.start_date.isoformat(),
                period_end=payload.end_date.isoformat(),
                holdout_start=task_config.splits.test.start.isoformat(),
                holdout_end=task_config.splits.test.end.isoformat(),
            )
        )
    contaminated = any(item["contaminated"] for item in exposures)
    store.append_event(
        "action",
        "MANUAL_BACKTEST_STARTED",
        "人工组合回测开始",
        f"回测 #{backtest_id} 使用 {len(records)} 个因子，结果与自动治理隔离。",
        payload={
            "backtest_id": backtest_id,
            **request,
            "scope": "MANUAL_NON_GOVERNANCE",
            "generation_ids": generation_ids,
            "holdout_contaminated": contaminated,
            "exposure_evidence_hashes": [item["evidence_hash"] for item in exposures],
        },
    )
    try:
        async with manual_backtest_lock:
            engine = ManualFactorBacktester(data_path, CONFIG_PATH)
            result = await asyncio.to_thread(
                engine.run,
                [factor_from_pool_record(record) for record in records],
                weights,
                ManualBacktestSpec(
                    start_date=payload.start_date,
                    end_date=payload.end_date,
                    initial_cash_cny=payload.initial_cash_cny,
                    gross_exposure=payload.gross_exposure,
                    holding_period_days=payload.holding_period_days,
                    backtest_preset=payload.backtest_preset,
                    backtest_engine=payload.backtest_engine,
                    execution_data_mode=payload.execution_data_mode,
                    rebalance_schedule=payload.rebalance_schedule,
                    vector_cost_model=payload.vector_cost_model,
                    product_template=payload.product_template,
                    selection_fraction=payload.selection_fraction,
                    maximum_positions=payload.maximum_positions,
                    lot_size=payload.lot_size,
                    maximum_volume_participation=payload.maximum_volume_participation,
                    opening_limit_threshold=payload.opening_limit_threshold,
                    commission_bps_each_side=payload.commission_bps_each_side,
                    stamp_duty_bps_sell=payload.stamp_duty_bps_sell,
                    transfer_fee_bps_each_side=payload.transfer_fee_bps_each_side,
                    minimum_commission_cny=payload.minimum_commission_cny,
                    slippage_bps_each_side=payload.slippage_bps_each_side,
                    use_historical_fee_schedule=payload.use_historical_fee_schedule,
                    cost_stress_multiplier=payload.cost_stress_multiplier,
                ),
                multiple_testing_trials=len(store.factor_pool(limit=5000)),
            )
            artifact_path = _write_manual_backtest_artifact(backtest_id, result)
            store.complete_manual_backtest(
                backtest_id,
                metrics=result["metrics"],
                artifact_path=str(artifact_path),
                result_hash=result["result_hash"],
            )
    except Exception as error:
        store.fail_manual_backtest(backtest_id, f"{type(error).__name__}: {error}")
        store.append_event(
            "audit",
            "MANUAL_BACKTEST_FAILED",
            "人工组合回测失败",
            f"回测 #{backtest_id}: {type(error).__name__}: {error}",
            level="ERROR",
            payload={"backtest_id": backtest_id, "scope": "MANUAL_NON_GOVERNANCE"},
        )
        raise HTTPException(status_code=422, detail=str(error)) from error
    store.append_event(
        "delivery",
        "MANUAL_BACKTEST_COMPLETED",
        "人工组合回测已交付",
        f"回测 #{backtest_id} 已生成完整净值曲线与评价指标。",
        payload={
            "backtest_id": backtest_id,
            "result_hash": result["result_hash"],
            "artifact_path": str(artifact_path),
            "metrics": result["metrics"],
            "trade_statement": result.get("trade_statement"),
            "scope": "MANUAL_NON_GOVERNANCE",
            "champion_updated": False,
            "memory_updated": False,
            "holdout_budget_consumed": False,
            "holdout_contaminated": contaminated,
            "same_generation_blind_test_blocked": contaminated,
        },
    )
    return {
        "id": backtest_id,
        **result,
        "metadata": {"favorite": False, "title": None, "notes": "", "tags": []},
    }


async def _settings_center_snapshot() -> dict[str, Any]:
    stored = store.settings()
    values = GlobalSettingsValues.from_store(stored, project_root=PROJECT_ROOT)
    combine_health = await asyncio.to_thread(_autocombine_health)
    operational: dict[str, Any]
    try:
        operational = {"valid": True, **validate_operational_paths(values)}
    except (FileNotFoundError, RuntimeError, TypeError, ValueError, OSError) as error:
        operational = {"valid": False, "error": f"{type(error).__name__}: {error}"}
    runtime = runtime_snapshot(
        runtime_root=RUNTIME_ROOT,
        config_path=CONFIG_PATH,
        research_concurrency=research_manager.maximum_concurrent_iterations,
        autocombine_health=combine_health,
    )
    requested_concurrency = {
        "research_concurrency": values.research_concurrency,
        "autocombine_concurrency": values.autocombine_concurrency,
    }
    effective_concurrency = {
        "research_concurrency": research_manager.maximum_concurrent_iterations,
        "autocombine_concurrency": (
            combine_health.get("maximum_concurrent_tasks") if combine_health else None
        ),
    }
    pending_restart = [
        key
        for key in RESTART_KEYS
        if effective_concurrency.get(key) is not None
        and requested_concurrency[key] != effective_concurrency[key]
    ]
    tasks = store.research_tasks()
    combine_tasks = combine_store.tasks()
    return {
        "values": values.model_dump(mode="json"),
        "catalog": settings_catalog(),
        "credentials": {
            "api_key_configured": vault.configured(),
            "api_key_source": (
                "environment" if os.getenv("AUTOALPHA_API_KEY") else "system_keychain"
            ),
            "tushare_token_configured": data_sync_worker.token_configured(),
            "tushare_token_source": (
                "environment" if os.getenv("TUSHARE_TOKEN") else "system_keychain"
            ),
            "secret_material_returned": False,
        },
        "operational": operational,
        "runtime": runtime,
        "services": {
            "autoalpha": {
                "status": "ok",
                "active_tasks": len(research_manager.active_task_ids()),
                "task_count": len(tasks),
            },
            "autocombine": {
                "status": "ok" if combine_health else "unreachable",
                "active_tasks": sum(
                    1 for task in combine_tasks if task["status"] in {"RUNNING", "STOPPING"}
                ),
                "task_count": len(combine_tasks),
                "health": combine_health,
            },
        },
        "activation": {
            "pending_restart_keys": sorted(pending_restart),
            "requested_concurrency": requested_concurrency,
            "effective_concurrency": effective_concurrency,
            "running_tasks_keep_frozen_protocols": True,
        },
        "revisions": store.settings_revisions(limit=40),
    }


async def _save_global_settings(
    payload: GlobalSettingsUpdate,
    *,
    changed_by: str = "settings-center",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        operational = validate_operational_paths(payload.values)
    except (FileNotFoundError, RuntimeError, TypeError, ValueError, OSError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    values = payload.values.model_copy(
        update={
            "data_path": operational["data_path"],
            "market_data_root": operational["market_data_root"],
        }
    )
    if payload.api_key:
        try:
            vault.set(payload.api_key)
        except (RuntimeError, ValueError) as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
    if payload.tushare_token:
        try:
            data_sync_worker.set_token(payload.tushare_token)
        except (RuntimeError, ValueError) as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
    revision = store.save_settings_revision(
        values.to_store(),
        change_note=payload.change_note,
        changed_by=changed_by,
        metadata={
            **(metadata or {}),
            "api_key_replaced": bool(payload.api_key),
            "tushare_token_replaced": bool(payload.tushare_token),
            "credential_backend": "system_keyring",
            "data_fingerprint": operational["data_fingerprint"],
        },
    )
    changed_keys = revision["changed_keys"] if revision else []
    store.append_event(
        "audit",
        "GLOBAL_SETTINGS_REVISION_CREATED",
        "全局设置版本已保存",
        payload.change_note,
        payload={
            "revision_id": revision["id"] if revision else None,
            "changed_keys": changed_keys,
            "pending_restart_keys": sorted(set(changed_keys) & RESTART_KEYS),
            "api_key_replaced": bool(payload.api_key),
            "tushare_token_replaced": bool(payload.tushare_token),
            "data_fingerprint": operational["data_fingerprint"],
        },
    )
    if store.state()["state"] == "WAITING_CONFIGURATION" and vault.configured():
        store.update_state(state="READY", phase="CONFIGURE", last_error=None)
    return await _settings_center_snapshot()


@app.get("/api/control-settings", dependencies=[Depends(_authorized)])
async def control_settings() -> dict[str, Any]:
    return await _settings_center_snapshot()


@app.put("/api/control-settings", dependencies=[Depends(_authorized)])
async def update_control_settings(payload: GlobalSettingsUpdate) -> dict[str, Any]:
    return await _save_global_settings(payload)


@app.post(
    "/api/control-settings/revisions/{revision_id}/restore",
    dependencies=[Depends(_authorized)],
)
async def restore_control_settings(
    revision_id: int, payload: SettingsRestoreRequest
) -> dict[str, Any]:
    revision = store.settings_revision(revision_id)
    if revision is None:
        raise HTTPException(status_code=404, detail="Settings revision not found")
    values = GlobalSettingsValues.from_store(revision["values"], project_root=PROJECT_ROOT)
    return await _save_global_settings(
        GlobalSettingsUpdate(values=values, change_note=payload.change_note),
        changed_by="settings-center-restore",
        metadata={"restored_from_revision": revision_id},
    )


@app.put("/api/settings", dependencies=[Depends(_authorized)])
async def update_settings(payload: SettingsRequest) -> dict[str, Any]:
    data_path = Path(payload.data_path).expanduser().resolve()
    if not data_path.exists() or not data_path.is_dir():
        raise HTTPException(status_code=422, detail=f"Data path does not exist: {data_path}")
    try:
        workspace = inspect_data_workspace(data_path)
        workspace.require_price_research()
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    if payload.api_key:
        try:
            vault.set(payload.api_key)
        except (RuntimeError, ValueError) as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
    market_data_root = Path(payload.market_data_root).expanduser().resolve()
    if not market_data_root.is_dir():
        raise HTTPException(
            status_code=422, detail=f"Market-data path does not exist: {market_data_root}"
        )
    if payload.tushare_token:
        try:
            data_sync_worker.set_token(payload.tushare_token)
        except (RuntimeError, ValueError) as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
    store.save_settings_revision(
        {
            "base_url": payload.base_url.rstrip("/"),
            "model": payload.model,
            "data_path": workspace.root_path,
            "iteration_interval_seconds": str(payload.iteration_interval_seconds),
            "temperature": str(payload.temperature),
            "maximum_active_factors": str(payload.maximum_active_factors),
            "full_llm_enabled": str(payload.full_llm_enabled).lower(),
            "market_data_root": str(market_data_root),
            "data_auto_update_enabled": str(payload.data_auto_update_enabled).lower(),
            "data_update_hour": str(payload.data_update_hour),
        },
        change_note="通过自动研究侧栏更新兼容配置",
        changed_by="legacy-settings-form",
        metadata={
            "api_key_replaced": bool(payload.api_key),
            "tushare_token_replaced": bool(payload.tushare_token),
        },
    )
    store.append_event(
        "audit",
        "SETTINGS_UPDATED",
        "服务配置已更新",
        "模型端点、模型名称、数据路径或迭代间隔发生变更。",
        payload={
            "base_url": payload.base_url.rstrip("/"),
            "model": payload.model,
            "data_path": workspace.root_path,
            "iteration_interval_seconds": payload.iteration_interval_seconds,
            "temperature": payload.temperature,
            "maximum_active_factors": payload.maximum_active_factors,
            "full_llm_enabled": payload.full_llm_enabled,
            "api_key_replaced": bool(payload.api_key),
            "tushare_token_replaced": bool(payload.tushare_token),
            "credential_backend": "system_keyring",
            "market_data_root": str(market_data_root),
            "data_auto_update_enabled": payload.data_auto_update_enabled,
            "data_update_hour": payload.data_update_hour,
            "data_fingerprint": workspace.fingerprint,
            "price_research_ready": workspace.price_research_ready,
            "institutional_pit_ready": workspace.institutional_pit_ready,
        },
    )
    if store.state()["state"] == "WAITING_CONFIGURATION":
        store.update_state(state="READY", phase="CONFIGURE", last_error=None)
    return (await snapshot())["settings"]


@app.put("/api/data-center/settings", dependencies=[Depends(_authorized)])
async def update_data_center_settings(payload: DataCenterSettingsRequest) -> dict[str, Any]:
    data_path = Path(payload.data_path).expanduser().resolve()
    market_data_root = Path(payload.market_data_root).expanduser().resolve()
    if not market_data_root.is_dir():
        raise HTTPException(
            status_code=422, detail=f"Market-data path does not exist: {market_data_root}"
        )
    try:
        workspace = inspect_data_workspace(data_path)
        workspace.require_price_research()
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    if payload.tushare_token:
        try:
            data_sync_worker.set_token(payload.tushare_token)
        except (RuntimeError, ValueError) as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
    store.save_settings_revision(
        {
            "data_path": workspace.root_path,
            "market_data_root": str(market_data_root),
            "data_auto_update_enabled": str(payload.data_auto_update_enabled).lower(),
            "data_update_hour": str(payload.data_update_hour),
            "data_product_ids": json.dumps(payload.data_product_ids),
        },
        change_note="通过数据中心更新数据配置",
        changed_by="data-center",
        metadata={"tushare_token_replaced": bool(payload.tushare_token)},
    )
    store.append_event(
        "audit",
        "DATA_CENTER_SETTINGS_UPDATED",
        "数据中心配置已更新",
        "数据路径、下载器路径或自动更新计划发生变更。",
        payload={
            "data_path": workspace.root_path,
            "market_data_root": str(market_data_root),
            "tushare_token_replaced": bool(payload.tushare_token),
            "credential_backend": "system_keyring",
            "data_auto_update_enabled": payload.data_auto_update_enabled,
            "data_update_hour": payload.data_update_hour,
            "data_product_ids": payload.data_product_ids,
            "data_fingerprint": workspace.fingerprint,
        },
    )
    return await data_center_snapshot()


@app.post("/api/data-sync/start", dependencies=[Depends(_authorized)])
async def start_data_sync(payload: DataSyncRequest) -> dict[str, Any]:
    if payload.run_now:
        try:
            return await data_sync_worker.start(
                trigger="manual",
                dataset_ids=payload.dataset_ids,
                start_date=payload.start_date,
                end_date=payload.end_date,
            )
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
    if data_sync_worker.alive:
        raise HTTPException(status_code=409, detail="A market-data sync is already running")
    if not data_sync_worker.token_configured():
        raise HTTPException(status_code=409, detail="Tushare Token is not configured")
    existing = _existing_system_job_by_type(
        "market_data_sync",
        queue=payload.queue,
        statuses=("QUEUED", "RUNNING", "PAUSED", "PAUSE_REQUESTED", "CANCEL_REQUESTED"),
    )
    if existing is not None:
        return {
            "mode": "queued",
            "deduplicated": True,
            "message": "Market data sync is already queued or running.",
            "job": existing,
            "sync": data_sync_worker.status(),
        }
    try:
        job = store.enqueue_system_job(
            job_id=f"job-market-data-sync-{uuid.uuid4().hex[:8]}",
            queue=payload.queue,
            job_type="market_data_sync",
            payload={
                "trigger": "manual",
                "dataset_ids": payload.dataset_ids,
                "start_date": payload.start_date.isoformat() if payload.start_date else None,
                "end_date": payload.end_date.isoformat() if payload.end_date else None,
            },
            priority=payload.priority,
            resource_group="market-data-sync",
            max_workers=1,
            progress_total=1,
            max_attempts=2,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    store.append_event(
        "action",
        "MARKET_DATA_SYNC_QUEUED",
        "市场数据同步已进入 Job Center",
        f"{job['job_id']} · market_data_sync 已排队，资源组 market-data-sync。",
        payload={
            "job_id": job["job_id"],
            "queue": payload.queue,
            "dataset_ids": payload.dataset_ids,
            "start_date": payload.start_date.isoformat() if payload.start_date else None,
            "end_date": payload.end_date.isoformat() if payload.end_date else None,
        },
    )
    return {
        "mode": "queued",
        "deduplicated": False,
        "message": "Market data sync queued in Job Center.",
        "job": job,
        "sync": data_sync_worker.status(),
    }


@app.post("/api/logs/manual", dependencies=[Depends(_authorized)])
async def append_manual_log(payload: ManualLogRequest) -> dict[str, Any]:
    state = store.state()
    event = store.append_event(
        payload.category,
        "MANUAL_NOTE",
        "人工研究备注",
        payload.content,
        run_id=state.get("run_id"),
        iteration=state.get("iteration"),
        payload={"source": "control_plane"},
    )
    return {"saved": True, "event_id": event["id"]}


@app.post(
    "/api/research-tasks/{task_id}/logs/manual",
    dependencies=[Depends(_authorized)],
)
async def append_research_task_manual_log(
    task_id: str, payload: ManualLogRequest
) -> dict[str, Any]:
    task = store.research_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Research task not found")
    task_view = _research_task_view(task)
    run_id = task_view.get("run_id")
    event = store.append_event(
        payload.category,
        "MANUAL_NOTE",
        "人工研究备注",
        payload.content,
        run_id=run_id,
        iteration=task_view.get("iteration"),
        payload={"source": "control_plane", "task_id": task_id},
    )
    return {"saved": True, "event_id": event["id"]}


@app.post("/api/run/start", dependencies=[Depends(_authorized)])
async def start_run() -> dict[str, Any]:
    if data_sync_worker.alive:
        raise HTTPException(status_code=409, detail="Wait for the market-data refresh to finish")
    try:
        return await research_manager.start("legacy-ashare")
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/api/run/stop", dependencies=[Depends(_authorized)])
async def stop_run() -> dict[str, Any]:
    return await research_manager.stop("legacy-ashare")


@app.post("/api/run/baseline", dependencies=[Depends(_authorized)])
async def run_baseline() -> dict[str, Any]:
    try:
        return await research_manager.run_genesis_baseline("legacy-ashare")
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/api/audit/verify", dependencies=[Depends(_authorized)])
async def verify_audit() -> dict[str, Any]:
    count = store.verify_events()
    state = store.state()
    store.append_event(
        "audit",
        "AUDIT_VERIFIED",
        "审计链验证通过",
        f"已验证 {count} 条日志记录。",
        run_id=state.get("run_id"),
        iteration=state.get("iteration"),
        payload={"verified_records": count},
    )
    return {"valid": True, "verified_records": count}


@app.post(
    "/api/research-tasks/{task_id}/audit/verify",
    dependencies=[Depends(_authorized)],
)
async def verify_research_task_audit(task_id: str) -> dict[str, Any]:
    task = store.research_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Research task not found")
    count = store.verify_events()
    task_view = _research_task_view(task)
    store.append_event(
        "audit",
        "AUDIT_VERIFIED",
        "审计链验证通过",
        f"已验证 {count} 条日志记录。",
        run_id=task_view.get("run_id"),
        iteration=task_view.get("iteration"),
        payload={"verified_records": count, "task_id": task_id},
    )
    return {"valid": True, "verified_records": count}


@app.get("/api/events", dependencies=[Depends(_authorized)])
async def event_stream(request: Request, after: int = 0) -> StreamingResponse:
    return _event_stream_response(request, after=after, run_scoped=False)


@app.get(
    "/api/research-tasks/{task_id}/events",
    dependencies=[Depends(_authorized)],
)
async def research_task_event_stream(
    task_id: str, request: Request, after: int = 0
) -> StreamingResponse:
    task = store.research_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Research task not found")
    run_id = _research_task_view(task).get("run_id")
    return _event_stream_response(
        request,
        after=after,
        run_id=run_id,
        task_id=task_id,
        run_scoped=True,
    )


def _event_stream_response(
    request: Request,
    *,
    after: int = 0,
    run_id: str | None = None,
    task_id: str | None = None,
    run_scoped: bool,
) -> StreamingResponse:
    async def generate() -> AsyncIterator[str]:
        cursor = after
        deadline = asyncio.get_running_loop().time() + 30
        yield "retry: 1000\n\n"
        while not await request.is_disconnected() and asyncio.get_running_loop().time() < deadline:
            events = (
                store.events(
                    after_id=cursor,
                    run_id=run_id,
                    task_id=task_id,
                    limit=200,
                )
                if run_scoped
                else store.events(after_id=cursor, limit=200)
            )
            for event in events:
                cursor = max(cursor, int(event["id"]))
                yield f"id: {event['id']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
            yield ": keepalive\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _session_value(token: str) -> str:
    return hashlib.sha256(f"autoalpha-session:{token}".encode()).hexdigest()


def _write_manual_backtest_artifact(backtest_id: int, result: dict[str, Any]) -> Path:
    directory = RUNTIME_ROOT / "artifacts" / "manual-backtests"
    directory.mkdir(parents=True, exist_ok=True)
    trade_rows = result.pop("_trade_statement_rows", [])
    if result.get("trade_statement", {}).get("available"):
        trade_destination = directory / f"backtest-{backtest_id:06d}-trades.csv"
        trade_temporary = trade_destination.with_suffix(".csv.tmp")
        with trade_temporary.open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=TRADE_STATEMENT_FIELDS)
            writer.writeheader()
            writer.writerows(trade_rows)
        trade_temporary.replace(trade_destination)
        trade_hash = hashlib.sha256(trade_destination.read_bytes()).hexdigest()
        result["trade_statement"].update(
            {
                "artifact_name": trade_destination.name,
                "sha256": trade_hash,
            }
        )
    hash_payload = {key: value for key, value in result.items() if key != "result_hash"}
    result["result_hash"] = hashlib.sha256(
        json.dumps(hash_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    destination = directory / f"backtest-{backtest_id:06d}.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def _manual_trade_artifact(
    backtest_id: int,
) -> tuple[dict[str, Any], dict[str, Any], Path | None]:
    record = store.manual_backtest(backtest_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Manual backtest not found")
    if record["status"] != "COMPLETED" or not record.get("artifact_path"):
        raise HTTPException(status_code=409, detail="Manual backtest is not completed")
    artifact_root = (RUNTIME_ROOT / "artifacts" / "manual-backtests").resolve()
    result_path = Path(record["artifact_path"]).resolve()
    if not result_path.is_relative_to(artifact_root) or not result_path.is_file():
        raise HTTPException(status_code=500, detail="Manual backtest artifact is unavailable")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    artifact_name = (result.get("trade_statement") or {}).get("artifact_name")
    if not artifact_name:
        return record, result, None
    trade_path = (artifact_root / str(artifact_name)).resolve()
    if not trade_path.is_relative_to(artifact_root) or not trade_path.is_file():
        raise HTTPException(status_code=500, detail="Trade statement artifact is unavailable")
    expected_hash = str(result["trade_statement"].get("sha256", ""))
    actual_hash = hashlib.sha256(trade_path.read_bytes()).hexdigest()
    if not expected_hash or not hmac.compare_digest(expected_hash, actual_hash):
        raise HTTPException(status_code=500, detail="Trade statement integrity check failed")
    return record, result, trade_path


def _typed_trade_row(row: dict[str, str]) -> dict[str, Any]:
    integer_fields = {"trade_id", "sleeve", "quantity"}
    float_fields = {
        "price_cny",
        "notional_cny",
        "commission_cny",
        "transfer_fee_cny",
        "stamp_duty_cny",
        "total_fees_cny",
        "net_cash_flow_cny",
        "sleeve_cash_after_cny",
    }
    return {
        key: int(value) if key in integer_fields else float(value) if key in float_fields else value
        for key, value in row.items()
    }


def _paper_event(event: str, title: str, portfolio: dict[str, Any]) -> None:
    store.append_event(
        "action",
        event,
        title,
        f"模拟组合 #{portfolio['id']} · {portfolio['name']} 已更新。",
        payload={
            "portfolio_id": portfolio["id"],
            "name": portfolio["name"],
            "status": portfolio["status"],
            "last_rebalanced_date": portfolio.get("last_rebalanced_date"),
        },
    )


def main() -> None:
    uvicorn.run(
        "autoalpha.service.app:app",
        host=os.getenv("AUTOALPHA_HOST", "127.0.0.1"),
        port=int(os.getenv("AUTOALPHA_PORT", "8788")),
        reload=False,
    )


if __name__ == "__main__":
    main()
