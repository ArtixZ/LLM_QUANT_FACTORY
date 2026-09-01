from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator, model_validator

from autoalpha.config import ResearchConfig
from autoalpha.data.product_catalog import (
    DEFAULT_PRODUCT_IDS,
    data_product_catalog,
    resolve_products,
)
from autoalpha.data.workspace import inspect_data_workspace
from autoalpha.service.autocombine import DEFAULT_BUDGET, DEFAULT_CONSTRUCTION, OBJECTIVE_PRESETS
from autoalpha.service.database_backend import database_runtime_config

OBJECTIVE_OPTIONS = tuple(OBJECTIVE_PRESETS)
RESTART_KEYS = {"research_concurrency", "autocombine_concurrency"}


class GlobalSettingsValues(BaseModel):
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-5.2"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    full_llm_enabled: bool = True
    iteration_interval_seconds: float = Field(default=5.0, ge=0.5, le=3600)
    proposal_batch_size: int = Field(default=3, ge=1, le=3)
    maximum_active_factors: int = Field(default=5, ge=1, le=12)
    research_concurrency: int = Field(default=2, ge=1, le=8)
    data_path: str
    market_data_root: str
    data_auto_update_enabled: bool = True
    data_update_hour: int = Field(default=18, ge=0, le=23)
    data_product_ids: list[str] = Field(default_factory=lambda: list(DEFAULT_PRODUCT_IDS))
    autocombine_default_objective: Literal[
        "ROBUST_ACTIVE_LONG_ONLY",
        "DRAWDOWN_FIRST",
        "PORTFOLIO_SHARPE_FIRST",
        "ABSOLUTE_LONG_ONLY",
        "LOW_TURNOVER",
        "DIVERSIFICATION_FIRST",
    ] = "DRAWDOWN_FIRST"
    autocombine_default_min_factors: int = Field(default=2, ge=1, le=12)
    autocombine_default_max_factors: int = Field(default=5, ge=1, le=12)
    autocombine_default_minimum_weight: float = Field(default=0.05, ge=0.01, le=0.50)
    autocombine_default_maximum_weight: float = Field(default=0.50, ge=0.10, le=1.0)
    autocombine_default_weight_step: float = Field(default=0.05, ge=0.01, le=0.25)
    autocombine_default_pool_limit: int = Field(default=30, ge=5, le=100)
    autocombine_default_maximum_experiments: int = Field(default=60, ge=1, le=5000)
    autocombine_default_llm_proposals: int = Field(default=20, ge=0, le=1000)
    autocombine_default_iteration_interval_seconds: float = Field(default=0.5, ge=0.0, le=3600)
    autocombine_concurrency: int = Field(default=2, ge=1, le=8)

    @field_validator("base_url", "model", "data_path", "market_data_root")
    @classmethod
    def required_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("value cannot be empty")
        return cleaned

    @field_validator("base_url")
    @classmethod
    def valid_base_url(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        return value.rstrip("/")

    @field_validator("data_product_ids")
    @classmethod
    def valid_products(cls, value: list[str]) -> list[str]:
        cleaned = list(dict.fromkeys(item.strip() for item in value if item.strip()))
        resolve_products(cleaned)
        if "core_market" not in cleaned:
            raise ValueError("core_market must remain enabled")
        return cleaned

    @model_validator(mode="after")
    def valid_combine_constraints(self) -> GlobalSettingsValues:
        if self.autocombine_default_min_factors > self.autocombine_default_max_factors:
            raise ValueError("AutoCombine minimum factor count exceeds maximum")
        if self.autocombine_default_minimum_weight > self.autocombine_default_maximum_weight:
            raise ValueError("AutoCombine minimum weight exceeds maximum")
        if (
            self.autocombine_default_minimum_weight * self.autocombine_default_max_factors
            > 1.0 + 1e-9
        ):
            raise ValueError("AutoCombine minimum weights make the factor count infeasible")
        if (
            self.autocombine_default_maximum_weight * self.autocombine_default_min_factors
            < 1.0 - 1e-9
        ):
            raise ValueError("AutoCombine maximum weights make the factor count infeasible")
        return self

    @classmethod
    def from_store(cls, settings: dict[str, str], *, project_root: Path) -> GlobalSettingsValues:
        defaults = default_settings(project_root)
        values: dict[str, Any] = {}
        for name, field in cls.model_fields.items():
            raw = settings.get(name, defaults[name])
            if name == "data_product_ids":
                try:
                    values[name] = json.loads(str(raw))
                except (TypeError, ValueError, json.JSONDecodeError):
                    values[name] = list(DEFAULT_PRODUCT_IDS)
            elif field.annotation is bool:
                values[name] = str(raw).casefold() == "true"
            else:
                values[name] = raw
        return cls.model_validate(values)

    def to_store(self) -> dict[str, str]:
        values = self.model_dump(mode="python")
        return {
            key: (
                json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                if isinstance(value, list)
                else str(value).lower()
                if isinstance(value, bool)
                else str(value)
            )
            for key, value in values.items()
        }


class GlobalSettingsUpdate(BaseModel):
    values: GlobalSettingsValues
    api_key: str | None = Field(default=None, max_length=10000)
    tushare_token: str | None = Field(default=None, max_length=10000)
    change_note: str = Field(default="更新全局设置", max_length=300)

    @field_validator("api_key", "tushare_token")
    @classmethod
    def clean_secret(cls, value: str | None) -> str | None:
        return value.strip() if value and value.strip() else None

    @field_validator("change_note")
    @classmethod
    def clean_note(cls, value: str) -> str:
        return value.strip() or "更新全局设置"


def default_settings(project_root: Path) -> dict[str, Any]:
    return {
        "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "model": os.getenv("AUTOALPHA_MODEL", "gpt-5.2"),
        "temperature": 0.7,
        "full_llm_enabled": True,
        "iteration_interval_seconds": 5.0,
        "proposal_batch_size": 3,
        "maximum_active_factors": 5,
        "research_concurrency": int(os.getenv("AUTOALPHA_MAX_CONCURRENT_RESEARCH", "2")),
        "data_path": os.getenv("AUTOALPHA_DATA_PATH", str(project_root.parent / "data")),
        "market_data_root": str(Path.home() / "MarketData" / "Ashare"),
        "data_auto_update_enabled": True,
        "data_update_hour": 18,
        "data_product_ids": list(DEFAULT_PRODUCT_IDS),
        "autocombine_default_objective": "DRAWDOWN_FIRST",
        "autocombine_default_min_factors": DEFAULT_CONSTRUCTION["min_factors"],
        "autocombine_default_max_factors": DEFAULT_CONSTRUCTION["max_factors"],
        "autocombine_default_minimum_weight": DEFAULT_CONSTRUCTION["minimum_weight"],
        "autocombine_default_maximum_weight": DEFAULT_CONSTRUCTION["maximum_weight"],
        "autocombine_default_weight_step": DEFAULT_CONSTRUCTION["weight_step"],
        "autocombine_default_pool_limit": DEFAULT_CONSTRUCTION["candidate_pool_limit"],
        "autocombine_default_maximum_experiments": DEFAULT_BUDGET["maximum_experiments"],
        "autocombine_default_llm_proposals": DEFAULT_BUDGET["maximum_llm_proposals"],
        "autocombine_default_iteration_interval_seconds": DEFAULT_BUDGET[
            "iteration_interval_seconds"
        ],
        "autocombine_concurrency": int(os.getenv("AUTOCOMBINE_CONCURRENCY", "2")),
    }


def validate_operational_paths(values: GlobalSettingsValues) -> dict[str, Any]:
    data_path = Path(values.data_path).expanduser().resolve()
    market_data_root = Path(values.market_data_root).expanduser().resolve()
    if not data_path.is_dir():
        raise ValueError(f"Data path does not exist: {data_path}")
    if not market_data_root.is_dir():
        raise ValueError(f"Market-data path does not exist: {market_data_root}")
    workspace = inspect_data_workspace(data_path)
    workspace.require_price_research()
    return {
        "data_path": workspace.root_path,
        "market_data_root": str(market_data_root),
        "data_fingerprint": workspace.fingerprint,
        "data_start": workspace.first_trade_date,
        "data_end": workspace.last_trade_date,
        "price_research_ready": workspace.price_research_ready,
        "institutional_pit_ready": workspace.institutional_pit_ready,
    }


def settings_catalog() -> list[dict[str, Any]]:
    objective_options = [
        {"value": key, "label": str(value["label"]), "description": value["description"]}
        for key, value in OBJECTIVE_PRESETS.items()
    ]
    return [
        _group(
            "provider",
            "AI Provider",
            "OpenAI Compatible 连接、模型与生成行为。凭证始终留在系统 Keychain。",
            [
                _field("base_url", "API Base URL", "url", "下次模型调用", "数据库"),
                _field("model", "模型名称", "text", "下次模型调用", "数据库"),
                _field("temperature", "生成温度", "number", "下次模型调用", "数据库", 0, 2, 0.05),
                _field("full_llm_enabled", "完整 LLM 团队", "boolean", "下一轮", "数据库"),
                _field("api_key", "API Key", "secret", "立即", "系统 Keychain"),
            ],
        ),
        _group(
            "autoalpha",
            "AutoAlpha",
            "自动因子研究的全局默认值；任务级时间协议与冻结快照不受反向修改。",
            [
                _field(
                    "iteration_interval_seconds",
                    "轮次间隔（秒）",
                    "number",
                    "下一轮",
                    "数据库",
                    0.5,
                    3600,
                    0.5,
                ),
                _field(
                    "proposal_batch_size",
                    "同方向批量提案数",
                    "number",
                    "下一轮",
                    "数据库",
                    1,
                    3,
                    1,
                ),
                _field(
                    "maximum_active_factors",
                    "冠军组合因子上限",
                    "number",
                    "下一轮",
                    "数据库",
                    1,
                    12,
                    1,
                ),
                _field(
                    "research_concurrency",
                    "并发研究轮次",
                    "number",
                    "重启服务",
                    "数据库 / 环境变量",
                    1,
                    8,
                    1,
                ),
            ],
        ),
        _group(
            "autocombine",
            "AutoCombine",
            "新建组合任务和因子库快速优化的默认值；已有任务继续使用冻结配置。",
            [
                _field(
                    "autocombine_default_objective",
                    "默认优化目标",
                    "select",
                    "新任务",
                    "数据库",
                    options=objective_options,
                ),
                _field(
                    "autocombine_default_min_factors",
                    "最少因子",
                    "number",
                    "新任务",
                    "数据库",
                    1,
                    12,
                    1,
                ),
                _field(
                    "autocombine_default_max_factors",
                    "最多因子",
                    "number",
                    "新任务",
                    "数据库",
                    1,
                    12,
                    1,
                ),
                _field(
                    "autocombine_default_minimum_weight",
                    "最小因子权重",
                    "number",
                    "新任务",
                    "数据库",
                    0.01,
                    0.5,
                    0.01,
                ),
                _field(
                    "autocombine_default_maximum_weight",
                    "最大因子权重",
                    "number",
                    "新任务",
                    "数据库",
                    0.1,
                    1,
                    0.05,
                ),
                _field(
                    "autocombine_default_weight_step",
                    "权重步长",
                    "number",
                    "新任务",
                    "数据库",
                    0.01,
                    0.25,
                    0.01,
                ),
                _field(
                    "autocombine_default_pool_limit",
                    "候选池上限",
                    "number",
                    "新任务",
                    "数据库",
                    5,
                    100,
                    1,
                ),
                _field(
                    "autocombine_default_maximum_experiments",
                    "实验预算",
                    "number",
                    "新任务",
                    "数据库",
                    1,
                    5000,
                    1,
                ),
                _field(
                    "autocombine_default_llm_proposals",
                    "LLM 提议预算",
                    "number",
                    "新任务",
                    "数据库",
                    0,
                    1000,
                    1,
                ),
                _field(
                    "autocombine_default_iteration_interval_seconds",
                    "组合实验间隔（秒）",
                    "number",
                    "新任务",
                    "数据库",
                    0,
                    3600,
                    0.5,
                ),
                _field(
                    "autocombine_concurrency",
                    "并发组合任务",
                    "number",
                    "重启服务",
                    "数据库 / 环境变量",
                    1,
                    8,
                    1,
                ),
            ],
        ),
        _group(
            "data",
            "数据与更新",
            "统一研究工作区、下载器和收盘后增量更新计划。",
            [
                _field("data_path", "研究数据工作区", "path", "新任务；旧任务冻结", "数据库"),
                _field("market_data_root", "市场数据下载器", "path", "下次同步", "数据库"),
                _field("data_auto_update_enabled", "自动增量更新", "boolean", "立即", "数据库"),
                _field(
                    "data_update_hour", "每日更新小时", "number", "下次调度", "数据库", 0, 23, 1
                ),
                _field("tushare_token", "Tushare Token", "secret", "立即", "系统 Keychain"),
                _field(
                    "data_product_ids",
                    "启用的数据产品",
                    "multiselect",
                    "下次同步",
                    "数据库",
                    options=[
                        {
                                "value": item["dataset_id"],
                                "label": item["label"],
                                "description": (
                                    "尚未接入可恢复下载契约；" + str(item["description"])
                                    if item["integration_state"] == "CATALOG"
                                    else item["description"]
                                ),
                                "disabled": item["dataset_id"] == "core_market"
                                or item["integration_state"] == "CATALOG",
                        }
                        for item in data_product_catalog()
                    ],
                ),
            ],
        ),
    ]


def runtime_snapshot(
    *,
    runtime_root: Path,
    config_path: Path,
    research_concurrency: int,
    autocombine_health: dict[str, Any] | None,
) -> dict[str, Any]:
    config = ResearchConfig.from_toml(config_path)
    database_config = database_runtime_config()
    autoalpha_address = (
        f"http://{os.getenv('AUTOALPHA_HOST', '127.0.0.1')}:"
        f"{os.getenv('AUTOALPHA_PORT', '8788')}"
    )
    autocombine_address = (
        f"http://{os.getenv('AUTOCOMBINE_HOST', '127.0.0.1')}:"
        f"{os.getenv('AUTOCOMBINE_PORT', '8888')}"
    )
    return {
        "runtime": [
            _runtime(
                "AutoAlpha 地址",
                autoalpha_address,
                "环境变量",
                "重启服务",
            ),
            _runtime(
                "AutoCombine 地址",
                autocombine_address,
                "环境变量",
                "重启服务",
            ),
            _runtime("运行目录", str(runtime_root.resolve()), "AUTOALPHA_RUNTIME", "重启服务"),
            _runtime("研究配置", str(config_path.resolve()), "AUTOALPHA_CONFIG", "重启服务"),
            _runtime(
                "数据库后端",
                database_config.backend,
                "AUTOALPHA_DATABASE_BACKEND",
                "重启服务",
            ),
            _runtime(
                "PostgreSQL 迁移阶段",
                database_config.migration_stage,
                "AUTOALPHA_DATABASE_URL",
                "重启服务",
            ),
            _runtime("AutoAlpha 并发", str(research_concurrency), "当前进程", "重启服务"),
            _runtime(
                "AutoCombine 并发",
                str((autocombine_health or {}).get("maximum_concurrent_tasks", "不可达")),
                "当前进程",
                "重启服务",
            ),
        ],
        "governance": {
            "protocol_version": config.governance.protocol_version,
            "portfolio_mode": "LONG_ONLY",
            "execution_protocol": config.strategy_evaluation.engine_protocol,
            "execution_data_mode": config.strategy_evaluation.execution_data_mode,
            "rebalance_schedule": config.strategy_evaluation.rebalance_schedule,
            "gross_exposure": config.strategy_evaluation.gross_exposure,
            "maximum_positions": config.strategy_evaluation.maximum_positions,
            "holding_period_days": config.portfolio.holding_period_days,
            "holdout_budget": config.governance.maximum_holdout_evaluations_per_generation,
        },
    }


def _group(key: str, title: str, description: str, fields: list[dict[str, Any]]) -> dict[str, Any]:
    return {"key": key, "title": title, "description": description, "fields": fields}


def _field(
    key: str,
    label: str,
    kind: str,
    effect: str,
    source: str,
    minimum: float | None = None,
    maximum: float | None = None,
    step: float | None = None,
    *,
    options: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "key": key,
        "label": label,
        "kind": kind,
        "effect": effect,
        "source": source,
        "minimum": minimum,
        "maximum": maximum,
        "step": step,
        "options": options or [],
    }


def _runtime(label: str, value: str, source: str, effect: str) -> dict[str, str]:
    return {"label": label, "value": value, "source": source, "effect": effect}
