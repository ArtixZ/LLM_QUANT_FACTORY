from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from datetime import date, timedelta
from typing import Any

from autoalpha.config import DateRange, ResearchConfig, SplitConfig, WalkForwardConfig

PROTOCOL_DATE_FIELDS = (
    "exploration_start",
    "exploration_end",
    "validation_start",
    "validation_end",
    "holdout_start",
    "holdout_end",
)


def default_task_protocol(
    data_start: str,
    data_end: str,
    base: ResearchConfig,
    *,
    preserve_base: bool = False,
) -> dict[str, Any]:
    if preserve_base:
        return {
            "exploration_start": base.splits.train.start.isoformat(),
            "exploration_end": base.splits.train.end.isoformat(),
            "validation_start": base.splits.validation.start.isoformat(),
            "validation_end": base.splits.validation.end.isoformat(),
            "holdout_start": base.splits.test.start.isoformat(),
            "holdout_end": base.splits.test.end.isoformat(),
            "minimum_folds": base.walk_forward.minimum_folds,
        }
    start = date.fromisoformat(data_start)
    end = date.fromisoformat(data_end)
    total_days = max(1, (end - start).days + 1)
    holdout_days = max(60, round(total_days * 0.20))
    validation_days = max(84, round(total_days * 0.30))
    if holdout_days + validation_days >= total_days:
        holdout_days = max(1, total_days // 5)
        validation_days = max(1, total_days // 3)
    holdout_start = end - timedelta(days=holdout_days - 1)
    validation_end = holdout_start - timedelta(days=1)
    validation_start = validation_end - timedelta(days=validation_days - 1)
    exploration_end = validation_start - timedelta(days=1)
    protocol = {
        "exploration_start": start.isoformat(),
        "exploration_end": exploration_end.isoformat(),
        "validation_start": validation_start.isoformat(),
        "validation_end": validation_end.isoformat(),
        "holdout_start": holdout_start.isoformat(),
        "holdout_end": end.isoformat(),
        "minimum_folds": _default_minimum_folds(validation_start, validation_end),
    }
    return protocol


def normalize_task_protocol(value: dict[str, Any]) -> dict[str, Any]:
    result = {
        name: date.fromisoformat(str(value[name])).isoformat()
        for name in PROTOCOL_DATE_FIELDS
    }
    result["minimum_folds"] = int(value.get("minimum_folds", 1))
    return result


def protocol_blockers(
    protocol: dict[str, Any], *, data_start: str, data_end: str
) -> list[str]:
    try:
        normalized = normalize_task_protocol(protocol)
    except (KeyError, TypeError, ValueError) as error:
        return [f"研究切分配置无效：{error}"]
    dates = {name: date.fromisoformat(normalized[name]) for name in PROTOCOL_DATE_FIELDS}
    coverage_start = date.fromisoformat(data_start)
    coverage_end = date.fromisoformat(data_end)
    blockers: list[str] = []
    if dates["exploration_start"] < coverage_start or dates["holdout_end"] > coverage_end:
        blockers.append(f"研究切分必须位于数据覆盖 {data_start} 至 {data_end} 内")
    if not (
        dates["exploration_start"] <= dates["exploration_end"]
        < dates["validation_start"] <= dates["validation_end"]
        < dates["holdout_start"] <= dates["holdout_end"]
    ):
        blockers.append("探索、公开验证和隐藏测试必须按时间先后排列且互不重叠")
    durations = {
        "探索区": (dates["exploration_end"] - dates["exploration_start"]).days + 1,
        "公开验证区": (dates["validation_end"] - dates["validation_start"]).days + 1,
        "隐藏测试区": (dates["holdout_end"] - dates["holdout_start"]).days + 1,
    }
    minimum_days = {"探索区": 120, "公开验证区": 84, "隐藏测试区": 60}
    for label, duration in durations.items():
        if duration < minimum_days[label]:
            blockers.append(f"{label}至少需要 {minimum_days[label]} 个自然日，当前为 {duration} 日")
    maximum_folds = _available_validation_folds(
        dates["validation_start"], dates["validation_end"]
    )
    if not 1 <= normalized["minimum_folds"] <= maximum_folds:
        blockers.append(f"最少验证折数应在 1 至 {maximum_folds} 之间")
    return blockers


def task_research_config(
    base: ResearchConfig,
    protocol: dict[str, Any],
    *,
    task_id: str,
) -> ResearchConfig:
    normalized = normalize_task_protocol(protocol)
    train = DateRange(
        date.fromisoformat(normalized["exploration_start"]),
        date.fromisoformat(normalized["exploration_end"]),
    )
    validation = DateRange(
        date.fromisoformat(normalized["validation_start"]),
        date.fromisoformat(normalized["validation_end"]),
    )
    test = DateRange(
        date.fromisoformat(normalized["holdout_start"]),
        date.fromisoformat(normalized["holdout_end"]),
    )
    fingerprint = protocol_fingerprint(normalized)
    train_years = max(1, math.ceil(((train.end - train.start).days + 1) / 365.25))
    walk_forward = WalkForwardConfig(
        train_years=train_years,
        validation_years=1,
        first_validation_year=validation.start.year,
        last_validation_year=validation.end.year,
        minimum_folds=normalized["minimum_folds"],
    )
    governance = replace(
        base.governance,
        protocol_version=f"{base.governance.protocol_version}-task-{fingerprint[:10]}",
    )
    return replace(
        base,
        name=f"{base.name}:{task_id}",
        generation=f"{base.generation}-task-{fingerprint[:10]}",
        splits=SplitConfig(
            train=train,
            validation=validation,
            test=test,
            embargo_days=base.splits.embargo_days,
        ),
        walk_forward=walk_forward,
        governance=governance,
    )


def protocol_fingerprint(protocol: dict[str, Any]) -> str:
    canonical = json.dumps(
        normalize_task_protocol(protocol), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _default_minimum_folds(start: date, end: date) -> int:
    return max(1, min(6, _available_validation_folds(start, end)))


def _available_validation_folds(start: date, end: date) -> int:
    count = 0
    for year in range(start.year, end.year + 1):
        segment_start = max(start, date(year, 1, 1))
        segment_end = min(end, date(year, 12, 31))
        if (segment_end - segment_start).days + 1 >= 84:
            count += 1
    return max(1, count)
