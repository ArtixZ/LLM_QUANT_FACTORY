from __future__ import annotations

import hashlib
import json
import math
from calendar import monthrange
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pyarrow.dataset as ds

from autoalpha.config import DateRange, ResearchConfig, SplitConfig, WalkForwardConfig

PROTOCOL_DATE_FIELDS = (
    "exploration_start",
    "exploration_end",
    "validation_start",
    "validation_end",
    "holdout_start",
    "holdout_end",
)
MINIMUM_FOLD_TRADING_DAYS = 60
RECENT_FIVE_YEAR_BACKWARD = "RECENT_FIVE_YEAR_BACKWARD"
CUSTOM_PROTOCOL_DESIGN = "CUSTOM"
PROTOCOL_DESIGNS = {CUSTOM_PROTOCOL_DESIGN, RECENT_FIVE_YEAR_BACKWARD}


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


def recent_five_year_task_protocol(
    data_start: str,
    data_end: str,
    *,
    exploration_years: int = 5,
    validation_years: int = 2,
    holdout_months: int = 6,
) -> dict[str, Any]:
    """Build a causal protocol by allocating windows backward from the latest date."""
    if not 2 <= exploration_years <= 10:
        raise ValueError("近期探索年数应在 2 至 10 年之间")
    if not 1 <= validation_years <= 5:
        raise ValueError("公开验证年数应在 1 至 5 年之间")
    if not 3 <= holdout_months <= 24:
        raise ValueError("隐藏测试月数应在 3 至 24 个月之间")
    coverage_start = date.fromisoformat(data_start)
    anchor = date.fromisoformat(data_end)
    if coverage_start > anchor:
        raise ValueError("任务数据起始日不能晚于结束日")

    holdout_start = _shift_months(anchor + timedelta(days=1), -holdout_months)
    validation_end = holdout_start - timedelta(days=1)
    validation_start = _shift_years(validation_end + timedelta(days=1), -validation_years)
    exploration_end = validation_start - timedelta(days=1)
    exploration_start = _shift_years(exploration_end + timedelta(days=1), -exploration_years)
    if exploration_start < coverage_start:
        required_years = exploration_years + validation_years
        raise ValueError(
            f"近期倒推协议至少需要约 {required_years} 年加 {holdout_months} 个月数据；"
            f"当前覆盖从 {coverage_start.isoformat()} 开始，所需起点为 "
            f"{exploration_start.isoformat()}"
        )
    return {
        "exploration_start": exploration_start.isoformat(),
        "exploration_end": exploration_end.isoformat(),
        "validation_start": validation_start.isoformat(),
        "validation_end": validation_end.isoformat(),
        "holdout_start": holdout_start.isoformat(),
        "holdout_end": anchor.isoformat(),
        "minimum_folds": _default_minimum_folds(validation_start, validation_end),
        "design": RECENT_FIVE_YEAR_BACKWARD,
        "anchor_date": anchor.isoformat(),
        "exploration_years": exploration_years,
        "validation_years": validation_years,
        "holdout_months": holdout_months,
    }


def normalize_task_protocol(value: dict[str, Any]) -> dict[str, Any]:
    result = {
        name: date.fromisoformat(str(value[name])).isoformat() for name in PROTOCOL_DATE_FIELDS
    }
    result["minimum_folds"] = int(value.get("minimum_folds", 1))
    if "design" in value:
        design = str(value.get("design") or CUSTOM_PROTOCOL_DESIGN).upper()
        if design not in PROTOCOL_DESIGNS:
            raise ValueError(f"未知研究协议模式：{design}")
        result["design"] = design
        if design == RECENT_FIVE_YEAR_BACKWARD:
            result["anchor_date"] = date.fromisoformat(str(value["anchor_date"])).isoformat()
            result["exploration_years"] = int(value.get("exploration_years", 5))
            result["validation_years"] = int(value.get("validation_years", 2))
            result["holdout_months"] = int(value.get("holdout_months", 6))
    return result


def protocol_blockers(protocol: dict[str, Any], *, data_start: str, data_end: str) -> list[str]:
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
        dates["exploration_start"]
        <= dates["exploration_end"]
        < dates["validation_start"]
        <= dates["validation_end"]
        < dates["holdout_start"]
        <= dates["holdout_end"]
    ):
        blockers.append("探索、公开验证和隐藏测试必须按时间先后排列且互不重叠")
    if normalized.get("design") == RECENT_FIVE_YEAR_BACKWARD:
        try:
            expected = recent_five_year_task_protocol(
                data_start,
                data_end,
                exploration_years=int(normalized["exploration_years"]),
                validation_years=int(normalized["validation_years"]),
                holdout_months=int(normalized["holdout_months"]),
            )
        except ValueError as error:
            blockers.append(str(error))
        else:
            mismatched = [
                field for field in PROTOCOL_DATE_FIELDS if normalized[field] != expected[field]
            ]
            if normalized.get("anchor_date") != data_end or mismatched:
                blockers.append(
                    "近期五年倒推模板必须锚定任务数据最新日；请重新应用模板，或切换为自定义协议"
                )
    durations = {
        "探索区": (dates["exploration_end"] - dates["exploration_start"]).days + 1,
        "公开验证区": (dates["validation_end"] - dates["validation_start"]).days + 1,
        "隐藏测试区": (dates["holdout_end"] - dates["holdout_start"]).days + 1,
    }
    minimum_days = {"探索区": 120, "公开验证区": 84, "隐藏测试区": 60}
    for label, duration in durations.items():
        if duration < minimum_days[label]:
            blockers.append(f"{label}至少需要 {minimum_days[label]} 个自然日，当前为 {duration} 日")
    maximum_folds = _available_validation_folds(dates["validation_start"], dates["validation_end"])
    if not 1 <= normalized["minimum_folds"] <= maximum_folds:
        blockers.append(f"最少验证折数应在 1 至 {maximum_folds} 之间")
    return blockers


def validation_fold_capacity(protocol: dict[str, Any], trading_dates: list[date]) -> dict[str, Any]:
    normalized = normalize_task_protocol(protocol)
    start = date.fromisoformat(normalized["validation_start"])
    end = date.fromisoformat(normalized["validation_end"])
    counts = {
        year: sum(start <= item <= end and item.year == year for item in trading_dates)
        for year in range(start.year, end.year + 1)
    }
    evaluable_years = [year for year, count in counts.items() if count >= MINIMUM_FOLD_TRADING_DAYS]
    return {
        "minimum_observations_per_fold": MINIMUM_FOLD_TRADING_DAYS,
        "maximum_folds": len(evaluable_years),
        "evaluable_years": evaluable_years,
        "observations_by_year": counts,
    }


def panel_validation_fold_capacity(protocol: dict[str, Any], panel_path: Path) -> dict[str, Any]:
    files = sorted(panel_path.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {panel_path}")
    dataset = ds.dataset(files, format="parquet")
    if "trade_date" not in dataset.schema.names:
        raise ValueError("Panel does not contain trade_date")
    values = dataset.to_table(columns=["trade_date"]).column("trade_date").unique().to_pylist()
    trading_dates = sorted(
        {
            item.date() if hasattr(item, "date") and not isinstance(item, date) else item
            for item in values
            if item is not None
        }
    )
    return validation_fold_capacity(protocol, trading_dates)


def protocol_data_blockers(protocol: dict[str, Any], panel_path: Path) -> list[str]:
    capacity = panel_validation_fold_capacity(protocol, panel_path)
    requested = normalize_task_protocol(protocol)["minimum_folds"]
    maximum = int(capacity["maximum_folds"])
    if requested <= maximum:
        return []
    counts = ", ".join(
        f"{year}={count}日" for year, count in capacity["observations_by_year"].items()
    )
    return [
        f"当前公开验证区只能形成 {maximum} 个有效 walk-forward 折，"
        f"但最少折数设置为 {requested}（每折至少 {MINIMUM_FOLD_TRADING_DAYS} 个交易日；"
        f"各年：{counts}）"
    ]


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
    canonical = json.dumps(normalize_task_protocol(protocol), sort_keys=True, separators=(",", ":"))
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


def _shift_years(value: date, years: int) -> date:
    target_year = value.year + years
    return value.replace(
        day=min(value.day, monthrange(target_year, value.month)[1]), year=target_year
    )


def _shift_months(value: date, months: int) -> date:
    month_index = value.year * 12 + value.month - 1 + months
    target_year, zero_based_month = divmod(month_index, 12)
    target_month = zero_based_month + 1
    return date(
        target_year,
        target_month,
        min(value.day, monthrange(target_year, target_month)[1]),
    )
