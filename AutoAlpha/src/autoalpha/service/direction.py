from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from autoalpha.config import ResearchConfig

MECHANISM_DOMAINS = (
    "VALUATION",
    "ORDER_FLOW",
    "CAPITALIZATION",
    "TURNOVER_LIQUIDITY",
    "PRICE_REVERSAL",
    "MOMENTUM_TREND",
    "VOLATILITY_RISK",
    "OTHER_INTERPRETABLE",
)


@dataclass(frozen=True)
class DirectionDefinition:
    direction: str
    title: str
    objective: str
    success_criteria: tuple[str, ...]
    avoid: tuple[str, ...]


@dataclass(frozen=True)
class DirectionPlan:
    definition: DirectionDefinition
    score: float
    rationale: tuple[str, ...]
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "direction": self.definition.direction,
            "title": self.definition.title,
            "objective": self.definition.objective,
            "success_criteria": list(self.definition.success_criteria),
            "avoid": list(self.definition.avoid),
            "diagnostic_score": self.score,
            "rationale": list(self.rationale),
            "evidence": self.evidence,
        }


DIRECTIONS = (
    DirectionDefinition(
        "RESTORE_STABILITY",
        "跨期稳定性修复",
        "降低年度收益离散并改善最差 walk-forward fold，不追逐全期最高夏普。",
        ("年度离散至少改善冻结步长", "最差折夏普不恶化并优先改善"),
        ("只改窗口参数", "用单一年份收益解释稳定性"),
    ),
    DirectionDefinition(
        "REDUCE_TAIL_RISK",
        "尾部风险压降",
        "改善组合最大回撤和下行路径，同时保持成本后收益与容量。",
        ("最大回撤至少改善冻结步长", "年化收益牺牲不越过迁移预算"),
        ("用更高集中度换取表面回撤改善", "忽略无法退出的残余持仓"),
    ),
    DirectionDefinition(
        "IMPROVE_EXECUTION_EFFICIENCY",
        "交易效率优化",
        "降低换手与成本压力，保留可承载的组合边际价值。",
        ("换手至少降低冻结比例", "成本压力净 IR 不恶化"),
        ("仅平滑同一表达式", "以覆盖率或容量恶化换低换手"),
    ),
    DirectionDefinition(
        "IMPROVE_ROBUST_RETURN",
        "稳健收益增强",
        "改善最差折、成本后年化或组合夏普，而不是提高单因子 IC。",
        ("最差折或年化达到最小经济改善", "回撤与成本门禁继续满足"),
        ("针对最近一年拟合", "用高 IC 替代组合增量"),
    ),
    DirectionDefinition(
        "EXPAND_TRADABLE_BREADTH",
        "可交易广度提升",
        "提高有效覆盖与容量，减少由窄股票池产生的虚假表现。",
        ("覆盖率或容量达到冻结改善步长", "收益与风险方向不反转"),
        ("用缺失值填充制造覆盖", "引入不可用或非 PIT 字段"),
    ),
    DirectionDefinition(
        "DIVERSIFY_FACTOR_LIBRARY",
        "因子库分散化",
        "寻找与当前组合机制不同且低相关的边际贡献。",
        ("相关性受控", "新增或替换后至少一项组合价值门禁改善"),
        ("同一树的参数变体", "把低相关本身当作收益证据"),
    ),
    DirectionDefinition(
        "EXPLORE_EXTENDED_DATA",
        "扩展数据机制探索",
        "使用至少一个已通过覆盖门禁的估值、换手或订单流字段，检验其对现有组合的独立边际价值。",
        ("候选明确引用已解锁扩展字段", "通过单因子筛选并形成可验证的组合边际证据"),
        (
            "使用仅已下载但未覆盖研究区的字段",
            "一次混合多个无关数据机制",
            "把字段新颖性当作收益证据",
        ),
    ),
    DirectionDefinition(
        "EXPLORE_NEW_MECHANISM",
        "新机制探索",
        "在其他方向冷却或证据不足时探索未充分覆盖的经济机制。",
        ("表达式结构与经济机制均有新意", "通过单因子确定性筛选"),
        ("无假设的随机公式", "重复近期失败因子族"),
    ),
)

_BY_ID = {item.direction: item for item in DIRECTIONS}


def diagnose_direction(
    incumbent: dict[str, Any],
    recent_metrics: list[dict[str, Any]],
    *,
    blocked_directions: set[str],
    config: ResearchConfig,
    data_experiment: dict[str, Any] | None = None,
) -> DirectionPlan:
    scores = {item.direction: 0.0 for item in DIRECTIONS}
    reasons: dict[str, list[str]] = {item.direction: [] for item in DIRECTIONS}
    failure_counts: Counter[str] = Counter()
    for metrics in recent_metrics[-config.adaptive_direction.recent_candidate_window :]:
        failure_counts.update(metrics.get("portfolio_action_gate_failures", []))
        failure_counts.update(metrics.get("exploratory_gate_failures", []))

    policy = config.evaluation
    _score_maximum_violation(
        scores,
        reasons,
        "RESTORE_STABILITY",
        incumbent.get("portfolio_annual_return_dispersion"),
        policy.maximum_annual_return_dispersion,
        "当前组合年度收益离散超过绝对门禁",
    )
    _score_minimum_violation(
        scores,
        reasons,
        "IMPROVE_ROBUST_RETURN",
        incumbent.get("portfolio_walk_forward_worst_sharpe"),
        policy.minimum_worst_fold_net_ir,
        "当前组合最差 walk-forward fold 不足",
    )
    _score_minimum_violation(
        scores,
        reasons,
        "EXPAND_TRADABLE_BREADTH",
        incumbent.get("portfolio_coverage"),
        policy.minimum_coverage,
        "当前组合覆盖率不足",
    )
    _score_minimum_violation(
        scores,
        reasons,
        "EXPAND_TRADABLE_BREADTH",
        incumbent.get("portfolio_capacity_cny"),
        policy.minimum_capacity_cny,
        "当前组合容量不足",
    )
    _score_maximum_violation(
        scores,
        reasons,
        "IMPROVE_EXECUTION_EFFICIENCY",
        incumbent.get("portfolio_annual_turnover"),
        policy.maximum_annual_turnover,
        "当前组合换手超过绝对门禁",
    )
    _score_minimum_violation(
        scores,
        reasons,
        "IMPROVE_EXECUTION_EFFICIENCY",
        incumbent.get("portfolio_cost_stress_net_ir"),
        policy.minimum_cost_stress_net_ir,
        "当前组合成本压力结果不足",
    )

    drawdown = _number(incumbent.get("portfolio_max_drawdown"))
    if drawdown is not None and drawdown < -0.10:
        scores["REDUCE_TAIL_RISK"] += min(35.0, abs(drawdown) * 100)
        reasons["REDUCE_TAIL_RISK"].append("当前组合公开区最大回撤需要独立审视")
    correlation = _number(incumbent.get("portfolio_max_factor_correlation"))
    if correlation is not None and correlation > 0.60 * policy.maximum_library_correlation:
        scores["DIVERSIFY_FACTOR_LIBRARY"] += 20.0 * (
            correlation / max(policy.maximum_library_correlation, 1e-12)
        )
        reasons["DIVERSIFY_FACTOR_LIBRARY"].append("现有因子相关性接近风险预算")

    mapping = {
        "annual_dispersion": "RESTORE_STABILITY",
        "unstable_annual_returns": "RESTORE_STABILITY",
        "walk_forward_positive_fraction": "RESTORE_STABILITY",
        "walk_forward_worst_sharpe": "IMPROVE_ROBUST_RETURN",
        "unstable_walk_forward": "IMPROVE_ROBUST_RETURN",
        "weak_worst_fold": "IMPROVE_ROBUST_RETURN",
        "portfolio_value": "IMPROVE_ROBUST_RETURN",
        "incremental_net_ir": "IMPROVE_ROBUST_RETURN",
        "incremental_annual_return": "IMPROVE_ROBUST_RETURN",
        "incremental_drawdown": "REDUCE_TAIL_RISK",
        "turnover": "IMPROVE_EXECUTION_EFFICIENCY",
        "excessive_turnover": "IMPROVE_EXECUTION_EFFICIENCY",
        "cost_stress": "IMPROVE_EXECUTION_EFFICIENCY",
        "capacity": "EXPAND_TRADABLE_BREADTH",
        "coverage": "EXPAND_TRADABLE_BREADTH",
        "insufficient_coverage": "EXPAND_TRADABLE_BREADTH",
        "factor_correlation": "DIVERSIFY_FACTOR_LIBRARY",
    }
    for failure, count in failure_counts.items():
        direction = mapping.get(failure)
        if direction:
            scores[direction] += min(24.0, 3.0 * count)
    for direction in scores:
        count = sum(
            failure_counts[failure]
            for failure, mapped_direction in mapping.items()
            if mapped_direction == direction
        )
        if count:
            reasons[direction].append(f"最近公开候选累计出现 {count} 次相关失败")

    if len(recent_metrics) < config.adaptive_direction.minimum_recent_candidates:
        scores["EXPLORE_NEW_MECHANISM"] += 30.0
        reasons["EXPLORE_NEW_MECHANISM"].append("当前世代公开样本不足，优先扩充机制证据")
    scores["EXPLORE_NEW_MECHANISM"] += 1.0

    data_experiment = data_experiment or {}
    eligible_extended = sorted(
        {str(field) for field in data_experiment.get("eligible_extended_fields", [])}
    )
    under_tested = sorted(
        {str(field) for field in data_experiment.get("under_tested_fields", [])}
        & set(eligible_extended)
    )
    recent_extended = int(data_experiment.get("recent_extended_experiments", 0) or 0)
    mechanism_counts = {
        str(key): int(value)
        for key, value in data_experiment.get("mechanism_counts", {}).items()
        if str(key) in MECHANISM_DOMAINS
    }
    target_mechanism = min(
        MECHANISM_DOMAINS,
        key=lambda mechanism: (
            mechanism_counts.get(mechanism, 0),
            MECHANISM_DOMAINS.index(mechanism),
        ),
    )
    if eligible_extended:
        scores["EXPLORE_EXTENDED_DATA"] += 34.0 if recent_extended == 0 else 8.0
        if under_tested:
            scores["EXPLORE_EXTENDED_DATA"] += min(16.0, 2.0 * len(under_tested))
            reasons["EXPLORE_EXTENDED_DATA"].append(
                f"{len(under_tested)} 个已解锁扩展字段尚无候选实验"
            )
        elif recent_extended == 0:
            reasons["EXPLORE_EXTENDED_DATA"].append("扩展数据已解锁但近期尚未形成机制实验")
        extended_domains = (
            "VALUATION",
            "ORDER_FLOW",
            "CAPITALIZATION",
            "TURNOVER_LIQUIDITY",
        )
        target_mechanism = min(
            extended_domains,
            key=lambda mechanism: (
                mechanism_counts.get(mechanism, 0),
                extended_domains.index(mechanism),
            ),
        )

    available = [
        item
        for item in DIRECTIONS
        if item.direction not in blocked_directions
        and (item.direction != "EXPLORE_EXTENDED_DATA" or bool(eligible_extended))
    ]
    if not available:
        available = [
            item
            for item in DIRECTIONS
            if item.direction != "EXPLORE_EXTENDED_DATA" or bool(eligible_extended)
        ]
    winner = max(available, key=lambda item: (scores[item.direction], -_direction_index(item)))
    rationale = tuple(reasons[winner.direction]) or ("没有主导绝对缺口，执行受预算的新机制探索",)
    return DirectionPlan(
        definition=winner,
        score=float(scores[winner.direction]),
        rationale=rationale,
        evidence={
            "recent_candidates_used": min(
                len(recent_metrics), config.adaptive_direction.recent_candidate_window
            ),
            "related_failure_counts": dict(sorted(failure_counts.items())),
            "blocked_by_cooldown": sorted(blocked_directions),
            "eligible_extended_fields": eligible_extended,
            "under_tested_extended_fields": under_tested,
            "recent_extended_experiments": recent_extended,
            "mechanism_counts": mechanism_counts,
            "target_mechanism": target_mechanism,
            "all_direction_scores": {key: round(value, 6) for key, value in scores.items()},
        },
    )


def assess_direction_outcome(
    direction: str,
    baseline: dict[str, Any],
    proposed: dict[str, Any],
    *,
    accepted: bool,
    candidate_eligible: bool,
    config: ResearchConfig,
    candidate_fields: set[str] | None = None,
    required_data_fields: set[str] | None = None,
) -> dict[str, Any]:
    changes: dict[str, float | None] = {}

    def change(key: str) -> float | None:
        before = _number(baseline.get(key))
        after = _number(proposed.get(key))
        value = None if before is None or after is None else after - before
        changes[key] = value
        return value

    dispersion_change = change("portfolio_annual_return_dispersion")
    worst_fold_change = change("portfolio_walk_forward_worst_sharpe")
    drawdown_change = change("portfolio_max_drawdown")
    change("portfolio_annual_turnover")
    cost_change = change("portfolio_cost_stress_net_ir")
    annual_change = change("portfolio_simple_annual_return")
    sharpe_change = change("portfolio_sharpe_ratio")
    coverage_change = change("portfolio_coverage")
    change("portfolio_capacity_cny")
    correlation_change = change("portfolio_max_factor_correlation")

    policy = config.evaluation
    adaptive = config.adaptive_direction
    observed_fields = set(candidate_fields or set())
    required_fields = set(required_data_fields or set())
    tests = {
        "RESTORE_STABILITY": (
            _at_most(dispersion_change, -policy.minimum_stability_dispersion_reduction)
            or _at_least(worst_fold_change, policy.minimum_stability_worst_fold_sharpe_improvement)
        ),
        "REDUCE_TAIL_RISK": _at_least(
            drawdown_change, policy.minimum_diversification_drawdown_improvement
        ),
        "IMPROVE_EXECUTION_EFFICIENCY": (
            _relative_reduction(
                baseline.get("portfolio_annual_turnover"),
                proposed.get("portfolio_annual_turnover"),
                adaptive.minimum_turnover_reduction_fraction,
            )
            and _at_least(cost_change, 0.0)
        ),
        "IMPROVE_ROBUST_RETURN": (
            _at_least(worst_fold_change, policy.minimum_stability_worst_fold_sharpe_improvement)
            or _at_least(annual_change, policy.minimum_incremental_annual_return)
            or _at_least(sharpe_change, policy.minimum_diversification_sharpe_improvement)
        ),
        "EXPAND_TRADABLE_BREADTH": (
            _at_least(coverage_change, adaptive.minimum_coverage_improvement)
            or _relative_growth(
                baseline.get("portfolio_capacity_cny"),
                proposed.get("portfolio_capacity_cny"),
                0.05,
            )
        ),
        "DIVERSIFY_FACTOR_LIBRARY": (
            _at_most(correlation_change, -adaptive.minimum_correlation_reduction)
            or (
                _number(proposed.get("portfolio_max_factor_correlation")) is not None
                and float(proposed["portfolio_max_factor_correlation"])
                <= policy.maximum_library_correlation
            )
        ),
        "EXPLORE_EXTENDED_DATA": (candidate_eligible and bool(observed_fields & required_fields)),
        "EXPLORE_NEW_MECHANISM": candidate_eligible,
    }
    exploratory_direction = direction in {"EXPLORE_EXTENDED_DATA", "EXPLORE_NEW_MECHANISM"}
    direction_improved = bool(
        tests.get(direction, False) and (candidate_eligible if exploratory_direction else accepted)
    )
    unresolved = proposed.get("portfolio_proposed_absolute_failures")
    objective_resolved = bool(
        direction_improved and isinstance(unresolved, list) and not unresolved
    )
    return {
        "direction": direction,
        "accepted_portfolio_action": accepted,
        "candidate_eligible": candidate_eligible,
        "direction_improved": direction_improved,
        "objective_resolved": objective_resolved,
        "metric_changes": changes,
        "unresolved_absolute_gates": unresolved if isinstance(unresolved, list) else [],
        "candidate_fields": sorted(observed_fields),
        "required_data_fields": sorted(required_fields),
        "required_data_field_used": bool(observed_fields & required_fields),
    }


def direction_definition(direction: str) -> DirectionDefinition:
    try:
        return _BY_ID[direction]
    except KeyError as error:
        raise ValueError(f"Unknown adaptive research direction: {direction}") from error


def classify_mechanism(
    *, fields: set[str], family: str = "", name: str = "", hypothesis: str = ""
) -> str:
    """Map a proposal to one stable economic mechanism before expensive evaluation."""
    lowered = f"{family} {name} {hypothesis}".casefold()
    if fields & {"pe", "pe_ttm", "pb", "ps", "ps_ttm", "dv_ratio", "dv_ttm"}:
        return "VALUATION"
    if any(field.startswith(("buy_", "sell_", "net_mf")) for field in fields):
        return "ORDER_FLOW"
    if fields & {"total_mv", "circ_mv", "total_share", "float_share", "free_share"}:
        return "CAPITALIZATION"
    if fields & {"turnover_rate", "turnover_rate_f", "volume_ratio"}:
        return "TURNOVER_LIQUIDITY"
    if any(token in lowered for token in ("reversal", "mean reversion", "反转", "均值回归")):
        return "PRICE_REVERSAL"
    if any(token in lowered for token in ("momentum", "trend", "动量", "趋势")):
        return "MOMENTUM_TREND"
    if any(token in lowered for token in ("volatility", "risk", "波动", "风险")):
        return "VOLATILITY_RISK"
    if fields & {"amount", "vol"}:
        return "TURNOVER_LIQUIDITY"
    return "OTHER_INTERPRETABLE"


def _score_minimum_violation(
    scores: dict[str, float],
    reasons: dict[str, list[str]],
    direction: str,
    raw_value: Any,
    threshold: float,
    reason: str,
) -> None:
    value = _number(raw_value)
    if value is not None and value < threshold:
        scale = max(abs(threshold), 0.10)
        scores[direction] += 100.0 + 25.0 * (threshold - value) / scale
        reasons[direction].append(reason)


def _score_maximum_violation(
    scores: dict[str, float],
    reasons: dict[str, list[str]],
    direction: str,
    raw_value: Any,
    threshold: float,
    reason: str,
) -> None:
    value = _number(raw_value)
    if value is not None and value > threshold:
        scale = max(abs(threshold), 0.10)
        scores[direction] += 100.0 + 25.0 * (value - threshold) / scale
        reasons[direction].append(reason)


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def _at_least(value: float | None, threshold: float) -> bool:
    return value is not None and value + 1e-12 >= threshold


def _at_most(value: float | None, threshold: float) -> bool:
    return value is not None and value - 1e-12 <= threshold


def _relative_reduction(before: Any, after: Any, fraction: float) -> bool:
    left = _number(before)
    right = _number(after)
    return left is not None and right is not None and right <= left * (1.0 - fraction)


def _relative_growth(before: Any, after: Any, fraction: float) -> bool:
    left = _number(before)
    right = _number(after)
    return left is not None and right is not None and right >= left * (1.0 + fraction)


def _direction_index(item: DirectionDefinition) -> int:
    return next(index for index, candidate in enumerate(DIRECTIONS) if candidate == item)
