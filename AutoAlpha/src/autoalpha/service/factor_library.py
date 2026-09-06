from __future__ import annotations

import math
from collections import Counter
from typing import Any

from autoalpha.service.mechanism import normalize_mechanism

CATEGORY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("价格反转与均值回归", ("reversal", "mean reversion", "mean_reversion", "反转")),
    ("趋势与动量", ("momentum", "trend", "breakout", "动量", "趋势")),
    (
        "流动性与交易活跃度",
        ("liquidity", "illiquidity", "activity", "volume stability", "dollar volume"),
    ),
    ("价格成交量关系", ("price-volume", "price volume", "volume-price", "force", "vwap")),
    ("波动率与风险", ("volatility", "variance", "risk", "波动")),
    ("成交量结构", ("volume", "turnover", "dollar volume", "amount")),
    ("多周期与路径结构", ("multi-horizon", "path", "lead-lag", "asymmetry")),
)

# Legacy long-short dimensions are retained as diagnostic scores only. Default
# ranking and strategy-bus leader selection must use the long-only dimensions.
SCORE_DIMENSIONS: dict[str, list[tuple[str, bool, float]]] = {
    "return": [
        ("sharpe_ratio", True, 0.55),
        ("simple_annual_return", True, 0.45),
    ],
    "robustness": [
        ("walk_forward_worst_sharpe", True, 0.35),
        ("walk_forward_positive_fraction", True, 0.25),
        ("deflated_sharpe_probability", True, 0.20),
        ("annual_return_dispersion", False, 0.20),
    ],
    "risk": [
        ("max_drawdown", True, 0.55),
        ("worst_year_incremental_return", True, 0.45),
    ],
    "execution": [
        ("annual_turnover", False, 0.35),
        ("coverage", True, 0.35),
        ("capacity_usd", True, 0.30),
    ],
    "information": [
        ("rank_ic_ir", True, 0.45),
        ("rank_ic_mean", True, 0.30),
        ("pearson_ic_mean", True, 0.25),
    ],
    "long_only_return": [
        ("long_only_active_information_ratio", True, 0.25),
        ("long_only_sharpe_ratio", True, 0.40),
        ("long_only_simple_annual_return", True, 0.25),
        ("long_only_compound_annual_return", True, 0.10),
    ],
    "long_only_robustness": [
        ("long_only_walk_forward_worst_sharpe", True, 0.35),
        ("long_only_walk_forward_positive_fraction", True, 0.25),
        ("long_only_deflated_sharpe_probability", True, 0.20),
        ("long_only_annual_return_dispersion", False, 0.20),
    ],
    "long_only_risk": [
        ("long_only_max_drawdown", True, 0.60),
        ("long_only_worst_year_return", True, 0.40),
    ],
    "long_only_execution": [
        ("long_only_annual_turnover", False, 0.35),
        ("long_only_coverage", True, 0.35),
        ("long_only_capacity_usd", True, 0.30),
    ],
    "recent_long_only_return": [
        ("recent_long_only_sharpe_ratio", True, 0.55),
        ("recent_long_only_simple_annual_return", True, 0.30),
        ("recent_long_only_compound_annual_return", True, 0.15),
    ],
    "recent_long_only_robustness": [
        ("recent_long_only_walk_forward_worst_sharpe", True, 0.35),
        ("recent_long_only_walk_forward_positive_fraction", True, 0.25),
        ("recent_long_only_deflated_sharpe_probability", True, 0.20),
        ("recent_long_only_annual_return_dispersion", False, 0.20),
    ],
    "recent_long_only_risk": [
        ("recent_long_only_max_drawdown", True, 0.60),
        ("recent_long_only_worst_year_return", True, 0.40),
    ],
    "recent_long_only_execution": [
        ("recent_long_only_annual_turnover", False, 0.35),
        ("recent_long_only_coverage", True, 0.35),
        ("recent_long_only_capacity_usd", True, 0.30),
    ],
}

SUMMARY_KEYS = (
    "sharpe_ratio",
    "simple_annual_return",
    "max_drawdown",
    "rank_ic_mean",
    "rank_ic_ir",
    "pearson_ic_mean",
    "annual_turnover",
    "coverage",
    "capacity_usd",
    "walk_forward_positive_fraction",
    "walk_forward_worst_sharpe",
    "annual_return_dispersion",
    "deflated_sharpe_probability",
    "long_only_sharpe_ratio",
    "long_only_simple_annual_return",
    "long_only_compound_annual_return",
    "long_only_max_drawdown",
    "long_only_annual_turnover",
    "long_only_coverage",
    "long_only_capacity_usd",
    "long_only_positive_year_ratio",
    "long_only_worst_year_return",
    "long_only_annual_return_dispersion",
    "long_only_walk_forward_positive_fraction",
    "long_only_walk_forward_folds",
    "long_only_walk_forward_median_sharpe",
    "long_only_walk_forward_worst_sharpe",
    "long_only_walk_forward_worst_drawdown",
    "long_only_deflated_sharpe_probability",
    "long_only_active_information_ratio",
    "long_only_active_simple_annual_return",
    "long_only_active_tracking_error",
    "long_only_market_beta",
    "long_only_active_walk_forward_positive_fraction",
    "long_only_active_walk_forward_worst_sharpe",
    "long_only_benchmark_simple_annual_return",
    "long_only_cost_stress_net_ir",
    "long_only_bankrupt",
    "long_only_bankruptcy_date",
    "exploratory_gate_passed",
    "exploratory_gate_failures",
    "return_convention",
    "long_only_return_convention",
    "recent_long_only_sharpe_ratio",
    "recent_long_only_simple_annual_return",
    "recent_long_only_compound_annual_return",
    "recent_long_only_max_drawdown",
    "recent_long_only_annual_turnover",
    "recent_long_only_coverage",
    "recent_long_only_capacity_usd",
    "recent_long_only_positive_year_ratio",
    "recent_long_only_worst_year_return",
    "recent_long_only_annual_return_dispersion",
    "recent_long_only_walk_forward_positive_fraction",
    "recent_long_only_walk_forward_folds",
    "recent_long_only_walk_forward_median_sharpe",
    "recent_long_only_walk_forward_worst_sharpe",
    "recent_long_only_deflated_sharpe_probability",
    "homogeneity_gate_passed",
    "homogeneity_gate_failure",
    "homogeneity_novelty_score",
    "homogeneity_cluster_id",
    "homogeneity_cluster_size",
    "homogeneity_nearest_factor_id",
    "homogeneity_nearest_similarity",
    "homogeneity_crowded_cluster",
    "homogeneity_redundancy_label",
    "canonical_main_start",
    "canonical_main_end",
    "canonical_recent_start",
    "canonical_recent_end",
)

RANKING_OPTIONS: tuple[dict[str, Any], ...] = (
    {
        "id": "overall",
        "label": "多空诊断综合分",
        "group": "多空诊断复合评分",
        "format": "score",
        "higher_is_better": True,
    },
    {
        "id": "robustness",
        "label": "多空稳健性分",
        "group": "多空诊断复合评分",
        "format": "score",
        "higher_is_better": True,
    },
    {
        "id": "return",
        "label": "多空收益质量分",
        "group": "多空诊断复合评分",
        "format": "score",
        "higher_is_better": True,
    },
    {
        "id": "risk",
        "label": "多空风险控制分",
        "group": "多空诊断复合评分",
        "format": "score",
        "higher_is_better": True,
    },
    {
        "id": "execution",
        "label": "多空可交易性分",
        "group": "多空诊断复合评分",
        "format": "score",
        "higher_is_better": True,
    },
    {
        "id": "information",
        "label": "信息诊断分",
        "group": "多空诊断复合评分",
        "format": "score",
        "higher_is_better": True,
    },
    {
        "id": "long_only_overall",
        "label": "统一主榜 · 稳健综合（2015–2024）",
        "group": "统一主榜（2015–2024）",
        "format": "score",
        "higher_is_better": True,
    },
    {
        "id": "recent_long_only_overall",
        "label": "近期榜 · 稳健综合（2020–2024）",
        "group": "近期榜（2020–2024）",
        "format": "score",
        "higher_is_better": True,
    },
    {
        "id": "long_only_return",
        "label": "统一主榜 · 收益质量",
        "group": "统一主榜（2015–2024）",
        "format": "score",
        "higher_is_better": True,
    },
    {
        "id": "long_only_robustness",
        "label": "统一主榜 · 稳健性",
        "group": "统一主榜（2015–2024）",
        "format": "score",
        "higher_is_better": True,
    },
    {
        "id": "long_only_risk",
        "label": "统一主榜 · 风险控制",
        "group": "统一主榜（2015–2024）",
        "format": "score",
        "higher_is_better": True,
    },
    {
        "id": "long_only_execution",
        "label": "统一主榜 · 可交易性",
        "group": "统一主榜（2015–2024）",
        "format": "score",
        "higher_is_better": True,
    },
    {
        "id": "recent_long_only_return",
        "label": "近期榜 · 收益质量",
        "group": "近期榜（2020–2024）",
        "format": "score",
        "higher_is_better": True,
    },
    {
        "id": "recent_long_only_robustness",
        "label": "近期榜 · 稳健性",
        "group": "近期榜（2020–2024）",
        "format": "score",
        "higher_is_better": True,
    },
    {
        "id": "recent_long_only_risk",
        "label": "近期榜 · 风险控制",
        "group": "近期榜（2020–2024）",
        "format": "score",
        "higher_is_better": True,
    },
    {
        "id": "recent_long_only_execution",
        "label": "近期榜 · 可交易性",
        "group": "近期榜（2020–2024）",
        "format": "score",
        "higher_is_better": True,
    },
    {
        "id": "long_only_active_information_ratio",
        "label": "纯多主动收益 IR",
        "group": "纯多原始指标",
        "format": "number",
        "higher_is_better": True,
    },
    {
        "id": "long_only_active_simple_annual_return",
        "label": "纯多主动年化",
        "group": "纯多原始指标",
        "format": "percent",
        "higher_is_better": True,
    },
    {
        "id": "long_only_sharpe_ratio",
        "label": "纯多夏普",
        "group": "纯多原始指标",
        "format": "number",
        "higher_is_better": True,
    },
    {
        "id": "long_only_simple_annual_return",
        "label": "纯多简单年化",
        "group": "纯多原始指标",
        "format": "percent",
        "higher_is_better": True,
    },
    {
        "id": "long_only_compound_annual_return",
        "label": "纯多复合年化",
        "group": "纯多原始指标",
        "format": "percent",
        "higher_is_better": True,
    },
    {
        "id": "long_only_max_drawdown",
        "label": "纯多最大回撤",
        "group": "纯多原始指标",
        "format": "percent",
        "higher_is_better": True,
    },
    {
        "id": "long_only_walk_forward_worst_sharpe",
        "label": "纯多最差折夏普",
        "group": "纯多原始指标",
        "format": "number",
        "higher_is_better": True,
    },
    {
        "id": "long_only_walk_forward_positive_fraction",
        "label": "纯多正收益窗口",
        "group": "纯多原始指标",
        "format": "percent",
        "higher_is_better": True,
    },
    {
        "id": "long_only_deflated_sharpe_probability",
        "label": "纯多 DSR 概率",
        "group": "纯多原始指标",
        "format": "percent",
        "higher_is_better": True,
    },
    {
        "id": "long_only_annual_turnover",
        "label": "纯多年化换手",
        "group": "纯多原始指标",
        "format": "number",
        "higher_is_better": False,
    },
    {
        "id": "long_only_coverage",
        "label": "纯多覆盖率",
        "group": "纯多原始指标",
        "format": "percent",
        "higher_is_better": True,
    },
    {
        "id": "long_only_capacity_usd",
        "label": "纯多容量",
        "group": "纯多原始指标",
        "format": "currency",
        "higher_is_better": True,
    },
    {
        "id": "recent_long_only_sharpe_ratio",
        "label": "近期纯多夏普",
        "group": "近期榜原始指标（2020–2024）",
        "format": "number",
        "higher_is_better": True,
    },
    {
        "id": "recent_long_only_simple_annual_return",
        "label": "近期纯多简单年化",
        "group": "近期榜原始指标（2020–2024）",
        "format": "percent",
        "higher_is_better": True,
    },
    {
        "id": "recent_long_only_max_drawdown",
        "label": "近期纯多最大回撤",
        "group": "近期榜原始指标（2020–2024）",
        "format": "percent",
        "higher_is_better": True,
    },
    {
        "id": "recent_long_only_walk_forward_worst_sharpe",
        "label": "近期纯多最差折夏普",
        "group": "近期榜原始指标（2020–2024）",
        "format": "number",
        "higher_is_better": True,
    },
    {
        "id": "sharpe_ratio",
        "label": "多空夏普",
        "group": "多空原始指标",
        "format": "number",
        "higher_is_better": True,
    },
    {
        "id": "simple_annual_return",
        "label": "多空简单年化",
        "group": "多空原始指标",
        "format": "percent",
        "higher_is_better": True,
    },
    {
        "id": "max_drawdown",
        "label": "多空最大回撤",
        "group": "多空原始指标",
        "format": "percent",
        "higher_is_better": True,
    },
    {
        "id": "walk_forward_worst_sharpe",
        "label": "多空最差折夏普",
        "group": "多空原始指标",
        "format": "number",
        "higher_is_better": True,
    },
    {
        "id": "walk_forward_positive_fraction",
        "label": "多空正收益窗口",
        "group": "多空原始指标",
        "format": "percent",
        "higher_is_better": True,
    },
    {
        "id": "deflated_sharpe_probability",
        "label": "多空 DSR 概率",
        "group": "多空原始指标",
        "format": "percent",
        "higher_is_better": True,
    },
    {
        "id": "rank_ic_mean",
        "label": "Rank IC",
        "group": "信息与交易",
        "format": "number4",
        "higher_is_better": True,
    },
    {
        "id": "rank_ic_ir",
        "label": "Rank IC IR",
        "group": "信息与交易",
        "format": "number",
        "higher_is_better": True,
    },
    {
        "id": "pearson_ic_mean",
        "label": "Pearson IC",
        "group": "信息与交易",
        "format": "number4",
        "higher_is_better": True,
    },
    {
        "id": "annual_turnover",
        "label": "多空年化换手",
        "group": "信息与交易",
        "format": "number",
        "higher_is_better": False,
    },
    {
        "id": "coverage",
        "label": "多空覆盖率",
        "group": "信息与交易",
        "format": "percent",
        "higher_is_better": True,
    },
    {
        "id": "capacity_usd",
        "label": "多空容量",
        "group": "信息与交易",
        "format": "currency",
        "higher_is_better": True,
    },
    {
        "id": "marginal_incremental_net_ir",
        "label": "组合边际净 IR",
        "group": "信息与交易",
        "format": "number",
        "higher_is_better": True,
    },
    {
        "id": "source_iteration",
        "label": "发现轮次",
        "group": "研究元数据",
        "format": "integer",
        "higher_is_better": False,
    },
)


def factor_category(family: str, name: str = "") -> str:
    value = f"{family} {name}".casefold()
    for category, keywords in CATEGORY_RULES:
        if any(keyword in value for keyword in keywords):
            return category
    return "其他机制"


def build_factor_library(
    records: list[dict[str, Any]],
    *,
    lifecycle_states: dict[str, dict[str, Any]] | None = None,
    contaminated_factor_ids: set[str] | None = None,
    research_diagnostics: dict[str, dict[str, Any]] | None = None,
    current_protocol: str | None = None,
    current_protocols: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a relative, multi-objective leaderboard without discarding rejected factors."""
    factors = []
    lifecycle_states = lifecycle_states or {}
    contaminated_factor_ids = contaminated_factor_ids or set()
    research_diagnostics = research_diagnostics or {}
    current_protocols = current_protocols or {}
    active_ids = {str(record["factor_id"]) for record in records if record["status"] == "ACTIVE"}

    def factor_protocol(record: dict[str, Any]) -> str | None:
        metrics = record.get("metrics", {})
        # A complete central-library reevaluation deliberately applies one
        # standard protocol across source tasks. Online task evidence remains
        # governed by the source task's own protocol.
        if metrics.get("reevaluation_batch_id") and current_protocol:
            return current_protocol
        return current_protocols.get(
            str(record.get("source_task_id", "legacy-ashare")), current_protocol
        )

    percentiles: dict[tuple[str, str], dict[str, float]] = {}
    for dimension, specs in SCORE_DIMENSIONS.items():
        for key, higher_is_better, _ in specs:
            values = {
                str(record["factor_id"]): (
                    _finite(record.get("metrics", {}).get(key))
                    if not factor_protocol(record)
                    or record.get("metrics", {}).get("evaluation_protocol")
                    == factor_protocol(record)
                    else None
                )
                for record in records
            }
            percentiles[(dimension, key)] = _percentile_scores(
                values, higher_is_better=higher_is_better
            )

    for record in records:
        factor_id = str(record["factor_id"])
        scores: dict[str, float] = {}
        for dimension, specs in SCORE_DIMENSIONS.items():
            weighted = 0.0
            available_weight = 0.0
            for key, _, weight in specs:
                if _finite(record.get("metrics", {}).get(key)) is None:
                    continue
                weighted += percentiles[(dimension, key)][factor_id] * weight
                available_weight += weight
            scores[dimension] = weighted / available_weight if available_weight else 0.0
        overall = (
            scores["return"] * 0.25
            + scores["robustness"] * 0.30
            + scores["risk"] * 0.15
            + scores["execution"] * 0.20
            + scores["information"] * 0.10
        )
        long_only_overall = (
            scores["long_only_return"] * 0.30
            + scores["long_only_robustness"] * 0.30
            + scores["long_only_risk"] * 0.20
            + scores["long_only_execution"] * 0.20
        )
        recent_long_only_overall = (
            scores["recent_long_only_return"] * 0.30
            + scores["recent_long_only_robustness"] * 0.30
            + scores["recent_long_only_risk"] * 0.20
            + scores["recent_long_only_execution"] * 0.20
        )
        proposal = record.get("proposal", {})
        metrics = record.get("metrics", {})
        metric_protocol = metrics.get("evaluation_protocol")
        source_task_id = str(record.get("source_task_id", "legacy-ashare"))
        factor_current_protocol = factor_protocol(record)
        protocol_stale = bool(
            factor_current_protocol and metric_protocol != factor_current_protocol
        )
        lifecycle = lifecycle_states.get(factor_id, {})
        diagnostic = research_diagnostics.get(factor_id, {})
        diagnostic_protocol = diagnostic.get("metrics", {}).get("evaluation_protocol")
        marginal = (
            None
            if protocol_stale
            or (factor_current_protocol and diagnostic_protocol != factor_current_protocol)
            else _marginal_contribution(diagnostic)
        )
        historical_metric_summary = {key: metrics.get(key) for key in SUMMARY_KEYS}
        long_only_available = (
            not protocol_stale
            and _finite(metrics.get("long_only_sharpe_ratio")) is not None
            and _finite(metrics.get("long_only_simple_annual_return")) is not None
        )
        recent_long_only_available = (
            not protocol_stale
            and _finite(metrics.get("recent_long_only_sharpe_ratio")) is not None
            and _finite(metrics.get("recent_long_only_simple_annual_return")) is not None
        )
        rounded_scores = {key: round(value, 2) for key, value in scores.items()}
        rounded_scores.update(
            {
                "overall": round(overall, 2),
                "long_only_overall": round(long_only_overall, 2),
                "recent_long_only_overall": round(recent_long_only_overall, 2),
            }
        )
        ranking_values: dict[str, float | int | None] = {}
        for definition in RANKING_OPTIONS:
            ranking_id = str(definition["id"])
            if ranking_id == "source_iteration":
                ranking_values[ranking_id] = int(record["source_iteration"])
            elif ranking_id == "marginal_incremental_net_ir":
                ranking_values[ranking_id] = (
                    marginal.get("incremental_net_ir") if marginal else None
                )
            elif ranking_id in rounded_scores:
                is_long_only_score = ranking_id.startswith(("long_only", "recent_long_only"))
                score_available = (
                    recent_long_only_available
                    if ranking_id.startswith("recent_long_only")
                    else long_only_available
                )
                ranking_values[ranking_id] = (
                    rounded_scores[ranking_id]
                    if not protocol_stale and (score_available or not is_long_only_score)
                    else None
                )
            else:
                ranking_values[ranking_id] = _finite(metrics.get(ranking_id))
        factors.append(
            {
                "factor_id": factor_id,
                "source_task_id": source_task_id,
                "source_iteration": int(record["source_iteration"]),
                "name": str(record.get("name", factor_id)),
                "family": str(record.get("family", "unknown")),
                "canonical_mechanism": normalize_mechanism(
                    proposal.get("canonical_mechanism") or record.get("family")
                ),
                "status": str(record.get("status", "SCREENED_OUT")),
                "status_reason": str(record.get("status_reason", "")),
                "created_at": record.get("created_at"),
                "updated_at": record.get("updated_at"),
                "category": factor_category(
                    str(record.get("family", "")), str(record.get("name", ""))
                ),
                "research_state": (
                    "STALE_PROTOCOL"
                    if protocol_stale
                    else "HOLDOUT_CONTAMINATED"
                    if factor_id in contaminated_factor_ids
                    and str(record.get("status", "")) == "ELIGIBLE"
                    else _research_state(str(record.get("status", "")), factor_id in active_ids)
                ),
                "metric_protocol": metric_protocol,
                "current_protocol": factor_current_protocol,
                "protocol_stale": protocol_stale,
                "promotion_eligible": (
                    str(record.get("status", "")) == "ELIGIBLE"
                    and bool(metrics.get("production_promotion_gate_passed", True))
                    and not protocol_stale
                    and factor_id not in contaminated_factor_ids
                ),
                "metric_evidence_scope": (
                    "HISTORICAL_STALE" if protocol_stale else "CURRENT_PROTOCOL"
                ),
                "library_admission_gate_passed": bool(
                    metrics.get(
                        "library_admission_gate_passed", record.get("status") == "ELIGIBLE"
                    )
                ),
                "production_promotion_gate_passed": bool(
                    metrics.get("production_promotion_gate_passed", True)
                ),
                "production_promotion_gate_failures": list(
                    metrics.get("production_promotion_gate_failures", [])
                ),
                "lifecycle_state": lifecycle.get("state", "RESEARCH"),
                "lifecycle_updated_at": lifecycle.get("created_at"),
                "holdout_contaminated": factor_id in contaminated_factor_ids,
                "marginal_contribution": marginal,
                "scores": {
                    **rounded_scores,
                },
                "ranking_values": ranking_values,
                "long_only_score_available": long_only_available,
                "recent_long_only_score_available": recent_long_only_available,
                "expression": proposal.get("expression"),
                "hypothesis": proposal.get("hypothesis", ""),
                "expected_direction": proposal.get("expected_direction", 1),
                "origin": proposal.get("origin", "AUTO_LLM_RESEARCH"),
                "tags": list(proposal.get("tags", [])),
                "metric_summary": (
                    {key: None for key in SUMMARY_KEYS}
                    if protocol_stale
                    else historical_metric_summary
                ),
                "historical_metric_summary": (
                    historical_metric_summary if protocol_stale else None
                ),
            }
        )

    factors.sort(
        key=lambda item: (
            not item["long_only_score_available"],
            -item["scores"]["long_only_overall"],
            -item["scores"]["recent_long_only_overall"],
            item["source_iteration"],
        )
    )
    _assign_similarity_clusters(factors)
    category_counts = Counter(item["category"] for item in factors)
    category_positions: Counter[str] = Counter()
    for rank, item in enumerate(factors, start=1):
        category_positions[item["category"]] += 1
        item["rank"] = rank
        item["category_rank"] = category_positions[item["category"]]
    return {
        "factors": factors,
        "ranking_options": [
            dict(option)
            for option in sorted(
                RANKING_OPTIONS,
                key=lambda option: (
                    {
                        "统一主榜（2015–2024）": 0,
                        "近期榜（2020–2024）": 1,
                        "纯多原始指标": 2,
                        "近期榜原始指标（2020–2024）": 3,
                        "信息与交易": 4,
                        "多空诊断复合评分": 5,
                        "多空原始指标": 6,
                        "研究元数据": 7,
                    }.get(str(option["group"]), 99),
                    list(RANKING_OPTIONS).index(option),
                ),
            )
        ],
        "categories": [
            {"name": category, "count": count}
            for category, count in sorted(
                category_counts.items(), key=lambda pair: (-pair[1], pair[0])
            )
        ],
        "summary": {
            "factor_count": len(factors),
            "active_count": sum(item["research_state"] == "ACTIVE" for item in factors),
            "eligible_count": sum(
                item["status"] == "ELIGIBLE" and item["promotion_eligible"] for item in factors
            ),
            "public_gate_eligible_count": sum(item["status"] == "ELIGIBLE" for item in factors),
            "observed_count": sum(item["status"] == "SCREENED_OUT" for item in factors),
            "category_count": len(category_counts),
            "cluster_count": len({item["cluster_id"] for item in factors}),
            "shadow_count": sum(item["lifecycle_state"] == "SHADOW" for item in factors),
            "contaminated_count": sum(item["holdout_contaminated"] for item in factors),
            "stale_protocol_count": sum(item["protocol_stale"] for item in factors),
            "long_only_evaluated_count": sum(item["long_only_score_available"] for item in factors),
            "recent_long_only_evaluated_count": sum(
                item["recent_long_only_score_available"] for item in factors
            ),
            "ranking_method": "US_EQUITY_LONG_ONLY_PRIMARY",
            "alpha_diagnostic_ranking_method": (
                "RETURN_25_ROBUSTNESS_30_RISK_15_EXECUTION_20_INFORMATION_10"
            ),
            "long_only_ranking_method": (
                "CANONICAL_2015_2024_RETURN_30_ROBUSTNESS_30_RISK_20_EXECUTION_20"
            ),
            "recent_long_only_ranking_method": (
                "RECENT_2020_2024_RETURN_30_ROBUSTNESS_30_RISK_20_EXECUTION_20"
            ),
        },
    }


def _marginal_contribution(diagnostic: dict[str, Any]) -> dict[str, Any] | None:
    metrics = diagnostic.get("metrics", {})
    options = [
        option
        for option in metrics.get("portfolio_option_diagnostics", [])
        if option.get("action") in {"ADD", "REPLACE"}
    ]
    if not options:
        return None
    best = max(
        options,
        key=lambda option: (
            bool(option.get("accepted")),
            -int(option.get("gate_failure_count", 999)),
            float(option.get("utility_change", -1e9)),
        ),
    )
    values = best.get("metrics", {})
    return {
        "action": best.get("action"),
        "accepted": bool(best.get("accepted")),
        "utility_change": _finite(best.get("utility_change")),
        "incremental_net_ir": _finite(values.get("portfolio_incremental_net_ir")),
        "annual_return_change": _finite(values.get("portfolio_annual_return_change")),
        "sharpe_change": _finite(values.get("portfolio_sharpe_change")),
        "max_drawdown_change": _finite(values.get("portfolio_max_drawdown_change")),
        "maximum_correlation": _finite(values.get("portfolio_max_factor_correlation")),
        "failed_gates": list(best.get("failed_gates", [])),
        "iteration": diagnostic.get("iteration"),
    }


def _assign_similarity_clusters(factors: list[dict[str, Any]]) -> None:
    parents = list(range(len(factors)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = root(left), root(right)
        if left_root != right_root:
            parents[right_root] = left_root

    signatures = [_expression_signature(item.get("expression")) for item in factors]
    for left in range(len(factors)):
        for right in range(left + 1, len(factors)):
            if factors[left]["category"] != factors[right]["category"]:
                continue
            union_score = _jaccard(signatures[left], signatures[right])
            same_family = factors[left]["family"].casefold() == factors[right]["family"].casefold()
            if union_score >= 0.82 or (same_family and union_score >= 0.67):
                union(left, right)
    members: dict[int, list[int]] = {}
    for index in range(len(factors)):
        members.setdefault(root(index), []).append(index)
    ordered = sorted(members.values(), key=lambda group: min(group))
    for cluster_number, group in enumerate(ordered, start=1):
        leader = max(
            group,
            key=lambda index: (
                factors[index]["long_only_score_available"],
                factors[index]["scores"]["long_only_overall"],
                factors[index]["scores"]["overall"],
            ),
        )
        cluster_id = f"C{cluster_number:03d}"
        for index in group:
            peer_scores = [
                (_jaccard(signatures[index], signatures[peer]), peer)
                for peer in group
                if peer != index
            ]
            nearest = max(peer_scores, default=(0.0, None))
            factors[index].update(
                {
                    "cluster_id": cluster_id,
                    "cluster_size": len(group),
                    "cluster_role": "LEADER" if index == leader else "MEMBER",
                    "nearest_factor_id": (
                        factors[nearest[1]]["factor_id"] if nearest[1] is not None else None
                    ),
                    "nearest_similarity": round(nearest[0], 4),
                    "similarity_method": "EXPRESSION_STRUCTURE_AND_MECHANISM_V1",
                }
            )


def _expression_signature(expression: Any) -> set[str]:
    if not isinstance(expression, dict):
        return set()
    operator = str(expression.get("operator", "unknown"))
    parameters = expression.get("parameters", {})
    signature = {f"operator:{operator}"}
    if operator == "field":
        signature.add(f"field:{parameters.get('name', 'unknown')}")
    for key in ("window", "periods"):
        if key in parameters:
            value = float(parameters[key])
            bucket = "short" if value <= 10 else "medium" if value <= 30 else "long"
            signature.add(f"horizon:{bucket}")
    for argument in expression.get("arguments", []):
        signature.update(_expression_signature(argument))
    return signature


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _research_state(status: str, active: bool) -> str:
    if active:
        return "ACTIVE"
    if status == "ELIGIBLE":
        return "QUALIFIED"
    return "OBSERVE"


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _percentile_scores(
    values: dict[str, float | None], *, higher_is_better: bool
) -> dict[str, float]:
    available = sorted(value for value in values.values() if value is not None)
    if not available:
        return {key: 0.0 for key in values}
    scores = {}
    denominator = max(1, len(available) - 1)
    for key, value in values.items():
        if value is None:
            scores[key] = 0.0
            continue
        lower = sum(candidate < value for candidate in available)
        equal = sum(candidate == value for candidate in available)
        percentile = (lower + (equal - 1) / 2) / denominator if len(available) > 1 else 0.5
        if not higher_is_better:
            percentile = 1.0 - percentile
        scores[key] = float(max(0.0, min(100.0, percentile * 100)))
    return scores
