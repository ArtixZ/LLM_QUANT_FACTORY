from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from autoalpha.service.mechanism import normalize_mechanism

NORMALIZATION_OPERATORS = {
    "cs_rank",
    "cs_zscore",
    "winsorize_mad",
    "winsorize_quantile",
}

ORDER_FLOW_FIELDS = {
    "net_mf_amount",
    "buy_sm_amount",
    "sell_sm_amount",
    "buy_md_amount",
    "sell_md_amount",
    "buy_lg_amount",
    "sell_lg_amount",
    "buy_elg_amount",
    "sell_elg_amount",
}
LIQUIDITY_FIELDS = {
    "amount",
    "vol",
    "volume",
    "turnover_rate",
    "turnover_rate_f",
    "volume_ratio",
}
CAPITAL_SUPPLY_FIELDS = {
    "free_share",
    "float_share",
    "total_share",
    "circ_mv",
    "total_mv",
}
VALUE_FIELDS = {
    "pe",
    "pe_ttm",
    "pb",
    "ps",
    "ps_ttm",
    "dv_ratio",
    "dv_ttm",
}
QUALITY_FIELDS = {
    "roe",
    "roe_dt",
    "roa",
    "grossprofit_margin",
    "netprofit_margin",
    "debt_to_assets",
    "ocf_to_or",
}
PRICE_FIELDS = {"open", "high", "low", "close", "adj_close", "pre_close"}


def enrich_factor_record(record: dict[str, Any]) -> dict[str, Any]:
    result = dict(record)
    proposal = result.get("proposal") or {}
    expression = proposal.get("expression") or {}
    fields = sorted(expression_fields(expression))
    windows = sorted(expression_windows(expression))
    mechanism = normalized_mechanism(proposal, fields)
    fingerprint = mechanism_fingerprint(
        expression, mechanism, expected_direction=int(proposal.get("expected_direction", 1))
    )
    semantic_cluster_id = f"SC_{fingerprint[:10]}"
    behavior_cluster_id = _clean_optional_identifier(result.get("behavior_cluster_id"))
    result.update(
        {
            "mechanism": mechanism,
            "mechanism_fingerprint": fingerprint,
            "semantic_cluster_id": semantic_cluster_id,
            "behavior_cluster_id": behavior_cluster_id,
            "search_cluster_id": behavior_cluster_id or semantic_cluster_id,
            "expression_fields": fields,
            "expression_windows": windows,
            "expression_summary": expression_summary(expression),
        }
    )
    return result


def factor_snapshot_homogeneity_summary(snapshot: list[dict[str, Any]]) -> dict[str, Any]:
    factor_count = len(snapshot)
    behavior_clusters = {
        str(item.get("behavior_cluster_id"))
        for item in snapshot
        if item.get("behavior_cluster_id")
    }
    semantic_clusters = {
        str(item.get("semantic_cluster_id"))
        for item in snapshot
        if item.get("semantic_cluster_id")
    }
    search_clusters = {
        str(item.get("search_cluster_id") or item.get("semantic_cluster_id"))
        for item in snapshot
        if item.get("search_cluster_id") or item.get("semantic_cluster_id")
    }
    crowded_search_clusters: dict[str, int] = {}
    for item in snapshot:
        cluster_id = str(item.get("search_cluster_id") or item.get("semantic_cluster_id") or "")
        if not cluster_id:
            continue
        crowded_search_clusters[cluster_id] = crowded_search_clusters.get(cluster_id, 0) + 1
    crowded_search_clusters = {
        cluster_id: count
        for cluster_id, count in sorted(
            crowded_search_clusters.items(), key=lambda value: (-value[1], value[0])
        )
        if count > 1
    }
    return {
        "protocol": "AUTOCOMBINE_FACTOR_SNAPSHOT_HOMOGENEITY_SUMMARY_V1",
        "factor_count": factor_count,
        "behavior_cluster_count": len(behavior_clusters),
        "semantic_cluster_count": len(semantic_clusters),
        "search_cluster_count": len(search_clusters),
        "behavior_cluster_coverage": (
            round(sum(1 for item in snapshot if item.get("behavior_cluster_id")) / factor_count, 6)
            if factor_count
            else 1.0
        ),
        "search_space_compression_ratio": (
            round(len(search_clusters) / factor_count, 6) if factor_count else 1.0
        ),
        "duplicate_search_cluster_factor_count": sum(
            max(0, count - 1) for count in crowded_search_clusters.values()
        ),
        "crowded_search_clusters": dict(list(crowded_search_clusters.items())[:12]),
    }


def expression_fields(expression: dict[str, Any]) -> set[str]:
    fields: set[str] = set()
    if str(expression.get("operator", "")) == "field":
        name = (expression.get("parameters") or {}).get("name")
        if name:
            fields.add(str(name))
    for argument in expression.get("arguments") or []:
        if isinstance(argument, dict):
            fields.update(expression_fields(argument))
    return fields


def expression_windows(expression: dict[str, Any]) -> set[int]:
    windows: set[int] = set()
    parameters = expression.get("parameters") or {}
    for key in ("window", "periods", "lookback", "span"):
        value = parameters.get(key)
        if isinstance(value, int | float) and int(value) > 0:
            windows.add(int(value))
    for argument in expression.get("arguments") or []:
        if isinstance(argument, dict):
            windows.update(expression_windows(argument))
    return windows


def normalized_mechanism(proposal: dict[str, Any], fields: list[str]) -> str:
    explicit = normalize_mechanism(proposal.get("canonical_mechanism"), default="")
    if explicit:
        return explicit
    field_set = set(fields)
    text = " ".join(
        str(proposal.get(key, "")) for key in ("name", "family", "hypothesis")
    ).casefold()
    if field_set & ORDER_FLOW_FIELDS:
        return normalize_mechanism("ORDER_FLOW")
    if field_set & QUALITY_FIELDS:
        return normalize_mechanism("QUALITY")
    if field_set & VALUE_FIELDS:
        return normalize_mechanism("VALUE")
    if field_set & CAPITAL_SUPPLY_FIELDS and not field_set & LIQUIDITY_FIELDS:
        return normalize_mechanism("CAPITAL_SUPPLY")
    if field_set & LIQUIDITY_FIELDS:
        return normalize_mechanism("LIQUIDITY_ACTIVITY")
    if any(token in text for token in ("reversal", "mean revert", "反转", "均值回归")):
        return normalize_mechanism("PRICE_REVERSAL")
    if any(token in text for token in ("momentum", "trend", "动量", "趋势")):
        return normalize_mechanism("PRICE_TREND")
    if any(token in text for token in ("volatility", "波动率", "low vol")):
        return normalize_mechanism("VOLATILITY")
    if field_set and field_set <= PRICE_FIELDS:
        return normalize_mechanism("PRICE_ACTION")
    return normalize_mechanism("OTHER")


def mechanism_fingerprint(
    expression: dict[str, Any], mechanism: str, *, expected_direction: int = 1
) -> str:
    body = {
        "mechanism": mechanism,
        "expected_direction": 1 if expected_direction >= 0 else -1,
        "structure": _expression_structure(expression),
    }
    encoded = json.dumps(body, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _expression_structure(expression: dict[str, Any]) -> Any:
    operator = str(expression.get("operator", "unknown"))
    arguments = [
        _expression_structure(argument)
        for argument in expression.get("arguments") or []
        if isinstance(argument, dict)
    ]
    if operator in NORMALIZATION_OPERATORS:
        return arguments[0] if arguments else operator
    parameters = expression.get("parameters") or {}
    stable_parameters = {
        key: parameters[key]
        for key in sorted(parameters)
        if key in {"name", "window", "periods", "lookback", "span"}
    }
    return [operator, stable_parameters, arguments]


def expression_summary(expression: dict[str, Any], *, maximum_length: int = 260) -> str:
    operator = str(expression.get("operator", "unknown"))
    parameters = expression.get("parameters") or {}
    if operator == "field":
        return str(parameters.get("name", "field"))
    arguments = [
        expression_summary(argument, maximum_length=maximum_length)
        for argument in expression.get("arguments") or []
        if isinstance(argument, dict)
    ]
    suffix = ",".join(
        f"{key}={parameters[key]}"
        for key in ("window", "periods", "lookback", "span", "threshold")
        if key in parameters
    )
    body = f"{operator}({', '.join(arguments)}{'; ' if suffix and arguments else ''}{suffix})"
    return body if len(body) <= maximum_length else f"{body[: maximum_length - 3]}..."


def _clean_optional_identifier(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def signal_independence_metrics(
    factor_ids: list[str], correlations: dict[str, float]
) -> dict[str, float | int]:
    count = len(factor_ids)
    if count <= 1:
        return {
            "portfolio_average_factor_correlation": 0.0,
            "portfolio_effective_factor_bets": 1.0,
            "portfolio_high_correlation_pair_count": 0,
        }
    matrix = np.eye(count, dtype=float)
    values: list[float] = []
    index = {factor_id: position for position, factor_id in enumerate(factor_ids)}
    for key, raw_value in correlations.items():
        left, right = key.split(":", 1)
        if left not in index or right not in index:
            continue
        value = float(np.clip(raw_value, -1.0, 1.0))
        matrix[index[left], index[right]] = value
        matrix[index[right], index[left]] = value
        values.append(abs(value))
    eigenvalues = np.clip(np.linalg.eigvalsh(matrix), 0.0, None)
    denominator = float(np.square(eigenvalues).sum())
    effective = float(np.square(eigenvalues.sum()) / denominator) if denominator else 1.0
    return {
        "portfolio_average_factor_correlation": float(np.mean(values)) if values else 0.0,
        "portfolio_effective_factor_bets": effective,
        "portfolio_high_correlation_pair_count": sum(value >= 0.70 for value in values),
    }


def mechanism_independence_metrics(
    records: dict[str, dict[str, Any]], factor_ids: list[str], weights: list[float]
) -> dict[str, Any]:
    mechanism_weights: dict[str, float] = defaultdict(float)
    fingerprint_counts: dict[str, int] = defaultdict(int)
    for factor_id, weight in zip(factor_ids, weights, strict=True):
        record = records[factor_id]
        mechanism_weights[str(record.get("mechanism", "OTHER"))] += float(weight)
        fingerprint_counts[str(record.get("mechanism_fingerprint", factor_id))] += 1
    hhi = sum(value * value for value in mechanism_weights.values())
    return {
        "portfolio_mechanism_weights": dict(sorted(mechanism_weights.items())),
        "portfolio_mechanism_count": len(mechanism_weights),
        "portfolio_effective_mechanisms": 1.0 / hhi if hhi > 0 else 1.0,
        "portfolio_maximum_mechanism_weight": max(mechanism_weights.values(), default=1.0),
        "portfolio_duplicate_semantic_factor_count": sum(
            max(0, count - 1) for count in fingerprint_counts.values()
        ),
    }


def write_return_artifact(
    root: Path,
    *,
    task_id: str,
    candidate_hash: str,
    net_returns: pd.Series,
    active_returns: pd.Series,
) -> tuple[str, str]:
    directory = root / task_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{candidate_hash}.parquet"
    frame = pd.DataFrame(
        {
            "net_return": net_returns.astype(float),
            "active_return": active_returns.reindex(net_returns.index).astype(float),
        }
    )
    frame.index.name = "trade_date"
    frame.to_parquet(path, compression="zstd")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return str(path.resolve()), digest


def load_return_artifact(path: str | Path) -> pd.DataFrame:
    frame = pd.read_parquet(Path(path))
    if not {"net_return", "active_return"} <= set(frame.columns):
        raise ValueError("AutoCombine return artifact has an invalid schema")
    return frame.sort_index()


def return_independence(left: pd.Series, right: pd.Series) -> dict[str, float | int]:
    frame = pd.concat([left.rename("left"), right.rename("right")], axis=1, join="inner").dropna()
    if len(frame) < 60:
        return {"pearson": 1.0, "spearman": 1.0, "observations": len(frame)}
    return {
        "pearson": float(frame["left"].corr(frame["right"], method="pearson")),
        "spearman": float(frame["left"].corr(frame["right"], method="spearman")),
        "observations": len(frame),
    }


def public_metric_bands(metrics: dict[str, Any]) -> dict[str, str]:
    return {
        "return": _band(float(metrics.get("portfolio_simple_annual_return", 0.0)), 0.05, 0.15),
        "sharpe": _band(float(metrics.get("portfolio_sharpe_ratio", 0.0)), 0.6, 1.2),
        "active_ir": _band(float(metrics.get("portfolio_active_information_ratio", 0.0)), 0.2, 0.6),
        "drawdown": _inverse_band(
            abs(float(metrics.get("portfolio_max_drawdown", -1.0))), 0.25, 0.15
        ),
        "worst_fold": _band(
            float(metrics.get("portfolio_walk_forward_worst_sharpe", -10.0)), -0.5, 0.0
        ),
        "independence": _inverse_band(
            float(metrics.get("portfolio_max_factor_correlation", 1.0)), 0.85, 0.65
        ),
    }


def _band(value: float, acceptable: float, strong: float) -> str:
    if not math.isfinite(value) or value < acceptable:
        return "WEAK"
    return "STRONG" if value >= strong else "ACCEPTABLE"


def _inverse_band(value: float, acceptable: float, strong: float) -> str:
    if not math.isfinite(value) or value > acceptable:
        return "WEAK"
    return "STRONG" if value <= strong else "ACCEPTABLE"
