from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from autoalpha.data.workspace import inspect_data_workspace
from autoalpha.dsl.compiler import FactorCompiler
from autoalpha.dsl.expression import Expression
from autoalpha.dsl.semantics import SemanticValidator
from autoalpha.service.batch_engine import MassiveBatchConfig
from autoalpha.service.factor_library import build_factor_library
from autoalpha.service.realistic_batch_engine import (
    RealisticAshareBatchConfig,
    RealisticAshareBatchEngine,
)

BEHAVIOR_PROTOCOL = "A_SHARE_FACTOR_BEHAVIOR_CLUSTER_V1"
DEFAULT_START = date(2015, 1, 1)
DEFAULT_END = date(2024, 12, 31)


@dataclass(frozen=True)
class BehaviorRunConfig:
    data_path: Path
    config_path: Path
    database_path: Path
    output_root: Path
    start_date: date = DEFAULT_START
    end_date: date = DEFAULT_END
    signal_projections: int = 12
    cluster_threshold: float = 0.74

    @property
    def artifact_root(self) -> Path:
        return self.output_root / "artifacts"

    @property
    def progress_path(self) -> Path:
        return self.output_root / "progress.json"

    @property
    def snapshot_path(self) -> Path:
        return self.output_root / "latest.json"

    @property
    def snapshot_root(self) -> Path:
        return self.output_root / "snapshots"


def load_behavior_snapshot(output_root: Path) -> dict[str, Any]:
    snapshot_path = output_root / "latest.json"
    progress_path = output_root / "progress.json"
    snapshot = _read_json(snapshot_path, {})
    progress = _read_json(progress_path, {})
    if not snapshot:
        return {
            "status": progress.get("status", "NOT_STARTED"),
            "protocol": BEHAVIOR_PROTOCOL,
            "progress": progress,
            "factors": {},
            "clusters": [],
        }
    snapshot["progress"] = progress
    return snapshot


class FactorBehaviorRunner:
    """Resumable full-library vector fingerprints and behavior clustering."""

    def __init__(self, config: BehaviorRunConfig, records: list[dict[str, Any]]) -> None:
        self.config = config
        self.records = records
        self.config.output_root.mkdir(parents=True, exist_ok=True)
        self.config.artifact_root.mkdir(parents=True, exist_ok=True)
        workspace = inspect_data_workspace(config.data_path)
        end = min(config.end_date, date.fromisoformat(workspace.last_trade_date))
        batch_config = RealisticAshareBatchConfig(
            **MassiveBatchConfig(
                data_path=config.data_path,
                config_path=config.config_path,
                start_date=config.start_date,
                end_date=end,
                workers=1,
                monte_carlo_samples=1_000,
            ).__dict__
        )
        self.engine = RealisticAshareBatchEngine(
            batch_config,
            config.output_root / "scratch",
            factor_family_size=max(1, len(records)),
        )
        self.compiler = FactorCompiler(
            SemanticValidator(
                self.engine.validator_fields,
                maximum_nodes=30,
                maximum_lookback=252,
            )
        )
        self.projector = self._projector(self.engine.fields["open"].columns)

    def run(self, *, resume: bool = True) -> dict[str, Any]:
        existing = _read_json(self.config.progress_path, {}) if resume else {}
        completed = set(existing.get("completed_factor_ids", []))
        failures = dict(existing.get("failures", {}))
        started_at = existing.get("started_at") or _now()
        self._write_progress(
            "RUNNING",
            completed,
            failures,
            started_at=started_at,
            current_factor_id=None,
        )
        for record in self.records:
            factor_id = str(record["factor_id"])
            artifact = self.config.artifact_root / f"{factor_id}.npz"
            if factor_id in completed and artifact.exists():
                continue
            self._write_progress(
                "RUNNING",
                completed,
                failures,
                started_at=started_at,
                current_factor_id=factor_id,
            )
            try:
                self._evaluate_factor(record, artifact)
                completed.add(factor_id)
                failures.pop(factor_id, None)
            except Exception as error:  # noqa: BLE001
                failures[factor_id] = f"{type(error).__name__}: {error}"
            self._write_progress(
                "RUNNING",
                completed,
                failures,
                started_at=started_at,
                current_factor_id=None,
            )
        try:
            snapshot = self._cluster(completed, failures, started_at)
            self.config.snapshot_root.mkdir(parents=True, exist_ok=True)
            _write_json_atomic(
                self.config.snapshot_root / f"{snapshot['snapshot_id']}.json",
                snapshot,
            )
            _write_json_atomic(self.config.snapshot_path, snapshot)
        except Exception:
            self._write_progress(
                "FAILED",
                completed,
                failures,
                started_at=started_at,
                current_factor_id=None,
            )
            raise
        self._write_progress(
            "COMPLETED",
            completed,
            failures,
            started_at=started_at,
            current_factor_id=None,
        )
        return snapshot

    def _evaluate_factor(self, record: dict[str, Any], artifact: Path) -> None:
        proposal = record["proposal"]
        expression = Expression.from_dict(proposal["expression"])
        signal = self.compiler.evaluate(expression, self.engine.fields)
        signal = signal * int(proposal.get("expected_direction", 1))
        outcome = self.engine._run_protocol(signal)  # noqa: SLF001
        rebalance_dates = outcome.path.index[outcome.path["rebalance"].gt(0)]
        lagged = signal.shift(1).reindex(rebalance_dates)
        ranked = lagged.rank(axis=1, method="average", pct=True).sub(0.5)
        coverage = ranked.notna().sum(axis=1).clip(lower=1).to_numpy(dtype=np.float32)
        sketch = ranked.fillna(0.0).to_numpy(dtype=np.float32) @ self.projector
        sketch /= np.sqrt(coverage[:, None])
        temporary = artifact.with_suffix(".tmp.npz")
        np.savez(
            temporary,
            net=outcome.path["net"].to_numpy(dtype=np.float32),
            signal_sketch=sketch.astype(np.float32),
            dates=outcome.path.index.to_numpy(dtype="datetime64[D]"),
            rebalance_dates=rebalance_dates.to_numpy(dtype="datetime64[D]"),
        )
        os.replace(temporary, artifact)

    def _projector(self, columns: pd.Index) -> np.ndarray:
        seed_material = "|".join(map(str, columns)).encode()
        seed = int(hashlib.sha256(seed_material).hexdigest()[:16], 16)
        rng = np.random.default_rng(seed)
        values = rng.choice(
            np.array([-1.0, 1.0], dtype=np.float32),
            size=(len(columns), self.config.signal_projections),
        )
        return values / math.sqrt(max(1, len(columns)))

    def _cluster(
        self,
        completed: set[str],
        failures: dict[str, str],
        started_at: str,
    ) -> dict[str, Any]:
        ordered = [record for record in self.records if str(record["factor_id"]) in completed]
        factor_ids = [str(record["factor_id"]) for record in ordered]
        returns = []
        sketches: list[np.ndarray] = []
        sketch_dates: list[np.ndarray] = []
        for factor_id in factor_ids:
            with np.load(self.config.artifact_root / f"{factor_id}.npz") as artifact:
                returns.append(np.asarray(artifact["net"], dtype=np.float64))
                sketches.append(np.asarray(artifact["signal_sketch"], dtype=np.float64))
                sketch_dates.append(np.asarray(artifact["rebalance_dates"], dtype="datetime64[D]"))
        if not factor_ids:
            raise RuntimeError("No factor behavior artifacts were produced")
        return_matrix = np.vstack(returns)
        residual_returns = return_matrix - np.nanmedian(return_matrix, axis=0, keepdims=True)
        signal_matrix = _align_signal_sketches(sketches, sketch_dates)
        return_correlation = _row_correlation(residual_returns)
        signal_correlation = _row_correlation(signal_matrix)
        similarity = np.clip(
            0.65 * np.maximum(signal_correlation, 0.0)
            + 0.35 * np.maximum(return_correlation, 0.0),
            0.0,
            1.0,
        )
        np.fill_diagonal(similarity, 1.0)
        labels = _hierarchical_labels(similarity, self.config.cluster_threshold)
        library = build_factor_library(ordered)
        library_lookup = {item["factor_id"]: item for item in library["factors"]}
        groups: dict[int, list[int]] = {}
        for index, label in enumerate(labels):
            groups.setdefault(int(label), []).append(index)
        factors: dict[str, dict[str, Any]] = {}
        clusters: list[dict[str, Any]] = []
        for members in groups.values():
            member_ids = [factor_ids[index] for index in members]
            member_hash = hashlib.sha256("|".join(sorted(member_ids)).encode()).hexdigest()
            cluster_id = "B_" + member_hash[:8]
            leader = max(members, key=lambda index: _leader_key(library_lookup[factor_ids[index]]))
            categories = Counter(library_lookup[factor_ids[index]]["category"] for index in members)
            dominant_category = categories.most_common(1)[0][0]
            cluster_similarity = [
                float(similarity[left, right])
                for offset, left in enumerate(members)
                for right in members[offset + 1 :]
            ]
            clusters.append(
                {
                    "cluster_id": cluster_id,
                    "label": f"{dominant_category}行为簇",
                    "size": len(members),
                    "leader_factor_id": factor_ids[leader],
                    "dominant_category": dominant_category,
                    "mean_internal_similarity": (
                        round(float(np.mean(cluster_similarity)), 4)
                        if cluster_similarity
                        else 1.0
                    ),
                }
            )
            for index in members:
                peers = [peer for peer in range(len(factor_ids)) if peer != index]
                nearest = max(peers, key=lambda peer: similarity[index, peer]) if peers else None
                nearest_similarity = (
                    float(similarity[index, nearest]) if nearest is not None else 0.0
                )
                nearest_signal = (
                    float(signal_correlation[index, nearest]) if nearest is not None else 0.0
                )
                nearest_return = (
                    float(return_correlation[index, nearest]) if nearest is not None else 0.0
                )
                factors[factor_ids[index]] = {
                    "behavior_cluster_id": cluster_id,
                    "behavior_cluster_label": f"{dominant_category}行为簇",
                    "behavior_cluster_size": len(members),
                    "behavior_cluster_role": "LEADER" if index == leader else "MEMBER",
                    "behavior_nearest_factor_id": (
                        factor_ids[nearest] if nearest is not None else None
                    ),
                    "behavior_nearest_similarity": round(nearest_similarity, 4),
                    "behavior_signal_correlation": round(nearest_signal, 4),
                    "behavior_return_correlation": round(nearest_return, 4),
                    "behavior_redundancy": _redundancy_label(
                        nearest_similarity, nearest_signal, nearest_return, len(members)
                    ),
                    "behavior_cluster_method": BEHAVIOR_PROTOCOL,
                }
        clusters.sort(key=lambda item: (-item["size"], item["cluster_id"]))
        snapshot_id = "behavior-" + hashlib.sha256(
            "|".join(
                [
                    BEHAVIOR_PROTOCOL,
                    self.engine.workspace.fingerprint,
                    self.config.start_date.isoformat(),
                    self.engine.config.end_date.isoformat(),
                    f"{self.config.cluster_threshold:.6f}",
                    *factor_ids,
                ]
            ).encode()
        ).hexdigest()[:16]
        return {
            "snapshot_id": snapshot_id,
            "status": "COMPLETED",
            "protocol": BEHAVIOR_PROTOCOL,
            "created_at": _now(),
            "started_at": started_at,
            "data_path": str(self.config.data_path.resolve()),
            "data_fingerprint": self.engine.workspace.fingerprint,
            "period_start": self.config.start_date.isoformat(),
            "period_end": self.engine.config.end_date.isoformat(),
            "factor_count": len(self.records),
            "evaluated_count": len(factor_ids),
            "failed_count": len(failures),
            "cluster_count": len(clusters),
            "cluster_threshold": self.config.cluster_threshold,
            "signal_weight": 0.65,
            "residual_return_weight": 0.35,
            "execution_protocol": self.engine.config.protocol,
            "alignment_contract": {
                "target_selection": "SHARED_TARGET_BOOK_V1",
                "signal_time": "END_OF_DAY_T",
                "execution_time": "NEXT_SCHEDULED_OPEN",
                "event_engine_is_gold_standard": True,
                "vector_scope": "FULL_LIBRARY_BEHAVIOR_SCREEN",
                "event_scope": "CLUSTER_LEADERS_AND_DISAGREEMENTS",
            },
            "failures": failures,
            "clusters": clusters,
            "factors": factors,
        }

    def _write_progress(
        self,
        status: str,
        completed: set[str],
        failures: dict[str, str],
        *,
        started_at: str,
        current_factor_id: str | None,
    ) -> None:
        _write_json_atomic(
            self.config.progress_path,
            {
                "status": status,
                "protocol": BEHAVIOR_PROTOCOL,
                "started_at": started_at,
                "updated_at": _now(),
                "total_count": len(self.records),
                "completed_count": len(completed),
                "failed_count": len(failures),
                "current_factor_id": current_factor_id,
                "completed_factor_ids": sorted(completed),
                "failures": failures,
            },
        )


def _row_correlation(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    matrix -= matrix.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 1e-12)
    result = np.clip(normalized @ normalized.T, -1.0, 1.0)
    np.fill_diagonal(result, 1.0)
    return result


def _align_signal_sketches(
    sketches: list[np.ndarray], dates: list[np.ndarray]
) -> np.ndarray:
    if not sketches or len(sketches) != len(dates):
        raise ValueError("Signal sketches and date indexes must be non-empty and aligned")
    projections = sketches[0].shape[1]
    if any(sketch.ndim != 2 or sketch.shape[1] != projections for sketch in sketches):
        raise ValueError("Signal sketches use inconsistent projection dimensions")
    calendar = np.unique(np.concatenate(dates))
    positions = {value: index for index, value in enumerate(calendar)}
    aligned = np.zeros((len(sketches), len(calendar), projections), dtype=np.float64)
    for factor_index, (sketch, factor_dates) in enumerate(zip(sketches, dates, strict=True)):
        if len(sketch) != len(factor_dates):
            raise ValueError("Signal sketch rows do not match its rebalance calendar")
        target = np.fromiter(
            (positions[value] for value in factor_dates),
            dtype=np.int64,
            count=len(factor_dates),
        )
        aligned[factor_index, target, :] = sketch
    return aligned.reshape(len(sketches), -1)


def _hierarchical_labels(similarity: np.ndarray, threshold: float) -> np.ndarray:
    if len(similarity) == 1:
        return np.ones(1, dtype=int)
    distance = np.clip(1.0 - similarity, 0.0, 2.0)
    np.fill_diagonal(distance, 0.0)
    condensed = squareform(distance, checks=False)
    tree = linkage(condensed, method="average", optimal_ordering=False)
    return fcluster(tree, t=1.0 - threshold, criterion="distance")


def _leader_key(factor: dict[str, Any]) -> tuple[bool, float, float, int]:
    return (
        bool(factor.get("long_only_score_available")),
        float(factor.get("scores", {}).get("long_only_overall", 0.0)),
        float(factor.get("scores", {}).get("overall", 0.0)),
        -int(factor.get("source_iteration", 0)),
    )


def _redundancy_label(
    similarity: float,
    signal_correlation: float,
    return_correlation: float,
    cluster_size: int,
) -> str:
    if signal_correlation >= 0.95 and return_correlation >= 0.90:
        return "NEAR_DUPLICATE"
    if similarity >= 0.82:
        return "SUBSTITUTE"
    if cluster_size > 1:
        return "RELATED"
    return "DISTINCT"


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _now() -> str:
    return datetime.now(UTC).isoformat()
