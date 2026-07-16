from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from autoalpha.dsl.expression import FactorDefinition


@dataclass(frozen=True)
class FactorCard:
    factor_id: str
    version: int
    name: str
    family: str
    hypothesis: str
    expression: dict[str, Any]
    expression_hash: str
    data_dependencies: tuple[str, ...]
    data_lag_days: int
    applicable_regimes: tuple[str, ...]
    failure_modes: tuple[str, ...]
    owner: str
    parent_ids: tuple[str, ...] = ()
    experiment_id: str = ""
    artifact_hash: str = ""


class FactorRegistry:
    def __init__(self, root: Path) -> None:
        self.root = root

    def publish(
        self,
        factor: FactorDefinition,
        *,
        data_dependencies: tuple[str, ...],
        data_lag_days: int,
        applicable_regimes: tuple[str, ...],
        failure_modes: tuple[str, ...],
        owner: str,
        experiment_id: str,
        parent_ids: tuple[str, ...] = (),
    ) -> FactorCard:
        existing = self.versions(factor.factor_id)
        version = len(existing) + 1
        base = FactorCard(
            factor_id=factor.factor_id,
            version=version,
            name=factor.name,
            family=factor.family,
            hypothesis=factor.hypothesis,
            expression=factor.expression.to_dict(),
            expression_hash=factor.expression.expression_hash,
            data_dependencies=tuple(sorted(data_dependencies)),
            data_lag_days=data_lag_days,
            applicable_regimes=applicable_regimes,
            failure_modes=failure_modes,
            owner=owner,
            parent_ids=parent_ids,
            experiment_id=experiment_id,
        )
        payload = asdict(base)
        payload["artifact_hash"] = _payload_hash(payload)
        card = FactorCard(**payload)
        path = self._path(card.factor_id, card.version)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        return card

    def versions(self, factor_id: str) -> tuple[FactorCard, ...]:
        directory = self.root / factor_id
        if not directory.exists():
            return ()
        cards = []
        for path in sorted(directory.glob("v*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            artifact_hash = payload.pop("artifact_hash")
            if _payload_hash(payload) != artifact_hash:
                raise RuntimeError(f"Factor artifact was modified: {path}")
            payload["artifact_hash"] = artifact_hash
            for field in (
                "data_dependencies",
                "applicable_regimes",
                "failure_modes",
                "parent_ids",
            ):
                payload[field] = tuple(payload[field])
            cards.append(FactorCard(**payload))
        return tuple(cards)

    def find_by_expression(self, expression_hash: str) -> tuple[FactorCard, ...]:
        return tuple(
            card
            for directory in self.root.glob("F_*")
            for card in self.versions(directory.name)
            if card.expression_hash == expression_hash
        )

    def _path(self, factor_id: str, version: int) -> Path:
        return self.root / factor_id / f"v{version:04d}.json"


def _payload_hash(payload: dict[str, Any]) -> str:
    hashable = {key: value for key, value in payload.items() if key != "artifact_hash"}
    canonical = json.dumps(hashable, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
