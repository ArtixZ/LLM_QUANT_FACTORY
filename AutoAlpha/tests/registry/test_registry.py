from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from autoalpha.dsl.expression import FactorDefinition, field
from autoalpha.registry.lifecycle import ALLOWED_TRANSITIONS, FactorState, LifecycleStore
from autoalpha.registry.metrics import assess_novelty, cluster_factors, neutralize_cross_section
from autoalpha.registry.store import FactorRegistry


def test_neutralization_removes_exposure() -> None:
    columns = ["A", "B", "C", "D", "E"]
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    size = pd.DataFrame([np.arange(5), np.arange(5)], index=dates, columns=columns)
    factor = 3.0 * size + pd.DataFrame(
        [[1, -1, 1, -1, 0], [-1, 1, -1, 1, 0]], index=dates, columns=columns
    )
    residual = neutralize_cross_section(factor, {"size": size})
    assert abs(residual.loc[dates[0]].corr(size.loc[dates[0]])) < 1e-12


def test_novelty_detects_duplicate_and_incremental_fit() -> None:
    index = pd.RangeIndex(100)
    base = pd.Series(np.linspace(-1, 1, 100), index=index)
    candidate = base * 0.99 + np.sin(np.arange(100)) * 0.001
    target = base + np.cos(np.arange(100)) * 0.1
    novelty = assess_novelty(candidate, pd.DataFrame({"base": base}), target)
    assert novelty.is_duplicate
    assert novelty.nearest_factor == "base"


def test_factor_clustering_finds_transitive_families() -> None:
    correlations = pd.DataFrame(
        [[1.0, 0.9, 0.1], [0.9, 1.0, 0.86], [0.1, 0.86, 1.0]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    assert cluster_factors(correlations) == (("A", "B", "C"),)


def test_factor_artifacts_are_versioned_and_integrity_checked(tmp_path) -> None:
    registry = FactorRegistry(tmp_path / "factors")
    definition = FactorDefinition("value", "value", "cheap stocks", field("book_to_price"))
    card = registry.publish(
        definition,
        data_dependencies=("book_to_price",),
        data_lag_days=1,
        applicable_regimes=("all",),
        failure_modes=("value trap",),
        owner="research",
        experiment_id="E1",
    )
    second = registry.publish(
        definition,
        data_dependencies=("book_to_price",),
        data_lag_days=1,
        applicable_regimes=("all",),
        failure_modes=("value trap",),
        owner="research",
        experiment_id="E2",
        parent_ids=(card.factor_id,),
    )
    assert second.version == 2
    assert len(registry.find_by_expression(definition.expression.expression_hash)) == 2

    path = tmp_path / "factors" / card.factor_id / "v0001.json"
    path.write_text(path.read_text().replace("cheap stocks", "tampered"))
    with pytest.raises(RuntimeError, match="modified"):
        registry.versions(card.factor_id)


def test_lifecycle_enforces_transitions_and_rejection_cooldown(tmp_path) -> None:
    lifecycle = LifecycleStore(tmp_path / "lifecycle.jsonl", rejection_cooldown_days=30)
    lifecycle.register("F1", actor="system", reason="new candidate")
    lifecycle.transition("F1", FactorState.REJECTED, actor="reviewer", reason="duplicate")
    with pytest.raises(ValueError, match="cooldown"):
        lifecycle.transition("F1", FactorState.RESEARCH, actor="human", reason="retry")

    rejected_at = datetime.fromisoformat(lifecycle.history("F1")[-1].timestamp_utc)
    event = lifecycle.transition(
        "F1",
        FactorState.RESEARCH,
        actor="human",
        reason="new evidence",
        now=rejected_at.astimezone(UTC) + timedelta(days=31),
    )
    assert event.state is FactorState.RESEARCH
    lifecycle.audit.verify()


def test_lifecycle_requires_shadow_stage_before_paper() -> None:
    assert FactorState.SHADOW in ALLOWED_TRANSITIONS[FactorState.QUALIFIED]
    assert FactorState.PAPER not in ALLOWED_TRANSITIONS[FactorState.QUALIFIED]
