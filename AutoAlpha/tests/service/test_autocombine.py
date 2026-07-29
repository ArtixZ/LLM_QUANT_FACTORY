from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from autoalpha.service import autocombine_app
from autoalpha.service.autocombine import (
    DEFAULT_BUDGET,
    DEFAULT_CONSTRUCTION,
    DEFAULT_OBJECTIVE,
    OBJECTIVE_PRESETS,
    AutoCombineWorker,
    CombineProposal,
    _CandidateEvaluationRejected,
    _gate_distance,
    _gate_failures,
    _portfolio_score,
    _recoverable_evaluation_failure_reason,
    _rejected_experiment_record,
    build_factor_snapshot,
    create_task_record,
    refresh_task_strategy_clusters,
)
from autoalpha.service.autocombine_intelligence import (
    enrich_factor_record,
    signal_independence_metrics,
    write_return_artifact,
)
from autoalpha.service.autocombine_store import AutoCombineStore
from autoalpha.service.store import ServiceStore


def _proposal(name: str, family: str, field: str) -> dict[str, object]:
    return {
        "name": name,
        "family": family,
        "hypothesis": f"Cross-sectional information in {field}",
        "expected_direction": 1,
        "expression": {"operator": "field", "parameters": {"name": field}, "arguments": []},
    }


def _store(tmp_path: Path) -> ServiceStore:
    store = ServiceStore(tmp_path / "autoalpha.sqlite3")
    for index, (name, family, field) in enumerate(
        (("Price", "Momentum", "close"), ("Volume", "Liquidity", "vol")), start=1
    ):
        store.upsert_factor_pool(
            factor_id=f"F_{index}",
            source_iteration=index,
            proposal=_proposal(name, family, field),
            metrics={
                "long_only_sharpe_ratio": float(index),
                "long_only_simple_annual_return": 0.05 * index,
                "long_only_walk_forward_worst_sharpe": 0.2,
                "long_only_max_drawdown": -0.1,
                "long_only_annual_turnover": 5.0,
            },
            status="ELIGIBLE",
            status_reason="test",
        )
    return store


def test_factor_snapshot_is_filtered_ranked_and_frozen(tmp_path: Path) -> None:
    store = _store(tmp_path)
    snapshot = build_factor_snapshot(
        store,
        {"statuses": ["ELIGIBLE"], "factor_ids": [], "source_task_ids": []},
        DEFAULT_CONSTRUCTION,
    )
    assert [item["factor_id"] for item in snapshot] == ["F_2", "F_1"]
    assert snapshot[0]["proposal"]["expression"]["parameters"] == {"name": "vol"}


def test_factor_snapshot_uses_behavior_cluster_as_search_cluster(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for factor_id in ("F_1", "F_2"):
        store.upsert_factor_knowledge(
            factor_id=factor_id,
            canonical_mechanism="TURNOVER_LIQUIDITY",
            mechanism_summary="test",
            tags=["test"],
            review={
                "expression_signature": factor_id,
                "parameter_family": "shared-parameter-family",
                "behavior_cluster_id": "B_SHARED",
            },
            falsification={},
            related_factors=[],
        )

    snapshot = build_factor_snapshot(
        store,
        {"statuses": ["ELIGIBLE"], "factor_ids": [], "source_task_ids": []},
        DEFAULT_CONSTRUCTION,
    )

    by_id = {item["factor_id"]: item for item in snapshot}
    assert by_id["F_1"]["behavior_cluster_id"] == "B_SHARED"
    assert by_id["F_2"]["search_cluster_id"] == "B_SHARED"
    assert by_id["F_1"]["semantic_cluster_id"] != by_id["F_2"]["semantic_cluster_id"]


def test_combine_task_view_exposes_homogeneity_summary(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for factor_id in ("F_1", "F_2"):
        store.upsert_factor_knowledge(
            factor_id=factor_id,
            canonical_mechanism="TURNOVER_LIQUIDITY",
            mechanism_summary="test",
            tags=["test"],
            review={
                "expression_signature": factor_id,
                "parameter_family": "shared-parameter-family",
                "behavior_cluster_id": "B_SHARED",
            },
            falsification={},
            related_factors=[],
        )
    combine_store = AutoCombineStore(store)
    task = combine_store.create_task(
        create_task_record(
            store,
            name="homogeneity summary",
            market="CN_A",
            data_path=str(tmp_path),
            protocol={
                "exploration_start": "2010-01-01",
                "exploration_end": "2017-12-31",
                "validation_start": "2018-01-01",
                "validation_end": "2024-12-31",
                "holdout_start": "2025-01-01",
                "holdout_end": "2026-07-16",
                "minimum_folds": 2,
            },
            scope={"statuses": ["ELIGIBLE"]},
            construction={**DEFAULT_CONSTRUCTION, "candidate_pool_limit": 10},
            objective=DEFAULT_OBJECTIVE,
            budget=DEFAULT_BUDGET,
            notes="test",
        )
    )

    summary = task["homogeneity_summary"]

    assert summary["factor_count"] == 2
    assert summary["behavior_cluster_count"] == 1
    assert summary["search_cluster_count"] == 1
    assert summary["duplicate_search_cluster_factor_count"] == 1


def test_manual_and_hybrid_factor_scope_preserve_user_intent(tmp_path: Path) -> None:
    store = _store(tmp_path)
    manual = build_factor_snapshot(
        store,
        {"mode": "MANUAL", "factor_ids": ["F_1"], "statuses": ["SCREENED_OUT"]},
        DEFAULT_CONSTRUCTION,
    )
    hybrid = build_factor_snapshot(
        store,
        {
            "mode": "HYBRID",
            "required_factor_ids": ["F_1"],
            "statuses": ["ELIGIBLE"],
        },
        DEFAULT_CONSTRUCTION,
    )
    assert [item["factor_id"] for item in manual] == ["F_1"]
    assert hybrid[0]["factor_id"] == "F_1"
    assert hybrid[0]["required"] is True


def test_autocombine_limits_real_parameter_family_duplicates() -> None:
    mechanism = {"F_A": "TURNOVER_LIQUIDITY", "F_B": "PRICE_REVERSAL", "F_C": "QUALITY"}
    search_cluster = {"F_A": "B_A", "F_B": "B_B", "F_C": "B_C"}
    construction = {
        **DEFAULT_CONSTRUCTION,
        "maximum_same_family": 3,
        "maximum_same_semantic_cluster": 2,
        "maximum_same_parameter_family": 1,
    }

    assert not AutoCombineWorker._family_limit_ok(
        ("F_A", "F_B"),
        mechanism,
        search_cluster,
        construction,
        parameter_family={"F_A": "window=20", "F_B": "window=20"},
    )
    assert AutoCombineWorker._family_limit_ok(
        ("F_A", "F_C"),
        mechanism,
        search_cluster,
        construction,
        parameter_family={"F_A": "window=20", "F_C": "NO_EXPLICIT_LOOKBACK"},
    )


def test_autocombine_audits_homogeneity_candidate_rejections(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for factor_id, mechanism, behavior in (
        ("F_1", "PRICE_REVERSAL", "B_PRICE"),
        ("F_2", "TURNOVER_LIQUIDITY", "B_LIQUIDITY"),
    ):
        store.upsert_factor_knowledge(
            factor_id=factor_id,
            canonical_mechanism=mechanism,
            mechanism_summary="test",
            tags=["test"],
            review={
                "behavior_cluster_id": behavior,
                "parameter_family": "window=20",
                "expression_signature": factor_id,
            },
            falsification={},
            related_factors=[],
        )
    combine_store = AutoCombineStore(store)
    task = combine_store.create_task(
        create_task_record(
            store,
            name="homogeneity rejected",
            market="CN_A",
            data_path=str(tmp_path),
            protocol={
                "exploration_start": "2010-01-01",
                "exploration_end": "2017-12-31",
                "validation_start": "2018-01-01",
                "validation_end": "2024-12-31",
                "holdout_start": "2025-01-01",
                "holdout_end": "2026-07-16",
                "minimum_folds": 2,
            },
            scope={"statuses": ["ELIGIBLE"]},
            construction={
                **DEFAULT_CONSTRUCTION,
                "candidate_pool_limit": 10,
                "maximum_same_family": 2,
                "maximum_same_semantic_cluster": 2,
                "maximum_same_parameter_family": 1,
            },
            objective=DEFAULT_OBJECTIVE,
            budget=DEFAULT_BUDGET,
            notes="test",
        )
    )
    worker = AutoCombineWorker(
        task["task_id"],
        store,
        combine_store,
        object(),  # type: ignore[arg-type]
        config_path=tmp_path / "research.toml",
    )

    with pytest.raises(
        _CandidateEvaluationRejected, match="HOMOGENEITY_DIVERSIFICATION_CONSTRAINT"
    ):
        worker._evaluate_proposal(
            object(),  # type: ignore[arg-type]
            task,
            CombineProposal(
                action="SEED",
                factor_ids=("F_1", "F_2"),
                rationale="same parameter family",
                hypothesis="should be rejected before evaluation",
                source="TEST",
            ),
            iteration=1,
        )

    events = combine_store.events(task["task_id"])
    assert events[0]["event"] == "COMBINE_HOMOGENEITY_CANDIDATE_REJECTED"
    assert events[0]["payload"]["dimension"] == "parameter_family"
    assert events[0]["payload"]["crowded_labels"] == {"window=20": 2}
    summary = autocombine_app._homogeneity_rejection_summary(events)
    assert summary == {
        "total": 1,
        "by_dimension": {"parameter_family": 1},
        "crowded_label_counts": {"parameter_family:window=20": 2},
    }


def test_autocombine_records_candidate_level_evaluation_rejection(tmp_path: Path) -> None:
    store = _store(tmp_path)
    combine_store = AutoCombineStore(store)
    task = combine_store.create_task(
        create_task_record(
            store,
            name="candidate rejection",
            market="CN_A",
            data_path=str(tmp_path),
            protocol={
                "exploration_start": "2010-01-01",
                "exploration_end": "2017-12-31",
                "validation_start": "2018-01-01",
                "validation_end": "2024-12-31",
                "holdout_start": "2025-01-01",
                "holdout_end": "2026-07-16",
                "minimum_folds": 2,
            },
            scope={"statuses": ["ELIGIBLE"]},
            construction={**DEFAULT_CONSTRUCTION, "candidate_pool_limit": 10},
            objective=DEFAULT_OBJECTIVE,
            budget=DEFAULT_BUDGET,
            notes="test",
        )
    )
    proposal = CombineProposal(
        factor_ids=("F_1", "F_2"),
        action="SEED",
        source="DETERMINISTIC",
        rationale="test",
        hypothesis="test",
        prompt_hash=None,
        response_hash=None,
    )
    assert (
        _recoverable_evaluation_failure_reason(
            ValueError("Only 5 walk-forward folds were evaluable; minimum=6")
        )
        == "INSUFFICIENT_WALK_FORWARD_FOLDS"
    )

    experiment = combine_store.record_experiment(
        task["task_id"],
        _rejected_experiment_record(
            task,
            proposal,
            1,
            _CandidateEvaluationRejected(
                "All weight candidates rejected: ['INSUFFICIENT_WALK_FORWARD_FOLDS']"
            ),
        ),
    )

    assert experiment["qualification"] == "CANDIDATE_EVALUATION_REJECTED"
    assert experiment["gate_status"] == "REJECTED"
    assert experiment["failed_gates"] == ["candidate_evaluation_rejected"]
    assert experiment["metrics"]["autocombine_hidden_metrics_exposed"] is False


def test_contaminated_factor_remains_available_with_audit_marker(tmp_path: Path) -> None:
    store = _store(tmp_path)
    backtest_id = store.create_manual_backtest({"factor_ids": ["F_1"]})
    store.record_manual_research_exposures(
        backtest_id=backtest_id,
        generation_id="test-generation",
        factor_ids=["F_1"],
        period_start="2025-01-01",
        period_end="2026-01-01",
        holdout_start="2025-01-01",
        holdout_end="2026-12-31",
    )
    snapshot = build_factor_snapshot(
        store,
        {"mode": "MANUAL", "factor_ids": ["F_1"]},
        DEFAULT_CONSTRUCTION,
    )
    assert [item["factor_id"] for item in snapshot] == ["F_1"]
    assert snapshot[0]["holdout_contaminated"] is True


def test_combine_store_persists_task_experiment_and_strategy(tmp_path: Path) -> None:
    base = _store(tmp_path)
    store = AutoCombineStore(base)
    record = create_task_record(
        base,
        name="Test combine",
        market="CN_A",
        data_path=str(tmp_path),
        protocol={
            "exploration_start": "2010-01-01",
            "exploration_end": "2017-12-31",
            "validation_start": "2018-01-01",
            "validation_end": "2024-12-31",
            "holdout_start": "2025-01-01",
            "holdout_end": "2026-07-16",
            "minimum_folds": 2,
        },
        scope={"statuses": ["ELIGIBLE"]},
        construction={**DEFAULT_CONSTRUCTION, "candidate_pool_limit": 10},
        objective=DEFAULT_OBJECTIVE,
        budget=DEFAULT_BUDGET,
        notes="test",
    )
    task = store.create_task(record)
    experiment = store.record_experiment(
        task["task_id"],
        {
            "iteration": 1,
            "candidate_hash": "candidate-1",
            "action": "SEED",
            "proposal_source": "DETERMINISTIC",
            "factor_ids": ["F_1", "F_2"],
            "weights": [0.5, 0.5],
            "rationale": "test",
            "hypothesis": "test",
            "metrics": {"portfolio_sharpe_ratio": 1.2},
            "score": 1.0,
            "gate_status": "PASSED",
            "failed_gates": [],
        },
    )
    store.update_task(
        task["task_id"],
        blind_verdict="BLIND_GENERALIZATION_PASSED",
        blind_evidence_hash="a" * 64,
        qualified_experiment_id=experiment["id"],
        production_candidate_experiment_id=experiment["id"],
        qualification_status="PRODUCTION_CANDIDATE",
    )
    strategy = store.promote_strategy(task["task_id"], experiment["id"], "Test strategy")
    assert store.task(task["task_id"])["factor_count"] == 2  # type: ignore[index]
    assert strategy["lifecycle"] == "QUALIFIED"
    assert strategy["specification"]["factor_weights"] == [0.5, 0.5]


def test_semantic_fingerprint_ignores_cosmetic_family_and_normalization() -> None:
    first = enrich_factor_record(
        {
            "proposal": {
                "name": "AmountStability",
                "family": "Liquidity Stability",
                "hypothesis": "stable amount",
                "expression": {
                    "operator": "cs_rank",
                    "parameters": {},
                    "arguments": [
                        {
                            "operator": "negate",
                            "parameters": {},
                            "arguments": [
                                {
                                    "operator": "rolling_std",
                                    "parameters": {"window": 60},
                                    "arguments": [
                                        {
                                            "operator": "field",
                                            "parameters": {"name": "amount"},
                                            "arguments": [],
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                },
            }
        }
    )
    second = enrich_factor_record(
        {
            "proposal": {
                "name": "LiquidityStability60D",
                "family": "Anything",
                "hypothesis": "same economic signal",
                "expression": {
                    "operator": "cs_zscore",
                    "parameters": {},
                    "arguments": [
                        {
                            "operator": "negate",
                            "parameters": {},
                            "arguments": [
                                {
                                    "operator": "rolling_std",
                                    "parameters": {"window": 60},
                                    "arguments": [
                                        {
                                            "operator": "field",
                                            "parameters": {"name": "amount"},
                                            "arguments": [],
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                },
            }
        }
    )
    assert first["mechanism"] == second["mechanism"] == "TURNOVER_LIQUIDITY"
    assert first["semantic_cluster_id"] == second["semantic_cluster_id"]


def test_signal_independence_reports_effective_bets() -> None:
    independent = signal_independence_metrics(["A", "B", "C"], {"A:B": 0.0, "A:C": 0.0, "B:C": 0.0})
    redundant = signal_independence_metrics(
        ["A", "B", "C"], {"A:B": 0.95, "A:C": 0.95, "B:C": 0.95}
    )
    assert independent["portfolio_effective_factor_bets"] == 3.0
    assert redundant["portfolio_effective_factor_bets"] < 1.2


def test_gate_distance_does_not_allow_score_to_hide_hard_failures() -> None:
    objective = {**DEFAULT_OBJECTIVE, "maximum_drawdown": 0.20}
    metrics = {
        "portfolio_coverage": 0.95,
        "portfolio_walk_forward_positive_fraction": 0.8,
        "portfolio_walk_forward_worst_sharpe": 0.2,
        "portfolio_max_drawdown": -0.30,
        "portfolio_annual_turnover": 5.0,
        "portfolio_max_factor_correlation": 0.2,
        "portfolio_cost_stress_net_ir": 1.0,
        "portfolio_simple_annual_return": 0.50,
    }
    assert "drawdown" in _gate_failures(metrics, objective)
    assert _gate_distance(metrics, objective) > 0


def test_task_leaders_are_clustered_by_active_returns(tmp_path: Path) -> None:
    base = _store(tmp_path)
    store = AutoCombineStore(base)
    task_ids: list[str] = []
    dates = pd.date_range("2020-01-01", periods=120, freq="B")
    for index, scale in enumerate((1.0, 1.01), start=1):
        record = create_task_record(
            base,
            name=f"Cluster task {index}",
            market="CN_A",
            data_path=str(tmp_path),
            protocol={
                "exploration_start": "2010-01-01",
                "exploration_end": "2017-12-31",
                "validation_start": "2018-01-01",
                "validation_end": "2024-12-31",
                "holdout_start": "2025-01-01",
                "holdout_end": "2026-07-16",
                "minimum_folds": 2,
            },
            scope={"statuses": ["ELIGIBLE"]},
            construction=DEFAULT_CONSTRUCTION,
            objective=DEFAULT_OBJECTIVE,
            budget=DEFAULT_BUDGET,
            notes="cluster test",
        )
        task = store.create_task(record)
        task_ids.append(task["task_id"])
        active = pd.Series(
            [scale * ((position % 7) - 3) / 1000 for position in range(len(dates))],
            index=dates,
        )
        path, digest = write_return_artifact(
            tmp_path / "artifacts",
            task_id=task["task_id"],
            candidate_hash=f"candidate-{index}",
            net_returns=active,
            active_returns=active,
        )
        experiment = store.record_experiment(
            task["task_id"],
            {
                "iteration": 1,
                "candidate_hash": f"candidate-{index}",
                "action": "SEED",
                "proposal_source": "DETERMINISTIC",
                "factor_ids": ["F_1", "F_2"],
                "weights": [0.5, 0.5],
                "rationale": "test",
                "hypothesis": "test",
                "metrics": {"autocombine_return_artifact_path": path},
                "score": 1.0,
                "gate_status": "REJECTED",
                "failed_gates": ["test"],
                "return_artifact_path": path,
                "return_artifact_hash": digest,
            },
        )
        store.update_task(task["task_id"], best_experiment_id=experiment["id"])
    clusters = refresh_task_strategy_clusters(store)
    assert (
        clusters[task_ids[0]]["strategy_cluster_id"] == clusters[task_ids[1]]["strategy_cluster_id"]
    )
    assert clusters[task_ids[0]]["nearest_active_return_correlation"] > 0.99


def test_gate_failures_and_score_prioritize_robust_public_metrics() -> None:
    good = {
        "portfolio_coverage": 0.95,
        "portfolio_walk_forward_positive_fraction": 0.8,
        "portfolio_walk_forward_worst_sharpe": 0.5,
        "portfolio_max_drawdown": -0.10,
        "portfolio_annual_turnover": 8.0,
        "portfolio_max_factor_correlation": 0.3,
        "portfolio_cost_stress_net_ir": 1.0,
        "portfolio_active_information_ratio": 1.2,
        "portfolio_active_simple_annual_return": 0.08,
        "portfolio_sharpe_ratio": 1.5,
        "portfolio_simple_annual_return": 0.12,
    }
    bad = {**good, "portfolio_walk_forward_worst_sharpe": -2.0, "portfolio_max_drawdown": -0.5}
    assert _gate_failures(good, DEFAULT_OBJECTIVE) == []
    assert set(_gate_failures(bad, DEFAULT_OBJECTIVE)) == {"worst_fold", "drawdown"}
    assert _portfolio_score(good, []) > _portfolio_score(bad, ["worst_fold", "drawdown"])


def test_objective_presets_change_portfolio_ranking() -> None:
    high_sharpe = {
        "portfolio_sharpe_ratio": 2.2,
        "portfolio_simple_annual_return": 0.18,
        "portfolio_max_drawdown": -0.18,
        "portfolio_walk_forward_worst_sharpe": 0.4,
        "portfolio_walk_forward_positive_fraction": 0.8,
        "portfolio_annual_turnover": 12.0,
        "portfolio_max_factor_correlation": 0.4,
    }
    low_drawdown = {
        **high_sharpe,
        "portfolio_sharpe_ratio": 1.2,
        "portfolio_simple_annual_return": 0.08,
        "portfolio_max_drawdown": -0.03,
    }
    sharpe_objective = OBJECTIVE_PRESETS["PORTFOLIO_SHARPE_FIRST"]
    drawdown_objective = OBJECTIVE_PRESETS["DRAWDOWN_FIRST"]
    assert _portfolio_score(high_sharpe, [], sharpe_objective) > _portfolio_score(
        low_drawdown, [], sharpe_objective
    )
    assert _portfolio_score(low_drawdown, [], drawdown_objective) > _portfolio_score(
        high_sharpe, [], drawdown_objective
    )
