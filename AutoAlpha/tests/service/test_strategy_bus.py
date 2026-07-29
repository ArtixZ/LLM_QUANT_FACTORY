from __future__ import annotations

from pathlib import Path

import pytest

from autoalpha.service.autocombine import (
    DEFAULT_BUDGET as AUTOCOMBINE_DEFAULT_BUDGET,
)
from autoalpha.service.autocombine import (
    DEFAULT_CONSTRUCTION,
    DEFAULT_OBJECTIVE,
    create_task_record,
)
from autoalpha.service.autocombine_store import AutoCombineStore
from autoalpha.service.external_jobs import external_job_id
from autoalpha.service.quantcombine import (
    DEFAULT_BUDGET,
    DEFAULT_ENGINE,
    create_quant_task_record,
)
from autoalpha.service.quantcombine_store import QuantCombineStore
from autoalpha.service.store import ServiceStore
from autoalpha.service.strategy_bus import (
    advance_formal_strategy_lifecycle,
    approve_formal_strategy_transition,
    build_strategy_bus_snapshot,
    create_formal_strategy_from_experiment,
    factor_knowledge_map,
    formal_strategy_library,
    promote_formal_strategy_lifecycle,
    publish_strategy_release_dossier,
    stable_experiment_id,
    strategy_execution_package,
    strategy_experiment_lineage,
    strategy_lifecycle_readiness,
    strategy_production_funnel,
    strategy_promotion_candidates,
    strategy_release_dossier,
)


def _proposal(name: str, family: str, field: str) -> dict[str, object]:
    return {
        "name": name,
        "family": family,
        "hypothesis": f"Cross-sectional information in {field}",
        "expected_direction": 1,
        "expression": {"operator": "field", "parameters": {"name": field}, "arguments": []},
    }


def _store_with_factors(tmp_path: Path) -> ServiceStore:
    store = ServiceStore(tmp_path / "autoalpha.sqlite3")
    for index, (name, family, field) in enumerate(
        (
            ("Volume Stability", "Liquidity", "vol"),
            ("Value Quality", "Value", "pb"),
        ),
        start=1,
    ):
        store.upsert_factor_pool(
            factor_id=f"F_{index}",
            source_iteration=index,
            source_task_id="task-test",
            proposal=_proposal(name, family, field),
            metrics={
                "long_only_sharpe_ratio": 1.0 + index / 10,
                "long_only_simple_annual_return": 0.08 + index / 100,
                "long_only_max_drawdown": -0.15,
                "long_only_walk_forward_worst_sharpe": 0.1,
                "long_only_annual_turnover": 5.0,
                "long_only_capacity_cny": 10_000_000,
                "online_behavior_cluster_id": "C001",
            },
            status="ELIGIBLE",
            status_reason="test",
        )
    return store


def test_strategy_bus_indexes_factor_candidates_and_clusters(tmp_path: Path) -> None:
    store = _store_with_factors(tmp_path)
    snapshot = build_strategy_bus_snapshot(
        store,
        autocombine_store=AutoCombineStore(store),
        quantcombine_store=QuantCombineStore(store),
    )

    assert snapshot["summary"]["by_stage"]["FACTOR_CANDIDATE"] == 2
    assert snapshot["summary"]["by_stage"]["FACTOR_CLUSTER"] >= 1
    factor_node = stable_experiment_id("AUTOALPHA", "F_1", "FACTOR_CANDIDATE")
    assert store.strategy_experiment_object(factor_node)["metrics"]["long_only_sharpe_ratio"] == 1.1  # type: ignore[index]
    assert any(edge["relation"] == "BELONGS_TO_CLUSTER" for edge in snapshot["edges"])


def test_factor_knowledge_map_ranks_clusters_by_long_only_metrics_not_legacy_sharpe(
    tmp_path: Path,
) -> None:
    store = ServiceStore(tmp_path / "autoalpha.sqlite3")
    store.upsert_factor_pool(
        factor_id="F_LEGACY_HIGH",
        source_iteration=1,
        source_task_id="task-test",
        proposal=_proposal("Legacy High", "Momentum", "close"),
        metrics={
            "sharpe_ratio": 99.0,
            "simple_annual_return": 9.0,
            "long_only_sharpe_ratio": 0.1,
            "long_only_simple_annual_return": 0.01,
            "long_only_max_drawdown": -0.30,
        },
        status="ELIGIBLE",
        status_reason="test",
    )
    store.upsert_factor_pool(
        factor_id="F_LONG_ONLY_HIGH",
        source_iteration=2,
        source_task_id="task-test",
        proposal=_proposal("Long Only High", "Momentum", "close"),
        metrics={
            "sharpe_ratio": 0.2,
            "simple_annual_return": 0.02,
            "long_only_sharpe_ratio": 1.6,
            "long_only_simple_annual_return": 0.18,
            "long_only_max_drawdown": -0.08,
        },
        status="ELIGIBLE",
        status_reason="test",
    )

    knowledge = factor_knowledge_map(
        store,
        behavior_snapshot={
            "factors": {
                "F_LEGACY_HIGH": {"behavior_cluster_id": "C001"},
                "F_LONG_ONLY_HIGH": {"behavior_cluster_id": "C001"},
            }
        },
    )

    cluster = knowledge["clusters"][0]
    assert cluster["leader_factor_id"] == "F_LONG_ONLY_HIGH"
    assert cluster["top_factors"][0]["factor_id"] == "F_LONG_ONLY_HIGH"


def test_formal_strategy_created_from_combination_experiment(tmp_path: Path) -> None:
    store = _store_with_factors(tmp_path)
    experiment_id = stable_experiment_id("TEST", "candidate-1", "COMBINATION_CANDIDATE")
    store.upsert_strategy_experiment_object(
        experiment_id=experiment_id,
        stage="COMBINATION_CANDIDATE",
        object_type="factor_combination",
        source_system="TEST",
        source_id="candidate-1",
        title="Test strategy candidate",
        status="RESEARCH_LEADER",
        market="CN_A",
        metrics={
            "portfolio_max_drawdown": -0.12,
            "portfolio_capacity_cny": 8_000_000,
        },
        evidence={
            "factor_ids": ["F_1", "F_2"],
            "weights": [0.6, 0.4],
            "maximum_positions": 30,
            "target_gross_exposure": 0.9,
        },
    )

    strategy = create_formal_strategy_from_experiment(store, experiment_id, name="Formal V1")

    assert strategy["name"] == "Formal V1"
    assert strategy["signal_policy"]["factor_ids"] == ["F_1", "F_2"]
    assert strategy["rebalance_policy"]["schedule"] == "WEEKLY_FIRST_SESSION"
    assert strategy["execution_policy"]["execution_time"] == "NEXT_SESSION_OPEN"
    assert strategy["cost_policy"]["stamp_duty_bps_sell"] == 5.0
    assert strategy["monitoring_policy"]["paper_first"] is True
    library = formal_strategy_library(store)
    summary = library["strategies"][0]["production_evidence_summary"]
    assert summary["protocol"] == "AUTOALPHA_STRATEGY_PRODUCTION_EVIDENCE_SUMMARY_V1"
    assert summary["evidence_state"] == "EVIDENCE_INCOMPLETE"
    assert "strict_pit_market_state_verified" in summary["missing_or_blocking_evidence"]


def test_strategy_execution_package_exposes_trade_rules_and_blockers(tmp_path: Path) -> None:
    store = _store_with_factors(tmp_path)
    experiment_id = stable_experiment_id("TEST", "candidate-package", "COMBINATION_CANDIDATE")
    store.upsert_strategy_experiment_object(
        experiment_id=experiment_id,
        stage="COMBINATION_CANDIDATE",
        object_type="factor_combination",
        source_system="TEST",
        source_id="candidate-package",
        title="Package strategy candidate",
        status="QUALIFIED_CHAMPION",
        market="CN_A",
        evidence={
            "factor_ids": ["F_1", "F_2"],
            "weights": [0.6, 0.4],
            "maximum_positions": 30,
            "target_gross_exposure": 0.9,
            "gate_status": "PASSED",
        },
    )
    strategy = create_formal_strategy_from_experiment(store, experiment_id)

    package = strategy_execution_package(store, strategy["strategy_uid"], strategy["version"])

    assert package["production_ready"] is False
    assert "strict_pit_market_state_not_verified" in package["production_blockers"]
    assert package["signal_contract"]["ranking_side"] == "LONG_ONLY_TOP_RANK"
    assert package["rebalance_contract"]["buy_rule"].startswith("BUY_TARGETS")
    assert package["execution_contract"]["execution_time"] == "NEXT_SESSION_OPEN"
    assert package["execution_contract"]["tradability"]["buy"] == "require_can_buy_open_or_proxy"
    paper_contract = package["paper_trading_contract"]
    assert paper_contract["protocol"] == "AUTOALPHA_STRATEGY_TO_PAPER_PORTFOLIO_SEED_V1"
    assert paper_contract["compatible_engine"] == "PaperTradingEngine"
    assert paper_contract["execution_protocol"] == "A_SHARE_PAPER_NEXT_OPEN_PROXY_EXECUTION_V2"
    assert paper_contract["proxy_only"] is True
    assert paper_contract["required_operator_inputs"] == ["initial_cash_cny", "as_of_date"]
    assert paper_contract["paper_portfolio_seed"]["factor_ids"] == ["F_1", "F_2"]
    assert paper_contract["paper_portfolio_seed"]["weights"] == [0.6, 0.4]
    assert paper_contract["paper_portfolio_seed"]["gross_exposure"] == 0.9
    assert paper_contract["timing"]["execution_time"] == "NEXT_SESSION_OPEN"
    assert paper_contract["tradability"]["t_plus_one_sell_lock"] is True
    playbook = package["trading_playbook"]
    assert playbook["protocol"] == "A_SHARE_LONG_ONLY_TRADING_PLAYBOOK_V1"
    assert playbook["portfolio_mode"] == "LONG_ONLY_CASH_EQUITY"
    assert playbook["signal_cutoff"] == "END_OF_DAY_AFTER_CLOSE"
    assert playbook["execution_window"] == "NEXT_SESSION_OPEN"
    assert playbook["capital_allocation"]["per_position_target_weight"] == pytest.approx(0.03)
    assert playbook["blocked_order_policy"]["sell_blocked"] == (
        "retain_position_until_next_tradable_rebalance"
    )
    assert "strict_pit_market_state_missing_for_production" in playbook["disable_conditions"]

    frozen = promote_formal_strategy_lifecycle(
        store,
        strategy["strategy_uid"],
        strategy["version"],
        target_lifecycle="FROZEN",
        evidence={
            "source_experiment_id": experiment_id,
            "factor_ids": ["F_1", "F_2"],
            "weights": [0.6, 0.4],
            "public_validation_passed": True,
        },
    )
    promote_formal_strategy_lifecycle(
        store,
        strategy["strategy_uid"],
        strategy["version"],
        target_lifecycle="HIDDEN_HOLDOUT",
        evidence={
            "frozen_specification_hash": frozen["specification_hash"],
            "holdout_evaluation_requested": True,
        },
    )
    promote_formal_strategy_lifecycle(
        store,
        strategy["strategy_uid"],
        strategy["version"],
        target_lifecycle="SHADOW",
        evidence={"hidden_holdout_passed": True, "holdout_evaluation_id": "HOLDOUT-1"},
    )
    promote_formal_strategy_lifecycle(
        store,
        strategy["strategy_uid"],
        strategy["version"],
        target_lifecycle="PAPER",
        evidence={"shadow_trading_days": 20, "shadow_execution_passed": True},
    )
    production = promote_formal_strategy_lifecycle(
        store,
        strategy["strategy_uid"],
        strategy["version"],
        target_lifecycle="PRODUCTION_CANDIDATE",
        evidence={
            "paper_trading_days": 20,
            "paper_trading_passed": True,
            "risk_approval": "APPROVED",
            "strict_pit_market_state_verified": True,
        },
    )
    production_package = strategy_execution_package(
        store, production["strategy_uid"], production["version"]
    )

    assert production_package["production_ready"] is True
    assert production_package["production_blockers"] == []
    assert production_package["audit_contract"]["strict_pit_market_state_verified"] is True


def test_strategy_release_dossier_binds_source_execution_and_factors(tmp_path: Path) -> None:
    store = _store_with_factors(tmp_path)
    experiment_id = stable_experiment_id("TEST", "candidate-dossier", "COMBINATION_CANDIDATE")
    store.upsert_strategy_experiment_object(
        experiment_id=experiment_id,
        stage="COMBINATION_CANDIDATE",
        object_type="factor_combination",
        source_system="TEST",
        source_id="candidate-dossier",
        title="Dossier strategy candidate",
        status="RESEARCH_LEADER",
        market="CN_A",
        metrics={
            "portfolio_sharpe_ratio": 1.2,
            "portfolio_simple_annual_return": 0.16,
            "portfolio_max_drawdown": -0.09,
        },
        evidence={
            "factor_ids": ["F_1", "F_2"],
            "weights": [0.6, 0.4],
            "gate_status": "REJECTED",
            "failed_gates": ["marginal_contribution"],
        },
    )
    strategy = create_formal_strategy_from_experiment(store, experiment_id)

    dossier = strategy_release_dossier(store, strategy["strategy_uid"], strategy["version"])

    assert dossier["dossier_protocol"] == "AUTOALPHA_FORMAL_STRATEGY_RELEASE_DOSSIER_V1"
    assert dossier["source"]["experiment_id"] == experiment_id
    assert dossier["source"]["failed_gates"] == ["marginal_contribution"]
    assert dossier["execution_package"]["execution_contract"]["execution_time"] == (
        "NEXT_SESSION_OPEN"
    )
    assert dossier["audit"]["release_decision"] == "EXPORT_ONLY_NOT_PRODUCTION_READY"
    assert [item["factor_id"] for item in dossier["factors"]] == ["F_1", "F_2"]
    assert [item["weight"] for item in dossier["factors"]] == [0.6, 0.4]


def test_strategy_release_dossier_export_is_immutable_and_idempotent(tmp_path: Path) -> None:
    store = _store_with_factors(tmp_path)
    experiment_id = stable_experiment_id("TEST", "candidate-export", "COMBINATION_CANDIDATE")
    store.upsert_strategy_experiment_object(
        experiment_id=experiment_id,
        stage="COMBINATION_CANDIDATE",
        object_type="factor_combination",
        source_system="TEST",
        source_id="candidate-export",
        title="Export strategy candidate",
        status="RESEARCH_LEADER",
        market="CN_A",
        evidence={"factor_ids": ["F_1", "F_2"], "weights": [0.6, 0.4]},
    )
    strategy = create_formal_strategy_from_experiment(store, experiment_id)

    first = publish_strategy_release_dossier(
        store, tmp_path / "artifacts", strategy["strategy_uid"], strategy["version"]
    )
    second = publish_strategy_release_dossier(
        store, tmp_path / "artifacts", strategy["strategy_uid"], strategy["version"]
    )
    payload_path = tmp_path / "artifacts" / first["artifact"]["payload_path"]

    assert first["artifact"]["artifact_id"] == second["artifact"]["artifact_id"]
    assert first["artifact"]["metadata"]["release_decision"] == (
        "EXPORT_ONLY_NOT_PRODUCTION_READY"
    )
    assert first["filename"].endswith("-release-dossier.json")
    assert "AUTOALPHA_FORMAL_STRATEGY_RELEASE_DOSSIER_V1" in payload_path.read_text(
        encoding="utf-8"
    )


def test_formal_strategy_creation_is_idempotent_per_source_experiment(tmp_path: Path) -> None:
    store = _store_with_factors(tmp_path)
    experiment_id = stable_experiment_id("TEST", "candidate-idempotent", "COMBINATION_CANDIDATE")
    store.upsert_strategy_experiment_object(
        experiment_id=experiment_id,
        stage="COMBINATION_CANDIDATE",
        object_type="factor_combination",
        source_system="TEST",
        source_id="candidate-idempotent",
        title="Idempotent strategy candidate",
        status="RESEARCH_LEADER",
        market="CN_A",
        evidence={"factor_ids": ["F_1", "F_2"], "weights": [0.7, 0.3]},
    )

    first = create_formal_strategy_from_experiment(store, experiment_id)
    second = create_formal_strategy_from_experiment(store, experiment_id)

    assert first["strategy_uid"] == second["strategy_uid"]
    assert first["version"] == second["version"]
    assert len(store.formal_strategy_versions(limit=10)) == 1


def test_formal_strategy_versions_are_indexed_on_strategy_bus(tmp_path: Path) -> None:
    store = _store_with_factors(tmp_path)
    experiment_id = stable_experiment_id("TEST", "candidate-bus", "COMBINATION_CANDIDATE")
    store.upsert_strategy_experiment_object(
        experiment_id=experiment_id,
        stage="COMBINATION_CANDIDATE",
        object_type="factor_combination",
        source_system="TEST",
        source_id="candidate-bus",
        title="Bus strategy candidate",
        status="RESEARCH_LEADER",
        market="CN_A",
        evidence={"factor_ids": ["F_1", "F_2"], "weights": [0.5, 0.5]},
    )
    strategy = create_formal_strategy_from_experiment(store, experiment_id)

    snapshot = build_strategy_bus_snapshot(
        store,
        autocombine_store=AutoCombineStore(store),
        quantcombine_store=QuantCombineStore(store),
    )

    assert snapshot["summary"]["by_stage"]["STRATEGY_VERSION"] == 1
    source_id = f"{strategy['strategy_uid']}@{strategy['version']}"
    strategy_node = stable_experiment_id(
        "FORMAL_STRATEGY_LIBRARY", source_id, "STRATEGY_VERSION"
    )
    assert store.strategy_experiment_object(strategy_node)["status"] == "RESEARCH"  # type: ignore[index]
    links = store.strategy_experiment_edges(experiment_id=strategy_node)
    assert any(edge["relation"] == "USED_IN" for edge in links)
    promoted_links = store.strategy_experiment_edges(experiment_id=experiment_id)
    assert any(edge["relation"] == "PROMOTED_TO_FORMAL_STRATEGY" for edge in promoted_links)


def test_strategy_lineage_marks_combination_as_formal_strategy_source(
    tmp_path: Path,
) -> None:
    store = _store_with_factors(tmp_path)
    experiment_id = stable_experiment_id(
        "TEST", "candidate-formal-lineage", "COMBINATION_CANDIDATE"
    )
    store.upsert_strategy_experiment_object(
        experiment_id=experiment_id,
        stage="COMBINATION_CANDIDATE",
        object_type="factor_combination",
        source_system="TEST",
        source_id="candidate-formal-lineage",
        title="Formal lineage candidate",
        status="RESEARCH_LEADER",
        market="CN_A",
        evidence={"factor_ids": ["F_1", "F_2"], "weights": [0.5, 0.5]},
    )
    strategy = create_formal_strategy_from_experiment(store, experiment_id)

    lineage = strategy_experiment_lineage(store, experiment_id)

    assert lineage["evidence_summary"]["has_formal_strategy_version"] is True
    assert lineage["evidence_summary"]["formal_strategy_count"] == 1
    assert lineage["evidence_summary"]["formal_strategy_lifecycles"] == {"RESEARCH": 1}
    assert lineage["center"]["formal_strategy_refs"][0]["strategy_uid"] == strategy["strategy_uid"]
    assert lineage["formal_strategy_refs"][experiment_id][0]["version"] == strategy["version"]


def test_autocombine_candidates_carry_job_center_lineage(tmp_path: Path) -> None:
    store = _store_with_factors(tmp_path)
    combine_store = AutoCombineStore(store)
    task = combine_store.create_task(
        create_task_record(
            store,
            name="lineage combine",
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
            budget=AUTOCOMBINE_DEFAULT_BUDGET,
            notes="test",
        )
    )
    experiment = combine_store.record_experiment(
        task["task_id"],
        {
            "iteration": 1,
            "candidate_hash": "candidate-lineage",
            "action": "ADD",
            "proposal_source": "TEST",
            "factor_ids": ["F_1", "F_2"],
            "weights": [0.5, 0.5],
            "metrics": {"portfolio_sharpe_ratio": 1.2},
            "score": 1.2,
            "gate_distance": 0.0,
            "qualification": "RESEARCH_LEADER",
            "gate_status": "REJECTED",
            "failed_gates": ["public_validation"],
        },
    )

    build_strategy_bus_snapshot(
        store,
        autocombine_store=combine_store,
        quantcombine_store=QuantCombineStore(store),
    )
    experiment_id = stable_experiment_id(
        "AUTOCOMBINE", str(experiment["id"]), "COMBINATION_CANDIDATE"
    )
    node = store.strategy_experiment_object(experiment_id)
    lineage = strategy_experiment_lineage(store, experiment_id)

    assert node is not None
    assert node["evidence"]["system_job_id"] == external_job_id(
        "autocombine", task["task_id"]
    )
    assert "queue=autocombine" in node["evidence"]["job_center_url"]
    assert lineage["center"]["system_job_id"] == node["evidence"]["system_job_id"]


def test_strategy_library_surfaces_promotion_candidates(tmp_path: Path) -> None:
    store = _store_with_factors(tmp_path)
    weak_id = stable_experiment_id("TEST", "weak", "COMBINATION_CANDIDATE")
    strong_id = stable_experiment_id("TEST", "strong", "COMBINATION_CANDIDATE")
    qualified_id = stable_experiment_id("TEST", "qualified", "COMBINATION_CANDIDATE")
    for experiment_id, title, sharpe in (
        (weak_id, "Weak candidate", 0.3),
        (strong_id, "Strong candidate", 1.5),
    ):
        store.upsert_strategy_experiment_object(
            experiment_id=experiment_id,
            stage="COMBINATION_CANDIDATE",
            object_type="factor_combination",
            source_system="TEST",
            source_id=title,
            title=title,
            status="RESEARCH_LEADER",
            market="CN_A",
            metrics={
                "portfolio_sharpe_ratio": sharpe,
                "portfolio_simple_annual_return": 0.12,
                "portfolio_max_drawdown": -0.08,
                "portfolio_walk_forward_worst_sharpe": 0.2,
            },
            evidence={"factor_ids": ["F_1", "F_2"], "weights": [0.6, 0.4]},
        )
    store.upsert_strategy_experiment_object(
        experiment_id=qualified_id,
        stage="COMBINATION_CANDIDATE",
        object_type="factor_combination",
        source_system="TEST",
        source_id="qualified",
        title="Qualified candidate",
        status="QUALIFIED_CHAMPION",
        market="CN_A",
        metrics={
            "portfolio_sharpe_ratio": 0.8,
            "portfolio_simple_annual_return": 0.10,
            "portfolio_max_drawdown": -0.05,
            "portfolio_walk_forward_worst_sharpe": 0.4,
        },
        evidence={
            "factor_ids": ["F_1", "F_2"],
            "weights": [0.5, 0.5],
            "gate_status": "PASSED",
        },
    )

    candidates = strategy_promotion_candidates(store)

    assert [item["experiment_id"] for item in candidates] == [
        strong_id,
        qualified_id,
        weak_id,
    ]
    assert candidates[0]["candidate_class"] == "RESEARCH_LEADER"
    assert candidates[0]["freeze_ready_after_creation"] is False
    assert candidates[0]["next_action"] == "CREATE_RESEARCH_VERSION_FOR_REVIEW"
    assert candidates[0]["public_validation_gap"]["root_causes"] == [
        "MISSING_GATE_TELEMETRY"
    ]
    assert candidates[0]["operator_hint"] == "INSPECT_SOURCE_CANDIDATE_GATE_TELEMETRY"
    assert candidates[0]["promotion_path"][0] == "CREATE_RESEARCH_VERSION"
    assert candidates[0]["production_evidence_summary"]["evidence_state"] == (
        "READY_TO_CREATE_RESEARCH_VERSION"
    )
    assert "public_validation_gate_passed" in candidates[0]["production_evidence_summary"][
        "missing_or_blocking_evidence"
    ]
    qualified = next(item for item in candidates if item["experiment_id"] == qualified_id)
    assert qualified["candidate_class"] == "QUALIFIED"
    assert qualified["freeze_ready_after_creation"] is True
    assert qualified["next_action"] == "CREATE_RESEARCH_VERSION_AND_FREEZE"
    assert qualified["public_validation_gap"] is None
    assert qualified["operator_hint"] == "READY_TO_FREEZE_PUBLIC_VALIDATION"
    assert qualified["production_evidence_summary"]["evidence_state"] == (
        "READY_TO_CREATE_AND_FREEZE"
    )
    assert qualified["production_evidence_summary"]["metric_coverage"][
        "has_portfolio_walk_forward_worst_sharpe"
    ] is True


def test_strategy_experiment_lineage_returns_upstream_and_downstream_evidence(
    tmp_path: Path,
) -> None:
    store = _store_with_factors(tmp_path)
    factor_id = stable_experiment_id("TEST", "factor", "FACTOR_CANDIDATE")
    cluster_id = stable_experiment_id("TEST", "cluster", "FACTOR_CLUSTER")
    combo_id = stable_experiment_id("TEST", "combo", "COMBINATION_CANDIDATE")
    for experiment_id, stage, title in (
        (factor_id, "FACTOR_CANDIDATE", "Factor"),
        (cluster_id, "FACTOR_CLUSTER", "Cluster"),
        (combo_id, "COMBINATION_CANDIDATE", "Combo"),
    ):
        store.upsert_strategy_experiment_object(
            experiment_id=experiment_id,
            stage=stage,
            object_type=stage.casefold(),
            source_system="TEST",
            source_id=title,
            title=title,
            status="ACTIVE",
            market="CN_A",
            metrics={"portfolio_sharpe_ratio": 1.2},
            evidence={"gate_status": "PASSED", "factor_ids": ["F_1"]},
        )
    store.upsert_strategy_experiment_edge(factor_id, cluster_id, "BELONGS_TO_CLUSTER")
    store.upsert_strategy_experiment_edge(cluster_id, combo_id, "USED_BY_COMBINATION")

    lineage = strategy_experiment_lineage(store, cluster_id, depth=2)

    assert lineage["protocol"] == "AUTOALPHA_STRATEGY_EXPERIMENT_LINEAGE_V1"
    assert lineage["center"]["experiment_id"] == cluster_id
    assert factor_id in lineage["upstream_experiment_ids"]
    assert combo_id in lineage["downstream_experiment_ids"]
    assert lineage["evidence_summary"]["node_count"] == 3
    assert lineage["evidence_summary"]["relations"] == {
        "BELONGS_TO_CLUSTER": 1,
        "USED_BY_COMBINATION": 1,
    }


def test_strategy_production_funnel_exposes_conversion_bottlenecks(tmp_path: Path) -> None:
    store = _store_with_factors(tmp_path)
    build_strategy_bus_snapshot(
        store,
        autocombine_store=AutoCombineStore(store),
        quantcombine_store=QuantCombineStore(store),
    )
    experiment_id = stable_experiment_id("TEST", "candidate-funnel", "COMBINATION_CANDIDATE")
    store.upsert_strategy_experiment_object(
        experiment_id=experiment_id,
        stage="COMBINATION_CANDIDATE",
        object_type="factor_combination",
        source_system="TEST",
        source_id="candidate-funnel",
        title="Funnel candidate",
        status="RESEARCH_LEADER",
        market="CN_A",
        evidence={
            "factor_ids": ["F_1", "F_2"],
            "weights": [0.6, 0.4],
            "failed_gates": ["drawdown", "correlation", "deflated_sharpe_probability"],
        },
    )
    strategy = create_formal_strategy_from_experiment(store, experiment_id)

    funnel = strategy_production_funnel(store)
    stages = {item["key"]: item for item in funnel["stages"]}

    assert funnel["protocol"] == "AUTOALPHA_STRATEGY_PRODUCTION_FUNNEL_V1"
    assert stages["factor_candidates"]["count"] == 2
    assert stages["combination_candidates"]["count"] == 1
    assert stages["formal_research"]["count"] == 1
    assert funnel["formal_lifecycle"] == {strategy["lifecycle"]: 1}
    assert ("drawdown", 1) in funnel["top_failed_gates"]
    assert ("RISK_CONSTRAINT_BREACH", 1) in funnel["top_root_causes"]
    assert ("FACTOR_INDEPENDENCE_INSUFFICIENT", 1) in funnel["top_root_causes"]
    assert (
        "REDUCE_EFFECTIVE_TRIAL_COUNT_AND_DEDUPLICATE_SEARCH_SPACE",
        1,
    ) in funnel["top_operator_hints"]
    assert any(item["key"] == "research_versions_not_frozen" for item in funnel["bottlenecks"])


def test_strategy_production_funnel_surfaces_gate_feedback_repair_tasks(
    tmp_path: Path,
) -> None:
    store = _store_with_factors(tmp_path)
    quant_store = QuantCombineStore(store)
    record = create_quant_task_record(
        store,
        name="Repair task",
        market="CN_A",
        data_path=str(tmp_path),
        protocol={
            "exploration_start": "2010-01-01",
            "exploration_end": "2017-12-31",
            "validation_start": "2018-01-01",
            "validation_end": "2024-12-31",
            "holdout_start": "2025-01-01",
            "holdout_end": "2026-07-16",
            "minimum_folds": 1,
        },
        scope={"statuses": ["ELIGIBLE"]},
        construction={"min_factors": 2, "max_factors": 2, "candidate_pool_limit": 5},
        objective={
            "profile": "DIVERSIFICATION_FIRST",
            "maximum_drawdown": 0.18,
            "maximum_factor_correlation": 0.60,
        },
        engine=DEFAULT_ENGINE,
        budget=DEFAULT_BUDGET,
        notes="[gate-feedback:GATE_FUNNEL_FEEDBACK_POLICY_V1] repair",
    )
    task = quant_store.create_task(record)

    funnel = strategy_production_funnel(store)
    stages = {item["key"]: item for item in funnel["stages"]}

    assert stages["repair_tasks"]["count"] == 1
    assert funnel["repair_tasks"][0]["task_id"] == task["task_id"]
    assert funnel["repair_tasks"][0]["objective_profile"] == "DIVERSIFICATION_FIRST"
    assert funnel["repair_tasks"][0]["maximum_drawdown"] == 0.18


def test_formal_strategy_creation_rejects_factor_candidate_source(tmp_path: Path) -> None:
    store = _store_with_factors(tmp_path)
    factor_experiment_id = stable_experiment_id("AUTOALPHA", "F_1", "FACTOR_CANDIDATE")
    build_strategy_bus_snapshot(
        store,
        autocombine_store=AutoCombineStore(store),
        quantcombine_store=QuantCombineStore(store),
    )

    with pytest.raises(ValueError, match="combination candidates"):
        create_formal_strategy_from_experiment(store, factor_experiment_id)


def test_factor_knowledge_map_groups_sources_and_failures(tmp_path: Path) -> None:
    store = _store_with_factors(tmp_path)
    result = factor_knowledge_map(store)

    assert result["factor_count"] == 2
    assert result["cluster_count"] >= 1
    assert result["primary_metric_policy"] == "long_only_first"


def test_factor_knowledge_map_uses_nested_long_only_scores(tmp_path: Path) -> None:
    store = _store_with_factors(tmp_path)
    result = factor_knowledge_map(store)
    cluster = next(item for item in result["clusters"] if item["cluster_id"] == "C001")

    assert cluster["leader_factor_id"] == "F_2"
    assert cluster["average_long_only_score"] > 0
    assert cluster["top_factors"][0]["score"] > cluster["top_factors"][1]["score"]


def test_factor_knowledge_map_exposes_research_map_views(tmp_path: Path) -> None:
    store = ServiceStore(tmp_path / "autoalpha.sqlite3")
    store.upsert_factor_pool(
        factor_id="F_WEAK_LEGACY",
        source_iteration=1,
        source_task_id="task-map",
        proposal={
            **_proposal("Legacy crowded", "Liquidity", "vol"),
            "expression": {
                "operator": "rolling_mean",
                "parameters": {"window": 20},
                "arguments": [
                    {"operator": "field", "parameters": {"name": "vol"}, "arguments": []}
                ],
            },
        },
        metrics={
            "sharpe_ratio": 20.0,
            "long_only_sharpe_ratio": 0.1,
            "long_only_simple_annual_return": 0.01,
            "long_only_max_drawdown": -0.40,
            "annual_returns": {"2020": -0.10, "2021": 0.02},
            "homogeneity_nearest_factor_id": "F_STRONG_LONG",
            "homogeneity_nearest_similarity": 0.91,
        },
        status="ELIGIBLE",
        status_reason="test",
    )
    store.upsert_factor_pool(
        factor_id="F_STRONG_LONG",
        source_iteration=2,
        source_task_id="task-map",
        proposal={
            **_proposal("Strong long", "liquidity", "amount"),
            "expression": {
                "operator": "rolling_mean",
                "parameters": {"window": 60},
                "arguments": [
                    {"operator": "field", "parameters": {"name": "amount"}, "arguments": []}
                ],
            },
        },
        metrics={
            "sharpe_ratio": 0.1,
            "long_only_sharpe_ratio": 1.8,
            "long_only_simple_annual_return": 0.22,
            "long_only_max_drawdown": -0.08,
            "annual_returns": {"2020": 0.04, "2021": 0.09},
        },
        status="ELIGIBLE",
        status_reason="test",
    )

    result = factor_knowledge_map(
        store,
        behavior_snapshot={
            "factors": {
                "F_WEAK_LEGACY": {
                    "behavior_cluster_id": "CROWDED_001",
                    "behavior_redundancy": "PARAMETER_VARIANT",
                },
                "F_STRONG_LONG": {
                    "behavior_cluster_id": "CROWDED_001",
                    "behavior_redundancy": "LEADER",
                },
            }
        },
    )

    assert result["research_map_protocol"] == "FACTOR_KNOWLEDGE_RESEARCH_MAP_V2"
    folded = result["homogeneity_fold_groups"][0]
    assert folded["cluster_id"] == "CROWDED_001"
    assert folded["leader_factor_id"] == "F_STRONG_LONG"
    assert folded["parameter_family_count"] == 2
    assert folded["redundancy_counts"]["PARAMETER_VARIANT"] == 1
    assert result["mechanism_map"][0]["leader_factor_id"] == "F_STRONG_LONG"
    assert {item["parameter_family"] for item in result["parameter_families"]} == {
        "window=20",
        "window=60",
    }
    heatmap = result["annual_heatmap"]
    assert heatmap["years"] == ["2020", "2021"]
    liquidity_row = heatmap["rows"][0]
    assert liquidity_row["mechanism"] == "TURNOVER_LIQUIDITY"
    assert liquidity_row["annual_returns"]["2020"] == pytest.approx(-0.03)
    assert liquidity_row["weak_years"] == ["2020"]


def test_formal_strategy_cannot_skip_initial_research_lifecycle(tmp_path: Path) -> None:
    store = _store_with_factors(tmp_path)
    experiment_id = stable_experiment_id("TEST", "candidate-prod", "COMBINATION_CANDIDATE")
    store.upsert_strategy_experiment_object(
        experiment_id=experiment_id,
        stage="COMBINATION_CANDIDATE",
        object_type="factor_combination",
        source_system="TEST",
        source_id="candidate-prod",
        title="Production skip attempt",
        status="RESEARCH_LEADER",
        market="CN_A",
        evidence={"factor_ids": ["F_1"], "weights": [1.0]},
    )

    with pytest.raises(ValueError, match="starts at RESEARCH"):
        create_formal_strategy_from_experiment(
            store,
            experiment_id,
            lifecycle="PRODUCTION_CANDIDATE",
        )


def test_formal_strategy_promotion_requires_ordered_evidence(tmp_path: Path) -> None:
    store = _store_with_factors(tmp_path)
    experiment_id = stable_experiment_id("TEST", "candidate-promote", "COMBINATION_CANDIDATE")
    store.upsert_strategy_experiment_object(
        experiment_id=experiment_id,
        stage="COMBINATION_CANDIDATE",
        object_type="factor_combination",
        source_system="TEST",
        source_id="candidate-promote",
        title="Promotion candidate",
        status="RESEARCH_LEADER",
        market="CN_A",
        evidence={"factor_ids": ["F_1", "F_2"], "weights": [0.5, 0.5]},
    )
    strategy = create_formal_strategy_from_experiment(store, experiment_id)

    with pytest.raises(ValueError, match="Invalid strategy lifecycle transition"):
        promote_formal_strategy_lifecycle(
            store,
            strategy["strategy_uid"],
            strategy["version"],
            target_lifecycle="SHADOW",
            evidence={"hidden_holdout_passed": True, "holdout_evaluation_id": "H1"},
        )
    readiness = strategy_lifecycle_readiness(store, strategy["strategy_uid"], strategy["version"])
    assert readiness["next_lifecycle"] == "FROZEN"
    assert readiness["ready"] is False
    assert "public_validation_passed" in readiness["missing_evidence"]
    assert readiness["public_validation_gap"]["source_status"] == "RESEARCH_LEADER"
    assert readiness["public_validation_gap"]["root_causes"] == ["MISSING_GATE_TELEMETRY"]

    with pytest.raises(ValueError, match="public_validation_passed"):
        promote_formal_strategy_lifecycle(
            store,
            strategy["strategy_uid"],
            strategy["version"],
            target_lifecycle="FROZEN",
            evidence={
                "source_experiment_id": experiment_id,
                "factor_ids": ["F_1", "F_2"],
                "weights": [0.5, 0.5],
            },
        )

    with pytest.raises(ValueError, match="source_experiment_public_validation_not_passed"):
        promote_formal_strategy_lifecycle(
            store,
            strategy["strategy_uid"],
            strategy["version"],
            target_lifecycle="FROZEN",
            evidence={
                "source_experiment_id": experiment_id,
                "factor_ids": ["F_1", "F_2"],
                "weights": [0.5, 0.5],
                "public_validation_passed": True,
            },
        )
    store.upsert_strategy_experiment_object(
        experiment_id=experiment_id,
        stage="COMBINATION_CANDIDATE",
        object_type="factor_combination",
        source_system="TEST",
        source_id="candidate-promote",
        title="Promotion candidate",
        status="QUALIFIED_CHAMPION",
        market="CN_A",
        evidence={
            "factor_ids": ["F_1", "F_2"],
            "weights": [0.5, 0.5],
            "gate_status": "PASSED",
        },
    )
    frozen = promote_formal_strategy_lifecycle(
        store,
        strategy["strategy_uid"],
        strategy["version"],
        target_lifecycle="FROZEN",
        evidence={
            "source_experiment_id": experiment_id,
            "factor_ids": ["F_1", "F_2"],
            "weights": [0.5, 0.5],
            "public_validation_passed": True,
        },
    )
    assert frozen["lifecycle"] == "FROZEN"
    assert frozen["evidence"]["last_transition_validation"]["source_gate_status"] == "PASSED"
    hidden_readiness = strategy_lifecycle_readiness(
        store, strategy["strategy_uid"], strategy["version"]
    )
    assert hidden_readiness["next_lifecycle"] == "HIDDEN_HOLDOUT"
    assert hidden_readiness["ready"] is False
    assert "holdout_evaluation_requested" in hidden_readiness["missing_evidence"]
    assert hidden_readiness["suggested_evidence"]["frozen_specification_hash"] == frozen[
        "specification_hash"
    ]


def test_formal_strategy_approval_records_human_evidence_without_bypassing_gates(
    tmp_path: Path,
) -> None:
    store = _store_with_factors(tmp_path)
    blocked_id = stable_experiment_id("TEST", "approval-blocked", "COMBINATION_CANDIDATE")
    store.upsert_strategy_experiment_object(
        experiment_id=blocked_id,
        stage="COMBINATION_CANDIDATE",
        object_type="factor_combination",
        source_system="TEST",
        source_id="approval-blocked",
        title="Blocked approval strategy",
        status="RESEARCH_LEADER",
        market="CN_A",
        evidence={
            "factor_ids": ["F_1", "F_2"],
            "weights": [0.5, 0.5],
            "gate_status": "REJECTED",
            "failed_gates": ["public_validation"],
        },
    )
    blocked = create_formal_strategy_from_experiment(store, blocked_id)

    with pytest.raises(ValueError, match="public_validation_passed"):
        approve_formal_strategy_transition(
            store,
            blocked["strategy_uid"],
            blocked["version"],
            approver="risk-owner",
            approval_type="PUBLIC_VALIDATION_REVIEW",
            notes="cannot override failed public gates",
        )

    passed_id = stable_experiment_id("TEST", "approval-passed", "COMBINATION_CANDIDATE")
    store.upsert_strategy_experiment_object(
        experiment_id=passed_id,
        stage="COMBINATION_CANDIDATE",
        object_type="factor_combination",
        source_system="TEST",
        source_id="approval-passed",
        title="Passed approval strategy",
        status="QUALIFIED_CHAMPION",
        market="CN_A",
        evidence={
            "factor_ids": ["F_1", "F_2"],
            "weights": [0.5, 0.5],
            "gate_status": "PASSED",
        },
    )
    passed = create_formal_strategy_from_experiment(store, passed_id)

    frozen = approve_formal_strategy_transition(
        store,
        passed["strategy_uid"],
        passed["version"],
        approver="risk-owner",
        approval_type="PUBLIC_VALIDATION_REVIEW",
        notes="public gates reviewed",
    )

    assert frozen["lifecycle"] == "FROZEN"
    assert frozen["evidence"]["human_approval"]["approver"] == "risk-owner"
    assert frozen["evidence"]["human_approval"]["approval_type"] == (
        "PUBLIC_VALIDATION_REVIEW"
    )
    assert frozen["evidence"]["public_validation_passed"] is True


def test_formal_strategy_auto_advance_uses_readiness_only(tmp_path: Path) -> None:
    store = _store_with_factors(tmp_path)
    blocked_id = stable_experiment_id("TEST", "blocked-auto", "COMBINATION_CANDIDATE")
    store.upsert_strategy_experiment_object(
        experiment_id=blocked_id,
        stage="COMBINATION_CANDIDATE",
        object_type="factor_combination",
        source_system="TEST",
        source_id="blocked-auto",
        title="Blocked auto strategy",
        status="RESEARCH_LEADER",
        market="CN_A",
        evidence={
            "factor_ids": ["F_1", "F_2"],
            "weights": [0.5, 0.5],
            "gate_status": "REJECTED",
            "failed_gates": ["strategy_independence"],
        },
    )
    blocked = create_formal_strategy_from_experiment(store, blocked_id)

    with pytest.raises(ValueError, match="public_validation_passed"):
        advance_formal_strategy_lifecycle(store, blocked["strategy_uid"], blocked["version"])

    passed_id = stable_experiment_id("TEST", "passed-auto", "COMBINATION_CANDIDATE")
    store.upsert_strategy_experiment_object(
        experiment_id=passed_id,
        stage="COMBINATION_CANDIDATE",
        object_type="factor_combination",
        source_system="TEST",
        source_id="passed-auto",
        title="Passed auto strategy",
        status="QUALIFIED_CHAMPION",
        market="CN_A",
        evidence={
            "factor_ids": ["F_1", "F_2"],
            "weights": [0.5, 0.5],
            "gate_status": "PASSED",
        },
    )
    passed = create_formal_strategy_from_experiment(store, passed_id)

    frozen = advance_formal_strategy_lifecycle(store, passed["strategy_uid"], passed["version"])
    assert frozen["lifecycle"] == "FROZEN"
    with pytest.raises(ValueError, match="holdout_evaluation_requested"):
        advance_formal_strategy_lifecycle(store, passed["strategy_uid"], passed["version"])
    hidden = promote_formal_strategy_lifecycle(
        store,
        passed["strategy_uid"],
        passed["version"],
        target_lifecycle="HIDDEN_HOLDOUT",
        evidence={
            "frozen_specification_hash": frozen["specification_hash"],
            "holdout_evaluation_requested": True,
        },
    )
    assert hidden["lifecycle"] == "HIDDEN_HOLDOUT"
    assert hidden["evidence"]["promotion_trail"][0]["evidence"]["public_validation_passed"] is True


def test_formal_strategy_transition_validates_stage_specific_evidence(
    tmp_path: Path,
) -> None:
    store = _store_with_factors(tmp_path)
    experiment_id = stable_experiment_id("TEST", "stage-specific", "COMBINATION_CANDIDATE")
    store.upsert_strategy_experiment_object(
        experiment_id=experiment_id,
        stage="COMBINATION_CANDIDATE",
        object_type="factor_combination",
        source_system="TEST",
        source_id="stage-specific",
        title="Stage Specific",
        status="QUALIFIED_CHAMPION",
        market="CN_A",
        evidence={"factor_ids": ["F_1", "F_2"], "weights": [0.5, 0.5], "gate_status": "PASSED"},
    )
    strategy = create_formal_strategy_from_experiment(store, experiment_id)
    frozen = advance_formal_strategy_lifecycle(store, strategy["strategy_uid"], strategy["version"])

    with pytest.raises(ValueError, match="frozen_specification_hash_mismatch"):
        promote_formal_strategy_lifecycle(
            store,
            strategy["strategy_uid"],
            strategy["version"],
            target_lifecycle="HIDDEN_HOLDOUT",
            evidence={
                "frozen_specification_hash": "bad-hash",
                "holdout_evaluation_requested": True,
            },
        )
    hidden = promote_formal_strategy_lifecycle(
        store,
        strategy["strategy_uid"],
        strategy["version"],
        target_lifecycle="HIDDEN_HOLDOUT",
        evidence={
            "frozen_specification_hash": frozen["specification_hash"],
            "holdout_evaluation_requested": True,
        },
    )
    assert hidden["lifecycle"] == "HIDDEN_HOLDOUT"

    with pytest.raises(ValueError, match="hidden_holdout_passed"):
        promote_formal_strategy_lifecycle(
            store,
            strategy["strategy_uid"],
            strategy["version"],
            target_lifecycle="SHADOW",
            evidence={"hidden_holdout_passed": False, "holdout_evaluation_id": "HOLDOUT-1"},
        )
    shadow = promote_formal_strategy_lifecycle(
        store,
        strategy["strategy_uid"],
        strategy["version"],
        target_lifecycle="SHADOW",
        evidence={"hidden_holdout_passed": True, "holdout_evaluation_id": "HOLDOUT-1"},
    )
    assert shadow["lifecycle"] == "SHADOW"

    with pytest.raises(ValueError, match="shadow_trading_days_must_be_positive"):
        promote_formal_strategy_lifecycle(
            store,
            strategy["strategy_uid"],
            strategy["version"],
            target_lifecycle="PAPER",
            evidence={"shadow_trading_days": 0, "shadow_execution_passed": True},
        )
    paper = promote_formal_strategy_lifecycle(
        store,
        strategy["strategy_uid"],
        strategy["version"],
        target_lifecycle="PAPER",
        evidence={"shadow_trading_days": 5, "shadow_execution_passed": True},
    )
    assert paper["lifecycle"] == "PAPER"

    with pytest.raises(ValueError, match="strict_pit_market_state_must_be_verified"):
        promote_formal_strategy_lifecycle(
            store,
            strategy["strategy_uid"],
            strategy["version"],
            target_lifecycle="PRODUCTION_CANDIDATE",
            evidence={
                "paper_trading_days": 5,
                "paper_trading_passed": True,
                "risk_approval": "APPROVED",
            },
        )


def test_system_job_queue_records_progress_and_summary(tmp_path: Path) -> None:
    store = ServiceStore(tmp_path / "autoalpha.sqlite3")
    job = store.enqueue_system_job(
        job_id="job-test",
        queue="batch",
        job_type="factor_reevaluation",
        payload={"factor_count": 100},
        max_workers=6,
        progress_total=100,
    )

    assert job["status"] == "QUEUED"
    updated = store.update_system_job(
        "job-test",
        status="RUNNING",
        progress_current=25,
        checkpoint={"offset": 25},
    )

    assert updated["checkpoint"] == {"offset": 25}
    assert store.system_job_summary()["queues"]["batch"]["RUNNING"] == 1


def test_system_job_claim_heartbeat_and_recover(tmp_path: Path) -> None:
    store = ServiceStore(tmp_path / "autoalpha.sqlite3")
    store.enqueue_system_job(
        job_id="job-claim",
        queue="batch",
        job_type="factor_reevaluation",
        payload={"factor_count": 100},
        max_workers=6,
        progress_total=100,
    )

    claimed = store.claim_system_job(queue="batch", worker_id="worker-1", lease_seconds=30)

    assert claimed is not None
    assert claimed["status"] == "RUNNING"
    assert claimed["lease_owner"] == "worker-1"
    heartbeat = store.heartbeat_system_job(
        "job-claim",
        worker_id="worker-1",
        progress_current=40,
        checkpoint={"offset": 40},
    )
    assert heartbeat["progress_current"] == 40
    assert heartbeat["checkpoint"] == {"offset": 40}
    store.update_system_job(
        "job-claim",
        lease_expires_at="2000-01-01T00:00:00+00:00",
    )
    summary_before_recovery = store.system_job_summary()

    assert summary_before_recovery["expired_running_count"] == 1
    assert summary_before_recovery["expired_running"] == [
        {"queue": "batch", "resource_group": "default", "count": 1}
    ]
    assert store.recover_expired_system_jobs(queue="batch") == 1
    assert store.system_job("job-claim")["status"] == "QUEUED"


def test_system_job_claim_respects_resource_group_capacity(tmp_path: Path) -> None:
    store = ServiceStore(tmp_path / "autoalpha.sqlite3")
    for job_id, group in (
        ("job-db-1", "sqlite-writer"),
        ("job-db-2", "sqlite-writer"),
        ("job-cpu-1", "cpu-batch"),
    ):
        store.enqueue_system_job(
            job_id=job_id,
            queue="batch",
            job_type="factor_reevaluation",
            payload={},
            resource_group=group,
            max_workers=1,
        )

    first = store.claim_system_job(
        queue="batch",
        worker_id="worker-db-1",
        resource_group="sqlite-writer",
    )
    blocked = store.claim_system_job(
        queue="batch",
        worker_id="worker-db-2",
        resource_group="sqlite-writer",
    )
    independent = store.claim_system_job(
        queue="batch",
        worker_id="worker-cpu-1",
        resource_group="cpu-batch",
    )
    summary = store.system_job_summary()

    assert first is not None
    assert first["job_id"] == "job-db-1"
    assert blocked is None
    assert independent is not None
    assert independent["job_id"] == "job-cpu-1"
    assert summary["resources"]["batch"]["sqlite-writer"]["statuses"]["RUNNING"] == 1
    assert summary["resources"]["batch"]["sqlite-writer"]["capacity"] == 1


def test_system_job_resource_group_capacity_uses_strictest_active_policy(tmp_path: Path) -> None:
    store = ServiceStore(tmp_path / "autoalpha.sqlite3")
    store.enqueue_system_job(
        job_id="job-strict",
        queue="batch",
        job_type="factor_reevaluation",
        payload={},
        resource_group="sqlite-writer",
        max_workers=1,
        priority=10,
    )
    store.enqueue_system_job(
        job_id="job-loose",
        queue="batch",
        job_type="factor_reevaluation",
        payload={},
        resource_group="sqlite-writer",
        max_workers=2,
        priority=20,
    )

    first = store.claim_system_job(queue="batch", worker_id="worker-1")
    second = store.claim_system_job(queue="batch", worker_id="worker-2")
    summary = store.system_job_summary()

    assert first is not None
    assert first["job_id"] == "job-strict"
    assert second is None
    assert store.system_job("job-loose")["status"] == "QUEUED"
    assert summary["resources"]["batch"]["sqlite-writer"]["capacity"] == 1
