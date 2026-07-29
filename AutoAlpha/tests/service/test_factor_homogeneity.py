from __future__ import annotations

from autoalpha.service.factor_homogeneity import (
    build_homogeneity_report,
    candidate_homogeneity_metrics,
    factor_homogeneity_integrity,
)


def _record(factor_id: str, mechanism: str = "TURNOVER_LIQUIDITY") -> dict:
    return {
        "factor_id": factor_id,
        "name": factor_id,
        "family": mechanism,
        "source_task_id": "task-aa",
        "proposal": {
            "canonical_mechanism": mechanism,
            "hypothesis": mechanism,
            "expression": {"operator": "field", "field": "turnover_rate"},
        },
        "metrics": {
            "recent_long_only_sharpe_ratio": 1.0,
            "long_only_sharpe_ratio": 1.0,
        },
    }


def test_homogeneity_report_marks_crowded_behavior_cluster() -> None:
    records = [_record(f"F_{index:02d}") for index in range(9)]
    behavior = {
        "protocol": "TEST",
        "factors": {
            record["factor_id"]: {
                "behavior_cluster_id": "B_CROWD",
                "behavior_cluster_role": "LEADER" if index == 0 else "MEMBER",
                "behavior_cluster_size": 9,
            }
            for index, record in enumerate(records)
        },
    }

    report = build_homogeneity_report(records, behavior, source_task_id="task-aa")

    assert report["crowded_cluster_count"] == 1
    assert report["crowded_clusters"][0]["cluster_id"] == "B_CROWD"
    assert report["llm_directive"]["must_avoid_cluster_ids"] == ["B_CROWD"]


def test_candidate_homogeneity_blocks_crowded_substitute() -> None:
    records = [_record(f"F_{index:02d}") for index in range(9)]
    behavior = {
        "factors": {
            record["factor_id"]: {
                "behavior_cluster_id": "B_CROWD",
                "behavior_cluster_size": 9,
            }
            for record in records
        }
    }

    metrics = candidate_homogeneity_metrics(
        factor_id="F_NEW",
        candidate_mechanism="TURNOVER_LIQUIDITY",
        candidate_fields={"turnover_rate"},
        pool_records=records,
        behavior_snapshot=behavior,
        redundancy_metrics={
            "library_signal_correlation_peer": "F_01",
            "library_signal_correlation_max": 0.86,
            "online_behavior_cluster_id": "B_CROWD",
            "online_behavior_cluster_size": 10,
            "online_behavior_redundancy": "SUBSTITUTE",
        },
    )

    assert metrics["homogeneity_gate_passed"] is False
    assert metrics["homogeneity_gate_failure"] == "CROWDED_CLUSTER_SUBSTITUTE"
    assert metrics["homogeneity_crowded_cluster"] is True


def test_candidate_homogeneity_allows_distinct_sparse_factor() -> None:
    metrics = candidate_homogeneity_metrics(
        factor_id="F_NEW",
        candidate_mechanism="ORDER_FLOW",
        candidate_fields={"net_mf_amount"},
        pool_records=[_record("F_OLD", "TURNOVER_LIQUIDITY")],
        behavior_snapshot={"factors": {}},
        redundancy_metrics={
            "library_signal_correlation_peer": "F_OLD",
            "library_signal_correlation_max": 0.30,
            "online_behavior_cluster_id": "PENDING_NEW_CLUSTER",
            "online_behavior_cluster_size": 1,
            "online_behavior_redundancy": "DISTINCT",
        },
    )

    assert metrics["homogeneity_gate_passed"] is True
    assert metrics["homogeneity_novelty_score"] > 0.5


def test_factor_homogeneity_integrity_reports_missing_review_fields() -> None:
    pool = [_record("F_COMPLETE"), _record("F_MISSING")]
    knowledge = [
        {
            "factor_id": "F_COMPLETE",
            "canonical_mechanism": "TURNOVER_LIQUIDITY",
            "review": {
                "expression_signature": "rank(turnover_rate)",
                "parameter_family": "turnover_rate:rank",
                "behavior_cluster_id": "C001",
            },
        },
        {
            "factor_id": "F_MISSING",
            "canonical_mechanism": "",
            "review": {
                "expression_signature": "rank(volume)",
            },
        },
    ]

    report = factor_homogeneity_integrity(pool, knowledge)

    assert report["protocol"] == "AUTOALPHA_FACTOR_HOMOGENEITY_INTEGRITY_V1"
    assert report["complete"] is False
    assert report["missing_field_count"] == 1
    assert report["missing_fields_by_factor"]["F_MISSING"] == [
        "canonical_mechanism",
        "parameter_family",
        "behavior_cluster_id",
    ]
    assert report["field_coverage"]["expression_signature"]["count"] == 2
    assert report["recommended_job_type"] == "factor_homogeneity_backfill"
