from autoalpha.service.factor_library import build_factor_library, factor_category


def _record(
    factor_id: str,
    *,
    status: str,
    family: str,
    sharpe: float,
    worst_fold: float,
    turnover: float,
) -> dict:
    return {
        "factor_id": factor_id,
        "source_iteration": int(factor_id.removeprefix("F_")),
        "name": f"factor_{factor_id}",
        "family": family,
        "status": status,
        "status_reason": "test",
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "proposal": {
            "hypothesis": "test hypothesis",
            "expected_direction": 1,
            "expression": {
                "operator": "field",
                "arguments": [],
                "parameters": {"name": "adj_close"},
            },
        },
        "metrics": {
            "sharpe_ratio": sharpe,
            "simple_annual_return": sharpe / 10,
            "walk_forward_worst_sharpe": worst_fold,
            "walk_forward_positive_fraction": 0.8,
            "deflated_sharpe_probability": 0.9,
            "annual_return_dispersion": 0.1,
            "max_drawdown": -0.1,
            "worst_year_incremental_return": -0.05,
            "annual_turnover": turnover,
            "coverage": 0.95,
            "capacity_cny": 50_000_000,
            "rank_ic_ir": 1.0,
            "rank_ic_mean": 0.02,
            "pearson_ic_mean": 0.01,
        },
    }


def test_library_retains_screened_out_factors_as_observation_assets() -> None:
    library = build_factor_library(
        [
            _record(
                "F_1",
                status="ACTIVE",
                family="Price-Volume Interaction",
                sharpe=2.0,
                worst_fold=0.5,
                turnover=10,
            ),
            _record(
                "F_2",
                status="SCREENED_OUT",
                family="Liquidity",
                sharpe=0.8,
                worst_fold=-0.2,
                turnover=5,
            ),
        ]
    )

    assert library["summary"]["factor_count"] == 2
    assert library["summary"]["observed_count"] == 1
    observed = next(item for item in library["factors"] if item["factor_id"] == "F_2")
    assert observed["research_state"] == "OBSERVE"
    assert observed["promotion_eligible"] is False
    assert observed["scores"]["overall"] >= 0


def test_factor_category_normalizes_model_family_names() -> None:
    assert factor_category("mean_reversion") == "价格反转与均值回归"
    assert factor_category("Dollar Volume Volatility") == "流动性与交易活跃度"
    assert factor_category("Momentum") == "趋势与动量"


def test_library_marks_metrics_from_an_old_protocol_as_stale() -> None:
    record = _record(
        "F_1",
        status="ACTIVE",
        family="Momentum",
        sharpe=2.0,
        worst_fold=0.5,
        turnover=10,
    )
    record["metrics"]["evaluation_protocol"] = "old-protocol"

    library = build_factor_library([record], current_protocol="new-protocol")

    factor = library["factors"][0]
    assert factor["protocol_stale"]
    assert factor["research_state"] == "STALE_PROTOCOL"
    assert not factor["promotion_eligible"]
    assert factor["metric_summary"]["sharpe_ratio"] is None
    assert factor["historical_metric_summary"]["sharpe_ratio"] == 2.0
    assert factor["scores"]["overall"] == 0.0
    assert library["summary"]["eligible_count"] == 0
    assert library["summary"]["stale_protocol_count"] == 1


def test_library_uses_each_factor_source_task_protocol() -> None:
    current = _record(
        "F_1",
        status="ELIGIBLE",
        family="Momentum",
        sharpe=2.0,
        worst_fold=0.5,
        turnover=10,
    )
    stale = _record(
        "F_2",
        status="ELIGIBLE",
        family="Liquidity",
        sharpe=1.0,
        worst_fold=0.1,
        turnover=5,
    )
    current["source_task_id"] = "task-a"
    stale["source_task_id"] = "task-b"
    current["metrics"]["evaluation_protocol"] = "protocol-a"
    stale["metrics"]["evaluation_protocol"] = "protocol-a"

    library = build_factor_library(
        [current, stale],
        current_protocols={"task-a": "protocol-a", "task-b": "protocol-b"},
    )

    factors = {factor["factor_id"]: factor for factor in library["factors"]}
    assert not factors["F_1"]["protocol_stale"]
    assert factors["F_2"]["protocol_stale"]
    assert factors["F_2"]["current_protocol"] == "protocol-b"


def test_library_blocks_holdout_contaminated_factor_promotion() -> None:
    record = _record(
        "F_1",
        status="ELIGIBLE",
        family="Momentum",
        sharpe=2.0,
        worst_fold=0.5,
        turnover=10,
    )
    record["metrics"]["evaluation_protocol"] = "current"

    library = build_factor_library(
        [record],
        contaminated_factor_ids={"F_1"},
        current_protocol="current",
    )

    factor = library["factors"][0]
    assert factor["research_state"] == "HOLDOUT_CONTAMINATED"
    assert factor["promotion_eligible"] is False


def test_library_adds_clusters_lifecycle_contamination_and_marginal_contribution() -> None:
    records = [
        _record(
            "F_1",
            status="ELIGIBLE",
            family="Momentum",
            sharpe=1.2,
            worst_fold=0.2,
            turnover=8,
        ),
        _record(
            "F_2",
            status="SCREENED_OUT",
            family="Momentum",
            sharpe=0.9,
            worst_fold=0.1,
            turnover=9,
        ),
    ]
    diagnostics = {
        "F_1": {
            "iteration": 1,
            "metrics": {
                "portfolio_option_diagnostics": [
                    {
                        "action": "ADD",
                        "accepted": False,
                        "gate_failure_count": 1,
                        "utility_change": 0.1,
                        "failed_gates": ["turnover"],
                        "metrics": {"portfolio_incremental_net_ir": 0.42},
                    }
                ]
            },
        }
    }
    library = build_factor_library(
        records,
        lifecycle_states={"F_1": {"state": "SHADOW"}},
        contaminated_factor_ids={"F_1"},
        research_diagnostics=diagnostics,
    )

    factor = next(item for item in library["factors"] if item["factor_id"] == "F_1")
    assert factor["cluster_id"].startswith("C")
    assert factor["lifecycle_state"] == "SHADOW"
    assert factor["holdout_contaminated"]
    assert factor["marginal_contribution"]["incremental_net_ir"] == 0.42
