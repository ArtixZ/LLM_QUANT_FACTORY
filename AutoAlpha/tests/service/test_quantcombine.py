from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from autoalpha.service.autocombine import DEFAULT_CONSTRUCTION, DEFAULT_OBJECTIVE
from autoalpha.service.quantcombine import (
    DEFAULT_BUDGET,
    DEFAULT_ENGINE,
    QuantCombineWorker,
    _complete_linkage_groups,
    _stage_evaluation_limits,
    _standalone_stability_score,
    create_quant_task_record,
    pareto_ranks,
)
from autoalpha.service.quantcombine_store import QuantCombineStore
from autoalpha.service.store import ServiceStore


def _proposal(name: str, family: str, field: str) -> dict[str, object]:
    return {
        "name": name,
        "family": family,
        "hypothesis": f"Cross-sectional information in {field}",
        "expected_direction": 1,
        "expression": {"operator": "field", "parameters": {"name": field}, "arguments": []},
    }


def _base_store(tmp_path: Path) -> ServiceStore:
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


def _task_record(base: ServiceStore, tmp_path: Path) -> dict[str, object]:
    return create_quant_task_record(
        base,
        name="Test quant combine",
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
            "min_factors": 2,
            "max_factors": 2,
            "candidate_pool_limit": 10,
        },
        objective=DEFAULT_OBJECTIVE,
        engine=DEFAULT_ENGINE,
        budget=DEFAULT_BUDGET,
        notes="test",
    )


def test_quant_task_freezes_factor_registry_and_engine(tmp_path: Path) -> None:
    base = _base_store(tmp_path)
    record = _task_record(base, tmp_path)
    assert record["task_id"].startswith("qcombine-")  # type: ignore[union-attr]
    assert record["engine"]["mode"] == "ENSEMBLE"  # type: ignore[index]
    assert len(record["factor_snapshot"]) == 2  # type: ignore[arg-type]
    assert record["snapshot_hash"]


def test_quant_store_persists_screen_candidate_and_strategy(tmp_path: Path) -> None:
    base = _base_store(tmp_path)
    store = QuantCombineStore(base)
    task = store.create_task(_task_record(base, tmp_path))
    store.upsert_factor_screen(
        task["task_id"],
        {
            "factor_id": "F_1",
            "cluster_id": "QC_1",
            "cluster_leader": True,
            "stability_score": 1.2,
            "metrics": {"portfolio_sharpe_ratio": 1.0},
        },
    )
    candidate = store.record_candidate(
        task["task_id"],
        {
            "iteration": 1,
            "stage": "SFFS",
            "algorithm": "STABILITY_SEED",
            "action": "SEED",
            "candidate_hash": "candidate-1",
            "factor_ids": ["F_1", "F_2"],
            "weights": [0.5, 0.5],
            "metrics": {"portfolio_sharpe_ratio": 1.4},
            "objectives": [1.4, 0.1, -0.1, 0.2, -5.0, -0.3, 1.8, 2.0],
            "score": 1.0,
            "gate_distance": 0.0,
            "gate_status": "PASSED",
            "failed_gates": [],
        },
    )
    store.update_task(
        task["task_id"],
        best_candidate_id=candidate["id"],
        qualified_candidate_id=candidate["id"],
        production_candidate_id=candidate["id"],
        qualification_status="PRODUCTION_CANDIDATE",
    )
    strategy = store.promote_strategy(task["task_id"], candidate["id"], "Quant strategy")
    assert store.factor_screen(task["task_id"])[0]["cluster_leader"] is True
    assert store.candidate_by_hash(task["task_id"], "candidate-1")["id"] == candidate["id"]  # type: ignore[index]
    assert strategy["strategy_id"].startswith("QS_")
    assert strategy["specification"]["source"] == "QUANTCOMBINE"


def test_pareto_rank_preserves_tradeoff_candidates() -> None:
    candidates = [
        {"id": 1, "objectives": [2.0, 0.1]},
        {"id": 2, "objectives": [1.0, 0.2]},
        {"id": 3, "objectives": [0.5, 0.1]},
    ]
    ranks = pareto_ranks(candidates)
    assert ranks[1][0] == 0
    assert ranks[2][0] == 0
    assert ranks[3][0] == 1


def test_stability_score_rewards_robust_long_only_metrics() -> None:
    robust = _standalone_stability_score(
        {
            "portfolio_sharpe_ratio": 1.5,
            "portfolio_simple_annual_return": 0.15,
            "portfolio_active_information_ratio": 0.8,
            "portfolio_walk_forward_worst_sharpe": 0.2,
            "portfolio_walk_forward_positive_fraction": 0.8,
            "portfolio_max_drawdown": -0.12,
            "portfolio_annual_turnover": 5.0,
            "portfolio_deflated_sharpe_probability": 0.8,
        }
    )
    fragile = _standalone_stability_score(
        {
            "portfolio_sharpe_ratio": 1.5,
            "portfolio_simple_annual_return": 0.15,
            "portfolio_active_information_ratio": 0.8,
            "portfolio_walk_forward_worst_sharpe": -2.0,
            "portfolio_walk_forward_positive_fraction": 0.3,
            "portfolio_max_drawdown": -0.45,
            "portfolio_annual_turnover": 50.0,
            "portfolio_deflated_sharpe_probability": 0.1,
        }
    )
    assert robust > fragile


def test_complete_linkage_does_not_merge_a_correlation_chain() -> None:
    similarities = {
        frozenset(("A", "B")): True,
        frozenset(("B", "C")): True,
        frozenset(("A", "C")): False,
    }
    groups = _complete_linkage_groups(
        ["A", "B", "C"], lambda left, right: similarities[frozenset((left, right))]
    )
    assert groups == [["A", "B"], ["C"]]


def test_cluster_leader_must_pass_single_factor_coverage(tmp_path: Path) -> None:
    base = _base_store(tmp_path)
    store = QuantCombineStore(base)
    task = store.create_task(_task_record(base, tmp_path))
    for factor_id, score, coverage in (
        ("F_1", 2.0, 0.50),
        ("F_2", 1.0, 0.95),
    ):
        store.upsert_factor_screen(
            task["task_id"],
            {
                "factor_id": factor_id,
                "stability_score": score,
                "metrics": {"portfolio_coverage": coverage},
            },
        )
    worker = QuantCombineWorker(
        task["task_id"], base, store, config_path=tmp_path / "research.toml"
    )
    index = pd.date_range("2020-01-01", periods=100, freq="B")
    returns = pd.Series(np.linspace(-0.01, 0.01, len(index)), index=index)
    worker._standalone_returns = {"F_1": returns, "F_2": returns * 1.01}
    worker._cluster_factors()
    clustered = {item["factor_id"]: item for item in store.factor_screen(task["task_id"])}
    assert clustered["F_1"]["exclusion_reason"] == "LOW_COVERAGE"
    assert clustered["F_1"]["cluster_leader"] is False
    assert clustered["F_2"]["cluster_leader"] is True


def test_ensemble_reserves_budget_for_all_search_stages() -> None:
    sffs_limit, evolution_limit = _stage_evaluation_limits("ENSEMBLE", 10, 160)
    assert sffs_limit == 70
    assert evolution_limit == 122
    assert evolution_limit < 160


def test_worker_reserves_final_evaluations_for_candidate_diagnostics(tmp_path: Path) -> None:
    base = _base_store(tmp_path)
    store = QuantCombineStore(base)
    record = _task_record(base, tmp_path)
    record["budget"]["maximum_evaluations"] = 20  # type: ignore[index]
    task = store.create_task(record)
    worker = QuantCombineWorker(
        task["task_id"], base, store, config_path=tmp_path / "research.toml"
    )

    limits: dict[str, int] = {}
    worker._screen_factors = lambda evaluator: None  # type: ignore[method-assign]
    worker._cluster_factors = lambda: None  # type: ignore[method-assign]
    worker._searchable_factor_ids = lambda: ["F_1", "F_2"]  # type: ignore[method-assign]
    worker._run_sffs = (  # type: ignore[method-assign]
        lambda evaluator, pool, *, evaluation_limit: limits.update(sffs=evaluation_limit)
    )
    worker._run_evolution = (  # type: ignore[method-assign]
        lambda evaluator, pool, *, evaluation_limit: limits.update(evolution=evaluation_limit)
    )
    worker._run_adaptive = (  # type: ignore[method-assign]
        lambda evaluator, pool, *, evaluation_limit: limits.update(adaptive=evaluation_limit)
    )
    worker._refresh_pareto = lambda: None  # type: ignore[method-assign]
    worker._qualify_best = lambda evaluator: None  # type: ignore[method-assign]

    worker._run_sync(object())  # type: ignore[arg-type]

    # max_factors=2, so the final two evaluations are not consumed by search.
    assert limits["adaptive"] == 18
    assert limits["evolution"] < limits["adaptive"]


def test_quantcombine_best_candidate_links_to_prefilled_workflows() -> None:
    static_root = Path(__file__).resolve().parents[2] / "src/autoalpha/service/static"
    html = (static_root / "quantcombine.html").read_text()
    script = (static_root / "quantcombine.js").read_text()
    screener = (static_root / "screener.js").read_text()
    backtest = (static_root / "backtest.js").read_text()

    assert 'id="quickScreenButton"' in html
    assert 'id="quickBacktestButton"' in html
    assert 'id="headerQuickScreenButton"' in html
    assert 'id="headerQuickBacktestButton"' in html
    assert 'url.port = "8788"' in script
    assert 'run: "1"' in script
    assert 'task.protocol.validation_start' in script
    assert 'backtest_preset: "A_SHARE_NON_PIT_PROXY_WEEKLY_V1"' in script
    assert 'backtest_engine: "EVENT_LEDGER"' in script
    assert 'execution_data_mode: "NON_PIT_PROXY"' in script
    assert 'queryParameters.get("weights")' in screener
    assert 'queryParameters.get("as_of_date")' in screener
    assert 'queryParameters.get("run") === "1"' in screener
    assert 'start_date: "startDate"' in backtest
    assert 'execution_data_mode: "executionDataMode"' in backtest
    assert 'product_template: "productTemplate"' in backtest
    assert 'rebalance_schedule: "rebalanceSchedule"' in backtest


def test_candidate_weight_search_does_not_cross_stage_limit(tmp_path: Path, monkeypatch) -> None:
    base = _base_store(tmp_path)
    store = QuantCombineStore(base)
    task = store.create_task(_task_record(base, tmp_path))
    store.update_task(task["task_id"], evaluation_count=5)
    worker = QuantCombineWorker(
        task["task_id"], base, store, config_path=tmp_path / "research.toml"
    )
    worker._weight_candidates = lambda factor_ids: [  # type: ignore[method-assign]
        (0.50, 0.50),
        (0.45, 0.55),
        (0.40, 0.60),
    ]
    worker._strategy_independence_metrics = lambda active: {}  # type: ignore[method-assign]
    monkeypatch.setattr(
        "autoalpha.service.quantcombine.write_return_artifact",
        lambda *args, **kwargs: (str(tmp_path / "returns.parquet"), "hash"),
    )

    index = pd.date_range("2020-01-01", periods=30, freq="B")
    calls = 0

    class Evaluator:
        def evaluate_portfolio(self, factors, *, weights):
            nonlocal calls
            calls += 1
            return SimpleNamespace(
                metrics={
                    "portfolio_sharpe_ratio": 1.0,
                    "portfolio_simple_annual_return": 0.1,
                    "portfolio_max_drawdown": -0.1,
                    "portfolio_walk_forward_worst_sharpe": 0.2,
                    "portfolio_walk_forward_positive_fraction": 1.0,
                    "portfolio_coverage": 1.0,
                    "portfolio_deflated_sharpe_probability": 1.0,
                },
                net_returns=pd.Series(0.001, index=index),
            )

        def _market_benchmark_returns(self, requested_index):
            return pd.Series(0.0, index=requested_index)

    worker._evaluate_subset(
        Evaluator(),  # type: ignore[arg-type]
        ("F_1", "F_2"),
        stage="ADAPTIVE",
        algorithm="BAYESIAN_INCLUSION",
        action="POSTERIOR_SAMPLE",
        evaluation_limit=7,
    )

    assert calls == 2
    assert store.task(task["task_id"])["evaluation_count"] == 7  # type: ignore[index]


def test_weight_optimizer_returns_bounded_discrete_portfolios(tmp_path: Path) -> None:
    base = _base_store(tmp_path)
    store = QuantCombineStore(base)
    task = store.create_task(_task_record(base, tmp_path))
    for factor_id, score in (("F_1", 0.5), ("F_2", 1.0)):
        store.upsert_factor_screen(
            task["task_id"],
            {
                "factor_id": factor_id,
                "stability_score": score,
                "metrics": {},
            },
        )
    worker = QuantCombineWorker(
        task["task_id"], base, store, config_path=tmp_path / "research.toml"
    )
    index = pd.date_range("2020-01-01", periods=200, freq="B")
    rng = np.random.default_rng(7)
    worker._standalone_returns = {
        "F_1": pd.Series(rng.normal(0.0005, 0.01, len(index)), index=index),
        "F_2": pd.Series(rng.normal(0.0004, 0.008, len(index)), index=index),
    }
    weights = worker._weight_candidates(("F_1", "F_2"))
    assert weights
    assert all(abs(sum(candidate) - 1.0) < 1e-8 for candidate in weights)
    assert all(0.05 <= value <= 0.50 for candidate in weights for value in candidate)
    assert all(
        abs((value / 0.05) - round(value / 0.05)) < 1e-8
        for candidate in weights
        for value in candidate
    )
