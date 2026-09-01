import json

import numpy as np
import pandas as pd

from autoalpha.data.contracts import FieldSpec, TableContract
from autoalpha.data.snapshot import SnapshotStore
from autoalpha.dsl.compiler import FactorCompiler
from autoalpha.dsl.expression import FactorDefinition, field, operation
from autoalpha.dsl.semantics import FieldDefinition, SemanticValidator
from autoalpha.execution.simulator import ExecutionSimulator, ExecutionStyle, Order
from autoalpha.operations.artifacts import ArtifactRegistry
from autoalpha.operations.pipeline import IdempotentPipeline
from autoalpha.portfolio.optimizer import PortfolioConstraints, PortfolioOptimizer
from autoalpha.registry.store import FactorRegistry
from autoalpha.research.evaluation import investment_diagnostics


def test_point_in_time_factor_to_execution_artifact_is_replayable(tmp_path) -> None:
    rng = np.random.default_rng(21)
    dates = pd.bdate_range("2024-01-02", periods=30)
    symbols = [f"S{i:03d}" for i in range(40)]
    close = pd.DataFrame(
        10 * np.exp(np.cumsum(rng.normal(scale=0.01, size=(30, 40)), axis=0)),
        index=dates,
        columns=symbols,
    )
    rows = close.stack().rename("close").reset_index()
    rows.columns = ["event_date", "symbol", "close"]
    rows["knowledge_time"] = pd.to_datetime(rows["event_date"], utc=True) + pd.Timedelta(hours=8)
    contract = TableContract(
        "prices",
        "1.0",
        (
            FieldSpec("event_date", "date", False, "trading date"),
            FieldSpec("symbol", "string", False, "security identifier"),
            FieldSpec("close", "float", False, "unadjusted close", "USD/share"),
            FieldSpec("knowledge_time", "timestamp", False, "source-visible timestamp"),
        ),
        primary_key=("event_date", "symbol", "knowledge_time"),
        entity_key=("event_date", "symbol"),
        event_time="event_date",
        knowledge_time="knowledge_time",
    )
    snapshot_store = SnapshotStore(tmp_path / "snapshots")
    snapshot = snapshot_store.write("S1", [(rows, contract)], source="synthetic-pit")
    snapshot_store.verify(snapshot.snapshot_id)

    expression = operation(
        "cs_rank", operation("negate", operation("returns", field("close"), periods=5))
    )
    definition = FactorDefinition(
        "five_day_reversal", "reversal", "recent losers revert", expression
    )
    compiler = FactorCompiler(SemanticValidator([FieldDefinition("close", "price")]))
    signal = compiler.evaluate(expression, {"close": close})
    forward = close.pct_change(fill_method=None).shift(-1)
    diagnostics = investment_diagnostics(
        signal.iloc[5:-1], {1: forward.iloc[5:-1]}, quantiles=5, minimum_names=30
    )
    assert diagnostics.coverage > 0.9

    factor_card = FactorRegistry(tmp_path / "factors").publish(
        definition,
        data_dependencies=("close",),
        data_lag_days=1,
        applicable_regimes=("all",),
        failure_modes=("momentum regime",),
        owner="integration-test",
        experiment_id="E1",
    )
    latest_signal = signal.iloc[-2].fillna(0.0)
    covariance = pd.DataFrame(np.eye(40) * 0.02, index=symbols, columns=symbols)
    current = pd.Series(1 / 40, index=symbols)
    optimized = PortfolioOptimizer(risk_aversion=1.0).optimize(
        latest_signal,
        covariance,
        current,
        current,
        pd.DataFrame({"beta": np.linspace(0.8, 1.2, 40)}, index=symbols),
        PortfolioConstraints(
            maximum_weight=0.05,
            maximum_active_weight=0.025,
            maximum_turnover=0.20,
            exposure_bounds={"beta": (-0.001, 0.001)},
        ),
    )
    assert optimized.success

    market_slices = pd.DataFrame(
        {"price": [10.0, 10.1], "volume": [2_000, 2_000], "can_trade": [True, True]},
        index=pd.date_range("2024-02-13 09:30", periods=2, freq="1h"),
    )
    execution = ExecutionSimulator().execute(
        Order("O1", symbols[0], "BUY", 200, 10.0, ExecutionStyle.TWAP, 0.1),
        market_slices,
        adv_shares=100_000,
        daily_volatility=0.02,
    )
    assert execution.filled_quantity == 200

    artifacts = ArtifactRegistry(tmp_path / "artifacts")
    pipeline = IdempotentPipeline(tmp_path / "pipeline")
    calls = 0

    def publish_report() -> dict[str, str]:
        nonlocal calls
        calls += 1
        report = {
            "snapshot_id": snapshot.snapshot_id,
            "factor_id": factor_card.factor_id,
            "factor_artifact_hash": factor_card.artifact_hash,
            "filled_quantity": execution.filled_quantity,
            "total_cost": execution.total_cost,
        }
        artifact = artifacts.publish(
            "research-report",
            json.dumps(report, sort_keys=True).encode(),
            owner="integration-test",
            source_ids=(snapshot.snapshot_id, factor_card.factor_id),
        )
        return {"artifact_id": artifact.artifact_id}

    inputs = {
        "snapshot": snapshot.manifest_hash,
        "factor": factor_card.artifact_hash,
        "config": "institutional_v2",
    }
    first = pipeline.run("institutional-workflow", inputs, publish_report)
    second = pipeline.run("institutional-workflow", inputs, publish_report)
    assert first.output == second.output
    assert second.cached and calls == 1
    assert artifacts.read(first.output["artifact_id"])
