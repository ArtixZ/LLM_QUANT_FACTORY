import json

import pytest

from autoalpha.operations.artifacts import ArtifactRegistry
from autoalpha.operations.monitoring import PaperTradingBook, ProductionMonitor
from autoalpha.operations.pipeline import IdempotentPipeline
from autoalpha.operations.release import ReleaseRegistry


def test_artifact_registry_is_content_addressed_and_detects_damage(tmp_path) -> None:
    registry = ArtifactRegistry(tmp_path / "artifacts")
    first = registry.publish("factor", b"payload", owner="quant", source_ids=("data-1",))
    second = registry.publish("factor", b"payload", owner="quant")
    assert first.artifact_id == second.artifact_id
    assert registry.read(first.artifact_id) == b"payload"
    payload_path = tmp_path / "artifacts" / registry.get(first.artifact_id).payload_path
    payload_path.write_bytes(b"damaged")
    with pytest.raises(RuntimeError, match="integrity"):
        registry.get(first.artifact_id)


def test_pipeline_is_idempotent_and_does_not_publish_failed_results(tmp_path) -> None:
    pipeline = IdempotentPipeline(tmp_path / "runs")
    calls = 0

    def execute() -> dict[str, int]:
        nonlocal calls
        calls += 1
        return {"value": 42}

    first = pipeline.run("factor", {"snapshot": "S1"}, execute)
    second = pipeline.run("factor", {"snapshot": "S1"}, execute)
    assert not first.cached and second.cached and calls == 1

    def fail() -> dict[str, int]:
        raise RuntimeError("failure")

    with pytest.raises(RuntimeError, match="failure"):
        pipeline.run("portfolio", {"date": "2024-01-01"}, fail)
    assert not list((tmp_path / "runs" / "portfolio").glob("*.json"))


def test_monitoring_and_paper_book_explain_research_live_gap(tmp_path) -> None:
    alerts = ProductionMonitor().evaluate(
        missing_fraction=0.10,
        rolling_ic=-0.02,
        exposure_zscore=4.0,
        shortfall_bps=25.0,
        pnl_deviation_zscore=4.0,
    )
    assert {alert.action for alert in alerts} == {"SUSPEND", "DEWEIGHT", "REVIEW", "ROLLBACK"}
    book = PaperTradingBook(tmp_path / "paper.jsonl")
    book.append(
        date="2024-01-02",
        research_return=0.01,
        paper_return=0.008,
        turnover=0.2,
        shortfall_bps=5,
    )
    book.append(
        date="2024-01-03",
        research_return=-0.005,
        paper_return=-0.006,
        turnover=0.1,
        shortfall_bps=7,
    )
    summary = book.summary()
    assert summary["days"] == 2
    assert summary["direction_agreement"] == 1


def test_release_requires_approval_and_supports_rollback(tmp_path) -> None:
    releases = ReleaseRegistry(tmp_path / "releases.jsonl")
    with pytest.raises(PermissionError):
        releases.promote("S1", "A1", approver="", approval_id="", allocation_fraction=0.1)
    releases.promote("S1", "A1", approver="risk", approval_id="P1", allocation_fraction=0.1)
    releases.promote("S1", "A2", approver="risk", approval_id="P2", allocation_fraction=0.25)
    rollback = releases.rollback("S1", approver="risk", approval_id="P3")
    assert rollback.artifact_id == "A1"
    releases.audit.verify()


def test_pipeline_task_key_is_stable_under_input_order(tmp_path) -> None:
    pipeline = IdempotentPipeline(tmp_path / "runs")
    first = pipeline.run("x", {"a": 1, "b": 2}, lambda: {"ok": True})
    second = pipeline.run("x", json.loads('{"b": 2, "a": 1}'), lambda: {"ok": False})
    assert first.task_key == second.task_key
    assert second.output == {"ok": True}
