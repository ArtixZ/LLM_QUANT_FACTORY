from __future__ import annotations

from pathlib import Path

from autoalpha.config import ResearchConfig
from autoalpha.service.direction import assess_direction_outcome, diagnose_direction
from autoalpha.service.store import ServiceStore


def _config() -> ResearchConfig:
    return ResearchConfig.from_toml(Path("config/research.toml"))


def test_absolute_stability_gap_determines_next_public_direction() -> None:
    config = _config()
    incumbent = {
        "portfolio_annual_return_dispersion": 0.22,
        "portfolio_walk_forward_worst_sharpe": 0.3,
        "portfolio_coverage": 0.95,
        "portfolio_capacity_cny": 100_000_000.0,
        "portfolio_annual_turnover": 12.0,
        "portfolio_cost_stress_net_ir": 1.0,
        "portfolio_max_drawdown": -0.08,
        "portfolio_max_factor_correlation": 0.25,
    }
    recent = [{"portfolio_action_gate_failures": ["annual_dispersion"]} for _ in range(6)]

    plan = diagnose_direction(incumbent, recent, blocked_directions=set(), config=config)

    assert plan.definition.direction == "RESTORE_STABILITY"
    assert plan.evidence["recent_candidates_used"] == 6


def test_campaign_cooldown_forces_a_different_direction() -> None:
    config = _config()
    plan = diagnose_direction(
        {"portfolio_annual_return_dispersion": 0.22},
        [{"portfolio_action_gate_failures": ["annual_dispersion", "portfolio_value"]}],
        blocked_directions={"RESTORE_STABILITY"},
        config=config,
    )

    assert plan.definition.direction != "RESTORE_STABILITY"
    assert "RESTORE_STABILITY" in plan.evidence["blocked_by_cooldown"]


def test_direction_progress_requires_an_accepted_public_action() -> None:
    config = _config()
    baseline = {
        "portfolio_annual_return_dispersion": 0.22,
        "portfolio_walk_forward_worst_sharpe": 0.1,
    }
    proposed = {
        "portfolio_annual_return_dispersion": 0.20,
        "portfolio_walk_forward_worst_sharpe": 0.12,
        "portfolio_proposed_absolute_failures": ["annual_dispersion"],
    }

    rejected = assess_direction_outcome(
        "RESTORE_STABILITY",
        baseline,
        proposed,
        accepted=False,
        candidate_eligible=True,
        config=config,
    )
    accepted = assess_direction_outcome(
        "RESTORE_STABILITY",
        baseline,
        proposed,
        accepted=True,
        candidate_eligible=True,
        config=config,
    )

    assert rejected["direction_improved"] is False
    assert accepted["direction_improved"] is True
    assert accepted["objective_resolved"] is False


def test_direction_campaign_stops_after_two_consecutive_misses(tmp_path: Path) -> None:
    store = ServiceStore(tmp_path / "service.sqlite3")
    store.ensure_generation(
        generation_id="g1",
        protocol_version="v5",
        maximum_candidates=20,
        maximum_holdout_attempts=2,
    )
    campaign = store.start_direction_campaign(
        generation_id="g1",
        direction="RESTORE_STABILITY",
        title="stability",
        objective="reduce dispersion",
        diagnostic_score=100.0,
        rationale=["dispersion failed"],
        evidence={"public_only": True},
        baseline={"portfolio_annual_return_dispersion": 0.22},
        maximum_attempts=3,
        started_iteration=1,
    )
    for iteration in (1, 2):
        store.reserve_direction_attempt(
            campaign_id=campaign["id"], iteration=iteration, baseline={"value": iteration}
        )
        campaign = store.complete_direction_attempt(
            iteration=iteration,
            candidate_id=f"factor-{iteration}",
            outcome="PUBLIC_DIRECTION_MISSED",
            improved=False,
            objective_resolved=False,
            diagnostics={"public_only": True},
            early_stop_consecutive_misses=2,
        )

    assert campaign["status"] == "EARLY_STOPPED"
    assert campaign["attempts_used"] == 2
    assert campaign["closure_reason"] == "CONSECUTIVE_DIRECTION_MISSES"
    assert store.active_direction_campaign("g1") is None


def test_direction_campaign_never_exceeds_three_attempts(tmp_path: Path) -> None:
    store = ServiceStore(tmp_path / "service.sqlite3")
    store.ensure_generation(
        generation_id="g1",
        protocol_version="v5",
        maximum_candidates=20,
        maximum_holdout_attempts=2,
    )
    campaign = store.start_direction_campaign(
        generation_id="g1",
        direction="EXPLORE_NEW_MECHANISM",
        title="novelty",
        objective="explore",
        diagnostic_score=1.0,
        rationale=["fallback"],
        evidence={},
        baseline={},
        maximum_attempts=3,
        started_iteration=1,
    )
    for iteration in (1, 2, 3):
        store.reserve_direction_attempt(
            campaign_id=campaign["id"], iteration=iteration, baseline={}
        )
        campaign = store.complete_direction_attempt(
            iteration=iteration,
            candidate_id=f"factor-{iteration}",
            outcome="PUBLIC_DIRECTION_MISSED",
            improved=False,
            objective_resolved=False,
            diagnostics={},
            early_stop_consecutive_misses=3,
        )

    assert campaign["status"] == "EXHAUSTED"
    assert campaign["attempts_used"] == 3
