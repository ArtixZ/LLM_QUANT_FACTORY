from autoalpha.research.ranking import CandidateValue, pareto_rank


def _candidate(
    candidate_id: str,
    *,
    net_ir: float,
    annual_return: float,
    drawdown: float,
    capacity: float,
    turnover: float,
    stress_loss: float,
    complexity: int,
) -> CandidateValue:
    return CandidateValue(
        candidate_id,
        net_ir,
        annual_return,
        drawdown,
        capacity,
        turnover,
        stress_loss,
        complexity,
    )


def test_pareto_ranking_does_not_collapse_tradeoffs_into_one_score() -> None:
    return_factor = _candidate(
        "return",
        net_ir=0.5,
        annual_return=0.03,
        drawdown=0.0,
        capacity=50e6,
        turnover=20,
        stress_loss=-0.05,
        complexity=8,
    )
    defensive_factor = _candidate(
        "defensive",
        net_ir=0.3,
        annual_return=0.01,
        drawdown=0.04,
        capacity=200e6,
        turnover=5,
        stress_loss=-0.01,
        complexity=3,
    )
    dominated = _candidate(
        "dominated",
        net_ir=0.2,
        annual_return=0.005,
        drawdown=-0.01,
        capacity=20e6,
        turnover=30,
        stress_loss=-0.08,
        complexity=10,
    )

    ranked = pareto_rank([return_factor, defensive_factor, dominated])
    fronts = {item.candidate.candidate_id: item.pareto_front for item in ranked}

    assert fronts["return"] == 1
    assert fronts["defensive"] == 1
    assert fronts["dominated"] == 2
