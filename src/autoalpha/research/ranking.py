from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateValue:
    candidate_id: str
    incremental_net_ir: float
    incremental_annual_return: float
    drawdown_improvement: float
    capacity_cny: float
    annual_turnover: float
    stress_loss: float
    complexity: int

    @property
    def objectives(self) -> tuple[float, ...]:
        return (
            self.incremental_net_ir,
            self.incremental_annual_return,
            self.drawdown_improvement,
            self.capacity_cny,
            -self.annual_turnover,
            self.stress_loss,
            -float(self.complexity),
        )


@dataclass(frozen=True)
class RankedCandidate:
    candidate: CandidateValue
    pareto_front: int


def pareto_rank(
    candidates: list[CandidateValue] | tuple[CandidateValue, ...],
) -> tuple[RankedCandidate, ...]:
    """Rank gate-passing candidates without collapsing unlike objectives into one score."""
    remaining = list(candidates)
    ranked: list[RankedCandidate] = []
    front_number = 1
    while remaining:
        front = [
            candidate
            for candidate in remaining
            if not any(
                _dominates(other.objectives, candidate.objectives)
                for other in remaining
                if other is not candidate
            )
        ]
        front.sort(
            key=lambda candidate: (
                candidate.drawdown_improvement,
                candidate.incremental_net_ir,
                candidate.capacity_cny,
                -candidate.complexity,
                candidate.candidate_id,
            ),
            reverse=True,
        )
        ranked.extend(RankedCandidate(candidate, front_number) for candidate in front)
        front_ids = {id(candidate) for candidate in front}
        remaining = [candidate for candidate in remaining if id(candidate) not in front_ids]
        front_number += 1
    return tuple(ranked)


def _dominates(left: tuple[float, ...], right: tuple[float, ...]) -> bool:
    return all(a >= b for a, b in zip(left, right, strict=True)) and any(
        a > b for a, b in zip(left, right, strict=True)
    )
