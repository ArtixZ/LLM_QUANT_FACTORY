from __future__ import annotations

from autoalpha.ibkr.pacing import HistoricalPacer


class FakeClock:
    """Deterministic monotonic clock whose sleep advances time."""

    def __init__(self) -> None:
        self.now = 1_000.0
        self.slept: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def make_pacer(clock: FakeClock, **kwargs: float) -> HistoricalPacer:
    return HistoricalPacer(clock=clock.time, sleep=clock.sleep, **kwargs)


def test_first_request_does_not_wait() -> None:
    clock = FakeClock()
    pacer = make_pacer(clock)
    assert pacer.acquire("a") == 0.0
    assert clock.slept == []


def test_minimum_interval_separates_consecutive_requests() -> None:
    clock = FakeClock()
    pacer = make_pacer(clock, minimum_interval_seconds=0.5, identical_cooldown_seconds=0.0)
    pacer.acquire("a")
    waited = pacer.acquire("b")
    assert waited == 0.5


def test_identical_request_waits_out_the_cooldown() -> None:
    clock = FakeClock()
    pacer = make_pacer(clock, minimum_interval_seconds=0.0, identical_cooldown_seconds=15.0)
    pacer.acquire("same")
    waited = pacer.acquire("same")
    assert waited == 15.0


def test_window_limit_blocks_the_sixty_first_request() -> None:
    clock = FakeClock()
    pacer = make_pacer(
        clock,
        request_limit=3,
        window_seconds=600.0,
        minimum_interval_seconds=0.0,
        identical_cooldown_seconds=0.0,
    )
    for index in range(3):
        pacer.acquire(f"k{index}")
    waited = pacer.acquire("k4")
    assert waited == 600.0


def test_expired_requests_leave_the_window() -> None:
    clock = FakeClock()
    pacer = make_pacer(
        clock,
        request_limit=2,
        window_seconds=100.0,
        minimum_interval_seconds=0.0,
        identical_cooldown_seconds=0.0,
    )
    pacer.acquire("a")
    pacer.acquire("b")
    clock.now += 101.0
    assert pacer.acquire("c") == 0.0
