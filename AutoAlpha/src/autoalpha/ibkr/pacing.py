from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

# Interactive Brokers rejects more than 60 historical-data requests in any rolling
# ten-minute window, and repeats an identical request only after a short cooldown.
HISTORICAL_REQUEST_LIMIT = 60
HISTORICAL_WINDOW_SECONDS = 600.0
IDENTICAL_REQUEST_COOLDOWN_SECONDS = 15.0


@dataclass
class HistoricalPacer:
    """Block until another historical-data request fits inside IBKR's pacing rules."""

    request_limit: int = HISTORICAL_REQUEST_LIMIT
    window_seconds: float = HISTORICAL_WINDOW_SECONDS
    identical_cooldown_seconds: float = IDENTICAL_REQUEST_COOLDOWN_SECONDS
    minimum_interval_seconds: float = 0.34
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    _timestamps: deque[float] = field(default_factory=deque, init=False, repr=False)
    _last_seen: dict[str, float] = field(default_factory=dict, init=False, repr=False)

    def acquire(self, key: str = "") -> float:
        """Wait as long as the pacing rules require, then record the request.

        Returns the number of seconds spent waiting, which callers surface in
        download telemetry so slow syncs are explainable.
        """
        waited = 0.0
        while True:
            now = self.clock()
            self._evict(now)
            delay = self._required_delay(now, key)
            if delay <= 0:
                break
            self.sleep(delay)
            waited += delay
        stamp = self.clock()
        self._timestamps.append(stamp)
        if key:
            self._last_seen[key] = stamp
        return waited

    def _required_delay(self, now: float, key: str) -> float:
        delays = [0.0]
        if len(self._timestamps) >= self.request_limit:
            delays.append(self._timestamps[0] + self.window_seconds - now)
        if self._timestamps:
            delays.append(self._timestamps[-1] + self.minimum_interval_seconds - now)
        if key and key in self._last_seen:
            delays.append(self._last_seen[key] + self.identical_cooldown_seconds - now)
        return max(delays)

    def _evict(self, now: float) -> None:
        horizon = now - self.window_seconds
        while self._timestamps and self._timestamps[0] <= horizon:
            self._timestamps.popleft()
        for key, stamp in list(self._last_seen.items()):
            if stamp <= now - self.identical_cooldown_seconds:
                del self._last_seen[key]
