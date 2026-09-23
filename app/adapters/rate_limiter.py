from __future__ import annotations

import time


class RateLimiter:
    """Blocks until at least min_interval_seconds has passed since the
    last call.

    arXiv's usage policy asks for no more than one request every 3 seconds.
    This is intentionally dead simple (single-process, blocking) since
    ingestion runs as a one-off CLI process, not a concurrent service.
    """

    def __init__(
            self, 
            min_interval_seconds: float, 
            sleep_fn= time.sleep, 
            clock=time.monotonic
        ):
        self._min_interval = min_interval_seconds
        self._sleep = sleep_fn
        self._clock = clock
        self._last_call: float | None = None

    def wait(self) -> None:
        now = self._clock()
        if self._last_call is not None:
            elapsed = now - self._last_call
            remaining = self._min_interval - elapsed
            if remaining > 0:
                self._sleep(remaining)
        self._last_call = self._clock()
