"""Per-API-key token-bucket rate limiter (in-memory, thread-safe)."""

from __future__ import annotations

import threading
import time


class RateLimiter:
    # TODO: in-memory only — a multi-worker / multi-replica gateway would need
    # shared buckets (e.g. Redis) so limits actually hold across processes.
    def __init__(self, per_minute: int = 60):
        self.per_minute = per_minute
        self._lock = threading.Lock()
        self._buckets: dict[str, tuple[float, float]] = {}  # key -> (tokens, last_ts)

    def allow(self, key: str) -> bool:
        now = time.time()
        with self._lock:
            tokens, last = self._buckets.get(key, (float(self.per_minute), now))
            tokens = min(float(self.per_minute),
                         tokens + (now - last) * (self.per_minute / 60.0))
            if tokens >= 1.0:
                self._buckets[key] = (tokens - 1.0, now)
                return True
            self._buckets[key] = (tokens, now)
            return False

    def reset(self, key: str | None = None) -> None:
        with self._lock:
            if key is None:
                self._buckets.clear()
            else:
                self._buckets.pop(key, None)
