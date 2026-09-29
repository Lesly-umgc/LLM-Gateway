"""Structured audit logging (JSONL) + in-memory metrics counters."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field


@dataclass
class Metrics:
    requests: int = 0
    blocked_input: int = 0
    blocked_output: int = 0
    cache_hits_exact: int = 0
    cache_hits_semantic: int = 0
    cache_misses: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_saved_cache: int = 0
    tokens_saved_compression: int = 0
    faithfulness_scored: int = 0
    faithfulness_flagged: int = 0
    faithfulness_last_score: float | None = None
    errors: int = 0

    def to_dict(self, cost_per_1k: float) -> dict:
        saved = self.tokens_saved_cache + self.tokens_saved_compression
        return {
            "requests": self.requests,
            "blocked": {"input": self.blocked_input, "output": self.blocked_output,
                        "total": self.blocked_input + self.blocked_output},
            "cache": {
                "hits_exact": self.cache_hits_exact,
                "hits_semantic": self.cache_hits_semantic,
                "misses": self.cache_misses,
                "hit_rate": (self.cache_hits_exact + self.cache_hits_semantic)
                            / max(1, self.cache_hits_exact + self.cache_hits_semantic
                                  + self.cache_misses),
            },
            "tokens": {
                "in": self.tokens_in, "out": self.tokens_out,
                "saved_cache": self.tokens_saved_cache,
                "saved_compression": self.tokens_saved_compression,
                "saved_total": saved,
            },
            "est_cost_saved_usd": round(saved / 1000.0 * cost_per_1k, 6),
            "faithfulness": {"scored": self.faithfulness_scored,
                            "flagged": self.faithfulness_flagged,
                            "last_score": self.faithfulness_last_score},
            "errors": self.errors,
        }


class AuditLog:
    """Append-only JSONL audit log of every request with rail verdicts."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()

    def record(self, entry: dict) -> None:
        entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **entry}
        line = json.dumps(entry, default=str)
        with self._lock:
            with open(self.path, "a") as f:
                f.write(line + "\n")


class GatewayState:
    def __init__(self, audit_path: str):
        self.metrics = Metrics()
        self.audit = AuditLog(audit_path)
        self._lock = threading.Lock()

    def bump(self, **kwargs) -> None:
        with self._lock:
            for k, v in kwargs.items():
                setattr(self.metrics, k, getattr(self.metrics, k) + v)
