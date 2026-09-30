"""Static-prefix cache: track repeated prompt prefixes, account the savings.

In chat workloads the same system prompt (persona, policies, instructions)
is resent with every request. A prefix-cache-capable serving stack (vLLM
prefix caching, OpenAI/Anthropic prompt caching) does not reprocess those
tokens at full price. This module hashes each request's static prefix and
records, on repeats, how many tokens become eligible for the cached input
rate instead of the full rate. Thread-safe.
"""

from __future__ import annotations

import hashlib
import re
import threading

from optimizer.compression import count_tokens


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


class PrefixCache:
    def __init__(self):
        self._lock = threading.Lock()
        self._seen: dict[str, int] = {}
        self.hits = 0
        self.misses = 0
        self.cached_tokens = 0  # tokens billed at cached rate, not full rate

    def check(self, prefix: str) -> tuple[bool, int]:
        """Check a static prefix. Returns (hit, prefix_tokens).

        Empty prefixes never hit and are not recorded.
        """
        tokens = count_tokens(_normalize(prefix))
        if not tokens:
            return False, 0
        key = hashlib.sha256(_normalize(prefix).encode()).hexdigest()
        with self._lock:
            if key in self._seen:
                self.hits += 1
                self.cached_tokens += tokens
                return True, tokens
            self._seen[key] = tokens
            self.misses += 1
            return False, tokens

    def stats(self) -> dict:
        with self._lock:
            return {
                "hits": self.hits,
                "misses": self.misses,
                "cached_tokens": self.cached_tokens,
                "unique_prefixes": len(self._seen),
            }
