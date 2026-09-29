"""Redis-backed token cache: exact-match + semantic-similarity, TTL.

Same interface and semantics as optimizer/cache.py (TokenCache), but entries
and embeddings live in Redis so the cache is shared across gateway replicas
and survives restarts. Used when LNM_REDIS_URL is set (e.g. docker compose).

Layout:
  lnm:exact:<sha>  JSON {response, prompt_tokens, completion_tokens, ts} (TTL)
  lnm:emb:<sha>    JSON embedding vector (TTL)

Semantic search scans lnm:emb:* and scores cosine similarity in Python —
fine for demo/edge scale; for large scale use Redis Stack / RediSearch.
"""

from __future__ import annotations

import json
import math
import os
import time

from optimizer.cache import CacheEntry, _normalize  # noqa: F401  (shared key fn)
from optimizer import embeddings as _emb

_CACHE_TTL = int(os.environ.get("LNM_CACHE_TTL_SECONDS", "3600"))
_SEM_THRESHOLD = float(os.environ.get("LNM_SEM_THRESHOLD", "0.95"))
_MIN_SEM_TOKENS = 8

_EXACT_PREFIX = "lnm:exact:"
_EMB_PREFIX = "lnm:emb:"


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class RedisTokenCache:
    """Drop-in replacement for TokenCache backed by Redis."""

    def __init__(self, url: str, ttl: int = _CACHE_TTL,
                 sem_threshold: float = _SEM_THRESHOLD):
        try:
            import redis
        except ImportError as e:
            raise RuntimeError(
                "redis package not installed (pip install redis)") from e
        self._redis_mod = redis
        self.url = url
        self.ttl = ttl
        self.sem_threshold = sem_threshold
        self._r = redis.from_url(url, socket_connect_timeout=5,
                                 socket_timeout=10)
        self._r.ping()  # raise now if Redis is unreachable
        self.hits_exact = 0
        self.hits_semantic = 0
        self.misses = 0
        self.tokens_saved = 0
        self.last_kind: str | None = None

    # ---- key helpers ----
    def _key(self, prompt: str, model: str) -> str:
        import hashlib
        return hashlib.sha256(f"{model}::{_normalize(prompt)}".encode()).hexdigest()

    def _entry(self, key: str) -> CacheEntry | None:
        raw = self._r.get(_EXACT_PREFIX + key)
        if not raw:
            return None
        d = json.loads(raw)
        if time.time() - d["ts"] > self.ttl:
            self._r.delete(_EXACT_PREFIX + key, _EMB_PREFIX + key)
            return None
        return CacheEntry(d["response"], d["prompt_tokens"],
                          d["completion_tokens"], d["ts"])

    # ---- core API (mirrors TokenCache) ----
    def get(self, prompt: str, model: str) -> CacheEntry | None:
        self.last_kind = None
        key = self._key(prompt, model)
        entry = self._entry(key)
        if entry is not None:
            self.hits_exact += 1
            self.tokens_saved += entry.prompt_tokens
            self.last_kind = "exact"
            return entry

        # semantic fallback
        norm = _normalize(prompt)
        if len(norm.split()) < _MIN_SEM_TOKENS:
            self.misses += 1
            return None
        q = _emb.encode(norm)
        if q is None:  # embeddings unavailable -> degrade to exact-only
            self.misses += 1
            return None
        best, best_key = 0.0, None
        try:
            for ek in self._r.scan_iter(_EMB_PREFIX + "*", count=200):
                ek = ek.decode() if isinstance(ek, bytes) else ek
                emb_raw = self._r.get(ek)
                if not emb_raw:
                    continue
                sim = _cosine(q, json.loads(emb_raw))
                if sim > best:
                    best, best_key = sim, ek[len(_EMB_PREFIX):]
        except Exception:
            self.misses += 1
            return None
        if best_key and best >= self.sem_threshold:
            entry = self._entry(best_key)
            if entry is not None:
                self.hits_semantic += 1
                from optimizer.compression import count_tokens
                self.tokens_saved += count_tokens(prompt)
                self.last_kind = "semantic"
                return entry
        self.misses += 1
        return None

    def put(self, prompt: str, model: str, response: str,
            prompt_tokens: int, completion_tokens: int) -> None:
        key = self._key(prompt, model)
        payload = json.dumps({
            "response": response, "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens, "ts": time.time(),
        })
        self._r.setex(_EXACT_PREFIX + key, self.ttl, payload)
        vec = _emb.encode(_normalize(prompt))
        if vec is not None:
            self._r.setex(_EMB_PREFIX + key, self.ttl, json.dumps(vec))

    @property
    def semantic_available(self) -> bool:
        return _emb.embed_error() is None

    def stats(self) -> dict:
        try:
            entries = sum(1 for _ in self._r.scan_iter(_EXACT_PREFIX + "*",
                                                      count=500))
        except Exception:
            entries = -1
        total = self.hits_exact + self.hits_semantic + self.misses
        return {
            "hits_exact": self.hits_exact,
            "hits_semantic": self.hits_semantic,
            "misses": self.misses,
            "lookups": total,
            "hit_rate": (self.hits_exact + self.hits_semantic) / total if total else 0.0,
            "tokens_saved": self.tokens_saved,
            "entries": entries,
            "semantic_available": self.semantic_available,
            "backend": "redis",
        }
