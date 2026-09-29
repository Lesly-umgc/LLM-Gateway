"""Token-optimizer cache: exact-match + semantic-similarity, TTL, thread-safe.

Exact key: sha256(normalized prompt + model). Semantic: all-MiniLM-L6-v2
embeddings (CPU OK), cosine similarity >= threshold, entry TTL.
Semantic lookups are skipped for very short prompts (< 8 tokens) to avoid
false positives. If the embedding model cannot load (offline), the cache
degrades to exact-match only and reports the degradation.
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
import time
from dataclasses import dataclass

_CACHE_TTL = int(os.environ.get("LNM_CACHE_TTL_SECONDS", "3600"))
_SEM_THRESHOLD = float(os.environ.get("LNM_SEM_THRESHOLD", "0.95"))
_MIN_SEM_TOKENS = 8


def _normalize(text: str) -> str:
    t = (text or "").strip().lower()
    return re.sub(r"\s+", " ", t)


@dataclass
class CacheEntry:
    response: str
    prompt_tokens: int      # input tokens that a hit avoids re-sending
    completion_tokens: int
    ts: float


class TokenCache:
    def __init__(self, ttl: int = _CACHE_TTL, sem_threshold: float = _SEM_THRESHOLD):
        self.ttl = ttl
        self.sem_threshold = sem_threshold
        self._lock = threading.Lock()
        self._store: dict[str, CacheEntry] = {}
        self._embeddings: dict[str, list[float]] = {}
        self.hits_exact = 0
        self.hits_semantic = 0
        self.misses = 0
        self.tokens_saved = 0
        self.last_kind: str | None = None  # "exact" | "semantic" | None

    @property
    def semantic_available(self) -> bool:
        from optimizer import embeddings as _emb
        return _emb.embed_error() is None

    # ---- core API ----
    def _key(self, prompt: str, model: str) -> str:
        return hashlib.sha256(f"{model}::{_normalize(prompt)}".encode()).hexdigest()

    def get(self, prompt: str, model: str) -> CacheEntry | None:
        self.last_kind = None
        key = self._key(prompt, model)
        now = time.time()
        with self._lock:
            entry = self._store.get(key)
            if entry and now - entry.ts <= self.ttl:
                self.hits_exact += 1
                self.tokens_saved += entry.prompt_tokens
                self.last_kind = "exact"
                return entry
            if entry:  # expired
                del self._store[key]
                self._embeddings.pop(key, None)

        # semantic fallback
        from optimizer import embeddings as _emb
        model_emb = _emb.get_embed_model()
        if model_emb is None:
            with self._lock:
                self.misses += 1
            return None
        norm = _normalize(prompt)
        if len(norm.split()) < _MIN_SEM_TOKENS:
            with self._lock:
                self.misses += 1
            return None
        try:
            import torch
            import torch.nn.functional as F

            q = torch.tensor(model_emb.encode(norm, convert_to_tensor=True))
            best, best_key = 0.0, None
            with self._lock:
                items = list(self._store.items())
            for k, e in items:
                if now - e.ts > self.ttl:
                    continue
                emb = self._embeddings.get(k)
                if emb is None:
                    continue
                sim = float(F.cosine_similarity(
                    q, torch.tensor(emb), dim=0))
                if sim > best:
                    best, best_key = sim, k
            if best_key and best >= self.sem_threshold:
                with self._lock:
                    entry = self._store.get(best_key)
                    if entry:
                        self.hits_semantic += 1
                        self.last_kind = "semantic"
                        # honest accounting: tokens we did NOT have to send this time
                        from optimizer.compression import count_tokens
                        saved = count_tokens(prompt)
                        self.tokens_saved += saved
                        return entry
        except Exception:
            pass
        with self._lock:
            self.misses += 1
        return None

    def put(self, prompt: str, model: str, response: str,
            prompt_tokens: int, completion_tokens: int) -> None:
        key = self._key(prompt, model)
        entry = CacheEntry(response, prompt_tokens, completion_tokens, time.time())
        with self._lock:
            self._store[key] = entry
        from optimizer import embeddings as _emb
        vec = _emb.encode(_normalize(prompt))
        if vec is not None:
            with self._lock:
                self._embeddings[key] = vec

    def stats(self) -> dict:
        with self._lock:
            total = self.hits_exact + self.hits_semantic + self.misses
            return {
                "hits_exact": self.hits_exact,
                "hits_semantic": self.hits_semantic,
                "misses": self.misses,
                "lookups": total,
                "hit_rate": (self.hits_exact + self.hits_semantic) / total if total else 0.0,
                "tokens_saved": self.tokens_saved,
                "entries": len(self._store),
                "semantic_available": self.semantic_available,
            }
