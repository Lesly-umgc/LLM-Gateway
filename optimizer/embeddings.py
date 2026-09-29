"""Shared lazy loader for the semantic-cache embedding model.

Both the in-process cache (optimizer/cache.py) and the Redis-backed cache
(optimizer/redis_cache.py) use all-MiniLM-L6-v2 on CPU. The model downloads
once (~90MB) and is then cached on disk.
"""

from __future__ import annotations

_model = None
_error: str | None = None


def get_embed_model():
    """Return the SentenceTransformer, or None if it cannot load (offline)."""
    global _model, _error
    if _model is None and _error is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:  # offline / download failure -> degrade
            _error = str(e)[:200]
    return _model


def embed_error() -> str | None:
    get_embed_model()
    return _error


def encode(text: str) -> list[float] | None:
    m = get_embed_model()
    if m is None:
        return None
    try:
        return m.encode(text).tolist()
    except Exception:
        return None
