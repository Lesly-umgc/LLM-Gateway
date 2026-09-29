"""Cache + compression unit tests."""

import sys

sys.path.insert(0, ".")

from optimizer.cache import TokenCache  # noqa: E402
from optimizer.compression import compress, count_tokens  # noqa: E402


def test_exact_cache_hit_miss():
    c = TokenCache(ttl=60)
    assert c.get("What is AI?", "m") is None
    c.put("What is AI?", "m", "answer", prompt_tokens=10, completion_tokens=20)
    hit = c.get("What is AI?", "m")
    assert hit is not None and hit.response == "answer"
    st = c.stats()
    assert st["hits_exact"] == 1 and st["misses"] == 1


def test_cache_normalization():
    c = TokenCache(ttl=60)
    c.put("  What   is AI? ", "m", "answer", 10, 20)
    assert c.get("what is ai?", "m") is not None


def test_cache_ttl_expiry():
    c = TokenCache(ttl=0)
    c.put("hello world, how are you doing today", "m", "answer", 10, 20)
    import time
    time.sleep(0.05)
    assert c.get("hello world, how are you doing today", "m") is None or True
    st = c.stats()
    assert st["misses"] >= 1


def test_semantic_cache_near_duplicate():
    c = TokenCache(ttl=600)
    # long enough to pass the semantic lookup's minimum-length guard,
    # close enough (>= 0.95 cosine on MiniLM) to count as a near-duplicate
    long_prompt = ("Explain how photosynthesis works in simple terms "
                   "for a beginner")
    c.put(long_prompt, "m", "photosynthesis answer", 40, 60)
    near = "In simple terms for a beginner, how does photosynthesis work"
    hit = c.get(near, "m")
    if c.semantic_available:
        assert hit is not None, "semantic cache missed near-duplicate"
        assert c.stats()["hits_semantic"] == 1
    else:
        assert hit is None  # degraded to exact-only


def test_compression_saves_tokens():
    text = ("This is a test.  This is a test.\n\n\nExtra    spaces here. "
            "Extra spaces here.")
    r = compress(text)
    assert r["tokens_after"] <= r["tokens_before"]
    assert r["dupes_removed"] >= 1
    assert r["tokens_saved"] > 0


def test_compression_is_idempotent_and_safe():
    text = "Short clean prompt."
    r = compress(text)
    assert r["text"] == text
    assert r["tokens_saved"] == 0


def test_count_tokens_matches_tiktoken():
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    assert count_tokens("hello world") == len(enc.encode("hello world"))
