"""RedisTokenCache tests — skipped when no Redis is reachable."""

import os
import sys

sys.path.insert(0, ".")

import pytest

redis = pytest.importorskip("redis")

from optimizer.redis_cache import RedisTokenCache  # noqa: E402

URL = os.environ.get("LNM_TEST_REDIS_URL", "redis://localhost:6379/15")


@pytest.fixture()
def cache():
    c = RedisTokenCache(URL, namespace="lnm:test")
    try:
        c._r.ping()
    except Exception:
        pytest.skip("no redis at %s" % URL)
    c._r.flushdb()
    yield c
    c._r.flushdb()


def test_put_get_roundtrip(cache):
    cache.put("hello world", "m", "hi there", 5, 3)
    e = cache.get("hello world", "m")
    assert e is not None and e.response == "hi there"
    assert cache.last_kind == "exact"
    assert cache.stats()["hits_exact"] == 1


def test_miss_and_model_isolation(cache):
    assert cache.get("nope", "m") is None
    cache.put("q", "m1", "a1", 4, 2)
    assert cache.get("q", "m2") is None  # different model, no hit


def test_ttl_expiry(cache):
    c = RedisTokenCache(URL, ttl=1, namespace="lnm:test")
    c._r.flushdb()
    c.put("quick", "m", "a", 2, 1)
    assert c.get("quick", "m") is not None
    import time
    time.sleep(1.2)
    assert c.get("quick", "m") is None
    c._r.flushdb()


def test_semantic_hit(cache):
    pytest.importorskip("sentence_transformers")
    cache.put("What is the capital of France?", "m", "Paris", 8, 2)
    e = cache.get("What is France's capital city?", "m")
    assert e is not None and e.response == "Paris"
    assert cache.last_kind == "semantic"
