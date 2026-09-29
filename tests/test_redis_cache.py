"""RedisTokenCache tests.

Uses a real Redis when one is reachable (LNM_TEST_REDIS_URL), otherwise a
fakeredis in-memory fake. Either way the cache logic gets exercised.
"""

import os
import sys

sys.path.insert(0, ".")

import pytest

redis = pytest.importorskip("redis")

from optimizer.redis_cache import RedisTokenCache  # noqa: E402

URL = os.environ.get("LNM_TEST_REDIS_URL", "redis://localhost:6379/15")


def _client():
    try:
        c = redis.from_url(URL, socket_connect_timeout=1)
        c.ping()
        return c
    except Exception:
        fake = pytest.importorskip("fakeredis")
        return fake.FakeRedis()


@pytest.fixture()
def cache(monkeypatch):
    client = _client()
    client.flushdb()

    real_from_url = redis.from_url

    def fake_from_url(url, **kw):
        return client

    monkeypatch.setattr(redis, "from_url", fake_from_url)
    c = RedisTokenCache(URL, namespace="lnm:test")
    yield c
    client.flushdb()


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
    c.put("quick", "m", "a", 2, 1)
    assert c.get("quick", "m") is not None
    import time
    time.sleep(1.2)
    assert c.get("quick", "m") is None


def test_semantic_hit(cache):
    pytest.importorskip("sentence_transformers")
    # long enough for the semantic lookup's minimum-length guard, and a
    # genuine >= 0.95 paraphrase on MiniLM
    cache.put("Explain how photosynthesis works in simple terms for a beginner",
              "m", "Photosynthesis converts light to chemical energy.", 14, 8)
    e = cache.get(
        "In simple terms for a beginner, how does photosynthesis work", "m")
    assert e is not None
    assert e.response.startswith("Photosynthesis")
    assert cache.last_kind == "semantic"
