"""Prefix cache + tiered router tests (all offline, no model needed)."""

import sys

sys.path.insert(0, ".")

from optimizer.prefix_cache import PrefixCache  # noqa: E402
from optimizer import router  # noqa: E402
from backends.routed import RoutedOllamaBackend  # noqa: E402


def test_prefix_cache_miss_then_hit():
    pc = PrefixCache()
    hit, tokens = pc.check("You are a helpful support agent.")
    assert hit is False and tokens > 0
    hit2, tokens2 = pc.check("You are a helpful support agent.")
    assert hit2 is True and tokens2 == tokens
    st = pc.stats()
    assert st["hits"] == 1 and st["misses"] == 1
    assert st["cached_tokens"] == tokens


def test_prefix_cache_empty_never_hits():
    pc = PrefixCache()
    assert pc.check("") == (False, 0)
    assert pc.check("   ") == (False, 0)
    assert pc.stats()["misses"] == 0  # not recorded


def test_prefix_cache_normalizes_whitespace():
    pc = PrefixCache()
    pc.check("You are  a helpful\nsupport agent.")
    hit, _ = pc.check("you are a helpful support agent.")
    assert hit is True


def test_router_simple_goes_small():
    tier, reason = router.route("What is your refund policy?")
    assert tier == router.SMALL
    assert reason


def test_router_complex_goes_large():
    tier, _ = router.route(
        "Compare the pro and enterprise plans for a 50-person team "
        "that needs SSO and audit logs, and explain why we should pick one.")
    assert tier == router.LARGE


def test_router_deterministic():
    q = "How do I reset my password?"
    assert router.route(q) == router.route(q)


class _StubBackend:
    def __init__(self, model):
        self.model = model
        self.calls = 0

    def generate(self, messages, **kwargs):
        self.calls += 1
        return "stub answer", {"prompt_tokens": 100, "completion_tokens": 10}


def test_routed_backend_picks_tier_and_logs():
    rb = RoutedOllamaBackend()
    rb.small = _StubBackend("small")
    rb.large = _StubBackend("large")
    msgs = [{"role": "system", "content": "You are support."},
            {"role": "user", "content": "What is your refund policy?"}]
    text, usage = rb.generate(msgs)
    assert text == "stub answer"
    assert usage["tier"] == router.SMALL
    assert usage["prefix_cache_hit"] is False  # first sight of prefix
    assert rb.small.calls == 1 and rb.large.calls == 0

    text2, usage2 = rb.generate(msgs)
    assert usage2["prefix_cache_hit"] is True  # prefix repeats
    assert len(rb.request_log) == 2
    assert rb.stats()["routed_small"] == 2


def test_routed_backend_complex_to_large():
    rb = RoutedOllamaBackend()
    rb.small = _StubBackend("small")
    rb.large = _StubBackend("large")
    msgs = [{"role": "user",
             "content": "Debug this error: Traceback 403 on API key rotation, "
                        "explain why it happens step by step and how to fix it"}]
    _, usage = rb.generate(msgs)
    assert usage["tier"] == router.LARGE
    assert rb.large.calls == 1
