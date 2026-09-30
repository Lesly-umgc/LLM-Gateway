"""Tiered Ollama backend: prefix cache + complexity routing, one interface.

Wraps two OllamaBackend tiers behind the standard Backend API so the
gateway request path needs no changes. Per request it records the static
system prefix (prefix-cache accounting) and routes by the complexity
heuristic, then delegates to the chosen tier. Per-request records and
aggregate stats are exposed for cost accounting.
"""

from __future__ import annotations

import os

from backends.base import Backend
from backends.ollama import OllamaBackend
from optimizer import prefix_cache as _pc
from optimizer import router as _router

DEFAULT_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


class RoutedOllamaBackend(Backend):
    name = "routed-ollama"

    def __init__(self, small_model: str = "llama3.2:3b",
                 large_model: str = "llama3.1:8b",
                 base_url: str | None = None,
                 num_ctx: int = 4096, timeout: int = 600):
        base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.small = OllamaBackend(base_url, small_model, num_ctx, timeout)
        self.large = OllamaBackend(base_url, large_model, num_ctx, timeout)
        self.prefixes = _pc.PrefixCache()
        self.request_log: list[dict] = []
        self.routed_small = 0
        self.routed_large = 0

    def _split(self, messages: list[dict]) -> tuple[str, str]:
        prefix = "\n".join(
            m.get("content", "") for m in messages
            if m.get("role") == "system" and m.get("content"))
        user_text = next(
            (m.get("content", "") for m in reversed(messages)
             if m.get("role") == "user"), "")
        if isinstance(user_text, list):  # content blocks
            user_text = " ".join(p.get("text", "") for p in user_text
                                 if isinstance(p, dict))
        return prefix, user_text or ""

    def generate(self, messages: list[dict], **kwargs) -> tuple[str, dict]:
        prefix, user_text = self._split(messages)
        prefix_hit, prefix_tokens = self.prefixes.check(prefix)
        tier, reason = _router.route(user_text)
        backend = self.small if tier == _router.SMALL else self.large
        if tier == _router.SMALL:
            self.routed_small += 1
        else:
            self.routed_large += 1
        text, usage = backend.generate(messages, **kwargs)
        usage.update({
            "tier": tier,
            "route_reason": reason,
            "model": backend.model,
            "prefix_cache_hit": prefix_hit,
            "prefix_tokens": prefix_tokens,
        })
        self.request_log.append({
            "tier": tier,
            "reason": reason,
            "prefix_hit": prefix_hit,
            "prefix_tokens": prefix_tokens,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        })
        return text, usage

    def stats(self) -> dict:
        return {
            "routed_small": self.routed_small,
            "routed_large": self.routed_large,
            "model_calls": len(self.request_log),
            "prefix": self.prefixes.stats(),
        }

    def check_connection(self) -> dict:
        s = self.small.check_connection()
        l = self.large.check_connection()
        return {"ok": bool(s.get("ok")) and bool(l.get("ok")),
                "detail": f"small={s.get('detail')} large={l.get('detail')}"}
