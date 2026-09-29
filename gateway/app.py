"""LNM Gateway — FastAPI application.

POST /v1/chat/completions  (OpenAI-compatible; Bearer API-key auth)
  pipeline: auth -> rate limit -> cache -> input rails -> compress ->
            backend -> output rails -> faithfulness sample -> cache store -> audit
GET  /admin/metrics        (admin key)
GET  /healthz

Extension: an optional top-level "context" field may accompany a chat request;
it is used as the grounding context for sampled RAGAS faithfulness scoring.
An optional "force_faithfulness_check": true requests an on-demand (non-sampled)
faithfulness evaluation for that request (needs "context" too).
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from gateway.auth import GatewayConfig, key_hash
from gateway.rate_limit import RateLimiter
from gateway.audit import GatewayState
from rails.engine import RailsEngine, REFUSAL_INPUT, REFUSAL_OUTPUT
from optimizer.cache import TokenCache
from optimizer.compression import compress, count_tokens
from monitors.faithfulness import FaithfulnessMonitor
from backends.gemini import GeminiBackend
from backends.vllm import VLLMBackend
from backends.ollama import OllamaBackend


def build_cache():
    """Redis-backed cache when LNM_REDIS_URL is set, else in-process.

    Falls back to the in-process cache (with a warning) if Redis is
    unreachable, so a bad REDIS_URL can't take the gateway down.
    """
    url = os.environ.get("LNM_REDIS_URL", "").strip()
    if url:
        try:
            from optimizer.redis_cache import RedisTokenCache
            cache = RedisTokenCache(url)
            print(f"[lnm] cache backend: redis ({url})", flush=True)
            return cache
        except Exception as e:
            print(f"[lnm] redis unavailable ({e}); using in-process cache",
                  flush=True)
    return TokenCache()


def build_backend(name: str):
    if name == "vllm":
        return VLLMBackend()
    if name == "gemini":
        return GeminiBackend()
    return OllamaBackend()


def create_app(config: GatewayConfig | None = None) -> FastAPI:
    config = config or GatewayConfig.from_env()
    app = FastAPI(title="LNM Gateway", version="0.1.0")

    state = GatewayState(config.audit_path)
    limiter = RateLimiter(config.rate_limit_per_min)
    rails = RailsEngine()
    cache = build_cache()
    backend = build_backend(config.backend_name)

    def on_faithfulness_flag(record: dict):
        state.bump(faithfulness_flagged=1)
        state.audit.record({
            "type": "faithfulness_flag",
            "request_id": record["request_id"],
            "faithfulness": record["faithfulness"],
        })

    def on_faithfulness_scored(record: dict):
        state.bump(faithfulness_scored=1)
        state.metrics.faithfulness_last_score = record["score"]

    monitor = FaithfulnessMonitor(on_flag=on_faithfulness_flag,
                                 on_scored=on_faithfulness_scored)

    app.state.lnm = {
        "config": config, "state": state, "limiter": limiter,
        "rails": rails, "cache": cache, "backend": backend, "monitor": monitor,
    }

    def _auth(request: Request) -> str | None:
        authz = request.headers.get("authorization", "")
        if not authz.lower().startswith("bearer "):
            return None
        key = authz[7:].strip()
        return key if config.is_valid_key(key) else None

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "backend": config.backend_name}

    @app.get("/admin/metrics")
    def admin_metrics(request: Request):
        authz = request.headers.get("authorization", "")
        key = authz[7:].strip() if authz.lower().startswith("bearer ") else ""
        if not config.is_admin(key):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return state.metrics.to_dict(config.cost_per_1k_tokens_usd)

    @app.post("/v1/chat/completions")
    def chat_completions(request: Request, body: dict[str, Any]):
        t0 = time.time()
        request_id = uuid.uuid4().hex[:12]

        key = _auth(request)
        if key is None:
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        if not limiter.allow(key):
            return JSONResponse({"error": "rate limit exceeded"}, status_code=429)

        messages = body.get("messages", [])
        if not isinstance(messages, list) or not messages:
            return JSONResponse({"error": "messages required"}, status_code=400)
        grounding = body.get("context") or ""
        model = body.get("model") or config.backend_name

        user_text = next(
            (m.get("content", "") for m in reversed(messages)
             if m.get("role") == "user"), "")
        if isinstance(user_text, list):
            user_text = " ".join(p.get("text", "") for p in user_text
                                 if isinstance(p, dict))
        input_tokens = count_tokens(user_text)
        khash = key_hash(key)

        audit_entry: dict[str, Any] = {
            "type": "request", "request_id": request_id, "key_hash": khash,
            "model": model, "backend": config.backend_name,
            "input_tokens": input_tokens,
        }

        # 1. cache lookup
        hit = cache.get(user_text, model)
        if hit is not None:
            kind = getattr(cache, "last_kind", None)
            bump_kw: dict[str, Any] = {
                "requests": 1,
                "tokens_out": hit.completion_tokens,
                "tokens_saved_cache": hit.prompt_tokens,
            }
            if kind == "semantic":
                bump_kw["cache_hits_semantic"] = 1
            else:
                bump_kw["cache_hits_exact"] = 1
            state.bump(**bump_kw)
            audit_entry.update({
                "cache": "hit", "output_tokens": hit.completion_tokens,
                "latency_ms": round((time.time() - t0) * 1000, 1),
            })
            state.audit.record(audit_entry)
            return _openai_response(request_id, model, hit.response, cached=True)

        state.bump(cache_misses=1)

        # 2. input rails
        verdict = rails.check_input(user_text)
        audit_entry["input_rail"] = {"blocked": verdict.blocked,
                                    "reason": verdict.reason,
                                    "method": verdict.method}
        if verdict.blocked:
            state.bump(requests=1, blocked_input=1)
            audit_entry["latency_ms"] = round((time.time() - t0) * 1000, 1)
            state.audit.record(audit_entry)
            return JSONResponse(
                {"error": {"message": REFUSAL_INPUT, "type": "rail_block",
                           "rail": "input", "reason": verdict.reason}},
                status_code=403)

        # 3. prompt compression
        comp = compress(user_text)
        state.bump(tokens_saved_compression=comp["tokens_saved"])
        audit_entry["compression"] = {k: comp[k] for k in
                                      ("tokens_before", "tokens_after",
                                       "tokens_saved")}

        # 4. backend (from app state so tests can inject a stub)
        backend = request.app.state.lnm["backend"]
        comp_messages = [dict(m) for m in messages]
        for m in reversed(comp_messages):
            if m.get("role") == "user":
                m["content"] = comp["text"]
                break
        try:
            text, usage = backend.generate(
                comp_messages,
                max_tokens=int(body.get("max_tokens", 512)),
                temperature=float(body.get("temperature", 0.2)),
            )
        except Exception as e:
            state.bump(requests=1, errors=1)
            audit_entry.update({"error": str(e)[:200],
                                "latency_ms": round((time.time() - t0) * 1000, 1)})
            state.audit.record(audit_entry)
            return JSONResponse({"error": {"message": "backend error",
                                           "detail": str(e)[:200]}},
                                status_code=502)
        out_tokens = usage.get("completion_tokens", count_tokens(text))

        # 5. output rails
        overdict = rails.check_output(text)
        audit_entry["output_rail"] = {"blocked": overdict.blocked,
                                     "reason": overdict.reason,
                                     "method": overdict.method,
                                     "redactions": overdict.redactions}
        if overdict.blocked:
            state.bump(requests=1, blocked_output=1, tokens_in=input_tokens)
            audit_entry["latency_ms"] = round((time.time() - t0) * 1000, 1)
            state.audit.record(audit_entry)
            return JSONResponse(
                {"error": {"message": REFUSAL_OUTPUT, "type": "rail_block",
                           "rail": "output", "reason": overdict.reason}},
                status_code=403)
        final_text = overdict.text

        # 6. faithfulness: sampled by default; a client can force an
        #    on-demand evaluation with "force_faithfulness_check": true
        #    (useful for testing / demos). Needs grounding context.
        force_check = bool(body.get("force_faithfulness_check"))
        monitor.maybe_check(request_id=request_id, answer=final_text,
                            context=grounding, force=force_check)
        audit_entry["faithfulness_sampled"] = bool(grounding)
        audit_entry["faithfulness_forced"] = force_check

        # 7. cache store + metrics + audit
        cache.put(user_text, model, final_text, input_tokens, out_tokens)
        state.bump(requests=1, tokens_in=input_tokens, tokens_out=out_tokens)
        audit_entry.update({
            "cache": "miss", "output_tokens": out_tokens,
            "latency_ms": round((time.time() - t0) * 1000, 1),
        })
        state.audit.record(audit_entry)
        return _openai_response(request_id, model, final_text)

    return app


def _openai_response(request_id: str, model: str, text: str,
                     cached: bool = False) -> dict:
    return {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": count_tokens(text) if cached else None,
                  "completion_tokens": None, "total_tokens": None},
        "lnm": {"cached": cached},
    }


app = create_app()
