"""Shared Gemini (free-tier) client used by rails actions, monitors, and the demo backend.

Auth: the raw key is never read by this process. We pass a surrogate through the
egress proxy via dynamic_credentials (connector id ``custom.google-gemini``);
the proxy swaps in the real credential. Only generativelanguage.googleapis.com
is ever contacted.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
import urllib.error

sys.path.insert(0, "/opt/hatch/skills/skill-creator/bin")
from dynamic_credentials import (  # noqa: E402
    url_with_surrogate_query_param,
    read_json_response,
    DynamicCredentialError,
)

HOST = "generativelanguage.googleapis.com"
BASE = f"https://{HOST}/v1beta"
DEFAULT_MODEL = os.environ.get("LNM_GEMINI_MODEL", "gemini-3.5-flash-lite")


def _url(model: str) -> str:
    try:
        return url_with_surrogate_query_param(
            f"{BASE}/models/{model}:generateContent",
            "custom.google-gemini",
            allowed_hosts=[HOST],
        )
    except DynamicCredentialError as e:  # pragma: no cover - env without connector
        raise RuntimeError(
            "Gemini connector 'custom.google-gemini' unavailable; "
            "LNM demo backend requires it. " + str(e)
        ) from e


def generate(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    max_output_tokens: int = 256,
    temperature: float = 0.0,
    retries: int = 4,
) -> tuple[str, dict]:
    """Generate text. Returns (text, usage_dict). Retries with backoff on 429/5xx."""
    body = json.dumps(
        {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": max_output_tokens,
                "temperature": temperature,
            },
        }
    ).encode()

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                _url(model),
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            data = read_json_response(urllib.request.urlopen(req, timeout=90))
            cand = data["candidates"][0]
            text = "".join(
                part.get("text", "") for part in cand["content"].get("parts", [])
            )
            usage = data.get("usageMetadata", {})
            return text, {
                "prompt_tokens": usage.get("promptTokenCount", 0),
                "completion_tokens": usage.get("candidatesTokenCount", 0),
            }
        except urllib.error.HTTPError as e:
            status = e.code
            try:
                detail = e.read().decode()[:300]
            except Exception:
                detail = ""
            last_err = RuntimeError(f"Gemini HTTP {status}: {detail}")
            if status in (429, 500, 502, 503) and attempt < retries - 1:
                time.sleep(2 ** attempt + 1)
                continue
            raise last_err from None
        except Exception as e:  # network etc.
            last_err = e
            if attempt < retries - 1:
                time.sleep(2 ** attempt + 1)
                continue
            raise
    raise last_err  # pragma: no cover


def classify(prompt: str, *, model: str = DEFAULT_MODEL) -> str:
    """One-word classification helper for judges. Returns raw text (caller parses)."""
    text, _ = generate(prompt, model=model, max_output_tokens=16, temperature=0.0)
    return text.strip()
