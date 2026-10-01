"""Shared Gemini client used by rails actions, monitors, and the demo backend.

Auth: plain Google AI Studio key from the GEMINI_API_KEY environment
variable (https://aistudio.google.com/apikey, free tier). Nothing
Hatch-specific here; the module works anywhere the key is set.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
import urllib.error

HOST = "generativelanguage.googleapis.com"
BASE = f"https://{HOST}/v1beta"
DEFAULT_MODEL = os.environ.get("LNM_GEMINI_MODEL", "gemini-3.5-flash-lite")


def _key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Get a free key at "
            "https://aistudio.google.com/apikey and export GEMINI_API_KEY."
        )
    return key


def generate(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    max_output_tokens: int = 256,
    temperature: float = 0.0,
    retries: int = 4,
) -> tuple[str, dict]:
    """Generate text. Returns (text, usage_dict). Retries with backoff on 429/5xx."""
    url = f"{BASE}/models/{model}:generateContent"
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
                url,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": _key(),
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = json.loads(resp.read().decode())
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
