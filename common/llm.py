"""Unified LLM helper for rails judges and the faithfulness monitor.

Routing (env LNM_JUDGE_BACKEND, default "ollama"):
  - "ollama": local Ollama model — no external calls, no extra credentials.
  - "gemini": free-tier Gemini via the GEMINI_API_KEY env var.
  - "auto"  : try Ollama, fall back to Gemini if unreachable.

See common/gemini.py for auth details.
"""

from __future__ import annotations

import os

def _backend() -> str:
    return os.environ.get("LNM_JUDGE_BACKEND", "ollama").lower()


_ollama = None


def _ollama_client():
    global _ollama
    if _ollama is None:
        from backends.ollama import OllamaBackend
        _ollama = OllamaBackend()
    return _ollama


def classify(prompt: str) -> str:
    """One-word classification. Returns raw text (caller parses)."""
    if _backend() in ("ollama", "auto"):
        try:
            return _ollama_client().classify(prompt)
        except Exception:
            if _backend() == "ollama":
                raise
    from common import gemini
    return gemini.classify(prompt)


def generate(prompt: str, *, max_output_tokens: int = 512,
             temperature: float = 0.0) -> tuple[str, dict]:
    """Free-form generation. Returns (text, usage)."""
    if _backend() in ("ollama", "auto"):
        try:
            client = _ollama_client()
            data = client._post("/api/generate", {
                "model": client.model,
                "prompt": prompt, "stream": False,
                "options": {"temperature": temperature,
                            "num_predict": max_output_tokens},
            })
            text = (data.get("response") or "").strip()
            return text, {"prompt_tokens": 0, "completion_tokens": 0}
        except Exception:
            if _backend() == "ollama":
                raise
    from common import gemini
    return gemini.generate(prompt, max_output_tokens=max_output_tokens,
                           temperature=temperature)


def judge_backend_name() -> str:
    return _backend()
