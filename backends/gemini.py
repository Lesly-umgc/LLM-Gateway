"""Gemini backend — free-tier Google AI Studio backend.

Auth: GEMINI_API_KEY env var (see common/gemini.py).
"""

from __future__ import annotations

import os

from backends.base import Backend
from common import gemini

MODEL = os.environ.get("LNM_GEMINI_MODEL", "gemini-3.5-flash-lite")


def _messages_to_prompt(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        role = (m.get("role") or "user").upper()
        content = m.get("content") or ""
        if isinstance(content, list):  # multimodal parts -> text only
            content = " ".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            )
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


class GeminiBackend(Backend):
    name = "gemini"

    def __init__(self, model: str = MODEL):
        self.model = model

    def generate(self, messages: list[dict], **kwargs) -> tuple[str, dict]:
        prompt = _messages_to_prompt(messages)
        max_tokens = int(kwargs.get("max_tokens", 512))
        temperature = float(kwargs.get("temperature", 0.2))
        text, usage = gemini.generate(
            prompt, model=self.model,
            max_output_tokens=max_tokens, temperature=temperature,
        )
        return text, usage

    def check_connection(self) -> dict:
        try:
            text, _ = gemini.generate(
                "Reply with exactly: OK", model=self.model, max_output_tokens=8)
            return {"ok": text.strip().upper().startswith("OK"),
                    "detail": f"model={self.model}"}
        except Exception as e:
            return {"ok": False, "detail": str(e)[:200]}
