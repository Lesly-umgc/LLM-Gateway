"""Backend interface for the gateway.

A backend turns an OpenAI-style message list into (text, usage).
usage = {"prompt_tokens": int, "completion_tokens": int}
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Backend(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, messages: list[dict], **kwargs) -> tuple[str, dict]:
        raise NotImplementedError

    def check_connection(self) -> dict:
        """Lightweight connectivity check. Returns {"ok": bool, "detail": str}."""
        raise NotImplementedError
