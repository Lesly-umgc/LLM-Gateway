"""Ollama backend — local LLM serving over Ollama's HTTP API.

This is the primary demo backend for LNM Gateway: it runs on CPU-only
machines (including Apple Silicon Macs), serves quantized models, and speaks
a simple /api/generate + /api/chat HTTP API.

Config via env:
  OLLAMA_BASE_URL   e.g. http://localhost:11434   (default)
  OLLAMA_MODEL      e.g. llama3.2:3b              (default; 1b fallback on tiny boxes)
  OLLAMA_NUM_CTX    context window, default 4096
"""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error

from backends.base import Backend

BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "4096"))


class OllamaBackend(Backend):
    name = "ollama"

    def __init__(self, base_url: str = BASE_URL, model: str = MODEL,
                 num_ctx: int = NUM_CTX, timeout: int = 300):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.num_ctx = num_ctx
        self.timeout = timeout

    def _post(self, path: str, payload: dict) -> dict:
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.base_url}{path}", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout)
            return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"Ollama HTTP {e.code}: {e.read().decode()[:300]}") from None
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.base_url}: {e.reason}. "
                f"Is `ollama serve` running and `{self.model}` pulled?") from None

    def generate(self, messages: list[dict], **kwargs) -> tuple[str, dict]:
        data = self._post("/api/chat", {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_ctx": self.num_ctx,
                "temperature": float(kwargs.get("temperature", 0.2)),
                "num_predict": int(kwargs.get("max_tokens", 512)),
            },
        })
        text = (data.get("message") or {}).get("content", "")
        return text, {
            "prompt_tokens": int(data.get("prompt_eval_count", 0)),
            "completion_tokens": int(data.get("eval_count", 0)),
        }

    def classify(self, prompt: str) -> str:
        """One-word classification helper for rails judges / monitors."""
        data = self._post("/api/generate", {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 16,
                        "num_ctx": self.num_ctx},
        })
        return (data.get("response") or "").strip()

    def check_connection(self) -> dict:
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags",
                                         method="GET")
            resp = urllib.request.urlopen(req, timeout=15)
            data = json.loads(resp.read().decode())
            models = [m.get("name") for m in data.get("models", [])]
            ok = any(self.model in (m or "") for m in models)
            return {"ok": ok,
                    "detail": f"model={self.model} available={models}"}
        except Exception as e:
            return {"ok": False, "detail": str(e)[:200]}
