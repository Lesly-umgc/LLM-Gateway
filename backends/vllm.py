"""vLLM backend — OpenAI-compatible client aimed at a vLLM server.

This is the production path: vLLM serves the (INT8-quantized) model on GPU
hardware, see serving/vllm_int8_example.yaml. The client is code-complete and
connection-tested against a mock OpenAI-compatible server (see tests), but was
NOT executed against a real vLLM server in this environment (no GPU) — that is
documented in the README under "What was NOT run here".

Config via env:
  VLLM_BASE_URL   e.g. http://vllm-svc:8000/v1   (default http://localhost:8000/v1)
  VLLM_MODEL      e.g. meta-llama/Meta-Llama-3-8B-Instruct-AWQ
  VLLM_API_KEY    optional bearer token if the server requires one
"""

from __future__ import annotations

import os
import urllib.request
import urllib.error
import json

from backends.base import Backend

BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1").rstrip("/")
MODEL = os.environ.get("VLLM_MODEL", "")
API_KEY = os.environ.get("VLLM_API_KEY", "")


class VLLMBackend(Backend):
    name = "vllm"

    def __init__(self, base_url: str = BASE_URL, model: str = MODEL,
                 api_key: str = API_KEY, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.model = model or "default"
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def generate(self, messages: list[dict], **kwargs) -> tuple[str, dict]:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": int(kwargs.get("max_tokens", 512)),
            "temperature": float(kwargs.get("temperature", 0.2)),
        }
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body, headers=self._headers(), method="POST",
        )
        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout)
            data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"vLLM server HTTP {e.code}: {e.read().decode()[:300]}") from None
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Cannot reach vLLM server at {self.base_url}: {e.reason}") from None
        choice = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return choice, {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        }

    def check_connection(self) -> dict:
        """GET /models — works against any OpenAI-compatible server (incl. mocks)."""
        req = urllib.request.Request(
            f"{self.base_url}/models", headers=self._headers(), method="GET")
        try:
            resp = urllib.request.urlopen(req, timeout=15)
            data = json.loads(resp.read().decode())
            models = [m.get("id") for m in data.get("data", [])]
            return {"ok": True, "detail": f"models={models[:5]}"}
        except Exception as e:
            return {"ok": False, "detail": str(e)[:200]}
