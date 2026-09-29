"""vLLM backend connection test against a local mock OpenAI-compatible server."""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, ".")

from backends.vllm import VLLMBackend  # noqa: E402


class MockVLLM(BaseHTTPRequestHandler):
    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        assert self.path == "/v1/models"
        self._send({"data": [{"id": "mock-llama-awq"}]})

    def do_POST(self):
        assert self.path == "/v1/chat/completions"
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        assert payload["model"] == "mock-llama-awq"
        self._send({
            "choices": [{"message": {"role": "assistant",
                                     "content": "mock reply"}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 4},
        })

    def log_message(self, *args):
        pass


def _server():
    srv = HTTPServer(("127.0.0.1", 0), MockVLLM)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_vllm_check_connection_against_mock():
    srv = _server()
    url = f"http://127.0.0.1:{srv.server_port}/v1"
    b = VLLMBackend(base_url=url, model="mock-llama-awq")
    res = b.check_connection()
    assert res["ok"], res
    assert "mock-llama-awq" in res["detail"]
    srv.shutdown()


def test_vllm_generate_against_mock():
    srv = _server()
    url = f"http://127.0.0.1:{srv.server_port}/v1"
    b = VLLMBackend(base_url=url, model="mock-llama-awq")
    text, usage = b.generate([{"role": "user", "content": "hi"}])
    assert text == "mock reply"
    assert usage == {"prompt_tokens": 12, "completion_tokens": 4}
    srv.shutdown()


def test_vllm_unreachable_reports_clean_error():
    b = VLLMBackend(base_url="http://127.0.0.1:1/v1", model="x", timeout=2)
    res = b.check_connection()
    assert not res["ok"]
    try:
        b.generate([{"role": "user", "content": "hi"}])
        raise AssertionError("should have raised")
    except RuntimeError as e:
        assert "Cannot reach vLLM server" in str(e)
