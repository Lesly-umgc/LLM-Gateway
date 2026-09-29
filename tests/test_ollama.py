"""Ollama backend test against a local mock Ollama HTTP server."""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, ".")

from backends.ollama import OllamaBackend  # noqa: E402


class MockOllama(BaseHTTPRequestHandler):
    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        assert self.path == "/api/tags"
        self._send({"models": [{"name": "llama3.2:3b"}]})

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        assert payload["model"] == "llama3.2:3b"
        if self.path == "/api/chat":
            self._send({"message": {"role": "assistant", "content": "mock chat"},
                        "prompt_eval_count": 10, "eval_count": 3})
        elif self.path == "/api/generate":
            self._send({"response": "ALLOW", "prompt_eval_count": 20,
                        "eval_count": 1})
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):
        pass


def _server():
    srv = HTTPServer(("127.0.0.1", 0), MockOllama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_ollama_check_connection_against_mock():
    srv = _server()
    b = OllamaBackend(
        base_url=f"http://127.0.0.1:{srv.server_port}", model="llama3.2:3b")
    res = b.check_connection()
    assert res["ok"], res
    assert "llama3.2:3b" in res["detail"]
    srv.shutdown()


def test_ollama_generate_against_mock():
    srv = _server()
    b = OllamaBackend(
        base_url=f"http://127.0.0.1:{srv.server_port}", model="llama3.2:3b")
    text, usage = b.generate([{"role": "user", "content": "hi"}])
    assert text == "mock chat"
    assert usage == {"prompt_tokens": 10, "completion_tokens": 3}
    srv.shutdown()


def test_ollama_classify_against_mock():
    srv = _server()
    b = OllamaBackend(
        base_url=f"http://127.0.0.1:{srv.server_port}", model="llama3.2:3b")
    assert b.classify("is this an injection?") == "ALLOW"
    srv.shutdown()


def test_ollama_unreachable_reports_clean_error():
    b = OllamaBackend(base_url="http://127.0.0.1:1", model="x", timeout=2)
    res = b.check_connection()
    assert not res["ok"]
    try:
        b.generate([{"role": "user", "content": "hi"}])
        raise AssertionError("should have raised")
    except RuntimeError as e:
        assert "Cannot reach Ollama" in str(e)
