"""Tests for the claude_oss loopback shim server."""

from __future__ import annotations

import json
import os
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

from ucode.agents.claude_oss.server import ModelRouter, make_server
from ucode.gateway_proxy import TokenCache


class TestModelRouter(unittest.TestCase):
    def test_exact_match_passes_through(self):
        router = ModelRouter(["databricks-glm-5-2", "databricks-kimi-k3"], "databricks-glm-5-2")
        self.assertEqual(router.resolve("databricks-kimi-k3"), "databricks-kimi-k3")

    def test_strips_the_1m_context_suffix_before_matching(self):
        router = ModelRouter(["databricks-kimi-k3"], "databricks-kimi-k3")
        self.assertEqual(router.resolve("databricks-kimi-k3[1m]"), "databricks-kimi-k3")

    def test_unrecognised_family_falls_back_to_default(self):
        router = ModelRouter(["databricks-glm-5-2"], "databricks-glm-5-2")
        self.assertEqual(router.resolve("claude-opus-4-8"), "databricks-glm-5-2")

    def test_bare_family_name_resolves_to_the_newest_match(self):
        router = ModelRouter(
            ["databricks-kimi-k2-7-code", "databricks-kimi-k3"], "databricks-kimi-k3"
        )
        self.assertEqual(router.resolve("kimi"), "databricks-kimi-k3")

    def test_empty_or_non_string_falls_back_to_default(self):
        router = ModelRouter(["databricks-glm-5-2"], "databricks-glm-5-2")
        self.assertEqual(router.resolve(None), "databricks-glm-5-2")
        self.assertEqual(router.resolve(123), "databricks-glm-5-2")


class _StubGateway(BaseHTTPRequestHandler):
    """Stands in for /ai-gateway/mlflow/v1/chat/completions."""

    response_body: dict = {}
    last_headers: dict[str, str] = {}

    def do_POST(self):  # noqa: N802
        type(self).last_headers = dict(self.headers)
        length = int(self.headers.get("Content-Length") or 0)
        self.request_body = json.loads(self.rfile.read(length).decode("utf-8"))
        body = json.dumps(type(self).response_body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        return


class TestMessagesEndpoint(unittest.TestCase):
    def setUp(self):
        _StubGateway.response_body = {
            "id": "chatcmpl-1",
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        }
        _StubGateway.last_headers = {}
        self.gateway = HTTPServer(("127.0.0.1", 0), _StubGateway)
        threading.Thread(target=self.gateway.serve_forever, daemon=True).start()
        gateway_host = f"http://127.0.0.1:{self.gateway.server_address[1]}"

        # A TokenCache pointed at a fake workspace never calls `databricks auth
        # token` here because get_databricks_token honours DATABRICKS_BEARER
        # first (see ucode.databricks.get_token precedence, mirrored by
        # get_databricks_token) — set it so no live CLI/workspace is needed.
        os.environ["DATABRICKS_BEARER"] = "stub-token"
        self.tokens = TokenCache(gateway_host, None)
        router = ModelRouter(["databricks-glm-5-2"], "databricks-glm-5-2")
        self.server = make_server(gateway_host, self.tokens, router)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.tokens.stop()
        self.gateway.shutdown()
        self.gateway.server_close()

    def _post(self, path: str, payload: dict) -> tuple[int, dict]:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={"Authorization": "Bearer whatever-claude-code-sent"},
        )
        with urllib.request.urlopen(request) as response:
            return response.status, json.loads(response.read().decode("utf-8"))

    def test_non_streaming_message_translates_both_ways(self):
        status, body = self._post(
            "/v1/messages",
            {
                "model": "databricks-glm-5-2",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["role"], "assistant")
        self.assertEqual(body["content"][0]["text"], "hi")
        self.assertEqual(body["stop_reason"], "end_turn")
        self.assertEqual(body["usage"]["input_tokens"], 10)

    def test_the_clients_own_authorization_header_is_never_forwarded(self):
        self._post(
            "/v1/messages",
            {
                "model": "databricks-glm-5-2",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        self.assertIn("Authorization", _StubGateway.last_headers)
        self.assertNotEqual(
            _StubGateway.last_headers["Authorization"], "Bearer whatever-claude-code-sent"
        )
        self.assertEqual(_StubGateway.last_headers["Authorization"], "Bearer stub-token")

    def test_count_tokens_endpoint(self):
        status, body = self._post(
            "/v1/messages/count_tokens",
            {"messages": [{"role": "user", "content": "hello there"}]},
        )
        self.assertEqual(status, 200)
        self.assertIn("input_tokens", body)
        self.assertGreater(body["input_tokens"], 0)

    def test_unknown_path_is_404(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._post("/not-a-real-path", {})
        self.assertEqual(ctx.exception.code, 404)
