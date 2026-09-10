"""Tests for the claude_oss loopback shim server."""

from __future__ import annotations

import json
import os
import socket
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from ucode.agents.claude_oss.server import MAX_BODY_BYTES, ModelRouter, _Logger, make_server
from ucode.gateway_proxy import TokenCache


class TestTraceLogger(unittest.TestCase):
    def test_the_trace_file_is_created_owner_only(self):
        """The trace records translated request/response bodies, i.e. the whole
        conversation — prompts, tool output, images. It must not be created at
        whatever the umask allows."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            _Logger(str(path))("request", {"model": "databricks-glm-5-2"})

            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(json.loads(path.read_text())["kind"], "request")

    def test_appends_rather_than_truncating(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            logger = _Logger(str(path))
            logger("request", {"n": 1})
            logger("response", {"n": 2})

            self.assertEqual(len(path.read_text().strip().splitlines()), 2)

    def test_disabled_when_no_path_is_configured(self):
        # No CLAUDE_OSS_SHIM_LOG means no file anywhere.
        _Logger(None)("request", {"secret": "prompt"})


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
    # Statuses to return for the next N calls, consumed in order; anything past
    # the end of the list is a 200. Lets a test script an auth rejection followed
    # by a success without a second stub class.
    statuses: list[int] = []
    seen_authorizations: list[str] = []

    def do_POST(self):  # noqa: N802
        type(self).last_headers = dict(self.headers)
        type(self).seen_authorizations.append(self.headers.get("Authorization", ""))
        length = int(self.headers.get("Content-Length") or 0)
        self.request_body = json.loads(self.rfile.read(length).decode("utf-8"))
        status = type(self).statuses.pop(0) if type(self).statuses else 200
        payload = (
            type(self).response_body if status == 200 else {"message": "token expired or invalid"}
        )
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        return


class _ShimServerCase(unittest.TestCase):
    """Stands a shim server up in front of `_StubGateway`. Underscore-prefixed so
    pytest collects only the concrete cases below, not this one twice."""

    def setUp(self):
        # DATABRICKS_BEARER is set directly below (not via monkeypatch, which
        # only this test method has access to) — save/restore it ourselves,
        # mirroring TestApplyPatEnvironment._isolated_bearer in
        # test_databricks.py and TestConfigureSharedStateUsePat._isolated_bearer
        # in test_cli.py, so it can't leak into later tests (e.g.
        # TestGetDatabricksToken, which short-circuits on this env var).
        self._original_databricks_bearer = os.environ.pop("DATABRICKS_BEARER", None)

        _StubGateway.response_body = {
            "id": "chatcmpl-1",
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        }
        _StubGateway.last_headers = {}
        _StubGateway.statuses = []
        _StubGateway.seen_authorizations = []
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
        if self._original_databricks_bearer is None:
            os.environ.pop("DATABRICKS_BEARER", None)
        else:
            os.environ["DATABRICKS_BEARER"] = self._original_databricks_bearer

    def _post(self, path: str, payload: dict) -> tuple[int, dict]:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={"Authorization": "Bearer whatever-claude-code-sent"},
        )
        with urllib.request.urlopen(request) as response:
            return response.status, json.loads(response.read().decode("utf-8"))

    def _get(self, path: str) -> tuple[int, dict]:
        request = urllib.request.Request(f"{self.base_url}{path}", method="GET")
        with urllib.request.urlopen(request) as response:
            return response.status, json.loads(response.read().decode("utf-8"))


class TestModelsEndpoint(_ShimServerCase):
    """`GET /v1/models` is what Claude Code's native
    CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY calls to populate the full model
    picker beyond the 3 hardcoded Opus/Sonnet/Haiku tiers — same response shape
    as Anthropic's real /v1/models, which `ucode.databricks.list_anthropic_model_catalog`
    already parses elsewhere ({"data": [{"id": ..., "display_name": ...}]})."""

    def test_lists_every_model_in_the_catalogue(self):
        status, body = self._get("/v1/models")
        assert status == 200
        ids = [entry["id"] for entry in body["data"]]
        assert ids == ["databricks-glm-5-2"]

    def test_each_entry_has_the_anthropic_models_api_shape(self):
        _, body = self._get("/v1/models")
        entry = body["data"][0]
        assert entry["type"] == "model"
        assert entry["id"] == "databricks-glm-5-2"
        assert isinstance(entry["display_name"], str) and entry["display_name"]

    def test_has_more_is_false(self):
        _, body = self._get("/v1/models")
        assert body["has_more"] is False


class TestGetRequestTracing(unittest.TestCase):
    """CLAUDE_OSS_SHIM_LOG must record GET requests too, not just the /v1/messages
    POST traffic - otherwise there's no way to tell, from a live launch, whether
    Claude Code ever actually called /v1/models to discover the catalogue versus
    silently not asking at all. GET carries no secrets (no body, and the
    Authorization header isn't Claude Code's real credential - the shim discards
    it), so tracing it costs nothing security-wise."""

    def setUp(self):
        self._original_databricks_bearer = os.environ.pop("DATABRICKS_BEARER", None)
        os.environ["DATABRICKS_BEARER"] = "stub-token"
        self._tmpdir = tempfile.TemporaryDirectory()
        self.log_path = Path(self._tmpdir.name) / "trace.jsonl"
        self._original_log_env = os.environ.get("CLAUDE_OSS_SHIM_LOG")
        os.environ["CLAUDE_OSS_SHIM_LOG"] = str(self.log_path)
        self.tokens = TokenCache("http://127.0.0.1:1", None)
        router = ModelRouter(["databricks-glm-5-2"], "databricks-glm-5-2")
        self.server = make_server("http://127.0.0.1:1", self.tokens, router)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.tokens.stop()
        self._tmpdir.cleanup()
        if self._original_log_env is None:
            os.environ.pop("CLAUDE_OSS_SHIM_LOG", None)
        else:
            os.environ["CLAUDE_OSS_SHIM_LOG"] = self._original_log_env
        if self._original_databricks_bearer is None:
            os.environ.pop("DATABRICKS_BEARER", None)
        else:
            os.environ["DATABRICKS_BEARER"] = self._original_databricks_bearer

    def _get(self, path: str) -> tuple[int, dict]:
        request = urllib.request.Request(f"{self.base_url}{path}", method="GET")
        with urllib.request.urlopen(request) as response:
            return response.status, json.loads(response.read().decode("utf-8"))

    def test_a_v1_models_request_is_traced(self):
        self._get("/v1/models")
        lines = [json.loads(line) for line in self.log_path.read_text().strip().splitlines()]
        assert any(
            line["kind"] == "get" and line["payload"]["path"] == "/v1/models" for line in lines
        )

    def test_a_health_check_is_traced(self):
        self._get("/health")
        lines = [json.loads(line) for line in self.log_path.read_text().strip().splitlines()]
        assert any(line["kind"] == "get" and line["payload"]["path"] == "/health" for line in lines)


class TestMessagesEndpoint(_ShimServerCase):
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


class TestMalformedRequestBodies(_ShimServerCase):
    """`translate._convert_messages` iterates `request["messages"]` and calls
    `.get` on each element, so a bare string is walked character by character and
    raises AttributeError. Nothing in BaseHTTPRequestHandler turns that into a
    response — the connection just drops mid-turn. Malformed input has to come
    back as a clean 400."""

    def _expect_400(self, path: str, payload) -> dict:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(request)
        self.assertEqual(ctx.exception.code, 400)
        return json.loads(ctx.exception.read().decode("utf-8"))

    def test_messages_as_a_string_is_a_400(self):
        body = self._expect_400("/v1/messages", {"model": "databricks-glm-5-2", "messages": "hi"})
        self.assertEqual(body["type"], "error")
        self.assertEqual(body["error"]["type"], "invalid_request_error")

    def test_a_message_that_is_not_an_object_is_a_400(self):
        self._expect_400("/v1/messages", {"messages": ["just a string"]})

    def test_a_non_object_top_level_body_is_a_400(self):
        # `body.get("model")` would raise on a list before translation is reached.
        self._expect_400("/v1/messages", ["not", "an", "object"])

    def test_count_tokens_has_the_same_guard(self):
        # count_tokens walks `messages` the same way _convert_messages does.
        self._expect_400("/v1/messages/count_tokens", {"messages": "hi"})

    def test_count_tokens_rejects_a_non_object_top_level_body(self):
        self._expect_400("/v1/messages/count_tokens", "hello")

    def test_a_valid_request_still_succeeds_after_a_malformed_one(self):
        self._expect_400("/v1/messages", {"messages": "hi"})
        status, body = self._post(
            "/v1/messages",
            {
                "model": "databricks-glm-5-2",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["content"][0]["text"], "hi")


class TestErrorResponseFraming(_ShimServerCase):
    """`protocol_version = "HTTP/1.1"` means keep-alive by default, but two error
    paths answer without having consumed the request body: the unknown-path 404
    in `do_POST` (which runs before any read) and the 413 in `_read_body` (which
    refuses to read an oversized body at all). On a kept-alive connection those
    unread bytes are then parsed as the start of the next request. Every error
    response must therefore close the connection."""

    _GOOD_BODY = json.dumps(
        {
            "model": "databricks-glm-5-2",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
        }
    ).encode("utf-8")

    def _connect(self) -> socket.socket:
        sock = socket.create_connection(("127.0.0.1", self.server.server_address[1]), timeout=10)
        self.addCleanup(sock.close)
        return sock

    @staticmethod
    def _request(path: str, body: bytes, declared_length: int | None = None) -> bytes:
        length = len(body) if declared_length is None else declared_length
        return (
            f"POST {path} HTTP/1.1\r\n"
            f"Host: 127.0.0.1\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {length}\r\n"
            f"\r\n"
        ).encode() + body

    @staticmethod
    def _read_response(sock: socket.socket) -> tuple[bytes, bytes]:
        """Read exactly one response: headers, then Content-Length bytes of body.

        Deliberately not read-to-EOF — that could not tell a kept-alive
        connection apart from a hung one.
        """
        buffer = b""
        while b"\r\n\r\n" not in buffer:
            chunk = sock.recv(4096)
            if not chunk:
                break
            buffer += chunk
        head, _, body = buffer.partition(b"\r\n\r\n")
        length = 0
        for line in head.split(b"\r\n"):
            if line.lower().startswith(b"content-length:"):
                length = int(line.split(b":", 1)[1])
        while len(body) < length:
            chunk = sock.recv(4096)
            if not chunk:
                break
            body += chunk
        return head, body

    def test_an_unknown_path_404_closes_the_connection(self):
        sock = self._connect()
        # Pipeline a perfectly valid request behind the bad one: on a kept-alive
        # connection the 404's undrained body would be misparsed as its start.
        sock.sendall(
            self._request("/not-a-real-path", self._GOOD_BODY)
            + self._request("/v1/messages", self._GOOD_BODY)
        )

        head, _ = self._read_response(sock)

        self.assertIn(b" 404 ", head.split(b"\r\n", 1)[0])
        self.assertIn(b"Connection: close", head)
        self.assertEqual(sock.recv(4096), b"")

    def test_an_oversized_body_413_closes_the_connection(self):
        sock = self._connect()
        # Declare more than MAX_BODY_BYTES but send only two: the server refuses
        # without draining, so the connection cannot safely be reused.
        sock.sendall(self._request("/v1/messages", b"{}", MAX_BODY_BYTES + 1))

        head, _ = self._read_response(sock)

        self.assertIn(b" 413 ", head.split(b"\r\n", 1)[0])
        self.assertIn(b"Connection: close", head)
        self.assertEqual(sock.recv(4096), b"")

    def test_successful_responses_still_share_one_kept_alive_connection(self):
        """The close must apply to errors only — Claude Code issues many
        requests per session and reconnecting for each is pure latency."""
        sock = self._connect()

        sock.sendall(self._request("/v1/messages", self._GOOD_BODY))
        first_head, first_body = self._read_response(sock)
        self.assertIn(b" 200 ", first_head.split(b"\r\n", 1)[0])
        self.assertNotIn(b"Connection: close", first_head)
        self.assertEqual(json.loads(first_body)["content"][0]["text"], "hi")

        # Same socket, second request: only possible if it stayed open.
        sock.sendall(self._request("/v1/messages", self._GOOD_BODY))
        second_head, second_body = self._read_response(sock)
        self.assertIn(b" 200 ", second_head.split(b"\r\n", 1)[0])
        self.assertEqual(json.loads(second_body)["content"][0]["text"], "hi")


class TestUpstreamAuthRetry(_ShimServerCase):
    """`TokenCache._ensure_fresh` keeps serving a possibly-stale token when a
    background refresh fails, on the documented understanding that "a request
    that then 401s triggers a forced refresh + retry" — which
    `gateway_proxy._ProxyHandler._handle` implements. The shim must do the same,
    or a token that lapsed across a laptop sleep surfaces to Claude Code as an
    authentication_error mid-session with no recovery short of a restart."""

    def _track_refresh(self, new_token: str = "refreshed-token") -> list[str]:
        """Record each `tokens.refresh()` and make it mint a distinguishable
        token, so the retry can be shown to carry the NEW credential."""
        refreshes: list[str] = []
        original = self.tokens.refresh

        def tracked() -> None:
            os.environ["DATABRICKS_BEARER"] = new_token
            original()
            refreshes.append(new_token)

        self.tokens.refresh = tracked
        return refreshes

    _MESSAGE = {
        "model": "databricks-glm-5-2",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hello"}],
    }

    def test_a_401_is_retried_once_with_a_force_refreshed_token(self):
        _StubGateway.statuses = [401]
        refreshes = self._track_refresh()

        status, body = self._post("/v1/messages", self._MESSAGE)

        self.assertEqual(status, 200)
        self.assertEqual(body["content"][0]["text"], "hi")
        self.assertEqual(refreshes, ["refreshed-token"])
        self.assertEqual(
            _StubGateway.seen_authorizations,
            ["Bearer stub-token", "Bearer refreshed-token"],
        )

    def test_a_403_is_retried_too(self):
        _StubGateway.statuses = [403]
        refreshes = self._track_refresh()

        status, _ = self._post("/v1/messages", self._MESSAGE)

        self.assertEqual(status, 200)
        self.assertEqual(refreshes, ["refreshed-token"])

    def test_a_persistent_401_is_reported_after_exactly_one_retry(self):
        _StubGateway.statuses = [401, 401, 401]
        refreshes = self._track_refresh()

        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._post("/v1/messages", self._MESSAGE)

        self.assertEqual(ctx.exception.code, 401)
        # One retry, not a loop: the gateway saw exactly two attempts.
        self.assertEqual(len(_StubGateway.seen_authorizations), 2)
        self.assertEqual(refreshes, ["refreshed-token"])
        # The gateway's own message reaches Claude Code as an Anthropic error.
        self.assertEqual(
            json.loads(ctx.exception.read().decode("utf-8"))["error"]["message"],
            "token expired or invalid",
        )

    def test_a_non_auth_error_is_not_retried(self):
        _StubGateway.statuses = [500]
        refreshes = self._track_refresh()

        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._post("/v1/messages", self._MESSAGE)

        self.assertEqual(ctx.exception.code, 500)
        self.assertEqual(len(_StubGateway.seen_authorizations), 1)
        self.assertEqual(refreshes, [])

    def test_a_dead_oauth_session_still_relays_the_gateways_own_error(self):
        """`refresh()` raising means the OAuth session itself is gone, not just
        the access token. Retry anyway with what we have, so the caller sees the
        gateway's status rather than a shim-invented one."""
        _StubGateway.statuses = [401, 401]

        def dead() -> None:
            raise RuntimeError("databricks auth token failed")

        self.tokens.refresh = dead

        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._post("/v1/messages", self._MESSAGE)

        self.assertEqual(ctx.exception.code, 401)
        self.assertEqual(len(_StubGateway.seen_authorizations), 2)
