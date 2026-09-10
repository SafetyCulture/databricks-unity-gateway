"""Loopback HTTP server presenting the Anthropic Messages API over Databricks
OSS chat models (GLM, Kimi, Inkling, ...).

Claude Code talks to this; it talks to `/ai-gateway/mlflow/v1/chat/completions`.
Ported and rebuilt from SafetyCulture/experimental#474/#475/#478 against this
repo's own token cache (`ucode.gateway_proxy.TokenCache`) and model discovery
(`ucode.databricks.discover_oss_models`/`model_token_limits`/`supports_vision`)
instead of duplicating them, so there is exactly one place these numbers live.

Security posture, matching `ucode.gateway_proxy`'s relayed-auth proxy:
  - Binds 127.0.0.1 only, never off-host.
  - The workspace token lives in memory and is refreshed off the request path.
  - The `Authorization` header Claude Code sends is discarded, not forwarded:
    the shim authenticates to Databricks with its own token, so no Anthropic
    credential can leak to the gateway.
  - Nothing is logged unless CLAUDE_OSS_SHIM_LOG is set, and tokens are never
    written even then.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from ucode.constants import LOOPBACK_HOST
from ucode.databricks import model_token_limits, newest, supports_vision
from ucode.gateway_proxy import log_token_refresh_failure

from . import translate

OSS_ROUTE = "/ai-gateway/mlflow/v1/chat/completions"
MAX_BODY_BYTES = 64 * 1024 * 1024


class ModelRouter:
    """Maps the model name Claude Code sends onto a Databricks model id.

    Claude Code always sends a model string, and its `/model` picker offers
    whatever the `ANTHROPIC_DEFAULT_*_MODEL` env vars name (see claude.py's
    `render_overlay`, extended in Task 4 to point those at real Databricks
    ids for OSS mode). The common case is an exact match that passes straight
    through, which is what makes in-session `/model` switching work.
    Anything unrecognised falls back to the default model rather than
    failing the turn.
    """

    def __init__(self, catalogue: list[str], default: str) -> None:
        self.catalogue = catalogue
        self.default = default

    def resolve(self, requested: Any) -> str:
        if not isinstance(requested, str) or not requested:
            return self.default
        # Claude Code appends context-window suffixes such as "[1m]".
        cleaned = requested.split("[", 1)[0].strip()
        if cleaned in self.catalogue:
            return cleaned
        for model in self.catalogue:
            if model.rsplit(".", 1)[-1] == cleaned:
                return model
        lowered = cleaned.lower()
        for family in ("glm", "kimi"):
            if family in lowered:
                match = newest(self.catalogue, family)
                if match:
                    return match
        return self.default


class _Logger:
    """Append-only JSONL tracing, enabled by CLAUDE_OSS_SHIM_LOG.

    Off by default. Records translated request/response bodies only — never
    the bearer token — which is what makes a translation bug diagnosable.

    Those bodies are the full conversation (prompts, tool output, images), so
    the file is created 0o600 rather than left to the process umask. The mode
    only applies at creation: a file the user already made keeps whatever mode
    they gave it."""

    _CREATE_MODE = 0o600

    def __init__(self, path: str | None) -> None:
        self.path = path
        self._lock = threading.Lock()

    def __call__(self, kind: str, payload: Any) -> None:
        if not self.path:
            return
        try:
            line = json.dumps({"kind": kind, "payload": payload}, default=str)
        except (TypeError, ValueError):
            return
        with self._lock:
            try:
                descriptor = os.open(
                    self.path,
                    os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                    self._CREATE_MODE,
                )
                with open(descriptor, "a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
            except OSError:
                pass


def _max_output_for(model: str) -> int | None:
    limits = model_token_limits(model)
    return limits["output"] if limits else None


class Handler(BaseHTTPRequestHandler):
    # Bound by make_server().
    host: str
    tokens: Any  # ucode.gateway_proxy.TokenCache
    router: ModelRouter
    trace: _Logger
    allow_images: bool
    reasoning: str

    protocol_version = "HTTP/1.1"
    server_version = "ucode-claude-oss-shim"

    def log_message(self, format: str, *args: object) -> None:
        return  # never log request lines; they can carry paths

    def _send_json(self, status: int, payload: dict, *, close: bool = False) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if close:
            # BaseHTTPRequestHandler.send_header also flips close_connection for
            # this one, so the socket is torn down after the response is flushed.
            self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: int, message: str) -> None:
        """Report an error and close the connection.

        `protocol_version = "HTTP/1.1"` makes keep-alive the default, but two
        error paths answer without having consumed the request body: the
        unknown-path 404 in `do_POST` (which runs before any read happens) and
        the 413 in `_read_body` (which refuses to read an oversized body at all).
        Those unread bytes are then parsed as the start of the next request on
        the same connection, desyncing every request after it — a corrupted
        session, not a clean failure.

        Closing on every error response, rather than only on those two, is the
        cheaper correctness argument: error responses are rare, so nothing is
        lost, and no future error path can reintroduce the desync by forgetting
        to drain. Success responses are untouched and stay kept-alive.
        """
        self.trace("error", {"status": status, "message": message})
        # Set here as well as via the header, so a write that fails partway
        # still leaves the connection marked for teardown.
        self.close_connection = True
        try:
            self._send_json(status, translate.error_body(status, message), close=True)
        except OSError:
            pass

    def _read_body(self) -> dict | None:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            self._send_error(400, "Empty request body.")
            return None
        if length > MAX_BODY_BYTES:
            self._send_error(413, "Request body too large.")
            return None
        try:
            parsed = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self._send_error(400, "Request body was not valid JSON.")
            return None
        # Both endpoints treat the body as a mapping (`body.get(...)`), so a bare
        # list or string has to be rejected here rather than raising downstream.
        if not isinstance(parsed, dict):
            self._send_error(400, "Request body must be a JSON object.")
            return None
        return parsed

    def _open_upstream(self, payload: dict, *, stream: bool):
        url = f"{self.host.rstrip('/')}{OSS_ROUTE}"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {self.tokens.token}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream" if stream else "application/json",
                "User-Agent": "ucode-claude-oss-shim",
                "x-databricks-use-coding-agent-mode": "true",
            },
        )
        return urllib.request.urlopen(request, timeout=900)

    def _upstream(self, payload: dict, *, stream: bool):
        """POST to the gateway route, retrying once with a force-refreshed token
        if the first attempt is rejected as unauthenticated.

        `TokenCache._ensure_fresh` deliberately keeps serving a possibly-stale
        token when a background refresh fails, on the stated understanding that
        "a request that then 401s triggers a forced refresh + retry" — which
        `gateway_proxy._ProxyHandler._handle` implements for the relayed proxy.
        Without the same retry here, a token that lapsed across a laptop sleep
        (the monotonic clock the refresher polls on stops advancing) reaches
        Claude Code as an `authentication_error` in the middle of a session, and
        the only recovery is restarting it.

        One retry, not a loop, matching `_handle`: if the second attempt is still
        rejected the credential really is bad, and the gateway's own message
        should reach the caller rather than being retried behind their back.
        """
        try:
            return self._open_upstream(payload, stream=stream)
        except urllib.error.HTTPError as exc:
            if exc.code not in (401, 403):
                raise
            # Drain the small error body so the connection can be released.
            try:
                exc.read()
                exc.close()
            except OSError:
                pass
        self.trace("token_refresh", {"reason": "upstream rejected the token"})
        try:
            self.tokens.refresh()
        except RuntimeError as exc:
            # The Databricks OAuth session is dead, not just the access token, and
            # cannot be re-minted non-interactively. Surface the `databricks auth
            # login` hint, then still retry so the gateway's own status and message
            # are what the caller sees.
            log_token_refresh_failure(exc)
        return self._open_upstream(payload, stream=stream)

    def do_GET(self) -> None:  # noqa: N802
        # Live-reproduced: the real Claude Code binary requests
        # "/v1/models?limit=1000", not the bare path - strip the query string
        # before matching, same as do_POST already does.
        path = self.path.split("?", 1)[0].rstrip("/")
        # Traced unconditionally (GET carries no request body and the shim
        # discards whatever Authorization header Claude Code sends, so there's
        # nothing secret here) - this is the only way to tell, from a live
        # CLAUDE_OSS_SHIM_LOG trace, whether Claude Code ever actually asked for
        # /v1/models rather than silently skipping discovery.
        self.trace("get", {"path": self.path})
        if path in ("/health", ""):
            self._send_json(200, {"status": "ok", "models": self.router.catalogue})
            return
        if path == "/v1/models":
            # Claude Code's native CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY
            # calls this to populate the full /model picker, rather than being
            # limited to the 3 hardcoded ANTHROPIC_DEFAULT_*_MODEL tiers. Shape
            # matches Anthropic's real /v1/models response, which
            # ucode.databricks.list_anthropic_model_catalog already parses
            # elsewhere ({"data": [{"id", "display_name", "type"}], "has_more"}).
            self._send_json(
                200,
                {
                    "data": [
                        {"type": "model", "id": model_id, "display_name": model_id}
                        for model_id in self.router.catalogue
                    ],
                    "has_more": False,
                },
            )
            return
        self._send_error(404, f"Unknown path: {self.path}")

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/")
        if path == "/v1/messages/count_tokens":
            self._count_tokens()
        elif path == "/v1/messages":
            self._messages()
        else:
            self._send_error(404, f"Unknown path: {self.path}")

    def _count_tokens(self) -> None:
        body = self._read_body()
        if body is None:
            return
        try:
            estimate = translate.count_tokens(body)
        except Exception as exc:  # noqa: BLE001 - see _messages
            self._send_error(400, f"Could not measure the request: {type(exc).__name__}: {exc}")
            return
        self._send_json(200, {"input_tokens": estimate})

    def _messages(self) -> None:
        body = self._read_body()
        if body is None:
            return
        # The translators assume Anthropic's shapes: `_convert_messages` iterates
        # `messages` and calls `.get` on each element, so a bare string is walked
        # character by character and raises AttributeError. BaseHTTPRequestHandler
        # turns nothing into a response, so the connection would simply drop
        # mid-turn. Catch broadly and answer 400: the request is unusable either
        # way, and a clean Anthropic error body is something Claude Code can
        # report, whereas a dropped socket is not.
        try:
            model = self.router.resolve(body.get("model"))
            payload = translate.anthropic_to_openai(
                body,
                model=model,
                max_output=_max_output_for(model),
                allow_images=self.allow_images and supports_vision(model),
            )
        except Exception as exc:  # noqa: BLE001 - malformed input must not drop the connection
            self._send_error(400, f"Could not translate the request: {type(exc).__name__}: {exc}")
            return
        self.trace("request", {"model_requested": body.get("model"), "upstream": payload})
        if body.get("stream"):
            self._stream(payload, model)
        else:
            self._once(payload, model)

    def _once(self, payload: dict, model: str) -> None:
        try:
            with self._upstream(payload, stream=False) as response:
                upstream = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            self._send_error(exc.code, _upstream_message(exc))
            return
        except (urllib.error.URLError, OSError, ValueError) as exc:
            self._send_error(502, f"Databricks gateway unreachable: {exc}")
            return
        self.trace("response", upstream)
        self._send_json(
            200, translate.openai_to_anthropic(upstream, model=model, reasoning=self.reasoning)
        )

    def _stream(self, payload: dict, model: str) -> None:
        try:
            response = self._upstream(payload, stream=True)
        except urllib.error.HTTPError as exc:
            self._send_error(exc.code, _upstream_message(exc))
            return
        except (urllib.error.URLError, OSError) as exc:
            self._send_error(502, f"Databricks gateway unreachable: {exc}")
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        translator = translate.StreamTranslator(model=model, reasoning=self.reasoning)
        try:
            with response:
                for line in response:
                    text = line.decode("utf-8", "replace").strip()
                    if not text.startswith("data:"):
                        continue
                    data = text[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except ValueError:
                        continue
                    self.trace("chunk", chunk)
                    for event in translator.feed(chunk):
                        self._write_event(*event)
            for event in translator.finish():
                self._write_event(*event)
        except (BrokenPipeError, ConnectionResetError):
            return
        except (urllib.error.URLError, OSError) as exc:
            try:
                self._write_event("error", translate.error_body(502, f"Stream interrupted: {exc}"))
                for event in translator.finish():
                    self._write_event(*event)
            except OSError:
                pass

    def _write_event(self, name: str, payload: dict) -> None:
        block = f"event: {name}\ndata: {json.dumps(payload)}\n\n".encode()
        self.wfile.write(block)
        self.wfile.flush()


def _upstream_message(exc: urllib.error.HTTPError) -> str:
    try:
        raw = exc.read().decode("utf-8", "replace")
    except OSError:
        return f"HTTP {exc.code} from Databricks gateway."
    try:
        parsed = json.loads(raw)
    except ValueError:
        return raw.strip()[:2000] or f"HTTP {exc.code} from Databricks gateway."
    for key in ("message", "error_message", "detail"):
        if isinstance(parsed.get(key), str):
            return parsed[key]
    error = parsed.get("error")
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return error["message"]
    if isinstance(error, str):
        return error
    return raw.strip()[:2000]


def make_server(
    host: str,
    tokens: Any,
    router: ModelRouter,
    *,
    port: int = 0,
    allow_images: bool = True,
    reasoning: str = "thinking",
) -> ThreadingHTTPServer:
    handler = type(
        "BoundHandler",
        (Handler,),
        {
            "host": host,
            "tokens": tokens,
            "router": router,
            "trace": _Logger(os.environ.get("CLAUDE_OSS_SHIM_LOG")),
            "allow_images": allow_images,
            "reasoning": reasoning,
        },
    )
    try:
        return ThreadingHTTPServer((LOOPBACK_HOST, port), handler)
    except OSError:
        # Requested port is taken (a stale shim from a killed session still
        # holding the socket) — let the OS pick one; the caller reads it back.
        return ThreadingHTTPServer((LOOPBACK_HOST, 0), handler)
