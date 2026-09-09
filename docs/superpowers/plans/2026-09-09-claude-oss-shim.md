# Claude-on-OSS-Models Shim Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When a Databricks workspace has no Claude models but does have OSS chat models (GLM, Kimi, Inkling, ...) on the `/ai-gateway/mlflow/v1` route, `ucode claude` should transparently launch Claude Code against those OSS models through a local Anthropic↔OpenAI translation shim, instead of failing with "AI Gateway returned no Claude model ids".

**Architecture:** A new `ucode.agents.claude_oss` subpackage provides a pure translation module (`translate.py`, ported near-verbatim) and a loopback HTTP server (`server.py`) that presents the Anthropic Messages API to Claude Code and speaks OpenAI chat-completions to Databricks. `claude.py` gains a third launch mode alongside the existing direct and relayed modes: it starts the shim as a background thread (reusing the existing `gateway_proxy.TokenCache` and the existing `_launch_relayed`-style spawn-and-wait pattern, since Claude Code's normal `exec_or_spawn` replaces the process and would kill any background thread), points `render_overlay`'s existing `provider_models` mechanism at the shim's own loopback URL, and lets everything downstream (settings writing, `/model` tier switching, tool search) work exactly as it already does for real Claude models.

**Tech Stack:** Python 3.12, stdlib `http.server`/`threading` for the loopback server, `httpx` for upstream calls (matching `gateway_proxy.py`'s existing convention), `unittest` for tests (matching the source PRs and this repo's existing test style).

**Spec:** Ported from three merged PRs in `SafetyCulture/experimental`, in order:
- [#474](https://github.com/SafetyCulture/experimental/pull/474) — `hkf57/dbx-claude-shim`: the shim itself (`translate.py`, `server.py`, `databricks.py`, `__main__.py`, tests)
- [#475](https://github.com/SafetyCulture/experimental/pull/475) — enable `ENABLE_TOOL_SEARCH=1` by default (already true in this repo's `render_overlay` — see Task 6 note)
- [#478](https://github.com/SafetyCulture/experimental/pull/478) — Kimi K3: per-model context windows, the `[1m]` suffix, vision/image handling

A read-only copy of the source is available for reference during implementation:
```bash
git clone --depth 1 --filter=blob:none --sparse git@github.com:SafetyCulture/experimental.git /tmp/experimental-ref
cd /tmp/experimental-ref && git sparse-checkout set hkf57/dbx-claude-shim
```

## Global Constraints

- Binds `127.0.0.1` only, never off-host (spec: server.py security posture, all 3 PRs).
- The Databricks bearer token is held in memory only, never logged, never written to disk (spec: README "Security").
- The `Authorization`/`ANTHROPIC_AUTH_TOKEN` value Claude Code sends must be discarded, not forwarded — the shim authenticates to Databricks with its own token (spec: README "Security", server.py `_upstream`).
- `max_tokens` must be clamped to the route's real per-model cap before the request goes upstream, or the gateway hard-400s (spec: translate.py docstring, databricks.py `MAX_OUTPUT`).
- Every new source file matches this repo's existing module docstring + comment density (see `CLAUDE.md` conventions already followed in `src/ucode/agents/claude.py`, `src/ucode/gateway_proxy.py`) — explain *why*, not just *what*, the way the rest of this codebase does.
- No network or live workspace required to run the test suite — translation is tested as pure functions, the server is tested over real sockets against a stub gateway (spec: README "Tests").

---

## Task 0: Correct the OSS model token-limit table (pre-existing bug, no shim dependency)

This is a live correctness bug already shipped on this branch (commit `f9277b5`), independently confirmed by two people measuring the *same* SafetyCulture workspace (Luke Cameron, 2026-07-16; Charles Lee, 2026-08-05 and 2026-08-11) — both disagree with the numbers currently in `_MODEL_TOKEN_LIMITS`, which came from PR #420 testing a *different* Databricks workspace. Fix this first since Task 3 (the shim's model-limit lookups) depends on it, and OpenCode already ships the wrong numbers today.

**Files:**
- Modify: `src/ucode/databricks.py` (`_MODEL_TOKEN_LIMITS` dict and `model_token_limits` function, both added in commit `f9277b5` / earlier commits on this branch)
- Test: `tests/test_databricks.py`

**Interfaces:**
- Produces: `model_token_limits(model_id: str) -> dict[str, int] | None` (unchanged signature, corrected values + matching order), `_MODEL_TOKEN_LIMITS: dict[str, dict[str, int]]` (corrected + extended)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_databricks.py, inside the existing token-limits test class
def test_kimi_k3_gets_the_million_token_window(self):
    # kimi-k3 specifically is 1M context (verified against safetyculture-safetyculture-production
    # by SafetyCulture/experimental#478, 2026-08-11), distinct from the general "kimi" family
    # (K2.7 Code, Inkling), which stays at the conservative 128k default.
    assert db_mod.model_token_limits("databricks-kimi-k3") == {"context": 1_000_000, "output": 65_536}

def test_kimi_k2_7_code_keeps_the_family_default(self):
    assert db_mod.model_token_limits("databricks-kimi-k2-7-code") == {"context": 128_000, "output": 65_536}

def test_glm_output_cap_matches_the_live_workspace_measurement(self):
    # Corrected from PR#420's 25_000/200_000 (measured against a different Databricks
    # workspace) to the value independently confirmed twice against our own workspace
    # (SafetyCulture/experimental#474, 2026-08-05: "I confirmed the output cap exactly by
    # tripping the gateway's rejection").
    assert db_mod.model_token_limits("databricks-glm-5-2") == {"context": 1_000_000, "output": 65_536}

def test_longest_family_key_wins_regardless_of_dict_order(self):
    # Guards the kimi vs kimi-k3 distinction: a naive first-match-in-iteration-order lookup
    # would let the shorter "kimi" key mask "kimi-k3" depending on dict insertion order.
    assert db_mod.model_token_limits("databricks-kimi-k3")["context"] == 1_000_000
    assert db_mod.model_token_limits("databricks-kimi-k2-7-code")["context"] == 128_000
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/work/third-party/databricks/unity-gateway && python3 -m pytest tests/test_databricks.py -k "kimi_k3 or kimi_k2_7 or glm_output_cap or longest_family" -v`
Expected: FAIL — `kimi-k3` currently resolves to the generic `kimi` entry (context 128_000, not 1_000_000), and `glm` currently resolves to `{"context": 200_000, "output": 25_000}`.

- [ ] **Step 3: Fix the table and the lookup function**

```python
# src/ucode/databricks.py — replace the existing _MODEL_TOKEN_LIMITS dict and model_token_limits

# Per-family token limits (context window + max output tokens), keyed by model-id
# substring. The gateway 400s a request whose max_tokens exceeds its cap, and
# Claude Code has no way to discover a Databricks model's real context window, so
# both numbers have to come from probing the gateway rather than an API.
#
# Values are workspace-specific: the same model can have a different cap on a
# different Databricks account. Every entry below was measured directly against
# safetyculture-safetyculture-production — do not copy caps from another
# workspace's ucode fork or PR without reprobing here first (see the glm/qwen/
# gpt-oss/llama/gemma caveat below).
_MODEL_TOKEN_LIMITS: dict[str, dict[str, int]] = {
    # glm and kimi/kimi-k3/inkling: probed against safetyculture-safetyculture-production
    # twice, independently, by tripping the gateway's max_tokens rejection —
    # lukecameron/ucode@fix/oss-serving-endpoints-fallback (2026-07-16) and
    # SafetyCulture/experimental#474 (2026-08-05, GLM) / #478 (2026-08-11, Kimi K3).
    # Context windows for glm and kimi-k3 come from each endpoint's own description
    # ("supports a context length of 1M tokens"), not a guess.
    "kimi-k3": {"context": 1_000_000, "output": 65_536},
    "kimi": {"context": 128_000, "output": 65_536},
    "glm": {"context": 1_000_000, "output": 65_536},
    "inkling": {"context": 128_000, "output": 65_536},
    # qwen/gpt-oss/llama-4-maverick/gemma: from databricks/unity-gateway#420, which
    # tested a workspace other than ours. None of these families have been seen on
    # safetyculture-safetyculture-production as of 2026-09-09 (discover_oss_models
    # has never returned one) — kept as a same-cohort placeholder so a family isn't
    # left with no limit at all if one of these models is later added here, but
    # reprobe before trusting the number if that happens.
    "qwen": {"context": 262_144, "output": 25_000},
    "gpt-oss": {"context": 131_072, "output": 25_000},
    "llama-4-maverick": {"context": 1_000_000, "output": 8_192},
    "gemma": {"context": 131_072, "output": 8_192},
}


def model_token_limits(model_id: str) -> dict[str, int] | None:
    """Return ``{"context": ..., "output": ...}`` limits for ``model_id``, or None.

    Matches by family substring (e.g. any ``*glm*`` id), longest key first so a
    more specific entry (``kimi-k3``) wins over a shorter one that would otherwise
    also match (``kimi``) regardless of dict insertion order. None means the model
    has no known limits and the agent should not pin any."""
    for family in sorted(_MODEL_TOKEN_LIMITS, key=len, reverse=True):
        if family in model_id:
            return dict(_MODEL_TOKEN_LIMITS[family])
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_databricks.py -v`
Expected: PASS, including the pre-existing tests in this file (the reorder must not change any other model's resolved limits — `llama-4-maverick` must still win over a bare `llama` if one is ever added, which the new longest-key-first sort now guarantees generically instead of relying on manual dict ordering).

- [ ] **Step 5: Lint and commit**

```bash
uvx ruff check src/ucode/databricks.py tests/test_databricks.py
git add src/ucode/databricks.py tests/test_databricks.py
git commit -m "databricks: correct glm/kimi-k3 token limits to workspace-verified values

PR#420's numbers were measured against a different Databricks workspace.
SafetyCulture/experimental#474 and #478 independently confirmed glm=1M/65536
and kimi-k3=1M/65536 against safetyculture-safetyculture-production directly,
agreeing with lukecameron/ucode's original July probe. Also fixes
model_token_limits to match by longest family key, needed for kimi-k3 to win
over the shorter generic kimi entry regardless of dict order."
```

---

## Task 1: Add `newest()` and vision-capability lookup to `databricks.py`

Two small, pure helpers the shim needs and this file doesn't have yet: picking the highest-versioned model in a family (for the Opus/Sonnet tier mapping — "newest glm", "newest kimi") and knowing which models accept image input.

**Files:**
- Modify: `src/ucode/databricks.py`
- Test: `tests/test_databricks.py`

**Interfaces:**
- Consumes: nothing new
- Produces: `newest(models: list[str], family: str) -> str | None`, `supports_vision(model_id: str) -> bool`, `VISION_FAMILIES: tuple[str, ...]` — all used by Task 3 (`server.py`) and Task 5 (`claude.py` tier mapping)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_databricks.py
class TestNewest:
    def test_picks_the_highest_versioned_model_in_a_family(self):
        models = ["databricks-glm-4-6", "databricks-glm-5-2", "databricks-kimi-k2-7-code"]
        assert db_mod.newest(models, "glm") == "databricks-glm-5-2"

    def test_returns_none_when_the_family_has_no_match(self):
        assert db_mod.newest(["databricks-glm-5-2"], "qwen") is None


class TestSupportsVision:
    def test_kimi_k3_supports_vision(self):
        assert db_mod.supports_vision("databricks-kimi-k3") is True

    def test_other_oss_models_do_not(self):
        assert db_mod.supports_vision("databricks-kimi-k2-7-code") is False
        assert db_mod.supports_vision("databricks-glm-5-2") is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_databricks.py -k "TestNewest or TestSupportsVision" -v`
Expected: FAIL with `AttributeError: module 'ucode.databricks' has no attribute 'newest'`.

- [ ] **Step 3: Implement**

```python
# src/ucode/databricks.py — add near model_token_limits

# Models with native image input on the mlflow chat-completions route. Everything
# else gets images stripped to a text placeholder rather than sent and wasted or
# rejected. Extend this if another model gains vision (SafetyCulture/experimental#478).
VISION_FAMILIES: tuple[str, ...] = ("kimi-k3",)


def supports_vision(model_id: str) -> bool:
    return any(family in model_id for family in VISION_FAMILIES)


def newest(models: list[str], family: str) -> str | None:
    """Pick the highest-versioned model in `family` from a discovered model list.

    Ids embed their version in the name (`glm-5-2`, `kimi-k3`), so a reverse
    lexicographic sort is enough to prefer the newest — same approach as
    SafetyCulture/experimental#474's `dbxclaude.databricks.newest`."""
    matches = sorted((m for m in models if family in m), reverse=True)
    return matches[0] if matches else None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_databricks.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
uvx ruff check src/ucode/databricks.py tests/test_databricks.py
git add src/ucode/databricks.py tests/test_databricks.py
git commit -m "databricks: add newest() and supports_vision() for OSS tier mapping"
```

---

## Task 2: Port `translate.py` — pure Anthropic↔OpenAI translation

This file has no dependency on the rest of the shim (no I/O, no globals — verified by reading `SafetyCulture/experimental#474`'s `dbxclaude/translate.py`), so it ports essentially verbatim. Copy it in whole; only the module path changes.

**Files:**
- Create: `src/ucode/agents/claude_oss/__init__.py` (empty — marks the package)
- Create: `src/ucode/agents/claude_oss/translate.py`
- Test: `tests/test_claude_oss_translate.py`

**Interfaces:**
- Produces: `anthropic_to_openai(request, *, model, max_output=None, allow_images=True) -> dict`, `openai_to_anthropic(response, *, model, reasoning="thinking") -> dict`, `StreamTranslator` (with `.feed(chunk)` and `.finish()`, each yielding `(event_name: str, payload: dict)` pairs), `error_body(status, message) -> dict`, `count_tokens(request) -> int`, `REASONING_MODES = ("thinking", "text", "drop")` — all consumed by Task 3 (`server.py`)

- [ ] **Step 1: Copy the source file**

```bash
mkdir -p src/ucode/agents/claude_oss
touch src/ucode/agents/claude_oss/__init__.py
cp /tmp/experimental-ref/hkf57/dbx-claude-shim/dbxclaude/translate.py src/ucode/agents/claude_oss/translate.py
```

Update only the module docstring's first line to name this codebase instead of the standalone tool (the rest of the docstring — the two properties of the gateway shape, the direction-of-travel diagram — stays as-is, it's still accurate):

```python
"""Pure translation between the Anthropic Messages API and OpenAI chat completions.

Ported from SafetyCulture/experimental#474/#478 (`dbx-claude-shim`), which
proved this translation against safetyculture-safetyculture-production before
this port. No I/O, no network, no globals — a function of its arguments only,
so it is unit-tested without a Databricks workspace.
...
```

- [ ] **Step 2: Copy the test file, adjusting the import**

```bash
cp /tmp/experimental-ref/hkf57/dbx-claude-shim/tests/test_translate.py tests/test_claude_oss_translate.py
```

Change the import line from `from dbxclaude import translate` to:
```python
from ucode.agents.claude_oss import translate
```

- [ ] **Step 3: Run the ported tests**

Run: `python3 -m pytest tests/test_claude_oss_translate.py -v`
Expected: PASS on all of them unchanged — this file has zero dependency on anything shim-specific, so a passing run here is a straight confirmation the copy was faithful, not new behavior being verified for the first time.

- [ ] **Step 4: Lint and commit**

```bash
uvx ruff check src/ucode/agents/claude_oss/translate.py tests/test_claude_oss_translate.py
git add src/ucode/agents/claude_oss/__init__.py src/ucode/agents/claude_oss/translate.py tests/test_claude_oss_translate.py
git commit -m "claude_oss: port translate.py from SafetyCulture/experimental#474/#478

Pure Anthropic<->OpenAI chat-completions translation, ported near-verbatim.
Verified against safetyculture-safetyculture-production by the source PRs:
streaming, tool calls, vision (Kimi K3), reasoning-as-thinking-blocks, the
gateway's strict unknown-field validator, and a stream ending without
finish_reason. Tests ported unchanged from tests/test_translate.py."
```

---

## Task 3: Build `server.py` — the loopback shim, wired to this repo's own primitives

Unlike `translate.py`, this file does NOT port verbatim: the source PR's `server.py` pairs with its own standalone `dbxclaude/databricks.py` (model discovery, token cache, family tables). This repo already has all of that — `discover_oss_models`, `model_token_limits`, `supports_vision` (Task 1), and a battle-tested background-refreshing `TokenCache` in `ucode.gateway_proxy` (used today by Claude's relayed-auth proxy) — so `server.py` is *rebuilt* against those instead of duplicating a second copy that can drift out of sync with the first (which is exactly how the Task 0 bug happened: two tables measuring the same thing, disagreeing, because nothing wired them together).

**Files:**
- Create: `src/ucode/agents/claude_oss/server.py`
- Test: `tests/test_claude_oss_server.py`

**Interfaces:**
- Consumes: `ucode.databricks.discover_oss_models(workspace, token) -> list[str]`, `ucode.databricks.model_token_limits(model_id) -> dict|None`, `ucode.databricks.supports_vision(model_id) -> bool`, `ucode.databricks.newest(models, family) -> str|None`, `ucode.gateway_proxy.TokenCache(workspace, profile, *, force_refresh_near_expiry=False)` (`.token` property, `.run_refresher()`, `.stop()`), `translate.anthropic_to_openai`/`translate.openai_to_anthropic`/`translate.StreamTranslator`/`translate.error_body`/`translate.count_tokens` (Task 2)
- Produces: `ModelRouter(catalogue: list[str], default: str)` with `.resolve(requested) -> str`, `make_server(host, tokens, router, *, port=0, allow_images=True, reasoning="thinking") -> ThreadingHTTPServer` — both consumed by Task 5 (`claude.py`'s `_launch_oss_shim`)

- [ ] **Step 1: Write the failing tests for `ModelRouter`**

```python
# tests/test_claude_oss_server.py
import unittest

from ucode.agents.claude_oss.server import ModelRouter


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
        router = ModelRouter(["databricks-kimi-k2-7-code", "databricks-kimi-k3"], "databricks-kimi-k3")
        self.assertEqual(router.resolve("kimi"), "databricks-kimi-k3")

    def test_empty_or_non_string_falls_back_to_default(self):
        router = ModelRouter(["databricks-glm-5-2"], "databricks-glm-5-2")
        self.assertEqual(router.resolve(None), "databricks-glm-5-2")
        self.assertEqual(router.resolve(123), "databricks-glm-5-2")
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_claude_oss_server.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ucode.agents.claude_oss.server'`

- [ ] **Step 3: Implement `ModelRouter` and the module skeleton**

```python
# src/ucode/agents/claude_oss/server.py
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

from typing import Any

from ucode.databricks import newest

from . import translate

OSS_ROUTE = "/ai-gateway/mlflow/v1/chat/completions"


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
```

- [ ] **Step 4: Run to verify `ModelRouter` tests pass**

Run: `python3 -m pytest tests/test_claude_oss_server.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Write the failing tests for the HTTP server, against a stub gateway**

This mirrors `SafetyCulture/experimental#474`'s `tests/test_server.py` approach: run the real `ThreadingHTTPServer` on a real socket, point its `_upstream` calls at a second, local stub HTTP server standing in for Databricks, and assert on the wire-level Anthropic responses.

```python
# tests/test_claude_oss_server.py (continued)
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

import urllib.request

from ucode.agents.claude_oss.server import ModelRouter, make_server
from ucode.gateway_proxy import TokenCache


class _StubGateway(BaseHTTPRequestHandler):
    """Stands in for /ai-gateway/mlflow/v1/chat/completions."""

    response_body: dict = {}

    def do_POST(self):  # noqa: N802
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
        self.gateway = HTTPServer(("127.0.0.1", 0), _StubGateway)
        threading.Thread(target=self.gateway.serve_forever, daemon=True).start()
        gateway_host = f"http://127.0.0.1:{self.gateway.server_address[1]}"

        # A TokenCache pointed at a fake workspace never calls `databricks auth
        # token` here because get_databricks_token honours DATABRICKS_BEARER
        # first (see ucode.databricks.get_token precedence, mirrored by
        # get_databricks_token) — set it so no live CLI/workspace is needed.
        import os

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
            {"model": "databricks-glm-5-2", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]},
        )
        sent_auth = self.gateway.RequestHandlerClass  # placeholder to keep flake happy
        # The stub records the last request's headers via BaseHTTPRequestHandler.headers,
        # captured on the instance during do_POST — assert the gateway received the
        # shim's own bearer, not the "Bearer whatever-claude-code-sent" the client sent.
        # (Captured explicitly below rather than trusting instance state across threads.)

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
```

Note on `test_the_clients_own_authorization_header_is_never_forwarded`: write this properly against a header captured by `_StubGateway.do_POST` (store `self.__class__.last_headers = dict(self.headers)` there) rather than the placeholder sketched above — the "Global Constraints" security rule for this feature is exactly what this test exists to prove, so it must actually assert on the captured header, not just call the endpoint. Fill this in for real during Step 5, not left as scaffolding.

- [ ] **Step 6: Run to verify failure**

Run: `python3 -m pytest tests/test_claude_oss_server.py -v`
Expected: FAIL — `ImportError: cannot import name 'make_server'`

- [ ] **Step 7: Implement the `Handler` and `make_server`**

```python
# src/ucode/agents/claude_oss/server.py (continued — add after ModelRouter)

import json
import os
import socket
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ucode.databricks import model_token_limits, supports_vision

MAX_BODY_BYTES = 64 * 1024 * 1024


class _Logger:
    """Append-only JSONL tracing, enabled by CLAUDE_OSS_SHIM_LOG.

    Off by default. Records translated request/response bodies only — never
    the bearer token — which is what makes a translation bug diagnosable."""

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
                with open(self.path, "a", encoding="utf-8") as handle:
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

    def log_message(self, fmt: str, *args: object) -> None:
        return  # never log request lines; they can carry paths

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: int, message: str) -> None:
        self.trace("error", {"status": status, "message": message})
        try:
            self._send_json(status, translate.error_body(status, message))
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
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self._send_error(400, "Request body was not valid JSON.")
            return None

    def _upstream(self, payload: dict, *, stream: bool):
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

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") in ("/health", ""):
            self._send_json(200, {"status": "ok", "models": self.router.catalogue})
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
        self._send_json(200, {"input_tokens": translate.count_tokens(body)})

    def _messages(self) -> None:
        body = self._read_body()
        if body is None:
            return
        model = self.router.resolve(body.get("model"))
        payload = translate.anthropic_to_openai(
            body,
            model=model,
            max_output=_max_output_for(model),
            allow_images=self.allow_images and supports_vision(model),
        )
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
        self._send_json(200, translate.openai_to_anthropic(upstream, model=model, reasoning=self.reasoning))

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
        block = f"event: {name}\ndata: {json.dumps(payload)}\n\n".encode("utf-8")
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
        return ThreadingHTTPServer(("127.0.0.1", port), handler)
    except OSError:
        # Requested port is taken (a stale shim from a killed session still
        # holding the socket) — let the OS pick one; the caller reads it back.
        return ThreadingHTTPServer(("127.0.0.1", 0), handler)
```

- [ ] **Step 8: Fill in the real header-capture assertion from Step 5's note, then run all tests**

Run: `python3 -m pytest tests/test_claude_oss_server.py -v`
Expected: PASS on all tests, including the never-forward-the-client-header test now asserting on a real captured header.

- [ ] **Step 9: Lint and commit**

```bash
uvx ruff check src/ucode/agents/claude_oss/server.py tests/test_claude_oss_server.py
git add src/ucode/agents/claude_oss/server.py tests/test_claude_oss_server.py
git commit -m "claude_oss: add the loopback shim server, wired to this repo's own
databricks.py and gateway_proxy.TokenCache

Rebuilt (not ported verbatim) from SafetyCulture/experimental#474/#478's
server.py: same HTTP handling and streaming relay, but sourced from
discover_oss_models/model_token_limits/supports_vision (Task 0-1) and
gateway_proxy.TokenCache instead of a second, independent copy of each -
avoiding the exact kind of two-tables-disagree bug Task 0 just fixed."
```

---

## Task 4: Extend `render_overlay` and `write_tool_config` with an OSS-shim base URL

Two new parameters threaded through the existing `relayed`/`provider`/`provider_models` machinery — no new code paths invented, just a third `base_url` source and a shared branch for pinning `ANTHROPIC_DEFAULT_*_MODEL` from a plain id dict.

**Files:**
- Modify: `src/ucode/agents/claude.py` (`render_overlay`, `write_tool_config`)
- Test: `tests/test_agent_claude.py`

**Interfaces:**
- Consumes: nothing new
- Produces: `render_overlay(..., oss_shim_base_url: str | None = None)`, `write_tool_config(..., oss_shim_base_url: str | None = None)` — both consumed by Task 5

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_claude.py
def test_render_overlay_points_at_the_oss_shim_when_given_a_base_url(self):
    overlay, keys = claude.render_overlay(
        "https://safetyculture-safetyculture-production.cloud.databricks.com",
        None,
        {},
        provider_models={"opus": "databricks-glm-5-2[1m]", "sonnet": "databricks-kimi-k3[1m]"},
        oss_shim_base_url="http://127.0.0.1:54321",
    )
    assert overlay["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:54321"
    assert overlay["env"]["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "databricks-glm-5-2[1m]"
    assert overlay["env"]["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "databricks-kimi-k3[1m]"
    # The shim authenticates to Databricks itself; Claude Code must not be
    # told to run a gateway apiKeyHelper that would try to reach it directly.
    assert "apiKeyHelper" not in overlay
    assert ["apiKeyHelper"] not in keys
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_agent_claude.py -k oss_shim -v`
Expected: FAIL — `TypeError: render_overlay() got an unexpected keyword argument 'oss_shim_base_url'`

- [ ] **Step 3: Implement**

In `src/ucode/agents/claude.py`, extend the `render_overlay` signature and its two relevant branches:

```python
def render_overlay(
    workspace: str,
    model: str | None,
    claude_models: dict[str, str] | None = None,
    disable_web_search: bool = False,
    profile: str | None = None,
    use_pat: bool = False,
    provider: str | None = None,
    provider_models: dict[str, str] | None = None,
    fable_enabled: bool = False,
    relayed: bool = False,
    relayed_base_url: str | None = None,
    route_root_model: str | None = None,
    custom_model: str | None = None,
    oss_shim_base_url: str | None = None,
) -> tuple[dict, list[list[str]]]:
    """... (existing docstring, plus:)

    When `oss_shim_base_url` is set, the workspace has no Claude models but does
    have OSS chat models (GLM, Kimi, ...) — `ucode.agents.claude_oss` is running
    a local Anthropic<->OpenAI translation shim there (see `_launch_oss_shim`).
    Like `relayed`, no `apiKeyHelper` is written: the shim authenticates to
    Databricks with its own token and discards whatever Claude Code sends, so a
    gateway apiKeyHelper here would be pointed at nothing. `provider_models`
    carries the OSS ids to pin per Claude Code tier, same mechanism a
    Bedrock-backed Model Provider Service already uses."""
    if relayed:
        if not relayed_base_url:
            raise RuntimeError("Relayed launch requires a proxy base URL.")
        base_url = relayed_base_url
    elif oss_shim_base_url:
        base_url = oss_shim_base_url
    else:
        base_url = build_tool_base_url("claude", workspace)
    ...
    if route_root_model:
        env["ANTHROPIC_MODEL"] = route_root_model
    elif provider_models and (provider or oss_shim_base_url):
        if provider_models.get("opus"):
            env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = provider_models["opus"]
        if provider_models.get("sonnet"):
            env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = provider_models["sonnet"]
        if provider_models.get("haiku"):
            env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = provider_models["haiku"]
    elif claude_models and not provider:
        ...  # unchanged
    ...
    overlay: dict = {"env": env}
    if relayed or oss_shim_base_url:
        keys = [["env", k] for k in env]
    else:
        overlay["apiKeyHelper"] = build_auth_shell_command(workspace, profile, use_pat=use_pat)
        keys = [["apiKeyHelper"]] + [["env", k] for k in env]
    ...
```

Only the three marked spots change: the `base_url` if/elif chain gets one new branch, the `provider_models` branch's guard becomes `provider_models and (provider or oss_shim_base_url)` instead of `provider and provider_models`, and the `apiKeyHelper` guard becomes `relayed or oss_shim_base_url` instead of just `relayed`. Everything else in the function is untouched.

Then thread the same parameter through `write_tool_config`:

```python
def write_tool_config(
    state: dict,
    model: str | None,
    provider: str | None = None,
    provider_models: dict[str, str] | None = None,
    relayed: bool = False,
    route_root_model: str | None = None,
    custom_model: str | None = None,
    coding_agent_config_defaults: dict[str, str] | None = None,
    oss_shim_base_url: str | None = None,
) -> dict:
    ...
    overlay, managed_keys = render_overlay(
        state["workspace"],
        model,
        state.get("claude_models") or {},
        disable_web_search=web_search_model is not None,
        profile=state.get("profile"),
        use_pat=bool(state.get("use_pat")),
        provider=provider,
        provider_models=provider_models,
        fable_enabled=bool(state.get("fable_enabled")),
        relayed=relayed,
        relayed_base_url=relayed_base_url,
        route_root_model=route_root_model,
        custom_model=custom_model,
        oss_shim_base_url=oss_shim_base_url,
    )
    ...
```

- [ ] **Step 4: Run to verify the new test passes and nothing else broke**

Run: `python3 -m pytest tests/test_agent_claude.py -v`
Expected: PASS on all tests, including every existing `render_overlay`/`write_tool_config` test (the `provider_models` guard change must not alter behavior for the real Bedrock-backed-provider case, where `provider` is set and `oss_shim_base_url` is `None` — verify by re-reading the existing provider tests in this file rather than assuming).

- [ ] **Step 5: Lint and commit**

```bash
uvx ruff check src/ucode/agents/claude.py tests/test_agent_claude.py
git add src/ucode/agents/claude.py tests/test_agent_claude.py
git commit -m "claude: thread an oss_shim_base_url through render_overlay/write_tool_config

Reuses the existing relayed/provider_models machinery rather than adding a
parallel config path: a shim base URL is a third base_url source (alongside
the direct gateway URL and the relayed proxy), and pins ANTHROPIC_DEFAULT_*
the same way a Bedrock-backed Model Provider Service already does."
```

---

## Task 5: `_launch_oss_shim` — start the shim and run Claude Code against it

Mirrors `_launch_relayed` almost exactly: both need a local companion server to outlive the Claude Code subprocess, so both spawn-and-wait instead of using `exec_or_spawn` (which does a POSIX `execve` that replaces ucode's own process — and any thread it was running — the instant Claude Code starts).

**Files:**
- Modify: `src/ucode/agents/claude.py` (`launch`, plus new `_launch_oss_shim`)
- Test: `tests/test_agent_claude.py`

**Interfaces:**
- Consumes: `claude_oss.server.ModelRouter`, `claude_oss.server.make_server` (Task 3), `gateway_proxy.TokenCache` (existing), `_maybe_add_1m_suffix` (existing), `_build_claude_argv` (existing), `newest` (Task 1)
- Produces: `_launch_oss_shim(state, binary, tool_args) -> NoReturn` (raises `SystemExit`)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_claude.py
def test_launch_oss_shim_starts_the_server_and_runs_claude_against_it(self, monkeypatch, tmp_path):
    state = {
        "workspace": "https://example.cloud.databricks.com",
        "oss_models": ["databricks-glm-5-2", "databricks-kimi-k3"],
    }
    started = {}

    class FakeProc:
        returncode = 0

        def wait(self):
            return 0

    def fake_popen(argv, **kwargs):
        started["argv"] = argv
        started["env"] = kwargs.get("env")
        return FakeProc()

    monkeypatch.setattr(claude.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(claude, "get_databricks_token", lambda *a, **k: "fake-token")
    monkeypatch.setattr(claude, "write_tool_config", lambda *a, **k: {})

    with pytest.raises(SystemExit) as exc:
        claude._launch_oss_shim(state, "claude", [])

    assert exc.value.code == 0
    assert started["env"]["ANTHROPIC_BASE_URL"].startswith("http://127.0.0.1:")
    assert started["env"]["ANTHROPIC_DEFAULT_OPUS_MODEL"].startswith("databricks-glm-5-2")
    assert started["env"]["ANTHROPIC_DEFAULT_SONNET_MODEL"].startswith("databricks-kimi-k3")
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_agent_claude.py -k oss_shim_starts -v`
Expected: FAIL — `AttributeError: module 'ucode.agents.claude' has no attribute '_launch_oss_shim'`

- [ ] **Step 3: Implement**

```python
# src/ucode/agents/claude.py — add near _launch_relayed

from ucode.agents.claude_oss.server import ModelRouter, make_server
from ucode.databricks import newest


def _launch_oss_shim(state: dict, binary: str, tool_args: list[str]) -> None:
    """OSS-model launch: the workspace has no Claude models but does have OSS
    chat models (GLM, Kimi, ...) on the mlflow gateway route. Start the local
    Anthropic<->OpenAI translation shim (`ucode.agents.claude_oss`), then run
    Claude Code alongside it — the shim must outlive the exec, so we
    spawn-and-wait rather than replacing the process, same as `_launch_relayed`.
    """
    workspace = state["workspace"]
    profile = state.get("profile")
    oss_models: list[str] = state.get("oss_models") or []

    glm = newest(oss_models, "glm") or oss_models[0]
    kimi = newest(oss_models, "kimi") or glm
    default = glm

    tokens = TokenCache(workspace, profile)
    router = ModelRouter(oss_models, default)
    server = make_server(workspace, tokens, router)
    bound_port = server.server_address[1]

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    refresher = threading.Thread(target=tokens.run_refresher, daemon=True)
    refresher.start()

    write_tool_config(
        state,
        None,
        provider_models={
            "opus": _maybe_add_1m_suffix(glm),
            "sonnet": _maybe_add_1m_suffix(kimi),
            "haiku": _maybe_add_1m_suffix(glm),
        },
        oss_shim_base_url=f"http://127.0.0.1:{bound_port}",
    )

    proc = subprocess.Popen(_build_claude_argv(binary, tool_args))
    try:
        returncode = proc.wait()
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
        returncode = proc.wait()
    finally:
        tokens.stop()
        server.shutdown()
        server.server_close()
    raise SystemExit(returncode)
```

Add the import for `TokenCache` alongside the existing `gateway_proxy` import at the top of the file (it's already imported as a module: `from ucode import gateway_proxy` — use `gateway_proxy.TokenCache(...)` in the implementation above instead of a bare `TokenCache` name, to match how `_launch_relayed` references it, and drop the redundant `from ucode.gateway_proxy import TokenCache` line from the snippet above).

- [ ] **Step 4: Wire it into `launch()`**

```python
# src/ucode/agents/claude.py — in launch(), before the exec_or_spawn call
def launch(
    state: dict,
    tool_args: list[str],
    *,
    options: LaunchOptions,
) -> None:
    binary = SPEC["binary"]
    workspace = state.get("workspace")
    if state.get("claude_relayed"):
        _launch_relayed(state, binary, tool_args)
        return
    if state.get("claude_oss_fallback"):
        _launch_oss_shim(state, binary, tool_args)
        return
    ...  # unchanged from here down
```

`claude_oss_fallback` is set in Task 6, at the point where discovery already knows whether real Claude models exist.

- [ ] **Step 5: Run to verify the test passes**

Run: `python3 -m pytest tests/test_agent_claude.py -v`
Expected: PASS

- [ ] **Step 6: Lint and commit**

```bash
uvx ruff check src/ucode/agents/claude.py tests/test_agent_claude.py
git add src/ucode/agents/claude.py tests/test_agent_claude.py
git commit -m "claude: add _launch_oss_shim, mirroring _launch_relayed's spawn-and-wait

Claude Code's normal launch path calls exec_or_spawn, which does a POSIX
execve that replaces ucode's own process (and any thread it was running) the
instant Claude Code starts. The OSS shim server has to keep running for the
whole session, so this launch mode spawns Claude Code as a child and waits,
exactly like the existing relayed-auth proxy already does, instead of
exec-ing."
```

---

## Task 6: Wire discovery to the launch decision

The pieces exist (Task 0's `discover_oss_models`, already merged on this branch; Tasks 4-5's shim launch); this task is the one line of policy connecting them: no Claude models + some OSS models = fall back, recorded once at configure time so `launch()` (which runs on every subsequent `ucode claude` without re-discovering) knows which mode to use.

**Files:**
- Modify: `src/ucode/cli.py` (`configure_shared_state`, `configure_workspace_command` or wherever `state["claude_oss_fallback"]` should be set — locate the exact spot per Step 1 below)
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `state["claude_models"]`, `state["oss_models"]` (both already populated by existing discovery)
- Produces: `state["claude_oss_fallback"]: bool`, consumed by Task 5's `launch()` check

- [ ] **Step 1: Locate the exact insertion point**

Read `configure_shared_state` in `src/ucode/cli.py` (the function that already sets `claude_models`/`oss_models` via `discover_model_services`/`discover_claude_models`/`discover_oss_models`, per the `want_claude`/`want_oss` blocks). Confirm where `state["claude_models"]` and `state["oss_models"]` are both set before `save_state(state)` is called, and insert the fallback flag there — this keeps the decision co-located with the data it depends on, rather than recomputed later from state that might have changed.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_cli.py
def test_configure_sets_oss_fallback_when_no_claude_models_but_oss_models_exist(self, monkeypatch):
    monkeypatch.setattr(cli, "discover_model_services", lambda *a, **k: ({}, [], [], [], None))
    monkeypatch.setattr(cli, "discover_claude_models", lambda *a, **k: ({}, None))
    monkeypatch.setattr(cli, "discover_oss_models", lambda *a, **k: (["databricks-glm-5-2"], None))
    state = cli.configure_shared_state("https://example.cloud.databricks.com", tools={"claude"})
    assert state["claude_oss_fallback"] is True

def test_configure_does_not_set_oss_fallback_when_claude_models_exist(self, monkeypatch):
    monkeypatch.setattr(cli, "discover_model_services", lambda *a, **k: ({}, [], [], [], None))
    monkeypatch.setattr(cli, "discover_claude_models", lambda *a, **k: ({"opus": "claude-opus-4-8"}, None))
    monkeypatch.setattr(cli, "discover_oss_models", lambda *a, **k: (["databricks-glm-5-2"], None))
    state = cli.configure_shared_state("https://example.cloud.databricks.com", tools={"claude"})
    assert state.get("claude_oss_fallback") is not True

def test_configure_does_not_set_oss_fallback_when_no_oss_models_either(self, monkeypatch):
    monkeypatch.setattr(cli, "discover_model_services", lambda *a, **k: ({}, [], [], [], None))
    monkeypatch.setattr(cli, "discover_claude_models", lambda *a, **k: ({}, None))
    monkeypatch.setattr(cli, "discover_oss_models", lambda *a, **k: ([], "no models"))
    state = cli.configure_shared_state("https://example.cloud.databricks.com", tools={"claude"})
    assert state.get("claude_oss_fallback") is not True
```

Adjust the monkeypatch targets and `configure_shared_state` call signature to match exactly what Step 1 found — the mock returns above assume the shape already read earlier in this session (`discover_model_services` returns a 5-tuple, `discover_claude_models`/`discover_oss_models` return `(result, reason)`); confirm against the actual current signatures before writing this, since they may have shifted since.

- [ ] **Step 3: Run to verify failure**

Run: `python3 -m pytest tests/test_cli.py -k oss_fallback -v`
Expected: FAIL — `state["claude_oss_fallback"]` raises `KeyError` or is falsy when it should be `True`.

- [ ] **Step 4: Implement**

At the insertion point found in Step 1, immediately after `claude_models`/`oss_models` are both known:

```python
if want_claude:
    state["claude_oss_fallback"] = not claude_models and bool(oss_models)
```

- [ ] **Step 5: Run to verify tests pass**

Run: `python3 -m pytest tests/test_cli.py -v`
Expected: PASS, and no existing `configure_shared_state`/`configure_workspace_command` test regresses (a Claude-models-available workspace must configure exactly as it does today — re-run the full file, not just the new tests).

- [ ] **Step 6: Confirm `ENABLE_TOOL_SEARCH` needs no separate change**

No code change here — this is a verification step. `render_overlay`'s base `env` dict already sets `"ENABLE_TOOL_SEARCH": "true"` unconditionally (this predates this plan), so PR #475's fix is already in effect for OSS-shim launches with zero extra work. Confirm this by reading the current `render_overlay` and noting it in the commit message below, so a future reader doesn't wonder why Task 6 doesn't touch tool search.

- [ ] **Step 7: Lint and commit**

```bash
uvx ruff check src/ucode/cli.py tests/test_cli.py
git add src/ucode/cli.py tests/test_cli.py
git commit -m "cli: set claude_oss_fallback when discovery finds OSS models but no Claude models

Connects Task 0's discover_oss_models to Task 5's _launch_oss_shim: 'ucode
claude' on a workspace with GLM/Kimi but no Claude models now configures for
the translation shim instead of leaving claude_models empty and failing at
launch with 'AI Gateway returned no Claude model ids'.

No change needed for tool search: render_overlay's base env already sets
ENABLE_TOOL_SEARCH=true unconditionally, so PR#475's fix already applies here."
```

---

## Task 7: An escape hatch, and `ug status`/`ug doctor` visibility

Automatic fallback is the chosen behavior, but a user who explicitly wants the "not available" error instead of a silently different model (e.g. scripting against a specific Claude version) needs a way to say so, and anyone looking at `ug status` after a fallback launch needs to be able to tell it happened.

**Files:**
- Modify: `src/ucode/cli.py` (a `--no-oss-fallback` flag on the `claude`/`configure` commands, threaded into `configure_shared_state`)
- Modify: `src/ucode/agents/__init__.py` or wherever `ug status`'s per-tool summary is built (`_provider_summary`, seen earlier reporting `(Provider: ...)` per tool)
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: Task 6's `state["claude_oss_fallback"]`
- Produces: a `--no-oss-fallback` CLI flag that forces `state["claude_oss_fallback"] = False` regardless of discovery; `_provider_summary` reporting e.g. `"Databricks (OSS shim: GLM 5.2 / Kimi K3)"` instead of the generic provider label when the flag is set

- [ ] **Step 1: Write the failing test for the flag**

```python
# tests/test_cli.py
def test_no_oss_fallback_flag_forces_the_flag_off_even_with_oss_models(self, monkeypatch):
    monkeypatch.setattr(cli, "discover_model_services", lambda *a, **k: ({}, [], [], [], None))
    monkeypatch.setattr(cli, "discover_claude_models", lambda *a, **k: ({}, None))
    monkeypatch.setattr(cli, "discover_oss_models", lambda *a, **k: (["databricks-glm-5-2"], None))
    state = cli.configure_shared_state(
        "https://example.cloud.databricks.com", tools={"claude"}, no_oss_fallback=True
    )
    assert state.get("claude_oss_fallback") is not True
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_cli.py -k no_oss_fallback_flag -v`
Expected: FAIL — `TypeError: configure_shared_state() got an unexpected keyword argument 'no_oss_fallback'`

- [ ] **Step 3: Implement the flag threading**

Add a `no_oss_fallback: bool = False` parameter to `configure_shared_state` (and up through whatever calls it from the `configure`/`claude` Typer commands — find the exact call chain from `ug claude --no-oss-fallback` down to `configure_shared_state`, matching how existing per-launch flags like `--enable-fable` are threaded, per `fable_enabled` in `render_overlay`'s signature seen in Task 4). At the Task 6 insertion point:

```python
if want_claude:
    state["claude_oss_fallback"] = not no_oss_fallback and not claude_models and bool(oss_models)
```

Register `--no-oss-fallback` on the `claude` subcommand's argparse/Typer definition, following the exact pattern used for the existing `--enable-fable`/`--disable-fable` flags (`ucode configure --enable-fable`).

- [ ] **Step 4: Run to verify it passes**

Run: `python3 -m pytest tests/test_cli.py -v`
Expected: PASS

- [ ] **Step 5: Report the mode in `ug status`**

Find `_provider_summary` (referenced in the `configure_workspace_command` panel: `f"[dim](Provider: {_provider_summary(tool_name, state)})[/dim]"`). Add a branch: when `state.get("claude_oss_fallback")`, return a string naming the two active models, e.g.:

```python
if tool_name == "claude" and state.get("claude_oss_fallback"):
    oss_models = state.get("oss_models") or []
    glm = newest(oss_models, "glm")
    kimi = newest(oss_models, "kimi")
    names = ", ".join(m for m in (glm, kimi) if m)
    return f"Databricks OSS shim ({names})" if names else "Databricks OSS shim"
```

Write the matching test alongside whatever test file already covers `_provider_summary` for the other providers (`provider`/`relayed` branches), following that file's existing pattern rather than inventing a new one.

- [ ] **Step 6: Lint and commit**

```bash
uvx ruff check src/ucode/cli.py
git add src/ucode/cli.py tests/test_cli.py
git commit -m "cli: add --no-oss-fallback escape hatch and report the shim in ug status"
```

---

## Task 8: `ucode claude --probe` — port the gateway diagnostic

`SafetyCulture/experimental#474`'s `probe` subcommand is the fastest way to answer "did the workspace's Anthropic-dialect route just start accepting this OSS model directly?" (at which point this whole shim becomes unnecessary for that model) and "is translation actually required right now?" — worth keeping as a diagnostic exposed the same way `ug doctor` is.

**Files:**
- Create: `src/ucode/agents/claude_oss/probe.py`
- Modify: `src/ucode/cli.py` (a `probe` subcommand under `claude`, or a top-level `ucode claude --probe` flag — match whichever pattern `ug doctor` already uses for a diagnostic-only subcommand)
- Test: `tests/test_claude_oss_probe.py`

**Interfaces:**
- Consumes: `ucode.databricks.get_databricks_token`, `ucode.databricks.discover_oss_models`, `ucode.databricks.model_token_limits`
- Produces: `run_probe(workspace: str, token: str, model: str | None) -> int` (return code, 0 if the probe completed regardless of pass/fail — matches `SafetyCulture/experimental#474`'s `cmd_probe` return convention)

- [ ] **Step 1: Port the core request/report helpers**

Copy `_post`/`_report` from `SafetyCulture/experimental#474`'s `dbxclaude/__main__.py` (lines implementing the raw `urllib` POST + PASS/FAIL reporting) into `probe.py` unchanged — these are generic HTTP diagnostic helpers with no dependency on the rest of that file.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_claude_oss_probe.py
from unittest.mock import patch

from ucode.agents.claude_oss.probe import run_probe


def test_probe_reports_translation_required_when_anthropic_route_rejects_oss_model():
    responses = iter(
        [
            (400, '{"message": "API type not supported"}'),  # anthropic route rejects
            (200, '{"id": "x", "choices": [{"message": {"content": "ok"}}]}'),  # oss non-streaming
            (200, "data: [DONE]\n"),  # oss streaming
            (200, '{"choices": [{"message": {"tool_calls": [{}]}}]}'),  # tool calling
            (400, '{"message": "max_tokens too large"}'),  # info-only oversized check
        ]
    )
    with patch("ucode.agents.claude_oss.probe._post", side_effect=lambda *a, **k: next(responses)):
        code = run_probe("https://example.cloud.databricks.com", "token", "databricks-glm-5-2")
    assert code == 0
```

- [ ] **Step 3: Run to verify failure**

Run: `python3 -m pytest tests/test_claude_oss_probe.py -v`
Expected: FAIL — module does not exist yet.

- [ ] **Step 4: Implement `run_probe`**

Port `cmd_probe` from `SafetyCulture/experimental#474`'s `dbxclaude/__main__.py`, renamed to `run_probe(workspace, token, model)`, dropping the `argparse.Namespace` parameter and the `_context`/`_pick_default` calls it used to resolve `host`/`profile`/`models` itself — those are replaced by this repo's own `resolve_host`/`discover_oss_models` at the CLI call site (Step 5), so `run_probe` receives an already-resolved `workspace`, `token`, and `model` directly.

- [ ] **Step 5: Wire the CLI command**

Add a `probe` action to the `claude` subcommand following whichever existing pattern `ug doctor` or `ug usage` uses for a standalone diagnostic command (both take no config-mutating side effects, matching what `probe` should do here too — read-only, never writes state or settings).

- [ ] **Step 6: Run to verify it passes, lint, and commit**

```bash
python3 -m pytest tests/test_claude_oss_probe.py -v
uvx ruff check src/ucode/agents/claude_oss/probe.py tests/test_claude_oss_probe.py
git add src/ucode/agents/claude_oss/probe.py tests/test_claude_oss_probe.py src/ucode/cli.py
git commit -m "claude_oss: port the gateway probe from SafetyCulture/experimental#474

Diagnostic only, no state changes: checks whether the Anthropic-dialect route
has started accepting OSS model ids directly (which would make this whole
shim unnecessary for that model) and confirms the mlflow route's actual
non-streaming/streaming/tool-calling/max_tokens behavior."
```

---

## Task 9: End-to-end verification against the real workspace

Every prior task's tests run against stubs or mocks. This task is the one the source PRs themselves did before merging (see each PR body's "Verified against the production workspace" table) — it cannot be automated into a unit test because it needs live Databricks access, but it is what actually proves the port works, and it's the task that should block calling this plan done.

**Files:** none (manual verification against the built branch)

- [ ] **Step 1: Reinstall from the branch**

```bash
uv tool install --reinstall ~/work/third-party/databricks/unity-gateway
ug --version   # confirm the new commit hash
```

- [ ] **Step 2: Configure against a workspace known to have no Claude models but real OSS models**

```bash
ug configure --agents claude
```
Expected: `ug status` reports the Databricks OSS shim provider (Task 7), not a hard failure.

- [ ] **Step 3: Run the actual verification checklist from the source PRs**

Reproduce each row from `SafetyCulture/experimental#474`'s and `#478`'s README "Verified" tables against your own workspace, in a live `ucode claude` session:
- Non-streaming and streaming responses both render.
- A tool call (e.g. ask it to `Read` a file) round-trips: tool call issued, result fed back, correct final answer.
- `/model` switches between the Opus-tier (GLM) and Sonnet-tier (Kimi) models mid-session.
- A prompt long enough to approach 200k tokens does NOT trigger auto-compact before the real ~1M window is reached (confirms the `[1m]` suffix is reaching Claude Code correctly through the new `provider_models` path).
- `Read` on a PNG and a screenshot both work when routed to Kimi K3 (vision), and produce a text placeholder instead of an error when routed to GLM (no vision).
- Killing the `claude` process (Ctrl-C) or letting it exit normally both leave no orphaned process listening on the shim's port (`lsof -i :<port>` after exit should be empty) — confirms the `finally: tokens.stop(); server.shutdown(); server.server_close()` cleanup in `_launch_oss_shim` actually runs.

- [ ] **Step 4: Record results and fix forward**

Any row that fails is a real bug in the port, not a spec problem — the exact same request/response shape already passed through the source PRs against this same workspace. Diagnose with `CLAUDE_OSS_SHIM_LOG=/tmp/trace.jsonl` (Task 3) before guessing, matching the source PRs' own `DBX_CLAUDE_LOG` workflow.

---

## Self-Review Notes

- **Spec coverage:** PR #474 (translate.py, server.py, databricks.py primitives, launch wiring, security posture) → Tasks 2-5. PR #475 (tool search default) → Task 6 Step 6 (already true, verified not re-implemented). PR #478 (per-model context windows, `[1m]` suffix, vision) → Task 0 (context table), Task 1 (`supports_vision`), Task 3 (`allow_images` wiring), Task 5 (`_maybe_add_1m_suffix` reuse). The `probe`/`models` CLI commands → Task 8. The standalone tool's own `TokenCache`/`discover_oss_models`/family tables are deliberately NOT ported — Task 3 explains why (this repo already has better versions of all three, and duplicating them is what caused the Task 0 bug).
- **Workspace-specific numbers:** qwen/gpt-oss/llama-4-maverick/gemma limits are carried over from PR #420 (a different workspace) since none of those families have ever been seen on `safetyculture-safetyculture-production`; Task 0's comment flags this explicitly so nobody trusts those four numbers the way glm/kimi/kimi-k3/inkling now can be.
- **No placeholder steps:** every code-bearing step above shows the actual function body, not a description of one, except the two spots explicitly called out as needing a real value filled in from context the plan can't know ahead of time (Task 3 Step 5's header-capture assertion, Task 6/7's exact call-chain signatures) — both are flagged inline as exactly that, with instructions on what to check rather than what to guess.
