"""Read-only diagnostic: does the workspace's Anthropic-dialect gateway route
now accept an OSS model id directly (making this whole shim unnecessary for
that model), and does the mlflow OSS route behave the way the shim assumes
(non-streaming, streaming, tool calling, `max_tokens` clamping)?

Ported from `SafetyCulture/experimental#474`'s `dbxclaude/__main__.py`:
`_post`/`_report` are copied unchanged (generic HTTP diagnostic helpers with
no dependency on the rest of that file); `cmd_probe` becomes `run_probe`,
with its own host/profile/model resolution (`_context`/`_pick_default`,
which used `argparse.Namespace`) dropped — this repo's CLI wiring
(`ucode.cli`) resolves workspace/token/model itself, via `load_state`,
`get_databricks_token`, and `discover_oss_models`, and passes the results in
directly. `run_probe` never touches `state.json` or `settings.json`; it only
issues read-only HTTP probes and prints a report.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

from ucode.databricks import build_tool_base_url, model_token_limits

from .server import OSS_ROUTE


def _post(url: str, token: str, payload: dict, *, stream: bool = False) -> tuple[int, str]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
            "anthropic-version": "2023-06-01",
            "x-databricks-use-coding-agent-mode": "true",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = response.read().decode("utf-8", "replace")
            return response.status, body
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError) as exc:
        return 0, str(exc)


def _report(name: str, status: int, body: str, *, limit: int = 400) -> bool:
    ok = status == 200
    mark = "PASS" if ok else "FAIL"
    print(f"\n[{mark}] {name}  (HTTP {status})")
    snippet = " ".join(body.split())[:limit]
    print(f"       {snippet}")
    return ok


def run_probe(workspace: str, token: str, model: str | None) -> int:
    """Run the gateway probe against an already-resolved workspace/token/model.

    Returns 0 once the probe has completed, regardless of whether individual
    checks passed or failed (matching `SafetyCulture/experimental#474`'s
    `cmd_probe` return convention) — 1 only if there was no model to probe.
    """
    if not model:
        print(
            "dbx-claude: no OSS model to probe. Pass --model, or run in a workspace "
            "with GLM/Kimi endpoints discoverable.",
            file=sys.stderr,
        )
        return 1

    print(f"workspace: {workspace}")
    print(f"probing with: {model}")

    results: dict[str, bool] = {}

    # The decisive question: if the Anthropic-dialect gateway route accepts an
    # OSS model id, Databricks translates server-side and this shim is
    # unnecessary — Claude Code could point ANTHROPIC_BASE_URL straight at it.
    status, body = _post(
        f"{build_tool_base_url('claude', workspace)}/v1/messages",
        token,
        {
            "model": model,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
        },
    )
    results["anthropic dialect accepts OSS model"] = _report(
        "Anthropic dialect route with an OSS model id", status, body
    )

    chat = f"{workspace}{OSS_ROUTE}"
    status, body = _post(
        chat,
        token,
        {
            "model": model,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
        },
    )
    results["oss non-streaming"] = _report("OSS route, non-streaming", status, body)

    status, body = _post(
        chat,
        token,
        {
            "model": model,
            "max_tokens": 16,
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": "Count: 1 2 3"}],
        },
        stream=True,
    )
    results["oss streaming"] = _report("OSS route, streaming", status, body)
    if status == 200:
        has_usage = '"usage"' in body
        print(f"       stream_options usage reported: {has_usage}")
        results["streaming usage"] = has_usage

    status, body = _post(
        chat,
        token,
        {
            "model": model,
            "max_tokens": 256,
            "messages": [
                {"role": "user", "content": "What is the weather in Sydney? Use the tool."}
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the current weather for a city.",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
            "tool_choice": "auto",
        },
    )
    ok = _report("OSS route, tool calling", status, body)
    results["oss tool calling"] = ok and "tool_calls" in body

    status, body = _post(
        chat,
        token,
        {
            "model": model,
            "max_tokens": 200_000,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    limits = model_token_limits(model)
    clamp = limits["output"] if limits else "unknown"
    print(f"\n[INFO] max_tokens=200000 rejection check  (HTTP {status})")
    print(f"       {' '.join(body.split())[:400]}")
    print(f"       shim clamps to {clamp}")

    print("\n--- summary ---")
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if results.get("anthropic dialect accepts OSS model"):
        print(
            "\nThe Anthropic-dialect route accepts this model directly.\n"
            "The shim is not needed: point ANTHROPIC_BASE_URL at "
            f"{build_tool_base_url('claude', workspace)} instead."
        )
    else:
        print("\nThe Anthropic-dialect route rejects OSS models, so translation is required.")
    return 0
