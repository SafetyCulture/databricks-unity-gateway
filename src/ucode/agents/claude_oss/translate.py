"""Pure translation between the Anthropic Messages API and OpenAI chat completions.

Ported from SafetyCulture/experimental#474/#478 (`dbx-claude-shim`), which
proved this translation against safetyculture-safetyculture-production before
this port. No I/O, no network, no globals — a function of its arguments only,
so it is unit-tested without a Databricks workspace.

Direction of travel:

    Claude Code  --(Anthropic Messages)-->  anthropic_to_openai()
                                                    |
                                     Databricks /ai-gateway/mlflow/v1
                                                    |
    Claude Code  <--(Anthropic Messages)--  openai_to_anthropic()
                                            StreamTranslator (SSE)

Two properties of the Databricks AI Gateway shape the code below:

1. Its request validator is strict and rejects unknown fields on tool
   definitions, so tools are rebuilt from a whitelist rather than passed
   through with unwanted keys removed.
2. The OSS route caps `max_tokens` well below the model's native output limit
   and rejects requests that exceed the cap, so max_tokens is clamped.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from typing import Any

# Anthropic stop_reason <- OpenAI finish_reason.
_FINISH_REASON = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "end_turn",
}

# Anthropic error.type <- HTTP status. Getting 429 right matters: it is what
# makes Claude Code back off and retry rather than fail the turn outright.
_ERROR_TYPE = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    408: "timeout_error",
    413: "request_too_large",
    422: "invalid_request_error",
    429: "rate_limit_error",
    500: "api_error",
    502: "api_error",
    503: "overloaded_error",
    504: "timeout_error",
    529: "overloaded_error",
}


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


# How to represent the `reasoning_content` GLM, Kimi, and Inkling emit.
#
#   "thinking" — Anthropic thinking blocks. Renders in Claude Code's UI the way
#                Pi renders these models' reasoning.
#   "text"     — ordinary text. Always safe, but the reasoning becomes part of
#                the assistant message and is replayed in later turns, which
#                costs context that is not cached on this route.
#   "drop"     — discard. Leaves the UI silent for the whole reasoning phase,
#                and yields an empty response if the model hits max_tokens
#                before it finishes reasoning.
REASONING_MODES = ("thinking", "text", "drop")

# Anthropic pairs every thinking block with a signature it issues, and replays
# require it. Nothing here can produce a real one, so a placeholder is emitted
# to keep the block well-formed; `_convert_messages` drops thinking blocks on
# the way back up, so it is never sent anywhere that would verify it.
_UNVERIFIED_SIGNATURE = "dbx-claude-shim-unverified"


def _reasoning_of(message: dict) -> str:
    """Pull reasoning text out of a message, whichever field the model used."""
    for key in ("reasoning_content", "reasoning"):
        value = message.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


# --- request: Anthropic -> OpenAI -------------------------------------------


def _text_of(blocks: Any) -> str:
    """Flatten an Anthropic content value to plain text.

    Accepts the string form, a list of blocks, or a single block. Non-text
    blocks that have no OpenAI equivalent in this position collapse to a short
    placeholder rather than vanishing silently, so a model that receives them
    can still tell something was there.
    """
    if blocks is None:
        return ""
    if isinstance(blocks, str):
        return blocks
    if isinstance(blocks, dict):
        blocks = [blocks]
    out: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            out.append(str(block))
            continue
        kind = block.get("type")
        if kind == "text":
            out.append(block.get("text") or "")
        elif kind == "image":
            out.append("[image omitted]")
        elif kind in ("thinking", "redacted_thinking"):
            continue
        else:
            out.append(f"[{kind or 'unknown'} block omitted]")
    return "\n".join(part for part in out if part)


def _system_text(system: Any) -> str:
    """Anthropic `system` is a string or a list of text blocks (with cache_control)."""
    return _text_of(system)


def _image_part(block: dict) -> dict | None:
    """Anthropic image block -> OpenAI image_url part, or None if unsupported."""
    source = block.get("source") or {}
    if source.get("type") == "base64":
        media = source.get("media_type") or "image/png"
        data = source.get("data") or ""
        return {"type": "image_url", "image_url": {"url": f"data:{media};base64,{data}"}}
    if source.get("type") == "url" and source.get("url"):
        return {"type": "image_url", "image_url": {"url": source["url"]}}
    return None


def _user_content(blocks: Any, *, allow_images: bool) -> Any:
    """Build the OpenAI content value for a user message.

    Returns a plain string when there is nothing but text, since some gateway
    validators are happier with the scalar form than a single-element array.
    """
    if isinstance(blocks, str):
        return blocks
    if not isinstance(blocks, list):
        return _text_of(blocks)

    parts: list[dict] = []
    for block in blocks:
        if not isinstance(block, dict):
            parts.append({"type": "text", "text": str(block)})
            continue
        kind = block.get("type")
        if kind == "text":
            text = block.get("text") or ""
            if text:
                parts.append({"type": "text", "text": text})
        elif kind == "image":
            part = _image_part(block) if allow_images else None
            parts.append(part or {"type": "text", "text": "[image omitted]"})
        elif kind in ("thinking", "redacted_thinking", "tool_result"):
            continue  # tool_result is hoisted separately by _convert_messages
        else:
            parts.append({"type": "text", "text": f"[{kind or 'unknown'} block omitted]"})

    if not parts:
        return ""
    if all(part["type"] == "text" for part in parts):
        return "\n".join(part["text"] for part in parts)
    return parts


def _convert_messages(messages: list[dict], *, allow_images: bool) -> list[dict]:
    """Flatten Anthropic messages into the OpenAI message sequence.

    The one structural difference that matters: Anthropic carries tool results
    as `tool_result` blocks inside the *user* message that follows the tool
    call, while OpenAI wants each result as its own `role: "tool"` message.
    Those are hoisted out and emitted before the remaining user text, which is
    the order OpenAI-dialect servers expect.
    """
    out: list[dict] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")

        if role == "assistant":
            text_parts: list[str] = []
            tool_calls: list[dict] = []
            blocks = [content] if isinstance(content, dict) else content
            if isinstance(blocks, str):
                text_parts.append(blocks)
            elif isinstance(blocks, list):
                for block in blocks:
                    if not isinstance(block, dict):
                        continue
                    kind = block.get("type")
                    if kind == "text":
                        if block.get("text"):
                            text_parts.append(block["text"])
                    elif kind == "tool_use":
                        tool_calls.append(
                            {
                                "id": block.get("id") or new_id("call"),
                                "type": "function",
                                "function": {
                                    "name": block.get("name") or "",
                                    "arguments": json.dumps(block.get("input") or {}),
                                },
                            }
                        )
                    # thinking / redacted_thinking are dropped: they are only
                    # replayable with the signature Anthropic issued, which no
                    # other provider can produce.

            entry: dict[str, Any] = {"role": "assistant"}
            entry["content"] = "\n".join(text_parts) if text_parts else None
            if tool_calls:
                entry["tool_calls"] = tool_calls
            # An assistant turn with neither text nor tool calls is not
            # meaningful to the model and some validators reject it.
            if entry["content"] or tool_calls:
                out.append(entry)
            continue

        # role == "user" (or anything else, treated as user).
        tool_results: list[dict] = []
        # Images returned by a tool (Claude Code's Read on a PNG, screenshots)
        # have nowhere to live: an OpenAI `role: "tool"` message is text-only.
        # They are collected here and re-attached as a following user message,
        # which is the only way a vision model ever sees them.
        hoisted_images: list[dict] = []
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    inner = block.get("content")
                    if allow_images and isinstance(inner, list):
                        for part in inner:
                            if isinstance(part, dict) and part.get("type") == "image":
                                image = _image_part(part)
                                if image:
                                    hoisted_images.append(image)
                    body = _text_of(inner)
                    if block.get("is_error"):
                        body = f"Error: {body}" if body else "Error"
                    if hoisted_images and not body.strip():
                        body = "(image returned, attached below)"
                    tool_results.append(
                        {
                            "role": "tool",
                            "tool_call_id": block.get("tool_use_id") or "",
                            "content": body or "(no output)",
                        }
                    )
        out.extend(tool_results)
        if hoisted_images:
            out.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Image returned by the tool call above:"},
                        *hoisted_images,
                    ],
                }
            )

        body = _user_content(content, allow_images=allow_images)
        if body:
            out.append({"role": "user", "content": body})

    return out


def _convert_tools(tools: list[dict]) -> list[dict]:
    """Rebuild tool definitions from a whitelist.

    The gateway's validator rejects unknown fields, and Claude Code decorates
    tool definitions with its own (`cache_control`, and client-SDK additions
    such as `eager_input_streaming`). Constructing the OpenAI form from only
    the three fields that exist in both dialects avoids having to chase each
    new decoration as it appears.
    """
    out: list[dict] = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        schema = tool.get("input_schema")
        if not name or not isinstance(schema, dict):
            # Server-side built-ins (web_search, computer use, …) declare a
            # `type` but no input_schema. There is no OSS equivalent to route
            # them to, so they are dropped.
            continue
        parameters = {k: v for k, v in schema.items() if k != "$schema"}
        function: dict[str, Any] = {"name": name, "parameters": parameters}
        if tool.get("description"):
            function["description"] = tool["description"]
        out.append({"type": "function", "function": function})
    return out


def _convert_tool_choice(choice: Any) -> Any:
    if not isinstance(choice, dict):
        return None
    kind = choice.get("type")
    if kind == "auto":
        return "auto"
    if kind == "any":
        return "required"
    if kind == "none":
        return "none"
    if kind == "tool" and choice.get("name"):
        return {"type": "function", "function": {"name": choice["name"]}}
    return None


def anthropic_to_openai(
    request: dict,
    *,
    model: str,
    max_output: int | None = None,
    allow_images: bool = True,
) -> dict:
    """Translate one Anthropic Messages request into an OpenAI chat completion.

    `model` is the Databricks model id to target; the incoming Anthropic model
    name is discarded because model selection is resolved by the caller (see
    `models.resolve`). `max_output` clamps `max_tokens` to what the gateway
    route accepts.
    """
    messages = _convert_messages(request.get("messages") or [], allow_images=allow_images)

    system = _system_text(request.get("system"))
    if system:
        messages.insert(0, {"role": "system", "content": system})

    max_tokens = request.get("max_tokens")
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        max_tokens = max_output or 8192
    if max_output:
        max_tokens = min(max_tokens, max_output)

    out: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }

    if request.get("stream"):
        out["stream"] = True
        # Without this the final chunk carries no usage and Claude Code's
        # context meter never advances.
        out["stream_options"] = {"include_usage": True}

    for key in ("temperature", "top_p"):
        if isinstance(request.get(key), (int, float)):
            out[key] = request[key]
    # top_k has no OpenAI equivalent and is dropped.

    stop = request.get("stop_sequences")
    if isinstance(stop, list) and stop:
        out["stop"] = stop

    tools = _convert_tools(request.get("tools") or [])
    if tools:
        out["tools"] = tools
        choice = _convert_tool_choice(request.get("tool_choice"))
        if choice is not None:
            out["tool_choice"] = choice

    # Deliberately not forwarded: metadata, cache_control, thinking, betas,
    # container, mcp_servers. None have an OSS-route equivalent and the strict
    # validator rejects unknown top-level fields.
    return out


# --- response: OpenAI -> Anthropic ------------------------------------------


def _parse_arguments(raw: Any) -> dict:
    """Parse a tool call's `arguments` string into an object.

    Anthropic's `tool_use.input` must be an object. A model that emits invalid
    JSON would otherwise break the whole turn, so the raw string is preserved
    under a sentinel key and the tool itself reports the error.
    """
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return {"__unparsed_arguments__": str(raw)}
    return parsed if isinstance(parsed, dict) else {"__unparsed_arguments__": raw}


def _usage(raw: Any) -> dict:
    usage = raw if isinstance(raw, dict) else {}
    out = {
        "input_tokens": int(usage.get("prompt_tokens") or 0),
        "output_tokens": int(usage.get("completion_tokens") or 0),
    }
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict) and details.get("cached_tokens"):
        out["cache_read_input_tokens"] = int(details["cached_tokens"])
    return out


def openai_to_anthropic(response: dict, *, model: str, reasoning: str = "thinking") -> dict:
    """Translate a non-streaming OpenAI chat completion into an Anthropic message."""
    choices = response.get("choices") or [{}]
    choice = choices[0] if isinstance(choices[0], dict) else {}
    message = choice.get("message") or {}

    content: list[dict] = []

    thoughts = _reasoning_of(message)
    if thoughts and reasoning == "thinking":
        content.append(
            {"type": "thinking", "thinking": thoughts, "signature": _UNVERIFIED_SIGNATURE}
        )
    elif thoughts and reasoning == "text":
        content.append({"type": "text", "text": thoughts})

    text = message.get("content")
    if isinstance(text, list):
        # Some providers return the multimodal part array even on output.
        text = _text_of(
            [
                part if isinstance(part, dict) else {"type": "text", "text": str(part)}
                for part in text
            ]
        )
    if text:
        content.append({"type": "text", "text": text})

    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        function = call.get("function") or {}
        content.append(
            {
                "type": "tool_use",
                "id": call.get("id") or new_id("toolu"),
                "name": function.get("name") or "",
                "input": _parse_arguments(function.get("arguments")),
            }
        )

    if not any(block["type"] in ("text", "tool_use") for block in content):
        # A message of nothing but a thinking block is not something clients
        # handle well, so always leave at least one text or tool_use block.
        content.append({"type": "text", "text": ""})

    stop_reason = _FINISH_REASON.get(choice.get("finish_reason") or "", "end_turn")
    if any(block["type"] == "tool_use" for block in content):
        stop_reason = "tool_use"

    return {
        "id": response.get("id") or new_id("msg"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": _usage(response.get("usage")),
    }


class StreamTranslator:
    """Turn an OpenAI SSE chunk stream into the Anthropic SSE event stream.

    Feed each decoded `data:` object to `feed()` and `finish()` once the
    upstream stream ends; both yield `(event_name, payload)` pairs ready to be
    written as SSE. The class owns the whole event contract Claude Code relies
    on: exactly one `message_start`, correctly indexed and balanced
    `content_block_start`/`stop` pairs, and a terminating
    `message_delta`/`message_stop`.

    Two upstream behaviours are handled explicitly because they are common on
    OSS routes:

    - A stream that ends without ever sending `finish_reason`. `finish()`
      infers one instead of leaving the client waiting, which is what produces
      "stream ended without finish_reason" errors in other harnesses.
    - Text and tool-call deltas interleaved in one chunk. Anthropic requires a
      content block be closed before the next opens, so text blocks are closed
      when a tool call starts.
    """

    def __init__(
        self,
        *,
        model: str,
        message_id: str | None = None,
        reasoning: str = "thinking",
    ) -> None:
        self.model = model
        self.message_id = message_id or new_id("msg")
        self.reasoning = reasoning if reasoning in REASONING_MODES else "thinking"
        self._started = False
        self._next_index = 0
        self._text_index: int | None = None
        self._thinking_index: int | None = None
        # OpenAI tool_call index -> Anthropic content block index.
        self._tool_index: dict[int, int] = {}
        self._finish_reason: str | None = None
        self._usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}
        self._saw_tool_use = False
        self._closed = False

    # -- helpers

    def _start(self) -> Iterator[tuple[str, dict]]:
        if self._started:
            return
        self._started = True
        yield (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": self.message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": self.model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    # Real values arrive with the final usage chunk and are
                    # reported in message_delta.
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            },
        )

    def _close_text(self) -> Iterator[tuple[str, dict]]:
        if self._text_index is None:
            return
        yield ("content_block_stop", {"type": "content_block_stop", "index": self._text_index})
        self._text_index = None

    def _close_thinking(self) -> Iterator[tuple[str, dict]]:
        if self._thinking_index is None:
            return
        # Anthropic closes a thinking block with its signature before the stop.
        yield (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": self._thinking_index,
                "delta": {"type": "signature_delta", "signature": _UNVERIFIED_SIGNATURE},
            },
        )
        yield ("content_block_stop", {"type": "content_block_stop", "index": self._thinking_index})
        self._thinking_index = None

    def _feed_reasoning(self, text: str) -> Iterator[tuple[str, dict]]:
        """Emit a reasoning delta in whichever representation is configured.

        Reasoning always precedes content, so a thinking block is closed as
        soon as real content starts.
        """
        if self.reasoning == "drop":
            return
        if self.reasoning == "text":
            yield from self._feed_text(text)
            return

        if self._thinking_index is None:
            self._thinking_index = self._next_index
            self._next_index += 1
            yield (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": self._thinking_index,
                    "content_block": {"type": "thinking", "thinking": ""},
                },
            )
        yield (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": self._thinking_index,
                "delta": {"type": "thinking_delta", "thinking": text},
            },
        )

    def _feed_text(self, text: str) -> Iterator[tuple[str, dict]]:
        if self._text_index is None:
            yield from self._close_thinking()
            self._text_index = self._next_index
            self._next_index += 1
            yield (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": self._text_index,
                    "content_block": {"type": "text", "text": ""},
                },
            )
        yield (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": self._text_index,
                "delta": {"type": "text_delta", "text": text},
            },
        )

    # -- public

    def feed(self, chunk: dict) -> Iterator[tuple[str, dict]]:
        if self._closed:
            return
        yield from self._start()

        if isinstance(chunk.get("usage"), dict):
            self._usage = _usage(chunk["usage"])

        choices = chunk.get("choices") or []
        if not choices:
            return
        choice = choices[0] if isinstance(choices[0], dict) else {}
        delta = choice.get("delta") or {}

        if choice.get("finish_reason"):
            self._finish_reason = choice["finish_reason"]

        tool_calls = delta.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                yield from self._feed_tool_call(call)

        # Reasoning is emitted before content, so handle it first.
        thoughts = _reasoning_of(delta)
        if thoughts:
            yield from self._feed_reasoning(thoughts)

        text = delta.get("content")
        if isinstance(text, str) and text:
            yield from self._feed_text(text)

    def _feed_tool_call(self, call: dict) -> Iterator[tuple[str, dict]]:
        position = call.get("index")
        if not isinstance(position, int):
            position = 0
        function = call.get("function") or {}

        if position not in self._tool_index:
            # A tool block cannot open while another block is still open.
            yield from self._close_thinking()
            yield from self._close_text()
            index = self._next_index
            self._next_index += 1
            self._tool_index[position] = index
            self._saw_tool_use = True
            yield (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {
                        "type": "tool_use",
                        "id": call.get("id") or new_id("toolu"),
                        "name": function.get("name") or "",
                        "input": {},
                    },
                },
            )

        arguments = function.get("arguments")
        if isinstance(arguments, str) and arguments:
            yield (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": self._tool_index[position],
                    "delta": {"type": "input_json_delta", "partial_json": arguments},
                },
            )

    def finish(self) -> Iterator[tuple[str, dict]]:
        """Emit the terminating events. Safe to call once; later calls are no-ops."""
        if self._closed:
            return
        yield from self._start()
        self._closed = True

        yield from self._close_thinking()
        yield from self._close_text()
        for index in sorted(self._tool_index.values()):
            yield ("content_block_stop", {"type": "content_block_stop", "index": index})

        # Infer a stop_reason when upstream never sent one, rather than
        # emitting null and leaving the client to error.
        stop_reason = _FINISH_REASON.get(self._finish_reason or "", None)
        if self._saw_tool_use:
            stop_reason = "tool_use"
        elif stop_reason is None:
            stop_reason = "end_turn"

        yield (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": self._usage,
            },
        )
        yield ("message_stop", {"type": "message_stop"})


# --- errors and token counting ---------------------------------------------


def error_body(status: int, message: str) -> dict:
    return {
        "type": "error",
        "error": {"type": _ERROR_TYPE.get(status, "api_error"), "message": message},
    }


def count_tokens(request: dict) -> int:
    """Estimate the prompt size for `/v1/messages/count_tokens`.

    The OSS route exposes no tokenizer, so this is a deliberate approximation
    (~4 characters per token plus per-block overhead) used only to drive
    Claude Code's context meter. It is never used for billing or for clamping
    a request.
    """
    chars = len(_system_text(request.get("system")))
    blocks = 0
    for message in request.get("messages") or []:
        content = message.get("content")
        if isinstance(content, list):
            blocks += len(content)
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    chars += len(json.dumps(block.get("input") or {}))
        else:
            blocks += 1
        chars += len(_text_of(content))
    for tool in request.get("tools") or []:
        if isinstance(tool, dict):
            chars += len(json.dumps(tool.get("input_schema") or {}))
            chars += len(tool.get("description") or "")
    return max(1, chars // 4 + blocks * 3)
