"""Translation tests. Standard library unittest, no network, no Databricks.

Run: python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ucode.agents.claude_oss import translate  # noqa: E402


class TestRequestTranslation(unittest.TestCase):
    def test_system_string_becomes_leading_system_message(self):
        out = translate.anthropic_to_openai(
            {"system": "Be terse.", "messages": [{"role": "user", "content": "hi"}]},
            model="system.ai.glm-5-2",
        )
        self.assertEqual(out["messages"][0], {"role": "system", "content": "Be terse."})
        self.assertEqual(out["model"], "system.ai.glm-5-2")

    def test_system_block_list_with_cache_control_is_flattened(self):
        out = translate.anthropic_to_openai(
            {
                "system": [
                    {"type": "text", "text": "Part one.", "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": "Part two."},
                ],
                "messages": [{"role": "user", "content": "hi"}],
            },
            model="m",
        )
        self.assertEqual(out["messages"][0]["content"], "Part one.\nPart two.")
        self.assertNotIn("cache_control", json.dumps(out))

    def test_tool_use_becomes_openai_tool_calls(self):
        out = translate.anthropic_to_openai(
            {
                "messages": [
                    {"role": "user", "content": "read it"},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "Reading."},
                            {
                                "type": "tool_use",
                                "id": "toolu_1",
                                "name": "Read",
                                "input": {"file_path": "/a.txt"},
                            },
                        ],
                    },
                ]
            },
            model="m",
        )
        assistant = out["messages"][-1]
        self.assertEqual(assistant["content"], "Reading.")
        call = assistant["tool_calls"][0]
        self.assertEqual(call["id"], "toolu_1")
        self.assertEqual(call["function"]["name"], "Read")
        self.assertEqual(json.loads(call["function"]["arguments"]), {"file_path": "/a.txt"})

    def test_tool_result_is_hoisted_into_its_own_tool_message(self):
        """Anthropic nests results inside the next user message; OpenAI wants
        them as separate `tool` messages, emitted before any user text."""
        out = translate.anthropic_to_openai(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_1",
                                "content": "file body",
                            },
                            {"type": "text", "text": "now summarise"},
                        ],
                    }
                ]
            },
            model="m",
        )
        self.assertEqual(
            out["messages"],
            [
                {"role": "tool", "tool_call_id": "toolu_1", "content": "file body"},
                {"role": "user", "content": "now summarise"},
            ],
        )

    def test_tool_result_error_is_marked_and_empty_result_has_placeholder(self):
        out = translate.anthropic_to_openai(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "t1",
                                "content": "no such file",
                                "is_error": True,
                            },
                            {"type": "tool_result", "tool_use_id": "t2", "content": ""},
                        ],
                    }
                ]
            },
            model="m",
        )
        self.assertEqual(out["messages"][0]["content"], "Error: no such file")
        self.assertEqual(out["messages"][1]["content"], "(no output)")

    def test_tool_result_with_block_list_content(self):
        out = translate.anthropic_to_openai(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "t1",
                                "content": [
                                    {"type": "text", "text": "line one"},
                                    {"type": "text", "text": "line two"},
                                ],
                            }
                        ],
                    }
                ]
            },
            model="m",
        )
        self.assertEqual(out["messages"][0]["content"], "line one\nline two")

    def test_thinking_blocks_are_dropped(self):
        out = translate.anthropic_to_openai(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": "hmm", "signature": "sig"},
                            {"type": "text", "text": "answer"},
                        ],
                    }
                ]
            },
            model="m",
        )
        self.assertEqual(out["messages"][0]["content"], "answer")
        self.assertNotIn("thinking", json.dumps(out))

    def test_assistant_turn_with_only_thinking_is_omitted(self):
        out = translate.anthropic_to_openai(
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": [{"type": "thinking", "thinking": "hmm"}]},
                ]
            },
            model="m",
        )
        self.assertEqual([m["role"] for m in out["messages"]], ["user"])

    def test_tools_are_rebuilt_from_a_whitelist(self):
        """The gateway's validator rejects unknown fields on tool definitions,
        so decorations Claude Code adds must not survive translation."""
        out = translate.anthropic_to_openai(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [
                    {
                        "name": "Read",
                        "description": "Read a file",
                        "input_schema": {
                            "$schema": "http://json-schema.org/draft-07/schema#",
                            "type": "object",
                            "properties": {"file_path": {"type": "string"}},
                        },
                        "cache_control": {"type": "ephemeral"},
                        "eager_input_streaming": True,
                    }
                ],
                "tool_choice": {"type": "auto"},
            },
            model="m",
        )
        self.assertEqual(list(out["tools"][0].keys()), ["type", "function"])
        self.assertEqual(
            sorted(out["tools"][0]["function"].keys()), ["description", "name", "parameters"]
        )
        self.assertNotIn("$schema", out["tools"][0]["function"]["parameters"])
        self.assertNotIn("eager_input_streaming", json.dumps(out))
        self.assertNotIn("cache_control", json.dumps(out))
        self.assertEqual(out["tool_choice"], "auto")

    def test_server_side_builtin_tools_are_dropped(self):
        out = translate.anthropic_to_openai(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "web_search_20250305", "name": "web_search"}],
            },
            model="m",
        )
        self.assertNotIn("tools", out)

    def test_tool_choice_variants(self):
        for given, expected in (
            ({"type": "any"}, "required"),
            ({"type": "none"}, "none"),
            ({"type": "tool", "name": "Read"}, {"type": "function", "function": {"name": "Read"}}),
        ):
            out = translate.anthropic_to_openai(
                {
                    "messages": [{"role": "user", "content": "hi"}],
                    "tools": [{"name": "Read", "input_schema": {"type": "object"}}],
                    "tool_choice": given,
                },
                model="m",
            )
            self.assertEqual(out["tool_choice"], expected)

    def test_max_tokens_is_clamped_to_the_route_cap(self):
        out = translate.anthropic_to_openai(
            {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 128_000},
            model="m",
            max_output=25_000,
        )
        self.assertEqual(out["max_tokens"], 25_000)

    def test_streaming_requests_ask_for_usage(self):
        out = translate.anthropic_to_openai(
            {"messages": [{"role": "user", "content": "hi"}], "stream": True}, model="m"
        )
        self.assertTrue(out["stream"])
        self.assertEqual(out["stream_options"], {"include_usage": True})

    def test_unsupported_top_level_fields_are_not_forwarded(self):
        out = translate.anthropic_to_openai(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "metadata": {"user_id": "u1"},
                "thinking": {"type": "enabled", "budget_tokens": 4096},
                "top_k": 40,
                "temperature": 0.3,
                "stop_sequences": ["STOP"],
            },
            model="m",
        )
        for key in ("metadata", "thinking", "top_k"):
            self.assertNotIn(key, out)
        self.assertEqual(out["temperature"], 0.3)
        self.assertEqual(out["stop"], ["STOP"])

    def test_images_convert_to_data_uri_or_are_stripped(self):
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this"},
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"},
                        },
                    ],
                }
            ]
        }
        allowed = translate.anthropic_to_openai(request, model="m", allow_images=True)
        parts = allowed["messages"][0]["content"]
        self.assertEqual(parts[1]["image_url"]["url"], "data:image/png;base64,AAAA")

        stripped = translate.anthropic_to_openai(request, model="m", allow_images=False)
        self.assertEqual(stripped["messages"][0]["content"], "what is this\n[image omitted]")


class TestResponseTranslation(unittest.TestCase):
    def test_text_response(self):
        out = translate.openai_to_anthropic(
            {
                "id": "chatcmpl-1",
                "choices": [
                    {"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            },
            model="system.ai.glm-5-2",
        )
        self.assertEqual(out["content"], [{"type": "text", "text": "hello"}])
        self.assertEqual(out["stop_reason"], "end_turn")
        self.assertEqual(out["model"], "system.ai.glm-5-2")
        self.assertEqual(out["usage"], {"input_tokens": 10, "output_tokens": 2})

    def test_tool_call_response_sets_tool_use_stop_reason(self):
        out = translate.openai_to_anthropic(
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "function": {"name": "Read", "arguments": '{"file_path":"/a"}'},
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
            model="m",
        )
        self.assertEqual(out["stop_reason"], "tool_use")
        block = out["content"][0]
        self.assertEqual(block["type"], "tool_use")
        self.assertEqual(block["name"], "Read")
        self.assertEqual(block["input"], {"file_path": "/a"})

    def test_malformed_tool_arguments_do_not_break_the_turn(self):
        out = translate.openai_to_anthropic(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {"id": "c1", "function": {"name": "Read", "arguments": "{not json"}}
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
            model="m",
        )
        self.assertEqual(out["content"][0]["input"], {"__unparsed_arguments__": "{not json"})

    def test_length_finish_reason_and_cached_tokens(self):
        out = translate.openai_to_anthropic(
            {
                "choices": [{"message": {"content": "cut off"}, "finish_reason": "length"}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "prompt_tokens_details": {"cached_tokens": 60},
                },
            },
            model="m",
        )
        self.assertEqual(out["stop_reason"], "max_tokens")
        self.assertEqual(out["usage"]["cache_read_input_tokens"], 60)

    def test_empty_response_still_yields_a_content_block(self):
        out = translate.openai_to_anthropic({"choices": [{"message": {}}]}, model="m")
        self.assertEqual(out["content"], [{"type": "text", "text": ""}])

    def test_reasoning_only_response_gets_a_thinking_block_plus_empty_text(self):
        """What GLM returns when it hits max_tokens before producing content."""
        out = translate.openai_to_anthropic(
            {
                "choices": [
                    {
                        "message": {"content": "", "reasoning_content": "still reasoning"},
                        "finish_reason": "length",
                    }
                ]
            },
            model="m",
        )
        self.assertEqual(out["content"][0]["type"], "thinking")
        self.assertEqual(out["content"][0]["thinking"], "still reasoning")
        self.assertEqual(out["content"][-1], {"type": "text", "text": ""})
        self.assertEqual(out["stop_reason"], "max_tokens")

    def test_reasoning_modes_for_non_streaming(self):
        response = {"choices": [{"message": {"content": "answer", "reasoning_content": "why"}}]}
        as_text = translate.openai_to_anthropic(response, model="m", reasoning="text")
        self.assertEqual(
            as_text["content"],
            [{"type": "text", "text": "why"}, {"type": "text", "text": "answer"}],
        )
        dropped = translate.openai_to_anthropic(response, model="m", reasoning="drop")
        self.assertEqual(dropped["content"], [{"type": "text", "text": "answer"}])


def drain(translator, chunks):
    events = []
    for chunk in chunks:
        events.extend(translator.feed(chunk))
    events.extend(translator.finish())
    return events


def names(events):
    return [name for name, _ in events]


class TestStreamTranslation(unittest.TestCase):
    def chunk(self, delta=None, finish=None, usage=None):
        out = {"choices": [{"delta": delta or {}, "finish_reason": finish}]}
        if usage:
            out["usage"] = usage
        return out

    def test_text_stream_event_sequence(self):
        translator = translate.StreamTranslator(model="m")
        events = drain(
            translator,
            [
                self.chunk({"role": "assistant"}),
                self.chunk({"content": "he"}),
                self.chunk({"content": "llo"}),
                self.chunk(finish="stop", usage={"prompt_tokens": 7, "completion_tokens": 3}),
            ],
        )
        self.assertEqual(
            names(events),
            [
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ],
        )
        payloads = dict(events[-2:])
        self.assertEqual(payloads["message_delta"]["delta"]["stop_reason"], "end_turn")
        self.assertEqual(
            payloads["message_delta"]["usage"], {"input_tokens": 7, "output_tokens": 3}
        )

    def test_exactly_one_message_start(self):
        translator = translate.StreamTranslator(model="m")
        events = drain(translator, [self.chunk({"content": "a"}), self.chunk({"content": "b"})])
        self.assertEqual(names(events).count("message_start"), 1)

    def test_tool_call_stream_produces_input_json_deltas(self):
        translator = translate.StreamTranslator(model="m")
        events = drain(
            translator,
            [
                self.chunk(
                    {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "Read"}}]}
                ),
                self.chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"file'}}]}),
                self.chunk(
                    {"tool_calls": [{"index": 0, "function": {"arguments": '_path":"/a"}'}}]}
                ),
                self.chunk(finish="tool_calls"),
            ],
        )
        starts = [p for n, p in events if n == "content_block_start"]
        self.assertEqual(starts[0]["content_block"]["type"], "tool_use")
        self.assertEqual(starts[0]["content_block"]["name"], "Read")
        self.assertEqual(starts[0]["content_block"]["input"], {})

        partials = [
            p["delta"]["partial_json"]
            for n, p in events
            if n == "content_block_delta" and p["delta"]["type"] == "input_json_delta"
        ]
        self.assertEqual(json.loads("".join(partials)), {"file_path": "/a"})
        self.assertEqual(dict(events)["message_delta"]["delta"]["stop_reason"], "tool_use")

    def test_text_block_is_closed_before_a_tool_block_opens(self):
        """Anthropic forbids overlapping content blocks."""
        translator = translate.StreamTranslator(model="m")
        events = drain(
            translator,
            [
                self.chunk({"content": "Let me look."}),
                self.chunk(
                    {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "Read"}}]}
                ),
                self.chunk(finish="tool_calls"),
            ],
        )
        order = names(events)
        first_stop = order.index("content_block_stop")
        second_start = [i for i, n in enumerate(order) if n == "content_block_start"][1]
        self.assertLess(first_stop, second_start)

        indices = [p["index"] for n, p in events if n == "content_block_start"]
        self.assertEqual(indices, [0, 1])

    def test_every_opened_block_is_closed(self):
        translator = translate.StreamTranslator(model="m")
        events = drain(
            translator,
            [
                self.chunk({"content": "text"}),
                self.chunk({"tool_calls": [{"index": 0, "id": "a", "function": {"name": "A"}}]}),
                self.chunk({"tool_calls": [{"index": 1, "id": "b", "function": {"name": "B"}}]}),
                self.chunk(finish="tool_calls"),
            ],
        )
        opened = sorted(p["index"] for n, p in events if n == "content_block_start")
        closed = sorted(p["index"] for n, p in events if n == "content_block_stop")
        self.assertEqual(opened, closed)
        self.assertEqual(opened, [0, 1, 2])

    def test_stream_ending_without_finish_reason_still_terminates(self):
        """The failure other harnesses report as 'stream ended without
        finish_reason'. A stop_reason is inferred rather than left null."""
        translator = translate.StreamTranslator(model="m")
        events = drain(translator, [self.chunk({"content": "partial"})])
        self.assertEqual(names(events)[-2:], ["message_delta", "message_stop"])
        self.assertEqual(dict(events)["message_delta"]["delta"]["stop_reason"], "end_turn")

    def test_finish_is_idempotent(self):
        translator = translate.StreamTranslator(model="m")
        drain(translator, [self.chunk({"content": "a"})])
        self.assertEqual(list(translator.finish()), [])

    def test_reasoning_becomes_a_thinking_block_by_default(self):
        """GLM emits reasoning_content heavily; dropping it leaves the UI silent."""
        translator = translate.StreamTranslator(model="m")
        events = drain(
            translator,
            [self.chunk({"reasoning_content": "step one"}), self.chunk({"content": "answer"})],
        )
        starts = [p for n, p in events if n == "content_block_start"]
        self.assertEqual(starts[0]["content_block"]["type"], "thinking")
        self.assertEqual(starts[1]["content_block"]["type"], "text")
        self.assertEqual(starts[0]["index"], 0)
        self.assertEqual(starts[1]["index"], 1)

        kinds = [p["delta"]["type"] for n, p in events if n == "content_block_delta"]
        self.assertEqual(kinds, ["thinking_delta", "signature_delta", "text_delta"])

    def test_thinking_block_is_closed_before_text_opens(self):
        translator = translate.StreamTranslator(model="m")
        events = drain(
            translator,
            [self.chunk({"reasoning_content": "hmm"}), self.chunk({"content": "answer"})],
        )
        order = names(events)
        self.assertLess(order.index("content_block_stop"), order.index("content_block_start", 2))
        opened = sorted(p["index"] for n, p in events if n == "content_block_start")
        closed = sorted(p["index"] for n, p in events if n == "content_block_stop")
        self.assertEqual(opened, closed)

    def test_thinking_block_is_closed_before_a_tool_call_opens(self):
        translator = translate.StreamTranslator(model="m")
        events = drain(
            translator,
            [
                self.chunk({"reasoning_content": "I should read the file"}),
                self.chunk(
                    {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "Read"}}]}
                ),
                self.chunk(finish="tool_calls"),
            ],
        )
        opened = sorted(p["index"] for n, p in events if n == "content_block_start")
        closed = sorted(p["index"] for n, p in events if n == "content_block_stop")
        self.assertEqual(opened, closed)
        self.assertEqual(dict(events)["message_delta"]["delta"]["stop_reason"], "tool_use")

    def test_reasoning_as_text_merges_into_one_text_block(self):
        translator = translate.StreamTranslator(model="m", reasoning="text")
        events = drain(
            translator,
            [self.chunk({"reasoning_content": "step one. "}), self.chunk({"content": "answer"})],
        )
        self.assertNotIn("thinking", json.dumps(events))
        text = "".join(p["delta"]["text"] for n, p in events if n == "content_block_delta")
        self.assertEqual(text, "step one. answer")
        self.assertEqual(names(events).count("content_block_start"), 1)

    def test_reasoning_can_be_dropped(self):
        translator = translate.StreamTranslator(model="m", reasoning="drop")
        events = drain(
            translator,
            [self.chunk({"reasoning_content": "step one"}), self.chunk({"content": "answer"})],
        )
        self.assertNotIn("thinking", json.dumps(events))
        deltas = [p["delta"]["text"] for n, p in events if n == "content_block_delta"]
        self.assertEqual(deltas, ["answer"])

    def test_reasoning_only_response_still_terminates_cleanly(self):
        """Hitting max_tokens mid-reasoning is common at small caps."""
        translator = translate.StreamTranslator(model="m")
        events = drain(
            translator, [self.chunk({"reasoning_content": "still going"}, finish="length")]
        )
        opened = sorted(p["index"] for n, p in events if n == "content_block_start")
        closed = sorted(p["index"] for n, p in events if n == "content_block_stop")
        self.assertEqual(opened, closed)
        self.assertEqual(names(events)[-2:], ["message_delta", "message_stop"])
        self.assertEqual(dict(events)["message_delta"]["delta"]["stop_reason"], "max_tokens")

    def test_alternate_reasoning_field_name(self):
        translator = translate.StreamTranslator(model="m")
        events = drain(translator, [self.chunk({"reasoning": "via the other field"})])
        deltas = [
            p["delta"]["thinking"]
            for n, p in events
            if n == "content_block_delta" and p["delta"]["type"] == "thinking_delta"
        ]
        self.assertEqual(deltas, ["via the other field"])

    def test_tool_call_without_id_gets_one_synthesised(self):
        translator = translate.StreamTranslator(model="m")
        events = drain(
            translator,
            [
                self.chunk({"tool_calls": [{"index": 0, "function": {"name": "Read"}}]}),
                self.chunk(finish="tool_calls"),
            ],
        )
        start = [p for n, p in events if n == "content_block_start"][0]
        self.assertTrue(start["content_block"]["id"].startswith("toolu_"))

    def test_usage_only_chunk_with_no_choices(self):
        translator = translate.StreamTranslator(model="m")
        events = drain(
            translator,
            [
                self.chunk({"content": "hi"}),
                {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 1}},
            ],
        )
        self.assertEqual(dict(events)["message_delta"]["usage"]["input_tokens"], 5)


class TestErrorsAndCounting(unittest.TestCase):
    def test_status_maps_to_anthropic_error_type(self):
        self.assertEqual(
            translate.error_body(429, "slow down")["error"]["type"], "rate_limit_error"
        )
        self.assertEqual(translate.error_body(401, "nope")["error"]["type"], "authentication_error")
        self.assertEqual(translate.error_body(418, "?")["error"]["type"], "api_error")

    def test_count_tokens_scales_with_input(self):
        small = translate.count_tokens({"messages": [{"role": "user", "content": "hi"}]})
        large = translate.count_tokens(
            {"messages": [{"role": "user", "content": "word " * 1000}], "system": "sys"}
        )
        self.assertGreater(large, small)
        self.assertGreaterEqual(small, 1)


if __name__ == "__main__":
    unittest.main()


class TestToolResultImages(unittest.TestCase):
    """Claude Code returns Read-on-an-image as a tool_result image block.
    OpenAI tool messages are text-only, so the image has to be hoisted."""

    REQUEST = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": "AAAA",
                                },
                            }
                        ],
                    }
                ],
            }
        ]
    }

    def test_image_is_hoisted_into_a_following_user_message(self):
        out = translate.anthropic_to_openai(self.REQUEST, model="m", allow_images=True)
        self.assertEqual([m["role"] for m in out["messages"]], ["tool", "user"])
        self.assertEqual(out["messages"][0]["tool_call_id"], "t1")
        parts = out["messages"][1]["content"]
        self.assertEqual(parts[1]["image_url"]["url"], "data:image/png;base64,AAAA")

    def test_tool_message_is_never_left_empty(self):
        out = translate.anthropic_to_openai(self.REQUEST, model="m", allow_images=True)
        self.assertTrue(out["messages"][0]["content"].strip())

    def test_stripped_for_text_only_models(self):
        out = translate.anthropic_to_openai(self.REQUEST, model="m", allow_images=False)
        self.assertEqual([m["role"] for m in out["messages"]], ["tool"])
        self.assertNotIn("image_url", json.dumps(out))

    def test_text_and_image_results_both_survive(self):
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": [
                                {"type": "text", "text": "screenshot taken"},
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/jpeg",
                                        "data": "BBBB",
                                    },
                                },
                            ],
                        }
                    ],
                }
            ]
        }
        out = translate.anthropic_to_openai(request, model="m", allow_images=True)
        self.assertIn("screenshot taken", out["messages"][0]["content"])
        self.assertEqual(
            out["messages"][1]["content"][1]["image_url"]["url"], "data:image/jpeg;base64,BBBB"
        )
