from unittest.mock import patch

from ucode.agents.claude_oss.probe import run_probe


def test_probe_reports_translation_required_when_anthropic_route_rejects_oss_model() -> None:
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


def test_probe_returns_error_code_when_no_model_to_probe() -> None:
    with patch("ucode.agents.claude_oss.probe._post") as post:
        code = run_probe("https://example.cloud.databricks.com", "token", None)
    post.assert_not_called()
    assert code == 1


def test_probe_calls_post_five_times_against_the_expected_routes_in_order() -> None:
    workspace = "https://example.cloud.databricks.com"
    model = "databricks-glm-5-2"
    responses = [
        (400, '{"message": "API type not supported"}'),  # anthropic route rejects
        (200, '{"choices": [{"message": {"content": "ok"}}]}'),  # oss non-streaming
        (200, "data: [DONE]\n"),  # oss streaming
        (200, '{"choices": [{"message": {"tool_calls": [{}]}}]}'),  # tool calling
        (400, '{"message": "max_tokens too large"}'),  # info-only oversized check
    ]
    with patch("ucode.agents.claude_oss.probe._post", side_effect=responses) as post:
        code = run_probe(workspace, "token", model)

    assert code == 0
    assert post.call_count == 5
    anthropic_call, oss_non_streaming, oss_streaming, tool_call, oversized_call = (
        post.call_args_list
    )

    assert anthropic_call.args[0] == f"{workspace}/ai-gateway/anthropic/v1/messages"
    oss_url = f"{workspace}/ai-gateway/mlflow/v1/chat/completions"
    for call in (oss_non_streaming, oss_streaming, tool_call, oversized_call):
        assert call.args[0] == oss_url

    # Only the streaming probe passes stream=True; every other call leaves it at the default.
    assert oss_streaming.kwargs.get("stream") is True
    for call in (anthropic_call, oss_non_streaming, tool_call, oversized_call):
        assert call.kwargs.get("stream", False) is False

    # The tool-calling probe is the only one that sends a `tools` payload.
    assert "tools" in tool_call.args[2]
    for call in (anthropic_call, oss_non_streaming, oss_streaming, oversized_call):
        assert "tools" not in call.args[2]

    # Every payload targets the already-resolved model; run_probe never rediscovers it.
    for call in post.call_args_list:
        assert call.args[2]["model"] == model


def test_probe_forces_the_tool_call_in_the_capability_check() -> None:
    """`tool_choice: "auto"` lets a tool-capable model answer in text instead
    of calling get_weather, and the probe would then report a false failure
    (it checks `"tool_calls" in body`). Force the specific function so the
    check can't be defeated by the model simply choosing not to call it."""
    workspace = "https://example.cloud.databricks.com"
    responses = [
        (400, '{"message": "API type not supported"}'),
        (200, '{"choices": [{"message": {"content": "ok"}}]}'),
        (200, 'data: {"choices": [{"delta": {"content": "1"}}]}\n\ndata: [DONE]\n'),
        (200, '{"choices": [{"message": {"tool_calls": [{}]}}]}'),
        (400, '{"message": "max_tokens too large"}'),
    ]
    with patch("ucode.agents.claude_oss.probe._post", side_effect=responses) as post:
        run_probe(workspace, "token", "databricks-glm-5-2")

    tool_call = post.call_args_list[3]
    assert tool_call.args[2]["tool_choice"] == {
        "type": "function",
        "function": {"name": "get_weather"},
    }


def test_probe_reports_streaming_failure_for_a_plain_json_response(capsys) -> None:
    """`server.py`'s _stream ignores anything that isn't an SSE `data:` line -
    a normal JSON body reaching Claude Code that way is silently dropped, not
    delivered. The probe must not report "PASS" for HTTP 200 alone."""
    workspace = "https://example.cloud.databricks.com"
    responses = [
        (400, '{"message": "API type not supported"}'),
        (200, '{"choices": [{"message": {"content": "ok"}}]}'),
        (200, '{"ok": true}'),  # HTTP 200, but no SSE "data:" line at all
        (200, '{"choices": [{"message": {"tool_calls": [{}]}}]}'),
        (400, '{"message": "max_tokens too large"}'),
    ]
    with patch("ucode.agents.claude_oss.probe._post", side_effect=responses):
        code = run_probe(workspace, "token", "databricks-glm-5-2")

    assert code == 0  # run_probe completes regardless of individual results
    out = capsys.readouterr().out
    assert "[FAIL] OSS route, streaming" in out


def test_probe_reports_streaming_success_for_a_real_sse_data_event(capsys) -> None:
    workspace = "https://example.cloud.databricks.com"
    responses = [
        (400, '{"message": "API type not supported"}'),
        (200, '{"choices": [{"message": {"content": "ok"}}]}'),
        (200, 'data: {"choices": [{"delta": {"content": "1"}}]}\n\ndata: [DONE]\n'),
        (200, '{"choices": [{"message": {"tool_calls": [{}]}}]}'),
        (400, '{"message": "max_tokens too large"}'),
    ]
    with patch("ucode.agents.claude_oss.probe._post", side_effect=responses):
        run_probe(workspace, "token", "databricks-glm-5-2")
    out = capsys.readouterr().out
    assert "[PASS] OSS route, streaming" in out
