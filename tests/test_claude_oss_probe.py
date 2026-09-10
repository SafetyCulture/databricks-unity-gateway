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


def test_probe_returns_error_code_when_no_model_to_probe():
    with patch("ucode.agents.claude_oss.probe._post") as post:
        code = run_probe("https://example.cloud.databricks.com", "token", None)
    post.assert_not_called()
    assert code == 1


def test_probe_calls_post_five_times_against_the_expected_routes_in_order():
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
