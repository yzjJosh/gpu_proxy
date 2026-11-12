"""Tests for the OpenAIChatRequestModifier plugin."""

import json
from typing import Callable

import pytest

from app.plugins.base import PluginBase
from app.plugins.openai_chat_modifier import OpenAIChatRequestModifier
from app.plugins.openai_token_stats import OpenAITokenStats
from tests.app.conftest import (
    CHAT_COMPLETIONS,
    RESPONSES,
    Wire,
    apply_on_request,
    json_body,
    make_chat_modifier,
    sse_usage,
)

_Config = dict[str, str | dict]
# The (title, html) pairs a plugin handed to its UI publish callback.
_Published = list[tuple[str, str]]

# The two paths the modifier claims, and two it must keep its hands off.
_COMPLETIONS = "/v1/completions"
_EMBEDDINGS = "/v1/embeddings"
_HEALTH = "/health"
_CHAT_COMPLETIONS_QUERY = "/v1/chat/completions?stream=true"

_REQUEST_ID = 1
_OK = 200
_JSON_HEADERS = {"Content-Type": "application/json"}
_SSE_HEADERS = {"Content-Type": "text/event-stream"}

# The temperature the modifier's config injects; no request body under test sets it.
_OVERRIDE_TEMPERATURE = 0.7
# A request-only field, large enough to be worth carrying through the merge.
_MAX_TOKENS = 100
# The usage an upstream stream reports, echoed back in the published stats row.
_PROMPT_TOKENS = 50
_COMPLETION_TOKENS = 25


def _recorder() -> tuple[Callable[[str, str], None], _Published]:
    """A UI publish callback paired with the panels it has recorded so far.

    Returns:
        The (callback, published) pair; the list grows as the plugin publishes.
    """
    published: _Published = []

    def publish(title: str, html: str) -> None:
        published.append((title, html))

    return publish, published


async def _stream_usage(plugin: PluginBase, usage: bytes) -> None:
    """Feed one usage-bearing SSE chunk, then the empty chunk that ends the stream.

    Args:
        plugin: The plugin watching the response.
        usage: The SSE event carrying the token counts.
    """
    await plugin.on_response_chunk(_REQUEST_ID, dict(_SSE_HEADERS), usage, CHAT_COMPLETIONS,
                             _OK, False)
    await plugin.on_response_chunk(_REQUEST_ID, dict(_SSE_HEADERS), b"", CHAT_COMPLETIONS,
                             _OK, True)


class TestDeepMergeNestedOverridesThroughPipeline:
    """Nested dict and list overrides work through the full plugin pipeline."""

    async def test_nested_overrides_through_full_pipeline(self) -> None:
        """Nesting survives the decode, the merge and the re-encode."""
        plugin = make_chat_modifier({
            "response_format": {"type": "json_schema"}, "tools": [{"name": "new_tool"}]})
        body = json_body({
            "response_format": {"type": "text", "schema": {"name": "old"}},
            "tools": [{"name": "old_tool", "desc": "find"}]})
        _, result, _ = await apply_on_request(plugin, body)
        assert isinstance(result, bytes)
        merged = json.loads(result)
        assert merged["response_format"] == {"type": "json_schema", "schema": {"name": "old"}}
        assert merged["tools"] == [{"name": "new_tool", "desc": "find"}]


class TestOpenAIChatRequestModifierAppliesOverridesToJsonBody:
    """Valid JSON body gets overrides merged."""

    async def test_on_request_applies_overrides_to_json_body(self) -> None:
        """Overridden and request-only keys both appear in the outgoing body."""
        plugin = make_chat_modifier({"model": "gpt-4", "temperature": _OVERRIDE_TEMPERATURE})
        _, result, _ = await apply_on_request(plugin, json_body({"model": "old", "max_tokens": _MAX_TOKENS}))
        assert json.loads(result) == {
            "model": "gpt-4",
            "temperature": _OVERRIDE_TEMPERATURE,
            "max_tokens": _MAX_TOKENS,
        }


class TestOpenAIChatRequestModifierPassesThroughOnInvalidJson:
    """Non-JSON body is left unchanged."""

    async def test_on_request_passes_through_on_invalid_json(self) -> None:
        """A body that will not parse is forwarded exactly as it arrived."""
        body = b"not valid json"
        _, result, path = await apply_on_request(make_chat_modifier({"model": "gpt-4"}), body)
        assert result == body
        assert path == CHAT_COMPLETIONS


class TestOpenAIChatRequestModifierPassesThroughOnEmptyOverrides:
    """No overrides means body is unchanged."""

    async def test_on_request_passes_through_on_empty_overrides(self) -> None:
        """A config with no overrides key at all leaves the body alone."""
        plugin = OpenAIChatRequestModifier({})
        body = json_body({"model": "old"})
        _, result, path = await apply_on_request(plugin, body)
        assert result is body
        assert path == CHAT_COMPLETIONS


class TestOpenAIChatRequestModifierReturnsModifiedBytes:
    """on_request returns modified bytes directly."""

    async def test_on_request_returns_modified_bytes(self) -> None:
        """The merged body comes back as bytes, not as a dict for someone else to encode."""
        _, result, _ = await apply_on_request(make_chat_modifier({"model": "gpt-4"}), json_body({"model": "old"}))
        assert isinstance(result, bytes)
        assert json.loads(result) == {"model": "gpt-4"}


class TestOpenAIChatRequestModifierName:
    """name() returns the configured name or default."""

    @pytest.mark.parametrize("config,expected", [
        ({"name": "my_modifier"}, "my_modifier"),
        ({}, "openai_chat_modifier"),
    ])
    async def test_name_returns_configured_or_default(self, config: _Config, expected: str) -> None:
        """A configured name wins; without one the plugin names itself."""
        assert OpenAIChatRequestModifier(config).name() == expected


class TestDeepMergeNonDictJsonPassThrough:
    """Non-dict JSON bodies pass through unchanged."""

    @pytest.mark.parametrize("body", [
        b'[1,2,3]',
        b'"string"',
        b'42',
        b'true',
        b'null',
    ])
    async def test_non_dict_json_passes_through(self, body: bytes) -> None:
        """There is nowhere to merge overrides into a JSON value that is not an object."""
        _, result, path = await apply_on_request(make_chat_modifier({"model": "gpt-4"}), body)
        assert result == body
        assert path == CHAT_COMPLETIONS


class TestDeepMergeEmptyBytesBody:
    """Empty bytes body passes through unchanged."""

    async def test_empty_bytes_body(self) -> None:
        """An empty body is not valid JSON, so it is forwarded untouched."""
        _, result, path = await apply_on_request(make_chat_modifier({"model": "gpt-4"}), b"")
        assert result == b""
        assert path == CHAT_COMPLETIONS


class TestOpenAIChatRequestModifierUpdatesContentLength:
    """on_request updates Content-Length header when body is modified."""

    async def test_content_length_updated_on_modification(self) -> None:
        """A rewritten body carries its own length, not the length it arrived with."""
        plugin = make_chat_modifier({"model": "gpt-4", "temperature": _OVERRIDE_TEMPERATURE})
        body = json_body({"model": "old"})
        headers = dict(_JSON_HEADERS)
        headers["Content-Length"] = str(len(body))
        result_headers, result, _ = await apply_on_request(plugin, body, headers=headers)
        assert result_headers["Content-Length"] == str(len(result))
        assert result_headers["Content-Length"] != str(len(body))
        assert result_headers["Content-Type"] == "application/json"

    async def test_content_length_not_updated_on_pass_through(self) -> None:
        """A body that was never rewritten keeps the length the client sent."""
        body = b"not json"
        headers = {"Content-Length": str(len(body))}
        result_headers, result, result_path = await apply_on_request(
            make_chat_modifier({"model": "gpt-4"}), body, headers=headers)
        assert result_headers["Content-Length"] == str(len(body))
        assert result is body
        assert result_path == CHAT_COMPLETIONS


class TestCompletionPathGating:
    """Overrides only apply to /v1/chat/completions and /v1/completions paths."""

    async def test_chat_completions_path_applies_overrides(self) -> None:
        """The chat completions path is one the modifier claims."""
        _, result, path = await apply_on_request(make_chat_modifier({"model": "gpt-4"}), json_body({"model": "old"}))
        assert json.loads(result) == {"model": "gpt-4"}
        assert path == CHAT_COMPLETIONS

    async def test_completions_path_applies_overrides(self) -> None:
        """The legacy completions path is claimed too."""
        _, result, path = await apply_on_request(make_chat_modifier({"model": "gpt-4"}), json_body({"model": "old"}), _COMPLETIONS)
        assert json.loads(result) == {"model": "gpt-4"}
        assert path == _COMPLETIONS

    async def test_embeddings_path_passes_through(self) -> None:
        """An embeddings request belongs to a different modifier, so this one skips it."""
        body = json_body({"model": "old"})
        _, result, path = await apply_on_request(make_chat_modifier({"model": "gpt-4"}), body, _EMBEDDINGS)
        assert result == body
        assert path == _EMBEDDINGS

    async def test_health_path_passes_through(self) -> None:
        """A non-OpenAI path is left alone even when its body would parse."""
        body = json_body({"model": "old"})
        _, result, path = await apply_on_request(make_chat_modifier({"model": "gpt-4"}), body, _HEALTH)
        assert result == body
        assert path == _HEALTH

    async def test_chat_completions_with_query_string_applies_overrides(self) -> None:
        """A query string after a claimed path does not stop the match."""
        _, result, _ = await apply_on_request(make_chat_modifier({"model": "gpt-4"}), json_body({"model": "old"}),
                              _CHAT_COMPLETIONS_QUERY)
        assert json.loads(result) == {"model": "gpt-4"}


class TestRegressionModifierWithTokenStats:
    """Regression: OpenAIChatRequestModifier and OpenAITokenStats coexist without interference."""

    async def test_modifier_and_token_stats_coexist_in_pipeline(self) -> None:
        """Both plugins work together: modifier transforms request, token stats tracks response."""
        modifier = make_chat_modifier({"model": "gpt-4", "temperature": _OVERRIDE_TEMPERATURE})
        token_stats = OpenAITokenStats({})
        body = json_body({"model": "old-model", "max_tokens": _MAX_TOKENS, "stream": True})

        new_headers, new_body, new_path = await apply_on_request(modifier, body, headers=dict(_JSON_HEADERS))
        assert json.loads(new_body) == {
            "model": "gpt-4",
            "temperature": _OVERRIDE_TEMPERATURE,
            "max_tokens": _MAX_TOKENS,
            "stream": True,
        }

        _, out_body, out_path = await apply_on_request(token_stats, new_body, new_path, new_headers)
        forwarded = json.loads(out_body)
        assert forwarded["model"] == "gpt-4"
        assert forwarded["temperature"] == _OVERRIDE_TEMPERATURE
        assert forwarded["max_tokens"] == _MAX_TOKENS
        assert forwarded["stream_options"] == {"include_usage": True}
        assert forwarded["stream_options"]["include_usage"] is True
        assert out_path == new_path

        publish, published = _recorder()
        token_stats.init_ui(publish)
        await _stream_usage(token_stats, sse_usage(
            prompt_tokens=_PROMPT_TOKENS, completion_tokens=_COMPLETION_TOKENS))
        token_stats.update_tps_and_publish()

        title, html = published[-1]
        assert title == "Token Usage"
        assert f"{_PROMPT_TOKENS} in" in html
        assert f"{_COMPLETION_TOKENS} out" in html

    @pytest.mark.parametrize("path,should_modify", [
        (CHAT_COMPLETIONS, True),
        (_COMPLETIONS, True),
        (_EMBEDDINGS, False),
        (_HEALTH, False),
    ])
    async def test_modifier_unaffected_by_new_hooks_on_various_paths(
        self, path: str, should_modify: bool,
    ) -> None:
        """OpenAIChatRequestModifier behavior is unchanged regardless of new hooks on PluginBase."""
        body = json_body({"model": "old"})
        _, result, _ = await apply_on_request(make_chat_modifier({"model": "gpt-4"}), body, path)
        if should_modify:
            assert json.loads(result) == {"model": "gpt-4"}
        else:
            assert result == body

    async def test_token_stats_does_not_affect_modifier_overrides(self) -> None:
        """Adding OpenAITokenStats to the pipeline does not change modifier output."""
        modifier = make_chat_modifier({
            "response_format": {"type": "json_schema"}, "tools": [{"name": "new_tool"}]})
        token_stats = OpenAITokenStats({})
        body = json_body({
            "response_format": {"type": "text", "schema": {"name": "old"}},
            "tools": [{"name": "old_tool", "desc": "find"}]})

        _, modified_body, _ = await apply_on_request(modifier, body)
        _, final_body, _ = await apply_on_request(token_stats, modified_body)

        forwarded = json.loads(final_body)
        assert forwarded["response_format"] == {"type": "json_schema", "schema": {"name": "old"}}
        assert forwarded["tools"] == [{"name": "new_tool", "desc": "find"}]

    async def test_token_stats_init_ui_does_not_break_modifier(self) -> None:
        """Calling init_ui on OpenAITokenStats has no effect on OpenAIChatRequestModifier."""
        publish, published = _recorder()
        modifier = make_chat_modifier({"model": "gpt-4"})
        token_stats = OpenAITokenStats({})
        token_stats.init_ui(publish)

        _, result, _ = await apply_on_request(modifier, json_body({"model": "old"}))
        assert json.loads(result) == {"model": "gpt-4"}
        assert published == []

        await apply_on_request(token_stats, json_body({"model": "gpt-4"}))
        await _stream_usage(token_stats, sse_usage(prompt_tokens=1, completion_tokens=2))
        token_stats.update_tps_and_publish()
        assert any("Generating" in html for _, html in published)
        assert any("Stats" in html for _, html in published)


class TestOngoingRequestLifecycle:
    """OpenAIChatRequestModifier tracks ongoing requests via base class lifecycle hooks."""

    async def test_on_request_adds_to_ongoing(self) -> None:
        """A request is only ongoing once it has been handed to the plugin."""
        plugin = OpenAIChatRequestModifier({})
        assert not plugin.is_request_ongoing(_REQUEST_ID)
        await apply_on_request(plugin, json_body({"model": "gpt-4"}))
        assert plugin.is_request_ongoing(_REQUEST_ID)

    async def test_on_response_chunk_final_removes_from_ongoing(self) -> None:
        """The final response chunk closes the request out."""
        plugin = OpenAIChatRequestModifier({})
        await apply_on_request(plugin, json_body({"model": "gpt-4"}))
        await plugin.on_response_chunk(_REQUEST_ID, {}, b"", CHAT_COMPLETIONS, _OK, is_final=True)
        assert not plugin.is_request_ongoing(_REQUEST_ID)

    async def test_on_cancelled_removes_from_ongoing(self) -> None:
        """A cancelled request stops being ongoing without any final chunk."""
        plugin = OpenAIChatRequestModifier({})
        await apply_on_request(plugin, json_body({"model": "gpt-4"}))
        plugin.on_cancelled(_REQUEST_ID)
        assert not plugin.is_request_ongoing(_REQUEST_ID)


# ── Same work on every API ──


_CAP = 100
_CLIENT_CAP = 64
_COMPLETION_CAP = 200
_NATIVE_CAP = 999


class TestApiParity:
    """Overrides apply the same way on every API."""

    async def test_overrides_apply_on_every_api(self, wire: Wire) -> None:
        """Configured overrides land in the body on every API."""
        plugin = make_chat_modifier({"model": "pinned", "temperature": _OVERRIDE_TEMPERATURE, "chat_template_kwargs": {"enable_thinking": False}})
        body = wire.request([{"role": "user", "content": "hi"}])
        headers, result, _ = await apply_on_request(plugin, body, wire.path)
        merged = json.loads(result)
        assert merged["model"] == "pinned"
        assert merged["temperature"] == _OVERRIDE_TEMPERATURE
        assert merged["chat_template_kwargs"] == {"enable_thinking": False}
        assert headers["Content-Length"] == str(len(result))

    @pytest.mark.parametrize("suffix", ["/input_tokens"])
    async def test_sub_path_with_json_body_gets_the_overrides(self, wire: Wire, suffix: str) -> None:
        """A sub path request with a JSON body gets the overrides."""
        plugin = make_chat_modifier({"model": "pinned"})
        _, result, _ = await apply_on_request(plugin, wire.request([{"role": "user", "content": "hi"}]), wire.path + suffix)
        assert json.loads(result)["model"] == "pinned"

    async def test_body_less_retrieve_passes_through(self) -> None:
        """A request with no body is passed on as it came."""
        body = b""
        _, result, _ = await apply_on_request(make_chat_modifier({"model": "pinned"}), body, RESPONSES + "/resp_1")
        assert result is body


# Configured overrides, the fields a Responses client sends, and the fields the
# Responses request then carries.
_RESPONSES_PRECEDENCE = [
    pytest.param({"max_tokens": _CAP}, {}, {"max_output_tokens": _CAP}, id="max-tokens-applies"),
    pytest.param({"max_tokens": _CAP}, {"max_tokens": _CLIENT_CAP}, {"max_output_tokens": _CAP}, id="max-tokens-over-client-max-tokens"),
    pytest.param({"max_tokens": _CAP}, {"max_output_tokens": _CLIENT_CAP}, {"max_output_tokens": _CLIENT_CAP}, id="max-tokens-yields"),
    pytest.param({"max_tokens": _CAP, "max_output_tokens": _NATIVE_CAP}, {}, {"max_output_tokens": _CAP}, id="native-beside-max-tokens-unwritten"),
    pytest.param({"max_tokens": _CAP, "max_output_tokens": _NATIVE_CAP}, {"max_output_tokens": _CLIENT_CAP}, {"max_output_tokens": _CLIENT_CAP}, id="native-does-not-force"),
    pytest.param({"max_completion_tokens": _COMPLETION_CAP}, {"max_output_tokens": _CLIENT_CAP}, {"max_output_tokens": _COMPLETION_CAP}, id="max-completion-forced"),
    pytest.param({"max_completion_tokens": _COMPLETION_CAP, "max_tokens": _CAP, "max_output_tokens": _NATIVE_CAP}, {}, {"max_output_tokens": _COMPLETION_CAP}, id="max-completion-wins"),
    pytest.param({"max_output_tokens": _NATIVE_CAP}, {"max_output_tokens": _CLIENT_CAP}, {"max_output_tokens": _NATIVE_CAP}, id="native-alone-forced"),
    pytest.param({"reasoning_effort": "low"}, {"reasoning": {"effort": "high", "summary": "auto"}}, {"reasoning": {"effort": "low", "summary": "auto"}}, id="effort-over-client"),
    pytest.param({"reasoning_effort": "low", "reasoning": {"effort": "high", "summary": "detailed"}}, {}, {"reasoning": {"effort": "low", "summary": "detailed"}}, id="effort-replaces-native"),
    pytest.param({"reasoning": {"effort": "high"}}, {}, {"reasoning": {"effort": "high"}}, id="native-effort-alone"),
]


class TestResponsesOverridePrecedence:
    """Configured overrides and client fields mix in a fixed order."""

    @pytest.mark.parametrize("overrides, client, expected", _RESPONSES_PRECEDENCE)
    async def test_precedence(self, overrides: dict, client: dict, expected: dict) -> None:
        """A Responses request gets the cap and effort chat would apply."""
        body = json_body({"model": "m", "input": "hi", **client})
        _, result, _ = await apply_on_request(make_chat_modifier(overrides), body, RESPONSES)
        merged = json.loads(result)
        assert {field: merged.get(field) for field in expected} == expected
        assert {"max_tokens", "max_completion_tokens", "reasoning_effort"} & set(merged) == set(client) & {"max_tokens"}

    async def test_yielding_only_override_leaves_the_body_alone(self) -> None:
        """Nothing to apply means the body is passed on as it came, not re-encoded."""
        body = json_body({"input": "hi", "max_output_tokens": _CLIENT_CAP})
        _, result, _ = await apply_on_request(make_chat_modifier({"max_tokens": _CAP}), body, RESPONSES)
        assert result is body

    @pytest.mark.parametrize(
        "overrides",
        [
            {"max_tokens": _CAP, "max_output_tokens": _NATIVE_CAP},
            {"max_completion_tokens": _COMPLETION_CAP, "max_tokens": _CAP},
            {"reasoning_effort": "low", "reasoning": {"effort": "high"}},
        ],
    )
    async def test_chat_merges_the_config_as_written(self, overrides: dict) -> None:
        """Chat merges the overrides with the names they were given."""
        body = json_body({"messages": [], "max_completion_tokens": _CLIENT_CAP})
        _, result, _ = await apply_on_request(make_chat_modifier(overrides), body)
        assert json.loads(result) == {"messages": [], "max_completion_tokens": _CLIENT_CAP, **overrides}
