"""Tests for the OpenAITokenStats plugin."""

import json
import re

import pytest
from tiktoken import get_encoding

from app.plugins.openai_token_stats import OpenAITokenStats
from tests.app.conftest import (
    CHAT_WIRE,
    RESPONSES,
    RESPONSES_WIRE,
    PublishedPanels,
    Wire,
    parse_sse_payloads,
    responses_request,
    sse_delta_content,
    sse_done,
    sse_event,
    sse_reasoning,
    sse_reasoning_content_null,
    sse_tool_call_null_fields,
    sse_usage,
)

_CHAT_COMPLETIONS = "/v1/chat/completions"
_COMPLETIONS = "/v1/completions"

_REQUEST_ID = 1
_OTHER_REQUEST_ID = 2
# A request the plugin was never told about, so it holds no state for it.
_UNKNOWN_REQUEST_ID = 99
_OK = 200

_PANEL_TITLE = "Token Usage"
# A chat request body with nothing in it the plugin needs to count.
_CHAT_BODY = b'{"messages": []}'

# Usage figures small enough to be rendered as plain digits.
_PROMPT_TOKENS = 10
_COMPLETION_TOKENS = 20
# A second, distinct pair, for the tests that add two usage reports together.
_MORE_PROMPT_TOKENS = 5
_MORE_COMPLETION_TOKENS = 15
# A third pair, for the second of two requests in flight at once.
_OTHER_PROMPT_TOKENS = 30
_OTHER_COMPLETION_TOKENS = 40
# The total upstream reports beside a pair, which it counts for itself.
_TOTAL_TOKENS = _PROMPT_TOKENS + _COMPLETION_TOKENS
# Large enough that the panel renders them compactly, as "1.2K" and "3.4K".
_BIG_PROMPT_TOKENS = 1200
_BIG_COMPLETION_TOKENS = 3400

# Enough generated text to make one update cycle's rate clearly non-zero.
_BURST = "hello world, this is a decent stretch of generated text"


async def _start(
    plugin: OpenAITokenStats, body: bytes = _CHAT_BODY, path: str = _CHAT_COMPLETIONS,
    request_id: int = _REQUEST_ID,
) -> bytes:
    """Announce one request to the plugin.

    Args:
        plugin: The plugin under test.
        body: The raw request body.
        path: The request path.
        request_id: The id the response chunks will quote.

    Returns:
        The body the plugin forwards upstream.
    """
    _, forwarded, _ = await plugin.on_request(request_id, {}, body, path)
    return forwarded


async def _chunk(
    plugin: OpenAITokenStats, data: bytes, path: str = _CHAT_COMPLETIONS,
    request_id: int = _REQUEST_ID, is_final: bool = False,
) -> bytes:
    """Feed one response chunk to the plugin.

    Args:
        plugin: The plugin under test.
        data: The raw response bytes.
        path: The request path.
        request_id: The request the chunk belongs to.
        is_final: Whether this chunk ends the response.

    Returns:
        The bytes the plugin forwards downstream.
    """
    return await plugin.on_response_chunk(request_id, {}, data, path, _OK, is_final)


async def _report_usage(
    plugin: OpenAITokenStats, prompt_tokens: int, completion_tokens: int,
    total_tokens: int = 0, path: str = _CHAT_COMPLETIONS, request_id: int = _REQUEST_ID,
) -> None:
    """Run one whole request whose only content is a closing usage report.

    Args:
        plugin: The plugin under test.
        prompt_tokens: The input tokens upstream reports.
        completion_tokens: The output tokens upstream reports.
        total_tokens: The total upstream reports, or 0 to omit the field.
        path: The request path.
        request_id: The id to run the request under.
    """
    body = _CHAT_BODY if path.startswith("/v1/chat") else b'{"prompt": "hi"}'
    await _start(plugin, body=body, path=path, request_id=request_id)
    await _chunk(plugin, sse_usage(prompt_tokens, completion_tokens, total_tokens), path=path,
           request_id=request_id, is_final=True)


def _last_panel(published: PublishedPanels) -> tuple[str, str]:
    """The most recent panel the plugin published.

    Args:
        published: Everything recorded so far.

    Returns:
        Its (title, html) pair.
    """
    return published[-1]


async def _panel_after_chunks(
    plugin: OpenAITokenStats, published: PublishedPanels, *chunks: bytes,
) -> tuple[str, str]:
    """Feed chunks to a request already under way, then read the panel they produce.

    Args:
        plugin: The plugin under test.
        published: The list its UI callback records into.
        chunks: The raw response bytes, in arrival order.

    Returns:
        The (title, html) pair of the panel published by the following flush.
    """
    for chunk in chunks:
        await _chunk(plugin, chunk)
    plugin.update_tps_and_publish()
    return _last_panel(published)


async def _panel_after_usage(
    plugin: OpenAITokenStats, published: PublishedPanels, prompt_tokens: int,
    completion_tokens: int,
) -> tuple[str, str]:
    """Run one request reporting the given usage, then read the panel it produces.

    Args:
        plugin: The plugin under test.
        published: The list its UI callback records into.
        prompt_tokens: The input tokens upstream reports.
        completion_tokens: The output tokens upstream reports.

    Returns:
        The (title, html) pair of the panel published by the closing flush.
    """
    await _report_usage(plugin, prompt_tokens, completion_tokens)
    plugin.update_tps_and_publish()
    return _last_panel(published)


def _stats(published: PublishedPanels) -> dict[str, float]:
    """The token counts shown by the last panel that carries a Stats row.

    Args:
        published: Everything recorded so far.

    Returns:
        The input, output and total figures, all zero when no panel showed a
        Stats row at all — which is how the plugin renders "nothing counted yet".
    """
    stats_html = ""
    for title, html in published:
        if title == _PANEL_TITLE and "Stats" in html:
            stats_html = html
    if not stats_html:
        return {"input": 0, "output": 0, "total": 0}
    return {
        field: _figure(stats_html, field)
        for field in ("input", "output", "total")
    }


def _figure(html: str, field: str) -> float:
    """One number out of a Stats row, as rendered.

    Args:
        html: The published panel body.
        field: Which figure to read: input, output or total.

    Returns:
        The number, or 0 when the row does not show that field.
    """
    label = {"input": "in", "output": "out", "total": "total"}[field]
    match = re.search(rf"([\d.]+)\s+{label}", html)
    return float(match.group(1)) if match else 0


def _tps(html: str, which: str) -> int:
    """The generating tokens-per-second figure a panel shows.

    Args:
        html: The published panel body.
        which: Either "current" or "peak".

    Returns:
        The rate.

    Raises:
        ValueError: If the panel has no generating row to read.
    """
    match = re.search(rf">Generating</span>.*?>(\d+) t/s {which}</span>", html)
    if match is None:
        raise ValueError(f"published panel shows no generating {which} rate")
    return int(match.group(1))


class TestOnRequestForceStream:
    """on_request forces stream=true and injects include_usage."""

    async def test_forces_stream_true(
        self, token_stats_plugin: tuple[OpenAITokenStats, PublishedPanels],
    ) -> None:
        """The plugin needs a stream with usage to count anything, so it asks for one."""
        plugin, _ = token_stats_plugin
        body = b'{"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}'
        data = json.loads(await _start(plugin, body=body))
        assert data["stream"] is True
        assert data["stream_options"]["include_usage"] is True

    async def test_non_completions_path_unchanged(
        self, token_stats_plugin: tuple[OpenAITokenStats, PublishedPanels],
    ) -> None:
        """A path the plugin does not watch is forwarded byte for byte."""
        plugin, _ = token_stats_plugin
        body = b'{"key": "value"}'
        assert await _start(plugin, body=body, path="/v1/models") == body

    async def test_invalid_json_unchanged(
        self, token_stats_plugin: tuple[OpenAITokenStats, PublishedPanels],
    ) -> None:
        """A body that will not parse is forwarded rather than rewritten."""
        plugin, _ = token_stats_plugin
        body = b"not json"
        assert await _start(plugin, body=body) == body

    async def test_includes_stream_options_when_missing(
        self, token_stats_plugin: tuple[OpenAITokenStats, PublishedPanels],
    ) -> None:
        """A request with no stream_options at all gets the whole key created."""
        plugin, _ = token_stats_plugin
        data = json.loads(await _start(plugin, body=b'{"model": "gpt-4", "messages": []}'))
        assert data["stream_options"] == {"include_usage": True}

    async def test_includes_usage_when_stream_options_present(
        self, token_stats_plugin: tuple[OpenAITokenStats, PublishedPanels],
    ) -> None:
        """An existing stream_options is added to, not replaced."""
        plugin, _ = token_stats_plugin
        body = b'{"model": "gpt-4", "stream_options": {"max_tokens": 100}, "messages": []}'
        data = json.loads(await _start(plugin, body=body))
        assert data["stream_options"] == {"max_tokens": 100, "include_usage": True}


class TestStateTracking:
    """Per-request state transitions verified through observable behavior."""

    async def test_streaming_passthrough_intermediate_chunk(
        self, token_stats_plugin: tuple[OpenAITokenStats, PublishedPanels],
    ) -> None:
        """A streaming client sees its chunks unchanged as they pass through."""
        plugin, _ = token_stats_plugin
        await _start(plugin, body=b'{"stream": true}')
        chunk = sse_delta_content("1", "x")
        assert await _chunk(plugin, chunk) == chunk

    async def test_state_removed_final_streaming(
        self, token_stats_plugin: tuple[OpenAITokenStats, PublishedPanels],
    ) -> None:
        """After the final chunk the plugin holds nothing, so late bytes pass through."""
        plugin, _ = token_stats_plugin
        await _start(plugin, body=b'{"stream": true}')
        await _chunk(plugin, sse_usage(_PROMPT_TOKENS, _COMPLETION_TOKENS), is_final=True)
        assert await _chunk(plugin, b"extra data") == b"extra data"

    async def test_state_removed_final_non_streaming(
        self, token_stats_plugin: tuple[OpenAITokenStats, PublishedPanels],
    ) -> None:
        """The same holds for a buffered request, which would otherwise swallow them."""
        plugin, _ = token_stats_plugin
        await _start(plugin, body=b'{"stream": false}')
        await _chunk(plugin, sse_usage(_PROMPT_TOKENS, _COMPLETION_TOKENS), is_final=True)
        assert await _chunk(plugin, b"extra data") == b"extra data"

    async def test_null_reasoning_content_does_not_raise(
        self, token_stats_plugin: tuple[OpenAITokenStats, PublishedPanels],
    ) -> None:
        """Content deltas with a null reasoning content relay through."""
        plugin, _ = token_stats_plugin
        await _start(plugin, body=b'{"stream": true}')
        chunk = sse_reasoning_content_null("1", "hello")
        assert await _chunk(plugin, chunk) == chunk

    async def test_null_tool_call_fields_do_not_raise(
        self, token_stats_plugin: tuple[OpenAITokenStats, PublishedPanels],
    ) -> None:
        """Tool call deltas with a null name and a null id relay through."""
        plugin, _ = token_stats_plugin
        await _start(plugin, body=b'{"stream": true}')
        chunk = sse_tool_call_null_fields("tc1", "read_file", {"path": "/etc/hostname"})
        assert await _chunk(plugin, chunk) == chunk

    async def test_missing_state_returns_chunk_unchanged(
        self, token_stats_plugin: tuple[OpenAITokenStats, PublishedPanels],
    ) -> None:
        """Chunks for a request the plugin never saw are none of its business."""
        plugin, _ = token_stats_plugin
        assert await _chunk(plugin, b"some data", request_id=_UNKNOWN_REQUEST_ID) == b"some data"


class TestSseUsageParsing:
    """Token counts extracted from SSE events and reflected in published stats."""

    async def test_extract_usage_from_sse_chunk(
        self, token_stats_plugin: tuple[OpenAITokenStats, PublishedPanels],
    ) -> None:
        """A usage event upstream sent becomes the figures the panel shows."""
        plugin, published = token_stats_plugin
        await _report_usage(plugin, _PROMPT_TOKENS, _COMPLETION_TOKENS, _TOTAL_TOKENS)
        plugin.update_tps_and_publish()
        stats = _stats(published)
        assert stats["input"] == _PROMPT_TOKENS
        assert stats["output"] == _COMPLETION_TOKENS
        assert stats["total"] == _TOTAL_TOKENS

    async def test_multiple_sse_events_in_one_chunk(
        self, token_stats_plugin: tuple[OpenAITokenStats, PublishedPanels],
    ) -> None:
        """Two usage events arriving in one chunk are both counted."""
        plugin, published = token_stats_plugin
        await _start(plugin)
        await _chunk(plugin,
               sse_usage(_PROMPT_TOKENS, _COMPLETION_TOKENS)
               + sse_usage(_MORE_PROMPT_TOKENS, _MORE_COMPLETION_TOKENS),
               is_final=True)
        plugin.update_tps_and_publish()
        stats = _stats(published)
        assert stats["input"] == _PROMPT_TOKENS + _MORE_PROMPT_TOKENS
        assert stats["output"] == _COMPLETION_TOKENS + _MORE_COMPLETION_TOKENS


class TestPathFiltering:
    """Non-chat/completions paths are ignored."""

    @pytest.mark.parametrize("path", ["/v1/models", "/v1/embeddings", "/"])
    async def test_skip_non_completions_path(
        self, token_stats_plugin: tuple[OpenAITokenStats, PublishedPanels], path: str,
    ) -> None:
        """Usage reported on a path the plugin does not watch is not counted.

        The request is announced on a watched path first, so that what stops the
        count is the path of the chunk itself rather than an unrecognised request.
        """
        plugin, published = token_stats_plugin
        await _start(plugin)
        await _chunk(plugin, sse_usage(_PROMPT_TOKENS, _COMPLETION_TOKENS), path=path, is_final=True)
        plugin.update_tps_and_publish()
        stats = _stats(published)
        assert stats["input"] == 0
        assert stats["output"] == 0


class TestStreamingAccumulation:
    """SSE usage accumulates across multiple chunks."""

    async def test_accumulate_across_chunks(
        self, token_stats_plugin: tuple[OpenAITokenStats, PublishedPanels],
    ) -> None:
        """Usage reported in separate chunks adds up rather than replacing."""
        plugin, published = token_stats_plugin
        await _start(plugin)
        await _chunk(plugin, sse_usage(_PROMPT_TOKENS, _COMPLETION_TOKENS))
        await _chunk(plugin, sse_usage(_MORE_PROMPT_TOKENS, _MORE_COMPLETION_TOKENS), is_final=True)
        plugin.update_tps_and_publish()
        stats = _stats(published)
        assert stats["input"] == _PROMPT_TOKENS + _MORE_PROMPT_TOKENS
        assert stats["output"] == _COMPLETION_TOKENS + _MORE_COMPLETION_TOKENS


class TestNonStreamingReassembly:
    """Non-streaming requests buffer SSE and reassemble on final chunk."""

    async def test_non_streaming_returns_assembled_json(
        self, token_stats_plugin: tuple[OpenAITokenStats, PublishedPanels],
    ) -> None:
        """A client that asked for no stream gets one whole chat completion back."""
        plugin, published = token_stats_plugin
        body = json.dumps({
            "model": "gpt-4", "stream": False,
            "messages": [{"role": "user", "content": "hi"}],
        }).encode()
        await _start(plugin, body=body)

        opening = sse_event({
            "id": "abc", "object": "chat.completion.chunk",
            "choices": [{"delta": {"role": "assistant", "content": "hello"}}],
        })
        assert await _chunk(plugin, opening) == b""

        closing = sse_event({"choices": [{"delta": {"content": " world"}, "finish_reason": "stop"}]})
        closing += sse_usage(_MORE_PROMPT_TOKENS, _PROMPT_TOKENS)
        closing += sse_done()
        result = await _chunk(plugin, closing, is_final=True)
        assert json.loads(result) == {
            "id": "abc",
            "object": "chat.completion",
            "created": 0,
            "model": "",
            "choices": [{
                "index": 0,
                "message": {"content": "hello world"},
                "finish_reason": "stop",
            }],
        }

        plugin.update_tps_and_publish()
        stats = _stats(published)
        assert stats["input"] == _MORE_PROMPT_TOKENS
        assert stats["output"] == _PROMPT_TOKENS

    async def test_non_streaming_text_completion_returns_text_field(
        self, token_stats_plugin: tuple[OpenAITokenStats, PublishedPanels],
    ) -> None:
        """Non-streaming /v1/completions returns text format, not message.format."""
        plugin, _ = token_stats_plugin
        body = json.dumps({"model": "gpt-4", "prompt": "hello", "stream": False}).encode()
        await _start(plugin, body=body, path=_COMPLETIONS)

        opening = sse_event({
            "id": "abc", "object": "text_completion.chunk",
            "choices": [{"delta": {"content": " world"}}],
        })
        assert await _chunk(plugin, opening, path=_COMPLETIONS) == b""

        closing = sse_event({"choices": [{"delta": {"content": "!"}, "finish_reason": "stop"}]})
        closing += sse_usage(_PROMPT_TOKENS, _COMPLETION_TOKENS)
        closing += sse_done()
        result = await _chunk(plugin, closing, path=_COMPLETIONS, is_final=True)
        assert json.loads(result) == {
            "id": "abc",
            "object": "text_completion",
            "created": 0,
            "model": "",
            "choices": [{"text": " world!", "index": 0, "finish_reason": "stop"}],
        }

    async def test_non_streaming_reassembles_tool_call_from_null_field_deltas(
        self, token_stats_plugin: tuple[OpenAITokenStats, PublishedPanels],
    ) -> None:
        """Tool call deltas with null fields reassemble into one whole call."""
        plugin, _ = token_stats_plugin
        await _start(plugin, body=b'{"stream": false}')

        assert await _chunk(
            plugin, sse_tool_call_null_fields("tc1", "read_file", {"path": "/etc/hostname"})
        ) == b""

        closing = sse_event({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})
        closing += sse_done()
        parsed = json.loads(await _chunk(plugin, closing, is_final=True))

        tool_calls = parsed["choices"][0]["message"]["tool_calls"]
        assert len(tool_calls) == 1
        call = tool_calls[0]
        assert call["id"] == "tc1"
        assert call["function"]["name"] == "read_file"
        assert json.loads(call["function"]["arguments"]) == {"path": "/etc/hostname"}

    async def test_non_streaming_reassembles_despite_null_reasoning_content(
        self, token_stats_plugin: tuple[OpenAITokenStats, PublishedPanels],
    ) -> None:
        """Content deltas with a null reasoning content reassemble into one body."""
        plugin, _ = token_stats_plugin
        await _start(plugin, body=b'{"stream": false}')

        opening = sse_event({
            "id": "abc", "object": "chat.completion.chunk",
            "choices": [{"delta": {
                "role": "assistant", "content": "hello", "reasoning_content": None,
            }}],
        })
        assert await _chunk(plugin, opening) == b""

        closing = sse_event({"choices": [{
            "delta": {"content": " world", "reasoning_content": None},
            "finish_reason": "stop",
        }]})
        closing += sse_done()
        result = await _chunk(plugin, closing, is_final=True)

        assert not result.startswith(b"data: ")
        parsed = json.loads(result)
        assert parsed["object"] == "chat.completion"
        assert parsed["choices"][0]["message"]["content"] == "hello world"
        assert parsed["choices"][0]["finish_reason"] == "stop"

    async def test_streaming_returns_chunk_as_is(
        self, token_stats_plugin: tuple[OpenAITokenStats, PublishedPanels],
    ) -> None:
        """A streaming request is never buffered, so nothing is reassembled."""
        plugin, _ = token_stats_plugin
        await _start(plugin, body=b'{"stream": true}')
        chunk = b'data: {"choices": [{"delta": {"content": "x"}}]}\n'
        assert await _chunk(plugin, chunk) == chunk

    async def test_non_streaming_returns_raw_buffer_when_no_events(
        self, token_stats_plugin: tuple[OpenAITokenStats, PublishedPanels],
    ) -> None:
        """With nothing parseable to reassemble, the buffered bytes are handed back."""
        plugin, _ = token_stats_plugin
        await _start(plugin, body=b'{"stream": false}')
        assert await _chunk(plugin, b"not valid sse at all", is_final=True) == b"not valid sse at all"


class TestPublishTiming:
    """Publish is called via update_tps_and_publish."""

    async def test_publish_called_via_update_tps_and_publish(
        self, token_stats_plugin: tuple[OpenAITokenStats, PublishedPanels],
    ) -> None:
        """The flush publishes a panel under the plugin's own title."""
        plugin, published = token_stats_plugin
        await _report_usage(plugin, _PROMPT_TOKENS, _COMPLETION_TOKENS)
        # Spelled out rather than routed through _panel_after_usage: the flush is
        # the subject of this test, not scaffolding for it.
        plugin.update_tps_and_publish()
        assert _last_panel(published)[0] == _PANEL_TITLE

    async def test_html_format_contains_tokens_per_second(
        self, token_stats_plugin: tuple[OpenAITokenStats, PublishedPanels],
    ) -> None:
        """The panel carries a generating row with both a current and a peak rate."""
        plugin, published = token_stats_plugin
        title, html = await _panel_after_usage(
            plugin, published, _PROMPT_TOKENS, _COMPLETION_TOKENS)
        assert title == _PANEL_TITLE
        assert "Generating" in html
        assert "t/s current" in html
        assert "t/s peak" in html

    async def test_html_format_contains_compact_numbers(
        self, token_stats_plugin: tuple[OpenAITokenStats, PublishedPanels],
    ) -> None:
        """Counts in the thousands are abbreviated rather than printed in full."""
        plugin, published = token_stats_plugin
        title, html = await _panel_after_usage(
            plugin, published, _BIG_PROMPT_TOKENS, _BIG_COMPLETION_TOKENS)
        assert title == _PANEL_TITLE
        assert "3.4K out" in html
        assert "1.2K in" in html


class TestSseDoneSkip:
    """SSE [DONE] marker is handled correctly."""

    async def test_skip_done_line(
        self, token_stats_plugin: tuple[OpenAITokenStats, PublishedPanels],
    ) -> None:
        """The terminator is not JSON, and must not disturb the counts beside it."""
        plugin, published = token_stats_plugin
        await _start(plugin)
        await _chunk(plugin, sse_usage(_PROMPT_TOKENS, _COMPLETION_TOKENS) + b"data: [DONE]\n",
               is_final=True)
        plugin.update_tps_and_publish()
        stats = _stats(published)
        assert stats["input"] == _PROMPT_TOKENS
        assert stats["output"] == _COMPLETION_TOKENS


class TestPublishNoneGuard:
    """Publishing is skipped entirely when no UI has been wired up."""

    async def test_no_crash_without_publish(self) -> None:
        """A plugin that was never given a publish callback still serves requests."""
        plugin = OpenAITokenStats({})
        await _start(plugin)
        result = await _chunk(plugin, sse_usage(_PROMPT_TOKENS, _COMPLETION_TOKENS), is_final=True)
        assert json.loads(result)["object"] == "chat.completion"
        assert plugin.is_request_ongoing(_REQUEST_ID) is False


class TestDeltaTextAccumulation:
    """SSE delta text is accumulated for TPS tracking."""

    async def test_accumulates_content(
        self, token_stats_plugin: tuple[OpenAITokenStats, PublishedPanels],
    ) -> None:
        """Content deltas feed the generating rate."""
        plugin, published = token_stats_plugin
        await _start(plugin)
        title, html = await _panel_after_chunks(plugin, published, sse_delta_content("1", _BURST))
        assert title == _PANEL_TITLE
        assert _tps(html, "current") > 0

    async def test_accumulates_reasoning(
        self, token_stats_plugin: tuple[OpenAITokenStats, PublishedPanels],
    ) -> None:
        """Reasoning deltas count towards the rate as much as content does."""
        plugin, published = token_stats_plugin
        await _start(plugin)
        title, html = await _panel_after_chunks(plugin, published, sse_reasoning("1", _BURST))
        assert title == _PANEL_TITLE
        assert _tps(html, "current") > 0

    async def test_accumulates_across_chunks(
        self, token_stats_plugin: tuple[OpenAITokenStats, PublishedPanels],
    ) -> None:
        """Text split over two chunks is rated as one stretch, not two."""
        plugin, published = token_stats_plugin
        await _start(plugin)
        title, html = await _panel_after_chunks(
            plugin, published, sse_delta_content("1", "hel"), sse_delta_content("1", "lo"))
        assert title == _PANEL_TITLE
        assert _tps(html, "current") == 1

    async def test_handles_none_content(
        self, token_stats_plugin: tuple[OpenAITokenStats, PublishedPanels],
    ) -> None:
        """An explicit null content is nothing to rate, and must not raise."""
        plugin, published = token_stats_plugin
        await _start(plugin)
        title, html = await _panel_after_chunks(
            plugin, published, sse_event({"choices": [{"delta": {"content": None}}]}))
        assert title == _PANEL_TITLE
        assert _tps(html, "current") == 0

    async def test_handles_missing_delta(
        self, token_stats_plugin: tuple[OpenAITokenStats, PublishedPanels],
    ) -> None:
        """An event with no delta at all is likewise nothing to rate."""
        plugin, published = token_stats_plugin
        await _start(plugin)
        title, html = await _panel_after_chunks(
            plugin, published, sse_usage(_PROMPT_TOKENS, _COMPLETION_TOKENS))
        assert title == _PANEL_TITLE
        assert _tps(html, "current") == 0

    async def test_handles_multiple_choices(
        self, token_stats_plugin: tuple[OpenAITokenStats, PublishedPanels],
    ) -> None:
        """Every choice in one event contributes its text to the rate.

        Measured against a one-choice cycle rather than against zero, because a
        rate above zero would not tell the second choice's text from the first's.
        """
        plugin, published = token_stats_plugin
        await _start(plugin)
        _, one_choice_html = await _panel_after_chunks(
            plugin, published, sse_event({"choices": [{"delta": {"content": _BURST}}]}))
        one_choice = _tps(one_choice_html, "current")

        title, html = await _panel_after_chunks(plugin, published, sse_event({
            "choices": [{"delta": {"content": _BURST}}, {"delta": {"content": _BURST}}],
        }))
        assert title == _PANEL_TITLE
        assert _tps(html, "current") > one_choice


class TestConcurrentIsolation:
    """Concurrent requests maintain isolated state."""

    async def test_concurrent_requests_independent_counts(
        self, token_stats_plugin: tuple[OpenAITokenStats, PublishedPanels],
    ) -> None:
        """Two requests in flight contribute to one running total."""
        plugin, published = token_stats_plugin
        await _start(plugin, request_id=_REQUEST_ID)
        await _start(plugin, request_id=_OTHER_REQUEST_ID)

        await _chunk(plugin, sse_usage(_PROMPT_TOKENS, _COMPLETION_TOKENS),
               request_id=_REQUEST_ID, is_final=True)
        await _chunk(plugin, sse_usage(_OTHER_PROMPT_TOKENS, _OTHER_COMPLETION_TOKENS),
               request_id=_OTHER_REQUEST_ID, is_final=True)

        plugin.update_tps_and_publish()
        stats = _stats(published)
        assert stats["input"] == _PROMPT_TOKENS + _OTHER_PROMPT_TOKENS
        assert stats["output"] == _COMPLETION_TOKENS + _OTHER_COMPLETION_TOKENS

    async def test_concurrent_requests_independent_decoders(
        self, token_stats_plugin: tuple[OpenAITokenStats, PublishedPanels],
    ) -> None:
        """A half-delivered event on one request cannot corrupt another's parse."""
        plugin, published = token_stats_plugin
        await _start(plugin, request_id=_REQUEST_ID)
        await _start(plugin, request_id=_OTHER_REQUEST_ID)

        await _chunk(plugin, b'data: {"', request_id=_REQUEST_ID)
        await _chunk(plugin, sse_usage(_MORE_PROMPT_TOKENS, _PROMPT_TOKENS),
               request_id=_OTHER_REQUEST_ID, is_final=True)

        plugin.update_tps_and_publish()
        stats = _stats(published)
        assert stats["input"] == _MORE_PROMPT_TOKENS
        assert stats["output"] == _PROMPT_TOKENS


class TestTPSLiveLoop:
    """TPS publishes via background loop on 1s intervals."""

    async def test_sse_without_is_final_publishes_nonzero_tps(
        self, token_stats_plugin: tuple[OpenAITokenStats, PublishedPanels],
    ) -> None:
        """A stream still in flight already has a rate worth showing."""
        plugin, published = token_stats_plugin
        await _start(plugin)
        title, html = await _panel_after_chunks(plugin, published, sse_delta_content("1", "hello world"))
        assert title == _PANEL_TITLE
        assert _tps(html, "current") > 0


class TestPublishStatsWithTPS:
    """The published panel renders a generating row, and a stats row once non-zero."""

    async def test_publish_has_stats_in_html(
        self, token_stats_plugin: tuple[OpenAITokenStats, PublishedPanels],
    ) -> None:
        """Once tokens have been counted, both rows are present."""
        plugin, published = token_stats_plugin
        title, html = await _panel_after_usage(
            plugin, published, _PROMPT_TOKENS, _COMPLETION_TOKENS)
        assert title == _PANEL_TITLE
        assert "Generating" in html
        assert "Stats" in html

    async def test_publish_hides_stats_row_when_zero(
        self, token_stats_plugin: tuple[OpenAITokenStats, PublishedPanels],
    ) -> None:
        """A report of zero tokens leaves the stats row out rather than showing zeros."""
        plugin, published = token_stats_plugin
        title, html = await _panel_after_usage(plugin, published, 0, 0)
        assert title == _PANEL_TITLE
        assert "Generating" in html
        assert "Stats" not in html

    async def test_publish_tps_zero(
        self, token_stats_plugin: tuple[OpenAITokenStats, PublishedPanels],
    ) -> None:
        """With no generated text, the rate reads zero rather than going missing."""
        plugin, published = token_stats_plugin
        title, html = await _panel_after_usage(plugin, published, 0, 0)
        assert title == _PANEL_TITLE
        assert _tps(html, "current") == 0

    async def test_peak_tps_tracks_maximum(
        self, token_stats_plugin: tuple[OpenAITokenStats, PublishedPanels],
    ) -> None:
        """Peak holds the best rate seen so far, not the most recent one."""
        plugin, published = token_stats_plugin
        await _start(plugin)
        _, burst_html = await _panel_after_chunks(plugin, published, sse_delta_content("1", _BURST))
        peak = _tps(burst_html, "peak")
        assert peak > 1

        title, html = await _panel_after_chunks(plugin, published, sse_delta_content("1", "x"))
        assert title == _PANEL_TITLE
        assert _tps(html, "current") == 1
        assert _tps(html, "peak") == peak


class TestName:
    """Plugin returns correct name."""

    async def test_name_returns_correct_value(
        self, token_stats_plugin: tuple[OpenAITokenStats, PublishedPanels],
    ) -> None:
        """The plugin names itself after its module."""
        plugin, _ = token_stats_plugin
        assert plugin.name() == "openai_token_stats"


class TestCompletionsPath:
    """Both /v1/chat/completions and /v1/completions paths are handled."""

    async def test_v1_completions_path_handled(
        self, token_stats_plugin: tuple[OpenAITokenStats, PublishedPanels],
    ) -> None:
        """The legacy completions path is counted the same as the chat one."""
        plugin, published = token_stats_plugin
        await _report_usage(plugin, _MORE_PROMPT_TOKENS, _PROMPT_TOKENS, path=_COMPLETIONS)
        plugin.update_tps_and_publish()
        stats = _stats(published)
        assert stats["input"] == _MORE_PROMPT_TOKENS
        assert stats["output"] == _PROMPT_TOKENS


class TestOngoingRequestLifecycle:
    """Request lifecycle tracks ongoing state via public is_request_ongoing."""

    async def test_on_request_adds_to_ongoing(
        self, token_stats_plugin: tuple[OpenAITokenStats, PublishedPanels],
    ) -> None:
        """on_request adds the request_id to ongoing tracking."""
        plugin, _ = token_stats_plugin
        assert plugin.is_request_ongoing(_REQUEST_ID) is False
        await _start(plugin, body=b'{"model": "gpt-4", "messages": []}')
        assert plugin.is_request_ongoing(_REQUEST_ID) is True

    async def test_on_response_chunk_final_removes_from_ongoing(
        self, token_stats_plugin: tuple[OpenAITokenStats, PublishedPanels],
    ) -> None:
        """on_response_chunk with is_final=True removes the request_id from ongoing tracking."""
        plugin, _ = token_stats_plugin
        await _start(plugin, body=b'{"model": "gpt-4", "messages": []}')
        await _chunk(plugin, b"", is_final=True)
        assert plugin.is_request_ongoing(_REQUEST_ID) is False

    async def test_on_cancelled_removes_from_ongoing(
        self, token_stats_plugin: tuple[OpenAITokenStats, PublishedPanels],
    ) -> None:
        """on_cancelled removes the request_id from ongoing tracking."""
        plugin, _ = token_stats_plugin
        await _start(plugin, body=b'{"model": "gpt-4", "messages": []}')
        plugin.on_cancelled(_REQUEST_ID)
        assert plugin.is_request_ongoing(_REQUEST_ID) is False


# ── Same work on every API ──


_ENCODING = get_encoding("cl100k_base")
_HISTORY = [
    {"role": "system", "content": "be brief"},
    {"role": "user", "content": "what is the weather"},
    {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
]


def _tokens(text: str) -> int:
    """The token count of the given text."""
    return len(_ENCODING.encode(text, allowed_special="all"))


def _stage_counts(html: str) -> tuple[int, int]:
    """How many requests a panel shows as prefilling and as generating."""
    prefilling = re.search(r"(\d+) prefilling", html)
    generating = re.search(r"(\d+) generating", html)
    return int(prefilling.group(1)), int(generating.group(1))


def _prefill_rate(html: str) -> int:
    """The prefill rate a panel shows for the last minute."""
    match = re.search(r"(\d+) t/s last min", html)
    return int(match.group(1))


async def _non_streaming_reply(plugin: OpenAITokenStats, wire: Wire) -> tuple[dict[str, str], dict, bytes]:
    """Send a non-streaming request and its streamed reply through the plugin.

    Returns:
        The response headers, the one reply the client gets, and the streamed reply.
    """
    await _start(plugin, wire.request(_HISTORY, stream=False), wire.path)
    headers = plugin.on_response_headers(_REQUEST_ID, {"Content-Type": "text/event-stream"}, _OK)
    reply = wire.reply("1", ("content", "answer"))
    assert await _chunk(plugin, reply, wire.path) == b""
    assembled = json.loads(await _chunk(plugin, b"", wire.path, is_final=True))
    return headers, assembled, reply


class TestApiParity:
    """Token stats are counted the same way on every API."""

    @pytest.mark.parametrize(
        "wire, has_stream_options",
        [pytest.param(CHAT_WIRE, True, id="chat"), pytest.param(RESPONSES_WIRE, False, id="responses")],
    )
    async def test_forces_stream(
        self, wire: Wire, has_stream_options: bool, token_stats_plugin: tuple[OpenAITokenStats, PublishedPanels],
    ) -> None:
        """The request is sent on as a stream, with stream options only on chat."""
        plugin, _ = token_stats_plugin
        forwarded = json.loads(await _start(plugin, wire.request(_HISTORY, stream=False), wire.path))
        assert forwarded["stream"] is True
        assert ("stream_options" in forwarded) is has_stream_options

    async def test_request_tokens_feed_the_prefill_rate(
        self, wire: Wire, token_stats_plugin: tuple[OpenAITokenStats, PublishedPanels],
    ) -> None:
        """Prompt tokens count toward the prefill rate."""
        plugin, published = token_stats_plugin
        await _start(plugin, wire.request(_HISTORY), wire.path)
        attempt = wire.attempt()
        await _chunk(plugin, attempt.start() + attempt.content(_BURST), wire.path)
        plugin.update_tps_and_publish()
        html = _last_panel(published)[1]
        assert _prefill_rate(html) == _tokens("be brief\nwhat is the weather\nsunny\n")
        assert _stage_counts(html) == (0, 1)

    @pytest.mark.parametrize(
        "piece, generated",
        [
            pytest.param(("content", _BURST), _BURST, id="content"),
            pytest.param(("reasoning", _BURST), _BURST, id="reasoning"),
            pytest.param(
                ("tool_call", "c1", "lookup_weather", {"city": _BURST}),
                "lookup_weather" + json.dumps({"city": _BURST}),
                id="tool_call",
            ),
        ],
    )
    async def test_generated_text_feeds_the_generate_rate(
        self, wire: Wire, piece: tuple, generated: str, token_stats_plugin: tuple[OpenAITokenStats, PublishedPanels],
    ) -> None:
        """Each kind of generated text counts toward the generate rate."""
        plugin, published = token_stats_plugin
        await _start(plugin, wire.request(_HISTORY), wire.path)
        attempt = wire.attempt()
        _, html = await _panel_after_chunks(plugin, published, attempt.start() + attempt.pieces([piece]))
        assert _tps(html, "current") == _tokens(generated)

    async def test_usage_is_counted(self, wire: Wire, token_stats_plugin: tuple[OpenAITokenStats, PublishedPanels]) -> None:
        """The usage the reply reports is shown."""
        plugin, published = token_stats_plugin
        await _start(plugin, wire.request(_HISTORY), wire.path)
        attempt = wire.attempt()
        reply = attempt.start() + attempt.content("hi") + attempt.finish("stop", (_PROMPT_TOKENS, _COMPLETION_TOKENS, _TOTAL_TOKENS))
        await _chunk(plugin, reply, wire.path, is_final=True)
        assert _stats(published) == {"input": _PROMPT_TOKENS, "output": _COMPLETION_TOKENS, "total": _TOTAL_TOKENS}

    async def test_non_streaming_chat_client_gets_one_reply(
        self, token_stats_plugin: tuple[OpenAITokenStats, PublishedPanels],
    ) -> None:
        """A chat client that did not ask to stream gets one JSON reply."""
        plugin, _ = token_stats_plugin
        headers, assembled, _ = await _non_streaming_reply(plugin, CHAT_WIRE)
        assert headers == {"Content-Type": "application/json"}
        assert assembled["choices"][0]["message"]["content"] == "answer"

    async def test_non_streaming_responses_client_gets_one_reply(
        self, token_stats_plugin: tuple[OpenAITokenStats, PublishedPanels],
    ) -> None:
        """A Responses client that did not ask to stream gets the final response as one JSON reply."""
        plugin, _ = token_stats_plugin
        headers, assembled, reply = await _non_streaming_reply(plugin, RESPONSES_WIRE)
        assert headers == {"Content-Type": "application/json"}
        assert assembled == parse_sse_payloads(reply)[-1]["response"]

    async def test_non_object_responses_item_counts_no_request_tokens(
        self, token_stats_plugin: tuple[OpenAITokenStats, PublishedPanels],
    ) -> None:
        """A Responses input holding a bare string item counts no prompt tokens and still streams."""
        plugin, published = token_stats_plugin
        forwarded = json.loads(await _start(plugin, responses_request(["loose"], stream=False), RESPONSES))
        assert forwarded["stream"] is True
        attempt = RESPONSES_WIRE.attempt()
        await _chunk(plugin, attempt.start() + attempt.content(_BURST), RESPONSES)
        plugin.update_tps_and_publish()
        html = _last_panel(published)[1]
        assert _prefill_rate(html) == 0
        assert _stage_counts(html) == (0, 1)

    @pytest.mark.parametrize("sub_path", ["/input_tokens"])
    async def test_sub_path_with_json_body_is_counted_and_passed_through(
        self, wire: Wire, sub_path: str, token_stats_plugin: tuple[OpenAITokenStats, PublishedPanels],
    ) -> None:
        """A sub path request is streamed and its plain JSON reply is passed on."""
        plugin, _ = token_stats_plugin
        path = wire.path + sub_path
        forwarded = json.loads(await _start(plugin, wire.request(_HISTORY, stream=None), path))
        assert forwarded["stream"] is True
        reply = b'{"object": "response.input_tokens", "input_tokens": 12}'
        assert await _chunk(plugin, reply, path) == b""
        assert await _chunk(plugin, b"", path, is_final=True) == reply

    async def test_body_less_retrieve_passes_through(self, token_stats_plugin: tuple[OpenAITokenStats, PublishedPanels]) -> None:
        """A request with no body and its reply pass through as they are."""
        plugin, _ = token_stats_plugin
        body = b""
        assert await _start(plugin, body, RESPONSES + "/resp_1") is body
        assert await _chunk(plugin, b'{"error": "not found"}', RESPONSES + "/resp_1", is_final=True) == b'{"error": "not found"}'


class TestGeneratingTransition:
    """The last choice of a chunk decides whether generation has started."""

    @pytest.mark.parametrize(
        "choices, stages",
        [
            ([{"delta": {"content": _BURST}}, {"delta": {}}], (1, 0)),
            ([{"delta": {}}, {"delta": {"content": _BURST}}], (0, 1)),
        ],
        ids=["last-choice-empty", "last-choice-text"],
    )
    async def test_last_choice_decides(
        self, choices: list[dict], stages: tuple[int, int], token_stats_plugin: tuple[OpenAITokenStats, PublishedPanels],
    ) -> None:
        """The last choice of a chunk decides whether the request shows as generating."""
        plugin, published = token_stats_plugin
        await _start(plugin)
        _, html = await _panel_after_chunks(plugin, published, sse_event({"choices": choices}))
        assert _stage_counts(html) == stages
        assert _tps(html, "current") == _tokens(_BURST)
