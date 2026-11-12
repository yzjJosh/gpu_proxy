import asyncio
import json
import logging
from typing import Awaitable, Callable, Optional
from unittest.mock import MagicMock, AsyncMock

import aiohttp.web as web
import pytest

from tests.app.conftest import (
    CHAT_WIRE,
    RESPONSES_WIRE,
    Wire,
    parse_sse_payloads,
    responses_request,
    sse_content,
    sse_done,
    sse_event,
    sse_finish_reason,
    sse_reasoning,
    sse_tool_call,
    sse_tool_call_null_fields,
    sse_usage,
    extract_sse_field,
    extract_sse_tool_call_args,
    extract_sse_tool_call_name,
)
from app.plugins.openai_guard_rail import OpenAIGuardRail
from app.responses import AbstractMutatingResponse

logging.getLogger("plugins.openai_guard_rail").setLevel(logging.DEBUG)


def _make_client_response() -> MagicMock:
    """Create a mock client response that records write calls."""
    response = MagicMock(spec=AbstractMutatingResponse)
    response.prepare = AsyncMock()
    response.write = AsyncMock()
    response.write_eof = AsyncMock()
    response.set_status = MagicMock()
    response.prepared = True
    response.headers = {}
    return response


def _make_web_request() -> MagicMock:
    """Create a mock web.Request."""
    return MagicMock()


def _request(messages: list[dict], stream: Optional[bool] = True) -> bytes:
    """Build a chat request body carrying the given message history.

    Args:
        messages: The conversation the client sends.
        stream: What the client asks for in the "stream" field, or None to
            leave the field out of the request entirely.

    Returns:
        The encoded request body.
    """
    body: dict[str, str | list[dict] | bool] = {"model": "test", "messages": messages}
    if stream is not None:
        body["stream"] = stream
    return json.dumps(body).encode()


def _user_message(content: str = "do something") -> dict[str, str]:
    """A user turn, as a fresh dict the caller may keep.

    Args:
        content: What the user asks for. Defaults to the opening request the
            multi-turn tests share.

    Returns:
        The user history message.
    """
    return {"role": "user", "content": content}


def _assistant_message(content: str, reasoning: str = "") -> dict[str, str]:
    """An assistant history message carrying prose and no tool call.

    Args:
        content: The answer the model gave, omitted from the message when empty.
        reasoning: The thinking the model emitted, omitted when empty.

    Returns:
        The assistant history message.
    """
    message: dict[str, str] = {"role": "assistant"}
    if reasoning:
        message["reasoning_content"] = reasoning
    if content:
        message["content"] = content
    return message


def _assistant_tool_call(
    name: str,
    arguments: str,
    tc_id: str = "tc1",
    reasoning: str = "",
    content: str = "",
) -> dict[str, str | list]:
    """An assistant history message carrying one tool call.

    Args:
        name: The tool the model asked to run.
        arguments: The tool arguments, already encoded as JSON text.
        tc_id: The id the model gave the tool call.
        reasoning: The thinking that preceded the call, omitted when empty.
        content: The prose that preceded the call, omitted when empty.

    Returns:
        The assistant history message.
    """
    message: dict[str, str | list] = {"role": "assistant"}
    if reasoning:
        message["reasoning_content"] = reasoning
    if content:
        message["content"] = content
    message["tool_calls"] = [
        {"id": tc_id, "type": "function", "function": {"name": name, "arguments": arguments}}
    ]
    return message


def _tool_result(content: str, tool_call_id: str = "") -> dict[str, str]:
    """A tool-result history message answering an earlier tool call.

    Args:
        content: What the tool returned.
        tool_call_id: The tool call being answered, omitted when empty.

    Returns:
        The tool history message.
    """
    message: dict[str, str] = {"role": "tool"}
    if tool_call_id:
        message["tool_call_id"] = tool_call_id
    message["content"] = content
    return message


STREAM_REQUEST = _request([])
NON_STREAM_REQUEST = _request([], stream=False)
NO_STREAM_FIELD_REQUEST = _request([], stream=None)
FIRST_TURN = _request([_user_message()])
TOOL_ARGS = '{"path":"config.yaml","mode":"strict_validation"}'
CHAT_COMPLETIONS = "/v1/chat/completions"
THINKING_ONLY = "thinking_only"

# Repeating blocks sized around the 50 characters a repeat must span before the
# guard rail calls it a loop.
LOOP_BLOCK = "repeating_loop_content_" * 3   # 69 chars
MIN_SIZE_BLOCK = "a" * 50                    # exactly at the minimum
LONG_BLOCK = "repeating_block_content_" * 4  # 96 chars

_ForwardFn = Callable[
    [dict[str, str], bytes, str, web.Request, AbstractMutatingResponse], Awaitable[Optional[bool]]
]


def _written(client: MagicMock) -> bytes:
    """Join every chunk the plugin wrote to the client response."""
    return b"".join(call[0][0] for call in client.write.call_args_list)


def _capture_ui(plugin: OpenAIGuardRail) -> list[tuple[str, str]]:
    """Attach a UI publisher to the plugin and return the list it fills."""
    published: list[tuple[str, str]] = []

    def publish(title: str, html: str) -> None:
        published.append((title, html))

    plugin.init_ui(publish)
    return published


class _Upstream:
    """An upstream that answers each attempt with the next scripted reply.

    The guard rail retries by issuing a fresh request, so a test lists one
    reply per attempt it expects and afterwards asserts how many it took. The
    last reply is repeated if the plugin retries more often than that.
    """

    def __init__(self, *replies: bytes) -> None:
        self._replies = replies
        self.calls = 0
        self.bodies: list[bytes] = []
        self.headers: list[dict[str, str]] = []

    async def forward(
        self,
        headers: dict[str, str],
        body: bytes,
        _path: str,
        web_req: web.Request,
        resp: AbstractMutatingResponse,
    ) -> None:
        """Record the request and stream back the reply scripted for it."""
        self.calls += 1
        self.bodies.append(body)
        self.headers.append(headers)
        await resp.prepare(web_req)
        await resp.write(self._replies[min(self.calls, len(self._replies)) - 1])
        await resp.write_eof()


def _forwarded_body(upstream: _Upstream) -> dict:
    """The request body the upstream received on its most recent attempt."""
    return json.loads(upstream.bodies[-1])


def _forwarded_messages(upstream: _Upstream) -> list[dict]:
    """The message history the upstream received on its most recent attempt."""
    return _forwarded_body(upstream)["messages"]


def _forwarded_assistant_messages(upstream: _Upstream) -> list[dict]:
    """The assistant messages the upstream received on its most recent attempt."""
    return [m for m in _forwarded_messages(upstream) if m.get("role") == "assistant"]


def _forwarded_system_message(upstream: _Upstream) -> str:
    """The system message the upstream received on its most recent attempt."""
    return next((m["content"] for m in _forwarded_messages(upstream) if m.get("role") == "system"), "")


def _assert_retry_names_tool_call(upstream: _Upstream, name: str) -> None:
    """Assert the retry told the model which tool call it took issue with."""
    system_msg = _forwarded_system_message(upstream)
    assert "The tool call was:" in system_msg
    assert name in system_msg


def _loop_reply(block: str = LOOP_BLOCK) -> bytes:
    """A reply whose content repeats often enough for the guard rail to call it a loop.

    Args:
        block: The text every repeat carries.

    Returns:
        The scripted reply.
    """
    return sse_content("1", block) * 3 + sse_done()


def _reasoning_only_reply(finish_reason: str = "stop") -> bytes:
    """A reply that thinks and then stops, leaving the turn with no final output.

    Args:
        finish_reason: The reason the reply finishes on, or "" to end the
            stream without announcing one at all.

    Returns:
        The scripted reply.
    """
    reply = sse_reasoning("1", THINKING_ONLY)
    if finish_reason:
        reply += sse_finish_reason("1", finish_reason)
    return reply + sse_done()


def _final_answer_reply() -> bytes:
    """A reply carrying the final output the previous attempt owed."""
    return (
        sse_content("1", "final_answer")
        + sse_finish_reason("1", "stop")
        + sse_done()
    )


def _recovered_reply(chat_id: str = "1", finish_reason: str = "") -> bytes:
    """A reply that answers cleanly, as a retry does once the guard rail intervened.

    Args:
        chat_id: The id every event of the reply carries.
        finish_reason: The reason the reply finishes on, or "" to end the
            stream without announcing one at all.

    Returns:
        The scripted reply.
    """
    reply = sse_content(chat_id, "recovered")
    if finish_reason:
        reply += sse_finish_reason(chat_id, finish_reason)
    return reply + sse_done()


def _tool_call_reply(
    name: str,
    chat_id: str,
    tc_id: str,
    reasoning: str = "",
    content: str = "",
    arguments: str = TOOL_ARGS,
) -> bytes:
    """A reply that optionally thinks and answers, then calls one tool.

    Args:
        name: The tool the model asks to run.
        chat_id: The id every event of the reply carries.
        tc_id: The id the model gives the tool call.
        reasoning: The thinking that precedes the call, omitted when empty.
        content: The prose that precedes the call, omitted when empty.
        arguments: The tool arguments, already encoded as JSON text.

    Returns:
        The scripted reply.
    """
    reply = b""
    if reasoning:
        reply += sse_reasoning(chat_id, reasoning)
    if content:
        reply += sse_content(chat_id, content)
    return (
        reply
        + sse_tool_call(tc_id, name, json.loads(arguments), chat_id)
        + sse_finish_reason(chat_id, "stop")
        + sse_done()
    )


def _repeat_history_tool_call_reply(name: str, arguments: str) -> bytes:
    """A reply that re-emits the tool call the history already carries.

    Args:
        name: The tool the history already called.
        arguments: The arguments the history already passed it, already encoded
            as JSON text.

    Returns:
        The scripted reply.
    """
    return sse_tool_call("tc2", name, json.loads(arguments)) + sse_done()


def _loop_upstream() -> _Upstream:
    """An upstream whose first attempt loops and whose retry recovers."""
    return _Upstream(
        _loop_reply(),
        _recovered_reply(),
    )


def _history_echo_upstream(text: str) -> _Upstream:
    """An upstream whose first attempt echoes history text and whose retry recovers.

    Args:
        text: The text from history the model repeats.

    Returns:
        The scripted upstream.
    """
    return _Upstream(
        sse_content("1", text) + sse_done(),
        _recovered_reply(),
    )


def _duplicate_tool_call_upstream(name: str, arguments: dict) -> _Upstream:
    """An upstream whose first attempt repeats one tool call and whose retry recovers."""
    return _Upstream(
        sse_tool_call("tc1", name, arguments)
        + sse_tool_call("tc2", name, arguments)
        + sse_finish_reason("1", "stop")
        + sse_done(),
        _recovered_reply(finish_reason="stop"),
    )


def _repeat_history_upstream(name: str, arguments: str) -> _Upstream:
    """An upstream whose only attempt re-emits the tool call the history already carries.

    Args:
        name: The tool the history already called.
        arguments: The arguments the history already passed it, already encoded
            as JSON text.

    Returns:
        The scripted upstream.
    """
    return _Upstream(
        _repeat_history_tool_call_reply(name, arguments),
    )


def _no_final_output_upstream(finish_reason: str = "stop") -> _Upstream:
    """An upstream whose first attempt only thinks and whose retry answers.

    Args:
        finish_reason: The reason the first attempt finishes on, or "" to end
            that attempt without announcing one at all.

    Returns:
        The scripted upstream.
    """
    return _Upstream(
        _reasoning_only_reply(finish_reason),
        _final_answer_reply(),
    )


def _repeat_tool_call_upstream(name: str, reasoning: str = "", content: str = "") -> _Upstream:
    """An upstream whose second turn calls a tool the first turn already called.

    Args:
        name: The tool the model calls again.
        reasoning: The thinking that precedes the repeat, omitted when empty.
        content: The prose that precedes the repeat, omitted when empty.

    Returns:
        The scripted upstream.
    """
    return _Upstream(
        _tool_call_reply(name, "2", "tc2", reasoning, content),
        _recovered_reply("2"),
    )


async def _drive_tool_call_turn(
    plugin: OpenAIGuardRail,
    name: str,
    reasoning: str = "",
    content: str = "",
) -> None:
    """Drive an opening turn the model answers with optional prose and then one tool call."""
    upstream = _Upstream(
        _tool_call_reply(name, "1", "tc1", reasoning, content),
    )
    await _drive_result(plugin, upstream.forward, _make_client_response(), FIRST_TURN)
    assert upstream.calls == 1


async def _single_panel(plugin: OpenAIGuardRail, upstream: _Upstream) -> str:
    """Drive one interruption off the given upstream and return the panel it published."""
    published = _capture_ui(plugin)
    await _drive_result(plugin, upstream.forward, _make_client_response())
    assert upstream.calls == 2
    assert len(published) == 1
    return published[0][1]


def _panel_count(html: str, label: str) -> int:
    """The number the telemetry panel shows on the tile carrying the given label."""
    label_tag = f">{label}</div>"
    assert label_tag in html, f"panel has no {label!r} tile"
    count_tag = html.split(label_tag, 1)[1]
    return int(count_tag.split(">", 1)[1].split("<", 1)[0])


async def _drive_result(
    plugin: OpenAIGuardRail,
    forward: _ForwardFn,
    client: MagicMock,
    body: bytes = STREAM_REQUEST,
    path: str = CHAT_COMPLETIONS,
) -> bool:
    """Run one request through the guard rail and return the plugin's own verdict."""
    return await plugin.handle_overridden_request(
        1,
        {},
        body,
        path,
        _make_web_request(),
        forward,
        client,
    )


async def _drive(
    plugin: OpenAIGuardRail,
    forward: _ForwardFn,
    client: MagicMock,
    body: bytes = STREAM_REQUEST,
    path: str = CHAT_COMPLETIONS,
) -> bytes:
    """Run one request through the guard rail, returning what reached the client."""
    await _drive_result(plugin, forward, client, body, path)
    return _written(client)


async def _assert_rejected_before_prepare(
    plugin: OpenAIGuardRail,
    status: int,
    reason: str,
    body: bytes = STREAM_REQUEST,
) -> None:
    """Assert a forward that rejects before preparing leaves the client untouched.

    Args:
        plugin: The guard rail under test.
        status: The status the rejecting forward sets.
        reason: The reason phrase the rejecting forward sets.
        body: The request body the client sends.
    """
    client = _make_client_response()

    async def reject(
        _headers: dict[str, str],
        _body: bytes,
        _path: str,
        _web_req: web.Request,
        resp: AbstractMutatingResponse,
    ) -> bool:
        resp.set_status(status, reason)
        return False

    result = await _drive_result(plugin, reject, client, body)
    assert result is False
    client.prepare.assert_not_called()
    client.write.assert_not_called()
    client.write_eof.assert_not_called()


class TestPluginBasics:
    async def test_name_configured(self) -> None:
        plugin = OpenAIGuardRail({"name": "my_guard"})
        assert plugin.name() == "my_guard"

    async def test_name_default(self) -> None:
        plugin = OpenAIGuardRail({})
        assert plugin.name() == "openai_guard_rail"

    async def test_overridden_paths(self) -> None:
        plugin = OpenAIGuardRail({})
        assert plugin.overridden_paths() == [CHAT_COMPLETIONS, "/v1/responses"]


class TestForceStream:
    async def test_non_streaming_request_forces_stream(self) -> None:
        """A client that asked for no streaming is still forwarded as a stream."""
        plugin = OpenAIGuardRail({})
        client = _make_client_response()

        upstream = _Upstream(
            sse_done(),
        )

        await _drive_result(plugin, upstream.forward, client, NON_STREAM_REQUEST)
        forwarded = _forwarded_body(upstream)
        assert forwarded["stream"] is True
        assert forwarded["stream_options"]["include_usage"] is True

    async def test_streaming_request_keeps_stream(self) -> None:
        """A client that asked for streaming keeps it, with usage requested."""
        plugin = OpenAIGuardRail({})
        client = _make_client_response()

        upstream = _Upstream(
            sse_done(),
        )

        await _drive_result(plugin, upstream.forward, client)
        forwarded = _forwarded_body(upstream)
        assert forwarded["stream"] is True
        assert forwarded["stream_options"]["include_usage"] is True

    async def test_content_length_updated(self) -> None:
        """Rewriting the body updates the Content-Length that goes upstream."""
        plugin = OpenAIGuardRail({})
        client = _make_client_response()

        upstream = _Upstream(
            sse_done(),
        )

        await _drive_result(plugin, upstream.forward, client, NON_STREAM_REQUEST)
        assert int(upstream.headers[-1]["Content-Length"]) == len(upstream.bodies[-1])


class TestStreamingPassthrough:
    async def test_content_events_pass_through(self) -> None:
        plugin = OpenAIGuardRail({})
        client = _make_client_response()

        upstream = _Upstream(
            sse_content("1", "hello") + sse_done(),
        )

        written = await _drive(plugin, upstream.forward, client)
        assert extract_sse_field(written) == "hello"
        assert client.headers["Content-Type"] == "text/event-stream"

    async def test_reasoning_events_pass_through(self) -> None:
        plugin = OpenAIGuardRail({})
        client = _make_client_response()

        upstream = _Upstream(
            sse_reasoning("1", "thinking") + sse_done(),
        )

        written = await _drive(plugin, upstream.forward, client)
        assert extract_sse_field(written, "reasoning") == "thinking"
        assert client.headers["Content-Type"] == "text/event-stream"

    async def test_tool_call_events_pass_through(self) -> None:
        plugin = OpenAIGuardRail({})
        client = _make_client_response()

        upstream = _Upstream(
            sse_tool_call("tc1", "se", {"query": "arch"}) + sse_done(),
        )

        written = await _drive(plugin, upstream.forward, client)
        assert extract_sse_tool_call_name(written) == "se"
        assert extract_sse_tool_call_args(written) == {"query": "arch"}
        assert client.headers["Content-Type"] == "text/event-stream"

    async def test_tool_call_with_explicit_null_fields_pass_through(self) -> None:
        """Tool call deltas with a null name and a null id relay through."""
        plugin = OpenAIGuardRail({})
        client = _make_client_response()

        upstream = _Upstream(
            sse_tool_call_null_fields("tc1", "read_file", {"path": "/etc/hostname"})
            + sse_done(),
        )

        written = await _drive(plugin, upstream.forward, client)
        assert extract_sse_tool_call_name(written) == "read_file"
        assert extract_sse_tool_call_args(written) == {"path": "/etc/hostname"}
        assert client.headers["Content-Type"] == "text/event-stream"

    async def test_mixed_event_types_pass_through(self) -> None:
        plugin = OpenAIGuardRail({})
        client = _make_client_response()

        upstream = _Upstream(
            sse_reasoning("1", "think")
            + sse_content("1", "hello")
            + sse_tool_call("tc1", "f")
            + sse_done(),
        )

        written = await _drive(plugin, upstream.forward, client)
        assert extract_sse_field(written, "reasoning") == "think"
        assert extract_sse_field(written, "content") == "hello"
        assert extract_sse_tool_call_name(written) == "f"
        assert client.headers["Content-Type"] == "text/event-stream"


class TestUsageEventPassthrough:
    async def test_usage_event_passes_through(self) -> None:
        """Test that usage events (no choices) pass through without being dropped."""
        plugin = OpenAIGuardRail({})
        client = _make_client_response()

        upstream = _Upstream(
            sse_content("1", "hello")
            + sse_usage(10, 20, 30)
            + sse_done(),
        )

        written = await _drive(plugin, upstream.forward, client)
        assert extract_sse_field(written) == "hello"
        assert b'"prompt_tokens": 10' in written


class TestFinishReasonPassthrough:
    async def test_finish_reason_event_with_empty_delta_passes_through(self) -> None:
        """Test that SSE events with finish_reason and empty delta are not dropped."""
        plugin = OpenAIGuardRail({})
        client = _make_client_response()

        upstream = _Upstream(
            sse_content("chatcmpl-1", "hello")
            + sse_finish_reason("chatcmpl-1", "tool_calls")
            + sse_done(),
        )

        written = await _drive(plugin, upstream.forward, client)
        assert extract_sse_field(written) == "hello"
        assert b'"finish_reason": "tool_calls"' in written

    async def test_finish_reason_stop_passes_through(self) -> None:
        """Test that stop finish_reason events pass through."""
        plugin = OpenAIGuardRail({})
        client = _make_client_response()

        upstream = _Upstream(
            sse_content("1", "done")
            + sse_finish_reason("1", "stop")
            + sse_done(),
        )

        written = await _drive(plugin, upstream.forward, client)
        assert extract_sse_field(written) == "done"
        assert b'"finish_reason": "stop"' in written


class TestNonStreamingAssembly:
    async def test_non_streaming_gets_json_notsse_event(self) -> None:
        plugin = OpenAIGuardRail({})
        client = _make_client_response()

        upstream = _Upstream(
            sse_content("1", "answer")
            + sse_finish_reason("1", "stop")
            + sse_done(),
        )

        written = await _drive(plugin, upstream.forward, client, NON_STREAM_REQUEST)
        result = json.loads(written)
        assert result["choices"][0]["message"]["content"] == "answer"
        assert result["choices"][0]["finish_reason"] == "stop"


class TestUpstreamErrorPropagation:
    async def test_upstream_500_propagated(self) -> None:
        plugin = OpenAIGuardRail({})
        client = _make_client_response()

        async def mock_forward(
            _headers: dict[str, str],
            _body: bytes,
            _path: str,
            web_req: web.Request,
            resp: AbstractMutatingResponse,
        ) -> None:
            resp.set_status(500, "Internal Server Error")
            await resp.prepare(web_req)
            await resp.write(b'{"error":"fail"}')
            await resp.write_eof()

        result = await _drive_result(plugin, mock_forward, client, NO_STREAM_FIELD_REQUEST)
        assert result is False
        status_arg = client.set_status.call_args[0][0]
        assert status_arg == 500
        written = _written(client)
        assert b'{"error":"fail"}' in written


class TestMissingReasonPhrase:
    """Upstream may omit the reason phrase, which HTTP/1.1 permits."""

    async def test_empty_reason_still_relays_the_response(self) -> None:
        """A 200 carrying no reason phrase reaches the client with the standard one."""
        plugin = OpenAIGuardRail({})
        client = _make_client_response()

        async def blank_reason(
            _headers: dict[str, str],
            _body: bytes,
            _path: str,
            web_req: web.Request,
            resp: AbstractMutatingResponse,
        ) -> None:
            resp.set_status(200, "")
            await resp.prepare(web_req)
            await resp.write(sse_content("1", "hello") + sse_done())
            await resp.write_eof()

        result = await _drive_result(plugin, blank_reason, client, STREAM_REQUEST)
        assert result is True
        assert client.set_status.call_args[0] == (200, "OK")
        assert extract_sse_field(_written(client)) == "hello"

    async def test_unknown_status_gets_no_invented_reason(self) -> None:
        """A status outside the registry keeps an empty reason rather than a made-up one."""
        plugin = OpenAIGuardRail({})
        client = _make_client_response()

        async def unknown_status(
            _headers: dict[str, str],
            _body: bytes,
            _path: str,
            web_req: web.Request,
            resp: AbstractMutatingResponse,
        ) -> None:
            resp.set_status(599, "")
            await resp.prepare(web_req)
            await resp.write(b'{"error":"odd"}')
            await resp.write_eof()

        result = await _drive_result(plugin, unknown_status, client, STREAM_REQUEST)
        assert result is False
        assert client.set_status.call_args[0] == (599, "")
        assert b'{"error":"odd"}' in _written(client)


class TestInvalidSsePassthrough:
    async def test_garbage_data_is_passed_through(self) -> None:
        plugin = OpenAIGuardRail({})
        client = _make_client_response()

        upstream = _Upstream(
            b'garbage data here\n\n',
        )

        written = await _drive(plugin, upstream.forward, client, NO_STREAM_FIELD_REQUEST)
        assert b"garbage data here" in written

    async def test_invalid_json_in_sse_is_passed_through(self) -> None:
        plugin = OpenAIGuardRail({})
        client = _make_client_response()

        upstream = _Upstream(
            b'data: not valid json\n\n',
        )

        written = await _drive(plugin, upstream.forward, client, NO_STREAM_FIELD_REQUEST)
        assert b"not valid json" in written


class TestPrepareError:
    async def test_response_prepare_exception_does_not_crash(self) -> None:
        """Test that an exception during response.prepare() doesn't cause UnboundLocalError."""
        plugin = OpenAIGuardRail({})
        client = _make_client_response()
        client.prepare = AsyncMock(side_effect=RuntimeError("prepare failed"))

        upstream = _Upstream(
            sse_done(),
        )

        result = await _drive_result(plugin, upstream.forward, client)
        assert result is False


class TestChatCompletionsPath:
    async def test_chat_completions(self) -> None:
        plugin = OpenAIGuardRail({})
        client = _make_client_response()

        upstream = _Upstream(
            sse_done(),
        )

        result = await _drive_result(plugin, upstream.forward, client, b'{}')
        assert result is True


class TestCompletionsPath:
    async def test_completions(self) -> None:
        plugin = OpenAIGuardRail({})
        client = _make_client_response()

        upstream = _Upstream(
            sse_done(),
        )

        result = await _drive_result(plugin, upstream.forward, client, b'{}', path='/v1/completions')
        assert result is True


class TestForwardWithoutPrepare:
    async def test_forward_returns_without_prepare_does_not_hang(self) -> None:
        """Test that forward returning early without calling prepare() doesn't cause a hang."""
        plugin = OpenAIGuardRail({})

        await _assert_rejected_before_prepare(plugin, 502, "Bad Gateway")

    async def test_forward_without_prepare_non_streaming(self) -> None:
        """Test forward without prepare on non-streaming request."""
        plugin = OpenAIGuardRail({})

        await _assert_rejected_before_prepare(plugin, 401, "Unauthorized", NON_STREAM_REQUEST)


class TestForwardCrashHang:
    async def test_forward_crash_without_write_eof_does_not_hang(self) -> None:
        """If forward crashes after prepare/write but before write_eof, relay must not hang."""
        plugin = OpenAIGuardRail({})
        client = _make_client_response()

        async def mock_forward(
            _headers: dict[str, str],
            _body: bytes,
            _path: str,
            web_req: web.Request,
            resp: AbstractMutatingResponse,
        ) -> None:
            await resp.prepare(web_req)
            await resp.write(sse_content("1", "ok"))
            raise ConnectionError("upstream crashed without write_eof")

        result = await asyncio.wait_for(
            _drive_result(plugin, mock_forward, client),
            timeout=5,
        )
        assert result is True


class TestStreamValidationAndRetry:
    async def test_loop_detection_triggers_retry(self) -> None:
        """Repeating content triggers retry with recovery."""
        plugin = OpenAIGuardRail({"break_loops": True})
        client = _make_client_response()
        upstream = _loop_upstream()

        written = await _drive(plugin, upstream.forward, client)
        assert upstream.calls == 2
        assert "recovered" in extract_sse_field(written)

    async def test_loop_causing_delta_not_in_output(self) -> None:
        """The delta that caused loop detection is dropped from output."""
        plugin = OpenAIGuardRail({"break_loops": True})
        client = _make_client_response()

        upstream = _Upstream(
            sse_content("1", "safe")
            + sse_content("1", MIN_SIZE_BLOCK) * 3
            + sse_done(),
            _recovered_reply(),
        )

        written = await _drive(plugin, upstream.forward, client)
        assert upstream.calls == 2
        content = extract_sse_field(written)
        assert "safe" in content
        assert "recovered" in content
        # Loop detection triggered on the repeating 'a' blocks
        assert content.count("a") == 99

    async def test_last_retry_suppresses_loop_detection(self) -> None:
        """Loop detection is disabled on the last retry attempt."""
        plugin = OpenAIGuardRail({"break_loops": True, "max_retry_attempt": 1})
        client = _make_client_response()

        upstream = _Upstream(
            _loop_reply(MIN_SIZE_BLOCK),
        )

        written = await _drive(plugin, upstream.forward, client)
        assert upstream.calls == 2
        assert MIN_SIZE_BLOCK in extract_sse_field(written)

    async def test_retry_message_appended_to_request(self) -> None:
        """Verify the retry message is appended to the request body."""
        plugin = OpenAIGuardRail({"break_loops": True})
        client = _make_client_response()

        upstream = _Upstream(
            _loop_reply(),
        )

        await _drive_result(plugin, upstream.forward, client)
        assert upstream.calls == 3
        messages = _forwarded_messages(upstream)
        assert any("repetition loop" in (m.get("content") or "").lower() for m in messages)

    async def test_retry_includes_partial_assistant_message(self) -> None:
        """The retry request includes the assistant's partial message up to the loop point."""
        plugin = OpenAIGuardRail({"break_loops": True})
        client = _make_client_response()

        safe_text = "this_is_safe_content_" * 3

        upstream = _Upstream(
            sse_content("1", safe_text)
            + sse_content("1", LOOP_BLOCK) * 3
            + sse_done(),
            _recovered_reply(),
        )

        await _drive_result(plugin, upstream.forward, client)
        assert upstream.calls == 2
        # The assistant message on retry has safe_text + 1 LOOP_BLOCK (triggering copy excluded)
        assistant_msgs = _forwarded_assistant_messages(upstream)
        assert len(assistant_msgs) >= 1
        last_assistant = assistant_msgs[-1]
        content = last_assistant.get("content", "")
        assert safe_text in content
        assert content.count(LOOP_BLOCK) == 1

    async def test_identical_thinking_and_content_no_loop(self) -> None:
        """Identical text in thinking then content doesn't trigger loop detection."""
        plugin = OpenAIGuardRail({"break_loops": True})
        client = _make_client_response()

        text = "this_is_the_same_text_in_both_sections_" * 3

        upstream = _Upstream(
            sse_reasoning("1", text)
            + sse_content("1", text)
            + sse_done(),
        )

        await _drive_result(plugin, upstream.forward, client)
        assert upstream.calls == 1

    async def test_history_repeated_in_response_triggers_retry(self) -> None:
        """Text from history repeated in new response triggers loop detection."""
        plugin = OpenAIGuardRail({"break_loops": True})
        client = _make_client_response()

        history_text = "history_block_content_" * 3
        upstream = _history_echo_upstream(history_text)

        await _drive_result(plugin, upstream.forward, client, _request([
            _assistant_message(history_text),
        ]))
        assert upstream.calls == 2

    async def test_suffix_of_history_repeated_triggers_retry(self) -> None:
        """A suffix of history repeated in new response triggers loop detection."""
        plugin = OpenAIGuardRail({"break_loops": True})
        client = _make_client_response()

        prefix = "unique_prefix_history_block_xyz_" * 5
        repeated = "repeated_suffix_content_here_" * 6  # 174 chars, well above min_block_size
        upstream = _history_echo_upstream(repeated)

        await _drive_result(plugin, upstream.forward, client, _request([
            _assistant_message(prefix + repeated),
        ]))
        assert upstream.calls == 2

    async def test_history_thinking_repeated_as_content_no_loop(self) -> None:
        """Text from history thinking repeated as content in new response doesn't trigger loop."""
        plugin = OpenAIGuardRail({"break_loops": True})
        client = _make_client_response()

        text = "history_thinking_block_" * 3

        upstream = _Upstream(
            sse_content("1", text) + sse_done(),
        )

        await _drive_result(plugin, upstream.forward, client, _request([
            _assistant_message("", reasoning=text),
        ]))
        assert upstream.calls == 1

    async def test_history_prefix_loop_excluded_no_retry(self) -> None:
        """Looping prefix in history is excluded; new non-repeating content doesn't trigger retry."""
        plugin = OpenAIGuardRail({"break_loops": True})
        client = _make_client_response()

        loop_unit = "abc_loop_unit_" * 10  # 140 chars
        safe_suffix = "xyz_safe_unique_suffix_" * 3  # different from loop_unit
        history = loop_unit + loop_unit + safe_suffix

        upstream = _Upstream(
            sse_content("1", "completely_new_content_here") + sse_done(),
        )

        await _drive_result(plugin, upstream.forward, client, _request([
            _assistant_message(history),
        ]))
        assert upstream.calls == 1

    async def test_history_safe_suffix_repeated_triggers_retry(self) -> None:
        """New content repeating the safe suffix of history triggers loop detection."""
        plugin = OpenAIGuardRail({"break_loops": True})
        client = _make_client_response()

        loop_unit = "abc_loop_prefix_" * 10  # looping prefix, excluded by init
        safe_suffix = "xyz_safe_suffix_content_" * 4  # 96 chars, stays in detector
        upstream = _history_echo_upstream(safe_suffix)

        await _drive_result(plugin, upstream.forward, client, _request([
            _assistant_message(loop_unit + loop_unit + safe_suffix),
        ]))
        assert upstream.calls == 2

    async def test_history_loop_suffix_repeated_triggers_retry(self) -> None:
        """History with looping pattern: new response repeating safe suffix triggers retry."""
        plugin = OpenAIGuardRail({"break_loops": True})
        client = _make_client_response()

        block = "history_loop_block_xyz_" * 4  # 92 chars, above min_block_size
        upstream = _history_echo_upstream(block)

        await _drive_result(plugin, upstream.forward, client, _request([
            _assistant_message(block + block),
        ]))
        assert upstream.calls == 2

    async def test_tool_call_repeated_in_response_triggers_retry(self) -> None:
        """History with tool call repeated in new response triggers loop detection."""
        plugin = OpenAIGuardRail({"break_loops": True})
        client = _make_client_response()

        tc_name = "write_file_v2"
        tc_args = '{"path":"test.txt","content":"' + "x" * 100 + '"}'  # long enough

        upstream = _Upstream(
            _repeat_history_tool_call_reply(tc_name, tc_args),
            _recovered_reply(),
        )

        await _drive_result(plugin, upstream.forward, client, _request([
            _assistant_tool_call(tc_name, tc_args),
        ]))
        assert upstream.calls == 2

    async def test_multi_section_history_repeated_triggers_retry(self) -> None:
        """History with thinking+content+tool_call repeated in new response triggers retry."""
        plugin = OpenAIGuardRail({"break_loops": True})
        client = _make_client_response()

        thinking = "thinking_section_text_" * 5  # 110 chars
        content = "content_section_text_" * 5    # 105 chars
        tc_name = "my_tool"
        tc_args = '{"arg":"' + "x" * 90 + '"}'  # 100 chars

        upstream = _Upstream(
            sse_reasoning("1", thinking)
            + sse_content("1", content)
            + sse_tool_call("tc2", tc_name, json.loads(tc_args))
            + sse_done(),
            _recovered_reply(),
        )

        await _drive_result(plugin, upstream.forward, client, _request([
            _assistant_tool_call(tc_name, tc_args, reasoning=thinking, content=content),
        ]))
        assert upstream.calls == 2

    async def test_last_retry_loop_content_passes_through(self) -> None:
        """On last retry, loop detection is suppressed and repeating content passes through."""
        plugin = OpenAIGuardRail({"break_loops": True, "max_retry_attempt": 1})
        client = _make_client_response()

        upstream = _Upstream(
            _loop_reply(LONG_BLOCK),
        )

        written = await _drive(plugin, upstream.forward, client)
        assert upstream.calls == 2
        assert LONG_BLOCK in extract_sse_field(written)

    async def test_break_loops_disabled_no_retry(self) -> None:
        """When break_loops is False (default), loop content passes through without retry."""
        plugin = OpenAIGuardRail({})
        client = _make_client_response()

        upstream = _Upstream(
            _loop_reply(LONG_BLOCK),
        )

        written = await _drive(plugin, upstream.forward, client)
        assert upstream.calls == 1
        assert LONG_BLOCK in extract_sse_field(written)

    async def test_force_final_output_no_content_triggers_retry(self) -> None:
        """When force_final_output is True and stop is reached with no content, retry."""
        plugin = OpenAIGuardRail({"force_final_output": True})
        client = _make_client_response()
        upstream = _no_final_output_upstream()

        await _drive_result(plugin, upstream.forward, client)
        assert upstream.calls == 2

    async def test_force_final_output_tool_calls_without_tool_calls_triggers_retry(self) -> None:
        """When finish_reason is tool_calls but no tool calls were sent, retry."""
        plugin = OpenAIGuardRail({"force_final_output": True})
        client = _make_client_response()
        upstream = _no_final_output_upstream("tool_calls")

        await _drive_result(plugin, upstream.forward, client)
        assert upstream.calls == 2

    async def test_force_final_output_missing_finish_reason_triggers_retry_if_reasoning_only(self) -> None:
        """When stream ends without finish_reason and no content/tool_calls, retry."""
        plugin = OpenAIGuardRail({"force_final_output": True})
        client = _make_client_response()
        upstream = _no_final_output_upstream("")

        await _drive_result(plugin, upstream.forward, client)
        assert upstream.calls == 2

    async def test_force_final_output_missing_finish_reason_not_triggers_retry_if_has_tool_call(self) -> None:
        """When the stream ends without finish_reason but a tool call was sent, no retry."""
        plugin = OpenAIGuardRail({"force_final_output": True})
        client = _make_client_response()

        upstream = _Upstream(
            sse_tool_call("tc1", "write_file", {"path": "a.txt"}) + sse_done(),
        )

        await _drive_result(plugin, upstream.forward, client)
        assert upstream.calls == 1

    async def test_force_final_output_tool_call_with_null_fields_no_retry(self) -> None:
        """A tool call sent with null fields counts as a tool call, so no retry."""
        plugin = OpenAIGuardRail({"force_final_output": True})
        client = _make_client_response()

        upstream = _Upstream(
            sse_tool_call_null_fields("tc1", "write_file", {"path": "a.txt"})
            + sse_finish_reason("1", "tool_calls")
            + sse_done(),
        )

        written = await _drive(plugin, upstream.forward, client)
        assert upstream.calls == 1
        assert extract_sse_tool_call_name(written) == "write_file"
        assert extract_sse_tool_call_args(written) == {"path": "a.txt"}
        assert b'"finish_reason": "tool_calls"' in written

    async def test_force_final_output_missing_finish_reason_not_triggers_retry_if_has_content(self) -> None:
        """When the stream ends without finish_reason but content was sent, no retry."""
        plugin = OpenAIGuardRail({"force_final_output": True})
        client = _make_client_response()

        upstream = _Upstream(
            sse_content("1", "content") + sse_done(),
        )

        await _drive_result(plugin, upstream.forward, client)
        assert upstream.calls == 1

    async def test_force_final_output_with_content_no_retry(self) -> None:
        """When force_final_output is True but content exists, no retry."""
        plugin = OpenAIGuardRail({"force_final_output": True})
        client = _make_client_response()

        upstream = _Upstream(
            sse_reasoning("1", "thinking")
            + sse_content("1", "valid_output")
            + sse_finish_reason("1", "stop")
            + sse_done(),
        )

        await _drive_result(plugin, upstream.forward, client)
        assert upstream.calls == 1

    async def test_force_final_output_disabled_no_retry(self) -> None:
        """When force_final_output is False (default), no content at stop doesn't retry."""
        plugin = OpenAIGuardRail({})
        client = _make_client_response()

        upstream = _Upstream(
            _reasoning_only_reply(),
        )

        written = await _drive(plugin, upstream.forward, client)
        assert upstream.calls == 1
        assert THINKING_ONLY in extract_sse_field(written, "reasoning")

    async def test_dedup_tool_calls_repeated_triggers_retry(self) -> None:
        plugin = OpenAIGuardRail({"dedup_tool_calls": True})
        client = _make_client_response()

        tc_name = "write_file"
        upstream = _duplicate_tool_call_upstream(tc_name, {"path": "a.txt"})

        await _drive_result(plugin, upstream.forward, client)
        assert upstream.calls == 2
        _assert_retry_names_tool_call(upstream, tc_name)

    async def test_dedup_tool_calls_different_no_retry(self) -> None:
        plugin = OpenAIGuardRail({"dedup_tool_calls": True})
        client = _make_client_response()

        upstream = _Upstream(
            sse_tool_call("tc1", "write_file", {"path": "a"})
            + sse_tool_call("tc2", "read_file", {"path": "b"})
            + sse_done(),
        )

        await _drive_result(plugin, upstream.forward, client)
        assert upstream.calls == 1

    async def test_dedup_tool_calls_prev_user_turn_not_deduped(self) -> None:
        plugin = OpenAIGuardRail({"dedup_tool_calls": True})
        client = _make_client_response()

        tc_name = "write_file"
        tc_args = '{"path":"a.txt"}'
        upstream = _repeat_history_upstream(tc_name, tc_args)

        await _drive_result(plugin, upstream.forward, client, _request([
            _user_message("first turn"),
            _assistant_tool_call(tc_name, tc_args),
            _user_message("second turn"),
        ]))
        assert upstream.calls == 1

    async def test_dedup_tool_calls_disabled_no_retry(self) -> None:
        plugin = OpenAIGuardRail({})
        client = _make_client_response()

        upstream = _Upstream(
            sse_tool_call("tc1", "write_file", {"path": "a"})
            + sse_tool_call("tc2", "write_file", {"path": "a"})
            + sse_done(),
        )

        await _drive_result(plugin, upstream.forward, client)
        assert upstream.calls == 1

    async def test_validate_tool_calls_invalid_json_triggers_retry(self) -> None:
        """When validate_tool_calls is True and tool call has invalid JSON, retry."""
        plugin = OpenAIGuardRail({"validate_tool_calls": True})
        client = _make_client_response()

        tc_name = "write_file"
        tc_data = {"id": "tc1", "type": "function", "function": {"name": tc_name, "arguments": "invalid json here"}}

        upstream = _Upstream(
            sse_event({"id": "1", "choices": [{"delta": {"tool_calls": [tc_data]}}]}) + sse_done(),
            _recovered_reply(finish_reason="stop"),
        )

        await _drive_result(plugin, upstream.forward, client)
        assert upstream.calls == 2
        _assert_retry_names_tool_call(upstream, tc_name)
        assert "invalid JSON arguments" in _forwarded_system_message(upstream)

    async def test_validate_tool_calls_valid_json_no_retry(self) -> None:
        """When validate_tool_calls is True but JSON is valid, no retry."""
        plugin = OpenAIGuardRail({"validate_tool_calls": True})
        client = _make_client_response()

        upstream = _Upstream(
            sse_tool_call("tc1", "write_file", {"path": "a.txt"}) + sse_done(),
        )

        await _drive_result(plugin, upstream.forward, client)
        assert upstream.calls == 1

    async def test_dedup_tool_call_from_same_turn_history_triggers_retry(self) -> None:
        """Tool call from current turn history repeated in new response triggers retry."""
        plugin = OpenAIGuardRail({"dedup_tool_calls": True})
        client = _make_client_response()

        tc_name = "write_file"
        tc_args = '{"path":"a.txt"}'

        upstream = _Upstream(
            sse_reasoning("1", "other_reasoning")
            + sse_tool_call("tc2", tc_name, json.loads(tc_args))
            + sse_finish_reason("1", "stop")
            + sse_done(),
            _recovered_reply(),
        )

        await _drive_result(plugin, upstream.forward, client, _request([
            _user_message("request"),
            _assistant_tool_call(tc_name, tc_args, reasoning="thinking"),
            _tool_result("result"),
        ]))
        assert upstream.calls == 2
        _assert_retry_names_tool_call(upstream, tc_name)

    async def test_dedup_tool_call_repeated_args_no_retry(self) -> None:
        """Tool call with repeated content in arguments should not trigger dedup."""
        plugin = OpenAIGuardRail({"dedup_tool_calls": True})
        client = _make_client_response()

        repeated = "this_is_a_repeated_argument_value_" * 3

        upstream = _Upstream(
            sse_tool_call("tc1", "process", {"a": repeated, "b": repeated, "c": repeated, "d": repeated})
            + sse_finish_reason("1", "stop")
            + sse_done(),
        )

        await _drive_result(plugin, upstream.forward, client)
        assert upstream.calls == 1

    async def test_dedup_tool_call_streaming_format_triggers_retry(self) -> None:
        """Real streaming format: first delta has id+name, subsequent deltas have only arguments.
        Duplicate calls in this format should still trigger retry."""
        plugin = OpenAIGuardRail({"dedup_tool_calls": True})
        client = _make_client_response()

        tc_name = "process"
        long_args = {"data": "long_argument_value_for_dedup_check_" * 3}
        upstream = _duplicate_tool_call_upstream(tc_name, long_args)

        await _drive_result(plugin, upstream.forward, client)
        assert upstream.calls == 2
        _assert_retry_names_tool_call(upstream, tc_name)

    async def test_dedup_tool_call_repeating_pattern_in_args_no_retry(self) -> None:
        """Tool call argument with repeating pattern should not trigger loop detector."""
        plugin = OpenAIGuardRail({"dedup_tool_calls": True, "break_loops": True})
        client = _make_client_response()

        repeated = "long_repeated_pattern_xyz_" * 5

        upstream = _Upstream(
            sse_tool_call("tc1", "process", repeated)
            + sse_finish_reason("1", "stop")
            + sse_done(),
        )

        await _drive_result(plugin, upstream.forward, client)
        assert upstream.calls == 1

    async def test_dedup_tool_call_before_last_user_turn_no_retry(self) -> None:
        """Tool call matching one before the last user turn should not trigger dedup."""
        plugin = OpenAIGuardRail({"dedup_tool_calls": True})
        client = _make_client_response()

        tc_name = "read_file"
        tc_args = '{"path":"data.txt"}'
        upstream = _repeat_history_upstream(tc_name, tc_args)

        await _drive_result(plugin, upstream.forward, client, _request([
            _user_message("first"),
            _assistant_tool_call(tc_name, tc_args),
            _tool_result("result1"),
            _user_message("last user turn"),
        ]))
        assert upstream.calls == 1


class TestCrossRequestToolCallDedup:
    """Test tool call dedup across requests using resumption state LRU cache."""

    async def test_history_has_duplicate_tool_calls_no_retry(self) -> None:
        """History already contains 2 duplicate tool calls. A new normal request
        with a different tool call should NOT trigger dedup retry."""
        plugin = OpenAIGuardRail({"dedup_tool_calls": True})
        client = _make_client_response()

        tc_name_dup = "read_file"
        tc_args_dup = '{"path":"data.txt"}'
        tc_name_new = "write_file"

        upstream = _Upstream(
            sse_tool_call("tc_new", tc_name_new, {"path": "output.txt", "content": "hello"})
            + sse_finish_reason("1", "stop")
            + sse_done(),
        )

        await _drive_result(plugin, upstream.forward, client, _request([
            _user_message("request"),
            _assistant_tool_call(tc_name_dup, tc_args_dup),
            _tool_result("result1"),
            _assistant_tool_call(tc_name_dup, tc_args_dup, tc_id="tc2"),
            _tool_result("result2"),
            _user_message("now write something"),
        ]))
        assert upstream.calls == 1

    async def test_tool_call_after_user_turn_same_as_history_no_retry(self) -> None:
        """History has a tool call, then a user message, then the same tool call in
        the current response. This should NOT trigger dedup retry because the user
        message resets the dedup window."""
        plugin = OpenAIGuardRail({"dedup_tool_calls": True})
        client = _make_client_response()

        tc_name = "read_file"
        tc_args = '{"path":"config.yaml"}'
        upstream = _repeat_history_upstream(tc_name, tc_args)

        await _drive_result(plugin, upstream.forward, client, _request([
            _user_message("first request"),
            _assistant_tool_call(tc_name, tc_args),
            _tool_result("result"),
            _assistant_message("I read the file."),
            _user_message("can you read it again?"),
        ]))
        assert upstream.calls == 1

    async def test_cross_request_duplicate_tool_call_triggers_retry(self) -> None:
        """Request 1: LLM generates tool call.
        Request 2: tool result without assistant tool call in history.
        LLM repeats same tool call → retry triggered via resumption state."""
        plugin = OpenAIGuardRail({"dedup_tool_calls": True})
        tc_name = "read_file"

        await _drive_tool_call_turn(plugin, tc_name)

        client = _make_client_response()
        upstream = _repeat_tool_call_upstream(tc_name)

        await _drive_result(plugin, upstream.forward, client, _request([
            _user_message(),
            _tool_result("result", "tc1"),
        ]))
        assert upstream.calls == 2

    async def test_cross_request_duplicate_tool_call_streaming_format_triggers_retry(self) -> None:
        """Request 1: LLM generates tool call in real streaming format (id in first delta, arguments in subsequent).
        Request 2: tool result without assistant tool call in history.
        LLM repeats same tool call in streaming format → retry triggered via resumption state."""
        plugin = OpenAIGuardRail({"dedup_tool_calls": True})
        tc_name = "read_file"

        await _drive_tool_call_turn(plugin, tc_name)

        client = _make_client_response()

        # the repeat arrives in real streaming shape: id first, arguments after
        upstream = _Upstream(
            sse_event({"id": "2", "choices": [{"delta": {"tool_calls": [{"id": "tc2", "type": "function", "index": 0, "function": {"name": tc_name}}]}}]})
            + sse_event({"id": "2", "choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": TOOL_ARGS}}]}}]})
            + sse_finish_reason("2", "stop")
            + sse_done(),
            _recovered_reply("2"),
        )

        await _drive_result(plugin, upstream.forward, client, _request([
            _user_message(),
            _tool_result("result", "tc1"),
        ]))
        assert upstream.calls == 2

    async def test_cross_turn_reasoning_before_repeat_triggers_retry(self) -> None:
        """Request 1: LLM generates reasoning + tool call.
        Request 2: history includes assistant message (with reasoning + tool_calls) + tool result.
        LLM generates reasoning then repeats the same tool call → the repeat is deduped and the turn retries."""
        plugin = OpenAIGuardRail({"dedup_tool_calls": True})
        tc_name = "read_file"
        reasoning1 = "i need to check the config file"

        await _drive_tool_call_turn(plugin, tc_name, reasoning=reasoning1)

        client = _make_client_response()
        upstream = _repeat_tool_call_upstream(tc_name, "config loaded, let me read it again to confirm")

        await _drive_result(plugin, upstream.forward, client, _request([
            _user_message(),
            _assistant_tool_call(tc_name, TOOL_ARGS, reasoning=reasoning1),
            _tool_result("result", "tc1"),
        ]))
        assert upstream.calls == 2

    async def test_cross_turn_reasoning_content_before_repeat_triggers_retry(self) -> None:
        """Request 1: LLM generates reasoning + content + tool call.
        Request 2: history includes assistant message (with reasoning + content + tool_calls) + tool result.
        LLM generates reasoning + content, then repeats the same tool call → should be deduped."""
        plugin = OpenAIGuardRail({"dedup_tool_calls": True})
        tc_name = "read_file"
        reasoning1 = "let me check the config"
        content1 = "I'll read the config file first."

        await _drive_tool_call_turn(plugin, tc_name, reasoning=reasoning1, content=content1)

        client = _make_client_response()

        upstream = _repeat_tool_call_upstream(
            tc_name,
            "config loaded, but let me verify",
            "Reading the file again to confirm...",
        )

        await _drive_result(plugin, upstream.forward, client, _request([
            _user_message(),
            _assistant_tool_call(tc_name, TOOL_ARGS, reasoning=reasoning1, content=content1),
            _tool_result("result", "tc1"),
        ]))
        assert upstream.calls == 2

    async def test_reasoning_resumption_with_cross_turn_dedup_triggers_retry(self) -> None:
        """Request 1: LLM generates reasoning + tool call.
        Request 2: client sends assistant message with tool_calls but WITHOUT reasoning_content,
        plus tool result. LLM generates reasoning + repeats same tool call.
        Verify BOTH: the reasoning from request 1 reappears in the forwarded body,
        AND the repeated tool call is deduped so the turn retries."""
        plugin = OpenAIGuardRail({"dedup_tool_calls": True})
        tc_name = "read_file"
        reasoning1 = "i need to check the config file first"

        await _drive_tool_call_turn(plugin, tc_name, reasoning=reasoning1)

        client = _make_client_response()
        upstream = _repeat_tool_call_upstream(tc_name, "let me verify the config again")

        # the client strips reasoning_content from the assistant message it sends back
        await _drive_result(plugin, upstream.forward, client, _request([
            _user_message(),
            _assistant_tool_call(tc_name, TOOL_ARGS),
            _tool_result("result", "tc1"),
        ]))
        assert upstream.calls == 2

        # Verify reasoning was recovered and included in the forwarded body
        assistant_msgs = _forwarded_assistant_messages(upstream)
        assert len(assistant_msgs) >= 1
        assert any(m.get("reasoning_content") == reasoning1 for m in assistant_msgs)

    async def test_rejected_tool_call_not_in_streamed_output(self) -> None:
        """A tool call rejected by dedup should not appear in the streamed output,
        while the accepted tool call from the retry should."""
        plugin = OpenAIGuardRail({"dedup_tool_calls": True})
        client = _make_client_response()

        rejected_name = "search"
        accepted_name = "read_file"

        upstream = _Upstream(
            sse_tool_call("tc1", rejected_name, {"query": "test"})
            + sse_tool_call("tc_dup", rejected_name, {"query": "test"})
            + sse_finish_reason("1", "stop")
            + sse_done(),
            sse_tool_call("tc2", accepted_name, {"path": "output.txt"})
            + sse_finish_reason("1", "stop")
            + sse_done(),
        )

        written = await _drive(plugin, upstream.forward, client, FIRST_TURN)
        assert upstream.calls == 2  # 1 rejected by dedup, 1 retry succeeds

        # The duplicate tool call (tc_dup) should NOT appear in streamed output
        assert b'tc_dup' not in written
        # First accepted tool call (tc1) was already streamed before dedup kicked in
        assert b'tc1' in written
        # Accepted tool call from retry SHOULD appear in streamed output
        assert b'tc2' in written
        assert "read_file" in extract_sse_tool_call_name(written)
        assert '{"path": "output.txt"}' in extract_sse_field(written, "arguments")


class TestMultiTurnReasoningPreservation:
    """Test reasoning_content recovery across multi-turn tool call scenarios
    when the client strips reasoning tokens from assistant messages."""

    async def test_reasoning_preserved_across_three_turns(self) -> None:
        """Turn 1: LLM reasons then generates tool call.
        Turn 2: tool response arrives (reasoning stripped from turn 1). LLM reasons again, generates next tool call.
        Turn 3: tool response arrives (reasoning stripped from both assistant msgs). LLM generates final output.
        Verify reasoning is recovered and forwarded to upstream in turn 2 and turn 3."""
        plugin = OpenAIGuardRail({})

        reasoning1 = "let me check the config file first"
        content1 = "I'll read the config"
        tc1_name = "read_file"

        reasoning2 = "config loaded, now I need to write the output"
        content2 = "Processing complete, writing result"
        tc2_name = "write_file"
        tc2_args = '{"path":"output.txt","data":"result"}'

        reasoning3 = "both operations completed successfully"
        content3 = "Done! Files processed."

        # Turn 1: LLM generates reasoning + tool call
        await _drive_tool_call_turn(plugin, tc1_name, reasoning=reasoning1, content=content1)

        # Turn 2: tool response arrives; reasoning from turn 1 is STRIPPED by client
        turn2_client = _make_client_response()

        turn2 = _Upstream(
            _tool_call_reply(tc2_name, "2", "tc2", reasoning2, content2, tc2_args),
        )

        await _drive_result(plugin, turn2.forward, turn2_client, _request([
            _user_message(),
            _assistant_tool_call(tc1_name, TOOL_ARGS, content=content1),
            _tool_result("file content", "tc1"),
        ]))

        # Verify reasoning from turn 1 was recovered in forwarded body
        assert turn2.calls == 1
        assistant_msgs = _forwarded_assistant_messages(turn2)
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0]["reasoning_content"] == reasoning1
        assert assistant_msgs[0]["content"] == content1

        # Turn 3: second tool response arrives; reasoning from BOTH assistant msgs is STRIPPED
        turn3_client = _make_client_response()

        turn3 = _Upstream(
            sse_reasoning("3", reasoning3)
            + sse_content("3", content3)
            + sse_finish_reason("3", "stop")
            + sse_done(),
        )

        await _drive_result(plugin, turn3.forward, turn3_client, _request([
            _user_message(),
            _assistant_tool_call(tc1_name, TOOL_ARGS, content=content1),
            _tool_result("file content", "tc1"),
            _assistant_tool_call(tc2_name, tc2_args, tc_id="tc2", content=content2),
            _tool_result("written", "tc2"),
        ]))

        # Verify reasoning from BOTH assistant turns was recovered
        assert turn3.calls == 1
        assistant_msgs = _forwarded_assistant_messages(turn3)
        assert len(assistant_msgs) == 2
        assert assistant_msgs[0]["reasoning_content"] == reasoning1
        assert assistant_msgs[0]["content"] == content1
        assert assistant_msgs[1]["reasoning_content"] == reasoning2
        assert assistant_msgs[1]["content"] == content2


class TestTelemetry:
    async def test_generic_loop_increases_count(self) -> None:
        """Trigger a generic loop interruption and verify UI count."""
        plugin = OpenAIGuardRail({"break_loops": True})
        upstream = _loop_upstream()

        html = await _single_panel(plugin, upstream)
        assert _panel_count(html, "Generic Loop") == 1
        assert _panel_count(html, "Total") == 1

    async def test_duplicate_tool_call_increases_count(self) -> None:
        """Trigger a duplicate tool call interruption and verify UI count."""
        plugin = OpenAIGuardRail({"dedup_tool_calls": True})
        upstream = _duplicate_tool_call_upstream("write_file", {"path": "a.txt"})

        html = await _single_panel(plugin, upstream)
        assert _panel_count(html, "Duplicate Tool Call") == 1

    async def test_no_final_output_increases_count(self) -> None:
        """Trigger a no-final-output interruption and verify UI count."""
        plugin = OpenAIGuardRail({"force_final_output": True})
        upstream = _no_final_output_upstream()

        html = await _single_panel(plugin, upstream)
        assert _panel_count(html, "No Final Output") == 1

    async def test_multiple_interruptions_accumulate(self) -> None:
        """Multiple interruptions from different types accumulate in total count."""
        plugin = OpenAIGuardRail({"break_loops": True, "force_final_output": True, "max_retry_attempt": 2})
        client = _make_client_response()
        published = _capture_ui(plugin)

        upstream = _Upstream(
            _loop_reply(),
            _reasoning_only_reply(),
            _final_answer_reply(),
        )

        await _drive_result(plugin, upstream.forward, client)
        assert upstream.calls == 3
        assert len(published) >= 2
        final_html = published[-1][1]
        assert _panel_count(final_html, "Total") == 2

    async def test_timestamps_show_after_interruption(self) -> None:
        """Timestamps show relative time after an interruption occurs."""
        plugin = OpenAIGuardRail({"break_loops": True})
        upstream = _loop_upstream()

        html = await _single_panel(plugin, upstream)
        assert "Last: just now" in html


class TestTelemetryIntegration:
    async def test_no_interruptions_keeps_zero_counts(self) -> None:
        plugin = OpenAIGuardRail({"break_loops": True, "force_final_output": True})
        client = _make_client_response()
        published = _capture_ui(plugin)

        upstream = _Upstream(
            sse_content("1", "ok") + sse_done(),
        )

        await _drive_result(plugin, upstream.forward, client)
        assert upstream.calls == 1
        assert len(published) == 0


class TestOngoingRequestLifecycle:
    async def test_on_request_adds_to_ongoing(self) -> None:
        """Verify on_request adds the request to the ongoing tracking set."""
        plugin = OpenAIGuardRail({})
        await plugin.on_request(1, {}, STREAM_REQUEST, CHAT_COMPLETIONS)
        assert plugin.is_request_ongoing(1) is True

    async def test_on_response_chunk_final_removes_from_ongoing(self) -> None:
        """Verify on_response_chunk with is_final=True removes the request from ongoing."""
        plugin = OpenAIGuardRail({})
        await plugin.on_request(1, {}, STREAM_REQUEST, CHAT_COMPLETIONS)
        await plugin.on_response_chunk(1, {}, b"", CHAT_COMPLETIONS, 200, is_final=True)
        assert plugin.is_request_ongoing(1) is False

    async def test_on_cancelled_removes_from_ongoing(self) -> None:
        """Verify on_cancelled removes the request from ongoing."""
        plugin = OpenAIGuardRail({})
        await plugin.on_request(1, {}, STREAM_REQUEST, CHAT_COMPLETIONS)
        plugin.on_cancelled(1)
        assert plugin.is_request_ongoing(1) is False


class TestDedupToolCallExclusions:
    """Test tool_call_dedup_exclusions keeping matching duplicates out of dedup retry."""

    async def test_excluded_by_name_no_retry(self) -> None:
        """A duplicate tool call whose name matches an exclusion is not retried."""
        plugin = OpenAIGuardRail(
            {
                "dedup_tool_calls": True,
                "tool_call_dedup_exclusions": [{"tool_name_regex": "(?i)read"}],
            }
        )
        client = _make_client_response()

        upstream = _duplicate_tool_call_upstream("read_file", {"path": "a.txt"})

        await _drive_result(plugin, upstream.forward, client)
        assert upstream.calls == 1

    async def test_excluded_by_args_no_retry(self) -> None:
        """A duplicate tool call whose arguments match an exclusion is not retried."""
        plugin = OpenAIGuardRail(
            {
                "dedup_tool_calls": True,
                "tool_call_dedup_exclusions": [{"tool_args_regex": "(?i)skills?\\.md"}],
            }
        )
        client = _make_client_response()

        upstream = _duplicate_tool_call_upstream("process", {"path": "skills.md"})

        await _drive_result(plugin, upstream.forward, client)
        assert upstream.calls == 1

    async def test_non_matching_exclusion_still_retries(self) -> None:
        """An exclusion that does not match the duplicate still triggers retry."""
        plugin = OpenAIGuardRail(
            {
                "dedup_tool_calls": True,
                "tool_call_dedup_exclusions": [{"tool_name_regex": "(?i)read"}],
            }
        )
        client = _make_client_response()

        upstream = _duplicate_tool_call_upstream("write_file", {"path": "a.txt"})

        await _drive_result(plugin, upstream.forward, client)
        assert upstream.calls == 2
        _assert_retry_names_tool_call(upstream, "write_file")

    async def test_entry_requires_both_patterns(self) -> None:
        """An entry carrying both patterns only excludes when both match."""
        plugin = OpenAIGuardRail(
            {
                "dedup_tool_calls": True,
                "tool_call_dedup_exclusions": [
                    {"tool_name_regex": "(?i)read", "tool_args_regex": "(?i)skills?\\.md"}
                ],
            }
        )
        client = _make_client_response()

        upstream = _duplicate_tool_call_upstream("read_file", {"path": "other.txt"})

        await _drive_result(plugin, upstream.forward, client)
        assert upstream.calls == 2
        _assert_retry_names_tool_call(upstream, "read_file")

    async def test_mixed_excluded_and_non_excluded(self) -> None:
        """Exclusion is per tool call: the non-excluded duplicate still triggers retry."""
        plugin = OpenAIGuardRail(
            {
                "dedup_tool_calls": True,
                "tool_call_dedup_exclusions": [{"tool_name_regex": "(?i)read"}],
            }
        )
        client = _make_client_response()

        upstream = _Upstream(
            sse_tool_call("tc1", "read_file", {"path": "a.txt"})
            + sse_tool_call("tc2", "read_file", {"path": "a.txt"})
            + sse_tool_call("tc3", "write_file", {"path": "b.txt"})
            + sse_tool_call("tc4", "write_file", {"path": "b.txt"})
            + sse_finish_reason("1", "stop")
            + sse_done(),
            _recovered_reply(finish_reason="stop"),
        )

        await _drive_result(plugin, upstream.forward, client)
        assert upstream.calls == 2
        _assert_retry_names_tool_call(upstream, "write_file")


# ── Same work on every API ──


# What a looping attempt sends of its second block before the loop is caught.
_LOOP_PARTIAL = LOOP_BLOCK[:66]
# What the second looping attempt sends, with the first retry pair in its history.
_SECOND_LOOP_PARTIAL = LOOP_BLOCK[:63]
# Assistant history text that the next reply repeats.
_HISTORY_TEXT = "history_block_content_" * 3


def _without_done(reply: bytes) -> bytes:
    """Drop the chat [DONE] sentinel from a reply, leaving what a client receives."""
    return reply.replace(sse_done(), b"")


async def _drive_wire(
    plugin: OpenAIGuardRail, wire: Wire, upstream: _Upstream, body: bytes
) -> tuple[bytes, MagicMock]:
    """Run one request of the wire's API through the guard rail."""
    client = _make_client_response()
    await _drive_result(plugin, upstream.forward, client, body, wire.path)
    return _written(client), client


def _first_turn(wire: Wire, stream: Optional[bool] = True) -> bytes:
    """The first request on the given wire."""
    return wire.request([_user_message()], stream)


class TestApiParityBasics:
    """The guard rail streams the same way on every API."""

    async def test_forces_stream(self, wire: Wire) -> None:
        """A request that did not ask to stream is sent on as a stream with its new length."""
        upstream = _Upstream(wire.reply("1", ("content", "hi")))
        await _drive_wire(OpenAIGuardRail({}), wire, upstream, _first_turn(wire, stream=False))
        assert _forwarded_body(upstream)["stream"] is True
        assert int(upstream.headers[-1]["Content-Length"]) == len(upstream.bodies[-1])

    async def test_responses_stream_is_forced_without_stream_options(self) -> None:
        """A forced Responses stream gets no stream options."""
        upstream = _Upstream(RESPONSES_WIRE.reply("1", ("content", "hi")))
        await _drive_wire(OpenAIGuardRail({}), RESPONSES_WIRE, upstream, _first_turn(RESPONSES_WIRE, stream=False))
        assert "stream_options" not in _forwarded_body(upstream)

    async def test_single_attempt_reaches_client_unchanged(self, wire: Wire) -> None:
        """A reply that passes every check reaches the client as it was sent."""
        reply = wire.reply("1", ("reasoning", "think"), ("content", "hello"), ("tool_call", "tc1", "read", {"p": 1}))
        written, client = await _drive_wire(
            OpenAIGuardRail({"break_loops": True, "dedup_tool_calls": True, "force_final_output": True}),
            wire, _Upstream(reply), _first_turn(wire),
        )
        assert written == _without_done(reply)
        assert client.write_eof.await_count == 1

    async def test_path_with_query_is_read_as_its_api(self, wire: Wire) -> None:
        """A request path with a query is checked in the format of its own API."""
        reply = wire.reply("1", ("reasoning", "think"), ("content", "hello"))
        upstream = _Upstream(reply)
        client = _make_client_response()
        await _drive_result(
            OpenAIGuardRail({"force_final_output": True}), upstream.forward, client,
            _first_turn(wire), wire.path + "?api-version=1",
        )
        assert upstream.calls == 1
        assert _written(client) == _without_done(reply)


class TestApiParityChecks:
    """Each check retries the reply on every API."""

    async def test_loop_is_retried(self, wire: Wire) -> None:
        """A looping reply is cut, retried, and the client gets the loop once."""
        upstream = _Upstream(
            wire.reply("1", ("content", LOOP_BLOCK), ("content", LOOP_BLOCK), ("content", LOOP_BLOCK), finish=None),
            wire.reply("2", ("content", "recovered")),
        )
        written, _ = await _drive_wire(OpenAIGuardRail({"break_loops": True}), wire, upstream, _first_turn(wire))
        assert upstream.calls == 2
        assert wire.content_of(written).endswith("recovered")
        assert wire.content_of(written).count(LOOP_BLOCK) == 1
        assert "repetition loop" in wire.system_texts(upstream.bodies[-1])[-1]

    async def test_duplicate_tool_call_is_retried(self, wire: Wire) -> None:
        """A repeated tool call is retried with a note that names the tool."""
        upstream = _Upstream(
            wire.reply("1", ("tool_call", "tc1", "read_file", {"path": "a"}), ("tool_call", "tc2", "read_file", {"path": "a"})),
            wire.reply("2", ("content", "recovered")),
        )
        await _drive_wire(OpenAIGuardRail({"dedup_tool_calls": True}), wire, upstream, _first_turn(wire))
        assert upstream.calls == 2
        system_text = wire.system_texts(upstream.bodies[-1])[-1]
        assert "repeating the same tool call" in system_text
        assert "read_file" in system_text

    async def test_invalid_tool_call_is_retried(self, wire: Wire) -> None:
        """A tool call with broken JSON arguments is retried."""
        upstream = _Upstream(
            wire.reply("1", ("tool_call", "tc1", "write_file", '{"path": ')),
            wire.reply("2", ("content", "recovered")),
        )
        await _drive_wire(OpenAIGuardRail({"validate_tool_calls": True}), wire, upstream, _first_turn(wire))
        assert upstream.calls == 2
        assert "invalid JSON arguments" in wire.system_texts(upstream.bodies[-1])[-1]

    @pytest.mark.parametrize("finish", ["stop", None], ids=["stop", "eof"])
    async def test_missing_final_output_is_retried(self, wire: Wire, finish: Optional[str]) -> None:
        """A reply with only reasoning is retried until it gives text."""
        upstream = _Upstream(
            wire.reply("1", ("reasoning", THINKING_ONLY), finish=finish),
            wire.reply("2", ("content", "final_answer")),
        )
        written, _ = await _drive_wire(OpenAIGuardRail({"force_final_output": True}), wire, upstream, _first_turn(wire))
        assert upstream.calls == 2
        assert wire.content_of(written) == "final_answer"
        assert "did not contain any text output" in wire.system_texts(upstream.bodies[-1])[-1]

    async def test_retry_body_holds_the_original_input_and_one_pair(self, wire: Wire) -> None:
        """Each retry sends the first input plus one partial reply and one note."""
        loop = wire.reply("1", ("content", LOOP_BLOCK), ("content", LOOP_BLOCK), ("content", LOOP_BLOCK), finish=None)
        upstream = _Upstream(loop, loop, wire.reply("3", ("content", "recovered")))
        body = wire.request([_user_message("first ask")])
        await _drive_wire(OpenAIGuardRail({"break_loops": True}), wire, upstream, body)
        assert upstream.calls == 3
        third = wire.history(upstream.bodies[2])
        assert [m["role"] for m in third] == ["user", "assistant", "system"]
        assert third[0]["content"] == "first ask"
        assert wire.history(upstream.bodies[1])[:1] == third[:1]
        assert third[1]["content"] == _SECOND_LOOP_PARTIAL

    async def test_history_primes_the_loop_detector(self, wire: Wire) -> None:
        """Assistant text in the history counts toward a loop in the new reply."""
        upstream = _Upstream(
            wire.reply("1", ("content", _HISTORY_TEXT), finish=None),
            wire.reply("2", ("content", "recovered")),
        )
        body = wire.request([_user_message(), _assistant_message(_HISTORY_TEXT)])
        await _drive_wire(OpenAIGuardRail({"break_loops": True}), wire, upstream, body)
        assert upstream.calls == 2

    async def test_responses_string_history_primes_the_loop_detector(self) -> None:
        """Assistant text given as a plain string in Responses input counts toward a loop."""
        upstream = _Upstream(
            RESPONSES_WIRE.reply("1", ("content", _HISTORY_TEXT), finish=None),
            RESPONSES_WIRE.reply("2", ("content", "recovered")),
        )
        body = responses_request([{"role": "user", "content": "do something"}, {"role": "assistant", "content": _HISTORY_TEXT}])
        await _drive_wire(OpenAIGuardRail({"break_loops": True}), RESPONSES_WIRE, upstream, body)
        assert upstream.calls == 2

    async def test_history_tool_call_primes_the_duplicate_detector(self, wire: Wire) -> None:
        """A tool call in the history counts as a repeat for the new reply."""
        upstream = _Upstream(
            wire.reply("1", ("tool_call", "tc2", "read_file", {"path": "a"})),
            wire.reply("2", ("content", "recovered")),
        )
        body = wire.request([_user_message(), _assistant_tool_call("read_file", '{"path": "a"}'), _tool_result("r", "tc1")])
        await _drive_wire(OpenAIGuardRail({"dedup_tool_calls": True}), wire, upstream, body)
        assert upstream.calls == 2

    async def test_generation_error_is_retried(self, wire: Wire) -> None:
        """A reply that fails while generating is retried."""
        failing = wire.attempt("1")
        upstream = _Upstream(failing.start() + failing.failure(), wire.reply("2", ("content", "recovered")))
        await _drive_wire(OpenAIGuardRail({"force_final_output": True}), wire, upstream, _first_turn(wire))
        assert upstream.calls == 2


async def _non_streaming_retry(wire: Wire) -> tuple[dict, MagicMock]:
    """Run a non-streaming request whose first attempt loops, and read the one reply."""
    upstream = _Upstream(
        wire.reply("1", ("content", LOOP_BLOCK), ("content", LOOP_BLOCK), ("content", LOOP_BLOCK), finish=None),
        wire.reply("2", ("content", "recovered")),
    )
    written, client = await _drive_wire(OpenAIGuardRail({"break_loops": True}), wire, upstream, _first_turn(wire, stream=False))
    return json.loads(written), client


class TestApiParityOutput:
    """A retried reply reaches the client as one reply on every API."""

    async def test_non_streaming_chat_retry_gets_one_reply(self) -> None:
        """A retried chat request that did not ask to stream gets one JSON reply with all the text."""
        reply, client = await _non_streaming_retry(CHAT_WIRE)
        assert client.headers["Content-Type"] == "application/json"
        assert reply["choices"][0]["message"]["content"] == LOOP_BLOCK + _LOOP_PARTIAL + "recovered"

    async def test_non_streaming_responses_retry_gets_one_reply(self) -> None:
        """A retried Responses request that did not ask to stream gets one reply holding every item."""
        reply, client = await _non_streaming_retry(RESPONSES_WIRE)
        assert client.headers["Content-Type"] == "application/json"
        assert reply["object"] == "response"
        assert [(item["status"], item["content"][0]["text"]) for item in reply["output"]] == [
            ("completed", LOOP_BLOCK), ("incomplete", _LOOP_PARTIAL), ("completed", "recovered"),
        ]

    async def test_retried_responses_stream_is_one_valid_stream(self) -> None:
        """A retried Responses stream reaches the client as one numbered stream."""
        wire = RESPONSES_WIRE
        upstream = _Upstream(
            wire.reply("1", ("reasoning", "plan"), ("content", LOOP_BLOCK), ("content", LOOP_BLOCK), ("content", LOOP_BLOCK), finish=None),
            wire.reply("2", ("content", "recovered")),
        )
        written, _ = await _drive_wire(OpenAIGuardRail({"break_loops": True}), wire, upstream, _first_turn(wire))
        payloads = parse_sse_payloads(written)
        assert [p["type"] for p in payloads].count("response.created") == 1
        assert [p["sequence_number"] for p in payloads] == list(range(len(payloads)))
        added = [p["output_index"] for p in payloads if p["type"] == "response.output_item.added"]
        assert added == list(range(len(added)))
        terminal = payloads[-1]["response"]
        assert terminal["id"] == "resp_1"
        assert [(item["type"], item["status"]) for item in terminal["output"]] == [
            ("reasoning", "completed"), ("message", "completed"), ("message", "incomplete"), ("message", "completed"),
        ]

    async def test_spliced_usage_is_the_last_attempts(self) -> None:
        """A retried Responses stream reports the usage of the last attempt."""
        wire = RESPONSES_WIRE
        retried = wire.attempt("2")
        upstream = _Upstream(
            wire.reply("1", ("content", LOOP_BLOCK), ("content", LOOP_BLOCK), ("content", LOOP_BLOCK), finish=None),
            retried.start() + retried.content("recovered") + retried.finish("stop", (7, 8, 15)),
        )
        written, _ = await _drive_wire(OpenAIGuardRail({"break_loops": True}), wire, upstream, _first_turn(wire))
        assert parse_sse_payloads(written)[-1]["response"]["usage"] == {"input_tokens": 7, "output_tokens": 8, "total_tokens": 15}


# The assistant turn the guard rail puts back for a call the client left out.
_INSERTED_TURN = {
    "role": "assistant", "reasoning_content": "plan the read",
    "tool_calls": [{"id": "call1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a"}'}}],
}


class TestApiParityResumption:
    """Saved turns go back into the next request on every API."""

    async def _first_turn_with_call(
        self, plugin: OpenAIGuardRail, wire: Wire, call_id: str, terminal_ids: Optional[dict[str, str]] = None
    ) -> None:
        """Run a first turn that ends with one call."""
        attempt = wire.attempt("1", terminal_ids)
        reply = attempt.start() + attempt.reasoning("plan the read") + attempt.tool_call(call_id, "read_file", {"path": "a"}) + attempt.finish("stop")
        upstream = _Upstream(reply)
        await _drive_wire(plugin, wire, upstream, _first_turn(wire))
        assert upstream.calls == 1

    @pytest.mark.parametrize(
        "wire, expected_turn",
        [
            pytest.param(CHAT_WIRE, {**_INSERTED_TURN, "content": ""}, id="chat"),
            pytest.param(RESPONSES_WIRE, _INSERTED_TURN, id="responses"),
        ],
    )
    async def test_missing_turn_is_inserted(self, wire: Wire, expected_turn: dict) -> None:
        """A tool result whose call turn the client left out gets that turn put back."""
        plugin = OpenAIGuardRail({})
        await self._first_turn_with_call(plugin, wire, "call_1")
        upstream = _Upstream(wire.reply("2", ("content", "done")))
        await _drive_wire(plugin, wire, upstream, wire.request([_user_message(), _tool_result("file", "call_1")]))
        assert wire.assistant_turns(upstream.bodies[-1]) == [expected_turn]

    async def test_reasoning_is_restored(self, wire: Wire) -> None:
        """Reasoning the client left out of a call turn is put back."""
        plugin = OpenAIGuardRail({})
        await self._first_turn_with_call(plugin, wire, "call_1")
        upstream = _Upstream(wire.reply("2", ("content", "done")))
        history = [_user_message(), _assistant_tool_call("read_file", '{"path": "a"}', tc_id="call_1"), _tool_result("file", "call_1")]
        await _drive_wire(plugin, wire, upstream, wire.request(history))
        turns = wire.assistant_turns(upstream.bodies[-1])
        assert len(turns) == 1
        assert turns[0]["reasoning_content"] == "plan the read"
        assert turns[0]["tool_calls"][0]["id"] == "call_1"

    async def test_terminal_ids_insert_the_turn_under_the_clients_id(self) -> None:
        """A missing turn is put back under the client's normalized call id."""
        wire = RESPONSES_WIRE
        plugin = OpenAIGuardRail({})
        await self._first_turn_with_call(plugin, wire, "call_1", {"call_1": "chatcmpl-tool-9"})
        upstream = _Upstream(wire.reply("2", ("content", "done")))
        await _drive_wire(plugin, wire, upstream, wire.request([_user_message(), _tool_result("file", "chatcmpl-tool-9")]))
        items = json.loads(upstream.bodies[-1])["input"]
        assert items[1:] == [
            {"type": "reasoning", "summary": [], "content": [{"type": "reasoning_text", "text": "plan the read"}]},
            {"type": "function_call", "call_id": "chatcmpltool9", "name": "read_file", "arguments": '{"path": "a"}'},
            {"type": "function_call_output", "call_id": "chatcmpl-tool-9", "output": "file"},
        ]

    async def test_terminal_ids_restore_reasoning_without_a_duplicate_turn(self) -> None:
        """Reasoning is put back on the turn the client sent under the final call id."""
        wire = RESPONSES_WIRE
        plugin = OpenAIGuardRail({})
        await self._first_turn_with_call(plugin, wire, "call_1", {"call_1": "chatcmpl-tool-9"})
        upstream = _Upstream(wire.reply("2", ("content", "done")))
        history = [_user_message(), _assistant_tool_call("read_file", '{"path": "a"}', tc_id="chatcmpl-tool-9"), _tool_result("file", "chatcmpl-tool-9")]
        await _drive_wire(plugin, wire, upstream, wire.request(history))
        turns = wire.assistant_turns(upstream.bodies[-1])
        assert len(turns) == 1
        assert turns[0]["reasoning_content"] == "plan the read"
        assert turns[0]["tool_calls"][0]["id"] == "chatcmpl-tool-9"

    async def test_call_completed_before_a_retry_is_resumed(self, wire: Wire) -> None:
        """A call sent before a retry is put back with the call from the retry."""
        plugin = OpenAIGuardRail({"dedup_tool_calls": True})
        upstream = _Upstream(
            wire.reply("1", ("tool_call", "callA", "read_file", {"path": "a"}), ("tool_call", "callB", "read_file", {"path": "a"})),
            wire.reply("2", ("tool_call", "callC", "write_file", {"path": "b"})),
        )
        await _drive_wire(plugin, wire, upstream, _first_turn(wire))
        assert upstream.calls == 2
        follow_up = _Upstream(wire.reply("3", ("content", "done")))
        await _drive_wire(plugin, wire, follow_up, wire.request([_user_message(), _tool_result("ra", "callA"), _tool_result("rc", "callC")]))
        history = wire.history(follow_up.bodies[-1])
        assert [m["role"] for m in history] == ["user", "assistant", "tool", "assistant", "tool"]
        assert [m["tool_calls"][0]["id"] for m in history if m["role"] == "assistant"] == ["callA", "callC"]


# Each row holds the pieces, the chat label sent alone, the chat label packed into the last chunk,
# the Responses terminal status, which is None for a failed generation, and the upstream calls expected.
_FINISH_TABLE = [
    pytest.param([("tool_call", "tc1", "run", {"a": 1})], "tool_calls", "tool_calls", "completed", 1, id="stop-call-auto"),
    pytest.param([("tool_call", "tc1", "run", {"a": 1})], "tool_calls", "stop", "completed", 1, id="stop-call-named"),
    pytest.param([("tool_call", "tc1", "run", {"a": 1})], "stop", "stop", "completed", 1, id="stop-alone-with-call"),
    pytest.param([("content", "say"), ("tool_call", "tc1", "run", {"a": 1})], "tool_calls", "stop", "completed", 1, id="stop-call-named-content"),
    pytest.param([("tool_call", "tc1", "run", {"a": 1})], "length", "tool_calls", "incomplete", 1, id="length-call"),
    pytest.param([("reasoning", "thinking")], "length", "length", "incomplete", 1, id="length-reasoning-only"),
    pytest.param([("content", "partial")], "length", "length", "incomplete", 1, id="length-content"),
    pytest.param([("reasoning", "thinking")], "stop", "stop", "completed", 2, id="stop-nothing"),
    pytest.param([("content", "answer")], "stop", "stop", "completed", 1, id="stop-content"),
    pytest.param([], None, None, None, 2, id="error-nothing"),
    pytest.param([("content", "answer")], None, None, None, 1, id="error-after-content"),
]


def _pack_finish(reply: bytes, reason: str) -> bytes:
    """Move a finish reason into the last chunk."""
    events = [event for event in reply.split(b"\n\n") if event]
    last = json.loads(events[-1][len(b"data: "):])
    last["choices"][0]["finish_reason"] = reason
    events[-1] = b"data: " + json.dumps(last).encode()
    return b"\n\n".join(events) + b"\n\n" + sse_done()


def _finish_reply(shape: str, pieces: list[tuple], alone: Optional[str], packed: Optional[str], status: Optional[str]) -> bytes:
    """A reply on the given wire shape that ends with the given finish."""
    if shape == "responses":
        attempt = RESPONSES_WIRE.attempt("1")
        reply = attempt.start() + attempt.pieces(pieces)
        return reply + (attempt.stream.completed(status) if status else attempt.failure())
    reply = CHAT_WIRE.attempt("1").pieces(pieces)
    if alone is None:
        return reply + CHAT_WIRE.attempt("1").failure()
    if shape == "chat_packed" and pieces:
        return _pack_finish(reply, packed)
    return reply + sse_finish_reason("1", alone) + sse_done()


class TestFinalOutputDecision:
    """The final-output check decides on what was generated, the same on every wire shape."""

    @pytest.mark.parametrize("shape", ["responses", "chat_alone", "chat_packed"])
    @pytest.mark.parametrize("pieces, alone, packed, status, calls", _FINISH_TABLE)
    async def test_decision_table(
        self, shape: str, pieces: list[tuple], alone: Optional[str], packed: Optional[str], status: Optional[str], calls: int
    ) -> None:
        """Each finish shape is retried or kept the same way on every wire shape."""
        wire = RESPONSES_WIRE if shape == "responses" else CHAT_WIRE
        upstream = _Upstream(_finish_reply(shape, pieces, alone, packed, status), wire.reply("2", ("content", "final")))
        await _drive_wire(OpenAIGuardRail({"force_final_output": True}), wire, upstream, _first_turn(wire))
        assert upstream.calls == calls

    @pytest.mark.parametrize("shape", ["chat_alone", "chat_packed"])
    async def test_tool_calls_without_a_call_is_retried(self, shape: str) -> None:
        """A tool calls finish with no call and no text is retried."""
        reply = _finish_reply(shape, [("reasoning", "thinking")], "tool_calls", "tool_calls", "completed")
        upstream = _Upstream(reply, CHAT_WIRE.reply("2", ("content", "final")))
        await _drive_wire(OpenAIGuardRail({"force_final_output": True}), CHAT_WIRE, upstream, _first_turn(CHAT_WIRE))
        assert upstream.calls == 2


# ── Every piece of a chat chunk ──


def _chunk(delta: dict, finish_reason: Optional[str] = None, chat_id: str = "1") -> bytes:
    """One chat chunk with the given delta and finish reason."""
    choice: dict = {"index": 0, "delta": delta}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    return sse_event({"id": chat_id, "choices": [choice]})


def _writes(client: MagicMock) -> list[bytes]:
    """The bytes written to the client, in order."""
    return [call[0][0] for call in client.write.call_args_list]


def _events_of(reply: bytes) -> list[bytes]:
    """The events of a reply."""
    return [event + b"\n\n" for event in reply.split(b"\n\n") if event]


class TestChatChunkPieces:
    """A chat chunk with many pieces is checked piece by piece."""

    async def test_packed_stop_with_reasoning_only_relays_the_chunk_then_retries(self) -> None:
        """A stop packed with reasoning is sent on, then the reply is retried."""
        attempt_one = sse_reasoning("1", "thin") + _chunk({"reasoning": "king"}, "stop") + sse_done()
        attempt_two = CHAT_WIRE.reply("2", ("content", "final"))
        upstream = _Upstream(attempt_one, attempt_two)
        written, _ = await _drive_wire(OpenAIGuardRail({"force_final_output": True}), CHAT_WIRE, upstream, _first_turn(CHAT_WIRE))
        assert upstream.calls == 2
        assert written == _without_done(attempt_one) + _without_done(attempt_two)

    @pytest.mark.parametrize("text_field", ["content", "reasoning"])
    @pytest.mark.parametrize("failure", ["duplicate", "invalid"])
    async def test_text_packed_with_a_failing_calls_opening(self, text_field: str, failure: str) -> None:
        """Text packed with the opening of a failing call reaches the client before the retry."""
        call_a = sse_tool_call("tcA", "read_file", {"path": "a"})
        packed = _chunk({text_field: "ok", "tool_calls": [{"index": 1, "id": "tcB", "type": "function", "function": {"name": "read_file"}}]})
        b_args = '{"path": "a"}' if failure == "duplicate" else '{"path": '
        b_rest = b"".join(_chunk({"tool_calls": [{"index": 1, "function": {"arguments": b_args[i:i + 3]}}]}) for i in range(0, len(b_args), 3))
        attempt_one = call_a + packed + b_rest + sse_finish_reason("1", "stop") + sse_done()
        attempt_two = CHAT_WIRE.reply("2", ("content", "final"))
        upstream = _Upstream(attempt_one, attempt_two)
        config = {"dedup_tool_calls": True} if failure == "duplicate" else {"validate_tool_calls": True}
        written, client = await _drive_wire(OpenAIGuardRail(config), CHAT_WIRE, upstream, _first_turn(CHAT_WIRE))
        assert upstream.calls == 2
        assert written == call_a + packed + _without_done(attempt_two)
        packed_index = len(_events_of(call_a))
        assert b"".join(_writes(client)[: packed_index + 1]) == call_a + packed

    async def test_reasoning_and_content_packed_reach_the_saved_turn(self) -> None:
        """Reasoning and content packed in one chunk are both saved for the turn."""
        plugin = OpenAIGuardRail({})
        reply = _chunk({"reasoning": "why", "content": "say"}) + sse_tool_call("call_1", "read_file", {"p": 1}) + sse_finish_reason("1", "stop") + sse_done()
        await _drive_wire(plugin, CHAT_WIRE, _Upstream(reply), _first_turn(CHAT_WIRE))
        upstream = _Upstream(CHAT_WIRE.reply("2", ("content", "done")))
        await _drive_wire(plugin, CHAT_WIRE, upstream, CHAT_WIRE.request([_user_message(), _tool_result("r", "call_1")]))
        turn = CHAT_WIRE.assistant_turns(upstream.bodies[-1])[0]
        assert turn["reasoning_content"] == "why"
        assert turn["content"] == "say"

    async def test_loop_in_a_reasoning_tail_withholds_the_chunk(self) -> None:
        """The second copy of the block arrives as the reasoning tail packed before content."""
        reasoning = sse_reasoning("1", LOOP_BLOCK)
        packed = _chunk({"reasoning": LOOP_BLOCK, "content": "x"})
        upstream = _Upstream(reasoning + packed + sse_done(), CHAT_WIRE.reply("2", ("content", "recovered")))
        written, _ = await _drive_wire(OpenAIGuardRail({"break_loops": True}), CHAT_WIRE, upstream, _first_turn(CHAT_WIRE))
        assert upstream.calls == 2
        assert packed not in written

    async def test_two_fragments_in_one_delta_are_one_valid_call(self) -> None:
        """Two argument parts in one chunk make one valid call."""
        plugin = OpenAIGuardRail({"validate_tool_calls": True})
        opening = _chunk({"tool_calls": [
            {"index": 0, "id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": '{"path":'}},
            {"index": 0, "function": {"arguments": ' "a"}'}},
        ]})
        upstream = _Upstream(opening + sse_finish_reason("1", "stop") + sse_done())
        await _drive_wire(plugin, CHAT_WIRE, upstream, _first_turn(CHAT_WIRE))
        assert upstream.calls == 1
        follow_up = _Upstream(CHAT_WIRE.reply("2", ("content", "done")))
        await _drive_wire(plugin, CHAT_WIRE, follow_up, CHAT_WIRE.request([_user_message(), _tool_result("r", "call_1")]))
        saved = CHAT_WIRE.assistant_turns(follow_up.bodies[-1])[0]["tool_calls"][0]["function"]
        assert saved == {"name": "read_file", "arguments": '{"path": "a"}'}

    async def test_loop_completed_by_content_withholds_a_chunk_whose_reasoning_passed(self) -> None:
        """A chunk whose content ends a loop is held back even when its reasoning passed."""
        first = _chunk({"reasoning": "r", "content": LOOP_BLOCK})
        second = _chunk({"reasoning": "r", "content": LOOP_BLOCK})
        attempt_two = CHAT_WIRE.reply("2", ("content", "recovered"))
        upstream = _Upstream(first + second + sse_done(), attempt_two)
        written, _ = await _drive_wire(OpenAIGuardRail({"break_loops": True}), CHAT_WIRE, upstream, _first_turn(CHAT_WIRE))
        assert upstream.calls == 2
        assert written == first + _without_done(attempt_two)

    async def test_two_choices_read_all_text_before_finish(self) -> None:
        """Text in a second choice counts before the finish of the first."""
        chunk = sse_event({"id": "1", "choices": [
            {"index": 0, "delta": {"reasoning": "r"}, "finish_reason": "stop"},
            {"index": 1, "delta": {"content": "answer"}, "finish_reason": "stop"},
        ]})
        upstream = _Upstream(chunk + sse_done(), CHAT_WIRE.reply("2", ("content", "final")))
        await _drive_wire(OpenAIGuardRail({"force_final_output": True}), CHAT_WIRE, upstream, _first_turn(CHAT_WIRE))
        assert upstream.calls == 1

    async def test_failing_call_end_packed_with_next_opening_is_withheld(self) -> None:
        """A chunk that ends a failing call is held back with the next opening."""
        call_x = sse_tool_call("tcX", "read_file", {"path": "a"})
        a_args = '{"path": "a"}'
        a_open = _chunk({"tool_calls": [{"index": 1, "id": "tcA", "type": "function", "function": {"name": "read_file", "arguments": a_args[:-1]}}]})
        packed = _chunk({"tool_calls": [
            {"index": 1, "function": {"arguments": a_args[-1]}},
            {"index": 2, "id": "tcB", "type": "function", "function": {"name": "other"}},
        ]})
        attempt_two = CHAT_WIRE.reply("2", ("content", "final"))
        upstream = _Upstream(call_x + a_open + packed + sse_done(), attempt_two)
        written, _ = await _drive_wire(OpenAIGuardRail({"dedup_tool_calls": True}), CHAT_WIRE, upstream, _first_turn(CHAT_WIRE))
        assert upstream.calls == 2
        assert written == call_x + _without_done(attempt_two)

    async def test_valid_call_end_packed_with_failing_next_opening_is_sent_with_its_call(self) -> None:
        """A chunk that ends a good call is sent even when the next call it opens fails."""
        a_args = '{"path": "a"}'
        a_open = _chunk({"tool_calls": [{"index": 0, "id": "tcA", "type": "function", "function": {"name": "read_file", "arguments": a_args[:-1]}}]})
        packed = _chunk({"tool_calls": [
            {"index": 0, "function": {"arguments": a_args[-1]}},
            {"index": 1, "id": "tcB", "type": "function", "function": {"name": "read_file"}},
        ]})
        b_rest = _chunk({"tool_calls": [{"index": 1, "function": {"arguments": a_args}}]})
        attempt_two = CHAT_WIRE.reply("2", ("content", "final"))
        upstream = _Upstream(a_open + packed + b_rest + sse_finish_reason("1", "stop") + sse_done(), attempt_two)
        written, client = await _drive_wire(OpenAIGuardRail({"dedup_tool_calls": True}), CHAT_WIRE, upstream, _first_turn(CHAT_WIRE))
        assert upstream.calls == 2
        assert written == a_open + packed + _without_done(attempt_two)
        assert b"".join(_writes(client)[:2]) == a_open + packed

    async def test_content_packed_with_a_valid_calls_opening_sends_packed_chunk_first(self) -> None:
        """Content packed with a good call opening is sent first, then the rest of the call."""
        packed = _chunk({"content": "ok", "tool_calls": [{"index": 0, "id": "tcB", "type": "function", "function": {"name": "read_file"}}]})
        b_rest = _chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"p": 1}'}}]})
        finish = sse_finish_reason("1", "stop")
        upstream = _Upstream(packed + b_rest + finish + sse_done())
        written, client = await _drive_wire(OpenAIGuardRail({"dedup_tool_calls": True}), CHAT_WIRE, upstream, _first_turn(CHAT_WIRE))
        assert written == packed + b_rest + finish
        assert _writes(client)[:3] == [packed, b"", b_rest + finish]
