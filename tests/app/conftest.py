"""Shared test fixtures for GPU priority proxy tests."""

import abc
import asyncio
import hashlib
import json
import re
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import (
    AsyncIterator,
    Awaitable,
    Callable,
    ClassVar,
    Iterator,
    NamedTuple,
    Optional,
    Self,
    override,
)
from unittest.mock import AsyncMock, MagicMock, patch

import pynvml
import pytest
from aiodocker.containers import DockerContainer
from aiohttp import web

from app.config import ProxyConfig, ServiceConfig
from app.gpu_monitor import ContainerVRAM, GPUStatus
from app.plugins.base import PluginBase
from app.plugins.openai_chat_modifier import OpenAIChatRequestModifier
from app.plugins.openai_token_stats import OpenAITokenStats
from app.responses import AbstractMutatingResponse
from app.service_state import ServiceState
from app.wait_queue import VRAMWaitQueue

_SSE_DATA_PREFIX = b"data: "

# Real providers emit a field a few characters at a time, so the helpers below
# slice their text the same way rather than sending it in one delta.
_DELTA_CHARS = 3

_TOTAL_VRAM_MIB = 24576
_SYSTEM_VRAM_MIB = 860
_FREE_VRAM_MIB = 22970
_LLAMACPP_VRAM_MIB = 746
_CPAI_VRAM_MIB = 396
_POLL_TIME = 1744749600.0

# Long enough for the upstream to receive the request, so the cancellation lands
# while the forward is awaiting a reply rather than before it starts.
_CANCEL_FORWARD_DELAY_S = 0.1

_ForwardFn = Callable[
    [dict[str, str], bytes, str, web.Request, AbstractMutatingResponse],
    Awaitable[bool],
]

# The (title, html) pairs a plugin handed to its UI publish callback.
PublishedPanels = list[tuple[str, str]]


# ── Clock ──


class FakeClock:
    """Injectable clock for deterministic timeout testing.

    This clock allows tests to advance time manually without real waits,
    making timeout tests fast and deterministic.
    """

    def __init__(self, start_time: float = 0.0) -> None:
        self._time = start_time
        self._sleep_waiters: list[asyncio.Event] = []

    def time(self) -> float:
        """Return current virtual time in seconds."""
        return self._time

    async def sleep(self, seconds: float) -> None:
        """Sleep virtually: advance time and wake waiters."""
        self.advance(seconds)

    def advance(self, seconds: float) -> None:
        """Advance virtual time by the given number of seconds."""
        self._time += seconds
        for event in self._sleep_waiters:
            event.set()
        self._sleep_waiters.clear()

    async def wait_for_sleepers(self) -> None:
        """Wait until at least one sleep() call is pending."""
        event = asyncio.Event()
        self._sleep_waiters.append(event)
        await event.wait()


# ── SSE helpers ──


def sse_event(data: dict) -> bytes:
    """Create a single SSE event."""
    return _SSE_DATA_PREFIX + json.dumps(data).encode() + b"\n\n"


def sse_delta_content(request_id: str, text: str) -> bytes:
    """Create a single SSE delta event with content, no chunking."""
    return sse_event({"id": request_id, "choices": [{"delta": {"content": text}}]})


def _sse_deltas(
    request_id: str, field: str, text: str, extra: dict | None = None
) -> bytes:
    """Create SSE events carrying one delta field, split into small chunks.

    Args:
        request_id: The id every event carries.
        field: The delta field that holds the text.
        text: The text to split.
        extra: More fields to put in every delta.
    """
    return b"".join(
        sse_event(
            {
                "id": request_id,
                "choices": [{"delta": {field: text[i:i + _DELTA_CHARS], **(extra or {})}}],
            }
        )
        for i in range(0, len(text), _DELTA_CHARS)
    )


def sse_content(request_id: str, text: str) -> bytes:
    """Create SSE content delta events, split into small chunks."""
    return _sse_deltas(request_id, "content", text)


def sse_reasoning(request_id: str, text: str) -> bytes:
    """Create SSE reasoning delta events, split into small chunks."""
    return _sse_deltas(request_id, "reasoning", text)


def sse_finish_reason(request_id: str, reason: str) -> bytes:
    """Create an SSE finish_reason event with empty delta."""
    return sse_event({"id": request_id, "choices": [{"delta": {}, "finish_reason": reason}]})


def sse_done() -> bytes:
    """Create the [DONE] sentinel."""
    return _SSE_DATA_PREFIX + b"[DONE]\n\n"


def _sse_tool_call_event(request_id: str, fragment: dict) -> bytes:
    """Create one SSE event carrying a tool call fragment."""
    return sse_event({"id": request_id, "choices": [{"delta": {"tool_calls": [fragment]}}]})


def _sse_tool_call_deltas(request_id: str, index: int, field: str, text: str) -> bytes:
    """Create SSE tool call events carrying one function field, split into small chunks."""
    return b"".join(
        _sse_tool_call_event(
            request_id, {"index": index, "function": {field: text[i:i + _DELTA_CHARS]}}
        )
        for i in range(0, len(text), _DELTA_CHARS)
    )


def sse_tool_call(
    tc_id: str,
    name: str,
    arguments: dict | str | None = None,
    request_id: str = "1",
    index: int = 0,
) -> bytes:
    """Generate realistic streaming tool call SSE events.

    In real OpenAI responses, id, name, and arguments appear in separate small deltas.
    The function dict always exists, as {} when empty.
    """
    args = {} if arguments is None else arguments
    opening = _sse_tool_call_event(
        request_id, {"id": tc_id, "type": "function", "index": index, "function": {}}
    )
    return (
        opening
        + _sse_tool_call_deltas(request_id, index, "name", name)
        + _sse_tool_call_deltas(request_id, index, "arguments", json.dumps(args))
    )


def sse_tool_call_null_fields(
    tc_id: str,
    name: str,
    arguments: dict | str | None = None,
    request_id: str = "1",
    index: int = 0,
) -> bytes:
    """Create tool call SSE events that send the unused fields as null.

    The id and the name are null on every delta after the first one.
    """
    args = {} if arguments is None else arguments
    text = json.dumps(args) if isinstance(args, dict) else args
    opening = _sse_tool_call_event(
        request_id,
        {
            "id": tc_id,
            "type": "function",
            "index": index,
            "function": {"name": name, "arguments": ""},
        },
    )
    return opening + b"".join(
        _sse_tool_call_event(
            request_id,
            {
                "index": index,
                "id": None,
                "function": {"name": None, "arguments": text[i:i + _DELTA_CHARS]},
            },
        )
        for i in range(0, len(text), _DELTA_CHARS)
    )


def sse_reasoning_content_null(request_id: str, text: str) -> bytes:
    """Create content SSE events that send the unused fields as null.

    The reasoning content and the tool calls are null on every delta.
    """
    return _sse_deltas(
        request_id,
        "content",
        text,
        {"role": "assistant", "reasoning_content": None, "tool_calls": None},
    )


def sse_usage(prompt_tokens: int, completion_tokens: int, total_tokens: int = 0) -> bytes:
    """Create an SSE usage event with prompt/completion/total token counts."""
    usage: dict[str, int] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }
    if total_tokens:
        usage["total_tokens"] = total_tokens
    return sse_event({"usage": usage})


def extract_sse_field(written: bytes, field: str = "content") -> str:
    """Extract and join all chunked values from written SSE bytes."""
    result = ""
    for raw_line in written.split(b"\n"):
        line = raw_line.strip()
        if not line.startswith(_SSE_DATA_PREFIX):
            continue
        data = json.loads(line[len(_SSE_DATA_PREFIX):])
        choices = data.get("choices", [])
        if not choices:
            continue
        delta = choices[0].get("delta", {})
        # A field the delta is not using can be null instead of missing.
        if delta.get(field) is not None:
            result += delta[field]
            continue
        for tool_call in delta.get("tool_calls") or []:
            function = tool_call.get("function", {})
            if function.get(field) is not None:
                result += function[field]
    return result


def extract_sse_tool_call_name(written: bytes) -> str:
    """Extract and join chunked tool call names."""
    return extract_sse_field(written, "name")


def extract_sse_tool_call_args(written: bytes) -> dict:
    """Extract and join chunked tool call arguments, returning parsed dict."""
    return json.loads(extract_sse_field(written, "arguments"))


# ── Completion modifier helpers ──


CHAT_COMPLETIONS = "/v1/chat/completions"
_REQUEST_ID = 1


def make_chat_modifier(overrides: dict) -> OpenAIChatRequestModifier:
    """A chat modifier configured to apply the given overrides.

    Args:
        overrides: The request fields the plugin should force.

    Returns:
        The configured plugin.
    """
    return OpenAIChatRequestModifier({"overrides": overrides})


def json_body(fields: dict) -> bytes:
    """A JSON request body carrying the given fields.

    Args:
        fields: The top-level mapping of the body.

    Returns:
        The encoded body.
    """
    return json.dumps(fields).encode()


async def apply_on_request(
    plugin: PluginBase,
    body: bytes,
    path: str = CHAT_COMPLETIONS,
    headers: Optional[dict[str, str]] = None,
) -> tuple[dict[str, str], bytes, str]:
    """Put one request through a plugin's on_request hook.

    Args:
        plugin: The plugin handling the request.
        body: The raw request body.
        path: The request path, a completions path by default.
        headers: The request headers, none by default.

    Returns:
        The (headers, body, path) triple the plugin returned.
    """
    return await plugin.on_request(
        _REQUEST_ID, {} if headers is None else headers, body, path
    )


# ── Responses API helpers ──


RESPONSES = "/v1/responses"


def responses_event(payload: dict) -> bytes:
    """Create one Responses SSE event with an event line."""
    return (
        b"event: " + payload["type"].encode() + b"\n"
        + _SSE_DATA_PREFIX + json.dumps(payload).encode() + b"\n\n"
    )


def responses_request(
    input_value: list | str,
    stream: Optional[bool] = True,
    instructions: Optional[str] = None,
) -> bytes:
    """Build a Responses request body.

    Args:
        input_value: The input items, or one string.
        stream: What the client asks for in "stream", or None to leave it out.
        instructions: The instructions field, left out when None.

    Returns:
        The encoded request body.
    """
    body: dict = {"model": "test", "input": input_value}
    if instructions is not None:
        body["instructions"] = instructions
    if stream is not None:
        body["stream"] = stream
    return json.dumps(body).encode()


def responses_input(messages: list[dict]) -> list[dict]:
    """Write a chat history as the Responses input items a client sends for it.

    Args:
        messages: Chat messages with string content.

    Returns:
        The input items.
    """
    items: list[dict] = []
    for message in messages:
        role = message["role"]
        if role == "tool":
            items.append({
                "type": "function_call_output",
                "call_id": message.get("tool_call_id", ""),
                "output": message["content"],
            })
        elif role != "assistant":
            items.append({"type": "message", "role": role, "content": message["content"]})
        else:
            items.extend(_responses_assistant_items(message))
    return items


def _responses_assistant_items(message: dict) -> list[dict]:
    """Write one assistant chat message as Responses items."""
    items: list[dict] = []
    if message.get("reasoning_content"):
        items.append({
            "type": "reasoning",
            "summary": [],
            "content": [{"type": "reasoning_text", "text": message["reasoning_content"]}],
        })
    if message.get("content"):
        items.append({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": message["content"]}],
        })
    for tool_call in message.get("tool_calls") or []:
        items.append({
            "type": "function_call",
            "call_id": tool_call["id"],
            "name": tool_call["function"]["name"],
            "arguments": tool_call["function"].get("arguments", ""),
        })
    return items


class ResponsesStream:
    """Builds the events of one Responses stream attempt, in vLLM's order.

    The terminal event lists every item the stream built.
    """

    def __init__(
        self,
        response_id: str = "resp_1",
        terminal_call_ids: Optional[dict[str, str]] = None,
    ) -> None:
        self._response_id = response_id
        self._terminal_call_ids = terminal_call_ids if terminal_call_ids is not None else {}
        self._sequence = 0
        self._output_index = 0
        self.output: list[dict] = []

    def _event(self, payload: dict) -> bytes:
        """Number one event and encode it."""
        payload["sequence_number"] = self._sequence
        self._sequence += 1
        return responses_event(payload)

    def _item_event(self, payload: dict) -> bytes:
        """Encode one event of the current item."""
        payload["output_index"] = self._output_index
        return self._event(payload)

    def _response(self, status: str) -> dict:
        """The response object with the given status."""
        return {"id": self._response_id, "object": "response", "status": status, "output": []}

    def start(self) -> bytes:
        """The events that open the response."""
        return (
            self._event({"type": "response.created", "response": self._response("in_progress")})
            + self._event({"type": "response.in_progress", "response": self._response("in_progress")})
        )

    def _text_item(self, item: dict, delta_type: str, part: dict, index_field: str, text: str) -> bytes:
        """The events that open an item and stream its text."""
        item_id = item["id"]
        events = self._item_event({"type": "response.output_item.added", "item": dict(item)})
        part_type = "response.reasoning_part" if part["type"] == "reasoning_text" else "response.content_part"
        events += self._item_event({
            "type": part_type + ".added", "item_id": item_id, index_field: 0, "part": dict(part),
        })
        for start in range(0, len(text), _DELTA_CHARS):
            events += self._item_event({
                "type": delta_type + ".delta", "item_id": item_id, index_field: 0,
                "delta": text[start:start + _DELTA_CHARS],
            })
        events += self._item_event({"type": delta_type + ".done", "item_id": item_id, index_field: 0, "text": text})
        events += self._item_event({
            "type": part_type + ".done", "item_id": item_id, index_field: 0, "part": {**part, "text": text},
        })
        return events

    def _finish_item(self, item: dict) -> bytes:
        """The event that ends an item, recording the item in the output."""
        events = self._item_event({"type": "response.output_item.done", "item": item})
        self.output.append(item)
        self._output_index += 1
        return events

    def reasoning(self, text: str, item_id: str = "") -> bytes:
        """A reasoning item streaming the given text."""
        item = {"id": item_id or f"rs_{self._output_index}", "type": "reasoning", "summary": [], "content": None}
        events = self._text_item(
            item, "response.reasoning_text", {"type": "reasoning_text", "text": ""}, "content_index", text,
        )
        done = {**item, "content": [{"type": "reasoning_text", "text": text}], "status": "completed"}
        return events + self._finish_item(done)

    def content(self, text: str, item_id: str = "") -> bytes:
        """A message item streaming the given text."""
        item = {
            "id": item_id or f"msg_{self._output_index}", "type": "message", "role": "assistant",
            "status": "in_progress", "content": [],
        }
        events = self._text_item(
            item, "response.output_text", {"type": "output_text", "text": "", "annotations": []},
            "content_index", text,
        )
        done = {
            **item, "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }
        return events + self._finish_item(done)

    def tool_call(self, call_id: str, name: str, arguments: dict | str | None = None) -> bytes:
        """A function call item streaming its arguments."""
        args = {} if arguments is None else arguments
        text = json.dumps(args) if isinstance(args, dict) else args
        return self.tool_call_opening(call_id, name) + self._tool_call_rest(call_id, name, text)

    def tool_call_opening(self, call_id: str, name: str) -> bytes:
        """The event that opens a function call item."""
        item = {
            "id": f"fc_{call_id}", "type": "function_call", "call_id": call_id, "name": name,
            "arguments": "", "status": "in_progress",
        }
        return self._item_event({"type": "response.output_item.added", "item": item})

    def _tool_call_rest(self, call_id: str, name: str, text: str) -> bytes:
        """The argument and closing events of a function call item."""
        item_id = f"fc_{call_id}"
        events = b""
        for start in range(0, len(text), _DELTA_CHARS):
            events += self._item_event({
                "type": "response.function_call_arguments.delta", "item_id": item_id,
                "delta": text[start:start + _DELTA_CHARS],
            })
        events += self._item_event({
            "type": "response.function_call_arguments.done", "item_id": item_id, "name": name, "arguments": text,
        })
        done = {
            "id": item_id, "type": "function_call", "call_id": self._terminal_call_ids.get(call_id, call_id),
            "name": name, "arguments": text, "status": "completed",
        }
        return events + self._finish_item(done)

    def completed(
        self,
        status: str = "completed",
        usage: Optional[dict[str, int]] = None,
        incomplete_reason: str = "",
    ) -> bytes:
        """The terminal event, listing every item this stream built."""
        response = self._response(status)
        response["output"] = list(self.output)
        response["usage"] = usage
        if incomplete_reason:
            response["incomplete_details"] = {"reason": incomplete_reason}
        return self._event({"type": "response.completed", "response": response})

    def failed(self) -> bytes:
        """The event a backend sends when generation fails."""
        response = self._response("failed")
        response["error"] = {"code": "server_error", "message": "boom"}
        return self._event({"type": "response.failed", "response": response})


def _responses_usage(input_tokens: int, output_tokens: int, total_tokens: int = 0) -> dict[str, int]:
    """A Responses usage object."""
    return {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens}


def parse_sse_payloads(written: bytes) -> list[dict]:
    """Parse every JSON data line in written SSE bytes, skipping the rest."""
    payloads: list[dict] = []
    for raw_line in written.split(b"\n"):
        line = raw_line.strip()
        if not line.startswith(_SSE_DATA_PREFIX):
            continue
        try:
            payloads.append(json.loads(line[len(_SSE_DATA_PREFIX):]))
        except json.JSONDecodeError:
            continue
    return payloads


def _extract_responses_text(written: bytes, delta_type: str = "response.output_text.delta") -> str:
    """Join the deltas of one event type in written Responses SSE bytes."""
    return "".join(
        payload.get("delta", "")
        for payload in parse_sse_payloads(written)
        if payload.get("type") == delta_type
    )


# ── API wire formats ──


class Attempt(abc.ABC):
    """Builds the upstream reply of one attempt in one API's wire format."""

    def start(self) -> bytes:
        """Bytes that open the reply."""
        return b""

    @abc.abstractmethod
    def content(self, text: str) -> bytes:
        """Content text."""

    @abc.abstractmethod
    def reasoning(self, text: str) -> bytes:
        """Reasoning text."""

    @abc.abstractmethod
    def tool_call(self, tc_id: str, name: str, arguments: dict | str | None = None) -> bytes:
        """One whole tool call."""

    @abc.abstractmethod
    def finish(self, reason: Optional[str], usage: Optional[tuple[int, int, int]] = None) -> bytes:
        """Bytes that end the reply, reporting the reason, or None to end without one."""

    @abc.abstractmethod
    def failure(self) -> bytes:
        """Bytes a backend sends when generation fails."""

    def pieces(self, pieces: tuple | list) -> bytes:
        """Bytes for a list of content, reasoning and tool call pieces.

        Args:
            pieces: Tuples whose first item is content, reasoning or tool_call, followed by
                that piece's values.

        Returns:
            The reply bytes.
        """
        reply = b""
        for piece in pieces:
            if piece[0] == "content":
                reply += self.content(piece[1])
            elif piece[0] == "reasoning":
                reply += self.reasoning(piece[1])
            else:
                reply += self.tool_call(*piece[1:])
        return reply


class _ChatAttempt(Attempt):
    def __init__(self, chat_id: str) -> None:
        self._chat_id = chat_id

    @override
    def content(self, text: str) -> bytes:
        return sse_content(self._chat_id, text)

    @override
    def reasoning(self, text: str) -> bytes:
        return sse_reasoning(self._chat_id, text)

    @override
    def tool_call(self, tc_id: str, name: str, arguments: dict | str | None = None) -> bytes:
        if isinstance(arguments, str):
            return sse_tool_call_null_fields(tc_id, name, arguments, self._chat_id)
        return sse_tool_call(tc_id, name, arguments, self._chat_id)

    @override
    def finish(self, reason: Optional[str], usage: Optional[tuple[int, int, int]] = None) -> bytes:
        events = sse_finish_reason(self._chat_id, reason) if reason else b""
        if usage is not None:
            events += sse_usage(*usage)
        return events + sse_done()

    @override
    def failure(self) -> bytes:
        return sse_event({"error": {"message": "boom", "code": 500}}) + sse_done()


class _ResponsesAttempt(Attempt):
    def __init__(self, response_id: str, terminal_call_ids: Optional[dict[str, str]]) -> None:
        self.stream = ResponsesStream(response_id, terminal_call_ids=terminal_call_ids)

    @override
    def start(self) -> bytes:
        return self.stream.start()

    @override
    def content(self, text: str) -> bytes:
        return self.stream.content(text)

    @override
    def reasoning(self, text: str) -> bytes:
        return self.stream.reasoning(text)

    @override
    def tool_call(self, tc_id: str, name: str, arguments: dict | str | None = None) -> bytes:
        return self.stream.tool_call(tc_id, name, arguments)

    @override
    def finish(self, reason: Optional[str], usage: Optional[tuple[int, int, int]] = None) -> bytes:
        if reason is None:
            return b""
        status = "incomplete" if reason == "length" else "completed"
        return self.stream.completed(status, _responses_usage(*usage) if usage else None)

    @override
    def failure(self) -> bytes:
        return self.stream.failed()


class Wire(abc.ABC):
    """One API's wire format.

    Attributes:
        name: The API name used as the test id.
        path: The request path of the API.
    """

    def __init__(self, name: str, path: str) -> None:
        self.name = name
        self.path = path

    @abc.abstractmethod
    def request(self, messages: list[dict], stream: Optional[bool] = True) -> bytes:
        """A request carrying the given chat history."""

    @abc.abstractmethod
    def attempt(self, attempt_id: str = "1", terminal_call_ids: Optional[dict[str, str]] = None) -> Attempt:
        """A builder for one upstream attempt.

        Args:
            attempt_id: The id the attempt's reply carries.
            terminal_call_ids: The call id the final reply gives each streamed call id.
        """

    def reply(self, attempt_id: str, *pieces: tuple, finish: Optional[str] = "stop") -> bytes:
        """A whole upstream reply made of the given pieces."""
        builder = self.attempt(attempt_id)
        return builder.start() + builder.pieces(pieces) + builder.finish(finish)

    @abc.abstractmethod
    def content_of(self, written: bytes) -> str:
        """The content text a client received."""

    @abc.abstractmethod
    def history(self, body: bytes) -> list[dict]:
        """The history a forwarded request carries, as chat style messages."""

    def system_texts(self, body: bytes) -> list[str]:
        """The system message texts a forwarded request carries."""
        return [m["content"] for m in self.history(body) if m.get("role") == "system"]

    def assistant_turns(self, body: bytes) -> list[dict]:
        """The assistant turns a forwarded request carries, as chat style messages."""
        return [m for m in self.history(body) if m.get("role") == "assistant"]


class _ChatWire(Wire):
    def __init__(self) -> None:
        super().__init__("chat", CHAT_COMPLETIONS)

    @override
    def request(self, messages: list[dict], stream: Optional[bool] = True) -> bytes:
        body: dict = {"model": "test", "messages": messages}
        if stream is not None:
            body["stream"] = stream
        return json.dumps(body).encode()

    @override
    def attempt(self, attempt_id: str = "1", terminal_call_ids: Optional[dict[str, str]] = None) -> Attempt:
        if terminal_call_ids:
            raise ValueError("a chat reply has no final call ids")
        return _ChatAttempt(attempt_id)

    @override
    def content_of(self, written: bytes) -> str:
        return extract_sse_field(written)

    @override
    def history(self, body: bytes) -> list[dict]:
        return json.loads(body)["messages"]


def _fold_responses_input(items: list[dict]) -> list[dict]:
    """Read Responses input items back as chat style messages, one per assistant run."""
    messages: list[dict] = []
    for item in items:
        item_type = item.get("type")
        if item_type in ("reasoning", "function_call") or item.get("role") == "assistant":
            if not messages or messages[-1].get("role") != "assistant":
                messages.append({"role": "assistant"})
            turn = messages[-1]
            if item_type == "reasoning":
                turn["reasoning_content"] = item["content"][0]["text"] if item.get("content") else ""
            elif item_type == "function_call":
                turn.setdefault("tool_calls", []).append({
                    "id": item["call_id"], "type": "function",
                    "function": {"name": item["name"], "arguments": item["arguments"]},
                })
            else:
                content = item["content"]
                turn["content"] = content if isinstance(content, str) else content[0]["text"]
        elif item_type == "function_call_output":
            messages.append({"role": "tool", "tool_call_id": item["call_id"], "content": item["output"]})
        else:
            messages.append({"role": item.get("role"), "content": item.get("content")})
    return messages


class _ResponsesWire(Wire):
    def __init__(self) -> None:
        super().__init__("responses", RESPONSES)

    @override
    def request(self, messages: list[dict], stream: Optional[bool] = True) -> bytes:
        return responses_request(responses_input(messages), stream)

    @override
    def attempt(self, attempt_id: str = "1", terminal_call_ids: Optional[dict[str, str]] = None) -> Attempt:
        return _ResponsesAttempt(f"resp_{attempt_id}", terminal_call_ids)

    @override
    def content_of(self, written: bytes) -> str:
        return _extract_responses_text(written)

    @override
    def history(self, body: bytes) -> list[dict]:
        return _fold_responses_input(json.loads(body)["input"])


CHAT_WIRE = _ChatWire()
RESPONSES_WIRE = _ResponsesWire()


@pytest.fixture(params=[CHAT_WIRE, RESPONSES_WIRE], ids=lambda wire: wire.name)
def wire(request: pytest.FixtureRequest) -> Wire:
    """Run the requesting test once per API wire format."""
    return request.param


# ── Test-only plugins ──


async def _respond(
    response: AbstractMutatingResponse,
    web_request: web.Request,
    body: bytes = b"",
    content_type: str = "",
) -> None:
    """Send a complete 200 response and close the stream."""
    response.set_status(200, reason="OK")
    if content_type:
        response.headers["Content-Type"] = content_type
    await response.prepare(web_request)
    if body:
        await response.write(body)
    await response.write_eof()


class _FakePlugin(PluginBase):
    """Base for the fake plugins below: named after its class, serving the paths it declares.

    The plugin manager registers a type under its class name, so a scenario that
    composes several plugins needs a distinct class per participant even when two
    of them behave identically. Shared behaviour lives on the private bases here.
    """

    paths: ClassVar[tuple[str, ...]] = ()

    @override
    def name(self) -> str:
        return self.__class__.__name__

    @override
    def overridden_paths(self) -> list[str]:
        return list(self.paths)


class _SuffixPlugin(_FakePlugin):
    """Base for fakes that append their configured suffix to what they are given."""

    def __init__(
        self, config: dict, service_config: ServiceConfig | None = None
    ) -> None:
        super().__init__(config, service_config)
        self._suffix: bytes = config.get("suffix", "").encode()


class _DelegatingPlugin(_FakePlugin):
    """Base for fakes that hand their overridden path straight back to the proxy."""

    @override
    async def handle_overridden_request(
        self,
        _request_id: int,
        headers: dict[str, str],
        body: bytes,
        path: str,
        web_request: web.Request,
        forward: _ForwardFn,
        response: AbstractMutatingResponse,
    ) -> bool:
        return await forward(headers, body, path, web_request, response)


class _ChunkCall(NamedTuple):
    """One on_response_chunk call, as the recording fakes below keep it."""

    chunk: bytes
    status_code: int
    is_final: bool


class _ChunkRecorder(_FakePlugin):
    """Base for fakes that record the chunk calls they saw and pass the chunk through."""

    def __init__(
        self, config: dict, service_config: ServiceConfig | None = None
    ) -> None:
        super().__init__(config, service_config)
        self.calls: list[_ChunkCall] = []

    @override
    async def on_response_chunk(
        self,
        _request_id: int,
        _headers: dict[str, str],
        chunk: bytes,
        _path: str,
        status_code: int,
        is_final: bool,
    ) -> bytes:
        self.calls.append(_ChunkCall(chunk, status_code, is_final))
        return chunk


class _ChunkCrashPlugin(_FakePlugin):
    """Base for fakes whose chunk hook raises, to prove the manager survives it."""

    error: ClassVar[type[Exception]] = RuntimeError
    message: ClassVar[str] = "crash"

    @override
    async def on_response_chunk(
        self,
        _request_id: int,
        _headers: dict[str, str],
        _chunk: bytes,
        _path: str,
        _status_code: int,
        _is_final: bool,
    ) -> bytes:
        raise self.error(self.message)


class OverridePlugin(_FakePlugin):
    """Claims a path of its own plus one the proxy already serves."""

    paths: ClassVar[tuple[str, ...]] = ("/custom", "/healthz")


class ScopedPlugin(_FakePlugin):
    """Claims a path, to be enabled on only some services."""

    paths: ClassVar[tuple[str, ...]] = ("/admin",)


class PathConflictPlugin(_FakePlugin):
    """Claims whatever path its config names, so two instances can collide."""

    def __init__(
        self, config: dict, service_config: ServiceConfig | None = None
    ) -> None:
        super().__init__(config, service_config)
        self._path: str = config.get("path", "/shared")

    @override
    def overridden_paths(self) -> list[str]:
        return [self._path]


class CustomResponsePlugin(_FakePlugin):
    """Answers its overridden path itself, without reaching the backend."""

    paths: ClassVar[tuple[str, ...]] = ("/custom",)

    @override
    async def handle_overridden_request(
        self,
        _request_id: int,
        _headers: dict[str, str],
        _body: bytes,
        _path: str,
        web_request: web.Request,
        _forward: _ForwardFn,
        response: AbstractMutatingResponse,
    ) -> bool:
        await _respond(response, web_request, b"custom response", "text/plain")
        return True


class CacheForwardPlugin(_FakePlugin):
    """Keeps the forward callable it was handed, so a test can call it after the handler returns."""

    cached_forward: ClassVar[Optional[_ForwardFn]] = None

    paths: ClassVar[tuple[str, ...]] = ("/cache-forward",)

    @override
    async def handle_overridden_request(
        self,
        _request_id: int,
        _headers: dict[str, str],
        _body: bytes,
        _path: str,
        web_request: web.Request,
        forward: _ForwardFn,
        response: AbstractMutatingResponse,
    ) -> bool:
        CacheForwardPlugin.cached_forward = forward
        await _respond(response, web_request)
        return True


class DelegatePlugin(_DelegatingPlugin):
    """Delegates a plugin-only path to the backend."""

    paths: ClassVar[tuple[str, ...]] = ("/delegate",)


class ForwardToBackupPlugin(_DelegatingPlugin):
    """Delegates a path whose service falls back to its backup target."""

    paths: ClassVar[tuple[str, ...]] = ("/forward-to-backup",)


class GpuDelegatePlugin(_DelegatingPlugin):
    """Delegates a GPU-routed path, so the delegation runs under VRAM admission."""

    paths: ClassVar[tuple[str, ...]] = ("/v1/completions",)


class UIPublisherPlugin(_FakePlugin):
    """Publishes the dashboard panel named in its config."""

    def __init__(
        self, config: dict, service_config: ServiceConfig | None = None
    ) -> None:
        super().__init__(config, service_config)
        self._title: str = config.get("title", "Untitled")
        self._content: str = config.get("content", "")

    @override
    def init_ui(self, publish: Callable[[str, str], None]) -> None:
        publish(self._title, self._content)


class BodyAppenderPlugin(_SuffixPlugin):
    """Appends its configured suffix to the request body."""

    @override
    async def on_request(
        self,
        _request_id: int,
        headers: dict[str, str],
        body: bytes,
        path: str,
    ) -> tuple[dict[str, str], bytes, str]:
        return headers, body + self._suffix, path


class HeaderPlugin(_FakePlugin):
    """Marks every request it sees with a header the backend can echo back."""

    @override
    async def on_request(
        self,
        _request_id: int,
        headers: dict[str, str],
        body: bytes,
        path: str,
    ) -> tuple[dict[str, str], bytes, str]:
        headers["X-Modified"] = "true"
        return headers, body, path


class ContentLengthPlugin(_FakePlugin):
    """Grows the JSON request body and keeps Content-Length in step with it."""

    @override
    async def on_request(
        self,
        _request_id: int,
        headers: dict[str, str],
        body: bytes,
        path: str,
    ) -> tuple[dict[str, str], bytes, str]:
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return headers, body, path
        data["injected"] = True
        new_body = json.dumps(data).encode()
        headers["Content-Length"] = str(len(new_body))
        return headers, new_body, path


class FailingPlugin(_FakePlugin):
    """Raises from on_request, to prove the request pipeline survives it."""

    @override
    async def on_request(
        self,
        _request_id: int,
        _headers: dict[str, str],
        _body: bytes,
        _path: str,
    ) -> tuple[dict[str, str], bytes, str]:
        raise RuntimeError("plugin error")


class MutatorPlugin(_FakePlugin):
    """Prepends an extra SSE event to every chunk."""

    @override
    async def on_response_chunk(
        self,
        _request_id: int,
        _headers: dict[str, str],
        chunk: bytes,
        _path: str,
        _status_code: int,
        _is_final: bool,
    ) -> bytes:
        return sse_event({"choices": [{"delta": {"content": "MUTATED"}}]}) + chunk


class ChunkReplacer(_FakePlugin):
    """Discards the upstream chunk and substitutes its own bytes."""

    @override
    async def on_response_chunk(
        self,
        _request_id: int,
        _headers: dict[str, str],
        _chunk: bytes,
        _path: str,
        _status_code: int,
        _is_final: bool,
    ) -> bytes:
        return b"replaced"


class ChunkAppender(_SuffixPlugin):
    """Appends its configured suffix to every chunk."""

    @override
    async def on_response_chunk(
        self,
        _request_id: int,
        _headers: dict[str, str],
        chunk: bytes,
        _path: str,
        _status_code: int,
        _is_final: bool,
    ) -> bytes:
        return chunk + self._suffix


class ChunkModifier(ChunkAppender):
    """Appends its configured suffix and remembers the chunk it was handed."""

    def __init__(
        self, config: dict, service_config: ServiceConfig | None = None
    ) -> None:
        super().__init__(config, service_config)
        self.received: bytes = b""

    @override
    async def on_response_chunk(
        self,
        request_id: int,
        headers: dict[str, str],
        chunk: bytes,
        path: str,
        status_code: int,
        is_final: bool,
    ) -> bytes:
        self.received = chunk
        return await super().on_response_chunk(
            request_id, headers, chunk, path, status_code, is_final
        )


class ChunkRecordingPlugin(_ChunkRecorder):
    """Reports the chunk, status code and final flag it was last called with."""

    @property
    def received(self) -> _ChunkCall:
        """Return the last call, which compares equal to the expected triple."""
        return self.calls[-1]


class ChunkEchoPlugin(_ChunkRecorder):
    """Reports the status code and final flag it was last called with."""

    @property
    def status(self) -> int:
        """Return the status code of the last call."""
        return self.calls[-1].status_code

    @property
    def final(self) -> bool:
        """Return the final flag of the last call."""
        return self.calls[-1].is_final


class GoodChunkPlugin(_ChunkRecorder):
    """Reports whether it ran, to prove a failing neighbour did not stop the chain."""

    @property
    def called(self) -> bool:
        """Return whether the chunk hook ran at all."""
        return bool(self.calls)


class ChunkCrasher(_ChunkCrashPlugin):
    """Raises mid-chain, between two plugins that modify the chunk."""


class BadChunkPlugin(_ChunkCrashPlugin):
    """Raises a different exception type, between two plugins that pass the chunk through."""

    error: ClassVar[type[Exception]] = ValueError
    message: ClassVar[str] = "boom"


class ChunkCapturePlugin(_FakePlugin):
    """Appends every chunk call to the JSON file its config names.

    The proxy owns the plugin instance, so the file is how a test reads the calls back.
    """

    def __init__(
        self, config: dict, service_config: ServiceConfig | None = None
    ) -> None:
        super().__init__(config, service_config)
        self._call_file: str = config["call_file"]

    @override
    async def on_response_chunk(
        self,
        _request_id: int,
        _headers: dict[str, str],
        chunk: bytes,
        path: str,
        status_code: int,
        is_final: bool,
    ) -> bytes:
        entry = {
            "chunk": chunk.decode("latin-1"),
            "path": path,
            "status_code": status_code,
            "is_final": is_final,
        }
        try:
            with open(self._call_file, "r") as f:
                calls = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            calls = []
        calls.append(entry)
        with open(self._call_file, "w") as f:
            json.dump(calls, f)
        return chunk


class CancelledForwardPlugin(_FakePlugin):
    """Cancels its own forward once upstream has the request.

    The guard rail does exactly this when it decides to retry: it runs
    forward() as a task and cancels it in a finally. Cancellation is a
    BaseException, so the proxy's forwarding callbacks are reached with no
    error message at all.
    """

    paths: ClassVar[tuple[str, ...]] = ("/cancel-forward",)

    @override
    async def handle_overridden_request(
        self,
        _request_id: int,
        headers: dict[str, str],
        body: bytes,
        path: str,
        web_request: web.Request,
        forward: _ForwardFn,
        response: AbstractMutatingResponse,
    ) -> bool:
        forwarding = asyncio.create_task(
            forward(headers, body, path, web_request, response)
        )
        await asyncio.sleep(_CANCEL_FORWARD_DELAY_S)
        forwarding.cancel()
        try:
            await forwarding
        except asyncio.CancelledError:
            pass
        return False


class CancelTrackingPlugin(_FakePlugin):
    """Records the requests it was told were cancelled."""

    def __init__(
        self, config: dict, service_config: ServiceConfig | None = None
    ) -> None:
        super().__init__(config, service_config)
        self.cancelled_requests: list[int] = []

    @override
    def on_cancelled(self, request_id: int) -> None:
        # Named rather than super(), so the base that stops tracking the request
        # stays visible if a fake base above ever grows an on_cancelled of its own.
        PluginBase.on_cancelled(self, request_id)
        self.cancelled_requests.append(request_id)


class CrashingCancelPlugin(CancelTrackingPlugin):
    """Raises from on_cancelled, to prove a later plugin is still told."""

    @override
    def on_cancelled(self, _request_id: int) -> None:
        raise RuntimeError("crash in on_cancelled")


class RequestIdRecorder(_FakePlugin):
    """Remembers the request id its request hook was called with."""

    def __init__(
        self, config: dict, service_config: ServiceConfig | None = None
    ) -> None:
        super().__init__(config, service_config)
        self.received_request_id = 0

    @override
    async def on_request(
        self,
        request_id: int,
        headers: dict[str, str],
        body: bytes,
        path: str,
    ) -> tuple[dict[str, str], bytes, str]:
        self.received_request_id = request_id
        return headers, body, path


class ChunkRequestIdRecorder(_FakePlugin):
    """Remembers the request id its chunk hook was called with."""

    def __init__(
        self, config: dict, service_config: ServiceConfig | None = None
    ) -> None:
        super().__init__(config, service_config)
        self.received_request_id = 0

    @override
    async def on_response_chunk(
        self,
        request_id: int,
        _headers: dict[str, str],
        chunk: bytes,
        _path: str,
        _status_code: int,
        _is_final: bool,
    ) -> bytes:
        self.received_request_id = request_id
        return chunk


class RequestIdCollector(_FakePlugin):
    """Collects every request id its request hook sees, in order."""

    def __init__(
        self, config: dict, service_config: ServiceConfig | None = None
    ) -> None:
        super().__init__(config, service_config)
        self.request_ids: list[int] = []

    @override
    async def on_request(
        self,
        request_id: int,
        headers: dict[str, str],
        body: bytes,
        path: str,
    ) -> tuple[dict[str, str], bytes, str]:
        self.request_ids.append(request_id)
        return headers, body, path


class PrepareTargetRecorder(_FakePlugin):
    """Records every prepare_target call it gets; optionally raises instead."""

    def __init__(
        self, config: dict, service_config: ServiceConfig | None = None
    ) -> None:
        super().__init__(config, service_config)
        self.prepared: list[tuple[int, str]] = []
        self._fail: bool = bool(config.get("fail", False))

    @override
    async def prepare_target(self, request_id: int, path: str) -> None:
        self.prepared.append((request_id, path))
        if self._fail:
            raise RuntimeError("prepare_target failed on purpose")


class AsyncSleepPlugin(_FakePlugin):
    """Passes everything through, after awaiting a yield so the hook's async-ness is real."""

    @override
    async def on_request(
        self,
        _request_id: int,
        headers: dict[str, str],
        body: bytes,
        path: str,
    ) -> tuple[dict[str, str], bytes, str]:
        await asyncio.sleep(0)
        return headers, body, path

    @override
    async def on_response_chunk(
        self,
        _request_id: int,
        _headers: dict[str, str],
        chunk: bytes,
        _path: str,
        _status_code: int,
        _is_final: bool,
    ) -> bytes:
        await asyncio.sleep(0)
        return chunk


class EchoPlugin(_FakePlugin):
    """Passes everything through untouched."""


class NoUIPlugin(_FakePlugin):
    """Passes everything through untouched and publishes no dashboard panel."""


# ── NVIDIA driver, Docker and /proc ──

BYTES_PER_MIB = 1024 * 1024
_MILLIWATTS_PER_WATT = 1000
_FAKE_GPU_HANDLE = "fake-gpu-0"
_HELD_READ_TIMEOUT_S = 5.0
_SHUTDOWN_WAIT_S = 0.5
_PROC_FILE_PATTERN = re.compile(r"/proc/(\d+)/(cgroup|cmdline)")


@dataclass
class FakeGpuProcess:
    """A process the fake driver lists on the GPU.

    A vram_bytes of None is a process whose memory the driver will not tell.
    A process with no container lives on the host. A process that has exited
    is still listed by the driver but is gone from /proc, and reading its files
    raises exit_error. A process that exits after its cgroup is read is gone
    by the time its command line is read. A hidden process has /proc files the
    host will not let anyone read.
    """

    pid: int
    vram_bytes: Optional[int]
    container: Optional[str] = None
    command: str = ""
    sm_utilization_pct: int = 0
    is_compute: bool = True
    is_graphics: bool = False
    cgroup: Optional[str] = None
    has_exited: bool = False
    exits_after_cgroup: bool = False
    exit_error: type[OSError] = FileNotFoundError
    is_hidden: bool = False

    @classmethod
    def from_mib(
        cls,
        pid: int,
        vram_mib: int,
        container: Optional[str] = None,
        command: str = "",
        sm_utilization_pct: int = 0,
    ) -> Self:
        """A process holding a whole number of MiB."""
        return cls(
            pid=pid,
            vram_bytes=vram_mib * BYTES_PER_MIB,
            container=container,
            command=command,
            sm_utilization_pct=sm_utilization_pct,
        )


class FakeNvidiaHost:
    """Scriptable stand-in for the NVIDIA driver, Docker and /proc of a GPU host.

    A test sets the readings and the GPU processes it wants, then installs the
    fake for the span of a with block. GPU memory is set here in MiB, with
    extra bytes for part MiB amounts, and power is set in watts. Counters
    record how often the driver was started, read, and shut down.
    """

    def __init__(
        self,
        total_vram_mib: int = _TOTAL_VRAM_MIB,
        used_vram_mib: int = 0,
        processes: Optional[list[FakeGpuProcess]] = None,
    ) -> None:
        self.total_vram_mib = total_vram_mib
        self.used_vram_mib = used_vram_mib
        self.reserved_vram_mib = 0
        self.extra_total_bytes = 0
        self.extra_used_bytes = 0
        self.processes: list[FakeGpuProcess] = [] if processes is None else processes
        self.stopped_containers: set[str] = set()
        self.gpu_util_pct = 0
        self.power_draw_w = 0.0
        self.power_limit_w = 0.0
        self.temperature_c = 0
        self.gr_clock_mhz = 0
        self.mem_clock_mhz = 0
        self.pstate = 0
        self.unsupported_queries: set[str] = set()
        self.init_error: Optional[pynvml.NVMLError] = None
        self.handle_error: Optional[pynvml.NVMLError] = None
        self.shutdown_error: Optional[pynvml.NVMLError] = None
        self.utilization_error: Optional[pynvml.NVMLError] = None
        self.docker_error: Optional[Exception] = None
        self.poll_fails = False
        self.inits = 0
        self.polls = 0
        self.closes = 0
        self.has_shut_down_during_read = False
        self._listed: dict[int, FakeGpuProcess] = {}
        self._read_gate: Optional[threading.Event] = None
        self._read_held = threading.Event()
        self._shutdown_called = threading.Event()

    @staticmethod
    def container_id(name: str) -> str:
        """The full Docker id the fake gives a container name."""
        return hashlib.sha256(name.encode()).hexdigest()

    def current_used_vram_mib(self) -> int:
        """Memory the driver reports in use, outside the reserved part."""
        return self.used_vram_mib

    def current_processes(self) -> list[FakeGpuProcess]:
        """Processes the driver lists on the GPU."""
        return self.processes

    def running_containers(self) -> list[str]:
        """Names of the containers Docker lists as running."""
        names = [p.container for p in self.current_processes() if p.container]
        return [n for n in dict.fromkeys(names) if n not in self.stopped_containers]

    def hold_reads(self) -> None:
        """Make each memory read wait until release_reads is called."""
        self._read_gate = threading.Event()

    def wait_for_held_read(self) -> None:
        """Block until a memory read is waiting to be released.

        Raises:
            TimeoutError: If no read arrives.
        """
        if not self._read_held.wait(_HELD_READ_TIMEOUT_S):
            raise TimeoutError("no GPU read arrived")

    def wait_for_shutdown(self) -> None:
        """Wait a short time for the driver to be shut down."""
        self._shutdown_called.wait(_SHUTDOWN_WAIT_S)

    def release_reads(self) -> None:
        """Let held memory reads finish, and stop holding new ones."""
        if self._read_gate is not None:
            self._read_gate.set()

    @contextmanager
    def installed(self) -> Iterator[None]:
        """Answer every driver, Docker and /proc call for the body of the block."""
        original_read_text = Path.read_text

        def read_text(path: Path, *args: str, **kwargs: str) -> str:
            match = _PROC_FILE_PATTERN.fullmatch(str(path))
            if match is None:
                return original_read_text(path, *args, **kwargs)
            return self._read_proc_file(int(match.group(1)), match.group(2))

        with (
            patch.multiple(
                pynvml,
                nvmlInit=self._init,
                nvmlShutdown=self._shutdown,
                nvmlDeviceGetHandleByIndex=self._get_handle,
                nvmlDeviceGetMemoryInfo=self._get_memory_info,
                nvmlDeviceGetUtilizationRates=self._get_utilization_rates,
                nvmlDeviceGetPowerUsage=self._get_power_usage,
                nvmlDeviceGetPowerManagementLimit=self._get_power_limit,
                nvmlDeviceGetClockInfo=self._get_clock_info,
                nvmlDeviceGetPowerState=self._get_power_state,
                nvmlDeviceGetTemperature=self._get_temperature,
                nvmlDeviceGetComputeRunningProcesses=self._list_compute_processes,
                nvmlDeviceGetGraphicsRunningProcesses=self._list_graphics_processes,
                nvmlDeviceGetProcessUtilization=self._get_process_utilization,
            ),
            patch("aiodocker.Docker", lambda: FakeDockerClient(self)),
            patch.object(Path, "read_text", read_text),
        ):
            yield

    def _check_supported(self, query: str) -> None:
        """Fail the way the driver does for a reading this GPU lacks."""
        if query in self.unsupported_queries:
            raise pynvml.NVMLError(pynvml.NVML_ERROR_NOT_SUPPORTED)

    def _init(self) -> None:
        self.inits += 1
        if self.init_error is not None:
            raise self.init_error

    def _shutdown(self) -> None:
        self.closes += 1
        self._shutdown_called.set()
        if self.shutdown_error is not None:
            raise self.shutdown_error

    def _get_handle(self, _index: int) -> str:
        if self.handle_error is not None:
            raise self.handle_error
        return _FAKE_GPU_HANDLE

    def _get_memory_info(self, _handle: str, version: Optional[int] = None) -> SimpleNamespace:
        """Report memory. The old format counts reserved memory as used."""
        self.polls += 1
        if self._read_gate is not None:
            closes_before_hold = self.closes
            self._read_held.set()
            self._read_gate.wait(_HELD_READ_TIMEOUT_S)
            if self.closes != closes_before_hold:
                self.has_shut_down_during_read = True
        if self.poll_fails:
            raise pynvml.NVMLError(pynvml.NVML_ERROR_UNKNOWN)
        total = self.total_vram_mib * BYTES_PER_MIB + self.extra_total_bytes
        reserved = self.reserved_vram_mib * BYTES_PER_MIB
        in_use = self.current_used_vram_mib() * BYTES_PER_MIB + self.extra_used_bytes
        free = total - reserved - in_use
        if version != pynvml.nvmlMemory_v2:
            in_use += reserved
        return SimpleNamespace(total=total, reserved=reserved, used=in_use, free=free)

    def _get_utilization_rates(self, _handle: str) -> SimpleNamespace:
        self._check_supported("nvmlDeviceGetUtilizationRates")
        return SimpleNamespace(gpu=self.gpu_util_pct, memory=0)

    def _get_power_usage(self, _handle: str) -> int:
        return round(self.power_draw_w * _MILLIWATTS_PER_WATT)

    def _get_power_limit(self, _handle: str) -> int:
        self._check_supported("nvmlDeviceGetPowerManagementLimit")
        return round(self.power_limit_w * _MILLIWATTS_PER_WATT)

    def _get_clock_info(self, _handle: str, clock: int) -> int:
        self._check_supported("nvmlDeviceGetClockInfo")
        if clock == pynvml.NVML_CLOCK_GRAPHICS:
            return self.gr_clock_mhz
        return self.mem_clock_mhz

    def _get_power_state(self, _handle: str) -> int:
        self._check_supported("nvmlDeviceGetPowerState")
        return self.pstate

    def _get_temperature(self, _handle: str, _sensor: int) -> int:
        self._check_supported("nvmlDeviceGetTemperature")
        return self.temperature_c

    def _list_processes(self, is_graphics: bool) -> list[SimpleNamespace]:
        """List compute or graphics processes and remember them for /proc."""
        listed = [
            p for p in self.current_processes()
            if (p.is_graphics if is_graphics else p.is_compute)
        ]
        self._listed.update({p.pid: p for p in listed})
        return [SimpleNamespace(pid=p.pid, usedGpuMemory=p.vram_bytes) for p in listed]

    def _list_compute_processes(self, _handle: str) -> list[SimpleNamespace]:
        return self._list_processes(is_graphics=False)

    def _list_graphics_processes(self, _handle: str) -> list[SimpleNamespace]:
        return self._list_processes(is_graphics=True)

    def _get_process_utilization(self, _handle: str, _since: int) -> list[SimpleNamespace]:
        """Report busy samples, failing with not found when there are none."""
        if self.utilization_error is not None:
            raise self.utilization_error
        samples = [
            SimpleNamespace(pid=p.pid, smUtil=p.sm_utilization_pct)
            for p in self.current_processes()
            if p.sm_utilization_pct
        ]
        if not samples:
            raise pynvml.NVMLError(pynvml.NVML_ERROR_NOT_FOUND)
        return samples

    def _read_proc_file(self, pid: int, name: str) -> str:
        """Serve /proc cgroup and cmdline files for processes the driver listed."""
        process = self._listed.get(pid)
        if process is None:
            raise FileNotFoundError(f"/proc/{pid}/{name}")
        if process.has_exited:
            raise process.exit_error(f"/proc/{pid}/{name}")
        if process.is_hidden:
            raise PermissionError(f"/proc/{pid}/{name}")
        if name == "cmdline":
            return "".join(f"{part}\0" for part in process.command.split())
        if process.exits_after_cgroup:
            process.has_exited = True
        if process.cgroup is not None:
            return process.cgroup
        if process.container is None:
            return "0::/user.slice/user-1000.slice/session-1.scope\n"
        return f"0::/docker/{self.container_id(process.container)}\n"


class FakeDockerClient:
    """The part of the Docker client that lists running containers."""

    def __init__(self, host: FakeNvidiaHost) -> None:
        self.containers = self
        self._host = host

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_exc_info: Optional[BaseException]) -> None:
        return None

    async def list(self) -> list[DockerContainer]:
        """List the containers the host has running, or fail if told to."""
        if self._host.docker_error is not None:
            raise self._host.docker_error
        return [
            DockerContainer(self, Id=FakeNvidiaHost.container_id(name), Names=[f"/{name}"])
            for name in self._host.running_containers()
        ]


# ── Fixtures ──


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the option that opts in to the live Responses gate tests."""
    parser.addoption(
        "--responses-gate",
        default="",
        choices=("", "conformance", "e2e"),
        help="run the live Responses gate phase: conformance or e2e",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers", "live: marks tests as live/integration tests (deselect with '-m not live')"
    )


@pytest.fixture
def sample_service_config() -> ServiceConfig:
    """Minimal ServiceConfig for plugin manager tests."""
    return ServiceConfig(
        name="svc1",
        priority=1,
        vram_budget=4096,
        hostname="svc1.local",
        target="http://localhost:8080",
        container="svc1",
    )


@pytest.fixture
def sample_service_llamacpp() -> ServiceConfig:
    """High-priority LLM service config."""
    return ServiceConfig(
        name="llamacpp-server",
        priority=1,
        vram_budget=16384,
        hostname="llamacpp.local",
        target="http://127.0.0.1:49863",
        container="llamacpp-server",
        gpu_routes=["/v1/completions", "/v1/chat/completions"],
        eviction_command="docker restart {container}",
        backup="http://127.0.0.1:49864",
    )


@pytest.fixture
def sample_service_cpai() -> ServiceConfig:
    """Low-priority CPAI service config."""
    return ServiceConfig(
        name="cpai",
        priority=2,
        vram_budget=4096,
        hostname="cpai.local",
        target="http://127.0.0.1:32168",
        container="cpai",
        gpu_routes=["/api/detect"],
        backup="http://127.0.0.1:32169",
        eviction_command="docker restart {container}",
    )


@pytest.fixture
def sample_config(
    sample_service_llamacpp: ServiceConfig,
    sample_service_cpai: ServiceConfig,
) -> ProxyConfig:
    """Valid proxy config with two services."""
    services = {
        "llamacpp-server": sample_service_llamacpp,
        "cpai": sample_service_cpai,
    }
    hostname_map = {
        "llamacpp.local": sample_service_llamacpp,
        "cpai.local": sample_service_cpai,
    }
    return ProxyConfig(
        listen_port=8080,
        total_vram_budget=20480,
        eviction_timeout=60,
        poll_interval=5,
        eviction_command_timeout=60,
        wait_queue_timeout=60,
        connection_timeout=30,
        dashboard_hostname="gpu-proxy.local",
        services=services,
        hostname_map=hostname_map,
    )


@pytest.fixture
def gpu_status_cold() -> GPUStatus:
    """GPU status with no containers using VRAM."""
    return GPUStatus(
        available_vram_mib=_FREE_VRAM_MIB,
        total_vram_mib=_TOTAL_VRAM_MIB,
        system_vram_mib=_SYSTEM_VRAM_MIB,
        docker_vram_mib=0,
        containers={},
        last_updated=_POLL_TIME,
        poll_failed=False,
        gr_clock_mhz=0,
        mem_clock_mhz=0,
        pstate="",
    )


@pytest.fixture
def gpu_status_llamacpp_hot() -> GPUStatus:
    """GPU status with llamacpp-server hot (using VRAM)."""
    return GPUStatus(
        available_vram_mib=_FREE_VRAM_MIB - _LLAMACPP_VRAM_MIB,
        total_vram_mib=_TOTAL_VRAM_MIB,
        system_vram_mib=_SYSTEM_VRAM_MIB,
        docker_vram_mib=_LLAMACPP_VRAM_MIB,
        containers={
            "llamacpp-server": ContainerVRAM(vram_mib=_LLAMACPP_VRAM_MIB, gpu_util_pct=0),
        },
        last_updated=_POLL_TIME,
        poll_failed=False,
        gr_clock_mhz=0,
        mem_clock_mhz=0,
        pstate="",
    )


@pytest.fixture
def gpu_status_both_hot() -> GPUStatus:
    """GPU status with both services hot."""
    return GPUStatus(
        available_vram_mib=_FREE_VRAM_MIB - _LLAMACPP_VRAM_MIB - _CPAI_VRAM_MIB,
        total_vram_mib=_TOTAL_VRAM_MIB,
        system_vram_mib=_SYSTEM_VRAM_MIB,
        docker_vram_mib=_LLAMACPP_VRAM_MIB + _CPAI_VRAM_MIB,
        containers={
            "llamacpp-server": ContainerVRAM(vram_mib=_LLAMACPP_VRAM_MIB, gpu_util_pct=0),
            "cpai": ContainerVRAM(vram_mib=_CPAI_VRAM_MIB, gpu_util_pct=0),
        },
        last_updated=_POLL_TIME,
        poll_failed=False,
        gr_clock_mhz=0,
        mem_clock_mhz=0,
        pstate="",
    )


@pytest.fixture
def gpu_status_poll_failed() -> GPUStatus:
    """GPU status indicating poll failure."""
    return GPUStatus(
        available_vram_mib=0,
        total_vram_mib=0,
        system_vram_mib=0,
        docker_vram_mib=0,
        containers={},
        last_updated=None,
        poll_failed=True,
        gr_clock_mhz=0,
        mem_clock_mhz=0,
        pstate="",
    )


@pytest.fixture
def service_states_cold(sample_config: ProxyConfig) -> dict[str, ServiceState]:
    """All services in cold state."""
    return {
        name: ServiceState(name, clock=FakeClock())
        for name in sample_config.services
    }


@pytest.fixture
async def wait_queue() -> AsyncIterator[VRAMWaitQueue]:
    """Empty VRAM wait queue with default timeout and proper teardown.

    Uses process_on_enqueue=False so entries stay in queue for unit testing.
    """
    queue = VRAMWaitQueue(clock=FakeClock(), timeout=60, process_on_enqueue=False)
    yield queue
    await queue.cancel_all()


@pytest.fixture
async def token_stats_plugin() -> AsyncIterator[tuple[OpenAITokenStats, PublishedPanels]]:
    """An OpenAITokenStats whose UI callback records every panel it publishes.

    Declared async so the plugin is built with a running event loop, which is
    what lets its constructor start the background TPS task.

    Yields:
        The (plugin, published) pair; published grows as the plugin publishes.
    """
    plugin = OpenAITokenStats({})
    published: PublishedPanels = []

    def publish(title: str, html: str) -> None:
        published.append((title, html))

    plugin.init_ui(publish)
    yield plugin, published


@pytest.fixture(autouse=True)
def mock_subprocess_exec() -> Iterator[None]:
    """Mock asyncio.create_subprocess_exec for all tests.

    Keeps tests from running real commands such as docker restart.

    Returns a mock subprocess that succeeds by default (exit code 0,
    empty stdout/stderr). Tests that need specific subprocess behavior
    (timeout, failure, specific output) should use their own
    `patch("asyncio.create_subprocess_exec", ...)` context manager,
    which will override this fixture for the duration of the `with` block.
    """
    async def _mock_create_subprocess_exec(*_args: str, **_kwargs: int) -> AsyncMock:
        proc = AsyncMock()
        proc.communicate = AsyncMock(return_value=(b"", b""))
        proc.returncode = 0
        proc.kill = MagicMock()
        return proc

    with patch("asyncio.create_subprocess_exec", side_effect=_mock_create_subprocess_exec):
        yield
