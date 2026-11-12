"""How the OpenAI Responses API spells its requests, streams and replies."""

import copy
import json
import logging
from typing import Callable, override

from app.plugins.openai_dialect import (
    ChatProjection,
    ConversationDialect,
    DeltaDecoder,
    OutputSplicer,
    TokenUsage,
    TranslatedOverrides,
    build_tool_call_piece,
    canonicalize_arguments,
)
from app.plugins.utils import SSEByteDecoder

_logger = logging.getLogger(__name__)

__all__ = ["ResponsesDialect"]

_CREATED = "response.created"
_IN_PROGRESS = "response.in_progress"
_COMPLETED = "response.completed"
_INCOMPLETE = "response.incomplete"
_FAILED = "response.failed"
_ITEM_ADDED = "response.output_item.added"
_ITEM_DONE = "response.output_item.done"
_OUTPUT_TEXT_DELTA = "response.output_text.delta"
_REASONING_TEXT_DELTA = "response.reasoning_text.delta"
_REASONING_SUMMARY_TEXT_DELTA = "response.reasoning_summary_text.delta"
_ARGUMENTS_DELTA = "response.function_call_arguments.delta"
_ARGUMENTS_DONE = "response.function_call_arguments.done"
_OUTPUT_TEXT_DONE = "response.output_text.done"
_REASONING_TEXT_DONE = "response.reasoning_text.done"
_REASONING_SUMMARY_TEXT_DONE = "response.reasoning_summary_text.done"
_CONTENT_PART_ADDED = "response.content_part.added"
_CONTENT_PART_DONE = "response.content_part.done"
_REASONING_PART_ADDED = "response.reasoning_part.added"
_REASONING_PART_DONE = "response.reasoning_part.done"
_REASONING_SUMMARY_PART_ADDED = "response.reasoning_summary_part.added"
_REASONING_SUMMARY_PART_DONE = "response.reasoning_summary_part.done"

# Every event type the decoder knows. It warns once about any other type.
_KNOWN_EVENT_TYPES = frozenset({
    _OUTPUT_TEXT_DELTA,
    _REASONING_TEXT_DELTA,
    _REASONING_SUMMARY_TEXT_DELTA,
    _ITEM_ADDED,
    _ARGUMENTS_DELTA,
    _COMPLETED,
    _INCOMPLETE,
    _FAILED,
    _CREATED,
    _IN_PROGRESS,
    _CONTENT_PART_ADDED,
    _CONTENT_PART_DONE,
    _REASONING_PART_ADDED,
    _REASONING_PART_DONE,
    _REASONING_SUMMARY_PART_ADDED,
    _REASONING_SUMMARY_PART_DONE,
    _OUTPUT_TEXT_DONE,
    _REASONING_TEXT_DONE,
    _REASONING_SUMMARY_TEXT_DONE,
    _ARGUMENTS_DONE,
    _ITEM_DONE,
    "response.output_text.annotation.added",
    "response.refusal.delta",
    "response.refusal.done",
    "error",
})

_TERMINAL_TYPES = frozenset({_COMPLETED, _INCOMPLETE, _FAILED})

# Delta events that carry generated text, and the piece type each one is.
_TEXT_DELTA_PIECES = {
    _OUTPUT_TEXT_DELTA: "content",
    _REASONING_TEXT_DELTA: "reasoning",
    _REASONING_SUMMARY_TEXT_DELTA: "reasoning",
}

# Each part shape as its part family, index field and part type.
_OUTPUT_TEXT_PART = ("content", "content_index", "output_text")
_REASONING_TEXT_PART = ("content", "content_index", "reasoning_text")
_SUMMARY_TEXT_PART = ("summary", "summary_index", "summary_text")

# Text events by type, each with its part shape.
_TEXT_EVENT_PARTS = {
    _OUTPUT_TEXT_DELTA: _OUTPUT_TEXT_PART,
    _REASONING_TEXT_DELTA: _REASONING_TEXT_PART,
    _REASONING_SUMMARY_TEXT_DELTA: _SUMMARY_TEXT_PART,
    _OUTPUT_TEXT_DONE: _OUTPUT_TEXT_PART,
    _REASONING_TEXT_DONE: _REASONING_TEXT_PART,
    _REASONING_SUMMARY_TEXT_DONE: _SUMMARY_TEXT_PART,
}

# Part events by type, each with its part shape.
_PART_EVENT_PARTS = {
    _CONTENT_PART_ADDED: _OUTPUT_TEXT_PART,
    _REASONING_PART_ADDED: _REASONING_TEXT_PART,
    _REASONING_SUMMARY_PART_ADDED: _SUMMARY_TEXT_PART,
    _CONTENT_PART_DONE: _OUTPUT_TEXT_PART,
    _REASONING_PART_DONE: _REASONING_TEXT_PART,
    _REASONING_SUMMARY_PART_DONE: _SUMMARY_TEXT_PART,
}

# The event that closes what each opening event starts.
_CLOSING_TYPES = {
    _OUTPUT_TEXT_DELTA: _OUTPUT_TEXT_DONE,
    _REASONING_TEXT_DELTA: _REASONING_TEXT_DONE,
    _REASONING_SUMMARY_TEXT_DELTA: _REASONING_SUMMARY_TEXT_DONE,
    _CONTENT_PART_ADDED: _CONTENT_PART_DONE,
    _REASONING_PART_ADDED: _REASONING_PART_DONE,
    _REASONING_SUMMARY_PART_ADDED: _REASONING_SUMMARY_PART_DONE,
}

_PLAIN_MESSAGE_ROLES = frozenset({"user", "system", "developer"})
_TEXT_BLOCK_TYPES = frozenset({"input_text", "output_text"})

_MESSAGE = "message"
_REASONING = "reasoning"
_FUNCTION_CALL = "function_call"
_FUNCTION_CALL_OUTPUT = "function_call_output"
_ASSISTANT_TEXT = "assistant_text"
_ASSISTANT_NO_TEXT = "assistant_no_text"

_INSTRUCTIONS = "instructions"
_STRING_INPUT = "string_input"
_SINGLE = "single"
_GROUP = "group"

_INCOMPLETE_STATUS = "incomplete"
_CANCELLED_STATUS = "cancelled"


# ── Request projection ──


def _reasoning_text(item: dict) -> str:
    for key in ("content", "summary"):
        blocks = item.get(key)
        if isinstance(blocks, list) and blocks and isinstance(blocks[0], dict):
            return blocks[0].get("text") or ""
    return ""


def _assistant_text(item: dict) -> str | None:
    content = item.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list) and content and isinstance(content[0], dict):
        text = content[0].get("text")
        if isinstance(text, str):
            return text
    return None


def _is_message_item(item: dict) -> bool:
    return item.get("type", _MESSAGE) == _MESSAGE


def _assistant_kind(item: dict | list | str) -> str | None:
    if not isinstance(item, dict):
        return None
    item_type = item.get("type")
    if item_type == _REASONING:
        return _REASONING
    if item_type == _FUNCTION_CALL:
        return _FUNCTION_CALL
    if item.get("role") == "assistant" and _is_message_item(item):
        return _ASSISTANT_TEXT if _assistant_text(item) is not None else _ASSISTANT_NO_TEXT
    return None


def _chat_content(content: str | list | None) -> str | list | None:
    if not isinstance(content, list):
        return content
    return [
        {"type": "text", "text": block.get("text")}
        if isinstance(block, dict) and block.get("type") in _TEXT_BLOCK_TYPES
        else block
        for block in content
    ]


def _project_single(item: dict | list | str) -> dict | list | str:
    if not isinstance(item, dict):
        return item
    if item.get("type") == _FUNCTION_CALL_OUTPUT:
        return {
            "role": "tool",
            "tool_call_id": item.get("call_id"),
            "content": _chat_content(item.get("output")),
        }
    if item.get("role") in _PLAIN_MESSAGE_ROLES and _is_message_item(item):
        return {"role": item["role"], "content": _chat_content(item.get("content"))}
    return {}


def _fold_into(message: dict, item: dict, kind: str) -> None:
    if kind == _REASONING:
        message["reasoning_content"] = _reasoning_text(item)
    elif kind == _FUNCTION_CALL:
        message.setdefault("tool_calls", []).append({
            "id": item.get("call_id"),
            "type": "function",
            "function": {"name": item.get("name"), "arguments": item.get("arguments")},
        })
    else:
        message["content"] = _assistant_text(item)


class _Segment:
    """One projected message and the request parts it came from."""

    def __init__(self, message: dict | list | str, items: list, kind: str) -> None:
        self.message = message
        self.items = items
        self.kind = kind
        self.snapshot: dict | list | str = copy.deepcopy(message)


class _Folder:
    """Joins reasoning, text and tool calls in a row into one assistant message."""

    def __init__(self) -> None:
        self.segments: list[_Segment] = []
        self._open: _Segment | None = None

    def can_join(self, item: dict | list | str) -> bool:
        """Tell whether an item would join the assistant message before it."""
        if self._open is None:
            return False
        message = self._open.message
        kind = _assistant_kind(item)
        if kind == _REASONING:
            return "reasoning_content" not in message
        if kind == _ASSISTANT_TEXT:
            return "content" not in message
        return kind == _FUNCTION_CALL

    def add(self, item: dict | list | str) -> None:
        """Add the next input item."""
        kind = _assistant_kind(item)
        if kind is None:
            self._open = None
            self.segments.append(_Segment(_project_single(item), [item], _SINGLE))
            return
        if not self.can_join(item):
            message: dict = {"role": "assistant"}
            if kind == _ASSISTANT_NO_TEXT:
                message["content"] = copy.deepcopy(item.get("content"))
            self._open = _Segment(message, [], _GROUP)
            self.segments.append(self._open)
            if kind == _ASSISTANT_NO_TEXT:
                self._open.items.append(item)
                return
        self._open.items.append(item)
        _fold_into(self._open.message, item, kind)
        self._open.snapshot = copy.deepcopy(self._open.message)


def _reasoning_blocks(text: str) -> list[dict]:
    return [{"type": "reasoning_text", "text": text}]


def _reasoning_item(text: str) -> dict:
    return {"type": _REASONING, "summary": [], "content": _reasoning_blocks(text)}


def _with_reasoning_text(item: dict, text: str) -> dict:
    patched = dict(item)
    patched["content"] = _reasoning_blocks(text)
    patched["summary"] = []
    return patched


def _render_assistant(message: dict) -> list[dict]:
    items: list[dict] = []
    reasoning = message.get("reasoning_content")
    if reasoning:
        items.append(_reasoning_item(reasoning))
    tool_calls = message.get("tool_calls") or []
    content = message.get("content")
    if content or not (items or tool_calls):
        items.append({
            "type": _MESSAGE,
            "role": "assistant",
            "content": [{"type": "output_text", "text": content or ""}],
        })
    for tool_call in tool_calls:
        function = tool_call.get("function") or {}
        items.append({
            "type": _FUNCTION_CALL,
            "call_id": tool_call.get("id"),
            "name": function.get("name"),
            "arguments": function.get("arguments"),
        })
    return items


def _render(message: dict) -> list[dict]:
    role = message.get("role")
    if role == "assistant":
        return _render_assistant(message)
    if role == "tool":
        return [{
            "type": _FUNCTION_CALL_OUTPUT,
            "call_id": message.get("tool_call_id"),
            "output": message.get("content"),
        }]
    return [{"type": _MESSAGE, "role": role, "content": message.get("content")}]


def _tool_call_function(message: dict, index: int) -> dict | None:
    tool_calls = message.get("tool_calls") or []
    if index >= len(tool_calls) or not isinstance(tool_calls[index], dict):
        return None
    return tool_calls[index].get("function")


def _patch_call(item: dict, message: dict, snapshot: dict, index: int) -> dict:
    function = _tool_call_function(message, index)
    if not isinstance(function, dict) or function == _tool_call_function(snapshot, index):
        return item
    patched = dict(item)
    patched["name"] = function.get("name")
    patched["arguments"] = function.get("arguments")
    return patched


def _patch_text(item: dict, message: dict, snapshot: dict) -> dict:
    text = message.get("content")
    if text == snapshot.get("content"):
        return item
    patched = dict(item)
    if isinstance(item.get("content"), str):
        patched["content"] = text
    else:
        blocks = list(item["content"])
        blocks[0] = {**blocks[0], "text": text}
        patched["content"] = blocks
    return patched


def _new_reasoning_index(items: list, folder: _Folder) -> int:
    """Give where a new reasoning item goes so it joins the group's own message."""
    kind = _assistant_kind(items[0])
    if kind == _ASSISTANT_NO_TEXT:
        return 1
    if kind == _ASSISTANT_TEXT and folder.can_join(_reasoning_item("")):
        return 1
    return 0


def _patch_reasoning(items: list, message: dict, snapshot: dict, folder: _Folder) -> list:
    is_set = "reasoning_content" in message
    if is_set == ("reasoning_content" in snapshot) and message.get(
        "reasoning_content"
    ) == snapshot.get("reasoning_content"):
        return items
    index = next(
        (position for position, item in enumerate(items) if _assistant_kind(item) == _REASONING),
        None,
    )
    if is_set:
        if index is None:
            items.insert(_new_reasoning_index(items, folder), _reasoning_item(message["reasoning_content"]))
            return items
        items[index] = _with_reasoning_text(items[index], message["reasoning_content"])
        return items
    if index is None or not _reasoning_text(items[index]):
        return items
    if index == 0 and len(items) > 1 and folder.can_join(items[1]):
        items[0] = _with_reasoning_text(items[0], "")
        return items
    return items[:index] + items[index + 1:]


def _patch_group(segment: _Segment, folder: _Folder) -> list:
    message, snapshot = segment.message, segment.snapshot
    patched: list = []
    call_index = 0
    for item in segment.items:
        kind = _assistant_kind(item)
        if kind == _FUNCTION_CALL:
            patched.append(_patch_call(item, message, snapshot, call_index))
            call_index += 1
        elif kind == _ASSISTANT_TEXT:
            patched.append(_patch_text(item, message, snapshot))
        else:
            patched.append(item)
    return _patch_reasoning(patched, message, snapshot, folder)


def _patch_single(segment: _Segment) -> list:
    item = segment.items[0]
    message = segment.message
    if message == segment.snapshot or not isinstance(item, dict) or "role" not in message:
        return [item]
    patched = dict(item)
    field = "output" if message["role"] == "tool" else "content"
    patched[field] = message.get("content")
    return [patched]


class _ResponsesProjection(ChatProjection):
    """A chat style view of a Responses request's instructions and input."""

    def __init__(self, body: dict) -> None:
        self._body = body
        self._segments: list[_Segment] = []
        super().__init__([])
        instructions = body.get("instructions")
        if isinstance(instructions, str) and instructions:
            self._segments.append(
                _Segment({"role": "system", "content": instructions}, [], _INSTRUCTIONS)
            )
        input_value = body["input"]
        if isinstance(input_value, str):
            self._segments.append(
                _Segment({"role": "user", "content": input_value}, [input_value], _STRING_INPUT)
            )
        else:
            folder = _Folder()
            for item in input_value:
                folder.add(item)
            self._segments.extend(folder.segments)
        self.messages.extend(segment.message for segment in self._segments)

    def _is_unchanged(self) -> bool:
        return len(self.messages) == len(self._segments) and all(
            message is segment.message and message == segment.snapshot
            for message, segment in zip(self.messages, self._segments)
        )

    def _write_segment(self, segment: _Segment, folder: _Folder) -> list:
        message = segment.message
        if segment.kind == _INSTRUCTIONS:
            if message != segment.snapshot:
                self._body["instructions"] = message.get("content")
            return []
        if segment.kind == _STRING_INPUT:
            return [{"type": _MESSAGE, "role": "user", "content": message.get("content")}]
        if segment.kind == _SINGLE:
            return _patch_single(segment)
        if message == segment.snapshot:
            return list(segment.items)
        return _patch_group(segment, folder)

    def _write_instructions(self, by_id: dict[int, _Segment]) -> list:
        """Write a gone instructions message back and give the messages left for input."""
        if not self._segments or self._segments[0].kind != _INSTRUCTIONS:
            return self.messages
        if any(message is self._segments[0].message for message in self.messages):
            return self.messages
        first = self.messages[0] if self.messages else None
        if isinstance(first, dict) and first.get("role") == "system" and id(first) not in by_id:
            self._body["instructions"] = first.get("content")
            return self.messages[1:]
        del self._body["instructions"]
        return self.messages

    @override
    def write_back(self) -> None:
        if self._is_unchanged():
            return
        by_id = {
            id(segment.message): segment
            for segment in self._segments
            if isinstance(segment.message, dict)
        }
        folder = _Folder()
        new_input: list = []
        for message in self._write_instructions(by_id):
            segment = by_id.get(id(message)) if isinstance(message, dict) else None
            if not isinstance(message, dict):
                items = [message]
            elif segment is None:
                items = _render(message)
            else:
                items = self._write_segment(segment, folder)
            for item in items:
                folder.add(item)
                new_input.append(item)
        self._body["input"] = new_input


# ── Stream reading ──


class _ResponsesDeltaDecoder(DeltaDecoder):
    """Reads the item events of one Responses stream."""

    def __init__(self) -> None:
        self._call_ids: dict[str, str] = {}
        self._streamed_calls: list[tuple[str, str]] = []
        self._warned_types: set[str] = set()
        self._response_id: str | None = None
        self._readers: dict[str, Callable[[dict], dict | None]] = {
            **dict.fromkeys(_TEXT_DELTA_PIECES, self._read_text),
            _ITEM_ADDED: self._read_item_added,
            _ARGUMENTS_DELTA: self._read_arguments,
            _COMPLETED: self._read_terminal,
            _INCOMPLETE: self._read_terminal,
        }

    @override
    def normalize(self, event: dict) -> tuple[list[dict], int | None]:
        if not isinstance(event, dict):
            return [], None
        event_type = event.get("type")
        if not isinstance(event_type, str):
            return [], None
        response = event.get("response")
        if self._response_id is None and isinstance(response, dict):
            self._response_id = response.get("id")
        reader = self._readers.get(event_type)
        if reader is not None:
            piece = reader(event)
            return ([piece], 0) if piece is not None else ([], None)
        if event_type not in _KNOWN_EVENT_TYPES and event_type not in self._warned_types:
            self._warned_types.add(event_type)
            _logger.warning(
                "[ResponsesDialect] Unknown event type %r in response %s",
                event_type,
                self._response_id,
            )
        return [], None

    def _read_text(self, event: dict) -> dict:
        return {"type": _TEXT_DELTA_PIECES[event["type"]], "text": event.get("delta") or ""}

    def _read_item_added(self, event: dict) -> dict | None:
        item = event.get("item")
        if not isinstance(item, dict) or item.get("type") != _FUNCTION_CALL:
            return None
        call_id = item.get("call_id") or ""
        name = item.get("name") or ""
        arguments = canonicalize_arguments(item.get("arguments") or "")
        if isinstance(item.get("id"), str):
            self._call_ids[item["id"]] = call_id
        self._streamed_calls.append((call_id, name))
        return build_tool_call_piece(name, arguments, call_id)

    def _read_arguments(self, event: dict) -> dict:
        arguments = canonicalize_arguments(event.get("delta") or "")
        return build_tool_call_piece("", arguments, self._call_ids.get(str(event.get("item_id")), ""))

    def _read_terminal(self, event: dict) -> dict:
        response = event.get("response")
        if not isinstance(response, dict):
            response = {}
        status = _INCOMPLETE_STATUS if event["type"] == _INCOMPLETE else response.get("status")
        return {
            "type": "finish_reason",
            "text": self._map_finish_reason(status, response),
            "tool_call_id_aliases": self._pair_call_ids(response),
        }

    def _map_finish_reason(self, status: str | None, response: dict) -> str:
        if status == _INCOMPLETE_STATUS:
            details = response.get("incomplete_details")
            is_filtered = isinstance(details, dict) and details.get("reason") == "content_filter"
            return "content_filter" if is_filtered else "length"
        if status == _CANCELLED_STATUS:
            return "abort"
        return "tool_calls" if self._streamed_calls else "stop"

    def _pair_call_ids(self, response: dict) -> list[tuple[str, str]]:
        terminal_calls = [
            (item.get("call_id") or "", item.get("name") or "")
            for item in response.get("output") or []
            if isinstance(item, dict) and item.get("type") == _FUNCTION_CALL
        ]
        is_paired = len(terminal_calls) == len(self._streamed_calls) and all(
            streamed[1] == terminal[1]
            for streamed, terminal in zip(self._streamed_calls, terminal_calls)
        )
        if not is_paired:
            _logger.warning(
                "[ResponsesDialect] Streamed calls %s do not match terminal calls %s, keeping no aliases",
                [call_id for call_id, _ in self._streamed_calls],
                [call_id for call_id, _ in terminal_calls],
            )
            return []
        return [
            (streamed[0], terminal[0])
            for streamed, terminal in zip(self._streamed_calls, terminal_calls)
        ]


# ── Item accumulation ──


class _Part:
    """One content or summary part of an output item."""

    def __init__(self, index_field: str, index: int, kind: str) -> None:
        self.index_field = index_field
        self.index = index
        self.kind = kind
        self.text = ""
        self.text_closing_type: str | None = None
        self.part_closing_type: str | None = None


def _part_payload(part: _Part) -> dict:
    """Describe a part the way an item lists it."""
    if part.kind == "output_text":
        return {"type": "output_text", "text": part.text, "annotations": []}
    return {"type": part.kind, "text": part.text}


def _close_part(part: _Part, item_id: str | None) -> list[dict]:
    """Build the events that close what a part left open."""
    events: list[dict] = []
    if part.text_closing_type is not None:
        events.append({
            "type": part.text_closing_type,
            "item_id": item_id,
            part.index_field: part.index,
            "text": part.text,
        })
    if part.part_closing_type is not None:
        events.append({
            "type": part.part_closing_type,
            "item_id": item_id,
            part.index_field: part.index,
            "part": _part_payload(part),
        })
    return events


class _ItemRecord:
    """One output item rebuilt from its events."""

    def __init__(self, item: dict, output_index: int | None) -> None:
        self.item = item
        self.output_index = output_index
        self.parts: dict[tuple[str, int], _Part] = {}
        self.arguments: str = item.get("arguments") or ""
        self.is_arguments_open = False
        self.is_done = False

    def get_part(self, family: str, index_field: str, index: int, kind: str) -> _Part:
        """Get a part, adding it on first use."""
        key = (family, index)
        if key not in self.parts:
            self.parts[key] = _Part(index_field, index, kind)
        return self.parts[key]

    def _ordered_parts(self, family: str | None = None) -> list[_Part]:
        return [self.parts[key] for key in sorted(self.parts) if family in (None, key[0])]

    def snapshot(self, status: str | None) -> dict:
        """Build the item as streamed so far."""
        if self.is_done:
            return copy.deepcopy(self.item)
        item = copy.deepcopy(self.item)
        for family in ("content", "summary"):
            payloads = [_part_payload(part) for part in self._ordered_parts(family)]
            if payloads:
                item[family] = payloads
        if item.get("type") == _FUNCTION_CALL:
            item["arguments"] = self.arguments
        if status is not None:
            item["status"] = status
        return item

    def close(self) -> list[dict]:
        """Close what the stream left open, and give the events that close it."""
        item_id = self.item.get("id")
        events: list[dict] = []
        for part in self._ordered_parts():
            events.extend(_close_part(part, item_id))
        if self.is_arguments_open:
            events.append({
                "type": _ARGUMENTS_DONE,
                "item_id": item_id,
                "name": self.item.get("name"),
                "arguments": self.arguments,
            })
        self.item = self.snapshot(_INCOMPLETE_STATUS)
        self.is_done = True
        events.append({"type": _ITEM_DONE, "item": copy.deepcopy(self.item)})
        if self.output_index is not None:
            for event in events:
                event["output_index"] = self.output_index
        return events


class _ItemAccumulator:
    """Rebuilds the output items of a Responses stream from its events."""

    def __init__(self) -> None:
        self.records: list[_ItemRecord] = []
        self._by_id: dict[tuple[int, str], _ItemRecord] = {}

    def feed(self, event: dict, attempt: int = 0) -> None:
        """Take the next event of the given attempt."""
        event_type = event.get("type")
        if event_type in (_ITEM_ADDED, _ITEM_DONE):
            self._feed_item(event, event_type == _ITEM_DONE, attempt)
            return
        record = self._by_id.get((attempt, str(event.get("item_id"))))
        if record is None:
            return
        if event_type in _TEXT_EVENT_PARTS:
            self._feed_text(record, event, event_type)
        elif event_type in _PART_EVENT_PARTS:
            self._feed_part(record, event, event_type)
        elif event_type == _ARGUMENTS_DELTA:
            record.arguments += event.get("delta") or ""
            record.is_arguments_open = True
        elif event_type == _ARGUMENTS_DONE:
            record.arguments = event.get("arguments") or record.arguments
            record.is_arguments_open = False

    def _feed_item(self, event: dict, is_done: bool, attempt: int) -> None:
        item = event.get("item")
        if not isinstance(item, dict):
            return
        item_id = (attempt, str(item.get("id")))
        record = self._by_id.get(item_id)
        if record is None or (record.is_done and not is_done):
            record = _ItemRecord(copy.deepcopy(item), event.get("output_index"))
            self.records.append(record)
            self._by_id[item_id] = record
        if is_done:
            record.item = copy.deepcopy(item)
            record.is_done = True

    def _feed_text(self, record: _ItemRecord, event: dict, event_type: str) -> None:
        family, index_field, kind = _TEXT_EVENT_PARTS[event_type]
        part = record.get_part(family, index_field, event.get(index_field, 0), kind)
        if event_type in _CLOSING_TYPES:
            part.text += event.get("delta") or ""
            part.text_closing_type = _CLOSING_TYPES[event_type]
        else:
            part.text = event.get("text") or part.text
            part.text_closing_type = None

    def _feed_part(self, record: _ItemRecord, event: dict, event_type: str) -> None:
        family, index_field, kind = _PART_EVENT_PARTS[event_type]
        payload = event.get("part")
        if isinstance(payload, dict):
            kind = payload.get("type") or kind
        part = record.get_part(family, index_field, event.get(index_field, 0), kind)
        if isinstance(payload, dict) and payload.get("text"):
            part.text = payload["text"]
        part.part_closing_type = _CLOSING_TYPES.get(event_type)


# ── Output splicing ──


def _read_payload(data: str) -> dict | None:
    """Read event data as a JSON object with a string type, or None if it is not one."""
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("type"), str):
        return None
    return payload


class _ResponsesSplicer(OutputSplicer):
    """Joins the approved output of every attempt into one Responses stream."""

    def __init__(self) -> None:
        self._items = _ItemAccumulator()
        self._attempt = 0
        self._is_verbatim = True
        self._is_retry_pending = False
        self._item_offset = 0
        self._last_sequence: int | None = None
        self._has_sequence = False
        self._has_event_lines = False
        self._response_id: str | None = None
        self._terminator = b""

    @override
    def splice(self, approved: bytes) -> bytes:
        if not approved:
            return b""
        output = bytearray()
        complete_end = 0
        for event_type, data, _event_id, raw_lines in SSEByteDecoder().feed(approved):
            raw = b"".join(raw_lines) + b"\n"
            complete_end += len(raw)
            self._splice_event(raw, _read_payload(data), event_type != "message", output)
        trailing = approved[complete_end:]
        if trailing:
            self._start_next_attempt(output)
            self._emit(output, trailing)
            self._terminator = b"\n" if trailing.endswith(b"\n") else b"\n\n"
        return bytes(output)

    @override
    def retry(self) -> None:
        self._is_verbatim = False
        self._is_retry_pending = True

    @override
    def finish(self) -> bytes:
        if not self._is_retry_pending:
            return b""
        output = bytearray()
        self._close_retry(output)
        return bytes(output)

    def _emit(self, output: bytearray, data: bytes) -> None:
        output.extend(self._terminator)
        self._terminator = b""
        output.extend(data)

    def _encode(self, payload: dict) -> bytes:
        data = b"data: " + json.dumps(payload, ensure_ascii=False).encode() + b"\n\n"
        if self._has_event_lines:
            return b"event: " + payload["type"].encode() + b"\n" + data
        return data

    def _next_sequence(self) -> int:
        self._last_sequence = 0 if self._last_sequence is None else self._last_sequence + 1
        return self._last_sequence

    def _close_retry(self, output: bytearray) -> None:
        for record in self._items.records:
            self._emit_record_closings(output, record)
        self._is_retry_pending = False

    def _emit_record_closings(self, output: bytearray, record: _ItemRecord) -> None:
        if record.is_done:
            return
        for event in record.close():
            if self._has_sequence:
                event["sequence_number"] = self._next_sequence()
            self._emit(output, self._encode(event))

    def _start_next_attempt(self, output: bytearray) -> None:
        if not self._is_retry_pending:
            return
        self._close_retry(output)
        self._attempt += 1
        self._item_offset = len(self._items.records)

    def _splice_event(self, raw: bytes, payload: dict | None, has_event_line: bool, output: bytearray) -> None:
        if payload is None:
            self._start_next_attempt(output)
            self._emit(output, raw)
            return
        self._has_event_lines = self._has_event_lines or has_event_line
        self._has_sequence = self._has_sequence or "sequence_number" in payload
        event_type = payload["type"]
        if event_type == _CREATED:
            self._start_next_attempt(output)
        if self._attempt > 0 and event_type in (_CREATED, _IN_PROGRESS):
            return
        if self._is_verbatim:
            self._record_verbatim(payload)
            self._emit(output, raw)
        else:
            payload = self._renumber(payload)
            self._emit(output, self._encode(payload))
        self._items.feed(payload, self._attempt)

    def _record_verbatim(self, payload: dict) -> None:
        sequence = payload.get("sequence_number")
        if isinstance(sequence, int):
            self._last_sequence = sequence
        response = payload.get("response")
        if self._response_id is None and isinstance(response, dict):
            self._response_id = response.get("id")

    def _renumber(self, payload: dict) -> dict:
        payload = copy.deepcopy(payload)
        if "sequence_number" in payload:
            payload["sequence_number"] = self._next_sequence()
        if isinstance(payload.get("output_index"), int):
            payload["output_index"] += self._item_offset
        response = payload.get("response")
        if payload["type"] in _TERMINAL_TYPES and isinstance(response, dict) and self._attempt > 0:
            if self._response_id is not None:
                response["id"] = self._response_id
            earlier = [record.snapshot(None) for record in self._items.records[:self._item_offset]]
            response["output"] = earlier + list(response.get("output") or [])
        return payload


# ── Dialect ──


class ResponsesDialect(ConversationDialect):
    """The Responses API."""

    path: str = "/v1/responses"
    history_key: str = "input"

    @override
    def translate_overrides(self, overrides: dict) -> TranslatedOverrides:
        renamed = ("max_tokens", "max_completion_tokens", "reasoning_effort")
        forced = {key: value for key, value in overrides.items() if key not in renamed}
        yielding: dict = {}
        if "max_completion_tokens" in overrides:
            forced["max_output_tokens"] = overrides["max_completion_tokens"]
        elif "max_tokens" in overrides:
            forced.pop("max_output_tokens", None)
            yielding["max_output_tokens"] = overrides["max_tokens"]
        if "reasoning_effort" in overrides:
            reasoning = forced.get("reasoning")
            reasoning = dict(reasoning) if isinstance(reasoning, dict) else {}
            reasoning["effort"] = overrides["reasoning_effort"]
            forced["reasoning"] = reasoning
        return TranslatedOverrides(forced, yielding)

    @override
    def force_stream(self, body: dict) -> None:
        body["stream"] = True

    @override
    def project(self, body: dict) -> ChatProjection | None:
        if not isinstance(body, dict) or not isinstance(body.get(self.history_key), (str, list)):
            return None
        return _ResponsesProjection(body)

    @override
    def delta_decoder(self) -> DeltaDecoder:
        return _ResponsesDeltaDecoder()

    @override
    def output_splicer(self) -> OutputSplicer:
        return _ResponsesSplicer()

    @override
    def generated_texts(self, event: dict) -> list[str]:
        event_type = event.get("type")
        if event_type in _TEXT_DELTA_PIECES or event_type == _ARGUMENTS_DELTA:
            return [event.get("delta") or ""]
        item = event.get("item")
        if event_type == _ITEM_ADDED and isinstance(item, dict) and item.get("type") == _FUNCTION_CALL:
            return [(item.get("name") or "") + (item.get("arguments") or "")]
        return []

    @override
    def usage(self, event: dict) -> TokenUsage | None:
        response = event.get("response")
        if event.get("type") not in _TERMINAL_TYPES or not isinstance(response, dict):
            return None
        usage = response.get("usage")
        if not isinstance(usage, dict):
            return None
        return TokenUsage(
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
            usage.get("total_tokens", 0),
        )

    @override
    def assemble_non_streaming(self, events: list[dict]) -> bytes | None:
        terminal: dict | None = None
        envelope: dict | None = None
        accumulator = _ItemAccumulator()
        for event in events:
            if not isinstance(event, dict):
                continue
            accumulator.feed(event)
            response = event.get("response")
            if isinstance(response, dict):
                envelope = response
                if event.get("type") in _TERMINAL_TYPES:
                    terminal = response
        if terminal is not None:
            return json.dumps(terminal).encode()
        if envelope is None and not accumulator.records:
            return None
        reply = copy.deepcopy(envelope) if envelope is not None else {}
        reply["output"] = [record.snapshot(None) for record in accumulator.records]
        return json.dumps(reply).encode()
