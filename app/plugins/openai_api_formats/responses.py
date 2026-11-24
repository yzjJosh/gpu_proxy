"""Request, stream, and response format for the OpenAI Responses API."""

import copy
import json
import logging
from typing import Callable, NamedTuple, override

from app.plugins.openai_api_formats.base import (
    ConversationAPIFormat,
    EditableMessageHistory,
    ParsedStreamEvent,
    RequestOverrides,
    RetryStreamCombiner,
    StreamEventParser,
    TokenUsage,
    build_tool_call_update,
    standardize_json_arguments,
)
from app.plugins.utils import SSEByteDecoder

_logger = logging.getLogger(__name__)

__all__ = ["ResponsesAPIFormat"]

_EVENT_RESPONSE_CREATED = "response.created"
_EVENT_RESPONSE_IN_PROGRESS = "response.in_progress"
_EVENT_RESPONSE_COMPLETED = "response.completed"
_EVENT_RESPONSE_INCOMPLETE = "response.incomplete"
_EVENT_RESPONSE_FAILED = "response.failed"
_EVENT_OUTPUT_ITEM_ADDED = "response.output_item.added"
_EVENT_OUTPUT_ITEM_DONE = "response.output_item.done"
_EVENT_OUTPUT_TEXT_DELTA = "response.output_text.delta"
_EVENT_REASONING_TEXT_DELTA = "response.reasoning_text.delta"
_EVENT_REASONING_SUMMARY_TEXT_DELTA = "response.reasoning_summary_text.delta"
_EVENT_FUNCTION_CALL_ARGUMENTS_DELTA = "response.function_call_arguments.delta"
_EVENT_FUNCTION_CALL_ARGUMENTS_DONE = "response.function_call_arguments.done"
_EVENT_OUTPUT_TEXT_DONE = "response.output_text.done"
_EVENT_REASONING_TEXT_DONE = "response.reasoning_text.done"
_EVENT_REASONING_SUMMARY_TEXT_DONE = "response.reasoning_summary_text.done"
_EVENT_CONTENT_PART_ADDED = "response.content_part.added"
_EVENT_CONTENT_PART_DONE = "response.content_part.done"
_EVENT_REASONING_PART_ADDED = "response.reasoning_part.added"
_EVENT_REASONING_PART_DONE = "response.reasoning_part.done"
_EVENT_REASONING_SUMMARY_PART_ADDED = "response.reasoning_summary_part.added"
_EVENT_REASONING_SUMMARY_PART_DONE = "response.reasoning_summary_part.done"

# Every Responses SSE event type the parser recognizes. Unknown types warn once.
_KNOWN_EVENT_TYPES = frozenset({
    _EVENT_OUTPUT_TEXT_DELTA,
    _EVENT_REASONING_TEXT_DELTA,
    _EVENT_REASONING_SUMMARY_TEXT_DELTA,
    _EVENT_OUTPUT_ITEM_ADDED,
    _EVENT_FUNCTION_CALL_ARGUMENTS_DELTA,
    _EVENT_RESPONSE_COMPLETED,
    _EVENT_RESPONSE_INCOMPLETE,
    _EVENT_RESPONSE_FAILED,
    _EVENT_RESPONSE_CREATED,
    _EVENT_RESPONSE_IN_PROGRESS,
    _EVENT_CONTENT_PART_ADDED,
    _EVENT_CONTENT_PART_DONE,
    _EVENT_REASONING_PART_ADDED,
    _EVENT_REASONING_PART_DONE,
    _EVENT_REASONING_SUMMARY_PART_ADDED,
    _EVENT_REASONING_SUMMARY_PART_DONE,
    _EVENT_OUTPUT_TEXT_DONE,
    _EVENT_REASONING_TEXT_DONE,
    _EVENT_REASONING_SUMMARY_TEXT_DONE,
    _EVENT_FUNCTION_CALL_ARGUMENTS_DONE,
    _EVENT_OUTPUT_ITEM_DONE,
    "response.output_text.annotation.added",
    "response.refusal.delta",
    "response.refusal.done",
    "error",
})

_TERMINAL_EVENT_TYPES = frozenset({
    _EVENT_RESPONSE_COMPLETED,
    _EVENT_RESPONSE_INCOMPLETE,
    _EVENT_RESPONSE_FAILED,
})

# Maps each text delta event type to the guard-rail update type it produces.
_TEXT_DELTA_UPDATE_TYPES = {
    _EVENT_OUTPUT_TEXT_DELTA: "content",
    _EVENT_REASONING_TEXT_DELTA: "reasoning",
    _EVENT_REASONING_SUMMARY_TEXT_DELTA: "reasoning",
}


class _OutputPartSpec(NamedTuple):
    """Describes where an output part lives inside a Responses output item."""

    container_field: str
    index_field: str
    part_type: str


_OUTPUT_TEXT_PART_SPEC = _OutputPartSpec("content", "content_index", "output_text")
_REASONING_TEXT_PART_SPEC = _OutputPartSpec("content", "content_index", "reasoning_text")
_SUMMARY_TEXT_PART_SPEC = _OutputPartSpec("summary", "summary_index", "summary_text")

_TEXT_EVENT_PART_SPECS = {
    _EVENT_OUTPUT_TEXT_DELTA: _OUTPUT_TEXT_PART_SPEC,
    _EVENT_REASONING_TEXT_DELTA: _REASONING_TEXT_PART_SPEC,
    _EVENT_REASONING_SUMMARY_TEXT_DELTA: _SUMMARY_TEXT_PART_SPEC,
    _EVENT_OUTPUT_TEXT_DONE: _OUTPUT_TEXT_PART_SPEC,
    _EVENT_REASONING_TEXT_DONE: _REASONING_TEXT_PART_SPEC,
    _EVENT_REASONING_SUMMARY_TEXT_DONE: _SUMMARY_TEXT_PART_SPEC,
}

_PART_LIFECYCLE_EVENT_SPECS = {
    _EVENT_CONTENT_PART_ADDED: _OUTPUT_TEXT_PART_SPEC,
    _EVENT_REASONING_PART_ADDED: _REASONING_TEXT_PART_SPEC,
    _EVENT_REASONING_SUMMARY_PART_ADDED: _SUMMARY_TEXT_PART_SPEC,
    _EVENT_CONTENT_PART_DONE: _OUTPUT_TEXT_PART_SPEC,
    _EVENT_REASONING_PART_DONE: _REASONING_TEXT_PART_SPEC,
    _EVENT_REASONING_SUMMARY_PART_DONE: _SUMMARY_TEXT_PART_SPEC,
}

# Maps each opening stream event type to the done event that closes it.
_OPENING_TO_CLOSING_EVENT_TYPES = {
    _EVENT_OUTPUT_TEXT_DELTA: _EVENT_OUTPUT_TEXT_DONE,
    _EVENT_REASONING_TEXT_DELTA: _EVENT_REASONING_TEXT_DONE,
    _EVENT_REASONING_SUMMARY_TEXT_DELTA: _EVENT_REASONING_SUMMARY_TEXT_DONE,
    _EVENT_CONTENT_PART_ADDED: _EVENT_CONTENT_PART_DONE,
    _EVENT_REASONING_PART_ADDED: _EVENT_REASONING_PART_DONE,
    _EVENT_REASONING_SUMMARY_PART_ADDED: _EVENT_REASONING_SUMMARY_PART_DONE,
}

_STANDARD_MESSAGE_ROLES = frozenset({"user", "system", "developer"})
_RESPONSES_TEXT_BLOCK_TYPES = frozenset({"input_text", "output_text"})

_ITEM_TYPE_MESSAGE = "message"
_ITEM_TYPE_REASONING = "reasoning"
_ITEM_TYPE_FUNCTION_CALL = "function_call"
_ITEM_TYPE_FUNCTION_CALL_OUTPUT = "function_call_output"

_ASSISTANT_ITEM_REASONING = "reasoning"
_ASSISTANT_ITEM_FUNCTION_CALL = "function_call"
_ASSISTANT_ITEM_TEXT_MESSAGE = "assistant_text_message"
_ASSISTANT_ITEM_EMPTY_MESSAGE = "assistant_empty_message"

_SOURCE_INSTRUCTIONS = "instructions"
_SOURCE_STRING_INPUT = "string_input"
_SOURCE_SINGLE_ITEM = "single_item"
_SOURCE_ASSISTANT_GROUP = "assistant_group"

_STATUS_INCOMPLETE = "incomplete"
_STATUS_CANCELLED = "cancelled"


# ── Request message history conversion ──


def _extract_reasoning_text(item: dict) -> str:
    for field_name in ("content", "summary"):
        blocks = item.get(field_name)
        if isinstance(blocks, list) and blocks and isinstance(blocks[0], dict):
            return blocks[0].get("text") or ""
    return ""


def _extract_assistant_message_text(item: dict) -> str | None:
    content = item.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list) and content and isinstance(content[0], dict):
        text = content[0].get("text")
        if isinstance(text, str):
            return text
    return None


def _is_message_item(item: dict) -> bool:
    return item.get("type", _ITEM_TYPE_MESSAGE) == _ITEM_TYPE_MESSAGE


def _classify_assistant_item(item: dict | list | str) -> str | None:
    if not isinstance(item, dict):
        return None
    item_type = item.get("type")
    if item_type == _ITEM_TYPE_REASONING:
        return _ASSISTANT_ITEM_REASONING
    if item_type == _ITEM_TYPE_FUNCTION_CALL:
        return _ASSISTANT_ITEM_FUNCTION_CALL
    if item.get("role") == "assistant" and _is_message_item(item):
        if _extract_assistant_message_text(item) is not None:
            return _ASSISTANT_ITEM_TEXT_MESSAGE
        return _ASSISTANT_ITEM_EMPTY_MESSAGE
    return None


def _convert_content_blocks_to_chat_format(content: str | list | None) -> str | list | None:
    if not isinstance(content, list):
        return content
    return [
        {"type": "text", "text": block.get("text")}
        if isinstance(block, dict) and block.get("type") in _RESPONSES_TEXT_BLOCK_TYPES
        else block
        for block in content
    ]


def _convert_non_assistant_item_to_chat_message(item: dict | list | str) -> dict | list | str:
    if not isinstance(item, dict):
        return item
    if item.get("type") == _ITEM_TYPE_FUNCTION_CALL_OUTPUT:
        return {
            "role": "tool",
            "tool_call_id": item.get("call_id"),
            "content": _convert_content_blocks_to_chat_format(item.get("output")),
        }
    if item.get("role") in _STANDARD_MESSAGE_ROLES and _is_message_item(item):
        return {
            "role": item["role"],
            "content": _convert_content_blocks_to_chat_format(item.get("content")),
        }
    return {}


def _merge_assistant_item_into_chat_message(
    chat_message: dict, item: dict, item_kind: str
) -> None:
    if item_kind == _ASSISTANT_ITEM_REASONING:
        chat_message["reasoning_content"] = _extract_reasoning_text(item)
    elif item_kind == _ASSISTANT_ITEM_FUNCTION_CALL:
        chat_message.setdefault("tool_calls", []).append({
            "id": item.get("call_id"),
            "type": "function",
            "function": {"name": item.get("name"), "arguments": item.get("arguments")},
        })
    else:
        chat_message["content"] = _extract_assistant_message_text(item)


class _TrackedChatMessage:
    """One chat-style message paired with the original Responses request items."""

    def __init__(
        self,
        chat_message: dict | list | str,
        source_items: list,
        source_kind: str,
    ) -> None:
        self.chat_message = chat_message
        self.source_items = source_items
        self.source_kind = source_kind
        self.original_snapshot: dict | list | str = copy.deepcopy(chat_message)


class _AssistantItemGrouper:
    """Groups consecutive reasoning, text, and tool call items into one assistant message."""

    def __init__(self) -> None:
        self.tracked_messages: list[_TrackedChatMessage] = []
        self._current_assistant_group: _TrackedChatMessage | None = None

    def can_merge_into_current_assistant_group(self, item: dict | list | str) -> bool:
        """Tell whether an item would merge into the open assistant message before it."""
        if self._current_assistant_group is None:
            return False
        chat_message = self._current_assistant_group.chat_message
        item_kind = _classify_assistant_item(item)
        if item_kind == _ASSISTANT_ITEM_REASONING:
            return "reasoning_content" not in chat_message
        if item_kind == _ASSISTANT_ITEM_TEXT_MESSAGE:
            return "content" not in chat_message
        return item_kind == _ASSISTANT_ITEM_FUNCTION_CALL

    def add_input_item(self, item: dict | list | str) -> None:
        """Add the next Responses input item to the grouped message list."""
        item_kind = _classify_assistant_item(item)
        if item_kind is None:
            self._current_assistant_group = None
            chat_message = _convert_non_assistant_item_to_chat_message(item)
            self.tracked_messages.append(
                _TrackedChatMessage(chat_message, [item], _SOURCE_SINGLE_ITEM)
            )
            return
        if not self.can_merge_into_current_assistant_group(item):
            chat_message = {"role": "assistant"}
            if item_kind == _ASSISTANT_ITEM_EMPTY_MESSAGE:
                chat_message["content"] = copy.deepcopy(item.get("content"))
            self._current_assistant_group = _TrackedChatMessage(
                chat_message, [], _SOURCE_ASSISTANT_GROUP
            )
            self.tracked_messages.append(self._current_assistant_group)
            if item_kind == _ASSISTANT_ITEM_EMPTY_MESSAGE:
                self._current_assistant_group.source_items.append(item)
                return
        self._current_assistant_group.source_items.append(item)
        _merge_assistant_item_into_chat_message(
            self._current_assistant_group.chat_message, item, item_kind
        )
        self._current_assistant_group.original_snapshot = copy.deepcopy(
            self._current_assistant_group.chat_message
        )


def _build_reasoning_content_blocks(text: str) -> list[dict]:
    return [{"type": "reasoning_text", "text": text}]


def _build_reasoning_item(text: str) -> dict:
    return {
        "type": _ITEM_TYPE_REASONING,
        "summary": [],
        "content": _build_reasoning_content_blocks(text),
    }


def _replace_reasoning_item_text(item: dict, text: str) -> dict:
    updated_item = dict(item)
    updated_item["content"] = _build_reasoning_content_blocks(text)
    updated_item["summary"] = []
    return updated_item


def _convert_assistant_chat_message_to_items(chat_message: dict) -> list[dict]:
    items: list[dict] = []
    reasoning = chat_message.get("reasoning_content")
    if reasoning:
        items.append(_build_reasoning_item(reasoning))
    tool_calls = chat_message.get("tool_calls") or []
    content = chat_message.get("content")
    if content or not (items or tool_calls):
        items.append({
            "type": _ITEM_TYPE_MESSAGE,
            "role": "assistant",
            "content": [{"type": "output_text", "text": content or ""}],
        })
    for tool_call in tool_calls:
        function = tool_call.get("function") or {}
        items.append({
            "type": _ITEM_TYPE_FUNCTION_CALL,
            "call_id": tool_call.get("id"),
            "name": function.get("name"),
            "arguments": function.get("arguments"),
        })
    return items


def _convert_new_chat_message_to_items(chat_message: dict) -> list[dict]:
    role = chat_message.get("role")
    if role == "assistant":
        return _convert_assistant_chat_message_to_items(chat_message)
    if role == "tool":
        return [{
            "type": _ITEM_TYPE_FUNCTION_CALL_OUTPUT,
            "call_id": chat_message.get("tool_call_id"),
            "output": chat_message.get("content"),
        }]
    return [{"type": _ITEM_TYPE_MESSAGE, "role": role, "content": chat_message.get("content")}]


def _get_tool_call_function(chat_message: dict, tool_call_index: int) -> dict | None:
    tool_calls = chat_message.get("tool_calls") or []
    if tool_call_index >= len(tool_calls) or not isinstance(tool_calls[tool_call_index], dict):
        return None
    return tool_calls[tool_call_index].get("function")


def _update_function_call_item(
    item: dict, chat_message: dict, original_snapshot: dict, tool_call_index: int
) -> dict:
    function = _get_tool_call_function(chat_message, tool_call_index)
    if not isinstance(function, dict) or function == _get_tool_call_function(
        original_snapshot, tool_call_index
    ):
        return item
    updated_item = dict(item)
    updated_item["name"] = function.get("name")
    updated_item["arguments"] = function.get("arguments")
    return updated_item


def _update_assistant_text_item(
    item: dict, chat_message: dict, original_snapshot: dict
) -> dict:
    text = chat_message.get("content")
    if text == original_snapshot.get("content"):
        return item
    updated_item = dict(item)
    if isinstance(item.get("content"), str):
        updated_item["content"] = text
    else:
        blocks = list(item["content"])
        blocks[0] = {**blocks[0], "text": text}
        updated_item["content"] = blocks
    return updated_item


def _find_reasoning_insertion_index(
    group_items: list, item_grouper: _AssistantItemGrouper
) -> int:
    """Return the insertion index for a new reasoning item."""
    first_item_kind = _classify_assistant_item(group_items[0])
    if first_item_kind == _ASSISTANT_ITEM_EMPTY_MESSAGE:
        return 1
    if (
        first_item_kind == _ASSISTANT_ITEM_TEXT_MESSAGE
        and item_grouper.can_merge_into_current_assistant_group(_build_reasoning_item(""))
    ):
        return 1
    return 0


def _update_reasoning_in_group_items(
    group_items: list,
    chat_message: dict,
    original_snapshot: dict,
    item_grouper: _AssistantItemGrouper,
) -> list:
    has_reasoning = "reasoning_content" in chat_message
    if has_reasoning == ("reasoning_content" in original_snapshot) and chat_message.get(
        "reasoning_content"
    ) == original_snapshot.get("reasoning_content"):
        return group_items
    reasoning_index = next(
        (
            position
            for position, item in enumerate(group_items)
            if _classify_assistant_item(item) == _ASSISTANT_ITEM_REASONING
        ),
        None,
    )
    if has_reasoning:
        if reasoning_index is None:
            insert_at = _find_reasoning_insertion_index(group_items, item_grouper)
            group_items.insert(insert_at, _build_reasoning_item(chat_message["reasoning_content"]))
            return group_items
        group_items[reasoning_index] = _replace_reasoning_item_text(
            group_items[reasoning_index], chat_message["reasoning_content"]
        )
        return group_items
    if reasoning_index is None or not _extract_reasoning_text(group_items[reasoning_index]):
        return group_items
    if (
        reasoning_index == 0
        and len(group_items) > 1
        and item_grouper.can_merge_into_current_assistant_group(group_items[1])
    ):
        group_items[0] = _replace_reasoning_item_text(group_items[0], "")
        return group_items
    return group_items[:reasoning_index] + group_items[reasoning_index + 1:]


def _update_assistant_group_items(
    tracked_message: _TrackedChatMessage, item_grouper: _AssistantItemGrouper
) -> list:
    chat_message = tracked_message.chat_message
    original_snapshot = tracked_message.original_snapshot
    updated_items: list = []
    tool_call_index = 0
    for item in tracked_message.source_items:
        item_kind = _classify_assistant_item(item)
        if item_kind == _ASSISTANT_ITEM_FUNCTION_CALL:
            updated_items.append(
                _update_function_call_item(item, chat_message, original_snapshot, tool_call_index)
            )
            tool_call_index += 1
        elif item_kind == _ASSISTANT_ITEM_TEXT_MESSAGE:
            updated_items.append(_update_assistant_text_item(item, chat_message, original_snapshot))
        else:
            updated_items.append(item)
    return _update_reasoning_in_group_items(
        updated_items, chat_message, original_snapshot, item_grouper
    )


def _update_single_item(tracked_message: _TrackedChatMessage) -> list:
    item = tracked_message.source_items[0]
    chat_message = tracked_message.chat_message
    if (
        chat_message == tracked_message.original_snapshot
        or not isinstance(item, dict)
        or "role" not in chat_message
    ):
        return [item]
    updated_item = dict(item)
    content_field = "output" if chat_message["role"] == "tool" else "content"
    updated_item[content_field] = chat_message.get("content")
    return [updated_item]


class _ResponsesMessageHistory(EditableMessageHistory):
    """A chat-style view of a Responses request's instructions and input."""

    def __init__(self, request_body: dict) -> None:
        self._request_body = request_body
        self._tracked_messages: list[_TrackedChatMessage] = []
        super().__init__([])
        instructions = request_body.get("instructions")
        if isinstance(instructions, str) and instructions:
            self._tracked_messages.append(
                _TrackedChatMessage(
                    {"role": "system", "content": instructions}, [], _SOURCE_INSTRUCTIONS
                )
            )
        input_value = request_body["input"]
        if isinstance(input_value, str):
            self._tracked_messages.append(
                _TrackedChatMessage(
                    {"role": "user", "content": input_value}, [input_value], _SOURCE_STRING_INPUT
                )
            )
        else:
            item_grouper = _AssistantItemGrouper()
            for item in input_value:
                item_grouper.add_input_item(item)
            self._tracked_messages.extend(item_grouper.tracked_messages)
        self.messages.extend(
            tracked_message.chat_message for tracked_message in self._tracked_messages
        )

    def _has_no_edits(self) -> bool:
        return len(self.messages) == len(self._tracked_messages) and all(
            message is tracked_message.chat_message
            and message == tracked_message.original_snapshot
            for message, tracked_message in zip(self.messages, self._tracked_messages)
        )

    def _serialize_tracked_message(
        self, tracked_message: _TrackedChatMessage, item_grouper: _AssistantItemGrouper
    ) -> list:
        chat_message = tracked_message.chat_message
        if tracked_message.source_kind == _SOURCE_INSTRUCTIONS:
            if chat_message != tracked_message.original_snapshot:
                self._request_body["instructions"] = chat_message.get("content")
            return []
        if tracked_message.source_kind == _SOURCE_STRING_INPUT:
            return [{"type": _ITEM_TYPE_MESSAGE, "role": "user", "content": chat_message.get("content")}]
        if tracked_message.source_kind == _SOURCE_SINGLE_ITEM:
            return _update_single_item(tracked_message)
        if chat_message == tracked_message.original_snapshot:
            return list(tracked_message.source_items)
        return _update_assistant_group_items(tracked_message, item_grouper)

    def _sync_removed_instructions(
        self, tracked_by_message_id: dict[int, _TrackedChatMessage]
    ) -> list:
        """Update instructions if its system message was removed or replaced, and return input messages."""
        if (
            not self._tracked_messages
            or self._tracked_messages[0].source_kind != _SOURCE_INSTRUCTIONS
        ):
            return self.messages
        if any(message is self._tracked_messages[0].chat_message for message in self.messages):
            return self.messages
        first_message = self.messages[0] if self.messages else None
        if (
            isinstance(first_message, dict)
            and first_message.get("role") == "system"
            and id(first_message) not in tracked_by_message_id
        ):
            self._request_body["instructions"] = first_message.get("content")
            return self.messages[1:]
        del self._request_body["instructions"]
        return self.messages

    @override
    def apply_to_request(self) -> None:
        if self._has_no_edits():
            return
        tracked_by_message_id = {
            id(tracked_message.chat_message): tracked_message
            for tracked_message in self._tracked_messages
            if isinstance(tracked_message.chat_message, dict)
        }
        item_grouper = _AssistantItemGrouper()
        updated_input_items: list = []
        for message in self._sync_removed_instructions(tracked_by_message_id):
            tracked_message = (
                tracked_by_message_id.get(id(message)) if isinstance(message, dict) else None
            )
            if not isinstance(message, dict):
                items = [message]
            elif tracked_message is None:
                items = _convert_new_chat_message_to_items(message)
            else:
                items = self._serialize_tracked_message(tracked_message, item_grouper)
            for item in items:
                item_grouper.add_input_item(item)
                updated_input_items.append(item)
        self._request_body["input"] = updated_input_items


# ── Stream event parsing ──


class _ResponsesStreamEventParser(StreamEventParser):
    """Parses the SSE events of one Responses API attempt."""

    def __init__(self) -> None:
        self._call_id_by_item_id: dict[str, str] = {}
        self._streamed_function_calls: list[tuple[str, str]] = []
        self._warned_unknown_event_types: set[str] = set()
        self._response_id: str | None = None
        self._event_handlers: dict[str, Callable[[dict], dict | None]] = {
            **dict.fromkeys(_TEXT_DELTA_UPDATE_TYPES, self._parse_text_delta_event),
            _EVENT_OUTPUT_ITEM_ADDED: self._parse_output_item_added_event,
            _EVENT_FUNCTION_CALL_ARGUMENTS_DELTA: self._parse_function_call_arguments_delta_event,
            _EVENT_RESPONSE_COMPLETED: self._parse_terminal_response_event,
            _EVENT_RESPONSE_INCOMPLETE: self._parse_terminal_response_event,
        }

    @override
    def parse_event(self, stream_event: dict) -> ParsedStreamEvent:
        if not isinstance(stream_event, dict):
            return ParsedStreamEvent([], None)
        event_type = stream_event.get("type")
        if not isinstance(event_type, str):
            return ParsedStreamEvent([], None)
        response = stream_event.get("response")
        if self._response_id is None and isinstance(response, dict):
            self._response_id = response.get("id")
        event_handler = self._event_handlers.get(event_type)
        if event_handler is not None:
            update = event_handler(stream_event)
            if update is not None:
                return ParsedStreamEvent([update], 0)
            return ParsedStreamEvent([], None)
        if (
            event_type not in _KNOWN_EVENT_TYPES
            and event_type not in self._warned_unknown_event_types
        ):
            self._warned_unknown_event_types.add(event_type)
            _logger.warning(
                "[ResponsesAPIFormat] Unknown event type %r in response %s",
                event_type,
                self._response_id,
            )
        return ParsedStreamEvent([], None)

    def _parse_text_delta_event(self, stream_event: dict) -> dict:
        return {
            "type": _TEXT_DELTA_UPDATE_TYPES[stream_event["type"]],
            "text": stream_event.get("delta") or "",
        }

    def _parse_output_item_added_event(self, stream_event: dict) -> dict | None:
        item = stream_event.get("item")
        if not isinstance(item, dict) or item.get("type") != _ITEM_TYPE_FUNCTION_CALL:
            return None
        call_id = item.get("call_id") or ""
        tool_name = item.get("name") or ""
        arguments = standardize_json_arguments(item.get("arguments") or "")
        if isinstance(item.get("id"), str):
            self._call_id_by_item_id[item["id"]] = call_id
        self._streamed_function_calls.append((call_id, tool_name))
        return build_tool_call_update(tool_name, arguments, call_id)

    def _parse_function_call_arguments_delta_event(self, stream_event: dict) -> dict:
        arguments = standardize_json_arguments(stream_event.get("delta") or "")
        call_id = self._call_id_by_item_id.get(str(stream_event.get("item_id")), "")
        return build_tool_call_update("", arguments, call_id)

    def _parse_terminal_response_event(self, stream_event: dict) -> dict:
        response = stream_event.get("response")
        if not isinstance(response, dict):
            response = {}
        status = (
            _STATUS_INCOMPLETE
            if stream_event["type"] == _EVENT_RESPONSE_INCOMPLETE
            else response.get("status")
        )
        return {
            "type": "finish_reason",
            "text": self._determine_finish_reason(status, response),
            "tool_call_id_aliases": self._match_streamed_and_final_call_ids(response),
        }

    def _determine_finish_reason(self, status: str | None, response: dict) -> str:
        if status == _STATUS_INCOMPLETE:
            details = response.get("incomplete_details")
            is_filtered = isinstance(details, dict) and details.get("reason") == "content_filter"
            return "content_filter" if is_filtered else "length"
        if status == _STATUS_CANCELLED:
            return "abort"
        return "tool_calls" if self._streamed_function_calls else "stop"

    def _match_streamed_and_final_call_ids(self, response: dict) -> list[tuple[str, str]]:
        final_calls = [
            (item.get("call_id") or "", item.get("name") or "")
            for item in response.get("output") or []
            if isinstance(item, dict) and item.get("type") == _ITEM_TYPE_FUNCTION_CALL
        ]
        is_aligned = len(final_calls) == len(self._streamed_function_calls) and all(
            streamed_call[1] == final_call[1]
            for streamed_call, final_call in zip(self._streamed_function_calls, final_calls)
        )
        if not is_aligned:
            _logger.warning(
                "[ResponsesAPIFormat] Streamed calls %s do not match final calls %s, keeping no aliases",
                [call_id for call_id, _ in self._streamed_function_calls],
                [call_id for call_id, _ in final_calls],
            )
            return []
        return [
            (streamed_call[0], final_call[0])
            for streamed_call, final_call in zip(self._streamed_function_calls, final_calls)
        ]


# ── Streamed output item tracking ──


class _StreamedOutputPart:
    """Tracks one streamed content or summary part inside an output item."""

    def __init__(self, index_field: str, part_index: int, part_type: str) -> None:
        self.index_field = index_field
        self.part_index = part_index
        self.part_type = part_type
        self.accumulated_text = ""
        self.pending_text_done_event_type: str | None = None
        self.pending_part_done_event_type: str | None = None


def _build_output_part_dict(part: _StreamedOutputPart) -> dict:
    """Build the part dict as it appears inside a Responses output item."""
    if part.part_type == "output_text":
        return {"type": "output_text", "text": part.accumulated_text, "annotations": []}
    return {"type": part.part_type, "text": part.accumulated_text}


def _build_part_closing_events(part: _StreamedOutputPart, item_id: str | None) -> list[dict]:
    """Build the done events needed to close an interrupted output part."""
    events: list[dict] = []
    if part.pending_text_done_event_type is not None:
        events.append({
            "type": part.pending_text_done_event_type,
            "item_id": item_id,
            part.index_field: part.part_index,
            "text": part.accumulated_text,
        })
    if part.pending_part_done_event_type is not None:
        events.append({
            "type": part.pending_part_done_event_type,
            "item_id": item_id,
            part.index_field: part.part_index,
            "part": _build_output_part_dict(part),
        })
    return events


class _StreamedOutputItem:
    """Rebuilds one Responses output item from its SSE events."""

    def __init__(self, item_data: dict, output_index: int | None) -> None:
        self.item_data = item_data
        self.output_index = output_index
        self.parts_by_key: dict[tuple[str, int], _StreamedOutputPart] = {}
        self.accumulated_arguments: str = item_data.get("arguments") or ""
        self.has_open_arguments_stream = False
        self.is_completed = False

    def get_or_create_part(
        self, container_field: str, index_field: str, part_index: int, part_type: str
    ) -> _StreamedOutputPart:
        """Return an existing tracked part, or create one on first use."""
        part_key = (container_field, part_index)
        if part_key not in self.parts_by_key:
            self.parts_by_key[part_key] = _StreamedOutputPart(index_field, part_index, part_type)
        return self.parts_by_key[part_key]

    def _list_parts_in_order(
        self, container_field: str | None = None
    ) -> list[_StreamedOutputPart]:
        return [
            self.parts_by_key[part_key]
            for part_key in sorted(self.parts_by_key)
            if container_field in (None, part_key[0])
        ]

    def build_item_snapshot(self, override_status: str | None) -> dict:
        """Return the output item reflecting everything streamed so far."""
        if self.is_completed:
            return copy.deepcopy(self.item_data)
        item = copy.deepcopy(self.item_data)
        for container_field in ("content", "summary"):
            part_dicts = [
                _build_output_part_dict(part)
                for part in self._list_parts_in_order(container_field)
            ]
            if part_dicts:
                item[container_field] = part_dicts
        if item.get("type") == _ITEM_TYPE_FUNCTION_CALL:
            item["arguments"] = self.accumulated_arguments
        if override_status is not None:
            item["status"] = override_status
        return item

    def build_closing_events(self) -> list[dict]:
        """Mark the item incomplete and return the SSE events that close it."""
        item_id = self.item_data.get("id")
        events: list[dict] = []
        for part in self._list_parts_in_order():
            events.extend(_build_part_closing_events(part, item_id))
        if self.has_open_arguments_stream:
            events.append({
                "type": _EVENT_FUNCTION_CALL_ARGUMENTS_DONE,
                "item_id": item_id,
                "name": self.item_data.get("name"),
                "arguments": self.accumulated_arguments,
            })
        self.item_data = self.build_item_snapshot(_STATUS_INCOMPLETE)
        self.is_completed = True
        events.append({"type": _EVENT_OUTPUT_ITEM_DONE, "item": copy.deepcopy(self.item_data)})
        if self.output_index is not None:
            for event in events:
                event["output_index"] = self.output_index
        return events


class _StreamedOutputItemTracker:
    """Tracks every output item across one or more Responses stream attempts."""

    def __init__(self) -> None:
        self.items: list[_StreamedOutputItem] = []
        self._items_by_attempt_and_id: dict[tuple[int, str], _StreamedOutputItem] = {}

    def record_event(self, stream_event: dict, attempt_number: int = 0) -> None:
        """Update tracked output items from the next SSE event."""
        event_type = stream_event.get("type")
        if event_type in (_EVENT_OUTPUT_ITEM_ADDED, _EVENT_OUTPUT_ITEM_DONE):
            self._record_item_lifecycle_event(
                stream_event, event_type == _EVENT_OUTPUT_ITEM_DONE, attempt_number
            )
            return
        output_item = self._items_by_attempt_and_id.get(
            (attempt_number, str(stream_event.get("item_id")))
        )
        if output_item is None:
            return
        if event_type in _TEXT_EVENT_PART_SPECS:
            self._record_text_event(output_item, stream_event, event_type)
        elif event_type in _PART_LIFECYCLE_EVENT_SPECS:
            self._record_part_lifecycle_event(output_item, stream_event, event_type)
        elif event_type == _EVENT_FUNCTION_CALL_ARGUMENTS_DELTA:
            output_item.accumulated_arguments += stream_event.get("delta") or ""
            output_item.has_open_arguments_stream = True
        elif event_type == _EVENT_FUNCTION_CALL_ARGUMENTS_DONE:
            output_item.accumulated_arguments = (
                stream_event.get("arguments") or output_item.accumulated_arguments
            )
            output_item.has_open_arguments_stream = False

    def _record_item_lifecycle_event(
        self, stream_event: dict, is_item_done: bool, attempt_number: int
    ) -> None:
        item_data = stream_event.get("item")
        if not isinstance(item_data, dict):
            return
        item_key = (attempt_number, str(item_data.get("id")))
        output_item = self._items_by_attempt_and_id.get(item_key)
        if output_item is None or (output_item.is_completed and not is_item_done):
            output_item = _StreamedOutputItem(
                copy.deepcopy(item_data), stream_event.get("output_index")
            )
            self.items.append(output_item)
            self._items_by_attempt_and_id[item_key] = output_item
        if is_item_done:
            output_item.item_data = copy.deepcopy(item_data)
            output_item.is_completed = True

    def _record_text_event(
        self, output_item: _StreamedOutputItem, stream_event: dict, event_type: str
    ) -> None:
        part_spec = _TEXT_EVENT_PART_SPECS[event_type]
        part = output_item.get_or_create_part(
            part_spec.container_field,
            part_spec.index_field,
            stream_event.get(part_spec.index_field, 0),
            part_spec.part_type,
        )
        if event_type in _OPENING_TO_CLOSING_EVENT_TYPES:
            part.accumulated_text += stream_event.get("delta") or ""
            part.pending_text_done_event_type = _OPENING_TO_CLOSING_EVENT_TYPES[event_type]
        else:
            part.accumulated_text = stream_event.get("text") or part.accumulated_text
            part.pending_text_done_event_type = None

    def _record_part_lifecycle_event(
        self, output_item: _StreamedOutputItem, stream_event: dict, event_type: str
    ) -> None:
        part_spec = _PART_LIFECYCLE_EVENT_SPECS[event_type]
        part_data = stream_event.get("part")
        part_type = part_spec.part_type
        if isinstance(part_data, dict):
            part_type = part_data.get("type") or part_type
        part = output_item.get_or_create_part(
            part_spec.container_field,
            part_spec.index_field,
            stream_event.get(part_spec.index_field, 0),
            part_type,
        )
        if isinstance(part_data, dict) and part_data.get("text"):
            part.accumulated_text = part_data["text"]
        part.pending_part_done_event_type = _OPENING_TO_CLOSING_EVENT_TYPES.get(event_type)


# ── Multi-attempt stream combining ──


def _parse_typed_sse_json(raw_data: str) -> dict | None:
    """Parse SSE event data as a JSON object with a string type, or return None."""
    try:
        payload = json.loads(raw_data)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("type"), str):
        return None
    return payload


class _ResponsesRetryStreamCombiner(RetryStreamCombiner):
    """Combines approved output from retried attempts into one valid Responses SSE stream."""

    def __init__(self) -> None:
        self._item_tracker = _StreamedOutputItemTracker()
        self._attempt_number = 0
        self._is_passthrough_mode = True
        self._is_retry_pending = False
        self._previous_attempts_item_count = 0
        self._last_sequence_number: int | None = None
        self._uses_sequence_numbers = False
        self._uses_sse_event_lines = False
        self._initial_response_id: str | None = None
        self._pending_line_separator = b""

    @override
    def feed_approved_bytes(self, approved_bytes: bytes) -> bytes:
        if not approved_bytes:
            return b""
        output = bytearray()
        consumed_byte_count = 0
        for event_type, data, _event_id, raw_lines in SSEByteDecoder().feed(approved_bytes):
            raw_event_bytes = b"".join(raw_lines) + b"\n"
            consumed_byte_count += len(raw_event_bytes)
            self._handle_sse_event(
                raw_event_bytes,
                _parse_typed_sse_json(data),
                event_type != "message",
                output,
            )
        trailing_bytes = approved_bytes[consumed_byte_count:]
        if trailing_bytes:
            self._advance_to_next_attempt_if_pending(output)
            self._write_with_pending_separator(output, trailing_bytes)
            self._pending_line_separator = b"\n" if trailing_bytes.endswith(b"\n") else b"\n\n"
        return bytes(output)

    @override
    def begin_retry(self) -> None:
        self._is_passthrough_mode = False
        self._is_retry_pending = True

    @override
    def flush_closing_events(self) -> bytes:
        if not self._is_retry_pending:
            return b""
        output = bytearray()
        self._emit_pending_closing_events(output)
        return bytes(output)

    def _write_with_pending_separator(self, output: bytearray, data: bytes) -> None:
        output.extend(self._pending_line_separator)
        self._pending_line_separator = b""
        output.extend(data)

    def _format_sse_event(self, payload: dict) -> bytes:
        data_line = b"data: " + json.dumps(payload, ensure_ascii=False).encode() + b"\n\n"
        if self._uses_sse_event_lines:
            return b"event: " + payload["type"].encode() + b"\n" + data_line
        return data_line

    def _allocate_sequence_number(self) -> int:
        if self._last_sequence_number is None:
            self._last_sequence_number = 0
        else:
            self._last_sequence_number += 1
        return self._last_sequence_number

    def _emit_pending_closing_events(self, output: bytearray) -> None:
        for output_item in self._item_tracker.items:
            self._emit_item_closing_events(output, output_item)
        self._is_retry_pending = False

    def _emit_item_closing_events(
        self, output: bytearray, output_item: _StreamedOutputItem
    ) -> None:
        if output_item.is_completed:
            return
        for closing_event in output_item.build_closing_events():
            if self._uses_sequence_numbers:
                closing_event["sequence_number"] = self._allocate_sequence_number()
            self._write_with_pending_separator(output, self._format_sse_event(closing_event))

    def _advance_to_next_attempt_if_pending(self, output: bytearray) -> None:
        if not self._is_retry_pending:
            return
        self._emit_pending_closing_events(output)
        self._attempt_number += 1
        self._previous_attempts_item_count = len(self._item_tracker.items)

    def _handle_sse_event(
        self,
        raw_event_bytes: bytes,
        payload: dict | None,
        has_event_line: bool,
        output: bytearray,
    ) -> None:
        if payload is None:
            self._advance_to_next_attempt_if_pending(output)
            self._write_with_pending_separator(output, raw_event_bytes)
            return
        self._uses_sse_event_lines = self._uses_sse_event_lines or has_event_line
        self._uses_sequence_numbers = self._uses_sequence_numbers or "sequence_number" in payload
        event_type = payload["type"]
        if event_type == _EVENT_RESPONSE_CREATED:
            self._advance_to_next_attempt_if_pending(output)
        if self._attempt_number > 0 and event_type in (
            _EVENT_RESPONSE_CREATED,
            _EVENT_RESPONSE_IN_PROGRESS,
        ):
            return
        if self._is_passthrough_mode:
            self._track_passthrough_metadata(payload)
            self._write_with_pending_separator(output, raw_event_bytes)
        else:
            payload = self._rewrite_event_for_combined_stream(payload)
            self._write_with_pending_separator(output, self._format_sse_event(payload))
        self._item_tracker.record_event(payload, self._attempt_number)

    def _track_passthrough_metadata(self, payload: dict) -> None:
        sequence_number = payload.get("sequence_number")
        if isinstance(sequence_number, int):
            self._last_sequence_number = sequence_number
        response = payload.get("response")
        if self._initial_response_id is None and isinstance(response, dict):
            self._initial_response_id = response.get("id")

    def _rewrite_event_for_combined_stream(self, payload: dict) -> dict:
        rewritten = copy.deepcopy(payload)
        if "sequence_number" in rewritten:
            rewritten["sequence_number"] = self._allocate_sequence_number()
        if isinstance(rewritten.get("output_index"), int):
            rewritten["output_index"] += self._previous_attempts_item_count
        response = rewritten.get("response")
        if (
            rewritten["type"] in _TERMINAL_EVENT_TYPES
            and isinstance(response, dict)
            and self._attempt_number > 0
        ):
            if self._initial_response_id is not None:
                response["id"] = self._initial_response_id
            earlier_items = [
                output_item.build_item_snapshot(None)
                for output_item in self._item_tracker.items[: self._previous_attempts_item_count]
            ]
            response["output"] = earlier_items + list(response.get("output") or [])
        return rewritten


# ── Responses API format ──


class ResponsesAPIFormat(ConversationAPIFormat):
    """The OpenAI Responses API format."""

    endpoint_path: str = "/v1/responses"
    history_field_name: str = "input"

    @override
    def convert_request_overrides(self, overrides: dict) -> RequestOverrides:
        translated_keys = ("max_tokens", "max_completion_tokens", "reasoning_effort")
        always_apply = {
            key: value for key, value in overrides.items() if key not in translated_keys
        }
        apply_if_missing: dict = {}
        if "max_completion_tokens" in overrides:
            always_apply["max_output_tokens"] = overrides["max_completion_tokens"]
        elif "max_tokens" in overrides:
            always_apply.pop("max_output_tokens", None)
            apply_if_missing["max_output_tokens"] = overrides["max_tokens"]
        if "reasoning_effort" in overrides:
            reasoning = always_apply.get("reasoning")
            reasoning = dict(reasoning) if isinstance(reasoning, dict) else {}
            reasoning["effort"] = overrides["reasoning_effort"]
            always_apply["reasoning"] = reasoning
        return RequestOverrides(always_apply, apply_if_missing)

    @override
    def enable_streaming(self, request_body: dict) -> None:
        request_body["stream"] = True

    @override
    def extract_message_history(self, request_body: dict) -> EditableMessageHistory | None:
        if not isinstance(request_body, dict) or not isinstance(
            request_body.get(self.history_field_name), (str, list)
        ):
            return None
        return _ResponsesMessageHistory(request_body)

    @override
    def create_stream_event_parser(self) -> StreamEventParser:
        return _ResponsesStreamEventParser()

    @override
    def create_retry_stream_combiner(self) -> RetryStreamCombiner:
        return _ResponsesRetryStreamCombiner()

    @override
    def extract_generated_texts(self, stream_event: dict) -> list[str]:
        event_type = stream_event.get("type")
        if event_type in _TEXT_DELTA_UPDATE_TYPES or event_type == _EVENT_FUNCTION_CALL_ARGUMENTS_DELTA:
            return [stream_event.get("delta") or ""]
        item = stream_event.get("item")
        if (
            event_type == _EVENT_OUTPUT_ITEM_ADDED
            and isinstance(item, dict)
            and item.get("type") == _ITEM_TYPE_FUNCTION_CALL
        ):
            return [(item.get("name") or "") + (item.get("arguments") or "")]
        return []

    @override
    def extract_token_usage(self, stream_event: dict) -> TokenUsage | None:
        response = stream_event.get("response")
        if stream_event.get("type") not in _TERMINAL_EVENT_TYPES or not isinstance(response, dict):
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
    def assemble_non_streaming_response(self, stream_events: list[dict]) -> bytes | None:
        terminal_response: dict | None = None
        latest_response_envelope: dict | None = None
        item_tracker = _StreamedOutputItemTracker()
        for stream_event in stream_events:
            if not isinstance(stream_event, dict):
                continue
            item_tracker.record_event(stream_event)
            response = stream_event.get("response")
            if isinstance(response, dict):
                latest_response_envelope = response
                if stream_event.get("type") in _TERMINAL_EVENT_TYPES:
                    terminal_response = response
        if terminal_response is not None:
            return json.dumps(terminal_response).encode()
        if latest_response_envelope is None and not item_tracker.items:
            return None
        assembled_response = (
            copy.deepcopy(latest_response_envelope) if latest_response_envelope is not None else {}
        )
        assembled_response["output"] = [
            output_item.build_item_snapshot(None) for output_item in item_tracker.items
        ]
        return json.dumps(assembled_response).encode()
