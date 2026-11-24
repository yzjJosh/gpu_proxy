"""Request, stream, and response format for the OpenAI Chat Completions API."""

from typing import override

from app.plugins.openai_api_formats.base import (
    ChoicesBasedAPIFormat,
    ConversationAPIFormat,
    EditableMessageHistory,
    ParsedStreamEvent,
    RetryStreamCombiner,
    StreamEventParser,
    build_tool_call_update,
    extract_chat_delta_reasoning,
    extract_chat_tool_call_fields,
)
from app.plugins.utils import assemble_non_streaming_response

__all__ = ["ChatCompletionsAPIFormat"]


def _build_chat_content_update(delta: dict) -> dict | None:
    content = delta.get("content")
    if content is None:
        return None
    return {"type": "content", "text": content}


def _build_chat_reasoning_update(delta: dict) -> dict | None:
    reasoning = extract_chat_delta_reasoning(delta)
    if reasoning is None:
        return None
    return {"type": "reasoning", "text": reasoning}


def _build_chat_tool_call_update(tool_call: dict) -> dict | None:
    tool_name, arguments = extract_chat_tool_call_fields(tool_call)
    tool_call_id = tool_call.get("id") or ""
    if not tool_name and not arguments and not tool_call_id:
        return None
    return build_tool_call_update(tool_name, arguments, tool_call_id)


def _build_chat_finish_reason_update(choice: dict) -> dict | None:
    finish_reason = choice.get("finish_reason")
    if finish_reason is None:
        return None
    return {"type": "finish_reason", "text": finish_reason}


class _ChatCompletionsStreamEventParser(StreamEventParser):
    """Extracts every update from a chat completions chunk, with finish reasons last."""

    @override
    def parse_event(self, stream_event: dict) -> ParsedStreamEvent:
        updates: list[dict] = []
        finish_reason_updates: list[dict] = []
        raw_bytes_owner: dict | None = None
        for choice in stream_event.get("choices") or []:
            delta = choice.get("delta") or {}
            reasoning_update = _build_chat_reasoning_update(delta)
            content_update = _build_chat_content_update(delta)
            tool_call_updates = [
                update
                for update in map(_build_chat_tool_call_update, delta.get("tool_calls") or [])
                if update is not None
            ]
            finish_update = _build_chat_finish_reason_update(choice)
            updates.extend(update for update in (reasoning_update, content_update) if update is not None)
            updates.extend(tool_call_updates)
            if finish_update is not None:
                finish_reason_updates.append(finish_update)
            if raw_bytes_owner is None:
                raw_bytes_owner = next(
                    (
                        update
                        for update in (content_update, reasoning_update, *tool_call_updates[:1], finish_update)
                        if update is not None
                    ),
                    None,
                )
        updates.extend(finish_reason_updates)
        if raw_bytes_owner is None:
            return ParsedStreamEvent(updates, None)
        owner_index = next(index for index, update in enumerate(updates) if update is raw_bytes_owner)
        return ParsedStreamEvent(updates, owner_index)


class _PassthroughRetryStreamCombiner(RetryStreamCombiner):
    """Forwards approved bytes unchanged across retry attempts."""

    @override
    def feed_approved_bytes(self, approved_bytes: bytes) -> bytes:
        return approved_bytes

    @override
    def begin_retry(self) -> None:
        return None

    @override
    def flush_closing_events(self) -> bytes:
        return b""


class _InPlaceChatMessageHistory(EditableMessageHistory):
    """Exposes a chat completions messages list directly in place."""

    @override
    def apply_to_request(self) -> None:
        return None


class ChatCompletionsAPIFormat(ChoicesBasedAPIFormat, ConversationAPIFormat):
    """The OpenAI Chat Completions API format."""

    endpoint_path: str = "/v1/chat/completions"
    history_field_name: str = "messages"

    @override
    def extract_message_history(self, request_body: dict) -> EditableMessageHistory | None:
        if not isinstance(request_body, dict):
            return None
        messages = request_body.get(self.history_field_name)
        if not isinstance(messages, list):
            return None
        return _InPlaceChatMessageHistory(messages)

    @override
    def create_stream_event_parser(self) -> StreamEventParser:
        return _ChatCompletionsStreamEventParser()

    @override
    def create_retry_stream_combiner(self) -> RetryStreamCombiner:
        return _PassthroughRetryStreamCombiner()

    @override
    def assemble_non_streaming_response(self, stream_events: list[dict]) -> bytes | None:
        return assemble_non_streaming_response(stream_events, is_chat=True)
