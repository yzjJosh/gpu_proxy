"""Shared base classes, data types, and helpers for OpenAI API formats."""

import abc
import json
from typing import NamedTuple, override

__all__ = [
    "ChoicesBasedAPIFormat",
    "ConversationAPIFormat",
    "EditableMessageHistory",
    "OpenAIAPIFormat",
    "ParsedStreamEvent",
    "RequestOverrides",
    "RetryStreamCombiner",
    "StreamEventParser",
    "TokenUsage",
    "build_tool_call_update",
    "extract_chat_delta_reasoning",
    "extract_chat_tool_call_fields",
    "find_api_format_by_exact_path",
    "find_api_format_by_path_prefix",
    "find_matching_path_prefix",
    "standardize_json_arguments",
]


class TokenUsage(NamedTuple):
    """Token counts reported by one OpenAI response."""

    input_tokens: int
    output_tokens: int
    total_tokens: int


class RequestOverrides(NamedTuple):
    """Request override fields prepared for one OpenAI API format.

    Attributes:
        always_apply: Fields merged into every request.
        apply_if_missing: Fields merged only when the request omits them.
    """

    always_apply: dict
    apply_if_missing: dict


class ParsedStreamEvent(NamedTuple):
    """Standardized updates extracted from one upstream SSE event.

    Attributes:
        updates: Extracted updates in validation order, with finish reasons last.
        raw_bytes_owner_index: Index of the update that carries the raw SSE bytes,
            or None if no updates were extracted.
    """

    updates: list[dict]
    raw_bytes_owner_index: int | None


class EditableMessageHistory(abc.ABC):
    """A chat-style view of one request's message history that can be written back.

    Attributes:
        messages: The conversation history as chat-style message dicts.
            Changes made here reach the request body when apply_to_request is called.
    """

    def __init__(self, messages: list) -> None:
        self.messages = messages

    @abc.abstractmethod
    def apply_to_request(self) -> None:
        """Write changes made to messages back into the request body."""


class StreamEventParser(abc.ABC):
    """Parses upstream SSE events for one guard-rail attempt."""

    @abc.abstractmethod
    def parse_event(self, stream_event: dict) -> ParsedStreamEvent:
        """Extract standardized updates from one parsed stream event.

        Args:
            stream_event: One parsed JSON event from the upstream SSE stream.

        Returns:
            The extracted updates in validation order, with finish reasons last,
            and the index of the update that carries the event's raw bytes.
        """


class RetryStreamCombiner(abc.ABC):
    """Combines approved output from one or more retry attempts into one valid stream."""

    @abc.abstractmethod
    def feed_approved_bytes(self, approved_bytes: bytes) -> bytes:
        """Accept approved raw SSE bytes and return the bytes to send to the client.

        Args:
            approved_bytes: One approved slice of upstream SSE bytes.

        Returns:
            The SSE bytes to forward to the client.
        """

    @abc.abstractmethod
    def begin_retry(self) -> None:
        """Mark that the current attempt was cut off and a retry attempt will follow."""

    @abc.abstractmethod
    def flush_closing_events(self) -> bytes:
        """Return any closing SSE events still needed at the end of the stream.

        Returns:
            The remaining closing SSE bytes, or empty bytes.
        """


class OpenAIAPIFormat(abc.ABC):
    """Reads and writes requests, streams, usage, and responses for one OpenAI API.

    Attributes:
        endpoint_path: The URL path of the API endpoint.
    """

    endpoint_path: str

    def convert_request_overrides(self, overrides: dict) -> RequestOverrides:
        """Convert chat-style override keys into the field names this API uses.

        Args:
            overrides: The configured request overrides.

        Returns:
            The overrides prepared for this API.
        """
        return RequestOverrides(dict(overrides), {})

    @abc.abstractmethod
    def enable_streaming(self, request_body: dict) -> None:
        """Modify the request body in place so the upstream streams its response.

        Args:
            request_body: The parsed JSON request body.
        """

    @abc.abstractmethod
    def extract_prompt_text(self, request_body: dict) -> str:
        """Extract the request text counted as prompt tokens.

        Args:
            request_body: The parsed JSON request body.

        Returns:
            The combined prompt text.

        Raises:
            ValueError: If the message history field is not a list.
        """

    @abc.abstractmethod
    def extract_generated_texts(self, stream_event: dict) -> list[str]:
        """Extract newly generated text chunks from one stream event.

        Args:
            stream_event: One parsed JSON event from the upstream SSE stream.

        Returns:
            One text string per generated choice or part, or an empty list.
        """

    @abc.abstractmethod
    def extract_token_usage(self, stream_event: dict) -> TokenUsage | None:
        """Extract token usage counts from one stream event.

        Args:
            stream_event: One parsed JSON event from the upstream SSE stream.

        Returns:
            The token usage counts, or None if the event has none.
        """

    @abc.abstractmethod
    def assemble_non_streaming_response(self, stream_events: list[dict]) -> bytes | None:
        """Build a full JSON response body for a client that did not request streaming.

        Args:
            stream_events: All parsed stream events in arrival order.

        Returns:
            The JSON-encoded response bytes, or None to pass raw bytes through.
        """


class ConversationAPIFormat(OpenAIAPIFormat):
    """An OpenAI API format whose requests carry a multi-turn message history.

    Attributes:
        history_field_name: The request body key that stores the message history.
    """

    history_field_name: str

    @abc.abstractmethod
    def extract_message_history(self, request_body: dict) -> EditableMessageHistory | None:
        """Return an editable chat-style view of the request's message history.

        Args:
            request_body: The parsed JSON request body.

        Returns:
            An editable message history, or None if the body has no usable history.
        """

    @abc.abstractmethod
    def create_stream_event_parser(self) -> StreamEventParser:
        """Create a fresh stream event parser for one upstream attempt.

        Returns:
            A new stream event parser.
        """

    @abc.abstractmethod
    def create_retry_stream_combiner(self) -> RetryStreamCombiner:
        """Create a fresh retry stream combiner for one guard-rail session.

        Returns:
            A new retry stream combiner.
        """

    def get_raw_history_field[V](self, request_body: dict[str, V]) -> V | None:
        """Return the raw history field value from the request body.

        Args:
            request_body: The parsed JSON request body.

        Returns:
            The raw field value, or None if it is absent.
        """
        return request_body.get(self.history_field_name)

    @override
    def extract_prompt_text(self, request_body: dict) -> str:
        message_history = self.extract_message_history(request_body)
        messages = (
            message_history.messages
            if message_history is not None
            else self.get_raw_history_field(request_body) or []
        )
        if not isinstance(messages, list):
            raise ValueError(f"{self.history_field_name} is not a list")
        text = ""
        for message in messages:
            content = message.get("content") or ""
            if isinstance(content, str):
                text += content
            elif isinstance(content, list):
                for block in content:
                    if block.get("type") == "text":
                        text += block.get("text", "")
            text += "\n"
        return text


def extract_chat_delta_reasoning(delta: dict) -> str | None:
    """Return the reasoning text from a chat delta, or None if absent.

    Args:
        delta: One chat choice delta dict.

    Returns:
        The reasoning string, or None.
    """
    return delta.get("reasoning") or delta.get("reasoning_content")


def extract_chat_tool_call_fields(tool_call: dict) -> tuple[str, str]:
    """Return the tool name and arguments from a chat tool call fragment.

    Args:
        tool_call: One tool call entry inside a chat delta.

    Returns:
        A pair of tool name and arguments strings, empty when missing.
    """
    function = tool_call.get("function") or {}
    return function.get("name") or "", function.get("arguments") or ""


class ChoicesBasedAPIFormat(OpenAIAPIFormat):
    """Shared base class for OpenAI APIs whose stream chunks carry a choices list."""

    @override
    def enable_streaming(self, request_body: dict) -> None:
        request_body["stream"] = True
        request_body.setdefault("stream_options", {})["include_usage"] = True

    @override
    def extract_generated_texts(self, stream_event: dict) -> list[str]:
        texts: list[str] = []
        for choice in stream_event.get("choices") or []:
            delta = choice.get("delta") or {}
            # Unused delta fields may be null instead of missing.
            text = delta.get("content") or ""
            text += extract_chat_delta_reasoning(delta) or ""
            for tool_call in delta.get("tool_calls") or []:
                tool_name, arguments = extract_chat_tool_call_fields(tool_call)
                text += tool_name + arguments
            # Legacy completion chunks carry their text directly on the choice.
            text += choice.get("text", "") or ""
            texts.append(text)
        return texts

    @override
    def extract_token_usage(self, stream_event: dict) -> TokenUsage | None:
        if "usage" not in stream_event:
            return None
        usage = stream_event["usage"]
        if not isinstance(usage, dict):
            return None
        return TokenUsage(
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
            usage.get("total_tokens", 0),
        )


def standardize_json_arguments(arguments: str) -> str:
    """Reformat valid JSON tool arguments into a standard JSON string.

    Args:
        arguments: The raw tool call arguments text.

    Returns:
        The standardized JSON string, or the original text if it is not valid JSON.
    """
    if arguments:
        try:
            return json.dumps(json.loads(arguments))
        except (json.JSONDecodeError, TypeError):
            pass
    return arguments


def build_tool_call_update(tool_name: str, arguments: str, tool_call_id: str) -> dict:
    """Create one standardized tool call update dict for guard-rail checks.

    Args:
        tool_name: The function name in this chunk.
        arguments: The function arguments text in this chunk.
        tool_call_id: The identifier of the tool call.

    Returns:
        A standardized tool call update dict.
    """
    return {
        "type": "tool_call",
        "text": tool_name + arguments,
        "tool_call_id": tool_call_id,
        "function": {"name": tool_name, "arguments": arguments},
    }


def find_api_format_by_path_prefix[F: OpenAIAPIFormat](
    api_formats: tuple[F, ...], request_path: str
) -> F | None:
    """Find the API format whose endpoint path is a prefix of request_path.

    Args:
        api_formats: The candidate API formats.
        request_path: The incoming HTTP request path.

    Returns:
        The matching API format, or None.
    """
    matching_prefix = find_matching_path_prefix(
        tuple(api_format.endpoint_path for api_format in api_formats), request_path
    )
    if matching_prefix is None:
        return None
    return find_api_format_by_exact_path(api_formats, matching_prefix)


def find_api_format_by_exact_path[F: OpenAIAPIFormat](
    api_formats: tuple[F, ...], request_path: str
) -> F | None:
    """Find the API format whose endpoint path equals request_path.

    Args:
        api_formats: The candidate API formats.
        request_path: The incoming HTTP request path.

    Returns:
        The matching API format, or None.
    """
    return next(
        (api_format for api_format in api_formats if api_format.endpoint_path == request_path),
        None,
    )


def find_matching_path_prefix(path_prefixes: tuple[str, ...], request_path: str) -> str | None:
    """Find the first path prefix that starts request_path.

    Args:
        path_prefixes: The candidate URL path prefixes.
        request_path: The incoming HTTP request path.

    Returns:
        The first matching prefix, or None.
    """
    return next((prefix for prefix in path_prefixes if request_path.startswith(prefix)), None)
