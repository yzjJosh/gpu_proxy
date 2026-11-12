"""How each OpenAI API spells its requests, streams and replies."""

import abc
import json
from typing import NamedTuple, override

from app.plugins.utils import assemble_non_streaming_response

__all__ = [
    "ChatCompletionsDialect",
    "ChatProjection",
    "CompletionsDialect",
    "ConversationDialect",
    "DeltaDecoder",
    "OpenAIDialect",
    "OutputSplicer",
    "TokenUsage",
    "TranslatedOverrides",
    "build_tool_call_piece",
    "canonicalize_arguments",
    "find_by_prefix",
    "find_exact",
    "find_prefix",
]


class TokenUsage(NamedTuple):
    """The token counts one reply reports."""

    input: int
    output: int
    total: int


class TranslatedOverrides(NamedTuple):
    """Configured overrides, named the way one API names them.

    Attributes:
        forced: Keys merged into every request.
        yielding: Keys merged only when the request does not set them.
    """

    forced: dict
    yielding: dict


class ChatProjection(abc.ABC):
    """A chat style view of one request's history that can be written back.

    Attributes:
        messages: The history as chat messages. Changes made to it reach the
            request when write_back is called.
    """

    def __init__(self, messages: list) -> None:
        self.messages = messages

    @abc.abstractmethod
    def write_back(self) -> None:
        """Put the changes made to messages back into the request body."""


class DeltaDecoder(abc.ABC):
    """Reads the stream events of one upstream attempt."""

    @abc.abstractmethod
    def normalize(self, event: dict) -> tuple[list[dict], int | None]:
        """Read the pieces one stream event carries.

        Args:
            event: One parsed stream event.

        Returns:
            The pieces in the order they must be checked, with finish reasons
            last, and the index of the piece that carries the event's bytes.
            An event with no piece gives an empty list and None.
        """


class OutputSplicer(abc.ABC):
    """Turns the approved output of every attempt into one valid stream."""

    @abc.abstractmethod
    def splice(self, approved: bytes) -> bytes:
        """Give the bytes to send in place of approved bytes.

        Args:
            approved: One run of approved bytes.

        Returns:
            The bytes to send in their place.
        """

    @abc.abstractmethod
    def retry(self) -> None:
        """Mark that the current attempt was cut off and a new one follows."""

    @abc.abstractmethod
    def finish(self) -> bytes:
        """Close what is still open once no more bytes come.

        Returns:
            The closing bytes still owed, or empty bytes.
        """


class OpenAIDialect(abc.ABC):
    """How one OpenAI API spells requests, streams, usage and full replies.

    Attributes:
        path: The URL path of the API.
    """

    path: str

    def translate_overrides(self, overrides: dict) -> TranslatedOverrides:
        """Rename chat style override keys to the names this API uses.

        Args:
            overrides: The configured overrides.

        Returns:
            The overrides to apply on this API.
        """
        return TranslatedOverrides(dict(overrides), {})

    @abc.abstractmethod
    def force_stream(self, body: dict) -> None:
        """Ask the upstream to stream its reply, changing the body in place.

        Args:
            body: The parsed request body.
        """

    @abc.abstractmethod
    def prompt_text(self, body: dict) -> str:
        """Get the request text whose tokens count as the prompt.

        Args:
            body: The parsed request body.

        Returns:
            The prompt text.

        Raises:
            ValueError: If the request history is not a list and cannot be read as one.
        """

    @abc.abstractmethod
    def generated_texts(self, event: dict) -> list[str]:
        """Get the generated text one stream event carries.

        Args:
            event: One parsed stream event.

        Returns:
            One text per generated part of the event, or an empty list.
        """

    @abc.abstractmethod
    def usage(self, event: dict) -> TokenUsage | None:
        """Get the token usage one stream event reports.

        Args:
            event: One parsed stream event.

        Returns:
            The usage, or None if the event reports none.
        """

    @abc.abstractmethod
    def assemble_non_streaming(self, events: list[dict]) -> bytes | None:
        """Build the full reply a client that did not stream expects.

        Args:
            events: Every parsed stream event, in order.

        Returns:
            The encoded reply, or None to pass the raw stream through.
        """


class ConversationDialect(OpenAIDialect):
    """An OpenAI API whose requests carry a message history.

    Attributes:
        history_key: The body field that holds the history.
    """

    history_key: str

    @abc.abstractmethod
    def project(self, body: dict) -> ChatProjection | None:
        """View the request history as chat messages.

        Args:
            body: The parsed request body.

        Returns:
            The view, or None if the body holds no usable history.
        """

    @abc.abstractmethod
    def delta_decoder(self) -> DeltaDecoder:
        """Make a reader for the stream of one upstream attempt.

        Returns:
            A new reader.
        """

    @abc.abstractmethod
    def output_splicer(self) -> OutputSplicer:
        """Make a splicer for the output of one guard rail session.

        Returns:
            A new splicer.
        """

    def history_value[V](self, body: dict[str, V]) -> V | None:
        """Get the history field as the request sent it.

        Args:
            body: The parsed request body.

        Returns:
            The field's value, or None if it is not set.
        """
        return body.get(self.history_key)

    @override
    def prompt_text(self, body: dict) -> str:
        projection = self.project(body)
        messages = projection.messages if projection is not None else self.history_value(body) or []
        if not isinstance(messages, list):
            raise ValueError(f"{self.history_key} is not a list")
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


def _delta_reasoning(delta: dict) -> str | None:
    """Give the reasoning text of a chat delta, or None if it has none."""
    return delta.get("reasoning") or delta.get("reasoning_content")


def _tool_call_fields(tool_call: dict) -> tuple[str, str]:
    """Give the name and arguments of a chat tool call, empty when missing."""
    function = tool_call.get("function") or {}
    return function.get("name") or "", function.get("arguments") or ""


class _ChoicesDialect(OpenAIDialect):
    """Reads streams made of choices chunks."""

    @override
    def force_stream(self, body: dict) -> None:
        body["stream"] = True
        body.setdefault("stream_options", {})["include_usage"] = True

    @override
    def generated_texts(self, event: dict) -> list[str]:
        texts: list[str] = []
        for choice in event.get("choices") or []:
            delta = choice.get("delta") or {}
            # A field the delta is not using can be null instead of missing.
            text = delta.get("content") or ""
            text += _delta_reasoning(delta) or ""
            for tool_call in delta.get("tool_calls") or []:
                name, arguments = _tool_call_fields(tool_call)
                text += name + arguments
            # Legacy completion chunks carry their text on the choice.
            text += choice.get("text", "") or ""
            texts.append(text)
        return texts

    @override
    def usage(self, event: dict) -> TokenUsage | None:
        if "usage" not in event:
            return None
        usage = event["usage"]
        if not isinstance(usage, dict):
            return None
        return TokenUsage(
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
            usage.get("total_tokens", 0),
        )


class CompletionsDialect(_ChoicesDialect):
    """The legacy completions API."""

    path: str = "/v1/completions"

    @override
    def prompt_text(self, body: dict) -> str:
        prompt = body.get("prompt", "")
        if isinstance(prompt, list):
            prompt = "".join(prompt)
        return prompt

    @override
    def assemble_non_streaming(self, events: list[dict]) -> bytes | None:
        return assemble_non_streaming_response(events, is_chat=False)


def canonicalize_arguments(arguments: str) -> str:
    """Write tool call arguments in one fixed JSON form when they parse.

    Args:
        arguments: The arguments text.

    Returns:
        The rewritten text, or the text as given if it is not JSON.
    """
    if arguments:
        try:
            return json.dumps(json.loads(arguments))
        except (json.JSONDecodeError, TypeError):
            pass
    return arguments


def build_tool_call_piece(name: str, arguments: str, call_id: str) -> dict:
    """Make the piece for one tool call chunk.

    Args:
        name: The tool name in this chunk.
        arguments: The arguments text in this chunk.
        call_id: The id of the call.

    Returns:
        The tool call piece.
    """
    return {
        "type": "tool_call",
        "text": name + arguments,
        "tool_call_id": call_id,
        "function": {"name": name, "arguments": arguments},
    }


def _chat_content_piece(delta: dict) -> dict | None:
    content = delta.get("content")
    if content is None:
        return None
    return {"type": "content", "text": content}


def _chat_reasoning_piece(delta: dict) -> dict | None:
    reasoning = _delta_reasoning(delta)
    if reasoning is None:
        return None
    return {"type": "reasoning", "text": reasoning}


def _chat_tool_call_piece(tool_call: dict) -> dict | None:
    name, arguments = _tool_call_fields(tool_call)
    arguments = canonicalize_arguments(arguments)
    tool_call_id = tool_call.get("id") or ""
    if not name and not arguments and not tool_call_id:
        return None
    return build_tool_call_piece(name, arguments, tool_call_id)


def _chat_finish_piece(choice: dict) -> dict | None:
    finish_reason = choice.get("finish_reason")
    if finish_reason is None:
        return None
    return {"type": "finish_reason", "text": finish_reason}


class _ChatDeltaDecoder(DeltaDecoder):
    """Reads every piece of a chat chunk, finish reasons last."""

    @override
    def normalize(self, event: dict) -> tuple[list[dict], int | None]:
        deltas: list[dict] = []
        finish_pieces: list[dict] = []
        carrier: dict | None = None
        for choice in event.get("choices") or []:
            delta = choice.get("delta") or {}
            reasoning = _chat_reasoning_piece(delta)
            content = _chat_content_piece(delta)
            tool_calls = [
                piece
                for piece in map(_chat_tool_call_piece, delta.get("tool_calls") or [])
                if piece is not None
            ]
            finish = _chat_finish_piece(choice)
            deltas.extend(piece for piece in (reasoning, content) if piece is not None)
            deltas.extend(tool_calls)
            if finish is not None:
                finish_pieces.append(finish)
            if carrier is None:
                carrier = next(
                    (
                        piece
                        for piece in (content, reasoning, *tool_calls[:1], finish)
                        if piece is not None
                    ),
                    None,
                )
        deltas.extend(finish_pieces)
        if carrier is None:
            return deltas, None
        return deltas, next(index for index, piece in enumerate(deltas) if piece is carrier)


class _IdentitySplicer(OutputSplicer):
    """Sends approved bytes as they are."""

    @override
    def splice(self, approved: bytes) -> bytes:
        return approved

    @override
    def retry(self) -> None:
        return None

    @override
    def finish(self) -> bytes:
        return b""


class _ChatProjection(ChatProjection):
    """The chat history itself."""

    @override
    def write_back(self) -> None:
        return None


class ChatCompletionsDialect(_ChoicesDialect, ConversationDialect):
    """The chat completions API."""

    path: str = "/v1/chat/completions"
    history_key: str = "messages"

    @override
    def project(self, body: dict) -> ChatProjection | None:
        if not isinstance(body, dict):
            return None
        messages = body.get(self.history_key)
        if not isinstance(messages, list):
            return None
        return _ChatProjection(messages)

    @override
    def delta_decoder(self) -> DeltaDecoder:
        return _ChatDeltaDecoder()

    @override
    def output_splicer(self) -> OutputSplicer:
        return _IdentitySplicer()

    @override
    def assemble_non_streaming(self, events: list[dict]) -> bytes | None:
        return assemble_non_streaming_response(events, is_chat=True)


def find_by_prefix[D: OpenAIDialect](dialects: tuple[D, ...], path: str) -> D | None:
    """Find the dialect whose path starts the request path.

    Args:
        dialects: The dialects to choose from.
        path: The request path.

    Returns:
        The matching dialect, or None.
    """
    prefix = find_prefix(tuple(dialect.path for dialect in dialects), path)
    return None if prefix is None else find_exact(dialects, prefix)


def find_exact[D: OpenAIDialect](dialects: tuple[D, ...], path: str) -> D | None:
    """Find the dialect whose path is the request path.

    Args:
        dialects: The dialects to choose from.
        path: The request path.

    Returns:
        The matching dialect, or None.
    """
    return next((dialect for dialect in dialects if dialect.path == path), None)


def find_prefix(prefixes: tuple[str, ...], path: str) -> str | None:
    """Find the first prefix that starts the request path.

    Args:
        prefixes: The prefixes to choose from.
        path: The request path.

    Returns:
        The matching prefix, or None.
    """
    return next((prefix for prefix in prefixes if path.startswith(prefix)), None)
