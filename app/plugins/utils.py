import json
import re
from dataclasses import dataclass

from app.config import ConfigError

__all__ = [
    "SSEByteDecoder",
    "ToolCallExclusion",
    "assemble_non_streaming_response",
    "decode_json_object",
    "is_tool_call_excluded",
    "parse_tool_call_exclusions",
]


_BASE_EXCLUSION_KEYS: frozenset[str] = frozenset({"tool_name_regex", "tool_args_regex"})


@dataclass
class ToolCallExclusion:
    """A compiled tool call exclusion entry.

    Both patterns may be present. A tool call matches only when every
    pattern that is set matches.
    """

    name_re: re.Pattern[str] | None
    args_re: re.Pattern[str] | None

    def matches(self, name: str, args: str) -> bool:
        """Tell whether a tool call matches this entry.

        Args:
            name: The tool name.
            args: The tool arguments text.

        Returns:
            True when every set pattern matches.
        """
        if self.name_re is not None and not self.name_re.search(name):
            return False
        if self.args_re is not None and not self.args_re.search(args):
            return False
        return True


def parse_tool_call_exclusions(
    raw: list,
    log_prefix: str,
    config_key: str,
    *,
    extra_allowed_keys: frozenset[str] = frozenset(),
) -> list[ToolCallExclusion]:
    """Parse a list of exclusion dicts into compiled patterns.

    Args:
        raw: A list of dicts with optional tool_name_regex and
            tool_args_regex keys.
        log_prefix: Prefix naming the plugin in error messages.
        config_key: The config key holding these exclusions.
        extra_allowed_keys: Additional entry keys accepted for this caller.

    Returns:
        The compiled exclusions.

    Raises:
        ConfigError: If raw is not a list, an entry is not a dict, an entry
            has an unrecognized key, an entry defines neither pattern, or a
            pattern is not a valid string regex.
    """
    if not isinstance(raw, list):
        raise ConfigError(
            f"{log_prefix}: {config_key} must be a list, "
            f"got {type(raw)}"
        )
    allowed_keys = _BASE_EXCLUSION_KEYS | extra_allowed_keys
    exclusions: list[ToolCallExclusion] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ConfigError(
                f"{log_prefix}: {config_key}[{index}] must be a dict, "
                f"got {type(entry)}"
            )
        unknown_keys = sorted(repr(key) for key in set(entry) - allowed_keys)
        if unknown_keys:
            joined = ", ".join(unknown_keys)
            raise ConfigError(
                f"{log_prefix}: {config_key}[{index}] has unrecognized "
                f"key(s): {joined}"
            )
        name_re = _compile_optional(
            entry.get("tool_name_regex"), "tool_name_regex", log_prefix
        )
        args_re = _compile_optional(
            entry.get("tool_args_regex"), "tool_args_regex", log_prefix
        )
        if name_re is None and args_re is None:
            raise ConfigError(
                f"{log_prefix}: {config_key}[{index}] must define "
                f"tool_name_regex or tool_args_regex"
            )
        exclusions.append(ToolCallExclusion(name_re, args_re))
    return exclusions


def is_tool_call_excluded(name: str, args: str, exclusions: list[ToolCallExclusion]) -> bool:
    """Return True if any exclusion entry matches."""
    return any(entry.matches(name, args) for entry in exclusions)


def _compile_optional(pattern: str | None, field: str, log_prefix: str) -> re.Pattern[str] | None:
    if pattern is None:
        return None
    if not isinstance(pattern, str):
        raise ConfigError(
            f"{log_prefix}: {field} must be a string regex, got {type(pattern)}"
        )
    try:
        return re.compile(pattern)
    except re.error as e:
        raise ConfigError(f"{log_prefix}: {field} is not a valid regex: {e}") from e


def decode_json_object(body: bytes) -> dict | None:
    """Parse a request or response body as a JSON object.

    Args:
        body: The raw bytes to parse.

    Returns:
        The decoded mapping, or None if the bytes are syntactically invalid
        JSON, are of a type the parser cannot accept, or decode to something
        other than an object.

    Raises:
        UnicodeDecodeError: If the bytes cannot be decoded under the encoding
            the parser detects — UTF-8 unless a BOM or null-byte pattern marks
            the body as UTF-16/32, in which case the error names that codec.
            Decoding happens before parsing, so a decode failure escapes
            rather than yielding None.
    """
    try:
        parsed = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def assemble_non_streaming_response(events: list[dict | None], is_chat: bool = True) -> bytes | None:
    """Assemble a complete OpenAI JSON response from accumulated SSE event dicts.

    Parses content/reasoning/tool_calls from deltas, and returns a
    properly-formatted non-streaming JSON response.

    Args:
        events: List of parsed JSON event dicts from OpenAI SSE stream
            (None entries are skipped).

    Returns:
        JSON-encoded bytes of the assembled response, or None if no events
        were found (indicating the raw bytes should be passed through as-is).

    Example:
        >>> events = [{"id": "abc", "choices": [{"delta": {"content": "hello"}}]}]
        >>> import json; json.loads(assemble_non_streaming_response(events))
        {'id': 'abc', 'object': 'chat.completion', 'choices': [...], ...}
    """
    if not events:
        return None

    content_parts, reasoning_parts, tool_calls_by_index, finish_reason, response_id, response_object = (
        _accumulate_stream(events)
    )
    message = _build_message(content_parts, reasoning_parts, tool_calls_by_index)
    resp = _build_response(
        is_chat, content_parts, message, finish_reason, response_id, response_object
    )
    return json.dumps(resp).encode()


def _accumulate_stream(
    events: list[dict | None],
) -> tuple[
    list[str],
    list[str],
    dict[int, dict],
    str | None,
    str | None,
    str | None,
]:
    """Accumulate content, reasoning, tool calls, and metadata from SSE events.

    Args:
        events: List of parsed JSON event dicts (None entries are skipped).

    Returns:
        A tuple of (content_parts, reasoning_parts, tool_calls_by_index,
        finish_reason, response_id, response_object).
    """
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls_by_index: dict[int, dict] = {}
    finish_reason: str | None = None
    response_id: str | None = None
    response_object: str | None = None

    for data in events:
        if data is None:
            continue
        if response_id is None and "id" in data:
            response_id = data["id"]
        if response_object is None and "object" in data:
            response_object = data["object"].replace(".chunk", "")
        for choice in data.get("choices") or []:
            if "text" in choice and choice["text"] is not None:
                content_parts.append(choice["text"])
            delta = choice.get("delta") or {}
            if "content" in delta and delta["content"] is not None:
                content_parts.append(delta["content"])
            reasoning = delta.get("reasoning")
            if reasoning is None:
                reasoning = delta.get("reasoning_content")
            if reasoning is not None:
                reasoning_parts.append(reasoning)
            for tc in delta.get("tool_calls") or []:
                _accumulate_tool_call(tool_calls_by_index, tc)
            fr = choice.get("finish_reason")
            if fr is not None:
                finish_reason = fr

    return (
        content_parts,
        reasoning_parts,
        tool_calls_by_index,
        finish_reason,
        response_id,
        response_object,
    )


def _accumulate_tool_call(tool_calls_by_index: dict[int, dict], tc: dict) -> None:
    """Merge a delta tool call into the per-index accumulated tool call.

    Args:
        tool_calls_by_index: Map of tool call index to accumulated dict.
        tc: The delta tool call fragment.
    """
    idx = tc.get("index", 0)
    existing = tool_calls_by_index.get(idx, {})
    # A field the delta is not using can be null instead of missing.
    if tc.get("id") is not None:
        existing["id"] = tc["id"]
    if tc.get("type") is not None:
        existing["type"] = tc["type"]
    func = tc.get("function") or {}
    if func.get("name") is not None or func.get("arguments") is not None:
        existing.setdefault("function", {})
        if func.get("name") is not None:
            existing["function"]["name"] = func["name"]
        if func.get("arguments") is not None:
            existing["function"]["arguments"] = (
                (existing["function"].get("arguments") or "") + func["arguments"]
            )
    tool_calls_by_index[idx] = existing


def _build_message(
    content_parts: list[str],
    reasoning_parts: list[str],
    tool_calls_by_index: dict[int, dict],
) -> dict:
    """Build the response message dict from accumulated stream parts.

    Args:
        content_parts: Accumulated content fragments.
        reasoning_parts: Accumulated reasoning fragments.
        tool_calls_by_index: Accumulated tool calls keyed by index.

    Returns:
        The message dict.
    """
    message: dict = {}
    if reasoning_parts:
        message["reasoning_content"] = "".join(reasoning_parts)
    if content_parts:
        message["content"] = "".join(content_parts)
    if tool_calls_by_index:
        message["tool_calls"] = list(tool_calls_by_index.values())
    if not message:
        message["content"] = ""
    return message


def _build_response(
    is_chat: bool,
    content_parts: list[str],
    message: dict,
    finish_reason: str | None,
    response_id: str | None,
    response_object: str | None,
) -> dict:
    """Build the outer OpenAI response dict.

    Args:
        is_chat: Whether to build a chat or text-completion response.
        content_parts: Accumulated content fragments.
        message: The assembled message dict.
        finish_reason: The finish reason, or None.
        response_id: The response id, or None.
        response_object: The response object type, or None.

    Returns:
        The response dict.
    """
    if is_chat:
        resp: dict = {
            "id": response_id or "",
            "object": response_object or "chat.completion",
            "created": 0,
            "model": "",
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
        }
    else:
        resp = {
            "id": response_id or "",
            "object": response_object or "text_completion",
            "created": 0,
            "model": "",
            "choices": [
                {
                    "text": "".join(content_parts),
                    "index": 0,
                    "finish_reason": finish_reason,
                }
            ],
        }
    return resp


class SSEByteDecoder:
    """Decode SSE protocol from a raw byte stream.

    Handles chunked TCP boundaries (partial lines split across feeds)
    and multi-line ``data`` fields joined with newlines.
    """

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._data_lines: list[str] = []
        self._event_type: str = "message"
        self._id: str | None = None
        self._raw_lines: list[bytes] = []
        self._closed = False

    def _parse_line(self, line: str) -> None:
        if line.startswith(":"):
            return
        if not line:
            return
        if ":" not in line:
            return
        field, value = line.split(":", 1)
        if value.startswith(" "):
            value = value[1:]
        if field == "data":
            self._data_lines.append(value)
        elif field == "event":
            self._event_type = value
        elif field == "id":
            self._id = value

    def _flush_event(self) -> tuple[str, str, str | None, list[bytes]]:
        data = "\n".join(self._data_lines)
        result = (self._event_type, data, self._id, self._raw_lines)
        self._data_lines = []
        self._event_type = "message"
        self._id = None
        self._raw_lines = []
        return result

    def feed(self, chunk: bytes) -> list[tuple[str, str, str | None, list[bytes]]]:
        """Feed a byte chunk into the decoder and return completed events.

        Parses ``data`` / ``event`` / ``id`` fields and yields events on
        blank-line boundaries. Raises if the decoder has already been closed.

        Args:
            chunk: Raw bytes to decode. May contain partial lines that will
                be buffered until the next ``feed`` call.

        Returns:
            List of ``(event_type, data, event_id, raw_lines)`` tuples for
            every complete SSE event terminated by a blank line in the data.

        Raises:
            RuntimeError: If called after ``close()`` has been invoked.
        """
        if self._closed:
            raise RuntimeError("Cannot feed a closed decoder")
        self._buffer.extend(chunk)
        events: list[tuple[str, str, str | None, list[bytes]]] = []
        while True:
            idx = self._buffer.find(b"\n")
            if idx == -1:
                break
            line_bytes = self._buffer[:idx]
            if line_bytes.endswith(b"\r"):
                line_bytes = line_bytes[:-1]
            self._buffer = self._buffer[idx + 1:]
            line = line_bytes.decode("utf-8", errors="replace")
            if not line:
                events.append(self._flush_event())
            else:
                self._parse_line(line)
                self._raw_lines.append(line_bytes + b"\n")
        return events

    def close(self) -> tuple[list[tuple[str, str, str | None, list[bytes]]], bytes]:
        """Flush pending data and finalize the decoder.

        Returns any incomplete event still in the buffer and clears the
        internal state. After this call, further ``feed`` calls raise
        ``RuntimeError``.

        Returns:
            A tuple of ``(events, leftover)`` where ``events`` contains any
            pending flushed event and ``leftover`` is the raw unconsumed
            bytes remaining in the buffer.
        """
        self._closed = True
        events = []
        if self._data_lines or self._id is not None:
            events.append(self._flush_event())
        leftover = bytes(self._buffer)
        self._buffer.clear()
        return events, leftover
