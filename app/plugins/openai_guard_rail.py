import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict, deque
from typing import AsyncIterator, Awaitable, Callable, Self
from urllib.parse import urlparse

from app.config import ServiceConfig
from app.responses import AbstractMutatingResponse, CaptureResponse

import aiohttp.web as web

from app.plugins.base import PluginBase
from app.plugins.openai_dialect import (
    ChatCompletionsDialect,
    ChatProjection,
    ConversationDialect,
    canonicalize_arguments,
    find_exact,
)
from app.plugins.openai_responses_dialect import ResponsesDialect
from app.plugins.utils import (
    SSEByteDecoder,
    ToolCallExclusion,
    is_tool_call_excluded,
    parse_tool_call_exclusions,
)

_TOOL_CALL_LOOP_DETECTOR_MIN_DISTANCE = 5
_RESUMPTION_LRU_SIZE = 50

_CHAT_DIALECT = ChatCompletionsDialect()
_DIALECTS: tuple[ConversationDialect, ...] = (_CHAT_DIALECT, ResponsesDialect())

__all__ = ["OpenAIGuardRail"]

_logger = logging.getLogger(__name__)


_RETRY_LOOP = 1
_RETRY_DUP_TOOL_CALL = 2
_RETRY_NO_FINAL_OUTPUT = 3
_RETRY_INVALID_TOOL_CALL = 4


class _Retry(Exception):
    __slots__ = ("message", "reason")

    def __init__(self, message: str, reason: int = _RETRY_LOOP) -> None:
        self.message = message
        self.reason = reason
        super().__init__(message)


class _Telemetry:
    __slots__ = (
        "generic_loop",
        "duplicate_tool_call",
        "no_final_output",
        "invalid_tool_call",
        "generic_loop_last",
        "duplicate_tool_call_last",
        "no_final_output_last",
        "invalid_tool_call_last",
    )

    def __init__(self) -> None:
        self.generic_loop = 0
        self.duplicate_tool_call = 0
        self.no_final_output = 0
        self.invalid_tool_call = 0
        self.generic_loop_last: float | None = None
        self.duplicate_tool_call_last: float | None = None
        self.no_final_output_last: float | None = None
        self.invalid_tool_call_last: float | None = None

    def record_generic_loop(self) -> None:
        self.generic_loop += 1
        self.generic_loop_last = time.time()

    def record_duplicate_tool_call(self) -> None:
        self.duplicate_tool_call += 1
        self.duplicate_tool_call_last = time.time()

    def record_no_final_output(self) -> None:
        self.no_final_output += 1
        self.no_final_output_last = time.time()

    def record_invalid_tool_call(self) -> None:
        self.invalid_tool_call += 1
        self.invalid_tool_call_last = time.time()

    @property
    def total(self) -> int:
        return (
            self.generic_loop
            + self.duplicate_tool_call
            + self.no_final_output
            + self.invalid_tool_call
        )


class _ResumptionState:
    """Accumulates deltas for the current turn. Saves reasoning, content, and tool calls."""

    def __init__(self) -> None:
        self.messages: list[dict] = []
        self.current_message: dict = {"role": "assistant", "content": ""}
        self._tool_calls: list[dict] = []
        self._current_tool_id: str = ""
        self._current_tc: dict = {}
        self._aliases: dict[str, str] = {}

    def resolve_tool_call_id(self, tc_id: str) -> str:
        """Normalize a client tool call id and map a terminal id to its streamed id."""
        norm_id = _normalize_tc_id(tc_id)
        return self._aliases.get(norm_id, norm_id)

    def find_aliases(self, tc_ids: list[str]) -> list[str]:
        """Return the terminal ids that stand for the given streamed ids."""
        return [alias for alias, tc_id in self._aliases.items() if tc_id in tc_ids]

    def _record_aliases(self, delta: dict) -> None:
        for streamed_id, terminal_id in delta.get("tool_call_id_aliases") or []:
            streamed = _normalize_tc_id(streamed_id)
            terminal = _normalize_tc_id(terminal_id)
            if streamed and terminal and streamed != terminal:
                self._aliases[terminal] = streamed

    @property
    def tool_calls(self) -> dict[str, str]:
        return {
            tc.get("id") or "": ((tc.get("function") or {}).get("name") or "")
            + ((tc.get("function") or {}).get("arguments") or "")
            for tc in self._tool_calls
            + ([self._current_tc] if self._current_tool_id else [])
        }

    def _flush_tool_call(self) -> None:
        if self._current_tool_id:
            self._tool_calls.append(dict(self._current_tc))
            self._current_tool_id = ""
            self._current_tc = {}

    def feed(self, delta: dict) -> None:
        tp = delta.get("type", "")
        if tp == "content":
            self.current_message.setdefault("content", "")
            self.current_message["content"] += delta.get("text", "")
            self._flush_tool_call()
        elif tp == "reasoning":
            self.current_message.setdefault("reasoning_content", "")
            self.current_message["reasoning_content"] += delta.get("text", "")
            self._flush_tool_call()
        elif tp == "tool_call":
            tc_id = _normalize_tc_id(delta.get("tool_call_id", ""))
            func = delta.get("function")
            if tc_id and tc_id != self._current_tool_id and self._current_tool_id:
                self._flush_tool_call()
            if tc_id and tc_id != self._current_tool_id:
                self._current_tool_id = tc_id
                self._current_tc = {
                    "id": tc_id,
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                }
            if self._current_tool_id:
                if func:
                    if "name" in func:
                        self._current_tc["function"]["name"] += func["name"]
                    if "arguments" in func:
                        self._current_tc["function"]["arguments"] += func["arguments"]
        else:
            if tp == "finish_reason":
                self._record_aliases(delta)
            self._flush_tool_call()

    def finalize_turn(self) -> dict:
        """Finalize the current turn, building a complete message dict."""
        self._flush_tool_call()
        msg = dict(self.current_message)
        if self._tool_calls:
            msg["tool_calls"] = list(self._tool_calls)
        self.messages.append(msg)
        self.current_message = {"role": "assistant", "content": ""}
        self._tool_calls = []
        self._current_tool_id = ""
        self._current_tc = {}
        return msg


class _LRUOrderedDict:
    """LRU cache keyed by tool_call_id, valued by _ResumptionState.

    Counts capacity by distinct _ResumptionState objects. Multiple IDs may
    share the same state. The state is evicted as a whole (all its IDs removed).
    """

    def __init__(self) -> None:
        self._states: OrderedDict[_ResumptionState, list[str]] = OrderedDict()
        self._id_to_state: dict[str, _ResumptionState] = {}

    def __setitem__(self, tc_id: str, state: _ResumptionState) -> None:
        if tc_id in self._id_to_state:
            old = self._id_to_state[tc_id]
            if old is state:
                if state in self._states:
                    self._states.move_to_end(state)
                return
            else:
                self.pop(tc_id)
        self._id_to_state[tc_id] = state
        if state in self._states:
            self._states[state].append(tc_id)
            self._states.move_to_end(state)
        else:
            if len(self._states) >= _RESUMPTION_LRU_SIZE:
                self._evict_oldest()
            self._states[state] = [tc_id]

    def pop(self, tc_id: str) -> _ResumptionState | None:
        state = self._id_to_state.pop(tc_id, None)
        if state is not None:
            ids = self._states.pop(state, [])
            for k in ids:
                self._id_to_state.pop(k, None)
        return state

    def len(self) -> int:
        return len(self._states)

    def _evict_oldest(self) -> None:
        if not self._states:
            return
        oldest_state, ids = next(iter(self._states.items()))
        self._states.pop(oldest_state)
        for k in ids:
            self._id_to_state.pop(k, None)

    def __contains__(self, tc_id: str) -> bool:
        return tc_id in self._id_to_state


class _ToolCallHasher:
    """Accumulates tool call fragments and produces a SHA-256 hash on ID boundary."""

    def __init__(self) -> None:
        self._accumulated: str = ""
        self._current_id: str | None = None

    def feed(self, delta_str: str, tc_id: str) -> str:
        result = ""
        if self._current_id != tc_id:
            self._current_id = tc_id
            result = self._flush_and_hash()
        self._accumulated += delta_str
        return result

    def _flush_and_hash(self) -> str:
        if not self._accumulated:
            return ""
        hash_len = max(len(self._accumulated), 32)
        h = hashlib.sha256(self._accumulated.encode()).hexdigest()
        self._accumulated = ""
        if len(h) >= hash_len:
            return h[:hash_len]
        # Extend with counter-based hashes to avoid repeating patterns
        result = h
        counter = 0
        while len(result) < hash_len:
            result += hashlib.sha256(h.encode() + str(counter).encode()).hexdigest()
            counter += 1
        return result[:hash_len]


class _StreamLoopDetector:
    """Detects when a streaming text output enters a repetition loop.

    Uses a sliding-window hash to find blocks that repeat at a fixed period.
    Once a loop is confirmed (hash match + string verification), returns True
    so the caller can abort the stream.
    """

    def __init__(self, min_block_size: int = 50) -> None:
        self.buffer = ""
        self.min_block_size = min_block_size
        self.seen_hashes = {}
        self._last_start_idx = 0
        self._last_added_hashes: dict[int, list[int]] = {}

    def feed(self, delta: str) -> bool:
        """Feed a chunk of text into the detector.

        Appends *delta* to the internal buffer and checks for repetition loops
        using a sliding-window hash. If a loop is detected, returns ``True``
        without undoing the feed — call ``pop_last()`` to revert if needed.

        Args:
            delta: New text chunk from the stream.

        Returns:
            ``True`` if a repetition loop is detected, ``False`` otherwise.
        """
        if not delta:
            return False

        self._last_start_idx = len(self.buffer)
        self._last_added_hashes = {}
        self.buffer += delta
        end_idx = len(self.buffer)

        for i in range(max(self.min_block_size, self._last_start_idx + 1), end_idx + 1):
            anchor_hash = hash(self.buffer[i - self.min_block_size : i])

            if anchor_hash in self.seen_hashes:
                for prev_end in self.seen_hashes[anchor_hash]:
                    distance = i - prev_end
                    if distance >= self.min_block_size and i >= 2 * distance:
                        if (
                            self.buffer[i - 2 * distance : i - distance]
                            == self.buffer[i - distance : i]
                        ):
                            return True

            if anchor_hash in self.seen_hashes:
                self.seen_hashes[anchor_hash].append(i)
                self._last_added_hashes.setdefault(anchor_hash, []).append(i)
            else:
                self.seen_hashes[anchor_hash] = [i]
                self._last_added_hashes[anchor_hash] = [i]

        return False

    def pop_last(self) -> bool:
        """Undo the most recent successful ``feed()`` call.

        Restores the buffer and hash state to what they were before the
        last ``feed()``. Only the most recent feed can be undone — earlier
        feeds cannot be recovered.

        Returns:
            ``True`` if a feed was undone, ``False`` if there was nothing to undo
            (no prior feed, or previous undo already consumed the state).
        """
        if not self._last_added_hashes:
            return False
        self.buffer = self.buffer[: self._last_start_idx]
        for h, indices in self._last_added_hashes.items():
            old = self.seen_hashes.get(h)
            for idx in indices:
                if old and idx in old:
                    old.remove(idx)
            if not old:
                del self.seen_hashes[h]
        self._last_added_hashes = {}
        return True


async def _iter_openai_sse_events(
    chunks: AsyncIterator[bytes],
) -> AsyncIterator[tuple[dict | None, bytes]]:
    """Async iterator that wraps SSEByteDecoder with OpenAI-specific parsing.

    Yields (parsed_json | None, raw_bytes) for each complete SSE event.
    Handles [DONE] sentinel, JSON parsing, and invalid SSE passthrough.
    """
    decoder = SSEByteDecoder()

    def _process_event(data: str, raw_lines: list[bytes]) -> tuple | None:
        if data == "[DONE]":
            _logger.debug("[OpenAIGuardRail] [DONE] sentinel received, skipping")
            return None
        raw_event = b"".join(raw_lines) + b"\n"
        parsed_json = None
        if data:
            try:
                parsed_json = json.loads(data)
            except json.JSONDecodeError:
                _logger.debug(
                    "[OpenAIGuardRail] Invalid JSON in SSE data, treating as passthrough"
                )
        return (parsed_json, raw_event)

    async for chunk in chunks:
        events = decoder.feed(chunk)
        for _event_type, data, _event_id, raw_lines in events:
            result = _process_event(data, raw_lines)
            if result is not None:
                yield result

    events, leftover = decoder.close()
    for _event_type, data, _event_id, raw_lines in events:
        result = _process_event(data, raw_lines)
        if result is not None:
            yield result

    if leftover:
        _logger.debug("[OpenAIGuardRail] SSE decoder leftover: %d bytes", len(leftover))
        yield (None, leftover)


def _decode_request_body(body: bytes) -> dict | None:
    """Parse a request body as a JSON object, or None if it is not one."""
    try:
        data = json.loads(body)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return data


def _encode_request_body(
    headers: dict[str, str], data: dict
) -> tuple[dict[str, str], bytes]:
    """Re-encode a request body, refreshing Content-Length to match."""
    modified_body = json.dumps(data).encode()
    modified_headers = dict(headers)
    modified_headers["Content-Length"] = str(len(modified_body))
    return modified_headers, modified_body


def _decode_request_projection(
    body: bytes, dialect: ConversationDialect
) -> tuple[dict, ChatProjection] | None:
    """Parse a request body and its chat messages, or None if it has none."""
    data = _decode_request_body(body)
    if data is None:
        return None
    projection = dialect.project(data)
    if projection is None:
        return None
    return data, projection


def _read_request_history[V](data: dict[str, V], dialect: ConversationDialect) -> list | V | None:
    """Read the history a request carries, as chat messages when it has a usable one."""
    projection = dialect.project(data)
    if projection is not None:
        return projection.messages
    return dialect.history_value(data)


def _safe_suffix_start(text: str, min_block_size: int, chunk_size: int) -> int:
    """Find where the longest loop-free trailing slice of text begins.

    Scans text backwards a chunk at a time with a throwaway detector and stops
    at the first chunk that closes a repetition loop, so history already stuck
    in a loop does not prime the permanent detector into firing immediately.

    Args:
        text: The serialized history to scan.
        min_block_size: Repetition block size the throwaway detector looks for.
        chunk_size: How much reversed text to feed per step.

    Returns:
        Index into text where the loop-free suffix starts; len(text) if none is.
    """
    temp = _StreamLoopDetector(min_block_size=min_block_size)
    safe_end_in_reversed = 0
    pos = 0
    reversed_text = text[::-1]
    while pos < len(reversed_text):
        end = min(pos + chunk_size, len(reversed_text))
        if temp.feed(reversed_text[pos:end]):
            temp.pop_last()
            break
        safe_end_in_reversed = end
        pos = end
    return len(text) - safe_end_in_reversed


def _normalize_tc_id(tc_id: str) -> str:
    """Normalize tool_call_id by stripping all non-alphanumeric characters.

    Upstream may send 'chatcmpl-tool-80e8d1086d9678d1' while client sends
    'chatcmpltool80e8d1086d9678d1'. Both normalize to 'chatcmpltool80e8d1086d9678d1'.
    """
    return "".join(c for c in tc_id if c.isalnum())


class _AsyncBytesStream:
    """Wrap raw bytes as an async iterable of chunks for SSE parsing."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._sent = False

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self

    async def __anext__(self) -> bytes:
        if self._sent:
            raise StopAsyncIteration
        self._sent = True
        return self._data


class _DeltaSerializer:
    """Wraps tool call hasher and section separators to produce loop detector feeds."""

    def __init__(self) -> None:
        self._hasher = _ToolCallHasher()
        self._last_separator: str | None = None
        self._current_tool_call_id: str = ""
        self._pending_tool_call_bytes: bytearray = bytearray()
        self._pending_tool_call_delta: dict = {"type": "tool_call"}
        self._results: deque[tuple[str, dict, bytes]] = deque()
        self._on_data: Callable[[str, dict, bytes], None] | None = None

    def _accumulate_tool_call_delta(self, delta: dict) -> None:
        """Merge a tool call delta into _pending_tool_call_delta."""
        tc_id = delta.get("tool_call_id", "")
        text = delta.get("text", "")
        func = delta.get("function")
        if tc_id:
            self._pending_tool_call_delta.setdefault("tool_call_id", tc_id)
        self._pending_tool_call_delta.setdefault("text", "")
        self._pending_tool_call_delta["text"] += text
        if func:
            self._pending_tool_call_delta.setdefault(
                "function", {"name": "", "arguments": ""}
            )
            if "name" in func:
                self._pending_tool_call_delta["function"]["name"] += func["name"]
            if "arguments" in func:
                self._pending_tool_call_delta["function"]["arguments"] += func[
                    "arguments"
                ]

    def feed(self, delta: dict, raw_bytes: bytes = b"") -> None:
        """Feed a delta for serialization."""
        text = delta.get("text", "")
        tp = delta.get("type", "")

        # Special handling for tool call: accmulate delta and lazy flush
        tool_call_id = delta.get("tool_call_id", "")
        if tp == "tool_call":
            if tool_call_id:
                self._current_tool_call_id = tool_call_id
            else:
                tool_call_id = self._current_tool_call_id
        # Accumulate tool call hashes until the entire tool call section is complete.
        # Always feed in order to trigger lazy flush even if the text is empty.
        tool_call_text = text if tp == "tool_call" else ""
        tool_call_hash = self._hasher.feed(tool_call_text, tool_call_id)
        if tool_call_hash:
            # Tool call completed. Flush the accumulated tool call hash and raw bytes.
            self._results.append(
                (
                    "[tool_call]" + tool_call_hash,
                    self._pending_tool_call_delta,
                    bytes(self._pending_tool_call_bytes),
                )
            )
            self._pending_tool_call_bytes.clear()
            self._pending_tool_call_delta = {"type": "tool_call"}

        if tp == "tool_call":
            # Valid tool call delta. Cache the raw bytes and delta until it completes.
            self._accumulate_tool_call_delta(delta)
            self._pending_tool_call_bytes.extend(raw_bytes)
        elif tp in ("content", "reasoning"):
            # Add separator for reasoning and content types
            separator = f"[{tp}]"
            if self._last_separator != separator:
                self._results.append((f"{separator}" + text, delta, raw_bytes))
            else:
                self._results.append((text, delta, raw_bytes))
            self._last_separator = separator
        else:
            # Other types. Pass through as is.
            self._results.append((text, delta, raw_bytes))
        self._flush()

    def poll(self) -> list[tuple[str, dict, bytes]]:
        result = []
        size = len(self._results)
        for _i in range(size):
            result.append(self._results.popleft())
        return result

    def finalize(self) -> Self:
        """Complete any pending tool call and return this serializer.

        Returns:
            This serializer, so the flushed results can be read with poll().
        """
        # Feeding empty delta will complete pending tool calls.
        self.feed({})
        return self

    def set_data_callback(self, callback: Callable[[str, dict, bytes], None]) -> None:
        """Add a callback to monitor structured output from this serializer without manual poll."""
        self._on_data = callback
        self._flush()

    def _flush(self) -> None:
        """Flush results to the callback."""
        if self._on_data:
            for serialized_text, delta, raw_bytes in self.poll():
                self._on_data(serialized_text, delta, raw_bytes)


class _Session:
    """Holds per-request state for the guard rail override path."""

    def __init__(
        self,
        is_stream: bool,
        max_retries: int,
        break_loops: bool,
        min_loop_distance: int,
        force_final_output: bool,
        dedup_tool_calls: bool,
        validate_tool_calls: bool,
        response: AbstractMutatingResponse,
        web_request: web.Request,
        dialect: ConversationDialect,
        messages: list[dict] | None = None,
        resumption_state: _ResumptionState | None = None,
        dedup_exclusions: list[ToolCallExclusion] | None = None,
    ) -> None:
        self.is_stream = is_stream
        self.dialect = dialect
        self._splicer = dialect.output_splicer()
        self.max_retries = max_retries
        self.break_loops = break_loops
        self.min_loop_distance = min_loop_distance
        self.force_final_output = force_final_output
        self.dedup_tool_calls = dedup_tool_calls
        self.validate_tool_calls = validate_tool_calls
        self.response = response
        self.web_request = web_request
        self.retries = 0
        self.approved_buffer = bytearray()
        self._resumption_state: _ResumptionState = (
            resumption_state if resumption_state is not None else _ResumptionState()
        )
        self._dedup_exclusions: list[ToolCallExclusion] = dedup_exclusions or []
        self.is_ok = False
        self.prepared = False
        self._upstream_streaming = False
        self._has_finish_reason = False
        self._loop_detector = _StreamLoopDetector(min_block_size=self.min_loop_distance)
        self._tool_call_dup_detector = _StreamLoopDetector(
            min_block_size=_TOOL_CALL_LOOP_DETECTOR_MIN_DISTANCE
        )
        self._delta_serializer: _DeltaSerializer = _DeltaSerializer()
        if messages and (break_loops or dedup_tool_calls):
            chat_history = self._serialize_history(messages)
            if break_loops:
                self._init_loop_history(chat_history)
            if dedup_tool_calls:
                self._init_tool_call_history(chat_history)
        self._delta_serializer.set_data_callback(self._on_serialized_data)

    def _serialize_history(self, messages: list[dict]) -> str:
        """Serialize chat history into a single string from the last user turn."""
        start = 0
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                start = i + 1
                break
        for m in messages[start:]:
            role = m.get("role")
            if role == "assistant":
                for key in ("reasoning_content", "content"):
                    text = m.get(key)
                    if text:
                        tp = "reasoning" if key == "reasoning_content" else "content"
                        self._delta_serializer.feed({"type": tp, "text": text})
                for tc in m.get("tool_calls") or []:
                    func = tc.get("function") or {}
                    name = func.get("name") or ""
                    args = canonicalize_arguments(func.get("arguments") or "")
                    tc_text = name + args
                    tc_id = tc.get("id") or ""
                    if tc_text:
                        self._delta_serializer.feed(
                            {
                                "type": "tool_call",
                                "text": tc_text,
                                "tool_call_id": tc_id,
                            }
                        )
        serializer = self._delta_serializer.finalize()
        return "".join([s for s, _, _ in serializer.poll()])

    def _init_loop_history(self, chat_history: str) -> None:
        """Initialize loop detector with non-looping assistant history suffix.

        Flattens assistant role content, feeds it backwards with a temporary
        detector to find the longest non-looping suffix, then feeds that suffix
        forward into the permanent detector.
        """
        if not chat_history:
            return

        min_bs = self.min_loop_distance
        safe_start = _safe_suffix_start(chat_history, min_bs, min_bs)
        if safe_start < len(chat_history):
            suffix = chat_history[safe_start:]
            self._loop_detector.feed(suffix)

    def _init_tool_call_history(self, chat_history: str) -> None:
        """Init tool call detector with [tool_call] sections from serialized history."""
        tc_text = ""
        pos = 0
        prefix = "[tool_call]"
        while pos < len(chat_history):
            start = chat_history.find(prefix, pos)
            if start < 0:
                break
            # Find next section marker to avoid false matches with `[` in content
            best_end = len(chat_history)
            for marker in ("[reasoning]", "[content]", "[tool_call]"):
                idx = chat_history.find(marker, start + len(prefix))
                if 0 <= idx < best_end:
                    best_end = idx
            tc_text += chat_history[start + len(prefix) : best_end]
            pos = best_end

        if not tc_text:
            return
        safe_start = _safe_suffix_start(
            tc_text, _TOOL_CALL_LOOP_DETECTOR_MIN_DISTANCE, 5
        )
        if safe_start < len(tc_text):
            self._tool_call_dup_detector.feed(tc_text[safe_start:])

    @staticmethod
    def _build_tool_call_hint(tc_text: str) -> str:
        truncated = (tc_text[:50] + "...") if tc_text and len(tc_text) > 50 else tc_text
        if truncated:
            return f" The tool call was: {truncated}."
        return ""

    def resumption_state(self) -> _ResumptionState:
        """Get the session-level resumption state."""
        return self._resumption_state

    def _on_serialized_data(
        self, serialized_str: str, delta: dict, raw_bytes: bytes
    ) -> None:
        """Handle serialized data."""
        if self.retries < self.max_retries:
            loop_detected = False
            dup_tool_call_detected = False
            try:
                # Loop detection
                if self.break_loops and self._loop_detector.feed(serialized_str):
                    loop_detected = True
                    raise _Retry(
                        "SYSTEM [EXTREMELY IMPORTANT DO NOT IGNORE]: Your last response "
                        "was interrupted because it entered a repetition loop. Stop repeating "
                        "the same content and continue with new, different content to address "
                        "the user's request.",
                        _RETRY_LOOP,
                    )
                # Tool call validation
                if serialized_str.startswith("[tool_call]"):
                    tc_hash = serialized_str[len("[tool_call]") :]
                    tc_text = delta.get("text", "")
                    tc_hint = self._build_tool_call_hint(tc_text)
                    arguments = (delta.get("function") or {}).get("arguments") or ""
                    name = (delta.get("function") or {}).get("name") or ""
                    if self.validate_tool_calls and arguments:
                        try:
                            json.loads(arguments)
                        except (json.JSONDecodeError, ValueError):
                            raise _Retry(
                                "SYSTEM [EXTREMELY IMPORTANT DO NOT IGNORE]: Your last response "
                                "was interrupted because a tool call had invalid JSON arguments. "
                                f"Always use valid JSON for tool call arguments.{tc_hint}",
                                _RETRY_INVALID_TOOL_CALL,
                            )
                    if (
                        self.dedup_tool_calls
                        and tc_hash
                        and not is_tool_call_excluded(name, arguments, self._dedup_exclusions)
                        and self._tool_call_dup_detector.feed(tc_hash)
                    ):
                        dup_tool_call_detected = True
                        raise _Retry(
                            "SYSTEM [EXTREMELY IMPORTANT DO NOT IGNORE]: Your last response "
                            "was interrupted because you are repeating the same tool call you "
                            f"already made or duplicate the same tool call multiple times in your response.{tc_hint} "
                            "If you want to edit a file, the edit is probably already applied — read it to confirm first. "
                            "If you want to read a file, you already read it previously and the result is in your context — refer to it. "
                            "If you want to execute a command, the command is already executed — refer to your previous context for the result.",
                            _RETRY_DUP_TOOL_CALL,
                        )
                # Finish reason check
                if self.force_final_output:
                    system_prompt_for_content_retry = (
                        "SYSTEM [EXTREMELY IMPORTANT DO NOT IGNORE]: Your last response "
                        "did not contain any text output. Generate a final response in the "
                        "content field to address the user's request."
                    )
                    if delta.get("type", "") == "finish_reason":
                        reason = delta.get("text", "")
                        if (
                            reason == "tool_calls"
                            and not self._resumption_state.tool_calls
                        ):
                            raise _Retry(
                                "SYSTEM [EXTREMELY IMPORTANT DO NOT IGNORE]: Your last response "
                                "has finish_reason set to tool_calls but didn't include any tool calls. "
                                "Either generate a proper tool call to address the user's request, or"
                                "generate a final response in the content field and finish with reason stop.",
                                _RETRY_NO_FINAL_OUTPUT,
                            )
                        elif (
                            reason == "stop"
                            and not self._resumption_state.current_message.get(
                                "content"
                            )
                            and not self._resumption_state.tool_calls
                        ):
                            raise _Retry(
                                system_prompt_for_content_retry,
                                _RETRY_NO_FINAL_OUTPUT,
                            )
                        else:
                            self._has_finish_reason = True
                    elif (
                        not self._upstream_streaming
                        and not delta
                        and not self._has_finish_reason
                        and not self._resumption_state.tool_calls
                        and not self._resumption_state.current_message.get("content")
                    ):
                        # Stream already finished and received EOF. Must has either tool_call or content. This check is needed in case finish_reason is missing.
                        raise _Retry(
                            system_prompt_for_content_retry,
                            _RETRY_NO_FINAL_OUTPUT,
                        )
            except:
                if loop_detected:
                    self._loop_detector.pop_last()
                if dup_tool_call_detected:
                    self._tool_call_dup_detector.pop_last()
                raise

        self._resumption_state.feed(delta)
        self._approve(raw_bytes)

    def _approve(self, raw_bytes: bytes) -> None:
        self.approved_buffer.extend(self._splicer.splice(raw_bytes))

    def on_retry(self) -> None:
        """Mark that the current attempt is being retried."""
        self._splicer.retry()

    async def on_prepared(
        self, status_code: int, reason: str, hdrs: dict[str, str]
    ) -> None:
        if not self.prepared:
            await self._on_first_prepared(status_code, reason, hdrs)
        self._upstream_streaming = True

    async def _on_first_prepared(
        self, status_code: int, reason: str, hdrs: dict[str, str]
    ) -> None:
        if self.prepared:
            return
        self.prepared = True
        self.is_ok = status_code == 200
        self.response.set_status(status_code, reason)
        self.response.headers.update(hdrs)
        if self.is_ok and not self.is_stream:
            self.response.headers["Content-Type"] = "application/json"
        await self.response.prepare(self.web_request)

    async def on_delta(
        self, deltas: list[dict], carrier: int | None, raw_bytes: bytes
    ) -> None:
        """Check the pieces of one stream event and send what is approved.

        Args:
            deltas: The pieces of the event, in the order they are checked.
            carrier: The index of the piece that carries the event's bytes.
            raw_bytes: The event's bytes.
        """
        if not self.prepared:
            return

        if not self.is_ok:
            # Pass through error message.
            self._approve(raw_bytes)
        elif not deltas:
            # Go through serializer for processing in-place.
            self._delta_serializer.feed({}, raw_bytes)
        else:
            for index, delta in enumerate(deltas):
                self._delta_serializer.feed(delta, raw_bytes if index == carrier else b"")

        # Immediately approved buffer in stream mode
        if self.is_stream:
            await self.response.write(bytes(self.approved_buffer))
            self.approved_buffer.clear()

    async def on_finish(self) -> None:
        self._upstream_streaming = False
        if self.prepared:
            # An empty event marks the end of the stream.
            await self.on_delta([], None, b"")

    async def flush(self) -> None:
        """Flush remaining approved buffer to the response and close."""
        if not self.prepared:
            return
        self.approved_buffer.extend(self._splicer.finish())
        if self.approved_buffer:
            approved_bytes = bytes(self.approved_buffer)
            if self.is_stream:
                _logger.debug(
                    "[OpenAIGuardRail] Flushing %d streaming bytes", len(approved_bytes)
                )
                await self.response.write(approved_bytes)
            else:
                json_response = await self._assemble_non_streaming(approved_bytes)
                if json_response is not None:
                    _logger.debug(
                        "[OpenAIGuardRail] Assembled non-streaming JSON, %d bytes",
                        len(json_response),
                    )
                    await self.response.write(json_response)
                else:
                    _logger.debug(
                        "[OpenAIGuardRail] Non-streaming pass-through, %d bytes",
                        len(approved_bytes),
                    )
                    await self.response.write(approved_bytes)
        await self.response.write_eof()
        self._resumption_state.finalize_turn()

    async def _assemble_non_streaming(self, raw_bytes: bytes) -> bytes | None:
        """Assemble a complete OpenAI JSON response from accumulated SSE events."""
        events: list[dict] = []
        async for parsed_json, _ in _iter_openai_sse_events(
            _AsyncBytesStream(raw_bytes)
        ):
            if parsed_json is not None:
                events.append(parsed_json)
        if not events:
            _logger.debug(
                "[OpenAIGuardRail] No SSE events in non-streaming response, passing through raw bytes"
            )
            return None
        _logger.debug(
            "[OpenAIGuardRail] Assembling non-streaming response from %d events",
            len(events),
        )
        return self.dialect.assemble_non_streaming(events)


# The plugin manager loads this module with spec.loader.exec_module and finds the
# class by name from config, so no production file imports it.
# lint-skip: coding.exported-api-unused
class OpenAIGuardRail(PluginBase):
    """Intercept OpenAI completions responses, buffer, validate, and flush approved content.

    V1: validate_batch approves all content. Override validate_batch() for actual validation.
    """

    def __init__(
        self, config: dict, service_config: ServiceConfig | None = None
    ) -> None:
        super().__init__(config, service_config)
        self._resumption_state: _LRUOrderedDict = _LRUOrderedDict()
        self._telemetry: _Telemetry = _Telemetry()
        self._dedup_exclusions: list[ToolCallExclusion] = parse_tool_call_exclusions(
            config.get("tool_call_dedup_exclusions", []),
            "OpenAIGuardRail",
            "tool_call_dedup_exclusions",
        )
        self._ui_publish: Callable[[str, str], None] | None = None
        self._ui_refresh_task: asyncio.Task[None] | None = None

    def _pop_resumption_state(
        self, messages: list[dict] | None
    ) -> _ResumptionState | None:
        if not messages:
            return None
        for m in messages:
            if m.get("role") == "tool":
                tc_id = m.get("tool_call_id", "")
                norm_id = _normalize_tc_id(tc_id)
                if norm_id and norm_id in self._resumption_state:
                    state = self._resumption_state.pop(norm_id)
                    _logger.debug(
                        "[OpenAIGuardRail] Popped resumption state for tc_id=%s, messages=%d, current_tool_calls=%d, lru_size=%d",
                        tc_id,
                        len(state.messages),
                        len(state.tool_calls),
                        self._resumption_state.len(),
                    )
                    return state
        _logger.debug(
            "[OpenAIGuardRail] No resumption state found for request: %d tool_call_ids in request, lru_size=%d",
            sum(1 for m in messages if m.get("role") == "tool"),
            self._resumption_state.len(),
        )
        return None

    def _push_resumption_state(self, state: _ResumptionState) -> None:
        tc_ids = (
            [
                _normalize_tc_id(tc.get("id") or "")
                for tc in state.messages[-1].get("tool_calls") or []
            ]
            if state.messages
            else []
        )
        if tc_ids:
            for tc_id in tc_ids + state.find_aliases(tc_ids):
                self._resumption_state[tc_id] = state
            _logger.debug(
                "[OpenAIGuardRail] Saving resumption state for %d tool_calls, lru_size=%d",
                len(tc_ids),
                self._resumption_state.len(),
            )
        else:
            _logger.debug(
                "[OpenAIGuardRail] No tool_calls in resumption state, not saving"
            )

    def init_ui(self, publish: Callable[[str, str], None]) -> None:
        """Initialize the UI publish callback.

        Args:
            publish: A callable that accepts (title: str, html: str)
                and publishes the HTML fragment to the UI.
        """
        self._ui_publish = publish

    async def _ui_refresh_loop(self) -> None:
        while True:
            await asyncio.sleep(60)
            self._publish_ui()

    def _relative_time(self, ts: float | None) -> str:
        if ts is None:
            return "\u2014"
        diff = time.time() - ts
        if diff < 60:
            return "just now"
        elif diff < 3600:
            return f"{int(diff // 60)} min ago"
        elif diff < 86400:
            return f"{int(diff // 3600)} hr ago"
        else:
            return f"{int(diff // 86400)} days ago"

    def _publish_ui(self) -> None:
        if self._ui_publish is None:
            return
        if self._ui_refresh_task is None:
            self._ui_refresh_task = asyncio.create_task(self._ui_refresh_loop())
        html = ""
        if self.config.get("break_loops"):
            html += '<div style="background:#fff3cd;border-left:3px solid #ffc107;'
            html += (
                'padding:6px 10px;border-radius:0 4px 4px 0;width:120px;margin:4px;">'
            )
            html += '<div style="font-size:11px;color:#856404;font-weight:600;">Generic Loop</div>'
            html += f'<div style="font-size:18px;font-weight:700;color:#856404;">{self._telemetry.generic_loop}</div>'
            html += f'<div style="font-size:10px;color:#999;">Last: {self._relative_time(self._telemetry.generic_loop_last)}</div>'
            html += "</div>"
        if self.config.get("dedup_tool_calls"):
            html += '<div style="background:#f8d7da;border-left:3px solid #dc3545;'
            html += (
                'padding:6px 10px;border-radius:0 4px 4px 0;width:120px;margin:4px;">'
            )
            html += '<div style="font-size:11px;color:#721c24;font-weight:600;">Duplicate Tool Call</div>'
            html += f'<div style="font-size:18px;font-weight:700;color:#721c24;">{self._telemetry.duplicate_tool_call}</div>'
            html += f'<div style="font-size:10px;color:#999;">Last: {self._relative_time(self._telemetry.duplicate_tool_call_last)}</div>'
            html += "</div>"
        if self.config.get("force_final_output"):
            html += '<div style="background:#d1ecf1;border-left:3px solid #17a2b8;'
            html += (
                'padding:6px 10px;border-radius:0 4px 4px 0;width:120px;margin:4px;">'
            )
            html += '<div style="font-size:11px;color:#0c5460;font-weight:600;">No Final Output</div>'
            html += f'<div style="font-size:18px;font-weight:700;color:#0c5460;">{self._telemetry.no_final_output}</div>'
            html += f'<div style="font-size:10px;color:#999;">Last: {self._relative_time(self._telemetry.no_final_output_last)}</div>'
            html += "</div>"
        if self.config.get("validate_tool_calls"):
            html += '<div style="background:#f5c6cb;border-left:3px solid #e83e8c;'
            html += (
                'padding:6px 10px;border-radius:0 4px 4px 0;width:120px;margin:4px;">'
            )
            html += '<div style="font-size:11px;color:#721c24;font-weight:600;">Invalid Tool Call</div>'
            html += f'<div style="font-size:18px;font-weight:700;color:#721c24;">{self._telemetry.invalid_tool_call}</div>'
            html += f'<div style="font-size:10px;color:#999;">Last: {self._relative_time(self._telemetry.invalid_tool_call_last)}</div>'
            html += "</div>"
        if html:
            html += '<div style="background:#e2e3e5;border-left:3px solid #6c757d;'
            html += (
                'padding:6px 10px;border-radius:0 4px 4px 0;width:120px;margin:4px;">'
            )
            html += (
                '<div style="font-size:11px;color:#383d41;font-weight:600;">Total</div>'
            )
            html += f'<div style="font-size:18px;font-weight:700;color:#383d41;">{self._telemetry.total}</div>'
            html += '<div style="font-size:10px;color:#999;">Since start</div>'
            html += "</div>"
            self._ui_publish("Guardrail Telemetry", html)

    def name(self) -> str:
        """Return the plugin name from config, defaulting to 'openai_guard_rail'.

        Returns:
            The plugin identifier string.
        """
        return self.config.get("name", "openai_guard_rail")

    def overridden_paths(self) -> list[str]:
        """Return the list of API paths this plugin intercepts.

        Returns:
            List of path strings that trigger interception.
        """
        return [dialect.path for dialect in _DIALECTS]

    def _pick_dialect(self, path: str) -> ConversationDialect:
        """Pick the API format of an intercepted path, chat unless it is another known one."""
        return find_exact(_DIALECTS, urlparse(path).path) or _CHAT_DIALECT

    def _is_streaming_request(self, body: bytes) -> bool:
        """Check if the original request asked for streaming."""
        try:
            return json.loads(body).get("stream", False) is True
        except Exception:
            return False

    def _force_stream(
        self, headers: dict[str, str], body: bytes, dialect: ConversationDialect
    ) -> tuple[dict[str, str], bytes]:
        """Ask the upstream to stream its reply."""
        data = _decode_request_body(body)
        if data is None:
            return headers, body
        dialect.force_stream(data)
        return _encode_request_body(headers, data)

    def _append_msg(
        self,
        headers: dict[str, str],
        body: bytes,
        messages: list[dict],
        dialect: ConversationDialect,
    ) -> tuple[dict[str, str], bytes]:
        """Append messages to request body for retry."""
        decoded = _decode_request_projection(body, dialect)
        if decoded is None:
            return headers, body
        data, projection = decoded
        projection.messages.extend(messages)
        projection.write_back()
        return _encode_request_body(headers, data)

    def _resume_states(
        self,
        headers: dict[str, str],
        body: bytes,
        resumption_state: _ResumptionState | None,
        dialect: ConversationDialect,
    ) -> tuple[dict[str, str], bytes]:
        """Recover reasoning and insert missing assistant messages from resumption state."""
        if not resumption_state or not resumption_state.messages:
            return headers, body
        decoded = _decode_request_projection(body, dialect)
        if decoded is None:
            return headers, body
        data, projection = decoded
        msgs = projection.messages
        _logger.debug(
            "[OpenAIGuardRail] _resume_states: %d msgs in request, %d saved messages in resumption state",
            len(msgs),
            len(resumption_state.messages),
        )
        resumed = False
        saved_by_tc = {}
        for saved in resumption_state.messages:
            for tc in saved.get("tool_calls") or []:
                tc_id = _normalize_tc_id(tc.get("id") or "")
                if tc_id:
                    saved_by_tc[tc_id] = saved
        _logger.debug(
            "[OpenAIGuardRail] _resume_states saved_by_tc keys: %s",
            list(saved_by_tc.keys()),
        )
        tc_ids_inserted = set()
        insertions = []
        existing_tc = set()
        client_ids: dict[str, str] = {}
        for i, m in enumerate(msgs):
            if m.get("role") == "assistant":
                for tc in m.get("tool_calls") or []:
                    tc_id = resumption_state.resolve_tool_call_id(tc.get("id") or "")
                    if tc_id:
                        existing_tc.add(tc_id)
            if m.get("role") == "tool":
                tc_id = resumption_state.resolve_tool_call_id(m.get("tool_call_id", ""))
                client_ids.setdefault(tc_id, _normalize_tc_id(m.get("tool_call_id", "")))
                if (
                    tc_id
                    and tc_id in saved_by_tc
                    and tc_id not in tc_ids_inserted
                    and tc_id not in existing_tc
                ):
                    saved = saved_by_tc[tc_id]
                    insertions.append((i, saved))
                    tc_ids_inserted.add(tc_id)
                    _logger.debug(
                        "[OpenAIGuardRail] Inserting saved assistant message before tool msg[%d] for tc_id=%s",
                        i,
                        tc_id,
                    )
                    resumed = True
        if insertions:
            for idx, saved in reversed(insertions):
                msg_to_insert = {k: v for k, v in saved.items() if k != "tool_call_ids"}
                if "tool_calls" in msg_to_insert:
                    msg_to_insert["tool_calls"] = [
                        {**tc, "id": client_ids.get(tc.get("id") or "", tc.get("id"))}
                        for tc in msg_to_insert["tool_calls"]
                    ]
                msgs.insert(idx, msg_to_insert)
                _logger.debug("[OpenAIGuardRail] Inserted message: %s", msg_to_insert)
        j = len(resumption_state.messages) - 1
        for i in range(len(msgs) - 1, -1, -1):
            if j < 0:
                break
            if msgs[i].get("role") == "assistant":
                saved = resumption_state.messages[j]
                saved_tc_ids = set(
                    _normalize_tc_id(tc.get("id") or "")
                    for tc in saved.get("tool_calls") or []
                )
                msg_tc_ids = set(
                    resumption_state.resolve_tool_call_id(tc.get("id") or "")
                    for tc in msgs[i].get("tool_calls") or []
                )
                if saved_tc_ids and msg_tc_ids == saved_tc_ids:
                    saved_reasoning = saved.get("reasoning_content", "")
                    if saved_reasoning and not msgs[i].get("reasoning_content"):
                        _logger.debug(
                            "[OpenAIGuardRail] Resumed %d chars reasoning from resumption state",
                            len(saved_reasoning),
                        )
                        msgs[i]["reasoning_content"] = saved_reasoning
                        resumed = True
                    j -= 1
        if not resumed:
            return headers, body
        projection.write_back()
        return _encode_request_body(headers, data)

    async def handle_overridden_request(
        self,
        _request_id: int,
        headers: dict[str, str],
        body: bytes,
        path: str,
        web_request: web.Request,
        forward: Callable[
            [dict[str, str], bytes, str, web.Request, AbstractMutatingResponse],
            Awaitable[bool],
        ],
        response: AbstractMutatingResponse,
    ) -> bool:
        """Intercept completions, forward with forced stream, validate and relay response.

        Args:
            _request_id: Unique identifier for this client request.
            headers: Original request headers.
            body: Original request body bytes.
            path: Original request path.
            web_request: The aiohttp web.Request.
            forward: Callable to forward the modified request upstream.
            response: The StreamResponse to write the validated output to.

        Returns:
            True if upstream returned 200, False otherwise.
        """
        _logger.debug("[OpenAIGuardRail] Intercepting %s", path)
        dialect = self._pick_dialect(path)
        is_streaming = self._is_streaming_request(body)
        resumption_state = self._pop_resumption_state(_read_request_history(json.loads(body), dialect))
        headers, body = self._resume_states(headers, body, resumption_state, dialect)
        headers, body = self._force_stream(headers, body, dialect)

        session = _Session(
            is_streaming,
            self.config.get("max_retry_attempt", 2),
            self.config.get("break_loops", False),
            self.config.get("min_loop_distance", 50),
            self.config.get("force_final_output", False),
            self.config.get("dedup_tool_calls", False),
            self.config.get("validate_tool_calls", False),
            response,
            web_request,
            dialect,
            _read_request_history(json.loads(body), dialect),
            resumption_state,
            self._dedup_exclusions,
        )

        result = False
        modified_headers, modified_body = headers, body
        while session.retries <= session.max_retries:
            try:
                result = await self._forward_request(
                    modified_headers,
                    modified_body,
                    path,
                    web_request,
                    forward,
                    session,
                )
                break
            except _Retry as e:
                if e.reason == _RETRY_LOOP:
                    self._telemetry.record_generic_loop()
                elif e.reason == _RETRY_DUP_TOOL_CALL:
                    self._telemetry.record_duplicate_tool_call()
                elif e.reason == _RETRY_NO_FINAL_OUTPUT:
                    self._telemetry.record_no_final_output()
                elif e.reason == _RETRY_INVALID_TOOL_CALL:
                    self._telemetry.record_invalid_tool_call()
                self._publish_ui()
                _logger.info("Loop detection retry triggered: %s", e.message)
                session.retries += 1
                session.on_retry()
                session.resumption_state().finalize_turn()
                modified_headers, modified_body = self._append_msg(
                    headers,
                    body,
                    [
                        session.resumption_state().messages[-1],
                        {"role": "system", "content": e.message},
                    ],
                    dialect,
                )
        await session.flush()
        if session.is_ok:
            self._push_resumption_state(session.resumption_state())
        return result

    async def _forward_request(
        self,
        headers: dict[str, str],
        body: bytes,
        path: str,
        web_request: web.Request,
        forward: Callable[
            [dict[str, str], bytes, str, web.Request, AbstractMutatingResponse],
            Awaitable[bool],
        ],
        session: _Session,
    ) -> bool:
        """Forward a request to the upstream service, capturing the response."""

        capture = CaptureResponse()
        forward_task = asyncio.create_task(
            forward(headers, body, path, web_request, capture)
        )
        prep_task = asyncio.create_task(capture.wait_prepared())

        is_ok = False
        retry_reason_str = ""
        try:
            done, _pending = await asyncio.wait(
                [prep_task, forward_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            if forward_task in done and not capture.prepared:
                _logger.debug(
                    "[OpenAIGuardRail] Forward returned before prepare. Abort."
                )
                return await forward_task

            _logger.debug(
                "[OpenAIGuardRail] Upstream status=%s reason=%s",
                capture.status_code,
                capture.reason,
            )

            await session.on_prepared(
                capture.status_code, capture.reason, capture.headers
            )

            is_ok = capture.status_code == 200
            event_count = 0

            def _on_forward_done(_task: asyncio.Task) -> None:
                asyncio.create_task(capture.close())

            forward_task.add_done_callback(_on_forward_done)

            decoder = session.dialect.delta_decoder()
            async for parsed_json, raw_bytes in _iter_openai_sse_events(
                capture.iter_chunks()
            ):
                event_count += 1
                deltas: list[dict] = []
                carrier: int | None = None
                if is_ok and parsed_json:
                    deltas, carrier = decoder.normalize(parsed_json)
                await session.on_delta(deltas, carrier, raw_bytes)
            await session.on_finish()
            _logger.debug("[OpenAIGuardRail] Relay complete, events=%s", event_count)

            forward_task_res = await forward_task
            _logger.debug(
                "[OpenAIGuardRail] Forward task complete, result=%s", forward_task_res
            )
        except Exception as e:
            _logger.error("[OpenAIGuardRail] Relay error: %s", e, exc_info=True)
            if isinstance(e, _Retry):
                retry_reason_str = {
                    _RETRY_LOOP: "loop_detection",
                    _RETRY_DUP_TOOL_CALL: "dup_tool_call",
                    _RETRY_NO_FINAL_OUTPUT: "no_final_output",
                    _RETRY_INVALID_TOOL_CALL: "invalid_tool_call",
                }.get(e.reason, "unknown")
                raise
        finally:
            prep_task.cancel()
            forward_task.cancel(retry_reason_str)
        return is_ok
