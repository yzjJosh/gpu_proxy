"""Tests for the OpenAI Chat Completions API format."""

import json

import pytest

from app.plugins.openai_api_formats.base import (
    ParsedStreamEvent,
    RequestOverrides,
    TokenUsage,
    standardize_json_arguments,
)
from app.plugins.openai_api_formats.chat_completions import ChatCompletionsAPIFormat

_CHAT_API_FORMAT = ChatCompletionsAPIFormat()

_PROMPT_TOKENS = 12
_COMPLETION_TOKENS = 34
_TOTAL_TOKENS = 46


def _build_chat_stream_chunk(*choices: dict) -> dict:
    """Build a chat completions SSE chunk containing the given choices."""
    return {"id": "1", "choices": list(choices)}


def _build_chat_choice(delta: dict, finish_reason: str | None = None) -> dict:
    """Build one choice entry inside a chat completions SSE chunk."""
    choice: dict = {"index": 0, "delta": delta}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    return choice


def _build_tool_call_delta(
    arguments: str = "", name: str = "", tool_call_id: str = ""
) -> dict:
    """Build one tool call fragment inside a chat completions delta."""
    tool_call_delta: dict = {"index": 0, "function": {}}
    if tool_call_id:
        tool_call_delta["id"] = tool_call_id
    if name:
        tool_call_delta["function"]["name"] = name
    if arguments:
        tool_call_delta["function"]["arguments"] = arguments
    return tool_call_delta


def _expected_content_update(text: str) -> dict:
    """Build an expected content update dict."""
    return {"type": "content", "text": text}


def _expected_reasoning_update(text: str) -> dict:
    """Build an expected reasoning update dict."""
    return {"type": "reasoning", "text": text}


def _expected_finish_reason_update(reason: str) -> dict:
    """Build an expected finish reason update dict."""
    return {"type": "finish_reason", "text": reason}


def _expected_tool_call_update(
    text: str, tool_call_id: str = "", name: str = "", arguments: str = ""
) -> dict:
    """Build an expected tool call update dict."""
    return {
        "type": "tool_call",
        "text": text,
        "tool_call_id": tool_call_id,
        "function": {"name": name, "arguments": arguments},
    }


def _extract_first_matching_update(stream_event: dict) -> dict:
    """Return the first matching update of a chat chunk."""
    for choice in stream_event.get("choices") or []:
        delta = choice.get("delta") or {}
        if delta.get("content") is not None:
            return _expected_content_update(delta["content"])
        reasoning = delta.get("reasoning") or delta.get("reasoning_content")
        if reasoning is not None:
            return _expected_reasoning_update(reasoning)
        for tool_call in delta.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            name = function.get("name") or ""
            arguments = standardize_json_arguments(function.get("arguments") or "")
            tool_call_id = tool_call.get("id") or ""
            if name + arguments or tool_call_id:
                return _expected_tool_call_update(name + arguments, tool_call_id, name, arguments)
        if choice.get("finish_reason") is not None:
            return _expected_finish_reason_update(choice["finish_reason"])
    return {}


# Each chat chunk paired with the updates the parser extracts in validation order.
_CHAT_CHUNK_PARSING_CASES = [
    pytest.param(
        _build_chat_stream_chunk(_build_chat_choice({"content": "hi"})),
        [_expected_content_update("hi")],
        id="content",
    ),
    pytest.param(
        _build_chat_stream_chunk(_build_chat_choice({"reasoning": "hm"})),
        [_expected_reasoning_update("hm")],
        id="reasoning",
    ),
    pytest.param(
        _build_chat_stream_chunk(_build_chat_choice({"reasoning_content": "hm"})),
        [_expected_reasoning_update("hm")],
        id="reasoning_content",
    ),
    pytest.param(
        _build_chat_stream_chunk(
            _build_chat_choice({"tool_calls": [_build_tool_call_delta(tool_call_id="c1", name="run")]})
        ),
        [_expected_tool_call_update("run", "c1", "run")],
        id="tool-call-opening",
    ),
    pytest.param(
        _build_chat_stream_chunk(_build_chat_choice({}, "stop")),
        [_expected_finish_reason_update("stop")],
        id="finish-alone",
    ),
    pytest.param(
        _build_chat_stream_chunk(_build_chat_choice({})),
        [],
        id="empty-delta",
    ),
    pytest.param({"usage": {"prompt_tokens": 1}}, [], id="usage-only"),
    pytest.param(
        _build_chat_stream_chunk(_build_chat_choice({"reasoning": "tail", "content": "head"})),
        [_expected_reasoning_update("tail"), _expected_content_update("head")],
        id="reasoning-and-content",
    ),
    pytest.param(
        _build_chat_stream_chunk(_build_chat_choice({"content": "end"}, "stop")),
        [_expected_content_update("end"), _expected_finish_reason_update("stop")],
        id="content-and-finish",
    ),
    pytest.param(
        _build_chat_stream_chunk(_build_chat_choice({"reasoning": "cut"}, "length")),
        [_expected_reasoning_update("cut"), _expected_finish_reason_update("length")],
        id="reasoning-and-length",
    ),
    pytest.param(
        _build_chat_stream_chunk(
            _build_chat_choice({"tool_calls": [_build_tool_call_delta('{"a": '), _build_tool_call_delta("1}")]})
        ),
        [
            _expected_tool_call_update('{"a": ', arguments='{"a": '),
            _expected_tool_call_update("1}", arguments="1}"),
        ],
        id="two-tool-call-entries",
    ),
    pytest.param(
        _build_chat_stream_chunk(
            _build_chat_choice({"reasoning": "r0", "content": "c0"}, "stop"),
            _build_chat_choice(
                {"content": "c1", "tool_calls": [_build_tool_call_delta(tool_call_id="c9", name="x")]},
                "tool_calls",
            ),
        ),
        [
            _expected_reasoning_update("r0"),
            _expected_content_update("c0"),
            _expected_content_update("c1"),
            _expected_tool_call_update("x", "c9", "x"),
            _expected_finish_reason_update("stop"),
            _expected_finish_reason_update("tool_calls"),
        ],
        id="two-choices",
    ),
    pytest.param(
        _build_chat_stream_chunk(
            _build_chat_choice({"content": None, "reasoning": None, "reasoning_content": "", "tool_calls": None})
        ),
        [_expected_reasoning_update("")],
        id="null-fields-and-empty-reasoning-content",
    ),
    pytest.param(
        _build_chat_stream_chunk(
            _build_chat_choice(
                {"content": "", "tool_calls": [{"index": 0, "id": None, "function": {"name": None, "arguments": None}}]}
            )
        ),
        [_expected_content_update("")],
        id="empty-content-and-blank-fragment",
    ),
    pytest.param(
        _build_chat_stream_chunk(_build_chat_choice({"reasoning": "", "reasoning_content": None})),
        [],
        id="empty-reasoning-falls-back-to-null",
    ),
    pytest.param(
        _build_chat_stream_chunk(
            _build_chat_choice(
                {"tool_calls": [_build_tool_call_delta("1}"), _build_tool_call_delta(tool_call_id="c2", name="next")]}
            )
        ),
        [
            _expected_tool_call_update("1}", arguments="1}"),
            _expected_tool_call_update("next", "c2", "next"),
        ],
        id="call-end-and-next-call",
    ),
    pytest.param(
        _build_chat_stream_chunk(
            _build_chat_choice({"content": "say", "tool_calls": [_build_tool_call_delta(tool_call_id="c2", name="next")]})
        ),
        [_expected_content_update("say"), _expected_tool_call_update("next", "c2", "next")],
        id="content-and-call-opening",
    ),
    pytest.param(
        _build_chat_stream_chunk(
            _build_chat_choice({"tool_calls": [_build_tool_call_delta(tool_call_id="c1", name="a")]}),
            _build_chat_choice({"content": "later"}),
        ),
        [_expected_tool_call_update("a", "c1", "a"), _expected_content_update("later")],
        id="first-choice-tool-call-only",
    ),
    pytest.param(
        _build_chat_stream_chunk(_build_chat_choice({}), _build_chat_choice({"reasoning": "second"}, "stop")),
        [_expected_reasoning_update("second"), _expected_finish_reason_update("stop")],
        id="first-choice-empty",
    ),
]


class TestChatStreamEventParserExtraction:
    """The chat stream event parser extracts every update with finish reasons last."""

    @pytest.mark.parametrize("stream_event, expected_updates", _CHAT_CHUNK_PARSING_CASES)
    def test_extracts_updates_in_validation_order(
        self, stream_event: dict, expected_updates: list[dict]
    ) -> None:
        """Each chat chunk yields its expected updates in validation order."""
        parsed = _CHAT_API_FORMAT.create_stream_event_parser().parse_event(stream_event)
        assert parsed.updates == expected_updates


# Expected index of the update that owns the raw SSE bytes for each parsing case.
_EXPECTED_RAW_BYTES_OWNER_INDEX_BY_CASE = {
    "content": 0,
    "reasoning": 0,
    "reasoning_content": 0,
    "tool-call-opening": 0,
    "finish-alone": 0,
    "empty-delta": None,
    "usage-only": None,
    "reasoning-and-content": 1,
    "content-and-finish": 0,
    "reasoning-and-length": 0,
    "two-tool-call-entries": 0,
    "two-choices": 1,
    "null-fields-and-empty-reasoning-content": 0,
    "empty-content-and-blank-fragment": 0,
    "empty-reasoning-falls-back-to-null": None,
    "call-end-and-next-call": 0,
    "content-and-call-opening": 0,
    "first-choice-tool-call-only": 0,
    "first-choice-empty": 0,
}

_RAW_BYTES_OWNER_INDEX_CASES = [
    pytest.param(row.values[0], _EXPECTED_RAW_BYTES_OWNER_INDEX_BY_CASE[row.id], id=row.id)
    for row in _CHAT_CHUNK_PARSING_CASES
]

_NON_EMPTY_CHUNK_EVENTS = [
    pytest.param(row.values[0], id=row.id)
    for row in _RAW_BYTES_OWNER_INDEX_CASES
    if row.values[1] is not None
]

_SINGLE_UPDATE_CHUNK_EVENTS = [
    pytest.param(row.values[0], id=row.id)
    for row in _CHAT_CHUNK_PARSING_CASES
    if len(row.values[1]) == 1
]


class TestChatStreamEventParserRawBytesOwner:
    """The raw SSE bytes belong to the update picked by the first-match priority order."""

    @pytest.mark.parametrize("stream_event, expected_owner_index", _RAW_BYTES_OWNER_INDEX_CASES)
    def test_raw_bytes_owner_index(
        self, stream_event: dict, expected_owner_index: int | None
    ) -> None:
        """Each chunk assigns its raw bytes to the expected update index."""
        parsed = _CHAT_API_FORMAT.create_stream_event_parser().parse_event(stream_event)
        assert parsed.raw_bytes_owner_index == expected_owner_index

    @pytest.mark.parametrize("stream_event", _NON_EMPTY_CHUNK_EVENTS)
    def test_raw_bytes_owner_matches_first_match_update(self, stream_event: dict) -> None:
        """The update that owns the raw bytes matches the first-match extraction rule."""
        parsed = _CHAT_API_FORMAT.create_stream_event_parser().parse_event(stream_event)
        assert parsed.updates[parsed.raw_bytes_owner_index] == _extract_first_matching_update(stream_event)

    @pytest.mark.parametrize("stream_event", _SINGLE_UPDATE_CHUNK_EVENTS)
    def test_single_update_chunk_matches_first_match_update(self, stream_event: dict) -> None:
        """A chunk with one update matches the first-match extraction rule."""
        parsed = _CHAT_API_FORMAT.create_stream_event_parser().parse_event(stream_event)
        assert parsed == ParsedStreamEvent([_extract_first_matching_update(stream_event)], 0)


class TestChatStreamEventParserToolArguments:
    """The chat stream parser standardizes valid JSON tool call arguments."""

    def test_reformats_valid_json_arguments(self) -> None:
        """Complete JSON tool arguments are reformatted into standard JSON spacing."""
        stream_event = _build_chat_stream_chunk(
            _build_chat_choice({"tool_calls": [_build_tool_call_delta('{"a":1,  "b": [2]}', tool_call_id="c1")]})
        )
        parsed = _CHAT_API_FORMAT.create_stream_event_parser().parse_event(stream_event)
        assert parsed.updates == [
            _expected_tool_call_update('{"a": 1, "b": [2]}', "c1", "", '{"a": 1, "b": [2]}')
        ]

    def test_preserves_partial_json_arguments(self) -> None:
        """Partial tool arguments that are not valid JSON stay unchanged."""
        stream_event = _build_chat_stream_chunk(
            _build_chat_choice({"tool_calls": [_build_tool_call_delta('{"a":', tool_call_id="c1")]})
        )
        parsed = _CHAT_API_FORMAT.create_stream_event_parser().parse_event(stream_event)
        assert parsed.updates == [_expected_tool_call_update('{"a":', "c1", "", '{"a":')]


class TestChatMessageHistoryAndRetryCombiner:
    """Chat message histories edit in place and the chat retry combiner passes bytes through."""

    def test_extract_message_history_edits_list_in_place(self) -> None:
        """The chat message history exposes the request's own messages list."""
        messages = [{"role": "user", "content": "hi"}]
        request_body = {"messages": messages}
        message_history = _CHAT_API_FORMAT.extract_message_history(request_body)
        assert message_history.messages is messages
        message_history.messages.append({"role": "system", "content": "added"})
        message_history.apply_to_request()
        assert request_body == {
            "messages": [{"role": "user", "content": "hi"}, {"role": "system", "content": "added"}]
        }
        assert request_body["messages"] is messages

    @pytest.mark.parametrize(
        "request_body",
        [{}, {"messages": "text"}, {"messages": {"a": 1}}, ["messages"]],
    )
    def test_missing_or_invalid_messages_returns_none(self, request_body: dict | list) -> None:
        """A request body without a messages list returns no editable history."""
        assert _CHAT_API_FORMAT.extract_message_history(request_body) is None

    def test_retry_stream_combiner_passes_bytes_unchanged(self) -> None:
        """The chat retry stream combiner forwards approved bytes untouched."""
        combiner = _CHAT_API_FORMAT.create_retry_stream_combiner()
        raw_bytes = b'data: {"choices": []}\n\ndata: [DO'
        assert combiner.feed_approved_bytes(raw_bytes) is raw_bytes
        combiner.begin_retry()
        assert combiner.feed_approved_bytes(b"data: x\n\n") == b"data: x\n\n"
        assert combiner.flush_closing_events() == b""


class TestChatCompletionsFormatOperations:
    """Chat completions prompt extraction, streaming, usage, assembly, and overrides."""

    _SAMPLE_CHAT_EVENTS: list[dict] = [
        {"id": "x", "object": "chat.completion.chunk", "choices": [{"delta": {"content": "hi"}, "text": "t"}]}
    ]

    def test_extract_prompt_text_includes_string_and_text_blocks_only(self) -> None:
        """Chat prompt text includes string content and text blocks, excluding reasoning."""
        request_body = {
            "messages": [
                {"role": "system", "content": "sys"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "a"},
                        {"type": "image_url", "image_url": {}},
                        {"type": "text", "text": "b"},
                    ],
                },
                {"role": "assistant", "content": None, "reasoning_content": "hidden"},
            ]
        }
        assert _CHAT_API_FORMAT.extract_prompt_text(request_body) == "sys\nab\n\n"

    def test_extract_prompt_text_without_messages_returns_empty_string(self) -> None:
        """A chat request without a messages field returns empty prompt text."""
        assert _CHAT_API_FORMAT.extract_prompt_text({"model": "x"}) == ""

    def test_extract_prompt_text_with_non_list_messages_raises_value_error(self) -> None:
        """A chat request whose messages field is not a list raises ValueError."""
        with pytest.raises(ValueError, match="messages is not a list"):
            _CHAT_API_FORMAT.extract_prompt_text({"messages": "abc"})

    def test_enable_streaming_sets_stream_and_include_usage(self) -> None:
        """Enabling streaming sets stream to True and include_usage to True."""
        request_body = {"stream": False, "stream_options": {"other": 1}}
        _CHAT_API_FORMAT.enable_streaming(request_body)
        assert request_body == {"stream": True, "stream_options": {"other": 1, "include_usage": True}}

    def test_extract_generated_texts_returns_one_entry_per_choice(self) -> None:
        """Each choice yields its combined generated text."""
        stream_event = _build_chat_stream_chunk(
            _build_chat_choice(
                {"content": "a", "reasoning": "b", "tool_calls": [{"function": {"name": "n", "arguments": "{}"}}]}
            ),
            {"index": 1, "text": "legacy"},
            _build_chat_choice({}),
        )
        assert _CHAT_API_FORMAT.extract_generated_texts(stream_event) == ["abn{}", "legacy", ""]

    def test_extract_generated_texts_ignores_null_fields(self) -> None:
        """Null delta fields contribute empty text."""
        stream_event = _build_chat_stream_chunk(
            _build_chat_choice({"content": None, "reasoning": None, "reasoning_content": "rc", "tool_calls": None})
        )
        assert _CHAT_API_FORMAT.extract_generated_texts(stream_event) == ["rc"]

    def test_extract_token_usage_reads_counts(self) -> None:
        """Usage counts map prompt, completion, and total tokens."""
        stream_event = {
            "usage": {
                "prompt_tokens": _PROMPT_TOKENS,
                "completion_tokens": _COMPLETION_TOKENS,
                "total_tokens": _TOTAL_TOKENS,
            }
        }
        assert _CHAT_API_FORMAT.extract_token_usage(stream_event) == TokenUsage(
            _PROMPT_TOKENS, _COMPLETION_TOKENS, _TOTAL_TOKENS
        )

    @pytest.mark.parametrize("stream_event", [{}, {"usage": None}, {"choices": []}])
    def test_extract_token_usage_returns_none_when_absent(self, stream_event: dict) -> None:
        """Events without a usage dict return None."""
        assert _CHAT_API_FORMAT.extract_token_usage(stream_event) is None

    def test_assemble_non_streaming_response_builds_chat_completion(self) -> None:
        """Chat events assemble into a chat.completion JSON response."""
        response = json.loads(_CHAT_API_FORMAT.assemble_non_streaming_response(self._SAMPLE_CHAT_EVENTS))
        assert response["object"] == "chat.completion"
        assert response["choices"][0]["message"]["content"] == "thi"

    def test_assemble_non_streaming_response_with_no_events_returns_none(self) -> None:
        """An empty event list returns None."""
        assert _CHAT_API_FORMAT.assemble_non_streaming_response([]) is None

    def test_convert_request_overrides_copies_as_always_apply(self) -> None:
        """Chat format returns a copy of overrides in always_apply."""
        overrides = {
            "max_tokens": 5,
            "max_output_tokens": 9,
            "reasoning_effort": "low",
            "reasoning": {"effort": "high"},
        }
        converted = _CHAT_API_FORMAT.convert_request_overrides(overrides)
        assert converted == RequestOverrides(overrides, {})
        assert converted.always_apply is not overrides
