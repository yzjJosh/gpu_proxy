"""Tests for plugins.utils module."""

import json

import pytest

from app.config import ConfigError
from app.plugins.utils import (
    SSEByteDecoder,
    assemble_non_streaming_response,
    decode_json_object,
    is_tool_call_excluded,
    parse_tool_call_exclusions,
)


class TestAssembleNonStreamingResponse:
    """Assemble SSE events into a complete non-streaming JSON response."""

    def test_empty_events_returns_none(self) -> None:
        """Empty event list returns None."""
        result = assemble_non_streaming_response([])
        assert result is None

    def test_content_only_assembles_correctly(self) -> None:
        """Content-only delta produces a message with content field."""
        events = [
            {"id": "abc", "object": "chat.completion.chunk", "created": 1, "model": "gpt-4"},
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "hello"},
                    }
                ],
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": " world"},
                        "finish_reason": "stop",
                    }
                ],
            },
        ]

        result = assemble_non_streaming_response(events)
        data = json.loads(result)

        assert data["id"] == "abc"
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["content"] == "hello world"
        assert data["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.parametrize("field", ["reasoning", "reasoning_content"])
    def test_reasoning_and_content_assembled(self, field: str) -> None:
        """Reasoning in either field spelling ends up in reasoning_content."""
        events = [
            {
                "choices": [
                    {"delta": {field: "thinking hard"}},
                ],
            },
            {
                "choices": [
                    {"delta": {"content": "answer"}},
                ],
            },
        ]

        result = assemble_non_streaming_response(events)
        data = json.loads(result)
        msg = data["choices"][0]["message"]

        assert msg.get("reasoning_content") == "thinking hard"
        assert msg["content"] == "answer"

    def test_tool_calls_assembled_by_index(self) -> None:
        """Tool call deltas are merged by index and placed in message.tool_calls."""
        events = [
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "tc1",
                                    "type": "function",
                                    "function": {"name": "read"},
                                }
                            ]
                        },
                    },
                ],
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": '{"path": "/f"}'},
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    },
                ],
            },
        ]

        result = assemble_non_streaming_response(events)
        data = json.loads(result)
        tool_calls = data["choices"][0]["message"]["tool_calls"]

        assert len(tool_calls) == 1
        assert tool_calls[0]["id"] == "tc1"
        assert tool_calls[0]["type"] == "function"
        assert tool_calls[0]["function"]["name"] == "read"
        assert tool_calls[0]["function"]["arguments"] == '{"path": "/f"}'
        assert data["choices"][0]["finish_reason"] == "tool_calls"

    def test_empty_delta_produces_empty_content(self) -> None:
        """No content/reasoning/tool_calls produces a message with empty content."""
        events = [{"choices": [{"finish_reason": "stop"}]}]

        result = assemble_non_streaming_response(events)
        data = json.loads(result)

        assert data["choices"][0]["message"]["content"] == ""

    def test_object_stripped_of_chunk_suffix(self) -> None:
        """Object field from chunk events is stripped of .chunk suffix."""
        events = [
            {"object": "chat.completion.chunk"},
            {"choices": [{"delta": {"content": "x"}}]},
        ]

        result = assemble_non_streaming_response(events)
        data = json.loads(result)

        assert data["object"] == "chat.completion"

    def test_response_id_from_first_event(self) -> None:
        """Response ID is taken from the first event that contains it."""
        events = [
            {"id": "first", "choices": [{"delta": {"content": "a"}}]},
            {"id": "second", "choices": [{"delta": {"content": "b"}}]},
        ]

        result = assemble_non_streaming_response(events)
        data = json.loads(result)

        assert data["id"] == "first"

    def test_multiple_tool_call_indices(self) -> None:
        """Multiple tool calls at different indices are all preserved."""
        events = [
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "tc0",
                                    "function": {"name": "a"},
                                },
                                {
                                    "index": 1,
                                    "id": "tc1",
                                    "function": {"name": "b"},
                                },
                            ],
                        },
                    },
                ],
            },
        ]

        result = assemble_non_streaming_response(events)
        data = json.loads(result)
        tool_calls = data["choices"][0]["message"]["tool_calls"]

        assert len(tool_calls) == 2
        assert tool_calls[0]["id"] == "tc0"
        assert tool_calls[1]["id"] == "tc1"

    def test_mixed_none_and_valid_events(self) -> None:
        """None entries in events list are skipped."""
        events = [
            {"id": "r1", "choices": [{"delta": {"content": "hi"}}]},
            None,
            {"choices": [{"delta": {"content": " there"}}]},
        ]

        result = assemble_non_streaming_response(events)
        data = json.loads(result)

        assert data["id"] == "r1"
        assert data["choices"][0]["message"]["content"] == "hi there"

    def test_returns_bytes(self) -> None:
        """Function returns bytes, not a string."""
        events = [{"choices": [{"delta": {"content": "x"}}]}]
        result = assemble_non_streaming_response(events)
        assert isinstance(result, bytes)

    def test_default_object_and_created_fields(self) -> None:
        """Missing object defaults to chat.completion, created defaults to 0."""
        events = [{"choices": [{"delta": {"content": "x"}}]}]

        result = assemble_non_streaming_response(events)
        data = json.loads(result)

        assert data["object"] == "chat.completion"
        assert data["created"] == 0

    def test_text_completion_format_with_is_chat_false(self) -> None:
        """is_chat=False produces text format instead of message.format."""
        events = [
            {"id": "abc", "object": "text_completion.chunk"},
            {"choices": [{"delta": {"content": "def"}}]},
            {"choices": [{"delta": {"content": " fibonacci"}, "finish_reason": "stop"}]},
        ]

        result = assemble_non_streaming_response(events, is_chat=False)
        data = json.loads(result)

        assert data["id"] == "abc"
        assert data["object"] == "text_completion"
        assert data["choices"][0]["text"] == "def fibonacci"
        assert data["choices"][0]["finish_reason"] == "stop"
        assert "message" not in data["choices"][0]

    def test_text_completion_empty_content(self) -> None:
        """is_chat=False with no content produces empty text field."""
        events = [{"choices": [{"finish_reason": "stop"}]}]

        result = assemble_non_streaming_response(events, is_chat=False)
        data = json.loads(result)

        assert data["choices"][0]["text"] == ""


class TestDecodeJsonObject:
    """Decode a request or response body into a JSON object."""

    def test_object_body_decodes_to_mapping(self) -> None:
        """A JSON object body decodes to a dict carrying its keys and values."""
        result = decode_json_object(b'{"model": "gpt-4", "stream": true, "n": 2}')
        assert result == {"model": "gpt-4", "stream": True, "n": 2}

    def test_nested_object_preserves_structure(self) -> None:
        """Objects and arrays nested inside the body survive decoding intact."""
        result = decode_json_object(b'{"messages": [{"role": "user", "content": "hi"}]}')
        assert result == {"messages": [{"role": "user", "content": "hi"}]}

    @pytest.mark.parametrize(
        "body",
        [
            pytest.param(b"[1, 2, 3]", id="array"),
            pytest.param(b'"a string"', id="string"),
            pytest.param(b"42", id="integer"),
            pytest.param(b"3.5", id="float"),
            pytest.param(b"true", id="boolean"),
            pytest.param(b"null", id="null"),
        ],
    )
    def test_non_object_json_returns_none(self, body: bytes) -> None:
        """Valid JSON that decodes to something other than an object returns None."""
        assert decode_json_object(body) is None

    @pytest.mark.parametrize(
        "body",
        [
            pytest.param(b"", id="empty"),
            pytest.param(b"{", id="truncated-object"),
            pytest.param(b"{'single': 'quotes'}", id="single-quoted"),
            pytest.param(b"not json at all", id="plain-text"),
        ],
    )
    def test_malformed_body_returns_none(self, body: bytes) -> None:
        """Bytes that are not parseable JSON return None."""
        assert decode_json_object(body) is None

    @pytest.mark.parametrize(
        "body",
        [
            pytest.param(None, id="none"),
            pytest.param(42, id="int"),
            pytest.param(["already", "decoded"], id="list"),
        ],
    )
    def test_non_bytes_body_returns_none(self, body: int | list[str] | None) -> None:
        """A body of a type the JSON parser cannot accept returns None."""
        assert decode_json_object(body) is None

    def test_invalid_utf8_propagates(self) -> None:
        """A body carrying a byte the detected codec cannot decode raises rather than returning None."""
        with pytest.raises(UnicodeDecodeError):
            decode_json_object(b'{"prompt": "\xff"}')


class TestSSEByteDecoder:
    """SSEByteDecoder handles chunked TCP boundaries correctly."""

    def test_simple_event(self) -> None:
        decoder = SSEByteDecoder()
        events = decoder.feed(b'data: {"content": "hello"}\n\n')
        assert len(events) == 1
        event_type, data, _event_id, _raw = events[0]
        assert event_type == "message"
        assert data == '{"content": "hello"}'

    def test_split_across_chunks(self) -> None:
        decoder = SSEByteDecoder()
        events = decoder.feed(b'data: {"')
        assert len(events) == 0
        events = decoder.feed(b'content": "hello"}\n\n')
        assert len(events) == 1
        assert events[0][1] == '{"content": "hello"}'

    def test_close_flushes_pending(self) -> None:
        decoder = SSEByteDecoder()
        decoder.feed(b'data: {"content": "x"}\n')
        events, _leftover = decoder.close()
        assert len(events) == 1
        assert events[0][1] == '{"content": "x"}'

    def test_close_returns_leftover(self) -> None:
        decoder = SSEByteDecoder()
        decoder.feed(b'partial')
        events, leftover = decoder.close()
        assert len(events) == 0
        assert leftover == b'partial'

    def test_multiline_data(self) -> None:
        decoder = SSEByteDecoder()
        events = decoder.feed(b'data: line1\ndata: line2\n\n')
        assert len(events) == 1
        assert events[0][1] == "line1\nline2"

    def test_feed_after_close_raises_runtime_error(self) -> None:
        decoder = SSEByteDecoder()
        decoder.close()
        with pytest.raises(RuntimeError, match="Cannot feed a closed decoder"):
            decoder.feed(b"hello")


class TestParseToolCallExclusions:
    """parse_tool_call_exclusions compiles valid entries and rejects malformed config."""

    def test_name_and_args_both_must_match(self) -> None:
        """With both patterns, a call is excluded only when name and args both match."""
        exclusions = parse_tool_call_exclusions(
            [{"tool_name_regex": "read", "tool_args_regex": "skills?\\.md"}],
            "Test",
            "tool_call_test_exclusions",
        )
        assert is_tool_call_excluded("read", "path=skills.md", exclusions)
        assert not is_tool_call_excluded("write", "path=skills.md", exclusions)
        assert not is_tool_call_excluded("read", "path=other.txt", exclusions)

    def test_name_only_wildcards_args(self) -> None:
        """A name-only entry excludes by name regardless of arguments."""
        exclusions = parse_tool_call_exclusions(
            [{"tool_name_regex": "read"}], "Test", "tool_call_test_exclusions"
        )
        assert is_tool_call_excluded("read", "any arguments", exclusions)
        assert not is_tool_call_excluded("write", "any arguments", exclusions)

    def test_args_only_wildcards_name(self) -> None:
        """An args-only entry excludes by arguments regardless of name."""
        exclusions = parse_tool_call_exclusions([{"tool_args_regex": "poll"}], "Test", "tool_call_test_exclusions")
        assert is_tool_call_excluded("any_tool", '{"action": "poll"}', exclusions)
        assert not is_tool_call_excluded("any_tool", '{"action": "run"}', exclusions)

    def test_empty_list_returns_empty(self) -> None:
        """An empty list is valid and yields no exclusions."""
        assert parse_tool_call_exclusions([], "Test", "tool_call_test_exclusions") == []

    @pytest.mark.parametrize(
        "raw",
        [
            pytest.param("read", id="string"),
            pytest.param(None, id="none"),
            pytest.param(42, id="int"),
        ],
    )
    def test_scalar_or_none_input_raises_config_error(self, raw: str | int | None) -> None:
        """Input that is not a list fails early instead of silently returning empty."""
        with pytest.raises(ConfigError, match="must be a list"):
            parse_tool_call_exclusions(raw, "Test", "tool_call_test_exclusions")

    def test_dict_input_raises_config_error(self) -> None:
        """A dict input fails early instead of silently returning empty."""
        with pytest.raises(ConfigError, match="must be a list"):
            parse_tool_call_exclusions(
                {"tool_name_regex": "read"}, "Test", "tool_call_test_exclusions"
            )

    def test_non_dict_entry_raises_config_error(self) -> None:
        """A list entry that is not a dict fails early."""
        with pytest.raises(ConfigError, match="must be a dict"):
            parse_tool_call_exclusions(["not-a-dict"], "Test", "tool_call_test_exclusions")

    def test_entry_with_no_patterns_raises_config_error(self) -> None:
        """An entry defining neither pattern is a no-op and fails early."""
        with pytest.raises(ConfigError, match="tool_name_regex or tool_args_regex"):
            parse_tool_call_exclusions([{}], "Test", "tool_call_test_exclusions")

    def test_non_string_name_pattern_raises_config_error(self) -> None:
        """A non-string name pattern fails early instead of being dropped."""
        with pytest.raises(ConfigError, match="tool_name_regex must be a string"):
            parse_tool_call_exclusions([{"tool_name_regex": 42}], "Test", "tool_call_test_exclusions")

    def test_non_string_args_pattern_raises_config_error(self) -> None:
        """A non-string args pattern fails early instead of being dropped."""
        with pytest.raises(ConfigError, match="tool_args_regex must be a string"):
            parse_tool_call_exclusions([{"tool_args_regex": ["x"]}], "Test", "tool_call_test_exclusions")

    def test_invalid_name_regex_raises_config_error(self) -> None:
        """An invalid name regex fails early instead of warning and continuing."""
        with pytest.raises(ConfigError, match="not a valid regex"):
            parse_tool_call_exclusions([{"tool_name_regex": "(unclosed"}], "Test", "tool_call_test_exclusions")

    def test_invalid_args_regex_raises_config_error(self) -> None:
        """An invalid args regex fails early instead of warning and continuing."""
        with pytest.raises(ConfigError, match="not a valid regex"):
            parse_tool_call_exclusions([{"tool_args_regex": "[bad"}], "Test", "tool_call_test_exclusions")

    def test_entry_matches_method_checks_set_patterns(self) -> None:
        """ToolCallExclusion.matches checks every pattern configured on the entry."""
        exclusions = parse_tool_call_exclusions(
            [
                {"tool_name_regex": "^task$", "tool_args_regex": '"action": "status"'},
                {"tool_name_regex": "^read$"},
                {"tool_args_regex": "poll"},
            ],
            "Test",
            "tool_call_test_exclusions",
        )
        assert exclusions[0].matches("task", '{"action": "status"}')
        assert not exclusions[0].matches("task", '{"action": "run"}')
        assert not exclusions[0].matches("other", '{"action": "status"}')
        assert exclusions[1].matches("read", "{}")
        assert not exclusions[1].matches("write", "{}")
        assert exclusions[2].matches("any", "poll")
        assert not exclusions[2].matches("any", "run")

    def test_unrecognized_key_rejected_by_default(self) -> None:
        """Unknown keys fail fast with plugin, config key, index, and key name."""
        with pytest.raises(
            ConfigError,
            match=r"Test: tool_call_test_exclusions\[0\].*unrecognized key.*max_dup_exclusion_count",
        ):
            parse_tool_call_exclusions(
                [{"tool_name_regex": "read", "max_dup_exclusion_count": 3}],
                "Test",
                "tool_call_test_exclusions",
            )

    def test_extra_allowed_keys_accepts_opted_in_key_only(self) -> None:
        """Opted-in extra keys pass while other unknown keys still raise ConfigError."""
        exclusions = parse_tool_call_exclusions(
            [{"tool_name_regex": "read", "max_dup_exclusion_count": 3}],
            "Test",
            "tool_call_test_exclusions",
            extra_allowed_keys=frozenset({"max_dup_exclusion_count"}),
        )
        assert len(exclusions) == 1
        assert exclusions[0].matches("read_file", "{}")

        with pytest.raises(
            ConfigError,
            match=r"Test: tool_call_test_exclusions\[0\].*unrecognized key.*typo_key",
        ):
            parse_tool_call_exclusions(
                [{"tool_name_regex": "read", "typo_key": 1}],
                "Test",
                "tool_call_test_exclusions",
                extra_allowed_keys=frozenset({"max_dup_exclusion_count"}),
            )
