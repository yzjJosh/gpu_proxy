"""Tests for shared OpenAI API format helpers and path lookups."""

import pytest

from app.plugins.openai_api_formats.base import (
    OpenAIAPIFormat,
    build_tool_call_update,
    extract_chat_delta_reasoning,
    extract_chat_tool_call_fields,
    find_api_format_by_exact_path,
    find_api_format_by_path_prefix,
    find_matching_path_prefix,
    standardize_json_arguments,
)
from app.plugins.openai_api_formats.chat_completions import ChatCompletionsAPIFormat
from app.plugins.openai_api_formats.legacy_completions import LegacyCompletionsAPIFormat
from app.plugins.openai_api_formats.responses import ResponsesAPIFormat

_CHAT_API_FORMAT = ChatCompletionsAPIFormat()
_LEGACY_COMPLETIONS_API_FORMAT = LegacyCompletionsAPIFormat()
_RESPONSES_API_FORMAT = ResponsesAPIFormat()
_ALL_API_FORMATS = (
    _CHAT_API_FORMAT,
    _LEGACY_COMPLETIONS_API_FORMAT,
    _RESPONSES_API_FORMAT,
)


class TestStandardizeJsonArguments:
    """Valid JSON tool arguments are reformatted, and non-JSON text stays unchanged."""

    @pytest.mark.parametrize(
        "arguments, expected",
        [
            ('{"x":1}', '{"x": 1}'),
            ("[1,2]", "[1, 2]"),
            ("not json", "not json"),
            ("", ""),
        ],
    )
    def test_reformats_only_valid_json(self, arguments: str, expected: str) -> None:
        """Only valid JSON strings are reformatted."""
        assert standardize_json_arguments(arguments) == expected


class TestBuildToolCallUpdate:
    """Standardized tool call update dicts combine name, arguments, and id."""

    def test_builds_tool_call_update_dict(self) -> None:
        """Name and arguments concatenate into text while preserving separate function fields."""
        assert build_tool_call_update("get_weather", '{"city": "Paris"}', "call_1") == {
            "type": "tool_call",
            "text": 'get_weather{"city": "Paris"}',
            "tool_call_id": "call_1",
            "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
        }


class TestChatDeltaFieldHelpers:
    """Chat delta helpers read reasoning and tool call fields."""

    @pytest.mark.parametrize(
        "delta, expected",
        [
            ({"reasoning": "a"}, "a"),
            ({"reasoning_content": "b"}, "b"),
            ({"reasoning": "a", "reasoning_content": "b"}, "a"),
            ({"reasoning": "", "reasoning_content": "b"}, "b"),
            ({"reasoning": None, "reasoning_content": None}, None),
            ({}, None),
        ],
    )
    def test_extract_chat_delta_reasoning(self, delta: dict, expected: str | None) -> None:
        """Reasoning falls back from reasoning to reasoning_content."""
        assert extract_chat_delta_reasoning(delta) == expected

    @pytest.mark.parametrize(
        "tool_call, expected",
        [
            ({"function": {"name": "run", "arguments": "{}"}}, ("run", "{}")),
            ({"function": {"name": None, "arguments": None}}, ("", "")),
            ({"function": None}, ("", "")),
            ({}, ("", "")),
        ],
    )
    def test_extract_chat_tool_call_fields(
        self, tool_call: dict, expected: tuple[str, str]
    ) -> None:
        """Missing or null function fields return empty strings."""
        assert extract_chat_tool_call_fields(tool_call) == expected


class TestFindAPIFormat:
    """Incoming request paths resolve to the matching API format or path prefix."""

    @pytest.mark.parametrize(
        "request_path, expected_format",
        [
            ("/v1/chat/completions", _CHAT_API_FORMAT),
            ("/v1/chat/completions/input_tokens", _CHAT_API_FORMAT),
            ("/v1/completions?x=1", _LEGACY_COMPLETIONS_API_FORMAT),
            ("/v1/responses/abc/cancel", _RESPONSES_API_FORMAT),
            ("/v1/embeddings", None),
        ],
    )
    def test_find_by_path_prefix(
        self, request_path: str, expected_format: OpenAIAPIFormat | None
    ) -> None:
        """A request path finds the API format whose endpoint path it starts with."""
        assert find_api_format_by_path_prefix(_ALL_API_FORMATS, request_path) is expected_format

    @pytest.mark.parametrize(
        "request_path, expected_format",
        [
            ("/v1/chat/completions", _CHAT_API_FORMAT),
            ("/v1/responses", _RESPONSES_API_FORMAT),
            ("/v1/chat/completions?x", None),
            ("/v1/responses/input_tokens", None),
        ],
    )
    def test_find_by_exact_path(
        self, request_path: str, expected_format: OpenAIAPIFormat | None
    ) -> None:
        """A request path finds only the API format with that exact endpoint path."""
        assert find_api_format_by_exact_path(_ALL_API_FORMATS, request_path) is expected_format

    @pytest.mark.parametrize(
        "request_path, expected_prefix",
        [
            ("/v1/embeddings", "/v1/embeddings"),
            ("/v1/rerank?x=1", "/v1/rerank"),
            ("/v1/chat/completions", None),
        ],
    )
    def test_find_matching_path_prefix(
        self, request_path: str, expected_prefix: str | None
    ) -> None:
        """A request path finds the first candidate prefix it starts with."""
        prefixes = ("/v1/embeddings", "/v1/rerank", "/v1/embeddings/x")
        assert find_matching_path_prefix(prefixes, request_path) == expected_prefix
