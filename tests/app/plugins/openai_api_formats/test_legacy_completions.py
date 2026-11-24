"""Tests for the OpenAI legacy completions API format."""

import json

import pytest

from app.plugins.openai_api_formats.base import RequestOverrides, TokenUsage
from app.plugins.openai_api_formats.legacy_completions import LegacyCompletionsAPIFormat

_LEGACY_COMPLETIONS_API_FORMAT = LegacyCompletionsAPIFormat()

_PROMPT_TOKENS = 12


class TestLegacyCompletionsAPIFormat:
    """Legacy completions prompt extraction, streaming, usage, assembly, and overrides."""

    @pytest.mark.parametrize("prompt, expected", [("one", "one"), (["a", "b"], "ab")])
    def test_extract_prompt_text_handles_string_and_list(
        self, prompt: str | list, expected: str
    ) -> None:
        """Legacy completions prompt text is the prompt string or its joined items."""
        assert _LEGACY_COMPLETIONS_API_FORMAT.extract_prompt_text({"prompt": prompt}) == expected

    def test_enable_streaming_sets_stream_and_include_usage(self) -> None:
        """Enabling streaming sets stream to True and include_usage to True."""
        request_body = {"stream": False, "stream_options": {"other": 1}}
        _LEGACY_COMPLETIONS_API_FORMAT.enable_streaming(request_body)
        assert request_body == {"stream": True, "stream_options": {"other": 1, "include_usage": True}}

    def test_extract_generated_texts_without_choices_returns_empty_list(self) -> None:
        """An event without choices produces no generated text entries."""
        assert _LEGACY_COMPLETIONS_API_FORMAT.extract_generated_texts({"usage": {}}) == []

    def test_extract_generated_texts_reads_choice_text(self) -> None:
        """Each legacy choice yields its text field."""
        stream_event = {"choices": [{"index": 0, "text": "hello"}, {"index": 1, "text": "world"}]}
        assert _LEGACY_COMPLETIONS_API_FORMAT.extract_generated_texts(stream_event) == [
            "hello",
            "world",
        ]

    def test_extract_token_usage_defaults_missing_counts_to_zero(self) -> None:
        """Missing usage counts default to zero."""
        assert _LEGACY_COMPLETIONS_API_FORMAT.extract_token_usage(
            {"usage": {"prompt_tokens": _PROMPT_TOKENS}}
        ) == TokenUsage(_PROMPT_TOKENS, 0, 0)

    def test_assemble_non_streaming_response_builds_text_completion(self) -> None:
        """Legacy events assemble into a text_completion JSON response."""
        response = json.loads(
            _LEGACY_COMPLETIONS_API_FORMAT.assemble_non_streaming_response(
                [{"id": "y", "choices": [{"text": "abc"}]}]
            )
        )
        assert response["object"] == "text_completion"
        assert response["choices"][0]["text"] == "abc"

    def test_assemble_non_streaming_response_with_no_events_returns_none(self) -> None:
        """An empty event list returns None."""
        assert _LEGACY_COMPLETIONS_API_FORMAT.assemble_non_streaming_response([]) is None

    def test_convert_request_overrides_copies_as_always_apply(self) -> None:
        """Legacy completions format returns a copy of overrides in always_apply."""
        overrides = {
            "max_tokens": 5,
            "max_output_tokens": 9,
            "reasoning_effort": "low",
            "reasoning": {"effort": "high"},
        }
        converted = _LEGACY_COMPLETIONS_API_FORMAT.convert_request_overrides(overrides)
        assert converted == RequestOverrides(overrides, {})
        assert converted.always_apply is not overrides
