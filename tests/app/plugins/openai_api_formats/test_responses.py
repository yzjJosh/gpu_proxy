"""Tests for the OpenAI Responses API format."""

import copy
import json
from pathlib import Path
from typing import Callable
from urllib import request as urllib_request

import pytest

from app.config import ProxyConfig, load_config
from app.plugins.openai_api_formats.base import ParsedStreamEvent, RequestOverrides, TokenUsage
from app.plugins.openai_api_formats.responses import ResponsesAPIFormat
from tests.app.conftest import ResponsesStream, parse_sse_payloads, responses_event

_RESPONSES_API_FORMAT = ResponsesAPIFormat()
_FIXTURES_DIR = Path(__file__).parents[1] / "fixtures" / "responses"
_BACKEND_FIXTURE_NAMES = ("vllm", "sglang", "llamacpp")
_NUMBERED_BACKEND_FIXTURES = ("vllm", "sglang")
_UNNUMBERED_BACKEND_FIXTURES = ("llamacpp",)

_INPUT_TOKENS = 384
_OUTPUT_TOKENS = 55
_TOTAL_TOKENS = 439

# All Responses API SSE event types from the specification.
_KNOWN_RESPONSES_EVENT_TYPES = frozenset({
    "response.created",
    "response.in_progress",
    "response.completed",
    "response.incomplete",
    "response.failed",
    "response.output_item.added",
    "response.output_item.done",
    "response.content_part.added",
    "response.content_part.done",
    "response.reasoning_part.added",
    "response.reasoning_part.done",
    "response.reasoning_summary_part.added",
    "response.reasoning_summary_part.done",
    "response.output_text.delta",
    "response.output_text.done",
    "response.reasoning_text.delta",
    "response.reasoning_text.done",
    "response.reasoning_summary_text.delta",
    "response.reasoning_summary_text.done",
    "response.function_call_arguments.delta",
    "response.function_call_arguments.done",
    "response.output_text.annotation.added",
    "response.refusal.delta",
    "response.refusal.done",
    "error",
})


def _read_stream_fixture(backend_name: str) -> bytes:
    """Read the raw bytes of one checked-in Responses SSE fixture."""
    return (_FIXTURES_DIR / f"{backend_name}.sse").read_bytes()


def _split_sse_events(raw_stream: bytes) -> list[bytes]:
    """Split a raw SSE stream into individual event byte blocks."""
    return [chunk + b"\n\n" for chunk in raw_stream.split(b"\n\n") if chunk.strip()]


def _build_reasoning_input_item(text: str, item_id: str = "", status: str = "") -> dict:
    """Build a Responses reasoning input item with the given text."""
    item: dict = {
        "type": "reasoning",
        "summary": [],
        "content": [{"type": "reasoning_text", "text": text}],
    }
    if item_id:
        item["id"] = item_id
    if status:
        item["status"] = status
    return item


def _build_message_input_item(role: str, text: str) -> dict:
    """Build a Responses message input item with the given role and text."""
    block_type = "output_text" if role == "assistant" else "input_text"
    return {
        "type": "message",
        "role": role,
        "content": [{"type": block_type, "text": text}],
    }


def _build_function_call_input_item(
    call_id: str, name: str = "run", arguments: str = "{}", item_id: str = ""
) -> dict:
    """Build a Responses function_call input item."""
    item = {"type": "function_call", "call_id": call_id, "name": name, "arguments": arguments}
    if item_id:
        item["id"] = item_id
    return item


def _build_function_call_output_input_item(call_id: str, output: str) -> dict:
    """Build a Responses function_call_output input item."""
    return {"type": "function_call_output", "call_id": call_id, "output": output}


def _extract_chat_messages(input_value: list | str, instructions: str | None = None) -> list:
    """Extract chat-style messages from a Responses request body."""
    request_body: dict = {"input": input_value}
    if instructions is not None:
        request_body["instructions"] = instructions
    return _RESPONSES_API_FORMAT.extract_message_history(request_body).messages


# ── Message history extraction ──


_INPUT_TO_CHAT_MESSAGES_CASES = [
    pytest.param("hello", None, [{"role": "user", "content": "hello"}], id="string-input"),
    pytest.param(
        [_build_message_input_item("user", "hi")],
        "be kind",
        [
            {"role": "system", "content": "be kind"},
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        ],
        id="instructions-first",
    ),
    pytest.param(
        [{"role": "user", "content": "plain"}],
        "",
        [{"role": "user", "content": "plain"}],
        id="empty-instructions",
    ),
    pytest.param(
        [
            {"role": "system", "content": "s"},
            {"role": "developer", "content": [{"type": "input_text", "text": "d"}]},
            {"role": "user", "content": "u"},
        ],
        None,
        [
            {"role": "system", "content": "s"},
            {"role": "developer", "content": [{"type": "text", "text": "d"}]},
            {"role": "user", "content": "u"},
        ],
        id="role-only-items",
    ),
    pytest.param(
        [
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": "x"},
                    {"type": "input_text", "text": "t"},
                ],
            }
        ],
        None,
        [
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": "x"},
                    {"type": "text", "text": "t"},
                ],
            }
        ],
        id="other-blocks-kept",
    ),
    pytest.param(
        [
            _build_reasoning_input_item("think"),
            _build_message_input_item("assistant", "say"),
            _build_function_call_input_item("c1"),
            _build_function_call_input_item("c2", "two", '{"a":1}'),
        ],
        None,
        [
            {
                "role": "assistant",
                "reasoning_content": "think",
                "content": "say",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "run", "arguments": "{}"}},
                    {
                        "id": "c2",
                        "type": "function",
                        "function": {"name": "two", "arguments": '{"a":1}'},
                    },
                ],
            }
        ],
        id="one-assistant-group",
    ),
    pytest.param(
        [{"role": "assistant", "content": "string answer"}],
        None,
        [{"role": "assistant", "content": "string answer"}],
        id="role-only-assistant-string",
    ),
    pytest.param(
        [{"type": "message", "role": "assistant", "content": "typed string"}],
        None,
        [{"role": "assistant", "content": "typed string"}],
        id="typed-assistant-string",
    ),
    pytest.param(
        [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "first"},
                    {"type": "output_text", "text": "second"},
                ],
            }
        ],
        None,
        [{"role": "assistant", "content": "first"}],
        id="assistant-first-block",
    ),
    pytest.param(
        [
            _build_message_input_item("assistant", "a"),
            {"role": "assistant", "content": []},
            _build_function_call_input_item("c1"),
        ],
        None,
        [
            {"role": "assistant", "content": "a"},
            {
                "role": "assistant",
                "content": [],
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "run", "arguments": "{}"}}
                ],
            },
        ],
        id="assistant-without-text-starts-group",
    ),
    pytest.param(
        [_build_function_call_input_item("c1"), _build_function_call_input_item("c2")],
        None,
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "run", "arguments": "{}"}},
                    {"id": "c2", "type": "function", "function": {"name": "run", "arguments": "{}"}},
                ],
            }
        ],
        id="group-without-assistant-item-has-no-content",
    ),
    pytest.param(
        [_build_message_input_item("assistant", "only")],
        None,
        [{"role": "assistant", "content": "only"}],
        id="group-without-reasoning-item",
    ),
    pytest.param(
        [
            {"type": "reasoning", "summary": [], "content": []},
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "sum"}]},
        ],
        None,
        [
            {"role": "assistant", "reasoning_content": ""},
            {"role": "assistant", "reasoning_content": "sum"},
        ],
        id="empty-reasoning-then-second-reasoning",
    ),
    pytest.param(
        [_build_message_input_item("assistant", "a"), _build_message_input_item("assistant", "b")],
        None,
        [{"role": "assistant", "content": "a"}, {"role": "assistant", "content": "b"}],
        id="second-message-starts-group",
    ),
    pytest.param(
        [_build_message_input_item("assistant", "a"), _build_reasoning_input_item("late")],
        None,
        [{"role": "assistant", "content": "a", "reasoning_content": "late"}],
        id="reasoning-joins-group-without-reasoning",
    ),
    pytest.param(
        [
            _build_function_call_input_item("c1"),
            _build_function_call_output_input_item("c1", "out"),
            _build_function_call_input_item("c2"),
        ],
        None,
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "run", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "out"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "c2", "type": "function", "function": {"name": "run", "arguments": "{}"}}
                ],
            },
        ],
        id="tool-result-ends-group",
    ),
    pytest.param(
        [
            {
                "type": "function_call_output",
                "call_id": "c1",
                "output": [{"type": "input_text", "text": "o"}],
            }
        ],
        None,
        [{"role": "tool", "tool_call_id": "c1", "content": [{"type": "text", "text": "o"}]}],
        id="tool-output-blocks",
    ),
    pytest.param(
        [{"type": "item_reference", "id": "x"}, 7, "loose"],
        None,
        [{}, 7, "loose"],
        id="opaque-and-non-object-items",
    ),
]


class TestExtractMessageHistory:
    """Responses instructions and input convert into chat-style messages."""

    @pytest.mark.parametrize(
        "input_value, instructions, expected_messages", _INPUT_TO_CHAT_MESSAGES_CASES
    )
    def test_converts_input_shapes_to_chat_messages(
        self, input_value: list | str, instructions: str | None, expected_messages: list
    ) -> None:
        """Each Responses input shape converts into the expected chat messages."""
        assert _extract_chat_messages(input_value, instructions) == expected_messages

    @pytest.mark.parametrize(
        "request_body",
        [{}, {"input": None}, {"input": 5}, {"input": {"role": "user"}}, ["input"]],
    )
    def test_missing_or_invalid_input_returns_none(self, request_body: dict | list) -> None:
        """A request body without a string or list input returns None."""
        assert _RESPONSES_API_FORMAT.extract_message_history(request_body) is None

    def test_non_string_instructions_are_ignored(self) -> None:
        """Non-string instructions do not add a system message."""
        request_body = {"instructions": ["x"], "input": "hi"}
        assert _RESPONSES_API_FORMAT.extract_message_history(request_body).messages == [
            {"role": "user", "content": "hi"}
        ]

    @pytest.mark.parametrize(
        "input_value, instructions, _expected_messages", _INPUT_TO_CHAT_MESSAGES_CASES
    )
    def test_apply_without_edits_leaves_request_body_untouched(
        self, input_value: list | str, instructions: str | None, _expected_messages: list
    ) -> None:
        """Applying an unedited message history leaves the request body untouched."""
        request_body: dict = {"input": input_value, "status_marker": 1}
        if instructions is not None:
            request_body["instructions"] = instructions
        original_body = copy.deepcopy(request_body)
        original_input = request_body["input"]
        _RESPONSES_API_FORMAT.extract_message_history(request_body).apply_to_request()
        assert request_body == original_body
        assert request_body["input"] is original_input
        assert request_body[_RESPONSES_API_FORMAT.history_field_name] is original_input


def _build_sample_conversation_request_body() -> dict:
    """Build a Responses request body with instructions, user, assistant group, and tool result."""
    return {
        "instructions": "rules",
        "input": [
            {"type": "message", "role": "user", "content": "go", "id": "u1"},
            {
                "type": "reasoning",
                "id": "rs1",
                "status": "completed",
                "summary": [{"type": "summary_text", "text": "s"}],
                "content": [{"type": "reasoning_text", "text": "think"}],
            },
            {
                "type": "message",
                "id": "m1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "say", "annotations": []}],
            },
            {
                "type": "function_call",
                "id": "f1",
                "call_id": "c1",
                "name": "read",
                "arguments": '{"path": "a"}',
                "status": "completed",
            },
            {"type": "function_call_output", "id": "o1", "call_id": "c1", "output": "long result"},
        ],
    }


def _edit_and_apply_message_history(
    request_body: dict, edit_messages: Callable[[list], None]
) -> list:
    """Apply an edit callback to the extracted chat messages, write back, and return them."""
    message_history = _RESPONSES_API_FORMAT.extract_message_history(request_body)
    edit_messages(message_history.messages)
    message_history.apply_to_request()
    return message_history.messages


# Positions of user, assistant group, and tool messages in the sample conversation history.
_USER_MESSAGE_INDEX = 1
_ASSISTANT_MESSAGE_INDEX = 2
_TOOL_MESSAGE_INDEX = 3


def _set_assistant_reasoning(messages: list) -> None:
    """Update the reasoning text of the assistant message."""
    messages[_ASSISTANT_MESSAGE_INDEX]["reasoning_content"] = "restored"


def _remove_assistant_reasoning(messages: list) -> None:
    """Remove the reasoning text from the assistant message."""
    messages[_ASSISTANT_MESSAGE_INDEX].pop("reasoning_content")


def _truncate_first_tool_call_arguments(messages: list) -> None:
    """Shorten the arguments of the first tool call."""
    messages[_ASSISTANT_MESSAGE_INDEX]["tool_calls"][0]["function"]["arguments"] = '{"path": "..."}'


def _prune_tool_result_content(messages: list) -> None:
    """Replace the tool result content with a pruned placeholder."""
    messages[_TOOL_MESSAGE_INDEX]["content"] = "[tool result pruned]"


def _update_user_message_content(messages: list) -> None:
    """Change the text of the user message."""
    messages[_USER_MESSAGE_INDEX]["content"] = "go now"


def _update_assistant_message_content(messages: list) -> None:
    """Change the text of the assistant message."""
    messages[_ASSISTANT_MESSAGE_INDEX]["content"] = "said"


def _insert_system_message_before_assistant(messages: list) -> None:
    """Insert a system message before the assistant turn."""
    messages.insert(_ASSISTANT_MESSAGE_INDEX, {"role": "system", "content": "again"})


def _insert_assistant_tool_turn_before_assistant(messages: list) -> None:
    """Insert a new assistant turn with reasoning and a tool call before the assistant turn."""
    messages.insert(
        _ASSISTANT_MESSAGE_INDEX,
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "why",
            "tool_calls": [
                {"id": "c9", "type": "function", "function": {"name": "grep", "arguments": "{}"}}
            ],
        },
    )


def _append_partial_reply_and_retry_instruction(messages: list) -> None:
    """Append a partial assistant reply and a retry system prompt."""
    messages.extend([
        {"role": "assistant", "content": "partial", "reasoning_content": "r"},
        {"role": "system", "content": "try again"},
    ])


class TestApplyMessageHistoryChanges:
    """Edits made on the chat-style message history write back to the matching Responses items."""

    def test_restoring_reasoning_updates_existing_reasoning_item(self) -> None:
        """Restored reasoning updates the group's reasoning item in place."""
        request_body = _build_sample_conversation_request_body()
        _edit_and_apply_message_history(request_body, _set_assistant_reasoning)
        assert request_body["input"][1] == {
            "type": "reasoning",
            "id": "rs1",
            "status": "completed",
            "summary": [],
            "content": [{"type": "reasoning_text", "text": "restored"}],
        }
        assert request_body["input"][2:] == _build_sample_conversation_request_body()["input"][2:]

    def test_restoring_reasoning_inserts_item_when_group_has_none(self) -> None:
        """Restored reasoning inserts a new reasoning item when the group has none."""
        request_body = {
            "input": [
                _build_message_input_item("user", "u"),
                _build_function_call_input_item("c1", item_id="f1"),
            ]
        }
        message_history = _RESPONSES_API_FORMAT.extract_message_history(request_body)
        message_history.messages[1]["reasoning_content"] = "new"
        message_history.apply_to_request()
        assert request_body["input"] == [
            _build_message_input_item("user", "u"),
            {"type": "reasoning", "summary": [], "content": [{"type": "reasoning_text", "text": "new"}]},
            _build_function_call_input_item("c1", item_id="f1"),
        ]

    def test_restoring_reasoning_places_item_after_leading_empty_assistant_message(self) -> None:
        """Restored reasoning is placed after a leading empty assistant message."""
        empty_message = {"type": "message", "role": "assistant", "content": []}
        request_body = {
            "input": [
                _build_message_input_item("user", "u"),
                empty_message,
                _build_function_call_input_item("c1"),
                _build_function_call_output_input_item("c1", "out"),
            ]
        }
        message_history = _RESPONSES_API_FORMAT.extract_message_history(request_body)
        message_history.messages[1]["reasoning_content"] = "think"
        message_history.apply_to_request()
        assert request_body["input"] == [
            _build_message_input_item("user", "u"),
            empty_message,
            _build_reasoning_input_item("think"),
            _build_function_call_input_item("c1"),
            _build_function_call_output_input_item("c1", "out"),
        ]
        assert _extract_chat_messages(request_body["input"]) == [
            {"role": "user", "content": [{"type": "text", "text": "u"}]},
            {
                "role": "assistant",
                "content": [],
                "reasoning_content": "think",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "run", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "out"},
        ]

    def test_restoring_reasoning_on_second_text_group_keeps_it_in_second_group(self) -> None:
        """Restored reasoning on a text group after another text group stays with its own text."""
        request_body = {
            "input": [
                _build_message_input_item("assistant", "a"),
                _build_message_input_item("assistant", "b"),
            ]
        }
        message_history = _RESPONSES_API_FORMAT.extract_message_history(request_body)
        message_history.messages[1]["reasoning_content"] = "why b"
        message_history.apply_to_request()
        assert request_body["input"] == [
            _build_message_input_item("assistant", "a"),
            _build_message_input_item("assistant", "b"),
            _build_reasoning_input_item("why b"),
        ]
        assert _extract_chat_messages(request_body["input"]) == [
            {"role": "assistant", "content": "a"},
            {"role": "assistant", "content": "b", "reasoning_content": "why b"},
        ]

    def test_restoring_reasoning_on_text_group_after_user_places_item_first(self) -> None:
        """Restored reasoning on a text group after a user message is placed before the text."""
        request_body = {
            "input": [
                _build_message_input_item("user", "u"),
                _build_message_input_item("assistant", "b"),
            ]
        }
        message_history = _RESPONSES_API_FORMAT.extract_message_history(request_body)
        message_history.messages[1]["reasoning_content"] = "why b"
        message_history.apply_to_request()
        assert request_body["input"] == [
            _build_message_input_item("user", "u"),
            _build_reasoning_input_item("why b"),
            _build_message_input_item("assistant", "b"),
        ]
        assert _extract_chat_messages(request_body["input"])[1] == {
            "role": "assistant",
            "content": "b",
            "reasoning_content": "why b",
        }

    def test_restoring_reasoning_into_empty_reasoning_item_keeps_one_group(self) -> None:
        """Restored reasoning written into an empty reasoning item keeps its calls grouped."""
        request_body = {
            "input": [
                _build_message_input_item("user", "u"),
                {"type": "reasoning", "id": "rs", "summary": [], "content": []},
                _build_function_call_input_item("c1"),
                _build_function_call_input_item("c2"),
            ]
        }
        message_history = _RESPONSES_API_FORMAT.extract_message_history(request_body)
        message_history.messages[1]["reasoning_content"] = "full"
        message_history.apply_to_request()
        assert [item["type"] for item in request_body["input"]] == [
            "message",
            "reasoning",
            "function_call",
            "function_call",
        ]
        assert request_body["input"][1] == {
            "type": "reasoning",
            "id": "rs",
            "summary": [],
            "content": [{"type": "reasoning_text", "text": "full"}],
        }
        assert _extract_chat_messages(request_body["input"])[1] == {
            "role": "assistant",
            "reasoning_content": "full",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "run", "arguments": "{}"}},
                {"id": "c2", "type": "function", "function": {"name": "run", "arguments": "{}"}},
            ],
        }

    def test_pruning_reasoning_removes_reasoning_item(self) -> None:
        """Pruning reasoning deletes the reasoning item from the input list."""
        request_body = _build_sample_conversation_request_body()
        _edit_and_apply_message_history(request_body, _remove_assistant_reasoning)
        assert [item["id"] for item in request_body["input"]] == ["u1", "m1", "f1", "o1"]

    def test_pruning_reasoning_after_another_group_keeps_empty_separator_item(self) -> None:
        """Pruning reasoning that separates two assistant groups leaves an empty reasoning item."""
        request_body = {
            "input": [
                _build_message_input_item("user", "u"),
                _build_reasoning_input_item("first thoughts"),
                _build_message_input_item("assistant", "first"),
                _build_reasoning_input_item("second thoughts", "rs2", "completed"),
                _build_function_call_input_item("c1", item_id="f1"),
            ]
        }
        message_history = _RESPONSES_API_FORMAT.extract_message_history(request_body)
        assert len(message_history.messages) == 3
        message_history.messages[2].pop("reasoning_content")
        message_history.apply_to_request()
        assert request_body["input"][3] == {
            "type": "reasoning",
            "id": "rs2",
            "status": "completed",
            "summary": [],
            "content": [{"type": "reasoning_text", "text": ""}],
        }
        re_extracted = _extract_chat_messages(request_body["input"])
        assert len(re_extracted) == 3
        assert re_extracted[2]["reasoning_content"] == ""
        assert re_extracted[2]["tool_calls"][0]["id"] == "c1"

    def test_pruning_already_empty_reasoning_item_leaves_it_unchanged(self) -> None:
        """Pruning an already empty reasoning item keeps the original object."""
        empty_reasoning = {"type": "reasoning", "id": "rs", "summary": [], "content": []}
        request_body = {
            "input": [
                _build_message_input_item("user", "u"),
                empty_reasoning,
                _build_function_call_input_item("c1"),
            ]
        }
        message_history = _RESPONSES_API_FORMAT.extract_message_history(request_body)
        message_history.messages[1].pop("reasoning_content")
        message_history.apply_to_request()
        assert request_body["input"][1] is empty_reasoning

    def test_truncating_arguments_updates_function_call_item(self) -> None:
        """Truncated tool arguments update the matching function_call item."""
        request_body = _build_sample_conversation_request_body()
        _edit_and_apply_message_history(request_body, _truncate_first_tool_call_arguments)
        assert request_body["input"][3] == {
            "type": "function_call",
            "id": "f1",
            "call_id": "c1",
            "name": "read",
            "arguments": '{"path": "..."}',
            "status": "completed",
        }

    def test_pruning_tool_result_updates_function_call_output_item(self) -> None:
        """A pruned tool result updates the matching function_call_output item."""
        request_body = _build_sample_conversation_request_body()
        _edit_and_apply_message_history(request_body, _prune_tool_result_content)
        assert request_body["input"][4] == {
            "type": "function_call_output",
            "id": "o1",
            "call_id": "c1",
            "output": "[tool result pruned]",
        }

    def test_editing_user_content_updates_user_item(self) -> None:
        """Changed user text updates the user message item."""
        request_body = _build_sample_conversation_request_body()
        _edit_and_apply_message_history(request_body, _update_user_message_content)
        assert request_body["input"][0] == {
            "type": "message",
            "role": "user",
            "content": "go now",
            "id": "u1",
        }

    def test_editing_instructions_system_message_updates_instructions(self) -> None:
        """Editing the leading system message updates the instructions field."""
        request_body = _build_sample_conversation_request_body()
        _edit_and_apply_message_history(
            request_body, lambda messages: messages[0].update(content="new rules")
        )
        assert request_body["instructions"] == "new rules"
        assert request_body["input"] == _build_sample_conversation_request_body()["input"]

    @pytest.mark.parametrize("new_rules", ["fresh rules", "other rules"])
    def test_replacing_instructions_system_message_updates_instructions(
        self, new_rules: str
    ) -> None:
        """Replacing the first system message with a new dict updates the instructions field."""
        request_body = _build_sample_conversation_request_body()

        def replace_instructions(messages: list) -> None:
            messages[0] = {"role": "system", "content": new_rules}

        messages = _edit_and_apply_message_history(request_body, replace_instructions)
        assert request_body["instructions"] == new_rules
        assert request_body["input"] == _build_sample_conversation_request_body()["input"]
        assert _RESPONSES_API_FORMAT.extract_message_history(request_body).messages == messages

    @pytest.mark.parametrize(
        "input_items",
        [
            _build_sample_conversation_request_body()["input"],
            [
                {"type": "message", "role": "system", "content": "keep"},
                {"type": "message", "role": "user", "content": "go"},
            ],
        ],
        ids=["group", "system-item-first"],
    )
    def test_deleting_instructions_system_message_removes_instructions(
        self, input_items: list
    ) -> None:
        """Deleting the instructions system message deletes instructions and keeps input."""
        request_body = {"instructions": "rules", "input": copy.deepcopy(input_items)}

        def delete_first(messages: list) -> None:
            del messages[0]

        messages = _edit_and_apply_message_history(request_body, delete_first)
        assert request_body == {"input": input_items}
        assert _RESPONSES_API_FORMAT.extract_message_history(request_body).messages == messages

    @pytest.mark.parametrize(
        "item, expected_content",
        [
            ({"role": "assistant", "content": "old", "id": "a"}, "new"),
            (
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "old", "annotations": [1]}],
                    "id": "a",
                },
                [{"type": "output_text", "text": "new", "annotations": [1]}],
            ),
        ],
        ids=["string", "blocks"],
    )
    def test_editing_assistant_content_preserves_item_content_shape(
        self, item: dict, expected_content: str | list
    ) -> None:
        """Changed assistant text updates its item while keeping string or block shape."""
        request_body = {"input": [_build_message_input_item("user", "u"), item]}
        message_history = _RESPONSES_API_FORMAT.extract_message_history(request_body)
        message_history.messages[1]["content"] = "new"
        message_history.apply_to_request()
        assert request_body["input"][1] == {**item, "content": expected_content}

    def test_inserting_system_message_adds_message_item(self) -> None:
        """An inserted system message serializes as a message item."""
        request_body = _build_sample_conversation_request_body()
        _edit_and_apply_message_history(request_body, _insert_system_message_before_assistant)
        assert request_body["input"][1] == {"type": "message", "role": "system", "content": "again"}
        assert len(request_body["input"]) == len(_build_sample_conversation_request_body()["input"]) + 1

    def test_inserting_assistant_turn_adds_reasoning_message_and_calls(self) -> None:
        """An inserted assistant turn serializes into reasoning, message, and function_call items."""
        request_body = {
            "input": [
                _build_message_input_item("user", "u"),
                _build_function_call_output_input_item("c1", "res"),
            ]
        }
        message_history = _RESPONSES_API_FORMAT.extract_message_history(request_body)
        message_history.messages.insert(
            1,
            {
                "role": "assistant",
                "content": "text",
                "reasoning_content": "why",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "grep", "arguments": '{"q": 1}'},
                    }
                ],
            },
        )
        message_history.apply_to_request()
        assert request_body["input"][1:4] == [
            {"type": "reasoning", "summary": [], "content": [{"type": "reasoning_text", "text": "why"}]},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "text"}]},
            {"type": "function_call", "call_id": "c1", "name": "grep", "arguments": '{"q": 1}'},
        ]

    @pytest.mark.parametrize(
        "new_message, expected_items",
        [
            (
                {"role": "assistant", "content": ""},
                [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": ""}]}],
            ),
            (
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "c", "function": {"name": "n", "arguments": "{}"}}],
                },
                [{"type": "function_call", "call_id": "c", "name": "n", "arguments": "{}"}],
            ),
            (
                {"role": "tool", "tool_call_id": "c", "content": "out"},
                [_build_function_call_output_input_item("c", "out")],
            ),
            (
                {"role": "user", "content": "more"},
                [{"type": "message", "role": "user", "content": "more"}],
            ),
        ],
        ids=["empty-turn", "calls-only", "tool", "user"],
    )
    def test_appending_new_message_shapes(self, new_message: dict, expected_items: list) -> None:
        """Each new chat message role serializes into the expected Responses items."""
        request_body = {"input": [_build_message_input_item("user", "u")]}
        message_history = _RESPONSES_API_FORMAT.extract_message_history(request_body)
        message_history.messages.append(new_message)
        message_history.apply_to_request()
        assert request_body["input"][1:] == expected_items

    def test_appending_to_string_input_converts_input_to_list(self) -> None:
        """Appending a message when input is a string rewrites input as an item list."""
        request_body = {"input": "question"}
        message_history = _RESPONSES_API_FORMAT.extract_message_history(request_body)
        message_history.messages.append({"role": "system", "content": "note"})
        message_history.apply_to_request()
        assert request_body["input"] == [
            {"type": "message", "role": "user", "content": "question"},
            {"type": "message", "role": "system", "content": "note"},
        ]

    def test_unknown_and_non_dict_items_are_preserved_verbatim(self) -> None:
        """Items the history editor does not recognize stay unchanged."""
        opaque_item = {"type": "item_reference", "id": "x"}
        request_body = {"input": [opaque_item, 7, _build_message_input_item("user", "u")]}
        message_history = _RESPONSES_API_FORMAT.extract_message_history(request_body)
        message_history.messages.append({"role": "system", "content": "s"})
        message_history.apply_to_request()
        assert request_body["input"][:3] == [opaque_item, 7, _build_message_input_item("user", "u")]
        assert request_body["input"][0] is opaque_item


def _strip_empty_assistant_content_fields(messages: list) -> list:
    """Remove empty content fields on assistant messages that emit no text item."""
    return [
        {
            key: value
            for key, value in message.items()
            if not (key == "content" and value == "" and message.get("role") == "assistant")
        }
        for message in copy.deepcopy(messages)
    ]


@pytest.mark.parametrize(
    "edit_messages",
    [
        _set_assistant_reasoning,
        _remove_assistant_reasoning,
        _truncate_first_tool_call_arguments,
        _prune_tool_result_content,
        _update_user_message_content,
        _update_assistant_message_content,
        _insert_system_message_before_assistant,
        _insert_assistant_tool_turn_before_assistant,
        _append_partial_reply_and_retry_instruction,
    ],
    ids=lambda edit_fn: edit_fn.__name__.strip("_"),
)
def test_re_extracting_after_apply_matches_edited_messages(
    edit_messages: Callable[[list], None]
) -> None:
    """Re-extracting message history after apply_to_request reflects the edits."""
    request_body = _build_sample_conversation_request_body()
    message_history = _RESPONSES_API_FORMAT.extract_message_history(request_body)
    edit_messages(message_history.messages)
    expected_messages = _strip_empty_assistant_content_fields(message_history.messages)
    message_history.apply_to_request()
    assert _RESPONSES_API_FORMAT.extract_message_history(request_body).messages == expected_messages


def test_appended_tool_call_turn_after_assistant_text_merges_into_same_group() -> None:
    """An appended call-only assistant turn after assistant text groups with that text."""
    request_body = {
        "input": [
            _build_message_input_item("user", "u"),
            _build_message_input_item("assistant", "end"),
        ]
    }
    message_history = _RESPONSES_API_FORMAT.extract_message_history(request_body)
    message_history.messages.append({
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "c1", "function": {"name": "n", "arguments": "{}"}}],
    })
    message_history.apply_to_request()
    assert _extract_chat_messages(request_body["input"]) == [
        {"role": "user", "content": [{"type": "text", "text": "u"}]},
        {
            "role": "assistant",
            "content": "end",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "n", "arguments": "{}"}}
            ],
        },
    ]


class TestExtractPromptText:
    """Prompt text is extracted from Responses request bodies."""

    def test_includes_instructions_messages_and_tool_outputs_only(self) -> None:
        """Prompt text includes instructions, messages, and tool outputs, not reasoning or calls."""
        request_body = {
            "instructions": "rules",
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "q"}]},
                _build_reasoning_input_item("hidden"),
                _build_function_call_input_item("c1", arguments='{"no": 1}'),
                _build_function_call_output_input_item("c1", "result"),
                {"type": "item_reference", "id": "x"},
            ],
        }
        assert _RESPONSES_API_FORMAT.extract_prompt_text(request_body) == "rules\nq\n\nresult\n\n"

    def test_string_input_counts_as_one_line(self) -> None:
        """String input produces one line of prompt text."""
        assert _RESPONSES_API_FORMAT.extract_prompt_text({"input": "hello"}) == "hello\n"

    def test_missing_input_returns_empty_string(self) -> None:
        """A request body without input returns empty prompt text."""
        assert _RESPONSES_API_FORMAT.extract_prompt_text({"model": "x"}) == ""

    def test_non_list_non_string_input_raises_value_error(self) -> None:
        """An input field that is neither a list nor a string raises ValueError."""
        with pytest.raises(ValueError, match="input is not a list"):
            _RESPONSES_API_FORMAT.extract_prompt_text({"input": 5})


# ── Stream event parsing ──


def _parse_all_stream_events(raw_stream: bytes) -> list[dict]:
    """Parse every SSE event in a raw Responses stream and return all extracted updates."""
    parser = _RESPONSES_API_FORMAT.create_stream_event_parser()
    all_updates: list[dict] = []
    for payload in parse_sse_payloads(raw_stream):
        parsed = parser.parse_event(payload)
        assert len(parsed.updates) <= 1
        assert parsed.raw_bytes_owner_index == (0 if parsed.updates else None)
        all_updates.extend(parsed.updates)
    return all_updates


def _extract_completed_response_output_items(raw_stream: bytes) -> list[dict]:
    """Return the output item list from the last response.completed event."""
    completed_events = [
        payload
        for payload in parse_sse_payloads(raw_stream)
        if payload["type"] == "response.completed"
    ]
    return completed_events[-1]["response"]["output"]


def _extract_output_item_text(output_item: dict) -> str:
    """Return the text stored in one Responses output item."""
    if output_item["type"] == "reasoning":
        blocks = output_item.get("content") or output_item.get("summary") or []
        return blocks[0]["text"] if blocks else ""
    return "".join(block["text"] for block in output_item["content"])


def _join_parsed_updates_text(parsed_updates: list[dict], update_type: str) -> str:
    """Join the text of every parsed update matching update_type."""
    return "".join(update["text"] for update in parsed_updates if update["type"] == update_type)


def _join_output_items_text(output_items: list[dict], item_type: str) -> str:
    """Join the text of every output item matching item_type."""
    return "".join(
        _extract_output_item_text(item) for item in output_items if item["type"] == item_type
    )


def _collect_parsed_tool_calls(parsed_updates: list[dict]) -> list[tuple[str, dict]]:
    """Reassemble tool call names and parsed JSON arguments from parsed updates in order."""
    calls_by_id: dict[str, list[str]] = {}
    for update in parsed_updates:
        if update["type"] == "tool_call":
            parts = calls_by_id.setdefault(update["tool_call_id"], ["", ""])
            parts[0] += update["function"]["name"]
            parts[1] += update["function"]["arguments"]
    return [(name, json.loads(arguments)) for name, arguments in calls_by_id.values()]


def _collect_completed_function_calls(output_items: list[dict]) -> list[tuple[str, dict]]:
    """Extract function call names and parsed JSON arguments from final output items."""
    return [
        (item["name"], json.loads(item["arguments"]))
        for item in output_items
        if item["type"] == "function_call"
    ]


def _is_generated_text_event(payload: dict) -> bool:
    """Tell whether one SSE event carries newly generated text."""
    if payload["type"].endswith(".delta"):
        return True
    return (
        payload["type"] == "response.output_item.added"
        and payload["item"]["type"] == "function_call"
    )


def _assert_parsed_stream_matches_final_output(raw_stream: bytes) -> list[dict]:
    """Verify that parsed content, reasoning, and tool calls match the completed response output."""
    parsed_updates = _parse_all_stream_events(raw_stream)
    final_output_items = _extract_completed_response_output_items(raw_stream)
    assert _join_parsed_updates_text(parsed_updates, "content") == _join_output_items_text(
        final_output_items, "message"
    )
    assert _join_parsed_updates_text(parsed_updates, "reasoning") == _join_output_items_text(
        final_output_items, "reasoning"
    )
    assert _collect_parsed_tool_calls(parsed_updates) == _collect_completed_function_calls(
        final_output_items
    )
    return parsed_updates


class TestStreamEventParserOnBackendFixtures:
    """Recorded backend Responses streams parse into their final output items."""

    @pytest.mark.parametrize("backend_name", _BACKEND_FIXTURE_NAMES)
    def test_parsed_updates_match_final_output_items(self, backend_name: str) -> None:
        """Parsed text and tool calls from each backend fixture match its completed response."""
        raw_stream = _read_stream_fixture(backend_name)
        assert {payload["type"] for payload in parse_sse_payloads(raw_stream)} <= _KNOWN_RESPONSES_EVENT_TYPES
        parsed_updates = _assert_parsed_stream_matches_final_output(raw_stream)
        assert _join_parsed_updates_text(parsed_updates, "reasoning").startswith("The user")

    @pytest.mark.parametrize("backend_name", _BACKEND_FIXTURE_NAMES)
    def test_emits_single_tool_calls_finish_reason(self, backend_name: str) -> None:
        """Each backend fixture ends with one tool_calls finish reason."""
        finish_updates = [
            update
            for update in _parse_all_stream_events(_read_stream_fixture(backend_name))
            if update["type"] == "finish_reason"
        ]
        assert [update["text"] for update in finish_updates] == ["tool_calls"]

    def test_vllm_pairs_streamed_and_final_call_ids_as_aliases(self) -> None:
        """The vLLM fixture pairs each streamed call id with its different final call id."""
        finish_update = [
            update
            for update in _parse_all_stream_events(_read_stream_fixture("vllm"))
            if update["type"] == "finish_reason"
        ][0]
        assert finish_update["tool_call_id_aliases"] == [
            ("call_b5821fd3f0ab9243", "chatcmpl-tool-bab2358b7fb9b9f3")
        ]

    @pytest.mark.parametrize("backend_name, call_id", [("sglang", "call_sgl1"), ("llamacpp", "call_lcpp1")])
    def test_matching_streamed_and_final_call_ids_pair_with_themselves(
        self, backend_name: str, call_id: str
    ) -> None:
        """Streams whose final call ids match the streamed ones pair each id with itself."""
        finish_update = [
            update
            for update in _parse_all_stream_events(_read_stream_fixture(backend_name))
            if update["type"] == "finish_reason"
        ][0]
        assert finish_update["tool_call_id_aliases"] == [(call_id, call_id)]

    def test_argument_delta_updates_carry_parent_call_id(self) -> None:
        """Argument updates carry the call_id of their parent function_call item."""
        tool_call_updates = [
            update
            for update in _parse_all_stream_events(_read_stream_fixture("vllm"))
            if update["type"] == "tool_call"
        ]
        assert {update["tool_call_id"] for update in tool_call_updates} == {"call_b5821fd3f0ab9243"}
        assert tool_call_updates[0]["text"] == "get_weather"
        assert "".join(update["function"]["arguments"] for update in tool_call_updates[1:]) == (
            '{"city": "Paris"}'
        )


def _extract_finish_reason_updates(raw_stream: bytes) -> list[dict]:
    """Parse a raw stream and return only its finish_reason updates."""
    parser = _RESPONSES_API_FORMAT.create_stream_event_parser()
    updates: list[dict] = []
    for payload in parse_sse_payloads(raw_stream):
        updates.extend(parser.parse_event(payload).updates)
    return [update for update in updates if update["type"] == "finish_reason"]


def _build_terminal_sse_event(
    status: str,
    incomplete_reason: str = "",
    event_type: str = "response.completed",
    output_items: list | None = None,
) -> bytes:
    """Build a terminal Responses SSE event with the given status and output."""
    response: dict = {"id": "r", "status": status, "output": output_items or []}
    if incomplete_reason:
        response["incomplete_details"] = {"reason": incomplete_reason}
    return responses_event({"type": event_type, "response": response})


class TestStreamEventParserFinishReasonMapping:
    """Terminal Responses events map to chat-style finish reasons."""

    @pytest.mark.parametrize(
        "status, has_streamed_call, incomplete_reason, expected_reason",
        [
            ("completed", False, "", "stop"),
            ("completed", True, "", "tool_calls"),
            ("incomplete", False, "max_output_tokens", "length"),
            ("incomplete", True, "max_output_tokens", "length"),
            ("incomplete", False, "content_filter", "content_filter"),
            ("incomplete", True, "content_filter", "content_filter"),
            ("cancelled", False, "", "abort"),
            ("cancelled", True, "", "abort"),
        ],
    )
    def test_maps_status_and_streamed_calls_to_finish_reason(
        self,
        status: str,
        has_streamed_call: bool,
        incomplete_reason: str,
        expected_reason: str,
    ) -> None:
        """The finish reason depends on terminal status and whether a call was streamed."""
        stream = ResponsesStream("r")
        raw_stream = stream.start() + (
            stream.tool_call("c1", "run") if has_streamed_call else stream.reasoning("x")
        )
        raw_stream += _build_terminal_sse_event(status, incomplete_reason, output_items=stream.output)
        assert [update["text"] for update in _extract_finish_reason_updates(raw_stream)] == [
            expected_reason
        ]

    def test_named_tool_choice_with_streamed_call_maps_to_tool_calls(self) -> None:
        """A forced tool choice with a streamed call still finishes with tool_calls."""
        stream = ResponsesStream("r")
        raw_stream = stream.start() + stream.tool_call("c1", "run")
        response = {
            "id": "r",
            "status": "completed",
            "tool_choice": {"type": "function", "name": "run"},
            "output": stream.output,
        }
        raw_stream += responses_event({"type": "response.completed", "response": response})
        assert [update["text"] for update in _extract_finish_reason_updates(raw_stream)] == [
            "tool_calls"
        ]

    def test_unstreamed_terminal_call_maps_to_stop_with_no_aliases(self) -> None:
        """A function call present only in the terminal response does not count as streamed."""
        stream = ResponsesStream("r")
        raw_stream = (
            stream.start()
            + stream.content("words")
            + _build_terminal_sse_event("completed", output_items=[_build_function_call_input_item("c1")])
        )
        finish_updates = _extract_finish_reason_updates(raw_stream)
        assert [update["text"] for update in finish_updates] == ["stop"]
        assert finish_updates[0]["tool_call_id_aliases"] == []

    def test_streamed_call_absent_from_terminal_keeps_tool_calls_with_no_aliases(self) -> None:
        """A streamed call missing from the terminal output still finishes with tool_calls."""
        stream = ResponsesStream("r")
        raw_stream = (
            stream.start()
            + stream.tool_call("c1", "run")
            + _build_terminal_sse_event("completed", output_items=[])
        )
        finish_updates = _extract_finish_reason_updates(raw_stream)
        assert [update["text"] for update in finish_updates] == ["tool_calls"]
        assert finish_updates[0]["tool_call_id_aliases"] == []

    def test_failed_event_produces_no_finish_reason(self) -> None:
        """A response.failed event produces no finish reason update."""
        stream = ResponsesStream("r")
        raw_stream = stream.start() + stream.content("partial") + stream.failed()
        assert _extract_finish_reason_updates(raw_stream) == []
        failed_payload = parse_sse_payloads(stream.failed())[0]
        assert _RESPONSES_API_FORMAT.create_stream_event_parser().parse_event(
            failed_payload
        ) == ParsedStreamEvent([], None)

    @pytest.mark.parametrize("has_streamed_call, expected_reason", [(False, "length"), (True, "length")])
    def test_incomplete_event_type_maps_to_length(
        self, has_streamed_call: bool, expected_reason: str
    ) -> None:
        """A response.incomplete event maps to length even if response.status says in_progress."""
        stream = ResponsesStream("r")
        raw_stream = stream.start() + (stream.tool_call("c1", "run") if has_streamed_call else b"")
        raw_stream += _build_terminal_sse_event("in_progress", event_type="response.incomplete")
        assert [update["text"] for update in _extract_finish_reason_updates(raw_stream)] == [
            expected_reason
        ]

    def test_mismatched_call_names_produce_no_aliases(self) -> None:
        """Streamed and final calls with different tool names are not paired."""
        stream = ResponsesStream("r", terminal_call_ids={"c1": "t1"})
        raw_stream = (
            stream.start()
            + stream.tool_call("c1", "run")
            + _build_terminal_sse_event(
                "completed", output_items=[_build_function_call_input_item("t1", name="other")]
            )
        )
        assert _extract_finish_reason_updates(raw_stream)[0]["tool_call_id_aliases"] == []

    def test_mismatched_call_counts_produce_no_aliases(self) -> None:
        """Streamed and final calls with different counts are not paired."""
        stream = ResponsesStream("r")
        raw_stream = (
            stream.start()
            + stream.tool_call("c1", "run")
            + _build_terminal_sse_event(
                "completed",
                output_items=[
                    _build_function_call_input_item("t1"),
                    _build_function_call_input_item("t2"),
                ],
            )
        )
        assert _extract_finish_reason_updates(raw_stream)[0]["tool_call_id_aliases"] == []

    def test_matching_calls_pair_in_arrival_order(self) -> None:
        """Matching streamed and final calls pair in the order they arrived."""
        stream = ResponsesStream("r", terminal_call_ids={"c1": "t1", "c2": "t2"})
        raw_stream = (
            stream.start()
            + stream.tool_call("c1", "run")
            + stream.tool_call("c2", "other")
            + stream.completed()
        )
        assert _extract_finish_reason_updates(raw_stream)[0]["tool_call_id_aliases"] == [
            ("c1", "t1"),
            ("c2", "t2"),
        ]


class TestStreamEventParserIndividualEvents:
    """Individual Responses SSE events convert into the expected updates."""

    def test_arguments_for_unknown_item_use_empty_call_id(self) -> None:
        """Argument deltas for an unseen item_id emit an empty tool_call_id."""
        payload = {
            "type": "response.function_call_arguments.delta",
            "item_id": "nope",
            "delta": "{}",
        }
        parsed = _RESPONSES_API_FORMAT.create_stream_event_parser().parse_event(payload)
        assert parsed == ParsedStreamEvent(
            [
                {
                    "type": "tool_call",
                    "text": "{}",
                    "tool_call_id": "",
                    "function": {"name": "", "arguments": "{}"},
                }
            ],
            0,
        )

    @pytest.mark.parametrize(
        "payload",
        [
            {"type": "response.made_up"},
            {"type": "response.output_text.done", "text": "already streamed"},
            {"type": "response.output_item.added", "item": {"type": "message", "id": "m"}},
            {"type": 5},
            {"no": "type"},
            [1, 2],
        ],
        ids=["unknown", "done-repeat", "non-call-item", "number-type", "no-type", "not-object"],
    )
    def test_non_update_events_return_empty(self, payload: dict | list) -> None:
        """Events without new text or terminal status return no updates."""
        assert _RESPONSES_API_FORMAT.create_stream_event_parser().parse_event(
            payload
        ) == ParsedStreamEvent([], None)

    @pytest.mark.parametrize(
        "event_type, expected_update_type",
        [
            ("response.output_text.delta", "content"),
            ("response.reasoning_text.delta", "reasoning"),
            ("response.reasoning_summary_text.delta", "reasoning"),
        ],
    )
    def test_text_delta_events_map_to_update_types(
        self, event_type: str, expected_update_type: str
    ) -> None:
        """Each text delta event produces one update of its corresponding type."""
        parsed = _RESPONSES_API_FORMAT.create_stream_event_parser().parse_event(
            {"type": event_type, "delta": "abc"}
        )
        assert parsed.updates == [{"type": expected_update_type, "text": "abc"}]

    def test_function_call_opening_standardizes_initial_arguments(self) -> None:
        """Initial JSON arguments on output_item.added are reformatted into standard JSON."""
        payload = {
            "type": "response.output_item.added",
            "item": {
                "type": "function_call",
                "id": "f",
                "call_id": "c",
                "name": "n",
                "arguments": '{"a":1}',
            },
        }
        parsed = _RESPONSES_API_FORMAT.create_stream_event_parser().parse_event(payload)
        assert parsed.updates == [
            {
                "type": "tool_call",
                "text": 'n{"a": 1}',
                "tool_call_id": "c",
                "function": {"name": "n", "arguments": '{"a": 1}'},
            }
        ]


class TestExtractGeneratedTexts:
    """Generated text extraction returns newly generated text per SSE event."""

    @pytest.mark.parametrize("backend_name", _BACKEND_FIXTURE_NAMES)
    def test_extracted_texts_match_final_output_items(self, backend_name: str) -> None:
        """Extracted generated texts across each backend fixture match its final output."""
        raw_stream = _read_stream_fixture(backend_name)
        payloads = parse_sse_payloads(raw_stream)
        texts = [
            text
            for payload in payloads
            for text in _RESPONSES_API_FORMAT.extract_generated_texts(payload)
        ]
        _assert_generated_texts_match_output_items(
            payloads, _extract_completed_response_output_items(raw_stream)
        )
        assert len(texts) == sum(1 for payload in payloads if _is_generated_text_event(payload))

    @pytest.mark.parametrize(
        "payload, expected_texts",
        [
            ({"type": "response.output_text.delta", "delta": "a"}, ["a"]),
            ({"type": "response.reasoning_text.delta", "delta": "b"}, ["b"]),
            ({"type": "response.reasoning_summary_text.delta", "delta": "c"}, ["c"]),
            ({"type": "response.function_call_arguments.delta", "delta": "{"}, ["{"]),
            (
                {
                    "type": "response.output_item.added",
                    "item": {"type": "function_call", "name": "n", "arguments": '{"x":1}'},
                },
                ['n{"x":1}'],
            ),
            ({"type": "response.output_item.added", "item": {"type": "message"}}, []),
            ({"type": "response.output_text.done", "text": "a"}, []),
            ({"type": "response.completed", "response": {}}, []),
        ],
    )
    def test_returns_text_only_for_generation_events(
        self, payload: dict, expected_texts: list[str]
    ) -> None:
        """Only delta events and function_call openings return generated text."""
        assert _RESPONSES_API_FORMAT.extract_generated_texts(payload) == expected_texts


class TestExtractTokenUsage:
    """Token usage is read from terminal Responses events."""

    def test_reads_usage_from_completed_event(self) -> None:
        """Completed backend events report input, output, and total tokens."""
        terminal_event = parse_sse_payloads(_read_stream_fixture("vllm"))[-1]
        assert _RESPONSES_API_FORMAT.extract_token_usage(terminal_event) == TokenUsage(
            _INPUT_TOKENS, _OUTPUT_TOKENS, _TOTAL_TOKENS
        )

    @pytest.mark.parametrize("event_type", ["response.incomplete", "response.failed"])
    def test_reads_usage_from_incomplete_and_failed_events(self, event_type: str) -> None:
        """Incomplete and failed terminal events also report token usage."""
        stream_event = {
            "type": event_type,
            "response": {"usage": {"input_tokens": 3, "output_tokens": 4}},
        }
        assert _RESPONSES_API_FORMAT.extract_token_usage(stream_event) == TokenUsage(3, 4, 0)

    @pytest.mark.parametrize(
        "stream_event",
        [
            {"type": "response.completed", "response": {"usage": None}},
            {"type": "response.created", "response": {"usage": {"input_tokens": 1}}},
            {"type": "response.completed"},
            {"usage": {"prompt_tokens": 1}},
        ],
    )
    def test_returns_none_when_usage_absent_or_non_terminal(self, stream_event: dict) -> None:
        """Non-terminal events and events without usage dicts return None."""
        assert _RESPONSES_API_FORMAT.extract_token_usage(stream_event) is None


class TestEnableStreaming:
    """Enabling streaming on Responses sets only the stream flag."""

    def test_sets_stream_true_only(self) -> None:
        """Enabling streaming sets stream to True and adds no extra fields."""
        request_body = {"stream": False}
        _RESPONSES_API_FORMAT.enable_streaming(request_body)
        assert request_body == {"stream": True}


# ── Request overrides ──


class TestConvertRequestOverrides:
    """Chat-style override keys convert into Responses field names."""

    @pytest.mark.parametrize(
        "overrides, expected_always_apply, expected_apply_if_missing",
        [
            ({"max_tokens": 10}, {}, {"max_output_tokens": 10}),
            ({"max_tokens": 10, "max_output_tokens": 99}, {}, {"max_output_tokens": 10}),
            ({"max_completion_tokens": 20}, {"max_output_tokens": 20}, {}),
            (
                {"max_completion_tokens": 20, "max_tokens": 10, "max_output_tokens": 99},
                {"max_output_tokens": 20},
                {},
            ),
            ({"max_output_tokens": 99}, {"max_output_tokens": 99}, {}),
            ({"reasoning_effort": "low"}, {"reasoning": {"effort": "low"}}, {}),
            (
                {"reasoning_effort": "low", "reasoning": {"effort": "high", "summary": "auto"}},
                {"reasoning": {"effort": "low", "summary": "auto"}},
                {},
            ),
            ({"reasoning": {"effort": "high"}}, {"reasoning": {"effort": "high"}}, {}),
            (
                {"temperature": 0.5, "min_p": 0.1, "chat_template_kwargs": {"x": True}},
                {"temperature": 0.5, "min_p": 0.1, "chat_template_kwargs": {"x": True}},
                {},
            ),
        ],
        ids=[
            "max-tokens-default",
            "max-tokens-overrides-native-max-output",
            "max-completion-forced",
            "max-completion-wins",
            "native-alone",
            "effort-renamed",
            "effort-replaces-native",
            "native-effort-alone",
            "others-unchanged",
        ],
    )
    def test_converts_chat_override_keys_to_responses_fields(
        self,
        overrides: dict,
        expected_always_apply: dict,
        expected_apply_if_missing: dict,
    ) -> None:
        """Configured chat override names convert into Responses always_apply and apply_if_missing fields."""
        assert _RESPONSES_API_FORMAT.convert_request_overrides(overrides) == RequestOverrides(
            expected_always_apply, expected_apply_if_missing
        )

    def test_does_not_mutate_input_overrides(self) -> None:
        """Converting overrides leaves the original config dict unchanged."""
        overrides = {"reasoning_effort": "low", "reasoning": {"effort": "high"}}
        _RESPONSES_API_FORMAT.convert_request_overrides(overrides)
        assert overrides == {"reasoning_effort": "low", "reasoning": {"effort": "high"}}


# ── Multi-attempt stream combining ──


def _run_retry_stream_combiner(segments: list) -> tuple[bytes, list[bytes]]:
    """Feed byte segments to a fresh retry stream combiner, using None to mark a retry boundary."""
    combiner = _RESPONSES_API_FORMAT.create_retry_stream_combiner()
    emitted_chunks: list[bytes] = []
    for segment in segments:
        if segment is None:
            combiner.begin_retry()
            continue
        emitted_chunks.append(combiner.feed_approved_bytes(segment))
    closing_tail = combiner.flush_closing_events()
    return b"".join(emitted_chunks) + closing_tail, emitted_chunks + [closing_tail]


def _strip_retry_rewritten_fields(payloads: list[dict]) -> list[dict]:
    """Remove numbering and terminal fields that a retry intentionally rewrites."""
    stripped: list[dict] = []
    for payload in payloads:
        event = {
            key: value
            for key, value in payload.items()
            if key not in ("sequence_number", "output_index")
        }
        if "response" in event:
            event["response"] = {
                key: value
                for key, value in event["response"].items()
                if key not in ("id", "output")
            }
        stripped.append(event)
    return stripped


def _split_sse_events_without_event_lines(raw_stream: bytes) -> list[bytes]:
    """Split a stream into event blocks after stripping all event header lines."""
    return [
        b"".join(
            line for line in chunk.splitlines(keepends=True) if not line.startswith(b"event:")
        )
        for chunk in _split_sse_events(raw_stream)
    ]


def _find_first_delta_event_index(sse_events: list[bytes], item_position: int) -> int | None:
    """Return the index of the first delta event belonging to item_position, or None."""
    seen_items = -1
    for index, event_bytes in enumerate(sse_events):
        payload = parse_sse_payloads(event_bytes)[0]
        if payload["type"] == "response.output_item.added":
            seen_items += 1
        if seen_items == item_position and payload["type"].endswith(".delta"):
            return index
    return None


def _find_cut_index_after_first_delta(sse_events: list[bytes], item_position: int) -> int:
    """Return the index right after the first delta event of item_position."""
    delta_index = _find_first_delta_event_index(sse_events, item_position)
    if delta_index is None:
        pytest.fail("item has no delta")
    return delta_index + 1


def _combine_interrupted_and_retried_fixture(
    backend_name: str, item_position: int
) -> tuple[bytes, list[bytes], int]:
    """Combine a fixture stream cut after item_position's first delta with a full retry attempt."""
    sse_events = _split_sse_events(_read_stream_fixture(backend_name))
    cut_index = _find_cut_index_after_first_delta(sse_events, item_position)
    segments = sse_events[:cut_index] + [None, sse_events[cut_index]] + sse_events
    combined_stream, emitted_chunks = _run_retry_stream_combiner(segments)
    return combined_stream, emitted_chunks, cut_index


class TestRetryStreamCombinerSingleAttempt:
    """A single attempt passes through the retry stream combiner unchanged."""

    @pytest.mark.parametrize("backend_name", _BACKEND_FIXTURE_NAMES)
    def test_single_attempt_is_byte_identical(self, backend_name: str) -> None:
        """A stream with no retries passes through byte for byte."""
        sse_events = _split_sse_events(_read_stream_fixture(backend_name))
        combined_stream, emitted_chunks = _run_retry_stream_combiner(sse_events)
        assert combined_stream == _read_stream_fixture(backend_name)
        assert emitted_chunks[-1] == b""

    def test_trailing_partial_event_without_retry_is_byte_identical(self) -> None:
        """A truncated final event without a retry passes through untouched."""
        segments = _split_sse_events(_read_stream_fixture("vllm"))[:5] + [
            b'data: {"type": "respon'
        ]
        combined_stream, emitted_chunks = _run_retry_stream_combiner(segments)
        assert combined_stream == b"".join(segments)
        assert emitted_chunks[-1] == b""

    def test_empty_inputs_emit_empty_bytes(self) -> None:
        """Empty byte feeds produce empty output before and after a retry."""
        combiner = _RESPONSES_API_FORMAT.create_retry_stream_combiner()
        assert combiner.feed_approved_bytes(b"") == b""
        combiner.begin_retry()
        assert combiner.feed_approved_bytes(b"") == b""
        assert combiner.flush_closing_events() == b""


class TestRetryStreamCombinerAcrossRetries:
    """Interrupted attempts and their retries combine into one valid Responses stream."""

    @pytest.mark.parametrize("backend_name", _BACKEND_FIXTURE_NAMES)
    @pytest.mark.parametrize("item_position", [0, 1])
    def test_combined_stream_has_one_opening_and_merged_terminal_output(
        self, backend_name: str, item_position: int
    ) -> None:
        """A cut and retried stream has one created event, one completed event, and merged output."""
        combined_stream, _, _ = _combine_interrupted_and_retried_fixture(backend_name, item_position)
        payloads = parse_sse_payloads(combined_stream)
        event_types = [payload["type"] for payload in payloads]
        assert event_types.count("response.created") == 1
        assert event_types.count("response.completed") == 1
        assert event_types[-1] == "response.completed"
        retry_payloads = parse_sse_payloads(_read_stream_fixture(backend_name))
        terminal_response = payloads[-1]["response"]
        assert terminal_response["id"] == retry_payloads[0]["response"]["id"]
        retry_output_items = retry_payloads[-1]["response"]["output"]
        earlier_items = terminal_response["output"][
            : len(terminal_response["output"]) - len(retry_output_items)
        ]
        assert terminal_response["output"][len(earlier_items):] == retry_output_items
        assert len(earlier_items) == item_position + 1
        assert earlier_items[-1]["status"] == "incomplete"
        assert terminal_response["usage"] == retry_payloads[-1]["response"]["usage"]

    @pytest.mark.parametrize("backend_name", _NUMBERED_BACKEND_FIXTURES)
    @pytest.mark.parametrize("item_position", [0, 1])
    def test_numbered_streams_keep_contiguous_sequence_and_output_indexes(
        self, backend_name: str, item_position: int
    ) -> None:
        """Numbered backend streams keep contiguous sequence_number and output_index values."""
        combined_stream, _, _ = _combine_interrupted_and_retried_fixture(backend_name, item_position)
        payloads = parse_sse_payloads(combined_stream)
        sequence_numbers = [payload["sequence_number"] for payload in payloads]
        assert sequence_numbers == list(range(len(sequence_numbers)))
        added_indexes = [
            payload["output_index"]
            for payload in payloads
            if payload["type"] == "response.output_item.added"
        ]
        assert added_indexes == list(range(len(added_indexes)))
        done_indexes = [
            payload["output_index"]
            for payload in payloads
            if payload["type"] == "response.output_item.done"
        ]
        assert done_indexes == list(range(len(done_indexes)))

    def test_two_consecutive_interrupted_attempts_keep_contiguous_output_indexes(self) -> None:
        """Two interrupted attempts and a third attempt keep contiguous output indexes."""
        first_stream = ResponsesStream("r1")
        attempt_one = _split_sse_events(first_stream.start() + first_stream.content("abcdefg"))[:5]
        second_stream = ResponsesStream("r2")
        attempt_two = _split_sse_events(second_stream.start() + second_stream.content("hijklmn"))[:5]
        third_stream = ResponsesStream("r3")
        attempt_three = _split_sse_events(
            third_stream.start() + third_stream.content("done") + third_stream.completed()
        )
        combined_stream, _ = _run_retry_stream_combiner(
            attempt_one + [None] + attempt_two + [None] + attempt_three
        )
        payloads = parse_sse_payloads(combined_stream)
        assert [payload["sequence_number"] for payload in payloads] == list(range(len(payloads)))
        added_indexes = [
            payload["output_index"]
            for payload in payloads
            if payload["type"] == "response.output_item.added"
        ]
        text_done_indexes = [
            payload["output_index"]
            for payload in payloads
            if payload["type"] == "response.output_text.done"
        ]
        part_done_indexes = [
            payload["output_index"]
            for payload in payloads
            if payload["type"] == "response.content_part.done"
        ]
        item_done_indexes = [
            payload["output_index"]
            for payload in payloads
            if payload["type"] == "response.output_item.done"
        ]
        assert added_indexes == [0, 1, 2]
        assert text_done_indexes == [0, 1, 2]
        assert part_done_indexes == [0, 1, 2]
        assert item_done_indexes == [0, 1, 2]
        done_statuses = [
            payload["item"]["status"]
            for payload in payloads
            if payload["type"] == "response.output_item.done"
        ]
        assert done_statuses == ["incomplete", "incomplete", "completed"]
        terminal_response = payloads[-1]["response"]
        assert terminal_response["id"] == "r1"
        assert [item["status"] for item in terminal_response["output"]] == [
            "incomplete",
            "incomplete",
            "completed",
        ]

    @pytest.mark.parametrize("backend_name", _UNNUMBERED_BACKEND_FIXTURES)
    @pytest.mark.parametrize("item_position", [0, 1])
    def test_unnumbered_streams_do_not_invent_numbers(
        self, backend_name: str, item_position: int
    ) -> None:
        """Streams without sequence_number or output_index stay unnumbered after retry."""
        combined_stream, _, _ = _combine_interrupted_and_retried_fixture(backend_name, item_position)
        assert all(
            "sequence_number" not in payload and "output_index" not in payload
            for payload in parse_sse_payloads(combined_stream)
        )

    @pytest.mark.parametrize("backend_name", _BACKEND_FIXTURE_NAMES)
    def test_interrupted_item_closes_with_already_approved_text(self, backend_name: str) -> None:
        """An interrupted item closes with the exact delta text already sent to the client."""
        sse_events = _split_sse_events(_read_stream_fixture(backend_name))
        cut_index = _find_cut_index_after_first_delta(sse_events, 0)
        sent_deltas = [
            parse_sse_payloads(event_bytes)[0]["delta"]
            for event_bytes in sse_events[: cut_index + 1]
            if parse_sse_payloads(event_bytes)[0]["type"].endswith(".delta")
        ]
        combined_stream, _ = _run_retry_stream_combiner(
            sse_events[:cut_index] + [None, sse_events[cut_index]] + sse_events
        )
        payloads = parse_sse_payloads(combined_stream)
        closed_item = [
            payload for payload in payloads if payload["type"] == "response.output_item.done"
        ][0]["item"]
        assert closed_item["status"] == "incomplete"
        assert _extract_output_item_text(closed_item) == "".join(sent_deltas)
        text_done_event = [
            payload for payload in payloads if payload["type"] == "response.reasoning_text.done"
        ][0]
        assert text_done_event["text"] == "".join(sent_deltas)

    def test_closing_events_emit_when_next_attempt_starts(self) -> None:
        """Trailing approved deltas stay in attempt one, and closings emit on attempt two's created event."""
        _combined_stream, emitted_chunks, cut_index = _combine_interrupted_and_retried_fixture("vllm", 0)
        assert [payload["type"] for payload in parse_sse_payloads(emitted_chunks[cut_index])] == [
            "response.reasoning_text.delta"
        ]
        assert [payload["type"] for payload in parse_sse_payloads(emitted_chunks[cut_index + 1])] == [
            "response.reasoning_text.done",
            "response.reasoning_part.done",
            "response.output_item.done",
        ]
        assert emitted_chunks[cut_index + 2] == b""

    def test_trailing_opening_after_retry_stays_in_first_attempt(self) -> None:
        """An item opening flushed right after begin_retry is numbered and closed under attempt one."""
        first_stream = ResponsesStream("r1")
        attempt_one = first_stream.start() + first_stream.content("abcdef")
        trailing_call_opening = first_stream.tool_call_opening("c1", "run")
        retry_stream = ResponsesStream("r2")
        attempt_two = retry_stream.start() + retry_stream.content("ok") + retry_stream.completed()
        combined_stream, _ = _run_retry_stream_combiner(
            _split_sse_events(attempt_one) + [None, trailing_call_opening] + _split_sse_events(attempt_two)
        )
        payloads = parse_sse_payloads(combined_stream)
        added_events = [
            payload for payload in payloads if payload["type"] == "response.output_item.added"
        ]
        assert [(payload["item"]["type"], payload["output_index"]) for payload in added_events] == [
            ("message", 0),
            ("function_call", 1),
            ("message", 2),
        ]
        opening_payload = added_events[1]
        previous_payload = payloads[payloads.index(opening_payload) - 1]
        assert opening_payload["sequence_number"] == previous_payload["sequence_number"] + 1
        closed_calls = [
            payload
            for payload in payloads
            if payload["type"] == "response.output_item.done"
            and payload["item"]["type"] == "function_call"
        ]
        assert closed_calls[0]["item"]["status"] == "incomplete"
        assert closed_calls[0]["output_index"] == 1
        terminal_response = payloads[-1]["response"]
        assert [item["type"] for item in terminal_response["output"]] == [
            "message",
            "function_call",
            "message",
        ]
        assert terminal_response["id"] == "r1"

    def test_non_sse_error_body_from_retry_passes_through_verbatim(self) -> None:
        """A raw JSON error body from a retry attempt is forwarded untouched after closing events."""
        first_stream = ResponsesStream("r1")
        attempt_one = _split_sse_events(first_stream.start() + first_stream.content("abcdefg"))
        error_body = b'{"error": {"message": "bad request"}}'
        third_stream = ResponsesStream("r3")
        attempt_three = _split_sse_events(
            third_stream.start() + third_stream.content("ok") + third_stream.completed()
        )
        combined_stream, emitted_chunks = _run_retry_stream_combiner(
            attempt_one[:5] + [None, error_body, None] + attempt_three
        )
        error_chunk = emitted_chunks[5]
        assert error_chunk.endswith(error_body)
        assert [
            payload["type"] for payload in parse_sse_payloads(error_chunk[: -len(error_body)])
        ] == [
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
        ]
        assert emitted_chunks[6:8] == [b"", b""]
        assert emitted_chunks[8].startswith(b"\n\nevent: response.output_item.added")
        assert parse_sse_payloads(combined_stream)[-1]["response"]["id"] == "r1"

    def test_partial_event_before_retry_gets_blank_line_before_closings(self) -> None:
        """A truncated event before retry is terminated with two newlines before closing events."""
        first_stream = ResponsesStream("r1")
        attempt_one = _split_sse_events(first_stream.start() + first_stream.content("abcdefg"))[:5]
        partial_event = b'data: {"type": "response.outp'
        retry_stream = ResponsesStream("r2")
        attempt_two = _split_sse_events(
            retry_stream.start() + retry_stream.content("ok") + retry_stream.completed()
        )
        _combined_stream, emitted_chunks = _run_retry_stream_combiner(
            attempt_one + [partial_event, None] + attempt_two
        )
        assert emitted_chunks[5] == partial_event
        assert emitted_chunks[6].startswith(b"\n\n")
        assert [payload["type"] for payload in parse_sse_payloads(emitted_chunks[6][2:])] == [
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
        ]
        assert emitted_chunks[7] == b""
        assert emitted_chunks[8].startswith(b"event: response.output_item.added")

    def test_flush_closing_events_emits_pending_closings_without_next_attempt(self) -> None:
        """Flushing after begin_retry with no retry attempt still emits the pending closing events."""
        stream = ResponsesStream("r1")
        sse_events = _split_sse_events(stream.start() + stream.reasoning("abcdef"))[:5]
        combined_stream, emitted_chunks = _run_retry_stream_combiner(sse_events + [None])
        assert [payload["type"] for payload in parse_sse_payloads(emitted_chunks[-1])] == [
            "response.reasoning_text.done",
            "response.reasoning_part.done",
            "response.output_item.done",
        ]
        assert combined_stream.startswith(b"".join(sse_events))

    def test_interrupted_item_reusing_completed_item_id_still_closes(self) -> None:
        """A second item in the same attempt that reuses a completed item id still closes."""
        stream = ResponsesStream("r1")
        sse_events = _split_sse_events(
            stream.start() + stream.content("AAA", "msg_x") + stream.content("BBBBBBBBBB", "msg_x")
        )
        cut_index = _find_cut_index_after_first_delta(sse_events, 1)
        _, emitted_chunks = _run_retry_stream_combiner(sse_events[:cut_index] + [None])
        first_delta = parse_sse_payloads(sse_events[cut_index - 1])[0]["delta"]
        closing_payloads = parse_sse_payloads(emitted_chunks[-1])
        assert [payload["type"] for payload in closing_payloads] == [
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
        ]
        assert closing_payloads[-1]["output_index"] == 1
        assert closing_payloads[-1]["item"]["content"] == [
            {"type": "output_text", "text": first_delta, "annotations": []}
        ]

    def test_single_newline_partial_event_before_retry_adds_one_newline(self) -> None:
        """A partial event ending with one newline gets one extra newline before closing events."""
        first_stream = ResponsesStream("r1")
        attempt_one = _split_sse_events(first_stream.start() + first_stream.content("abcdefg"))[:5]
        partial_line = b'data: {"type": "response.outp\n'
        retry_stream = ResponsesStream("r2")
        attempt_two = _split_sse_events(
            retry_stream.start() + retry_stream.content("ok") + retry_stream.completed()
        )
        _combined_stream, emitted_chunks = _run_retry_stream_combiner(
            attempt_one + [partial_line, None] + attempt_two
        )
        assert emitted_chunks[5] == partial_line
        assert emitted_chunks[6].startswith(b"\nevent: response.output_text.done\n")
        assert [payload["type"] for payload in parse_sse_payloads(emitted_chunks[6][1:])] == [
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
        ]

    def test_data_only_stream_stays_without_event_lines_after_retry(self) -> None:
        """A data-only SSE stream stays free of event lines across retries."""
        first_stream = ResponsesStream("r1")
        attempt_one = _split_sse_events_without_event_lines(
            first_stream.start() + first_stream.content("abcdefg")
        )[:5]
        retry_stream = ResponsesStream("r2")
        attempt_two = _split_sse_events_without_event_lines(
            retry_stream.start() + retry_stream.content("ok") + retry_stream.completed()
        )
        combined_stream, _ = _run_retry_stream_combiner(attempt_one + [None] + attempt_two)
        non_data_lines = [
            line
            for line in combined_stream.split(b"\n")
            if line and not line.startswith(b"data: ")
        ]
        assert non_data_lines == []
        payloads = parse_sse_payloads(combined_stream)
        assert [payload["sequence_number"] for payload in payloads] == list(range(len(payloads)))
        assert payloads[5]["type"] == "response.output_text.done"

    def test_interrupted_part_closes_with_initial_part_text(self) -> None:
        """A content part interrupted before any delta closes with its initial part text."""
        opening_events = [
            responses_event({"type": "response.created", "response": {"id": "r1", "status": "in_progress"}}),
            responses_event(
                {"type": "response.output_item.added", "item": {"id": "m1", "type": "message", "content": []}}
            ),
            responses_event({
                "type": "response.content_part.added",
                "item_id": "m1",
                "content_index": 0,
                "part": {"type": "output_text", "text": "Hi"},
            }),
        ]
        combined_stream, _ = _run_retry_stream_combiner(opening_events + [None])
        payloads = parse_sse_payloads(combined_stream)
        assert [payload["type"] for payload in payloads[3:]] == [
            "response.content_part.done",
            "response.output_item.done",
        ]
        assert payloads[3]["part"] == {"type": "output_text", "text": "Hi", "annotations": []}
        assert payloads[4]["item"]["content"] == [
            {"type": "output_text", "text": "Hi", "annotations": []}
        ]

    def test_interrupted_part_with_text_done_closes_using_done_text(self) -> None:
        """A part whose text finished before interruption closes with its output_text.done text."""
        opening_events = [
            responses_event({"type": "response.created", "response": {"id": "r1", "status": "in_progress"}}),
            responses_event(
                {"type": "response.output_item.added", "item": {"id": "m1", "type": "message", "content": []}}
            ),
            responses_event({
                "type": "response.content_part.added",
                "item_id": "m1",
                "content_index": 0,
                "part": {"type": "output_text", "text": ""},
            }),
            responses_event(
                {"type": "response.output_text.delta", "item_id": "m1", "content_index": 0, "delta": "ab"}
            ),
            responses_event(
                {"type": "response.output_text.done", "item_id": "m1", "content_index": 0, "text": "abcd"}
            ),
        ]
        combined_stream, _ = _run_retry_stream_combiner(opening_events + [None])
        payloads = parse_sse_payloads(combined_stream)
        assert [payload["type"] for payload in payloads[5:]] == [
            "response.content_part.done",
            "response.output_item.done",
        ]
        assert payloads[5]["part"]["text"] == "abcd"
        assert payloads[6]["item"]["content"][0]["text"] == "abcd"

    def test_interrupted_function_call_with_done_arguments_keeps_done_arguments(self) -> None:
        """A function call interrupted after arguments.done closes with those final arguments."""
        stream = ResponsesStream("r1")
        opening_events = _split_sse_events(stream.start() + stream.tool_call_opening("c1", "run")) + [
            responses_event({
                "type": "response.function_call_arguments.done",
                "item_id": "fc_c1",
                "name": "run",
                "arguments": '{"x": 1}',
            }),
        ]
        combined_stream, _ = _run_retry_stream_combiner(opening_events + [None])
        payloads = parse_sse_payloads(combined_stream)
        assert [payload["type"] for payload in payloads[4:]] == ["response.output_item.done"]
        assert payloads[4]["item"]["arguments"] == '{"x": 1}'
        assert payloads[4]["item"]["status"] == "incomplete"

    @pytest.mark.parametrize("backend_name", _BACKEND_FIXTURE_NAMES)
    def test_retry_preserves_event_bodies_and_drops_only_retry_openings(
        self, backend_name: str
    ) -> None:
        """Bytes before retry stay verbatim, later events keep their content, and duplicate openings drop."""
        sse_events = _split_sse_events(_read_stream_fixture(backend_name))
        cut_index = _find_cut_index_after_first_delta(sse_events, 1)
        _, emitted_chunks = _run_retry_stream_combiner(
            sse_events[:cut_index] + [None, sse_events[cut_index]] + sse_events
        )
        input_segments = sse_events[: cut_index + 1] + sse_events
        opening_event_types = ("response.created", "response.in_progress")
        assert emitted_chunks[:cut_index] == input_segments[:cut_index]
        post_retry_pairs = list(zip(input_segments, emitted_chunks))[cut_index:]
        kept_pairs = [
            (input_chunk, output_chunk)
            for input_chunk, output_chunk in post_retry_pairs
            if parse_sse_payloads(input_chunk)[0]["type"] not in opening_event_types
        ]
        dropped_outputs = [
            output_chunk
            for input_chunk, output_chunk in post_retry_pairs
            if parse_sse_payloads(input_chunk)[0]["type"] in opening_event_types
        ]
        assert [
            _strip_retry_rewritten_fields(parse_sse_payloads(output_chunk))
            for _, output_chunk in kept_pairs
        ] == [
            _strip_retry_rewritten_fields(parse_sse_payloads(input_chunk))
            for input_chunk, _ in kept_pairs
        ]
        assert len(dropped_outputs) == sum(
            1
            for event_bytes in sse_events
            if parse_sse_payloads(event_bytes)[0]["type"] in opening_event_types
        )
        assert not (
            {
                payload["type"]
                for output_chunk in dropped_outputs
                for payload in parse_sse_payloads(output_chunk)
            }
            & set(opening_event_types)
        )

    @pytest.mark.parametrize("backend_name", _BACKEND_FIXTURE_NAMES)
    def test_combined_stream_matches_golden_fixture(self, backend_name: str) -> None:
        """A cut and retried fixture matches the saved golden combined stream."""
        combined_stream, _, _ = _combine_interrupted_and_retried_fixture(backend_name, 1)
        assert combined_stream == (_FIXTURES_DIR / f"spliced_{backend_name}.sse").read_bytes()


# ── Non-streaming response assembly ──


class TestAssembleNonStreamingResponse:
    """Responses SSE events assemble into a single non-streaming JSON response."""

    def test_returns_terminal_response_object(self) -> None:
        """A complete stream assembles into the terminal event's response object."""
        payloads = parse_sse_payloads(_read_stream_fixture("vllm"))
        assembled = json.loads(_RESPONSES_API_FORMAT.assemble_non_streaming_response(payloads))
        assert assembled == payloads[-1]["response"]

    def test_combined_retry_stream_includes_items_from_both_attempts(self) -> None:
        """Assembling a combined retry stream includes items from both attempts."""
        combined_stream, _, _ = _combine_interrupted_and_retried_fixture("vllm", 1)
        assembled = json.loads(
            _RESPONSES_API_FORMAT.assemble_non_streaming_response(
                parse_sse_payloads(combined_stream)
            )
        )
        assert [item["type"] for item in assembled["output"]] == [
            "reasoning",
            "function_call",
            "reasoning",
            "function_call",
        ]
        assert assembled["output"][1]["status"] == "incomplete"

    def test_truncated_stream_rebuilds_output_items_from_deltas(self) -> None:
        """A stream with no terminal event rebuilds its response output from streamed deltas."""
        sse_events = _split_sse_events(_read_stream_fixture("sglang"))
        cut_index = _find_cut_index_after_first_delta(sse_events, 1)
        assembled = json.loads(
            _RESPONSES_API_FORMAT.assemble_non_streaming_response(
                parse_sse_payloads(b"".join(sse_events[:cut_index]))
            )
        )
        assert assembled["id"] == "resp_sgl1"
        assert assembled["status"] == "in_progress"
        assert [item["type"] for item in assembled["output"]] == ["reasoning", "message"]
        assert assembled["output"][1]["content"] == [
            {"type": "output_text", "text": "Checking", "annotations": []}
        ]
        assert assembled["output"][1]["status"] == "in_progress"

    def test_interrupted_retries_reusing_item_ids_keep_each_attempt_item(self) -> None:
        """Interrupted retries that reuse an item id still keep every attempt's item."""
        first_stream = ResponsesStream("r1")
        second_stream = ResponsesStream("r1")
        segments = [
            first_stream.start() + first_stream.content("AAA", "msg_x"),
            None,
            second_stream.start() + second_stream.content("BBB", "msg_x"),
            None,
        ]
        combined_stream, _ = _run_retry_stream_combiner(segments)
        assembled = json.loads(
            _RESPONSES_API_FORMAT.assemble_non_streaming_response(
                parse_sse_payloads(combined_stream)
            )
        )
        assert [(item["id"], _extract_output_item_text(item)) for item in assembled["output"]] == [
            ("msg_x", "AAA"),
            ("msg_x", "BBB"),
        ]

    def test_stream_cut_after_created_returns_response_with_empty_output(self) -> None:
        """A stream cut immediately after response.created returns that response with empty output."""
        created_response = {"id": "r1", "object": "response", "status": "in_progress"}
        assembled = _RESPONSES_API_FORMAT.assemble_non_streaming_response(
            [{"type": "response.created", "response": created_response}]
        )
        assert json.loads(assembled) == {**created_response, "output": []}

    def test_empty_events_returns_none(self) -> None:
        """An empty event list returns None."""
        assert _RESPONSES_API_FORMAT.assemble_non_streaming_response([]) is None

    def test_unrecognized_events_return_none(self) -> None:
        """Events without Responses structure return None."""
        assert _RESPONSES_API_FORMAT.assemble_non_streaming_response([{"error": "x"}, 5]) is None


# ── Live backend conformance tests, opt in ──


_LIVE_TIMEOUT_S = 120
_CONFIG_PATH = Path(__file__).parents[4] / "config.yaml"
_GATED_TARGET = pytest.param(
    None,
    marks=pytest.mark.skip(reason="needs --responses-gate=conformance"),
    id="gated",
)
_PROBE_TOOL = {
    "type": "function",
    "name": "get_weather",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}
_PROBE_HISTORY = [
    {"role": "user", "content": "What is the weather in Paris?"},
    {"type": "reasoning", "summary": [], "content": [{"type": "reasoning_text", "text": "Use the tool."}]},
    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Checking."}]},
    {"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": '{"city": "Paris"}'},
    {"type": "function_call_output", "call_id": "call_1", "output": "Sunny, 20C"},
    {"type": "message", "role": "system", "content": "Answer in one line."},
]
_LIVE_PROBES = {
    "content": {"input": "Say hello in one short line.", "stream": True},
    "tool": {"input": "Use get_weather for Paris.", "tools": [_PROBE_TOOL], "stream": True},
    "history": {"input": _PROBE_HISTORY, "tools": [_PROBE_TOOL], "stream": True},
    "tool_named": {
        "input": "Use get_weather for Paris.",
        "tools": [_PROBE_TOOL],
        "tool_choice": {"type": "function", "name": "get_weather"},
        "stream": True,
    },
    "length": {"input": "Count from one to fifty in words.", "max_output_tokens": 16, "stream": True},
}


def _discover_live_responses_targets(config: ProxyConfig) -> list[tuple[str, str, str]]:
    """List each Responses backend as name, URL, and Host header."""
    targets: list[tuple[str, str, str]] = []
    for service_name, service in config.services.items():
        if "/v1/responses" not in service.gpu_routes:
            continue
        targets.append(
            (service_name, f"http://127.0.0.1:{config.listen_port}/v1/responses", service.hostname)
        )
        if service.backup:
            targets.append(
                (f"{service_name}-backup", service.backup.rstrip("/") + "/v1/responses", "")
            )
    return targets


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrize the live conformance test across probes and configured Responses backends."""
    if "target" not in metafunc.fixturenames:
        return
    metafunc.parametrize("probe", sorted(_LIVE_PROBES))
    if metafunc.config.getoption("--responses-gate") != "conformance":
        metafunc.parametrize("target", [_GATED_TARGET])
        return
    targets = _discover_live_responses_targets(load_config(str(_CONFIG_PATH)))
    metafunc.parametrize("target", [pytest.param(target, id=target[0]) for target in targets])


@pytest.fixture(scope="module")
def live_config() -> ProxyConfig:
    """Load the deployment config used by live backend conformance tests."""
    return load_config(str(_CONFIG_PATH))


def _post_json_request(url: str, host_header: str, request_body: dict) -> tuple[int, str, bytes]:
    """Post a JSON body and return HTTP status, content type, and raw response bytes."""
    request = urllib_request.Request(url, data=json.dumps(request_body).encode(), method="POST")
    request.add_header("Content-Type", "application/json")
    if host_header:
        request.add_header("Host", host_header)
    with urllib_request.urlopen(request, timeout=_LIVE_TIMEOUT_S) as response:
        return response.status, response.headers.get("Content-Type", ""), response.read()


def _fetch_service_routing_counters(config: ProxyConfig, service_name: str) -> dict[str, int]:
    """Read the dashboard routing counters for one service."""
    request = urllib_request.Request(f"http://127.0.0.1:{config.listen_port}/?format=json")
    request.add_header("Host", config.dashboard_hostname)
    with urllib_request.urlopen(request, timeout=_LIVE_TIMEOUT_S) as response:
        return json.loads(response.read())["services"][service_name]["routing"]


def _classify_probe_routing_attribution(
    config: ProxyConfig, service_name: str, counters_before: dict[str, int] | None
) -> str:
    """Tell whether the target, backup, or proxy answered a live probe."""
    if counters_before is None:
        return "direct"
    counters_after = _fetch_service_routing_counters(config, service_name)
    if (
        counters_after["backup"] == counters_before["backup"]
        and counters_after["target"] > counters_before["target"]
    ):
        return "target"
    if counters_after == counters_before:
        return "not admitted"
    return "unattributed"


def _collect_generated_text_per_item(payloads: list[dict]) -> list[str]:
    """Collect the generated text extracted for each streamed item in first-seen order."""
    texts_by_item_id: dict[str, str] = {}
    current_item_id = ""
    for payload in payloads:
        if payload["type"] == "response.output_item.added":
            current_item_id = payload["item"].get("id", "")
            texts_by_item_id.setdefault(current_item_id, "")
        item_id = payload.get("item_id", current_item_id)
        for text in _RESPONSES_API_FORMAT.extract_generated_texts(payload):
            texts_by_item_id[item_id] = texts_by_item_id.get(item_id, "") + text
    return list(texts_by_item_id.values())


def _assert_generated_texts_match_output_items(
    payloads: list[dict], output_items: list[dict]
) -> None:
    """Verify that each terminal output item matches the generated text streamed for it."""
    streamed_texts = _collect_generated_text_per_item(payloads)
    assert len(streamed_texts) == len(output_items)
    for text, item in zip(streamed_texts, output_items):
        if item["type"] == "function_call":
            assert text.startswith(item["name"])
            assert json.loads(text[len(item["name"]):]) == json.loads(item["arguments"])
        elif item["type"] in ("message", "reasoning"):
            assert text == _extract_output_item_text(item)


def _assert_retry_combiner_framing_for_every_item(
    raw_stream: bytes, output_items: list[dict]
) -> int:
    """Cut the stream inside each item, retry, and verify the combined stream framing."""
    sse_events = _split_sse_events(raw_stream)
    verified_cuts = 0
    for item_position in range(len(output_items)):
        delta_index = _find_first_delta_event_index(sse_events, item_position)
        if delta_index is None:
            continue
        cut_index = delta_index + 1
        combined_stream, _ = _run_retry_stream_combiner(
            sse_events[:cut_index] + [None, sse_events[cut_index]] + sse_events
        )
        payloads = parse_sse_payloads(combined_stream)
        assert [payload["type"] for payload in payloads].count("response.created") == 1
        added_indexes = [
            payload["output_index"]
            for payload in payloads
            if payload["type"] == "response.output_item.added" and "output_index" in payload
        ]
        assert added_indexes == list(range(len(added_indexes)))
        sequence_numbers = [
            payload["sequence_number"] for payload in payloads if "sequence_number" in payload
        ]
        assert sequence_numbers == sorted(set(sequence_numbers))
        assert len(payloads[-1]["response"]["output"]) == item_position + 1 + len(output_items)
        verified_cuts += 1
    return verified_cuts


@pytest.mark.live
def test_live_backend_conformance(
    live_config: ProxyConfig,
    record_property: Callable[[str, str], None],
    target: tuple[str, str, str],
    probe: str,
) -> None:
    """A live backend's Responses stream satisfies every contract ResponsesAPIFormat relies on."""
    service_name, url, host_header = target
    counters_before = _fetch_service_routing_counters(live_config, service_name) if host_header else None
    try:
        status_code, _content_type, raw_stream = _post_json_request(
            url, host_header, _LIVE_PROBES[probe]
        )
    except TimeoutError:
        pytest.skip("no answer")
    attribution = _classify_probe_routing_attribution(live_config, service_name, counters_before)
    record_property("attribution", attribution)
    if attribution == "not admitted":
        pytest.skip("not admitted")
    assert status_code == 200
    payloads = parse_sse_payloads(raw_stream)
    assert {payload["type"] for payload in payloads} <= _KNOWN_RESPONSES_EVENT_TYPES
    terminal_events = [
        payload
        for payload in payloads
        if payload["type"] in ("response.completed", "response.incomplete")
    ]
    if not terminal_events:
        pytest.skip("failed generation")
    final_output_items = terminal_events[-1]["response"]["output"]
    parsed_updates = _assert_parsed_stream_matches_final_output(raw_stream)
    record_property("verified_content", str(bool(_join_parsed_updates_text(parsed_updates, "content"))))
    record_property("verified_reasoning", str(bool(_join_parsed_updates_text(parsed_updates, "reasoning"))))
    record_property("verified_tool_call", str(bool(_collect_parsed_tool_calls(parsed_updates))))
    finish_updates = [update for update in parsed_updates if update["type"] == "finish_reason"]
    assert len(finish_updates) == 1
    record_property("finish", finish_updates[0]["text"])
    record_property("status", terminal_events[-1]["response"].get("status", ""))
    usage = _RESPONSES_API_FORMAT.extract_token_usage(terminal_events[-1])
    assert usage.input_tokens > 0
    assert usage.output_tokens > 0
    _assert_generated_texts_match_output_items(payloads, final_output_items)
    record_property(
        "cut_points",
        str(_assert_retry_combiner_framing_for_every_item(raw_stream, final_output_items)),
    )
