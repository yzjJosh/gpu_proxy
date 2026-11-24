"""Tests for the Responses API dialect."""

import copy
import json
from pathlib import Path
from typing import Callable
from urllib import request as urllib_request

import pytest

from app.config import ProxyConfig, load_config
from app.plugins.openai_dialect import TokenUsage, TranslatedOverrides
from app.plugins.openai_responses_dialect import ResponsesDialect
from tests.app.conftest import ResponsesStream, parse_sse_payloads, responses_event

_DIALECT = ResponsesDialect()
_FIXTURES = Path(__file__).parent / "fixtures" / "responses"
_STREAM_NAMES = ("vllm", "sglang", "llamacpp")
_NUMBERED_STREAMS = ("vllm", "sglang")
_UNNUMBERED_STREAMS = ("llamacpp",)

_INPUT_TOKENS = 384
_OUTPUT_TOKENS = 55
_TOTAL_TOKENS = 439

# The event types the Responses reader knows, from the API's event list.
_SPEC_KNOWN_TYPES = frozenset({
    "response.created", "response.in_progress", "response.completed", "response.incomplete",
    "response.failed", "response.output_item.added", "response.output_item.done",
    "response.content_part.added", "response.content_part.done",
    "response.reasoning_part.added", "response.reasoning_part.done",
    "response.reasoning_summary_part.added", "response.reasoning_summary_part.done",
    "response.output_text.delta", "response.output_text.done",
    "response.reasoning_text.delta", "response.reasoning_text.done",
    "response.reasoning_summary_text.delta", "response.reasoning_summary_text.done",
    "response.function_call_arguments.delta", "response.function_call_arguments.done",
    "response.output_text.annotation.added", "response.refusal.delta", "response.refusal.done",
    "error",
})


def _raw(name: str) -> bytes:
    """The bytes of one checked-in stream fixture."""
    return (_FIXTURES / f"{name}.sse").read_bytes()


def _event_chunks(raw: bytes) -> list[bytes]:
    """Cut a stream into its events, each ending at its blank line."""
    return [chunk + b"\n\n" for chunk in raw.split(b"\n\n") if chunk.strip()]


def _reasoning_item(text: str, item_id: str = "", status: str = "") -> dict:
    """A reasoning input item with the given text."""
    item: dict = {"type": "reasoning", "summary": [], "content": [{"type": "reasoning_text", "text": text}]}
    if item_id:
        item["id"] = item_id
    if status:
        item["status"] = status
    return item


def _message_item(role: str, text: str) -> dict:
    """A message input item with the given role and text."""
    return {"type": "message", "role": role, "content": [{"type": "output_text" if role == "assistant" else "input_text", "text": text}]}


def _call_item(call_id: str, name: str = "run", arguments: str = "{}", item_id: str = "") -> dict:
    """A function call input item."""
    item = {"type": "function_call", "call_id": call_id, "name": name, "arguments": arguments}
    if item_id:
        item["id"] = item_id
    return item


def _output_item(call_id: str, output: str) -> dict:
    """A function call output input item."""
    return {"type": "function_call_output", "call_id": call_id, "output": output}


def _project(input_value: list | str, instructions: str | None = None) -> list:
    """The chat messages projected from the given input."""
    body: dict = {"input": input_value}
    if instructions is not None:
        body["instructions"] = instructions
    return _DIALECT.project(body).messages


# ── Projection ──


_FOLD_TABLE = [
    pytest.param("hello", None, [{"role": "user", "content": "hello"}], id="string-input"),
    pytest.param(
        [_message_item("user", "hi")], "be kind",
        [{"role": "system", "content": "be kind"}, {"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        id="instructions-first",
    ),
    pytest.param([{"role": "user", "content": "plain"}], "", [{"role": "user", "content": "plain"}], id="empty-instructions"),
    pytest.param(
        [{"role": "system", "content": "s"}, {"role": "developer", "content": [{"type": "input_text", "text": "d"}]}, {"role": "user", "content": "u"}],
        None,
        [{"role": "system", "content": "s"}, {"role": "developer", "content": [{"type": "text", "text": "d"}]}, {"role": "user", "content": "u"}],
        id="role-only-items",
    ),
    pytest.param(
        [{"role": "user", "content": [{"type": "input_image", "image_url": "x"}, {"type": "input_text", "text": "t"}]}],
        None,
        [{"role": "user", "content": [{"type": "input_image", "image_url": "x"}, {"type": "text", "text": "t"}]}],
        id="other-blocks-kept",
    ),
    pytest.param(
        [_reasoning_item("think"), _message_item("assistant", "say"), _call_item("c1"), _call_item("c2", "two", '{"a":1}')],
        None,
        [{
            "role": "assistant", "reasoning_content": "think", "content": "say",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "run", "arguments": "{}"}},
                {"id": "c2", "type": "function", "function": {"name": "two", "arguments": '{"a":1}'}},
            ],
        }],
        id="one-assistant-group",
    ),
    pytest.param(
        [{"role": "assistant", "content": "string answer"}],
        None, [{"role": "assistant", "content": "string answer"}], id="role-only-assistant-string",
    ),
    pytest.param(
        [{"type": "message", "role": "assistant", "content": "typed string"}],
        None, [{"role": "assistant", "content": "typed string"}], id="typed-assistant-string",
    ),
    pytest.param(
        [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "first"}, {"type": "output_text", "text": "second"}]}],
        None, [{"role": "assistant", "content": "first"}], id="assistant-first-block",
    ),
    pytest.param(
        [_message_item("assistant", "a"), {"role": "assistant", "content": []}, _call_item("c1")],
        None,
        [{"role": "assistant", "content": "a"}, {"role": "assistant", "content": [], "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "run", "arguments": "{}"}}]}],
        id="assistant-without-text-starts-group",
    ),
    pytest.param(
        [_call_item("c1"), _call_item("c2")],
        None,
        [{"role": "assistant", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "run", "arguments": "{}"}},
            {"id": "c2", "type": "function", "function": {"name": "run", "arguments": "{}"}},
        ]}],
        id="group-without-assistant-item-has-no-content",
    ),
    pytest.param(
        [_message_item("assistant", "only")], None, [{"role": "assistant", "content": "only"}],
        id="group-without-reasoning-item",
    ),
    pytest.param(
        [{"type": "reasoning", "summary": [], "content": []}, {"type": "reasoning", "summary": [{"type": "summary_text", "text": "sum"}]}],
        None,
        [{"role": "assistant", "reasoning_content": ""}, {"role": "assistant", "reasoning_content": "sum"}],
        id="empty-reasoning-then-second-reasoning",
    ),
    pytest.param(
        [_message_item("assistant", "a"), _message_item("assistant", "b")],
        None, [{"role": "assistant", "content": "a"}, {"role": "assistant", "content": "b"}],
        id="second-message-starts-group",
    ),
    pytest.param(
        [_message_item("assistant", "a"), _reasoning_item("late")],
        None, [{"role": "assistant", "content": "a", "reasoning_content": "late"}],
        id="reasoning-joins-group-without-reasoning",
    ),
    pytest.param(
        [_call_item("c1"), _output_item("c1", "out"), _call_item("c2")],
        None,
        [
            {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "run", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "out"},
            {"role": "assistant", "tool_calls": [{"id": "c2", "type": "function", "function": {"name": "run", "arguments": "{}"}}]},
        ],
        id="tool-result-ends-group",
    ),
    pytest.param(
        [{"type": "function_call_output", "call_id": "c1", "output": [{"type": "input_text", "text": "o"}]}],
        None, [{"role": "tool", "tool_call_id": "c1", "content": [{"type": "text", "text": "o"}]}],
        id="tool-output-blocks",
    ),
    pytest.param(
        [{"type": "item_reference", "id": "x"}, 7, "loose"],
        None, [{}, 7, "loose"], id="opaque-and-non-object-items",
    ),
]


class TestProjection:
    """Responses input reads as chat messages."""

    @pytest.mark.parametrize("input_value, instructions, expected", _FOLD_TABLE)
    def test_fold_table(self, input_value: list | str, instructions: str | None, expected: list) -> None:
        """Each input shape reads as the chat messages it stands for."""
        assert _project(input_value, instructions) == expected

    @pytest.mark.parametrize(
        "body",
        [{}, {"input": None}, {"input": 5}, {"input": {"role": "user"}}, ["input"]],
    )
    def test_no_usable_history_gives_none(self, body: dict | list) -> None:
        """A body with no input list or string gives no messages."""
        assert _DIALECT.project(body) is None

    def test_non_string_instructions_project_nothing(self) -> None:
        """Instructions that are not a string add no system message."""
        body = {"instructions": ["x"], "input": "hi"}
        assert _DIALECT.project(body).messages == [{"role": "user", "content": "hi"}]

    @pytest.mark.parametrize("input_value, instructions, _expected", _FOLD_TABLE)
    def test_round_trip_without_edits_leaves_body_untouched(
        self, input_value: list | str, instructions: str | None, _expected: list
    ) -> None:
        """Writing back with no edits leaves the body as it was."""
        body: dict = {"input": input_value, "status_marker": 1}
        if instructions is not None:
            body["instructions"] = instructions
        original = copy.deepcopy(body)
        original_input = body["input"]
        _DIALECT.project(body).write_back()
        assert body == original
        assert body["input"] is original_input
        assert body[_DIALECT.history_key] is original_input


def _group_body() -> dict:
    """A body with a user turn, an assistant group and its tool result."""
    return {
        "instructions": "rules",
        "input": [
            {"type": "message", "role": "user", "content": "go", "id": "u1"},
            {"type": "reasoning", "id": "rs1", "status": "completed", "summary": [{"type": "summary_text", "text": "s"}], "content": [{"type": "reasoning_text", "text": "think"}]},
            {"type": "message", "id": "m1", "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": "say", "annotations": []}]},
            {"type": "function_call", "id": "f1", "call_id": "c1", "name": "read", "arguments": '{"path": "a"}', "status": "completed"},
            {"type": "function_call_output", "id": "o1", "call_id": "c1", "output": "long result"},
        ],
    }


def _edit_and_write(body: dict, edit: Callable[[list], None]) -> list:
    """Apply an edit to the projected messages, write it back, and return the messages."""
    projection = _DIALECT.project(body)
    edit(projection.messages)
    projection.write_back()
    return projection.messages


# Positions of the user turn, the assistant group and the tool result once projected.
_USER = 1
_GROUP = 2
_TOOL = 3


def _set_reasoning(messages: list) -> None:
    """Set the reasoning of the assistant group."""
    messages[_GROUP]["reasoning_content"] = "restored"


def _pop_reasoning(messages: list) -> None:
    """Remove the reasoning of the assistant group."""
    messages[_GROUP].pop("reasoning_content")


def _truncate_args(messages: list) -> None:
    """Shorten the arguments of the first call."""
    messages[_GROUP]["tool_calls"][0]["function"]["arguments"] = '{"path": "..."}'


def _prune_tool(messages: list) -> None:
    """Replace the tool result with a pruned note."""
    messages[_TOOL]["content"] = "[tool result pruned]"


def _change_user(messages: list) -> None:
    """Change the text of the user turn."""
    messages[_USER]["content"] = "go now"


def _change_assistant(messages: list) -> None:
    """Change the text of the assistant group."""
    messages[_GROUP]["content"] = "said"


def _insert_system(messages: list) -> None:
    """Insert a system message before the assistant group."""
    messages.insert(_GROUP, {"role": "system", "content": "again"})


def _insert_turn(messages: list) -> None:
    """Insert an assistant turn with one call before the assistant group."""
    messages.insert(_GROUP, {"role": "assistant", "content": "", "reasoning_content": "why", "tool_calls": [
        {"id": "c9", "type": "function", "function": {"name": "grep", "arguments": "{}"}},
    ]})


def _append_retry_pair(messages: list) -> None:
    """Append a partial reply and a retry note."""
    messages.extend([
        {"role": "assistant", "content": "partial", "reasoning_content": "r"},
        {"role": "system", "content": "try again"},
    ])


class TestWriteBack:
    """Each edit the shared rules make lands on the one field it maps to."""

    def test_restoring_reasoning_patches_the_reasoning_item(self) -> None:
        """Restored reasoning is written into the group's reasoning item."""
        body = _group_body()
        _edit_and_write(body, _set_reasoning)
        item = body["input"][1]
        assert item == {"type": "reasoning", "id": "rs1", "status": "completed", "summary": [], "content": [{"type": "reasoning_text", "text": "restored"}]}
        assert body["input"][2:] == _group_body()["input"][2:]

    def test_restoring_reasoning_adds_an_item_when_the_group_has_none(self) -> None:
        """Restored reasoning adds a reasoning item to a group that has none."""
        body = {"input": [_message_item("user", "u"), _call_item("c1", item_id="f1")]}
        projection = _DIALECT.project(body)
        projection.messages[1]["reasoning_content"] = "new"
        projection.write_back()
        assert body["input"] == [
            _message_item("user", "u"),
            {"type": "reasoning", "summary": [], "content": [{"type": "reasoning_text", "text": "new"}]},
            _call_item("c1", item_id="f1"),
        ]

    def test_restoring_reasoning_goes_after_a_leading_message_without_text(self) -> None:
        """Restored reasoning follows a leading message without text and keeps one group."""
        empty_message = {"type": "message", "role": "assistant", "content": []}
        body = {"input": [_message_item("user", "u"), empty_message, _call_item("c1"), _output_item("c1", "out")]}
        projection = _DIALECT.project(body)
        projection.messages[1]["reasoning_content"] = "think"
        projection.write_back()
        assert body["input"] == [
            _message_item("user", "u"),
            empty_message,
            _reasoning_item("think"),
            _call_item("c1"),
            _output_item("c1", "out"),
        ]
        assert _project(body["input"]) == [
            {"role": "user", "content": [{"type": "text", "text": "u"}]},
            {
                "role": "assistant", "content": [], "reasoning_content": "think",
                "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "run", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "out"},
        ]

    def test_restoring_reasoning_after_a_text_group_stays_in_its_own_group(self) -> None:
        """Restored reasoning on a text group right after another text group stays with its own text."""
        body = {"input": [_message_item("assistant", "a"), _message_item("assistant", "b")]}
        projection = _DIALECT.project(body)
        projection.messages[1]["reasoning_content"] = "why b"
        projection.write_back()
        assert body["input"] == [_message_item("assistant", "a"), _message_item("assistant", "b"), _reasoning_item("why b")]
        assert _project(body["input"]) == [
            {"role": "assistant", "content": "a"},
            {"role": "assistant", "content": "b", "reasoning_content": "why b"},
        ]

    def test_restoring_reasoning_on_a_text_group_after_a_user_turn_goes_first(self) -> None:
        """Restored reasoning on a text group after a user turn goes before the text."""
        body = {"input": [_message_item("user", "u"), _message_item("assistant", "b")]}
        projection = _DIALECT.project(body)
        projection.messages[1]["reasoning_content"] = "why b"
        projection.write_back()
        assert body["input"] == [_message_item("user", "u"), _reasoning_item("why b"), _message_item("assistant", "b")]
        assert _project(body["input"])[1] == {"role": "assistant", "content": "b", "reasoning_content": "why b"}

    def test_restoring_into_an_empty_reasoning_item_keeps_one_group(self) -> None:
        """Reasoning put into an empty reasoning item keeps its calls in one group."""
        body = {"input": [
            _message_item("user", "u"),
            {"type": "reasoning", "id": "rs", "summary": [], "content": []},
            _call_item("c1"),
            _call_item("c2"),
        ]}
        projection = _DIALECT.project(body)
        projection.messages[1]["reasoning_content"] = "full"
        projection.write_back()
        assert [item["type"] for item in body["input"]] == ["message", "reasoning", "function_call", "function_call"]
        assert body["input"][1] == {"type": "reasoning", "id": "rs", "summary": [], "content": [{"type": "reasoning_text", "text": "full"}]}
        assert _project(body["input"])[1] == {
            "role": "assistant", "reasoning_content": "full",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "run", "arguments": "{}"}},
                {"id": "c2", "type": "function", "function": {"name": "run", "arguments": "{}"}},
            ],
        }

    def test_pruning_reasoning_removes_the_item(self) -> None:
        """Pruned reasoning removes the reasoning item."""
        body = _group_body()
        _edit_and_write(body, _pop_reasoning)
        assert [item["id"] for item in body["input"]] == ["u1", "m1", "f1", "o1"]

    def test_pruning_reasoning_after_another_group_keeps_an_emptied_item(self) -> None:
        """Pruned reasoning that follows another group leaves an empty reasoning item."""
        body = {"input": [
            _message_item("user", "u"),
            _reasoning_item("first thoughts"),
            _message_item("assistant", "first"),
            _reasoning_item("second thoughts", "rs2", "completed"),
            _call_item("c1", item_id="f1"),
        ]}
        projection = _DIALECT.project(body)
        assert len(projection.messages) == 3
        projection.messages[2].pop("reasoning_content")
        projection.write_back()
        assert body["input"][3] == {"type": "reasoning", "id": "rs2", "status": "completed", "summary": [], "content": [{"type": "reasoning_text", "text": ""}]}
        reprojected = _project(body["input"])
        assert len(reprojected) == 3
        assert reprojected[2]["reasoning_content"] == ""
        assert reprojected[2]["tool_calls"][0]["id"] == "c1"

    def test_pruning_an_already_empty_reasoning_item_leaves_it(self) -> None:
        """An empty reasoning item that is pruned stays as it was."""
        empty = {"type": "reasoning", "id": "rs", "summary": [], "content": []}
        body = {"input": [_message_item("user", "u"), empty, _call_item("c1")]}
        projection = _DIALECT.project(body)
        projection.messages[1].pop("reasoning_content")
        projection.write_back()
        assert body["input"][1] is empty

    def test_truncated_arguments_patch_the_call(self) -> None:
        """Shortened call arguments are written into the call item."""
        body = _group_body()
        _edit_and_write(body, _truncate_args)
        assert body["input"][3] == {"type": "function_call", "id": "f1", "call_id": "c1", "name": "read", "arguments": '{"path": "..."}', "status": "completed"}

    def test_pruned_tool_result_patches_the_output(self) -> None:
        """A pruned tool result is written into the output item."""
        body = _group_body()
        _edit_and_write(body, _prune_tool)
        assert body["input"][4] == {"type": "function_call_output", "id": "o1", "call_id": "c1", "output": "[tool result pruned]"}

    def test_user_content_patches_the_item(self) -> None:
        """Changed user text is written into the user item."""
        body = _group_body()
        _edit_and_write(body, _change_user)
        assert body["input"][0] == {"type": "message", "role": "user", "content": "go now", "id": "u1"}

    def test_instructions_message_patches_instructions(self) -> None:
        """A changed first system message is written into the instructions."""
        body = _group_body()
        _edit_and_write(body, lambda messages: messages[0].update(content="new rules"))
        assert body["instructions"] == "new rules"
        assert body["input"] == _group_body()["input"]

    @pytest.mark.parametrize("rules", ["fresh rules", "other rules"])
    def test_replaced_instructions_message_becomes_instructions(self, rules: str) -> None:
        """A new first system message in place of the instructions one is written as the instructions."""
        body = _group_body()

        def replace(messages: list) -> None:
            messages[0] = {"role": "system", "content": rules}

        messages = _edit_and_write(body, replace)
        assert body["instructions"] == rules
        assert body["input"] == _group_body()["input"]
        assert _DIALECT.project(body).messages == messages

    @pytest.mark.parametrize(
        "input_value",
        [
            _group_body()["input"],
            [{"type": "message", "role": "system", "content": "keep"}, {"type": "message", "role": "user", "content": "go"}],
        ],
        ids=["group", "system-item-first"],
    )
    def test_deleted_instructions_message_removes_instructions(self, input_value: list) -> None:
        """Deleting the instructions message removes the instructions and keeps the input."""
        body = {"instructions": "rules", "input": copy.deepcopy(input_value)}

        def delete(messages: list) -> None:
            del messages[0]

        messages = _edit_and_write(body, delete)
        assert body == {"input": input_value}
        assert _DIALECT.project(body).messages == messages

    @pytest.mark.parametrize(
        "item, expected_content",
        [
            ({"role": "assistant", "content": "old", "id": "a"}, "new"),
            ({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "old", "annotations": [1]}], "id": "a"}, [{"type": "output_text", "text": "new", "annotations": [1]}]),
        ],
        ids=["string", "blocks"],
    )
    def test_assistant_content_patches_its_item(self, item: dict, expected_content: str | list) -> None:
        """Changed assistant text is written into its item in the same shape."""
        body = {"input": [_message_item("user", "u"), item]}
        projection = _DIALECT.project(body)
        projection.messages[1]["content"] = "new"
        projection.write_back()
        assert body["input"][1] == {**item, "content": expected_content}

    def test_inserted_system_message_renders_one_item(self) -> None:
        """An added system message becomes one message item."""
        body = _group_body()
        _edit_and_write(body, _insert_system)
        assert body["input"][1] == {"type": "message", "role": "system", "content": "again"}
        assert len(body["input"]) == len(_group_body()["input"]) + 1

    def test_inserted_assistant_turn_renders_reasoning_message_and_calls(self) -> None:
        """An added assistant turn becomes reasoning, message and call items."""
        body = {"input": [_message_item("user", "u"), _output_item("c1", "res")]}
        projection = _DIALECT.project(body)
        projection.messages.insert(1, {
            "role": "assistant", "content": "text", "reasoning_content": "why",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "grep", "arguments": '{"q": 1}'}}],
        })
        projection.write_back()
        assert body["input"][1:4] == [
            {"type": "reasoning", "summary": [], "content": [{"type": "reasoning_text", "text": "why"}]},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "text"}]},
            {"type": "function_call", "call_id": "c1", "name": "grep", "arguments": '{"q": 1}'},
        ]

    @pytest.mark.parametrize(
        "message, expected",
        [
            ({"role": "assistant", "content": ""}, [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": ""}]}]),
            ({"role": "assistant", "content": "", "tool_calls": [{"id": "c", "function": {"name": "n", "arguments": "{}"}}]}, [{"type": "function_call", "call_id": "c", "name": "n", "arguments": "{}"}]),
            ({"role": "tool", "tool_call_id": "c", "content": "out"}, [_output_item("c", "out")]),
            ({"role": "user", "content": "more"}, [{"type": "message", "role": "user", "content": "more"}]),
        ],
        ids=["empty-turn", "calls-only", "tool", "user"],
    )
    def test_inserted_message_shapes(self, message: dict, expected: list) -> None:
        """Each kind of added message becomes the items it stands for."""
        body = {"input": [_message_item("user", "u")]}
        projection = _DIALECT.project(body)
        projection.messages.append(message)
        projection.write_back()
        assert body["input"][1:] == expected

    def test_insert_into_string_input_writes_list_form(self) -> None:
        """A message added to string input turns the input into a list."""
        body = {"input": "question"}
        projection = _DIALECT.project(body)
        projection.messages.append({"role": "system", "content": "note"})
        projection.write_back()
        assert body["input"] == [
            {"type": "message", "role": "user", "content": "question"},
            {"type": "message", "role": "system", "content": "note"},
        ]

    def test_opaque_and_non_object_items_are_kept_verbatim(self) -> None:
        """Items the reader does not know are kept as they were."""
        opaque = {"type": "item_reference", "id": "x"}
        body = {"input": [opaque, 7, _message_item("user", "u")]}
        projection = _DIALECT.project(body)
        projection.messages.append({"role": "system", "content": "s"})
        projection.write_back()
        assert body["input"][:3] == [opaque, 7, _message_item("user", "u")]
        assert body["input"][0] is opaque


@pytest.mark.parametrize(
    "edit",
    [_set_reasoning, _pop_reasoning, _truncate_args, _prune_tool, _change_user, _change_assistant, _insert_system, _insert_turn, _append_retry_pair],
    ids=lambda edit: edit.__name__.strip("_"),
)
def test_reprojecting_written_back_body_gives_edited_messages(edit: Callable[[list], None]) -> None:
    """A second projection reads the edit made before write-back."""
    body = _group_body()
    projection = _DIALECT.project(body)
    edit(projection.messages)
    edited = _without_empty_assistant_content(projection.messages)
    projection.write_back()
    assert _DIALECT.project(body).messages == edited


def _without_empty_assistant_content(messages: list) -> list:
    """The edited messages, minus the empty content an inserted turn renders no item for."""
    return [
        {k: v for k, v in m.items() if not (k == "content" and v == "" and m.get("role") == "assistant")}
        for m in copy.deepcopy(messages)
    ]


def test_turn_inserted_right_after_a_group_folds_like_the_backend() -> None:
    """A rendered turn placed straight after an assistant group joins that group."""
    body = {"input": [_message_item("user", "u"), _message_item("assistant", "end")]}
    projection = _DIALECT.project(body)
    projection.messages.append({"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "function": {"name": "n", "arguments": "{}"}}]})
    projection.write_back()
    assert _project(body["input"]) == [
        {"role": "user", "content": [{"type": "text", "text": "u"}]},
        {"role": "assistant", "content": "end", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "n", "arguments": "{}"}}]},
    ]


class TestPromptText:
    """The prompt text is read from a Responses body."""

    def test_counts_instructions_messages_and_tool_output(self) -> None:
        """Prompt text holds instructions, message text and tool output, not reasoning or calls."""
        body = {"instructions": "rules", "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "q"}]},
            _reasoning_item("hidden"),
            _call_item("c1", arguments='{"no": 1}'),
            _output_item("c1", "result"),
            {"type": "item_reference", "id": "x"},
        ]}
        assert _DIALECT.prompt_text(body) == "rules\nq\n\nresult\n\n"

    def test_string_input(self) -> None:
        """String input is counted as one line of prompt text."""
        assert _DIALECT.prompt_text({"input": "hello"}) == "hello\n"

    def test_missing_input_counts_nothing(self) -> None:
        """A body with no input gives no prompt text."""
        assert _DIALECT.prompt_text({"model": "x"}) == ""

    def test_malformed_input_raises(self) -> None:
        """Input that is not a list or string raises."""
        with pytest.raises(ValueError, match="input is not a list"):
            _DIALECT.prompt_text({"input": 5})


# ── Stream reading ──


def _decode(raw: bytes) -> list[dict]:
    """The pieces decoded from a raw stream."""
    decoder = _DIALECT.delta_decoder()
    deltas: list[dict] = []
    for payload in parse_sse_payloads(raw):
        pieces, carrier = decoder.normalize(payload)
        assert len(pieces) <= 1
        assert carrier == (0 if pieces else None)
        deltas.extend(pieces)
    return deltas


def _terminal_output(raw: bytes) -> list[dict]:
    """The output items of the last completed event."""
    return [p for p in parse_sse_payloads(raw) if p["type"] == "response.completed"][-1]["response"]["output"]


def _item_text(item: dict) -> str:
    """The text of one output item."""
    if item["type"] == "reasoning":
        blocks = item.get("content") or item.get("summary") or []
        return blocks[0]["text"] if blocks else ""
    return "".join(block["text"] for block in item["content"])


def _decoded_text(deltas: list[dict], kind: str) -> str:
    """The joined text of the decoded pieces of one kind."""
    return "".join(d["text"] for d in deltas if d["type"] == kind)


def _items_text(output: list[dict], item_type: str) -> str:
    """The joined text of the output items of one type."""
    return "".join(_item_text(item) for item in output if item["type"] == item_type)


def _decoded_calls(deltas: list[dict]) -> list[tuple[str, dict]]:
    """The name and parsed arguments of each decoded call, in the order first seen."""
    calls: dict[str, list[str]] = {}
    for delta in deltas:
        if delta["type"] == "tool_call":
            call = calls.setdefault(delta["tool_call_id"], ["", ""])
            call[0] += delta["function"]["name"]
            call[1] += delta["function"]["arguments"]
    return [(name, json.loads(arguments)) for name, arguments in calls.values()]


def _terminal_calls(output: list[dict]) -> list[tuple[str, dict]]:
    """The name and parsed arguments of each call in the output."""
    return [(item["name"], json.loads(item["arguments"])) for item in output if item["type"] == "function_call"]


def _is_text_event(payload: dict) -> bool:
    """Tell whether an event adds generated text."""
    if payload["type"].endswith(".delta"):
        return True
    return payload["type"] == "response.output_item.added" and payload["item"]["type"] == "function_call"


def _assert_reconstruction(raw: bytes) -> list[dict]:
    """Check that decoded content, reasoning and calls equal the terminal output, and return the deltas."""
    deltas = _decode(raw)
    output = _terminal_output(raw)
    assert _decoded_text(deltas, "content") == _items_text(output, "message")
    assert _decoded_text(deltas, "reasoning") == _items_text(output, "reasoning")
    assert _decoded_calls(deltas) == _terminal_calls(output)
    return deltas


class TestDecoderOnBackendStreams:
    """Real backend streams decode to their final items."""

    @pytest.mark.parametrize("name", _STREAM_NAMES)
    def test_decoded_text_equals_terminal_items(self, name: str) -> None:
        """The text read from each backend stream matches its final items."""
        raw = _raw(name)
        assert {p["type"] for p in parse_sse_payloads(raw)} <= _SPEC_KNOWN_TYPES
        deltas = _assert_reconstruction(raw)
        assert _decoded_text(deltas, "reasoning").startswith("The user")

    @pytest.mark.parametrize("name", _STREAM_NAMES)
    def test_one_finish_reason_reports_the_streamed_call(self, name: str) -> None:
        """Each backend stream ends with one tool calls finish reason."""
        finishes = [d for d in _decode(_raw(name)) if d["type"] == "finish_reason"]
        assert [d["text"] for d in finishes] == ["tool_calls"]

    def test_vllm_terminal_ids_alias_streamed_ids(self) -> None:
        """The vLLM stream pairs each streamed call id with its final call id."""
        finish = [d for d in _decode(_raw("vllm")) if d["type"] == "finish_reason"][0]
        assert finish["tool_call_id_aliases"] == [("call_b5821fd3f0ab9243", "chatcmpl-tool-bab2358b7fb9b9f3")]

    @pytest.mark.parametrize("name, call_id", [("sglang", "call_sgl1"), ("llamacpp", "call_lcpp1")])
    def test_equal_ids_pair_with_themselves(self, name: str, call_id: str) -> None:
        """A stream whose final call ids match the streamed ones pairs each id with itself."""
        finish = [d for d in _decode(_raw(name)) if d["type"] == "finish_reason"][0]
        assert finish["tool_call_id_aliases"] == [(call_id, call_id)]

    def test_argument_deltas_carry_the_call_id_of_their_item(self) -> None:
        """Argument pieces carry the call id of the item they belong to."""
        calls = [d for d in _decode(_raw("vllm")) if d["type"] == "tool_call"]
        assert {d["tool_call_id"] for d in calls} == {"call_b5821fd3f0ab9243"}
        assert calls[0]["text"] == "get_weather"
        assert "".join(d["function"]["arguments"] for d in calls[1:]) == '{"city": "Paris"}'


def _finish_of(raw: bytes) -> list[dict]:
    """The finish reason pieces decoded from a raw stream."""
    decoder = _DIALECT.delta_decoder()
    pieces: list[dict] = []
    for payload in parse_sse_payloads(raw):
        pieces.extend(decoder.normalize(payload)[0])
    return [p for p in pieces if p["type"] == "finish_reason"]


def _terminal(status: str, incomplete_reason: str = "", event_type: str = "response.completed", output: list | None = None) -> bytes:
    """A final event with the given status and output."""
    response: dict = {"id": "r", "status": status, "output": output or []}
    if incomplete_reason:
        response["incomplete_details"] = {"reason": incomplete_reason}
    return responses_event({"type": event_type, "response": response})


class TestTerminalMapping:
    """The final event maps to a chat finish reason."""

    @pytest.mark.parametrize(
        "status, has_call, incomplete_reason, expected",
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
    def test_reason_from_status_and_streamed_call(self, status: str, has_call: bool, incomplete_reason: str, expected: str) -> None:
        """The finish reason comes from the final status and whether a call was streamed."""
        stream = ResponsesStream("r")
        raw = stream.start() + (stream.tool_call("c1", "run") if has_call else stream.reasoning("x"))
        raw += _terminal(status, incomplete_reason, output=stream.output)
        assert [f["text"] for f in _finish_of(raw)] == [expected]

    def test_named_tool_choice_with_streamed_call(self) -> None:
        """A forced tool choice with a streamed call still finishes with tool calls."""
        stream = ResponsesStream("r")
        raw = stream.start() + stream.tool_call("c1", "run")
        response = {"id": "r", "status": "completed", "tool_choice": {"type": "function", "name": "run"}, "output": stream.output}
        raw += responses_event({"type": "response.completed", "response": response})
        assert [f["text"] for f in _finish_of(raw)] == ["tool_calls"]

    def test_terminal_call_that_was_not_streamed_gives_stop(self) -> None:
        """A call that only the final event lists does not count as a streamed call."""
        stream = ResponsesStream("r")
        raw = stream.start() + stream.content("words") + _terminal("completed", output=[_call_item("c1")])
        finish = _finish_of(raw)
        assert [f["text"] for f in finish] == ["stop"]
        assert finish[0]["tool_call_id_aliases"] == []

    def test_streamed_call_missing_from_terminal_gives_tool_calls(self) -> None:
        """A streamed call missing from the final event still finishes with tool calls."""
        stream = ResponsesStream("r")
        raw = stream.start() + stream.tool_call("c1", "run") + _terminal("completed", output=[])
        finish = _finish_of(raw)
        assert [f["text"] for f in finish] == ["tool_calls"]
        assert finish[0]["tool_call_id_aliases"] == []

    def test_failed_gives_no_piece(self) -> None:
        """A failed stream gives no finish reason."""
        stream = ResponsesStream("r")
        raw = stream.start() + stream.content("partial") + stream.failed()
        assert _finish_of(raw) == []
        assert _DIALECT.delta_decoder().normalize(parse_sse_payloads(stream.failed())[0]) == ([], None)

    @pytest.mark.parametrize("has_call, expected", [(False, "length"), (True, "length")])
    def test_incomplete_event_type_maps_like_incomplete_status(self, has_call: bool, expected: str) -> None:
        """An incomplete event finishes with length, like an incomplete status."""
        stream = ResponsesStream("r")
        raw = stream.start() + (stream.tool_call("c1", "run") if has_call else b"")
        raw += _terminal("in_progress", event_type="response.incomplete")
        assert [f["text"] for f in _finish_of(raw)] == [expected]

    def test_name_mismatch_records_no_aliases(self) -> None:
        """Calls whose names differ are not paired."""
        stream = ResponsesStream("r", terminal_call_ids={"c1": "t1"})
        raw = stream.start() + stream.tool_call("c1", "run") + _terminal("completed", output=[_call_item("t1", name="other")])
        assert _finish_of(raw)[0]["tool_call_id_aliases"] == []

    def test_count_mismatch_records_no_aliases(self) -> None:
        """Calls are not paired when the counts differ."""
        stream = ResponsesStream("r")
        raw = stream.start() + stream.tool_call("c1", "run") + _terminal("completed", output=[_call_item("t1"), _call_item("t2")])
        assert _finish_of(raw)[0]["tool_call_id_aliases"] == []

    def test_matching_calls_pair_in_order(self) -> None:
        """Matching calls are paired in the order they came."""
        stream = ResponsesStream("r", terminal_call_ids={"c1": "t1", "c2": "t2"})
        raw = stream.start() + stream.tool_call("c1", "run") + stream.tool_call("c2", "other") + stream.completed()
        assert _finish_of(raw)[0]["tool_call_id_aliases"] == [("c1", "t1"), ("c2", "t2")]


class TestDecoderEvents:
    """Each stream event gives the pieces it carries."""

    def test_unknown_item_gives_empty_call_id(self) -> None:
        """Arguments for an item the reader never saw get an empty call id."""
        payload = {"type": "response.function_call_arguments.delta", "item_id": "nope", "delta": "{}"}
        deltas, carrier = _DIALECT.delta_decoder().normalize(payload)
        assert deltas == [{"type": "tool_call", "text": "{}", "tool_call_id": "", "function": {"name": "", "arguments": "{}"}}]
        assert carrier == 0

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
    def test_events_without_pieces(self, payload: dict | list) -> None:
        """Events that carry no new text give no pieces."""
        assert _DIALECT.delta_decoder().normalize(payload) == ([], None)

    @pytest.mark.parametrize(
        "event_type, piece_type",
        [
            ("response.output_text.delta", "content"),
            ("response.reasoning_text.delta", "reasoning"),
            ("response.reasoning_summary_text.delta", "reasoning"),
        ],
    )
    def test_text_deltas(self, event_type: str, piece_type: str) -> None:
        """Each kind of text event gives one piece of its kind."""
        deltas, _ = _DIALECT.delta_decoder().normalize({"type": event_type, "delta": "abc"})
        assert deltas == [{"type": piece_type, "text": "abc"}]

    def test_call_opening_canonicalizes_initial_arguments(self) -> None:
        """Arguments sent with a call opening are rewritten in one JSON form."""
        payload = {"type": "response.output_item.added", "item": {"type": "function_call", "id": "f", "call_id": "c", "name": "n", "arguments": '{"a":1}'}}
        deltas, _ = _DIALECT.delta_decoder().normalize(payload)
        assert deltas == [{"type": "tool_call", "text": 'n{"a": 1}', "tool_call_id": "c", "function": {"name": "n", "arguments": '{"a": 1}'}}]


class TestGeneratedTexts:
    """Each event gives the text it adds."""

    @pytest.mark.parametrize("name", _STREAM_NAMES)
    def test_texts_match_terminal_items(self, name: str) -> None:
        """The generated texts of each backend stream match its final items."""
        raw = _raw(name)
        texts = [t for p in parse_sse_payloads(raw) for t in _DIALECT.generated_texts(p)]
        _assert_texts_match_items(parse_sse_payloads(raw), _terminal_output(raw))
        assert len(texts) == sum(1 for p in parse_sse_payloads(raw) if _is_text_event(p))

    @pytest.mark.parametrize(
        "payload, expected",
        [
            ({"type": "response.output_text.delta", "delta": "a"}, ["a"]),
            ({"type": "response.reasoning_text.delta", "delta": "b"}, ["b"]),
            ({"type": "response.reasoning_summary_text.delta", "delta": "c"}, ["c"]),
            ({"type": "response.function_call_arguments.delta", "delta": "{"}, ["{"]),
            ({"type": "response.output_item.added", "item": {"type": "function_call", "name": "n", "arguments": '{"x":1}'}}, ['n{"x":1}']),
            ({"type": "response.output_item.added", "item": {"type": "message"}}, []),
            ({"type": "response.output_text.done", "text": "a"}, []),
            ({"type": "response.completed", "response": {}}, []),
        ],
    )
    def test_one_entry_per_generated_text_event(self, payload: dict, expected: list[str]) -> None:
        """Each event gives one generated text entry when it carries new text."""
        assert _DIALECT.generated_texts(payload) == expected


class TestUsage:
    """Token counts are read from the final event."""

    def test_terminal_usage(self) -> None:
        """Usage is read from the final event."""
        assert _DIALECT.usage(parse_sse_payloads(_raw("vllm"))[-1]) == TokenUsage(_INPUT_TOKENS, _OUTPUT_TOKENS, _TOTAL_TOKENS)

    @pytest.mark.parametrize("event_type", ["response.incomplete", "response.failed"])
    def test_other_terminals(self, event_type: str) -> None:
        """Usage is read from incomplete and failed final events too."""
        event = {"type": event_type, "response": {"usage": {"input_tokens": 3, "output_tokens": 4}}}
        assert _DIALECT.usage(event) == TokenUsage(3, 4, 0)

    @pytest.mark.parametrize(
        "event",
        [
            {"type": "response.completed", "response": {"usage": None}},
            {"type": "response.created", "response": {"usage": {"input_tokens": 1}}},
            {"type": "response.completed"},
            {"usage": {"prompt_tokens": 1}},
        ],
    )
    def test_no_usage(self, event: dict) -> None:
        """Events without usage give none."""
        assert _DIALECT.usage(event) is None


class TestForceStream:
    """Forcing a stream sets only the stream field."""

    def test_sets_stream_only(self) -> None:
        """Forcing a stream sets stream and adds nothing else."""
        body = {"stream": False}
        _DIALECT.force_stream(body)
        assert body == {"stream": True}


# ── Overrides ──


class TestTranslateOverrides:
    """Chat style overrides are renamed for Responses."""

    @pytest.mark.parametrize(
        "overrides, forced, yielding",
        [
            ({"max_tokens": 10}, {}, {"max_output_tokens": 10}),
            ({"max_tokens": 10, "max_output_tokens": 99}, {}, {"max_output_tokens": 10}),
            ({"max_completion_tokens": 20}, {"max_output_tokens": 20}, {}),
            ({"max_completion_tokens": 20, "max_tokens": 10, "max_output_tokens": 99}, {"max_output_tokens": 20}, {}),
            ({"max_output_tokens": 99}, {"max_output_tokens": 99}, {}),
            ({"reasoning_effort": "low"}, {"reasoning": {"effort": "low"}}, {}),
            ({"reasoning_effort": "low", "reasoning": {"effort": "high", "summary": "auto"}}, {"reasoning": {"effort": "low", "summary": "auto"}}, {}),
            ({"reasoning": {"effort": "high"}}, {"reasoning": {"effort": "high"}}, {}),
            ({"temperature": 0.5, "min_p": 0.1, "chat_template_kwargs": {"x": True}}, {"temperature": 0.5, "min_p": 0.1, "chat_template_kwargs": {"x": True}}, {}),
        ],
        ids=[
            "max-tokens-yields", "max-tokens-beats-native", "max-completion-forced", "max-completion-wins",
            "native-alone", "effort-renamed", "effort-replaces-native", "native-effort-alone", "others-unchanged",
        ],
    )
    def test_renames(self, overrides: dict, forced: dict, yielding: dict) -> None:
        """Chat override names become the names Responses uses."""
        assert _DIALECT.translate_overrides(overrides) == TranslatedOverrides(forced, yielding)

    def test_never_changes_the_config(self) -> None:
        """Translating overrides leaves the given overrides as they were."""
        overrides = {"reasoning_effort": "low", "reasoning": {"effort": "high"}}
        _DIALECT.translate_overrides(overrides)
        assert overrides == {"reasoning_effort": "low", "reasoning": {"effort": "high"}}


# ── Output splicing ──


def _splice_runs(runs: list) -> tuple[bytes, list[bytes]]:
    """Feed runs to a fresh splicer. A None run stands for a retry."""
    splicer = _DIALECT.output_splicer()
    outputs: list[bytes] = []
    for run in runs:
        if run is None:
            splicer.retry()
            continue
        outputs.append(splicer.splice(run))
    tail = splicer.finish()
    return b"".join(outputs) + tail, outputs + [tail]


def _without_numbers(payloads: list[dict]) -> list[dict]:
    """The events without the numbers and terminal fields a retry gives new values."""
    stripped = []
    for payload in payloads:
        event = {k: v for k, v in payload.items() if k not in ("sequence_number", "output_index")}
        if "response" in event:
            event["response"] = {k: v for k, v in event["response"].items() if k not in ("id", "output")}
        stripped.append(event)
    return stripped


def _data_only_chunks(raw: bytes) -> list[bytes]:
    """Cut a stream into its events with the event lines left out."""
    return [
        b"".join(line for line in chunk.splitlines(keepends=True) if not line.startswith(b"event:"))
        for chunk in _event_chunks(raw)
    ]


def _first_delta_index(chunks: list[bytes], item_position: int) -> int | None:
    """Index of the first text or argument delta of the n-th item, or None if it has none."""
    seen = -1
    for index, chunk in enumerate(chunks):
        payload = parse_sse_payloads(chunk)[0]
        if payload["type"] == "response.output_item.added":
            seen += 1
        if seen == item_position and payload["type"].endswith(".delta"):
            return index
    return None


def _cut_after_first_delta(chunks: list[bytes], item_position: int) -> int:
    """Index just past the first delta of the n-th item, which must have one."""
    index = _first_delta_index(chunks, item_position)
    if index is None:
        pytest.fail("item has no delta")
    return index + 1


def _cut_and_retry(name: str, item_position: int) -> tuple[bytes, list[bytes], int]:
    """Splice a stream cut after its first delta with a full retry."""
    chunks = _event_chunks(_raw(name))
    cut = _cut_after_first_delta(chunks, item_position)
    runs = chunks[:cut] + [None, chunks[cut]] + chunks
    output, outputs = _splice_runs(runs)
    return output, outputs, cut


class TestSplicerSingleAttempt:
    """One attempt passes through the splicer unchanged."""

    @pytest.mark.parametrize("name", _STREAM_NAMES)
    def test_byte_identical(self, name: str) -> None:
        """One attempt passes through byte for byte."""
        chunks = _event_chunks(_raw(name))
        output, outputs = _splice_runs(chunks)
        assert output == _raw(name)
        assert outputs[-1] == b""

    def test_trailing_leftover_without_retry_is_identical(self) -> None:
        """A cut off last event without a retry passes through as it was."""
        chunks = _event_chunks(_raw("vllm"))[:5] + [b"data: {\"type\": \"respon"]
        output, outputs = _splice_runs(chunks)
        assert output == b"".join(chunks)
        assert outputs[-1] == b""

    def test_empty_run_changes_nothing(self) -> None:
        """Empty runs give empty output."""
        splicer = _DIALECT.output_splicer()
        assert splicer.splice(b"") == b""
        splicer.retry()
        assert splicer.splice(b"") == b""
        assert splicer.finish() == b""


class TestSplicerRetry:
    """Retried attempts join into one valid stream."""

    @pytest.mark.parametrize("name", _STREAM_NAMES)
    @pytest.mark.parametrize("item_position", [0, 1])
    def test_one_valid_stream(self, name: str, item_position: int) -> None:
        """A stream cut and retried reads as one stream that ends with the last attempt."""
        output, _, _ = _cut_and_retry(name, item_position)
        payloads = parse_sse_payloads(output)
        types = [p["type"] for p in payloads]
        assert types.count("response.created") == 1
        assert types.count("response.completed") == 1
        assert types[-1] == "response.completed"
        attempt_two = parse_sse_payloads(_raw(name))
        terminal = payloads[-1]["response"]
        assert terminal["id"] == attempt_two[0]["response"]["id"]
        own = attempt_two[-1]["response"]["output"]
        earlier = terminal["output"][: len(terminal["output"]) - len(own)]
        assert terminal["output"][len(earlier):] == own
        assert len(earlier) == item_position + 1
        assert earlier[-1]["status"] == "incomplete"
        assert terminal["usage"] == attempt_two[-1]["response"]["usage"]

    @pytest.mark.parametrize("name", _NUMBERED_STREAMS)
    @pytest.mark.parametrize("item_position", [0, 1])
    def test_numbered_stream_counts_on(self, name: str, item_position: int) -> None:
        """A numbered stream cut and retried keeps its event and item numbers counting up."""
        output, _, _ = _cut_and_retry(name, item_position)
        payloads = parse_sse_payloads(output)
        sequence = [p["sequence_number"] for p in payloads]
        assert sequence == list(range(len(sequence)))
        added = [p["output_index"] for p in payloads if p["type"] == "response.output_item.added"]
        assert added == list(range(len(added)))
        done = [p["output_index"] for p in payloads if p["type"] == "response.output_item.done"]
        assert done == list(range(len(done)))

    @pytest.mark.parametrize("name", _UNNUMBERED_STREAMS)
    @pytest.mark.parametrize("item_position", [0, 1])
    def test_unnumbered_stream_stays_unnumbered(self, name: str, item_position: int) -> None:
        """A stream without numbers gets none added when cut and retried."""
        output, _, _ = _cut_and_retry(name, item_position)
        assert all("sequence_number" not in p and "output_index" not in p for p in parse_sse_payloads(output))

    @pytest.mark.parametrize("name", _STREAM_NAMES)
    def test_cut_item_is_closed_with_the_approved_text(self, name: str) -> None:
        """The cut item is closed with the text the client already got."""
        chunks = _event_chunks(_raw(name))
        cut = _cut_after_first_delta(chunks, 0)
        sent_deltas = [parse_sse_payloads(c)[0]["delta"] for c in chunks[: cut + 1] if parse_sse_payloads(c)[0]["type"].endswith(".delta")]
        output, _ = _splice_runs(chunks[:cut] + [None, chunks[cut]] + chunks)
        payloads = parse_sse_payloads(output)
        closed = [p for p in payloads if p["type"] == "response.output_item.done"][0]["item"]
        assert closed["status"] == "incomplete"
        assert _item_text(closed) == "".join(sent_deltas)
        text_done = [p for p in payloads if p["type"] == "response.reasoning_text.done"][0]
        assert text_done["text"] == "".join(sent_deltas)

    def test_closing_comes_before_the_next_attempt(self) -> None:
        """The carried-over delta is the first attempt's. The closings ride on the next attempt's first run."""
        _output, outputs, cut = _cut_and_retry("vllm", 0)
        assert [p["type"] for p in parse_sse_payloads(outputs[cut])] == ["response.reasoning_text.delta"]
        assert [p["type"] for p in parse_sse_payloads(outputs[cut + 1])] == [
            "response.reasoning_text.done", "response.reasoning_part.done", "response.output_item.done",
        ]
        assert outputs[cut + 2] == b""

    def test_carried_over_opening_is_numbered_and_closed_as_the_first_attempt(self) -> None:
        """An opening sent after the retry is numbered and closed as part of the first attempt."""
        stream = ResponsesStream("r1")
        attempt_one = stream.start() + stream.content("abcdef")
        opening = stream.tool_call_opening("c1", "run")
        retry_stream = ResponsesStream("r2")
        attempt_two = retry_stream.start() + retry_stream.content("ok") + retry_stream.completed()
        output, _ = _splice_runs(_event_chunks(attempt_one) + [None, opening] + _event_chunks(attempt_two))
        payloads = parse_sse_payloads(output)
        added = [p for p in payloads if p["type"] == "response.output_item.added"]
        assert [(p["item"]["type"], p["output_index"]) for p in added] == [("message", 0), ("function_call", 1), ("message", 2)]
        opening_payload = [p for p in payloads if p["type"] == "response.output_item.added"][1]
        before = payloads[payloads.index(opening_payload) - 1]
        assert opening_payload["sequence_number"] == before["sequence_number"] + 1
        closed = [p for p in payloads if p["type"] == "response.output_item.done" and p["item"]["type"] == "function_call"]
        assert closed[0]["item"]["status"] == "incomplete"
        assert closed[0]["output_index"] == 1
        terminal = payloads[-1]["response"]
        assert [item["type"] for item in terminal["output"]] == ["message", "function_call", "message"]
        assert terminal["id"] == "r1"

    def test_later_error_body_is_relayed_verbatim(self) -> None:
        """An error body from a later attempt is passed on as it was."""
        stream = ResponsesStream("r1")
        attempt_one = _event_chunks(stream.start() + stream.content("abcdefg"))
        error_body = b'{"error": {"message": "bad request"}}'
        retry_stream = ResponsesStream("r3")
        attempt_three = _event_chunks(retry_stream.start() + retry_stream.content("ok") + retry_stream.completed())
        output, outputs = _splice_runs(attempt_one[:5] + [None, error_body, None] + attempt_three)
        error_out = outputs[5]
        assert error_out.endswith(error_body)
        assert [p["type"] for p in parse_sse_payloads(error_out[: -len(error_body)])] == [
            "response.output_text.done", "response.content_part.done", "response.output_item.done",
        ]
        assert outputs[6:8] == [b"", b""]
        assert outputs[8].startswith(b"\n\nevent: response.output_item.added")
        assert parse_sse_payloads(output)[-1]["response"]["id"] == "r1"

    def test_leftover_then_retry(self) -> None:
        """A cut off event is passed on before the closings of a retry."""
        stream = ResponsesStream("r1")
        attempt_one = _event_chunks(stream.start() + stream.content("abcdefg"))[:5]
        leftover = b"data: {\"type\": \"response.outp"
        retry_stream = ResponsesStream("r2")
        attempt_two = _event_chunks(retry_stream.start() + retry_stream.content("ok") + retry_stream.completed())
        _output, outputs = _splice_runs(attempt_one + [leftover, None] + attempt_two)
        assert outputs[5] == leftover
        assert outputs[6].startswith(b"\n\n")
        assert [p["type"] for p in parse_sse_payloads(outputs[6][2:])] == [
            "response.output_text.done", "response.content_part.done", "response.output_item.done",
        ]
        assert outputs[7] == b""
        assert outputs[8].startswith(b"event: response.output_item.added")

    def test_finish_emits_owed_closings(self) -> None:
        """Finishing after a retry with no new attempt sends the owed closings."""
        stream = ResponsesStream("r1")
        chunks = _event_chunks(stream.start() + stream.reasoning("abcdef"))[:5]
        output, outputs = _splice_runs(chunks + [None])
        assert [p["type"] for p in parse_sse_payloads(outputs[-1])] == [
            "response.reasoning_text.done", "response.reasoning_part.done", "response.output_item.done",
        ]
        assert output.startswith(b"".join(chunks))

    def test_cut_item_reusing_a_done_id_is_closed(self) -> None:
        """A cut item whose id an earlier done item of the attempt used is still closed."""
        stream = ResponsesStream("r1")
        chunks = _event_chunks(stream.start() + stream.content("AAA", "msg_x") + stream.content("BBBBBBBBBB", "msg_x"))
        cut = _cut_after_first_delta(chunks, 1)
        _, outputs = _splice_runs(chunks[:cut] + [None])
        first_delta = parse_sse_payloads(chunks[cut - 1])[0]["delta"]
        closings = parse_sse_payloads(outputs[-1])
        assert [p["type"] for p in closings] == [
            "response.output_text.done", "response.content_part.done", "response.output_item.done",
        ]
        assert closings[-1]["output_index"] == 1
        assert closings[-1]["item"]["content"] == [{"type": "output_text", "text": first_delta, "annotations": []}]

    def test_line_leftover_then_retry(self) -> None:
        """A last event cut after a whole line gets one more line end before the closings."""
        stream = ResponsesStream("r1")
        attempt_one = _event_chunks(stream.start() + stream.content("abcdefg"))[:5]
        leftover = b"data: {\"type\": \"response.outp\n"
        retry_stream = ResponsesStream("r2")
        attempt_two = _event_chunks(retry_stream.start() + retry_stream.content("ok") + retry_stream.completed())
        _output, outputs = _splice_runs(attempt_one + [leftover, None] + attempt_two)
        assert outputs[5] == leftover
        assert outputs[6].startswith(b"\nevent: response.output_text.done\n")
        assert [p["type"] for p in parse_sse_payloads(outputs[6][1:])] == [
            "response.output_text.done", "response.content_part.done", "response.output_item.done",
        ]

    def test_data_only_stream_stays_data_only(self) -> None:
        """A stream without event lines gets none added when cut and retried."""
        stream = ResponsesStream("r1")
        attempt_one = _data_only_chunks(stream.start() + stream.content("abcdefg"))[:5]
        retry_stream = ResponsesStream("r2")
        attempt_two = _data_only_chunks(retry_stream.start() + retry_stream.content("ok") + retry_stream.completed())
        output, _ = _splice_runs(attempt_one + [None] + attempt_two)
        assert [line for line in output.split(b"\n") if line and not line.startswith(b"data: ")] == []
        payloads = parse_sse_payloads(output)
        assert [p["sequence_number"] for p in payloads] == list(range(len(payloads)))
        assert payloads[5]["type"] == "response.output_text.done"

    def test_cut_part_is_closed_with_its_opening_text(self) -> None:
        """A part cut before any delta is closed with the text its opening carried."""
        opening = [
            responses_event({"type": "response.created", "response": {"id": "r1", "status": "in_progress"}}),
            responses_event({"type": "response.output_item.added", "item": {"id": "m1", "type": "message", "content": []}}),
            responses_event({
                "type": "response.content_part.added", "item_id": "m1", "content_index": 0,
                "part": {"type": "output_text", "text": "Hi"},
            }),
        ]
        output, _ = _splice_runs(opening + [None])
        payloads = parse_sse_payloads(output)
        assert [p["type"] for p in payloads[3:]] == ["response.content_part.done", "response.output_item.done"]
        assert payloads[3]["part"] == {"type": "output_text", "text": "Hi", "annotations": []}
        assert payloads[4]["item"]["content"] == [{"type": "output_text", "text": "Hi", "annotations": []}]

    def test_cut_part_is_closed_with_its_done_text(self) -> None:
        """A part whose text is done but not closed is closed with the done text."""
        opening = [
            responses_event({"type": "response.created", "response": {"id": "r1", "status": "in_progress"}}),
            responses_event({"type": "response.output_item.added", "item": {"id": "m1", "type": "message", "content": []}}),
            responses_event({
                "type": "response.content_part.added", "item_id": "m1", "content_index": 0,
                "part": {"type": "output_text", "text": ""},
            }),
            responses_event({"type": "response.output_text.delta", "item_id": "m1", "content_index": 0, "delta": "ab"}),
            responses_event({"type": "response.output_text.done", "item_id": "m1", "content_index": 0, "text": "abcd"}),
        ]
        output, _ = _splice_runs(opening + [None])
        payloads = parse_sse_payloads(output)
        assert [p["type"] for p in payloads[5:]] == ["response.content_part.done", "response.output_item.done"]
        assert payloads[5]["part"]["text"] == "abcd"
        assert payloads[6]["item"]["content"][0]["text"] == "abcd"

    def test_cut_call_is_closed_with_its_done_arguments(self) -> None:
        """A function call whose arguments are done but not closed keeps the done arguments."""
        stream = ResponsesStream("r1")
        opening = _event_chunks(stream.start() + stream.tool_call_opening("c1", "run")) + [
            responses_event({
                "type": "response.function_call_arguments.done", "item_id": "fc_c1", "name": "run",
                "arguments": "{\"x\": 1}",
            }),
        ]
        output, _ = _splice_runs(opening + [None])
        payloads = parse_sse_payloads(output)
        assert [p["type"] for p in payloads[4:]] == ["response.output_item.done"]
        assert payloads[4]["item"]["arguments"] == "{\"x\": 1}"
        assert payloads[4]["item"]["status"] == "incomplete"

    @pytest.mark.parametrize("name", _STREAM_NAMES)
    def test_splice_keeps_every_byte_but_the_dropped_openings(self, name: str) -> None:
        """Bytes before the retry come out as sent, later events keep their content, and only later openings go."""
        chunks = _event_chunks(_raw(name))
        cut = _cut_after_first_delta(chunks, 1)
        _, outputs = _splice_runs(chunks[:cut] + [None, chunks[cut]] + chunks)
        runs = chunks[:cut + 1] + chunks
        openings = ("response.created", "response.in_progress")
        assert outputs[:cut] == runs[:cut]
        later = list(zip(runs, outputs))[cut:]
        kept = [(run, out) for run, out in later if parse_sse_payloads(run)[0]["type"] not in openings]
        dropped = [out for run, out in later if parse_sse_payloads(run)[0]["type"] in openings]
        assert [_without_numbers(parse_sse_payloads(out)) for _, out in kept] == [_without_numbers(parse_sse_payloads(run)) for run, _ in kept]
        assert len(dropped) == sum(1 for c in chunks if parse_sse_payloads(c)[0]["type"] in openings)
        assert not {p["type"] for out in dropped for p in parse_sse_payloads(out)} & set(openings)

    @pytest.mark.parametrize("name", _STREAM_NAMES)
    def test_golden_splice(self, name: str) -> None:
        """A cut and retried stream matches its saved spliced copy."""
        output, _, _ = _cut_and_retry(name, 1)
        assert output == (_FIXTURES / f"spliced_{name}.sse").read_bytes()


# ── Non-streaming reply ──


class TestAssembleNonStreaming:
    """Stream events are put back together into one reply."""

    def test_returns_the_terminal_response(self) -> None:
        """The reply is the response of the final event."""
        payloads = parse_sse_payloads(_raw("vllm"))
        assert json.loads(_DIALECT.assemble_non_streaming(payloads)) == payloads[-1]["response"]

    def test_spliced_buffer_holds_earlier_items(self) -> None:
        """The reply of a retried stream holds the items of both attempts."""
        output, _, _ = _cut_and_retry("vllm", 1)
        reply = json.loads(_DIALECT.assemble_non_streaming(parse_sse_payloads(output)))
        assert [item["type"] for item in reply["output"]] == ["reasoning", "function_call", "reasoning", "function_call"]
        assert reply["output"][1]["status"] == "incomplete"

    def test_cut_stream_rebuilds_items(self) -> None:
        """A stream with no final event is rebuilt into a reply from its items."""
        chunks = _event_chunks(_raw("sglang"))
        cut = _cut_after_first_delta(chunks, 1)
        reply = json.loads(_DIALECT.assemble_non_streaming(parse_sse_payloads(b"".join(chunks[:cut]))))
        assert reply["id"] == "resp_sgl1"
        assert reply["status"] == "in_progress"
        assert [item["type"] for item in reply["output"]] == ["reasoning", "message"]
        assert reply["output"][1]["content"] == [{"type": "output_text", "text": "Checking", "annotations": []}]
        assert reply["output"][1]["status"] == "in_progress"

    def test_cut_retries_with_repeated_ids_keep_every_attempt(self) -> None:
        """Attempts that reuse an item id each keep their own item when retries run out."""
        first, second = ResponsesStream("r1"), ResponsesStream("r1")
        runs = [first.start() + first.content("AAA", "msg_x"), None, second.start() + second.content("BBB", "msg_x"), None]
        output, _ = _splice_runs(runs)
        reply = json.loads(_DIALECT.assemble_non_streaming(parse_sse_payloads(output)))
        assert [(item["id"], _item_text(item)) for item in reply["output"]] == [("msg_x", "AAA"), ("msg_x", "BBB")]

    def test_stream_cut_after_created(self) -> None:
        """A stream cut right after it opened gives its response with no output."""
        response = {"id": "r1", "object": "response", "status": "in_progress"}
        reply = _DIALECT.assemble_non_streaming([{"type": "response.created", "response": response}])
        assert json.loads(reply) == {**response, "output": []}

    def test_no_events(self) -> None:
        """No events give no reply."""
        assert _DIALECT.assemble_non_streaming([]) is None

    def test_events_without_responses_shape(self) -> None:
        """Events that are not Responses events give no reply."""
        assert _DIALECT.assemble_non_streaming([{"error": "x"}, 5]) is None


# ── Live gate, opt in ──


_LIVE_TIMEOUT_S = 120
_CONFIG_PATH = Path(__file__).parents[3] / "config.yaml"
_GATED_TARGET = pytest.param(None, marks=pytest.mark.skip(reason="needs --responses-gate=conformance"), id="gated")
_PROBE_TOOL = {
    "type": "function",
    "name": "get_weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}
_PROBE_HISTORY = [
    {"role": "user", "content": "What is the weather in Paris?"},
    {"type": "reasoning", "summary": [], "content": [{"type": "reasoning_text", "text": "Use the tool."}]},
    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Checking."}]},
    {"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": '{"city": "Paris"}'},
    {"type": "function_call_output", "call_id": "call_1", "output": "Sunny, 20C"},
    {"type": "message", "role": "system", "content": "Answer in one line."},
]
_PROBES = {
    "content": {"input": "Say hello in one short line.", "stream": True},
    "tool": {"input": "Use get_weather for Paris.", "tools": [_PROBE_TOOL], "stream": True},
    "history": {"input": _PROBE_HISTORY, "tools": [_PROBE_TOOL], "stream": True},
    "tool_named": {
        "input": "Use get_weather for Paris.", "tools": [_PROBE_TOOL],
        "tool_choice": {"type": "function", "name": "get_weather"}, "stream": True,
    },
    "length": {"input": "Count from one to fifty in words.", "max_output_tokens": 16, "stream": True},
}


def _live_targets(config: ProxyConfig) -> list[tuple[str, str, str]]:
    """List each Responses backend with its name, url and host header. GPU services go through the proxy."""
    targets: list[tuple[str, str, str]] = []
    for name, service in config.services.items():
        if "/v1/responses" not in service.gpu_routes:
            continue
        targets.append((name, f"http://127.0.0.1:{config.listen_port}/v1/responses", service.hostname))
        if service.backup:
            targets.append((f"{name}-backup", service.backup.rstrip("/") + "/v1/responses", ""))
    return targets


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Give the conformance test one case per probe and Responses backend when its phase runs."""
    if "target" not in metafunc.fixturenames:
        return
    metafunc.parametrize("probe", sorted(_PROBES))
    if metafunc.config.getoption("--responses-gate") != "conformance":
        metafunc.parametrize("target", [_GATED_TARGET])
        return
    targets = _live_targets(load_config(str(_CONFIG_PATH)))
    metafunc.parametrize("target", [pytest.param(target, id=target[0]) for target in targets])


@pytest.fixture(scope="module")
def live_config() -> ProxyConfig:
    """The deployment config the live tests read."""
    return load_config(str(_CONFIG_PATH))


def _post(url: str, host: str, body: dict) -> tuple[int, str, bytes]:
    """Post a JSON body and return the status, content type and reply."""
    req = urllib_request.Request(url, data=json.dumps(body).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    if host:
        req.add_header("Host", host)
    with urllib_request.urlopen(req, timeout=_LIVE_TIMEOUT_S) as reply:
        return reply.status, reply.headers.get("Content-Type", ""), reply.read()


def _routing(config: ProxyConfig, service: str) -> dict[str, int]:
    """The dashboard's routing counters of one service."""
    req = urllib_request.Request(f"http://127.0.0.1:{config.listen_port}/?format=json")
    req.add_header("Host", config.dashboard_hostname)
    with urllib_request.urlopen(req, timeout=_LIVE_TIMEOUT_S) as reply:
        return json.loads(reply.read())["services"][service]["routing"]


def _attribution(config: ProxyConfig, service: str, before: dict[str, int] | None) -> str:
    """Tell whether the target, a backup, or the proxy itself answered a probe."""
    if before is None:
        return "direct"
    after = _routing(config, service)
    if after["backup"] == before["backup"] and after["target"] > before["target"]:
        return "target"
    if after == before:
        return "not admitted"
    return "unattributed"


def _streamed_items(payloads: list[dict]) -> list[str]:
    """The text generated_texts gives for each streamed item, in first-seen order."""
    texts: dict[str, str] = {}
    current = ""
    for payload in payloads:
        if payload["type"] == "response.output_item.added":
            current = payload["item"].get("id", "")
            texts.setdefault(current, "")
        item_id = payload.get("item_id", current)
        for text in _DIALECT.generated_texts(payload):
            texts[item_id] = texts.get(item_id, "") + text
    return list(texts.values())


def _assert_texts_match_items(payloads: list[dict], output: list[dict]) -> None:
    """Each terminal item's text is what generated_texts read for its streamed item."""
    streamed = _streamed_items(payloads)
    assert len(streamed) == len(output)
    for text, item in zip(streamed, output):
        if item["type"] == "function_call":
            assert text.startswith(item["name"])
            assert json.loads(text[len(item["name"]):]) == json.loads(item["arguments"])
        elif item["type"] in ("message", "reasoning"):
            assert text == _item_text(item)


def _assert_splice_framing(raw: bytes, output: list[dict]) -> int:
    """Cut the stream inside each item, retry, and check the spliced framing."""
    chunks = _event_chunks(raw)
    cuts = 0
    for position in range(len(output)):
        index = _first_delta_index(chunks, position)
        if index is None:
            continue
        cut = index + 1
        spliced, _ = _splice_runs(chunks[:cut] + [None, chunks[cut]] + chunks)
        payloads = parse_sse_payloads(spliced)
        assert [p["type"] for p in payloads].count("response.created") == 1
        indexes = [p["output_index"] for p in payloads if p["type"] == "response.output_item.added" and "output_index" in p]
        assert indexes == list(range(len(indexes)))
        sequence = [p["sequence_number"] for p in payloads if "sequence_number" in p]
        assert sequence == sorted(set(sequence))
        assert len(payloads[-1]["response"]["output"]) == position + 1 + len(output)
        cuts += 1
    return cuts


@pytest.mark.live
def test_live_backend_conformance(
    live_config: ProxyConfig, record_property: Callable[[str, str], None], target: tuple[str, str, str], probe: str
) -> None:
    """A backend's Responses stream meets every criterion the dialect relies on."""
    name, url, host = target
    before = _routing(live_config, name) if host else None
    try:
        status, _content_type, raw = _post(url, host, _PROBES[probe])
    except TimeoutError:
        pytest.skip("no answer")
    attribution = _attribution(live_config, name, before)
    record_property("attribution", attribution)
    if attribution == "not admitted":
        pytest.skip("not admitted")
    assert status == 200
    payloads = parse_sse_payloads(raw)
    assert {p["type"] for p in payloads} <= _SPEC_KNOWN_TYPES
    terminals = [p for p in payloads if p["type"] in ("response.completed", "response.incomplete")]
    if not terminals:
        pytest.skip("failed generation")
    output = terminals[-1]["response"]["output"]
    deltas = _assert_reconstruction(raw)
    record_property("verified_content", str(bool(_decoded_text(deltas, "content"))))
    record_property("verified_reasoning", str(bool(_decoded_text(deltas, "reasoning"))))
    record_property("verified_tool_call", str(bool(_decoded_calls(deltas))))
    finishes = [d for d in deltas if d["type"] == "finish_reason"]
    assert len(finishes) == 1
    record_property("finish", finishes[0]["text"])
    record_property("status", terminals[-1]["response"].get("status", ""))
    usage = _DIALECT.usage(terminals[-1])
    assert usage.input > 0
    assert usage.output > 0
    _assert_texts_match_items(payloads, output)
    record_property("cut_points", str(_assert_splice_framing(raw, output)))
