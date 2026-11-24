"""Tests for OpenAIContextModifier plugin."""

import json
from pathlib import Path
from typing import Optional

import pytest

from app.config import ConfigError, load_config
from app.plugins.openai_chat_modifier import OpenAIChatRequestModifier
from app.plugins.openai_context_modifier import OpenAIContextModifier
from tests.app.conftest import (
    CHAT_COMPLETIONS,
    CHAT_WIRE,
    RESPONSES,
    RESPONSES_WIRE,
    Wire,
    responses_input,
    responses_request,
)

_Message = dict[str, str | list[dict]]
_ToolCall = dict[str, str | dict[str, str | int]]
_Config = dict

COMPLETIONS = "/v1/chat/completions"
PRUNED = "[tool result pruned]"

# The ratio every test uses unless it is the thing under test: the prune window
# grows by half a turn for each turn the conversation gains.
HALF_RATIO = 0.5

SYSTEM_PROMPT = "You are helpful"
HELPER_PROMPT = "You are a helper."
LONG_TOOL_RESULT = "x" * 200
SEARCH_FOO = '{"query":"foo"}'
LONG_ARGS = '{"input": "' + "x" * 200 + '"}'
TRUNCATED_ARGS = json.dumps({"input": "x" * 100 + "..."})
MD_EXCLUSION = [{"tool_name_regex": "(?i)read", "tool_args_regex": r"(?i)(skill|agents)\.md"}]
READ_PATHS = ["SKILL.md", "agents.md", "readme.txt"]


def _system(content: str = SYSTEM_PROMPT) -> _Message:
    """A system message.

    Args:
        content: The system prompt text.

    Returns:
        The system history message.
    """
    return {"role": "system", "content": content}


def _user(content: str) -> _Message:
    """A user message.

    Args:
        content: What the user asked.

    Returns:
        The user history message.
    """
    return {"role": "user", "content": content}


def _assistant(content: str, reasoning: str = "") -> _Message:
    """An assistant message with no tool calls.

    Args:
        content: The answer the model gave.
        reasoning: The thinking the model emitted, omitted when empty.

    Returns:
        The assistant history message.
    """
    message: _Message = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning_content"] = reasoning
    return message


def _tool_call(tc_id: str, name: str = "s", arguments: str | int = "") -> _ToolCall:
    """One entry of an assistant message's tool_calls list.

    Args:
        tc_id: The id the matching tool result will quote.
        name: The function the model asked for.
        arguments: The raw arguments value the client sent, omitted when empty.
            Typed loosely because a client may send something that is not a string.

    Returns:
        The tool call entry.
    """
    function: dict[str, str | int] = {"name": name}
    if arguments != "":
        function["arguments"] = arguments
    return {"id": tc_id, "type": "function", "function": function}


def _assistant_calling(*calls: _ToolCall, content: str = "", reasoning: str = "") -> _Message:
    """An assistant message that issued the given tool calls.

    Args:
        calls: The tool call entries, as built by _tool_call.
        content: The prose the model emitted alongside the calls.
        reasoning: The thinking the model emitted, omitted when empty.

    Returns:
        The assistant history message.
    """
    message = _assistant(content, reasoning)
    message["tool_calls"] = list(calls)
    return message


def _tool_result(tc_id: str, content: str) -> _Message:
    """A tool result answering a tool call.

    Args:
        tc_id: The tool call this result answers, omitted when empty so the
            plugin cannot match it to a call.
        content: The output the tool produced.

    Returns:
        The tool history message.
    """
    message: _Message = {"role": "tool", "content": content}
    if tc_id:
        message["tool_call_id"] = tc_id
    return message


def _long_result(turn: int) -> str:
    """The tool output for a turn, long enough that pruning replaces it.

    Args:
        turn: The turn the output belongs to, so each turn is distinguishable.

    Returns:
        The tool output text.
    """
    return f"this is a long tool result number {turn} that exceeds placeholder length"


def _tool_turns(
    count: int,
    name: str = "s",
    arguments: str = "",
    content: Optional[str] = None,
) -> list[_Message]:
    """Turns of a user question, an assistant tool call, and its result.

    Args:
        count: How many turns to build, numbered from one.
        name: The function every tool call asks for.
        arguments: The arguments every tool call carries.
        content: The output every tool returns, or None for per-turn long output.

    Returns:
        The message history, three messages per turn.
    """
    messages: list[_Message] = []
    for i in range(1, count + 1):
        messages.append(_user(f"Q{i}"))
        messages.append(_assistant_calling(_tool_call(f"t{i}", name, arguments)))
        messages.append(_tool_result(f"t{i}", _long_result(i) if content is None else content))
    return messages


def _read_tool_turns(count: int, paths: list[str]) -> list[_Message]:
    """Turns whose read tool call cycles through the given file paths.

    Args:
        count: How many turns to build, numbered from one.
        paths: The paths to cycle through, one per turn.

    Returns:
        The message history, three messages per turn.
    """
    messages: list[_Message] = []
    for i in range(1, count + 1):
        arguments = json.dumps({"path": paths[(i - 1) % len(paths)]})
        messages.append(_user(f"Q{i}"))
        messages.append(_assistant_calling(_tool_call(f"t{i}", "read", arguments)))
        messages.append(_tool_result(f"t{i}", _long_result(i)))
    return messages


def _ids_reading(count: int, paths: list[str], wanted: str) -> set[str]:
    """The tool call ids from _read_tool_turns whose read targeted one path.

    Args:
        count: The turn count the history was built with.
        paths: The paths the history cycled through.
        wanted: The path to select.

    Returns:
        The matching tool call ids.
    """
    return {f"t{i}" for i in range(1, count + 1) if paths[(i - 1) % len(paths)] == wanted}


def _alternating_tool_turns(count: int) -> list[_Message]:
    """Turns alternating a read_file call on odd turns with a search on even ones.

    Args:
        count: How many turns to build, numbered from one.

    Returns:
        The message history, three messages per turn.
    """
    messages: list[_Message] = []
    for i in range(1, count + 1):
        if i % 2 == 1:
            call = _tool_call(f"t{i}", "read_file", '{"path":"f.txt"}')
        else:
            call = _tool_call(f"t{i}", "search", '{"query":"q"}')
        messages.append(_user(f"Q{i}"))
        messages.append(_assistant_calling(call))
        messages.append(_tool_result(f"t{i}", _long_result(i)))
    return messages


def _chat_turns(count: int, question: str = "u", answer: str = "a") -> list[_Message]:
    """Turns of a plain user question and assistant answer, with no tools.

    Args:
        count: How many turns to build, numbered from one.
        question: Prefix for each user message's content.
        answer: Prefix for each assistant message's content.

    Returns:
        The message history, two messages per turn.
    """
    messages: list[_Message] = []
    for i in range(1, count + 1):
        messages.append(_user(f"{question}{i}"))
        messages.append(_assistant(f"{answer}{i}"))
    return messages


def _reasoning_conversation() -> list[_Message]:
    """A 4-turn conversation with reasoning on every assistant message.

    Returns:
        The message history.
    """
    messages: list[_Message] = [_system()]
    for i in range(1, 5):
        messages.append(_user(f"Question {i}"))
        messages.append(_assistant(f"Answer {i}", reasoning=f"Thinking {i}"))
    return messages


def _amortized_config(exclusions: Optional[list[_Config]] = None) -> _Config:
    """Config that prunes tool results older than a ten-turn window.

    Args:
        exclusions: Entries for tool_call_prune_exclusions, or None to omit the key.

    Returns:
        The plugin config.
    """
    config: _Config = {
        "preserve_tool_calls_min_window_size": 10,
        "preserve_tool_calls_max_window_size_ratio": HALF_RATIO,
    }
    if exclusions is not None:
        config["tool_call_prune_exclusions"] = exclusions
    return config


def _repeat_config(min_window: int = 5, ratio: float = HALF_RATIO) -> _Config:
    """Config with only the system prompt repeat enabled.

    Args:
        min_window: Turns from the end within which the repeat never triggers.
        ratio: How fast the window grows with the conversation.

    Returns:
        The plugin config.
    """
    return {
        "preserve_thinking_tokens_from_previous_turns": -1,
        "preserve_tool_calls_min_window_size": -1,
        "repeat_system_prompt_min_window": min_window,
        "repeat_system_prompt_max_window_ratio": ratio,
    }


def _request_body(messages: list[_Message]) -> bytes:
    """Encode a chat request carrying the given history.

    Args:
        messages: The conversation the client sends.

    Returns:
        The encoded request body.
    """
    return json.dumps({"messages": messages}).encode()


async def _rewritten_messages(
    plugin: OpenAIContextModifier, messages: list[_Message]
) -> list[_Message]:
    """Put one request through the plugin and return the history it forwards.

    Args:
        plugin: The plugin under test.
        messages: The conversation the client sends.

    Returns:
        The messages the plugin hands to the backend.
    """
    _, body, _ = await plugin.on_request(1, {}, _request_body(messages), COMPLETIONS)
    return json.loads(body)["messages"]


def _with_role(messages: list[dict], role: str) -> list[dict]:
    """Every message of one role, skipping entries a client sent as non-dicts.

    Args:
        messages: The forwarded history.
        role: The role to select.

    Returns:
        The matching messages, in order.
    """
    return [m for m in messages if isinstance(m, dict) and m.get("role") == role]


def _assistant_reasonings(messages: list[dict]) -> list[Optional[str]]:
    """The reasoning each assistant message still carries, None where it has none.

    The plugin strips reasoning by removing the key, so None distinguishes a
    stripped message from one still carrying an empty reasoning string.

    Args:
        messages: The forwarded history.

    Returns:
        One entry per assistant message, in order.
    """
    return [m.get("reasoning_content") for m in _with_role(messages, "assistant")]


def _tool_contents(messages: list[_Message]) -> list[str]:
    """The content of every tool result, in order.

    Args:
        messages: The forwarded history.

    Returns:
        One entry per tool message, in order.
    """
    return [m["content"] for m in _with_role(messages, "tool")]


def _pruned_ids(messages: list[_Message]) -> set[str]:
    """The ids of the tool results the plugin replaced with the placeholder.

    Args:
        messages: The forwarded history.

    Returns:
        The tool call ids whose result was pruned.
    """
    return {
        m["tool_call_id"] for m in _with_role(messages, "tool") if m["content"] == PRUNED
    }


def _tool_result_for(messages: list[_Message], tc_id: str) -> str:
    """The content of the tool result answering one tool call.

    Args:
        messages: The forwarded history.
        tc_id: The tool call id to look up.

    Returns:
        The tool result content, or "" when the history holds no such result.
    """
    matches = [m["content"] for m in messages if m.get("tool_call_id") == tc_id]
    return matches[0] if matches else ""


def _system_messages(messages: list[_Message]) -> list[_Message]:
    """Every system message in the forwarded history.

    Args:
        messages: The forwarded history.

    Returns:
        The system messages, in order.
    """
    return _with_role(messages, "system")


def _sole_tool_call(messages: list[_Message]) -> _ToolCall:
    """The first tool call of the first assistant message that issued one.

    Args:
        messages: The forwarded history.

    Returns:
        The tool call entry.
    """
    return [m for m in messages if m.get("tool_calls")][0]["tool_calls"][0]


def _capture_ui(plugin: OpenAIContextModifier) -> list[tuple[str, str]]:
    """Record every panel the plugin publishes.

    Args:
        plugin: The plugin to attach the recorder to.

    Returns:
        A list that grows with each (title, html) the plugin publishes.
    """
    published: list[tuple[str, str]] = []
    plugin.init_ui(lambda title, html: published.append((title, html)))
    return published


async def _tool_pruning_panels() -> list[tuple[str, str]]:
    """Panels published for a conversation whose only pruning is of tool results.

    Reasoning pruning is off and no message carries reasoning, so whatever the
    panel says about reasoning is what an untouched counter renders as.

    Returns:
        Every (title, html) the plugin published.
    """
    plugin = OpenAIContextModifier({
        "preserve_thinking_tokens_from_previous_turns": -1,
        "preserve_tool_calls_min_window_size": 1,
        "preserve_tool_calls_max_window_size_ratio": HALF_RATIO,
    })
    published = _capture_ui(plugin)
    await _rewritten_messages(
        plugin, [_system()] + _tool_turns(14) + [_user("Q15"), _assistant("A15")]
    )
    return published


async def _send_distinct_conversations(plugin: OpenAIContextModifier, count: int) -> None:
    """Send one-turn conversations, each with a conversation id of its own.

    Args:
        plugin: The plugin under test.
        count: How many conversations to send.
    """
    for i in range(1, count + 1):
        await _rewritten_messages(
            plugin,
            [_system(f"System {i}"), _user(f"User {i}"), _assistant(f"A{i}")],
        )


def _reasoning_tool_turns(count: int) -> list[_Message]:
    """Tool turns whose assistant messages also carry reasoning.

    Args:
        count: How many turns to build, numbered from one.

    Returns:
        The message history, three messages per turn.
    """
    messages: list[_Message] = []
    for i in range(1, count + 1):
        messages.append(_user(f"Q{i}"))
        messages.append(_assistant_calling(_tool_call(f"t{i}"), reasoning=f"R{i}"))
        messages.append(_tool_result(f"t{i}", _long_result(i)))
    return messages


def _duplicate_call_history() -> list[_Message]:
    """A turn issuing the same tool call twice, followed by a plain turn.

    Returns:
        The message history.
    """
    return [
        _system(HELPER_PROMPT),
        _user("Turn 1"),
        _assistant_calling(
            _tool_call("tc1", arguments='{"q":"foo"}'),
            _tool_call("tc2", arguments='{"q":"foo"}'),
            content="a",
        ),
        _tool_result("tc1", LONG_TOOL_RESULT),
        _tool_result("tc2", LONG_TOOL_RESULT),
        _user("Turn 2"),
        _assistant("b"),
    ]


def _exclusion_history(
    tool_name: str = "search", tool_args: str = SEARCH_FOO, num_turns: int = 22
) -> list[_Message]:
    """A history long enough for amortized pruning to reach its early turns.

    Args:
        tool_name: The function every tool call asks for.
        tool_args: The arguments every tool call carries.
        num_turns: How many tool turns to build.

    Returns:
        The message history.
    """
    return (
        [_system()]
        + _tool_turns(num_turns, name=tool_name, arguments=tool_args)
        + [_assistant("A")]
    )


_CONFIG_YAML = """
listen_port: 8080
total_vram_budget: 20480
dashboard_hostname: gpu-proxy.local
services:
  svc1:
    priority: 1
    vram_budget: 10240
    hostname: svc1.local
    target: http://127.0.0.1:8001
    container: svc1
    gpu_routes: ["/v1/completions"]
    backup: http://127.0.0.1:9001
plugins:
  - name: context_modifier
    type: OpenAIContextModifier
    config:
      preserve_tool_calls_min_window_size: 10
      preserve_tool_calls_max_window_size_ratio: 0.5
      tool_call_prune_exclusions:
        - tool_name_regex: '(?i)read'
          tool_args_regex: '(?i)(skill|agents)\\.md'
"""


def _prune_zone_history(arguments: str) -> list[_Message]:
    """A history whose one tool call sits far enough back to be pruned.

    Args:
        arguments: The arguments the tool call carries.

    Returns:
        The message history.
    """
    return [
        _user("start"),
        _assistant_calling(_tool_call("t1", "search", arguments)),
        _tool_result("t1", "old result"),
        _user("f1"),
        _user("f2"),
        _user("f3"),
        _user("end"),
    ]


def _recent_history(arguments: str) -> list[_Message]:
    """A history whose one tool call is recent enough to be left alone.

    Args:
        arguments: The arguments the tool call carries.

    Returns:
        The message history.
    """
    return [
        _user("start"),
        _assistant_calling(_tool_call("t1", "search", arguments)),
        _tool_result("t1", "result"),
        _user("end"),
    ]


class TestPreserveTurnsNegativeOne:
    """preserve_turns=-1 (default): no pruning."""

    async def test_no_pruning_with_default(self) -> None:
        plugin = OpenAIContextModifier({})
        messages = await _rewritten_messages(plugin, _reasoning_conversation())
        assert _assistant_reasonings(messages) == [
            "Thinking 1",
            "Thinking 2",
            "Thinking 3",
            "Thinking 4",
        ]


class TestPreserveTurnsZero:
    """preserve_turns=0: strip ALL reasoning."""

    async def test_all_reasoning_stripped(self) -> None:
        plugin = OpenAIContextModifier({"preserve_thinking_tokens_from_previous_turns": 0})
        messages = await _rewritten_messages(plugin, _reasoning_conversation())
        assert _assistant_reasonings(messages) == [None, None, None, None]


class TestPreserveTurnsOne:
    """preserve_turns=1: keep only current turn's reasoning."""

    async def test_only_last_assistant_reasoning_kept(self) -> None:
        plugin = OpenAIContextModifier({"preserve_thinking_tokens_from_previous_turns": 1})
        messages = await _rewritten_messages(plugin, _reasoning_conversation())
        assert _assistant_reasonings(messages) == [None, None, None, "Thinking 4"]


class TestPreserveTurnsTwo:
    """preserve_turns=2: keep current + 1 previous turn."""

    async def test_last_two_turns_kept(self) -> None:
        plugin = OpenAIContextModifier({"preserve_thinking_tokens_from_previous_turns": 2})
        messages = await _rewritten_messages(plugin, _reasoning_conversation())
        assert _assistant_reasonings(messages) == [None, None, "Thinking 3", "Thinking 4"]


class TestMixedRoles:
    """Non-user roles don't affect turn counting."""

    async def test_tool_and_function_roles_dont_count(self) -> None:
        plugin = OpenAIContextModifier({"preserve_thinking_tokens_from_previous_turns": 1})
        history = [
            _user("Q1"),
            _assistant("A1", reasoning="R1"),
            _tool_result("", "tool output"),
            _user("Q2"),
            _assistant("A2", reasoning="R2"),
        ]
        messages = await _rewritten_messages(plugin, history)
        assert _assistant_reasonings(messages) == [None, "R2"]


class TestAssistantWithToolCalls:
    """Assistant messages with tool_calls also get reasoning stripped."""

    async def test_assistant_with_tool_calls_reasoning_stripped(self) -> None:
        plugin = OpenAIContextModifier({"preserve_thinking_tokens_from_previous_turns": 1})
        history = [
            _user("Q1"),
            _assistant_calling(_tool_call("t1", "search"), reasoning="R1"),
            _user("Q2"),
            _assistant("A2", reasoning="R2"),
        ]
        messages = await _rewritten_messages(plugin, history)
        assert _assistant_reasonings(messages) == [None, "R2"]
        assert messages[1]["tool_calls"] == [_tool_call("t1", "search")]


class TestMessagesWithoutReasoning:
    """Messages without reasoning field are unaffected."""

    async def test_no_reasoning_field_unchanged(self) -> None:
        plugin = OpenAIContextModifier({"preserve_thinking_tokens_from_previous_turns": 0})
        history = [_user("Q1"), _assistant("A1")]
        messages = await _rewritten_messages(plugin, history)
        assert messages[1] == {"role": "assistant", "content": "A1"}


class TestEmptyMessages:
    """Empty messages array passes through."""

    async def test_empty_messages_array(self) -> None:
        plugin = OpenAIContextModifier({"preserve_thinking_tokens_from_previous_turns": 2})
        assert await _rewritten_messages(plugin, []) == []


class TestPathFiltering:
    """Non-completions paths pass through unchanged."""

    @pytest.mark.parametrize("path", [
        "/v1/embeddings",
        "/v1/models",
        "/health",
        "/admin/status",
    ])
    async def test_non_completions_path_passthrough(self, path: str) -> None:
        plugin = OpenAIContextModifier({"preserve_thinking_tokens_from_previous_turns": 0})
        body = _request_body(_reasoning_conversation())
        _, result, result_path = await plugin.on_request(1, {}, body, path)
        assert result == body
        assert result_path == path


class TestInvalidJsonPassthrough:
    """Invalid JSON body passes through unchanged."""

    async def test_invalid_json_passthrough(self) -> None:
        plugin = OpenAIContextModifier({"preserve_thinking_tokens_from_previous_turns": 0})
        body = b"not valid json"
        _, result, _ = await plugin.on_request(1, {}, body, COMPLETIONS)
        assert result == body


class TestMissingMessagesKey:
    """Missing messages key passes through unchanged."""

    async def test_no_messages_key_passthrough(self) -> None:
        plugin = OpenAIContextModifier({"preserve_thinking_tokens_from_previous_turns": 0})
        body = b'{"model": "gpt-4"}'
        _, result, _ = await plugin.on_request(1, {}, body, COMPLETIONS)
        assert result == body


class TestNonArrayMessages:
    """Non-array messages passes through unchanged."""

    async def test_string_messages_passthrough(self) -> None:
        plugin = OpenAIContextModifier({"preserve_thinking_tokens_from_previous_turns": 0})
        body = b'{"messages": "not an array"}'
        _, result, _ = await plugin.on_request(1, {}, body, COMPLETIONS)
        assert result == body


class TestContentLengthUpdate:
    """Content-Length header updated after modification."""

    async def test_content_length_updated(self) -> None:
        plugin = OpenAIContextModifier({"preserve_thinking_tokens_from_previous_turns": 0})
        history = [
            _user("Q1"),
            _assistant("A1", reasoning="Very long thinking process"),
        ]
        body = _request_body(history)
        headers = {"Content-Length": str(len(body))}
        _, result, _ = await plugin.on_request(1, headers, body, COMPLETIONS)
        assert headers["Content-Length"] == str(len(result))
        assert len(result) < len(body)


class TestPluginName:
    """name() returns configured name or default."""

    @pytest.mark.parametrize("config,expected", [
        ({"name": "my_modifier"}, "my_modifier"),
        ({}, "openai_context_modifier"),
    ])
    async def test_name_returns_configured_or_default(self, config: _Config, expected: str) -> None:
        plugin = OpenAIContextModifier(config)
        assert plugin.name() == expected


class TestMissingRoleOnMessage:
    """Messages without role key are skipped without crashing."""

    async def test_message_without_role_skipped(self) -> None:
        plugin = OpenAIContextModifier({"preserve_thinking_tokens_from_previous_turns": 0})
        history = [
            _user("Q1"),
            {"content": "orphan message"},
            _assistant("A1", reasoning="R1"),
        ]
        messages = await _rewritten_messages(plugin, history)
        assert _assistant_reasonings(messages) == [None]
        assert messages[1] == {"content": "orphan message"}


class TestNonDictMessageEntry:
    """Non-dict entries in messages array are skipped."""

    async def test_string_entry_in_messages_skipped(self) -> None:
        plugin = OpenAIContextModifier({"preserve_thinking_tokens_from_previous_turns": 0})
        history = [_user("Q1"), "invalid entry", _assistant("A1", reasoning="R1")]
        messages = await _rewritten_messages(plugin, history)
        assert _assistant_reasonings(messages) == [None]
        assert messages[1] == "invalid entry"


class TestToolCallsWithReasoningDisabled:
    """Tool call pruning works even when reasoning pruning is off."""

    async def test_tool_calls_pruned_when_reasoning_disabled(self) -> None:
        plugin = OpenAIContextModifier({
            "preserve_thinking_tokens_from_previous_turns": -1,
            "preserve_tool_calls_min_window_size": 1,
            "preserve_tool_calls_max_window_size_ratio": HALF_RATIO,
        })
        history = [
            _user("Q1"),
            _assistant_calling(_tool_call("t1", "search")),
            _tool_result("t1", "large search results here"),
            _user("Q2"),
            _assistant("A2"),
        ]
        messages = await _rewritten_messages(plugin, history)
        assert messages[1]["tool_calls"] == [_tool_call("t1", "search")]
        assert messages[2]["tool_call_id"] == "t1"
        assert messages[-1]["content"] == "A2"


class TestIntegrationWithChatModifier:
    """ContextModifier and ChatModifier coexist without interference."""

    async def test_context_modifier_and_chat_modifier_coexist_in_pipeline(self) -> None:
        chat_modifier = OpenAIChatRequestModifier({
            "overrides": {"model": "gpt-4", "temperature": 0.7}
        })
        context_modifier = OpenAIContextModifier({
            "preserve_thinking_tokens_from_previous_turns": 1
        })

        history = [
            _user("Q1"),
            _assistant("A1", reasoning="R1"),
            _user("Q2"),
            _assistant("A2", reasoning="R2"),
        ]
        body = json.dumps({"model": "old-model", "messages": history}).encode()
        headers = {"Content-Type": "application/json"}

        headers, result_body, result_path = await chat_modifier.on_request(
            1, headers, body, COMPLETIONS
        )
        _, final_body, final_path = await context_modifier.on_request(
            1, headers, result_body, result_path
        )

        parsed = json.loads(final_body)
        assert parsed["model"] == "gpt-4"
        assert parsed["temperature"] == 0.7
        assert _assistant_reasonings(parsed["messages"]) == [None, "R2"]
        assert final_path == COMPLETIONS


class TestAmortizedToolCallsBasic:
    """Amortized pruning triggers when unpruned exceeds max_window."""

    async def test_prune_triggers_at_correct_point(self) -> None:
        plugin = OpenAIContextModifier(_amortized_config())
        history = _tool_turns(22) + [_user("Q23"), _assistant("A23")]
        messages = await _rewritten_messages(plugin, history)
        kept = [_long_result(i) for i in range(14, 23)]
        assert _tool_contents(messages) == [PRUNED] * 13 + kept

    async def test_does_not_prune_when_unpruned_equals_max_window(self) -> None:
        """Keep full tool results while unpruned turns equal max_window, then prune once."""
        plugin = OpenAIContextModifier({
            "preserve_tool_calls_min_window_size": 5,
            "preserve_tool_calls_max_window_size_ratio": HALF_RATIO,
        })

        at_nine = await _rewritten_messages(plugin, [_system()] + _tool_turns(9))
        assert _tool_contents(at_nine) == [_long_result(i) for i in range(1, 10)]

        at_ten = await _rewritten_messages(plugin, [_system()] + _tool_turns(10))
        assert _tool_contents(at_ten) == [_long_result(i) for i in range(1, 11)]

        at_eleven = await _rewritten_messages(plugin, [_system()] + _tool_turns(11))
        assert _tool_contents(at_eleven) == [PRUNED] * 6 + [_long_result(i) for i in range(7, 12)]

        at_twelve = await _rewritten_messages(plugin, [_system()] + _tool_turns(12))
        assert _tool_contents(at_twelve) == [PRUNED] * 6 + [_long_result(i) for i in range(7, 13)]


class TestAmortizedToolCallsDisabled:
    """Amortized pruning disabled when min_window=-1."""

    async def test_no_pruning_when_disabled(self) -> None:
        plugin = OpenAIContextModifier({"preserve_tool_calls_min_window_size": -1})
        history = _tool_turns(1) + [_user("Q2"), _assistant("A2")]
        messages = await _rewritten_messages(plugin, history)
        assert _tool_contents(messages) == [_long_result(1)]


class TestAmortizedToolCallsUnderMinWindow:
    """No pruning when total turns <= min_window."""

    async def test_no_pruning_under_min_window(self) -> None:
        plugin = OpenAIContextModifier(_amortized_config())
        history = _tool_turns(7) + [_assistant("A7")]
        messages = await _rewritten_messages(plugin, history)
        assert _tool_contents(messages) == [_long_result(i) for i in range(1, 8)]


class TestAmortizedToolCallsShortContent:
    """Short tool results are not pruned even when in prune zone."""

    async def test_short_content_kept_in_amortized(self) -> None:
        plugin = OpenAIContextModifier(_amortized_config())
        history = _tool_turns(20, content="ok") + [_user("Q21"), _assistant("A21")]
        messages = await _rewritten_messages(plugin, history)
        assert _tool_contents(messages) == ["ok"] * 20


class TestMultiRequestStatefulPruning:
    """State persists across requests on the same plugin instance."""

    async def test_stable_prefix_repruned_on_subsequent_requests(self) -> None:
        """Client always sends full unpruned history. Plugin must re-prune the stable prefix."""
        plugin = OpenAIContextModifier(_amortized_config())
        base = [_system()] + _tool_turns(21)

        first = base + [_user("Q22"), _assistant("A22")]
        assert (await _rewritten_messages(plugin, first))[3]["content"] == PRUNED

        second = first + [_user("Q23"), _assistant("A23")]
        assert (await _rewritten_messages(plugin, second))[3]["content"] == PRUNED


class TestLRUEviction:
    """LRUOrderedDict evicts oldest entry when exceeding 20 entries."""

    async def test_oldest_evicted_after_21_conversations(self) -> None:
        """After 21 distinct conversations, the oldest is evicted.

        Verified by observing that a follow-up request for the evicted
        conversation behaves as fresh state: previously-remembered pruning
        state is lost, causing more aggressive pruning on old turns.
        """
        plugin = OpenAIContextModifier({
            "preserve_tool_calls_min_window_size": 1,
            "preserve_tool_calls_max_window_size_ratio": HALF_RATIO,
        })

        # Establishes pruning state for the conversation that must be evicted.
        await _rewritten_messages(plugin, [_system("System 0")] + _tool_turns(22))

        # Twenty more distinct conversations fill the cache, evicting the first.
        await _send_distinct_conversations(plugin, 20)

        # With state remembered, turns 23-24 sit inside the window and are kept
        # (pruned_up_to_turn=21, unpruned_turns=3, max_window=13, threshold=3).
        # With state lost, the threshold falls back to min_window=1 and t23 goes.
        messages = await _rewritten_messages(plugin, [_system("System 0")] + _tool_turns(24))
        assert _tool_result_for(messages, "t23") == PRUNED


class TestStatefulEdgeCases:
    """Stateful pruning handles edge cases gracefully."""

    async def test_no_system_message_still_prunes(self) -> None:
        plugin = OpenAIContextModifier(_amortized_config())
        history = _tool_turns(21) + [_user("Q22"), _assistant("A22")]
        messages = await _rewritten_messages(plugin, history)
        assert messages[2]["content"] == PRUNED

    async def test_no_first_user_message_skips_stateful_pruning(self) -> None:
        plugin = OpenAIContextModifier({
            "preserve_tool_calls_min_window_size": 1,
            "preserve_tool_calls_max_window_size_ratio": HALF_RATIO,
        })
        history = [_system(), _assistant("Hi")]
        assert await _rewritten_messages(plugin, history) == history


async def test_init_with_system_prompt_config() -> None:
    """A configured window and ratio decide when the prompt is repeated.

    Over 14 turns the configured ratio of 0.3 puts the head of the conversation
    outside the window, where the default ratio of 0.5 would still hold it in.
    """
    plugin = OpenAIContextModifier(_repeat_config(min_window=7, ratio=0.3))
    messages = await _rewritten_messages(plugin, [_system()] + _chat_turns(14))
    assert len(_system_messages(messages)) == 2


async def test_init_with_default_system_prompt_config() -> None:
    """Left unconfigured, the repeat never fires, and its ratio defaults to 0.5."""
    # 30 turns leaves behind the window a small default window would give
    # (anything under 15), so a repeat here would mean the feature is on by default.
    default_plugin = OpenAIContextModifier({})
    long_history = [_system()] + _chat_turns(30)
    assert _system_messages(await _rewritten_messages(default_plugin, long_history)) == [_system()]

    # Only the window is configured, so the ratio falls back to 0.5, which over
    # 14 turns keeps the prompt inside the window and leaves the history alone.
    default_ratio_plugin = OpenAIContextModifier({
        "repeat_system_prompt_min_window": 7,
    })
    history = [_system()] + _chat_turns(14)
    assert _system_messages(await _rewritten_messages(default_ratio_plugin, history)) == [_system()]


async def test_on_request_repeats_system_prompt() -> None:
    plugin = OpenAIContextModifier(_repeat_config())
    history = [_system("you are helpful")] + _chat_turns(20) + [_user("u21")]
    messages = await _rewritten_messages(plugin, history)

    assert messages[0]["content"] == "you are helpful"
    assert len(messages) == 43
    assert messages[41] == _system("you are helpful")
    assert messages[42] == _user("u21")


class TestSystemPromptRepeatNoSystemMessage:
    """No system message: no crash, no system prompt inserted."""

    async def test_no_system_message_unchanged(self) -> None:
        plugin = OpenAIContextModifier(_repeat_config())
        assert await _rewritten_messages(plugin, [_user("hello")]) == [_user("hello")]


class TestSystemPromptRepeatWithinMinWindow:
    """System prompt stays at head when within min_window."""

    async def test_system_prompt_stays_at_head(self) -> None:
        plugin = OpenAIContextModifier(_repeat_config())
        history = [_system("sys"), _user("u1"), _assistant("a1")]
        assert await _rewritten_messages(plugin, history) == history


class TestSystemPromptRepeatCopyNearLastUser:
    """System prompt is copied near last user message while original is kept."""

    async def test_system_prompt_copied_to_last_user(self) -> None:
        plugin = OpenAIContextModifier(_repeat_config())
        history = [_system()] + _chat_turns(20) + [_user("u21")]
        messages = await _rewritten_messages(plugin, history)

        assert messages[0]["content"] == SYSTEM_PROMPT
        assert len(messages) == 43
        assert messages[41] == _system()
        assert messages[42] == _user("u21")


class TestSystemPromptRepeatMultiRequestCopy:
    """System prompt copy advances when conversation grows."""

    async def test_system_prompt_copy_advances_on_longer_conversation(self) -> None:
        plugin = OpenAIContextModifier(_repeat_config())

        short = [_system()] + _chat_turns(20) + [_user("u21")]
        first = await _rewritten_messages(plugin, short)
        assert first[0]["content"] == SYSTEM_PROMPT
        assert len(first) == 43
        assert first[41] == _system()
        assert first[42] == _user("u21")

        long_history = [_system()] + _chat_turns(60) + [_user("u61")]
        second = await _rewritten_messages(plugin, long_history)
        assert second[0]["content"] == SYSTEM_PROMPT
        assert len(second) == 123
        assert second[121] == _system()
        assert second[122] == _user("u61")


class TestSystemPromptRepeatStaleStateReset:
    """Stale system_prompt_index from longer conversation resets on shorter request."""

    async def test_stale_state_does_not_crash(self) -> None:
        plugin = OpenAIContextModifier(_repeat_config())
        long_history = [_system("sys")] + _chat_turns(20) + [_user("u21")]
        await _rewritten_messages(plugin, long_history)

        short_history = [_system("sys"), _user("u1")]
        assert await _rewritten_messages(plugin, short_history) == short_history


class TestSystemPromptRepeatIdempotent:
    """Same messages produce same output across requests (VLLM prefix cache stability)."""

    async def test_same_messages_produce_same_result(self) -> None:
        plugin = OpenAIContextModifier(_repeat_config())
        history = [_system("you are helpful")] + _chat_turns(20) + [_user("u21")]

        first = await _rewritten_messages(plugin, history)
        second = await _rewritten_messages(plugin, history)

        assert first == second
        assert second[0]["content"] == "you are helpful"


class TestCountersViaUI:
    """Counter values are observable through the published UI panel."""

    async def test_reasoning_counter_shown_in_ui(self) -> None:
        plugin = OpenAIContextModifier({"preserve_thinking_tokens_from_previous_turns": 0})
        published = _capture_ui(plugin)

        history = [
            _user("Q1"),
            _assistant("A1", reasoning="thinking about the answer"),
        ]
        await _rewritten_messages(plugin, history)

        assert len(published) == 1
        assert "reasoning toks pruned" in published[0][1]

    async def test_reasoning_counter_accumulates_in_ui(self) -> None:
        plugin = OpenAIContextModifier({"preserve_thinking_tokens_from_previous_turns": 0})
        published = _capture_ui(plugin)

        history = [
            _user("Q1"),
            _assistant("A1", reasoning="first reasoning"),
            _user("Q2"),
            _assistant("A2", reasoning="second reasoning"),
        ]
        await _rewritten_messages(plugin, history)

        assert len(published) == 1
        assert "reasoning toks pruned" in published[0][1]

    async def test_tool_counter_shown_in_ui(self) -> None:
        published = await _tool_pruning_panels()
        assert len(published) == 1
        assert "tool toks pruned" in published[0][1]


class TestPublishUI:
    """The published panel carries the plugin's own title and rendered chips."""

    async def test_ui_publish_receives_html(self) -> None:
        plugin = OpenAIContextModifier({"preserve_thinking_tokens_from_previous_turns": 0})
        published = _capture_ui(plugin)

        history = [
            _user("Test question"),
            _assistant("A1", reasoning="thinking about this answer"),
        ]
        await _rewritten_messages(plugin, history)

        assert len(published) == 1
        title, html = published[0]
        assert title == "Context Modifier"
        assert "reasoning toks pruned" in html
        assert "style=" in html

    async def test_ui_has_colored_chips(self) -> None:
        plugin = OpenAIContextModifier({
            "preserve_thinking_tokens_from_previous_turns": 0,
            "preserve_tool_calls_min_window_size": 1,
            "preserve_tool_calls_max_window_size_ratio": HALF_RATIO,
        })
        published = _capture_ui(plugin)

        history = [_system()] + _reasoning_tool_turns(14) + [_user("Q15"), _assistant("A15")]
        await _rewritten_messages(plugin, history)

        assert len(published) == 1
        html = published[0][1]
        assert "reasoning toks pruned" in html
        assert "tool toks pruned" in html


class TestZeroMetricsHidden:
    """Zero metrics are hidden from UI output."""

    async def test_zero_reasoning_not_shown(self) -> None:
        published = await _tool_pruning_panels()
        assert len(published) == 1
        html = published[0][1]
        assert "reasoning toks pruned" not in html
        assert "tool toks pruned" in html


class TestAllZeroConversationsProduceNoUI:
    """All-zero conversations produce no UI output."""

    async def test_no_pruning_produces_no_ui(self) -> None:
        plugin = OpenAIContextModifier({"preserve_thinking_tokens_from_previous_turns": 2})
        published = _capture_ui(plugin)

        history = [_user("Q1"), _assistant("A1", reasoning="R1")]
        await _rewritten_messages(plugin, history)

        assert published == []

    async def test_no_ui_publish_when_none(self) -> None:
        plugin = OpenAIContextModifier({"preserve_thinking_tokens_from_previous_turns": 0})
        history = [_user("Q1"), _assistant("A1", reasoning="R1")]
        messages = await _rewritten_messages(plugin, history)
        assert _assistant_reasonings(messages) == [None]


class TestRepeatSystemPromptPreservesOriginal:
    """Verify repeat_system_prompt preserves all original messages."""

    async def test_all_original_messages_preserved_on_repeat(self) -> None:
        """When system prompt is repeated, original conversation messages must be preserved."""
        history = [_system()] + _chat_turns(21, question="Q", answer="A")
        original_contents = [m["content"] for m in history[1:]]

        plugin = OpenAIContextModifier(_repeat_config())

        messages = await _rewritten_messages(plugin, history)
        assert len(messages) == 44
        assert messages[0]["content"] == SYSTEM_PROMPT
        assert messages[41] == _system()
        assert messages[42] == _user("Q21")
        non_system = [m["content"] for m in messages if m.get("role") != "system"]
        assert non_system == original_contents

        # The same history a second time must land in the same place.
        repeated = await _rewritten_messages(plugin, history)
        assert len(repeated) == 44
        assert repeated[0]["content"] == SYSTEM_PROMPT
        assert repeated[41] == _system()
        assert repeated[42] == _user("Q21")


class TestWhitespaceSystemPrompt:
    """Whitespace-only system content is left untouched."""

    async def test_whitespace_system_prompt_unchanged(self) -> None:
        """A system prompt that is only whitespace should not be repeated or modified."""
        plugin = OpenAIContextModifier(_repeat_config())
        blank = "   \t  \n  "
        history = [_system(blank)] + _chat_turns(20) + [_user("u21")]
        messages = await _rewritten_messages(plugin, history)

        assert messages[0]["content"] == blank
        assert _system_messages(messages) == [_system(blank)]


class TestPrunedUpToTurnStaleClamping:
    """When conversation shrinks, pruned_up_to_turn is clamped to prevent over-pruning."""

    async def test_no_over_pruning_after_conversation_shrinks(self) -> None:
        """Send a long conversation that triggers pruning, then a short one on the same conv_id.
        Verify the short conversation is not over-pruned due to stale state."""
        plugin = OpenAIContextModifier({
            "preserve_thinking_tokens_from_previous_turns": -1,
            "preserve_tool_calls_min_window_size": 5,
            "preserve_tool_calls_max_window_size_ratio": HALF_RATIO,
        })

        opening = [_system("sys"), _user("first user")]
        long_history = opening + _tool_turns(22)
        assert _pruned_ids(await _rewritten_messages(plugin, long_history))

        # The same conversation id, now only one turn long. min_window=5 exceeds
        # the turn count, so nothing may be pruned despite the remembered state.
        short_history = opening + _tool_turns(1)
        assert await _rewritten_messages(plugin, short_history) == short_history

        # Long enough to prune, but far shorter than the conversation the state
        # was remembered from. Unless that state is clamped down to what this
        # conversation can support, the window computes negative and every tool
        # result is pruned instead of the three oldest.
        shrunk_history = opening + _tool_turns(8)
        kept = [_long_result(i) for i in range(4, 9)]
        assert _tool_contents(await _rewritten_messages(plugin, shrunk_history)) == [PRUNED] * 3 + kept


class TestOpenAIContextModifier:
    """Per-turn tool call deduplication."""

    async def test_dedup_keeps_last_identical_in_turn(self) -> None:
        plugin = OpenAIContextModifier({"dedup_tool_calls_per_turn": True})
        history = [
            _system(HELPER_PROMPT),
            _user("Search for foo three times"),
            _assistant_calling(
                _tool_call("tc1", "search", SEARCH_FOO),
                _tool_call("tc2", "search", SEARCH_FOO),
                _tool_call("tc3", "search", SEARCH_FOO),
                content="I'll search multiple times.",
            ),
            _tool_result("tc1", LONG_TOOL_RESULT),
            _tool_result("tc2", LONG_TOOL_RESULT),
            _tool_result("tc3", "short"),
        ]
        messages = await _rewritten_messages(plugin, history)
        assert _tool_contents(messages) == [PRUNED, PRUNED, "short"]

    async def test_dedup_different_args_not_pruned(self) -> None:
        plugin = OpenAIContextModifier({"dedup_tool_calls_per_turn": True})
        history = [
            _system(HELPER_PROMPT),
            _user("Search for different things"),
            _assistant_calling(
                _tool_call("tc1", "search", SEARCH_FOO),
                _tool_call("tc2", "search", '{"query":"bar"}'),
                content="Searching...",
            ),
            _tool_result("tc1", LONG_TOOL_RESULT),
            _tool_result("tc2", LONG_TOOL_RESULT),
        ]
        messages = await _rewritten_messages(plugin, history)
        assert _tool_contents(messages) == [LONG_TOOL_RESULT, LONG_TOOL_RESULT]

    async def test_dedup_different_functions_not_pruned(self) -> None:
        plugin = OpenAIContextModifier({"dedup_tool_calls_per_turn": True})
        history = [
            _system(HELPER_PROMPT),
            _user("Do two things"),
            _assistant_calling(
                _tool_call("tc1", "search", '{"q":"x"}'),
                _tool_call("tc2", "fetch", '{"q":"x"}'),
                content="Running both.",
            ),
            _tool_result("tc1", LONG_TOOL_RESULT),
            _tool_result("tc2", LONG_TOOL_RESULT),
        ]
        messages = await _rewritten_messages(plugin, history)
        assert _tool_contents(messages) == [LONG_TOOL_RESULT, LONG_TOOL_RESULT]

    async def test_dedup_across_multiple_turns(self) -> None:
        plugin = OpenAIContextModifier({"dedup_tool_calls_per_turn": True})
        history = [
            _system(HELPER_PROMPT),
            _user("Search in turn 1"),
            _assistant_calling(
                _tool_call("tc1", "search", SEARCH_FOO), content="Searching..."
            ),
            _tool_result("tc1", LONG_TOOL_RESULT),
            _user("Search again in turn 2"),
            _assistant_calling(
                _tool_call("tc2", "search", SEARCH_FOO), content="Searching again..."
            ),
            _tool_result("tc2", LONG_TOOL_RESULT),
        ]
        messages = await _rewritten_messages(plugin, history)
        assert _tool_contents(messages) == [LONG_TOOL_RESULT, LONG_TOOL_RESULT]

    async def test_dedup_disabled_noop(self) -> None:
        plugin = OpenAIContextModifier({})
        history = [
            _system(HELPER_PROMPT),
            _user("Search for foo"),
            _assistant_calling(
                _tool_call("tc1", "search", SEARCH_FOO),
                _tool_call("tc2", "search", SEARCH_FOO),
                content="Searching...",
            ),
            _tool_result("tc1", LONG_TOOL_RESULT),
            _tool_result("tc2", LONG_TOOL_RESULT),
        ]
        messages = await _rewritten_messages(plugin, history)
        assert _tool_contents(messages) == [LONG_TOOL_RESULT, LONG_TOOL_RESULT]

    async def test_dedup_shorter_than_placeholder_kept(self) -> None:
        plugin = OpenAIContextModifier({"dedup_tool_calls_per_turn": True})
        history = [
            _system(HELPER_PROMPT),
            _user("Search"),
            _assistant_calling(
                _tool_call("tc1", "search", SEARCH_FOO),
                _tool_call("tc2", "search", SEARCH_FOO),
                content="Searching...",
            ),
            _tool_result("tc1", "short"),
            _tool_result("tc2", "short"),
        ]
        messages = await _rewritten_messages(plugin, history)
        assert _tool_contents(messages) == ["short", "short"]

    async def test_dedup_missing_tool_call_id_kept(self) -> None:
        plugin = OpenAIContextModifier({"dedup_tool_calls_per_turn": True})
        history = [
            _system(HELPER_PROMPT),
            _user("Search"),
            _assistant_calling(
                _tool_call("tc1", "search", SEARCH_FOO), content="Searching..."
            ),
            _tool_result("", LONG_TOOL_RESULT),
        ]
        messages = await _rewritten_messages(plugin, history)
        assert _tool_contents(messages) == [LONG_TOOL_RESULT]

    async def test_dedup_orphan_tool_message_kept(self) -> None:
        plugin = OpenAIContextModifier({"dedup_tool_calls_per_turn": True})
        history = [
            _system(HELPER_PROMPT),
            _user("Search"),
            _assistant_calling(content="Searching..."),
            _tool_result("nonexistent", LONG_TOOL_RESULT),
        ]
        messages = await _rewritten_messages(plugin, history)
        assert _tool_contents(messages) == [LONG_TOOL_RESULT]

    async def test_dedup_no_tool_calls_noop(self) -> None:
        plugin = OpenAIContextModifier({"dedup_tool_calls_per_turn": True})
        history = [_system(HELPER_PROMPT), _user("Hello"), _assistant("Hi there!")]
        assert await _rewritten_messages(plugin, history) == history

    async def test_dedup_invalid_args_kept(self) -> None:
        plugin = OpenAIContextModifier({"dedup_tool_calls_per_turn": True})
        history = [
            _system(HELPER_PROMPT),
            _user("Search"),
            _assistant_calling(
                _tool_call("tc1", "search", arguments=123), content="Searching..."
            ),
            _tool_result("tc1", LONG_TOOL_RESULT),
        ]
        messages = await _rewritten_messages(plugin, history)
        assert _tool_contents(messages) == [LONG_TOOL_RESULT]

    async def test_dedup_combined_with_amortized(self) -> None:
        plugin = OpenAIContextModifier({
            "dedup_tool_calls_per_turn": True,
            "preserve_tool_calls_min_window_size": 1,
            "preserve_tool_calls_max_window_size_ratio": HALF_RATIO,
        })
        history = [
            _system(HELPER_PROMPT),
            _user("Turn 1"),
            _assistant_calling(
                _tool_call("tc1", arguments='{"q":"foo"}'),
                _tool_call("tc2", arguments='{"q":"foo"}'),
                content="a",
            ),
            _tool_result("tc1", LONG_TOOL_RESULT),
            _tool_result("tc2", LONG_TOOL_RESULT),
            _user("Turn 2"),
            _assistant("mid"),
            _user("Turn 3"),
            _assistant_calling(_tool_call("tc3", arguments='{"q":"foo"}'), content="b"),
            _tool_result("tc3", LONG_TOOL_RESULT),
        ]
        messages = await _rewritten_messages(plugin, history)
        assert _tool_contents(messages) == [PRUNED, PRUNED, LONG_TOOL_RESULT]

    async def test_dedup_ui_token_count_correct(self) -> None:
        plugin = OpenAIContextModifier({
            "dedup_tool_calls_per_turn": True,
            "preserve_tool_calls_min_window_size": 1,
            "preserve_tool_calls_max_window_size_ratio": HALF_RATIO,
        })
        published = _capture_ui(plugin)
        await _rewritten_messages(plugin, _duplicate_call_history())

        assert published[-1][0] == "Context Modifier"
        assert "tool toks pruned" in published[-1][1]

    async def test_dedup_token_count_does_not_accumulate(self) -> None:
        plugin = OpenAIContextModifier({"dedup_tool_calls_per_turn": True})
        published = _capture_ui(plugin)
        await _rewritten_messages(plugin, _duplicate_call_history())
        assert "19 tool toks pruned" in published[-1][1]

        # The same conversation id with nothing left to dedup must not add to
        # the count already shown, nor reset it.
        no_duplicates = [
            _system(HELPER_PROMPT),
            _user("Turn 1"),
            _assistant("just a response"),
        ]
        await _rewritten_messages(plugin, no_duplicates)
        assert "19 tool toks pruned" in published[-1][1]


class TestToolCallPruneExclusions:
    """tool_call_prune_exclusions prevents matching tool results from being pruned."""

    async def test_exclusion_by_name_keeps_tool_result(self) -> None:
        plugin = OpenAIContextModifier(
            _amortized_config(exclusions=[{"tool_name_regex": "search"}])
        )
        history = _exclusion_history(tool_name="search")
        assert _pruned_ids(await _rewritten_messages(plugin, history)) == set()

    async def test_exclusion_by_name_and_args_keeps_tool_result(self) -> None:
        plugin = OpenAIContextModifier(_amortized_config(
            exclusions=[{"tool_name_regex": "search", "tool_args_regex": "query"}]
        ))
        history = _exclusion_history(tool_name="search", tool_args=SEARCH_FOO)
        assert _pruned_ids(await _rewritten_messages(plugin, history)) == set()

    async def test_exclusion_by_args_only_keeps_tool_result(self) -> None:
        plugin = OpenAIContextModifier(
            _amortized_config(exclusions=[{"tool_args_regex": "important"}])
        )
        history = _exclusion_history(tool_args='{"flag":"important"}')
        assert _pruned_ids(await _rewritten_messages(plugin, history)) == set()

    async def test_no_exclusion_still_prunes(self) -> None:
        plugin = OpenAIContextModifier(_amortized_config())
        assert _pruned_ids(await _rewritten_messages(plugin, _exclusion_history()))

    async def test_name_match_but_args_no_match_still_prunes(self) -> None:
        """Both name and args regex must match within a single exclusion entry."""
        plugin = OpenAIContextModifier(_amortized_config(
            exclusions=[{"tool_name_regex": "search", "tool_args_regex": "notpresent"}]
        ))
        history = _exclusion_history(tool_name="search", tool_args=SEARCH_FOO)
        assert _pruned_ids(await _rewritten_messages(plugin, history))

    async def test_multiple_exclusions_any_match_excludes(self) -> None:
        """A tool matching any exclusion entry should be excluded."""
        plugin = OpenAIContextModifier(_amortized_config(exclusions=[
            {"tool_name_regex": "read_file"},
            {"tool_name_regex": "search"},
        ]))
        history = _exclusion_history(tool_name="search")
        assert _pruned_ids(await _rewritten_messages(plugin, history)) == set()

    async def test_invalid_regex_raises_config_error(self) -> None:
        """An invalid regex entry fails fast at construction instead of being silently skipped."""
        with pytest.raises(ConfigError, match="tool_name_regex is not a valid regex"):
            OpenAIContextModifier(
                _amortized_config(exclusions=[{"tool_name_regex": "[invalid"}])
            )

    async def test_malformed_entry_error_names_prune_config_key(self) -> None:
        """A malformed entry error names tool_call_prune_exclusions, not another plugin's key."""
        with pytest.raises(ConfigError, match="tool_call_prune_exclusions"):
            OpenAIContextModifier(_amortized_config(exclusions=[{"no_pattern": "x"}]))

    async def test_exclusion_does_not_affect_dedup(self) -> None:
        """Exclusions apply to amortized pruning, not dedup. Dedup still works."""
        plugin = OpenAIContextModifier({
            "dedup_tool_calls_per_turn": True,
            "tool_call_prune_exclusions": [{"tool_name_regex": "search"}],
        })
        history = [
            _system(HELPER_PROMPT),
            _user("Search for foo three times"),
            _assistant_calling(
                _tool_call("tc1", "search", SEARCH_FOO),
                _tool_call("tc2", "search", SEARCH_FOO),
                _tool_call("tc3", "search", SEARCH_FOO),
                content="I'll search multiple times.",
            ),
            _tool_result("tc1", LONG_TOOL_RESULT),
            _tool_result("tc2", LONG_TOOL_RESULT),
            _tool_result("tc3", "short"),
        ]
        messages = await _rewritten_messages(plugin, history)
        assert _tool_contents(messages) == [PRUNED, PRUNED, "short"]

    async def test_mixed_excluded_and_normal_tools(self) -> None:
        """Excluded tools are kept, non-excluded tools are pruned normally."""
        plugin = OpenAIContextModifier(
            _amortized_config(exclusions=[{"tool_name_regex": "^read_file$"}])
        )
        history = [_system()] + _alternating_tool_turns(32) + [_assistant("A")]
        pruned = _pruned_ids(await _rewritten_messages(plugin, history))

        read_file_ids = {f"t{i}" for i in range(1, 33) if i % 2 == 1}
        assert pruned.isdisjoint(read_file_ids)
        assert pruned

    async def test_production_config_excludes_skill_and_agents_md(self) -> None:
        """The config.yaml exclusion keeps skill.md and agents.md reads from pruning."""
        config = _amortized_config(exclusions=MD_EXCLUSION)
        plugin = OpenAIContextModifier(config)
        history = [_system()] + _read_tool_turns(22, READ_PATHS) + [_assistant("A")]
        pruned = _pruned_ids(await _rewritten_messages(plugin, history))

        excluded = _ids_reading(22, READ_PATHS, "SKILL.md") | _ids_reading(
            22, READ_PATHS, "agents.md"
        )
        assert pruned.isdisjoint(excluded)

    async def test_production_config_prunes_non_matching_file(self) -> None:
        """Non-matching files are pruned normally by the production config."""
        config = _amortized_config(exclusions=MD_EXCLUSION)
        plugin = OpenAIContextModifier(config)
        history = [_system()] + _read_tool_turns(22, ["readme.txt"]) + [_assistant("A")]
        assert _pruned_ids(await _rewritten_messages(plugin, history))

    async def test_yaml_loaded_regex_respected_in_pruning(self, tmp_path: Path) -> None:
        """A regex loaded from actual YAML config is effective and respected in pruning."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(_CONFIG_YAML)

        config = load_config(str(config_file))
        plugin_configs = [
            p["config"] for p in config.plugins if p["type"] == "OpenAIContextModifier"
        ]
        assert len(plugin_configs) == 1
        assert plugin_configs[0]["tool_call_prune_exclusions"] == MD_EXCLUSION

        plugin = OpenAIContextModifier(plugin_configs[0])
        history = [_system()] + _read_tool_turns(22, READ_PATHS) + [_assistant("A")]
        pruned = _pruned_ids(await _rewritten_messages(plugin, history))

        excluded = _ids_reading(22, READ_PATHS, "SKILL.md") | _ids_reading(
            22, READ_PATHS, "agents.md"
        )
        assert pruned.isdisjoint(excluded)
        assert pruned & _ids_reading(22, READ_PATHS, "readme.txt")


class TestToolArgsTruncation:
    """Tool args truncation in the prune zone and exclusion behavior."""

    async def test_args_truncated_in_prune_zone(self) -> None:
        plugin = OpenAIContextModifier({"preserve_tool_calls_min_window_size": 2})
        messages = await _rewritten_messages(plugin, _prune_zone_history(LONG_ARGS))
        assert _sole_tool_call(messages)["function"]["arguments"] == TRUNCATED_ARGS

    async def test_args_not_truncated_when_short(self) -> None:
        short_args = '{"input": "hello"}'
        plugin = OpenAIContextModifier({"preserve_tool_calls_min_window_size": 2})
        messages = await _rewritten_messages(plugin, _prune_zone_history(short_args))
        assert _sole_tool_call(messages)["function"]["arguments"] == short_args

    async def test_args_not_truncated_outside_prune_zone(self) -> None:
        plugin = OpenAIContextModifier({"preserve_tool_calls_min_window_size": 2})
        messages = await _rewritten_messages(plugin, _recent_history(LONG_ARGS))
        assert _sole_tool_call(messages)["function"]["arguments"] == LONG_ARGS

    async def test_args_exclusion_name_only_does_not_exempt(self) -> None:
        plugin = OpenAIContextModifier({
            "preserve_tool_calls_min_window_size": 2,
            "tool_call_prune_exclusions": [{"tool_name_regex": "search"}],
        })
        messages = await _rewritten_messages(plugin, _prune_zone_history(LONG_ARGS))
        assert _sole_tool_call(messages)["function"]["arguments"] == TRUNCATED_ARGS

    async def test_args_exclusion_args_only_does_not_exempt(self) -> None:
        plugin = OpenAIContextModifier({
            "preserve_tool_calls_min_window_size": 2,
            "tool_call_prune_exclusions": [{"tool_args_regex": '"input"'}],
        })
        messages = await _rewritten_messages(plugin, _prune_zone_history(LONG_ARGS))
        assert _sole_tool_call(messages)["function"]["arguments"] == TRUNCATED_ARGS

    async def test_args_exclusion_both_match_exempts(self) -> None:
        plugin = OpenAIContextModifier({
            "preserve_tool_calls_min_window_size": 2,
            "tool_call_prune_exclusions": [
                {"tool_name_regex": "search", "tool_args_regex": '"input"'}
            ],
        })
        messages = await _rewritten_messages(plugin, _prune_zone_history(LONG_ARGS))
        assert _sole_tool_call(messages)["function"]["arguments"] == LONG_ARGS


class TestOngoingRequestLifecycle:
    """Ongoing request tracking via on_request, on_response_chunk, and on_cancelled."""

    async def test_on_request_adds_to_ongoing(self) -> None:
        """on_request adds the request_id to the ongoing set."""
        plugin = OpenAIContextModifier({})
        await _rewritten_messages(plugin, [_user("hello")])
        assert plugin.is_request_ongoing(1) is True

    async def test_on_response_chunk_final_removes_from_ongoing(self) -> None:
        """on_response_chunk with is_final=True removes the request_id from ongoing."""
        plugin = OpenAIContextModifier({})
        await _rewritten_messages(plugin, [_user("hello")])
        await plugin.on_response_chunk(1, {}, b"", COMPLETIONS, 200, is_final=True)
        assert plugin.is_request_ongoing(1) is False

    async def test_on_cancelled_removes_from_ongoing(self) -> None:
        """on_cancelled removes the request_id from ongoing."""
        plugin = OpenAIContextModifier({})
        await _rewritten_messages(plugin, [_user("hello")])
        plugin.on_cancelled(1)
        assert plugin.is_request_ongoing(1) is False


# ── Same work on every API ──


def _comparable(messages: list[dict]) -> list[dict]:
    """The messages without empty assistant content."""
    return [
        {k: v for k, v in m.items() if not (k == "content" and m["role"] == "assistant" and v == "")}
        for m in messages
    ]


async def _wire_rewrite(
    config: _Config, wire: Wire, messages: list[_Message]
) -> tuple[list[dict], list[tuple[str, str]], bytes, dict[str, str]]:
    """Put one request of the wire's API through a fresh plugin."""
    plugin = OpenAIContextModifier(config)
    published = _capture_ui(plugin)
    headers, body, _ = await plugin.on_request(1, {}, wire.request(messages), wire.path)
    return wire.history(body), published, body, headers


_PARITY_CASES = [
    pytest.param({"preserve_thinking_tokens_from_previous_turns": 1}, _reasoning_conversation(), id="prune-reasoning"),
    pytest.param(
        {**_amortized_config(), "preserve_tool_calls_min_window_size": 2},
        _tool_turns(6, "read", LONG_ARGS),
        id="prune-tool-calls",
    ),
    pytest.param(
        {**_amortized_config(MD_EXCLUSION), "preserve_tool_calls_min_window_size": 1},
        _read_tool_turns(4, READ_PATHS),
        id="prune-with-exclusions",
    ),
    pytest.param(
        {"dedup_tool_calls_per_turn": True},
        [_user("Q"), _assistant_calling(_tool_call("t1", "s", SEARCH_FOO)), _tool_result("t1", LONG_TOOL_RESULT),
         _assistant_calling(_tool_call("t2", "s", SEARCH_FOO)), _tool_result("t2", LONG_TOOL_RESULT)],
        id="dedup-tool-calls",
    ),
    pytest.param(_repeat_config(), [_system()] + _chat_turns(20) + [_user("u21")], id="repeat-system-prompt"),
]


class TestApiParity:
    """Context edits are the same on every API."""

    @pytest.mark.parametrize("config, messages", _PARITY_CASES)
    async def test_same_edits_on_every_api(self, config: _Config, messages: list[_Message]) -> None:
        """Each history edit is the same on chat and Responses, with the same panels."""
        chat_messages, chat_panels, _, _ = await _wire_rewrite(config, CHAT_WIRE, messages)
        responses_messages, responses_panels, _, _ = await _wire_rewrite(config, RESPONSES_WIRE, messages)
        assert _comparable(responses_messages) == _comparable(chat_messages)
        assert _comparable(chat_messages) != _comparable(messages)
        assert responses_panels == chat_panels

    async def test_body_is_reencoded_with_its_length(self, wire: Wire) -> None:
        """The rewritten body carries its own length."""
        _, _, body, headers = await _wire_rewrite(_repeat_config(), wire, [_system(), _user("u1")])
        assert headers["Content-Length"] == str(len(body))

    async def test_responses_edits_keep_item_fields(self) -> None:
        """Pruned Responses input keeps the fields of the items left."""
        config = {"preserve_thinking_tokens_from_previous_turns": 0}
        reasoning = {"type": "reasoning", "id": "rs_1", "status": "completed", "summary": [], "content": [{"type": "reasoning_text", "text": "deep"}]}
        message = {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": "A"}]}
        user = {"type": "message", "role": "user", "content": "Q"}
        plugin = OpenAIContextModifier(config)
        _, body, _ = await plugin.on_request(1, {}, responses_request([user, reasoning, message, user]), RESPONSES)
        assert json.loads(body)["input"] == [user, message, user]

    async def test_instructions_are_the_system_prompt_to_repeat(self) -> None:
        """Responses instructions are the system prompt that gets repeated."""
        plugin = OpenAIContextModifier(_repeat_config())
        items = responses_input(_chat_turns(20) + [_user("u21")])
        _, body, _ = await plugin.on_request(1, {}, responses_request(items, instructions=SYSTEM_PROMPT), RESPONSES)
        forwarded = json.loads(body)
        assert forwarded["instructions"] == SYSTEM_PROMPT
        assert forwarded["input"][-2:] == [
            {"type": "message", "role": "system", "content": SYSTEM_PROMPT},
            {"type": "message", "role": "user", "content": "u21"},
        ]

    async def test_conversation_state_is_shared_across_apis(self) -> None:
        """A conversation that moves from chat to Responses keeps its pruning window."""
        plugin = OpenAIContextModifier({**_amortized_config(), "preserve_tool_calls_min_window_size": 2})
        published = _capture_ui(plugin)
        await plugin.on_request(1, {}, CHAT_WIRE.request(_tool_turns(6, arguments=SEARCH_FOO)), CHAT_COMPLETIONS)
        await plugin.on_request(2, {}, RESPONSES_WIRE.request(_tool_turns(7, arguments=SEARCH_FOO)), RESPONSES)
        assert len(published) == 2
        assert published[-1][1].count('<div style="width:100%') == 1

    @pytest.mark.parametrize("suffix", ["?stream=1", "/input_tokens"])
    async def test_only_the_exact_path_is_handled(self, wire: Wire, suffix: str) -> None:
        """A path with anything after the API path is left alone."""
        plugin = OpenAIContextModifier({"preserve_thinking_tokens_from_previous_turns": 0})
        body = wire.request(_reasoning_conversation())
        _, result, _ = await plugin.on_request(1, {}, body, wire.path + suffix)
        assert result is body
