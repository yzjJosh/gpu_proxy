"""OpenAI context modifier plugin for reasoning token management."""

from html import escape
import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable
import tiktoken
from app.config import ServiceConfig
from app.plugins.base import PluginBase
from app.plugins.openai_dialect import ChatCompletionsDialect, ConversationDialect, find_exact
from app.plugins.openai_responses_dialect import ResponsesDialect
from app.plugins.utils import (
    ToolCallExclusion,
    decode_json_object,
    is_tool_call_excluded,
    parse_tool_call_exclusions,
)

__all__ = ["OpenAIContextModifier"]

_tiktoken_encoder = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    """Count tokens in text using cl100k_base encoding."""
    if not text:
        return 0
    return len(_tiktoken_encoder.encode(text, allowed_special="all"))


_DIALECTS: tuple[ConversationDialect, ...] = (ChatCompletionsDialect(), ResponsesDialect())
_TOOL_RESULT_PLACEHOLDER = "[tool result pruned]"
_TRUNCATE_ARGS_AT = 100
_TRUNCATE_ARGS_PLACEHOLDER = "..."
_TRUNCATE_ARGS_TOTAL = _TRUNCATE_ARGS_AT + len(_TRUNCATE_ARGS_PLACEHOLDER)

_MAX_STATE_ENTRIES = 20


def _fmt_num(n: int) -> str:
    if n >= 1000:
        val = n / 1000
        if val == int(val):
            return f"{int(val)}K"
        return f"{val:.1f}K"
    return str(n)

@dataclass
class _ConversationState:
    pruned_up_to_turn: int = 0
    system_prompt_index: int = 0
    first_user_content: str = ""
    tokens_reasoning_pruned: int = 0
    tokens_tool_pruned: int = 0
    tokens_tool_deduped: int = 0
    system_prompt_turns_advanced: int = 0


def _conversation_id(messages: list) -> str | None:
    """Derive a stable conversation ID from system + first user message.

    Returns SHA256 hex digest, or None if no first user message exists.
    """
    system_content = ""
    first_user_content = None

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "system" and not system_content:
            system_content = msg.get("content", "") or ""
        elif role == "user" and first_user_content is None:
            first_user_content = msg.get("content", "") or ""

    if first_user_content is None:
        return None

    raw = f"{system_content}\x00{first_user_content}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

class _LRUOrderedDict(OrderedDict[str, _ConversationState]):
    """OrderedDict that evicts oldest entry when capacity is exceeded."""

    def __setitem__(self, key: str, value: _ConversationState) -> None:
        super().__setitem__(key, value)
        if len(self) > _MAX_STATE_ENTRIES:
            self.popitem(last=False)


# The plugin manager loads this module with spec.loader.exec_module and finds the
# class by name from config, so no production file imports it.
# lint-skip: coding.exported-api-unused
class OpenAIContextModifier(PluginBase):
    """Plugin that prunes reasoning tokens from assistant messages in conversation history.

    Preserves reasoning for the last N turns (user messages) and strips reasoning
    from older assistant messages to prevent context dilution.
    """

    def __init__(
        self, config: dict, service_config: ServiceConfig | None = None
    ) -> None:
        super().__init__(config, service_config)
        self._preserve_thinking_turns: int = config.get(
            "preserve_thinking_tokens_from_previous_turns", -1
        )
        self._tool_calls_min_window: int = config.get(
            "preserve_tool_calls_min_window_size", -1
        )
        self._tool_calls_max_window_ratio: float = config.get(
            "preserve_tool_calls_max_window_size_ratio", 0.5
        )
        self._repeat_min_window: int = config.get("repeat_system_prompt_min_window", -1)
        self._repeat_max_window_ratio: float = config.get("repeat_system_prompt_max_window_ratio", 0.5)
        self._should_dedup_tool_calls_per_turn: bool = config.get(
            "dedup_tool_calls_per_turn", False
        )
        self._prune_exclusions: list[ToolCallExclusion] = parse_tool_call_exclusions(
            config.get("tool_call_prune_exclusions", []),
            "OpenAIContextModifier",
            "tool_call_prune_exclusions",
        )
        self._state: _LRUOrderedDict = _LRUOrderedDict()
        self._ui_publish: Callable[[str, str], None] | None = None

    def name(self) -> str:
        return self.config.get("name", "openai_context_modifier")

    def init_ui(self, publish: Callable[[str, str], None]) -> None:
        """Initialize the UI publish callback.

        Args:
            publish: A callable that accepts (plugin_name: str, html: str)
                and publishes the HTML fragment to the UI.
        """
        self._ui_publish = publish

    def _publish_ui(self) -> None:
        if self._ui_publish is None:
            return
        lines: list[str] = []
        for conv_id, state in reversed(list(self._state.items())):
            state: _ConversationState = state
            if len(lines) >= 10:
                break
            label = escape(state.first_user_content)
            if not label:
                label = conv_id[:8]
            chips: list[str] = []
            if state.tokens_reasoning_pruned:
                chips.append(
                    f'<span style="background:#e8f4fd;color:#0077b6;padding:2px 6px;border-radius:3px;font-size:0.8rem;margin-left:4px">{_fmt_num(state.tokens_reasoning_pruned)} reasoning toks pruned</span>'
                )
            if state.tokens_tool_pruned or state.tokens_tool_deduped:
                chips.append(
                    f'<span style="background:#fff3cd;color:#856404;padding:2px 6px;border-radius:3px;font-size:0.8rem;margin-left:4px">{_fmt_num(state.tokens_tool_pruned+state.tokens_tool_deduped)} tool toks pruned</span>'
                )
            if state.system_prompt_turns_advanced:
                chips.append(
                    f'<span style="background:#d4edda;color:#155724;padding:2px 6px;border-radius:3px;font-size:0.8rem;margin-left:4px">system prompt advanced {_fmt_num(state.system_prompt_turns_advanced)} turns</span>'
                )
            if not chips:
                continue
            lines.append(
                f'<div style="width:100%;margin-bottom:2px"><span style="color:#888;font-size:0.85rem">"{label}":</span> ' + " ".join(chips) + '</div>'
            )
            if len(lines) >= 10:
                break
        if lines:
            self._ui_publish("Context Modifier", "".join(lines))

    def _get_state(self, conv_id: str) -> _ConversationState:
        """Get or create state for a conversation."""
        if conv_id in self._state:
            self._state.move_to_end(conv_id)
            return self._state[conv_id]
        state = _ConversationState(
            pruned_up_to_turn=0,
            system_prompt_index=0,
            first_user_content="",
            tokens_reasoning_pruned=0,
            tokens_tool_pruned=0,
            system_prompt_turns_advanced=0,
        )
        self._state[conv_id] = state
        return state

    async def on_request(
        self, request_id: int, headers: dict[str, str], body: bytes, path: str
    ) -> tuple[dict[str, str], bytes, str]:
        headers, body, path = await super().on_request(request_id, headers, body, path)
        dialect = find_exact(_DIALECTS, path)
        if dialect is None:
            return headers, body, path

        if (
            self._preserve_thinking_turns == -1
            and self._tool_calls_min_window == -1
            and self._repeat_min_window == -1
            and not self._should_dedup_tool_calls_per_turn
        ):
            return headers, body, path

        parsed = decode_json_object(body)
        if parsed is None:
            return headers, body, path

        projection = dialect.project(parsed)
        if projection is None:
            return headers, body, path
        messages = projection.messages

        conv_id = _conversation_id(messages)
        state: _ConversationState | None = None
        if conv_id is not None:
            state = self._get_state(conv_id)
            if not state.first_user_content:
                for msg in messages:
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        content = msg.get("content", "") or ""
                        if isinstance(content, list):
                            content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
                        content = content.replace("\n", " ").strip()
                        if content:
                            state.first_user_content = content[:60] + ("..." if len(content) > 60 else "")
                        break
        dedup_tokens = 0
        if self._should_dedup_tool_calls_per_turn:
            dedup_tokens = _dedup_tool_calls_per_turn(messages)
        if conv_id is not None and dedup_tokens > 0:
            state = self._get_state(conv_id)
            state.tokens_tool_deduped = dedup_tokens
        if self._preserve_thinking_turns != -1:
            reasoning_state = state if state else _ConversationState()
            _prune_reasoning(messages, self._preserve_thinking_turns, reasoning_state)
        if self._tool_calls_min_window > 0 and conv_id is not None:
            state = self._get_state(conv_id)
            _prune_tool_calls_amortized(
                messages,
                self._tool_calls_min_window,
                self._tool_calls_max_window_ratio,
                state,
                self._prune_exclusions,
            )
        if self._repeat_min_window > 0 and conv_id is not None:
            state = self._get_state(conv_id)
            _repeat_system_prompt(
                messages,
                self._repeat_min_window,
                self._repeat_max_window_ratio,
                state,
            )
        projection.write_back()
        new_body = json.dumps(parsed, ensure_ascii=False).encode("utf-8")
        headers["Content-Length"] = str(len(new_body))
        try:
            self._publish_ui()
        except Exception:
            pass
        return headers, new_body, path


def _prune_reasoning(messages: list, preserve_thinking_turns: int, state: _ConversationState) -> None:
    """Strip reasoning from assistant messages beyond the N-turn window.

    Iterates messages from end to start, counting user messages as turn
    boundaries. Assistant messages with user_turns_seen >= preserve_thinking_turns
    have their reasoning field removed. Pruned token counts are accumulated
    in state.tokens_reasoning_pruned.

    Args:
        messages: List of message dicts from the request body.
        preserve_thinking_turns: Number of turns to preserve (0 = strip all).
        state: Conversation state to accumulate pruned token counts.
    """
    user_turns_seen = 0

    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if not isinstance(msg, dict):
            continue

        role = msg.get("role")
        if role == "user":
            user_turns_seen += 1
        elif role == "assistant" and user_turns_seen >= preserve_thinking_turns:
            reasoning = msg.pop("reasoning_content", None)
            if reasoning:
                state.tokens_reasoning_pruned += _count_tokens(reasoning)


def _build_tool_call_lookup(messages: list) -> dict[str, tuple[str, str]]:
    """Build tool_call_id -> (name, arguments) lookup from assistant messages."""
    result: dict[str, tuple[str, str]] = {}
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for tc in (msg.get("tool_calls") or []):
            if not isinstance(tc, dict):
                continue
            tc_id = tc.get("id")
            if tc_id is None:
                continue
            func = tc.get("function")
            if isinstance(func, dict):
                name = func.get("name", "") or ""
                args = func.get("arguments", "") or ""
            else:
                name = tc.get("name", "") or ""
                args = tc.get("arguments", "") or ""
            if not isinstance(name, str) or not isinstance(args, str):
                continue
            result[tc_id] = (name, args)
    return result


def _is_tool_result_excluded(msg: dict, tool_call_lookup: dict[str, tuple[str, str]], exclusions: list[ToolCallExclusion]) -> bool:
    """Check if a tool message matches any prune exclusion entry."""
    tc_id = msg.get("tool_call_id")
    if tc_id is None:
        return False
    name_args = tool_call_lookup.get(tc_id)
    if name_args is None:
        return False
    name, args = name_args
    return is_tool_call_excluded(name, args, exclusions)


def _is_tool_call_excluded(name: str, args: str, exclusions: list[ToolCallExclusion]) -> bool:
    """Check if both tool name AND arguments match an exclusion entry."""
    for entry in exclusions:
        if entry.name_re and entry.name_re.search(name):
            if entry.args_re and entry.args_re.search(args):
                return True
    return False


def _prune_tool_calls_amortized(
    messages: list, min_window: int, max_window_ratio: float, state: _ConversationState, prune_exclusions: list[ToolCallExclusion] | None = None
) -> None:
    """Replace content of old tool responses using amortized window pruning with state.

    Uses state.pruned_up_to_turn to compute unpruned_turns correctly, since the
    client always sends full unpruned history.

    Args:
        messages: List of message dicts from the request body.
        min_window: Minimum turns to always keep.
        max_window_ratio: Ratio controlling max window growth per turn.
        state: Conversation state tracking pruned_up_to_turn.
    """
    total_turns = 0
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            total_turns += 1

    if total_turns <= min_window:
        return

    state.pruned_up_to_turn = max(0, min(state.pruned_up_to_turn, total_turns - min_window))
    unpruned_turns = total_turns - state.pruned_up_to_turn
    max_window = int(total_turns * max_window_ratio) + min_window

    if unpruned_turns > max_window:
        prune_threshold = min_window
    else:
        prune_threshold = total_turns - state.pruned_up_to_turn

    tool_call_lookup = _build_tool_call_lookup(messages)
    placeholder_len = len(_TOOL_RESULT_PLACEHOLDER)

    user_turns_seen = 0
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if not isinstance(msg, dict):
            continue

        role = msg.get("role")
        if role == "user":
            user_turns_seen += 1
        elif role == "assistant" and user_turns_seen >= prune_threshold:
            _truncate_tool_args(msg, prune_exclusions or [], state)
        elif role == "tool" and user_turns_seen >= prune_threshold:
            if prune_exclusions and _is_tool_result_excluded(msg, tool_call_lookup, prune_exclusions):
                continue
            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > placeholder_len:
                original = content
                msg["content"] = _TOOL_RESULT_PLACEHOLDER
                state.tokens_tool_pruned += _count_tokens(original) - _count_tokens(_TOOL_RESULT_PLACEHOLDER)

    if unpruned_turns > max_window:
        state.pruned_up_to_turn = total_turns - min_window


def _truncate_string(value: str, max_len: int) -> str:
    """Truncate an overlong string, appending the placeholder."""
    if len(value) <= max_len:
        return value
    return value[:max_len] + _TRUNCATE_ARGS_PLACEHOLDER


def _truncate_json_values(doc: dict | list, max_len: int) -> None:
    """Truncate overlong string values in a parsed JSON document, in place."""
    if isinstance(doc, dict):
        for key, value in doc.items():
            if isinstance(value, (dict, list)):
                _truncate_json_values(value, max_len)
            elif isinstance(value, str):
                doc[key] = _truncate_string(value, max_len)
        return
    for i, value in enumerate(doc):
        if isinstance(value, (dict, list)):
            _truncate_json_values(value, max_len)
        elif isinstance(value, str):
            doc[i] = _truncate_string(value, max_len)


def _truncate_tool_args(msg: dict, exclusions: list[ToolCallExclusion], state: _ConversationState) -> None:
    """Truncate oversized function.arguments in assistant tool calls.

    Parses arguments as JSON, truncates each string value, and re-encodes.
    """
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        args = fn.get("arguments", "")
        if not isinstance(args, str) or len(args) <= _TRUNCATE_ARGS_TOTAL:
            continue
        name = fn.get("name", "") or ""
        if _is_tool_call_excluded(name, args, exclusions):
            continue
        try:
            parsed = json.loads(args)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, (dict, list)):
            _truncate_json_values(parsed, _TRUNCATE_ARGS_AT)
        elif isinstance(parsed, str):
            parsed = _truncate_string(parsed, _TRUNCATE_ARGS_AT)
        truncated_json = json.dumps(parsed, ensure_ascii=False)
        if len(truncated_json) < len(args):
            state.tokens_tool_pruned += _count_tokens(args) - _count_tokens(truncated_json)
            tc["function"]["arguments"] = truncated_json


def _dedup_tool_calls_per_turn(messages: list) -> int:
    """Deduplicate tool results within each user turn.

    For each function_name + arguments combination within a single user turn,
    only the last tool result is kept. Earlier duplicates are replaced with
    _TOOL_RESULT_PLACEHOLDER if longer.

    Returns the number of tokens pruned (original tokens minus placeholder tokens).
    """
    tool_call_lookup = _build_tool_call_lookup(messages)

    # Group tool messages by (turn_id, dedup_key). Backward scan, first seen = last occurrence = keep.
    turn_seen: set[str] = set()
    tokens_pruned = 0
    placeholder_tokens = _count_tokens(_TOOL_RESULT_PLACEHOLDER)

    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if not isinstance(msg, dict):
            continue

        role = msg.get("role")
        if role == "user":
            turn_seen = set()
        elif role == "tool":
            tc_id = msg.get("tool_call_id")
            if tc_id is None:
                continue
            name_args = tool_call_lookup.get(tc_id)
            if name_args is None:
                continue
            name, args = name_args
            dedup_key = hashlib.sha256((name + "\x00" + args).encode("utf-8")).hexdigest()

            if dedup_key in turn_seen:
                content = msg.get("content", "")
                if isinstance(content, str) and _count_tokens(content) > placeholder_tokens:
                    tokens_pruned += _count_tokens(content) - placeholder_tokens
                    msg["content"] = _TOOL_RESULT_PLACEHOLDER
            else:
                turn_seen.add(dedup_key)

    return tokens_pruned


def _find_last_user_index(messages: list) -> int:
    """Return index of the last user message, or -1 if none exists."""
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], dict) and messages[i].get("role") == "user":
            return i
    return -1


def _repeat_system_prompt(
    messages: list, min_window: int, max_window_ratio: float, state: _ConversationState
) -> None:
    """Repeat system prompt near recent turns to keep it close to active conversation.

    Args:
        messages: List of message dicts from the request body.
        min_window: Never trigger repeat if within this many user turns from end.
        max_window_ratio: Max window = floor(total_turns * ratio) + min_window.
        state: Conversation state with system_prompt_index.
    """
    if min_window < 0:
        return
    if not messages:
        return
    if messages[0].get("role") != "system":
        return

    # Pre-check: stale state reset
    if state.system_prompt_index >= len(messages) - 1:
        state.system_prompt_turns_advanced = 0
        state.system_prompt_index = 0

    original_system_content = messages[0].get("content", "")
    if not original_system_content.strip():
        return

    total_turns = 0
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            total_turns += 1

    distance = 0
    for i in range(state.system_prompt_index, len(messages)):
        if isinstance(messages[i], dict) and messages[i].get("role") == "user":
            distance += 1

    max_window = int(total_turns * max_window_ratio) + min_window

    if distance > max_window:
        # Inject a copy before the last user message
        last_user_index = _find_last_user_index(messages)
        if last_user_index >= 0:
            messages.insert(last_user_index, {"role": "system", "content": original_system_content})
            state.system_prompt_turns_advanced = total_turns - 1
            state.system_prompt_index = last_user_index
    elif state.system_prompt_index > 0:
        # Re-apply: inject at tracked position
        messages.insert(state.system_prompt_index, {"role": "system", "content": original_system_content})
