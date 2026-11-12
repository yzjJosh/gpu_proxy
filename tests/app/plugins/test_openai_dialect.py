"""Tests for the chat and legacy completions dialects and the dialect helpers."""

import json

import pytest

from app.plugins.openai_dialect import (
    ChatCompletionsDialect,
    CompletionsDialect,
    OpenAIDialect,
    TokenUsage,
    TranslatedOverrides,
    canonicalize_arguments,
    find_by_prefix,
    find_exact,
    find_prefix,
)
from app.plugins.openai_responses_dialect import ResponsesDialect

_CHAT = ChatCompletionsDialect()
_COMPLETIONS = CompletionsDialect()
_RESPONSES = ResponsesDialect()
_ALL = (_CHAT, _COMPLETIONS, _RESPONSES)

_PROMPT_TOKENS = 12
_COMPLETION_TOKENS = 34
_TOTAL_TOKENS = 46


def _chunk(*choices: dict) -> dict:
    """A chat chunk carrying the given choices."""
    return {"id": "1", "choices": list(choices)}


def _choice(delta: dict, finish_reason: str | None = None) -> dict:
    """One chat choice."""
    choice: dict = {"index": 0, "delta": delta}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    return choice


def _fragment(text_args: str = "", name: str = "", tc_id: str = "") -> dict:
    """One tool call entry of a chat delta."""
    fragment: dict = {"index": 0, "function": {}}
    if tc_id:
        fragment["id"] = tc_id
    if name:
        fragment["function"]["name"] = name
    if text_args:
        fragment["function"]["arguments"] = text_args
    return fragment


def _content(text: str) -> dict:
    """A content piece with the given text."""
    return {"type": "content", "text": text}


def _reasoning(text: str) -> dict:
    """A reasoning piece with the given text."""
    return {"type": "reasoning", "text": text}


def _finish(reason: str) -> dict:
    """A finish reason piece with the given reason."""
    return {"type": "finish_reason", "text": reason}


def _tool(text: str, tc_id: str = "", name: str = "", args: str = "") -> dict:
    """A tool call piece with the given text, id, name and arguments."""
    return {"type": "tool_call", "text": text, "tool_call_id": tc_id, "function": {"name": name, "arguments": args}}


def _read_first_piece(event: dict) -> dict:
    """Read the first matching piece of a chat chunk."""
    for choice in event.get("choices") or []:
        delta = choice.get("delta") or {}
        if delta.get("content") is not None:
            return _content(delta["content"])
        reasoning = delta.get("reasoning") or delta.get("reasoning_content")
        if reasoning is not None:
            return _reasoning(reasoning)
        for tool_call in delta.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            name = function.get("name") or ""
            args = canonicalize_arguments(function.get("arguments") or "")
            tc_id = tool_call.get("id") or ""
            if name + args or tc_id:
                return _tool(name + args, tc_id, name, args)
        if choice.get("finish_reason") is not None:
            return _finish(choice["finish_reason"])
    return {}


# Each chunk shape with the pieces the chat reader yields, in check order.
_READER_TABLE = [
    pytest.param(_chunk(_choice({"content": "hi"})), [_content("hi")], id="content"),
    pytest.param(_chunk(_choice({"reasoning": "hm"})), [_reasoning("hm")], id="reasoning"),
    pytest.param(_chunk(_choice({"reasoning_content": "hm"})), [_reasoning("hm")], id="reasoning_content"),
    pytest.param(
        _chunk(_choice({"tool_calls": [_fragment(tc_id="c1", name="run")]})),
        [_tool("run", "c1", "run")],
        id="tool-call-opening",
    ),
    pytest.param(_chunk(_choice({}, "stop")), [_finish("stop")], id="finish-alone"),
    pytest.param(_chunk(_choice({})), [], id="empty-delta"),
    pytest.param({"usage": {"prompt_tokens": 1}}, [], id="usage-only"),
    pytest.param(
        _chunk(_choice({"reasoning": "tail", "content": "head"})),
        [_reasoning("tail"), _content("head")],
        id="reasoning-and-content",
    ),
    pytest.param(
        _chunk(_choice({"content": "end"}, "stop")),
        [_content("end"), _finish("stop")],
        id="content-and-finish",
    ),
    pytest.param(
        _chunk(_choice({"reasoning": "cut"}, "length")),
        [_reasoning("cut"), _finish("length")],
        id="reasoning-and-length",
    ),
    pytest.param(
        _chunk(_choice({"tool_calls": [_fragment('{"a": '), _fragment("1}")]})),
        [_tool('{"a": ', args='{"a": '), _tool("1}", args="1}")],
        id="two-tool-call-entries",
    ),
    pytest.param(
        _chunk(
            _choice({"reasoning": "r0", "content": "c0"}, "stop"),
            _choice({"content": "c1", "tool_calls": [_fragment(tc_id="c9", name="x")]}, "tool_calls"),
        ),
        [_reasoning("r0"), _content("c0"), _content("c1"), _tool("x", "c9", "x"), _finish("stop"), _finish("tool_calls")],
        id="two-choices",
    ),
    pytest.param(
        _chunk(_choice({"content": None, "reasoning": None, "reasoning_content": "", "tool_calls": None})),
        [_reasoning("")],
        id="null-fields-and-empty-reasoning-content",
    ),
    pytest.param(
        _chunk(_choice({"content": "", "tool_calls": [{"index": 0, "id": None, "function": {"name": None, "arguments": None}}]})),
        [_content("")],
        id="empty-content-and-blank-fragment",
    ),
    pytest.param(
        _chunk(_choice({"reasoning": "", "reasoning_content": None})),
        [],
        id="empty-reasoning-falls-back-to-null",
    ),
    pytest.param(
        _chunk(_choice({"tool_calls": [_fragment("1}"), _fragment(tc_id="c2", name="next")]})),
        [_tool("1}", args="1}"), _tool("next", "c2", "next")],
        id="call-end-and-next-call",
    ),
    pytest.param(
        _chunk(_choice({"content": "say", "tool_calls": [_fragment(tc_id="c2", name="next")]})),
        [_content("say"), _tool("next", "c2", "next")],
        id="content-and-call-opening",
    ),
    pytest.param(
        _chunk(_choice({"tool_calls": [_fragment(tc_id="c1", name="a")]}), _choice({"content": "later"})),
        [_tool("a", "c1", "a"), _content("later")],
        id="first-choice-tool-call-only",
    ),
    pytest.param(
        _chunk(_choice({}), _choice({"reasoning": "second"}, "stop")),
        [_reasoning("second"), _finish("stop")],
        id="first-choice-empty",
    ),
]


class TestChatDeltaDecoderReadsEveryPiece:
    """The chat reader yields every piece of a chunk, finish reasons last."""

    @pytest.mark.parametrize("event, expected", _READER_TABLE)
    def test_pieces_in_check_order(self, event: dict, expected: list[dict]) -> None:
        """Each chunk shape gives its pieces in check order."""
        deltas, _carrier = _CHAT.delta_decoder().normalize(event)
        assert deltas == expected


# The index of the piece each reader table chunk's bytes ride on, or None when it has no piece.
_READER_CARRIERS = {
    "content": 0, "reasoning": 0, "reasoning_content": 0, "tool-call-opening": 0, "finish-alone": 0,
    "empty-delta": None, "usage-only": None, "reasoning-and-content": 1, "content-and-finish": 0,
    "reasoning-and-length": 0, "two-tool-call-entries": 0, "two-choices": 1,
    "null-fields-and-empty-reasoning-content": 0, "empty-content-and-blank-fragment": 0,
    "empty-reasoning-falls-back-to-null": None, "call-end-and-next-call": 0, "content-and-call-opening": 0,
    "first-choice-tool-call-only": 0, "first-choice-empty": 0,
}

# Each chunk shape with the index of the piece its bytes ride on.
_CARRIER_TABLE = [pytest.param(row.values[0], _READER_CARRIERS[row.id], id=row.id) for row in _READER_TABLE]

# The chunks of the carrier table that have a piece to carry their bytes.
_CARRIED_EVENTS = [pytest.param(row.values[0], id=row.id) for row in _CARRIER_TABLE if row.values[1] is not None]

# The chunks of the reader table that hold exactly one piece.
_SINGLE_PIECE_EVENTS = [pytest.param(row.values[0], id=row.id) for row in _READER_TABLE if len(row.values[1]) == 1]


class TestChatDeltaDecoderCarrier:
    """The event's bytes ride on the piece the first-match reading picks."""

    @pytest.mark.parametrize("event, expected_carrier", _CARRIER_TABLE)
    def test_carrier_index(self, event: dict, expected_carrier: int | None) -> None:
        """The bytes of a chunk ride on the piece at the expected index."""
        _deltas, carrier = _CHAT.delta_decoder().normalize(event)
        assert carrier == expected_carrier

    @pytest.mark.parametrize("event", _CARRIED_EVENTS)
    def test_carrier_is_first_match_piece(self, event: dict) -> None:
        """The piece that carries the bytes is the one the first-match reading picks."""
        deltas, carrier = _CHAT.delta_decoder().normalize(event)
        assert deltas[carrier] == _read_first_piece(event)

    @pytest.mark.parametrize("event", _SINGLE_PIECE_EVENTS)
    def test_single_piece_chunk_matches_first_match_reading(self, event: dict) -> None:
        """A chunk with one piece reads as its first matching piece."""
        deltas, _carrier = _CHAT.delta_decoder().normalize(event)
        assert deltas == [_read_first_piece(event)]


class TestChatDeltaDecoderArguments:
    """The chat reader writes tool call arguments in one form."""

    def test_arguments_are_written_in_one_json_form(self) -> None:
        """Whole JSON arguments are rewritten in one JSON form."""
        event = _chunk(_choice({"tool_calls": [_fragment('{"a":1,  "b": [2]}', tc_id="c1")]}))
        deltas, _ = _CHAT.delta_decoder().normalize(event)
        assert deltas == [_tool('{"a": 1, "b": [2]}', "c1", "", '{"a": 1, "b": [2]}')]

    def test_partial_arguments_are_kept_as_sent(self) -> None:
        """Arguments that are not whole JSON are kept as sent."""
        event = _chunk(_choice({"tool_calls": [_fragment('{"a":', tc_id="c1")]}))
        deltas, _ = _CHAT.delta_decoder().normalize(event)
        assert deltas == [_tool('{"a":', "c1", "", '{"a":')]


class TestCanonicalizeArguments:
    """Tool call arguments are rewritten only when they are JSON."""

    @pytest.mark.parametrize(
        "arguments, expected",
        [
            ('{"x":1}', '{"x": 1}'),
            ('[1,2]', "[1, 2]"),
            ("not json", "not json"),
            ("", ""),
        ],
    )
    def test_rewrites_only_valid_json(self, arguments: str, expected: str) -> None:
        """Only valid JSON is rewritten."""
        assert canonicalize_arguments(arguments) == expected


class TestChatProjection:
    """The chat messages list is read and written in place."""

    def test_messages_are_the_body_list(self) -> None:
        """The chat projection is the client's own messages list."""
        messages = [{"role": "user", "content": "hi"}]
        body = {"messages": messages}
        projection = _CHAT.project(body)
        assert projection.messages is messages
        projection.messages.append({"role": "system", "content": "added"})
        projection.write_back()
        assert body == {"messages": [{"role": "user", "content": "hi"}, {"role": "system", "content": "added"}]}
        assert body["messages"] is messages

    @pytest.mark.parametrize("body", [{}, {"messages": "text"}, {"messages": {"a": 1}}, ["messages"]])
    def test_no_usable_history_gives_none(self, body: dict | list) -> None:
        """A body with no messages list gives no messages."""
        assert _CHAT.project(body) is None

    def test_splicer_is_identity(self) -> None:
        """The chat splicer passes every byte on as it was."""
        splicer = _CHAT.output_splicer()
        raw = b'data: {"choices": []}\n\ndata: [DO'
        assert splicer.splice(raw) is raw
        splicer.retry()
        assert splicer.splice(b"data: x\n\n") == b"data: x\n\n"
        assert splicer.finish() == b""


class TestPromptText:
    """The prompt text is read from each kind of request body."""

    def test_chat_counts_string_and_text_blocks(self) -> None:
        """Chat prompt text holds string content and text blocks, not reasoning."""
        body = {"messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "image_url", "image_url": {}}, {"type": "text", "text": "b"}]},
            {"role": "assistant", "content": None, "reasoning_content": "hidden"},
        ]}
        assert _CHAT.prompt_text(body) == "sys\nab\n\n"

    def test_chat_without_history_counts_nothing(self) -> None:
        """A chat body with no messages gives no prompt text."""
        assert _CHAT.prompt_text({"model": "x"}) == ""

    def test_chat_with_malformed_history_raises(self) -> None:
        """Chat messages that are not a list raise."""
        with pytest.raises(ValueError, match="messages is not a list"):
            _CHAT.prompt_text({"messages": "abc"})

    @pytest.mark.parametrize("prompt, expected", [("one", "one"), (["a", "b"], "ab")])
    def test_completions_counts_prompt(self, prompt: str | list, expected: str) -> None:
        """Legacy prompt text is the prompt string or its joined parts."""
        assert _COMPLETIONS.prompt_text({"prompt": prompt}) == expected


class TestForceStream:
    """Forcing a stream also asks for token usage."""

    @pytest.mark.parametrize("dialect", [_CHAT, _COMPLETIONS], ids=["chat", "completions"])
    def test_choices_dialects_ask_for_usage(self, dialect: OpenAIDialect) -> None:
        """Forcing a stream on chat and legacy also asks for usage."""
        body = {"stream": False, "stream_options": {"other": 1}}
        dialect.force_stream(body)
        assert body == {"stream": True, "stream_options": {"other": 1, "include_usage": True}}


class TestGeneratedTexts:
    """Each chunk gives the text it adds for each choice."""

    def test_one_entry_per_choice(self) -> None:
        """Each choice gives one generated text entry."""
        event = _chunk(
            _choice({"content": "a", "reasoning": "b", "tool_calls": [{"function": {"name": "n", "arguments": "{}"}}]}),
            {"index": 1, "text": "legacy"},
            _choice({}),
        )
        assert _CHAT.generated_texts(event) == ["abn{}", "legacy", ""]

    def test_null_fields_count_as_empty(self) -> None:
        """Null fields count as no text."""
        event = _chunk(_choice({"content": None, "reasoning": None, "reasoning_content": "rc", "tool_calls": None}))
        assert _CHAT.generated_texts(event) == ["rc"]

    def test_event_without_choices(self) -> None:
        """An event with no choices gives no entries."""
        assert _COMPLETIONS.generated_texts({"usage": {}}) == []


class TestUsage:
    """Token counts are read from the usage chunk."""

    def test_reads_prompt_and_completion_tokens(self) -> None:
        """Usage is read from the prompt, completion and total counts."""
        event = {"usage": {"prompt_tokens": _PROMPT_TOKENS, "completion_tokens": _COMPLETION_TOKENS, "total_tokens": _TOTAL_TOKENS}}
        assert _CHAT.usage(event) == TokenUsage(_PROMPT_TOKENS, _COMPLETION_TOKENS, _TOTAL_TOKENS)

    def test_missing_counts_are_zero(self) -> None:
        """Missing counts read as zero."""
        assert _COMPLETIONS.usage({"usage": {"prompt_tokens": _PROMPT_TOKENS}}) == TokenUsage(_PROMPT_TOKENS, 0, 0)

    @pytest.mark.parametrize("event", [{}, {"usage": None}, {"choices": []}])
    def test_no_usage(self, event: dict) -> None:
        """Events without usage give none."""
        assert _CHAT.usage(event) is None


class TestAssembleNonStreaming:
    """Stream chunks are put back together into one reply."""

    _EVENTS: list[dict] = [{"id": "x", "object": "chat.completion.chunk", "choices": [{"delta": {"content": "hi"}, "text": "t"}]}]

    def test_chat_builds_chat_completion(self) -> None:
        """Chat events become one chat completion reply."""
        reply = json.loads(_CHAT.assemble_non_streaming(self._EVENTS))
        assert reply["object"] == "chat.completion"
        assert reply["choices"][0]["message"]["content"] == "thi"

    def test_completions_builds_text_completion(self) -> None:
        """Legacy events become one text completion reply."""
        reply = json.loads(_COMPLETIONS.assemble_non_streaming([{"id": "y", "choices": [{"text": "abc"}]}]))
        assert reply["object"] == "text_completion"
        assert reply["choices"][0]["text"] == "abc"

    def test_no_events(self) -> None:
        """No events give no reply."""
        assert _CHAT.assemble_non_streaming([]) is None


class TestTranslateOverrides:
    """Chat and legacy overrides keep their names."""

    @pytest.mark.parametrize("dialect", [_CHAT, _COMPLETIONS], ids=["chat", "completions"])
    def test_identity(self, dialect: OpenAIDialect) -> None:
        """Chat and legacy overrides keep their names and are copied."""
        overrides = {"max_tokens": 5, "max_output_tokens": 9, "reasoning_effort": "low", "reasoning": {"effort": "high"}}
        translated = dialect.translate_overrides(overrides)
        assert translated == TranslatedOverrides(overrides, {})
        assert translated.forced is not overrides


class TestFindDialect:
    """A request path finds its dialect."""

    @pytest.mark.parametrize(
        "path, expected",
        [
            ("/v1/chat/completions", _CHAT),
            ("/v1/chat/completions/input_tokens", _CHAT),
            ("/v1/completions?x=1", _COMPLETIONS),
            ("/v1/responses/abc/cancel", _RESPONSES),
            ("/v1/embeddings", None),
        ],
    )
    def test_by_prefix(self, path: str, expected: OpenAIDialect | None) -> None:
        """A path finds the dialect whose path it starts with."""
        assert find_by_prefix(_ALL, path) is expected

    @pytest.mark.parametrize(
        "path, expected",
        [
            ("/v1/chat/completions", _CHAT),
            ("/v1/responses", _RESPONSES),
            ("/v1/chat/completions?x", None),
            ("/v1/responses/input_tokens", None),
        ],
    )
    def test_exact(self, path: str, expected: OpenAIDialect | None) -> None:
        """A path finds only the dialect with that exact path."""
        assert find_exact(_ALL, path) is expected

    @pytest.mark.parametrize(
        "path, expected",
        [
            ("/v1/embeddings", "/v1/embeddings"),
            ("/v1/rerank?x=1", "/v1/rerank"),
            ("/v1/chat/completions", None),
        ],
    )
    def test_prefix(self, path: str, expected: str | None) -> None:
        """A path finds the first prefix it starts with."""
        assert find_prefix(("/v1/embeddings", "/v1/rerank", "/v1/embeddings/x"), path) == expected
