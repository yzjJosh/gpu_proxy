"""Tests for the OpenAIRequestModifier plugin's public on_request hook."""

import json

import pytest

from tests.app.conftest import CHAT_COMPLETIONS, apply_on_request, json_body, make_chat_modifier

_Body = dict

_NON_TARGET = "/v1/embeddings"

# A request value the overrides never mention, so the merge must leave it alone.
_UNTOUCHED_TEMPERATURE = 0.5
# A value that exists only in the overrides, so the merge has to introduce it.
_ADDED_TEMPERATURE = 0.7
# A request-only field, large enough to be worth carrying through the merge.
_MAX_TOKENS = 100


def _decoded(body: bytes) -> _Body:
    """Decode a response body back to a dict.

    Args:
        body: The raw JSON body.

    Returns:
        The parsed dict.
    """
    return json.loads(body)


class TestRequestOverride:
    """on_request merges configured overrides into matching request bodies."""

    async def test_scalar_override(self) -> None:
        """A config scalar replaces the request's value at the same key."""
        _, body, _ = await apply_on_request(make_chat_modifier({"model": "gpt-4"}), json_body({"model": "old"}))
        assert _decoded(body) == {"model": "gpt-4"}

    async def test_request_only_keys_preserved(self) -> None:
        """Keys the overrides never mention survive the merge."""
        _, body, _ = await apply_on_request(
            make_chat_modifier({"temperature": _ADDED_TEMPERATURE}),
            json_body({"model": "gpt-4", "max_tokens": _MAX_TOKENS}),
        )
        assert _decoded(body) == {
            "model": "gpt-4",
            "max_tokens": _MAX_TOKENS,
            "temperature": _ADDED_TEMPERATURE,
        }

    async def test_config_only_keys_added(self) -> None:
        """Keys present only in the overrides are added to the result."""
        _, body, _ = await apply_on_request(
            make_chat_modifier({"response_format": {"type": "json_schema"}}),
            json_body({"model": "gpt-4"}),
        )
        assert _decoded(body) == {
            "model": "gpt-4",
            "response_format": {"type": "json_schema"},
        }

    async def test_nested_dict_recurse(self) -> None:
        """A nested dict is merged key by key rather than replaced wholesale."""
        _, body, _ = await apply_on_request(
            make_chat_modifier({"response_format": {"type": "json_schema", "json_schema": {"name": "my_schema"}}}),
            json_body({
                "model": "old",
                "temperature": _UNTOUCHED_TEMPERATURE,
                "response_format": {
                    "type": "text",
                    "json_schema": {"name": "x", "strict": True},
                },
            }),
        )
        assert _decoded(body) == {
            "model": "old",
            "temperature": _UNTOUCHED_TEMPERATURE,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "my_schema", "strict": True},
            },
        }

    async def test_deeply_nested_merge(self) -> None:
        """Recursion keeps going past three levels of nesting."""
        _, body, _ = await apply_on_request(
            make_chat_modifier({"a": {"b": {"c": {"d": 10}}}}),
            json_body({"a": {"b": {"c": {"d": 1, "e": 2}}}}),
        )
        assert _decoded(body) == {"a": {"b": {"c": {"d": 10, "e": 2}}}}


class TestRequestOverrideLists:
    """Lists are merged element by element, by index."""

    async def test_list_by_index_merge(self) -> None:
        """An override element is merged into the request element at its index."""
        _, body, _ = await apply_on_request(
            make_chat_modifier({"tools": [{"name": "new"}]}),
            json_body({"tools": [{"name": "old", "desc": "find"}]}),
        )
        assert _decoded(body) == {"tools": [{"name": "new", "desc": "find"}]}

    async def test_shorter_config_preserves_remaining(self) -> None:
        """Request elements past the end of the override list are kept."""
        _, body, _ = await apply_on_request(make_chat_modifier({"items": [10]}), json_body({"items": [1, 2, 3]}))
        assert _decoded(body) == {"items": [10, 2, 3]}

    async def test_longer_config_appends(self) -> None:
        """Override elements past the end of the request list are appended."""
        _, body, _ = await apply_on_request(make_chat_modifier({"items": [10, 20, 30]}), json_body({"items": [1]}))
        assert _decoded(body) == {"items": [10, 20, 30]}

    async def test_nested_list_in_dict(self) -> None:
        """A list reached through a dict key still merges by index."""
        _, body, _ = await apply_on_request(
            make_chat_modifier({"config": {"items": [10, 20]}}),
            json_body({"config": {"items": [1, 2, 3]}}),
        )
        assert _decoded(body) == {"config": {"items": [10, 20, 3]}}


class TestRequestTypeMismatch:
    """When the two sides disagree on type at a key, the override wins outright."""

    @pytest.mark.parametrize("request_val,override_val,expected", [
        ({"nested": True}, "scalar", "scalar"),
        ("scalar", {"nested": True}, {"nested": True}),
        ([1, 2], "scalar", "scalar"),
        ("scalar", [1, 2], [1, 2]),
    ])
    async def test_type_mismatch_config_wins(
        self, request_val: dict | str | list, override_val: dict | str | list, expected: dict | str | list
    ) -> None:
        """A dict, list or scalar on one side never merges into the other kind."""
        _, body, _ = await apply_on_request(
            make_chat_modifier({"key": override_val}), json_body({"key": request_val})
        )
        assert _decoded(body) == {"key": expected}


class TestRequestMergeCopies:
    """A nested merge keeps the request's sibling keys intact."""

    async def test_output_matches_expected(self) -> None:
        """Sibling keys around the overridden key survive the merge unchanged."""
        _, body, _ = await apply_on_request(
            make_chat_modifier({"nested": {"a": {"scale": 10}}}),
            json_body({"model": "old", "nested": {"a": 1, "b": 2}}),
        )
        assert _decoded(body) == {"model": "old", "nested": {"a": {"scale": 10}, "b": 2}}


class TestRequestPathHandling:
    """on_request leaves bodies alone when the path is not a target."""

    async def test_non_target_path_untouched(self) -> None:
        """A request on a path the plugin does not claim is returned unchanged."""
        headers, body, path = await apply_on_request(make_chat_modifier({"model": "gpt-4"}), json_body({"model": "old"}), _NON_TARGET)
        assert path == _NON_TARGET
        assert body == json_body({"model": "old"})
        assert headers == {}

    async def test_empty_overrides_return_unchanged(self) -> None:
        """A plugin with no overrides passes the request through untouched."""
        headers, body, path = await apply_on_request(make_chat_modifier({}), json_body({"model": "old"}))
        assert _decoded(body) == {"model": "old"}
        assert headers == {}
        assert path == CHAT_COMPLETIONS

    async def test_content_length_updated(self) -> None:
        """Content-Length reflects the merged body when it changes."""
        headers, body, _ = await apply_on_request(
            make_chat_modifier({"model": "gpt-4"}),
            json_body({"model": "old", "max_tokens": _MAX_TOKENS}),
        )
        assert headers["Content-Length"] == str(len(body))
