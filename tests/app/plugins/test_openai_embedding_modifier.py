"""Tests for OpenAIEmbeddingRequestModifier plugin."""

import json

import pytest

from app.plugins.openai_embedding_modifier import OpenAIEmbeddingRequestModifier

# The three paths the plugin treats as embedding endpoints.
_EMBEDDINGS_PATH = "/v1/embeddings"
_EMBEDDING_PATH = "/v1/embedding"
_BARE_EMBEDDING_PATH = "/embedding"
# Paths that belong to other plugins and must be left alone.
_CHAT_PATH = "/v1/chat/completions"
_RERANK_PATH = "/v1/rerank"
_HEALTH_PATH = "/health"

# The request id the plugin tracks a request under; the value itself is opaque.
_REQUEST_ID = 1

# Models named in the config overrides, distinct from the one the request carries
# so an override that is silently dropped cannot pass as a success.
_OVERRIDE_MODEL = "text-embedding-3-small"
_CONFIGURED_MODEL = "Qwen3-Embedding-0.6B"
# The model the incoming request body names, which an override has to displace.
_REQUEST_MODEL = "old"
# Dimensions exist only in the overrides, so the merge has to introduce the key.
_OVERRIDE_DIMENSIONS = 256


async def _run_request(
    plugin: OpenAIEmbeddingRequestModifier,
    body: bytes,
    path: str,
    headers: dict[str, str] | None = None,
) -> tuple[dict[str, str], bytes, str]:
    """Run one request through the plugin's request hook.

    Args:
        plugin: The modifier under test.
        body: The raw request body bytes the client sent.
        path: The request path, which decides whether overrides apply.
        headers: Request headers, defaulting to an empty header dict.

    Returns:
        The plugin's (headers, body, path) triple, unchanged for a path or a
        body it declines to touch.
    """
    return await plugin.on_request(
        _REQUEST_ID, headers if headers is not None else {}, body, path
    )


class TestOpenAIEmbeddingRequestModifierAppliesOverrides:
    """Valid JSON body gets overrides merged on embedding paths."""

    async def test_embeddings_path_applies_overrides(self) -> None:
        """Every configured override lands, and untouched request fields survive."""
        plugin = OpenAIEmbeddingRequestModifier({
            "overrides": {"model": _OVERRIDE_MODEL, "encoding_format": "float"}
        })
        body = json.dumps({"model": _REQUEST_MODEL, "input": "hello world"}).encode()
        _result_headers, result, result_path = await _run_request(
            plugin, body, _EMBEDDINGS_PATH
        )
        result_data = json.loads(result)
        assert result_data == {
            "model": _OVERRIDE_MODEL,
            "encoding_format": "float",
            "input": "hello world",
        }
        assert result_path == _EMBEDDINGS_PATH

    async def test_embedding_singular_path_applies_overrides(self) -> None:
        """The singular /v1/embedding spelling is an embedding path too."""
        plugin = OpenAIEmbeddingRequestModifier({
            "overrides": {"model": "default-embedding"}
        })
        body = json.dumps({"model": _REQUEST_MODEL, "input": ["doc1", "doc2"]}).encode()
        _result_headers, result, result_path = await _run_request(
            plugin, body, _EMBEDDING_PATH
        )
        assert json.loads(result)["model"] == "default-embedding"
        assert result_path == _EMBEDDING_PATH

    async def test_bare_embedding_path_applies_overrides(self) -> None:
        """Bare /embedding path from config.yaml gpu_routes also gets overrides."""
        plugin = OpenAIEmbeddingRequestModifier({
            "overrides": {"model": _CONFIGURED_MODEL}
        })
        body = json.dumps({"model": _REQUEST_MODEL, "input": "test"}).encode()
        _result_headers, result, result_path = await _run_request(
            plugin, body, _BARE_EMBEDDING_PATH
        )
        assert json.loads(result)["model"] == _CONFIGURED_MODEL
        assert result_path == _BARE_EMBEDDING_PATH


class TestOpenAIEmbeddingRequestModifierReturnsModifiedBytes:
    """on_request returns modified bytes directly."""

    async def test_on_request_returns_modified_bytes(self) -> None:
        """The modified body comes back as bytes, not as a decoded object."""
        plugin = OpenAIEmbeddingRequestModifier({
            "overrides": {"model": _CONFIGURED_MODEL}
        })
        body = json.dumps({"model": _REQUEST_MODEL, "input": "test"}).encode()
        _result_headers, result, _result_path = await _run_request(
            plugin, body, _EMBEDDINGS_PATH
        )
        assert isinstance(result, bytes)
        assert json.loads(result) == {"model": _CONFIGURED_MODEL, "input": "test"}


class TestOpenAIEmbeddingRequestModifierPathGating:
    """Overrides only apply to embedding paths."""

    async def test_chat_completions_passes_through(self) -> None:
        """A chat request keeps its own model rather than the embedding override."""
        plugin = OpenAIEmbeddingRequestModifier({
            "overrides": {"model": _OVERRIDE_MODEL}
        })
        body = json.dumps({"model": "gpt-4"}).encode()
        _result_headers, result, result_path = await _run_request(plugin, body, _CHAT_PATH)
        assert result == body
        assert result_path == _CHAT_PATH

    async def test_rerank_passes_through(self) -> None:
        """A rerank request belongs to another modifier and is left alone."""
        plugin = OpenAIEmbeddingRequestModifier({
            "overrides": {"model": _OVERRIDE_MODEL}
        })
        body = json.dumps({"model": "rerank-v1"}).encode()
        _result_headers, result, result_path = await _run_request(plugin, body, _RERANK_PATH)
        assert result == body
        assert result_path == _RERANK_PATH

    async def test_health_passes_through(self) -> None:
        """A non-API path carrying a JSON body is still not an embedding path."""
        plugin = OpenAIEmbeddingRequestModifier({
            "overrides": {"model": _OVERRIDE_MODEL}
        })
        body = json.dumps({"model": _REQUEST_MODEL}).encode()
        _result_headers, result, result_path = await _run_request(plugin, body, _HEALTH_PATH)
        assert result == body
        assert result_path == _HEALTH_PATH

    async def test_embeddings_with_query_string_applies_overrides(self) -> None:
        """A query string does not stop the path from matching."""
        plugin = OpenAIEmbeddingRequestModifier({
            "overrides": {"model": _OVERRIDE_MODEL}
        })
        body = json.dumps({"model": _REQUEST_MODEL}).encode()
        _result_headers, result, _result_path = await _run_request(
            plugin, body, f"{_EMBEDDINGS_PATH}?user=test"
        )
        assert json.loads(result) == {"model": _OVERRIDE_MODEL}


class TestOpenAIEmbeddingRequestModifierInvalidJson:
    """Non-JSON body passes through unchanged."""

    async def test_invalid_json_passes_through(self) -> None:
        """A body that is not JSON at all is forwarded byte for byte."""
        plugin = OpenAIEmbeddingRequestModifier({
            "overrides": {"model": _OVERRIDE_MODEL}
        })
        body = b"not valid json"
        _result_headers, result, result_path = await _run_request(
            plugin, body, _EMBEDDINGS_PATH
        )
        assert result == body
        assert result_path == _EMBEDDINGS_PATH

    async def test_empty_body_passes_through(self) -> None:
        """An empty body is forwarded rather than turned into the overrides."""
        plugin = OpenAIEmbeddingRequestModifier({
            "overrides": {"model": _OVERRIDE_MODEL}
        })
        _result_headers, result, result_path = await _run_request(
            plugin, b"", _EMBEDDINGS_PATH
        )
        assert result == b""
        assert result_path == _EMBEDDINGS_PATH


class TestOpenAIEmbeddingRequestModifierEmptyOverrides:
    """No overrides means body is unchanged."""

    async def test_no_overrides_passes_through(self) -> None:
        """A plugin configured with nothing to apply leaves the body alone.

        Asserted by identity: re-encoding this body would produce equal bytes,
        so only the original object proves the merge was skipped rather than
        applied and round-tripped.
        """
        plugin = OpenAIEmbeddingRequestModifier({})
        body = json.dumps({"model": _REQUEST_MODEL, "input": "test"}).encode()
        _result_headers, result, result_path = await _run_request(
            plugin, body, _EMBEDDINGS_PATH
        )
        assert result is body
        assert result_path == _EMBEDDINGS_PATH


class TestOpenAIEmbeddingRequestModifierName:
    """name() returns the configured name or default."""

    @pytest.mark.parametrize("config,expected", [
        ({"name": "my_embedding_modifier"}, "my_embedding_modifier"),
        ({}, "openai_embedding_modifier"),
    ])
    async def test_name_returns_configured_or_default(
        self, config: dict[str, str], expected: str
    ) -> None:
        """A configured name wins; without one the plugin falls back to its own."""
        plugin = OpenAIEmbeddingRequestModifier(config)
        assert plugin.name() == expected


class TestOpenAIEmbeddingRequestModifierContentLength:
    """on_request updates Content-Length header when body is modified."""

    async def test_content_length_updated_on_modification(self) -> None:
        """A rewritten body carries a Content-Length matching its new size."""
        plugin = OpenAIEmbeddingRequestModifier({
            "overrides": {"model": _OVERRIDE_MODEL, "dimensions": _OVERRIDE_DIMENSIONS}
        })
        body = json.dumps({"model": _REQUEST_MODEL}).encode()
        headers = {"Content-Type": "application/json", "Content-Length": str(len(body))}
        result_headers, result, _result_path = await _run_request(
            plugin, body, _EMBEDDINGS_PATH, headers
        )
        assert result_headers["Content-Length"] == str(len(result))
        assert result_headers["Content-Type"] == "application/json"

    async def test_content_length_not_updated_on_pass_through(self) -> None:
        """A body left alone keeps the client's own Content-Length."""
        plugin = OpenAIEmbeddingRequestModifier({
            "overrides": {"model": _OVERRIDE_MODEL}
        })
        body = b"not json"
        headers = {"Content-Length": str(len(body))}
        result_headers, result, result_path = await _run_request(
            plugin, body, _EMBEDDINGS_PATH, headers
        )
        assert result_headers["Content-Length"] == str(len(body))
        assert result is body
        assert result_path == _EMBEDDINGS_PATH


class TestOpenAIEmbeddingRequestModifierDeepMerge:
    """deep_merge works correctly for nested embedding request fields."""

    async def test_nested_user_field_merge(self) -> None:
        """Nested overrides replace their own keys and keep the request's siblings."""
        plugin = OpenAIEmbeddingRequestModifier({
            "overrides": {
                "model": _OVERRIDE_MODEL,
                "user": {"session_id": "abc123", "tier": "pro"},
            }
        })
        body = json.dumps({
            "model": _REQUEST_MODEL,
            "input": "hello",
            "user": {"session_id": "old_session", "metadata": "keep"},
        }).encode()
        _result_headers, result, _result_path = await _run_request(
            plugin, body, _EMBEDDINGS_PATH
        )
        result_data = json.loads(result)
        assert result_data["model"] == _OVERRIDE_MODEL
        assert result_data["user"]["session_id"] == "abc123"
        assert result_data["user"]["tier"] == "pro"
        assert result_data["user"]["metadata"] == "keep"


class TestOpenAIEmbeddingRequestModifierNonDictJson:
    """Non-dict JSON bodies pass through unchanged."""

    @pytest.mark.parametrize("body", [
        b'[1,2,3]',
        b'"string"',
        b'42',
        b'true',
        b'null',
    ])
    async def test_non_dict_json_passes_through(self, body: bytes) -> None:
        """Valid JSON that is not an object has no keys to merge overrides into."""
        plugin = OpenAIEmbeddingRequestModifier({"overrides": {"model": "new"}})
        _result_headers, result, result_path = await _run_request(
            plugin, body, _EMBEDDINGS_PATH
        )
        assert result == body
        assert result_path == _EMBEDDINGS_PATH


class TestOngoingRequestLifecycle:
    """Requests are tracked as ongoing until response completes or is cancelled."""

    async def test_on_request_adds_to_ongoing(self) -> None:
        """Handling a request marks it ongoing until something ends it."""
        plugin = OpenAIEmbeddingRequestModifier({"overrides": {}})
        body = b'{"model": "text-embedding-3-small", "input": "hello"}'
        await _run_request(plugin, body, _EMBEDDINGS_PATH)
        assert plugin.is_request_ongoing(_REQUEST_ID) is True

    async def test_on_response_chunk_final_removes_from_ongoing(self) -> None:
        """The final response chunk clears the request from the ongoing set."""
        plugin = OpenAIEmbeddingRequestModifier({"overrides": {}})
        body = b'{"model": "text-embedding-3-small", "input": "hello"}'
        await _run_request(plugin, body, _EMBEDDINGS_PATH)
        await plugin.on_response_chunk(
            _REQUEST_ID, {}, b"", _EMBEDDINGS_PATH, 200, is_final=True
        )
        assert plugin.is_request_ongoing(_REQUEST_ID) is False

    async def test_on_cancelled_removes_from_ongoing(self) -> None:
        """A cancelled request stops being ongoing even with no response at all."""
        plugin = OpenAIEmbeddingRequestModifier({"overrides": {}})
        body = b'{"model": "text-embedding-3-small", "input": "hello"}'
        await _run_request(plugin, body, _EMBEDDINGS_PATH)
        plugin.on_cancelled(_REQUEST_ID)
        assert plugin.is_request_ongoing(_REQUEST_ID) is False
