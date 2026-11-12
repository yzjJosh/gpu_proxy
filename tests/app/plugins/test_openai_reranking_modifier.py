"""Tests for OpenAIRerankingRequestModifier plugin."""

import json
import pytest

from app.plugins.openai_reranking_modifier import OpenAIRerankingRequestModifier

# The two paths the plugin treats as reranking endpoints.
_RERANK_PATH = "/v1/rerank"
_RERANKING_PATH = "/reranking"
# Paths that belong to other plugins and must be left alone.
_CHAT_PATH = "/v1/chat/completions"
_EMBEDDINGS_PATH = "/v1/embeddings"
_HEALTH_PATH = "/health"

# The request id the plugin tracks a request under; the value itself is opaque.
_REQUEST_ID = 1

# Models named in the config overrides, distinct from the one the request carries
# so an override that is silently dropped cannot pass as a success.
_OVERRIDE_MODEL = "rerank-english-v3.0"
_MULTILINGUAL_MODEL = "rerank-multilingual-v3.0"
_CONFIGURED_MODEL = "Qwen3-Reranker-0.6B"
_SHORT_OVERRIDE_MODEL = "rerank-v1"
# The model the incoming request body names, which an override has to displace.
_REQUEST_MODEL = "old"
# Fields that exist only in the overrides, so the merge has to introduce them.
_OVERRIDE_TOP_N = 5
_OVERRIDE_MAX_CHUNKS = 10


async def _run_request(
    plugin: OpenAIRerankingRequestModifier,
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


class TestOpenAIRerankingRequestModifierAppliesOverrides:
    """Valid JSON body gets overrides merged on reranking paths."""

    async def test_rerank_path_applies_overrides(self) -> None:
        """Every configured override lands, and untouched request fields survive."""
        plugin = OpenAIRerankingRequestModifier({
            "overrides": {"model": _OVERRIDE_MODEL, "top_n": _OVERRIDE_TOP_N}
        })
        body = json.dumps({
            "model": _REQUEST_MODEL,
            "query": "search term",
            "documents": ["doc1"],
        }).encode()
        _result_headers, result, result_path = await _run_request(plugin, body, _RERANK_PATH)
        result_data = json.loads(result)
        assert result_data == {
            "model": _OVERRIDE_MODEL,
            "top_n": _OVERRIDE_TOP_N,
            "query": "search term",
            "documents": ["doc1"],
        }
        assert result_path == _RERANK_PATH

    async def test_reranking_path_applies_overrides(self) -> None:
        """The bare /reranking spelling is a reranking path too."""
        plugin = OpenAIRerankingRequestModifier({
            "overrides": {"model": _MULTILINGUAL_MODEL}
        })
        body = json.dumps({
            "model": _REQUEST_MODEL,
            "query": "test",
            "documents": ["a", "b"],
        }).encode()
        _result_headers, result, result_path = await _run_request(
            plugin, body, _RERANKING_PATH
        )
        assert json.loads(result)["model"] == _MULTILINGUAL_MODEL
        assert result_path == _RERANKING_PATH


class TestOpenAIRerankingRequestModifierReturnsModifiedBytes:
    """on_request returns modified bytes directly."""

    async def test_on_request_returns_modified_bytes(self) -> None:
        """The modified body comes back as bytes, not as a decoded object."""
        plugin = OpenAIRerankingRequestModifier({
            "overrides": {"model": _CONFIGURED_MODEL}
        })
        body = json.dumps({
            "model": _REQUEST_MODEL,
            "query": "test",
            "documents": ["doc1"],
        }).encode()
        _result_headers, result, _result_path = await _run_request(
            plugin, body, _RERANK_PATH
        )
        assert isinstance(result, bytes)
        assert json.loads(result) == {
            "model": _CONFIGURED_MODEL,
            "query": "test",
            "documents": ["doc1"],
        }


class TestOpenAIRerankingRequestModifierPathGating:
    """Overrides only apply to reranking paths."""

    async def test_chat_completions_passes_through(self) -> None:
        """A chat request keeps its own model rather than the reranking override."""
        plugin = OpenAIRerankingRequestModifier({
            "overrides": {"model": _SHORT_OVERRIDE_MODEL}
        })
        body = json.dumps({"model": "gpt-4"}).encode()
        _result_headers, result, result_path = await _run_request(plugin, body, _CHAT_PATH)
        assert result == body
        assert result_path == _CHAT_PATH

    async def test_embeddings_passes_through(self) -> None:
        """An embeddings request belongs to another modifier and is left alone."""
        plugin = OpenAIRerankingRequestModifier({
            "overrides": {"model": _SHORT_OVERRIDE_MODEL}
        })
        body = json.dumps({"model": "text-embedding-3-small"}).encode()
        _result_headers, result, result_path = await _run_request(
            plugin, body, _EMBEDDINGS_PATH
        )
        assert result == body
        assert result_path == _EMBEDDINGS_PATH

    async def test_health_passes_through(self) -> None:
        """A non-API path carrying a JSON body is still not a reranking path."""
        plugin = OpenAIRerankingRequestModifier({
            "overrides": {"model": _SHORT_OVERRIDE_MODEL}
        })
        body = json.dumps({"model": _REQUEST_MODEL}).encode()
        _result_headers, result, result_path = await _run_request(plugin, body, _HEALTH_PATH)
        assert result == body
        assert result_path == _HEALTH_PATH

    async def test_rerank_with_query_string_applies_overrides(self) -> None:
        """A query string does not stop the path from matching."""
        plugin = OpenAIRerankingRequestModifier({
            "overrides": {"model": _SHORT_OVERRIDE_MODEL}
        })
        body = json.dumps({"model": _REQUEST_MODEL}).encode()
        _result_headers, result, _result_path = await _run_request(
            plugin, body, f"{_RERANK_PATH}?top_n=3"
        )
        assert json.loads(result) == {"model": _SHORT_OVERRIDE_MODEL}


class TestOpenAIRerankingRequestModifierInvalidJson:
    """Non-JSON body passes through unchanged."""

    async def test_invalid_json_passes_through(self) -> None:
        """A body that is not JSON at all is forwarded byte for byte."""
        plugin = OpenAIRerankingRequestModifier({
            "overrides": {"model": _SHORT_OVERRIDE_MODEL}
        })
        body = b"not valid json"
        _result_headers, result, result_path = await _run_request(plugin, body, _RERANK_PATH)
        assert result == body
        assert result_path == _RERANK_PATH

    async def test_empty_body_passes_through(self) -> None:
        """An empty body is forwarded rather than turned into the overrides."""
        plugin = OpenAIRerankingRequestModifier({
            "overrides": {"model": _SHORT_OVERRIDE_MODEL}
        })
        _result_headers, result, result_path = await _run_request(plugin, b"", _RERANK_PATH)
        assert result == b""
        assert result_path == _RERANK_PATH


class TestOpenAIRerankingRequestModifierEmptyOverrides:
    """No overrides means body is unchanged."""

    async def test_no_overrides_passes_through(self) -> None:
        """A plugin configured with nothing to apply leaves the body alone.

        Asserted by identity: re-encoding this body would produce equal bytes,
        so only the original object proves the merge was skipped rather than
        applied and round-tripped.
        """
        plugin = OpenAIRerankingRequestModifier({})
        body = json.dumps({"model": _REQUEST_MODEL, "query": "test"}).encode()
        _result_headers, result, result_path = await _run_request(plugin, body, _RERANK_PATH)
        assert result is body
        assert result_path == _RERANK_PATH


class TestOpenAIRerankingRequestModifierName:
    """name() returns the configured name or default."""

    @pytest.mark.parametrize("config,expected", [
        ({"name": "my_rerank_modifier"}, "my_rerank_modifier"),
        ({}, "openai_reranking_modifier"),
    ])
    async def test_name_returns_configured_or_default(
        self, config: dict[str, str], expected: str
    ) -> None:
        """A configured name wins; without one the plugin falls back to its own."""
        plugin = OpenAIRerankingRequestModifier(config)
        assert plugin.name() == expected


class TestOpenAIRerankingRequestModifierContentLength:
    """on_request updates Content-Length header when body is modified."""

    async def test_content_length_updated_on_modification(self) -> None:
        """A rewritten body carries a Content-Length matching its new size."""
        plugin = OpenAIRerankingRequestModifier({
            "overrides": {
                "model": _OVERRIDE_MODEL,
                "max_chunks_per_doc": _OVERRIDE_MAX_CHUNKS,
            }
        })
        body = json.dumps({"model": _REQUEST_MODEL}).encode()
        headers = {"Content-Type": "application/json", "Content-Length": str(len(body))}
        result_headers, result, _result_path = await _run_request(
            plugin, body, _RERANK_PATH, headers
        )
        assert result_headers["Content-Length"] == str(len(result))
        assert result_headers["Content-Type"] == "application/json"

    async def test_content_length_not_updated_on_pass_through(self) -> None:
        """A body left alone keeps the client's own Content-Length."""
        plugin = OpenAIRerankingRequestModifier({
            "overrides": {"model": _SHORT_OVERRIDE_MODEL}
        })
        body = b"not json"
        headers = {"Content-Length": str(len(body))}
        result_headers, result, result_path = await _run_request(
            plugin, body, _RERANK_PATH, headers
        )
        assert result_headers["Content-Length"] == str(len(body))
        assert result is body
        assert result_path == _RERANK_PATH


class TestOpenAIRerankingRequestModifierDeepMerge:
    """deep_merge works correctly for nested reranking request fields."""

    async def test_documents_list_merge(self) -> None:
        """A list override merges element-wise into the request's own elements.

        The request element carries a key the override does not, so replacing
        the element wholesale instead of merging into it would drop that key.
        """
        plugin = OpenAIRerankingRequestModifier({
            "overrides": {
                "model": "rerank-v3",
                "return_documents": True,
                "rank_fields": [{"field": "title", "weight": 2}],
            }
        })
        body = json.dumps({
            "model": _REQUEST_MODEL,
            "query": "search",
            "documents": [{"text": "doc1"}, {"text": "doc2"}],
            "rank_fields": [{"field": "body", "boost": 1}],
        }).encode()
        _result_headers, result, _result_path = await _run_request(plugin, body, _RERANK_PATH)
        result_data = json.loads(result)
        assert result_data["model"] == "rerank-v3"
        assert result_data["return_documents"] is True
        assert result_data["rank_fields"][0] == {
            "field": "title",
            "boost": 1,
            "weight": 2,
        }
        assert len(result_data["documents"]) == 2


class TestOpenAIRerankingRequestModifierNonDictJson:
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
        plugin = OpenAIRerankingRequestModifier({"overrides": {"model": "new"}})
        _result_headers, result, result_path = await _run_request(plugin, body, _RERANK_PATH)
        assert result == body
        assert result_path == _RERANK_PATH


class TestOngoingRequestLifecycle:
    """OpenAIRerankingRequestModifier tracks ongoing requests via base class lifecycle hooks."""

    async def test_on_request_adds_to_ongoing(self) -> None:
        """on_request adds the request_id to ongoing tracking."""
        plugin = OpenAIRerankingRequestModifier({})
        body = b'{"model": "rerank-v3", "query": "search"}'
        await _run_request(plugin, body, _RERANK_PATH)
        assert plugin.is_request_ongoing(_REQUEST_ID)

    async def test_on_response_chunk_final_removes_from_ongoing(self) -> None:
        """A final response chunk removes the request_id from ongoing tracking."""
        plugin = OpenAIRerankingRequestModifier({})
        body = b'{"model": "rerank-v3", "query": "search"}'
        await _run_request(plugin, body, _RERANK_PATH)
        await plugin.on_response_chunk(
            _REQUEST_ID, {}, b"", _RERANK_PATH, 200, is_final=True
        )
        assert not plugin.is_request_ongoing(_REQUEST_ID)

    async def test_on_cancelled_removes_from_ongoing(self) -> None:
        """on_cancelled removes the request_id from ongoing tracking."""
        plugin = OpenAIRerankingRequestModifier({})
        body = b'{"model": "rerank-v3", "query": "search"}'
        await _run_request(plugin, body, _RERANK_PATH)
        plugin.on_cancelled(_REQUEST_ID)
        assert not plugin.is_request_ongoing(_REQUEST_ID)
