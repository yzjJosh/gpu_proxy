"""Tests for PluginBase abstract class."""

from typing import Callable, NamedTuple, override

import pytest
from aiohttp import web

from app.config import ServiceConfig
from app.plugins.base import PluginBase, forward_handler, make_forward_guard
from app.responses import AbstractMutatingResponse
from tests.app.conftest import CancelTrackingPlugin

_REQUEST_ID = 1
_OTHER_REQUEST_ID = 2
_UNKNOWN_REQUEST_ID = 99
_CANCELLED_REQUEST_ID = 42

_PATH = "/v1/chat/completions"
_OTHER_PATH = "/v1/embeddings"
_OK = 200
_SERVER_ERROR = 500

_Publications = list[tuple[str, str]]


class _ChunkArgs(NamedTuple):
    """Every argument one on_response_chunk call was given."""

    request_id: int
    headers: dict[str, str]
    chunk: bytes
    path: str
    status_code: int
    is_final: bool


class ConcretePlugin(PluginBase):
    """A plugin that only implements name(), inheriting every other hook.

    Instantiating this class proves the remaining hooks are concrete on the
    base, since a concrete subclass is required to exercise them.
    """

    @override
    def name(self) -> str:
        return "concrete"


class _NamelessPlugin(PluginBase):
    """A subclass that leaves PluginBase's one abstract method unimplemented."""


class _TransformingPlugin(PluginBase):
    """Rewrites the request body the base hook returns, and restates its length."""

    @override
    def name(self) -> str:
        return "transforming"

    @override
    async def on_request(
        self, request_id: int, headers: dict[str, str], body: bytes, path: str
    ) -> tuple[dict[str, str], bytes, str]:
        headers, body, path = await super().on_request(request_id, headers, body, path)
        new_body = body + b'"extra": true'
        headers["Content-Length"] = str(len(new_body))
        return headers, new_body, path


class _ModifyingPlugin(PluginBase):
    """Upper-cases every response chunk the base hook returns."""

    @override
    def name(self) -> str:
        return "modifying"

    @override
    async def on_response_chunk(
        self,
        request_id: int,
        headers: dict[str, str],
        chunk: bytes,
        path: str,
        status_code: int,
        is_final: bool,
    ) -> bytes:
        return (
            await super().on_response_chunk(
                request_id, headers, chunk, path, status_code, is_final
            )
        ).upper()


class _PassthroughPlugin(PluginBase):
    """Overrides the chunk hook but hands back exactly what the base returns."""

    @override
    def name(self) -> str:
        return "passthrough"

    @override
    async def on_response_chunk(
        self,
        request_id: int,
        headers: dict[str, str],
        chunk: bytes,
        path: str,
        status_code: int,
        is_final: bool,
    ) -> bytes:
        return await super().on_response_chunk(
            request_id, headers, chunk, path, status_code, is_final
        )


class _CapturingPlugin(PluginBase):
    """Records the arguments of every chunk call, then defers to the base hook."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.calls: list[_ChunkArgs] = []

    @override
    def name(self) -> str:
        return "capturing"

    @override
    async def on_response_chunk(
        self,
        request_id: int,
        headers: dict[str, str],
        chunk: bytes,
        path: str,
        status_code: int,
        is_final: bool,
    ) -> bytes:
        self.calls.append(
            _ChunkArgs(request_id, headers, chunk, path, status_code, is_final)
        )
        return await super().on_response_chunk(
            request_id, headers, chunk, path, status_code, is_final
        )


_VRAM_BUDGET_MIB = 1024


def _service_config() -> ServiceConfig:
    """A minimal service config a plugin can be built against."""
    return ServiceConfig(
        name="test-svc",
        priority=1,
        vram_budget=_VRAM_BUDGET_MIB,
        hostname="test-svc.local",
        target="http://127.0.0.1:9000/test-svc",
        container="test-svc",
    )


def _no_publish(_title: str, _html: str) -> None:
    """Stand in for the callback a plugin has not been handed yet."""


class _UIPlugin(PluginBase):
    """Keeps the publish callback it was handed, so a test can call it back."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.captured_publish: Callable[[str, str], None] = _no_publish

    @override
    def name(self) -> str:
        return "ui_plugin"

    @override
    def init_ui(self, publish: Callable[[str, str], None]) -> None:
        self.captured_publish = publish


class _ForwardRecorder:
    """A forward function that records whether the guard let it run."""

    def __init__(self) -> None:
        self.called = False

    async def __call__(
        self,
        _headers: dict[str, str],
        _body: bytes,
        _path: str,
        _request: web.Request,
        _response: AbstractMutatingResponse,
    ) -> bool:
        self.called = True
        return True


def _recorder() -> tuple[Callable[[str, str], None], _Publications]:
    """A publish callback and the list it appends every publication to."""
    published: _Publications = []

    def publish(title: str, html: str) -> None:
        published.append((title, html))

    return publish, published


class TestPluginBaseAbstract:
    """Test PluginBase abstract class behavior."""

    def test_plugin_base_is_abstract(self) -> None:
        """A subclass that never implements name() cannot be instantiated either."""
        with pytest.raises(TypeError, match="name"):
            _NamelessPlugin({})

    def test_cannot_instantiate_plugin_base_directly(self) -> None:
        """Test direct instantiation of PluginBase raises TypeError."""
        with pytest.raises(TypeError):
            PluginBase({})


class TestConcretePlugin:
    """Test concrete subclass behavior."""

    def test_concrete_plugin_instantiates_with_config(self) -> None:
        """A plugin keeps the config it was built with, copied rather than aliased."""
        config = {"key": "value"}
        plugin = ConcretePlugin(config)
        assert plugin.config == config
        assert plugin.config is not config

    @pytest.mark.parametrize(
        "config",
        [
            {},
            {"host": "localhost", "port": 8080},
            {"enabled": True, "timeout": 30},
        ],
    )
    def test_config_stored_on_instance(self, config: dict[str, str | int | bool]) -> None:
        """Test config dict is stored on the plugin instance."""
        plugin = ConcretePlugin(config)
        assert plugin.config == config

    def test_service_config_stored_on_instance(self) -> None:
        """A plugin keeps the service config it was built with, by identity."""
        service = _service_config()
        plugin = ConcretePlugin({}, service)
        assert plugin.service_config is service

    def test_service_config_defaults_to_none(self) -> None:
        """A plugin built without a service config exposes none."""
        plugin = ConcretePlugin({})
        assert plugin.service_config is None

    async def test_on_request_passes_through_unchanged(self) -> None:
        """Test on_request returns headers and body unchanged."""
        headers = {"Content-Type": "application/json"}
        body = b'{"test": true}'
        plugin = ConcretePlugin({})
        result_headers, result_body, result_path = await plugin.on_request(
            _REQUEST_ID, headers, body, "/test"
        )
        assert result_headers is headers
        assert result_body is body
        assert result_path == "/test"

    async def test_on_request_can_transform_body_and_headers(self) -> None:
        """Test on_request can transform both body and headers."""
        plugin = _TransformingPlugin({})
        headers, result, result_path = await plugin.on_request(
            _REQUEST_ID, {"Content-Length": "17"}, b'{"test": true, ', "/api/v1"
        )
        assert result == b'{"test": true, "extra": true'
        assert headers["Content-Length"] == "28"
        assert result_path == "/api/v1"


class TestDefaultHookBehavior:
    """Test on_response_chunk and init_ui hooks on PluginBase."""

    async def test_on_response_chunk_exists_and_is_not_abstract(self) -> None:
        """A plugin that overrides only on_request still inherits the chunk hook."""
        plugin = _TransformingPlugin({})
        assert await plugin.on_response_chunk(
            _REQUEST_ID, {}, b"chunk", _PATH, _OK, is_final=False
        ) == b"chunk"

    async def test_on_response_chunk_signature(self) -> None:
        """The base hook takes every chunk argument by keyword, and by no other name."""
        plugin = ConcretePlugin({})
        await plugin.on_request(_REQUEST_ID, {}, b"", _PATH)
        returned = await plugin.on_response_chunk(
            request_id=_REQUEST_ID,
            _headers={"content-type": "application/json"},
            chunk=b"body",
            _path=_PATH,
            _status_code=_OK,
            is_final=True,
        )
        assert returned == b"body"
        assert not plugin.is_request_ongoing(_REQUEST_ID)
        with pytest.raises(TypeError, match="unexpected keyword argument 'req_id'"):
            plugin.on_response_chunk(req_id=_REQUEST_ID)

    async def test_on_response_chunk_defaults_to_passthrough(self) -> None:
        """Test on_response_chunk returns chunk unchanged by default."""
        plugin = ConcretePlugin({})
        result = await plugin.on_response_chunk(
            request_id=_REQUEST_ID,
            _headers={"content-type": "text/plain"},
            chunk=b"hello",
            _path=_PATH,
            _status_code=_OK,
            is_final=False,
        )
        assert result == b"hello"

    async def test_on_response_chunk_can_return_modified_chunk(self) -> None:
        """Test on_response_chunk can return modified bytes to override the response."""
        plugin = _ModifyingPlugin({})
        result = await plugin.on_response_chunk(
            _REQUEST_ID, {}, b"hello world", _PATH, _OK, False
        )
        assert result == b"HELLO WORLD"

    async def test_on_response_chunk_passthrough_keeps_original(self) -> None:
        """Test returning chunk unchanged passes it through."""
        plugin = _PassthroughPlugin({})
        result = await plugin.on_response_chunk(
            _REQUEST_ID, {}, b"original", _PATH, _OK, True
        )
        assert result == b"original"

    def test_init_ui_exists_and_is_not_abstract(self) -> None:
        """A plugin that overrides only the chunk hook still inherits init_ui."""
        plugin = _ModifyingPlugin({})
        publish, published = _recorder()
        plugin.init_ui(publish)
        assert published == []

    def test_init_ui_signature(self) -> None:
        """The base init_ui takes its callback as _publish, and by no other name."""
        plugin = ConcretePlugin({})
        publish, published = _recorder()
        plugin.init_ui(_publish=publish)
        assert published == []
        with pytest.raises(TypeError, match="unexpected keyword argument 'publish'"):
            plugin.init_ui(publish=publish)

    def test_init_ui_defaults_to_noop(self) -> None:
        """The default init_ui publishes nothing at all, rather than an empty panel."""
        plugin = ConcretePlugin({})
        publish, published = _recorder()
        assert plugin.init_ui(publish) is None
        assert published == []

    async def test_prepare_target_defaults_to_noop(self) -> None:
        """The default prepare_target does nothing, so it is awaitable and returns None."""
        plugin = ConcretePlugin({})
        assert await plugin.prepare_target(_REQUEST_ID, _PATH) is None

    async def test_prepare_target_does_not_track_requests(self) -> None:
        """prepare_target is not the request entry point, so it tracks no request."""
        plugin = ConcretePlugin({})
        await plugin.prepare_target(_REQUEST_ID, _PATH)
        assert not plugin.is_request_ongoing(_REQUEST_ID)


class TestOnResponseChunkOverride:
    """Test subclass override of on_response_chunk hook."""

    @pytest.mark.parametrize(
        ("chunk", "is_final", "status_code"),
        [
            (b'{"text": "hello"}', False, 200),
            (b"", True, 200),
            (b"error payload", False, 500),
        ],
    )
    async def test_override_receives_all_arguments(
        self, chunk: bytes, is_final: bool, status_code: int
    ) -> None:
        """Test overridden on_response_chunk receives all arguments correctly."""
        plugin = _CapturingPlugin({})
        headers = {"content-type": "application/json", "x-custom": "value"}
        await plugin.on_response_chunk(
            _REQUEST_ID,
            headers=headers,
            chunk=chunk,
            path=_PATH,
            status_code=status_code,
            is_final=is_final,
        )
        assert plugin.calls == [
            _ChunkArgs(_REQUEST_ID, headers, chunk, _PATH, status_code, is_final)
        ]
        assert plugin.calls[0].headers is headers

    async def test_empty_chunk_with_is_final_true(self) -> None:
        """Test handling of empty chunk as finalization signal."""
        plugin = _CapturingPlugin({})
        await plugin.on_response_chunk(
            _REQUEST_ID,
            headers={},
            chunk=b"",
            path="/api/v1",
            status_code=_OK,
            is_final=True,
        )
        assert len(plugin.calls) == 1
        assert plugin.calls[0].chunk == b""
        assert plugin.calls[0].is_final is True

    async def test_large_status_code_passed_through(self) -> None:
        """Test server error status codes are passed through unchanged."""
        plugin = _CapturingPlugin({})
        await plugin.on_response_chunk(
            _REQUEST_ID,
            headers={"content-type": "text/plain"},
            chunk=b"Internal Server Error",
            path=_OTHER_PATH,
            status_code=_SERVER_ERROR,
            is_final=False,
        )
        assert plugin.calls[0].status_code == _SERVER_ERROR

    async def test_headers_dict_passed_through(self) -> None:
        """Test response headers dict is passed through to override."""
        plugin = _CapturingPlugin({})
        headers = {
            "content-type": "text/event-stream",
            "cache-control": "no-cache",
            "x-request-id": "abc-123",
        }
        await plugin.on_response_chunk(
            _REQUEST_ID,
            headers=headers,
            chunk=b"data: test\n",
            path=_PATH,
            status_code=_OK,
            is_final=False,
        )
        assert plugin.calls[0].headers == headers


class TestInitUIOverride:
    """Test subclass override of init_ui hook."""

    def test_init_ui_override_receives_callable(self) -> None:
        """Test overridden init_ui receives a callable publish function."""
        plugin = _UIPlugin({})
        publish, _ = _recorder()
        plugin.init_ui(publish=publish)
        assert plugin.captured_publish is publish

    def test_init_ui_publish_callback_invokes_without_error(self) -> None:
        """The callback an override captured is the live one: invoking it publishes."""
        plugin = _UIPlugin({})
        publish, published = _recorder()
        plugin.init_ui(publish)
        plugin.captured_publish("Dashboard", "<div>stats</div>")
        assert published == [("Dashboard", "<div>stats</div>")]


class TestOngoingRequestTracking:
    """Test ongoing-request tracking in PluginBase."""

    async def test_on_request_adds_to_ongoing(self) -> None:
        """on_request marks the request ongoing."""
        plugin = ConcretePlugin({})
        await plugin.on_request(_REQUEST_ID, {}, b"", "/test")
        assert plugin.is_request_ongoing(_REQUEST_ID)

    def test_on_request_noop_without_request(self) -> None:
        """is_request_ongoing returns False for unknown request_id."""
        plugin = ConcretePlugin({})
        assert not plugin.is_request_ongoing(_UNKNOWN_REQUEST_ID)

    async def test_on_response_chunk_final_removes_from_ongoing(self) -> None:
        """on_response_chunk with is_final=True ends the request."""
        plugin = ConcretePlugin({})
        await plugin.on_request(_REQUEST_ID, {}, b"", "/test")
        await plugin.on_response_chunk(_REQUEST_ID, {}, b"chunk", "/test", _OK, is_final=True)
        assert not plugin.is_request_ongoing(_REQUEST_ID)

    async def test_on_response_chunk_non_final_keeps_ongoing(self) -> None:
        """on_response_chunk with is_final=False keeps request ongoing."""
        plugin = ConcretePlugin({})
        await plugin.on_request(_REQUEST_ID, {}, b"", "/test")
        await plugin.on_response_chunk(_REQUEST_ID, {}, b"chunk", "/test", _OK, is_final=False)
        assert plugin.is_request_ongoing(_REQUEST_ID)

    async def test_multiple_requests_tracked_independently(self) -> None:
        """Multiple requests are tracked independently."""
        plugin = ConcretePlugin({})
        await plugin.on_request(_REQUEST_ID, {}, b"", "/a")
        await plugin.on_request(_OTHER_REQUEST_ID, {}, b"", "/b")
        assert plugin.is_request_ongoing(_REQUEST_ID)
        assert plugin.is_request_ongoing(_OTHER_REQUEST_ID)
        await plugin.on_response_chunk(_REQUEST_ID, {}, b"", "/a", _OK, is_final=True)
        assert not plugin.is_request_ongoing(_REQUEST_ID)
        assert plugin.is_request_ongoing(_OTHER_REQUEST_ID)


class TestOnCancelledHook:
    """Test on_cancelled hook on PluginBase."""

    async def test_on_cancelled_cleans_up_ongoing_requests(self) -> None:
        """on_cancelled ends the request it names."""
        plugin = ConcretePlugin({})
        await plugin.on_request(_REQUEST_ID, {}, b"", "/test")
        assert plugin.is_request_ongoing(_REQUEST_ID)
        plugin.on_cancelled(_REQUEST_ID)
        assert not plugin.is_request_ongoing(_REQUEST_ID)

    async def test_on_cancelled_not_abstract(self) -> None:
        """A plugin that overrides only the chunk hook still inherits on_cancelled."""
        plugin = _PassthroughPlugin({})
        await plugin.on_request(_REQUEST_ID, {}, b"", _PATH)
        plugin.on_cancelled(_REQUEST_ID)
        assert not plugin.is_request_ongoing(_REQUEST_ID)

    async def test_on_cancelled_signature(self) -> None:
        """The cancelled request is named request_id, and by no other name."""
        plugin = ConcretePlugin({})
        await plugin.on_request(_REQUEST_ID, {}, b"", "/test")
        plugin.on_cancelled(request_id=_REQUEST_ID)
        assert not plugin.is_request_ongoing(_REQUEST_ID)
        with pytest.raises(TypeError, match="unexpected keyword argument 'rid'"):
            plugin.on_cancelled(rid=_REQUEST_ID)

    def test_plugin_can_override_on_cancelled(self) -> None:
        """Plugins can override on_cancelled to track cancellations."""
        plugin = CancelTrackingPlugin({})
        plugin.on_cancelled(_CANCELLED_REQUEST_ID)
        assert plugin.cancelled_requests == [_CANCELLED_REQUEST_ID]

    async def test_on_cancelled_does_not_affect_other_requests(self) -> None:
        """on_cancelled only removes the cancelled request_id."""
        plugin = ConcretePlugin({})
        await plugin.on_request(_REQUEST_ID, {}, b"", "/a")
        await plugin.on_request(_OTHER_REQUEST_ID, {}, b"", "/b")
        plugin.on_cancelled(_REQUEST_ID)
        assert not plugin.is_request_ongoing(_REQUEST_ID)
        assert plugin.is_request_ongoing(_OTHER_REQUEST_ID)

    async def test_on_cancelled_on_non_ongoing_request_is_safe(self) -> None:
        """Cancelling a request that was never started changes nothing."""
        plugin = ConcretePlugin({})
        await plugin.on_request(_REQUEST_ID, {}, b"", "/a")
        plugin.on_cancelled(_UNKNOWN_REQUEST_ID)
        assert not plugin.is_request_ongoing(_UNKNOWN_REQUEST_ID)
        assert plugin.is_request_ongoing(_REQUEST_ID)


class TestPluginBaseDefaults:
    """Test the hooks PluginBase implements so subclasses need not."""

    def test_default_overridden_paths_returns_empty(self) -> None:
        """A plugin that claims no paths overrides none."""
        assert ConcretePlugin({}).overridden_paths() == []

    async def test_default_handle_overridden_request_returns_false(self) -> None:
        """A plugin that handles no path declines the request it is offered."""
        plugin = ConcretePlugin({})
        handled = await plugin.handle_overridden_request(
            _REQUEST_ID, {}, b"", "/test", None, None, None)
        assert handled is False


class TestForwardGuard:
    """Test the guard that confines forward() to an override handler."""

    async def test_forward_guard_raises_outside_handler(self) -> None:
        """Called outside a handler, the guard raises and never runs the function.

        Outside covers both before any handler has run and after one has
        finished, so a handler that ended must leave the guard closed again.
        """
        recorder = _ForwardRecorder()
        guarded = make_forward_guard(recorder)
        with pytest.raises(RuntimeError, match=r"forward\(\) can only be called"):
            await guarded({}, b"", "/test", None, None)
        with forward_handler():
            pass
        with pytest.raises(RuntimeError, match=r"forward\(\) can only be called"):
            await guarded({}, b"", "/test", None, None)
        assert recorder.called is False

    async def test_forward_guard_allows_within_handler(self) -> None:
        """Called inside a handler, the guard runs the function and returns its result."""
        recorder = _ForwardRecorder()
        guarded = make_forward_guard(recorder)
        with forward_handler():
            result = await guarded({}, b"", "/test", None, None)
        assert result is True
        assert recorder.called is True
