"""Tests for PluginManager discovery, instantiation, and pipeline."""

import inspect
from pathlib import Path
from typing import Callable, Optional

import pytest

from app.config import ConfigError, ProxyConfig, ServiceConfig
from app.plugins.base import PluginBase
from app.plugins.manager import PluginManager
from tests.app.conftest import (
    AsyncSleepPlugin,
    BadChunkPlugin,
    BodyAppenderPlugin,
    CancelTrackingPlugin,
    ChunkAppender,
    ChunkRequestIdRecorder,
    ChunkCrasher,
    ChunkEchoPlugin,
    ChunkModifier,
    ChunkReplacer,
    ChunkRecordingPlugin,
    CrashingCancelPlugin,
    EchoPlugin,
    FailingPlugin,
    GoodChunkPlugin,
    HeaderPlugin,
    NoUIPlugin,
    PathConflictPlugin,
    OverridePlugin,
    PrepareTargetRecorder,
    RequestIdCollector,
    RequestIdRecorder,
    ScopedPlugin,
    UIPublisherPlugin,
)

# The service `_make_config` builds when a test passes `service_plugins`.
_SERVICE = "test-svc"
# The service the tests that build their own `services` dict use.
_OTHER_SERVICE = "svc1"

_VRAM_BUDGET_MIB = 1024
_OK = 200
_PATH = "/"
_CHUNK = b"original"
_JSON_HEADERS = {"Content-Type": "application/json"}

_PluginSpec = dict[str, str | dict]
_Types = list[type[PluginBase]]


def _write_plugin_file(tmp_path: Path, filename: str, source: str) -> None:
    """Write a Python file into the directory the manager will scan."""
    (tmp_path / filename).write_text(source)


def _plugin_source(*class_names: str) -> str:
    """Source for a plugin module declaring one PluginBase subclass per name."""
    header = "from app.plugins.base import PluginBase\n"
    bodies = [
        f"\n\nclass {name}(PluginBase):\n    def name(self):\n        return {name!r}\n"
        for name in class_names
    ]
    return header + "".join(bodies)


def _service(name: str, plugins: Optional[list[str]] = None) -> ServiceConfig:
    """A service with the named plugin instances enabled on it."""
    return ServiceConfig(
        name=name,
        priority=1,
        vram_budget=_VRAM_BUDGET_MIB,
        hostname=f"{name}.local",
        target=f"http://127.0.0.1:9000/{name}",
        container=name,
        plugins=plugins if plugins is not None else [],
    )


def _spec(name: str, plugin_type: str, **config: str | int | bool) -> _PluginSpec:
    """One entry of the top-level `plugins:` list, naming a type and its config."""
    return {"name": name, "type": plugin_type, "config": dict(config)}


def _make_config(
    plugins: Optional[list[_PluginSpec]] = None,
    services: Optional[dict[str, ServiceConfig]] = None,
    service_plugins: Optional[list[str]] = None,
) -> ProxyConfig:
    """A minimal ProxyConfig; `service_plugins` builds the default test service."""
    svc_dict = dict(services or {})
    if service_plugins is not None and _SERVICE not in svc_dict:
        svc_dict[_SERVICE] = _service(_SERVICE, service_plugins)
    return ProxyConfig(plugins=plugins or [], services=svc_dict)


def _one_service_config(
    plugins: list[_PluginSpec],
    service_plugins: list[str],
) -> ProxyConfig:
    """A config whose only service is `_OTHER_SERVICE`, running the named plugins."""
    return _make_config(
        plugins=plugins,
        services={_OTHER_SERVICE: _service(_OTHER_SERVICE, service_plugins)},
    )


def _manager(
    tmp_path: Path,
    config: ProxyConfig,
    types: Optional[_Types] = None,
    dashboard_refresh: Optional[Callable[[], None]] = None,
) -> PluginManager:
    """A manager scanning tmp_path, with the given types registered directly."""
    return PluginManager(str(tmp_path), config, dashboard_refresh, test_plugins=types or [])


async def _run_request(
    manager: PluginManager,
    service_name: str = _SERVICE,
    headers: Optional[dict[str, str]] = None,
    body: bytes = b'{"test": true}',
    path: str = _PATH,
    request_id: int = 1,
) -> tuple[dict[str, str], bytes, str]:
    """Push one request through a service's request pipeline."""
    return await manager.run_request_pipeline(
        request_id, headers if headers is not None else {}, body, path, service_name)


async def _run_chunk(
    manager: PluginManager,
    service_name: str = _OTHER_SERVICE,
    chunk: bytes = _CHUNK,
    status_code: int = _OK,
    is_final: bool = False,
    request_id: int = 1,
) -> bytes:
    """Push one response chunk through a service's chunk pipeline."""
    return await manager.run_response_chunk_pipeline(
        request_id, {}, chunk, _PATH, status_code, service_name, is_final)


class TestDiscoverTypes:
    """Test _discover_types method."""

    def test_discovers_plugin_subclasses(self, tmp_path: Path) -> None:
        """Test type discovery finds PluginBase subclasses from .py files."""
        _write_plugin_file(tmp_path, "plugin_a.py", _plugin_source("TestPluginA"))
        manager = _manager(tmp_path, _make_config())
        assert "TestPluginA" in manager.get_discovered_types()

    def test_skips_init_file(self, tmp_path: Path) -> None:
        """Test type discovery skips __init__.py."""
        _write_plugin_file(tmp_path, "__init__.py", _plugin_source("InitPlugin"))
        manager = _manager(tmp_path, _make_config())
        assert "InitPlugin" not in manager.get_discovered_types()

    def test_skips_non_plugin_classes(self, tmp_path: Path) -> None:
        """Test type discovery skips non-PluginBase classes."""
        _write_plugin_file(tmp_path, "utils.py", "class Helper:\n    pass\n")
        manager = _manager(tmp_path, _make_config())
        assert "Helper" not in manager.get_discovered_types()

    def test_discovers_multiple_classes_across_files(self, tmp_path: Path) -> None:
        """Test type discovery handles multiple classes across multiple files."""
        _write_plugin_file(
            tmp_path, "plugins_a.py", _plugin_source("AlphaPlugin", "BetaPlugin"))
        _write_plugin_file(tmp_path, "plugins_b.py", _plugin_source("GammaPlugin"))
        manager = _manager(tmp_path, _make_config())
        discovered = manager.get_discovered_types()
        assert "AlphaPlugin" in discovered
        assert "BetaPlugin" in discovered
        assert "GammaPlugin" in discovered


class TestCreateInstances:
    """Test plugin instance creation from config."""

    def test_instance_creation_with_config(self, tmp_path: Path) -> None:
        """Test instance creation passes config to __init__."""
        config = _make_config(
            plugins=[_spec("my_inst", "BodyAppenderPlugin", key="val")],
            service_plugins=["my_inst"],
        )
        manager = _manager(tmp_path, config, [BodyAppenderPlugin])
        assert manager.get_plugin_instance(_SERVICE, "my_inst").config == {"key": "val"}

    def test_multiple_instances_same_type_different_configs(self, tmp_path: Path) -> None:
        """Test multiple instances of the same type with different configs work."""
        config = _make_config(
            plugins=[
                _spec("inst1", "BodyAppenderPlugin", port=8080),
                _spec("inst2", "BodyAppenderPlugin", port=9090),
            ],
            service_plugins=["inst1", "inst2"],
        )
        manager = _manager(tmp_path, config, [BodyAppenderPlugin])
        first = manager.get_plugin_instance(_SERVICE, "inst1")
        second = manager.get_plugin_instance(_SERVICE, "inst2")
        assert first.config == {"port": 8080}
        assert second.config == {"port": 9090}
        assert first is not second

    def test_instance_receives_service_config(self, tmp_path: Path) -> None:
        """A created instance exposes the ServiceConfig of the service it belongs to."""
        config = _make_config(
            plugins=[_spec("echo", "EchoPlugin")],
            service_plugins=["echo"],
        )
        manager = _manager(tmp_path, config, [EchoPlugin])
        instance = manager.get_plugin_instance(_SERVICE, "echo")
        assert instance.service_config is config.services[_SERVICE]

    def test_unknown_plugin_type_raises_config_error(self, tmp_path: Path) -> None:
        """Test unknown plugin type raises ConfigError."""
        config = _make_config(
            plugins=[_spec("bad", "NonExistent")],
            service_plugins=["bad"],
        )
        with pytest.raises(ConfigError, match="NonExistent"):
            _manager(tmp_path, config)

    def test_plugin_not_defined_raises_config_error(self, tmp_path: Path) -> None:
        """Test service referencing undefined plugin name raises ConfigError."""
        config = _make_config(
            plugins=[_spec("p1", "BodyAppenderPlugin")],
            services={_SERVICE: _service(_SERVICE, ["undefined-plugin"])},
        )
        with pytest.raises(ConfigError, match="undefined-plugin"):
            _manager(tmp_path, config, [BodyAppenderPlugin])


class TestServicePluginValidation:
    """Test service plugin reference validation."""

    def test_service_unknown_plugin_raises_config_error(self, tmp_path: Path) -> None:
        """Test service referencing unknown plugin name raises ConfigError."""
        config = _one_service_config([_spec("p1", "BodyAppenderPlugin")], ["nonexistent"])
        with pytest.raises(ConfigError, match="nonexistent"):
            _manager(tmp_path, config, [BodyAppenderPlugin])


class TestRunPipeline:
    """Test run_pipeline method."""

    @pytest.mark.parametrize("plugin_order,expected_suffix", [
        (["a", "b"], b'"A"\"B"'),
        (["b", "a"], b'"B"\"A"'),
        (["a"], b'"A"'),
        (["b"], b'"B"'),
    ])
    async def test_pipeline_executes_in_declared_order(
        self, tmp_path: Path, plugin_order: list[str], expected_suffix: bytes,
    ) -> None:
        """Test run_pipeline executes plugins in declared order."""
        config = _make_config(
            plugins=[
                _spec("a", "BodyAppenderPlugin", suffix='"A"'),
                _spec("b", "BodyAppenderPlugin", suffix='"B"'),
            ],
            service_plugins=plugin_order,
        )
        manager = _manager(tmp_path, config, [BodyAppenderPlugin])
        headers = dict(_JSON_HEADERS)
        result_headers, result, _ = await _run_request(manager, headers=headers, body=b'"start"')
        assert result == b'"start"' + expected_suffix
        assert result_headers == _JSON_HEADERS

    async def test_pipeline_empty_list_returns_unchanged(self, tmp_path: Path) -> None:
        """Test run_pipeline with empty list returns headers and body unchanged."""
        manager = _manager(tmp_path, _make_config(service_plugins=[]))
        headers = dict(_JSON_HEADERS)
        body = b'{"test": true}'
        result_headers, result, _ = await _run_request(manager, headers=headers, body=body)
        assert result_headers is headers
        assert result is body

    async def test_pipeline_empty_bytes_returns_empty(self, tmp_path: Path) -> None:
        """Test run_pipeline with empty bytes and empty list returns empty bytes."""
        manager = _manager(tmp_path, _make_config(service_plugins=[]))
        _, result, _ = await _run_request(manager, body=b"")
        assert result == b""

    async def test_pipeline_headers_propagate_through_plugins(self, tmp_path: Path) -> None:
        """Test headers are passed through and can be modified by plugins."""
        config = _make_config(
            plugins=[_spec("h1", "HeaderPlugin")],
            service_plugins=["h1"],
        )
        manager = _manager(tmp_path, config, [HeaderPlugin])
        result_headers, _, _ = await _run_request(manager, headers=dict(_JSON_HEADERS))
        assert result_headers["X-Modified"] == "true"
        assert result_headers["Content-Type"] == "application/json"

    def test_request_pipeline_is_awaitable(self, tmp_path: Path) -> None:
        """The request pipeline is a coroutine function, so callers await it."""
        manager = _manager(tmp_path, _make_config())
        assert inspect.iscoroutinefunction(manager.run_request_pipeline)

    async def test_request_pipeline_awaits_async_hook(self, tmp_path: Path) -> None:
        """The request pipeline awaits each hook, so a hook can suspend and resume."""
        config = _make_config(
            plugins=[_spec("a1", "AsyncSleepPlugin")],
            service_plugins=["a1"],
        )
        manager = _manager(tmp_path, config, [AsyncSleepPlugin])
        result_headers, result, _ = await _run_request(manager, body=b"async")
        assert result == b"async"
        assert result_headers == {}

    def test_chunk_pipeline_is_awaitable(self, tmp_path: Path) -> None:
        """The chunk pipeline is a coroutine function, so callers await it."""
        manager = _manager(tmp_path, _make_config())
        assert inspect.iscoroutinefunction(manager.run_response_chunk_pipeline)

    async def test_chunk_pipeline_awaits_async_hook(self, tmp_path: Path) -> None:
        """The chunk pipeline awaits each hook, so a hook can suspend and resume."""
        config = _one_service_config(
            [_spec("a1", "AsyncSleepPlugin")], ["a1"]
        )
        manager = _manager(tmp_path, config, [AsyncSleepPlugin])
        result = await _run_chunk(manager, chunk=b"async")
        assert isinstance(result, (bytes, bytearray, memoryview))
        assert bytes(result) == b"async"


class TestMissingPluginsDirectory:
    """Test behavior when plugins directory doesn't exist."""

    def test_missing_plugins_directory_handled_gracefully(self, tmp_path: Path) -> None:
        """Test missing plugins directory is handled gracefully (no crash, empty registry)."""
        manager = _manager(tmp_path / "nonexistent", _make_config())
        assert manager.get_discovered_types() == {}
        assert manager.list_plugin_names(_SERVICE) == []


class TestDuplicateTypeName:
    """Test detection of duplicate plugin type names across files."""

    def test_duplicate_type_name_raises_config_error(self, tmp_path: Path) -> None:
        """Writing two .py files with the same class name raises ConfigError."""
        _write_plugin_file(tmp_path, "plugin_a.py", _plugin_source("SameNamePlugin"))
        _write_plugin_file(tmp_path, "plugin_b.py", _plugin_source("SameNamePlugin"))
        with pytest.raises(ConfigError, match="Duplicate plugin type name"):
            _manager(tmp_path, _make_config())


class TestModuleLoadingErrors:
    """Test error handling for invalid plugin module files."""

    def test_invalid_syntax_raises_config_error(self, tmp_path: Path) -> None:
        """A plugin file with invalid Python syntax raises ConfigError."""
        _write_plugin_file(tmp_path, "broken.py", "this is not valid python {{{")
        with pytest.raises(ConfigError, match="Failed to load plugin module"):
            _manager(tmp_path, _make_config())


class TestPluginRuntimeError:
    """Test plugin runtime error propagation during pipeline execution."""

    async def test_plugin_on_request_exception_propagates(self, tmp_path: Path) -> None:
        """If on_request raises, the exception propagates during run_pipeline."""
        config = _make_config(
            plugins=[_spec("failing_inst", "FailingPlugin")],
            service_plugins=["failing_inst"],
        )
        manager = _manager(tmp_path, config, [FailingPlugin])
        with pytest.raises(RuntimeError, match="plugin error"):
            await _run_request(manager)


class TestOnRequestSignatureIncludesPath:
    """Verify on_request accepts and returns path."""

    async def test_plugin_receives_and_returns_path(self, tmp_path: Path) -> None:
        """The path a plugin returns is the one the pipeline reports."""
        config = _make_config(
            plugins=[_spec("e1", "EchoPlugin")],
            service_plugins=["e1"],
        )
        manager = _manager(tmp_path, config, [EchoPlugin])
        _, _, path = await _run_request(manager, path="/test/path")
        assert path == "/test/path"


class TestInitUiAndPublish:
    """Test init_ui wiring and publish callback in PluginManager."""

    def test_init_ui_called_at_startup(self, tmp_path: Path) -> None:
        """Test that init_ui is called for each plugin during manager construction."""
        config = _make_config(
            plugins=[_spec(
                "ui1", "UIPublisherPlugin", title="UI Title", content="<div>hello</div>")],
            service_plugins=["ui1"],
        )
        manager = _manager(tmp_path, config, [UIPublisherPlugin])
        assert manager.get_plugin_ui(_SERVICE, "ui1") == ("UI Title", "<div>hello</div>")

    def test_publish_caches_title_and_html_per_plugin(self, tmp_path: Path) -> None:
        """Test that publish caches (title, html) keyed by plugin instance name."""
        config = _make_config(
            plugins=[_spec(
                "cache1", "UIPublisherPlugin", title="Stats", content="<h1>stats</h1>")],
            service_plugins=["cache1"],
        )
        manager = _manager(tmp_path, config, [UIPublisherPlugin])
        assert manager.get_plugin_ui(_SERVICE, "cache1") == ("Stats", "<h1>stats</h1>")

    def test_publish_none_when_init_ui_not_overridden(self, tmp_path: Path) -> None:
        """Test that plugins not overriding init_ui have None in _plugin_uis."""
        config = _make_config(
            plugins=[_spec("b1", "BodyAppenderPlugin")],
            service_plugins=["b1"],
        )
        manager = _manager(tmp_path, config, [BodyAppenderPlugin])
        assert manager.get_plugin_ui(_SERVICE, "b1") is None

    def test_dashboard_refresh_fn_called_on_publish(self, tmp_path: Path) -> None:
        """Test that dashboard_refresh_fn is called when publish is invoked."""
        refreshes: list[int] = []

        config = _make_config(
            plugins=[_spec(
                "r1", "UIPublisherPlugin", title="Refresh", content="<span>ok</span>")],
            service_plugins=["r1"],
        )
        _manager(
            tmp_path, config, [UIPublisherPlugin], dashboard_refresh=lambda: refreshes.append(1))
        assert refreshes == [1]

    @pytest.mark.parametrize("service_plugins,expected_titles", [
        (["p1", "p2"], ["First", "Second"]),
        (["p2", "p1"], ["Second", "First"]),
        (["p1"], ["First"]),
        (["p2"], ["Second"]),
    ])
    def test_get_plugin_uis_returns_ordered_list(
        self, tmp_path: Path, service_plugins: list[str], expected_titles: list[str],
    ) -> None:
        """Test get_plugin_uis returns cached UIs in service plugin order."""
        config = _one_service_config(
            [
                _spec("p1", "UIPublisherPlugin", title="First", content="<div>1</div>"),
                _spec("p2", "UIPublisherPlugin", title="Second", content="<div>2</div>"),
            ],
            service_plugins,
        )
        manager = _manager(tmp_path, config, [UIPublisherPlugin])
        assert [title for title, _ in manager.get_plugin_uis(_OTHER_SERVICE)] == expected_titles

    def test_get_plugin_uis_skips_plugins_without_publish(self, tmp_path: Path) -> None:
        """Test get_plugin_uis skips plugins where publish was never called."""
        config = _one_service_config(
            [
                _spec(
                    "with_inst", "UIPublisherPlugin",
                    title="Active", content="<div>active</div>"),
                _spec("without_inst", "NoUIPlugin"),
            ],
            ["with_inst", "without_inst"],
        )
        manager = _manager(tmp_path, config, [UIPublisherPlugin, NoUIPlugin])
        assert manager.get_plugin_uis(_OTHER_SERVICE) == [("Active", "<div>active</div>")]

    def test_get_plugin_uis_empty_service_plugins(self, tmp_path: Path) -> None:
        """Test get_plugin_uis returns empty list for service with no plugins."""
        config = _one_service_config([], [])
        manager = _manager(tmp_path, config)
        assert manager.get_plugin_uis(_OTHER_SERVICE) == []

    def test_dashboard_refresh_none_does_not_crash(self, tmp_path: Path) -> None:
        """Test that omitting dashboard_refresh_fn doesn't crash on publish."""
        config = _make_config(
            plugins=[_spec(
                "nr1", "UIPublisherPlugin", title="Title", content="<div>html</div>")],
            service_plugins=["nr1"],
        )
        manager = _manager(tmp_path, config, [UIPublisherPlugin])
        assert manager.get_plugin_ui(_SERVICE, "nr1") == ("Title", "<div>html</div>")


class TestRunResponseChunkPipeline:
    """Test run_response_chunk_pipeline method."""

    async def test_chains_through_multiple_plugins_in_order(self, tmp_path: Path) -> None:
        """Test on_response_chunk is called on each plugin in service config order."""
        config = _one_service_config(
            [
                _spec("inst_a", "ChunkRecordingPlugin"),
                _spec("inst_b", "ChunkRecordingPlugin"),
            ],
            ["inst_a", "inst_b"],
        )
        manager = _manager(tmp_path, config, [ChunkRecordingPlugin])
        result = await _run_chunk(manager, chunk=b"data: hello\n")
        assert manager.get_plugin_instance(
            _OTHER_SERVICE, "inst_a").received == (b"data: hello\n", _OK, False)
        assert manager.get_plugin_instance(
            _OTHER_SERVICE, "inst_b").received == (b"data: hello\n", _OK, False)
        assert result == b"data: hello\n"

    @pytest.mark.parametrize("is_final,status_code", [
        (True, 200),
        (False, 200),
        (True, 500),
        (False, 401),
    ])
    async def test_forwards_is_final_and_status_code(
        self, tmp_path: Path, is_final: bool, status_code: int,
    ) -> None:
        """Test is_final and status_code are correctly forwarded to plugins."""
        config = _one_service_config([_spec("echo1", "ChunkEchoPlugin")], ["echo1"])
        manager = _manager(tmp_path, config, [ChunkEchoPlugin])
        result = await _run_chunk(manager, status_code=status_code, is_final=is_final)
        plugin = manager.get_plugin_instance(_OTHER_SERVICE, "echo1")
        assert plugin.status == status_code
        assert plugin.final is is_final
        assert result == _CHUNK

    async def test_failing_plugin_does_not_stop_chain(self, tmp_path: Path) -> None:
        """Test a plugin raising an exception doesn't prevent subsequent plugins from running."""
        config = _one_service_config(
            [
                _spec("good1", "GoodChunkPlugin"),
                _spec("bad1", "BadChunkPlugin"),
                _spec("good2", "GoodChunkPlugin"),
            ],
            ["good1", "bad1", "good2"],
        )
        manager = _manager(tmp_path, config, [GoodChunkPlugin, BadChunkPlugin])
        result = await _run_chunk(manager, chunk=b"x", is_final=True)
        assert manager.get_plugin_instance(_OTHER_SERVICE, "good1").called is True
        assert manager.get_plugin_instance(_OTHER_SERVICE, "good2").called is True
        assert result == b"x"

    async def test_unknown_service_returns_original_chunk(self, tmp_path: Path) -> None:
        """Test unknown service_name returns original chunk unchanged."""
        manager = _manager(tmp_path, _make_config())
        assert await _run_chunk(manager, service_name="nonexistent", is_final=True) == _CHUNK

    async def test_service_with_no_plugins_returns_original_chunk(self, tmp_path: Path) -> None:
        """Test service with empty plugin list returns original chunk unchanged."""
        manager = _manager(tmp_path, _one_service_config([], []))
        assert await _run_chunk(manager, is_final=True) == _CHUNK

    async def test_plugin_can_override_chunk(self, tmp_path: Path) -> None:
        """Test a plugin can return modified bytes that replace the chunk."""
        config = _one_service_config([_spec("ov1", "ChunkReplacer")], ["ov1"])
        manager = _manager(tmp_path, config, [ChunkReplacer])
        assert await _run_chunk(manager) == b"replaced"

    async def test_plugins_chain_modifications(self, tmp_path: Path) -> None:
        """Test multiple plugins can each modify the chunk in sequence."""
        config = _one_service_config(
            [
                _spec("a1", "ChunkAppender", suffix="A"),
                _spec("b1", "ChunkAppender", suffix="B"),
            ],
            ["a1", "b1"],
        )
        manager = _manager(tmp_path, config, [ChunkAppender])
        assert await _run_chunk(manager, chunk=b"start") == b"startAB"

    async def test_passthrough_followed_by_override(self, tmp_path: Path) -> None:
        """Test a plugin returning chunk unchanged passes it to next plugin which overrides."""
        config = _one_service_config(
            [
                _spec("pt1", "ChunkEchoPlugin"),
                _spec("rp1", "ChunkReplacer"),
            ],
            ["pt1", "rp1"],
        )
        manager = _manager(tmp_path, config, [ChunkEchoPlugin, ChunkReplacer])
        assert await _run_chunk(manager) == b"replaced"

    async def test_failing_plugin_preserves_current_chunk(self, tmp_path: Path) -> None:
        """Test that when a modifying plugin fails, subsequent plugins receive the last successful chunk."""
        config = _one_service_config(
            [
                _spec("mod1", "ChunkModifier", suffix="M"),
                _spec("crash1", "ChunkCrasher"),
                _spec("mod2", "ChunkModifier", suffix="M"),
            ],
            ["mod1", "crash1", "mod2"],
        )
        manager = _manager(tmp_path, config, [ChunkModifier, ChunkCrasher])
        result = await _run_chunk(manager, chunk=b"x")
        assert manager.get_plugin_instance(_OTHER_SERVICE, "mod1").received == b"x"
        assert manager.get_plugin_instance(_OTHER_SERVICE, "mod2").received == b"xM"
        assert result == b"xMM"


class TestRunTargetPipeline:
    """Test run_target_pipeline method."""

    async def test_calls_each_plugin_in_declared_order(
        self, tmp_path: Path
    ) -> None:
        """prepare_target runs on every plugin of the service, with its id and path."""
        manager = _manager(
            tmp_path,
            _make_config(
                plugins=[
                    _spec("first", "PrepareTargetRecorder"),
                    _spec("second", "PrepareTargetRecorder"),
                ],
                service_plugins=["first", "second"],
            ),
            types=[PrepareTargetRecorder],
        )
        await manager.run_target_pipeline(7, "/v1/models", _SERVICE)
        first = manager.get_plugin_instance(_SERVICE, "first")
        second = manager.get_plugin_instance(_SERVICE, "second")
        assert first.prepared == [(7, "/v1/models")]
        assert second.prepared == [(7, "/v1/models")]

    async def test_error_propagates(self, tmp_path: Path) -> None:
        """A plugin that raises fails the pipeline so the caller can log and continue."""
        manager = _manager(
            tmp_path,
            _make_config(
                plugins=[_spec("broken", "PrepareTargetRecorder", fail=True)],
                service_plugins=["broken"],
            ),
            types=[PrepareTargetRecorder],
        )
        with pytest.raises(RuntimeError, match="prepare_target failed"):
            await manager.run_target_pipeline(1, _PATH, _SERVICE)

    async def test_unknown_service_is_noop(self, tmp_path: Path) -> None:
        """A service the manager knows nothing about prepares nothing."""
        manager = _manager(tmp_path, _make_config(), types=[PrepareTargetRecorder])
        assert (
            await manager.run_target_pipeline(1, _PATH, "nonexistent") is None
        )


class TestPluginBaseOverriddenRequest:
    """Test the path-override map PluginManager builds from its plugins."""

    def test_path_overrides_populated_from_plugins(self, tmp_path: Path) -> None:
        """Every path a service's plugin claims is routed back to that plugin."""
        config = _one_service_config(
            [_spec("override-plugin", "OverridePlugin")], ["override-plugin"])
        manager = _manager(tmp_path, config, [OverridePlugin])
        overrides = manager.get_path_overrides(_OTHER_SERVICE)
        assert sorted(overrides) == ["/custom", "/healthz"]
        assert overrides["/custom"].name() == "OverridePlugin"

    def test_duplicate_path_first_plugin_wins(self, tmp_path: Path) -> None:
        """When two plugins claim one path, the one listed first keeps it."""
        config = _one_service_config(
            [
                _spec("first", "PathConflictPlugin", path="/shared"),
                _spec("second", "PathConflictPlugin", path="/shared"),
            ],
            ["first", "second"],
        )
        manager = _manager(tmp_path, config, [PathConflictPlugin])
        overrides = manager.get_path_overrides(_OTHER_SERVICE)
        assert list(overrides) == ["/shared"]
        assert overrides["/shared"] is manager.get_plugin_instance(_OTHER_SERVICE, "first")

    def test_path_overrides_scoped_to_service(self, tmp_path: Path) -> None:
        """A path claimed on one service is not served on another."""
        config = _make_config(
            plugins=[_spec("scoped-plugin", "ScopedPlugin")],
            services={
                "svc-a": _service("svc-a", ["scoped-plugin"]),
                "svc-b": _service("svc-b", []),
            },
        )
        manager = _manager(tmp_path, config, [ScopedPlugin])
        assert list(manager.get_path_overrides("svc-a")) == ["/admin"]
        assert manager.get_path_overrides("svc-b") == {}


class TestRequestIdPropagation:
    """Test that request_id is correctly propagated through plugin pipelines."""

    async def test_request_id_passed_to_on_request(self, tmp_path: Path) -> None:
        """Test request_id reaches on_request unchanged."""
        config = _make_config(
            plugins=[_spec("rec1", "RequestIdRecorder")],
            service_plugins=["rec1"],
        )
        manager = _manager(tmp_path, config, [RequestIdRecorder])
        await _run_request(manager, path="/test", request_id=42)
        assert manager.get_plugin_instance(_SERVICE, "rec1").received_request_id == 42

    async def test_request_id_passed_to_on_response_chunk(self, tmp_path: Path) -> None:
        """Test request_id reaches on_response_chunk unchanged."""
        config = _one_service_config([_spec("rec1", "ChunkRequestIdRecorder")], ["rec1"])
        manager = _manager(tmp_path, config, [ChunkRequestIdRecorder])
        await _run_chunk(manager, chunk=b"chunk", request_id=99)
        assert manager.get_plugin_instance(
            _OTHER_SERVICE, "rec1").received_request_id == 99

    async def test_different_request_ids_for_concurrent_requests(self, tmp_path: Path) -> None:
        """Test each request carries its own id to the same plugin instance."""
        config = _make_config(
            plugins=[_spec("list1", "RequestIdCollector")],
            service_plugins=["list1"],
        )
        manager = _manager(tmp_path, config, [RequestIdCollector])
        await _run_request(manager, request_id=1)
        await _run_request(manager, request_id=2)
        await _run_request(manager, request_id=3)
        assert manager.get_plugin_instance(_SERVICE, "list1").request_ids == [1, 2, 3]


class TestRunCancelledPipeline:
    """Test run_cancelled_pipeline on PluginManager."""

    async def test_only_calls_on_cancelled_for_ongoing_requests(self, tmp_path: Path) -> None:
        """run_cancelled_pipeline only calls on_cancelled for plugins with ongoing requests."""
        config = _make_config(
            plugins=[
                _spec("tracker-a", "CancelTrackingPlugin"),
                _spec("tracker-b", "CancelTrackingPlugin"),
            ],
            service_plugins=["tracker-a", "tracker-b"],
        )
        manager = _manager(tmp_path, config, [CancelTrackingPlugin])
        plugin_a = manager.get_plugin_instance(_SERVICE, "tracker-a")
        plugin_b = manager.get_plugin_instance(_SERVICE, "tracker-b")

        # Only plugin_a has an ongoing request.
        await plugin_a.on_request(1, {}, b"", "/test")

        manager.run_cancelled_pipeline(1, _SERVICE)

        assert plugin_a.cancelled_requests == [1]
        assert plugin_b.cancelled_requests == []

    async def test_skips_plugins_that_completed_final_chunk(self, tmp_path: Path) -> None:
        """run_cancelled_pipeline skips plugins that already saw is_final=True."""
        config = _make_config(
            plugins=[_spec("tracker", "CancelTrackingPlugin")],
            service_plugins=["tracker"],
        )
        manager = _manager(tmp_path, config, [CancelTrackingPlugin])
        plugin = manager.get_plugin_instance(_SERVICE, "tracker")

        await plugin.on_request(1, {}, b"", "/test")
        await plugin.on_response_chunk(1, {}, b"", "/test", _OK, is_final=True)

        manager.run_cancelled_pipeline(1, _SERVICE)

        assert plugin.cancelled_requests == []

    async def test_skips_unknown_service(self, tmp_path: Path) -> None:
        """run_cancelled_pipeline is a no-op for unknown service."""
        config = _make_config(
            plugins=[_spec("tracker", "CancelTrackingPlugin")],
            service_plugins=["tracker"],
        )
        manager = _manager(tmp_path, config, [CancelTrackingPlugin])
        plugin = manager.get_plugin_instance(_SERVICE, "tracker")
        await plugin.on_request(1, {}, b"", "/test")

        manager.run_cancelled_pipeline(1, "nonexistent")

        assert plugin.cancelled_requests == []

    async def test_errors_in_on_cancelled_do_not_break_pipeline(self, tmp_path: Path) -> None:
        """An exception in one plugin's on_cancelled doesn't prevent others from being called."""
        config = _make_config(
            plugins=[
                _spec("crasher", "CrashingCancelPlugin"),
                _spec("tracker", "CancelTrackingPlugin"),
            ],
            service_plugins=["crasher", "tracker"],
        )
        manager = _manager(tmp_path, config, [CrashingCancelPlugin, CancelTrackingPlugin])
        crasher = manager.get_plugin_instance(_SERVICE, "crasher")
        tracker = manager.get_plugin_instance(_SERVICE, "tracker")

        await crasher.on_request(1, {}, b"", "/test")
        await tracker.on_request(1, {}, b"", "/test")

        manager.run_cancelled_pipeline(1, _SERVICE)

        assert tracker.cancelled_requests == [1]
