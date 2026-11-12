"""Plugin manager for discovery, instantiation, and pipeline execution."""

import importlib.util
import inspect
import logging
import os
import sys
from typing import Callable, Optional, Sequence


from app.config import ConfigError, PluginConfig, ProxyConfig, ServiceConfig
from app.plugins.base import PluginBase

__all__ = ["PluginManager"]

_logger = logging.getLogger(__name__)

_PLUGIN_MODULE_PREFIX = "plugin_"


class PluginManager:
    """Manages plugin lifecycle: discovery, instantiation, and execution.

    Discovers PluginBase subclasses from a directory, creates instances
    from configuration, validates service references, and executes
    plugin pipelines.
    """

    def __init__(
        self,
        plugins_dir: str,
        config: ProxyConfig,
        dashboard_refresh_fn: Optional[Callable[[], None]] = None,
        test_plugins: Optional[Sequence[type[PluginBase]]] = None,
    ) -> None:
        """Initialize plugin manager with discovery and validation.

        Performs two phases:
        1. Discovers PluginBase subclasses from the plugins directory
        2. Creates per-service plugin instances from configuration

        Args:
            plugins_dir: Directory containing plugin Python files
            config: Proxy configuration with plugin definitions
            dashboard_refresh_fn: Optional callback to trigger dashboard refresh
                when a plugin publishes new UI data
            test_plugins: Optional sequence of plugin classes to register
                directly in the type registry, bypassing file discovery.
        """
        self._config = config
        self._dashboard_refresh_fn = dashboard_refresh_fn
        self._type_registry: dict[str, type[PluginBase]] = {}
        self._service_plugins: dict[str, dict[str, PluginBase]] = {}
        self._plugin_uis: dict[tuple[str, str], Optional[tuple[str, str]]] = {}
        self._discover_types(plugins_dir)
        if test_plugins is not None:
            for cls in test_plugins:
                self._register_type(cls)
        self._create_instances(config)

    def _register_type(self, cls: type[PluginBase]) -> None:
        name = cls.__name__
        if name in self._type_registry:
            raise ConfigError(f"Duplicate plugin type name: {name}")
        self._type_registry[name] = cls

    def _discover_types(self, plugins_dir: str) -> None:
        """Scan directory for PluginBase subclasses.

        Imports each .py file (except __init__.py) and collects
        all classes that are subclasses of PluginBase.

        Args:
            plugins_dir: Directory to scan for plugin files
        """
        if not os.path.isdir(plugins_dir):
            return

        for filename in sorted(os.listdir(plugins_dir)):
            if not filename.endswith(".py") or filename == "__init__.py":
                continue

            filepath = os.path.join(plugins_dir, filename)
            module_name = f"{_PLUGIN_MODULE_PREFIX}{filename.replace('.py', '')}"

            spec = importlib.util.spec_from_file_location(module_name, filepath)
            if spec is None or spec.loader is None:
                continue

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            except Exception as e:
                raise ConfigError(
                    f"Failed to load plugin module '{filename}': {e}"
                ) from e

            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (
                    issubclass(obj, PluginBase)
                    and obj is not PluginBase
                    and inspect.getmodule(obj) is module
                ):
                    if name in self._type_registry:
                        raise ConfigError(
                            f"Duplicate plugin type name: {name}"
                        )
                    self._type_registry[name] = obj

    def _create_instances(self, config: ProxyConfig) -> None:
        """Create per-service plugin instances from configuration.

        For each service, instantiates a fresh plugin instance for every
        plugin name referenced by that service. A single plugin definition
        therefore produces one instance per service that uses it.

        Args:
            config: Proxy configuration with plugin definitions and services

        Raises:
            ConfigError: If a service references a plugin name that has no
                matching definition or an unknown plugin type
        """
        plugin_defs: dict[str, PluginConfig] = {
            defn["name"]: defn for defn in config.plugins
        }
        for service_name, service in config.services.items():
            service_plugins: dict[str, PluginBase] = {}
            for plugin_name in service.plugins:
                if plugin_name not in plugin_defs:
                    raise ConfigError(
                        f"Plugin '{plugin_name}' referenced by service "
                        f"'{service_name}' is not defined"
                    )
                plugin_def = plugin_defs[plugin_name]
                type_name = plugin_def["type"]
                plugin_config = plugin_def.get("config", {})

                if type_name not in self._type_registry:
                    raise ConfigError(
                        f"Unknown plugin type: {type_name}"
                    )

                plugin = self._type_registry[type_name](plugin_config, service)
                service_plugins[plugin_name] = plugin
                self._plugin_uis[(service_name, plugin_name)] = None
                publish_closure = self._make_publish(service_name, plugin_name)
                plugin.init_ui(publish_closure)
            self._service_plugins[service_name] = service_plugins

    def _make_publish(self, service_name: str, plugin_name: str) -> Callable[[str, str], None]:
        """Create a publish callback that caches UI data and triggers refresh.

        Args:
            service_name: The service the plugin instance belongs to
            plugin_name: The instance name of the plugin publishing UI data

        Returns:
            A callable accepting (title, html) that caches the pair
            and optionally calls dashboard_refresh_fn
        """
        def publish(title: str, html: str) -> None:
            self._plugin_uis[(service_name, plugin_name)] = (title, html)
            if self._dashboard_refresh_fn is not None:
                self._dashboard_refresh_fn()

        return publish

    def get_discovered_types(self) -> dict[str, type[PluginBase]]:
        """Return a copy of the discovered plugin type registry."""
        return dict(self._type_registry)

    def get_plugin_instance(self, service_name: str, name: str) -> PluginBase:
        """Return the plugin instance for a service.

        Args:
            service_name: The service the plugin instance belongs to
            name: The plugin definition name

        Returns:
            The plugin instance

        Raises:
            KeyError: If the service or plugin is not found
        """
        return self._service_plugins[service_name][name]

    def list_plugin_names(self, service_name: str) -> list[str]:
        """Return a sorted list of plugin names for a service."""
        return sorted(self._service_plugins.get(service_name, {}).keys())

    def get_plugin_ui(self, service_name: str, name: str) -> Optional[tuple[str, str]]:
        """Return the cached UI entry for a plugin instance of a service.

        Args:
            service_name: The service the plugin instance belongs to
            name: The plugin definition name

        Returns:
            A (title, html) tuple if the plugin published UI data, else None
        """
        return self._plugin_uis.get((service_name, name))

    def get_path_overrides(self, service_name: str) -> dict[str, PluginBase]:
        """Return path-override map for plugins enabled on the given service.

        Only considers plugins assigned to the service. First plugin wins
        when multiple plugins claim the same path.
        """
        plugins = self._service_plugins.get(service_name)
        if not plugins:
            return {}
        overrides: dict[str, PluginBase] = {}
        service = self._config.services.get(service_name)
        if service is None:
            return {}
        for plugin_name in service.plugins:
            plugin = plugins.get(plugin_name)
            if plugin is None:
                continue
            for path in plugin.overridden_paths():
                if path not in overrides:
                    overrides[path] = plugin
        return overrides

    def get_plugin_uis(self, service_name: str) -> list[tuple[str, str]]:
        """Get cached UI entries for plugins assigned to a service.

        Resolves plugin names from the service's configuration, looks up
        each in the UI cache, and returns only published entries in order.

        Args:
            service_name: Name of the service to get UI entries for

        Returns:
            Ordered list of (title, html) tuples for plugins that called publish
        """
        service = self._config.services.get(service_name)
        if service is None:
            return []
        result: list[tuple[str, str]] = []
        for plugin_name in service.plugins:
            entry = self._plugin_uis.get((service_name, plugin_name))
            if entry is not None:
                result.append(entry)
        return result

    def _resolve_service_plugins(
        self, service_name: str
    ) -> tuple[ServiceConfig | None, dict[str, PluginBase] | None]:
        """Resolve a service's config and plugin map.

        Args:
            service_name: Name of the service to resolve plugins for

        Returns:
            The (service, plugins) pair, or (None, None) for an unknown service
        """
        service = self._config.services.get(service_name)
        if service is None:
            return None, None
        return service, self._service_plugins.get(service_name)

    async def run_request_pipeline(
        self,
        request_id: int,
        headers: dict[str, str],
        body: bytes,
        path: str,
        service_name: str,
    ) -> tuple[dict[str, str], bytes, str]:
        """Execute a pipeline of plugins in order.

        Iterates through plugin names and awaits on_request on each,
        passing the result to the next plugin in the chain.

        Args:
            request_id: Unique identifier for this client request.
            headers: The request headers dict
            body: The raw request body bytes
            path: The request URL path
            service_name: Name of the service to resolve plugins for

        Returns:
            Tuple of (headers, body, path) after all transformations
        """
        service, plugins = self._resolve_service_plugins(service_name)
        if service is None or not plugins:
            return headers, body, path
        for name in service.plugins:
            plugin = plugins.get(name)
            if plugin is None:
                raise RuntimeError(f"Plugin not found: {name}")
            headers, body, path = await plugin.on_request(request_id, headers, body, path)
        return headers, body, path

    async def run_target_pipeline(
        self, request_id: int, path: str, service_name: str
    ) -> None:
        """Execute prepare_target on every plugin of a service, in order.

        Args:
            request_id: Unique identifier for this client request.
            path: The request URL path
            service_name: Name of the service to resolve plugins for

        Raises:
            RuntimeError: If a configured plugin instance is missing.
        """
        service, plugins = self._resolve_service_plugins(service_name)
        if service is None or not plugins:
            return
        for name in service.plugins:
            plugin = plugins.get(name)
            if plugin is None:
                raise RuntimeError(f"Plugin not found: {name}")
            await plugin.prepare_target(request_id, path)

    def run_response_headers_pipeline(
        self,
        request_id: int,
        headers: dict[str, str],
        status_code: int,
        service_name: str,
    ) -> dict[str, str]:
        """Let the service plugins update the response headers.

        Args:
            request_id: The client request id.
            headers: The response headers.
            status_code: The response status code.
            service_name: The service name.

        Returns:
            The response headers.
        """
        service, plugins = self._resolve_service_plugins(service_name)
        if service is None or not plugins:
            return headers
        for plugin_name in service.plugins:
            plugin = plugins.get(plugin_name)
            if plugin is None:
                raise RuntimeError(f"Plugin not found: {plugin_name}")
            headers = plugin.on_response_headers(request_id, headers, status_code)
        return headers

    async def run_response_chunk_pipeline(
        self,
        request_id: int,
        headers: dict[str, str],
        chunk: bytes,
        path: str,
        status_code: int,
        service_name: str,
        is_final: bool,
    ) -> bytes:
        """Execute on_response_chunk through all plugins for a service.

        Resolves plugin names from the service's configuration and awaits
        on_response_chunk on each plugin in order. If a plugin returns
        non-None bytes, that becomes the new chunk passed to subsequent
        plugins. Errors are caught and logged without breaking the chain.

        Args:
            request_id: Unique identifier for this client request.
            headers: The response headers dict
            chunk: The raw response chunk bytes
            path: The original request URL path
            status_code: The HTTP status code of the response
            service_name: Name of the service to resolve plugins for
            is_final: Whether this is the final chunk of the response

        Returns:
            The (possibly modified) response chunk after all plugins have
            had a chance to transform it.
        """
        service, plugins = self._resolve_service_plugins(service_name)
        if service is None or not plugins:
            return chunk
        current_chunk = chunk
        for plugin_name in service.plugins:
            plugin = plugins.get(plugin_name)
            if plugin is None:
                continue
            try:
                current_chunk = await plugin.on_response_chunk(
                    request_id, headers, current_chunk, path, status_code, is_final
                )
            except Exception as e:
                _logger.error(
                    "Plugin '%s' on_response_chunk error: %s", plugin_name, e
                )
        return current_chunk

    def run_cancelled_pipeline(
        self, request_id: int, service_name: str
    ) -> None:
        """Call on_cancelled() on plugins that have an ongoing request.

        Only invokes on_cancelled() on plugins where is_request_ongoing(request_id)
        returns True, meaning on_request() ran but on_response_chunk(is_final=True)
        did not complete.

        Args:
            request_id: Unique identifier for this client request.
            service_name: Name of the service to resolve plugins for.
        """
        service, plugins = self._resolve_service_plugins(service_name)
        if service is None or not plugins:
            return
        for plugin_name in service.plugins:
            plugin = plugins.get(plugin_name)
            if plugin is None:
                continue
            if not plugin.is_request_ongoing(request_id):
                continue
            try:
                plugin.on_cancelled(request_id)
            except Exception as e:
                _logger.error(
                    "Plugin '%s' on_cancelled error: %s", plugin_name, e
                )
