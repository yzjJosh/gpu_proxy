"""Proxy configuration loading and validation."""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from urllib.parse import urlparse
from typing import Optional, TypedDict

__all__ = ["ConfigError", "PluginConfig", "ProxyConfig", "ServiceConfig", "load_config"]


class PluginConfig(TypedDict):
    """One entry of the top-level plugins list."""

    name: str
    type: str
    config: dict


@dataclass(frozen=True)
class ServiceConfig:
    """Configuration for a single proxied service."""

    name: str
    priority: int
    vram_budget: int
    hostname: str
    target: str
    container: str
    backup: Optional[str] = None
    fallback_on_error: bool = True
    gpu_routes: list[str] = field(default_factory=list)
    eviction_command: str = "docker restart {container}"
    external: bool = False
    is_persistent: bool = False
    health_check_command: Optional[str] = None
    health_check_interval: int = 5
    plugins: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ProxyConfig:
    """Top-level proxy configuration."""

    listen_port: int = 8080
    total_vram_budget: int = 20480
    eviction_timeout: int = 60
    poll_interval: int = 5
    eviction_command_timeout: int = 60
    wait_queue_timeout: int = 60
    connection_timeout: int = 30
    sock_read_timeout: int = 120
    dashboard_hostname: str = "gpu-proxy.local"
    log_level: str = "INFO"
    services: dict[str, ServiceConfig] = field(default_factory=dict)
    hostname_map: dict[str, ServiceConfig] = field(default_factory=dict)
    plugins: list[PluginConfig] = field(default_factory=list)


class ConfigError(Exception):
    """Raised when configuration validation fails."""


def _is_valid_url(url: str) -> bool:
    """Check if URL has a scheme and netloc."""
    try:
        parsed = urlparse(url)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False


def _validate_external_service(name: str, svc_data: dict) -> tuple[int, int, str]:
    if "vram_budget" in svc_data:
        raise ConfigError(f"Service '{name}': vram_budget is banned for external services")
    if "container" in svc_data:
        raise ConfigError(f"Service '{name}': container is banned for external services")
    if "priority" in svc_data:
        raise ConfigError(f"Service '{name}': priority is banned for external services")
    if "gpu_routes" in svc_data:
        raise ConfigError(f"Service '{name}': gpu_routes is banned for external services")
    return 0, 0, ""


def _validate_local_service(name: str, svc_data: dict, total_vram_budget: int, priorities: set[int]) -> tuple[int, int, str]:
    if "container" not in svc_data:
        raise ConfigError(f"Service '{name}' missing required field: container")
    if "vram_budget" not in svc_data:
        raise ConfigError(f"Service '{name}' missing required field: vram_budget")
    if "priority" not in svc_data:
        raise ConfigError(f"Service '{name}' missing required field: priority")

    priority = svc_data["priority"]
    vram_budget = svc_data["vram_budget"]
    container = svc_data["container"]

    if not isinstance(priority, int) or isinstance(priority, bool):
        raise ConfigError(f"Service '{name}' priority must be an integer")
    if not isinstance(vram_budget, int) or isinstance(vram_budget, bool):
        raise ConfigError(f"Service '{name}' vram_budget must be an integer")

    if priority <= 0:
        raise ConfigError(f"Service '{name}' priority must be > 0")
    if vram_budget <= 0:
        raise ConfigError(f"Service '{name}' vram_budget must be > 0")
    if vram_budget > total_vram_budget:
        raise ConfigError(
            f"Service '{name}' vram_budget ({vram_budget}) exceeds total_vram_budget ({total_vram_budget})"
        )

    if priority in priorities:
        raise ConfigError(f"Duplicate priority {priority} found (services must have unique priorities)")
    priorities.add(priority)

    return priority, vram_budget, container


def _parse_plugins(plugins_raw: list) -> list[PluginConfig]:
    """Parse and validate the global plugins list from config data."""
    if not isinstance(plugins_raw, list):
        raise ConfigError("plugins must be a list")

    plugins: list[PluginConfig] = []
    seen_plugin_names: set[str] = set()
    for plugin_def in plugins_raw:
        if not isinstance(plugin_def, dict):
            raise ConfigError("Each plugin entry must be a mapping")
        for req in ("name", "type"):
            if req not in plugin_def:
                raise ConfigError(f"Plugin entry missing required field: {req}")
        pname = plugin_def["name"]
        ptype = plugin_def["type"]
        if not isinstance(pname, str):
            raise ConfigError(f"Plugin name must be a string, got {type(pname)}")
        if not isinstance(ptype, str):
            raise ConfigError(f"Plugin type must be a string, got {type(ptype)}")
        pconfig = plugin_def.get("config", {})
        if not isinstance(pconfig, dict):
            raise ConfigError(f"Plugin '{pname}' config must be a dict")
        if pname in seen_plugin_names:
            raise ConfigError(f"Duplicate plugin name: {pname}")
        seen_plugin_names.add(pname)
        plugins.append({
            "name": pname,
            "type": ptype,
            "config": pconfig,
        })
    return plugins


def load_config(path: str) -> ProxyConfig:
    """Load and validate proxy configuration from YAML file."""
    data = _load_yaml(path)
    top = _top_level_fields(data)

    services_data = data.get("services")
    if not services_data or not isinstance(services_data, dict) or len(services_data) == 0:
        raise ConfigError("At least one service must be defined")

    services, hostname_map = _build_services(
        services_data,
        top.total_vram_budget,
        top.dashboard_hostname,
    )

    plugins = _parse_plugins(data.get("plugins", []))

    return ProxyConfig(
        listen_port=top.listen_port,
        total_vram_budget=top.total_vram_budget,
        eviction_timeout=top.eviction_timeout,
        poll_interval=top.poll_interval,
        eviction_command_timeout=top.eviction_command_timeout,
        wait_queue_timeout=top.wait_queue_timeout,
        connection_timeout=top.connection_timeout,
        sock_read_timeout=top.sock_read_timeout,
        dashboard_hostname=top.dashboard_hostname,
        log_level=top.log_level,
        services=services,
        hostname_map=hostname_map,
        plugins=plugins,
    )


def _load_yaml(path: str) -> dict:
    """Read and parse the YAML config file.

    Args:
        path: Path to the YAML file.

    Returns:
        The parsed YAML document.

    Raises:
        ConfigError: The file is not valid YAML or is not a mapping.
    """
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML: {e}") from e
    except FileNotFoundError:
        raise
    if not isinstance(data, dict):
        raise ConfigError("Config must be a YAML mapping")
    return data


@dataclass(frozen=True)
class _TopLevelFields:
    """The top-level config fields extracted from the YAML document."""

    listen_port: int
    total_vram_budget: int
    eviction_timeout: int
    poll_interval: int
    eviction_command_timeout: int
    wait_queue_timeout: int
    connection_timeout: int
    sock_read_timeout: int
    dashboard_hostname: str
    log_level: str


def _top_level_fields(data: dict) -> _TopLevelFields:
    """Extract the top-level config fields with their defaults.

    Args:
        data: The parsed config mapping.

    Returns:
        A _TopLevelFields instance holding the resolved values.
    """
    return _TopLevelFields(
        listen_port=data.get("listen_port", 8080),
        total_vram_budget=data.get("total_vram_budget", 20480),
        eviction_timeout=data.get("eviction_timeout", 60),
        poll_interval=data.get("poll_interval", 5),
        eviction_command_timeout=data.get("eviction_command_timeout", 60),
        wait_queue_timeout=data.get("wait_queue_timeout", 60),
        connection_timeout=data.get("connection_timeout", 30),
        sock_read_timeout=data.get("sock_read_timeout", 120),
        dashboard_hostname=data.get("dashboard_hostname", "gpu-proxy.local"),
        log_level=data.get("log_level", "INFO"),
    )


def _build_services(
    services_data: dict,
    total_vram_budget: int,
    dashboard_hostname: str,
) -> tuple[dict[str, ServiceConfig], dict[str, ServiceConfig]]:
    """Validate each service and build the services and hostname maps.

    Args:
        services_data: The raw services mapping.
        total_vram_budget: The global VRAM budget for budget validation.
        dashboard_hostname: The dashboard hostname reserved from services.

    Returns:
        A tuple of (services, hostname_map).

    Raises:
        ConfigError: A service fails validation.
    """
    services: dict[str, ServiceConfig] = {}
    hostname_map: dict[str, ServiceConfig] = {}
    priorities: set[int] = set()
    hostnames: set[str] = set()

    for name, svc_data in services_data.items():
        if not isinstance(svc_data, dict):
            raise ConfigError(f"Service '{name}' must be a mapping")
        svc = _validate_service(
            name, svc_data, total_vram_budget, priorities, hostnames, dashboard_hostname
        )
        services[name] = svc
        hostname_map[svc.hostname] = svc

    return services, hostname_map


def _validate_service(
    name: str,
    svc_data: dict,
    total_vram_budget: int,
    priorities: set[int],
    hostnames: set[str],
    dashboard_hostname: str,
) -> ServiceConfig:
    """Validate and build a single ServiceConfig.

    Args:
        name: The service name.
        svc_data: The raw service mapping.
        total_vram_budget: The global VRAM budget.
        priorities: Set of already-used priorities.
        hostnames: Set of already-used hostnames.
        dashboard_hostname: The dashboard hostname reserved from services.

    Returns:
        The validated ServiceConfig.

    Raises:
        ConfigError: A validation check fails.
    """
    external = svc_data.get("external", False)

    if external:
        priority, vram_budget, container = _validate_external_service(name, svc_data)
    else:
        priority, vram_budget, container = _validate_local_service(
            name, svc_data, total_vram_budget, priorities
        )

    is_persistent = svc_data.get("is_persistent", False)
    if not isinstance(is_persistent, bool):
        raise ConfigError(f"Service '{name}' is_persistent must be a boolean")

    fallback_on_error = svc_data.get("fallback_on_error", True)
    if not isinstance(fallback_on_error, bool):
        raise ConfigError(f"Service '{name}' fallback_on_error must be a boolean")

    _require_fields(name, svc_data)
    plugins = _service_plugins(name, svc_data)
    health_check_command = _service_health_check_command(name, svc_data)
    health_check_interval = _service_health_check_interval(name, svc_data)
    hostname = _service_hostname(name, svc_data)
    target = _service_target(name, svc_data)
    backup = _service_backup(name, svc_data)
    _register_hostname(hostname, hostnames, dashboard_hostname)

    return ServiceConfig(
        name=name,
        priority=priority,
        vram_budget=vram_budget,
        hostname=hostname,
        target=target,
        container=container,
        backup=backup,
        fallback_on_error=fallback_on_error,
        gpu_routes=svc_data.get("gpu_routes", []) or [],
        eviction_command=svc_data.get("eviction_command", "docker restart {container}"),
        external=external,
        is_persistent=is_persistent,
        health_check_command=health_check_command,
        health_check_interval=health_check_interval,
        plugins=plugins,
    )


def _require_fields(name: str, svc_data: dict) -> None:
    """Raise when the service mapping lacks a required field."""
    for key in ("hostname", "target"):
        if key not in svc_data:
            raise ConfigError(f"Service '{name}' missing required field: {key}")


def _service_hostname(name: str, svc_data: dict) -> str:
    """Read and validate the service hostname."""
    hostname = svc_data["hostname"]
    if not isinstance(hostname, str) or not hostname.strip():
        raise ConfigError(f"Service '{name}' hostname must be a non-empty string")
    return hostname


def _service_target(name: str, svc_data: dict) -> str:
    """Read and validate the service target URL."""
    target = svc_data["target"]
    if not _is_valid_url(target):
        raise ConfigError(f"Service '{name}' target is not a valid URL: {target}")
    return target


def _service_backup(name: str, svc_data: dict) -> str | None:
    """Read and validate the optional service backup URL."""
    backup = svc_data.get("backup")
    if backup is not None and not _is_valid_url(backup):
        raise ConfigError(f"Service '{name}' backup is not a valid URL: {backup}")
    return backup


def _service_health_check_command(name: str, svc_data: dict) -> str | None:
    """Read and validate the optional health check command."""
    command = svc_data.get("health_check_command")
    if command is not None and not isinstance(command, str):
        raise ConfigError(f"Service '{name}' health_check_command must be a string")
    return command


def _service_health_check_interval(name: str, svc_data: dict) -> int:
    """Read and validate the health check interval."""
    interval = svc_data.get("health_check_interval", 5)
    if not isinstance(interval, int) or isinstance(interval, bool):
        raise ConfigError(f"Service '{name}' health_check_interval must be an integer")
    if interval < 1:
        raise ConfigError(f"Service '{name}' health_check_interval must be >= 1")
    return interval


def _service_plugins(name: str, svc_data: dict) -> list[str]:
    """Read and validate the service's plugin list."""
    plugins = svc_data.get("plugins", [])
    if not isinstance(plugins, list):
        raise ConfigError(f"Service '{name}' plugins must be a list")
    for plugin in plugins:
        if not isinstance(plugin, str):
            raise ConfigError(
                f"Service '{name}' plugins entries must be strings, got {type(plugin)}"
            )
    return plugins


def _register_hostname(
    hostname: str,
    hostnames: set[str],
    dashboard_hostname: str,
) -> None:
    """Register a service hostname, checking duplicates and dashboard collision.

    Args:
        hostname: The validated hostname.
        hostnames: Set of already-registered hostnames, mutated in place.
        dashboard_hostname: The dashboard hostname reserved from services.

    Raises:
        ConfigError: The hostname is duplicated or collides with the dashboard.
    """
    if hostname in hostnames:
        raise ConfigError(f"Duplicate hostname '{hostname}' found")
    hostnames.add(hostname)

    if hostname == dashboard_hostname:
        raise ConfigError(f"Service hostname '{hostname}' collides with dashboard_hostname")
