"""Tests for config module: loading, validation, and edge cases."""

from pathlib import Path
from textwrap import indent

import pytest
import yaml

from app.config import ConfigError, ProxyConfig, ServiceConfig, load_config

_LISTEN_PORT = 8080
_ALTERNATE_LISTEN_PORT = 9090
_TOTAL_VRAM_MIB = 20480
_ALTERNATE_TOTAL_VRAM_MIB = 24576
_REDUCED_TOTAL_VRAM_MIB = 8192
_LARGE_SERVICE_VRAM_MIB = 16384
_SMALL_SERVICE_VRAM_MIB = 4096
_SERVICE_VRAM_MIB = 10240
_TINY_SERVICE_VRAM_MIB = 1024
_OVERSIZED_VRAM_MIB = 12288
_NEGATIVE_VRAM_MIB = -1024

_EVICTION_TIMEOUT_S = 60
_ALTERNATE_EVICTION_TIMEOUT_S = 90
_EVICTION_COMMAND_TIMEOUT_S = 60
_ALTERNATE_EVICTION_COMMAND_TIMEOUT_S = 75
_WAIT_QUEUE_TIMEOUT_S = 60
_ALTERNATE_WAIT_QUEUE_TIMEOUT_S = 45
_CONNECTION_TIMEOUT_S = 30
_ALTERNATE_CONNECTION_TIMEOUT_S = 20
_POLL_INTERVAL_S = 5
_ALTERNATE_POLL_INTERVAL_S = 15
_SOCK_READ_TIMEOUT_S = 120
_ALTERNATE_SOCK_READ_TIMEOUT_S = 200
_HEALTH_CHECK_INTERVAL_S = 10
_DEFAULT_HEALTH_CHECK_INTERVAL_S = 5
_NEGATIVE_HEALTH_CHECK_INTERVAL_S = -5

_DASHBOARD_HOSTNAME = "gpu-proxy.local"
_ALTERNATE_DASHBOARD_HOSTNAME = "dashboard.internal"
_DEFAULT_EVICTION_COMMAND = "docker restart {container}"
_CUSTOM_EVICTION_COMMAND = "systemctl restart {container}"
_HEALTH_CHECK_COMMAND = "curl -f http://localhost:8001/health"
_INVALID_URL = "not-a-valid-url"
_HOSTLESS_URL = "http://"
_NON_HTTP_URL = "ftp://127.0.0.1:8001"
_DEFAULT_LOG_LEVEL = "INFO"
_ALTERNATE_LOG_LEVEL = "DEBUG"

_DEFAULT_GPU_ROUTES: list[str] = ["/v1/completions"]


def _service(
    priority: int = 1,
    vram_budget: int = _SERVICE_VRAM_MIB,
    hostname: str = "svc1.local",
    target: str = "http://127.0.0.1:8001",
    container: str = "svc1",
    gpu_routes: list[str] | None = _DEFAULT_GPU_ROUTES,
    backup: str | None = "http://127.0.0.1:9001",
    eviction_command: str | None = None,
    health_check_command: str | None = None,
    health_check_interval: int | None = None,
    is_persistent: bool | None = None,
    fallback_on_error: bool | None = None,
    plugins: list[str] | None = None,
) -> dict:
    """A minimal valid local service. Pass `None` to leave an optional field out."""
    service: dict = {
        "priority": priority,
        "vram_budget": vram_budget,
        "hostname": hostname,
        "target": target,
        "container": container,
    }
    for field, value in (
        ("gpu_routes", gpu_routes),
        ("backup", backup),
        ("eviction_command", eviction_command),
        ("health_check_command", health_check_command),
        ("health_check_interval", health_check_interval),
        ("is_persistent", is_persistent),
        ("fallback_on_error", fallback_on_error),
        ("plugins", plugins),
    ):
        if value is not None:
            service[field] = value
    return service


def _external_service(
    hostname: str = "ext.local",
    target: str = "https://api.example.com",
    priority: int | None = None,
    vram_budget: int | None = None,
    container: str | None = None,
    gpu_routes: list[str] | None = None,
) -> dict:
    """A minimal valid external service, which owns no GPU and no container.

    Its target is an off-box https endpoint, the shape an external service
    normally takes. The proxy dials it rather than starting it. A banned field
    is declared by passing it, for the rejection tests.
    """
    service: dict = {
        "external": True,
        "hostname": hostname,
        "target": target,
    }
    for field, value in (
        ("priority", priority),
        ("vram_budget", vram_budget),
        ("container", container),
        ("gpu_routes", gpu_routes),
    ):
        if value is not None:
            service[field] = value
    return service


def _document(
    listen_port: int = _LISTEN_PORT,
    total_vram_budget: int = _TOTAL_VRAM_MIB,
    dashboard_hostname: str = _DASHBOARD_HOSTNAME,
    services: dict | None = None,
    plugins: list | str | None = None,
    log_level: str | None = None,
    eviction_timeout: int | None = None,
    poll_interval: int | None = None,
    eviction_command_timeout: int | None = None,
    wait_queue_timeout: int | None = None,
    connection_timeout: int | None = None,
    sock_read_timeout: int | None = None,
) -> dict:
    """A minimal valid config document. Pass `None` to leave an optional field out."""
    if services is None:
        services = {"svc1": _service()}
    document: dict = {
        "listen_port": listen_port,
        "total_vram_budget": total_vram_budget,
        "dashboard_hostname": dashboard_hostname,
        "services": services,
    }
    for field, value in (
        ("plugins", plugins),
        ("log_level", log_level),
        ("eviction_timeout", eviction_timeout),
        ("poll_interval", poll_interval),
        ("eviction_command_timeout", eviction_command_timeout),
        ("wait_queue_timeout", wait_queue_timeout),
        ("connection_timeout", connection_timeout),
        ("sock_read_timeout", sock_read_timeout),
    ):
        if value is not None:
            document[field] = value
    return document


def _two_services() -> dict:
    """Two local services sharing one GPU, the arrangement the proxy is built for."""
    return {
        "llamacpp-server": _service(
            priority=1,
            vram_budget=_LARGE_SERVICE_VRAM_MIB,
            hostname="llamacpp.local",
            target="http://127.0.0.1:49863",
            container="llamacpp-server",
            gpu_routes=["/v1/completions", "/v1/chat/completions"],
            eviction_command=_DEFAULT_EVICTION_COMMAND,
            backup="http://127.0.0.1:49864",
        ),
        "cpai": _service(
            priority=2,
            vram_budget=_SMALL_SERVICE_VRAM_MIB,
            hostname="cpai.local",
            target="http://127.0.0.1:32168",
            container="cpai",
            gpu_routes=["/api/detect"],
            backup="http://127.0.0.1:32169",
        ),
    }


def _load_text(tmp_path: Path, text: str) -> ProxyConfig:
    """Load config from YAML written exactly as given."""
    path = tmp_path / "config.yaml"
    path.write_text(text)
    return load_config(str(path))


def _load(tmp_path: Path, document: dict) -> ProxyConfig:
    """Load config from a document rendered to YAML."""
    return _load_text(tmp_path, yaml.safe_dump(document))


class TestLoadConfig:
    """Tests for load_config with valid and invalid configurations."""

    def test_valid_config_loads(self, tmp_path: Path) -> None:
        """Valid config loads, keeping every top-level field and both services."""
        config = _load(
            tmp_path,
            _document(
                services=_two_services(),
                listen_port=_ALTERNATE_LISTEN_PORT,
                total_vram_budget=_ALTERNATE_TOTAL_VRAM_MIB,
                eviction_timeout=_ALTERNATE_EVICTION_TIMEOUT_S,
                poll_interval=_ALTERNATE_POLL_INTERVAL_S,
                eviction_command_timeout=_ALTERNATE_EVICTION_COMMAND_TIMEOUT_S,
                wait_queue_timeout=_ALTERNATE_WAIT_QUEUE_TIMEOUT_S,
                connection_timeout=_ALTERNATE_CONNECTION_TIMEOUT_S,
                sock_read_timeout=_ALTERNATE_SOCK_READ_TIMEOUT_S,
                dashboard_hostname=_ALTERNATE_DASHBOARD_HOSTNAME,
                log_level=_ALTERNATE_LOG_LEVEL,
            ),
        )
        assert config.listen_port == _ALTERNATE_LISTEN_PORT
        assert config.total_vram_budget == _ALTERNATE_TOTAL_VRAM_MIB
        assert config.eviction_timeout == _ALTERNATE_EVICTION_TIMEOUT_S
        assert config.poll_interval == _ALTERNATE_POLL_INTERVAL_S
        assert config.eviction_command_timeout == _ALTERNATE_EVICTION_COMMAND_TIMEOUT_S
        assert config.wait_queue_timeout == _ALTERNATE_WAIT_QUEUE_TIMEOUT_S
        assert config.connection_timeout == _ALTERNATE_CONNECTION_TIMEOUT_S
        assert config.sock_read_timeout == _ALTERNATE_SOCK_READ_TIMEOUT_S
        assert config.dashboard_hostname == _ALTERNATE_DASHBOARD_HOSTNAME
        assert config.log_level == _ALTERNATE_LOG_LEVEL
        assert set(config.services) == {"llamacpp-server", "cpai"}
        llamacpp = config.services["llamacpp-server"]
        assert llamacpp.priority == 1
        assert llamacpp.vram_budget == _LARGE_SERVICE_VRAM_MIB
        assert llamacpp.hostname == "llamacpp.local"
        assert llamacpp.target == "http://127.0.0.1:49863"
        assert llamacpp.container == "llamacpp-server"
        assert llamacpp.gpu_routes == ["/v1/completions", "/v1/chat/completions"]
        assert llamacpp.backup == "http://127.0.0.1:49864"
        assert llamacpp.external is False
        cpai = config.services["cpai"]
        assert cpai.priority == 2
        assert cpai.vram_budget == _SMALL_SERVICE_VRAM_MIB
        assert cpai.hostname == "cpai.local"
        assert cpai.target == "http://127.0.0.1:32168"

    def test_hostname_map_populated(self, tmp_path: Path) -> None:
        """hostname_map indexes the same services by hostname for O(1) lookup."""
        config = _load(tmp_path, _document(services=_two_services()))
        assert set(config.hostname_map) == {"llamacpp.local", "cpai.local"}
        assert config.hostname_map["llamacpp.local"].name == "llamacpp-server"
        assert config.hostname_map["cpai.local"].name == "cpai"
        assert config.hostname_map["llamacpp.local"] is config.services["llamacpp-server"]

    def test_duplicate_priorities_rejected(self, tmp_path: Path) -> None:
        """Two local services claiming the same priority are rejected."""
        services = {
            "svc1": _service(priority=1),
            "svc2": _service(
                priority=1,
                vram_budget=_SMALL_SERVICE_VRAM_MIB,
                hostname="svc2.local",
                target="http://127.0.0.1:8002",
                container="svc2",
            ),
        }
        with pytest.raises(ConfigError, match="Duplicate priority 1"):
            _load(tmp_path, _document(services=services))

    def test_duplicate_hostnames_rejected(self, tmp_path: Path) -> None:
        """Two services answering on the same hostname are rejected."""
        services = {
            "svc1": _service(hostname="same.local"),
            "svc2": _service(
                priority=2,
                vram_budget=_SMALL_SERVICE_VRAM_MIB,
                hostname="same.local",
                target="http://127.0.0.1:8002",
                container="svc2",
            ),
        }
        with pytest.raises(ConfigError, match="Duplicate hostname 'same.local'"):
            _load(tmp_path, _document(services=services))

    def test_dashboard_hostname_collision_rejected(self, tmp_path: Path) -> None:
        """A service hostname equal to the dashboard's is rejected."""
        document = _document(dashboard_hostname="svc1.local")
        with pytest.raises(ConfigError, match="collides with dashboard_hostname"):
            _load(tmp_path, document)

    def test_vram_budget_exceeds_total_rejected(self, tmp_path: Path) -> None:
        """A service asking for more VRAM than the config's own budget is rejected."""
        document = _document(
            total_vram_budget=_REDUCED_TOTAL_VRAM_MIB,
            services={"svc1": _service(vram_budget=_OVERSIZED_VRAM_MIB)},
        )
        with pytest.raises(ConfigError, match="exceeds total_vram_budget"):
            _load(tmp_path, document)

    def test_gpu_routes_empty_allowed(self, tmp_path: Path) -> None:
        """An explicitly empty gpu_routes list is valid: the service is passthrough-only."""
        services = {
            "passthrough": _service(
                vram_budget=_TINY_SERVICE_VRAM_MIB,
                hostname="pass.local",
                container="passthrough",
                gpu_routes=[],
            )
        }
        config = _load(tmp_path, _document(services=services))
        assert config.services["passthrough"].gpu_routes == []

    def test_vram_budget_zero_rejected(self, tmp_path: Path) -> None:
        """A service claiming zero VRAM is rejected."""
        document = _document(services={"svc1": _service(vram_budget=0)})
        with pytest.raises(ConfigError, match="vram_budget must be > 0"):
            _load(tmp_path, document)

    def test_vram_budget_negative_rejected(self, tmp_path: Path) -> None:
        """A service claiming negative VRAM is rejected."""
        document = _document(services={"svc1": _service(vram_budget=_NEGATIVE_VRAM_MIB)})
        with pytest.raises(ConfigError, match="vram_budget must be > 0"):
            _load(tmp_path, document)

    def test_priority_zero_rejected(self, tmp_path: Path) -> None:
        """Priority zero is reserved for external services and rejected for local ones."""
        document = _document(services={"svc1": _service(priority=0)})
        with pytest.raises(ConfigError, match="priority must be > 0"):
            _load(tmp_path, document)

    @pytest.mark.parametrize("missing", ["target", "vram_budget", "priority"])
    def test_missing_required_field_rejected(self, tmp_path: Path, missing: str) -> None:
        """A service missing a required field is rejected, naming the field it lacks."""
        service = _service()
        service.pop(missing)
        with pytest.raises(ConfigError, match=f"missing required field: {missing}"):
            _load(tmp_path, _document(services={"svc1": service}))

    def test_no_services_rejected(self, tmp_path: Path) -> None:
        """A config defining no services at all is rejected."""
        with pytest.raises(ConfigError, match="At least one service must be defined"):
            _load(tmp_path, _document(services={}))

    def test_eviction_command_default(self, tmp_path: Path) -> None:
        """An omitted eviction_command falls back to restarting the container."""
        config = _load(tmp_path, _document())
        assert config.services["svc1"].eviction_command == _DEFAULT_EVICTION_COMMAND

    def test_custom_eviction_command(self, tmp_path: Path) -> None:
        """A service's own eviction_command is kept in place of the default."""
        document = _document(
            services={"svc1": _service(eviction_command=_CUSTOM_EVICTION_COMMAND)}
        )
        config = _load(tmp_path, document)
        assert config.services["svc1"].eviction_command == _CUSTOM_EVICTION_COMMAND

    def test_defaults_applied(self, tmp_path: Path) -> None:
        """Every top-level field a config leaves out falls back to its default."""
        document = _document()
        document.pop("listen_port")
        document.pop("total_vram_budget")
        document.pop("dashboard_hostname")
        config = _load(tmp_path, document)
        assert config.listen_port == _LISTEN_PORT
        assert config.total_vram_budget == _TOTAL_VRAM_MIB
        assert config.dashboard_hostname == _DASHBOARD_HOSTNAME
        assert config.poll_interval == _POLL_INTERVAL_S
        assert config.sock_read_timeout == _SOCK_READ_TIMEOUT_S
        assert config.log_level == _DEFAULT_LOG_LEVEL
        assert config.eviction_timeout == _EVICTION_TIMEOUT_S
        assert config.eviction_command_timeout == _EVICTION_COMMAND_TIMEOUT_S
        assert config.wait_queue_timeout == _WAIT_QUEUE_TIMEOUT_S
        assert config.connection_timeout == _CONNECTION_TIMEOUT_S

    def test_config_is_immutable(self, sample_config: ProxyConfig) -> None:
        """ProxyConfig is frozen after creation."""
        with pytest.raises(AttributeError):
            sample_config.listen_port = _ALTERNATE_LISTEN_PORT

    def test_service_config_is_immutable(
        self, sample_service_llamacpp: ServiceConfig
    ) -> None:
        """ServiceConfig is frozen after creation."""
        with pytest.raises(AttributeError):
            sample_service_llamacpp.priority = 99

    def test_file_not_found(self, tmp_path: Path) -> None:
        """Loading a config file that is not there raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config(str(tmp_path / "no-such-config.yaml"))

    def test_config_example_validates(self) -> None:
        """The config.example.yaml shipped with the repo passes validation."""
        example_path = Path(__file__).resolve().parents[2] / "config.example.yaml"
        config = load_config(str(example_path))
        assert set(config.services) == {"llamacpp-server", "cpai", "external-llm"}
        assert [service.name for service in config.services.values()] == list(config.services)

    def test_eviction_command_timeout_default(self, tmp_path: Path) -> None:
        """eviction_command_timeout defaults to 60s when omitted."""
        config = _load(tmp_path, _document())
        assert config.eviction_command_timeout == _EVICTION_COMMAND_TIMEOUT_S

    def test_connection_timeout_default(self, tmp_path: Path) -> None:
        """connection_timeout defaults to 30s when omitted."""
        config = _load(tmp_path, _document())
        assert config.connection_timeout == _CONNECTION_TIMEOUT_S

    def test_negative_priority_rejected(self, tmp_path: Path) -> None:
        """A service with a negative priority is rejected."""
        document = _document(
            services={"svc1": _service(priority=-1, vram_budget=_TINY_SERVICE_VRAM_MIB)}
        )
        with pytest.raises(ConfigError, match="priority must be > 0"):
            _load(tmp_path, document)

    def test_empty_hostname_rejected(self, tmp_path: Path) -> None:
        """A service with an empty hostname is rejected."""
        document = _document(services={"svc1": _service(hostname="")})
        with pytest.raises(ConfigError, match="hostname must be a non-empty string"):
            _load(tmp_path, document)

    def test_whitespace_hostname_rejected(self, tmp_path: Path) -> None:
        """A hostname of only whitespace is rejected, not silently stripped."""
        document = _document(services={"svc1": _service(hostname="   ")})
        with pytest.raises(ConfigError, match="hostname must be a non-empty string"):
            _load(tmp_path, document)

    def test_malformed_yaml_rejected(self, tmp_path: Path) -> None:
        """Syntactically invalid YAML is reported as a config error, not a parser crash."""
        with pytest.raises(ConfigError, match="Invalid YAML"):
            _load_text(tmp_path, "this: is: not: valid: yaml: [[[")

    def test_missing_hostname_rejected(self, tmp_path: Path) -> None:
        """A service without a hostname is rejected, naming the field it lacks."""
        service = _service()
        service.pop("hostname")
        document = _document(services={"svc1": service})
        with pytest.raises(ConfigError, match="missing required field: hostname"):
            _load(tmp_path, document)

    def test_missing_container_rejected(self, tmp_path: Path) -> None:
        """A local service without a container is rejected, naming the field it lacks.

        An empty container is present rather than missing, so it loads; and it
        loads as a local service, because only the `external` flag makes a
        service external, never the absence of a container.
        """
        service = _service()
        service.pop("container")
        document = _document(services={"svc1": service})
        with pytest.raises(ConfigError, match="missing required field: container"):
            _load(tmp_path, document)
        empty = _document(services={"svc1": _service(container="")})
        service = _load(tmp_path, empty).services["svc1"]
        assert service.container == ""
        assert service.external is False

    def test_missing_backup_accepted(self, tmp_path: Path) -> None:
        """A service may omit backup, which loads as None rather than failing."""
        document = _document(
            services={"svc1": _service(vram_budget=_TINY_SERVICE_VRAM_MIB, backup=None)}
        )
        config = _load(tmp_path, document)
        assert config.services["svc1"].backup is None

    def test_missing_gpu_routes_defaults_to_empty(self, tmp_path: Path) -> None:
        """An omitted gpu_routes defaults to empty, making the service passthrough-only."""
        document = _document(
            services={
                "svc1": _service(vram_budget=_TINY_SERVICE_VRAM_MIB, gpu_routes=None)
            }
        )
        config = _load(tmp_path, document)
        assert config.services["svc1"].gpu_routes == []

    def test_invalid_url_for_target_rejected(self, tmp_path: Path) -> None:
        """A target that is not a URL the proxy can dial is rejected, quoting the value."""
        document = _document(services={"svc1": _service(target=_INVALID_URL)})
        with pytest.raises(ConfigError, match=f"target is not a valid URL: {_INVALID_URL}"):
            _load(tmp_path, document)
        hostless = _document(services={"svc1": _service(target=_HOSTLESS_URL)})
        with pytest.raises(ConfigError, match="target is not a valid URL"):
            _load(tmp_path, hostless)
        wrong_scheme = _document(services={"svc1": _service(target=_NON_HTTP_URL)})
        with pytest.raises(ConfigError, match="target is not a valid URL"):
            _load(tmp_path, wrong_scheme)

    def test_invalid_url_for_backup_rejected(self, tmp_path: Path) -> None:
        """A backup that is not a URL is rejected, quoting the value."""
        document = _document(services={"svc1": _service(backup=_INVALID_URL)})
        with pytest.raises(ConfigError, match=f"backup is not a valid URL: {_INVALID_URL}"):
            _load(tmp_path, document)

    def test_vram_budget_exactly_equals_total_budget(self, tmp_path: Path) -> None:
        """A service may claim the whole GPU: the budget check is inclusive."""
        document = _document(services={"svc1": _service(vram_budget=_TOTAL_VRAM_MIB)})
        config = _load(tmp_path, document)
        assert config.services["svc1"].vram_budget == _TOTAL_VRAM_MIB

    def test_duplicate_service_names_rejected(self, tmp_path: Path) -> None:
        """A name defined twice collapses to one service, the later definition winning.

        YAML mappings cannot hold a repeated key, so the config format makes a
        duplicate name unrepresentable rather than rejecting it downstream.
        """
        def block(hostname: str) -> str:
            return indent(yaml.safe_dump({"svc1": _service(hostname=hostname)}), "  ")

        base = _document()
        base.pop("services")
        text = (
            yaml.safe_dump(base)
            + "services:\n"
            + block(hostname="first.local")
            + block(hostname="second.local")
        )
        config = _load_text(tmp_path, text)
        assert set(config.services) == {"svc1"}
        assert config.services["svc1"].hostname == "second.local"
        assert set(config.hostname_map) == {"second.local"}

    def test_wrong_type_for_vram_budget_rejected(self, tmp_path: Path) -> None:
        """A vram_budget that is not an integer is rejected, and nor is a bool one.

        Python makes `bool` a subclass of `int`, so `vram_budget: true` satisfies
        a plain integer check and would claim a budget of one MiB.
        """
        service = _service()
        service["vram_budget"] = "not_a_number"
        document = _document(services={"svc1": service})
        with pytest.raises(ConfigError, match="vram_budget must be an integer"):
            _load(tmp_path, document)
        boolean_service = _service()
        boolean_service["vram_budget"] = True
        boolean = _document(services={"svc1": boolean_service})
        with pytest.raises(ConfigError, match="vram_budget must be an integer"):
            _load(tmp_path, boolean)

    def test_wrong_type_for_priority_rejected(self, tmp_path: Path) -> None:
        """A priority that is not an integer is rejected, and nor is a bool one.

        Python makes `bool` a subclass of `int`, so `priority: true` satisfies a
        plain integer check and would rank the service at priority one.
        """
        service = _service()
        service["priority"] = "high"
        document = _document(services={"svc1": service})
        with pytest.raises(ConfigError, match="priority must be an integer"):
            _load(tmp_path, document)
        boolean_service = _service()
        boolean_service["priority"] = True
        boolean = _document(services={"svc1": boolean_service})
        with pytest.raises(ConfigError, match="priority must be an integer"):
            _load(tmp_path, boolean)

    def test_external_service_no_container(self, tmp_path: Path) -> None:
        """An external service needs no container and holds no GPU claim."""
        config = _load(tmp_path, _document(services={"ext": _external_service()}))
        service = config.services["ext"]
        assert service.external is True
        assert service.vram_budget == 0
        assert service.container == ""
        assert service.priority == 0

    def test_external_service_banned_priority(self, tmp_path: Path) -> None:
        """An external service may not claim a priority."""
        document = _document(services={"ext": _external_service(priority=1)})
        with pytest.raises(ConfigError, match="priority is banned for external services"):
            _load(tmp_path, document)

    def test_external_service_banned_budget(self, tmp_path: Path) -> None:
        """An external service may not claim a VRAM budget."""
        document = _document(
            services={"ext": _external_service(vram_budget=_TINY_SERVICE_VRAM_MIB)}
        )
        with pytest.raises(
            ConfigError, match="vram_budget is banned for external services"
        ):
            _load(tmp_path, document)

    def test_health_check_fields_loaded(self, tmp_path: Path) -> None:
        """A service's health check command and interval are carried through.

        A service that asks for no health check is left with none, so the
        command is carried through in both directions rather than invented.
        """
        document = _document(
            services={
                "svc1": _service(
                    vram_budget=_TINY_SERVICE_VRAM_MIB,
                    health_check_command=_HEALTH_CHECK_COMMAND,
                    health_check_interval=_HEALTH_CHECK_INTERVAL_S,
                )
            }
        )
        config = _load(tmp_path, document)
        service = config.services["svc1"]
        assert service.health_check_command == _HEALTH_CHECK_COMMAND
        assert service.health_check_interval == _HEALTH_CHECK_INTERVAL_S
        without_interval = _document(
            services={"svc1": _service(health_check_command=_HEALTH_CHECK_COMMAND)}
        )
        defaulted = _load(tmp_path, without_interval).services["svc1"]
        assert defaulted.health_check_interval == _DEFAULT_HEALTH_CHECK_INTERVAL_S
        unchecked = _load(tmp_path, _document()).services["svc1"]
        assert unchecked.health_check_command is None

    def test_external_service_banned_container(self, tmp_path: Path) -> None:
        """An external service may not name a container."""
        document = _document(
            services={"ext": _external_service(container="banned-cont")}
        )
        with pytest.raises(ConfigError, match="container is banned for external services"):
            _load(tmp_path, document)

    def test_multiple_external_services_allowed(self, tmp_path: Path) -> None:
        """External services all sit at priority 0, so several may coexist."""
        services = {
            "ext1": _external_service(hostname="ext1.local"),
            "ext2": _external_service(
                hostname="ext2.local", target="http://127.0.0.1:8002"
            ),
        }
        config = _load(tmp_path, _document(services=services))
        assert config.services["ext1"].external is True
        assert config.services["ext2"].external is True
        assert config.services["ext1"].priority == 0
        assert config.services["ext2"].priority == 0

    def test_health_check_interval_zero_rejected(self, tmp_path: Path) -> None:
        """A health check interval of zero is rejected."""
        document = _document(
            services={
                "svc1": _service(
                    health_check_command=_HEALTH_CHECK_COMMAND, health_check_interval=0
                )
            }
        )
        with pytest.raises(ConfigError, match="health_check_interval must be >= 1"):
            _load(tmp_path, document)

    def test_health_check_interval_negative_rejected(self, tmp_path: Path) -> None:
        """A negative health check interval is rejected."""
        document = _document(
            services={
                "svc1": _service(
                    health_check_command=_HEALTH_CHECK_COMMAND,
                    health_check_interval=_NEGATIVE_HEALTH_CHECK_INTERVAL_S,
                )
            }
        )
        with pytest.raises(ConfigError, match="health_check_interval must be >= 1"):
            _load(tmp_path, document)

    def test_external_service_banned_gpu_routes(self, tmp_path: Path) -> None:
        """An external service may not declare GPU routes."""
        document = _document(
            services={"ext": _external_service(gpu_routes=["/v1/completions"])}
        )
        with pytest.raises(ConfigError, match="gpu_routes is banned for external services"):
            _load(tmp_path, document)


class TestPluginConfig:
    """Tests for plugin configuration parsing."""

    def test_global_plugins_section_parses(self, tmp_path: Path) -> None:
        """Every global plugin keeps its name, type, and config, in order."""
        plugins = [
            {"name": "auth_plugin", "type": "AuthPlugin", "config": {"api_key": "secret123"}},
            {
                "name": "rate_limiter",
                "type": "RateLimitPlugin",
                "config": {"max_requests": 100},
            },
        ]
        plugins.append({"name": "bare", "type": "BarePlugin"})
        config = _load(tmp_path, _document(plugins=plugins))
        assert config.plugins[:2] == plugins[:2]
        assert config.plugins[2] == {"name": "bare", "type": "BarePlugin", "config": {}}

    def test_service_level_plugins_list_parses(self, tmp_path: Path) -> None:
        """A service's own plugin list is kept in the order it was written."""
        document = _document(
            services={"svc1": _service(plugins=["auth_plugin", "rate_limiter"])}
        )
        config = _load(tmp_path, document)
        assert config.services["svc1"].plugins == ["auth_plugin", "rate_limiter"]

    def test_config_without_plugins_defaults_empty(self, tmp_path: Path) -> None:
        """With no plugins named anywhere, both plugin lists load empty."""
        config = _load(tmp_path, _document())
        assert config.plugins == []
        assert config.services["svc1"].plugins == []

    @pytest.mark.parametrize("missing_field", ["name", "type"])
    def test_missing_required_plugin_field_raises(
        self, tmp_path: Path, missing_field: str
    ) -> None:
        """A plugin entry missing name or type is rejected, naming the field."""
        plugin = {"name": "my_plugin", "type": "MyPlugin", "config": {}}
        del plugin[missing_field]
        with pytest.raises(
            ConfigError, match=f"Plugin entry missing required field: {missing_field}"
        ):
            _load(tmp_path, _document(plugins=[plugin]))

    def test_duplicate_plugin_name_raises(self, tmp_path: Path) -> None:
        """Two plugin instances sharing a name are rejected."""
        plugins = [
            {"name": "my_plugin", "type": "MyPlugin", "config": {}},
            {"name": "my_plugin", "type": "OtherPlugin", "config": {}},
        ]
        with pytest.raises(ConfigError, match="Duplicate plugin name: my_plugin"):
            _load(tmp_path, _document(plugins=plugins))

    @pytest.mark.parametrize("cfg_type", ["string", "int", "list"])
    def test_plugin_config_must_be_dict(self, tmp_path: Path, cfg_type: str) -> None:
        """A plugin config that is not a mapping is rejected."""
        configs: dict[str, str | int | list] = {
            "string": "just_a_string",
            "int": 42,
            "list": [1, 2, 3],
        }
        plugin = {"name": "my_plugin", "type": "MyPlugin", "config": configs[cfg_type]}
        with pytest.raises(ConfigError, match="config must be a dict"):
            _load(tmp_path, _document(plugins=[plugin]))

    def test_plugins_must_be_list(self, tmp_path: Path) -> None:
        """A plugins section that is not a list is rejected."""
        with pytest.raises(ConfigError, match="plugins must be a list"):
            _load(tmp_path, _document(plugins="string"))

    def test_plugin_entry_must_be_mapping(self, tmp_path: Path) -> None:
        """A plugin entry that is a bare string rather than a mapping is rejected."""
        with pytest.raises(ConfigError, match="Each plugin entry must be a mapping"):
            _load(tmp_path, _document(plugins=["string_entry"]))

    @pytest.mark.parametrize("field,value", [
        ("name", "123"),
        ("type", "42"),
        ("name", "true"),
        ("type", "true"),
    ])
    def test_plugin_name_and_type_must_be_strings(
        self, tmp_path: Path, field: str, value: str
    ) -> None:
        """A plugin name or type that YAML reads as a number or bool is rejected."""
        plugin = {"name": "valid_name", "type": "ValidType", "config": {}}
        plugin[field] = yaml.safe_load(value)
        assert not isinstance(plugin[field], str)
        with pytest.raises(ConfigError, match=f"Plugin {field} must be a string"):
            _load(tmp_path, _document(plugins=[plugin]))

    @pytest.mark.parametrize("plugins_yaml", [
        "plugins: [123]",
        "plugins: [null]",
        "plugins: [true]",
    ])
    def test_service_plugins_non_string_entries_rejected(
        self, tmp_path: Path, plugins_yaml: str
    ) -> None:
        """A service plugin list holding anything but strings is rejected."""
        service = _service()
        service["plugins"] = yaml.safe_load(plugins_yaml)["plugins"]
        with pytest.raises(ConfigError, match="plugins entries must be strings"):
            _load(tmp_path, _document(services={"svc1": service}))

    def test_plugin_config_regex_backslashes_preserved(self, tmp_path: Path) -> None:
        """Single-quoted YAML keeps a regex's backslashes intact through loading.

        Written as literal YAML rather than dumped from a Python string, because
        the escaping this pins down is the YAML parser's, not the dumper's.
        """
        text = yaml.safe_dump(_document()) + """
plugins:
  - name: context_modifier
    type: OpenAIContextModifier
    config:
      tool_call_prune_exclusions:
        - tool_name_regex: '(?i)read'
          tool_args_regex: '(?i)(skill|agents)\\.md'
"""
        config = _load_text(tmp_path, text)
        exclusion = config.plugins[0]["config"]["tool_call_prune_exclusions"][0]
        assert exclusion["tool_name_regex"] == "(?i)read"
        assert exclusion["tool_args_regex"] == r"(?i)(skill|agents)\.md"


class TestPersistentFlag:
    """Loading and validating the per-service persistent flag."""

    def test_persistent_defaults_to_false(self, tmp_path: Path) -> None:
        """A service that omits persistent is not persistent."""
        config = _load(tmp_path, _document())
        assert config.services["svc1"].is_persistent is False

    def test_persistent_true_loaded(self, tmp_path: Path) -> None:
        """An explicit persistent: true is carried through."""
        document = _document(services={"svc1": _service(is_persistent=True)})
        config = _load(tmp_path, document)
        assert config.services["svc1"].is_persistent is True

    def test_persistent_wrong_type_rejected(self, tmp_path: Path) -> None:
        """A persistent value that is not a bool is rejected."""
        service = _service()
        service["is_persistent"] = "yes"
        document = _document(services={"svc1": service})
        with pytest.raises(ConfigError, match="is_persistent must be a boolean"):
            _load(tmp_path, document)


class TestFallbackOnError:
    """Loading and validating the per-service fallback_on_error flag."""

    def test_fallback_defaults_to_true(self, tmp_path: Path) -> None:
        """A service that omits fallback_on_error still retries on the backup."""
        config = _load(tmp_path, _document())
        assert config.services["svc1"].fallback_on_error is True

    def test_fallback_false_loaded(self, tmp_path: Path) -> None:
        """An explicit fallback_on_error: false is carried through."""
        document = _document(services={"svc1": _service(fallback_on_error=False)})
        config = _load(tmp_path, document)
        assert config.services["svc1"].fallback_on_error is False

    def test_fallback_true_loaded(self, tmp_path: Path) -> None:
        """An explicit fallback_on_error: true is carried through."""
        document = _document(services={"svc1": _service(fallback_on_error=True)})
        config = _load(tmp_path, document)
        assert config.services["svc1"].fallback_on_error is True

    @pytest.mark.parametrize("bad_value", ["yes", 1])
    def test_fallback_wrong_type_rejected(
        self, tmp_path: Path, bad_value: str | int
    ) -> None:
        """A fallback_on_error value that is not a real bool is rejected."""
        service = _service()
        service["fallback_on_error"] = bad_value
        document = _document(services={"svc1": service})
        with pytest.raises(ConfigError, match="fallback_on_error must be a boolean"):
            _load(tmp_path, document)
