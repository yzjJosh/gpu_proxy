"""Tests for the LazyStartContainer plugin."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import TracebackType
from typing import Callable

import pytest

from app.config import ServiceConfig
from app.plugins.lazy_start_container import LazyStartContainer

_VRAM_BUDGET_MIB = 1024
_SHORT_START_TIMEOUT = 0.05


def _service(container: str = "lazy-svc") -> ServiceConfig:
    """A minimal service config for building the plugin against."""
    return ServiceConfig(
        name="lazy",
        priority=1,
        vram_budget=_VRAM_BUDGET_MIB,
        hostname="lazy.local",
        target="http://127.0.0.1:9000/lazy",
        container=container,
    )


def _started_ago(seconds: float) -> str:
    """An RFC3339 start timestamp seconds before now."""
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()


def _state(
    running: bool, health: str | None = None, started_at: str | None = None
) -> dict[str, dict]:
    """One aiodocker show() payload with the given state and health status."""
    state: dict[str, bool | str | dict] = {"Running": running}
    if health is not None:
        state["Health"] = {"Status": health}
    if started_at is not None:
        state["StartedAt"] = started_at
    return {"State": state}


class FakeDockerContainer:
    """A scriptable container: show() walks a state list, start() counts."""

    def __init__(self, states: list[dict[str, dict]]) -> None:
        self._states = list(states)
        self.shows: int = 0
        self.starts: int = 0

    async def show(self) -> dict[str, dict]:
        self.shows += 1
        current = self._states[0]
        if len(self._states) > 1:
            self._states.pop(0)
        return current

    async def start(self) -> None:
        self.starts += 1


class FakeDockerContainers:
    """Hands out the single container, or raises the configured error."""

    def __init__(
        self,
        container: FakeDockerContainer | None,
        error: Exception | None = None,
    ) -> None:
        self._container = container
        self._error = error

    async def get(self, _name: str) -> FakeDockerContainer:
        if self._error is not None:
            raise self._error
        if self._container is None:
            raise RuntimeError("fake docker has no container configured")
        return self._container


class FakeDocker:
    """Stands in for aiodocker.docker.Docker, which the plugin constructs itself."""

    def __init__(
        self,
        container: FakeDockerContainer | None,
        error: Exception | None = None,
        constructed: list[FakeDocker] | None = None,
    ) -> None:
        self.containers = FakeDockerContainers(container, error)
        if constructed is not None:
            constructed.append(self)

    async def __aenter__(self) -> FakeDocker:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        return False


def _patch_docker(
    monkeypatch: pytest.MonkeyPatch,
    container: FakeDockerContainer | None = None,
    error: Exception | None = None,
    constructed: list[FakeDocker] | None = None,
) -> None:
    """Point the plugin's Docker at a fake factory."""
    factory: Callable[[], FakeDocker] = lambda: FakeDocker(
        container, error, constructed
    )
    monkeypatch.setattr("app.plugins.lazy_start_container.Docker", factory)


async def _prepare_stopped(
    monkeypatch: pytest.MonkeyPatch, after_start: dict[str, dict]
) -> FakeDockerContainer:
    """Run prepare_target once: the container is stopped, then reports after_start."""
    container = FakeDockerContainer([_state(False), after_start])
    _patch_docker(monkeypatch, container)
    plugin = LazyStartContainer({}, _service())
    await plugin.prepare_target(1, "/v1/models")
    return container


class TestLazyStartContainer:
    """prepare_target brings a stopped container up before traffic reaches it."""

    async def test_ready_container_needs_no_start(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A running, healthy container passes through untouched."""
        container = FakeDockerContainer([_state(True, "healthy")])
        _patch_docker(monkeypatch, container)
        plugin = LazyStartContainer({}, _service())
        await plugin.prepare_target(1, "/v1/models")
        assert container.starts == 0
        assert container.shows == 1

    async def test_running_container_without_healthcheck_passes_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without a healthcheck, running is all that is required."""
        container = FakeDockerContainer([_state(True)])
        _patch_docker(monkeypatch, container)
        plugin = LazyStartContainer({}, _service())
        await plugin.prepare_target(1, "/v1/models")
        assert container.starts == 0

    async def test_stopped_container_is_started_and_waited_for(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A stopped container is started, then polled until it reports healthy."""
        container = FakeDockerContainer(
            [
                _state(False),
                _state(True, "starting", _started_ago(0.0)),
                _state(True, "healthy", _started_ago(0.0)),
            ]
        )
        _patch_docker(monkeypatch, container)
        plugin = LazyStartContainer({}, _service())
        await plugin.prepare_target(1, "/v1/models")
        assert container.starts == 1
        assert container.shows == 3

    async def test_stopped_container_without_healthcheck_starts_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A healthcheck-less container only needs to reach the running state."""
        container = await _prepare_stopped(monkeypatch, _state(True))
        assert container.starts == 1
        assert container.shows == 2

    async def test_timeout_gives_up_and_returns(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A container whose start attempt is already past the timeout is abandoned."""
        container = FakeDockerContainer(
            [_state(False), _state(True, "starting", _started_ago(1.0))]
        )
        _patch_docker(monkeypatch, container)
        plugin = LazyStartContainer({"start_timeout": _SHORT_START_TIMEOUT}, _service())
        await plugin.prepare_target(1, "/v1/models")
        assert container.starts == 1
        assert container.shows == 2

    async def test_followup_after_deadline_passes_through_without_waiting(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Once the global deadline elapses, later requests forward without waiting."""
        container = FakeDockerContainer(
            [_state(False), _state(True, "starting", _started_ago(1.0))]
        )
        _patch_docker(monkeypatch, container)
        plugin = LazyStartContainer({"start_timeout": _SHORT_START_TIMEOUT}, _service())
        await plugin.prepare_target(1, "/v1/models")
        await plugin.prepare_target(2, "/v1/models")
        assert container.starts == 1
        assert container.shows == 3

    async def test_running_unhealthy_with_aged_start_passes_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An old start attempt gets no fresh wait: the request forwards at once."""
        container = FakeDockerContainer(
            [_state(True, "starting", _started_ago(60.0))]
        )
        _patch_docker(monkeypatch, container)
        plugin = LazyStartContainer({"start_timeout": _SHORT_START_TIMEOUT}, _service())
        await plugin.prepare_target(1, "/v1/models")
        assert container.starts == 0
        assert container.shows == 1

    async def test_new_stop_restarts_and_waits_again(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A container observed stopped again is a new attempt with a fresh start."""
        container = FakeDockerContainer(
            [
                _state(False),
                _state(True, "starting", _started_ago(1.0)),
                _state(False),
                _state(True, "starting", _started_ago(1.0)),
            ]
        )
        _patch_docker(monkeypatch, container)
        plugin = LazyStartContainer({"start_timeout": _SHORT_START_TIMEOUT}, _service())
        await plugin.prepare_target(1, "/v1/models")
        await plugin.prepare_target(2, "/v1/models")
        assert container.starts == 2
        assert container.shows == 4

    async def test_later_stop_waits_again(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After a ready container is stopped again, lazy start works fully again."""
        container = FakeDockerContainer(
            [
                _state(False),
                _state(True, "healthy", _started_ago(1.0)),
                _state(False),
                _state(True, "healthy", _started_ago(1.0)),
            ]
        )
        _patch_docker(monkeypatch, container)
        plugin = LazyStartContainer({}, _service())
        await plugin.prepare_target(1, "/v1/models")
        await plugin.prepare_target(2, "/v1/models")
        assert container.starts == 2
        assert container.shows == 4

    async def test_running_unhealthy_without_attempt_waits_out_deadline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A container that just started but is unhealthy is waited on, not started."""
        container = FakeDockerContainer(
            [_state(True, "starting", _started_ago(0.0))]
        )
        _patch_docker(monkeypatch, container)
        plugin = LazyStartContainer({"start_timeout": _SHORT_START_TIMEOUT}, _service())
        await plugin.prepare_target(1, "/v1/models")
        assert container.starts == 0

    @pytest.mark.parametrize(
        "started_at",
        [
            pytest.param(None, id="missing-start-time"),
            pytest.param("not-a-timestamp", id="unparseable-start-time"),
        ],
    )
    async def test_no_readable_start_time_forwards_immediately(
        self, monkeypatch: pytest.MonkeyPatch, started_at: str | None
    ) -> None:
        """With no usable start time there is no deadline, so it forwards at once."""
        container = await _prepare_stopped(
            monkeypatch, _state(True, "starting", started_at)
        )
        assert container.starts == 1
        assert container.shows == 2

    async def test_concurrent_calls_start_the_container_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Racing requests serialize on one start; the latecomer re-checks state."""
        container = FakeDockerContainer(
            [_state(False), _state(True, "healthy", _started_ago(0.0))]
        )
        _patch_docker(monkeypatch, container)
        plugin = LazyStartContainer({}, _service())
        await asyncio.gather(
            plugin.prepare_target(1, "/v1/models"),
            plugin.prepare_target(2, "/v1/models"),
        )
        assert container.starts == 1
        assert container.shows == 3

    async def test_missing_service_config_is_a_noop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A plugin built without a service config never touches docker."""
        constructed: list[FakeDocker] = []
        _patch_docker(monkeypatch, constructed=constructed)
        plugin = LazyStartContainer({})
        await plugin.prepare_target(1, "/v1/models")
        assert constructed == []

    async def test_empty_container_name_is_a_noop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A service with no container (e.g. external) never touches docker."""
        constructed: list[FakeDocker] = []
        _patch_docker(monkeypatch, constructed=constructed)
        plugin = LazyStartContainer({}, _service(container=""))
        await plugin.prepare_target(1, "/v1/models")
        assert constructed == []

    async def test_docker_errors_propagate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A docker failure propagates so the proxy can log and forward anyway."""
        _patch_docker(monkeypatch, error=RuntimeError("docker down"))
        plugin = LazyStartContainer({}, _service())
        with pytest.raises(RuntimeError, match="docker down"):
            await plugin.prepare_target(1, "/v1/models")


class TestLazyStartContainerName:
    """The plugin registers under a name the config can reference."""

    def test_name_defaults_to_its_own_identity(self) -> None:
        """Without a configured name the plugin identifies itself by slug."""
        plugin = LazyStartContainer({})
        assert plugin.name() == "lazy_start_container"

    def test_name_follows_the_config(self) -> None:
        """A configured name overrides the default."""
        plugin = LazyStartContainer({"name": "lazy-start"})
        assert plugin.name() == "lazy-start"
