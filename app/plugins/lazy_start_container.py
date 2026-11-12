"""Start a service's container on demand before traffic reaches it."""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import TypedDict, override

from aiodocker.docker import Docker

from app.config import ServiceConfig
from app.plugins.base import PluginBase

_logger = logging.getLogger(__name__)

__all__ = ["LazyStartContainer"]

_DEFAULT_START_TIMEOUT = 120.0
_HEALTH_POLL_INTERVAL = 1.0


class _ContainerState(TypedDict, total=False):
    Running: bool
    StartedAt: str
    Health: dict[str, str | int]


class _ContainerInfo(TypedDict):
    State: _ContainerState


def _is_ready(data: _ContainerInfo) -> bool:
    """A container is ready when it runs and, if health-checked, reports healthy."""
    state = data["State"]
    if not state["Running"]:
        return False
    health = state.get("Health")
    return health is None or health.get("Status") == "healthy"


def _ready_deadline(data: _ContainerInfo, timeout: float) -> datetime | None:
    """End of the container's current start attempt, from its last start time."""
    started_at = data["State"].get("StartedAt")
    if not isinstance(started_at, str):
        return None
    try:
        return datetime.fromisoformat(started_at) + timedelta(seconds=timeout)
    except ValueError:
        return None


# The plugin manager loads this module with spec.loader.exec_module and finds the
# class by name from config, so no production file imports it.
# lint-skip: coding.exported-api-unused
class LazyStartContainer(PluginBase):
    """Brings the service's container up before traffic is dispatched to it.

    prepare_target passes through when the container already runs (and is
    healthy, when it has a healthcheck). Otherwise the container is started
    and polled until it is ready or the global start deadline elapses. The
    deadline is the container's last start time plus start_timeout, so it is
    shared by every request: follow-ups wait only the remaining time and
    pass straight through once it has passed.
    """

    def __init__(
        self, config: dict, service_config: ServiceConfig | None = None
    ) -> None:
        super().__init__(config, service_config)
        self._start_timeout: float = float(
            config.get("start_timeout", _DEFAULT_START_TIMEOUT)
        )
        self._lock = asyncio.Lock()

    @override
    def name(self) -> str:
        return self.config.get("name", "lazy_start_container")

    @override
    async def prepare_target(self, request_id: int, _path: str) -> None:
        service = self.service_config
        if service is None or not service.container:
            return
        async with self._lock:
            await self._ensure_running(service.container, request_id)

    async def _ensure_running(self, container: str, request_id: int) -> None:
        async with Docker() as docker:
            target = await docker.containers.get(container)
            data = await target.show()
            if _is_ready(data):
                return
            if not data["State"]["Running"]:
                _logger.info(
                    "Starting container %s for request %d", container, request_id
                )
                await target.start()
                data = await target.show()
            deadline = _ready_deadline(data, self._start_timeout)
            if deadline is None:
                _logger.warning(
                    "Container %s has no readable start time; forwarding anyway",
                    container,
                )
                return
            while not _is_ready(data):
                if datetime.now(timezone.utc) >= deadline:
                    _logger.warning(
                        "Container %s not ready within %ss of its start;"
                        " forwarding anyway",
                        container,
                        self._start_timeout,
                    )
                    return
                await asyncio.sleep(_HEALTH_POLL_INTERVAL)
                data = await target.show()
