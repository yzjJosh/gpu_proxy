"""GPU monitor: background coroutine that polls GPU status via nvidia_docker_usage library."""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

_NVIDIA_DOCKER_USAGE_PATH = "/usr/local/bin"

sys.path.insert(0, str(Path(_NVIDIA_DOCKER_USAGE_PATH).resolve()))
from nvidia_docker_usage import (
    GpuUsageMonitor as LibGpuUsageMonitor,
    GpuStats as LibGpuStats,
    ContainerUsage as LibContainerUsage,
)

_logger = logging.getLogger("gpu_monitor")

__all__ = ["ClockProtocol", "ContainerVRAM", "GPUStatus", "GpuMonitor", "RealClock"]


class ClockProtocol:
    """Protocol for injectable clocks used in timeout and scheduling logic."""

    def time(self) -> float:
        """Return current time in seconds (Unix epoch for production, virtual for tests)."""
        raise NotImplementedError

    async def sleep(self, _seconds: float) -> None:
        """Sleep for the given number of seconds (may be virtual in tests)."""
        raise NotImplementedError


class RealClock(ClockProtocol):
    """Default clock using real asyncio time and sleep."""

    def time(self) -> float:
        """Return current wall-clock time as Unix epoch seconds."""
        return time.time()

    async def sleep(self, seconds: float) -> None:
        """Sleep for real seconds."""
        await asyncio.sleep(seconds)


@dataclass
class ContainerVRAM:
    """VRAM usage for a single container."""

    vram_mib: int = 0
    gpu_util_pct: int = 0
    processes: list = field(default_factory=list)


@dataclass
class GPUStatus:
    """Current GPU status snapshot from nvidia_docker_usage.py."""

    available_vram_mib: int = 0
    total_vram_mib: int = 0
    system_vram_mib: int = 0
    docker_vram_mib: int = 0
    gpu_util_pct: int = 0
    power_usage_w: float = 0.0
    power_limit_w: float = 0.0
    temperature_c: int = 0
    gr_clock_mhz: int = 0
    mem_clock_mhz: int = 0
    pstate: str = ""
    containers: dict[str, ContainerVRAM] = field(default_factory=dict)
    last_updated: Optional[float] = None
    poll_failed: bool = True


def _convert_stats(
    stats: LibGpuStats,
    containers: list[LibContainerUsage],
    clock: ClockProtocol,
) -> GPUStatus:
    """Convert library dataclasses to our GPUStatus/ContainerVRAM types."""
    docker_vram_mib = sum(c.total_vram_mib for c in containers)
    system_vram_mib = stats.used_vram_mib - docker_vram_mib

    status = GPUStatus(
        available_vram_mib=stats.available_vram_mib,
        total_vram_mib=stats.total_vram_mib,
        system_vram_mib=system_vram_mib,
        docker_vram_mib=docker_vram_mib,
        gpu_util_pct=stats.gpu_utilization_pct,
        power_usage_w=stats.power_draw_w,
        power_limit_w=stats.power_limit_w,
        temperature_c=stats.temperature_c,
        gr_clock_mhz=stats.gr_clock_mhz,
        mem_clock_mhz=stats.mem_clock_mhz,
        pstate=stats.pstate,
        containers={
            c.name: ContainerVRAM(
                vram_mib=c.total_vram_mib,
                gpu_util_pct=c.total_gpu_utilization_pct,
                processes=c.processes,
            )
            for c in containers
        },
        last_updated=clock.time(),
        poll_failed=False,
    )
    return status


class GpuMonitor:
    """Background coroutine that periodically polls GPU status."""

    def __init__(
        self,
        clock: ClockProtocol,
        poll_interval: int = 5,
        on_update: Optional[Callable[[GPUStatus], None]] = None,
    ) -> None:
        self._poll_interval = poll_interval
        self._on_update = on_update
        self._clock = clock
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._monitor: Optional[LibGpuUsageMonitor] = None

    async def start(self) -> None:
        """Start the background polling task."""
        try:
            self._monitor = LibGpuUsageMonitor()
            await self._monitor.init()
        except Exception:
            _logger.exception("Failed to initialize GPU monitor")
            self._monitor = None
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        """Stop the background polling task."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        if self._monitor is not None:
            await self._monitor.close()
            self._monitor = None

    async def _poll_loop(self) -> None:
        """Main polling loop."""
        while self._running:
            try:
                await self.poll_once()
            except asyncio.CancelledError:
                break
            except Exception:
                pass
            try:
                await self._clock.sleep(self._poll_interval)
            except asyncio.CancelledError:
                break

    async def poll_once(self) -> GPUStatus:
        """Run a single GPU poll via the nvidia_docker_usage library."""
        if self._monitor is None:
            status = GPUStatus(poll_failed=True)
        else:
            try:
                stats = await self._monitor.get_stats()
                containers = await self._monitor.get_container_usage()
                status = _convert_stats(stats, containers, self._clock)
            except Exception:
                _logger.warning("GPU poll failed", exc_info=True)
                status = GPUStatus(poll_failed=True)
        if self._on_update:
            self._on_update(status)
        return status
