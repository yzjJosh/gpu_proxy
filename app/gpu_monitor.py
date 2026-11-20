"""Check the GPU and the VRAM of each container again and again."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from app.gpu_usage import (
    ContainerUsage,
    GpuProcess,
    GpuStats,
    GpuUsageError,
    GpuUsageReader,
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
    processes: list[GpuProcess] = field(default_factory=list)


@dataclass
class GPUStatus:
    """Current GPU status snapshot."""

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
    stats: GpuStats,
    containers: list[ContainerUsage],
    clock: ClockProtocol,
) -> GPUStatus:
    """Turn a GPU usage reading into a GPU status."""
    docker_vram_mib = sum(c.vram_mib for c in containers)
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
                vram_mib=c.vram_mib,
                gpu_util_pct=c.gpu_utilization_pct,
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
        self._reader: Optional[GpuUsageReader] = None

    async def start(self) -> None:
        """Start the background polling task."""
        self._reader = GpuUsageReader()
        try:
            await self._reader.open()
        except GpuUsageError:
            _logger.exception("Failed to initialize GPU monitor")
            self._reader = None
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
        if self._reader is not None:
            await self._reader.close()
            self._reader = None

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
        """Run a single GPU poll."""
        if self._reader is None:
            status = GPUStatus(poll_failed=True)
        else:
            try:
                stats, containers = await self._reader.read()
                status = _convert_stats(stats, containers, self._clock)
            except Exception:
                _logger.warning("GPU poll failed", exc_info=True)
                status = GPUStatus(poll_failed=True)
        if self._on_update:
            self._on_update(status)
        return status
