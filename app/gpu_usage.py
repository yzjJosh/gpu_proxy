"""Read memory and load for one GPU and for each Docker container on it."""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import aiodocker
import pynvml

_logger = logging.getLogger("gpu_usage")

__all__ = ["ContainerUsage", "GpuProcess", "GpuStats", "GpuUsageError", "GpuUsageReader"]

_MIB = 1024 * 1024
_MW_PER_W = 1000
_ALL_UTILIZATION_SAMPLES = 0
_PROC_ROOT = Path("/proc")
_CONTAINER_ID_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass
class GpuStats:
    """GPU wide readings from one read."""

    total_vram_mib: int
    used_vram_mib: int
    available_vram_mib: int
    gpu_utilization_pct: int
    power_draw_w: float
    power_limit_w: float
    gr_clock_mhz: int
    mem_clock_mhz: int
    pstate: str
    temperature_c: int


@dataclass
class GpuProcess:
    """One process in a container that holds GPU memory."""

    pid: int
    command: str
    vram_mib: int
    sm_utilization_pct: int


@dataclass
class ContainerUsage:
    """GPU use of one Docker container, added up over its processes."""

    name: str
    processes: list[GpuProcess]
    vram_mib: int
    gpu_utilization_pct: int


class GpuUsageError(Exception):
    """Reading the GPU or Docker failed."""


@dataclass
class _DeviceProcess:
    """One process the driver lists on the GPU."""

    pid: int
    vram_bytes: Optional[int]
    sm_utilization_pct: int


class GpuUsageReader:
    """Read one GPU and split its memory among Docker containers.

    A container gets the memory the driver reports for its own processes and
    nothing more. Memory the driver gives to no process stays out of every container.
    """

    def __init__(self, gpu_index: int = 0) -> None:
        self._gpu_index = gpu_index
        self._handle: Optional[pynvml.c_nvmlDevice_t] = None
        self._driver_lock = threading.Lock()

    async def open(self) -> None:
        """Connect to the GPU driver.

        Raises:
            GpuUsageError: If the driver cannot start or the GPU is not found.
        """
        try:
            await asyncio.to_thread(pynvml.nvmlInit)
        except pynvml.NVMLError as e:
            raise GpuUsageError(f"Failed to initialize NVML: {e}") from e
        try:
            self._handle = await asyncio.to_thread(
                pynvml.nvmlDeviceGetHandleByIndex, self._gpu_index
            )
        except pynvml.NVMLError as e:
            await asyncio.to_thread(_shutdown_nvml)
            raise GpuUsageError(
                f"Failed to get GPU handle for index {self._gpu_index}: {e}"
            ) from e

    async def close(self) -> None:
        """Disconnect from the GPU driver. Calling it again does nothing."""
        if self._handle is None:
            return
        self._handle = None
        await asyncio.to_thread(self._shutdown_after_reads)

    async def read(self) -> tuple[GpuStats, list[ContainerUsage]]:
        """Read the GPU and the GPU use of each running container.

        Returns:
            The GPU wide readings, and the containers that hold GPU memory,
            biggest first.

        Raises:
            GpuUsageError: If the reader is not open, the driver, the Docker
                daemon or a process file gives an error, or the driver does not
                say how much memory a process in a container holds.
        """
        if self._handle is None:
            raise GpuUsageError("GPU usage reader is not open")
        try:
            (stats, processes), container_names = await asyncio.gather(
                asyncio.to_thread(self._read_device_while_open),
                _list_container_names(),
            )
            containers = await asyncio.to_thread(
                _group_by_container, processes, container_names
            )
        except (pynvml.NVMLError, aiodocker.DockerError, OSError) as e:
            raise GpuUsageError(f"Failed to read GPU usage: {e}") from e
        return stats, containers

    def _read_device_while_open(self) -> tuple[GpuStats, list[_DeviceProcess]]:
        """Read the GPU, unless the reader was closed first.

        Raises:
            GpuUsageError: If the reader was closed before the read began.
        """
        with self._driver_lock:
            # The reader can be closed while this waits for the lock.
            if self._handle is None:
                raise GpuUsageError("GPU usage reader is not open")
            return _read_device(self._handle)

    def _shutdown_after_reads(self) -> None:
        """Disconnect from the GPU driver once no read is using it."""
        with self._driver_lock:
            _shutdown_nvml()


def _shutdown_nvml() -> None:
    """Disconnect from the GPU driver, logging a failure instead of raising it."""
    try:
        pynvml.nvmlShutdown()
    except pynvml.NVMLError:
        _logger.warning("NVML shutdown failed", exc_info=True)


def _read_device(
    handle: pynvml.c_nvmlDevice_t,
) -> tuple[GpuStats, list[_DeviceProcess]]:
    """Read the GPU memory and the processes on the GPU.

    Args:
        handle: The GPU to read.

    Returns:
        The GPU wide readings and every process on the GPU.
    """
    memory = pynvml.nvmlDeviceGetMemoryInfo(handle, version=pynvml.nvmlMemory_v2)
    processes = _list_device_processes(handle)
    stats = GpuStats(
        total_vram_mib=memory.total // _MIB,
        used_vram_mib=round(memory.used / _MIB),
        available_vram_mib=memory.free // _MIB,
        gpu_utilization_pct=_read_optional(
            lambda: pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        ),
        power_draw_w=pynvml.nvmlDeviceGetPowerUsage(handle) / _MW_PER_W,
        power_limit_w=_read_optional(
            lambda: pynvml.nvmlDeviceGetPowerManagementLimit(handle)
        ) / _MW_PER_W,
        gr_clock_mhz=_read_optional(
            lambda: pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
        ),
        mem_clock_mhz=_read_optional(
            lambda: pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
        ),
        pstate=f"P{_read_optional(lambda: pynvml.nvmlDeviceGetPowerState(handle))}",
        temperature_c=_read_optional(
            lambda: pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        ),
    )
    return stats, processes


def _read_optional(query: Callable[[], int]) -> int:
    """Read a number some GPUs do not have.

    Args:
        query: Reads the number.

    Returns:
        The number, or 0 when the GPU does not have it.
    """
    try:
        return query()
    except pynvml.NVMLError:
        return 0


def _list_device_processes(handle: pynvml.c_nvmlDevice_t) -> list[_DeviceProcess]:
    """List each process on the GPU once, with the memory the driver gives it.

    Args:
        handle: The GPU to read.

    Returns:
        Every compute and graphics process on the GPU.
    """
    utilization = _read_process_utilization(handle)
    processes: dict[int, _DeviceProcess] = {}
    for list_processes in (
        pynvml.nvmlDeviceGetComputeRunningProcesses,
        pynvml.nvmlDeviceGetGraphicsRunningProcesses,
    ):
        for info in list_processes(handle):
            processes[info.pid] = _DeviceProcess(
                pid=info.pid,
                vram_bytes=info.usedGpuMemory,
                sm_utilization_pct=utilization.get(info.pid, 0),
            )
    return list(processes.values())


def _read_process_utilization(handle: pynvml.c_nvmlDevice_t) -> dict[int, int]:
    """Read how busy each process keeps the GPU.

    Args:
        handle: The GPU to read.

    Returns:
        Busy percent by process id. Empty when the GPU has nothing to report.
    """
    try:
        samples = pynvml.nvmlDeviceGetProcessUtilization(handle, _ALL_UTILIZATION_SAMPLES)
    except pynvml.NVMLError_NotFound:
        return {}
    except pynvml.NVMLError:
        _logger.warning("NVML error querying process utilization", exc_info=True)
        return {}
    return {sample.pid: sample.smUtil for sample in samples}


async def _list_container_names() -> dict[str, str]:
    """List the running Docker containers.

    Returns:
        Container name by full container id.
    """
    async with aiodocker.Docker() as docker:
        containers = await docker.containers.list()
    return {container.id: container["Names"][0].lstrip("/") for container in containers}


def _group_by_container(
    processes: list[_DeviceProcess], container_names: dict[str, str]
) -> list[ContainerUsage]:
    """Add up GPU use for each container from the processes inside it.

    Args:
        processes: Every process on the GPU.
        container_names: Container name by full container id.

    Returns:
        The containers that hold GPU memory, biggest first.

    Raises:
        GpuUsageError: If the driver does not say how much memory a process
            in a container holds.
    """
    container_processes: dict[str, list[GpuProcess]] = defaultdict(list)
    container_bytes: dict[str, int] = defaultdict(int)
    for process in processes:
        name = _find_container(process.pid, container_names)
        if name is None:
            continue
        if process.vram_bytes is None:
            raise GpuUsageError(
                f"NVML does not report GPU memory for pid {process.pid} "
                f"in container {name}"
            )
        container_bytes[name] += process.vram_bytes
        container_processes[name].append(GpuProcess(
            pid=process.pid,
            command=_read_command(process.pid),
            vram_mib=process.vram_bytes // _MIB,
            sm_utilization_pct=process.sm_utilization_pct,
        ))
    usages = [
        ContainerUsage(
            name=name,
            processes=members,
            vram_mib=container_bytes[name] // _MIB,
            gpu_utilization_pct=sum(p.sm_utilization_pct for p in members),
        )
        for name, members in container_processes.items()
    ]
    usages.sort(key=lambda usage: usage.vram_mib, reverse=True)
    return usages


def _find_container(pid: int, container_names: dict[str, str]) -> Optional[str]:
    """Find the running container a process lives in.

    Args:
        pid: The process to look up.
        container_names: Container name by full container id.

    Returns:
        The container name, or None for a process outside every running
        container or a process that has ended.
    """
    try:
        cgroup = (_PROC_ROOT / str(pid) / "cgroup").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return None
    id_match = _CONTAINER_ID_PATTERN.search(cgroup)
    if id_match is None:
        return None
    return container_names.get(id_match.group())


def _read_command(pid: int) -> str:
    """Read the command line of a process.

    Args:
        pid: The process to read.

    Returns:
        The command line, or an empty string if the process has ended.
    """
    try:
        cmdline = (_PROC_ROOT / str(pid) / "cmdline").read_text(errors="replace")
    except (FileNotFoundError, ProcessLookupError):
        return ""
    return cmdline.replace("\0", " ").strip()
