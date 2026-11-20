"""Tests for GPU readings and the exact VRAM of each container."""

import asyncio
from typing import NamedTuple

import aiodocker
import pynvml
import pytest

from app.gpu_usage import ContainerUsage, GpuProcess, GpuStats, GpuUsageError, GpuUsageReader
from tests.app.conftest import BYTES_PER_MIB, FakeGpuProcess, FakeNvidiaHost

_QUARTER_MIB_BYTES = BYTES_PER_MIB // 4
_THREE_QUARTER_MIB_BYTES = BYTES_PER_MIB - _QUARTER_MIB_BYTES

_TOTAL_VRAM_MIB = 24576
_USED_VRAM_MIB = 4000

_LLAMACPP = "llamacpp-server"
_LLAMACPP_PID = 101
_LLAMACPP_VRAM_MIB = 1000
_LLAMACPP_UTIL_PCT = 7
_CPAI = "cpai"
_CPAI_PID = 202
_CPAI_VRAM_MIB = 500
_CPAI_UTIL_PCT = 3
_HOST_PID = 303
_HOST_VRAM_MIB = 300
_SECOND_LLAMACPP_PID = 104
_SECOND_LLAMACPP_VRAM_MIB = 200
_SECOND_LLAMACPP_UTIL_PCT = 11

_SMALL_SPLIT_PROCESS_MIB = 100
_SMALL_SPLIT_CONTAINER_MIB = 201
_LARGE_SPLIT_PROCESS_MIB = 300
_LARGE_SPLIT_CONTAINER_MIB = 601

_RESERVED_VRAM_MIB = 640
_LARGE_RESERVED_VRAM_MIB = 1024

_COMMAND = "python -m serve --port 8080"


class _Reading(NamedTuple):
    """GPU wide readings to set on a fake host."""

    used_mib: int
    util_pct: int
    power_w: float
    limit_w: float
    gr_clock_mhz: int
    mem_clock_mhz: int
    pstate: int
    temp_c: int


_IDLE_READING = _Reading(860, 0, 30.0, 250.0, 210, 405, 8, 40)
_BUSY_READING = _Reading(18000, 93, 280.5, 300.0, 2610, 14001, 2, 77)


def _llamacpp_process() -> FakeGpuProcess:
    """The llamacpp server process."""
    return FakeGpuProcess.from_mib(
        _LLAMACPP_PID, _LLAMACPP_VRAM_MIB, _LLAMACPP, sm_utilization_pct=_LLAMACPP_UTIL_PCT
    )


def _cpai_process() -> FakeGpuProcess:
    """The cpai process."""
    return FakeGpuProcess.from_mib(
        _CPAI_PID, _CPAI_VRAM_MIB, _CPAI, sm_utilization_pct=_CPAI_UTIL_PCT
    )


def _host(*processes: FakeGpuProcess, used_vram_mib: int = _USED_VRAM_MIB) -> FakeNvidiaHost:
    """A GPU host running the given processes."""
    return FakeNvidiaHost(
        total_vram_mib=_TOTAL_VRAM_MIB,
        used_vram_mib=used_vram_mib,
        processes=list(processes),
    )


async def _read(host: FakeNvidiaHost) -> tuple[GpuStats, list[ContainerUsage]]:
    """Open a reader on the host, read it once, and close it."""
    reader = GpuUsageReader()
    with host.installed():
        await reader.open()
        try:
            return await reader.read()
        finally:
            await reader.close()


async def _read_containers(host: FakeNvidiaHost) -> list[ContainerUsage]:
    """The containers one read of the host reports."""
    _, containers = await _read(host)
    return containers


class TestContainerAttribution:
    """Tests for how GPU memory is split among containers."""

    async def test_container_vram_is_exactly_what_the_driver_reports(self) -> None:
        """Memory no process owns is never handed out to a container."""
        host = _host(
            _llamacpp_process(),
            _cpai_process(),
            FakeGpuProcess.from_mib(_HOST_PID, _HOST_VRAM_MIB),
        )
        containers = await _read_containers(host)
        assert [(c.name, c.vram_mib) for c in containers] == [
            (_LLAMACPP, _LLAMACPP_VRAM_MIB),
            (_CPAI, _CPAI_VRAM_MIB),
        ]

    async def test_unowned_memory_does_not_change_container_vram(self) -> None:
        """Two readings that differ only in unowned memory give the same containers."""
        quiet = await _read_containers(_host(_llamacpp_process(), used_vram_mib=_LLAMACPP_VRAM_MIB))
        busy = await _read_containers(_host(_llamacpp_process(), used_vram_mib=_TOTAL_VRAM_MIB))
        assert quiet == busy
        assert busy[0].vram_mib == _LLAMACPP_VRAM_MIB

    async def test_container_adds_up_all_its_processes(self) -> None:
        """A container with two processes holds the memory and load of both."""
        second = FakeGpuProcess.from_mib(
            _SECOND_LLAMACPP_PID,
            _SECOND_LLAMACPP_VRAM_MIB,
            _LLAMACPP,
            sm_utilization_pct=_SECOND_LLAMACPP_UTIL_PCT,
        )
        containers = await _read_containers(_host(_llamacpp_process(), second))
        assert len(containers) == 1
        assert containers[0].vram_mib == _LLAMACPP_VRAM_MIB + _SECOND_LLAMACPP_VRAM_MIB
        assert containers[0].gpu_utilization_pct == (
            _LLAMACPP_UTIL_PCT + _SECOND_LLAMACPP_UTIL_PCT
        )
        assert [p.pid for p in containers[0].processes] == [_LLAMACPP_PID, _SECOND_LLAMACPP_PID]

    @pytest.mark.parametrize(
        ("process_mib", "container_mib"),
        [
            (_SMALL_SPLIT_PROCESS_MIB, _SMALL_SPLIT_CONTAINER_MIB),
            (_LARGE_SPLIT_PROCESS_MIB, _LARGE_SPLIT_CONTAINER_MIB),
        ],
        ids=["small", "large"],
    )
    async def test_container_total_adds_bytes_before_rounding(
        self, process_mib: int, container_mib: int
    ) -> None:
        """Part MiB amounts of each process add up, then drop to the whole MiB below."""
        split_bytes = process_mib * BYTES_PER_MIB + _THREE_QUARTER_MIB_BYTES
        processes = [
            FakeGpuProcess(pid=pid, vram_bytes=split_bytes, container=_LLAMACPP)
            for pid in (_LLAMACPP_PID, _SECOND_LLAMACPP_PID)
        ]
        containers = await _read_containers(_host(*processes))
        assert containers[0].vram_mib == container_mib
        assert [p.vram_mib for p in containers[0].processes] == [process_mib, process_mib]

    async def test_process_listed_as_compute_and_graphics_counts_once(self) -> None:
        """A process the driver lists twice is not counted twice."""
        process = _llamacpp_process()
        process.is_graphics = True
        containers = await _read_containers(_host(process))
        assert containers[0].vram_mib == _LLAMACPP_VRAM_MIB
        assert len(containers[0].processes) == 1

    async def test_graphics_only_process_counts(self) -> None:
        """A process listed only as graphics still belongs to its container."""
        process = _llamacpp_process()
        process.is_compute = False
        process.is_graphics = True
        containers = await _read_containers(_host(process))
        assert [(c.name, c.vram_mib) for c in containers] == [(_LLAMACPP, _LLAMACPP_VRAM_MIB)]

    async def test_host_processes_are_left_out(self) -> None:
        """Processes outside every container belong to no container."""
        containers = await _read_containers(_host(FakeGpuProcess.from_mib(_HOST_PID, _HOST_VRAM_MIB)))
        assert containers == []

    async def test_containers_come_biggest_first(self) -> None:
        """Containers are listed from most memory to least."""
        containers = await _read_containers(_host(_cpai_process(), _llamacpp_process()))
        assert [c.name for c in containers] == [_LLAMACPP, _CPAI]

    async def test_systemd_cgroup_path_finds_container(self) -> None:
        """A process in a systemd scope cgroup is matched to its container."""
        process = _llamacpp_process()
        process.cgroup = (
            f"0::/system.slice/docker-{FakeNvidiaHost.container_id(_LLAMACPP)}.scope\n"
        )
        containers = await _read_containers(_host(process))
        assert [(c.name, c.vram_mib) for c in containers] == [(_LLAMACPP, _LLAMACPP_VRAM_MIB)]

    async def test_process_of_stopped_container_is_left_out(self) -> None:
        """A process whose container Docker no longer runs belongs to no container."""
        host = _host(_llamacpp_process(), _cpai_process())
        host.stopped_containers = {_CPAI}
        containers = await _read_containers(host)
        assert [c.name for c in containers] == [_LLAMACPP]

    async def test_exited_process_is_left_out(self) -> None:
        """A process that ends during the read belongs to no container."""
        process = _cpai_process()
        process.has_exited = True
        containers = await _read_containers(_host(_llamacpp_process(), process))
        assert [c.name for c in containers] == [_LLAMACPP]

    async def test_process_ending_as_its_cgroup_is_read_is_left_out(self) -> None:
        """A process that ends while its cgroup file is open belongs to no container."""
        process = _cpai_process()
        process.has_exited = True
        process.exit_error = ProcessLookupError
        containers = await _read_containers(_host(_llamacpp_process(), process))
        assert [c.name for c in containers] == [_LLAMACPP]

    @pytest.mark.parametrize(
        "exit_error",
        [FileNotFoundError, ProcessLookupError],
        ids=["gone", "ending"],
    )
    async def test_process_ending_before_its_command_is_read_keeps_its_memory(
        self, exit_error: type[OSError]
    ) -> None:
        """A process that ends after its cgroup is read stays, with no command line."""
        process = _llamacpp_process()
        process.command = _COMMAND
        process.exits_after_cgroup = True
        process.exit_error = exit_error
        containers = await _read_containers(_host(process))
        assert containers[0].processes == [
            GpuProcess(
                pid=_LLAMACPP_PID,
                command="",
                vram_mib=_LLAMACPP_VRAM_MIB,
                sm_utilization_pct=_LLAMACPP_UTIL_PCT,
            )
        ]
        assert containers[0].vram_mib == _LLAMACPP_VRAM_MIB

    async def test_container_process_with_hidden_memory_fails_the_read(self) -> None:
        """A container process whose memory the driver hides is an error, not zero."""
        process = _llamacpp_process()
        process.vram_bytes = None
        with pytest.raises(GpuUsageError, match=f"pid {_LLAMACPP_PID} in container {_LLAMACPP}"):
            await _read(_host(process))

    async def test_host_process_with_hidden_memory_is_ignored(self) -> None:
        """A host process whose memory the driver hides does not stop the read."""
        hidden = FakeGpuProcess(pid=_HOST_PID, vram_bytes=None)
        containers = await _read_containers(_host(_llamacpp_process(), hidden))
        assert [(c.name, c.vram_mib) for c in containers] == [(_LLAMACPP, _LLAMACPP_VRAM_MIB)]

    async def test_process_details_are_reported(self) -> None:
        """Each container process carries its id, command line, memory and load."""
        process = _llamacpp_process()
        process.command = _COMMAND
        containers = await _read_containers(_host(process))
        assert containers[0].processes == [
            GpuProcess(
                pid=_LLAMACPP_PID,
                command=_COMMAND,
                vram_mib=_LLAMACPP_VRAM_MIB,
                sm_utilization_pct=_LLAMACPP_UTIL_PCT,
            )
        ]

    async def test_no_load_samples_reads_zero_load(self) -> None:
        """With no load samples from the driver, every container reads idle."""
        containers = await _read_containers(
            _host(FakeGpuProcess.from_mib(_LLAMACPP_PID, _LLAMACPP_VRAM_MIB, _LLAMACPP))
        )
        assert containers[0].gpu_utilization_pct == 0
        assert containers[0].processes[0].sm_utilization_pct == 0

    async def test_load_query_failure_reads_zero_load(self) -> None:
        """A failed load query still reports memory, with every container idle."""
        host = _host(_llamacpp_process())
        host.utilization_error = pynvml.NVMLError(pynvml.NVML_ERROR_NOT_SUPPORTED)
        containers = await _read_containers(host)
        assert [(c.name, c.vram_mib, c.gpu_utilization_pct) for c in containers] == [
            (_LLAMACPP, _LLAMACPP_VRAM_MIB, 0)
        ]


class TestGpuStats:
    """Tests for the GPU wide readings."""

    @pytest.mark.parametrize("reading", [_IDLE_READING, _BUSY_READING], ids=["idle", "busy"])
    async def test_stats_carry_every_reading(self, reading: _Reading) -> None:
        """Every GPU wide reading is reported in the reader's units."""
        host = _host(used_vram_mib=reading.used_mib)
        host.gpu_util_pct = reading.util_pct
        host.power_draw_w = reading.power_w
        host.power_limit_w = reading.limit_w
        host.gr_clock_mhz = reading.gr_clock_mhz
        host.mem_clock_mhz = reading.mem_clock_mhz
        host.pstate = reading.pstate
        host.temperature_c = reading.temp_c
        stats, _ = await _read(host)
        assert stats == GpuStats(
            total_vram_mib=_TOTAL_VRAM_MIB,
            used_vram_mib=reading.used_mib,
            available_vram_mib=_TOTAL_VRAM_MIB - reading.used_mib,
            gpu_utilization_pct=reading.util_pct,
            power_draw_w=reading.power_w,
            power_limit_w=reading.limit_w,
            gr_clock_mhz=reading.gr_clock_mhz,
            mem_clock_mhz=reading.mem_clock_mhz,
            pstate=f"P{reading.pstate}",
            temperature_c=reading.temp_c,
        )

    async def test_used_vram_leaves_out_reserved_memory(self) -> None:
        """Memory the driver keeps for itself is not counted as used."""
        host = _host()
        host.reserved_vram_mib = _RESERVED_VRAM_MIB
        stats, _ = await _read(host)
        assert stats.used_vram_mib == _USED_VRAM_MIB

    @pytest.mark.parametrize(
        ("reserved_mib", "extra_used_bytes"),
        [(_RESERVED_VRAM_MIB, _QUARTER_MIB_BYTES), (_LARGE_RESERVED_VRAM_MIB, _THREE_QUARTER_MIB_BYTES)],
        ids=["small-reserve", "large-reserve"],
    )
    async def test_available_vram_leaves_out_reserved_memory(
        self, reserved_mib: int, extra_used_bytes: int
    ) -> None:
        """Memory the driver keeps for itself is not available, and part MiBs are dropped."""
        host = _host()
        host.reserved_vram_mib = reserved_mib
        host.extra_used_bytes = extra_used_bytes
        stats, _ = await _read(host)
        assert stats.available_vram_mib == _TOTAL_VRAM_MIB - _USED_VRAM_MIB - reserved_mib - 1

    @pytest.mark.parametrize(
        ("extra_bytes", "used_mib"),
        [(_QUARTER_MIB_BYTES, _USED_VRAM_MIB), (_THREE_QUARTER_MIB_BYTES, _USED_VRAM_MIB + 1)],
        ids=["rounds-down", "rounds-up"],
    )
    async def test_used_vram_rounds_to_the_nearest_mib(
        self, extra_bytes: int, used_mib: int
    ) -> None:
        """Used memory that is part of a MiB goes to the nearest whole MiB."""
        host = _host()
        host.extra_used_bytes = extra_bytes
        stats, _ = await _read(host)
        assert stats.used_vram_mib == used_mib

    @pytest.mark.parametrize(
        "extra_bytes",
        [_QUARTER_MIB_BYTES, _THREE_QUARTER_MIB_BYTES],
        ids=["quarter", "three-quarters"],
    )
    async def test_total_vram_drops_part_of_a_mib(self, extra_bytes: int) -> None:
        """Total memory that is part of a MiB is cut down to the whole MiB below."""
        host = _host()
        host.extra_total_bytes = extra_bytes
        stats, _ = await _read(host)
        assert stats.total_vram_mib == _TOTAL_VRAM_MIB
        assert stats.available_vram_mib == _TOTAL_VRAM_MIB - _USED_VRAM_MIB

    async def test_missing_optional_readings_read_zero(self) -> None:
        """Readings a GPU lacks come back as zero instead of failing the read."""
        host = _host()
        host.gpu_util_pct = _BUSY_READING.util_pct
        host.power_limit_w = _BUSY_READING.limit_w
        host.gr_clock_mhz = _BUSY_READING.gr_clock_mhz
        host.mem_clock_mhz = _BUSY_READING.mem_clock_mhz
        host.pstate = _BUSY_READING.pstate
        host.temperature_c = _BUSY_READING.temp_c
        host.unsupported_queries = {
            "nvmlDeviceGetUtilizationRates",
            "nvmlDeviceGetPowerManagementLimit",
            "nvmlDeviceGetClockInfo",
            "nvmlDeviceGetPowerState",
            "nvmlDeviceGetTemperature",
        }
        stats, _ = await _read(host)
        assert (stats.gpu_utilization_pct, stats.power_limit_w) == (0, 0)
        assert (stats.gr_clock_mhz, stats.mem_clock_mhz) == (0, 0)
        assert (stats.pstate, stats.temperature_c) == ("P0", 0)


class TestReaderLifecycle:
    """Tests for opening, closing and failing reads."""

    async def test_open_failure_raises(self) -> None:
        """A driver that will not start fails open and is never shut down."""
        host = _host()
        host.init_error = pynvml.NVMLError(pynvml.NVML_ERROR_DRIVER_NOT_LOADED)
        with host.installed(), pytest.raises(GpuUsageError, match="initialize NVML"):
            await GpuUsageReader().open()
        assert host.closes == 0

    async def test_missing_gpu_raises_and_disconnects(self) -> None:
        """A GPU index with no GPU fails open and shuts the driver down again."""
        host = _host()
        host.handle_error = pynvml.NVMLError(pynvml.NVML_ERROR_INVALID_ARGUMENT)
        host.shutdown_error = pynvml.NVMLError(pynvml.NVML_ERROR_UNKNOWN)
        with host.installed(), pytest.raises(GpuUsageError, match="index 3"):
            await GpuUsageReader(gpu_index=3).open()
        assert host.closes == 1

    async def test_read_before_open_raises(self) -> None:
        """Reading a reader that was never opened fails without touching the GPU."""
        host = _host()
        with host.installed(), pytest.raises(GpuUsageError, match="not open"):
            await GpuUsageReader().read()
        assert host.polls == 0

    async def test_read_after_close_raises(self) -> None:
        """A closed reader no longer reads the GPU."""
        host = _host()
        reader = GpuUsageReader()
        with host.installed():
            await reader.open()
            await reader.close()
            with pytest.raises(GpuUsageError, match="not open"):
                await reader.read()
        assert host.polls == 0

    async def test_close_twice_disconnects_once(self) -> None:
        """Closing again does nothing, even when the first shutdown failed."""
        host = _host()
        host.shutdown_error = pynvml.NVMLError(pynvml.NVML_ERROR_UNKNOWN)
        reader = GpuUsageReader()
        with host.installed():
            await reader.open()
            await reader.close()
            await reader.close()
        assert host.closes == 1

    async def test_close_waits_for_a_read_in_progress(self) -> None:
        """Closing mid read waits for it, and a read queued behind it never starts."""
        host = _host(_llamacpp_process())
        host.hold_reads()
        reader = GpuUsageReader()
        with host.installed():
            await reader.open()
            in_progress = asyncio.create_task(reader.read())
            await asyncio.to_thread(host.wait_for_held_read)
            queued = asyncio.create_task(reader.read())
            await asyncio.sleep(0)
            closing = asyncio.create_task(reader.close())
            await asyncio.sleep(0)
            await asyncio.to_thread(host.wait_for_shutdown)
            host.release_reads()
            _, containers = await in_progress
            with pytest.raises(GpuUsageError, match="not open"):
                await queued
            await closing
        assert [(c.name, c.vram_mib) for c in containers] == [(_LLAMACPP, _LLAMACPP_VRAM_MIB)]
        assert host.polls == 1
        assert host.closes == 1
        assert host.has_shut_down_during_read is False

    async def test_close_without_open_does_nothing(self) -> None:
        """Closing a reader that never opened does not touch the driver."""
        host = _host()
        with host.installed():
            await GpuUsageReader().close()
        assert host.closes == 0

    async def test_driver_failure_fails_the_read(self) -> None:
        """A failed driver query is reported as a read error."""
        host = _host(_llamacpp_process())
        host.poll_fails = True
        with pytest.raises(GpuUsageError, match="Failed to read GPU usage"):
            await _read(host)

    @pytest.mark.parametrize(
        "error",
        [aiodocker.DockerError(500, "daemon error"), TimeoutError()],
        ids=["docker-error", "timeout"],
    )
    async def test_docker_failure_fails_the_read(self, error: Exception) -> None:
        """A Docker daemon that fails or does not answer is reported as a read error."""
        host = _host(_llamacpp_process())
        host.docker_error = error
        with pytest.raises(GpuUsageError, match="Failed to read GPU usage"):
            await _read(host)

    async def test_unreadable_process_fails_the_read(self) -> None:
        """A process the host will not let us look at is reported as a read error."""
        process = _llamacpp_process()
        process.is_hidden = True
        with pytest.raises(GpuUsageError, match="Failed to read GPU usage"):
            await _read(_host(process))
