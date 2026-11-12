"""Tests for GPU monitor: library-based polling, conversion, and lifecycle."""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator, Callable, Optional
from unittest.mock import patch

from app.gpu_monitor import (
    ClockProtocol,
    GPUStatus,
    GpuMonitor,
    LibContainerUsage,
    LibGpuStats,
)

# Importing app.gpu_monitor is what puts the library on sys.path, so this follows it.
from nvidia_docker_usage import ContainerProcess, NvmlError

_POLL_TIME = 1000.0
_ALTERNATE_POLL_TIME = 42.0
_SLEEP_TIMEOUT_S = 1.0
_STOP_TIMEOUT_S = 1.0

_DEFAULT_POLL_INTERVAL_S = 5
_CUSTOM_POLL_INTERVAL_S = 10

_TOTAL_VRAM_MIB = 24576
_IDLE_USED_VRAM_MIB = 860
_IDLE_FREE_VRAM_MIB = 23716
_LOADED_USED_VRAM_MIB = 1606
_LOADED_FREE_VRAM_MIB = 22970
_SOLE_CONTAINER_FREE_VRAM_MIB = 23830
_OVERSUBSCRIBED_USED_VRAM_MIB = 500

_LLAMACPP = "llamacpp-server"
_LLAMACPP_VRAM_MIB = 746
_LLAMACPP_UTIL_PCT = 5
_CPAI = "cpai"
_CPAI_VRAM_MIB = 860

_IDLE_POWER_W = 30.0
_POWER_LIMIT_W = 250.0
_IDLE_TEMP_C = 40
_IDLE_GR_CLOCK_MHZ = 210
_IDLE_MEM_CLOCK_MHZ = 405
_IDLE_PSTATE = "P8"

_BUSY_UTIL_PCT = 10
_BUSY_GR_CLOCK_MHZ = 1980
_BUSY_MEM_CLOCK_MHZ = 3122
_BUSY_PSTATE = "P2"

_PROCESS_PID = 1234
_PROCESS_COMMAND = "/app/llama-server"
_PROCESS_VRAM_MIB = 746
_PROCESS_UTIL_PCT = 5


class PollClock(ClockProtocol):
    """A clock whose sleeps park until the test lets them finish.

    Parking the monitor's polling loop keeps a test in charge of when the next
    poll happens, and recording how long each sleep asked for is how a test
    reads back the poll interval the monitor is working to.
    """

    def __init__(self, start_time: float = _POLL_TIME) -> None:
        self._time = start_time
        self._sleeps: list[float] = []
        self._sleeping = asyncio.Event()
        self._resume = asyncio.Event()

    def time(self) -> float:
        """Return the current virtual time in seconds."""
        return self._time

    async def sleep(self, seconds: float) -> None:
        """Record the requested sleep and park here until the test releases it."""
        self._sleeps.append(seconds)
        self._sleeping.set()
        await self._resume.wait()
        self._resume.clear()
        self._time += seconds

    async def next_sleep(self) -> float:
        """Wait for the caller to ask to sleep and return the length it asked for.

        Raises:
            TimeoutError: If nothing asks to sleep, so a monitor that stopped
                polling fails the waiting test instead of hanging the suite.
        """
        await asyncio.wait_for(self._sleeping.wait(), _SLEEP_TIMEOUT_S)
        self._sleeping.clear()
        return self._sleeps[-1]

    def release(self) -> None:
        """Let a parked sleep finish, so whatever is waiting on it runs again."""
        self._resume.set()


class FakeGpuLibrary:
    """Scriptable stand-in for the NVIDIA library the monitor polls.

    Reports the readings a test configured, counts what the monitor asked of
    it, and can be told to fail at any of the three points the real library
    can fail: opening the connection, querying stats, querying containers.
    """

    def __init__(
        self,
        stats: Optional[LibGpuStats] = None,
        containers: Optional[list[LibContainerUsage]] = None,
        init_error: Optional[Exception] = None,
        stats_error: Optional[Exception] = None,
        containers_error: Optional[Exception] = None,
    ) -> None:
        self.stats = _lib_stats() if stats is None else stats
        self.containers = [] if containers is None else containers
        self.init_error = init_error
        self.stats_error = stats_error
        self.containers_error = containers_error
        self.inits = 0
        self.polls = 0
        self.closes = 0

    async def init(self) -> None:
        """Open the GPU connection, failing if the test asked it to."""
        self.inits += 1
        if self.init_error is not None:
            raise self.init_error

    async def close(self) -> None:
        """Shut the GPU connection down."""
        self.closes += 1

    async def get_stats(self) -> LibGpuStats:
        """Return the GPU-wide reading the test configured."""
        self.polls += 1
        if self.stats_error is not None:
            raise self.stats_error
        return self.stats

    async def get_container_usage(self) -> list[LibContainerUsage]:
        """Return the per-container readings the test configured."""
        if self.containers_error is not None:
            raise self.containers_error
        return self.containers


def _lib_stats(
    total_vram_mib: int = _TOTAL_VRAM_MIB,
    used_vram_mib: int = _IDLE_USED_VRAM_MIB,
    available_vram_mib: int = _IDLE_FREE_VRAM_MIB,
    gpu_utilization_pct: int = 0,
    power_draw_w: float = _IDLE_POWER_W,
    power_limit_w: float = _POWER_LIMIT_W,
    gr_clock_mhz: int = _IDLE_GR_CLOCK_MHZ,
    mem_clock_mhz: int = _IDLE_MEM_CLOCK_MHZ,
    pstate: str = _IDLE_PSTATE,
    temperature_c: int = _IDLE_TEMP_C,
) -> LibGpuStats:
    """A GPU-wide reading as the library reports it, idle unless overridden."""
    return LibGpuStats(
        total_vram_mib=total_vram_mib,
        used_vram_mib=used_vram_mib,
        available_vram_mib=available_vram_mib,
        gpu_utilization_pct=gpu_utilization_pct,
        power_draw_w=power_draw_w,
        power_limit_w=power_limit_w,
        gr_clock_mhz=gr_clock_mhz,
        mem_clock_mhz=mem_clock_mhz,
        pstate=pstate,
        temperature_c=temperature_c,
        reserved_vram_mib=0,
    )


def _lib_container(
    name: str = _LLAMACPP,
    total_vram_mib: int = _LLAMACPP_VRAM_MIB,
    total_gpu_utilization_pct: int = _LLAMACPP_UTIL_PCT,
    processes: Optional[list[ContainerProcess]] = None,
) -> LibContainerUsage:
    """One container's usage as the library reports it."""
    return LibContainerUsage(
        name=name,
        processes=[] if processes is None else processes,
        total_vram_mib=total_vram_mib,
        total_gpu_utilization_pct=total_gpu_utilization_pct,
    )


async def _stop(monitor: GpuMonitor) -> None:
    """Stop a monitor, failing the test rather than hanging if it will not halt.

    The stop is awaited directly rather than handed to a task, so it begins
    before the event loop can run anything else — the same standing start the
    proxy gives it, and the only one under which a polling task that has not
    had its first turn yet is the thing being cancelled. Awaiting it directly
    also hands whatever stop() raised straight back to the test, so a crash on
    the way down is a failure rather than a discarded task.

    Missing the deadline cancels the stop where it is stuck, and a monitor that
    swallows that cancellation returns looking as if it had stopped cleanly, so
    it is the expiry of the deadline and not the escape of a TimeoutError that
    decides whether the stop hung.

    Raises:
        TimeoutError: If stop() does not return promptly.
        Exception: Whatever stop() itself raised.
    """
    async with asyncio.timeout(_STOP_TIMEOUT_S) as deadline:
        await monitor.stop()
    if deadline.expired():
        raise TimeoutError("monitor.stop() did not return")


@asynccontextmanager
async def _started(
    library: FakeGpuLibrary,
    clock: Optional[ClockProtocol] = None,
    poll_interval: int = _DEFAULT_POLL_INTERVAL_S,
    on_update: Optional[Callable[[GPUStatus], None]] = None,
) -> AsyncIterator[GpuMonitor]:
    """Run a monitor against the given fake library for the body of the block.

    The monitor is started on the way in and always stopped on the way out,
    which is the lifecycle the proxy gives it in production.
    """
    monitor = GpuMonitor(
        clock=PollClock() if clock is None else clock,
        poll_interval=poll_interval,
        on_update=on_update,
    )
    with patch("app.gpu_monitor.LibGpuUsageMonitor", lambda: library):
        await monitor.start()
        try:
            yield monitor
        finally:
            await _stop(monitor)


async def _poll(
    stats: Optional[LibGpuStats] = None,
    containers: Optional[list[LibContainerUsage]] = None,
    clock: Optional[ClockProtocol] = None,
) -> GPUStatus:
    """The status a running monitor reports for one poll of the given readings."""
    library = FakeGpuLibrary(stats=stats, containers=containers)
    async with _started(library, clock=clock) as monitor:
        return await monitor.poll_once()


class TestConvertStats:
    """Tests for the GPUStatus a poll builds out of the library's readings."""

    async def test_convert_basic_stats(self) -> None:
        """An idle GPU with no containers reports every headline figure."""
        status = await _poll(_lib_stats())
        assert status.total_vram_mib == _TOTAL_VRAM_MIB
        assert status.available_vram_mib == _IDLE_FREE_VRAM_MIB
        assert status.system_vram_mib == _IDLE_USED_VRAM_MIB
        assert status.docker_vram_mib == 0
        assert status.gpu_util_pct == 0
        assert status.power_usage_w == _IDLE_POWER_W
        assert status.power_limit_w == _POWER_LIMIT_W
        assert status.temperature_c == _IDLE_TEMP_C
        assert status.poll_failed is False
        assert status.last_updated == _POLL_TIME

    async def test_convert_with_containers(self) -> None:
        """Container VRAM is attributed to Docker and taken off the system's share."""
        status = await _poll(
            _lib_stats(
                used_vram_mib=_LOADED_USED_VRAM_MIB,
                available_vram_mib=_LOADED_FREE_VRAM_MIB,
            ),
            [
                _lib_container(),
                _lib_container(
                    name=_CPAI,
                    total_vram_mib=_CPAI_VRAM_MIB,
                    total_gpu_utilization_pct=0,
                ),
            ],
        )
        docker_vram_mib = _LLAMACPP_VRAM_MIB + _CPAI_VRAM_MIB
        assert status.docker_vram_mib == docker_vram_mib
        assert status.system_vram_mib == _LOADED_USED_VRAM_MIB - docker_vram_mib
        assert set(status.containers) == {_LLAMACPP, _CPAI}
        assert status.containers[_LLAMACPP].vram_mib == _LLAMACPP_VRAM_MIB
        assert status.containers[_LLAMACPP].gpu_util_pct == _LLAMACPP_UTIL_PCT
        assert status.containers[_CPAI].vram_mib == _CPAI_VRAM_MIB
        assert status.containers[_CPAI].gpu_util_pct == 0

    async def test_convert_no_containers(self) -> None:
        """With nothing running in Docker, all used VRAM belongs to the system."""
        status = await _poll(_lib_stats(used_vram_mib=_IDLE_USED_VRAM_MIB))
        assert status.docker_vram_mib == 0
        assert status.system_vram_mib == _IDLE_USED_VRAM_MIB
        assert status.containers == {}

    async def test_convert_sets_last_updated_from_clock(self) -> None:
        """last_updated carries the clock's reading, not a wall-clock timestamp."""
        status = await _poll(clock=PollClock(start_time=_ALTERNATE_POLL_TIME))
        assert status.last_updated == _ALTERNATE_POLL_TIME

    async def test_convert_processes_preserved(self) -> None:
        """The processes the library attributes to a container survive conversion."""
        process = ContainerProcess(
            pid=_PROCESS_PID,
            command=_PROCESS_COMMAND,
            vram_mib=_PROCESS_VRAM_MIB,
            sm_utilization_pct=_PROCESS_UTIL_PCT,
        )
        status = await _poll(
            _lib_stats(
                used_vram_mib=_LLAMACPP_VRAM_MIB,
                available_vram_mib=_SOLE_CONTAINER_FREE_VRAM_MIB,
            ),
            [_lib_container(processes=[process])],
        )
        assert status.containers[_LLAMACPP].processes == [process]

    async def test_convert_gpu_utilization(self) -> None:
        """GPU utilization is carried across under the name GPUStatus gives it."""
        status = await _poll(_lib_stats(gpu_utilization_pct=_BUSY_UTIL_PCT))
        assert status.gpu_util_pct == _BUSY_UTIL_PCT

    async def test_convert_clocks_and_pstate(self) -> None:
        """Graphics clock, memory clock, and P-State come across untouched."""
        status = await _poll(
            _lib_stats(
                gr_clock_mhz=_BUSY_GR_CLOCK_MHZ,
                mem_clock_mhz=_BUSY_MEM_CLOCK_MHZ,
                pstate=_BUSY_PSTATE,
            )
        )
        assert status.gr_clock_mhz == _BUSY_GR_CLOCK_MHZ
        assert status.mem_clock_mhz == _BUSY_MEM_CLOCK_MHZ
        assert status.pstate == _BUSY_PSTATE

    async def test_convert_negative_system_vram_docker_exceeds_used(self) -> None:
        """System VRAM is allowed to go negative when containers exceed reported use."""
        status = await _poll(
            _lib_stats(used_vram_mib=_OVERSUBSCRIBED_USED_VRAM_MIB),
            [_lib_container()],
        )
        assert status.system_vram_mib == _OVERSUBSCRIBED_USED_VRAM_MIB - _LLAMACPP_VRAM_MIB
        assert status.system_vram_mib < 0
        assert status.docker_vram_mib == _LLAMACPP_VRAM_MIB


class TestGpuMonitorInit:
    """Tests for how a monitor is configured when it is built."""

    async def test_default_params(self) -> None:
        """A monitor built without a poll interval waits five seconds between polls."""
        clock = PollClock()
        async with _started(FakeGpuLibrary(), clock=clock):
            assert await clock.next_sleep() == _DEFAULT_POLL_INTERVAL_S

    async def test_custom_params(self) -> None:
        """A monitor built with a poll interval waits exactly that long between polls."""
        clock = PollClock()
        async with _started(
            FakeGpuLibrary(), clock=clock, poll_interval=_CUSTOM_POLL_INTERVAL_S
        ):
            assert await clock.next_sleep() == _CUSTOM_POLL_INTERVAL_S

    async def test_monitor_initially_none(self) -> None:
        """Building a monitor touches no GPU: only start() opens the library."""
        library = FakeGpuLibrary()
        with patch("app.gpu_monitor.LibGpuUsageMonitor", lambda: library):
            monitor = GpuMonitor(clock=PollClock())
            status = await monitor.poll_once()
        assert library.inits == 0
        assert library.polls == 0
        assert status.poll_failed is True


class TestGpuMonitorPoll:
    """Tests for GPU polling behavior."""

    async def test_poll_once_with_monitor_returns_status(self) -> None:
        """A poll against a working library reports the reading it got back."""
        status = await _poll(_lib_stats(used_vram_mib=_IDLE_USED_VRAM_MIB))
        assert status.poll_failed is False
        assert status.total_vram_mib == _TOTAL_VRAM_MIB
        assert status.system_vram_mib == _IDLE_USED_VRAM_MIB
        assert status.last_updated == _POLL_TIME

    async def test_poll_once_without_monitor_returns_failed(self) -> None:
        """With no library connection, a poll reports failure and no reading at all."""
        library = FakeGpuLibrary(init_error=RuntimeError("NVML not available"))
        async with _started(library) as monitor:
            status = await monitor.poll_once()
        assert status.poll_failed is True
        assert status.last_updated is None
        assert status.total_vram_mib == 0

    async def test_poll_once_exception_returns_failed(self) -> None:
        """A stats query that raises is reported as a failed poll, not propagated."""
        library = FakeGpuLibrary(stats_error=RuntimeError("NVML error"))
        async with _started(library) as monitor:
            status = await monitor.poll_once()
        assert status.poll_failed is True
        assert status.last_updated is None

    async def test_poll_once_container_exception_returns_failed(self) -> None:
        """A container query that raises discards the stats reading that did succeed."""
        library = FakeGpuLibrary(containers_error=RuntimeError("Docker error"))
        async with _started(library) as monitor:
            status = await monitor.poll_once()
        assert status.poll_failed is True
        assert status.total_vram_mib == 0
        assert status.last_updated is None

    async def test_on_update_callback_invoked_after_poll(self) -> None:
        """The callback receives the very status a successful poll returns."""
        seen: list[GPUStatus] = []
        async with _started(FakeGpuLibrary(), on_update=seen.append) as monitor:
            status = await monitor.poll_once()
            assert seen[-1] is status
            assert seen[-1].poll_failed is False
            assert seen[-1].total_vram_mib == _TOTAL_VRAM_MIB

    async def test_on_update_callback_invoked_on_failure(self) -> None:
        """The callback hears about a failed poll too, and hears that it failed."""
        seen: list[GPUStatus] = []
        library = FakeGpuLibrary(stats_error=RuntimeError("fail"))
        async with _started(library, on_update=seen.append) as monitor:
            status = await monitor.poll_once()
            assert seen[-1] is status
            assert seen[-1].poll_failed is True

    async def test_gpu_poll_does_not_directly_modify_service_states(self) -> None:
        """Polling only reports: what it hands the callback is all it does.

        The monitor keeps nothing of its own between polls, so a later reading
        never arrives shadowed by an earlier one — deciding what a reading means
        for a service is entirely the caller's business.
        """
        seen: list[GPUStatus] = []
        library = FakeGpuLibrary()
        async with _started(library, on_update=seen.append) as monitor:
            first = await monitor.poll_once()
            assert seen[-1] is first
            library.stats = _lib_stats(gpu_utilization_pct=_BUSY_UTIL_PCT)
            later = await monitor.poll_once()
            assert seen[-1] is later
        assert first.gpu_util_pct == 0
        assert later.gpu_util_pct == _BUSY_UTIL_PCT

    async def test_first_poll_failed_no_cache_routes_all_to_backup(self) -> None:
        """A failed first poll reports no capacity, so nothing reads spare VRAM into it."""
        library = FakeGpuLibrary(init_error=RuntimeError("no GPU"))
        async with _started(library) as monitor:
            status = await monitor.poll_once()
        assert status.poll_failed is True
        assert status.last_updated is None
        assert status.available_vram_mib == 0
        assert status.containers == {}


class TestGpuMonitorStartStop:
    """Tests for starting and stopping the monitor."""

    async def test_start_initializes_monitor(self) -> None:
        """start() opens the library connection once and polls through it."""
        library = FakeGpuLibrary()
        clock = PollClock()
        async with _started(library, clock=clock) as monitor:
            assert await clock.next_sleep() == _DEFAULT_POLL_INTERVAL_S
            assert library.inits == 1
            assert library.polls == 1
            status = await monitor.poll_once()
        assert status.poll_failed is False

    async def test_start_failure_sets_monitor_none(self) -> None:
        """A library that fails to open is dropped, unpolled and never closed."""
        library = FakeGpuLibrary(init_error=RuntimeError("NVML not available"))
        async with _started(library) as monitor:
            status = await monitor.poll_once()
        assert status.poll_failed is True
        assert library.polls == 0
        assert library.closes == 0

    async def test_start_nvml_error_sets_monitor_none(self) -> None:
        """An NvmlError while opening the library never reaches the caller of start()."""
        library = FakeGpuLibrary(init_error=NvmlError("NVML init failed"))
        async with _started(library) as monitor:
            status = await monitor.poll_once()
        assert status.poll_failed is True
        assert library.polls == 0

    async def test_stop_closes_monitor(self) -> None:
        """stop() closes the library, and polls afterwards report failure."""
        library = FakeGpuLibrary()
        monitor = GpuMonitor(clock=PollClock())
        with patch("app.gpu_monitor.LibGpuUsageMonitor", lambda: library):
            await monitor.start()
            running = await monitor.poll_once()
            await _stop(monitor)
        assert running.poll_failed is False
        assert library.closes == 1
        stopped = await monitor.poll_once()
        assert stopped.poll_failed is True

    async def test_start_and_stop(self) -> None:
        """A started monitor polls on its own schedule and stops polling once stopped.

        Starting a second time replaces the polling schedule, so stopping has to
        silence every schedule the monitor has ever started and not just the last
        one: a poll that outlives stop() still reports to the update callback.
        """
        seen: list[GPUStatus] = []
        library = FakeGpuLibrary()
        clock = PollClock()
        async with _started(library, clock=clock, on_update=seen.append) as monitor:
            await clock.next_sleep()
            assert library.polls == 1
            clock.release()
            await clock.next_sleep()
            assert library.polls == 2
            await monitor.start()
            await clock.next_sleep()
            assert library.polls == 3
        clock.release()
        await asyncio.sleep(0)
        assert library.polls == 3
        assert len(seen) == 3

    async def test_stop_is_idempotent(self) -> None:
        """Stopping an already-stopped monitor closes the library no more than once."""
        library = FakeGpuLibrary()
        async with _started(library) as monitor:
            await _stop(monitor)
            await _stop(monitor)
            assert library.closes == 1
        assert library.closes == 1
