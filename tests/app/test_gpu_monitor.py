"""Tests for polling, converting, starting and stopping the GPU monitor."""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator, Callable, Optional

import aiodocker
import pynvml

from app.gpu_monitor import ClockProtocol, GPUStatus, GpuMonitor
from app.gpu_usage import GpuProcess
from tests.app.conftest import FakeGpuProcess, FakeNvidiaHost

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
_IDLE_PSTATE = 8

_BUSY_UTIL_PCT = 10
_BUSY_GR_CLOCK_MHZ = 1980
_BUSY_MEM_CLOCK_MHZ = 3122
_BUSY_PSTATE = 2
_BUSY_PSTATE_NAME = "P2"

_LLAMACPP_PID = 1200
_CPAI_PID = 1300
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


def _host(
    used_vram_mib: int = _IDLE_USED_VRAM_MIB,
    processes: Optional[list[FakeGpuProcess]] = None,
    gpu_util_pct: int = 0,
    gr_clock_mhz: int = _IDLE_GR_CLOCK_MHZ,
    mem_clock_mhz: int = _IDLE_MEM_CLOCK_MHZ,
    pstate: int = _IDLE_PSTATE,
) -> FakeNvidiaHost:
    """A GPU host with the given processes, idle unless overridden."""
    host = FakeNvidiaHost(
        total_vram_mib=_TOTAL_VRAM_MIB,
        used_vram_mib=used_vram_mib,
        processes=processes,
    )
    host.gpu_util_pct = gpu_util_pct
    host.power_draw_w = _IDLE_POWER_W
    host.power_limit_w = _POWER_LIMIT_W
    host.temperature_c = _IDLE_TEMP_C
    host.gr_clock_mhz = gr_clock_mhz
    host.mem_clock_mhz = mem_clock_mhz
    host.pstate = pstate
    return host


def _host_without_driver() -> FakeNvidiaHost:
    """A GPU host whose driver will not start."""
    host = _host()
    host.init_error = pynvml.NVMLError(pynvml.NVML_ERROR_DRIVER_NOT_LOADED)
    return host


def _llamacpp_process() -> FakeGpuProcess:
    """The llamacpp server process."""
    return FakeGpuProcess.from_mib(
        _LLAMACPP_PID, _LLAMACPP_VRAM_MIB, _LLAMACPP, sm_utilization_pct=_LLAMACPP_UTIL_PCT
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
    host: FakeNvidiaHost,
    clock: Optional[ClockProtocol] = None,
    poll_interval: int = _DEFAULT_POLL_INTERVAL_S,
    on_update: Optional[Callable[[GPUStatus], None]] = None,
) -> AsyncIterator[GpuMonitor]:
    """Run a monitor against the given fake GPU host for the body of the block.

    The monitor is started on the way in and always stopped on the way out,
    which is the lifecycle the proxy gives it in production.
    """
    monitor = GpuMonitor(
        clock=PollClock() if clock is None else clock,
        poll_interval=poll_interval,
        on_update=on_update,
    )
    with host.installed():
        await monitor.start()
        try:
            yield monitor
        finally:
            await _stop(monitor)


async def _poll(
    host: Optional[FakeNvidiaHost] = None,
    clock: Optional[ClockProtocol] = None,
) -> GPUStatus:
    """The status a running monitor reports for one poll of the given host."""
    async with _started(_host() if host is None else host, clock=clock) as monitor:
        return await monitor.poll_once()


class TestConvertStats:
    """Tests for the GPUStatus a poll builds out of the GPU host's readings."""

    async def test_convert_basic_stats(self) -> None:
        """An idle GPU with no containers reports every headline figure."""
        status = await _poll(_host())
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
            _host(
                used_vram_mib=_LOADED_USED_VRAM_MIB,
                processes=[
                    _llamacpp_process(),
                    FakeGpuProcess.from_mib(_CPAI_PID, _CPAI_VRAM_MIB, _CPAI),
                ],
            )
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
        status = await _poll(_host(used_vram_mib=_IDLE_USED_VRAM_MIB))
        assert status.docker_vram_mib == 0
        assert status.system_vram_mib == _IDLE_USED_VRAM_MIB
        assert status.containers == {}

    async def test_convert_sets_last_updated_from_clock(self) -> None:
        """last_updated carries the clock's reading, not a wall-clock timestamp."""
        status = await _poll(clock=PollClock(start_time=_ALTERNATE_POLL_TIME))
        assert status.last_updated == _ALTERNATE_POLL_TIME

    async def test_convert_processes_preserved(self) -> None:
        """The processes attributed to a container survive conversion."""
        status = await _poll(
            _host(
                used_vram_mib=_LLAMACPP_VRAM_MIB,
                processes=[
                    FakeGpuProcess.from_mib(
                        _PROCESS_PID,
                        _PROCESS_VRAM_MIB,
                        _LLAMACPP,
                        command=_PROCESS_COMMAND,
                        sm_utilization_pct=_PROCESS_UTIL_PCT,
                    )
                ],
            )
        )
        assert status.containers[_LLAMACPP].processes == [
            GpuProcess(
                pid=_PROCESS_PID,
                command=_PROCESS_COMMAND,
                vram_mib=_PROCESS_VRAM_MIB,
                sm_utilization_pct=_PROCESS_UTIL_PCT,
            )
        ]

    async def test_convert_gpu_utilization(self) -> None:
        """GPU utilization is carried across under the name GPUStatus gives it."""
        status = await _poll(_host(gpu_util_pct=_BUSY_UTIL_PCT))
        assert status.gpu_util_pct == _BUSY_UTIL_PCT

    async def test_convert_clocks_and_pstate(self) -> None:
        """Graphics clock, memory clock, and P-State come across untouched."""
        status = await _poll(
            _host(
                gr_clock_mhz=_BUSY_GR_CLOCK_MHZ,
                mem_clock_mhz=_BUSY_MEM_CLOCK_MHZ,
                pstate=_BUSY_PSTATE,
            )
        )
        assert status.gr_clock_mhz == _BUSY_GR_CLOCK_MHZ
        assert status.mem_clock_mhz == _BUSY_MEM_CLOCK_MHZ
        assert status.pstate == _BUSY_PSTATE_NAME

    async def test_convert_negative_system_vram_docker_exceeds_used(self) -> None:
        """System VRAM is allowed to go negative when containers exceed reported use."""
        status = await _poll(
            _host(
                used_vram_mib=_OVERSUBSCRIBED_USED_VRAM_MIB,
                processes=[_llamacpp_process()],
            )
        )
        assert status.system_vram_mib == _OVERSUBSCRIBED_USED_VRAM_MIB - _LLAMACPP_VRAM_MIB
        assert status.system_vram_mib < 0
        assert status.docker_vram_mib == _LLAMACPP_VRAM_MIB


class TestGpuMonitorInit:
    """Tests for how a monitor is configured when it is built."""

    async def test_default_params(self) -> None:
        """A monitor built without a poll interval waits five seconds between polls."""
        clock = PollClock()
        async with _started(_host(), clock=clock):
            assert await clock.next_sleep() == _DEFAULT_POLL_INTERVAL_S

    async def test_custom_params(self) -> None:
        """A monitor built with a poll interval waits exactly that long between polls."""
        clock = PollClock()
        async with _started(
            _host(), clock=clock, poll_interval=_CUSTOM_POLL_INTERVAL_S
        ):
            assert await clock.next_sleep() == _CUSTOM_POLL_INTERVAL_S

    async def test_monitor_initially_none(self) -> None:
        """Building a monitor does not touch the GPU. Only start opens the driver."""
        host = _host()
        with host.installed():
            monitor = GpuMonitor(clock=PollClock())
            status = await monitor.poll_once()
        assert host.inits == 0
        assert host.polls == 0
        assert status.poll_failed is True


class TestGpuMonitorPoll:
    """Tests for GPU polling behavior."""

    async def test_poll_once_with_monitor_returns_status(self) -> None:
        """A poll against a working GPU reports the reading it got back."""
        status = await _poll(_host(used_vram_mib=_IDLE_USED_VRAM_MIB))
        assert status.poll_failed is False
        assert status.total_vram_mib == _TOTAL_VRAM_MIB
        assert status.system_vram_mib == _IDLE_USED_VRAM_MIB
        assert status.last_updated == _POLL_TIME

    async def test_poll_once_without_monitor_returns_failed(self) -> None:
        """With no driver connection, a poll reports failure and no reading at all."""
        host = _host_without_driver()
        async with _started(host) as monitor:
            status = await monitor.poll_once()
        assert status.poll_failed is True
        assert status.last_updated is None
        assert status.total_vram_mib == 0

    async def test_poll_once_exception_returns_failed(self) -> None:
        """A driver query that raises is reported as a failed poll, not propagated."""
        host = _host()
        host.poll_fails = True
        async with _started(host) as monitor:
            status = await monitor.poll_once()
        assert status.poll_failed is True
        assert status.last_updated is None

    async def test_poll_once_container_exception_returns_failed(self) -> None:
        """A Docker query that raises discards the GPU reading that did succeed."""
        host = _host()
        host.docker_error = aiodocker.DockerError(500, "Docker error")
        async with _started(host) as monitor:
            status = await monitor.poll_once()
        assert status.poll_failed is True
        assert status.total_vram_mib == 0
        assert status.last_updated is None

    async def test_on_update_callback_invoked_after_poll(self) -> None:
        """The callback receives the very status a successful poll returns."""
        seen: list[GPUStatus] = []
        async with _started(_host(), on_update=seen.append) as monitor:
            status = await monitor.poll_once()
            assert seen[-1] is status
            assert seen[-1].poll_failed is False
            assert seen[-1].total_vram_mib == _TOTAL_VRAM_MIB

    async def test_on_update_callback_invoked_on_failure(self) -> None:
        """The callback hears about a failed poll too, and hears that it failed."""
        seen: list[GPUStatus] = []
        host = _host()
        host.poll_fails = True
        async with _started(host, on_update=seen.append) as monitor:
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
        host = _host()
        async with _started(host, on_update=seen.append) as monitor:
            first = await monitor.poll_once()
            assert seen[-1] is first
            host.gpu_util_pct = _BUSY_UTIL_PCT
            later = await monitor.poll_once()
            assert seen[-1] is later
        assert first.gpu_util_pct == 0
        assert later.gpu_util_pct == _BUSY_UTIL_PCT

    async def test_first_poll_failed_no_cache_routes_all_to_backup(self) -> None:
        """A failed first poll reports no capacity, so nothing reads spare VRAM into it."""
        async with _started(_host_without_driver()) as monitor:
            status = await monitor.poll_once()
        assert status.poll_failed is True
        assert status.last_updated is None
        assert status.available_vram_mib == 0
        assert status.containers == {}


class TestGpuMonitorStartStop:
    """Tests for starting and stopping the monitor."""

    async def test_start_initializes_monitor(self) -> None:
        """Start opens the driver once and polls through it."""
        host = _host()
        clock = PollClock()
        async with _started(host, clock=clock) as monitor:
            assert await clock.next_sleep() == _DEFAULT_POLL_INTERVAL_S
            assert host.inits == 1
            assert host.polls == 1
            status = await monitor.poll_once()
        assert status.poll_failed is False

    async def test_start_failure_sets_monitor_none(self) -> None:
        """A driver that fails to start is dropped, unpolled and never shut down."""
        host = _host_without_driver()
        async with _started(host) as monitor:
            status = await monitor.poll_once()
        assert status.poll_failed is True
        assert host.polls == 0
        assert host.closes == 0

    async def test_start_nvml_error_sets_monitor_none(self) -> None:
        """A missing GPU at start does not reach the caller."""
        host = _host()
        host.handle_error = pynvml.NVMLError(pynvml.NVML_ERROR_INVALID_ARGUMENT)
        async with _started(host) as monitor:
            status = await monitor.poll_once()
        assert status.poll_failed is True
        assert host.polls == 0
        assert host.closes == 1

    async def test_stop_closes_monitor(self) -> None:
        """Stop closes the driver, and later polls fail."""
        host = _host()
        monitor = GpuMonitor(clock=PollClock())
        with host.installed():
            await monitor.start()
            running = await monitor.poll_once()
            await _stop(monitor)
        assert running.poll_failed is False
        assert host.closes == 1
        stopped = await monitor.poll_once()
        assert stopped.poll_failed is True

    async def test_start_and_stop(self) -> None:
        """A started monitor polls on its own schedule and stops polling once stopped.

        Starting a second time replaces the polling schedule, so stopping has to
        silence every schedule the monitor has ever started and not just the last
        one: a poll that outlives stop() still reports to the update callback.
        """
        seen: list[GPUStatus] = []
        host = _host()
        clock = PollClock()
        async with _started(host, clock=clock, on_update=seen.append) as monitor:
            await clock.next_sleep()
            assert host.polls == 1
            clock.release()
            await clock.next_sleep()
            assert host.polls == 2
            await monitor.start()
            await clock.next_sleep()
            assert host.polls == 3
        clock.release()
        await asyncio.sleep(0)
        assert host.polls == 3
        assert len(seen) == 3

    async def test_stop_is_idempotent(self) -> None:
        """Stopping an already-stopped monitor closes the driver no more than once."""
        host = _host()
        async with _started(host) as monitor:
            await _stop(monitor)
            await _stop(monitor)
            assert host.closes == 1
        assert host.closes == 1
