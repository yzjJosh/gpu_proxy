"""Tests for watchdog module."""

import logging
import threading
import time

import pytest

from app.watchdog import EventLoopWatchdog

# Short enough to keep the lifecycle tests instant; they never wait it out.
_SHORT_TIMEOUT_S = 0.1
# The gap between pings in the keep-alive test. Same value as the timeout
# above by coincidence, not by meaning: this one paces the caller.
_PING_INTERVAL_S = 0.1
# Long enough that five pings spaced _PING_INTERVAL_S apart overrun it.
_KEEPALIVE_TIMEOUT_S = 0.2
_KEEPALIVE_PINGS = 5


def _ping_repeatedly(
    watchdog: EventLoopWatchdog, times: int, interval_s: float
) -> None:
    """Ping the watchdog a number of times, pausing between pings."""
    for _ in range(times):
        watchdog.ping()
        time.sleep(interval_s)


def _wait_for_warning(
    caplog: pytest.LogCaptureFixture, deadline_s: float
) -> str | None:
    """Poll caplog for a watchdog warning until it appears or time runs out.

    Returns the warning message, or None if none appeared before the deadline.
    Polling (rather than a fixed sleep) keeps the test deterministic on slow
    machines: it returns as soon as the background thread logs.
    """
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        for record in caplog.records:
            if record.name == "watchdog" and record.levelno == logging.WARNING:
                return record.message
        time.sleep(0.01)
    return None


class TestEventLoopWatchdogLifecycle:
    """Tests for start/stop lifecycle through public API."""

    def test_start_stop_no_exception(self) -> None:
        """Starting and stopping completes without raising."""
        watchdog = EventLoopWatchdog(timeout=_SHORT_TIMEOUT_S)
        watchdog.start()
        watchdog.stop()

    def test_stop_without_start_is_safe(self) -> None:
        """Calling stop() on an unstarted watchdog doesn't raise."""
        watchdog = EventLoopWatchdog()
        watchdog.stop()

    def test_stop_twice_is_safe(self) -> None:
        """Calling stop() twice doesn't raise."""
        watchdog = EventLoopWatchdog(timeout=_SHORT_TIMEOUT_S)
        watchdog.start()
        watchdog.stop()
        watchdog.stop()

    def test_start_then_ping_then_stop(self) -> None:
        """start -> ping -> stop completes without raising."""
        watchdog = EventLoopWatchdog(timeout=_SHORT_TIMEOUT_S)
        watchdog.start()
        watchdog.ping()
        watchdog.stop()


class TestEventLoopWatchdogPing:
    """Tests for ping mechanism through observable behavior."""

    def test_regular_ping_prevents_timeout(self) -> None:
        """Regular pinging keeps the watchdog alive without error."""
        watchdog = EventLoopWatchdog(timeout=_KEEPALIVE_TIMEOUT_S)
        watchdog.start()
        _ping_repeatedly(watchdog, _KEEPALIVE_PINGS, _PING_INTERVAL_S)
        watchdog.stop()


class TestEventLoopWatchdogTimeout:
    """Tests for timeout detection through observable output.

    The watchdog's only observable output is logging — the warning IS
    the documented contract, analogous to CLI stdout being the contract.
    """

    def test_timeout_without_ping_warns_with_stack(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Timeout without ping logs a warning with the main thread stack."""
        watchdog = EventLoopWatchdog(timeout=_SHORT_TIMEOUT_S)
        watchdog.start()
        try:
            warning = _wait_for_warning(caplog, deadline_s=1.0)
        finally:
            watchdog.stop()

        assert "Event loop blocked" in warning
        assert "main thread stack" in warning


class TestEventLoopWatchdogConcurrency:
    """Tests for concurrent ping/stop safety."""

    def test_ping_during_stop_is_safe(self) -> None:
        """Pinging while stop is in progress doesn't crash."""
        watchdog = EventLoopWatchdog(timeout=_SHORT_TIMEOUT_S)
        watchdog.start()

        def ping_loop() -> None:
            for _ in range(100):
                watchdog.ping()
                time.sleep(0.001)

        thread = threading.Thread(target=ping_loop)
        thread.start()
        watchdog.stop()
        thread.join(timeout=1.0)