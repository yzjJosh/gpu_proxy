"""Event loop watchdog: background thread that detects asyncio blocking."""

from __future__ import annotations

import logging
import sys
import threading
import traceback
from typing import Optional

__all__ = ["EventLoopWatchdog"]

_logger = logging.getLogger("watchdog")

_TIMEOUT = 60


class EventLoopWatchdog:
    """Monitors the asyncio event loop for blocking.

    Runs a background thread that expects a periodic ping from the main
    event loop. If no ping arrives within the configured interval, logs
    a warning with a stack trace of the main thread.
    """

    def __init__(self, timeout: float = _TIMEOUT) -> None:
        self._timeout = timeout
        self._ping_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the watchdog thread."""
        if self._thread is not None:
            return
        self._ping_event.set()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor, name="loop-watchdog", daemon=True)
        self._thread.start()
        _logger.info("Watchdog started (timeout=%ds)", self._timeout)

    def stop(self) -> None:
        """Stop the watchdog thread."""
        self._stop_event.set()
        self._ping_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._timeout)
            self._thread = None
        _logger.info("Watchdog stopped")

    def ping(self) -> None:
        """Signal that the event loop is alive."""
        self._ping_event.set()

    def _monitor(self) -> None:
        """Background thread: waits for pings and logs warnings on timeout."""
        while not self._stop_event.is_set():
            if not self._ping_event.wait(timeout=self._timeout):
                # lint-skip: coding.access-private
                # sys._current_frames() is the only way to capture the main
                # thread's stack from this background thread; CPython exposes
                # no public API for reading another thread's frames. Skip
                # approved by the project owner.
                frames = sys._current_frames()
                main_frame = frames.get(threading.main_thread().ident)
                if main_frame:
                    stack = traceback.extract_stack(main_frame)
                    stack_text = "".join(traceback.format_list(stack))
                    _logger.warning(
                        "Event loop blocked for >%ds — main thread stack:\n%s",
                        self._timeout,
                        stack_text,
                    )
                else:
                    _logger.warning("Event loop blocked for >%ds — main thread frame unavailable", self._timeout)
            else:
                self._ping_event.clear()