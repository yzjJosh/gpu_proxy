"""VRAM wait queue with priority-ordered wake-up."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from types import TracebackType
from typing import Callable, Optional

from app.budget import compute_effective_remaining, get_eviction_candidates
from app.gpu_monitor import ClockProtocol, GPUStatus
from app.config import ProxyConfig, ServiceConfig
from app.service_state import ServiceState, ServiceStateEnum

__all__ = ["VRAMWaitQueue"]


@dataclass
class _WaitEntry:
    """A single entry in the VRAM wait queue."""

    service_name: str
    priority: int
    future: asyncio.Future = field(default_factory=asyncio.Future)
    waiters: int = 0
    _timeout_task: Optional[asyncio.Task] = None

    def attach_timeout(self, task: asyncio.Task) -> None:
        """Attach the task that will time this entry out if it waits too long."""
        self._timeout_task = task

    def cancel_timeout(self) -> None:
        """Cancel the timeout task if one is attached and still pending."""
        if self._timeout_task and not self._timeout_task.done():
            self._timeout_task.cancel()


class _ReentrantLock:
    """Simple reentrant async lock for single-threaded asyncio."""

    def __init__(self) -> None:
        self._base_lock = asyncio.Lock()
        self._owner: Optional[asyncio.Task] = None
        self._count: int = 0

    def _get_current(self) -> Optional[asyncio.Task]:
        try:
            return asyncio.current_task()
        except RuntimeError:
            return None

    async def acquire(self) -> None:
        current = self._get_current()
        if current is self._owner:
            self._count += 1
            return
        await self._base_lock.acquire()
        self._owner = current
        self._count = 1

    def release(self) -> None:
        current = self._get_current()
        if current is not self._owner:
            raise RuntimeError("release without acquire")
        self._count -= 1
        if self._count == 0:
            self._owner = None
            self._base_lock.release()

    async def __aenter__(self) -> _ReentrantLock:
        await self.acquire()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.release()


def _decide_eviction_route(
    service: ServiceConfig,
    gpu_status: GPUStatus,
    config: ProxyConfig,
    service_states: dict[str, ServiceState],
    wait_queue: VRAMWaitQueue,
) -> tuple[str, str]:
    """Decide where to route a request the current budget cannot satisfy.

    Args:
        service: The service the request is addressed to.
        gpu_status: The most recent GPU poll.
        config: The proxy configuration.
        service_states: Current state of every configured service.
        wait_queue: The queue, needed to account for already-waiting requests.

    Returns:
        A (decision, reason) pair: "wait" if evicting others would free enough
        VRAM, otherwise "backup".
    """
    _evictable, hypothetical_remaining, has_candidates = get_eviction_candidates(
        service, gpu_status, config, service_states, wait_queue
    )
    if hypothetical_remaining >= service.vram_budget:
        return "wait", "eviction_triggered"
    if has_candidates:
        return "backup", "impossible_eviction"
    # No evictable services -> route to backup
    return "backup", "no_evictable_services"


def _gpu_gate(
    service: ServiceConfig,
    gpu_status: GPUStatus,
    config: ProxyConfig,
    service_states: dict[str, ServiceState],
    wait_queue: VRAMWaitQueue,
) -> tuple[str, str]:
    """Determine routing decision for a GPU request."""

    state = service_states.get(service.name)
    if state is None:
        return "backup", "service_not_found"

    # Fail fast: poll_failed -> route ALL local GPU services to backup
    if gpu_status.poll_failed:
        return "backup", "gpu_poll_failed"

    # Fail fast: unhealthy service -> route to backup.
    if state.state == ServiceStateEnum.unhealthy:
        return "backup", "unhealthy_service"

    # Success fast: already hot -> route to target immediately
    if state.state == ServiceStateEnum.hot:
        return "target", "hot_service_pass_through"

    # Other states - compute effective remaining budget
    # Exclude requesting service from budget to avoid double-counting its own VRAM usage
    effective_remaining = compute_effective_remaining(
        gpu_status,
        config,
        service_states,
        wait_queue,
        requesting_service_name=service.name,
    )

    # Route to target if all conditions are met:
    #   - GPU budget is sufficient
    #   - Service is not evicting or restarting
    is_state_allowed = not (
        state.state in (ServiceStateEnum.evicting, ServiceStateEnum.restarting)
    )
    if effective_remaining >= service.vram_budget:
        if is_state_allowed:
            return "target", "request_approved_budget_sufficient"
        else:
            return "wait", "service_state_disallowed"

    # Budget insufficient: check if services can be evicted to satisfy the budget
    return _decide_eviction_route(
        service, gpu_status, config, service_states, wait_queue
    )


class VRAMWaitQueue:
    """Priority-ordered wait queue for GPU budget approval."""

    def __init__(
        self,
        clock: ClockProtocol,
        timeout: int = 60,
        process_on_enqueue: bool = True,
    ) -> None:
        self._entries: list[_WaitEntry] = []
        self._lock: _ReentrantLock = _ReentrantLock()
        self._timeout: int = timeout
        self._clock = clock
        self._process_on_enqueue = process_on_enqueue

    async def enqueue(
        self,
        service_name: str,
        gpu_status: GPUStatus,
        config: ProxyConfig,
        service_states: dict[str, ServiceState],
    ) -> _WaitEntry:
        """Add or join a wait queue entry for a service."""
        async with self._lock:
            # Check for existing entry (deduplication)
            for entry in self._entries:
                if entry.service_name == service_name:
                    # Entry already exists (deduplication)
                    return entry

            # Create new entry
            svc_cfg = config.services.get(service_name)
            priority = svc_cfg.priority if svc_cfg else 0
            if not svc_cfg:
                raise Exception(f"Service {service_name} not found in config")
            entry = _WaitEntry(service_name=service_name, priority=priority)

            # Insert in priority order (ascending = higher priority first)
            inserted = False
            for i, existing in enumerate(self._entries):
                if priority < existing.priority:
                    self._entries.insert(i, entry)
                    inserted = True
                    break
            if not inserted:
                self._entries.append(entry)

            # Start timeout task
            entry.attach_timeout(asyncio.create_task(self._timeout_entry(entry)))

            # Enqueue a new entry may lead to immediate rejection of existing entries. So re-evaluate wake up queue.
            if self._process_on_enqueue:
                await self.process_wake_up(gpu_status, config, service_states)
            return entry

    async def _timeout_entry(self, entry: _WaitEntry) -> None:
        """Timeout handler for a wait queue entry."""
        try:
            await self._clock.sleep(self._timeout)
        except asyncio.CancelledError:
            return
        async with self._lock:
            if not entry.future.done():
                entry.future.set_result(("backup", "timeout"))
                if entry in self._entries:
                    self._entries.remove(entry)
            entry.cancel_timeout()

    def _remove_locked(self, service_name: str) -> Optional[_WaitEntry]:
        """Remove a service's entry from the queue. Caller must hold self._lock."""
        for i, entry in enumerate(self._entries):
            if entry.service_name == service_name:
                self._entries.pop(i)
                entry.cancel_timeout()
                return entry
        return None

    async def cancel(self, service_name: str) -> None:
        """Cancel a service's entry from the queue."""
        async with self._lock:
            entry = self._remove_locked(service_name)
            if entry and not entry.future.done():
                entry.future.cancel()

    async def wait(
        self,
        entry: _WaitEntry,
        on_removed: Optional[Callable[[], None]] = None,
    ) -> tuple[str, str]:
        """Wait for a queue entry decision and clean up if the last waiter leaves.

        Args:
            entry: The queue entry to wait on.
            on_removed: Optional callback run when an unresolved entry is removed.

        Returns:
            The routing decision and reason.
        """
        if entry.future.done():
            return entry.future.result()
        entry.waiters += 1
        try:
            return await asyncio.shield(entry.future)
        finally:
            entry.waiters -= 1
            if entry.waiters == 0 and entry in self._entries:
                self._discard_entry(entry)
                if on_removed is not None:
                    on_removed()

    async def cancel_all(self) -> None:
        """Cancel all pending timeout tasks in the queue."""
        async with self._lock:
            for entry in list(self._entries):
                self._discard_entry(entry)

    def _discard_entry(self, entry: _WaitEntry) -> None:
        """Remove an entry and cancel its timeout and future."""
        if entry in self._entries:
            self._entries.remove(entry)
        entry.cancel_timeout()
        if not entry.future.done():
            entry.future.cancel()

    async def process_wake_up(
        self,
        gpu_status: GPUStatus,
        config: ProxyConfig,
        service_states: dict[str, ServiceState],
    ) -> None:
        """Process queue entries in priority order after VRAM freed."""
        async with self._lock:
            entries_to_remove = []

            for entry in list(self._entries):
                service_config = config.services.get(entry.service_name)
                if service_config is None:
                    if not entry.future.done():
                        entry.future.set_result(("backup", "service_not_found"))
                    entries_to_remove.append(entry)
                    continue
                decision, reason = _gpu_gate(
                    service_config, gpu_status, config, service_states, self
                )
                if decision == "wait":
                    # Continue waiting for this entry, but process remaining entries
                    continue
                if not entry.future.done():
                    entry.future.set_result((decision, reason))
                entries_to_remove.append(entry)

            # Remove all processed entries
            for entry in entries_to_remove:
                self._remove_locked(entry.service_name)

    @property
    def entries(self) -> list[_WaitEntry]:
        """Current entries in the queue."""
        return self._entries
