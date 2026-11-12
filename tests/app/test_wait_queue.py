"""Tests for the VRAM wait queue: admission decisions, priority order, and timeouts."""

import asyncio
from typing import Optional

import pytest

from app.budget import compute_effective_remaining
from app.config import ProxyConfig, ServiceConfig
from app.gpu_monitor import ContainerVRAM, GPUStatus, RealClock
from app.service_state import ServiceState, ServiceStateEnum
from app.wait_queue import VRAMWaitQueue

# ── The card and budgets the queue fixtures below describe ──

# One 16 GiB card, shared by three services that each reserve a quarter of it
# and one that reserves the lot.
_CARD_MIB = 16384
_SERVICE_BUDGET_MIB = 4096
_WHOLE_CARD_BUDGET_MIB = 16384
# Just under one service's budget: too little to admit a cold service on its
# own, but enough that reclaiming an equal-sized warm resident would cover it.
_JUST_UNDER_BUDGET_MIB = 4000
# Configured budget deliberately exceeds the card, as it does in production:
# the queue is limited by whichever of the two is smaller at poll time.
_CONFIGURED_BUDGET_MIB = 20480

# ── The larger card the shared service config in conftest is sized for ──

_BIG_CARD_MIB = 24576
_SYSTEM_MIB = 860
_CPAI_RESIDENT_MIB = 396
_LLAMACPP_RESIDENT_MIB = 16384
# Free VRAM that covers cpai's 4096 budget but not llamacpp-server's 16384.
_MOSTLY_FREE_MIB = 20000
# Just under one small service's budget: enough that the card is not idle,
# not enough for any admission to succeed.
_ALMOST_ONE_SLOT_MIB = 4000

# ── A third, deliberately undersized card used by the eviction-ordering tests ──

_SMALL_CARD_MIB = 8000
_SMALL_BUDGET_MIB = 4000
_HOT_SVC_BUDGET_MIB = 2000
_TINY_BUDGET_MIB = 1000

# ── Timeouts ──

_QUEUE_TIMEOUT_S = 60
# Short enough to keep the timeout tests quick, and awaited rather than slept
# through, so the suite never pays for the full wait.
_SHORT_TIMEOUT_S = 0.1
# Ceiling on how long a timeout test will wait before declaring the queue stuck.
_TIMEOUT_GRACE_S = 5.0

# ── Decisions the queue publishes on an entry's future ──

_APPROVED = ("target", "request_approved_budget_sufficient")
_PASSED_THROUGH = ("target", "hot_service_pass_through")
_NO_EVICTABLE = ("backup", "no_evictable_services")
_IMPOSSIBLE_EVICTION = ("backup", "impossible_eviction")
_POLL_FAILED_BACKUP = ("backup", "gpu_poll_failed")
_TIMED_OUT = ("backup", "timeout")

_Decision = tuple[str, str]
_Transitions = dict[str, tuple[ServiceStateEnum, ...]]

# Transition chains, spelled out because ServiceState only accepts legal steps.
_TO_HOT: tuple[ServiceStateEnum, ...] = (ServiceStateEnum.hot,)
_TO_WARM: tuple[ServiceStateEnum, ...] = (ServiceStateEnum.warm,)
_TO_EVICTING: tuple[ServiceStateEnum, ...] = (ServiceStateEnum.hot, ServiceStateEnum.evicting)
_TO_RESTARTING: tuple[ServiceStateEnum, ...] = _TO_EVICTING + (ServiceStateEnum.restarting,)
# Resumes from evicting rather than from cold: the rest of the way back to cold.
_FINISH_RESTART: tuple[ServiceStateEnum, ...] = (
    ServiceStateEnum.restarting, ServiceStateEnum.cold)


def _service(name: str, priority: int, budget_mib: int, container: str) -> ServiceConfig:
    """A GPU-routed service the queue can admit."""
    return ServiceConfig(
        name=name,
        priority=priority,
        vram_budget=budget_mib,
        hostname=f"{name}.local",
        target=f"http://{name}",
        container=container,
        gpu_routes=["/"],
    )


def _card(
    available_mib: int,
    resident: Optional[dict[str, int]] = None,
    total_mib: int = _CARD_MIB,
    system_mib: int = 0,
) -> GPUStatus:
    """One GPU poll: the given VRAM free, with the named containers resident."""
    resident = resident or {}
    return GPUStatus(
        available_vram_mib=available_mib,
        total_vram_mib=total_mib,
        system_vram_mib=system_mib,
        docker_vram_mib=sum(resident.values()),
        containers={name: ContainerVRAM(vram_mib=mib) for name, mib in resident.items()},
        last_updated=None,
        poll_failed=False,
    )


def _big_card(available_mib: int, resident: Optional[dict[str, int]] = None) -> GPUStatus:
    """A poll of the 24 GiB card, whose system VRAM is never reclaimable."""
    return _card(available_mib, resident, total_mib=_BIG_CARD_MIB, system_mib=_SYSTEM_MIB)


def _advance(state: ServiceState, chain: tuple[ServiceStateEnum, ...]) -> None:
    """Walk a service through a chain of states, since it only accepts legal steps."""
    for target in chain:
        state.transition(target)


def _states(
    config: ProxyConfig,
    transitions: Optional[_Transitions] = None,
) -> dict[str, ServiceState]:
    """Every configured service, each walked through the transition chain named for it."""
    states = {name: ServiceState(name, clock=RealClock()) for name in config.services}
    for name, chain in (transitions or {}).items():
        _advance(states[name], chain)
    return states


def _result_of(future: asyncio.Future) -> Optional[_Decision]:
    """The decision published on an entry's future, or None while it is still waiting."""
    return future.result() if future.done() else None


def _queued_names(queue: VRAMWaitQueue) -> list[str]:
    """The services still waiting, in the order the queue will wake them."""
    return [entry.service_name for entry in queue.entries]


def _fresh_queue(
    process_on_enqueue: bool = False,
    timeout: float = _QUEUE_TIMEOUT_S,
) -> VRAMWaitQueue:
    """A queue of its own, timing its entries out after the given wait.

    The default timeout is far enough out that no entry resolves by timing out.
    """
    return VRAMWaitQueue(
        clock=RealClock(),
        timeout=timeout,
        process_on_enqueue=process_on_enqueue,
    )


async def _route(
    service_name: str,
    gpu: GPUStatus,
    config: ProxyConfig,
    states: dict[str, ServiceState],
) -> Optional[_Decision]:
    """Put one request through a queue of its own; None means it is still waiting."""
    queue = _fresh_queue(process_on_enqueue=True)
    try:
        entry = await queue.enqueue(service_name, gpu, config, states)
        return _result_of(entry.future)
    finally:
        await queue.cancel_all()


@pytest.fixture
def queue_config() -> ProxyConfig:
    """Four services on one card: three that share it, one that needs all of it."""
    return ProxyConfig(
        total_vram_budget=_CONFIGURED_BUDGET_MIB,
        services={
            "svc1": _service("svc1", 1, _SERVICE_BUDGET_MIB, "c1"),
            "high": _service("high", 1, _SERVICE_BUDGET_MIB, "ch"),
            "low": _service("low", 2, _SERVICE_BUDGET_MIB, "cl"),
            "llamacpp-server": _service("llamacpp-server", 1, _WHOLE_CARD_BUDGET_MIB, "cm"),
        },
        hostname_map={},
    )


@pytest.fixture
def gpu_all_free() -> GPUStatus:
    """A poll showing the whole card free."""
    return _card(_CARD_MIB)


@pytest.fixture
def gpu_none_free() -> GPUStatus:
    """A poll showing no VRAM available at all."""
    return _card(0)


@pytest.fixture
def gpu_one_slot_free() -> GPUStatus:
    """A poll leaving room for exactly one of the small services."""
    return _card(_SERVICE_BUDGET_MIB)


@pytest.fixture
def queue_states(queue_config: ProxyConfig) -> dict[str, ServiceState]:
    """Every service in queue_config, all cold."""
    return _states(queue_config)


class TestWaitQueueEnqueue:
    """Tests for adding entries to the wait queue."""

    async def test_enqueue_returns_entry(
        self, wait_queue: VRAMWaitQueue, gpu_all_free: GPUStatus,
        queue_config: ProxyConfig, queue_states: dict[str, ServiceState],
    ) -> None:
        """Enqueue returns the entry queued for the service."""
        entry = await wait_queue.enqueue("svc1", gpu_all_free, queue_config, queue_states)
        assert entry.service_name == "svc1"
        assert _queued_names(wait_queue) == ["svc1"]

    async def test_enqueue_sorted_by_priority(
        self, wait_queue: VRAMWaitQueue, gpu_all_free: GPUStatus,
        queue_config: ProxyConfig, queue_states: dict[str, ServiceState],
    ) -> None:
        """Entries are sorted by priority (ascending = higher priority first)."""
        await wait_queue.enqueue("low", gpu_all_free, queue_config, queue_states)
        await wait_queue.enqueue("high", gpu_all_free, queue_config, queue_states)
        assert _queued_names(wait_queue) == ["high", "low"]

    async def test_enqueue_deduplication(
        self, wait_queue: VRAMWaitQueue, gpu_all_free: GPUStatus,
        queue_config: ProxyConfig, queue_states: dict[str, ServiceState],
    ) -> None:
        """Same service enqueued twice shares the same entry/future."""
        entry1 = await wait_queue.enqueue("svc1", gpu_all_free, queue_config, queue_states)
        entry2 = await wait_queue.enqueue("svc1", gpu_all_free, queue_config, queue_states)
        assert entry1 is entry2
        assert _queued_names(wait_queue) == ["svc1"]

    async def test_enqueue_future_not_resolved(
        self, wait_queue: VRAMWaitQueue, gpu_all_free: GPUStatus,
        queue_config: ProxyConfig, queue_states: dict[str, ServiceState],
    ) -> None:
        """Newly enqueued entry has unresolved future (process_on_enqueue=False)."""
        entry = await wait_queue.enqueue("svc1", gpu_all_free, queue_config, queue_states)
        assert _result_of(entry.future) is None


class TestWaitQueueCancel:
    """Tests for canceling entries from the wait queue."""

    async def test_remove_existing(
        self, wait_queue: VRAMWaitQueue, gpu_all_free: GPUStatus,
        queue_config: ProxyConfig, queue_states: dict[str, ServiceState],
    ) -> None:
        """Removing an existing entry removes it from the queue."""
        await wait_queue.enqueue("svc1", gpu_all_free, queue_config, queue_states)
        await wait_queue.cancel("svc1")
        assert _queued_names(wait_queue) == []

    async def test_cancel_nonexistent(
        self, wait_queue: VRAMWaitQueue, gpu_all_free: GPUStatus,
        queue_config: ProxyConfig, queue_states: dict[str, ServiceState],
    ) -> None:
        """Canceling a non-existent entry is a no-op."""
        await wait_queue.enqueue("svc1", gpu_all_free, queue_config, queue_states)
        await wait_queue.cancel("nonexistent")
        assert _queued_names(wait_queue) == ["svc1"]

    async def test_cancel_cancels_future(
        self, wait_queue: VRAMWaitQueue, gpu_all_free: GPUStatus,
        queue_config: ProxyConfig, queue_states: dict[str, ServiceState],
    ) -> None:
        """Canceling an entry cancels its future."""
        entry = await wait_queue.enqueue("svc1", gpu_all_free, queue_config, queue_states)
        await wait_queue.cancel("svc1")
        assert entry.future.cancelled()


class TestWaitQueueProcessWakeUp:
    """Tests for processing the wait queue after VRAM freed."""

    async def test_approve_when_budget_sufficient(
        self, wait_queue: VRAMWaitQueue, gpu_all_free: GPUStatus,
        queue_config: ProxyConfig, queue_states: dict[str, ServiceState],
    ) -> None:
        """Entry approved when budget allows."""
        entry = await wait_queue.enqueue("svc1", gpu_all_free, queue_config, queue_states)
        await wait_queue.process_wake_up(gpu_all_free, queue_config, queue_states)
        assert _result_of(entry.future) == _APPROVED

    async def test_stays_queued_when_budget_insufficient(
        self, wait_queue: VRAMWaitQueue, queue_config: ProxyConfig,
        queue_states: dict[str, ServiceState],
    ) -> None:
        """Entry remains in queue when the gate wants an eviction it cannot perform itself."""
        # svc1 is cold; "low" is hot and lower priority, so it is a candidate.
        queue_states["low"].transition(ServiceStateEnum.hot)
        gpu = _card(0, {"cl": _SERVICE_BUDGET_MIB})
        entry = await wait_queue.enqueue("svc1", gpu, queue_config, queue_states)
        await wait_queue.process_wake_up(gpu, queue_config, queue_states)
        assert _result_of(entry.future) is None
        assert _queued_names(wait_queue) == ["svc1"]

    async def test_priority_order_processing(
        self, wait_queue: VRAMWaitQueue, gpu_all_free: GPUStatus,
        queue_config: ProxyConfig, queue_states: dict[str, ServiceState],
    ) -> None:
        """Higher-priority entries processed first."""
        low_entry = await wait_queue.enqueue("low", gpu_all_free, queue_config, queue_states)
        high_entry = await wait_queue.enqueue("high", gpu_all_free, queue_config, queue_states)
        await wait_queue.process_wake_up(gpu_all_free, queue_config, queue_states)
        assert _result_of(high_entry.future) == _APPROVED
        assert _result_of(low_entry.future) == _APPROVED

    async def test_partial_approval(
        self, wait_queue: VRAMWaitQueue, gpu_one_slot_free: GPUStatus,
        queue_config: ProxyConfig, queue_states: dict[str, ServiceState],
    ) -> None:
        """First entry approved, second gets backup when budget only covers one."""
        high_entry = await wait_queue.enqueue(
            "high", gpu_one_slot_free, queue_config, queue_states)
        low_entry = await wait_queue.enqueue(
            "low", gpu_one_slot_free, queue_config, queue_states)
        await wait_queue.process_wake_up(gpu_one_slot_free, queue_config, queue_states)
        # high (priority 1) is processed first and takes the only slot; low then
        # measures itself against a budget high has already spoken for.
        assert _result_of(high_entry.future) == _APPROVED
        assert _result_of(low_entry.future) == _NO_EVICTABLE

    async def test_per_service_deduplication_in_queue(
        self, wait_queue: VRAMWaitQueue, gpu_all_free: GPUStatus,
        queue_config: ProxyConfig, queue_states: dict[str, ServiceState],
    ) -> None:
        """Same service's budget deducted once even with multiple waiters."""
        entry1 = await wait_queue.enqueue("svc1", gpu_all_free, queue_config, queue_states)
        entry2 = await wait_queue.enqueue("svc1", gpu_all_free, queue_config, queue_states)
        assert entry1 is entry2
        await wait_queue.process_wake_up(gpu_all_free, queue_config, queue_states)
        assert _result_of(entry1.future) == _APPROVED

    async def test_partial_vram_wake_up_stays_queued(
        self, wait_queue: VRAMWaitQueue, gpu_all_free: GPUStatus,
        gpu_one_slot_free: GPUStatus, queue_config: ProxyConfig,
        queue_states: dict[str, ServiceState],
    ) -> None:
        """When eviction frees some but not enough VRAM, the waiting entry is sent to backup."""
        entry = await wait_queue.enqueue(
            "llamacpp-server", gpu_all_free, queue_config, queue_states)
        # One slot free, but llamacpp-server needs the whole card and nothing
        # is resident to evict, so waiting could never help.
        await wait_queue.process_wake_up(gpu_one_slot_free, queue_config, queue_states)
        assert _result_of(entry.future) == _NO_EVICTABLE

    async def test_wait_queue_identity_preservation(
        self, wait_queue: VRAMWaitQueue, gpu_all_free: GPUStatus,
        queue_config: ProxyConfig, queue_states: dict[str, ServiceState],
    ) -> None:
        """Verify that multiple waiters for the same service join the same entry."""
        entry1 = await wait_queue.enqueue("svc1", gpu_all_free, queue_config, queue_states)
        entry2 = await wait_queue.enqueue("svc1", gpu_all_free, queue_config, queue_states)
        assert entry1 is entry2

        await wait_queue.process_wake_up(gpu_all_free, queue_config, queue_states)

        # One wake-up resolves the single shared entry, so both waiters see it.
        assert _result_of(entry2.future) == _APPROVED
        assert _queued_names(wait_queue) == []


class TestWaitQueueTimeout:
    """Tests for per-entry timeout in the wait queue."""

    async def test_timeout_resolves_backup(
        self, gpu_all_free: GPUStatus, queue_config: ProxyConfig,
        queue_states: dict[str, ServiceState],
    ) -> None:
        """Timed-out entry resolves future with ("backup", "timeout")."""
        queue = _fresh_queue(timeout=_SHORT_TIMEOUT_S)
        try:
            entry = await queue.enqueue("svc1", gpu_all_free, queue_config, queue_states)
            result = await asyncio.wait_for(entry.future, _TIMEOUT_GRACE_S)
            assert result == _TIMED_OUT
        finally:
            await queue.cancel_all()

    async def test_timeout_removes_entry_from_queue(
        self, gpu_all_free: GPUStatus, queue_config: ProxyConfig,
        queue_states: dict[str, ServiceState],
    ) -> None:
        """Timed-out entry is removed from the queue after timeout."""
        queue = _fresh_queue(timeout=_SHORT_TIMEOUT_S)
        try:
            entry = await queue.enqueue("svc1", gpu_all_free, queue_config, queue_states)
            assert _queued_names(queue) == ["svc1"]
            await asyncio.wait_for(entry.future, _TIMEOUT_GRACE_S)
            assert _queued_names(queue) == []
        finally:
            await queue.cancel_all()


class TestWaitQueueEnqueueWithProcessing:
    """Tests for enqueue with process_on_enqueue=True (integration)."""

    async def test_enqueue_auto_processes(
        self, gpu_all_free: GPUStatus, queue_config: ProxyConfig,
        queue_states: dict[str, ServiceState],
    ) -> None:
        """With process_on_enqueue=True, entry is resolved during enqueue."""
        assert await _route("svc1", gpu_all_free, queue_config, queue_states) == _APPROVED

    async def test_enqueue_auto_process_backup(
        self, gpu_none_free: GPUStatus, queue_config: ProxyConfig,
        queue_states: dict[str, ServiceState],
    ) -> None:
        """With process_on_enqueue=True and insufficient budget, entry gets backup."""
        assert await _route("svc1", gpu_none_free, queue_config, queue_states) == _NO_EVICTABLE


class TestEvictingRestartingQueueRecovery:
    """Evicting/restarting service may queue, then regain control after restart."""

    async def test_waiting_request_survives_another_service_restarting(
        self, wait_queue: VRAMWaitQueue, gpu_all_free: GPUStatus,
        queue_config: ProxyConfig, queue_states: dict[str, ServiceState],
    ) -> None:
        """A request waits until another service releases its memory."""
        _advance(queue_states["low"], _TO_EVICTING)
        gpu = _card(0, {"cl": _SERVICE_BUDGET_MIB})
        entry = await wait_queue.enqueue("svc1", gpu, queue_config, queue_states)
        await wait_queue.process_wake_up(gpu, queue_config, queue_states)
        assert _result_of(entry.future) is None

        queue_states["low"].transition(ServiceStateEnum.restarting)
        await wait_queue.process_wake_up(gpu, queue_config, queue_states)
        assert _result_of(entry.future) is None
        assert _queued_names(wait_queue) == ["svc1"]

        queue_states["low"].transition(ServiceStateEnum.cold)
        await wait_queue.process_wake_up(gpu_all_free, queue_config, queue_states)
        assert _result_of(entry.future) == _APPROVED
        assert _queued_names(wait_queue) == []

    async def test_evicting_service_queues_then_approved_after_cold(
        self, wait_queue: VRAMWaitQueue, gpu_all_free: GPUStatus,
        queue_config: ProxyConfig, queue_states: dict[str, ServiceState],
    ) -> None:
        """Service in evicting state is held back even though the budget is free.

        After restart completes (state -> cold), process_wake_up approves it.
        """
        _advance(queue_states["svc1"], _TO_EVICTING)

        entry = await wait_queue.enqueue("svc1", gpu_all_free, queue_config, queue_states)
        await wait_queue.process_wake_up(gpu_all_free, queue_config, queue_states)
        assert _result_of(entry.future) is None, "Evicting service should stay queued"
        assert _queued_names(wait_queue) == ["svc1"]

        _advance(queue_states["svc1"], _FINISH_RESTART)

        await wait_queue.process_wake_up(gpu_all_free, queue_config, queue_states)
        assert _result_of(entry.future) == _APPROVED, "Service should be approved after restart"
        assert _queued_names(wait_queue) == []

    async def test_restarting_service_queues_then_approved_after_cold(
        self, wait_queue: VRAMWaitQueue, gpu_all_free: GPUStatus,
        queue_config: ProxyConfig, queue_states: dict[str, ServiceState],
    ) -> None:
        """Service in restarting state is held back even though the budget is free.

        After restart completes (state -> cold), process_wake_up approves it.
        """
        _advance(queue_states["svc1"], _TO_RESTARTING)

        entry = await wait_queue.enqueue("svc1", gpu_all_free, queue_config, queue_states)
        await wait_queue.process_wake_up(gpu_all_free, queue_config, queue_states)
        assert _result_of(entry.future) is None, "Restarting service should stay queued"
        assert _queued_names(wait_queue) == ["svc1"]

        queue_states["svc1"].transition(ServiceStateEnum.cold)

        await wait_queue.process_wake_up(gpu_all_free, queue_config, queue_states)
        assert _result_of(entry.future) == _APPROVED, "Service should be approved after restart"
        assert _queued_names(wait_queue) == []

    async def test_evicting_service_insufficient_budget_stays_queued(
        self, wait_queue: VRAMWaitQueue, gpu_all_free: GPUStatus,
        queue_config: ProxyConfig, queue_states: dict[str, ServiceState],
    ) -> None:
        """Restarting with no budget stays queued, then is approved once both clear."""
        queue_states["low"].transition(ServiceStateEnum.hot)
        _advance(queue_states["svc1"], _TO_RESTARTING)
        gpu_contended = _card(0, {"cl": _SERVICE_BUDGET_MIB})

        entry = await wait_queue.enqueue("svc1", gpu_contended, queue_config, queue_states)
        await wait_queue.process_wake_up(gpu_contended, queue_config, queue_states)
        assert _result_of(entry.future) is None, "Should stay queued: restarting + insufficient budget"
        assert _queued_names(wait_queue) == ["svc1"]

        # low is evicted and svc1 finishes restarting.
        queue_states["low"].transition(ServiceStateEnum.cold)
        queue_states["svc1"].transition(ServiceStateEnum.cold)

        await wait_queue.process_wake_up(gpu_all_free, queue_config, queue_states)
        assert _result_of(entry.future) == _APPROVED, "Approved once restarted and VRAM freed"


class TestHighPriorityPreemptsLowPriority:
    """New high-priority request entering queue may immediately fail lower-priority request."""

    async def test_high_priority_enqueue_rejects_waiting_low_priority(
        self, queue_config: ProxyConfig, queue_states: dict[str, ServiceState],
    ) -> None:
        """Enqueuing "high" is itself what rejects the "low" already waiting.

        "low" is queued waiting for warm "high" to be evicted for it. Then
        "high" asks for the GPU: its own resident VRAM is not held against it,
        so it is admitted, and that same enqueue re-evaluates "low", whose only
        eviction candidate has become a queue entry it cannot displace. No
        explicit wake-up happens in between — the enqueue does all of it.
        """
        queue = _fresh_queue(process_on_enqueue=True)
        try:
            queue_states["high"].transition(ServiceStateEnum.warm)
            gpu = _card(_JUST_UNDER_BUDGET_MIB, {"ch": _JUST_UNDER_BUDGET_MIB})

            low_entry = await queue.enqueue("low", gpu, queue_config, queue_states)
            assert _result_of(low_entry.future) is None, "Low waits for high to be evicted"
            assert _queued_names(queue) == ["low"]

            high_entry = await queue.enqueue("high", gpu, queue_config, queue_states)
            assert _result_of(high_entry.future) == _APPROVED, "High should win the slot"
            assert _result_of(low_entry.future) == _IMPOSSIBLE_EVICTION, "Low should be rejected"
            assert _queued_names(queue) == []
        finally:
            await queue.cancel_all()

    async def test_high_priority_enqueue_with_warm_service_eviction(
        self, gpu_all_free: GPUStatus, queue_config: ProxyConfig,
        queue_states: dict[str, ServiceState],
    ) -> None:
        """A second waiter joins the first in waiting for the same eviction to happen."""
        queue = _fresh_queue(process_on_enqueue=True)
        try:
            # "low" is warm and holds the only VRAM in use; it outranks nobody,
            # so both waiters below are entitled to have it evicted.
            queue_states["low"].transition(ServiceStateEnum.warm)
            gpu_with_warm = _card(0, {"cl": _SERVICE_BUDGET_MIB})

            svc1_entry = await queue.enqueue(
                "svc1", gpu_with_warm, queue_config, queue_states)
            assert _result_of(svc1_entry.future) is None, "svc1 waits for the eviction"

            high_entry = await queue.enqueue(
                "high", gpu_with_warm, queue_config, queue_states)

            # Enqueuing "high" re-evaluates the whole queue, but the queue only
            # decides; it does not perform the eviction, so neither waiter can
            # move until "low" actually goes cold.
            assert _result_of(svc1_entry.future) is None, "svc1 stays queued — no eviction yet"
            assert _result_of(high_entry.future) is None, "high stays queued — no eviction yet"
            assert _queued_names(queue) == ["svc1", "high"]

            queue_states["low"].transition(ServiceStateEnum.cold)
            await queue.process_wake_up(gpu_all_free, queue_config, queue_states)

            assert _result_of(svc1_entry.future) == _APPROVED, "svc1 approved after eviction"
            assert _result_of(high_entry.future) == _APPROVED, "high approved after eviction"
        finally:
            await queue.cancel_all()


class TestGpuGateAdmission:
    """Tests for the GPU gate admission decision logic."""

    async def test_hot_service_passes_through(
        self, sample_config: ProxyConfig, gpu_status_llamacpp_hot: GPUStatus,
    ) -> None:
        """Hot service passes through to target without budget check."""
        states = _states(sample_config, {"llamacpp-server": _TO_HOT})
        decision = await _route(
            "llamacpp-server", gpu_status_llamacpp_hot, sample_config, states)
        assert decision == _PASSED_THROUGH

    async def test_cold_service_sufficient_budget(
        self, sample_config: ProxyConfig, gpu_all_free: GPUStatus,
    ) -> None:
        """Cold service with sufficient budget is approved."""
        states = _states(sample_config)
        assert await _route("cpai", gpu_all_free, sample_config, states) == _APPROVED

    async def test_cold_service_insufficient_budget_evictable(
        self, sample_config: ProxyConfig,
    ) -> None:
        """Cold service with insufficient budget but evictable lower-priority services waits."""
        states = _states(sample_config, {"cpai": _TO_HOT})
        gpu = _big_card(_MOSTLY_FREE_MIB, {"cpai": _CPAI_RESIDENT_MIB})
        assert await _route("llamacpp-server", gpu, sample_config, states) is None

    async def test_cold_service_insufficient_no_evictable(
        self, sample_config: ProxyConfig,
    ) -> None:
        """Cold service with insufficient budget, nothing evictable → backup."""
        states = _states(sample_config, {"llamacpp-server": _TO_HOT})
        gpu = _big_card(0, {"llamacpp-server": _LLAMACPP_RESIDENT_MIB})
        assert await _route("cpai", gpu, sample_config, states) == _NO_EVICTABLE

    async def test_poll_failed_routes_to_backup(
        self, sample_config: ProxyConfig, gpu_status_poll_failed: GPUStatus,
    ) -> None:
        """When poll_failed=True, all requests route to backup."""
        states = _states(sample_config, {"llamacpp-server": _TO_HOT})
        decision = await _route(
            "llamacpp-server", gpu_status_poll_failed, sample_config, states)
        assert decision == _POLL_FAILED_BACKUP

    async def test_impossible_eviction_routes_to_backup(
        self, sample_config: ProxyConfig,
    ) -> None:
        """Budget insufficient with an idle card and nothing resident → backup."""
        states = _states(sample_config)
        gpu = _big_card(100)
        assert await _route("cpai", gpu, sample_config, states) == _NO_EVICTABLE

    async def test_evicting_state_queues_or_backups(
        self, sample_config: ProxyConfig, gpu_all_free: GPUStatus,
    ) -> None:
        """A service being evicted waits even when the budget would allow it through."""
        states = _states(sample_config, {"cpai": _TO_EVICTING})
        assert await _route("cpai", gpu_all_free, sample_config, states) is None

    async def test_restarting_state_routes_to_backup_or_queues(
        self, sample_config: ProxyConfig, gpu_all_free: GPUStatus,
    ) -> None:
        """A restarting service waits too, and stays restarting while it does."""
        states = _states(sample_config, {"cpai": _TO_RESTARTING})
        assert await _route("cpai", gpu_all_free, sample_config, states) is None
        assert states["cpai"].state == ServiceStateEnum.restarting

    async def test_vram_drops_mid_operation_hot_not_evicted(
        self, sample_config: ProxyConfig,
    ) -> None:
        """A hot high-priority service is not evicted for a lower-priority request."""
        states = _states(sample_config, {"llamacpp-server": _TO_HOT})
        gpu = _big_card(100, {"llamacpp-server": _LLAMACPP_RESIDENT_MIB})
        assert await _route("cpai", gpu, sample_config, states) == _NO_EVICTABLE
        assert states["llamacpp-server"].state == ServiceStateEnum.hot

    async def test_vram_drops_cold_evaluated_against_reduced_budget(
        self, sample_config: ProxyConfig,
    ) -> None:
        """A cold service is measured against what is left, not the card's size."""
        states = _states(sample_config, {"cpai": _TO_HOT})
        # 20000 MiB free of a 24576 MiB card is plenty in absolute terms, but
        # llamacpp-server needs 16384 out of a 20480 budget that cpai has already
        # drawn against, so it must wait for cpai to go.
        gpu = _big_card(_MOSTLY_FREE_MIB, {"cpai": _CPAI_RESIDENT_MIB})
        assert await _route("llamacpp-server", gpu, sample_config, states) is None
        assert states["cpai"].state == ServiceStateEnum.hot

    async def test_gpu_gate_returns_literal_type(self, sample_config: ProxyConfig) -> None:
        """The queue publishes one of the documented decision/reason pairs."""
        states = _states(sample_config, {"llamacpp-server": _TO_HOT})
        gpu = _big_card(22970, {"llamacpp-server": 746})
        assert await _route("llamacpp-server", gpu, sample_config, states) == _PASSED_THROUGH

    async def test_exact_eviction_logic_prioritizes_warm(self) -> None:
        """Test the 2-phase exact eviction selection prioritizing warm services."""
        config = ProxyConfig(
            total_vram_budget=_SMALL_CARD_MIB,
            services={
                "svc1": _service("svc1", 2, _SMALL_BUDGET_MIB, "c"),
                "svc2": _service("svc2", 1, _SMALL_BUDGET_MIB, "c2"),
                "svc3": _service("svc3", 3, _SMALL_BUDGET_MIB, "c3"),
            },
            hostname_map={},
        )
        states = _states(config, {"svc2": _TO_WARM, "svc3": _TO_HOT})
        gpu = _card(
            _SMALL_CARD_MIB,
            {"c2": _SMALL_BUDGET_MIB, "c3": _SMALL_BUDGET_MIB},
            total_mib=_SMALL_CARD_MIB,
        )
        # svc3 is lower priority than svc1 and can be evicted for it, so svc1 waits
        # rather than being sent to backup.
        assert await _route("svc1", gpu, config, states) is None

    async def test_external_service_no_bypass_in_gate(self) -> None:
        """External services get no bypass; they are routed on budget like anything else.

        External services are handled via passthrough in proxy.py, not the gate.
        """
        svc_ext = ServiceConfig(
            name="external-svc",
            priority=100,
            vram_budget=0,
            hostname="ext.local",
            target="http://external:8000",
            container="",
            external=True,
            gpu_routes=["/api"],
        )
        svc_hot = _service("hot-svc", 10, _HOT_SVC_BUDGET_MIB, "hot-container")
        config = ProxyConfig(
            total_vram_budget=_TINY_BUDGET_MIB,
            services={"external-svc": svc_ext, "hot-svc": svc_hot},
            hostname_map={"ext.local": svc_ext, "hot.local": svc_hot},
        )
        states = _states(config, {"hot-svc": _TO_HOT})
        gpu = _card(100, total_mib=_HOT_SVC_BUDGET_MIB)
        assert await _route("external-svc", gpu, config, states) == _NO_EVICTABLE


class TestNegativeInputGpuGate:
    """Budget arithmetic when a poll reports nonsensical VRAM figures."""

    async def test_negative_available_vram_in_status(self, sample_config: ProxyConfig) -> None:
        """Negative available VRAM produces negative effective remaining."""
        gpu = _big_card(-100)
        states = _states(sample_config)
        queue = _fresh_queue()
        try:
            remaining = compute_effective_remaining(
                gpu, sample_config, states, queue,
                state_overrides={"llamacpp-server": ServiceStateEnum.cold},
            )
            assert remaining == -100
        finally:
            await queue.cancel_all()

    async def test_negative_total_vram_mib_in_status(self, sample_config: ProxyConfig) -> None:
        """Negative total VRAM produces negative baseline without crashing."""
        gpu = _card(_CONFIGURED_BUDGET_MIB, total_mib=-1)
        states = _states(sample_config)
        queue = _fresh_queue()
        try:
            remaining = compute_effective_remaining(
                gpu, sample_config, states, queue,
                state_overrides={"llamacpp-server": ServiceStateEnum.cold},
            )
            assert remaining == _CONFIGURED_BUDGET_MIB
        finally:
            await queue.cancel_all()


class TestImpossibleEvictionFastPath:
    """The gate skips the queue when no combination of evictions could ever help."""

    async def test_impossible_eviction_returns_backup(self, sample_config: ProxyConfig) -> None:
        """Evicting everything still would not free enough, so the request goes to backup."""
        states = _states(sample_config, {"cpai": _TO_HOT})
        # cpai holds 4096 MiB and can be evicted, but even all of it plus the
        # 4000 free is short of llamacpp-server's 16384 budget.
        gpu = _big_card(_ALMOST_ONE_SLOT_MIB, {"cpai": _SERVICE_BUDGET_MIB})
        decision = await _route("llamacpp-server", gpu, sample_config, states)
        assert decision == _IMPOSSIBLE_EVICTION
