"""Tests for the VRAM budget formula and eviction-candidate selection."""

from typing import Optional

import pytest

from app.budget import compute_effective_remaining, get_eviction_candidates
from app.config import ProxyConfig, ServiceConfig
from app.gpu_monitor import ContainerVRAM, GPUStatus, RealClock
from app.service_state import ServiceState, ServiceStateEnum
from app.wait_queue import VRAMWaitQueue

# ── The card and budgets behind conftest's `sample_config` ──

_CARD_MIB = 24576
_SYSTEM_MIB = 860
_FREE_MIB = 22970
# The global cap, which sits below the card: the formula is limited by the
# smaller of what the card has free and what the config allows.
_GLOBAL_BUDGET_MIB = 20480
_LLAMACPP_BUDGET_MIB = 16384
_CPAI_BUDGET_MIB = 4096
# Free VRAM low enough that the global cap never binds, so releasing a service's
# budget shows up in the result one MiB for one MiB.
_FREE_BELOW_CAP_MIB = 4000

# ── The smaller, self-contained config the later tests build for themselves ──

_SMALL_BUDGET_MIB = 10000
# Everything below is sized against what the requesting service needs, so the
# tables can express "exactly enough", "short by half", and so on.
_REQ_BUDGET_MIB = 4000
_HALF_REQ_MIB = 2000
# What a _HALF_REQ_MIB service is really using when that differs from its budget.
_UNDER_HALF_MIB = 1000
_OVER_HALF_MIB = 3000
# The budget reserved by each service that shares its container with others.
_SHARED_CONTAINER_BUDGET_MIB = 4000
# A cold service is never charged, so its budget only has to be a legal size.
_COLD_SERVICE_BUDGET_MIB = 4000

# ── A deliberately oversized service, for the unhealthy-eviction tests ──

_BIG_BUDGET_MIB = 96000
_UNHEALTHY_BUDGET_MIB = 92000
_UNHEALTHY_RESIDENT_MIB = 90000

_LLAMACPP = "llamacpp-server"
_CPAI = "cpai"
_REQ_SERVICE = "req_svc"
# Middle of the 1 (highest) to 5 (lowest) range, so tests can put candidates
# on either side of the requester.
_REQ_PRIORITY = 3

_Transitions = dict[str, ServiceStateEnum]
# (name, priority, vram_budget) for a service a test adds to the GPU.
_ServiceSpec = tuple[str, int, int]


def _queue() -> VRAMWaitQueue:
    """An empty wait queue, for the budget calls that do not exercise queueing."""
    return VRAMWaitQueue(clock=RealClock())


def _service(
    name: str,
    priority: int,
    budget_mib: int,
    container: Optional[str] = None,
    external: bool = False,
    is_persistent: bool = False,
) -> ServiceConfig:
    """A service reserving the given budget, in a container named after it."""
    return ServiceConfig(
        name=name,
        priority=priority,
        vram_budget=budget_mib,
        hostname=f"{name}.local",
        target=f"http://127.0.0.1:8001/{name}",
        container=name if container is None else container,
        external=external,
        is_persistent=is_persistent,
    )


def _card(
    available_mib: int,
    resident: Optional[dict[str, int]] = None,
    system_mib: int = _SYSTEM_MIB,
) -> GPUStatus:
    """One GPU poll of the 24 GiB card, with the named containers resident."""
    resident = resident or {}
    return GPUStatus(
        available_vram_mib=available_mib,
        total_vram_mib=_CARD_MIB,
        system_vram_mib=system_mib,
        docker_vram_mib=sum(resident.values()),
        containers={name: ContainerVRAM(vram_mib=mib) for name, mib in resident.items()},
        last_updated=None,
        poll_failed=False,
    )


def _states(
    config: ProxyConfig,
    transitions: Optional[_Transitions] = None,
) -> dict[str, ServiceState]:
    """Every configured service, each moved into the state named for it."""
    states = {name: ServiceState(name, clock=RealClock()) for name in config.services}
    for name, target in (transitions or {}).items():
        states[name].transition(target)
    return states


def _remaining(
    gpu: GPUStatus,
    config: ProxyConfig,
    states: dict[str, ServiceState],
    queue: Optional[VRAMWaitQueue] = None,
    state_overrides: Optional[_Transitions] = None,
    requesting_service_name: Optional[str] = None,
) -> int:
    """The effective remaining budget, measured against an empty queue by default."""
    return compute_effective_remaining(
        gpu,
        config,
        states,
        queue if queue is not None else _queue(),
        state_overrides=state_overrides,
        requesting_service_name=requesting_service_name,
    )


def _small_config(services: dict[str, ServiceConfig]) -> ProxyConfig:
    """A config over the smaller budget the self-contained tests use."""
    return ProxyConfig(
        total_vram_budget=_SMALL_BUDGET_MIB, services=services, hostname_map={})


def _sole_service_config(service: ServiceConfig, budget_mib: int) -> ProxyConfig:
    """A config whose only service is the one given."""
    return ProxyConfig(
        total_vram_budget=budget_mib,
        services={service.name: service},
        hostname_map={service.hostname: service},
    )


def _add_service(
    services: dict[str, ServiceConfig],
    states: dict[str, ServiceState],
    gpu: GPUStatus,
    spec: _ServiceSpec,
    state: ServiceStateEnum,
    resident_mib: Optional[int] = None,
) -> None:
    """Register one service in the given state, resident on the GPU.

    It occupies its whole budget unless `resident_mib` says otherwise, which is
    how a test separates what a service reserved from what it actually uses.
    """
    name, priority, budget_mib = spec
    services[name] = _service(name, priority, budget_mib)
    states[name] = ServiceState(name, clock=RealClock())
    states[name].transition(state)
    gpu.containers[name] = ContainerVRAM(
        vram_mib=budget_mib if resident_mib is None else resident_mib)


def _add_services(
    services: dict[str, ServiceConfig],
    states: dict[str, ServiceState],
    gpu: GPUStatus,
    specs: list[_ServiceSpec],
    state: ServiceStateEnum,
) -> None:
    """Register each spec as a service in the given state, resident on the GPU."""
    for spec in specs:
        _add_service(services, states, gpu, spec, state)


def _evicted_names(evictable: list[tuple[ServiceConfig, ServiceState]]) -> list[str]:
    """The service names selected for eviction, in the order they were chosen."""
    return [svc.name for svc, _ in evictable]


@pytest.fixture
def requesting_service() -> ServiceConfig:
    """The service asking for budget in the eviction tests, at middle priority."""
    return _service(_REQ_SERVICE, _REQ_PRIORITY, _REQ_BUDGET_MIB, container="req_cont")


@pytest.fixture
def gpu_status() -> GPUStatus:
    """An idle card sized to the smaller budget, for tests that add containers to it."""
    return GPUStatus(
        available_vram_mib=_SMALL_BUDGET_MIB,
        total_vram_mib=_SMALL_BUDGET_MIB,
        containers={},
        poll_failed=False,
    )


class TestComputeEffectiveRemaining:
    """Tests for the canonical effective remaining budget formula."""

    def test_no_services_using_vram(
        self, sample_config: ProxyConfig, gpu_status_cold: GPUStatus,
    ) -> None:
        """No services using VRAM: effective = min(available, budget)."""
        # min(22970, 20480) = 20480, no hot/warm/queued
        assert _remaining(
            gpu_status_cold, sample_config, _states(sample_config)) == _GLOBAL_BUDGET_MIB

    def test_hot_service_uses_max_of_real_or_budget(
        self, sample_config: ProxyConfig, gpu_status_llamacpp_hot: GPUStatus,
    ) -> None:
        """Hot service uses max(real_gpu_used, vram_budget) in budget calc."""
        states = _states(sample_config, {_LLAMACPP: ServiceStateEnum.hot})
        # min(22224, 20480) - max(746, 16384) = 20480 - 16384 = 4096
        assert _remaining(gpu_status_llamacpp_hot, sample_config, states) == (
            _GLOBAL_BUDGET_MIB - _LLAMACPP_BUDGET_MIB)

    def test_hot_service_real_exceeds_budget(self, sample_config: ProxyConfig) -> None:
        """Hot service using more VRAM than budget: account for real usage."""
        oversized_mib = 20000
        free_mib = 4096
        gpu = _card(free_mib, {_LLAMACPP: oversized_mib})
        states = _states(sample_config, {_LLAMACPP: ServiceStateEnum.hot})
        # min(4096 + 20000, 20480) - max(20000, 16384) = 20480 - 20000 = 480
        assert _remaining(gpu, sample_config, states) == _GLOBAL_BUDGET_MIB - oversized_mib

    def test_warm_service_uses_real_vram(self, sample_config: ProxyConfig) -> None:
        """Warm service uses real_used (not budget)."""
        resident_mib = 1500
        gpu = _card(_FREE_MIB, {_CPAI: resident_mib})
        states = _states(sample_config, {_CPAI: ServiceStateEnum.warm})
        # min(22970, 20480) - 1500 = 20480 - 1500 = 18980
        remaining = _remaining(
            gpu, sample_config, states,
            state_overrides={_LLAMACPP: ServiceStateEnum.cold})
        assert remaining == _GLOBAL_BUDGET_MIB - resident_mib

    def test_budget_released_after_warm_to_evicting(self, sample_config: ProxyConfig) -> None:
        """After warm → evicting, the service's VRAM is no longer deducted."""
        resident_mib = 1500
        gpu = _card(_FREE_MIB, {_CPAI: resident_mib})
        states = _states(sample_config, {_CPAI: ServiceStateEnum.warm})
        queue = _queue()

        assert _remaining(gpu, sample_config, states, queue) == (
            _GLOBAL_BUDGET_MIB - resident_mib)

        states[_CPAI].transition(ServiceStateEnum.evicting)
        assert _remaining(gpu, sample_config, states, queue) == (
            _GLOBAL_BUDGET_MIB - resident_mib)

    def test_both_hot_services(
        self, sample_config: ProxyConfig, gpu_status_both_hot: GPUStatus,
    ) -> None:
        """Both services hot: deduct max(real, budget) for each."""
        states = _states(
            sample_config,
            {_LLAMACPP: ServiceStateEnum.hot, _CPAI: ServiceStateEnum.hot},
        )
        # min(21828, 20480) - max(746, 16384) - max(396, 4096) = 20480 - 16384 - 4096 = 0
        assert _remaining(gpu_status_both_hot, sample_config, states) == (
            _GLOBAL_BUDGET_MIB - _LLAMACPP_BUDGET_MIB - _CPAI_BUDGET_MIB)

    def test_available_vram_no_double_subtract_system(self, sample_config: ProxyConfig) -> None:
        """available_vram_mib already excludes system VRAM; no double-subtraction."""
        gpu = _card(_FREE_MIB)
        # min(22970, 20480) = 20480 (no double-subtract of 860)
        remaining = _remaining(
            gpu, sample_config, _states(sample_config),
            state_overrides={_LLAMACPP: ServiceStateEnum.cold})
        assert remaining == _GLOBAL_BUDGET_MIB

    def test_per_service_deduplication(
        self, sample_config: ProxyConfig, gpu_status_llamacpp_hot: GPUStatus,
    ) -> None:
        """Same service's budget counted only once regardless of request count."""
        states = _states(sample_config, {_LLAMACPP: ServiceStateEnum.hot})
        # Repeating the call stands in for several in-flight requests: the
        # deduction is a property of the service, not of the request count.
        first = _remaining(gpu_status_llamacpp_hot, sample_config, states)
        second = _remaining(gpu_status_llamacpp_hot, sample_config, states)
        assert first == _GLOBAL_BUDGET_MIB - _LLAMACPP_BUDGET_MIB
        assert second == first

    async def test_wait_queue_entries_accounted(
        self, sample_config: ProxyConfig, gpu_status_cold: GPUStatus,
    ) -> None:
        """Higher-priority wait queue entries are deducted from budget."""
        queue = VRAMWaitQueue(clock=RealClock(), timeout=60, process_on_enqueue=False)
        try:
            entry = await queue.enqueue(_LLAMACPP, gpu_status_cold, sample_config, _states(sample_config))
            states = _states(sample_config)
            assert [e.service_name for e in queue.entries] == [_LLAMACPP]

            # A caller who is not the queued service must respect its claim.
            assert _remaining(gpu_status_cold, sample_config, states, queue) == (
                _GLOBAL_BUDGET_MIB - _LLAMACPP_BUDGET_MIB)

            # The queued service itself is not charged for its own claim.
            assert _remaining(
                gpu_status_cold, sample_config, states, queue,
                requesting_service_name=entry.service_name) == _GLOBAL_BUDGET_MIB

            entry2 = await queue.enqueue(_CPAI, gpu_status_cold, sample_config, states)
            # cpai is lower priority, so it is charged for llamacpp-server but not itself.
            assert _remaining(
                gpu_status_cold, sample_config, states, queue,
                requesting_service_name=entry2.service_name) == (
                    _GLOBAL_BUDGET_MIB - _LLAMACPP_BUDGET_MIB)

            # Queueing cpai behind it must not change what llamacpp-server sees.
            assert _remaining(
                gpu_status_cold, sample_config, states, queue,
                requesting_service_name=entry.service_name) == _GLOBAL_BUDGET_MIB
        finally:
            await queue.cancel_all()


class TestComputeEffectiveRemainingStateOverrides:
    """Tests for the hypothetical states a caller can ask the formula to assume."""

    def test_compute_effective_remaining_accepts_state_overrides(
        self, sample_config: ProxyConfig, gpu_status_llamacpp_hot: GPUStatus,
    ) -> None:
        """An override naming nobody is accepted and changes nothing."""
        states = _states(sample_config, {_LLAMACPP: ServiceStateEnum.hot})
        without = _remaining(gpu_status_llamacpp_hot, sample_config, states)
        with_empty = _remaining(
            gpu_status_llamacpp_hot, sample_config, states, state_overrides={})
        assert without == _GLOBAL_BUDGET_MIB - _LLAMACPP_BUDGET_MIB
        assert with_empty == without

    def test_state_overrides_type_hint(
        self, sample_config: ProxyConfig, gpu_status_cold: GPUStatus,
    ) -> None:
        """An override carrying a ServiceStateEnum is honoured for a cold service too."""
        states = _states(sample_config)
        # Nothing is resident, so charging llamacpp-server can only come from
        # the override being read as a state in its own right.
        remaining = _remaining(
            gpu_status_cold, sample_config, states,
            state_overrides={_LLAMACPP: ServiceStateEnum.hot})
        assert remaining == _GLOBAL_BUDGET_MIB - _LLAMACPP_BUDGET_MIB

    def test_state_overrides_functional(self, sample_config: ProxyConfig) -> None:
        """Overriding a hot service to cold releases the VRAM it was charged for."""
        gpu = _card(_FREE_BELOW_CAP_MIB, {_CPAI: _CPAI_BUDGET_MIB})
        states = _states(sample_config, {_CPAI: ServiceStateEnum.hot})

        remaining_without = _remaining(gpu, sample_config, states)
        remaining_with = _remaining(
            gpu, sample_config, states,
            state_overrides={_CPAI: ServiceStateEnum.cold})

        assert remaining_without == _FREE_BELOW_CAP_MIB
        assert remaining_with == _FREE_BELOW_CAP_MIB + _CPAI_BUDGET_MIB


class TestBoundaryConditions:
    """Boundary condition tests for VRAM budget calculations and priority values."""

    def test_vram_exactly_at_budget(self) -> None:
        """Service vram_budget == total_vram_budget: exactly at the boundary, should be allowed."""
        svc = _service("big-service", 1, _GLOBAL_BUDGET_MIB)
        config = _sole_service_config(svc, _GLOBAL_BUDGET_MIB)
        gpu = _card(_CARD_MIB, system_mib=0)
        assert _remaining(gpu, config, _states(config)) == _GLOBAL_BUDGET_MIB

    def test_zero_available_vram(self, sample_config: ProxyConfig) -> None:
        """available_vram_mib == 0: no VRAM available, all cold requests route to backup."""
        gpu = _card(0, {_LLAMACPP: _CARD_MIB})
        states = _states(sample_config, {_LLAMACPP: ServiceStateEnum.hot})
        # min(0 + 24576, 20480) - max(24576, 16384) = 20480 - 24576
        assert _remaining(gpu, sample_config, states) == _GLOBAL_BUDGET_MIB - _CARD_MIB

    def test_vram_exactly_at_remaining_budget_boundary(self, sample_config: ProxyConfig) -> None:
        """Cold service vram_budget == effective_remaining: exactly at boundary."""
        reserved_system_mib = 4096
        gpu = _card(_GLOBAL_BUDGET_MIB, system_mib=reserved_system_mib)
        assert _remaining(gpu, sample_config, _states(sample_config)) == _GLOBAL_BUDGET_MIB

    def test_max_priority_value(self) -> None:
        """Very large priority value still works correctly."""
        svc = _service("low-priority", 999999, 1024)
        config = _sole_service_config(svc, _GLOBAL_BUDGET_MIB)
        gpu = _card(_GLOBAL_BUDGET_MIB, system_mib=0)
        assert _remaining(gpu, config, _states(config)) == _GLOBAL_BUDGET_MIB

    def test_min_priority_value_boundary(self) -> None:
        """Priority = 1 (minimum valid) works correctly."""
        svc = _service("high-priority", 1, 1024)
        config = _sole_service_config(svc, _GLOBAL_BUDGET_MIB)
        gpu = _card(_GLOBAL_BUDGET_MIB, system_mib=0)
        assert _remaining(gpu, config, _states(config)) == _GLOBAL_BUDGET_MIB

    def test_service_vram_budget_equals_total(self, sample_config: ProxyConfig) -> None:
        """Service with vram_budget == total_vram_budget is a valid boundary."""
        gpu = _card(_CARD_MIB, {_LLAMACPP: _LLAMACPP_BUDGET_MIB}, system_mib=0)
        remaining = _remaining(
            gpu, sample_config, _states(sample_config),
            state_overrides={_LLAMACPP: ServiceStateEnum.cold})
        assert remaining == _GLOBAL_BUDGET_MIB


class TestComputeEffectiveRemainingExtensive:
    """Deduction amounts for each combination of warm and hot residents."""

    @pytest.mark.parametrize("warm_vram, hot_vram_real, hot_vram_budget, expected_deduction", [
        (_UNDER_HALF_MIB, 0, 0, _UNDER_HALF_MIB),
        (0, 500, _HALF_REQ_MIB, _HALF_REQ_MIB),
        (0, _OVER_HALF_MIB, _HALF_REQ_MIB, _OVER_HALF_MIB),
        (_UNDER_HALF_MIB, 500, _HALF_REQ_MIB, _UNDER_HALF_MIB + _HALF_REQ_MIB),
        (_UNDER_HALF_MIB, _OVER_HALF_MIB, _HALF_REQ_MIB, _UNDER_HALF_MIB + _OVER_HALF_MIB),
    ])
    def test_deductions(
        self, gpu_status: GPUStatus, warm_vram: int, hot_vram_real: int,
        hot_vram_budget: int, expected_deduction: int,
    ) -> None:
        """Verify correct deduction amounts for warm, hot, and combined service states."""
        services: dict[str, ServiceConfig] = {}
        states: dict[str, ServiceState] = {}

        if warm_vram > 0:
            _add_service(
                services, states, gpu_status, ("warm_svc", 1, _HALF_REQ_MIB),
                ServiceStateEnum.warm, resident_mib=warm_vram)

        if hot_vram_budget > 0:
            _add_service(
                services, states, gpu_status, ("hot_svc", 1, hot_vram_budget),
                ServiceStateEnum.hot, resident_mib=hot_vram_real)

        config = _small_config(services)
        remaining = _remaining(
            gpu_status, config, states, requesting_service_name="req")
        assert remaining == _SMALL_BUDGET_MIB - expected_deduction


class TestFormulaCornerCases:
    """Cases where the container, not the service, is the unit of accounting."""

    def test_shared_container_deduplication(self, gpu_status: GPUStatus) -> None:
        """Verify that multiple services sharing a container are not double-counted in the baseline."""
        free_mib = 1000
        shared_mib = 3000
        services = {
            "svc1": _service("svc1", 1, _SHARED_CONTAINER_BUDGET_MIB, container="shared"),
            "svc2": _service("svc2", 2, _SHARED_CONTAINER_BUDGET_MIB, container="shared"),
        }
        config = _small_config(services)

        gpu_status.available_vram_mib = free_mib
        gpu_status.containers = {"shared": ContainerVRAM(vram_mib=shared_mib)}

        states = _states(config, {
            "svc1": ServiceStateEnum.hot,
            "svc2": ServiceStateEnum.warm,
        })

        # Baseline = 1000 free + 3000 resident = 4000.
        # One container, so one deduction: max(hot budget 4000, warm real 3000).
        remaining = _remaining(
            gpu_status, config, states, requesting_service_name="req")
        assert remaining == 0

    def test_negative_vram_safety(self, gpu_status: GPUStatus) -> None:
        """A poll reporting negative free VRAM still yields the arithmetic result."""
        resident_mib = 500
        deficit_mib = 1000
        gpu_status.available_vram_mib = -deficit_mib
        gpu_status.containers = {"c1": ContainerVRAM(vram_mib=resident_mib)}
        config = _small_config(
            {"svc1": _service("svc1", 1, _COLD_SERVICE_BUDGET_MIB, container="c1")})
        # min(-1000 + 500, 10000) = -500, and a cold service deducts nothing.
        assert _remaining(gpu_status, config, _states(config)) == -resident_mib


async def test_wait_queue_shared_container_deduplication(gpu_status_cold: GPUStatus) -> None:
    """Verify that multiple services sharing a container in the wait queue are not double-counted."""
    services = {
        name: _service(name, priority, _SHARED_CONTAINER_BUDGET_MIB, container="shared")
        for name, priority in (("svc1", 1), ("svc2", 2), ("svc3", 3))
    }
    config = _small_config(services)

    queue = VRAMWaitQueue(clock=RealClock(), process_on_enqueue=False)
    try:
        states = _states(config)
        entry1 = await queue.enqueue("svc1", gpu_status_cold, config, states)
        entry2 = await queue.enqueue("svc2", gpu_status_cold, config, states)

        # svc1 leads the queue: nothing ahead of it, and it is not charged for itself.
        assert _remaining(
            gpu_status_cold, config, states, queue,
            requesting_service_name=entry1.service_name) == _SMALL_BUDGET_MIB

        # svc2 is charged for svc1 ahead of it, but not for itself.
        assert _remaining(
            gpu_status_cold, config, states, queue,
            requesting_service_name=entry2.service_name) == (
                _SMALL_BUDGET_MIB - _SHARED_CONTAINER_BUDGET_MIB)

        # A newcomer is charged for both — but they share one container, so the
        # deduction is one budget, not two.
        assert _remaining(gpu_status_cold, config, states, queue) == (
            _SMALL_BUDGET_MIB - _SHARED_CONTAINER_BUDGET_MIB)
    finally:
        await queue.cancel_all()


class TestGetEvictionCandidatesExtensive:
    """Which services the eviction search selects, and in what order."""

    async def test_no_candidates(
        self, requesting_service: ServiceConfig, gpu_status: GPUStatus,
    ) -> None:
        """With nothing else configured there is nothing to evict."""
        config = _small_config({_REQ_SERVICE: requesting_service})
        evictable, hypothetical_remaining, has_candidates = get_eviction_candidates(
            requesting_service, gpu_status, config, _states(config), _queue()
        )
        assert _evicted_names(evictable) == []
        assert hypothetical_remaining == _SMALL_BUDGET_MIB
        assert has_candidates is False

    @pytest.mark.parametrize("scenario, warm_configs, hot_configs, expected_evicted", [
        (
            "tie_breaking_warm",
            [("w1", 5, _HALF_REQ_MIB), ("w2", 4, _HALF_REQ_MIB)],  # Priorities 5 (lowest) and 4
            [],
            ["w1", "w2"]
        ),
        (
            "exact_match_warm",
            [("w1", 5, _REQ_BUDGET_MIB), ("w2", 4, _HALF_REQ_MIB)],
            [],
            ["w1"]
        ),
        (
            "warm_then_hot",
            [("w1", 1, _HALF_REQ_MIB)],  # Warm but HIGH priority (1) -> still evictable!
            [("h1", 5, _HALF_REQ_MIB)],  # Hot but LOW priority (5)
            ["w1", "h1"]
        ),
        (
            "hot_equal_priority_not_evictable",
            [],
            [("h1", 3, _REQ_BUDGET_MIB)],  # Hot, priority 3 (same as req_svc) -> not evictable
            []
        ),
        (
            "hot_higher_priority_not_evictable",
            [],
            [("h1", 1, _REQ_BUDGET_MIB)],  # Hot, priority 1 (higher) -> not evictable
            []
        ),
        (
            "tie_breaking_hot",
            [],
            [("h1", 4, _HALF_REQ_MIB), ("h2", 5, _HALF_REQ_MIB)],  # Priorities 4 and 5 (both lower than 3)
            ["h2", "h1"]
        )
    ])
    def test_eviction_scenarios(
        self, requesting_service: ServiceConfig, gpu_status: GPUStatus, scenario: str,
        warm_configs: list[_ServiceSpec], hot_configs: list[_ServiceSpec],
        expected_evicted: list[str],
    ) -> None:
        """Test eviction candidate selection across multiple warm/hot service configurations."""
        services = {_REQ_SERVICE: requesting_service}
        states = {_REQ_SERVICE: ServiceState(_REQ_SERVICE, clock=RealClock())}
        _add_services(services, states, gpu_status, warm_configs, ServiceStateEnum.warm)
        _add_services(services, states, gpu_status, hot_configs, ServiceStateEnum.hot)

        config = _small_config(services)
        gpu_status.available_vram_mib = 0  # capacity

        evictable, _, has_candidates = get_eviction_candidates(
            requesting_service, gpu_status, config, states, _queue()
        )

        assert _evicted_names(evictable) == expected_evicted, f"Failed on {scenario}"
        assert has_candidates is bool(expected_evicted), f"Failed on {scenario}"

    def test_get_eviction_candidates_skips_external(
        self, requesting_service: ServiceConfig, gpu_status: GPUStatus,
    ) -> None:
        """External services should never be targeted for eviction, whether warm or hot."""
        services = {
            _REQ_SERVICE: requesting_service,
            # Both would qualify on state and priority; only `external` excludes them.
            "ext_warm": _service("ext_warm", 5, 0, container="", external=True),
            "ext_hot": _service("ext_hot", 4, 0, container="", external=True),
        }
        config = _small_config(services)
        states = _states(config, {
            "ext_warm": ServiceStateEnum.warm,
            "ext_hot": ServiceStateEnum.hot,
        })
        gpu_status.available_vram_mib = 0  # Forces eviction search

        evictable, _, has_candidates = get_eviction_candidates(
            requesting_service, gpu_status, config, states, _queue()
        )

        assert _evicted_names(evictable) == []
        assert has_candidates is False

    def _build_unhealthy_setup(
        self,
        requesting_service: ServiceConfig,
        gpu_status: GPUStatus,
        containers: dict[str, ContainerVRAM],
    ) -> tuple[ProxyConfig, dict[str, ServiceState]]:
        """Build an unhealthy eviction-candidate setup sharing state across tests."""
        unhealthy = _service(
            "llm_heavy_raw", 4, _UNHEALTHY_BUDGET_MIB, container="llamacpp-llm")
        services = {_REQ_SERVICE: requesting_service, "llm_heavy_raw": unhealthy}

        gpu_status.available_vram_mib = 0  # forces eviction search
        gpu_status.containers = containers

        config = ProxyConfig(
            total_vram_budget=_BIG_BUDGET_MIB, services=services, hostname_map={})
        return config, _states(config, {"llm_heavy_raw": ServiceStateEnum.unhealthy})

    def test_unhealthy_using_vram_is_evictable_when_container_differs_from_name(
        self, requesting_service: ServiceConfig, gpu_status: GPUStatus,
    ) -> None:
        """An unhealthy service using VRAM must be evictable even when its container
        name differs from its service name (real config uses distinct names)."""
        # Container is keyed by CONTAINER name ("llamacpp-llm"), not service name.
        config, states = self._build_unhealthy_setup(
            requesting_service, gpu_status,
            {"llamacpp-llm": ContainerVRAM(vram_mib=_UNHEALTHY_RESIDENT_MIB)},
        )

        evictable, hypothetical_remaining, has_candidates = get_eviction_candidates(
            requesting_service, gpu_status, config, states, _queue()
        )

        assert _evicted_names(evictable) == ["llm_heavy_raw"]
        assert has_candidates is True
        assert hypothetical_remaining >= requesting_service.vram_budget

    def test_unhealthy_without_container_is_not_evictable(
        self, requesting_service: ServiceConfig, gpu_status: GPUStatus,
    ) -> None:
        """An unhealthy service with no container VRAM usage must NOT be evictable."""
        # No container present in GPU status -> real_used == 0.
        config, states = self._build_unhealthy_setup(requesting_service, gpu_status, {})

        evictable, _, has_candidates = get_eviction_candidates(
            requesting_service, gpu_status, config, states, _queue()
        )

        assert _evicted_names(evictable) == []
        assert has_candidates is False


class TestPersistentEvictionCandidates:
    """Persistent services are protected from lower-priority eviction."""

    async def test_is_persistent_warm_not_evicted_by_lower_priority(
        self, requesting_service: ServiceConfig, gpu_status: GPUStatus,
    ) -> None:
        """A persistent warm service is safe from a lower-priority requester."""
        persistent_svc = _service("persist", 1, _HALF_REQ_MIB, is_persistent=True)
        services = {_REQ_SERVICE: requesting_service, "persist": persistent_svc}
        config = _small_config(services)
        states = _states(config, {"persist": ServiceStateEnum.warm})
        gpu_status.available_vram_mib = 0  # forces eviction search

        evictable, _, has_candidates = get_eviction_candidates(
            requesting_service, gpu_status, config, states, _queue()
        )

        assert _evicted_names(evictable) == []
        assert has_candidates is False

    async def test_is_persistent_warm_evicted_by_higher_priority(
        self, gpu_status: GPUStatus,
    ) -> None:
        """A higher-priority requester may still evict a persistent service."""
        high_requester = _service("high_req", 1, _REQ_BUDGET_MIB)
        persistent_svc = _service("persist", 2, _HALF_REQ_MIB, is_persistent=True)
        services = {"high_req": high_requester, "persist": persistent_svc}
        config = _small_config(services)
        states = _states(config, {"persist": ServiceStateEnum.warm})
        gpu_status.available_vram_mib = 0

        evictable, _, has_candidates = get_eviction_candidates(
            high_requester, gpu_status, config, states, _queue()
        )

        assert _evicted_names(evictable) == ["persist"]
        assert has_candidates is True

    async def test_non_persistent_selected_before_persistent(
        self, requesting_service: ServiceConfig, gpu_status: GPUStatus,
    ) -> None:
        """At equal priority, non-persistent candidates are evicted before persistent."""
        persistent_svc = _service("persist_svc", 1, _HALF_REQ_MIB, is_persistent=True)
        plain_svc = _service("plain_svc", 1, _REQ_BUDGET_MIB)
        services = {
            _REQ_SERVICE: requesting_service,
            "persist_svc": persistent_svc,
            "plain_svc": plain_svc,
        }
        config = _small_config(services)
        states = _states(config, {
            "persist_svc": ServiceStateEnum.warm,
            "plain_svc": ServiceStateEnum.warm,
        })
        gpu_status.available_vram_mib = 0

        evictable, _, has_candidates = get_eviction_candidates(
            requesting_service, gpu_status, config, states, _queue()
        )

        assert _evicted_names(evictable) == ["plain_svc"]
        assert has_candidates is True

    async def test_is_persistent_hot_not_evicted_by_lower_priority(
        self, requesting_service: ServiceConfig, gpu_status: GPUStatus,
    ) -> None:
        """A persistent hot service is safe from a lower-priority requester."""
        persistent_svc = _service("persist", 1, _HALF_REQ_MIB, is_persistent=True)
        services = {_REQ_SERVICE: requesting_service, "persist": persistent_svc}
        config = _small_config(services)
        states = _states(config, {"persist": ServiceStateEnum.hot})
        gpu_status.available_vram_mib = 0  # forces eviction search

        evictable, _, has_candidates = get_eviction_candidates(
            requesting_service, gpu_status, config, states, _queue()
        )

        assert _evicted_names(evictable) == []
        assert has_candidates is False
