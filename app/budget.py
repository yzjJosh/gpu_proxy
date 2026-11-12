"""VRAM budget computation logic."""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.service_state import ServiceStateEnum

if TYPE_CHECKING:
    from app.gpu_monitor import GPUStatus
    from app.config import ProxyConfig, ServiceConfig
    from app.service_state import ServiceState
    from app.wait_queue import VRAMWaitQueue

__all__ = ["compute_effective_remaining", "get_eviction_candidates"]


def _sum_proxied_usage(gpu_status: GPUStatus, config: ProxyConfig) -> int:
    """Sum real VRAM usage of all UNIQUE proxied containers.

    Args:
        gpu_status: Current snapshot of GPU and container VRAM usage.
        config: Global proxy configuration.

    Returns:
        Total real VRAM usage of the proxied containers, counted once each.
    """
    proxied_usage_real = 0
    seen_containers = set()
    for svc in config.services.values():
        if (
            svc.container in gpu_status.containers
            and svc.container not in seen_containers
        ):
            proxied_usage_real += gpu_status.containers[svc.container].vram_mib
            seen_containers.add(svc.container)
    return proxied_usage_real


def _active_service_deductions(
    gpu_status: GPUStatus,
    config: ProxyConfig,
    service_states: dict[str, ServiceState],
    state_overrides: dict[str, ServiceStateEnum] | None,
    requesting_service_name: str | None,
    container_deductions: dict[str, int],
) -> None:
    """Add active-service deductions for each unique container.

    The requesting service is always treated as cold to avoid self-blocking.

    Args:
        gpu_status: Current snapshot of GPU and container VRAM usage.
        config: Global proxy configuration.
        service_states: Map of service name to its mutable state.
        state_overrides: Optional mapping of service name to hypothetical state.
        requesting_service_name: Name of the service requesting budget.
        container_deductions: Map being filled with container -> deduction.
    """
    for name, state in service_states.items():
        if name == requesting_service_name:
            effective_state = ServiceStateEnum.cold
        else:
            effective_state = (state_overrides or {}).get(name, state.state)

        svc = config.services.get(name)
        if not svc:
            continue

        real_used = 0
        if svc.container in gpu_status.containers:
            real_used = gpu_status.containers[svc.container].vram_mib

        if effective_state in (ServiceStateEnum.hot, ServiceStateEnum.cooling_down):
            container_deductions[svc.container] = max(
                container_deductions.get(svc.container, 0), real_used, svc.vram_budget
            )
        elif effective_state in (
            ServiceStateEnum.warm,
            ServiceStateEnum.evicting,
            ServiceStateEnum.restarting,
            ServiceStateEnum.unhealthy,
        ):
            container_deductions[svc.container] = max(
                container_deductions.get(svc.container, 0), real_used
            )


def _wait_queue_deductions(
    config: ProxyConfig,
    wait_queue: VRAMWaitQueue,
    requesting_service_name: str | None,
    container_deductions: dict[str, int],
) -> None:
    """Add higher-precedence wait queue deductions for each container.

    Args:
        config: Global proxy configuration.
        wait_queue: The VRAM wait queue for priority-ordered admission.
        requesting_service_name: Name of the service requesting budget.
        container_deductions: Map being filled with container -> deduction.
    """
    requesting_service_cfg = config.services.get(requesting_service_name)
    for entry in wait_queue.entries:
        if requesting_service_cfg and requesting_service_cfg.priority <= entry.priority:
            continue

        svc = config.services.get(entry.service_name)
        if svc:
            container_deductions[svc.container] = max(
                container_deductions.get(svc.container, 0), svc.vram_budget
            )


def compute_effective_remaining(
    gpu_status: GPUStatus,
    config: ProxyConfig,
    service_states: dict[str, ServiceState],
    wait_queue: VRAMWaitQueue,
    state_overrides: dict[str, ServiceStateEnum] | None = None,
    requesting_service_name: str | None = None,
) -> int:
    """Compute effective remaining VRAM budget using the canonical formula.

    Formula: min(PhysicalFree + ProxiedRealUsed, GlobalLimit) - Deductions
    where Deductions are calculated per unique container as the max of
    active budgets or real usage.

    Args:
        gpu_status: Current snapshot of GPU and container VRAM usage.
        config: Global proxy configuration.
        service_states: Map of service name to its mutable state.
        wait_queue: The VRAM wait queue for priority-ordered admission.
        state_overrides: Optional mapping of service name to hypothetical state.
        requesting_service_name: Name of the service requesting budget. This
            service's current state will always be overridden to 'cold' to
            prevent self-blocking and allow VRAM reuse.

    Returns:
        The effective remaining VRAM budget in MiB.
    """
    total_available_to_proxy = (
        gpu_status.available_vram_mib
        + _sum_proxied_usage(gpu_status, config)
    )
    effective_budget = min(total_available_to_proxy, config.total_vram_budget)

    container_deductions: dict[str, int] = {}
    _active_service_deductions(
        gpu_status,
        config,
        service_states,
        state_overrides,
        requesting_service_name,
        container_deductions,
    )
    _wait_queue_deductions(config, wait_queue, requesting_service_name, container_deductions)

    for deduction_mib in container_deductions.values():
        effective_budget -= deduction_mib

    return effective_budget


def get_eviction_candidates(
    requesting_service: ServiceConfig,
    gpu_status: GPUStatus,
    config: ProxyConfig,
    service_states: dict[str, ServiceState],
    wait_queue: VRAMWaitQueue,
) -> tuple[list[tuple[ServiceConfig, ServiceState]], int, bool]:
    """Determine the exact subset of services to evict to meet the VRAM budget.

    Prioritizes 'warm' services (Phase 1) before 'hot' services (Phase 2),
    sorting each phase by priority descending (lowest priority first).

    Args:
        requesting_service: The service configuration needing VRAM.
        gpu_status: Current snapshot of GPU and container VRAM usage.
        config: Global proxy configuration.
        service_states: Map of service name to its mutable state.
        wait_queue: The VRAM wait queue for priority-ordered admission.

    Returns:
        A tuple containing:
            - A list of (ServiceConfig, ServiceState) pairs selected for eviction.
            - The hypothetical remaining VRAM budget after those evictions.
            - A boolean indicating if ANY evictable candidates were found.
    """
    candidates = _build_eviction_candidates(
        requesting_service, gpu_status, config, service_states
    )
    effective_remaining = compute_effective_remaining(
        gpu_status,
        config,
        service_states,
        wait_queue,
        requesting_service_name=requesting_service.name,
    )
    evictable, hypothetical_remaining = _select_evictable(
        requesting_service,
        gpu_status,
        config,
        service_states,
        wait_queue,
        candidates,
        effective_remaining,
    )
    return evictable, hypothetical_remaining, len(candidates) > 0


def _build_eviction_candidates(
    requesting_service: ServiceConfig,
    gpu_status: GPUStatus,
    config: ProxyConfig,
    service_states: dict[str, ServiceState],
) -> list[tuple[ServiceConfig, ServiceState, int]]:
    """Collect the services eligible for eviction, tagged with their priority tier.

    Args:
        requesting_service: The service configuration needing VRAM.
        gpu_status: Current snapshot of GPU and container VRAM usage.
        config: Global proxy configuration.
        service_states: Map of service name to its mutable state.

    Returns:
        A list of (ServiceConfig, ServiceState, tier) tuples, sorted warm-first
        then by priority descending.
    """
    candidates: list[tuple[ServiceConfig, ServiceState, int]] = []
    for name, svc in config.services.items():
        if name == requesting_service.name or svc.external:
            continue
        if svc.is_persistent and requesting_service.priority > svc.priority:
            continue
        st = service_states.get(name)
        if not st:
            continue
        real_used = gpu_status.containers[svc.container].vram_mib if (
            svc.container in gpu_status.containers
        ) else 0
        if st.state in (
            ServiceStateEnum.warm,
            ServiceStateEnum.evicting,
            ServiceStateEnum.restarting,
            ServiceStateEnum.cooling_down,
        ):
            candidates.append((svc, st, 1))
        elif st.state == ServiceStateEnum.unhealthy and real_used > 0:
            candidates.append((svc, st, 1))
        elif (
            st.state == ServiceStateEnum.hot
            and svc.priority > requesting_service.priority
        ):
            candidates.append((svc, st, 2))

    candidates.sort(key=lambda x: (x[2], -x[0].priority))
    return candidates


def _select_evictable(
    requesting_service: ServiceConfig,
    gpu_status: GPUStatus,
    config: ProxyConfig,
    service_states: dict[str, ServiceState],
    wait_queue: VRAMWaitQueue,
    candidates: list[tuple[ServiceConfig, ServiceState, int]],
    effective_remaining: int,
) -> tuple[list[tuple[ServiceConfig, ServiceState]], int]:
    """Select the smallest set of candidates whose eviction frees enough VRAM.

    Args:
        requesting_service: The service configuration needing VRAM.
        gpu_status: Current snapshot of GPU and container VRAM usage.
        config: Global proxy configuration.
        service_states: Map of service name to its mutable state.
        wait_queue: The VRAM wait queue for priority-ordered admission.
        candidates: The candidate list, warm-first sorted.
        effective_remaining: The remaining budget before any eviction.

    Returns:
        A tuple of the evicted (ServiceConfig, ServiceState) pairs and the
        hypothetical remaining budget after those evictions.
    """
    evictable: list[tuple[ServiceConfig, ServiceState]] = []
    state_overrides: dict[str, ServiceStateEnum] = {}
    hypothetical_remaining = effective_remaining

    for svc, st, _ in candidates:
        if hypothetical_remaining >= requesting_service.vram_budget:
            break
        evictable.append((svc, st))
        state_overrides[svc.name] = ServiceStateEnum.cold
        hypothetical_remaining = compute_effective_remaining(
            gpu_status,
            config,
            service_states,
            wait_queue,
            state_overrides=state_overrides,
            requesting_service_name=requesting_service.name,
        )

    return evictable, hypothetical_remaining
