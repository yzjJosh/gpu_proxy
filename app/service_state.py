"""Service state machine and in-flight request tracking."""

from __future__ import annotations

import asyncio
import logging
from enum import Enum
from typing import Optional

from app.eviction import EvictionEventType, EvictionLog
from app.gpu_monitor import ClockProtocol

_logger = logging.getLogger("gpu_proxy")

__all__ = ["ServiceState", "ServiceStateEnum"]


class ServiceStateEnum(Enum):
    """Possible states for a proxied service."""

    cold = "cold"
    warm = "warm"  # VRAM > 0, in_flight = 0
    hot = "hot"  # in_flight > 0
    cooling_down = "cooling_down" # after hot and waiting for next gpu poll
    evicting = "evicting"
    restarting = "restarting"
    unhealthy = "unhealthy"


# Only these transitions are valid; others would corrupt state
_VALID_TRANSITIONS = {
    ServiceStateEnum.cold: {ServiceStateEnum.warm, ServiceStateEnum.hot, ServiceStateEnum.unhealthy},
    ServiceStateEnum.warm: {
        ServiceStateEnum.hot,
        ServiceStateEnum.cold,
        ServiceStateEnum.evicting,
        ServiceStateEnum.unhealthy,
    },
    ServiceStateEnum.hot: {
        ServiceStateEnum.evicting,
        ServiceStateEnum.cooling_down,
        ServiceStateEnum.unhealthy,
    },
    ServiceStateEnum.cooling_down: {
        ServiceStateEnum.hot,
        ServiceStateEnum.warm,
        ServiceStateEnum.cold,
        ServiceStateEnum.evicting,
        ServiceStateEnum.unhealthy,
    },
    ServiceStateEnum.evicting: {ServiceStateEnum.restarting},
    ServiceStateEnum.restarting: {ServiceStateEnum.cold, ServiceStateEnum.warm, ServiceStateEnum.unhealthy},
    ServiceStateEnum.unhealthy: {
        ServiceStateEnum.cold,
        ServiceStateEnum.warm,
        ServiceStateEnum.hot,
        ServiceStateEnum.restarting,
    },
}


class ServiceState:
    """Per-service mutable state with in-flight tracking."""

    def __init__(
        self,
        name: str,
        clock: ClockProtocol,
        eviction_log: Optional[EvictionLog] = None,
    ) -> None:
        self.name = name
        self.state: ServiceStateEnum = ServiceStateEnum.cold
        self.last_state_change_time = clock.time()
        self.is_healthy: bool = True
        self.in_flight: int = 0
        self.in_flight_zero: asyncio.Event = asyncio.Event()
        self.in_flight_zero.set()  # in_flight starts at 0
        self.eviction_complete: asyncio.Event = asyncio.Event()
        self._clock = clock
        self._eviction_log = eviction_log

    def increment_in_flight(self) -> None:
        """Increment in-flight counter for target requests."""
        self.in_flight += 1
        self.in_flight_zero.clear()

    def decrement_in_flight(self) -> None:
        """Decrement in-flight counter; set in_flight_zero event when hits 0."""
        self.in_flight = max(0, self.in_flight - 1)
        if self.in_flight == 0:
            self.in_flight_zero.set()

    def transition(
        self, new_state: ServiceStateEnum, details: Optional[str] = None
    ) -> bool:
        """Transition to a new state with validation."""
        if self.state == new_state:
            # Unchanged.
            return False
        # Special case: hot → cold/warm rejected when in_flight > 0
        if (
            self.state == ServiceStateEnum.hot
            and new_state in (ServiceStateEnum.cold, ServiceStateEnum.warm)
            and self.in_flight > 0
        ):
            return False

        valid_targets = _VALID_TRANSITIONS.get(self.state, set())
        if new_state not in valid_targets:
            return False

        old_state = self.state
        self.state = new_state
        self.last_state_change_time = self._clock.time()

        # Side effects on transition
        if old_state == ServiceStateEnum.evicting:
            # Eviction completed or stopped.
            self.eviction_complete.set()
        if new_state == ServiceStateEnum.evicting:
            # Eviction starting.
            self.eviction_complete.clear()

        # Log to EvictionLog if available
        if self._eviction_log is not None:
            msg = f"{old_state.value} → {new_state.value}"
            if details:
                msg = f"{msg} ({details})"
            self._eviction_log.append(
                EvictionEventType.state_transition, self.name, msg
            )

        _logger.info(
            "State transition for %s: %s → %s%s",
            self.name,
            old_state.value,
            new_state.value,
            f" ({details})" if details else "",
        )
        return True
