"""Tests for service state machine and in-flight tracking."""

import pytest

from app.service_state import ServiceState, ServiceStateEnum
from tests.app.conftest import FakeClock

_SERVICE_NAME = "test-service"
_SHARED_NAME = "shared-name"
_FIRST_SERVICE_NAME = "test-service-1"
_SECOND_SERVICE_NAME = "test-service-2"

_START_TIME = 100.0
_ELAPSED_S = 5.0
_OTHER_START_TIME = 250.0
_OTHER_ELAPSED_S = 7.0

# How to drive a fresh service into each state, using valid transitions only.
_PATH_TO_STATE: dict[ServiceStateEnum, tuple[ServiceStateEnum, ...]] = {
    ServiceStateEnum.cold: (),
    ServiceStateEnum.warm: (ServiceStateEnum.warm,),
    ServiceStateEnum.hot: (ServiceStateEnum.hot,),
    ServiceStateEnum.cooling_down: (ServiceStateEnum.hot, ServiceStateEnum.cooling_down),
    ServiceStateEnum.evicting: (ServiceStateEnum.hot, ServiceStateEnum.evicting),
    ServiceStateEnum.restarting: (
        ServiceStateEnum.hot,
        ServiceStateEnum.evicting,
        ServiceStateEnum.restarting,
    ),
    ServiceStateEnum.unhealthy: (ServiceStateEnum.unhealthy,),
}


def _make_state(name: str = _SERVICE_NAME) -> ServiceState:
    """A fresh service, cold, on a clock the test controls."""
    return ServiceState(name, clock=FakeClock(start_time=_START_TIME))


def _in_state(state: ServiceStateEnum) -> ServiceState:
    """A service driven through valid transitions until it sits in the given state.

    Every step is checked, so a path that stops being valid fails here rather
    than leaving a test quietly asserting against the wrong starting state.
    """
    service = _make_state()
    for step in _PATH_TO_STATE[state]:
        assert service.transition(step) is True
    assert service.state == state
    return service


class TestServiceStateInit:
    """Tests for ServiceState initialization."""

    def test_initial_state_is_cold(self) -> None:
        """New ServiceState starts as cold."""
        state = _make_state()
        assert state.state == ServiceStateEnum.cold

    def test_no_new_hack_in_service_state(self) -> None:
        """Two states for one service name are separate objects, not a shared one.

        Construction must stay ordinary: each service reports back the name it
        was given, and there is no interning of instances by name, in __new__ or
        anywhere else, or one service's transitions would leak into every other
        holder of that name.
        """
        first = _make_state(_SHARED_NAME)
        second = _make_state(_SHARED_NAME)
        first.transition(ServiceStateEnum.hot)
        assert first is not second
        assert second.state == ServiceStateEnum.cold
        assert first.name == _SHARED_NAME
        assert second.name == _SHARED_NAME
        # A differently named service reports its own name, so no fixed string
        # can stand in for the one construction was given.
        assert _make_state().name == _SERVICE_NAME

    def test_initial_in_flight_zero(self) -> None:
        """New ServiceState has in_flight=0."""
        state = _make_state()
        assert state.in_flight == 0

    def test_in_flight_zero_event_set_on_init(self) -> None:
        """in_flight_zero event is set initially (in_flight is already 0)."""
        state = _make_state()
        assert state.in_flight_zero.is_set()

    def test_initial_is_healthy_is_true(self) -> None:
        """New ServiceState has is_healthy=True."""
        state = _make_state()
        assert state.is_healthy is True


class TestInFlightTracking:
    """Tests for in-flight request counting."""

    def test_increment_in_flight(self) -> None:
        """Increment in_flight counter."""
        state = _make_state()
        state.increment_in_flight()
        assert state.in_flight == 1

    def test_increment_multiple(self) -> None:
        """Multiple increments tracked correctly."""
        state = _make_state()
        state.increment_in_flight()
        state.increment_in_flight()
        state.increment_in_flight()
        assert state.in_flight == 3

    def test_decrement_in_flight(self) -> None:
        """Decrement after increment returns to 0."""
        state = _make_state()
        state.increment_in_flight()
        state.decrement_in_flight()
        assert state.in_flight == 0

    def test_decrement_fires_in_flight_zero(self) -> None:
        """in_flight_zero event fires when counter reaches 0."""
        state = _make_state()
        event = state.in_flight_zero
        state.increment_in_flight()
        assert not event.is_set()
        state.decrement_in_flight()
        # A waiter holds this object, so firing must set it rather than swap in
        # a fresh event nobody is waiting on.
        assert state.in_flight_zero is event
        assert event.is_set()

    def test_decrement_partial_does_not_fire(self) -> None:
        """in_flight_zero not set when counter > 0 after decrement."""
        state = _make_state()
        state.increment_in_flight()
        state.increment_in_flight()
        state.decrement_in_flight()
        assert state.in_flight == 1
        assert not state.in_flight_zero.is_set()

    def test_in_flight_never_negative(self) -> None:
        """Decrementing past zero clamps at zero rather than going negative."""
        state = _make_state()
        state.decrement_in_flight()
        assert state.in_flight == 0
        assert state.in_flight_zero.is_set()

    def test_only_target_requests_counted(self) -> None:
        """Only the in-flight calls move the counter; changing state never does."""
        state = _make_state()
        state.increment_in_flight()
        state.transition(ServiceStateEnum.hot)
        state.transition(ServiceStateEnum.evicting)
        state.transition(ServiceStateEnum.restarting)
        assert state.in_flight == 1
        assert not state.in_flight_zero.is_set()


class TestStateTransitions:
    """Tests for all valid and invalid state transitions."""

    def test_cold_to_warm(self) -> None:
        """cold -> warm: VRAM detected but no requests."""
        state = _make_state()
        assert state.transition(ServiceStateEnum.warm) is True
        assert state.state == ServiceStateEnum.warm

    def test_cold_to_hot(self) -> None:
        """cold -> hot: request admitted directly to hot."""
        state = _make_state()
        assert state.transition(ServiceStateEnum.hot) is True
        assert state.state == ServiceStateEnum.hot

    def test_warm_to_hot(self) -> None:
        """warm -> hot: request admitted directly to hot."""
        state = _in_state(ServiceStateEnum.warm)
        assert state.transition(ServiceStateEnum.hot) is True
        assert state.state == ServiceStateEnum.hot

    def test_warm_to_cold(self) -> None:
        """warm -> cold: VRAM drops to 0."""
        state = _in_state(ServiceStateEnum.warm)
        assert state.transition(ServiceStateEnum.cold) is True
        assert state.state == ServiceStateEnum.cold

    def test_warm_to_evicting(self) -> None:
        """warm -> evicting: selected for eviction."""
        state = _in_state(ServiceStateEnum.warm)
        assert state.transition(ServiceStateEnum.evicting) is True
        assert state.state == ServiceStateEnum.evicting

    def test_hot_to_evicting(self) -> None:
        """hot -> evicting: lower-priority active service evicted."""
        state = _in_state(ServiceStateEnum.hot)
        assert state.transition(ServiceStateEnum.evicting) is True
        assert state.state == ServiceStateEnum.evicting

    def test_hot_to_cooling_down_to_cold(self) -> None:
        """hot -> cooling_down -> cold: in_flight hits 0 and VRAM == 0."""
        state = _in_state(ServiceStateEnum.cooling_down)
        assert state.transition(ServiceStateEnum.cold) is True
        assert state.state == ServiceStateEnum.cold

    def test_hot_to_cooling_down_to_warm(self) -> None:
        """hot -> cooling_down -> warm: in_flight hits 0 and VRAM > 0."""
        state = _in_state(ServiceStateEnum.cooling_down)
        assert state.transition(ServiceStateEnum.warm) is True
        assert state.state == ServiceStateEnum.warm

    def test_hot_transitions_to_cooling_down_when_in_flight_zero_then_proceeds(
        self,
    ) -> None:
        """hot → cooling_down when in_flight drops to 0, then cooling_down → cold/warm.

        Draining to zero is the scenario, not the cause: the transition table
        lists cooling_down as a target of hot unconditionally, so the move is
        allowed at any in-flight count. Both exits from cooling_down are
        covered: the GPU may report the VRAM released (cold) or still held
        (warm) by the time the service settles.
        """
        drained = _in_state(ServiceStateEnum.hot)
        drained.increment_in_flight()
        drained.decrement_in_flight()
        assert drained.transition(ServiceStateEnum.cooling_down) is True
        assert drained.transition(ServiceStateEnum.cold) is True
        assert drained.state == ServiceStateEnum.cold

        still_resident = _in_state(ServiceStateEnum.cooling_down)
        assert still_resident.transition(ServiceStateEnum.warm) is True
        assert still_resident.state == ServiceStateEnum.warm

    def test_evicting_to_restarting(self) -> None:
        """evicting -> restarting: the eviction finished and the service comes back."""
        state = _in_state(ServiceStateEnum.evicting)
        assert state.transition(ServiceStateEnum.restarting) is True
        assert state.state == ServiceStateEnum.restarting

    def test_restarting_to_cold(self) -> None:
        """restarting -> cold: container back, model not loaded."""
        state = _in_state(ServiceStateEnum.restarting)
        assert state.transition(ServiceStateEnum.cold) is True
        assert state.state == ServiceStateEnum.cold


class TestInvalidTransitions:
    """Tests for invalid state transitions that return False."""

    @pytest.mark.parametrize("from_state,to_state", [
        # cold → invalid targets
        (ServiceStateEnum.cold, ServiceStateEnum.evicting),
        (ServiceStateEnum.cold, ServiceStateEnum.restarting),
        # warm → invalid targets
        (ServiceStateEnum.warm, ServiceStateEnum.restarting),
        # hot → invalid targets
        (ServiceStateEnum.hot, ServiceStateEnum.restarting),
        (ServiceStateEnum.hot, ServiceStateEnum.warm),
        (ServiceStateEnum.hot, ServiceStateEnum.cold),
        # evicting → invalid targets
        (ServiceStateEnum.evicting, ServiceStateEnum.cold),
        (ServiceStateEnum.evicting, ServiceStateEnum.warm),
        (ServiceStateEnum.evicting, ServiceStateEnum.hot),
        (ServiceStateEnum.evicting, ServiceStateEnum.unhealthy),
        # restarting → invalid targets
        (ServiceStateEnum.restarting, ServiceStateEnum.hot),
        (ServiceStateEnum.restarting, ServiceStateEnum.evicting),
        (ServiceStateEnum.restarting, ServiceStateEnum.restarting),
        # unhealthy → invalid targets
        (ServiceStateEnum.unhealthy, ServiceStateEnum.evicting),
    ], ids=lambda p: f"{p.value}_to_invalid" if isinstance(p, ServiceStateEnum) else str(p))
    def test_invalid_transition_raises_valueerror(
        self, from_state: ServiceStateEnum, to_state: ServiceStateEnum
    ) -> None:
        """Invalid state transitions return False and leave state unchanged."""
        svc = _in_state(from_state)
        result = svc.transition(to_state)
        assert result is False
        assert svc.state == from_state

    @pytest.mark.parametrize("state", [
        ServiceStateEnum.hot,
        ServiceStateEnum.evicting,
        ServiceStateEnum.restarting,
    ], ids=lambda s: f"{s.value}_self_transition")
    def test_self_transition_is_noop(self, state: ServiceStateEnum) -> None:
        """Self-transitions are idempotent (no-op, not an error)."""
        svc = _in_state(state)
        result = svc.transition(state)
        assert result is False
        assert svc.state == state


class TestEvictionCompleteEvent:
    """Tests for eviction_complete event coordination."""

    def test_eviction_complete_event_shared(self) -> None:
        """Waiters keep one event: eviction reuses the object rather than replacing it."""
        state = _make_state()
        event = state.eviction_complete
        assert not event.is_set()
        state.transition(ServiceStateEnum.hot)
        # Only leaving evicting may set the event; an unrelated move must not.
        assert not event.is_set()
        state.transition(ServiceStateEnum.evicting)
        state.transition(ServiceStateEnum.restarting)
        assert state.eviction_complete is event
        assert event.is_set()

    def test_eviction_complete_set_on_evicting_to_restarting(self) -> None:
        """Leaving evicting sets the event; entering evicting clears it again."""
        state = _in_state(ServiceStateEnum.evicting)
        assert not state.eviction_complete.is_set()
        state.transition(ServiceStateEnum.restarting)
        assert state.eviction_complete.is_set()
        # A second eviction must make waiters wait again, not pass straight through.
        assert state.transition(ServiceStateEnum.cold) is True
        assert state.transition(ServiceStateEnum.hot) is True
        assert state.transition(ServiceStateEnum.evicting) is True
        assert not state.eviction_complete.is_set()


class TestClockProtocolTypeHints:
    """Tests for the clock a ServiceState is required to be given."""

    def test_service_state_clock_parameter_type(self) -> None:
        """A service needs a clock, and every timestamp it reports comes from it."""
        with pytest.raises(TypeError, match="clock"):
            ServiceState(_SERVICE_NAME)
        clock = FakeClock(start_time=_START_TIME)
        state = ServiceState(_SERVICE_NAME, clock=clock)
        assert state.last_state_change_time == _START_TIME
        clock.advance(_ELAPSED_S)
        state.transition(ServiceStateEnum.warm)
        assert state.last_state_change_time == _START_TIME + _ELAPSED_S
        # A second service on a clock reading different times: no fixed value
        # satisfies both services, so each timestamp comes from its own clock.
        other_clock = FakeClock(start_time=_OTHER_START_TIME)
        other = ServiceState(_SECOND_SERVICE_NAME, clock=other_clock)
        assert other.last_state_change_time == _OTHER_START_TIME
        other_clock.advance(_OTHER_ELAPSED_S)
        other.transition(ServiceStateEnum.warm)
        assert other.last_state_change_time == _OTHER_START_TIME + _OTHER_ELAPSED_S


class TestConcurrentTransitions:
    """Tests for transitions attempted while a request is still in flight.

    Nothing here runs concurrently: a ServiceState takes no lock and yields at
    no point, so the interleavings a caller worries about reduce to the order
    the calls are made in, which these tests spell out directly.
    """

    async def test_gpu_poll_update_while_request_processing(self) -> None:
        """GPU poll update arriving while a request is being processed.

        Scenario: Service is hot with in_flight > 0. GPU poll shows VRAM == 0.
        The service must stay hot, and what forbids the move is the transition
        table: hot lists only evicting, cooling_down and unhealthy as targets,
        so cold and warm are rejected at any in-flight count. The request in
        flight is the scenario, not the cause — transition() also guards
        hot → cold/warm on in_flight > 0, but the table already rejects every
        case that guard covers, so it can change no outcome. A repeated poll
        changes nothing, and eviction is never blocked this way: a busy service
        must still be evictable for a higher-priority request.
        """
        state = _in_state(ServiceStateEnum.hot)
        # Request in flight
        state.increment_in_flight()
        # GPU poll shows VRAM == 0, but hot has no cold/warm target → stays hot
        assert state.transition(ServiceStateEnum.cold) is False
        assert state.transition(ServiceStateEnum.warm) is False
        assert state.transition(ServiceStateEnum.cold) is False
        assert state.state == ServiceStateEnum.hot
        assert state.in_flight == 1
        # The same in-flight request does not stand in the way of eviction.
        assert state.transition(ServiceStateEnum.evicting) is True
        assert state.state == ServiceStateEnum.evicting

    async def test_eviction_timeout_vs_in_flight_zero_race(self) -> None:
        """A drained and a still-busy service both leave evicting the same way.

        Two services reach evicting, one whose request has finished and one
        whose request is still open, so in_flight_zero is set for the first and
        clear for the second. Restarting is the one target evicting allows,
        whatever the count, so both reach it with eviction_complete set.
        """
        clock = FakeClock(start_time=_START_TIME)
        # Path 1: the request finishes, so in_flight_zero is set
        drained = ServiceState(_FIRST_SERVICE_NAME, clock=clock)
        drained.transition(ServiceStateEnum.hot)
        drained.increment_in_flight()
        drained.transition(ServiceStateEnum.evicting)
        drained.decrement_in_flight()
        assert drained.in_flight_zero.is_set()
        assert drained.transition(ServiceStateEnum.restarting) is True
        assert drained.eviction_complete.is_set()

        # Path 2: the request is still open, so in_flight_zero stays clear
        timed_out = ServiceState(_SECOND_SERVICE_NAME, clock=clock)
        timed_out.transition(ServiceStateEnum.hot)
        timed_out.increment_in_flight()
        timed_out.transition(ServiceStateEnum.evicting)
        assert not timed_out.in_flight_zero.is_set()
        assert timed_out.transition(ServiceStateEnum.restarting) is True
        assert timed_out.eviction_complete.is_set()


class TestEvictionTimeoutPath:
    """Tests that evicting → restarting is permitted at any in-flight count.

    A ServiceState owns no timer. Whatever decides how long to wait for
    in_flight to drain before giving up lives in the caller, so all that can be
    asked here is whether the move out of evicting is allowed and what it sets.
    """

    async def test_eviction_timeout_triggers_restarting(self) -> None:
        """Leaving evicting while still busy completes the eviction anyway.

        The pending request is the scenario, not the cause: the transition
        table lists restarting as the one target of evicting, unconditionally,
        so a service that never drained still leaves evicting, and doing so
        releases the waiters on eviction_complete.
        """
        state = _in_state(ServiceStateEnum.hot)
        state.increment_in_flight()
        state.transition(ServiceStateEnum.evicting)
        # in_flight > 0, so in_flight_zero is not set
        assert not state.in_flight_zero.is_set()
        assert state.transition(ServiceStateEnum.restarting) is True
        assert state.eviction_complete.is_set()
