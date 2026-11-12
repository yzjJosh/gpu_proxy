"""Tests for eviction context, eviction log, and routing stats."""

from datetime import datetime, timezone

from app.eviction import EvictionEventType, EvictionLog

# Bound small enough that a test can overrun it and see the oldest entries drop.
_SMALL_LOG_BOUND = 5


def _append_repeatedly(
    log: EvictionLog,
    event_type: EvictionEventType,
    service_name: str | None,
    times: int,
) -> None:
    """Append the same event the given number of times."""
    for _ in range(times):
        log.append(event_type, service_name)


def _append_for_numbered_services(
    log: EvictionLog, event_type: EvictionEventType, count: int
) -> None:
    """Append one event per service, each named svc0, svc1, and so on."""
    for i in range(count):
        log.append(event_type, f"svc{i}")


def _append_every_event_type(log: EvictionLog, service_name: str) -> None:
    """Append one entry of every event type the log defines."""
    for event_type in EvictionEventType:
        log.append(event_type, service_name)


class TestEvictionLog:
    """Tests for the bounded eviction event log."""

    def test_append_entry(self) -> None:
        """Appending an entry adds it to the log."""
        log = EvictionLog()
        log.append(EvictionEventType.eviction_started, "svc1", "evicted by svc2")
        assert len(log.entries) == 1
        assert log.entries[0].event_type == EvictionEventType.eviction_started
        assert log.entries[0].service_name == "svc1"
        assert log.entries[0].details == "evicted by svc2"

    def test_append_sets_timestamp(self) -> None:
        """An entry is stamped with the UTC moment it was appended."""
        log = EvictionLog()
        before = datetime.now(timezone.utc)
        log.append(EvictionEventType.request_approved, "svc1")
        after = datetime.now(timezone.utc)
        assert before <= log.entries[0].timestamp <= after
        assert log.entries[0].timestamp.tzinfo == timezone.utc

    def test_bounded_deque(self) -> None:
        """Log is bounded; oldest entries auto-dropped at max_entries."""
        log = EvictionLog(max_entries=_SMALL_LOG_BOUND)
        _append_for_numbered_services(
            log, EvictionEventType.eviction_started, _SMALL_LOG_BOUND * 2
        )
        assert len(log.entries) == _SMALL_LOG_BOUND
        # Should keep the newest 5
        assert log.entries[0].service_name == "svc5"

    def test_all_event_types(self) -> None:
        """All defined event types can be logged."""
        log = EvictionLog()
        _append_every_event_type(log, "svc1")
        assert len(log.entries) == len(EvictionEventType)

    def test_noisy_events_not_in_quiet_buffer(self) -> None:
        """Noisy events go to entries but not quiet_entries."""
        log = EvictionLog()
        log.append(EvictionEventType.request_approved, "svc1")
        log.append(EvictionEventType.state_transition, "svc1", "cooling_down → hot")
        log.append(EvictionEventType.state_transition, "svc1", "hot → cooling_down")
        log.append(EvictionEventType.state_transition, "svc1", "cold → warm (VRAM detected)")
        assert len(log.entries) == 4
        assert len(log.quiet_entries) == 1

    def test_significant_events_in_quiet_buffer(self) -> None:
        """Significant events go to both entries and quiet_entries."""
        log = EvictionLog()
        log.append(EvictionEventType.eviction_started, "svc1")
        log.append(EvictionEventType.request_rejected, "svc1")
        log.append(EvictionEventType.target_unreachable, "svc1")
        assert len(log.entries) == 3
        assert len(log.quiet_entries) == 3

    def test_mixed_events_separate_buffers(self) -> None:
        """Mixed noisy and significant events are split correctly."""
        log = EvictionLog()
        log.append(EvictionEventType.request_approved, "svc1")
        log.append(EvictionEventType.eviction_started, "svc1")
        log.append(EvictionEventType.state_transition, "svc1", "cooling_down → hot")
        log.append(EvictionEventType.backup_unreachable, "svc1")
        assert len(log.entries) == 4
        assert len(log.quiet_entries) == 2

    def test_warm_to_hot_is_noisy(self) -> None:
        """warm → hot is a noisy event and excluded from the quiet buffer."""
        log = EvictionLog()
        log.append(EvictionEventType.state_transition, "svc1", "warm → hot")
        assert len(log.entries) == 1
        assert len(log.quiet_entries) == 0

    def test_quiet_buffer_is_bounded(self) -> None:
        """quiet_entries is also bounded by max_entries."""
        log = EvictionLog(max_entries=_SMALL_LOG_BOUND)
        _append_repeatedly(
            log, EvictionEventType.eviction_started, "svc1", _SMALL_LOG_BOUND * 2
        )
        assert len(log.quiet_entries) == _SMALL_LOG_BOUND

    def test_state_transition_non_noisy_in_quiet(self) -> None:
        """Non-noisy state transitions go to quiet buffer."""
        log = EvictionLog()
        log.append(EvictionEventType.state_transition, "svc1", "cold → warm (VRAM detected)")
        log.append(EvictionEventType.state_transition, "svc1", "cooling_down → cold (VRAM dropped to 0)")
        assert len(log.quiet_entries) == 2

    def test_repeated_target_unreachable_deduplicated_in_quiet(self) -> None:
        """Repeated target_unreachable for same service only records first in quiet buffer."""
        log = EvictionLog()
        _append_repeatedly(log, EvictionEventType.target_unreachable, "svc1", 5)
        assert len(log.entries) == 5
        assert len(log.quiet_entries) == 1

    def test_repeated_backup_unreachable_deduplicated_in_quiet(self) -> None:
        """Repeated backup_unreachable for same service only records first in quiet buffer."""
        log = EvictionLog()
        _append_repeatedly(log, EvictionEventType.backup_unreachable, "svc1", 5)
        assert len(log.entries) == 5
        assert len(log.quiet_entries) == 1

    def test_unreachable_state_change_recorded_in_quiet(self) -> None:
        """Transition from target_unreachable to backup_unreachable is recorded."""
        log = EvictionLog()
        log.append(EvictionEventType.target_unreachable, "svc1")
        log.append(EvictionEventType.target_unreachable, "svc1")
        log.append(EvictionEventType.backup_unreachable, "svc1")
        log.append(EvictionEventType.backup_unreachable, "svc1")
        log.append(EvictionEventType.target_unreachable, "svc1")
        assert len(log.entries) == 5
        assert len(log.quiet_entries) == 3

    def test_other_event_resets_unreachable_state(self) -> None:
        """A non-unreachable event resets the dedup state, allowing next unreachable to record."""
        log = EvictionLog()
        log.append(EvictionEventType.target_unreachable, "svc1")
        log.append(EvictionEventType.target_unreachable, "svc1")
        log.append(EvictionEventType.eviction_started, "svc1", "evicted")
        log.append(EvictionEventType.target_unreachable, "svc1")
        assert len(log.entries) == 4
        assert len(log.quiet_entries) == 3

    def test_different_services_deduplicated_independently(self) -> None:
        """Unreachable dedup is per-service, not global."""
        log = EvictionLog()
        log.append(EvictionEventType.target_unreachable, "svc1")
        log.append(EvictionEventType.target_unreachable, "svc2")
        log.append(EvictionEventType.target_unreachable, "svc1")
        log.append(EvictionEventType.target_unreachable, "svc2")
        assert len(log.entries) == 4
        assert len(log.quiet_entries) == 2

    def test_unreachable_with_none_service_not_deduplicated(self) -> None:
        """Entries with None service_name are never deduplicated."""
        log = EvictionLog()
        _append_repeatedly(log, EvictionEventType.target_unreachable, None, 3)
        assert len(log.quiet_entries) == 3
