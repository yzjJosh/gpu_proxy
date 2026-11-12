"""Eviction context tracking and eviction log."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

__all__ = ["EvictionEventType", "EvictionLog"]


class EvictionEventType(Enum):
    """Types of events logged in EvictionLog."""
    eviction_started = "eviction_started"
    eviction_completed = "eviction_completed"
    eviction_timeout = "eviction_timeout"
    eviction_command_completed = "eviction_command_completed"
    eviction_command_failed = "eviction_command_failed"
    gpu_poll_failed = "gpu_poll_failed"
    backup_unreachable = "backup_unreachable"
    backup_failed = "backup_failed"
    target_unreachable = "target_unreachable"
    target_failed = "target_failed"
    become_healthy = "become_healthy"
    become_unhealthy = "become_unhealthy"
    request_approved = "request_approved"
    request_rejected = "request_rejected"
    state_transition = "state_transition"


@dataclass
class _LogEntry:
    """A single entry in the eviction log."""

    timestamp: datetime
    event_type: EvictionEventType
    service_name: Optional[str] = None
    details: str = ""

def _entry_row(entry: _LogEntry) -> dict[str, str | None]:
    """One log entry as a dashboard row."""
    return {
        "timestamp": entry.timestamp.isoformat() if entry.timestamp else None,
        "event": entry.event_type.value,
        "service": entry.service_name,
        "details": entry.details,
    }


class EvictionLog:
    """Bounded log of eviction and routing events.

    Two buffers: entries (all events), quiet_entries (non-noisy only).
    In quiet mode, repeated target_unreachable/backup_unreachable entries
    for the same service are deduplicated — only state changes are recorded.
    """

    def __init__(self, max_entries: int = 1000) -> None:
        self.entries: deque[_LogEntry] = deque(maxlen=max_entries)
        self.quiet_entries: deque[_LogEntry] = deque(maxlen=max_entries)
        self._last_unreachable: dict[str, Optional[str]] = {}

    @staticmethod
    def _is_noisy(event_type: EvictionEventType, details: str) -> bool:
        if event_type == EvictionEventType.request_approved:
            return True
        if event_type == EvictionEventType.state_transition:
            if 'hot → cooling_down' in details:
                return True
            if 'cooling_down → hot' in details:
                return True
            if 'cooling_down → warm' in details:
                return True
            if 'warm → hot' in details:
                return True
        return False

    def _is_duplicate_unreachable(self, event_type: EvictionEventType, service_name: Optional[str]) -> bool:
        """Check if this unreachable event is a duplicate for the same service.

        Tracks per-service unreachable state. A new entry is recorded only
        when the state changes (e.g., target_unreachable → backup_unreachable,
        or unreachable → any other event type).
        """
        if service_name is None:
            return False
        if event_type not in (EvictionEventType.target_unreachable, EvictionEventType.backup_unreachable):
            self._last_unreachable.pop(service_name, None)
            return False
        last = self._last_unreachable.get(service_name)
        if last == event_type.value:
            return True
        self._last_unreachable[service_name] = event_type.value
        return False

    def append(self, event_type: EvictionEventType, service_name: Optional[str] = None, details: str = "") -> None:
        """Append a new log entry."""
        entry = _LogEntry(
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            service_name=service_name,
            details=details,
        )
        self.entries.append(entry)
        if not self._is_noisy(event_type, details):
            if not self._is_duplicate_unreachable(event_type, service_name):
                self.quiet_entries.append(entry)

    def rows(self) -> list[dict[str, str | None]]:
        """All entries as dashboard rows, oldest first."""
        return [_entry_row(e) for e in self.entries]

    def quiet_rows(self) -> list[dict[str, str | None]]:
        """Non-noisy entries as dashboard rows, oldest first."""
        return [_entry_row(e) for e in self.quiet_entries]


