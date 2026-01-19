"""
Hive Mind - Event System

Provides event-based communication for topology transitions and agent coordination.
Enables graceful transitions between topologies with proper notification.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Coroutine
from uuid import uuid4

logger = logging.getLogger("titan.hive.events")


class EventType(str, Enum):
    """Types of events in the Hive Mind."""

    # Topology events
    TOPOLOGY_CHANGING = "topology.changing"      # Before switch starts
    TOPOLOGY_CHANGED = "topology.changed"        # After switch completes
    TOPOLOGY_MIGRATION_START = "topology.migration.start"
    TOPOLOGY_MIGRATION_COMPLETE = "topology.migration.complete"

    # Agent events
    AGENT_JOINING = "agent.joining"
    AGENT_JOINED = "agent.joined"
    AGENT_LEAVING = "agent.leaving"
    AGENT_LEFT = "agent.left"
    AGENT_MIGRATING = "agent.migrating"
    AGENT_MIGRATED = "agent.migrated"

    # Task events
    TASK_ASSIGNED = "task.assigned"
    TASK_STARTED = "task.started"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"

    # Learning events
    EPISODE_RECORDED = "learning.episode.recorded"
    MODEL_UPDATED = "learning.model.updated"


@dataclass
class Event:
    """Represents an event in the system."""

    event_type: EventType
    payload: dict[str, Any]
    source_id: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    event_id: str = field(default_factory=lambda: str(uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize event to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "source_id": self.source_id,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Event:
        """Deserialize event from dictionary."""
        return cls(
            event_id=data["event_id"],
            event_type=EventType(data["event_type"]),
            source_id=data.get("source_id"),
            payload=data.get("payload", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
        )


# Type alias for event handlers
EventHandler = Callable[[Event], Coroutine[Any, Any, None]]


class EventBus:
    """
    Central event bus for the Hive Mind.

    Supports:
    - Async event handlers
    - Multiple handlers per event type
    - Event filtering
    - Event history for debugging
    """

    def __init__(self, max_history: int = 1000) -> None:
        self._handlers: dict[EventType, list[EventHandler]] = {}
        self._history: list[Event] = []
        self._max_history = max_history
        self._paused = False

    def subscribe(
        self,
        event_type: EventType,
        handler: EventHandler,
    ) -> Callable[[], None]:
        """
        Subscribe to an event type.

        Args:
            event_type: Type of event to subscribe to
            handler: Async function to call when event occurs

        Returns:
            Unsubscribe function
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []

        self._handlers[event_type].append(handler)
        logger.debug(f"Subscribed handler to {event_type.value}")

        def unsubscribe() -> None:
            if handler in self._handlers.get(event_type, []):
                self._handlers[event_type].remove(handler)
                logger.debug(f"Unsubscribed handler from {event_type.value}")

        return unsubscribe

    async def publish(self, event: Event) -> None:
        """
        Publish an event to all subscribers.

        Args:
            event: Event to publish
        """
        if self._paused:
            logger.debug(f"Event bus paused, skipping {event.event_type.value}")
            return

        # Add to history
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        handlers = self._handlers.get(event.event_type, [])
        if not handlers:
            logger.debug(f"No handlers for {event.event_type.value}")
            return

        logger.debug(f"Publishing {event.event_type.value} to {len(handlers)} handlers")

        # Run handlers concurrently
        tasks = [handler(event) for handler in handlers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Log any errors
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Handler error for {event.event_type.value}: {result}")

    async def emit(
        self,
        event_type: EventType,
        payload: dict[str, Any],
        source_id: str | None = None,
        **metadata: Any,
    ) -> Event:
        """
        Convenience method to create and publish an event.

        Args:
            event_type: Type of event
            payload: Event data
            source_id: ID of the event source
            **metadata: Additional metadata

        Returns:
            The created event
        """
        event = Event(
            event_type=event_type,
            payload=payload,
            source_id=source_id,
            metadata=metadata,
        )
        await self.publish(event)
        return event

    def pause(self) -> None:
        """Pause event processing."""
        self._paused = True
        logger.info("Event bus paused")

    def resume(self) -> None:
        """Resume event processing."""
        self._paused = False
        logger.info("Event bus resumed")

    def get_history(
        self,
        event_type: EventType | None = None,
        limit: int = 100,
    ) -> list[Event]:
        """
        Get event history.

        Args:
            event_type: Filter by event type (optional)
            limit: Maximum events to return

        Returns:
            List of events (most recent last)
        """
        events = self._history
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return events[-limit:]

    def clear_history(self) -> None:
        """Clear event history."""
        self._history.clear()
        logger.debug("Event history cleared")


# Singleton event bus
_default_event_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """Get the default event bus."""
    global _default_event_bus
    if _default_event_bus is None:
        _default_event_bus = EventBus()
    return _default_event_bus


# Topology-specific event helpers
async def emit_topology_changing(
    event_bus: EventBus,
    old_type: str,
    new_type: str,
    agent_count: int,
) -> Event:
    """Emit event before topology switch."""
    return await event_bus.emit(
        EventType.TOPOLOGY_CHANGING,
        {
            "old_type": old_type,
            "new_type": new_type,
            "agent_count": agent_count,
        },
        source_id="topology_engine",
    )


async def emit_topology_changed(
    event_bus: EventBus,
    old_type: str,
    new_type: str,
    agent_count: int,
    duration_ms: float,
) -> Event:
    """Emit event after topology switch."""
    return await event_bus.emit(
        EventType.TOPOLOGY_CHANGED,
        {
            "old_type": old_type,
            "new_type": new_type,
            "agent_count": agent_count,
            "duration_ms": duration_ms,
        },
        source_id="topology_engine",
    )


async def emit_agent_migrating(
    event_bus: EventBus,
    agent_id: str,
    from_topology: str,
    to_topology: str,
) -> Event:
    """Emit event when agent starts migration."""
    return await event_bus.emit(
        EventType.AGENT_MIGRATING,
        {
            "agent_id": agent_id,
            "from_topology": from_topology,
            "to_topology": to_topology,
        },
        source_id=agent_id,
    )


async def emit_agent_migrated(
    event_bus: EventBus,
    agent_id: str,
    new_topology: str,
    new_role: str | None,
) -> Event:
    """Emit event when agent completes migration."""
    return await event_bus.emit(
        EventType.AGENT_MIGRATED,
        {
            "agent_id": agent_id,
            "new_topology": new_topology,
            "new_role": new_role,
        },
        source_id=agent_id,
    )
