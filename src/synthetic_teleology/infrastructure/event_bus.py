"""Event bus infrastructure for the Synthetic Teleology framework.

Provides synchronous and asynchronous pub-sub event buses, plus an in-memory
event store for replay and debugging.  All buses accept ``DomainEvent``
instances and dispatch them to registered handlers, catching and logging errors
so that a single failing subscriber never breaks the publish pipeline.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

from synthetic_teleology.domain.events import DomainEvent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
SyncHandler = Callable[[DomainEvent], None]
AsyncHandler = Callable[[DomainEvent], Any]  # may be sync or async callable


# ===================================================================== #
#  Synchronous Event Bus                                                 #
# ===================================================================== #

class EventBus:
    """Thread-safe synchronous pub-sub for domain events.

    Handlers are invoked **in registration order**.  A handler that raises is
    logged and skipped; subsequent handlers still execute.

    Usage::

        bus = EventBus()
        bus.subscribe(GoalCreated, my_handler)
        bus.publish(GoalCreated(...))
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._handlers: dict[type[DomainEvent], list[SyncHandler]] = defaultdict(list)
        self._global_handlers: list[SyncHandler] = []

    # -- subscription -------------------------------------------------------

    def subscribe(
        self,
        event_type: type[DomainEvent],
        handler: SyncHandler,
    ) -> None:
        """Register *handler* for a specific *event_type*."""
        with self._lock:
            self._handlers[event_type].append(handler)

    def subscribe_all(self, handler: SyncHandler) -> None:
        """Register *handler* to receive **every** published event."""
        with self._lock:
            self._global_handlers.append(handler)

    def unsubscribe(
        self,
        event_type: type[DomainEvent],
        handler: SyncHandler,
    ) -> bool:
        """Remove *handler* from *event_type*. Returns ``True`` if found."""
        with self._lock:
            handlers = self._handlers.get(event_type, [])
            try:
                handlers.remove(handler)
                return True
            except ValueError:
                return False

    def unsubscribe_all(self, handler: SyncHandler) -> bool:
        """Remove a global handler. Returns ``True`` if found."""
        with self._lock:
            try:
                self._global_handlers.remove(handler)
                return True
            except ValueError:
                return False

    # -- publishing ---------------------------------------------------------

    def publish(self, event: DomainEvent) -> None:
        """Publish *event* to all matching handlers (global first, then typed)."""
        with self._lock:
            global_snapshot = list(self._global_handlers)
            typed_snapshot = list(self._handlers.get(type(event), []))

        for handler in global_snapshot:
            try:
                handler(event)
            except Exception:
                logger.exception("Error in global event handler %r", handler)

        for handler in typed_snapshot:
            try:
                handler(event)
            except Exception:
                logger.exception(
                    "Error in handler %r for %s", handler, type(event).__name__
                )

    def publish_many(self, events: Sequence[DomainEvent]) -> None:
        """Publish a batch of events in order."""
        for event in events:
            self.publish(event)

    # -- introspection / lifecycle ------------------------------------------

    def handler_count(self, event_type: type[DomainEvent] | None = None) -> int:
        """Return the number of handlers registered.

        If *event_type* is ``None``, returns the total across all types
        plus globals.
        """
        with self._lock:
            if event_type is not None:
                return len(self._handlers.get(event_type, []))
            total = sum(len(hs) for hs in self._handlers.values())
            return total + len(self._global_handlers)

    def clear(self) -> None:
        """Remove all registered handlers."""
        with self._lock:
            self._handlers.clear()
            self._global_handlers.clear()


# ===================================================================== #
#  Asynchronous Event Bus                                                #
# ===================================================================== #

class AsyncEventBus:
    """Async-compatible event bus for LLM-based agents.

    Handlers may be either regular callables or async coroutines; the bus
    inspects each handler at dispatch time and ``await``s coroutines
    transparently.

    Usage::

        bus = AsyncEventBus()
        bus.subscribe(GoalRevised, my_async_handler)
        await bus.publish(GoalRevised(...))
    """

    def __init__(self) -> None:
        self._handlers: dict[type[DomainEvent], list[AsyncHandler]] = defaultdict(list)
        self._global_handlers: list[AsyncHandler] = []

    # -- subscription -------------------------------------------------------

    def subscribe(
        self,
        event_type: type[DomainEvent],
        handler: AsyncHandler,
    ) -> None:
        """Register *handler* (sync or async) for *event_type*."""
        self._handlers[event_type].append(handler)

    def subscribe_all(self, handler: AsyncHandler) -> None:
        """Register *handler* (sync or async) for all event types."""
        self._global_handlers.append(handler)

    def unsubscribe(
        self,
        event_type: type[DomainEvent],
        handler: AsyncHandler,
    ) -> bool:
        """Remove *handler* from *event_type*. Returns ``True`` if found."""
        handlers = self._handlers.get(event_type, [])
        try:
            handlers.remove(handler)
            return True
        except ValueError:
            return False

    def unsubscribe_all(self, handler: AsyncHandler) -> bool:
        """Remove a global handler. Returns ``True`` if found."""
        try:
            self._global_handlers.remove(handler)
            return True
        except ValueError:
            return False

    # -- publishing ---------------------------------------------------------

    async def publish(self, event: DomainEvent) -> None:
        """Publish *event* to all matching handlers (global first)."""
        for handler in list(self._global_handlers):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception:
                logger.exception("Error in async global event handler %r", handler)

        for handler in list(self._handlers.get(type(event), [])):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception:
                logger.exception(
                    "Error in async handler %r for %s",
                    handler,
                    type(event).__name__,
                )

    async def publish_many(self, events: Sequence[DomainEvent]) -> None:
        """Publish a batch of events in order."""
        for event in events:
            await self.publish(event)

    # -- introspection / lifecycle ------------------------------------------

    def handler_count(self, event_type: type[DomainEvent] | None = None) -> int:
        """Return the number of handlers registered."""
        if event_type is not None:
            return len(self._handlers.get(event_type, []))
        total = sum(len(hs) for hs in self._handlers.values())
        return total + len(self._global_handlers)

    def clear(self) -> None:
        """Remove all registered handlers."""
        self._handlers.clear()
        self._global_handlers.clear()


# ===================================================================== #
#  Event Store                                                           #
# ===================================================================== #

class EventStore:
    """In-memory append-only event store for replay and debugging.

    Can be wired to an ``EventBus`` via ``subscribe_all`` so that every
    published event is automatically persisted::

        store = EventStore()
        bus = EventBus()
        bus.subscribe_all(store.append)
    """

    def __init__(self, max_size: int = 0) -> None:
        """Create a store.

        Parameters
        ----------
        max_size:
            Maximum number of events to keep.  ``0`` means unlimited.
        """
        self._events: list[DomainEvent] = []
        self._max_size = max_size
        self._lock = threading.Lock()

    def append(self, event: DomainEvent) -> None:
        """Append a single event.  Evicts the oldest event if *max_size* is
        exceeded."""
        with self._lock:
            self._events.append(event)
            if self._max_size > 0 and len(self._events) > self._max_size:
                self._events = self._events[-self._max_size:]

    def query(
        self,
        event_type: type[DomainEvent] | None = None,
        since: float | None = None,
        limit: int = 0,
    ) -> Sequence[DomainEvent]:
        """Return events matching the optional filters.

        Parameters
        ----------
        event_type:
            If given, only return events that are instances of this type.
        since:
            If given, only return events whose ``timestamp >= since``.
        limit:
            Maximum number of events to return (0 = unlimited).
        """
        with self._lock:
            result: list[DomainEvent] = list(self._events)

        if event_type is not None:
            result = [e for e in result if isinstance(e, event_type)]
        if since is not None:
            result = [e for e in result if e.timestamp >= since]
        if limit > 0:
            result = result[-limit:]
        return result

    @property
    def latest(self) -> DomainEvent | None:
        """Return the most recently appended event, or ``None``."""
        with self._lock:
            return self._events[-1] if self._events else None

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)

    def __bool__(self) -> bool:
        return len(self) > 0

    def clear(self) -> None:
        """Remove all stored events."""
        with self._lock:
            self._events.clear()
