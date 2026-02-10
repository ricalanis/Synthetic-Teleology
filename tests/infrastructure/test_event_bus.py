"""Tests for EventBus, AsyncEventBus, and EventStore."""

from __future__ import annotations

import asyncio
import time

import pytest

from synthetic_teleology.domain.events import (
    DomainEvent,
    GoalCreated,
    GoalRevised,
    EvaluationCompleted,
)
from synthetic_teleology.domain.values import EvalSignal, GoalRevision
from synthetic_teleology.infrastructure.event_bus import (
    AsyncEventBus,
    EventBus,
    EventStore,
)


class TestEventBus:
    """Test synchronous EventBus subscribe, publish, unsubscribe."""

    def test_subscribe_and_publish(self) -> None:
        bus = EventBus()
        received: list[DomainEvent] = []
        bus.subscribe(GoalCreated, lambda e: received.append(e))

        event = GoalCreated(
            source_id="test",
            goal_id="g1",
            goal_name="goal-1",
            timestamp=time.time(),
        )
        bus.publish(event)
        assert len(received) == 1
        assert received[0] is event

    def test_typed_subscription_filters_events(self) -> None:
        bus = EventBus()
        goal_events: list[DomainEvent] = []
        bus.subscribe(GoalCreated, lambda e: goal_events.append(e))

        # Publish a GoalCreated (should be received)
        bus.publish(GoalCreated(
            source_id="test",
            goal_id="g1",
            goal_name="goal-1",
            timestamp=time.time(),
        ))

        # Publish an EvaluationCompleted (should NOT be received)
        bus.publish(EvaluationCompleted(
            source_id="test",
            goal_id="g1",
            eval_signal=EvalSignal(score=0.5),
            revision_triggered=False,
            timestamp=time.time(),
        ))

        assert len(goal_events) == 1

    def test_subscribe_all(self) -> None:
        bus = EventBus()
        all_events: list[DomainEvent] = []
        bus.subscribe_all(lambda e: all_events.append(e))

        bus.publish(GoalCreated(
            source_id="test",
            goal_id="g1",
            goal_name="goal-1",
            timestamp=time.time(),
        ))
        bus.publish(EvaluationCompleted(
            source_id="test",
            goal_id="g1",
            eval_signal=EvalSignal(score=0.5),
            revision_triggered=False,
            timestamp=time.time(),
        ))

        assert len(all_events) == 2

    def test_unsubscribe(self) -> None:
        bus = EventBus()
        received: list[DomainEvent] = []
        handler = lambda e: received.append(e)
        bus.subscribe(GoalCreated, handler)

        assert bus.unsubscribe(GoalCreated, handler) is True

        bus.publish(GoalCreated(
            source_id="test",
            goal_id="g1",
            goal_name="goal-1",
            timestamp=time.time(),
        ))
        assert len(received) == 0

    def test_unsubscribe_not_found(self) -> None:
        bus = EventBus()
        handler = lambda e: None
        assert bus.unsubscribe(GoalCreated, handler) is False

    def test_unsubscribe_all(self) -> None:
        bus = EventBus()
        handler = lambda e: None
        bus.subscribe_all(handler)
        assert bus.unsubscribe_all(handler) is True
        assert bus.unsubscribe_all(handler) is False

    def test_handler_error_does_not_break_others(self) -> None:
        bus = EventBus()
        received: list[DomainEvent] = []

        def bad_handler(e: DomainEvent) -> None:
            raise RuntimeError("handler failed")

        bus.subscribe(GoalCreated, bad_handler)
        bus.subscribe(GoalCreated, lambda e: received.append(e))

        bus.publish(GoalCreated(
            source_id="test",
            goal_id="g1",
            goal_name="goal-1",
            timestamp=time.time(),
        ))

        # Second handler should still receive the event
        assert len(received) == 1

    def test_handler_count(self) -> None:
        bus = EventBus()
        bus.subscribe(GoalCreated, lambda e: None)
        bus.subscribe(GoalCreated, lambda e: None)
        bus.subscribe_all(lambda e: None)

        assert bus.handler_count(GoalCreated) == 2
        assert bus.handler_count() == 3  # 2 typed + 1 global

    def test_clear(self) -> None:
        bus = EventBus()
        bus.subscribe(GoalCreated, lambda e: None)
        bus.subscribe_all(lambda e: None)
        bus.clear()
        assert bus.handler_count() == 0

    def test_publish_many(self) -> None:
        bus = EventBus()
        received: list[DomainEvent] = []
        bus.subscribe_all(lambda e: received.append(e))

        events = [
            GoalCreated(source_id="test", goal_id=f"g{i}", goal_name=f"goal-{i}", timestamp=time.time())
            for i in range(3)
        ]
        bus.publish_many(events)
        assert len(received) == 3

    def test_global_handlers_called_before_typed(self) -> None:
        bus = EventBus()
        order: list[str] = []

        bus.subscribe_all(lambda e: order.append("global"))
        bus.subscribe(GoalCreated, lambda e: order.append("typed"))

        bus.publish(GoalCreated(
            source_id="test",
            goal_id="g1",
            goal_name="goal-1",
            timestamp=time.time(),
        ))
        assert order == ["global", "typed"]


class TestAsyncEventBus:
    """Test async event bus."""

    @pytest.mark.asyncio
    async def test_async_subscribe_and_publish(self) -> None:
        bus = AsyncEventBus()
        received: list[DomainEvent] = []
        bus.subscribe(GoalCreated, lambda e: received.append(e))

        event = GoalCreated(
            source_id="test",
            goal_id="g1",
            goal_name="goal-1",
            timestamp=time.time(),
        )
        await bus.publish(event)
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_async_handler(self) -> None:
        bus = AsyncEventBus()
        received: list[DomainEvent] = []

        async def async_handler(e: DomainEvent) -> None:
            received.append(e)

        bus.subscribe(GoalCreated, async_handler)
        await bus.publish(GoalCreated(
            source_id="test",
            goal_id="g1",
            goal_name="goal-1",
            timestamp=time.time(),
        ))
        assert len(received) == 1

    def test_async_handler_count(self) -> None:
        bus = AsyncEventBus()
        bus.subscribe(GoalCreated, lambda e: None)
        bus.subscribe_all(lambda e: None)
        assert bus.handler_count(GoalCreated) == 1
        assert bus.handler_count() == 2

    def test_async_clear(self) -> None:
        bus = AsyncEventBus()
        bus.subscribe(GoalCreated, lambda e: None)
        bus.clear()
        assert bus.handler_count() == 0


class TestEventStore:
    """Test EventStore append, query, max_size."""

    def test_append_and_len(self) -> None:
        store = EventStore()
        event = GoalCreated(
            source_id="test",
            goal_id="g1",
            goal_name="goal-1",
            timestamp=time.time(),
        )
        store.append(event)
        assert len(store) == 1
        assert store.latest is event

    def test_query_all(self) -> None:
        store = EventStore()
        for i in range(5):
            store.append(GoalCreated(
                source_id="test",
                goal_id=f"g{i}",
                goal_name=f"goal-{i}",
                timestamp=float(i),
            ))
        result = store.query()
        assert len(result) == 5

    def test_query_by_type(self) -> None:
        store = EventStore()
        store.append(GoalCreated(
            source_id="test",
            goal_id="g1",
            goal_name="goal-1",
            timestamp=time.time(),
        ))
        store.append(EvaluationCompleted(
            source_id="test",
            goal_id="g1",
            eval_signal=EvalSignal(score=0.5),
            revision_triggered=False,
            timestamp=time.time(),
        ))

        goal_events = store.query(event_type=GoalCreated)
        assert len(goal_events) == 1

    def test_query_since_timestamp(self) -> None:
        store = EventStore()
        for i in range(5):
            store.append(GoalCreated(
                source_id="test",
                goal_id=f"g{i}",
                goal_name=f"goal-{i}",
                timestamp=float(i),
            ))
        result = store.query(since=3.0)
        assert len(result) == 2  # timestamps 3.0 and 4.0

    def test_query_with_limit(self) -> None:
        store = EventStore()
        for i in range(10):
            store.append(GoalCreated(
                source_id="test",
                goal_id=f"g{i}",
                goal_name=f"goal-{i}",
                timestamp=float(i),
            ))
        result = store.query(limit=3)
        assert len(result) == 3

    def test_max_size_eviction(self) -> None:
        store = EventStore(max_size=3)
        for i in range(5):
            store.append(GoalCreated(
                source_id="test",
                goal_id=f"g{i}",
                goal_name=f"goal-{i}",
                timestamp=float(i),
            ))
        assert len(store) == 3
        # Should keep the last 3
        result = store.query()
        assert len(result) == 3

    def test_empty_store(self) -> None:
        store = EventStore()
        assert len(store) == 0
        assert store.latest is None
        assert not store

    def test_bool(self) -> None:
        store = EventStore()
        assert not store
        store.append(GoalCreated(
            source_id="test",
            goal_id="g1",
            goal_name="goal-1",
            timestamp=time.time(),
        ))
        assert store

    def test_clear(self) -> None:
        store = EventStore()
        store.append(GoalCreated(
            source_id="test",
            goal_id="g1",
            goal_name="goal-1",
            timestamp=time.time(),
        ))
        store.clear()
        assert len(store) == 0

    def test_wired_to_event_bus(self) -> None:
        store = EventStore()
        bus = EventBus()
        bus.subscribe_all(store.append)

        bus.publish(GoalCreated(
            source_id="test",
            goal_id="g1",
            goal_name="goal-1",
            timestamp=time.time(),
        ))
        assert len(store) == 1
