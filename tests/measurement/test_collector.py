"""Tests for EventCollector and AgentLog."""

from __future__ import annotations

import time

import pytest

from synthetic_teleology.domain.events import (
    ActionExecuted,
    ConstraintViolated,
    EvaluationCompleted,
    GoalRevised,
    LoopStepCompleted,
    ReflectionTriggered,
)
from synthetic_teleology.domain.values import ActionSpec, EvalSignal, GoalRevision
from synthetic_teleology.infrastructure.event_bus import EventBus
from synthetic_teleology.measurement.collector import (
    AgentLog,
    AgentLogEntry,
    EventCollector,
)


def _make_collector() -> tuple[EventCollector, EventBus]:
    """Create a collector wired to a fresh event bus."""
    bus = EventBus()
    collector = EventCollector(bus)
    return collector, bus


class TestAgentLogEntry:
    """Test AgentLogEntry data structure."""

    def test_creation(self) -> None:
        entry = AgentLogEntry(step=1, timestamp=time.time())
        assert entry.step == 1
        assert entry.eval_score == 0.0
        assert entry.goal_revised is False

    def test_fields_mutable(self) -> None:
        entry = AgentLogEntry(step=0, timestamp=0.0)
        entry.eval_score = 0.75
        entry.action_name = "move"
        assert entry.eval_score == 0.75
        assert entry.action_name == "move"


class TestAgentLog:
    """Test AgentLog derived properties."""

    def test_empty_log(self) -> None:
        log = AgentLog(agent_id="a1")
        assert log.num_steps == 0
        assert log.revision_count == 0
        assert log.get_scores() == []
        assert log.get_costs() == []
        assert log.get_action_names() == []

    def test_with_entries(self) -> None:
        log = AgentLog(agent_id="a1")
        log.entries.append(AgentLogEntry(step=1, timestamp=1.0, eval_score=0.5, action_name="a", action_cost=1.0))
        log.entries.append(AgentLogEntry(step=2, timestamp=2.0, eval_score=0.8, action_name="b", action_cost=2.0))
        assert log.num_steps == 2
        assert log.get_scores() == [0.5, 0.8]
        assert log.get_costs() == [1.0, 2.0]
        assert log.get_action_names() == ["a", "b"]

    def test_steps_where(self) -> None:
        log = AgentLog(agent_id="a1")
        log.entries.append(AgentLogEntry(step=0, timestamp=0.0, goal_revised=True))
        log.entries.append(AgentLogEntry(step=1, timestamp=1.0, goal_revised=False))
        log.entries.append(AgentLogEntry(step=2, timestamp=2.0, goal_revised=True))
        assert log.steps_where("goal_revised") == [0, 2]

    def test_steps_where_constraint_violated(self) -> None:
        log = AgentLog(agent_id="a1")
        log.entries.append(AgentLogEntry(step=0, timestamp=0.0, constraint_violated=True))
        log.entries.append(AgentLogEntry(step=1, timestamp=1.0, constraint_violated=False))
        assert log.steps_where("constraint_violated") == [0]

    def test_revision_count(self) -> None:
        log = AgentLog(agent_id="a1")
        log.goal_revisions.append(
            GoalRevised(
                source_id="a1",
                revision=GoalRevision(
                    previous_goal_id="g1",
                    new_goal_id="g2",
                    reason="update",
                ),
                previous_objective=None,
                new_objective=None,
                timestamp=time.time(),
            )
        )
        assert log.revision_count == 1


class TestEventCollector:
    """Test EventCollector subscribes to events and builds AgentLog."""

    def test_creates_log_on_step_event(self) -> None:
        collector, bus = _make_collector()
        bus.publish(LoopStepCompleted(
            source_id="agent-1",
            step_number=1,
            agent_state=None,
            elapsed_ms=10.0,
            goal_id="g1",
            timestamp=time.time(),
        ))
        log = collector.get_log("agent-1")
        assert log is not None
        assert log.num_steps == 1
        assert log.entries[0].step == 1
        assert log.entries[0].goal_id == "g1"

    def test_evaluation_captured(self) -> None:
        collector, bus = _make_collector()
        signal = EvalSignal(score=0.7, confidence=0.9)
        bus.publish(EvaluationCompleted(
            source_id="agent-1",
            goal_id="g1",
            eval_signal=signal,
            revision_triggered=False,
            timestamp=time.time(),
        ))
        bus.publish(LoopStepCompleted(
            source_id="agent-1",
            step_number=1,
            agent_state=None,
            elapsed_ms=5.0,
            goal_id="g1",
            timestamp=time.time(),
        ))
        log = collector.get_log("agent-1")
        assert log is not None
        assert log.entries[0].eval_score == pytest.approx(0.7)
        assert log.entries[0].eval_confidence == pytest.approx(0.9)

    def test_action_captured(self) -> None:
        collector, bus = _make_collector()
        action = ActionSpec(name="move_right", cost=1.5)
        bus.publish(ActionExecuted(
            source_id="agent-1",
            action=action,
            timestamp=time.time(),
        ))
        bus.publish(LoopStepCompleted(
            source_id="agent-1",
            step_number=1,
            agent_state=None,
            elapsed_ms=5.0,
            goal_id="g1",
            timestamp=time.time(),
        ))
        log = collector.get_log("agent-1")
        assert log is not None
        assert log.entries[0].action_name == "move_right"
        assert log.entries[0].action_cost == pytest.approx(1.5)

    def test_goal_revision_captured(self) -> None:
        collector, bus = _make_collector()
        revision = GoalRevision(
            previous_goal_id="g1",
            new_goal_id="g2",
            reason="teleological_update",
        )
        bus.publish(GoalRevised(
            source_id="agent-1",
            revision=revision,
            previous_objective=None,
            new_objective=None,
            timestamp=time.time(),
        ))
        bus.publish(LoopStepCompleted(
            source_id="agent-1",
            step_number=1,
            agent_state=None,
            elapsed_ms=5.0,
            goal_id="g2",
            timestamp=time.time(),
        ))
        log = collector.get_log("agent-1")
        assert log is not None
        assert log.revision_count == 1
        assert log.entries[0].goal_revised is True

    def test_constraint_violation_captured(self) -> None:
        collector, bus = _make_collector()
        bus.publish(ConstraintViolated(
            source_id="agent-1",
            constraint_name="safety",
            violation_details="out of bounds",
            timestamp=time.time(),
        ))
        bus.publish(LoopStepCompleted(
            source_id="agent-1",
            step_number=1,
            agent_state=None,
            elapsed_ms=5.0,
            goal_id="g1",
            timestamp=time.time(),
        ))
        log = collector.get_log("agent-1")
        assert log is not None
        assert log.entries[0].constraint_violated is True
        assert len(log.constraint_violations) == 1

    def test_reflection_captured(self) -> None:
        collector, bus = _make_collector()
        bus.publish(ReflectionTriggered(
            source_id="agent-1",
            trigger_reason="periodic",
            timestamp=time.time(),
        ))
        bus.publish(LoopStepCompleted(
            source_id="agent-1",
            step_number=1,
            agent_state=None,
            elapsed_ms=5.0,
            goal_id="g1",
            timestamp=time.time(),
        ))
        log = collector.get_log("agent-1")
        assert log is not None
        assert log.entries[0].reflection_triggered is True

    def test_multiple_agents(self) -> None:
        collector, bus = _make_collector()
        for agent_id in ["a1", "a2"]:
            bus.publish(LoopStepCompleted(
                source_id=agent_id,
                step_number=1,
                agent_state=None,
                elapsed_ms=5.0,
                goal_id="g1",
                timestamp=time.time(),
            ))
        assert set(collector.agent_ids()) == {"a1", "a2"}
        assert len(collector.get_all_logs()) == 2

    def test_get_log_nonexistent_returns_none(self) -> None:
        collector, _ = _make_collector()
        assert collector.get_log("missing") is None

    def test_clear(self) -> None:
        collector, bus = _make_collector()
        bus.publish(LoopStepCompleted(
            source_id="agent-1",
            step_number=1,
            agent_state=None,
            elapsed_ms=5.0,
            goal_id="g1",
            timestamp=time.time(),
        ))
        collector.clear()
        assert collector.get_log("agent-1") is None
        assert collector.agent_ids() == []

    def test_multiple_steps_accumulate(self) -> None:
        collector, bus = _make_collector()
        for i in range(1, 4):
            bus.publish(EvaluationCompleted(
                source_id="agent-1",
                goal_id="g1",
                eval_signal=EvalSignal(score=float(i) * 0.1),
                revision_triggered=False,
                timestamp=time.time(),
            ))
            bus.publish(LoopStepCompleted(
                source_id="agent-1",
                step_number=i,
                agent_state=None,
                elapsed_ms=5.0,
                goal_id="g1",
                timestamp=time.time(),
            ))
        log = collector.get_log("agent-1")
        assert log is not None
        assert log.num_steps == 3
        assert log.get_scores() == pytest.approx([0.1, 0.2, 0.3])
