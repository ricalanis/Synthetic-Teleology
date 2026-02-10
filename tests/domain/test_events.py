"""Tests for domain events."""

from __future__ import annotations

import time

import pytest

from synthetic_teleology.domain.enums import AgentState, NegotiationStrategy, RevisionReason
from synthetic_teleology.domain.values import EvalSignal, ObjectiveVector
from synthetic_teleology.domain.enums import Direction
from synthetic_teleology.domain.events import (
    ActionExecuted,
    AgentRegistered,
    ConsensusReached,
    ConstraintRestored,
    ConstraintViolated,
    DomainEvent,
    EvaluationCompleted,
    GoalAbandoned,
    GoalAchieved,
    GoalCreated,
    GoalRevised,
    LoopStepCompleted,
    NegotiationStarted,
    PerturbationInjected,
    PlanGenerated,
    ReflectionTriggered,
    StateChanged,
)


class TestDomainEvent:
    """Test base DomainEvent and subclass creation, immutability, timestamp."""

    def test_base_event_has_timestamp(self) -> None:
        before = time.time()
        event = DomainEvent(source_id="test")
        after = time.time()
        assert before <= event.timestamp <= after

    def test_event_immutability(self) -> None:
        event = DomainEvent(source_id="test")
        with pytest.raises(AttributeError):
            event.source_id = "changed"  # type: ignore[misc]

    def test_goal_created(self) -> None:
        obj = ObjectiveVector(
            values=(1.0, 2.0),
            directions=(Direction.APPROACH, Direction.APPROACH),
        )
        event = GoalCreated(
            source_id="agent-1",
            goal_id="g1",
            goal_name="my-goal",
            objective=obj,
        )
        assert event.goal_id == "g1"
        assert event.goal_name == "my-goal"
        assert event.objective is obj

    def test_goal_revised(self) -> None:
        event = GoalRevised(
            source_id="agent-1",
            reason=RevisionReason.THRESHOLD_EXCEEDED,
        )
        assert event.reason == RevisionReason.THRESHOLD_EXCEEDED

    def test_goal_abandoned(self) -> None:
        event = GoalAbandoned(
            source_id="agent-1",
            goal_id="g1",
            reason="infeasible",
        )
        assert event.goal_id == "g1"
        assert event.reason == "infeasible"

    def test_goal_achieved(self) -> None:
        sig = EvalSignal(score=0.95, confidence=1.0)
        event = GoalAchieved(
            source_id="agent-1",
            goal_id="g1",
            final_eval=sig,
        )
        assert event.final_eval is sig

    def test_evaluation_completed(self) -> None:
        sig = EvalSignal(score=0.5)
        event = EvaluationCompleted(
            source_id="agent-1",
            goal_id="g1",
            eval_signal=sig,
            revision_triggered=True,
        )
        assert event.revision_triggered is True

    def test_loop_step_completed(self) -> None:
        event = LoopStepCompleted(
            step_number=10,
            agent_state=AgentState.IDLE,
            elapsed_ms=50.0,
            goal_id="g1",
        )
        assert event.step_number == 10
        assert event.agent_state == AgentState.IDLE

    def test_constraint_violated(self) -> None:
        event = ConstraintViolated(
            constraint_id="c1",
            constraint_name="safety",
            violation_details="dim[0] out of bounds",
            severity=0.9,
        )
        assert event.severity == 0.9

    def test_constraint_restored(self) -> None:
        event = ConstraintRestored(
            constraint_id="c1",
            constraint_name="safety",
        )
        assert event.constraint_name == "safety"

    def test_perturbation_injected(self) -> None:
        event = PerturbationInjected(
            perturbation_type="shift",
            magnitude=5.0,
            affected_dimensions=(0, 1),
        )
        assert event.magnitude == 5.0
        assert event.affected_dimensions == (0, 1)

    def test_state_changed(self) -> None:
        event = StateChanged(
            agent_id="a1",
            previous_state=AgentState.IDLE,
            new_state=AgentState.PERCEIVING,
        )
        assert event.previous_state == AgentState.IDLE
        assert event.new_state == AgentState.PERCEIVING

    def test_negotiation_started(self) -> None:
        event = NegotiationStarted(
            negotiation_id="n1",
            participant_ids=("a1", "a2"),
            strategy=NegotiationStrategy.CONSENSUS,
        )
        assert event.participant_ids == ("a1", "a2")

    def test_consensus_reached(self) -> None:
        event = ConsensusReached(
            negotiation_id="n1",
            rounds=3,
        )
        assert event.rounds == 3

    def test_agent_registered(self) -> None:
        event = AgentRegistered(agent_id="a1", agent_name="Agent One")
        assert event.agent_name == "Agent One"

    def test_reflection_triggered(self) -> None:
        event = ReflectionTriggered(
            goal_id="g1",
            trigger_reason="low confidence",
        )
        assert event.trigger_reason == "low confidence"

    def test_action_executed(self) -> None:
        event = ActionExecuted(success=True, error_message="")
        assert event.success is True

    def test_plan_generated(self) -> None:
        event = PlanGenerated(goal_id="g1", planning_time_ms=12.5)
        assert event.planning_time_ms == 12.5
