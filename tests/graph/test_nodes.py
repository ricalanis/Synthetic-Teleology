"""Tests for individual graph node functions."""

from __future__ import annotations

from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.values import (
    ActionSpec,
    EvalSignal,
    PolicySpec,
    StateSnapshot,
)
from synthetic_teleology.graph.nodes import (
    act_node,
    check_constraints_node,
    evaluate_node,
    filter_policy_node,
    perceive_node,
    plan_node,
    reflect_node,
    revise_node,
)


class TestPerceiveNode:

    def test_increments_step(self, initial_state: dict) -> None:
        result = perceive_node(initial_state)
        assert result["step"] == 1

    def test_returns_state_snapshot(self, initial_state: dict) -> None:
        result = perceive_node(initial_state)
        assert isinstance(result["state_snapshot"], StateSnapshot)

    def test_step_increments_from_existing(self, initial_state: dict) -> None:
        initial_state["step"] = 5
        result = perceive_node(initial_state)
        assert result["step"] == 6


class TestEvaluateNode:

    def test_returns_eval_signal(self, initial_state: dict) -> None:
        initial_state.update(perceive_node(initial_state))
        result = evaluate_node(initial_state)
        assert isinstance(result["eval_signal"], EvalSignal)

    def test_appends_to_eval_history(self, initial_state: dict) -> None:
        initial_state.update(perceive_node(initial_state))
        result = evaluate_node(initial_state)
        assert len(result["eval_history"]) == 1

    def test_emits_event(self, initial_state: dict) -> None:
        initial_state.update(perceive_node(initial_state))
        result = evaluate_node(initial_state)
        assert len(result["events"]) == 1
        assert result["events"][0]["type"] == "evaluation_completed"


class TestReviseNode:

    def test_no_revision_returns_empty(self, initial_state: dict) -> None:
        initial_state.update(perceive_node(initial_state))
        initial_state.update(evaluate_node(initial_state))
        # With a moderate eval score, updater may not trigger
        initial_state["eval_signal"] = EvalSignal(score=0.3, confidence=0.9)
        result = revise_node(initial_state)
        # ThresholdUpdater only revises when |score| >= threshold (0.8)
        assert result == {} or "goal" not in result

    def test_revision_when_signal_high(self, initial_state: dict) -> None:
        initial_state.update(perceive_node(initial_state))
        initial_state["eval_signal"] = EvalSignal(score=-0.9, confidence=0.95)
        result = revise_node(initial_state)
        if "goal" in result:
            assert isinstance(result["goal"], Goal)
            assert len(result["goal_history"]) == 1


class TestCheckConstraintsNode:

    def test_no_violations_with_empty_pipeline(self, initial_state: dict) -> None:
        initial_state.update(perceive_node(initial_state))
        result = check_constraints_node(initial_state)
        assert result["constraints_ok"] is True
        assert result["constraint_violations"] == []


class TestPlanNode:

    def test_returns_policy(self, initial_state: dict) -> None:
        initial_state.update(perceive_node(initial_state))
        result = plan_node(initial_state)
        assert isinstance(result["policy"], PolicySpec)
        assert result["policy"].size > 0

    def test_emits_plan_event(self, initial_state: dict) -> None:
        initial_state.update(perceive_node(initial_state))
        result = plan_node(initial_state)
        assert result["events"][0]["type"] == "plan_generated"


class TestFilterPolicyNode:

    def test_passes_through_with_empty_pipeline(self, initial_state: dict) -> None:
        initial_state.update(perceive_node(initial_state))
        initial_state.update(plan_node(initial_state))
        result = filter_policy_node(initial_state)
        assert isinstance(result["filtered_policy"], PolicySpec)
        # With no constraint checkers, all actions pass through
        assert result["filtered_policy"].size == initial_state["policy"].size


class TestActNode:

    def test_returns_executed_action(self, initial_state: dict) -> None:
        initial_state.update(perceive_node(initial_state))
        initial_state.update(plan_node(initial_state))
        initial_state.update(filter_policy_node(initial_state))
        result = act_node(initial_state)
        assert result["executed_action"] is not None
        assert isinstance(result["executed_action"], ActionSpec)

    def test_appends_to_action_history(self, initial_state: dict) -> None:
        initial_state.update(perceive_node(initial_state))
        initial_state.update(plan_node(initial_state))
        initial_state.update(filter_policy_node(initial_state))
        result = act_node(initial_state)
        assert len(result["action_history"]) == 1

    def test_act_with_no_act_fn(self, initial_state: dict) -> None:
        initial_state["act_fn"] = None
        initial_state.update(perceive_node(initial_state))
        initial_state.update(plan_node(initial_state))
        initial_state.update(filter_policy_node(initial_state))
        result = act_node(initial_state)
        # Default: picks first action
        assert result["executed_action"] is not None


class TestReflectNode:

    def test_no_stop_when_under_max_steps(self, initial_state: dict) -> None:
        initial_state["step"] = 5
        initial_state["eval_signal"] = EvalSignal(score=0.3, confidence=0.8)
        initial_state["filtered_policy"] = PolicySpec(actions=(ActionSpec(name="a"),))
        result = reflect_node(initial_state)
        assert "stop_reason" not in result

    def test_stop_at_max_steps(self, initial_state: dict) -> None:
        initial_state["step"] = 20
        initial_state["max_steps"] = 20
        initial_state["eval_signal"] = EvalSignal(score=0.3, confidence=0.8)
        result = reflect_node(initial_state)
        assert result["stop_reason"] == "max_steps"

    def test_stop_on_goal_achieved_score(self, initial_state: dict) -> None:
        initial_state["step"] = 5
        initial_state["eval_signal"] = EvalSignal(score=0.95, confidence=0.99)
        result = reflect_node(initial_state)
        assert result["stop_reason"] == "goal_achieved"

    def test_stop_on_empty_policy(self, initial_state: dict) -> None:
        initial_state["step"] = 5
        initial_state["eval_signal"] = EvalSignal(score=0.3, confidence=0.8)
        initial_state["filtered_policy"] = PolicySpec(actions=())
        result = reflect_node(initial_state)
        assert result["stop_reason"] == "empty_policy"

    def test_stop_on_abandoned_goal(self, initial_state: dict) -> None:
        initial_state["goal"].abandon()
        initial_state["step"] = 5
        initial_state["eval_signal"] = EvalSignal(score=0.3, confidence=0.8)
        result = reflect_node(initial_state)
        assert result["stop_reason"] == "goal_abandoned"
