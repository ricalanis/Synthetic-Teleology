"""Tests for GraphBuilder in LLM mode."""

from __future__ import annotations

from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.graph.builder import GraphBuilder
from synthetic_teleology.services.llm_evaluation import EvaluationOutput, LLMEvaluator
from synthetic_teleology.services.llm_planning import LLMPlanner
from tests.helpers.mock_llm import MockStructuredChatModel


def _make_model() -> MockStructuredChatModel:
    """Create a MockStructuredChatModel for testing builder construction."""
    return MockStructuredChatModel(structured_responses=[
        EvaluationOutput(score=0.5, confidence=0.8, reasoning="test"),
    ])


class TestGraphBuilderLLMMode:

    def test_llm_mode_detection(self) -> None:
        builder = GraphBuilder("agent").with_model(_make_model()).with_goal("Test goal")
        assert builder._is_llm_mode is True

    def test_numeric_mode_detection(self) -> None:
        builder = GraphBuilder("agent").with_objective((5.0,))
        assert builder._is_llm_mode is False

    def test_build_llm_mode(self) -> None:
        app, state = (
            GraphBuilder("llm-agent")
            .with_model(_make_model())
            .with_goal("Increase revenue", criteria=["Revenue > $120k"])
            .with_max_steps(20)
            .build()
        )

        assert app is not None
        assert state["goal"].description == "Increase revenue"
        assert state["goal"].success_criteria == ["Revenue > $120k"]
        assert state["max_steps"] == 20
        assert state["model"] is not None
        assert isinstance(state["evaluator"], LLMEvaluator)
        assert isinstance(state["planner"], LLMPlanner)

    def test_build_llm_mode_with_tools(self) -> None:
        class MockTool:
            name = "search"
            description = "Search the web"

        app, state = (
            GraphBuilder("tool-agent")
            .with_model(_make_model())
            .with_goal("Research topic")
            .with_tools(MockTool())
            .build()
        )

        assert len(state["tools"]) == 1

    def test_build_llm_mode_with_constraints(self) -> None:
        app, state = (
            GraphBuilder("constrained-agent")
            .with_model(_make_model())
            .with_goal("Test goal")
            .with_constraints("No budget overruns", "Respect privacy")
            .build()
        )

        assert state["constraint_pipeline"] is not None

    def test_build_llm_mode_defaults(self) -> None:
        app, state = (
            GraphBuilder("defaults-agent")
            .with_model(_make_model())
            .with_goal("Simple goal")
            .build()
        )

        assert state["step"] == 0
        assert state["max_steps"] == 100
        assert state["goal_achieved_threshold"] == 0.9
        assert state["events"] == []
        assert state["reasoning_trace"] == []
        assert callable(state["perceive_fn"])

    def test_build_llm_mode_custom_threshold(self) -> None:
        app, state = (
            GraphBuilder("thresh-agent")
            .with_model(_make_model())
            .with_goal("Goal")
            .with_goal_achieved_threshold(0.75)
            .build()
        )

        assert state["goal_achieved_threshold"] == 0.75

    def test_build_llm_mode_custom_evaluator(self) -> None:
        from synthetic_teleology.services.evaluation import NumericEvaluator

        custom = NumericEvaluator(max_distance=20.0)
        app, state = (
            GraphBuilder("custom-eval-agent")
            .with_model(_make_model())
            .with_goal("Goal")
            .with_evaluator(custom)
            .build()
        )

        assert state["evaluator"] is custom

    def test_build_llm_mode_with_environment(self) -> None:
        def perceive():
            return None

        def act(p, s):
            return None

        def transition(a):
            return None

        app, state = (
            GraphBuilder("env-agent")
            .with_model(_make_model())
            .with_goal("Goal")
            .with_environment(perceive_fn=perceive, act_fn=act, transition_fn=transition)
            .build()
        )

        assert state["perceive_fn"] is perceive
        assert state["act_fn"] is act
        assert state["transition_fn"] is transition

    def test_with_goal_accepts_goal_entity_in_llm_mode(self) -> None:
        goal = Goal(name="explicit", description="Explicit goal desc")
        app, state = (
            GraphBuilder("entity-llm-agent")
            .with_model(_make_model())
            .with_goal(goal)
            .build()
        )

        assert state["goal"].name == "explicit"
        assert state["goal"].description == "Explicit goal desc"

    def test_with_num_hypotheses(self) -> None:
        app, state = (
            GraphBuilder("hyp-agent")
            .with_model(_make_model())
            .with_goal("Goal")
            .with_num_hypotheses(5)
            .build()
        )

        assert state["num_hypotheses"] == 5

    def test_with_temperature(self) -> None:
        builder = (
            GraphBuilder("temp-agent")
            .with_model(_make_model())
            .with_goal("Goal")
            .with_temperature(0.3)
        )
        assert builder._temperature == 0.3

    def test_repr_llm_mode(self) -> None:
        builder = (
            GraphBuilder("repr-agent")
            .with_model(_make_model())
            .with_goal("My cool goal that is quite long actually")
        )
        r = repr(builder)
        assert "repr-agent" in r
        assert "mode=llm" in r

    def test_default_perceive_fn_in_llm_mode(self) -> None:
        app, state = (
            GraphBuilder("no-env-agent")
            .with_model(_make_model())
            .with_goal("Goal")
            .build()
        )

        snapshot = state["perceive_fn"]()
        assert snapshot is not None

    def test_with_metadata_llm_mode(self) -> None:
        app, state = (
            GraphBuilder("meta-agent")
            .with_model(_make_model())
            .with_goal("Goal")
            .with_metadata(experiment="llm-test")
            .build()
        )

        assert state["metadata"]["experiment"] == "llm-test"
