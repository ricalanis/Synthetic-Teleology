"""Tests for prebuilt agent constructors."""

from __future__ import annotations

from synthetic_teleology.environments.numeric import NumericEnvironment
from synthetic_teleology.graph.prebuilt import (
    create_llm_teleological_agent,
    create_react_teleological_agent,
    create_teleological_agent,
)


class TestCreateTeleologicalAgent:

    def test_basic_creation(self) -> None:
        env = NumericEnvironment(dimensions=2, initial_state=(0.0, 0.0))
        app, state = create_teleological_agent(
            target_values=(5.0, 5.0),
            perceive_fn=lambda: env.observe(),
            transition_fn=lambda a: env.step(a) if a else None,
            max_steps=10,
        )
        assert app is not None
        assert state["max_steps"] == 10
        assert state["goal"].objective.values == (5.0, 5.0)

    def test_invoke_runs(self) -> None:
        env = NumericEnvironment(dimensions=2, initial_state=(0.0, 0.0))
        app, state = create_teleological_agent(
            target_values=(2.0, 2.0),
            perceive_fn=lambda: env.observe(),
            transition_fn=lambda a: env.step(a) if a else None,
            max_steps=5,
        )
        result = app.invoke(state)
        assert result["step"] == 5

    def test_custom_threshold(self) -> None:
        env = NumericEnvironment(dimensions=1)
        app, state = create_teleological_agent(
            target_values=(1.0,),
            perceive_fn=lambda: env.observe(),
            goal_achieved_threshold=0.5,
            max_steps=5,
        )
        assert state["goal_achieved_threshold"] == 0.5

    def test_custom_step_size(self) -> None:
        env = NumericEnvironment(dimensions=2)
        app, state = create_teleological_agent(
            target_values=(1.0, 1.0),
            perceive_fn=lambda: env.observe(),
            step_size=0.1,
            max_steps=5,
        )
        assert app is not None


class TestCreateLLMTeleologicalAgent:

    def test_creation_with_mock_model(self) -> None:
        env = NumericEnvironment(dimensions=2)
        mock_model = "mock-llm"
        app, state = create_llm_teleological_agent(
            model=mock_model,
            target_values=(3.0, 3.0),
            perceive_fn=lambda: env.observe(),
            max_steps=5,
        )
        assert state["metadata"]["llm_model"] == "mock-llm"

    def test_invoke_runs(self) -> None:
        env = NumericEnvironment(dimensions=2, initial_state=(0.0, 0.0))
        app, state = create_llm_teleological_agent(
            model="mock",
            target_values=(1.0, 1.0),
            perceive_fn=lambda: env.observe(),
            transition_fn=lambda a: env.step(a) if a else None,
            max_steps=3,
        )
        result = app.invoke(state)
        assert result["step"] == 3


class TestCreateReActTeleologicalAgent:

    def test_creation_with_mock(self) -> None:
        env = NumericEnvironment(dimensions=1)
        app, state = create_react_teleological_agent(
            model="mock-llm",
            tools=["tool_a", "tool_b"],
            goal_description="Research quantum computing",
            perceive_fn=lambda: env.observe(),
            max_steps=3,
        )
        assert state["metadata"]["tools"] == ["tool_a", "tool_b"]
        assert state["metadata"]["goal_description"] == "Research quantum computing"

    def test_invoke_runs(self) -> None:
        env = NumericEnvironment(dimensions=1, initial_state=(0.0,))
        app, state = create_react_teleological_agent(
            model="mock",
            tools=[],
            goal_description="Test",
            perceive_fn=lambda: env.observe(),
            transition_fn=lambda a: env.step(a) if a else None,
            max_steps=3,
        )
        result = app.invoke(state)
        assert result["step"] > 0
