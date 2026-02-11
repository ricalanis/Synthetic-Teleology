"""Tests for closure-based strategy injection in the graph layer.

Verifies that strategies are injected via closures (not stored in state),
that backward compatibility is maintained, and that closure-based graphs
run correctly.
"""

from __future__ import annotations

import time
from typing import Any

from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.values import (
    ActionSpec,
    StateSnapshot,
)
from synthetic_teleology.graph.builder import GraphBuilder
from synthetic_teleology.graph.graph import build_teleological_graph
from synthetic_teleology.services.constraint_engine import ConstraintPipeline, PolicyFilter
from synthetic_teleology.services.evaluation import NumericEvaluator
from synthetic_teleology.services.goal_revision import ThresholdUpdater
from synthetic_teleology.services.planning import GreedyPlanner


def _make_env(values: tuple[float, ...] = (1.0, 1.0)):
    """Create a simple perceive/transition pair for testing."""
    state_values = list(values)

    def perceive() -> StateSnapshot:
        return StateSnapshot(
            timestamp=time.time(),
            values=tuple(state_values),
        )

    def transition(action: ActionSpec) -> None:
        effect = action.parameters.get("effect")
        if effect and isinstance(effect, (list, tuple)):
            for i in range(min(len(state_values), len(effect))):
                state_values[i] += effect[i]

    return perceive, transition


class TestClosureStrategiesNotInState:

    def test_graph_closure_strategies_no_strategies_in_state(self) -> None:
        """Build via GraphBuilder, verify state has no evaluator/planner keys."""
        perceive, transition = _make_env()
        app, state = (
            GraphBuilder("closure-agent")
            .with_objective((5.0, 5.0))
            .with_environment(perceive_fn=perceive, transition_fn=transition)
            .with_max_steps(3)
            .build()
        )

        # Strategies should NOT be in state (they're in closures)
        assert "evaluator" not in state
        assert "planner" not in state
        assert "goal_updater" not in state
        assert "constraint_pipeline" not in state
        assert "policy_filter" not in state

        # Graph should still be valid
        assert app is not None


class TestBackwardCompatStateStrategies:

    def test_graph_backward_compat_state_strategies(self) -> None:
        """Build without kwargs, strategies in state, still works."""
        perceive, transition = _make_env()
        evaluator = NumericEvaluator()
        updater = ThresholdUpdater()
        action_space = [
            ActionSpec(name="noop", parameters={"effect": (0.0, 0.0), "delta": (0.0, 0.0)}),
        ]
        planner = GreedyPlanner(action_space=action_space)
        pipeline = ConstraintPipeline(checkers=[])
        pf = PolicyFilter(pipeline)

        # Build graph WITHOUT strategy kwargs (backward compat path)
        app = build_teleological_graph()

        state: dict[str, Any] = {
            "step": 0,
            "max_steps": 2,
            "goal_achieved_threshold": 0.99,
            "goal": Goal(name="test", objective=__import__(
                "synthetic_teleology.domain.values", fromlist=["ObjectiveVector"]
            ).ObjectiveVector(values=(5.0, 5.0), directions=("APPROACH", "APPROACH"))),
            "evaluator": evaluator,
            "goal_updater": updater,
            "planner": planner,
            "constraint_pipeline": pipeline,
            "policy_filter": pf,
            "perceive_fn": perceive,
            "transition_fn": transition,
            "act_fn": None,
            "events": [],
            "goal_history": [],
            "eval_history": [],
            "action_history": [],
            "reasoning_trace": [],
            "action_feedback": [],
            "metadata": {},
        }

        result = app.invoke(state)
        assert result["step"] == 2


class TestClosureRunsCorrectly:

    def test_graph_closure_runs_correctly(self) -> None:
        """Full invoke with closure strategies, verify result."""
        perceive, transition = _make_env((1.0, 1.0))
        app, state = (
            GraphBuilder("closure-run-agent")
            .with_objective((5.0, 5.0))
            .with_environment(perceive_fn=perceive, transition_fn=transition)
            .with_max_steps(3)
            .build()
        )

        result = app.invoke(state)
        assert result["step"] == 3
        assert len(result["eval_history"]) == 3
        assert result["goal"] is not None
