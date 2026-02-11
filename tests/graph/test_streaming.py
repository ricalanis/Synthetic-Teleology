"""Tests for stream event formatters."""

from __future__ import annotations

from synthetic_teleology.environments.numeric import NumericEnvironment
from synthetic_teleology.graph.graph import build_teleological_graph
from synthetic_teleology.graph.streaming import (
    collect_stream_events,
    format_stream_events,
    stream_to_agent_log_entries,
)


class TestStreamFormatters:

    def _make_state(self) -> dict:
        from synthetic_teleology.domain.entities import Goal
        from synthetic_teleology.domain.enums import Direction
        from synthetic_teleology.domain.values import ActionSpec, ObjectiveVector
        from synthetic_teleology.services.constraint_engine import ConstraintPipeline, PolicyFilter
        from synthetic_teleology.services.evaluation import NumericEvaluator
        from synthetic_teleology.services.goal_revision import ThresholdUpdater
        from synthetic_teleology.services.planning import GreedyPlanner

        env = NumericEnvironment(dimensions=2, initial_state=(0.0, 0.0))
        actions: list[ActionSpec] = []
        for d in range(2):
            for sign, label in [(0.5, "pos"), (-0.5, "neg")]:
                effect = tuple(sign if i == d else 0.0 for i in range(2))
                actions.append(ActionSpec(
                    name=f"s{d}_{label}",
                    parameters={"effect": effect, "delta": effect},
                ))
        actions.append(ActionSpec(
            name="noop",
            parameters={"effect": (0.0, 0.0), "delta": (0.0, 0.0)},
        ))

        pipeline = ConstraintPipeline(checkers=[])
        return {
            "step": 0,
            "max_steps": 3,
            "goal_achieved_threshold": 0.9,
            "goal": Goal(name="test", objective=ObjectiveVector(
                values=(5.0, 5.0), directions=(Direction.APPROACH, Direction.APPROACH)
            )),
            "evaluator": NumericEvaluator(max_distance=10.0),
            "goal_updater": ThresholdUpdater(threshold=0.99),
            "planner": GreedyPlanner(action_space=actions),
            "constraint_pipeline": pipeline,
            "policy_filter": PolicyFilter(pipeline),
            "perceive_fn": lambda: env.observe(),
            "act_fn": lambda p, s: p.actions[0] if p.size > 0 else None,
            "transition_fn": lambda a: env.step(a) if a else None,
            "events": [],
            "goal_history": [],
            "eval_history": [],
            "action_history": [],
            "metadata": {},
        }, env

    def test_format_stream_events(self) -> None:
        state, _ = self._make_state()
        app = build_teleological_graph()
        stream = app.stream(state, stream_mode="updates")
        events = list(format_stream_events(stream))
        assert len(events) > 0
        assert all("node" in e for e in events)
        assert all("timestamp" in e for e in events)

    def test_collect_stream_events(self) -> None:
        state, _ = self._make_state()
        app = build_teleological_graph()
        stream = app.stream(state, stream_mode="updates")
        events = collect_stream_events(stream)
        assert isinstance(events, list)
        assert len(events) > 0

    def test_stream_to_agent_log_entries(self) -> None:
        state, _ = self._make_state()
        app = build_teleological_graph()
        stream = app.stream(state, stream_mode="updates")
        entries = stream_to_agent_log_entries(stream)
        assert isinstance(entries, list)
        # Should have at least some reflection or action entries
        assert len(entries) > 0
