"""Tests for multi-agent coordination graph."""

from __future__ import annotations

from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.enums import Direction
from synthetic_teleology.domain.values import ActionSpec, ObjectiveVector
from synthetic_teleology.environments.numeric import NumericEnvironment
from synthetic_teleology.graph.multi_agent import AgentConfig, build_multi_agent_graph
from synthetic_teleology.services.constraint_engine import ConstraintPipeline, PolicyFilter
from synthetic_teleology.services.evaluation import NumericEvaluator
from synthetic_teleology.services.goal_revision import ThresholdUpdater
from synthetic_teleology.services.planning import GreedyPlanner


def _make_action_space() -> list[ActionSpec]:
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
    return actions


class TestBuildMultiAgentGraph:

    def test_compiles(self) -> None:
        env1 = NumericEnvironment(dimensions=2, initial_state=(0.0, 0.0))
        env2 = NumericEnvironment(dimensions=2, initial_state=(10.0, 10.0))
        configs = [
            AgentConfig(
                agent_id="agent-1",
                goal=Goal(name="g1", objective=ObjectiveVector(
                    values=(5.0, 5.0), directions=(Direction.APPROACH, Direction.APPROACH)
                )),
                perceive_fn=lambda: env1.observe(),
                transition_fn=lambda a: env1.step(a) if a else None,
                max_steps_per_round=3,
            ),
            AgentConfig(
                agent_id="agent-2",
                goal=Goal(name="g2", objective=ObjectiveVector(
                    values=(7.0, 7.0), directions=(Direction.APPROACH, Direction.APPROACH)
                )),
                perceive_fn=lambda: env2.observe(),
                transition_fn=lambda a: env2.step(a) if a else None,
                max_steps_per_round=3,
            ),
        ]
        app = build_multi_agent_graph(configs, max_rounds=2)
        assert app is not None

    def test_invoke_runs(self) -> None:
        env1 = NumericEnvironment(dimensions=2, initial_state=(0.0, 0.0))
        env2 = NumericEnvironment(dimensions=2, initial_state=(10.0, 10.0))

        actions = _make_action_space()
        pipeline = ConstraintPipeline(checkers=[])

        configs = [
            AgentConfig(
                agent_id="agent-1",
                goal=Goal(name="g1", objective=ObjectiveVector(
                    values=(5.0, 5.0), directions=(Direction.APPROACH, Direction.APPROACH)
                )),
                perceive_fn=lambda: env1.observe(),
                transition_fn=lambda a: env1.step(a) if a else None,
                act_fn=lambda p, s: p.actions[0] if p.size > 0 else None,
                max_steps_per_round=3,
            ),
            AgentConfig(
                agent_id="agent-2",
                goal=Goal(name="g2", objective=ObjectiveVector(
                    values=(7.0, 7.0), directions=(Direction.APPROACH, Direction.APPROACH)
                )),
                perceive_fn=lambda: env2.observe(),
                transition_fn=lambda a: env2.step(a) if a else None,
                act_fn=lambda p, s: p.actions[0] if p.size > 0 else None,
                max_steps_per_round=3,
            ),
        ]

        app = build_multi_agent_graph(configs, max_rounds=2)
        result = app.invoke({
            "evaluator": NumericEvaluator(max_distance=15.0),
            "goal_updater": ThresholdUpdater(threshold=0.99),
            "planner": GreedyPlanner(action_space=actions),
            "constraint_pipeline": pipeline,
            "policy_filter": PolicyFilter(pipeline),
            "goal_achieved_threshold": 0.9,
            "max_rounds": 2,
            "events": [],
            "agent_results": {},
            "negotiation_round": 0,
        })

        assert "agent_results" in result
        assert len(result["agent_results"]) == 2
        assert len(result["events"]) > 0

    def test_negotiation_produces_shared_objective(self) -> None:
        env1 = NumericEnvironment(dimensions=2, initial_state=(0.0, 0.0))
        env2 = NumericEnvironment(dimensions=2, initial_state=(10.0, 10.0))

        actions = _make_action_space()
        pipeline = ConstraintPipeline(checkers=[])

        configs = [
            AgentConfig(
                agent_id="a1",
                goal=Goal(name="g1", objective=ObjectiveVector(
                    values=(3.0, 3.0), directions=(Direction.APPROACH, Direction.APPROACH)
                )),
                perceive_fn=lambda: env1.observe(),
                transition_fn=lambda a: env1.step(a) if a else None,
                act_fn=lambda p, s: p.actions[0] if p.size > 0 else None,
                max_steps_per_round=2,
            ),
            AgentConfig(
                agent_id="a2",
                goal=Goal(name="g2", objective=ObjectiveVector(
                    values=(7.0, 7.0), directions=(Direction.APPROACH, Direction.APPROACH)
                )),
                perceive_fn=lambda: env2.observe(),
                transition_fn=lambda a: env2.step(a) if a else None,
                act_fn=lambda p, s: p.actions[0] if p.size > 0 else None,
                max_steps_per_round=2,
            ),
        ]

        app = build_multi_agent_graph(configs, max_rounds=3)
        result = app.invoke({
            "evaluator": NumericEvaluator(max_distance=15.0),
            "goal_updater": ThresholdUpdater(threshold=0.99),
            "planner": GreedyPlanner(action_space=actions),
            "constraint_pipeline": pipeline,
            "policy_filter": PolicyFilter(pipeline),
            "goal_achieved_threshold": 0.9,
            "max_rounds": 3,
            "events": [],
            "agent_results": {},
            "negotiation_round": 0,
        })

        # Should have negotiation rounds
        assert result.get("negotiation_round", 0) >= 1
