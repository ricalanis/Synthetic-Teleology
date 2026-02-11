"""Integration tests for the full teleological graph."""

from __future__ import annotations

from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.enums import Direction
from synthetic_teleology.domain.values import (
    ActionSpec,
    ObjectiveVector,
    StateSnapshot,
)
from synthetic_teleology.environments.numeric import NumericEnvironment
from synthetic_teleology.graph.graph import build_teleological_graph
from synthetic_teleology.services.constraint_engine import (
    ConstraintPipeline,
    PolicyFilter,
    SafetyChecker,
)
from synthetic_teleology.services.evaluation import NumericEvaluator
from synthetic_teleology.services.goal_revision import ThresholdUpdater
from synthetic_teleology.services.loop import SyncAgenticLoop
from synthetic_teleology.services.planning import GreedyPlanner


class TestBuildTeleologicalGraph:

    def test_graph_compiles(self) -> None:
        app = build_teleological_graph()
        assert app is not None

    def test_graph_with_checkpointer(self) -> None:
        from langgraph.checkpoint.memory import MemorySaver
        app = build_teleological_graph(checkpointer=MemorySaver())
        assert app is not None


class TestGraphInvoke:

    def test_basic_invocation(self, initial_state: dict) -> None:
        app = build_teleological_graph()
        result = app.invoke(initial_state)
        assert result["step"] > 0
        assert len(result["events"]) > 0

    def test_reaches_max_steps(self, initial_state: dict) -> None:
        initial_state["max_steps"] = 5
        app = build_teleological_graph()
        result = app.invoke(initial_state)
        assert result["step"] == 5
        assert result["stop_reason"] == "max_steps"

    def test_achieves_goal_early(self) -> None:
        """Goal at (0.5, 0.5), start at (0, 0), should converge quickly."""
        env = NumericEnvironment(dimensions=2, initial_state=(0.0, 0.0))
        objective = ObjectiveVector(
            values=(0.5, 0.5),
            directions=(Direction.APPROACH, Direction.APPROACH),
        )
        goal = Goal(name="close-target", objective=objective)

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

        evaluator = NumericEvaluator(max_distance=10.0)
        updater = ThresholdUpdater(threshold=0.99)
        planner = GreedyPlanner(action_space=actions)
        pipeline = ConstraintPipeline(checkers=[])
        pf = PolicyFilter(pipeline)

        app = build_teleological_graph()
        result = app.invoke({
            "step": 0,
            "max_steps": 50,
            "goal_achieved_threshold": 0.9,
            "goal": goal,
            "evaluator": evaluator,
            "goal_updater": updater,
            "planner": planner,
            "constraint_pipeline": pipeline,
            "policy_filter": pf,
            "perceive_fn": lambda: env.observe(),
            "act_fn": lambda p, s: p.actions[0] if p.size > 0 else None,
            "transition_fn": lambda a: env.step(a) if a else None,
            "events": [],
            "goal_history": [],
            "eval_history": [],
            "action_history": [],
            "metadata": {},
        })

        assert result["stop_reason"] == "goal_achieved"
        assert result["step"] < 50

    def test_accumulates_events(self, initial_state: dict) -> None:
        initial_state["max_steps"] = 3
        app = build_teleological_graph()
        result = app.invoke(initial_state)
        # Each step emits multiple events (evaluate, plan, act, reflect)
        assert len(result["events"]) >= 3

    def test_accumulates_eval_history(self, initial_state: dict) -> None:
        initial_state["max_steps"] = 5
        app = build_teleological_graph()
        result = app.invoke(initial_state)
        assert len(result["eval_history"]) == 5

    def test_accumulates_action_history(self, initial_state: dict) -> None:
        initial_state["max_steps"] = 5
        app = build_teleological_graph()
        result = app.invoke(initial_state)
        # Each step that doesn't stop early produces an action
        assert len(result["action_history"]) >= 1

    def test_state_snapshot_updated(self, initial_state: dict) -> None:
        initial_state["max_steps"] = 3
        app = build_teleological_graph()
        result = app.invoke(initial_state)
        assert isinstance(result["state_snapshot"], StateSnapshot)

    def test_with_constraints(self) -> None:
        """Run with a SafetyChecker active."""
        env = NumericEnvironment(dimensions=2, initial_state=(5.0, 5.0))
        objective = ObjectiveVector(
            values=(10.0, 10.0),
            directions=(Direction.APPROACH, Direction.APPROACH),
        )
        goal = Goal(name="constrained-target", objective=objective)

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

        checker = SafetyChecker(
            lower_bounds=(0.0, 0.0),
            upper_bounds=(20.0, 20.0),
        )
        pipeline = ConstraintPipeline(checkers=[checker])
        pf = PolicyFilter(pipeline)

        app = build_teleological_graph()
        result = app.invoke({
            "step": 0,
            "max_steps": 10,
            "goal_achieved_threshold": 0.9,
            "goal": goal,
            "evaluator": NumericEvaluator(max_distance=15.0),
            "goal_updater": ThresholdUpdater(threshold=0.8),
            "planner": GreedyPlanner(action_space=actions),
            "constraint_pipeline": pipeline,
            "policy_filter": pf,
            "perceive_fn": lambda: env.observe(),
            "act_fn": lambda p, s: p.actions[0] if p.size > 0 else None,
            "transition_fn": lambda a: env.step(a) if a else None,
            "events": [],
            "goal_history": [],
            "eval_history": [],
            "action_history": [],
            "metadata": {},
        })

        assert result["step"] > 0
        assert result["constraints_ok"] is True


class TestGraphParityWithSyncLoop:
    """Verify the LangGraph produces comparable results to SyncAgenticLoop."""

    def test_same_direction_convergence(self) -> None:
        """Both should converge toward the target from the same start."""
        # Setup shared config
        start = (1.0, 1.0)
        target = (5.0, 5.0)
        max_steps = 20

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

        evaluator = NumericEvaluator(max_distance=10.0)
        updater = ThresholdUpdater(threshold=0.99)  # High threshold = no revisions
        planner = GreedyPlanner(action_space=actions)
        pipeline = ConstraintPipeline(checkers=[])
        pf = PolicyFilter(pipeline)

        objective = ObjectiveVector(
            values=target,
            directions=(Direction.APPROACH, Direction.APPROACH),
        )

        # Run SyncAgenticLoop
        env1 = NumericEnvironment(dimensions=2, initial_state=start)
        goal1 = Goal(name="sync-target", objective=objective)
        loop = SyncAgenticLoop(
            evaluator=evaluator,
            goal_updater=updater,
            planner=planner,
            constraint_pipeline=pipeline,
            max_steps=max_steps,
            goal_achieved_threshold=0.9,
            perceive_fn=lambda: env1.observe(),
            act_fn=lambda p, s: p.actions[0] if p.size > 0 else None,
            transition_fn=lambda a: env1.step(a) if a else None,
        )
        sync_result = loop.run(goal1)

        # Run LangGraph
        env2 = NumericEnvironment(dimensions=2, initial_state=start)
        goal2 = Goal(name="graph-target", objective=objective)
        app = build_teleological_graph()
        graph_result = app.invoke({
            "step": 0,
            "max_steps": max_steps,
            "goal_achieved_threshold": 0.9,
            "goal": goal2,
            "evaluator": evaluator,
            "goal_updater": updater,
            "planner": planner,
            "constraint_pipeline": pipeline,
            "policy_filter": pf,
            "perceive_fn": lambda: env2.observe(),
            "act_fn": lambda p, s: p.actions[0] if p.size > 0 else None,
            "transition_fn": lambda a: env2.step(a) if a else None,
            "events": [],
            "goal_history": [],
            "eval_history": [],
            "action_history": [],
            "metadata": {},
        })

        # Both should have similar step counts and final scores
        assert abs(sync_result.steps_completed - graph_result["step"]) <= 2
        sync_score = sync_result.metadata.get("last_eval_score", 0)
        graph_score = graph_result["eval_signal"].score
        assert abs(sync_score - graph_score) < 0.15
