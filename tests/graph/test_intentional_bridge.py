"""Tests for Intentional State Mapping — LangGraph bridge."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.values import (
    ActionSpec,
    EvalSignal,
    PolicySpec,
    StateSnapshot,
)
from synthetic_teleology.graph.intentional_bridge import (
    build_intentional_teleological_graph,
    make_intentional_perceive_node,
    make_intentional_plan_node,
    make_intentional_revise_node,
)


def _mock_intentional_agent(**kwargs):
    """Create a mock intentional-state agent with configurable attributes."""
    agent = SimpleNamespace(
        beliefs={},
        desires=kwargs.get("desires", []),
        intentions=kwargs.get("intentions"),
    )
    agent.reconsider = kwargs.get("reconsider", lambda s, e: None)
    agent.revise_goal = kwargs.get("revise_goal", lambda s, e: None)
    return agent


class TestMakeIntentionalPerceiveNode:
    def test_updates_beliefs(self) -> None:
        agent = _mock_intentional_agent()
        node_fn = make_intentional_perceive_node(agent)

        snapshot = StateSnapshot(values=(1.0, 2.0, 3.0), timestamp=0.0)
        state = {
            "perceive_fn": lambda: snapshot,
            "step": 0,
        }

        result = node_fn(state)
        assert result["step"] == 1
        assert result["state_snapshot"] is snapshot
        assert agent.beliefs["state_values"] == (1.0, 2.0, 3.0)
        assert agent.beliefs["step"] == 1

    def test_builds_observation_from_values(self) -> None:
        agent = _mock_intentional_agent()
        node_fn = make_intentional_perceive_node(agent)

        snapshot = StateSnapshot(values=(5.0,), timestamp=0.0)
        state = {"perceive_fn": lambda: snapshot, "step": 0}
        result = node_fn(state)
        assert "5.0" in result["observation"]


class TestMakeIntentionalReviseNode:
    def test_fallback_to_standard_updater(self) -> None:
        """When desires don't change, falls back to standard updater."""
        agent = _mock_intentional_agent()
        node_fn = make_intentional_revise_node(agent)

        goal = Goal(name="test")
        updater = MagicMock()
        updater.update.return_value = None  # No revision

        state = {
            "goal_updater": updater,
            "goal": goal,
            "state_snapshot": StateSnapshot(values=(1.0,), timestamp=0.0),
            "eval_signal": EvalSignal(score=0.5, confidence=0.8),
            "step": 1,
        }

        result = node_fn(state)
        assert result == {}
        updater.update.assert_called_once()

    def test_desire_reconsideration_triggers_revision(self) -> None:
        """When desires change, uses agent goal revision."""
        revised_goal = Goal(name="revised")

        desires_before = [(1, "desire_a")]
        desires_after = [(1, "desire_b")]

        agent = _mock_intentional_agent(desires=desires_before)

        call_count = [0]
        def reconsider(s, e):
            call_count[0] += 1
            agent.desires = desires_after

        agent.reconsider = reconsider
        agent.revise_goal = lambda s, e: revised_goal

        node_fn = make_intentional_revise_node(agent)

        state = {
            "goal_updater": MagicMock(update=MagicMock(return_value=None)),
            "goal": Goal(name="original"),
            "state_snapshot": StateSnapshot(values=(1.0,), timestamp=0.0),
            "eval_signal": EvalSignal(score=0.3, confidence=0.5),
            "step": 1,
        }

        result = node_fn(state)
        assert call_count[0] == 1
        assert result.get("goal") is revised_goal
        events = result.get("events", [])
        assert any(e.get("source") == "intentional_desire_reconsideration" for e in events)

    def test_updates_beliefs_with_eval(self) -> None:
        agent = _mock_intentional_agent()
        node_fn = make_intentional_revise_node(agent)

        state = {
            "goal_updater": MagicMock(update=MagicMock(return_value=None)),
            "goal": Goal(name="test"),
            "state_snapshot": StateSnapshot(values=(1.0,), timestamp=0.0),
            "eval_signal": EvalSignal(score=0.7, confidence=0.9),
            "step": 1,
        }

        node_fn(state)
        assert agent.beliefs["eval_score"] == 0.7
        assert agent.beliefs["eval_confidence"] == 0.9


class TestMakeIntentionalPlanNode:
    def test_reuses_intention(self) -> None:
        """When agent has valid intention, reuse it."""
        policy = PolicySpec(actions=(ActionSpec(name="act1"),))
        agent = _mock_intentional_agent(intentions=policy)
        node_fn = make_intentional_plan_node(agent)

        planner = MagicMock()
        state = {
            "planner": planner,
            "goal": Goal(name="test"),
            "state_snapshot": StateSnapshot(values=(1.0,), timestamp=0.0),
            "step": 1,
        }

        result = node_fn(state)
        assert result["policy"] is policy
        planner.plan.assert_not_called()

    def test_fallback_to_planner_when_no_intention(self) -> None:
        """When agent has no intention, falls back to planner."""
        agent = _mock_intentional_agent(intentions=None)
        node_fn = make_intentional_plan_node(agent)

        new_policy = PolicySpec(actions=(ActionSpec(name="planned"),))
        planner = MagicMock()
        planner.plan.return_value = new_policy

        state = {
            "planner": planner,
            "goal": Goal(name="test"),
            "state_snapshot": StateSnapshot(values=(1.0,), timestamp=0.0),
            "step": 1,
        }

        result = node_fn(state)
        assert result["policy"] is new_policy
        planner.plan.assert_called_once()
        # Intention should be updated
        assert agent.intentions is new_policy

    def test_empty_intention_falls_back(self) -> None:
        """Empty action list in intention triggers fallback."""
        agent = _mock_intentional_agent(intentions=PolicySpec(actions=()))
        node_fn = make_intentional_plan_node(agent)

        new_policy = PolicySpec(actions=(ActionSpec(name="fallback"),))
        planner = MagicMock()
        planner.plan.return_value = new_policy

        state = {
            "planner": planner,
            "goal": Goal(name="test"),
            "state_snapshot": StateSnapshot(values=(1.0,), timestamp=0.0),
            "step": 1,
        }

        result = node_fn(state)
        assert result["policy"] is new_policy


class TestBuildIntentionalTeleologicalGraph:
    def test_compiles(self) -> None:
        agent = _mock_intentional_agent()
        graph = build_intentional_teleological_graph(agent)
        assert graph is not None

    def test_compiles_with_interrupt(self) -> None:
        agent = _mock_intentional_agent()
        graph = build_intentional_teleological_graph(
            agent,
            interrupt_before=["act"],
        )
        assert graph is not None


class TestBackwardCompatShims:
    """Verify that deprecated BDI names still work."""

    def test_bdi_bridge_re_exports(self) -> None:
        from synthetic_teleology.graph.bdi_bridge import (
            build_bdi_teleological_graph,
            make_bdi_perceive_node,
            make_bdi_plan_node,
            make_bdi_revise_node,
        )
        assert build_bdi_teleological_graph is build_intentional_teleological_graph
        assert make_bdi_perceive_node is make_intentional_perceive_node
        assert make_bdi_plan_node is make_intentional_plan_node
        assert make_bdi_revise_node is make_intentional_revise_node

    def test_bdi_agent_alias(self) -> None:
        from synthetic_teleology.agents.bdi import BDIAgent
        from synthetic_teleology.agents.intentional import IntentionalStateAgent
        assert BDIAgent is IntentionalStateAgent

    def test_graph_init_exports_both(self) -> None:
        from synthetic_teleology.graph import (
            build_bdi_teleological_graph,
            build_intentional_teleological_graph,
            make_bdi_perceive_node,
            make_intentional_perceive_node,
        )
        assert build_bdi_teleological_graph is build_intentional_teleological_graph
        assert make_bdi_perceive_node is make_intentional_perceive_node

    def test_builder_deprecated_with_bdi_agent(self) -> None:
        from synthetic_teleology.graph.builder import GraphBuilder
        builder = GraphBuilder("test")
        mock_agent = _mock_intentional_agent()
        result = builder.with_bdi_agent(mock_agent)
        assert result is builder
        assert builder._intentional_agent is mock_agent


class TestBenchmarkUpgrades:
    """Verify benchmark enhancement params (carried from old test file)."""

    def test_distribution_shift_modes(self) -> None:
        from synthetic_teleology.measurement.benchmarks.distribution_shift import (
            DistributionShiftBenchmark,
        )

        # sudden (default)
        b = DistributionShiftBenchmark()
        assert b._shift_mode == "sudden"

        # gradual
        b = DistributionShiftBenchmark(shift_mode="gradual", transition_steps=10)
        assert b._shift_mode == "gradual"
        assert b._transition_steps == 10

        # shift_type
        b = DistributionShiftBenchmark(shift_type="dynamics")
        assert b._shift_type == "dynamics"

        with pytest.raises(ValueError, match="shift_mode"):
            DistributionShiftBenchmark(shift_mode="invalid")

        with pytest.raises(ValueError, match="shift_type"):
            DistributionShiftBenchmark(shift_type="invalid")

    def test_conflicting_obj_tradeoff(self) -> None:
        from synthetic_teleology.measurement.benchmarks.conflicting_obj import (
            ConflictingObjectivesBenchmark,
        )

        b = ConflictingObjectivesBenchmark(tradeoff_step=25, tradeoff_multiplier=2.0)
        assert b._tradeoff_step == 25
        assert b._tradeoff_multiplier == 2.0

    def test_negotiation_strategy(self) -> None:
        from synthetic_teleology.measurement.benchmarks.negotiation import (
            NegotiationBenchmark,
        )

        b = NegotiationBenchmark(strategy="consensus")
        assert b._strategy == "consensus"

        b = NegotiationBenchmark(strategy="voting")
        assert b._strategy == "voting"

        with pytest.raises(ValueError, match="strategy"):
            NegotiationBenchmark(strategy="invalid")

    def test_knowledge_synthesis_critic(self) -> None:
        from synthetic_teleology.measurement.benchmarks.knowledge_synthesis import (
            KnowledgeSynthesisBenchmark,
        )

        b = KnowledgeSynthesisBenchmark(critic_interval=5)
        assert b._critic_interval == 5
