"""Tests for BDI-LangGraph bridge."""

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
from synthetic_teleology.graph.bdi_bridge import (
    build_bdi_teleological_graph,
    make_bdi_perceive_node,
    make_bdi_plan_node,
    make_bdi_revise_node,
)


def _mock_bdi_agent(**kwargs):
    """Create a mock BDI agent with configurable attributes."""
    agent = SimpleNamespace(
        beliefs={},
        desires=kwargs.get("desires", []),
        intentions=kwargs.get("intentions"),
    )
    agent.reconsider = kwargs.get("reconsider", lambda s, e: None)
    agent.revise_goal = kwargs.get("revise_goal", lambda s, e: None)
    return agent


class TestMakeBdiPerceiveNode:
    def test_updates_beliefs(self) -> None:
        bdi = _mock_bdi_agent()
        node_fn = make_bdi_perceive_node(bdi)

        snapshot = StateSnapshot(values=(1.0, 2.0, 3.0), timestamp=0.0)
        state = {
            "perceive_fn": lambda: snapshot,
            "step": 0,
        }

        result = node_fn(state)
        assert result["step"] == 1
        assert result["state_snapshot"] is snapshot
        assert bdi.beliefs["state_values"] == (1.0, 2.0, 3.0)
        assert bdi.beliefs["step"] == 1

    def test_builds_observation_from_values(self) -> None:
        bdi = _mock_bdi_agent()
        node_fn = make_bdi_perceive_node(bdi)

        snapshot = StateSnapshot(values=(5.0,), timestamp=0.0)
        state = {"perceive_fn": lambda: snapshot, "step": 0}
        result = node_fn(state)
        assert "5.0" in result["observation"]


class TestMakeBdiReviseNode:
    def test_fallback_to_standard_updater(self) -> None:
        """When BDI doesn't trigger, falls back to standard updater."""
        bdi = _mock_bdi_agent()
        node_fn = make_bdi_revise_node(bdi)

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

    def test_bdi_desire_reconsideration_triggers_revision(self) -> None:
        """When BDI desires change, uses BDI goal revision."""
        revised_goal = Goal(name="revised")

        desires_before = [(1, "desire_a")]
        desires_after = [(1, "desire_b")]

        bdi = _mock_bdi_agent(desires=desires_before)

        call_count = [0]
        def reconsider(s, e):
            call_count[0] += 1
            bdi.desires = desires_after

        bdi.reconsider = reconsider
        bdi.revise_goal = lambda s, e: revised_goal

        node_fn = make_bdi_revise_node(bdi)

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
        assert any(e.get("source") == "bdi_desire_reconsideration" for e in events)

    def test_updates_beliefs_with_eval(self) -> None:
        bdi = _mock_bdi_agent()
        node_fn = make_bdi_revise_node(bdi)

        state = {
            "goal_updater": MagicMock(update=MagicMock(return_value=None)),
            "goal": Goal(name="test"),
            "state_snapshot": StateSnapshot(values=(1.0,), timestamp=0.0),
            "eval_signal": EvalSignal(score=0.7, confidence=0.9),
            "step": 1,
        }

        node_fn(state)
        assert bdi.beliefs["eval_score"] == 0.7
        assert bdi.beliefs["eval_confidence"] == 0.9


class TestMakeBdiPlanNode:
    def test_reuses_bdi_intention(self) -> None:
        """When BDI has valid intention, reuse it."""
        policy = PolicySpec(actions=(ActionSpec(name="act1"),))
        bdi = _mock_bdi_agent(intentions=policy)
        node_fn = make_bdi_plan_node(bdi)

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
        """When BDI has no intention, falls back to planner."""
        bdi = _mock_bdi_agent(intentions=None)
        node_fn = make_bdi_plan_node(bdi)

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
        # BDI intention should be updated
        assert bdi.intentions is new_policy

    def test_empty_intention_falls_back(self) -> None:
        """Empty action list in intention triggers fallback."""
        bdi = _mock_bdi_agent(intentions=PolicySpec(actions=()))
        node_fn = make_bdi_plan_node(bdi)

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


class TestBuildBdiTeleologicalGraph:
    def test_compiles(self) -> None:
        bdi = _mock_bdi_agent()
        graph = build_bdi_teleological_graph(bdi)
        assert graph is not None

    def test_compiles_with_interrupt(self) -> None:
        bdi = _mock_bdi_agent()
        graph = build_bdi_teleological_graph(
            bdi,
            interrupt_before=["act"],
        )
        assert graph is not None


class TestBenchmarkUpgrades:
    """Verify benchmark enhancement params."""

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
