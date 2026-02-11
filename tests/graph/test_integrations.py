"""Tests for wired integrations: EvolvingConstraintManager and KnowledgeStore."""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock

from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.values import (
    ActionSpec,
    EvalSignal,
    StateSnapshot,
)
from synthetic_teleology.graph.nodes import (
    _build_enriched_observation,
    evolve_constraints_node,
    reflect_node,
)
from synthetic_teleology.infrastructure.knowledge_store import KnowledgeStore
from synthetic_teleology.services.evolving_constraints import (
    EvolutionResult,
    EvolvingConstraintManager,
)


def _make_env(values: tuple[float, ...] = (1.0, 1.0)):
    state_values = list(values)

    def perceive() -> StateSnapshot:
        return StateSnapshot(timestamp=time.time(), values=tuple(state_values))

    def transition(action: ActionSpec) -> None:
        effect = action.parameters.get("effect")
        if effect and isinstance(effect, (list, tuple)):
            for i in range(min(len(state_values), len(effect))):
                state_values[i] += effect[i]

    return perceive, transition


class TestEvolvingConstraintsWired:

    def test_evolving_constraints_wired(self) -> None:
        """Builder with evolving constraints passes to graph, node calls step()."""
        # Create a mock manager
        manager = MagicMock(spec=EvolvingConstraintManager)
        manager.step.return_value = None

        state: dict[str, Any] = {
            "step": 3,
            "evolving_constraint_manager": manager,
            "constraint_violations": ["violation A"],
        }

        evolve_constraints_node(state)

        manager.record_violations.assert_called_once_with(["violation A"])
        manager.step.assert_called_once()


class TestEvolvingConstraintsRecordsViolations:

    def test_evolving_constraints_records_violations(self) -> None:
        """Violations are recorded before step is called."""
        manager = MagicMock(spec=EvolvingConstraintManager)
        evolution = EvolutionResult(
            reasoning="Added new constraint",
            new_constraints=["Stay focused"],
        )
        manager.step.return_value = evolution

        state: dict[str, Any] = {
            "step": 6,
            "evolving_constraint_manager": manager,
            "constraint_violations": ["v1", "v2"],
        }

        result = evolve_constraints_node(state)

        manager.record_violations.assert_called_once_with(["v1", "v2"])
        assert len(result["reasoning_trace"]) == 1
        assert result["reasoning_trace"][0]["reasoning"] == "Added new constraint"


class TestKnowledgeStoreInEnrichment:

    def test_knowledge_store_in_enrichment(self) -> None:
        """Observation includes knowledge entries when store is present."""
        store = KnowledgeStore()
        store.put("insight_1", "Market is growing", source="evaluator", tags=("market",))

        state: dict[str, Any] = {
            "step": 3,
            "knowledge_store": store,
            "action_feedback": [],
            "eval_history": [],
            "goal_history": [],
        }

        enriched = _build_enriched_observation("Base obs", state)

        assert "Knowledge store" in enriched
        assert "insight_1" in enriched

    def test_knowledge_store_absent_no_error(self) -> None:
        """Without knowledge store, enrichment works normally."""
        state: dict[str, Any] = {
            "step": 3,
            "action_feedback": [],
            "eval_history": [],
            "goal_history": [],
        }

        enriched = _build_enriched_observation("Base obs", state)
        assert enriched == "Base obs"


class TestReflectWritesKnowledge:

    def test_reflect_writes_knowledge(self) -> None:
        """reflect_node populates knowledge store with reflection data."""
        store = KnowledgeStore()
        goal = Goal(name="test-goal")

        state: dict[str, Any] = {
            "step": 5,
            "goal": goal,
            "eval_signal": EvalSignal(score=0.7, confidence=0.9),
            "max_steps": 100,
            "goal_achieved_threshold": 0.9,
            "knowledge_store": store,
        }

        reflect_node(state)

        assert "reflection_step_5" in store
        entry = store.get("reflection_step_5")
        assert entry is not None
        assert entry.value["eval_score"] == 0.7
        assert entry.source == "reflect_node"
