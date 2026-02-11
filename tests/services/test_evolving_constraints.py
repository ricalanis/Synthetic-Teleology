"""Tests for evolving constraints and intentional grounding."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.enums import GoalOrigin
from synthetic_teleology.services.evolving_constraints import (
    ConstraintEvolution,
    EvolutionResult,
    EvolvingConstraintManager,
)
from synthetic_teleology.services.goal_grounding import (
    ExternalDirective,
    GoalSource,
    IntentionalGroundingManager,
)


def _mock_model(responses: list[str]) -> MagicMock:
    model = MagicMock()
    call_count = [0]

    def invoke(prompt):
        idx = min(call_count[0], len(responses) - 1)
        call_count[0] += 1
        msg = MagicMock()
        msg.content = responses[idx]
        return msg

    model.invoke = invoke
    return model


# ===================================================================== #
#  EvolvingConstraintManager tests                                       #
# ===================================================================== #


class TestConstraintEvolution:
    def test_defaults(self) -> None:
        ev = ConstraintEvolution(evolution_type="add", constraint_text="be safe")
        assert ev.evolution_type == "add"
        assert ev.confidence == 0.5

    def test_custom_fields(self) -> None:
        ev = ConstraintEvolution(
            evolution_type="modify",
            constraint_text="new constraint",
            previous_constraint="old constraint",
            confidence=0.9,
        )
        assert ev.previous_constraint == "old constraint"


class TestEvolvingConstraintManager:
    def test_initial_constraints(self) -> None:
        model = _mock_model(["{}"])
        mgr = EvolvingConstraintManager(model, initial_constraints=["c1", "c2"])
        assert mgr.constraints == ["c1", "c2"]

    def test_should_evolve_frequency(self) -> None:
        model = _mock_model(["{}"])
        mgr = EvolvingConstraintManager(model, evolution_frequency=3)
        assert not mgr.should_evolve()  # round 0
        mgr._round_counter = 1
        assert not mgr.should_evolve()
        mgr._round_counter = 3
        assert mgr.should_evolve()
        mgr._round_counter = 6
        assert mgr.should_evolve()

    def test_record_violations(self) -> None:
        model = _mock_model(["{}"])
        mgr = EvolvingConstraintManager(model)
        mgr.record_violations(["v1", "v2"], {"step": 1})
        assert len(mgr._violation_history) == 1

    def test_step_triggers_evolution(self) -> None:
        resp = json.dumps({
            "evolutions": [
                {"type": "add", "constraint": "new rule", "reasoning": "needed", "confidence": 0.8}
            ],
            "overall_reasoning": "improvement",
        })
        model = _mock_model([resp])
        mgr = EvolvingConstraintManager(model, evolution_frequency=2)

        # Step 1: no evolution
        result = mgr.step()
        assert result is None

        # Step 2: should evolve
        result = mgr.step()
        assert result is not None
        assert len(result.new_constraints) == 1
        assert "new rule" in mgr.constraints

    def test_add_constraint(self) -> None:
        resp = json.dumps({
            "evolutions": [
                {"type": "add", "constraint": "be ethical",
                 "reasoning": "important", "confidence": 0.9}
            ],
            "overall_reasoning": "ethics",
        })
        model = _mock_model([resp])
        mgr = EvolvingConstraintManager(model)
        result = mgr.evolve()
        assert "be ethical" in mgr.constraints
        assert len(result.evolutions) == 1

    def test_remove_constraint(self) -> None:
        resp = json.dumps({
            "evolutions": [
                {"type": "remove", "constraint": "obsolete",
                 "reasoning": "no longer needed", "confidence": 0.7}
            ],
            "overall_reasoning": "cleanup",
        })
        model = _mock_model([resp])
        mgr = EvolvingConstraintManager(model, initial_constraints=["obsolete", "keep"])
        result = mgr.evolve()
        assert "obsolete" not in mgr.constraints
        assert "keep" in mgr.constraints
        assert len(result.removed_constraints) == 1

    def test_modify_constraint(self) -> None:
        resp = json.dumps({
            "evolutions": [{
                "type": "modify",
                "constraint": "be very safe",
                "previous": "be safe",
                "reasoning": "strengthen",
                "confidence": 0.8,
            }],
            "overall_reasoning": "refinement",
        })
        model = _mock_model([resp])
        mgr = EvolvingConstraintManager(model, initial_constraints=["be safe"])
        result = mgr.evolve()
        assert "be very safe" in mgr.constraints
        assert "be safe" not in mgr.constraints
        assert len(result.modified_constraints) == 1

    def test_max_constraints_limit(self) -> None:
        resp = json.dumps({
            "evolutions": [
                {"type": "add", "constraint": "extra", "reasoning": "test", "confidence": 0.5}
            ],
            "overall_reasoning": "test",
        })
        model = _mock_model([resp])
        mgr = EvolvingConstraintManager(model, max_constraints=2, initial_constraints=["c1", "c2"])
        mgr.evolve()
        assert len(mgr.constraints) == 2  # capped
        assert "extra" not in mgr.constraints

    def test_evolution_failure_graceful(self) -> None:
        model = MagicMock()
        model.invoke = MagicMock(side_effect=Exception("LLM error"))
        mgr = EvolvingConstraintManager(model)
        result = mgr.evolve()
        assert isinstance(result, EvolutionResult)
        assert "failed" in result.reasoning.lower()

    def test_evolution_history(self) -> None:
        resp = json.dumps({"evolutions": [], "overall_reasoning": "nothing"})
        model = _mock_model([resp])
        mgr = EvolvingConstraintManager(model)
        mgr.evolve()
        assert len(mgr.evolution_history) == 1


# ===================================================================== #
#  IntentionalGroundingManager tests                                     #
# ===================================================================== #


class TestExternalDirective:
    def test_defaults(self) -> None:
        d = ExternalDirective(source=GoalSource.USER, content="do X")
        assert d.source == GoalSource.USER
        assert d.priority == 0.5

    def test_goal_source_enum(self) -> None:
        assert GoalSource.USER.value == "user"
        assert GoalSource.NORMATIVE.value == "normative"
        assert GoalSource.NEGOTIATED.value == "negotiated"


class TestIntentionalGroundingManager:
    def test_add_directive(self) -> None:
        mgr = IntentionalGroundingManager()
        mgr.add_directive(ExternalDirective(source=GoalSource.USER, content="go fast"))
        assert len(mgr.directives) == 1

    def test_convenience_methods(self) -> None:
        mgr = IntentionalGroundingManager()
        mgr.add_user_directive("be fast")
        mgr.add_normative_directive("be safe")
        mgr.add_negotiated_directive("balance speed and safety")
        assert len(mgr.directives) == 3
        assert mgr.directives[0].priority == 0.8
        assert mgr.directives[1].priority == 0.6

    def test_max_directives_limit(self) -> None:
        mgr = IntentionalGroundingManager(max_directives=3)
        for i in range(5):
            mgr.add_user_directive(f"directive {i}")
        assert len(mgr.directives) == 3

    def test_no_directives_returns_none(self) -> None:
        mgr = IntentionalGroundingManager()
        goal = Goal(name="test", description="test goal")
        result = mgr.ground(goal)
        assert result is None

    def test_rule_based_grounding_no_model(self) -> None:
        mgr = IntentionalGroundingManager(grounding_threshold=0.5)
        mgr.add_user_directive("prioritize safety", priority=0.9)
        goal = Goal(name="test", description="test goal")
        result = mgr.ground(goal)
        assert result is not None
        assert "safety" in result.description
        assert result.provenance is not None
        assert result.provenance.origin == GoalOrigin.ENDOGENOUS

    def test_rule_based_grounding_below_threshold(self) -> None:
        mgr = IntentionalGroundingManager(grounding_threshold=0.8)
        mgr.add_directive(ExternalDirective(
            source=GoalSource.ENVIRONMENTAL, content="low priority", priority=0.3,
        ))
        goal = Goal(name="test", description="test goal")
        result = mgr.ground(goal)
        assert result is None

    def test_llm_grounding(self) -> None:
        resp = json.dumps({
            "should_ground": True,
            "adjusted_description": "test goal with safety focus",
            "adjusted_criteria": ["be safe"],
            "influence_score": 0.8,
            "reasoning": "safety directive is high priority",
        })
        model = _mock_model([resp])
        mgr = IntentionalGroundingManager(model=model, grounding_threshold=0.5)
        mgr.add_user_directive("prioritize safety")

        goal = Goal(name="test", description="test goal")
        result = mgr.ground(goal)
        assert result is not None
        assert "safety" in result.description
        assert result.goal_id != goal.goal_id

    def test_llm_grounding_no_ground(self) -> None:
        resp = json.dumps({
            "should_ground": False,
            "influence_score": 0.2,
            "reasoning": "no significant influence",
        })
        model = _mock_model([resp])
        mgr = IntentionalGroundingManager(model=model)
        mgr.add_directive(ExternalDirective(
            source=GoalSource.ENVIRONMENTAL, content="minor", priority=0.2,
        ))

        goal = Goal(name="test", description="test goal")
        result = mgr.ground(goal)
        assert result is None

    def test_llm_grounding_failure_fallback(self) -> None:
        model = MagicMock()
        model.invoke = MagicMock(side_effect=Exception("LLM error"))
        mgr = IntentionalGroundingManager(model=model, grounding_threshold=0.5)
        mgr.add_user_directive("be safe", priority=0.9)

        goal = Goal(name="test", description="test goal")
        result = mgr.ground(goal)
        # Falls back to rule-based, which should trigger
        assert result is not None

    def test_clear_directives(self) -> None:
        mgr = IntentionalGroundingManager()
        mgr.add_user_directive("test")
        mgr.clear_directives()
        assert len(mgr.directives) == 0

    def test_grounding_history_tracked(self) -> None:
        mgr = IntentionalGroundingManager(grounding_threshold=0.5)
        mgr.add_user_directive("be safe", priority=0.9)
        goal = Goal(name="test", description="test goal")
        mgr.ground(goal)
        assert len(mgr.grounding_history) == 1
