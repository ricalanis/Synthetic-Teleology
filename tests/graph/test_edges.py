"""Tests for conditional edge functions."""

from __future__ import annotations

from synthetic_teleology.domain.values import EvalSignal
from synthetic_teleology.graph.edges import should_continue, should_revise


class TestShouldContinue:

    def test_continue_when_no_stop_reason(self) -> None:
        state = {"stop_reason": None}
        assert should_continue(state) == "perceive"

    def test_continue_when_stop_reason_missing(self) -> None:
        state: dict = {}
        assert should_continue(state) == "perceive"

    def test_end_when_stop_reason_set(self) -> None:
        state = {"stop_reason": "max_steps"}
        assert should_continue(state) == "__end__"

    def test_end_on_goal_achieved(self) -> None:
        state = {"stop_reason": "goal_achieved"}
        assert should_continue(state) == "__end__"


class TestShouldRevise:

    def test_no_revise_on_high_positive_score(self) -> None:
        """Good scores should NOT trigger revision (paper alignment)."""
        state = {"eval_signal": EvalSignal(score=0.8, confidence=0.9)}
        assert should_revise(state) == "check_constraints"

    def test_revise_on_negative_score(self) -> None:
        state = {"eval_signal": EvalSignal(score=-0.5, confidence=0.9)}
        assert should_revise(state) == "revise"

    def test_skip_revise_on_moderate_score(self) -> None:
        state = {"eval_signal": EvalSignal(score=0.3, confidence=0.9)}
        assert should_revise(state) == "check_constraints"

    def test_skip_revise_when_no_signal(self) -> None:
        state: dict = {}
        assert should_revise(state) == "check_constraints"

    def test_no_revise_at_old_boundary(self) -> None:
        """score=0.5 is good performance — should not revise."""
        state = {"eval_signal": EvalSignal(score=0.5, confidence=0.9)}
        assert should_revise(state) == "check_constraints"

    def test_revise_at_negative_boundary(self) -> None:
        """Exactly -0.3 should trigger revision."""
        state = {"eval_signal": EvalSignal(score=-0.3, confidence=0.9)}
        assert should_revise(state) == "revise"

    def test_no_revise_just_above_negative_boundary(self) -> None:
        state = {"eval_signal": EvalSignal(score=-0.29, confidence=0.9)}
        assert should_revise(state) == "check_constraints"

    def test_revise_on_very_negative_score(self) -> None:
        state = {"eval_signal": EvalSignal(score=-0.4, confidence=0.9)}
        assert should_revise(state) == "revise"
