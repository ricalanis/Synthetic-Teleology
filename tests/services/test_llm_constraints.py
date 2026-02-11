"""Tests for LLMConstraintChecker with mocked LLM."""

from __future__ import annotations

from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.values import ActionSpec, StateSnapshot
from synthetic_teleology.services.llm_constraints import (
    ConstraintAssessment,
    ConstraintCheckOutput,
    LLMConstraintChecker,
)
from tests.helpers.mock_llm import MockStructuredChatModel


class TestLLMConstraintChecker:

    def test_all_constraints_satisfied(self) -> None:
        model = MockStructuredChatModel(structured_responses=[
            ConstraintCheckOutput(
                assessments=[
                    ConstraintAssessment(
                        constraint_name="budget",
                        is_satisfied=True,
                        severity=0.0,
                        reasoning="Within budget",
                    ),
                ],
                overall_safe=True,
                overall_reasoning="All constraints met",
            )
        ])
        checker = LLMConstraintChecker(
            model=model, constraints=["Stay within $10k budget"]
        )
        goal = Goal(name="g", description="test")
        state = StateSnapshot(timestamp=0.0, observation="Current spend: $5k")

        passed, message = checker.check(goal, state)
        assert passed is True
        assert message == ""

    def test_constraint_violated(self) -> None:
        model = MockStructuredChatModel(structured_responses=[
            ConstraintCheckOutput(
                assessments=[
                    ConstraintAssessment(
                        constraint_name="budget",
                        is_satisfied=False,
                        severity=0.8,
                        reasoning="Over budget by $2k",
                        suggested_mitigation="Reduce spending",
                    ),
                ],
                overall_safe=False,
                overall_reasoning="Budget constraint violated",
            )
        ])
        checker = LLMConstraintChecker(
            model=model, constraints=["Stay within $10k budget"]
        )
        goal = Goal(name="g", description="test")
        state = StateSnapshot(timestamp=0.0, observation="Current spend: $12k")

        passed, message = checker.check(goal, state)
        assert passed is False
        assert "budget" in message.lower()
        assert "severity=0.8" in message

    def test_check_with_action(self) -> None:
        model = MockStructuredChatModel(structured_responses=[
            ConstraintCheckOutput(
                assessments=[
                    ConstraintAssessment(
                        constraint_name="no_weekends",
                        is_satisfied=True,
                        severity=0.0,
                        reasoning="Action is weekday",
                    ),
                ],
                overall_safe=True,
                overall_reasoning="Safe",
            )
        ])
        checker = LLMConstraintChecker(
            model=model, constraints=["No weekend actions"]
        )
        goal = Goal(name="g", description="test")
        state = StateSnapshot(timestamp=0.0, observation="Monday")
        action = ActionSpec(name="send_email", description="Send report email")

        passed, _ = checker.check(goal, state, action=action)
        assert passed is True

    def test_multiple_constraints(self) -> None:
        model = MockStructuredChatModel(structured_responses=[
            ConstraintCheckOutput(
                assessments=[
                    ConstraintAssessment(
                        constraint_name="budget",
                        is_satisfied=True,
                        severity=0.0,
                        reasoning="OK",
                    ),
                    ConstraintAssessment(
                        constraint_name="ethics",
                        is_satisfied=False,
                        severity=0.9,
                        reasoning="Potential privacy concern",
                    ),
                ],
                overall_safe=False,
                overall_reasoning="Ethics constraint violated",
            )
        ])
        checker = LLMConstraintChecker(
            model=model,
            constraints=["Stay within budget", "Respect user privacy"],
        )
        goal = Goal(name="g", description="test")
        state = StateSnapshot(timestamp=0.0, observation="state")

        passed, message = checker.check(goal, state)
        assert passed is False
        assert "ethics" in message.lower()

    def test_check_detailed_returns_full_output(self) -> None:
        expected = ConstraintCheckOutput(
            assessments=[
                ConstraintAssessment(
                    constraint_name="budget",
                    is_satisfied=True,
                    severity=0.0,
                    reasoning="OK",
                ),
            ],
            overall_safe=True,
            overall_reasoning="All good",
        )
        model = MockStructuredChatModel(structured_responses=[expected])
        checker = LLMConstraintChecker(
            model=model, constraints=["Stay within budget"]
        )
        goal = Goal(name="g", description="test")
        state = StateSnapshot(timestamp=0.0, observation="state")

        result = checker.check_detailed(goal, state)
        assert result is not None
        assert result.overall_safe is True
        assert len(result.assessments) == 1
