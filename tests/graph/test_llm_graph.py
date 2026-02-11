"""Tests for the full teleological graph in LLM mode with mocked models.

These tests verify that the complete graph runs end-to-end using mocked
LLM services, ensuring the LLM mode integration works without real API calls.
"""

from __future__ import annotations

from synthetic_teleology.graph.builder import GraphBuilder
from synthetic_teleology.graph.prebuilt import create_llm_agent
from synthetic_teleology.services.llm_evaluation import EvaluationOutput, LLMEvaluator
from synthetic_teleology.services.llm_planning import (
    ActionProposal,
    LLMPlanner,
    PlanHypothesis,
    PlanningOutput,
)
from synthetic_teleology.services.llm_revision import LLMReviser, RevisionOutput
from tests.helpers.mock_llm import MockStructuredChatModel


def _eval_output(score: float = 0.5) -> EvaluationOutput:
    return EvaluationOutput(
        score=score,
        confidence=0.8,
        reasoning=f"Score {score}",
    )


def _plan_output() -> PlanningOutput:
    return PlanningOutput(
        hypotheses=[
            PlanHypothesis(
                actions=[ActionProposal(name="action_1", description="Do something")],
                reasoning="Only viable plan",
                expected_outcome="Progress",
                confidence=0.7,
            ),
        ],
        selected_index=0,
        selection_reasoning="Best option",
    )


def _revision_output(should_revise: bool = False) -> RevisionOutput:
    return RevisionOutput(
        should_revise=should_revise,
        reasoning="Goal is fine" if not should_revise else "Need to adjust",
    )


class TestLLMGraphIntegration:

    def _make_services(self, eval_score: float = 0.5, n: int = 50):
        """Create mocked LLM services for graph testing."""
        eval_model = MockStructuredChatModel(
            structured_responses=[_eval_output(eval_score)] * n
        )
        plan_model = MockStructuredChatModel(
            structured_responses=[_plan_output()] * n
        )
        rev_model = MockStructuredChatModel(
            structured_responses=[_revision_output()] * n
        )

        return (
            LLMEvaluator(model=eval_model),
            LLMPlanner(model=plan_model, num_hypotheses=1),
            LLMReviser(model=rev_model),
        )

    def test_graph_runs_to_max_steps(self) -> None:
        evaluator, planner, reviser = self._make_services(eval_score=0.3)

        app, state = (
            GraphBuilder("llm-test")
            .with_model(MockStructuredChatModel(structured_responses=[_eval_output()]))
            .with_goal("Test goal", criteria=["test criterion"])
            .with_evaluator(evaluator)
            .with_planner(planner)
            .with_goal_updater(reviser)
            .with_max_steps(3)
            .build()
        )

        result = app.invoke(state)
        assert result["step"] == 3
        assert len(result["events"]) > 0

    def test_graph_stops_on_goal_achieved(self) -> None:
        evaluator, planner, reviser = self._make_services(eval_score=0.95)

        app, state = (
            GraphBuilder("achieved-test")
            .with_model(MockStructuredChatModel(structured_responses=[_eval_output()]))
            .with_goal("Easy goal")
            .with_evaluator(evaluator)
            .with_planner(planner)
            .with_goal_updater(reviser)
            .with_max_steps(20)
            .with_goal_achieved_threshold(0.9)
            .build()
        )

        result = app.invoke(state)
        assert result["step"] <= 20
        assert result.get("stop_reason") == "goal_achieved"

    def test_graph_produces_reasoning_trace(self) -> None:
        evaluator, planner, reviser = self._make_services(eval_score=0.5)

        app, state = (
            GraphBuilder("trace-test")
            .with_model(MockStructuredChatModel(structured_responses=[_eval_output()]))
            .with_goal("Goal with trace")
            .with_evaluator(evaluator)
            .with_planner(planner)
            .with_goal_updater(reviser)
            .with_max_steps(2)
            .build()
        )

        result = app.invoke(state)
        assert "reasoning_trace" in result
        assert len(result["reasoning_trace"]) > 0

    def test_graph_records_eval_history(self) -> None:
        evaluator, planner, reviser = self._make_services(eval_score=0.4)

        app, state = (
            GraphBuilder("history-test")
            .with_model(MockStructuredChatModel(structured_responses=[_eval_output()]))
            .with_goal("History goal")
            .with_evaluator(evaluator)
            .with_planner(planner)
            .with_goal_updater(reviser)
            .with_max_steps(3)
            .build()
        )

        result = app.invoke(state)
        assert len(result["eval_history"]) == 3

    def test_create_llm_agent_prebuilt(self) -> None:
        evaluator, planner, reviser = self._make_services(eval_score=0.3)

        app, state = create_llm_agent(
            model=MockStructuredChatModel(structured_responses=[_eval_output()]),
            goal="Prebuilt goal",
            max_steps=2,
        )

        state["evaluator"] = evaluator
        state["planner"] = planner
        state["goal_updater"] = reviser

        result = app.invoke(state)
        assert result["step"] == 2


class TestReasoningQualityMetric:

    def test_reasoning_quality_with_llm_log(self) -> None:
        from synthetic_teleology.measurement.collector import AgentLog, AgentLogEntry
        from synthetic_teleology.measurement.metrics import ReasoningQuality

        log = AgentLog(agent_id="llm-agent")
        log.entries.append(
            AgentLogEntry(
                step=0,
                timestamp=1.0,
                eval_score=0.3,
                reasoning="Initial assessment shows partial progress toward goal",
                hypotheses_count=3,
            )
        )
        log.entries.append(
            AgentLogEntry(
                step=1,
                timestamp=2.0,
                eval_score=0.5,
                reasoning="Market data indicates improving trends after strategy change",
                hypotheses_count=3,
            )
        )
        log.entries.append(
            AgentLogEntry(
                step=2,
                timestamp=3.0,
                eval_score=0.7,
                reasoning="Significant progress achieved through targeted actions",
                hypotheses_count=2,
            )
        )

        metric = ReasoningQuality()
        result = metric.compute(log)

        assert result.name == "reasoning_quality"
        assert 0.0 < result.value <= 1.0

    def test_reasoning_quality_empty_reasoning(self) -> None:
        from synthetic_teleology.measurement.collector import AgentLog, AgentLogEntry
        from synthetic_teleology.measurement.metrics import ReasoningQuality

        log = AgentLog(agent_id="numeric-agent")
        log.entries.append(AgentLogEntry(step=0, timestamp=1.0, eval_score=0.3))
        log.entries.append(AgentLogEntry(step=1, timestamp=2.0, eval_score=0.5))

        metric = ReasoningQuality()
        result = metric.compute(log)

        assert result.value == 0.0
