"""LLM-based evaluation strategy using LangChain structured output.

Replaces ``NumericEvaluator`` as the default in LLM mode.  Uses
``model.with_structured_output()`` for reliable parsing of the LLM's
assessment into an ``EvalSignal``.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.values import EvalSignal, StateSnapshot
from synthetic_teleology.services.evaluation import BaseEvaluator

logger = logging.getLogger(__name__)

# -- Structured output schemas -----------------------------------------------


class EvaluationOutput(BaseModel):
    """Structured output schema for LLM evaluation."""

    score: float = Field(ge=-1, le=1, description="Overall goal alignment score [-1, 1]")
    confidence: float = Field(ge=0, le=1, description="Confidence in the assessment [0, 1]")
    reasoning: str = Field(description="Chain-of-thought reasoning for the score")
    criteria_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Per-criterion scores (criterion name -> score in [-1, 1])",
    )


# -- Prompt ------------------------------------------------------------------

_EVALUATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an evaluation critic for a goal-directed AI agent. "
            "Assess how well the current state aligns with the goal. "
            "Consider each success criterion individually, then produce an "
            "overall score.\n\n"
            "Scoring guide:\n"
            "  1.0 = goal fully achieved\n"
            "  0.0 = neutral / no progress\n"
            " -1.0 = maximum regression from goal\n\n"
            "Be precise and quantitative where possible.",
        ),
        (
            "human",
            "## Goal\n"
            "**Name**: {goal_name}\n"
            "**Description**: {goal_description}\n"
            "**Success Criteria**: {success_criteria}\n\n"
            "## Current State\n"
            "**Observation**: {observation}\n"
            "**Values**: {state_values}\n"
            "**Context**: {context}\n\n"
            "Evaluate the current state against the goal and criteria.",
        ),
    ]
)


# -- LLMEvaluator -----------------------------------------------------------


class LLMEvaluator(BaseEvaluator):
    """LLM-based evaluation using structured output.

    Uses ``BaseChatModel.with_structured_output(EvaluationOutput)`` to get
    reliable, typed responses from the LLM.

    Parameters
    ----------
    model:
        A LangChain chat model (e.g. ``ChatAnthropic``, ``ChatOpenAI``).
    criteria:
        Optional additional evaluation criteria (merged with goal criteria).
    prompt:
        Optional custom ``ChatPromptTemplate`` to replace the default.
    """

    def __init__(
        self,
        model: BaseChatModel,
        criteria: list[str] | None = None,
        prompt: ChatPromptTemplate | None = None,
    ) -> None:
        self.model = model
        self.criteria = criteria or []
        self._prompt = prompt or _EVALUATION_PROMPT
        self._chain = self._build_chain()

    def _build_chain(self) -> Any:
        """Build the evaluation chain with structured output."""
        structured_model = self.model.with_structured_output(EvaluationOutput)
        return self._prompt | structured_model

    def validate(self, goal: Goal, state: StateSnapshot) -> bool:
        """LLM evaluator can handle any goal with a description."""
        return bool(goal.description or goal.name or goal.objective is not None)

    def evaluate(self, goal: Goal, state: StateSnapshot) -> EvalSignal:
        """Evaluate goal alignment using the LLM.

        Returns an ``EvalSignal`` with reasoning and per-criterion scores.
        Falls back to a neutral signal on error.
        """
        # Merge goal criteria with evaluator criteria
        all_criteria = list(goal.success_criteria) + self.criteria

        # Build observation text
        observation = state.observation or (
            f"State values: {state.values}" if state.values else "No observation available"
        )

        try:
            result: EvaluationOutput = self._chain.invoke(
                {
                    "goal_name": goal.name or goal.goal_id,
                    "goal_description": goal.description or str(goal.objective),
                    "success_criteria": (
                        "\n".join(f"- {c}" for c in all_criteria)
                        if all_criteria
                        else "No specific criteria defined"
                    ),
                    "observation": observation,
                    "state_values": str(state.values) if state.values else "N/A",
                    "context": str(dict(state.context)) if state.context else "N/A",
                }
            )

            return EvalSignal(
                score=max(-1.0, min(1.0, result.score)),
                confidence=max(0.0, min(1.0, result.confidence)),
                reasoning=result.reasoning,
                criteria_scores=result.criteria_scores,
                explanation=f"LLMEvaluator: {result.reasoning[:200]}",
            )

        except Exception as exc:
            logger.warning("LLMEvaluator: evaluation failed: %s", exc)
            return EvalSignal(
                score=0.0,
                confidence=0.1,
                explanation=f"LLMEvaluator: error: {exc}",
            )
