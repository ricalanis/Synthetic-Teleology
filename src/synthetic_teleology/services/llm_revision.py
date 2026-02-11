"""LLM-based goal revision strategy using LangChain structured output.

The LLM reasons about whether the goal should be revised based on evaluation
feedback, and if so, proposes a new description and/or success criteria.
"""

from __future__ import annotations

import concurrent.futures
import logging
from typing import TYPE_CHECKING, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.enums import RevisionReason
from synthetic_teleology.domain.values import EvalSignal, StateSnapshot
from synthetic_teleology.services.goal_revision import BaseGoalUpdater

if TYPE_CHECKING:
    from synthetic_teleology.domain.aggregates import ConstraintSet

logger = logging.getLogger(__name__)


# -- Structured output schemas -----------------------------------------------


class RevisionOutput(BaseModel):
    """Structured output schema for goal revision."""

    should_revise: bool = Field(description="Whether the goal should be revised")
    reasoning: str = Field(description="Reasoning for the decision")
    new_description: str | None = Field(
        default=None,
        description="Revised goal description (if revising)",
    )
    new_criteria: list[str] | None = Field(
        default=None,
        description="Revised success criteria (if revising)",
    )
    new_values: list[float] | None = Field(
        default=None,
        description="Revised numeric objective values (if applicable)",
    )


# -- Prompt ------------------------------------------------------------------

_REVISION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a goal-revision expert for a goal-directed AI agent. "
            "Analyze the evaluation feedback and determine whether the goal "
            "should be revised.\n\n"
            "Reasons to revise:\n"
            "- The goal is unrealistic given the current state\n"
            "- The success criteria are misaligned with outcomes\n"
            "- The environment has changed significantly\n"
            "- Persistent low scores suggest a fundamental mismatch\n\n"
            "Reasons NOT to revise:\n"
            "- Progress is being made (positive scores)\n"
            "- Low scores are temporary or expected early on\n"
            "- The goal is well-specified and achievable\n\n"
            "If revising, propose concrete changes. Keep the core intent.",
        ),
        (
            "human",
            "## Current Goal\n"
            "**Name**: {goal_name}\n"
            "**Description**: {goal_description}\n"
            "**Success Criteria**: {success_criteria}\n"
            "**Version**: {version}\n\n"
            "## Evaluation Feedback\n"
            "**Score**: {eval_score}\n"
            "**Confidence**: {eval_confidence}\n"
            "**Reasoning**: {eval_reasoning}\n"
            "**Criteria Scores**: {criteria_scores}\n\n"
            "## Current State\n"
            "**Observation**: {observation}\n"
            "**Values**: {state_values}\n\n"
            "Should this goal be revised? If so, propose specific changes.",
        ),
    ]
)


# -- LLMReviser -------------------------------------------------------------


class LLMReviser(BaseGoalUpdater):
    """LLM-based goal revision strategy.

    Uses the LLM to reason about whether the goal should be revised based on
    evaluation feedback, and produces a revised goal if warranted.

    Parameters
    ----------
    model:
        A LangChain chat model.
    prompt:
        Optional custom ``ChatPromptTemplate``.
    """

    def __init__(
        self,
        model: BaseChatModel,
        prompt: ChatPromptTemplate | None = None,
        timeout: float | None = None,
    ) -> None:
        self.model = model
        self._prompt = prompt or _REVISION_PROMPT
        self._timeout = timeout
        self._chain = self._build_chain()

    def _build_chain(self) -> Any:
        """Build the revision chain with structured output."""
        structured_model = self.model.with_structured_output(RevisionOutput)
        return self._prompt | structured_model

    def _invoke_with_timeout(self, inputs: dict[str, Any]) -> Any:
        """Invoke the chain with optional timeout."""
        if self._timeout is None:
            return self._chain.invoke(inputs)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self._chain.invoke, inputs)
            return future.result(timeout=self._timeout)

    def update(
        self,
        goal: Goal,
        state: StateSnapshot,
        eval_signal: EvalSignal,
        constraints: ConstraintSet | None = None,
    ) -> Goal | None:
        """Use the LLM to decide whether and how to revise the goal.

        Returns a revised ``Goal`` or ``None`` if no revision is needed.
        """
        observation = state.observation or (
            f"State values: {state.values}" if state.values else "No observation"
        )

        try:
            result: RevisionOutput = self._invoke_with_timeout(
                {
                    "goal_name": goal.name or goal.goal_id,
                    "goal_description": goal.description or str(goal.objective),
                    "success_criteria": (
                        "\n".join(f"- {c}" for c in goal.success_criteria)
                        if goal.success_criteria
                        else "No criteria defined"
                    ),
                    "version": goal.version,
                    "eval_score": f"{eval_signal.score:.4f}",
                    "eval_confidence": f"{eval_signal.confidence:.4f}",
                    "eval_reasoning": eval_signal.reasoning or eval_signal.explanation,
                    "criteria_scores": str(dict(eval_signal.criteria_scores)) or "N/A",
                    "observation": observation,
                    "state_values": str(state.values) if state.values else "N/A",
                }
            )

            if not result.should_revise:
                logger.debug("LLMReviser: no revision needed — %s", result.reasoning)
                return None

            # Build new objective if numeric values were proposed
            new_objective = None
            if (
                result.new_values
                and goal.objective is not None
                and len(result.new_values) == goal.objective.dimension
            ):
                    new_objective = goal.objective.with_values(
                        tuple(float(v) for v in result.new_values)
                    )

            new_goal, _ = goal.revise(
                new_objective=new_objective,
                reason=f"{RevisionReason.EVALUATION_FEEDBACK.value}: {result.reasoning[:200]}",
                eval_signal=eval_signal,
                new_description=result.new_description,
                new_criteria=result.new_criteria,
            )

            logger.info(
                "LLMReviser: revised goal %s -> %s: %s",
                goal.goal_id,
                new_goal.goal_id,
                result.reasoning[:100],
            )
            return new_goal

        except Exception as exc:
            logger.warning("LLMReviser: revision failed: %s", exc)
            return None
