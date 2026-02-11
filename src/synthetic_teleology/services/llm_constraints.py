"""LLM-based soft constraint checking using LangChain structured output.

Instead of boolean predicate checks, the LLM evaluates constraint compliance
with nuance, severity scores, and suggested mitigations.
"""

from __future__ import annotations

import concurrent.futures
import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.values import ActionSpec, StateSnapshot
from synthetic_teleology.services.constraint_engine import BaseConstraintChecker

logger = logging.getLogger(__name__)


# -- Structured output schemas -----------------------------------------------


class ConstraintAssessment(BaseModel):
    """Assessment of a single constraint."""

    constraint_name: str = Field(description="Name of the constraint")
    is_satisfied: bool = Field(description="Whether the constraint is satisfied")
    severity: float = Field(
        ge=0,
        le=1,
        description="Violation severity (0=trivial, 1=critical)",
    )
    reasoning: str = Field(description="Reasoning for the assessment")
    suggested_mitigation: str = Field(
        default="",
        description="How to mitigate the violation (if any)",
    )


class ConstraintCheckOutput(BaseModel):
    """Full constraint check output."""

    assessments: list[ConstraintAssessment] = Field(
        description="Per-constraint assessments"
    )
    overall_safe: bool = Field(description="Whether all constraints are satisfied")
    overall_reasoning: str = Field(description="Overall safety reasoning")


# -- Prompt ------------------------------------------------------------------

_CONSTRAINT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a constraint checker for a goal-directed AI agent. "
            "Evaluate whether the current state and proposed action comply "
            "with the given constraints.\n\n"
            "For each constraint:\n"
            "1. Assess whether it is satisfied\n"
            "2. Rate the severity of any violation (0=trivial, 1=critical)\n"
            "3. Explain your reasoning\n"
            "4. Suggest mitigations for violations\n\n"
            "Be thorough but fair. Minor deviations may be acceptable for "
            "soft constraints.",
        ),
        (
            "human",
            "## Constraints\n{constraints_list}\n\n"
            "## Current State\n"
            "**Goal**: {goal_description}\n"
            "**Observation**: {observation}\n"
            "**Values**: {state_values}\n\n"
            "## Proposed Action\n{action_info}\n\n"
            "Evaluate each constraint.",
        ),
    ]
)


# -- LLMConstraintChecker ---------------------------------------------------


class LLMConstraintChecker(BaseConstraintChecker):
    """LLM-based soft constraint checker.

    Evaluates natural language constraints using an LLM for nuanced
    assessment instead of rigid boolean predicates.

    Parameters
    ----------
    model:
        A LangChain chat model.
    constraints:
        Natural language constraint descriptions.
    prompt:
        Optional custom ``ChatPromptTemplate``.
    """

    def __init__(
        self,
        model: BaseChatModel,
        constraints: list[str],
        prompt: ChatPromptTemplate | None = None,
        timeout: float | None = None,
    ) -> None:
        self.model = model
        self.constraints = constraints
        self._prompt = prompt or _CONSTRAINT_PROMPT
        self._timeout = timeout
        self._chain = self._build_chain()

    def _build_chain(self) -> Any:
        """Build the constraint checking chain."""
        structured_model = self.model.with_structured_output(ConstraintCheckOutput)
        return self._prompt | structured_model

    def _invoke_with_timeout(self, inputs: dict[str, Any]) -> Any:
        """Invoke the chain with optional timeout."""
        if self._timeout is None:
            return self._chain.invoke(inputs)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self._chain.invoke, inputs)
            return future.result(timeout=self._timeout)

    def check(
        self,
        goal: Goal,
        state: StateSnapshot,
        action: ActionSpec | None = None,
    ) -> tuple[bool, str]:
        """Check constraints using LLM reasoning.

        Returns ``(passed, violation_message)``.
        """
        observation = state.observation or (
            f"State values: {state.values}" if state.values else "No observation"
        )

        action_info = "No action proposed"
        if action is not None:
            action_info = (
                f"**Name**: {action.name}\n"
                f"**Description**: {action.description}\n"
                f"**Parameters**: {dict(action.parameters)}"
            )

        try:
            result: ConstraintCheckOutput = self._invoke_with_timeout(
                {
                    "constraints_list": "\n".join(
                        f"{i + 1}. {c}" for i, c in enumerate(self.constraints)
                    ),
                    "goal_description": goal.description or goal.name or str(goal.objective),
                    "observation": observation,
                    "state_values": str(state.values) if state.values else "N/A",
                    "action_info": action_info,
                }
            )

            if result.overall_safe:
                return True, ""

            # Build violation message from failed assessments
            violations = []
            for a in result.assessments:
                if not a.is_satisfied:
                    violations.append(
                        f"{a.constraint_name} (severity={a.severity:.1f}): "
                        f"{a.reasoning}"
                    )

            message = (
                f"LLMConstraintChecker: {result.overall_reasoning} | "
                + " | ".join(violations)
            )
            return False, message

        except Exception as exc:
            logger.warning("LLMConstraintChecker: check failed: %s", exc)
            # Fail closed — reject on error to maintain safety
            return False, f"LLMConstraintChecker: constraint check failed: {exc}"

    def check_detailed(
        self,
        goal: Goal,
        state: StateSnapshot,
        action: ActionSpec | None = None,
    ) -> ConstraintCheckOutput | None:
        """Check constraints and return the full detailed assessment.

        Returns ``None`` on error.
        """
        observation = state.observation or (
            f"State values: {state.values}" if state.values else "No observation"
        )

        action_info = "No action proposed"
        if action is not None:
            action_info = (
                f"**Name**: {action.name}\n"
                f"**Description**: {action.description}\n"
                f"**Parameters**: {dict(action.parameters)}"
            )

        try:
            return self._invoke_with_timeout(
                {
                    "constraints_list": "\n".join(
                        f"{i + 1}. {c}" for i, c in enumerate(self.constraints)
                    ),
                    "goal_description": goal.description or goal.name or str(goal.objective),
                    "observation": observation,
                    "state_values": str(state.values) if state.values else "N/A",
                    "action_info": action_info,
                }
            )
        except Exception as exc:
            logger.warning("LLMConstraintChecker: detailed check failed: %s", exc)
            return None
