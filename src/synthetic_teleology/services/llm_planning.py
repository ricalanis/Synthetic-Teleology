"""LLM-based multi-hypothesis planning using LangChain structured output.

Generates N candidate plans (hypotheses), each with a confidence score,
reasoning, and expected outcome.  Returns a probabilistic ``PolicySpec``
where action selection can be stochastic.
"""

from __future__ import annotations

import concurrent.futures
import logging
import math
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.values import (
    ActionSpec,
    Hypothesis,
    PolicySpec,
    StateSnapshot,
)
from synthetic_teleology.services.planning import BasePlanner

logger = logging.getLogger(__name__)


# -- Structured output schemas -----------------------------------------------


class ActionProposal(BaseModel):
    """A single action proposed by the LLM."""

    name: str = Field(description="Action name")
    description: str = Field(default="", description="What this action does")
    tool_name: str | None = Field(default=None, description="LangChain tool to invoke")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Action parameters")
    cost: float = Field(default=0.0, ge=0, description="Estimated cost")


class PlanHypothesis(BaseModel):
    """A candidate plan hypothesis."""

    actions: list[ActionProposal] = Field(description="Ordered actions for this plan")
    reasoning: str = Field(description="Why this plan was chosen")
    expected_outcome: str = Field(description="Expected result of executing this plan")
    confidence: float = Field(ge=0, le=1, description="Confidence in this plan [0, 1]")
    risk: str = Field(default="", description="Risk assessment")


class PlanningOutput(BaseModel):
    """Full planning output with multiple hypotheses."""

    hypotheses: list[PlanHypothesis] = Field(description="Candidate plans")
    selected_index: int = Field(description="Index of the recommended plan")
    selection_reasoning: str = Field(description="Why this plan was selected")


# -- Prompt ------------------------------------------------------------------

_PLANNING_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strategic planner for a goal-directed AI agent. "
            "Given a goal and current state, generate {num_hypotheses} distinct "
            "plan hypotheses. Each plan should be a concrete sequence of actions. "
            "Consider different approaches with varying risk/reward profiles.\n\n"
            "{tools_description}"
            "For each hypothesis:\n"
            "1. Propose concrete, executable actions\n"
            "2. Explain your reasoning\n"
            "3. Describe the expected outcome\n"
            "4. Assess confidence (0-1) and risks\n\n"
            "Then select the best hypothesis and explain why.",
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
            "Generate {num_hypotheses} plan hypotheses.",
        ),
    ]
)


def _softmax(values: list[float], temperature: float = 1.0) -> list[float]:
    """Compute softmax with temperature."""
    if not values:
        return []
    scaled = [v / temperature for v in values]
    max_val = max(scaled)
    exps = [math.exp(v - max_val) for v in scaled]
    total = sum(exps)
    return [e / total for e in exps]


# -- LLMPlanner -------------------------------------------------------------


class LLMPlanner(BasePlanner):
    """LLM-based multi-hypothesis planner.

    Generates N candidate plans via the LLM, assigns confidence-based
    probabilities, and returns a probabilistic ``PolicySpec``.

    Parameters
    ----------
    model:
        A LangChain chat model.
    tools:
        Optional list of LangChain tools the agent can use.
    num_hypotheses:
        Number of plan candidates to generate (default 3).
    temperature:
        Softmax temperature for probability assignment (default 1.0).
    prompt:
        Optional custom ``ChatPromptTemplate``.
    """

    def __init__(
        self,
        model: BaseChatModel,
        tools: list[Any] | None = None,
        num_hypotheses: int = 3,
        temperature: float = 1.0,
        prompt: ChatPromptTemplate | None = None,
        timeout: float | None = None,
    ) -> None:
        self.model = model
        self.tools = tools or []
        self.num_hypotheses = num_hypotheses
        self.temperature = temperature
        self._prompt = prompt or _PLANNING_PROMPT
        self._timeout = timeout
        self._chain = self._build_chain()

    def _build_chain(self) -> Any:
        """Build the planning chain with structured output."""
        structured_model = self.model.with_structured_output(PlanningOutput)
        return self._prompt | structured_model

    def _invoke_with_timeout(self, inputs: dict[str, Any]) -> Any:
        """Invoke the chain with optional timeout."""
        if self._timeout is None:
            return self._chain.invoke(inputs)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self._chain.invoke, inputs)
            return future.result(timeout=self._timeout)

    def _get_tools_description(self) -> str:
        """Generate tool descriptions for the prompt."""
        if not self.tools:
            return ""
        lines = ["Available tools:\n"]
        for tool in self.tools:
            name = getattr(tool, "name", str(tool))
            desc = getattr(tool, "description", "")
            lines.append(f"  - {name}: {desc}")
        lines.append("\n")
        return "\n".join(lines)

    def plan(self, goal: Goal, state: StateSnapshot) -> PolicySpec:
        """Generate a multi-hypothesis plan using the LLM.

        Returns a ``PolicySpec`` with probabilistic action selection.
        """
        observation = state.observation or (
            f"State values: {state.values}" if state.values else "No observation"
        )
        criteria = goal.success_criteria

        try:
            result: PlanningOutput = self._invoke_with_timeout(
                {
                    "num_hypotheses": self.num_hypotheses,
                    "tools_description": self._get_tools_description(),
                    "goal_name": goal.name or goal.goal_id,
                    "goal_description": goal.description or str(goal.objective),
                    "success_criteria": (
                        "\n".join(f"- {c}" for c in criteria)
                        if criteria
                        else "No specific criteria"
                    ),
                    "observation": observation,
                    "state_values": str(state.values) if state.values else "N/A",
                    "context": str(dict(state.context)) if state.context else "N/A",
                }
            )

            if not result.hypotheses:
                return PolicySpec()

            # Convert hypotheses to domain objects
            hypotheses = []
            for h in result.hypotheses:
                actions = tuple(
                    ActionSpec(
                        name=a.name,
                        description=a.description,
                        tool_name=a.tool_name,
                        parameters=a.parameters,
                        cost=a.cost,
                    )
                    for a in h.actions
                )
                hypotheses.append(
                    Hypothesis(
                        actions=actions,
                        confidence=h.confidence,
                        reasoning=h.reasoning,
                        expected_outcome=h.expected_outcome,
                        risk_assessment=h.risk,
                    )
                )

            # Build probabilistic policy from selected hypothesis actions
            selected_idx = min(result.selected_index, len(hypotheses) - 1)
            selected = hypotheses[selected_idx]

            # If multiple hypotheses, create probabilistic policy over first actions
            if len(hypotheses) > 1:
                confidences = [h.confidence for h in hypotheses]
                probabilities = _softmax(confidences, self.temperature)

                # Collect the first action from each hypothesis
                first_actions = []
                valid_probs = []
                for h, p in zip(hypotheses, probabilities, strict=True):
                    if h.actions:
                        first_actions.append(h.actions[0])
                        valid_probs.append(p)

                if first_actions:
                    # Renormalize
                    total = sum(valid_probs)
                    valid_probs = [p / total for p in valid_probs]

                    return PolicySpec(
                        actions=tuple(first_actions),
                        probabilities=tuple(valid_probs),
                        metadata={
                            "planner": "LLMPlanner",
                            "hypotheses": [
                                {
                                    "reasoning": h.reasoning,
                                    "confidence": h.confidence,
                                    "expected_outcome": h.expected_outcome,
                                }
                                for h in hypotheses
                            ],
                            "selected_index": selected_idx,
                            "selection_reasoning": result.selection_reasoning,
                        },
                    )

            # Single hypothesis or fallback: deterministic policy
            return PolicySpec(
                actions=selected.actions,
                metadata={
                    "planner": "LLMPlanner",
                    "reasoning": selected.reasoning,
                    "confidence": selected.confidence,
                    "expected_outcome": selected.expected_outcome,
                },
            )

        except Exception as exc:
            logger.warning("LLMPlanner: planning failed: %s", exc)
            noop_action = ActionSpec(
                name="noop_fallback",
                description="No-op fallback due to planning error",
            )
            return PolicySpec(
                actions=(noop_action,),
                metadata={
                    "planner": "LLMPlanner",
                    "llm_error": True,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
