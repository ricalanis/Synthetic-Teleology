"""Distributed intentional grounding.

Implements the intentional grounding mechanism from Haidemariam (2026)
Section 5.4.7: goals must be grounded in external directives (user,
normative, negotiated) to maintain legitimacy.

The ``IntentionalGroundingManager`` accumulates directives from multiple
sources and uses an LLM to assess their influence on the current goal.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.enums import GoalOrigin
from synthetic_teleology.domain.values import EvalSignal, GoalProvenance

logger = logging.getLogger(__name__)


class GoalSource(Enum):
    """Source type for external directives."""

    USER = "user"
    NORMATIVE = "normative"
    NEGOTIATED = "negotiated"
    SYSTEM = "system"
    ENVIRONMENTAL = "environmental"


@dataclass
class ExternalDirective:
    """An external directive that may influence goal formation."""

    source: GoalSource
    content: str
    priority: float = 0.5  # 0-1, higher = more influential
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class IntentionalGroundingManager:
    """Manages distributed intentional grounding for goals.

    Accumulates directives from multiple sources and uses an LLM to
    assess their influence on the current goal. When grounding triggers,
    the goal may be adjusted to reflect external directives.

    Parameters
    ----------
    model:
        A LangChain chat model for grounding assessment.
    grounding_threshold:
        Minimum directive influence score to trigger grounding (0-1).
    max_directives:
        Maximum number of directives to retain.
    """

    def __init__(
        self,
        model: Any | None = None,
        grounding_threshold: float = 0.5,
        max_directives: int = 50,
    ) -> None:
        self._model = model
        self._grounding_threshold = grounding_threshold
        self._max_directives = max_directives
        self._directives: list[ExternalDirective] = []
        self._grounding_history: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    @property
    def directives(self) -> list[ExternalDirective]:
        """Current accumulated directives."""
        return list(self._directives)

    @property
    def grounding_history(self) -> list[dict[str, Any]]:
        """History of grounding events."""
        return list(self._grounding_history)

    def add_directive(self, directive: ExternalDirective) -> None:
        """Add an external directive.

        If the directive limit is reached, oldest directives are removed.
        """
        with self._lock:
            self._directives.append(directive)
            if len(self._directives) > self._max_directives:
                self._directives = self._directives[-self._max_directives:]

    def add_user_directive(self, content: str, priority: float = 0.8) -> None:
        """Convenience: add a user directive."""
        self.add_directive(ExternalDirective(
            source=GoalSource.USER,
            content=content,
            priority=priority,
        ))

    def add_normative_directive(self, content: str, priority: float = 0.6) -> None:
        """Convenience: add a normative/ethical directive."""
        self.add_directive(ExternalDirective(
            source=GoalSource.NORMATIVE,
            content=content,
            priority=priority,
        ))

    def add_negotiated_directive(self, content: str, priority: float = 0.7) -> None:
        """Convenience: add a directive from multi-agent negotiation."""
        self.add_directive(ExternalDirective(
            source=GoalSource.NEGOTIATED,
            content=content,
            priority=priority,
        ))

    def ground(self, goal: Goal, eval_signal: EvalSignal | None = None) -> Goal | None:
        """Assess directives and optionally ground (adjust) the goal.

        Parameters
        ----------
        goal:
            Current goal to potentially ground.
        eval_signal:
            Optional evaluation signal providing context.

        Returns
        -------
        Goal or None
            A new grounded goal, or None if no grounding needed.
        """
        with self._lock:
            if not self._directives:
                return None

            if self._model is not None:
                return self._llm_ground(goal, eval_signal)
            return self._rule_based_ground(goal, eval_signal)

    def _llm_ground(self, goal: Goal, eval_signal: EvalSignal | None) -> Goal | None:
        """LLM-based grounding assessment."""
        directives_text = "\n".join(
            f"- [{d.source.value}] (priority={d.priority:.1f}): {d.content}"
            for d in self._directives[-10:]  # Last 10 directives
        )

        goal_desc = goal.description or goal.name or str(goal)
        eval_info = ""
        if eval_signal is not None:
            eval_info = (
                f"\nCurrent eval: score={eval_signal.score:.2f},"
                f" confidence={eval_signal.confidence:.2f}"
            )

        prompt = (
            f"Current goal: {goal_desc}{eval_info}\n\n"
            f"External directives:\n{directives_text}\n\n"
            f"Assess whether the goal should be adjusted based on these directives. "
            f"Consider the priority and source of each directive.\n\n"
            f"Respond with JSON: {{\"should_ground\": true|false, "
            f"\"adjusted_description\": \"...\", \"adjusted_criteria\": [...], "
            f"\"influence_score\": 0.0-1.0, \"reasoning\": \"...\"}}"
        )

        try:
            response = self._model.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            data = self._parse_json(content)

            influence = float(data.get("influence_score", 0.0))
            should_ground = (
                data.get("should_ground", False)
                and influence >= self._grounding_threshold
            )

            self._grounding_history.append({
                "influence_score": influence,
                "should_ground": should_ground,
                "reasoning": data.get("reasoning", ""),
                "timestamp": time.time(),
            })

            if not should_ground:
                return None

            adjusted_desc = data.get("adjusted_description", goal.description)
            new_goal, _revision = goal.revise(
                new_description=adjusted_desc,
                reason="grounding",
            )
            # Attach provenance to the new goal (frozen — use replace)
            new_goal = dataclasses.replace(
                new_goal,
                provenance=GoalProvenance(
                    origin=GoalOrigin.ENDOGENOUS,
                    source_description="Intentional grounding from external directives",
                ),
            )
            return new_goal

        except Exception as exc:
            logger.warning("IntentionalGroundingManager: LLM grounding failed: %s", exc)
            return self._rule_based_ground(goal, eval_signal)

    def _rule_based_ground(self, goal: Goal, eval_signal: EvalSignal | None) -> Goal | None:
        """Rule-based grounding fallback (no LLM)."""
        if not self._directives:
            return None

        # Calculate weighted influence of high-priority directives
        high_priority = [d for d in self._directives if d.priority >= self._grounding_threshold]
        if not high_priority:
            return None

        avg_priority = sum(d.priority for d in high_priority) / len(high_priority)
        if avg_priority < self._grounding_threshold:
            return None

        # Combine high-priority directive contents into goal adjustment
        directive_summary = "; ".join(d.content for d in high_priority[-3:])
        current_desc = goal.description or goal.name or ""
        adjusted_desc = f"{current_desc} [Grounded: {directive_summary}]"

        self._grounding_history.append({
            "influence_score": avg_priority,
            "should_ground": True,
            "reasoning": "Rule-based grounding from high-priority directives",
            "timestamp": time.time(),
        })

        new_goal, _revision = goal.revise(
            new_description=adjusted_desc,
            reason="grounding",
        )
        new_goal = dataclasses.replace(
            new_goal,
            provenance=GoalProvenance(
                origin=GoalOrigin.ENDOGENOUS,
                source_description="Rule-based intentional grounding",
            ),
        )
        return new_goal

    def clear_directives(self) -> None:
        """Clear all accumulated directives."""
        self._directives.clear()

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        """Extract JSON from LLM response."""
        cleaned = text.strip()
        if "```json" in cleaned:
            start = cleaned.index("```json") + 7
            end = cleaned.index("```", start)
            cleaned = cleaned[start:end].strip()
        elif "```" in cleaned:
            start = cleaned.index("```") + 3
            end = cleaned.index("```", start)
            cleaned = cleaned[start:end].strip()
        return json.loads(cleaned)
