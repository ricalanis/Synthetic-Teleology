"""Co-evolutionary constraint management.

Implements evolving constraints per Haidemariam (2026) Section 5.4.6:
constraints are not fixed — they co-evolve with the agent's goals through
LLM-based analysis of violation patterns and interaction history.

The ``EvolvingConstraintManager`` periodically proposes additions, removals,
or modifications of constraints based on accumulated evidence.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ConstraintEvolution:
    """Record of a single constraint evolution event."""

    evolution_type: str  # "add", "remove", "modify"
    constraint_text: str
    reasoning: str = ""
    confidence: float = 0.5
    timestamp: float = field(default_factory=time.time)
    previous_constraint: str | None = None


@dataclass
class EvolutionResult:
    """Result of a constraint evolution round."""

    evolutions: list[ConstraintEvolution] = field(default_factory=list)
    new_constraints: list[str] = field(default_factory=list)
    removed_constraints: list[str] = field(default_factory=list)
    modified_constraints: list[tuple[str, str]] = field(default_factory=list)
    reasoning: str = ""


class EvolvingConstraintManager:
    """LLM-based co-evolutionary constraint manager.

    Analyzes violation patterns and interaction history to propose
    constraint evolution: additions, removals, or modifications.

    Parameters
    ----------
    model:
        A LangChain chat model for reasoning about constraints.
    initial_constraints:
        Starting set of constraint texts.
    evolution_frequency:
        Evolve constraints every N rounds (default 3).
    max_constraints:
        Maximum number of active constraints (default 20).
    """

    def __init__(
        self,
        model: Any,
        initial_constraints: list[str] | None = None,
        evolution_frequency: int = 3,
        max_constraints: int = 20,
    ) -> None:
        self._model = model
        self._constraints: list[str] = list(initial_constraints or [])
        self._evolution_frequency = evolution_frequency
        self._max_constraints = max_constraints
        self._violation_history: list[dict[str, Any]] = []
        self._evolution_history: list[EvolutionResult] = []
        self._round_counter = 0

    @property
    def constraints(self) -> list[str]:
        """Current active constraints."""
        return list(self._constraints)

    @property
    def evolution_history(self) -> list[EvolutionResult]:
        """History of constraint evolution events."""
        return list(self._evolution_history)

    @property
    def round_counter(self) -> int:
        """Number of rounds processed."""
        return self._round_counter

    def record_violations(
        self, violations: list[str], context: dict[str, Any] | None = None,
    ) -> None:
        """Record constraint violations for analysis."""
        self._violation_history.append({
            "violations": violations,
            "context": context or {},
            "timestamp": time.time(),
        })

    def should_evolve(self) -> bool:
        """Check if it's time for constraint evolution."""
        return (
            self._round_counter > 0
            and self._round_counter % self._evolution_frequency == 0
        )

    def step(self, agent_results: dict[str, Any] | None = None) -> EvolutionResult | None:
        """Process one round: record data and optionally evolve.

        Parameters
        ----------
        agent_results:
            Optional dict of agent results from a multi-agent round.

        Returns
        -------
        EvolutionResult or None
            The evolution result if evolution was triggered, else None.
        """
        self._round_counter += 1

        if not self.should_evolve():
            return None

        return self.evolve(agent_results)

    def evolve(self, agent_results: dict[str, Any] | None = None) -> EvolutionResult:
        """Run LLM-based constraint evolution analysis.

        Parameters
        ----------
        agent_results:
            Optional agent results providing context.

        Returns
        -------
        EvolutionResult
            Proposed constraint evolutions.
        """
        violation_summary = self._summarize_violations()
        current = "\n".join(f"- {c}" for c in self._constraints) if self._constraints else "None"

        prompt = (
            f"Current constraints:\n{current}\n\n"
            f"Recent violation patterns:\n{violation_summary}\n\n"
            f"Analyze whether any constraints should be:\n"
            f"1. Added (new constraints needed based on patterns)\n"
            f"2. Removed (over-restrictive or obsolete)\n"
            f"3. Modified (needs refinement)\n\n"
            f"Respond with JSON: {{\"evolutions\": ["
            f"{{\"type\": \"add\"|\"remove\"|\"modify\", "
            f"\"constraint\": \"...\", \"reasoning\": \"...\", "
            f"\"confidence\": 0.0-1.0, \"previous\": null|\"...\"}}], "
            f"\"overall_reasoning\": \"...\"}}"
        )

        try:
            response = self._model.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            data = self._parse_json(content)

            result = self._apply_evolutions(data)
            self._evolution_history.append(result)
            return result

        except Exception as exc:
            logger.warning("EvolvingConstraintManager: evolution failed: %s", exc)
            result = EvolutionResult(reasoning=f"Evolution failed: {exc}")
            self._evolution_history.append(result)
            return result

    def _summarize_violations(self) -> str:
        """Summarize recent violations for the LLM prompt."""
        if not self._violation_history:
            return "No violations recorded."

        # Take last 10 entries
        recent = self._violation_history[-10:]
        lines = []
        for entry in recent:
            vs = "; ".join(entry["violations"]) if entry["violations"] else "none"
            lines.append(f"- {vs}")
        return "\n".join(lines)

    def _apply_evolutions(self, data: dict[str, Any]) -> EvolutionResult:
        """Apply parsed evolutions to the constraint set."""
        result = EvolutionResult(reasoning=data.get("overall_reasoning", ""))
        evolutions_data = data.get("evolutions", [])

        for ev in evolutions_data:
            ev_type = ev.get("type", "")
            constraint = ev.get("constraint", "")
            reasoning = ev.get("reasoning", "")
            confidence = float(ev.get("confidence", 0.5))
            previous = ev.get("previous")

            evolution = ConstraintEvolution(
                evolution_type=ev_type,
                constraint_text=constraint,
                reasoning=reasoning,
                confidence=confidence,
                previous_constraint=previous,
            )
            result.evolutions.append(evolution)

            if ev_type == "add" and constraint:
                if len(self._constraints) < self._max_constraints:
                    self._constraints.append(constraint)
                    result.new_constraints.append(constraint)
            elif ev_type == "remove" and constraint:
                if constraint in self._constraints:
                    self._constraints.remove(constraint)
                    result.removed_constraints.append(constraint)
            elif ev_type == "modify" and constraint and previous and previous in self._constraints:
                idx = self._constraints.index(previous)
                self._constraints[idx] = constraint
                result.modified_constraints.append((previous, constraint))

        return result

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
