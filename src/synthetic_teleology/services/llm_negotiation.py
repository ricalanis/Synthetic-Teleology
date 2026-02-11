"""LLM-powered multi-agent dialogue negotiation.

Implements a 3-phase negotiation protocol per Haidemariam (2026):
1. **Propose** — Each agent's results -> LLM generates a proposal per agent.
2. **Critique** — LLM identifies agreements/disagreements across proposals.
3. **Synthesize** — LLM produces consensus direction + revised criteria.

Operates on ``agent_results`` dict (not BaseAgent objects), bridging the
graph and service layers cleanly.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class GoalProposal:
    """A single agent's proposal for the shared direction."""

    agent_id: str = ""
    proposed_direction: str = ""
    reasoning: str = ""
    priority_dimensions: list[str] = field(default_factory=list)
    confidence: float = 0.5


@dataclass
class NegotiationCritique:
    """LLM analysis of agreements and disagreements across proposals."""

    agreements: list[str] = field(default_factory=list)
    disagreements: list[str] = field(default_factory=list)
    synthesis_hints: list[str] = field(default_factory=list)


@dataclass
class NegotiationConsensus:
    """Final consensus from the negotiation dialogue."""

    shared_direction: str = ""
    revised_criteria: list[str] = field(default_factory=list)
    confidence: float = 0.5
    reasoning: str = ""
    round_count: int = 0


class LLMNegotiator:
    """LLM-powered multi-agent goal negotiation.

    Parameters
    ----------
    model:
        A LangChain chat model supporting ``with_structured_output``.
    max_dialogue_rounds:
        Maximum rounds of propose-critique-synthesize. Default 3.
    temperature:
        LLM sampling temperature. Default 0.7.
    """

    def __init__(
        self,
        model: Any,
        max_dialogue_rounds: int = 3,
        temperature: float = 0.7,
        timeout: float | None = None,
    ) -> None:
        self._model = model
        self._max_rounds = max_dialogue_rounds
        self._temperature = temperature
        self._timeout = timeout
        self._executor = (
            concurrent.futures.ThreadPoolExecutor(max_workers=1)
            if timeout is not None else None
        )

    def _invoke_with_timeout(self, prompt: str) -> Any:
        """Invoke the model with optional timeout."""
        if self._timeout is None:
            return self._model.invoke(prompt)
        future = self._executor.submit(self._model.invoke, prompt)
        return future.result(timeout=self._timeout)

    def shutdown(self) -> None:
        """Shut down the internal thread pool executor."""
        if self._executor is not None:
            self._executor.shutdown(wait=False)

    def negotiate(
        self,
        agent_results: dict[str, dict[str, Any]],
    ) -> NegotiationConsensus:
        """Run the 3-phase negotiation protocol.

        Parameters
        ----------
        agent_results:
            Dict mapping agent_id to their run results (containing
            ``final_goal``, ``eval_signal``, ``reasoning_trace``, etc.).

        Returns
        -------
        NegotiationConsensus
            The negotiation outcome with shared direction.
        """
        if len(agent_results) < 2:
            # Single agent — no negotiation needed
            agent_id = next(iter(agent_results)) if agent_results else "unknown"
            result = agent_results.get(agent_id, {})
            goal = result.get("final_goal")
            direction = ""
            if goal is not None:
                direction = getattr(goal, "description", str(goal))
            return NegotiationConsensus(
                shared_direction=direction,
                confidence=1.0,
                round_count=0,
            )

        # Phase 1: Propose
        proposals = self._phase_propose(agent_results)

        # Phase 2: Critique
        critique = self._phase_critique(proposals)

        # Phase 3: Synthesize
        consensus = self._phase_synthesize(proposals, critique)

        return consensus

    def _phase_propose(
        self,
        agent_results: dict[str, dict[str, Any]],
    ) -> list[GoalProposal]:
        """Generate a proposal per agent."""
        proposals = []

        for agent_id, result in agent_results.items():
            goal = result.get("final_goal")
            eval_signal = result.get("eval_signal")

            goal_desc = ""
            if goal is not None:
                goal_desc = getattr(goal, "description", "") or str(goal)

            eval_info = ""
            if eval_signal is not None:
                eval_info = (
                    f"score={eval_signal.score:.2f},"
                    f" confidence={eval_signal.confidence:.2f}"
                )

            prompt = (
                f"Agent '{agent_id}' completed a round with:\n"
                f"- Goal: {goal_desc}\n"
                f"- Evaluation: {eval_info}\n"
                f"- Steps: {result.get('steps', 0)}\n"
                f"- Stop reason: {result.get('stop_reason', 'unknown')}\n\n"
                f"Generate a proposal for the shared direction. "
                f"Respond with JSON: {{\"direction\": \"...\", \"reasoning\": \"...\", "
                f"\"priority_dimensions\": [...], \"confidence\": 0.0-1.0}}"
            )

            try:
                response = self._invoke_with_timeout(prompt)
                content = response.content if hasattr(response, "content") else str(response)
                data = self._parse_json(content)

                proposals.append(GoalProposal(
                    agent_id=agent_id,
                    proposed_direction=data.get("direction", goal_desc),
                    reasoning=data.get("reasoning", ""),
                    priority_dimensions=data.get("priority_dimensions", []),
                    confidence=float(data.get("confidence", 0.5)),
                ))
            except Exception as exc:
                logger.warning("LLMNegotiator: propose failed for %s: %s", agent_id, exc)
                proposals.append(GoalProposal(
                    agent_id=agent_id,
                    proposed_direction=goal_desc,
                    confidence=0.3,
                ))

        return proposals

    def _phase_critique(self, proposals: list[GoalProposal]) -> NegotiationCritique:
        """Identify agreements and disagreements."""
        proposal_summary = "\n".join(
            f"- {p.agent_id}: {p.proposed_direction} (confidence={p.confidence:.2f})"
            for p in proposals
        )

        prompt = (
            f"Multiple agents have proposed directions:\n{proposal_summary}\n\n"
            f"Analyze the proposals and identify:\n"
            f"1. Agreements (common themes)\n"
            f"2. Disagreements (conflicting priorities)\n"
            f"3. Synthesis hints (how to reconcile)\n\n"
            f"Respond with JSON: {{\"agreements\": [...], \"disagreements\": [...], "
            f"\"synthesis_hints\": [...]}}"
        )

        try:
            response = self._invoke_with_timeout(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            data = self._parse_json(content)

            return NegotiationCritique(
                agreements=data.get("agreements", []),
                disagreements=data.get("disagreements", []),
                synthesis_hints=data.get("synthesis_hints", []),
            )
        except Exception as exc:
            logger.warning("LLMNegotiator: critique failed: %s", exc)
            return NegotiationCritique()

    def _phase_synthesize(
        self,
        proposals: list[GoalProposal],
        critique: NegotiationCritique,
    ) -> NegotiationConsensus:
        """Produce a consensus direction."""
        proposal_summary = "\n".join(
            f"- {p.agent_id}: {p.proposed_direction}"
            for p in proposals
        )
        agreements = ", ".join(critique.agreements) if critique.agreements else "none identified"
        disagreements = (
            ", ".join(critique.disagreements) if critique.disagreements else "none identified"
        )
        hints = ", ".join(critique.synthesis_hints) if critique.synthesis_hints else "none"

        prompt = (
            f"Synthesize a consensus from these proposals:\n{proposal_summary}\n\n"
            f"Agreements: {agreements}\n"
            f"Disagreements: {disagreements}\n"
            f"Hints: {hints}\n\n"
            f"Produce a shared direction that balances all perspectives. "
            f"Respond with JSON: {{\"shared_direction\": \"...\", "
            f"\"revised_criteria\": [...], \"confidence\": 0.0-1.0, "
            f"\"reasoning\": \"...\"}}"
        )

        try:
            response = self._invoke_with_timeout(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            data = self._parse_json(content)

            return NegotiationConsensus(
                shared_direction=data.get("shared_direction", ""),
                revised_criteria=data.get("revised_criteria", []),
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", ""),
                round_count=1,
            )
        except Exception as exc:
            logger.warning("LLMNegotiator: synthesize failed: %s", exc)
            # Fallback: combine proposals
            combined = "; ".join(p.proposed_direction for p in proposals if p.proposed_direction)
            return NegotiationConsensus(
                shared_direction=combined,
                confidence=0.3,
                round_count=1,
            )

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
