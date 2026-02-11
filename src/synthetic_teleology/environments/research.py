"""Simulated knowledge synthesis environment.

``ResearchEnvironment`` models a world where agents pursue knowledge
through research, synthesis, and exploration.  State tracks per-topic
knowledge levels, synthesis quality, and novelty -- representing the
kind of epistemic landscape a research-oriented agent might navigate.

Typical use: testing teleological agents in domains where the objective
is knowledge accumulation or creative synthesis rather than numeric
optimisation.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from synthetic_teleology.domain.values import ActionSpec, StateSnapshot
from synthetic_teleology.environments.base import BaseEnvironment
from synthetic_teleology.infrastructure.event_bus import EventBus

logger = logging.getLogger(__name__)

# Default topics if none specified
_DEFAULT_TOPICS = ("topic_a", "topic_b", "topic_c")


class ResearchEnvironment(BaseEnvironment):
    """Simulated research/knowledge synthesis environment.

    The state vector is structured as::

        [knowledge_0, knowledge_1, ..., knowledge_N-1, synthesis_quality, novelty]

    where ``N`` is the number of topics.

    Three action types are supported:

    * ``"research"`` -- increase knowledge in a specific topic.
      Parameters: ``{"topic": int, "effort": float}``.
    * ``"synthesize"`` -- combine existing knowledge into synthesis quality.
      Parameters: ``{"effort": float}``.
    * ``"explore"`` -- venture into unknown territory, boosting novelty.
      Parameters: ``{"effort": float}``.

    Knowledge follows diminishing returns (logarithmic growth).
    Synthesis quality depends on the breadth and depth of accumulated
    knowledge.  Novelty decays over time unless actively maintained
    through exploration.

    Parameters
    ----------
    topics:
        Tuple of topic names.  Determines the number of knowledge dimensions.
    max_knowledge:
        Maximum knowledge level per topic.
    knowledge_decay:
        Per-step knowledge decay rate (simulates forgetting). [0, 1].
    novelty_decay:
        Per-step novelty decay rate. [0, 1].
    synthesis_bonus:
        Multiplier for synthesis quality gains from diverse knowledge.
    noise_std:
        Standard deviation of Gaussian noise on all state changes.
    event_bus:
        Event bus for domain events.
    env_id:
        Identifier for this environment.

    Example
    -------
    ::

        env = ResearchEnvironment(topics=("ml", "physics", "biology"))
        state = env.reset()
        action = ActionSpec(name="research", parameters={"topic": 0, "effort": 0.5})
        new_state = env.step(action)
    """

    def __init__(
        self,
        topics: tuple[str, ...] = _DEFAULT_TOPICS,
        max_knowledge: float = 10.0,
        knowledge_decay: float = 0.01,
        novelty_decay: float = 0.05,
        synthesis_bonus: float = 1.5,
        noise_std: float = 0.02,
        event_bus: EventBus | None = None,
        env_id: str = "research-env",
    ) -> None:
        super().__init__(event_bus=event_bus, env_id=env_id)

        if len(topics) < 1:
            raise ValueError("At least one topic is required")

        self._topics = topics
        self._num_topics = len(topics)
        self._max_knowledge = max_knowledge
        self._knowledge_decay = max(0.0, min(1.0, knowledge_decay))
        self._novelty_decay = max(0.0, min(1.0, novelty_decay))
        self._synthesis_bonus = synthesis_bonus
        self._noise_std = noise_std

        # State: [knowledge_per_topic..., synthesis_quality, novelty]
        # Total dimensions = num_topics + 2
        self._state_dim = self._num_topics + 2
        self._knowledge: NDArray[np.float64] = np.zeros(self._num_topics, dtype=np.float64)
        self._synthesis_quality: float = 0.0
        self._novelty: float = 0.0

        # Track research history for diversity calculations
        self._research_counts: NDArray[np.float64] = np.zeros(
            self._num_topics, dtype=np.float64
        )

    # -- properties ---------------------------------------------------------

    @property
    def topics(self) -> tuple[str, ...]:
        """The topic names."""
        return self._topics

    @property
    def num_topics(self) -> int:
        """Number of knowledge topics."""
        return self._num_topics

    @property
    def knowledge_levels(self) -> NDArray[np.float64]:
        """Current per-topic knowledge levels (read-only copy)."""
        return self._knowledge.copy()

    @property
    def synthesis_quality(self) -> float:
        """Current synthesis quality."""
        return self._synthesis_quality

    @property
    def novelty(self) -> float:
        """Current novelty level."""
        return self._novelty

    @property
    def knowledge_diversity(self) -> float:
        """Shannon diversity of knowledge across topics.

        Higher values indicate more balanced knowledge distribution.
        Returns 0.0 if total knowledge is zero.
        """
        total = float(self._knowledge.sum())
        if total <= 0:
            return 0.0
        proportions = self._knowledge / total
        # Avoid log(0)
        proportions = proportions[proportions > 0]
        return float(-np.sum(proportions * np.log(proportions + 1e-12)))

    # -- BaseEnvironment implementation -------------------------------------

    def step(self, action: ActionSpec, **kwargs: Any) -> StateSnapshot:
        """Apply a research, synthesis, or exploration action.

        Parameters
        ----------
        action:
            Action with ``name`` in ``{"research", "synthesize", "explore"}``.
            See class docstring for expected parameters.

        Returns
        -------
        StateSnapshot
            Updated state after applying the action and natural dynamics.
        """
        action_name = action.name.lower()
        effort = max(0.0, float(action.parameters.get("effort", 0.5)))

        if action_name == "research":
            self._do_research(action, effort)
        elif action_name == "synthesize":
            self._do_synthesize(effort)
        elif action_name == "explore":
            self._do_explore(effort)
        else:
            logger.warning(
                "ResearchEnvironment: unknown action %r, treating as no-op",
                action_name,
            )

        # Natural dynamics: decay
        self._knowledge *= (1.0 - self._knowledge_decay)
        self._novelty *= (1.0 - self._novelty_decay)

        # Add noise
        if self._noise_std > 0.0:
            knowledge_noise = np.random.normal(0.0, self._noise_std, self._num_topics)
            self._knowledge = np.clip(
                self._knowledge + knowledge_noise,
                0.0,
                self._max_knowledge,
            )
            self._synthesis_quality = max(
                0.0,
                self._synthesis_quality + np.random.normal(0.0, self._noise_std),
            )
            self._novelty = max(
                0.0,
                self._novelty + np.random.normal(0.0, self._noise_std),
            )

        snapshot = self._build_snapshot(
            metadata={
                "step": self._step_count + 1,
                "action_name": action_name,
                "effort": effort,
                "diversity": self.knowledge_diversity,
            }
        )
        self._post_step(snapshot)
        return snapshot

    def observe(self, **kwargs: Any) -> StateSnapshot:
        """Return current research state.

        Returns
        -------
        StateSnapshot
            Knowledge levels, synthesis quality, and novelty.
        """
        return self._build_snapshot(
            metadata={
                "step": self._step_count,
                "diversity": self.knowledge_diversity,
                "topics": self._topics,
            }
        )

    def reset(self) -> StateSnapshot:
        """Reset all research progress to zero.

        Returns
        -------
        StateSnapshot
            The blank-slate initial state.
        """
        self._knowledge = np.zeros(self._num_topics, dtype=np.float64)
        self._synthesis_quality = 0.0
        self._novelty = 0.0
        self._research_counts = np.zeros(self._num_topics, dtype=np.float64)
        self._post_reset()

        return self._build_snapshot(
            metadata={"step": 0, "reset": True}
        )

    # -- state serialization -------------------------------------------------

    def _state_dict_impl(self) -> dict[str, Any]:
        return {
            "knowledge": list(float(v) for v in self._knowledge),
            "synthesis_quality": self._synthesis_quality,
            "novelty": self._novelty,
            "research_counts": list(float(v) for v in self._research_counts),
        }

    def _load_state_dict_impl(self, state: dict[str, Any]) -> None:
        if "knowledge" in state:
            self._knowledge = np.array(state["knowledge"], dtype=np.float64)
        if "synthesis_quality" in state:
            self._synthesis_quality = float(state["synthesis_quality"])
        if "novelty" in state:
            self._novelty = float(state["novelty"])
        if "research_counts" in state:
            self._research_counts = np.array(state["research_counts"], dtype=np.float64)

    # -- action implementations ---------------------------------------------

    def _do_research(self, action: ActionSpec, effort: float) -> None:
        """Apply a research action to increase knowledge in a topic.

        Knowledge gain follows diminishing returns:
        ``gain = effort / (1 + current_level / max_knowledge)``
        """
        topic_idx = int(action.parameters.get("topic", 0))
        if not (0 <= topic_idx < self._num_topics):
            logger.warning(
                "ResearchEnvironment: invalid topic index %d, clamping to [0, %d)",
                topic_idx,
                self._num_topics,
            )
            topic_idx = max(0, min(topic_idx, self._num_topics - 1))

        current = self._knowledge[topic_idx]
        # Diminishing returns: harder to learn more as knowledge grows
        gain = effort / (1.0 + current / self._max_knowledge)
        self._knowledge[topic_idx] = min(self._max_knowledge, current + gain)
        self._research_counts[topic_idx] += 1.0

    def _do_synthesize(self, effort: float) -> None:
        """Apply a synthesis action to improve synthesis quality.

        Synthesis quality depends on both the total knowledge and its
        diversity across topics.  A diversity bonus rewards breadth.
        """
        total_knowledge = float(self._knowledge.sum())
        diversity = self.knowledge_diversity
        max_diversity = math.log(self._num_topics + 1e-12)

        # Normalize diversity to [0, 1]
        diversity_factor = diversity / max_diversity if max_diversity > 0 else 0.0

        # Synthesis gain: effort * knowledge-base * diversity-bonus
        knowledge_factor = total_knowledge / (self._num_topics * self._max_knowledge)
        gain = effort * knowledge_factor * (1.0 + self._synthesis_bonus * diversity_factor)

        self._synthesis_quality += gain

    def _do_explore(self, effort: float) -> None:
        """Apply an exploration action to boost novelty.

        Exploration also grants small random knowledge gains across
        under-researched topics, simulating serendipitous discovery.
        """
        # Direct novelty boost
        self._novelty += effort * 0.8

        # Serendipitous knowledge gain in least-researched topics
        if self._num_topics > 1:
            # Find topics below median research count
            median_count = float(np.median(self._research_counts))
            for i in range(self._num_topics):
                if self._research_counts[i] <= median_count:
                    serendipity_gain = effort * 0.1 * np.random.random()
                    self._knowledge[i] = min(
                        self._max_knowledge,
                        self._knowledge[i] + serendipity_gain,
                    )

    # -- perturbation -------------------------------------------------------

    def _apply_perturbation(self, perturbation: dict[str, Any]) -> None:
        """Apply research-specific perturbations.

        Supported keys:

        * ``"knowledge_boost"`` -- dict of ``{topic_idx: amount}`` added.
        * ``"knowledge_reset"`` -- list of topic indices to reset to zero.
        * ``"synthesis_shock"`` -- float added to synthesis quality.
        * ``"novelty_injection"`` -- float added to novelty.
        * ``"paradigm_shift"`` -- redistributes knowledge randomly.

        Parameters
        ----------
        perturbation:
            Dictionary describing the perturbation.
        """
        if "knowledge_boost" in perturbation:
            for idx_key, amount in perturbation["knowledge_boost"].items():
                idx = int(idx_key)
                if 0 <= idx < self._num_topics:
                    self._knowledge[idx] = min(
                        self._max_knowledge,
                        self._knowledge[idx] + float(amount),
                    )

        if "knowledge_reset" in perturbation:
            for idx in perturbation["knowledge_reset"]:
                idx = int(idx)
                if 0 <= idx < self._num_topics:
                    self._knowledge[idx] = 0.0

        if "synthesis_shock" in perturbation:
            self._synthesis_quality = max(
                0.0,
                self._synthesis_quality + float(perturbation["synthesis_shock"]),
            )

        if "novelty_injection" in perturbation:
            self._novelty = max(
                0.0,
                self._novelty + float(perturbation["novelty_injection"]),
            )

        if "paradigm_shift" in perturbation:
            # Randomly redistribute total knowledge
            total = float(self._knowledge.sum())
            if total > 0:
                distribution = np.random.dirichlet(
                    np.ones(self._num_topics)
                )
                self._knowledge = distribution * total
                self._knowledge = np.minimum(self._knowledge, self._max_knowledge)

    # -- internal helpers ---------------------------------------------------

    def _build_snapshot(self, metadata: dict[str, Any] | None = None) -> StateSnapshot:
        """Construct a state snapshot from current internal state.

        The state vector is ``[knowledge_0, ..., knowledge_N, synthesis, novelty]``.
        """
        values = (
            tuple(float(v) for v in self._knowledge)
            + (self._synthesis_quality, self._novelty)
        )
        return self._make_snapshot(values=values, metadata=metadata or {})

    def __repr__(self) -> str:
        avg_knowledge = float(self._knowledge.mean()) if self._num_topics > 0 else 0.0
        return (
            f"ResearchEnvironment("
            f"topics={self._num_topics}, "
            f"avg_knowledge={avg_knowledge:.2f}, "
            f"synthesis={self._synthesis_quality:.2f}, "
            f"novelty={self._novelty:.2f}, "
            f"steps={self._step_count})"
        )
