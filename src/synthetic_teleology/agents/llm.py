"""LLM-powered teleological agent for the Synthetic Teleology framework.

``LLMAgent`` extends :class:`TeleologicalAgent` by wiring LLM-based
strategies for evaluation, goal revision, and planning.  It delegates
each phase to the corresponding LLM service:

* :class:`LLMCriticEvaluator`  -- computes Delta(G_t, S_t) via an LLM critic
* :class:`LLMGoalEditor`       -- proposes goal revisions via an LLM editor
* :class:`LLMPlanner`          -- generates action policies via an LLM planner

This corresponds to **Section 5.7** of the Synthetic Teleology paper:
full LLM instantiation of the teleological loop.

Example
-------
::

    from synthetic_teleology.infrastructure.llm import LLMProvider, LLMConfig
    from synthetic_teleology.agents.llm import LLMAgent

    provider = MyLLMProvider(api_key="...")
    agent = LLMAgent.from_provider(
        agent_id="research-agent",
        provider=provider,
        target_values=(5.0, 5.0),
        directions=(Direction.APPROACH, Direction.APPROACH),
    )
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Sequence

from synthetic_teleology.agents.teleological import TeleologicalAgent
from synthetic_teleology.domain.aggregates import ConstraintSet
from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.enums import Direction
from synthetic_teleology.domain.values import ActionSpec, ObjectiveVector
from synthetic_teleology.infrastructure.event_bus import EventBus
from synthetic_teleology.services.evaluation import LLMCriticEvaluator
from synthetic_teleology.services.goal_revision import LLMGoalEditor
from synthetic_teleology.services.planning import LLMPlanner

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ===================================================================== #
#  LLM Agent Configuration                                               #
# ===================================================================== #


class LLMAgentConfig:
    """Configuration container for an LLM-powered teleological agent.

    Bundles all LLM-specific settings (prompts, temperatures, available
    actions) into a single object that can be serialised, logged, and
    passed around.

    Parameters
    ----------
    eval_system_prompt:
        System prompt for the LLM critic evaluator.
    eval_temperature:
        Temperature for evaluation calls.
    revision_system_prompt:
        System prompt for the LLM goal editor.
    revision_temperature:
        Temperature for goal-revision calls.
    planning_system_prompt:
        System prompt for the LLM planner.
    planning_temperature:
        Temperature for planning calls.
    available_actions:
        Action space for the LLM planner to choose from.
    """

    def __init__(
        self,
        eval_system_prompt: str = "",
        eval_temperature: float = 0.3,
        revision_system_prompt: str = "",
        revision_temperature: float = 0.4,
        planning_system_prompt: str = "",
        planning_temperature: float = 0.5,
        available_actions: Sequence[ActionSpec] | None = None,
    ) -> None:
        self.eval_system_prompt = eval_system_prompt
        self.eval_temperature = eval_temperature
        self.revision_system_prompt = revision_system_prompt
        self.revision_temperature = revision_temperature
        self.planning_system_prompt = planning_system_prompt
        self.planning_temperature = planning_temperature
        self.available_actions = list(available_actions) if available_actions else []

    def __repr__(self) -> str:
        return (
            f"LLMAgentConfig("
            f"eval_temp={self.eval_temperature}, "
            f"revision_temp={self.revision_temperature}, "
            f"planning_temp={self.planning_temperature}, "
            f"actions={len(self.available_actions)})"
        )


# ===================================================================== #
#  LLM Agent                                                              #
# ===================================================================== #


class LLMAgent(TeleologicalAgent):
    """Teleological agent with LLM-powered evaluation, revision, and planning.

    Extends :class:`TeleologicalAgent` by injecting :class:`LLMCriticEvaluator`,
    :class:`LLMGoalEditor`, and :class:`LLMPlanner` as the strategy
    implementations for the three core phases of the teleological loop.

    This is the **Section 5.7 full LLM instantiation**: every cognitive
    component (evaluate Delta, revise G, plan pi) is delegated to an LLM
    provider, making the agent's behaviour fully driven by language-model
    reasoning.

    Parameters
    ----------
    agent_id:
        Unique identifier for this agent.
    initial_goal:
        The starting goal.
    event_bus:
        Event bus for publishing domain events.
    provider:
        An LLM provider instance.  Must expose a
        ``generate(prompt, system_prompt=..., temperature=...) -> str``
        interface.
    config:
        LLM-specific configuration (prompts, temperatures, actions).
        If ``None``, defaults are used.
    constraints:
        Optional constraint set.

    Example
    -------
    ::

        agent = LLMAgent(
            agent_id="llm-agent-1",
            initial_goal=goal,
            event_bus=EventBus(),
            provider=my_provider,
            config=LLMAgentConfig(
                eval_temperature=0.2,
                planning_temperature=0.6,
            ),
        )
    """

    def __init__(
        self,
        agent_id: str,
        initial_goal: Goal,
        event_bus: EventBus,
        provider: Any,
        config: LLMAgentConfig | None = None,
        constraints: ConstraintSet | None = None,
    ) -> None:
        cfg = config or LLMAgentConfig()
        self._provider = provider
        self._llm_config = cfg

        # Build LLM-backed strategies
        evaluator = LLMCriticEvaluator(
            provider=provider,
            system_prompt=cfg.eval_system_prompt,
            temperature=cfg.eval_temperature,
        )
        updater = LLMGoalEditor(
            provider=provider,
            system_prompt=cfg.revision_system_prompt,
            temperature=cfg.revision_temperature,
        )
        planner = LLMPlanner(
            provider=provider,
            available_actions=cfg.available_actions or None,
            system_prompt=cfg.planning_system_prompt,
            temperature=cfg.planning_temperature,
        )

        super().__init__(
            agent_id=agent_id,
            initial_goal=initial_goal,
            event_bus=event_bus,
            evaluator=evaluator,
            updater=updater,
            planner=planner,
            constraints=constraints,
        )

        logger.info(
            "LLMAgent %s initialised with provider=%s, config=%s",
            agent_id,
            type(provider).__name__,
            cfg,
        )

    # -- public accessors ---------------------------------------------------

    @property
    def provider(self) -> Any:
        """The LLM provider backing this agent."""
        return self._provider

    @property
    def llm_config(self) -> LLMAgentConfig:
        """The LLM-specific configuration for this agent."""
        return self._llm_config

    # -- factory methods ----------------------------------------------------

    @classmethod
    def from_provider(
        cls,
        agent_id: str,
        provider: Any,
        target_values: tuple[float, ...],
        directions: tuple[Direction, ...] | None = None,
        weights: tuple[float, ...] | None = None,
        goal_name: str = "",
        event_bus: EventBus | None = None,
        config: LLMAgentConfig | None = None,
        constraints: ConstraintSet | None = None,
    ) -> LLMAgent:
        """Convenience factory: create an LLMAgent from raw objective parameters.

        Builds the :class:`Goal` and :class:`ObjectiveVector` from the
        provided target values, then constructs the agent with the given
        LLM provider.

        Parameters
        ----------
        agent_id:
            Unique identifier.
        provider:
            LLM provider instance.
        target_values:
            Target values for the objective vector.
        directions:
            Optimization direction per dimension.  Defaults to ``APPROACH``.
        weights:
            Optional per-dimension weights.
        goal_name:
            Optional name for the initial goal.
        event_bus:
            Event bus.  If ``None``, a new one is created.
        config:
            LLM-specific configuration.
        constraints:
            Optional constraint set.

        Returns
        -------
        LLMAgent
            A fully configured LLM-powered teleological agent.
        """
        dirs = directions or tuple(Direction.APPROACH for _ in target_values)
        objective = ObjectiveVector(
            values=target_values,
            directions=dirs,
            weights=weights,
        )
        goal = Goal(
            name=goal_name or f"{agent_id}-goal",
            objective=objective,
        )
        bus = event_bus or EventBus()

        return cls(
            agent_id=agent_id,
            initial_goal=goal,
            event_bus=bus,
            provider=provider,
            config=config,
            constraints=constraints,
        )

    # -- hooks (LLM-specific overrides) -------------------------------------

    def on_goal_revised(self, revision: Any) -> None:
        """Log LLM-driven goal revision at INFO level."""
        logger.info(
            "LLMAgent %s: LLM-driven goal revision (%s -> %s), reason: %s",
            self.id,
            revision.previous_goal_id,
            revision.new_goal_id,
            revision.reason,
        )

    def on_reflection_triggered(self) -> None:
        """Log reflection with LLM context."""
        logger.debug(
            "LLMAgent %s: reflection triggered at step %d "
            "(provider=%s, score=%.3f)",
            self.id,
            self.step_count,
            type(self._provider).__name__,
            self._last_eval.score if self._last_eval else 0.0,
        )

    # -- dunder helpers -----------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"LLMAgent("
            f"id={self.id!r}, "
            f"state={self._state.value}, "
            f"goal={self.current_goal.goal_id!r}, "
            f"provider={type(self._provider).__name__}, "
            f"steps={self._step_count})"
        )
