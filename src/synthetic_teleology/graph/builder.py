"""Fluent builder for assembling a compiled teleological graph.

``GraphBuilder`` supports two modes:

1. **LLM Mode** (new in v1.0): Call ``.with_model()`` and ``.with_goal("description")``.
   All services default to LLM-backed implementations.
2. **Numeric Mode** (backward compatible): Call ``.with_objective((values,))``.
   Falls back to ``NumericEvaluator``, ``GreedyPlanner``, ``ThresholdUpdater``.

The builder detects which mode based on whether ``.with_model()`` was called.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.enums import Direction
from synthetic_teleology.domain.values import ActionSpec, ObjectiveVector
from synthetic_teleology.graph.graph import build_teleological_graph
from synthetic_teleology.services.constraint_engine import (
    ConstraintPipeline,
    PolicyFilter,
)
from synthetic_teleology.services.evaluation import BaseEvaluator, NumericEvaluator
from synthetic_teleology.services.goal_revision import BaseGoalUpdater, ThresholdUpdater
from synthetic_teleology.services.planning import BasePlanner, GreedyPlanner


def _build_default_action_space(dimensions: int, step_size: float = 0.5) -> list[ActionSpec]:
    """Generate a default action space for an N-dimensional continuous space."""
    actions: list[ActionSpec] = []
    for d in range(dimensions):
        effect_pos = tuple(step_size if i == d else 0.0 for i in range(dimensions))
        actions.append(
            ActionSpec(
                name=f"step_dim{d}_pos",
                parameters={"effect": effect_pos, "delta": effect_pos},
            )
        )
        effect_neg = tuple(-step_size if i == d else 0.0 for i in range(dimensions))
        actions.append(
            ActionSpec(
                name=f"step_dim{d}_neg",
                parameters={"effect": effect_neg, "delta": effect_neg},
            )
        )
    noop_effect = tuple(0.0 for _ in range(dimensions))
    actions.append(
        ActionSpec(name="noop", parameters={"effect": noop_effect, "delta": noop_effect})
    )
    return actions


class GraphBuilder:
    """Fluent builder for assembling a compiled teleological LangGraph.

    Example — LLM mode::

        from langchain_anthropic import ChatAnthropic

        app, initial = (
            GraphBuilder("agent-1")
            .with_model(ChatAnthropic(model="claude-sonnet-4-5-20250929"))
            .with_goal("Increase revenue by 20%", criteria=["Revenue > $120k"])
            .with_tools(search_tool, calculator_tool)
            .with_constraints("Never exceed $10k budget", "No weekend actions")
            .with_max_steps(20)
            .build()
        )
        result = app.invoke(initial)

    Example — Numeric mode::

        app, initial = (
            GraphBuilder("agent-1")
            .with_objective((5.0, 5.0))
            .with_environment(perceive_fn=obs, transition_fn=step)
            .build()
        )
        result = app.invoke(initial)
    """

    def __init__(self, agent_id: str) -> None:
        self._agent_id = agent_id
        self._goal: Goal | None = None
        self._model: Any | None = None
        self._tools: list[Any] = []
        self._constraint_descriptions: list[str] = []
        self._num_hypotheses: int = 3
        self._temperature: float = 0.7
        self._evaluator: BaseEvaluator | None = None
        self._updater: BaseGoalUpdater | None = None
        self._planner: BasePlanner | None = None
        self._constraint_checkers: list[Any] = []
        self._checkpointer: Any | None = None
        self._action_step_size: float = 0.5
        self._max_steps: int = 100
        self._goal_achieved_threshold: float = 0.9
        self._perceive_fn: Callable | None = None
        self._act_fn: Callable | None = None
        self._transition_fn: Callable | None = None
        self._metadata: dict[str, Any] = {}
        self._knowledge_store: Any | None = None
        self._audit_trail: Any | None = None
        self._grounding_manager: Any | None = None
        self._human_approval_before: list[str] = []
        self._human_approval_after: list[str] = []
        self._bdi_agent: Any | None = None

    # -- NEW: LLM-first API --------------------------------------------------

    def with_model(self, model: Any) -> GraphBuilder:
        """Set the LangChain chat model (enables LLM mode).

        Parameters
        ----------
        model:
            A ``BaseChatModel`` instance (e.g. ``ChatAnthropic``, ``ChatOpenAI``).
        """
        self._model = model
        return self

    def with_goal(
        self,
        description_or_goal: str | Goal,
        criteria: list[str] | None = None,
        values: tuple[float, ...] | None = None,
        directions: tuple[Direction, ...] | None = None,
        name: str = "",
    ) -> GraphBuilder:
        """Set goal from natural language description or a Goal entity.

        Parameters
        ----------
        description_or_goal:
            Either a natural language goal description (str) or a ``Goal``
            entity directly.
        criteria:
            Success criteria the LLM evaluates against.
        values:
            Optional numeric target values.
        directions:
            Per-dimension optimization directions (if values provided).
        name:
            Goal name (defaults to agent_id + "-goal").
        """
        if isinstance(description_or_goal, Goal):
            self._goal = description_or_goal
            return self

        description = description_or_goal
        objective = None
        if values is not None:
            dirs = directions or tuple(Direction.APPROACH for _ in values)
            objective = ObjectiveVector(values=values, directions=dirs)

        self._goal = Goal(
            name=name or f"{self._agent_id}-goal",
            description=description,
            objective=objective,
            success_criteria=list(criteria or []),
        )
        return self

    def with_tools(self, *tools: Any) -> GraphBuilder:
        """Add LangChain tools the agent can use as actions."""
        self._tools.extend(tools)
        return self

    def with_constraints(self, *constraints: str) -> GraphBuilder:
        """Add natural language constraints for LLM-based checking."""
        self._constraint_descriptions.extend(constraints)
        return self

    def with_num_hypotheses(self, n: int) -> GraphBuilder:
        """How many plan candidates the LLM should generate."""
        self._num_hypotheses = n
        return self

    def with_temperature(self, temp: float) -> GraphBuilder:
        """LLM sampling temperature (higher = more exploratory)."""
        self._temperature = temp
        return self

    # -- PRESERVED: backward-compatible numeric API ---------------------------

    def with_objective(
        self,
        values: tuple[float, ...],
        directions: tuple[Direction, ...] | None = None,
        weights: tuple[float, ...] | None = None,
        goal_name: str = "",
    ) -> GraphBuilder:
        """Create a goal from raw objective parameters (numeric mode)."""
        dirs = directions or tuple(Direction.APPROACH for _ in values)
        objective = ObjectiveVector(values=values, directions=dirs, weights=weights)
        self._goal = Goal(
            name=goal_name or f"{self._agent_id}-goal",
            objective=objective,
        )
        return self

    def with_evaluator(self, evaluator: BaseEvaluator) -> GraphBuilder:
        """Override the default evaluator."""
        self._evaluator = evaluator
        return self

    def with_goal_updater(self, updater: BaseGoalUpdater) -> GraphBuilder:
        """Override the default goal-revision strategy."""
        self._updater = updater
        return self

    def with_planner(self, planner: BasePlanner) -> GraphBuilder:
        """Override the default planner."""
        self._planner = planner
        return self

    def with_constraint_checkers(self, *checkers: Any) -> GraphBuilder:
        """Add constraint checkers (numeric mode)."""
        self._constraint_checkers.extend(checkers)
        return self

    def with_checkpointer(self, checkpointer: Any) -> GraphBuilder:
        """Set a LangGraph checkpointer for persistence."""
        self._checkpointer = checkpointer
        return self

    def with_action_step_size(self, step_size: float) -> GraphBuilder:
        """Set step size for auto-generated action space (numeric mode)."""
        self._action_step_size = step_size
        return self

    def with_max_steps(self, max_steps: int) -> GraphBuilder:
        """Set maximum loop iterations."""
        self._max_steps = max_steps
        return self

    def with_goal_achieved_threshold(self, threshold: float) -> GraphBuilder:
        """Set the eval score threshold for goal achievement."""
        self._goal_achieved_threshold = threshold
        return self

    def with_environment(
        self,
        perceive_fn: Callable | None = None,
        act_fn: Callable | None = None,
        transition_fn: Callable | None = None,
    ) -> GraphBuilder:
        """Set environment interaction functions."""
        if perceive_fn is not None:
            self._perceive_fn = perceive_fn
        if act_fn is not None:
            self._act_fn = act_fn
        if transition_fn is not None:
            self._transition_fn = transition_fn
        return self

    def with_knowledge_store(self, store: Any) -> GraphBuilder:
        """Attach a KnowledgeStore for metacognitive commons."""
        self._knowledge_store = store
        return self

    def with_audit_trail(self, trail: Any) -> GraphBuilder:
        """Attach a GoalAuditTrail for revision tracking."""
        self._audit_trail = trail
        return self

    def with_grounding_manager(self, manager: Any) -> GraphBuilder:
        """Attach an IntentionalGroundingManager."""
        self._grounding_manager = manager
        return self

    def with_human_approval(
        self,
        before: list[str] | None = None,
        after: list[str] | None = None,
    ) -> GraphBuilder:
        """Set human-in-the-loop interrupt points.

        Parameters
        ----------
        before:
            Node names to interrupt *before* execution.
        after:
            Node names to interrupt *after* execution.
        """
        if before:
            self._human_approval_before = before
        if after:
            self._human_approval_after = after
        return self

    def with_bdi_agent(self, bdi_agent: Any) -> GraphBuilder:
        """Attach a BDI agent for BDI-LangGraph bridge."""
        self._bdi_agent = bdi_agent
        return self

    def with_metadata(self, **kwargs: Any) -> GraphBuilder:
        """Set additional metadata."""
        self._metadata.update(kwargs)
        return self

    # -- build ----------------------------------------------------------------

    @property
    def _is_llm_mode(self) -> bool:
        """True if a model has been set (LLM mode)."""
        return self._model is not None

    def build(self) -> tuple[Any, dict[str, Any]]:
        """Validate and build the compiled graph + initial state.

        In LLM mode (``with_model()`` was called), creates LLM-backed services
        by default.  In numeric mode, falls back to ``NumericEvaluator``,
        ``GreedyPlanner``, and ``ThresholdUpdater``.

        Returns
        -------
        tuple[CompiledStateGraph, dict]
            The compiled graph and the initial state dict.

        Raises
        ------
        ValueError
            If no goal has been specified.
        RuntimeError
            If in numeric mode and no perceive_fn has been specified.
        """
        if self._goal is None:
            raise ValueError(
                "GraphBuilder requires a goal. "
                "Call .with_goal(description) or .with_objective(...) before .build()."
            )

        if self._is_llm_mode:
            return self._build_llm_mode()
        return self._build_numeric_mode()

    def _build_llm_mode(self) -> tuple[Any, dict[str, Any]]:
        """Build in LLM mode with LLM-backed services."""
        from synthetic_teleology.services.llm_evaluation import LLMEvaluator
        from synthetic_teleology.services.llm_planning import (
            LLMPlanner as LLMHypothesisPlanner,
        )
        from synthetic_teleology.services.llm_revision import LLMReviser

        model = self._model

        evaluator = self._evaluator or LLMEvaluator(model=model)
        updater = self._updater or LLMReviser(model=model)
        planner = self._planner or LLMHypothesisPlanner(
            model=model,
            tools=self._tools,
            num_hypotheses=self._num_hypotheses,
            temperature=self._temperature,
        )

        # Build constraint pipeline
        checkers = list(self._constraint_checkers)
        if self._constraint_descriptions:
            from synthetic_teleology.services.llm_constraints import LLMConstraintChecker

            checkers.append(
                LLMConstraintChecker(
                    model=model,
                    constraints=self._constraint_descriptions,
                )
            )
        pipeline = ConstraintPipeline(checkers=checkers)
        policy_filter = PolicyFilter(pipeline)

        # Create a default perceive_fn for LLM mode if not provided
        perceive_fn = self._perceive_fn
        if perceive_fn is None:
            import time

            from synthetic_teleology.domain.values import StateSnapshot

            def _default_llm_perceive() -> StateSnapshot:
                return StateSnapshot(
                    timestamp=time.time(),
                    observation="Awaiting observation from environment or tools.",
                )

            perceive_fn = _default_llm_perceive

        app = build_teleological_graph(
            checkpointer=self._checkpointer,
            interrupt_before=self._human_approval_before or None,
            interrupt_after=self._human_approval_after or None,
            enable_grounding=self._grounding_manager is not None,
        )

        initial_state: dict[str, Any] = {
            "step": 0,
            "max_steps": self._max_steps,
            "goal_achieved_threshold": self._goal_achieved_threshold,
            "goal": self._goal,
            "evaluator": evaluator,
            "goal_updater": updater,
            "planner": planner,
            "constraint_pipeline": pipeline,
            "policy_filter": policy_filter,
            "model": model,
            "tools": self._tools,
            "num_hypotheses": self._num_hypotheses,
            "perceive_fn": perceive_fn,
            "act_fn": self._act_fn,
            "transition_fn": self._transition_fn,
            "events": [],
            "goal_history": [],
            "eval_history": [],
            "action_history": [],
            "reasoning_trace": [],
            "metadata": self._metadata,
        }

        if self._knowledge_store is not None:
            initial_state["knowledge_store"] = self._knowledge_store
        if self._audit_trail is not None:
            initial_state["audit_trail"] = self._audit_trail
        if self._grounding_manager is not None:
            initial_state["grounding_manager"] = self._grounding_manager

        return app, initial_state

    def _build_numeric_mode(self) -> tuple[Any, dict[str, Any]]:
        """Build in numeric mode (backward compatible with v0.2.x)."""
        if self._perceive_fn is None:
            raise RuntimeError(
                "GraphBuilder requires a perceive_fn in numeric mode. "
                "Call .with_environment(perceive_fn=...) before .build()."
            )

        evaluator = self._evaluator or NumericEvaluator()
        updater = self._updater or ThresholdUpdater()

        if self._planner is not None:
            planner = self._planner
        else:
            ndims = (
                self._goal.objective.dimension
                if self._goal.objective is not None
                else 2
            )
            action_space = _build_default_action_space(ndims, self._action_step_size)
            planner = GreedyPlanner(action_space=action_space)

        pipeline = ConstraintPipeline(checkers=self._constraint_checkers)
        policy_filter = PolicyFilter(pipeline)

        app = build_teleological_graph(
            checkpointer=self._checkpointer,
            interrupt_before=self._human_approval_before or None,
            interrupt_after=self._human_approval_after or None,
            enable_grounding=self._grounding_manager is not None,
        )

        initial_state: dict[str, Any] = {
            "step": 0,
            "max_steps": self._max_steps,
            "goal_achieved_threshold": self._goal_achieved_threshold,
            "goal": self._goal,
            "evaluator": evaluator,
            "goal_updater": updater,
            "planner": planner,
            "constraint_pipeline": pipeline,
            "policy_filter": policy_filter,
            "perceive_fn": self._perceive_fn,
            "act_fn": self._act_fn,
            "transition_fn": self._transition_fn,
            "events": [],
            "goal_history": [],
            "eval_history": [],
            "action_history": [],
            "reasoning_trace": [],
            "metadata": self._metadata,
        }

        if self._knowledge_store is not None:
            initial_state["knowledge_store"] = self._knowledge_store
        if self._audit_trail is not None:
            initial_state["audit_trail"] = self._audit_trail
        if self._grounding_manager is not None:
            initial_state["grounding_manager"] = self._grounding_manager

        return app, initial_state

    def __repr__(self) -> str:
        parts = [f"agent_id={self._agent_id!r}"]
        if self._model is not None:
            parts.append(f"model={type(self._model).__name__}")
        if self._goal is not None:
            desc = self._goal.description[:30] if self._goal.description else self._goal.goal_id
            parts.append(f"goal={desc!r}")
        mode = "llm" if self._is_llm_mode else "numeric"
        parts.append(f"mode={mode}")
        return f"GraphBuilder({', '.join(parts)})"
