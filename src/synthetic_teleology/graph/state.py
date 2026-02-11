"""LangGraph state definition for the teleological loop.

Defines ``TeleologicalState``, a ``TypedDict`` that flows through the
LangGraph ``StateGraph``.  Append-only channels use ``Annotated[list, operator.add]``
so that each node can emit new items without overwriting previous entries.

Note: We intentionally do NOT use ``from __future__ import annotations`` because
LangGraph needs to resolve type hints at runtime via ``get_type_hints()``.
"""

import operator
from collections.abc import Callable
from typing import Annotated, Any, TypedDict

from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.values import (
    ActionSpec,
    EvalSignal,
    Hypothesis,
    PolicySpec,
    StateSnapshot,
)
from synthetic_teleology.services.constraint_engine import (
    ConstraintPipeline,
    PolicyFilter,
)
from synthetic_teleology.services.evaluation import BaseEvaluator
from synthetic_teleology.services.goal_revision import BaseGoalUpdater
from synthetic_teleology.services.planning import BasePlanner


def _make_bounded_add(max_size: int) -> Callable[[list, list], list]:
    """Create a reducer that appends and keeps only the last ``max_size`` items.

    Parameters
    ----------
    max_size:
        Maximum number of items to retain.  When the combined list
        exceeds this, the oldest entries (head) are dropped.

    Returns
    -------
    Callable
        A reducer compatible with ``Annotated[list, reducer]``.
    """

    def bounded_add(current: list, new: list) -> list:
        combined = current + new
        if len(combined) > max_size:
            return combined[-max_size:]
        return combined

    return bounded_add


def make_bounded_state(max_history: int = 500) -> type:
    """Create a ``BoundedTeleologicalState`` TypedDict with capped accumulation channels.

    All append-only channels (``events``, ``eval_history``, ``action_history``,
    ``reasoning_trace``, ``action_feedback``, ``goal_history``) are bounded
    to at most ``max_history`` entries each.

    Parameters
    ----------
    max_history:
        Maximum entries per accumulation channel.

    Returns
    -------
    type
        A TypedDict subclass with bounded reducers.
    """
    bounded = _make_bounded_add(max_history)

    class BoundedTeleologicalState(TypedDict, total=False):
        """TeleologicalState with bounded accumulation channels."""

        # -- Loop control
        step: int
        max_steps: int
        goal_achieved_threshold: float
        stop_reason: str

        # -- Core teleological state
        goal: Goal
        observation: str
        state_snapshot: StateSnapshot
        eval_signal: EvalSignal
        hypotheses: list[Hypothesis]
        selected_plan: Hypothesis | None
        policy: PolicySpec
        filtered_policy: PolicySpec
        executed_action: ActionSpec | None

        # -- Constraints
        constraints_ok: bool
        constraint_violations: list[str]
        constraint_assessments: list[dict]

        # -- Injected strategies
        evaluator: BaseEvaluator
        goal_updater: BaseGoalUpdater
        planner: BasePlanner
        constraint_pipeline: ConstraintPipeline
        policy_filter: PolicyFilter

        # -- LLM configuration
        model: Any
        tools: list[Any]
        num_hypotheses: int

        # -- Environment callables
        perceive_fn: Callable
        act_fn: Callable | None
        transition_fn: Callable | None

        # -- Bounded accumulation channels
        events: Annotated[list, bounded]
        goal_history: Annotated[list, bounded]
        eval_history: Annotated[list, bounded]
        action_history: Annotated[list, bounded]
        reasoning_trace: Annotated[list, bounded]
        action_feedback: Annotated[list, bounded]

        # -- Optional integrations
        knowledge_store: Any
        audit_trail: Any
        grounding_manager: Any
        evolving_constraint_manager: Any

        # -- Extensibility
        metadata: dict[str, Any]

    return BoundedTeleologicalState


class TeleologicalState(TypedDict, total=False):
    """State flowing through the teleological LangGraph.

    Fields are grouped into:

    * **Loop control** -- step counter, limits, termination info.
    * **Core teleological state** -- mutated each iteration by nodes.
    * **Constraints** -- result of the constraint check node.
    * **Injected strategies** -- set once at invocation, read by nodes.
    * **Environment callables** -- functions for world interaction.
    * **LLM configuration** -- model, tools, hypothesis count.
    * **Accumulation channels** -- append-reducers for history.
    * **Extensibility** -- free-form metadata dict.
    """

    # -- Loop control --------------------------------------------------------
    step: int
    max_steps: int
    goal_achieved_threshold: float
    stop_reason: str

    # -- Core teleological state (mutated per iteration) ---------------------
    goal: Goal
    observation: str                               # NEW: natural language state description
    state_snapshot: StateSnapshot
    eval_signal: EvalSignal
    hypotheses: list[Hypothesis]                   # NEW: multi-hypothesis plans
    selected_plan: Hypothesis | None         # NEW: chosen plan
    policy: PolicySpec
    filtered_policy: PolicySpec
    executed_action: ActionSpec | None

    # -- Constraints ---------------------------------------------------------
    constraints_ok: bool
    constraint_violations: list[str]
    constraint_assessments: list[dict]             # NEW: soft constraint reasoning

    # -- Injected strategies (set once, read by nodes) -----------------------
    # DEPRECATED: prefer passing strategies as kwargs to build_teleological_graph()
    # for closure-based injection (enables LangGraph checkpointing).
    # These fields are kept for backward compatibility with code that stores
    # strategies in state directly.
    evaluator: BaseEvaluator
    goal_updater: BaseGoalUpdater
    planner: BasePlanner
    constraint_pipeline: ConstraintPipeline
    policy_filter: PolicyFilter

    # -- LLM configuration (set once) ----------------------------------------
    model: Any                                     # NEW: BaseChatModel
    tools: list[Any]                               # NEW: LangChain tools
    num_hypotheses: int                            # NEW: hypothesis count

    # -- Environment callables -----------------------------------------------
    perceive_fn: Callable
    act_fn: Callable | None
    transition_fn: Callable | None

    # -- Accumulation channels (append-reducers) -----------------------------
    events: Annotated[list, operator.add]
    goal_history: Annotated[list, operator.add]
    eval_history: Annotated[list, operator.add]
    action_history: Annotated[list, operator.add]
    reasoning_trace: Annotated[list, operator.add]  # NEW: all LLM reasoning
    action_feedback: Annotated[list, operator.add]  # NEW: structured action results

    # -- Optional integrations ------------------------------------------------
    knowledge_store: Any                             # KnowledgeStore instance
    audit_trail: Any                                 # GoalAuditTrail instance
    grounding_manager: Any                           # IntentionalGroundingManager instance
    evolving_constraint_manager: Any                 # EvolvingConstraintManager instance

    # -- Extensibility -------------------------------------------------------
    metadata: dict[str, Any]
