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

    # -- Optional integrations ------------------------------------------------
    knowledge_store: Any                             # KnowledgeStore instance
    audit_trail: Any                                 # GoalAuditTrail instance
    grounding_manager: Any                           # IntentionalGroundingManager instance

    # -- Extensibility -------------------------------------------------------
    metadata: dict[str, Any]
