"""Graph wiring for the Data Pipeline Fixer agent.

Uses LLM mode (with_model + with_goal) with a custom PipelineEvaluator
that reads simulated PipelineState. The LLM planner and reviser are
provided via MockStructuredChatModel so the example runs without API keys.

Demonstrates evolving constraints: an EvolvingConstraintManager with its
own mock model proposes new constraints as schema drift is detected.

Mock response ordering (main model):
  Steps 1-7:  PlanningOutput  (normal monitoring)
  Step 8:     RevisionOutput  (schema drift triggers revision)
  Steps 9-25: PlanningOutput  (repair and adaptation)

Evolving constraint manager mock:
  Returns JSON strings for constraint evolution proposals.
"""

import json
import os

from synthetic_teleology.graph import GraphBuilder, WorkingMemory
from synthetic_teleology.services.evolving_constraints import EvolvingConstraintManager
from synthetic_teleology.services.llm_planning import (
    ActionProposal,
    PlanHypothesis,
    PlanningOutput,
)
from synthetic_teleology.services.llm_revision import RevisionOutput
from synthetic_teleology.testing import MockStructuredChatModel

from .models import PipelineState
from .strategies import BudgetConstraintChecker, PipelineEvaluator, SafetyConstraintChecker
from .tools import build_tools

# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

def _get_model():
    """Get a real or mock LLM model.

    Tries Anthropic, then OpenAI. Falls back to None so the caller
    can build a domain-specific mock.
    """
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0.3)
        except ImportError:
            pass
    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model="gpt-4o", temperature=0.3)
        except ImportError:
            pass
    return None


# ---------------------------------------------------------------------------
# Mock builders
# ---------------------------------------------------------------------------

def _make_monitoring_plan(component: str, step_label: str) -> PlanningOutput:
    """Create a PlanningOutput for routine health monitoring."""
    return PlanningOutput(
        hypotheses=[
            PlanHypothesis(
                actions=[
                    ActionProposal(
                        name=f"check_health_{step_label}",
                        description=f"Check health of {component}",
                        tool_name="check_health",
                        parameters={"component": component},
                        cost=0.0,
                    ),
                ],
                reasoning=f"Routine health check on {component}",
                expected_outcome="Health metrics within normal range",
                confidence=0.85,
            ),
        ],
        selected_index=0,
        selection_reasoning=f"Standard monitoring cycle for {component}",
    )


def _make_diagnostic_plan(target: str) -> PlanningOutput:
    """Create a PlanningOutput for running diagnostics."""
    return PlanningOutput(
        hypotheses=[
            PlanHypothesis(
                actions=[
                    ActionProposal(
                        name=f"diagnose_{target}",
                        description=f"Run diagnostic on {target}",
                        tool_name="run_diagnostic",
                        parameters={"target": target},
                        cost=0.0,
                    ),
                ],
                reasoning=f"Schema drift detected, diagnosing {target}",
                expected_outcome="Identify root cause of schema incompatibility",
                confidence=0.80,
            ),
        ],
        selected_index=0,
        selection_reasoning=f"Need to understand drift impact on {target}",
    )


def _make_fix_plan(fix_type: str, target: str) -> PlanningOutput:
    """Create a PlanningOutput for applying a fix."""
    return PlanningOutput(
        hypotheses=[
            PlanHypothesis(
                actions=[
                    ActionProposal(
                        name=f"fix_{fix_type}_{target}",
                        description=f"Apply {fix_type} to {target}",
                        tool_name="apply_fix",
                        parameters={"fix_type": fix_type, "target": target},
                        cost=1.0,
                    ),
                ],
                reasoning=f"Apply {fix_type} to repair {target}",
                expected_outcome=f"Improve health after {fix_type} on {target}",
                confidence=0.75,
            ),
        ],
        selected_index=0,
        selection_reasoning=f"Best available fix for {target}: {fix_type}",
    )


def _make_rollback_plan(target: str) -> PlanningOutput:
    """Create a PlanningOutput for rolling back a component."""
    return PlanningOutput(
        hypotheses=[
            PlanHypothesis(
                actions=[
                    ActionProposal(
                        name=f"rollback_{target}",
                        description=f"Rollback {target} to v1.0",
                        tool_name="rollback",
                        parameters={"target": target, "version": "v1.0"},
                        cost=1.0,
                    ),
                ],
                reasoning=f"Roll back {target} to known-good schema v1.0",
                expected_outcome="Stabilize pipeline while planning migration",
                confidence=0.70,
            ),
        ],
        selected_index=0,
        selection_reasoning=f"Rollback {target} to restore immediate stability",
    )


def _build_main_mock() -> MockStructuredChatModel:
    """Build the main mock model for planner + reviser.

    The mock ``_call_index`` is shared across **all** ``with_structured_output``
    runnables (planner and reviser both use the same model instance).

    Call flow per graph step:
    - Normal step (score > -0.3): planner only = 1 call
    - Revision step (score <= -0.3): reviser + planner = 2 calls

    Timeline detail:
    ---------------------------------------------------------------
    Steps 1-8  : planner only (1 call each)             -> idx 0-7
    Step 8 act : check_health call #8 triggers schema drift
    Step 9     : eval drops to -0.4 -> revision fires
                 reviser  (idx 8  = RevisionOutput should_revise=True)
                 planner  (idx 9  = PlanningOutput diagnose orders)
    Steps 10-12: eval still -0.4 -> revision fires each step
                 step 10: reviser (idx 10 = RevisionOutput should_revise=False)
                          planner (idx 11 = PlanningOutput rollback orders)
                 step 11: reviser (idx 12 = RevisionOutput should_revise=False)
                          planner (idx 13 = PlanningOutput add_adapter events)
                 step 12: reviser (idx 14 = RevisionOutput should_revise=False)
                          planner (idx 15 = PlanningOutput schema_migration orders)
    Step 13+   : eval >= -0.3, planner only             -> idx 16, 17, ...
    ---------------------------------------------------------------
    """
    responses: list = []

    # Indices 0-7: steps 1-8 routine monitoring (planner calls)
    components = [
        "pipeline", "ingestion", "transform", "load",
        "users_table", "orders_table", "events_table",
        "pipeline",  # step 8: last check before drift triggers in act
    ]
    for comp in components:
        responses.append(_make_monitoring_plan(comp, comp))

    # --- Step 9: revision fires (score -0.4) ---
    # Index 8: reviser -> should_revise=True
    responses.append(
        RevisionOutput(
            should_revise=True,
            reasoning=(
                "Schema drift detected. Pipeline health degraded significantly "
                "(health=0.55, error_rate=0.18). Upstream producer migrated to "
                "schema v2.0 without coordination. Must adapt pipeline to new "
                "schema while maintaining backward compatibility."
            ),
            new_description=(
                "Adapt pipeline to schema v2.0 while maintaining "
                "backward compatibility and restoring health above 95%"
            ),
            new_criteria=[
                "Pipeline health score above 0.95",
                "Error rate below 2%",
                "Throughput within 90% of baseline",
                "All schema validations passing",
                "Complete schema v2.0 migration with backward compatibility",
            ],
        )
    )
    # Index 9: planner after revision
    responses.append(_make_diagnostic_plan("orders"))

    # --- Step 10: score still -0.4 -> revision fires again ---
    # Index 10: reviser -> should_revise=False (already revised)
    responses.append(
        RevisionOutput(
            should_revise=False,
            reasoning="Goal already revised to include schema migration. Continuing repairs.",
        )
    )
    # Index 11: planner
    responses.append(_make_rollback_plan("orders"))

    # --- Step 11: score still -0.4 -> revision fires again ---
    # Index 12: reviser -> should_revise=False
    responses.append(
        RevisionOutput(
            should_revise=False,
            reasoning="Repairs in progress. Health improving. No further goal revision needed.",
        )
    )
    # Index 13: planner
    responses.append(_make_fix_plan("add_adapter", "events"))

    # --- Step 12: score still -0.4 -> revision fires again ---
    # Index 14: reviser -> should_revise=False
    responses.append(
        RevisionOutput(
            should_revise=False,
            reasoning="Schema migration underway. Awaiting health recovery.",
        )
    )
    # Index 15: planner
    responses.append(_make_fix_plan("schema_migration", "orders"))

    # --- Steps 13+: score rises above -0.3, planner only ---
    responses.append(_make_fix_plan("schema_migration", "events"))          # idx 16 step 13
    responses.append(_make_fix_plan("retry_failed", "orders"))              # idx 17 step 14
    responses.append(_make_fix_plan("retry_failed", "events"))              # idx 18 step 15
    responses.append(_make_fix_plan("increase_throughput", "pipeline"))     # idx 19 step 16
    responses.append(_make_monitoring_plan("pipeline", "post_fix_1"))       # idx 20 step 17
    responses.append(_make_fix_plan("add_adapter", "sessions"))             # idx 21 step 18
    responses.append(_make_fix_plan("schema_migration", "sessions"))        # idx 22 step 19
    responses.append(_make_fix_plan("retry_failed", "pipeline"))            # idx 23 step 20
    responses.append(_make_monitoring_plan("pipeline", "post_fix_2"))       # idx 24 step 21
    responses.append(_make_fix_plan("increase_throughput", "ingestion"))    # idx 25 step 22
    responses.append(_make_monitoring_plan("pipeline", "final"))            # idx 26 step 23
    responses.append(_make_fix_plan("add_adapter", "analytics"))            # idx 27 step 24
    responses.append(_make_fix_plan("add_adapter", "payments"))             # idx 28 step 25

    return MockStructuredChatModel(structured_responses=responses)


def _build_evolving_mock() -> MockStructuredChatModel:
    """Build a mock model for the EvolvingConstraintManager.

    The manager calls model.invoke(prompt) and parses JSON from the
    response content. So we provide plain JSON strings that will be
    returned via _generate as AIMessage content.
    """
    evolution_responses = [
        # First evolution: add schema monitoring constraint
        json.dumps({
            "evolutions": [
                {
                    "type": "add",
                    "constraint": "Schema changes must be validated before deployment",
                    "reasoning": (
                        "Schema drift caused pipeline degradation; "
                        "proactive validation needed"
                    ),
                    "confidence": 0.85,
                    "previous": None,
                },
            ],
            "overall_reasoning": (
                "Adding schema validation constraint based on observed drift pattern"
            ),
        }),
        # Second evolution: modify health threshold
        json.dumps({
            "evolutions": [
                {
                    "type": "modify",
                    "constraint": "Pipeline health must stay above 90% during migration",
                    "reasoning": (
                        "Original 80% threshold too lenient "
                        "during active schema migration"
                    ),
                    "confidence": 0.80,
                    "previous": "Pipeline health must stay above 80%",
                },
            ],
            "overall_reasoning": (
                "Tightening health constraint during active migration period"
            ),
        }),
        # Third evolution: add backward compat constraint
        json.dumps({
            "evolutions": [
                {
                    "type": "add",
                    "constraint": (
                        "All schema migrations must maintain "
                        "backward compatibility for 7 days"
                    ),
                    "reasoning": "Downstream consumers need time to adapt to schema changes",
                    "confidence": 0.90,
                    "previous": None,
                },
            ],
            "overall_reasoning": (
                "Ensuring backward compatibility window for downstream consumers"
            ),
        }),
        # Fourth evolution: no changes needed
        json.dumps({
            "evolutions": [],
            "overall_reasoning": "Current constraints are adequate; no changes needed",
        }),
    ]

    return MockStructuredChatModel(structured_responses=evolution_responses)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_pipeline_agent(max_steps: int = 25):
    """Build a LangGraph data pipeline fixer agent.

    Returns ``(app, initial_state, pipeline_state)`` tuple.

    The agent uses:
    - Custom PipelineEvaluator (deterministic, reads PipelineState)
    - LLM planner + reviser via MockStructuredChatModel
    - Simulated tools that mutate PipelineState
    - EvolvingConstraintManager with its own mock model
    """
    pipeline_state = PipelineState()
    tools = build_tools(pipeline_state)

    # Working memory bridges perception <-> action
    memory = WorkingMemory(
        initial_context=(
            f"Data pipeline monitoring agent. "
            f"Pipeline has {len(pipeline_state.tables)} tables: "
            f"{', '.join(pipeline_state.tables)}. "
            f"Initial health: {pipeline_state.health_score:.2f}, "
            f"error_rate: {pipeline_state.error_rate:.2f}, "
            f"throughput: {pipeline_state.throughput:.0f} rec/s, "
            f"schema: {pipeline_state.schema_version}."
        ),
        max_entries=30,
    )

    # --- Models ---
    model = _get_model() or _build_main_mock()

    # Evolving constraint manager with its own mock
    evolving_model = _get_model() or _build_evolving_mock()
    constraint_manager = EvolvingConstraintManager(
        model=evolving_model,
        initial_constraints=[
            "Pipeline health must stay above 80%",
            "Error rate must stay below 5%",
        ],
        evolution_frequency=3,
    )

    # --- Strategies ---
    evaluator = PipelineEvaluator(pipeline_state, baseline_throughput=1000.0)
    budget_checker = BudgetConstraintChecker(pipeline_state, max_fixes=15)
    safety_checker = SafetyConstraintChecker()

    # --- Build graph ---
    app, initial_state = (
        GraphBuilder("pipeline-fixer")
        .with_model(model)
        .with_goal(
            "Maintain pipeline health above 95% accuracy",
            criteria=[
                "Pipeline health score above 0.95",
                "Error rate below 2%",
                "Throughput within 90% of baseline",
                "All schema validations passing",
            ],
        )
        .with_tools(*tools)
        .with_evaluator(evaluator)
        .with_constraint_checkers(budget_checker, safety_checker)
        .with_evolving_constraints(constraint_manager)
        .with_environment(
            perceive_fn=memory.perceive,
            transition_fn=memory.record,
        )
        .with_max_steps(max_steps)
        .with_num_hypotheses(1)
        .build()
    )

    return app, initial_state, pipeline_state
