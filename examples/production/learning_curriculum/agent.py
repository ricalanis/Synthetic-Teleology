"""Graph wiring for the Adaptive Learning Curriculum agent.

Uses LLM mode (with_model + with_goal) with a custom evaluator that reads
the simulated CurriculumState directly.  The LLM planner and reviser are
backed by MockStructuredChatModel, providing PlanningOutput and
RevisionOutput responses in a fixed sequence that drives curriculum
progression, quiz failure, and remedial revision.

In simulated mode (no API key), uses MockStructuredChatModel.
"""

import os

from synthetic_teleology.graph import GraphBuilder, WorkingMemory
from synthetic_teleology.infrastructure.knowledge_store import KnowledgeStore
from synthetic_teleology.services.audit_trail import GoalAuditTrail

from .models import REACT_CURRICULUM, CurriculumState, create_default_learner
from .strategies import CurriculumEvaluator, PrerequisiteChecker, TimeBudgetChecker
from .tools import create_tools

# ---------------------------------------------------------------------------
# Mock response factory
# ---------------------------------------------------------------------------


def _build_mock_responses() -> list:
    """Build the ordered mock responses for ~35-step curriculum run.

    The mock model is only called for PlanningOutput (by LLMPlanner) and
    RevisionOutput (by LLMReviser).  The custom CurriculumEvaluator does
    NOT call the model.

    Flow per step (when score > -0.3, no revision):
        perceive -> evaluate (custom) -> check_constraints
        -> plan (MOCK) -> filter -> act -> reflect

    Flow when score <= -0.3 (revision):
        perceive -> evaluate (custom) -> revise (MOCK)
        -> check_constraints -> plan (MOCK) -> filter -> act -> reflect

    Sequence:
        Steps  1-14: PlanningOutput x14  (progressive topics)
        Step  15:     RevisionOutput x1, PlanningOutput x1  (state_management failure + remedial)
        Steps 16-35:  PlanningOutput x20  (remedial + continued progression)
    """
    from synthetic_teleology.services.llm_planning import (
        ActionProposal,
        PlanHypothesis,
        PlanningOutput,
    )
    from synthetic_teleology.services.llm_revision import RevisionOutput

    def _plan(tool_name: str, topic: str, reasoning: str, **extra_params) -> PlanningOutput:
        """Helper to build a single-hypothesis PlanningOutput."""
        params: dict = {"topic": topic}
        params.update(extra_params)
        return PlanningOutput(
            hypotheses=[
                PlanHypothesis(
                    actions=[
                        ActionProposal(
                            name=f"{tool_name}_{topic}",
                            description=f"{tool_name.replace('_', ' ').title()} for {topic}",
                            tool_name=tool_name,
                            parameters=params,
                        ),
                    ],
                    reasoning=reasoning,
                    expected_outcome=f"Progress on {topic}",
                    confidence=0.8,
                ),
            ],
            selected_index=0,
            selection_reasoning=reasoning,
        )

    responses: list = []

    # ======================================================================
    # Mastery tracking (initial values from create_default_learner):
    #   jsx_basics=0.80, components=0.70, props=0.55,
    #   state_management=0.20, hooks=0.30, effects=0.30,
    #   context=0.15, routing=0.40, forms=0.30, testing=0.05
    #
    # Tool effects: generate_lesson +0.10, search_resources +0.05
    # Quiz passes when mastery >= 0.50
    # ======================================================================

    # ---- Steps 1-3: jsx_basics -------------------------------------------
    # Mastery: 0.80 -> lesson 0.90 -> quiz PASSES (0.90) -> resources 0.95
    responses.append(_plan(
        "generate_lesson", "jsx_basics",
        "Start with the foundation: JSX syntax and expressions",
    ))
    responses.append(_plan(
        "create_quiz", "jsx_basics",
        "Verify JSX basics comprehension before moving on",
    ))
    responses.append(_plan(
        "search_resources", "jsx_basics",
        "Supplement JSX learning with additional materials",
        difficulty="beginner",
    ))

    # ---- Steps 4-6: components -------------------------------------------
    # Mastery: 0.70 -> lesson 0.80 -> quiz PASSES (0.80) -> resources 0.85
    responses.append(_plan(
        "generate_lesson", "components",
        "Teach component structure and composition patterns",
    ))
    responses.append(_plan(
        "create_quiz", "components",
        "Assess understanding of React components",
    ))
    responses.append(_plan(
        "search_resources", "components",
        "Reinforce component concepts with practice exercises",
        difficulty="beginner",
    ))

    # ---- Steps 7-9: props ------------------------------------------------
    # Mastery: 0.55 -> lesson 0.65 -> quiz PASSES (0.65) -> resources 0.70
    responses.append(_plan(
        "generate_lesson", "props",
        "Cover props passing, destructuring, and prop types",
    ))
    responses.append(_plan(
        "create_quiz", "props",
        "Test props knowledge before advancing to state",
    ))
    responses.append(_plan(
        "search_resources", "props",
        "Additional exercises on prop patterns",
        difficulty="intermediate",
    ))

    # ---- Steps 10-14: state_management (prep + fail) ---------------------
    # Mastery: 0.20 -> lesson 0.30 -> assess (no change) -> lesson 0.40
    #          -> resources 0.45 -> quiz FAILS (0.45 < 0.50)
    responses.append(_plan(
        "generate_lesson", "state_management",
        "Introduce useState and state update patterns",
    ))
    responses.append(_plan(
        "assess_performance", "state_management",
        "Check overall learner progress before state management quiz",
    ))
    responses.append(_plan(
        "generate_lesson", "state_management",
        "Reinforce state management with more examples before quiz",
    ))
    responses.append(_plan(
        "search_resources", "state_management",
        "Find additional state management resources",
        difficulty="intermediate",
    ))
    responses.append(_plan(
        "create_quiz", "state_management",
        "Quiz on state management — critical checkpoint",
    ))

    # ---- Step 15: REVISION -----------------------------------------------
    # Quiz at step 14 fails (mastery 0.45 < 0.50).
    # CurriculumEvaluator returns score = -0.5, triggering should_revise.
    # LLMReviser consumes RevisionOutput, then flow resumes with PlanningOutput.
    responses.append(RevisionOutput(
        should_revise=True,
        reasoning=(
            "Learner failed state_management quiz, revealing a fundamental gap. "
            "Need remedial modules before advancing."
        ),
        new_description=(
            "Teach user React fundamentals with remedial state management modules"
        ),
        new_criteria=[
            "Complete core topics with mastery >= 70%",
            "Pass all topic quizzes",
            "Cover at least 7 of 10 curriculum topics",
            "Maintain learning efficiency (lessons/quizzes ratio)",
            "Complete remedial state management exercises with mastery >= 60%",
        ],
    ))
    # After revision -> check_constraints -> plan (consumes next PlanningOutput)
    responses.append(_plan(
        "generate_lesson", "state_management",
        "Remedial lesson: revisit state management fundamentals from scratch",
    ))

    # ---- Steps 16-18: remedial + retry -----------------------------------
    # Mastery: 0.45 -> lesson 0.55 -> resources 0.60 -> lesson 0.70
    #          -> quiz PASSES (0.70)
    responses.append(_plan(
        "search_resources", "state_management",
        "Find beginner-friendly state management tutorials for remediation",
        difficulty="beginner",
    ))
    responses.append(_plan(
        "generate_lesson", "state_management",
        "Hands-on state management workshop with guided exercises",
    ))
    responses.append(_plan(
        "create_quiz", "state_management",
        "Retry state management quiz after remedial modules",
    ))

    # ---- Steps 19-22: hooks (need extra lessons) -------------------------
    # Mastery: 0.30 -> lesson 0.40 -> lesson 0.50 -> resources 0.55
    #          -> quiz PASSES (0.55)
    responses.append(_plan(
        "generate_lesson", "hooks",
        "Teach custom hooks and the rules of hooks",
    ))
    responses.append(_plan(
        "generate_lesson", "hooks",
        "Advanced hook patterns: useReducer, custom hooks",
    ))
    responses.append(_plan(
        "search_resources", "hooks",
        "Hooks practice exercises and cheat sheets",
        difficulty="intermediate",
    ))
    responses.append(_plan(
        "create_quiz", "hooks",
        "Assess hooks understanding after intensive prep",
    ))

    # ---- Steps 23-26: effects (need extra lessons) -----------------------
    # Mastery: 0.30 -> lesson 0.40 -> lesson 0.50 -> resources 0.55
    #          -> quiz PASSES (0.55)
    responses.append(_plan(
        "generate_lesson", "effects",
        "Cover useEffect, cleanup, and dependency arrays",
    ))
    responses.append(_plan(
        "generate_lesson", "effects",
        "Advanced effects: race conditions, abort controllers",
    ))
    responses.append(_plan(
        "search_resources", "effects",
        "Supplementary materials on effect patterns",
        difficulty="intermediate",
    ))
    responses.append(_plan(
        "create_quiz", "effects",
        "Quiz on side effects and lifecycle",
    ))

    # ---- Steps 27-30: routing (moderate prep needed) ---------------------
    # Mastery: 0.40 -> lesson 0.50 -> resources 0.55 -> quiz PASSES (0.55)
    #          -> assess performance
    responses.append(_plan(
        "generate_lesson", "routing",
        "Teach React Router and navigation patterns",
    ))
    responses.append(_plan(
        "search_resources", "routing",
        "Additional routing exercises and examples",
        difficulty="intermediate",
    ))
    responses.append(_plan(
        "create_quiz", "routing",
        "Verify routing knowledge",
    ))
    responses.append(_plan(
        "assess_performance", "routing",
        "Full progress assessment — 7 topics targeted",
    ))

    # ---- Steps 31-34: forms (moderate prep needed) -----------------------
    # Mastery: 0.30 -> lesson 0.40 -> lesson 0.50 -> resources 0.55
    #          -> quiz PASSES (0.55)
    responses.append(_plan(
        "generate_lesson", "forms",
        "Controlled and uncontrolled forms, validation patterns",
    ))
    responses.append(_plan(
        "generate_lesson", "forms",
        "Form libraries and advanced validation techniques",
    ))
    responses.append(_plan(
        "search_resources", "forms",
        "Form pattern exercises and best practices",
        difficulty="intermediate",
    ))
    responses.append(_plan(
        "create_quiz", "forms",
        "Forms and input handling quiz",
    ))

    # ---- Step 35: final assessment ---------------------------------------
    responses.append(_plan(
        "assess_performance", "forms",
        "Final comprehensive learner assessment",
    ))

    return responses


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def _get_model():
    """Get a real LLM model if API keys are available, else return None."""
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0.5)
        except ImportError:
            pass
    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model="gpt-4o", temperature=0.5)
        except ImportError:
            pass
    return None  # triggers mock path


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def _build_mock_model():
    """Build a MockStructuredChatModel with curriculum progression responses."""
    from synthetic_teleology.testing import MockStructuredChatModel

    return MockStructuredChatModel(structured_responses=_build_mock_responses())


def build_curriculum_agent(max_steps: int = 35):
    """Build a LangGraph curriculum agent.

    Returns ``(app, initial_state, curriculum_state, knowledge_store, audit_trail)``
    tuple.
    """
    # Domain state
    learner = create_default_learner()
    curriculum_state = CurriculumState(
        available_topics=list(REACT_CURRICULUM),
        learner=learner,
    )

    # Tools bound to the shared state
    tools = create_tools(curriculum_state)

    # Strategies
    evaluator = CurriculumEvaluator(curriculum_state)
    prereq_checker = PrerequisiteChecker(curriculum_state)
    time_checker = TimeBudgetChecker(curriculum_state, max_time=180.0)

    # Working memory for the perception-action loop
    topic_names = [t.name for t in REACT_CURRICULUM]
    initial_context = (
        f"React Curriculum ({len(REACT_CURRICULUM)} topics): "
        f"{', '.join(topic_names)}\n"
        f"Learner profile: strong on basics (jsx_basics=0.8, components=0.7), "
        f"weak on advanced (state_management=0.2, hooks=0.3).\n"
        f"Goal: bring learner to competency across core topics."
    )
    memory = WorkingMemory(initial_context=initial_context, max_entries=40)

    # Infrastructure
    knowledge_store = KnowledgeStore()
    audit_trail = GoalAuditTrail(knowledge_store=knowledge_store)

    # Seed initial knowledge
    knowledge_store.put(
        key="learner_initial_profile",
        value={
            "strengths": dict(learner.strengths),
            "weaknesses": list(learner.weaknesses),
        },
        source="curriculum_agent",
        tags=("learner", "profile"),
    )

    # Model
    model = _get_model() or _build_mock_model()

    app, initial_state = (
        GraphBuilder("learning-curriculum")
        .with_model(model)
        .with_goal(
            "Teach user React fundamentals to competency level",
            criteria=[
                "Complete core topics with mastery >= 70%",
                "Pass all topic quizzes",
                "Cover at least 7 of 10 curriculum topics",
                "Maintain learning efficiency (lessons/quizzes ratio)",
            ],
        )
        .with_tools(*tools)
        .with_evaluator(evaluator)
        .with_constraint_checkers(prereq_checker, time_checker)
        .with_knowledge_store(knowledge_store)
        .with_audit_trail(audit_trail)
        .with_environment(
            perceive_fn=memory.perceive,
            transition_fn=memory.record,
        )
        .with_max_steps(max_steps)
        .with_num_hypotheses(1)
        .build()
    )

    return app, initial_state, curriculum_state, knowledge_store, audit_trail
