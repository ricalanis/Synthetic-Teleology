"""Graph wiring for the Competitive Research agent.

Uses LLM mode (with_model + with_goal) with a custom evaluator that reads
simulated environment state. The LLM (real or mock) drives the planner and
reviser only.

Demonstrates: goal revision, tool routing, knowledge store, audit trail.
"""

import os
from typing import Any

from synthetic_teleology.graph import GraphBuilder, WorkingMemory
from synthetic_teleology.infrastructure.knowledge_store import KnowledgeStore
from synthetic_teleology.services.audit_trail import GoalAuditTrail

from .models import ResearchState
from .strategies import ResearchEvaluator, SourceDiversityChecker
from .tools import DataExtractorTool, DocumentReaderTool, WebSearchTool


def _get_model() -> Any:
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


def _build_mock_model() -> Any:
    """Build a MockStructuredChatModel with realistic research responses.

    The mock drives the LLM planner and reviser. The shared call index
    means we must account for BOTH services consuming from the same list.

    Per step the call order is:
        1. evaluate (custom, no model call)
        2. should_revise edge: if score <= -0.3 -> revise node calls model
        3. plan node calls model
        4. act node executes tool (no model call)

    Score trajectory (score rises as topics accumulate):
        Step 1 eval: 0 findings -> score = -1.0    -> revise + plan = 2 calls
        Step 2 eval: 1 finding  -> score ~ -0.78    -> revise + plan = 2 calls
        Step 3 eval: 2 findings -> score ~ -0.52    -> revise + plan = 2 calls
        Step 4 eval: 3 findings -> score ~ -0.26    -> plan only   = 1 call
        ...steps 4-8: score > -0.3, plan only
        Step 9 eval: pivot_discovered, not analyzed -> score = -0.35
                     -> revise (should_revise=True) + plan = 2 calls
        Step 10+: pivot_analyzed=True after tool, score normal -> plan only
    """
    from synthetic_teleology.services.llm_planning import (
        ActionProposal,
        PlanHypothesis,
        PlanningOutput,
    )
    from synthetic_teleology.services.llm_revision import RevisionOutput
    from synthetic_teleology.testing import MockStructuredChatModel

    def _plan(
        action_name: str,
        action_desc: str,
        tool_name: str,
        params: dict[str, Any],
        reasoning: str,
        expected: str,
        confidence: float = 0.8,
    ) -> PlanningOutput:
        return PlanningOutput(
            hypotheses=[
                PlanHypothesis(
                    actions=[
                        ActionProposal(
                            name=action_name,
                            description=action_desc,
                            tool_name=tool_name,
                            parameters=params,
                            cost=0.0,
                        ),
                    ],
                    reasoning=reasoning,
                    expected_outcome=expected,
                    confidence=confidence,
                ),
            ],
            selected_index=0,
            selection_reasoning=reasoning,
        )

    _no_revise = RevisionOutput(
        should_revise=False,
        reasoning="Research is in early stages, goal is well-specified, no revision needed.",
    )

    responses: list[Any] = [
        # === Step 1: score=-1.0, revise fires (no-op) + plan ===
        _no_revise,
        _plan(
            "search_products", "Search TechRival product portfolio",
            "web_search", {"query": "TechRival product portfolio offerings"},
            "Start by mapping the competitor's product landscape.",
            "Comprehensive view of TechRival's product suite.",
        ),
        # === Step 2: score~-0.78, revise fires (no-op) + plan ===
        _no_revise,
        _plan(
            "extract_market_share", "Extract market share metrics",
            "data_extractor", {"source": "industry_reports", "metric": "market_share"},
            "Quantify competitive positioning with market share data.",
            "Numerical market share breakdown across competitors.",
        ),
        # === Step 3: score~-0.52, revise fires (no-op) + plan ===
        _no_revise,
        _plan(
            "read_10k", "Analyze annual SEC filing",
            "document_reader", {"url": "sec.gov/techrival/10-K/2025"},
            "Deep-dive into financials for revenue and risk analysis.",
            "Detailed financial metrics and risk factors.",
        ),
        # === Step 4: score~-0.26 > -0.3, plan only ===
        _plan(
            "extract_revenue", "Extract revenue breakdown by segment",
            "data_extractor", {"source": "financial_data", "metric": "revenue breakdown"},
            "Understand revenue composition across product lines.",
            "Segment-level revenue data for competitive comparison.",
        ),
        # === Step 5: score~0.0, plan only ===
        _plan(
            "search_leadership", "Research leadership team",
            "web_search", {"query": "TechRival leadership team executives"},
            "Map key decision makers and recent leadership changes.",
            "Leadership profiles and organizational direction signals.",
        ),
        # === Step 6: score~0.22, plan only ===
        _plan(
            "read_10q", "Analyze latest quarterly filing",
            "document_reader", {"url": "sec.gov/techrival/10-Q/Q3-2025"},
            "Get most recent performance trajectory.",
            "Latest quarterly trends and forward guidance.",
            confidence=0.85,
        ),
        # === Step 7: score~0.37, plan only ===
        _plan(
            "extract_growth", "Extract growth and efficiency metrics",
            "data_extractor", {"source": "analyst_reports", "metric": "growth rate ARR"},
            "Quantify growth trajectory for competitive assessment.",
            "Growth rates, rule-of-40, and efficiency metrics.",
        ),
        # === Step 8: score~0.50, plan only ===
        # This step searches with competitor terms; with api_calls=8 after
        # tool increment, WebSearchTool discovers the pivot and sets
        # pivot_discovered=True on the ResearchState.
        _plan(
            "search_strategic", "Search TechRival strategic direction",
            "web_search", {"query": "TechRival strategic direction competitor AI"},
            "Investigate the competitor's strategic direction.",
            "Strategic direction analysis and market moves.",
        ),
        # === Step 9: evaluator sees pivot_discovered=True, pivot_analyzed=False ===
        #   -> score = -0.35, revise fires with should_revise=True, then plan
        RevisionOutput(
            should_revise=True,
            reasoning=(
                "A major strategic pivot to AI by TechRival has been discovered. "
                "The original goal focused on standard competitive analysis, but "
                "this pivot fundamentally changes the competitive landscape. "
                "The goal must be revised to include analysis of the AI pivot "
                "and its implications."
            ),
            new_description=(
                "Build comprehensive competitive analysis of TechRival Inc, "
                "including deep analysis of their strategic pivot to AI "
                "and its implications for competitive positioning"
            ),
            new_criteria=[
                "Map competitor product portfolio and positioning",
                "Identify market share and revenue estimates",
                "Analyze strategic direction and recent moves",
                "Assess competitive threats and opportunities",
                "Analyze AI pivot: investment scope, technology strategy, and timeline",
                "Assess impact of AI pivot on competitive dynamics and market position",
            ],
        ),
        _plan(
            "search_ai_pivot", "Deep search on TechRival AI strategy",
            "web_search", {"query": "TechRival AI artificial intelligence strategy pivot"},
            "Investigate the newly discovered AI pivot in detail.",
            "Detailed intelligence on AI pivot scope and implications.",
            confidence=0.9,
        ),
        # === Step 10+: pivot_analyzed=True (set by web_search above), normal scores ===
        _plan(
            "read_ai_press", "Read AI-related press releases",
            "document_reader", {"url": "techrival.com/press/ai-strategy-announcement"},
            "Get official communications about the AI strategy.",
            "Official positioning and messaging around AI pivot.",
        ),
        _plan(
            "extract_tech", "Extract AI technology capabilities",
            "data_extractor", {"source": "tech_analysis", "metric": "technology AI ML stack"},
            "Assess technical depth of AI investment.",
            "Technology assessment of AI/ML capabilities.",
        ),
        _plan(
            "search_competitive", "Research competitive positioning",
            "web_search", {"query": "TechRival competitive position vs AcmeTech"},
            "Assess how the pivot changes competitive dynamics.",
            "Updated competitive positioning analysis.",
        ),
        _plan(
            "read_earnings", "Analyze earnings call for AI strategy details",
            "document_reader", {"url": "techrival.com/investor/earnings-call-Q3"},
            "Extract management commentary on AI monetization.",
            "Forward-looking AI strategy from management team.",
        ),
        _plan(
            "extract_sentiment", "Extract analyst and market sentiment",
            "data_extractor", {"source": "analyst_reports", "metric": "sentiment analyst"},
            "Gauge market reaction to competitive moves.",
            "Sentiment analysis from analysts and customers.",
        ),
        _plan(
            "search_customers", "Research customer base and satisfaction",
            "web_search", {"query": "TechRival customer reviews satisfaction NPS"},
            "Understand customer perspective on competitive position.",
            "Customer loyalty metrics and satisfaction data.",
        ),
        _plan(
            "read_patents", "Analyze patent portfolio for AI signals",
            "document_reader", {"url": "uspto.gov/techrival/patent-filings-2025"},
            "Assess depth of AI R&D through patent activity.",
            "Patent landscape revealing AI technology investments.",
        ),
        _plan(
            "extract_competitive", "Extract competitive positioning metrics",
            "data_extractor", {"source": "gartner_reports", "metric": "competitive positioning"},
            "Finalize competitive positioning assessment.",
            "Gartner/Forrester competitive positioning data.",
        ),
        _plan(
            "search_weaknesses", "Research competitive weaknesses",
            "web_search", {"query": "TechRival weakness risk challenges"},
            "Identify vulnerabilities for threat assessment.",
            "Key risk factors and competitive weaknesses.",
        ),
        _plan(
            "search_partnerships", "Research strategic partnerships",
            "web_search", {"query": "TechRival partnership alliances ecosystem"},
            "Map the partnership ecosystem and alliances.",
            "Partnership network and ecosystem positioning.",
        ),
        _plan(
            "search_pricing", "Research pricing strategy",
            "web_search", {"query": "TechRival pricing strategy enterprise"},
            "Understand pricing dynamics and competitive pricing.",
            "Pricing model analysis and competitive comparison.",
        ),
        _plan(
            "final_synthesis", "Synthesize competitive analysis",
            "web_search", {"query": "TechRival strategic direction 2026 outlook"},
            "Consolidate all findings into final competitive assessment.",
            "Comprehensive competitive thesis with AI pivot implications.",
            confidence=0.95,
        ),
    ]

    return MockStructuredChatModel(structured_responses=responses)


def build_research_agent(
    max_steps: int = 25,
) -> tuple[Any, dict[str, Any], ResearchState, KnowledgeStore, GoalAuditTrail]:
    """Build competitive research agent.

    Returns
    -------
    tuple
        (app, initial_state, research_state, knowledge_store, audit_trail)
    """
    # --- Shared mutable state ---
    research_state = ResearchState()

    # --- Tools ---
    web_search = WebSearchTool(research_state)
    doc_reader = DocumentReaderTool(research_state)
    data_extractor = DataExtractorTool(research_state)
    tools = [web_search, doc_reader, data_extractor]

    # --- Strategies ---
    evaluator = ResearchEvaluator(research_state)
    source_checker = SourceDiversityChecker(research_state)

    # --- Infrastructure ---
    knowledge_store = KnowledgeStore()
    audit_trail = GoalAuditTrail(knowledge_store=knowledge_store)

    # --- Working memory for the perception loop ---
    memory = WorkingMemory(
        initial_context=(
            "Competitive Research Agent initialized. "
            "Target: comprehensive analysis of TechRival Inc. "
            "Available tools: web_search, document_reader, data_extractor."
        ),
        max_entries=30,
    )

    # --- Model ---
    model = _get_model()
    if model is None:
        model = _build_mock_model()

    # --- Build graph ---
    app, initial_state = (
        GraphBuilder("competitive-research")
        .with_model(model)
        .with_goal(
            "Build comprehensive competitive analysis of TechRival Inc",
            criteria=[
                "Map competitor product portfolio and positioning",
                "Identify market share and revenue estimates",
                "Analyze strategic direction and recent moves",
                "Assess competitive threats and opportunities",
            ],
        )
        .with_tools(*tools)
        .with_evaluator(evaluator)
        .with_constraint_checkers(source_checker)
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

    # Seed knowledge store with initial context
    knowledge_store.put(
        key="research:target",
        value="TechRival Inc",
        source="agent_init",
        tags=("research", "target"),
        confidence=1.0,
    )
    knowledge_store.put(
        key="research:scope",
        value={
            "topics": [
                "product_portfolio", "market_share", "revenue",
                "leadership", "strategy", "threats",
            ],
            "min_sources": 5,
        },
        source="agent_init",
        tags=("research", "scope"),
        confidence=1.0,
    )

    return app, initial_state, research_state, knowledge_store, audit_trail
