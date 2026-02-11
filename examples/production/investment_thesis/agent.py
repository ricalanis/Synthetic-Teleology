"""Graph wiring for the Investment Thesis Builder agent.

Uses LLM mode (with_model + with_goal) with a custom evaluator.
The custom evaluator reads simulated environment state (ThesisState)
for deterministic scoring.  The LLM (or mock) handles planning and
revision only.

In simulated mode (no API key), uses MockStructuredChatModel.
The mock provides PlanningOutput + RevisionOutput responses in sequence:
  - Steps 1-9: PlanningOutput for normal research actions
  - Step 10: evaluator detects lawsuit -> RevisionOutput triggers goal update
  - Steps 10-25: PlanningOutput for continued analysis including risk sections
"""

import os

from synthetic_teleology.graph import GraphBuilder, WorkingMemory
from synthetic_teleology.infrastructure.knowledge_store import KnowledgeStore
from synthetic_teleology.services.audit_trail import GoalAuditTrail

from .models import ThesisState
from .strategies import SourceDiversityChecker, ThesisEvaluator
from .tools import (
    FilingAnalyzerTool,
    FinancialDataTool,
    NewsSearchTool,
    SentimentTool,
)


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


def _build_mock_model():
    """Build mock responses for ~25-step thesis building agent.

    The custom ThesisEvaluator handles scoring, so the mock only needs
    PlanningOutput and RevisionOutput responses.

    Sequence:
      PlanningOutput x 8 (steps 1-8, normal research; step 8 triggers lawsuit)
      RevisionOutput x 1 (step 9, evaluator detects lawsuit -> revision)
      PlanningOutput x 17 (steps 9-25, risk analysis + continued research)
    """
    from synthetic_teleology.services.llm_planning import (
        ActionProposal,
        PlanHypothesis,
        PlanningOutput,
    )
    from synthetic_teleology.services.llm_revision import RevisionOutput
    from synthetic_teleology.testing import MockStructuredChatModel

    def _plan(action_name: str, tool_name: str, params: dict,
              reasoning: str, outcome: str, confidence: float = 0.8) -> PlanningOutput:
        return PlanningOutput(
            hypotheses=[
                PlanHypothesis(
                    actions=[
                        ActionProposal(
                            name=action_name,
                            description=reasoning,
                            tool_name=tool_name,
                            parameters=params,
                        )
                    ],
                    reasoning=reasoning,
                    expected_outcome=outcome,
                    confidence=confidence,
                ),
            ],
            selected_index=0,
            selection_reasoning=reasoning,
        )

    responses = [
        # --- Steps 1-8: Normal research phase ---
        # Each step: evaluate (custom, no mock) -> plan (mock PlanningOutput)
        # Step 8's news_search is the 3rd call, triggering lawsuit discovery.

        # Step 1: Fetch core revenue data
        _plan(
            "fetch_revenue",
            "financial_data",
            {"company": "NovaTech Corp", "metric": "revenue"},
            "Start with core revenue figures to establish baseline",
            "Annual revenue data for NovaTech Corp",
        ),
        # Step 2: Get revenue growth
        _plan(
            "fetch_growth",
            "financial_data",
            {"company": "NovaTech Corp", "metric": "revenue_growth"},
            "Assess growth trajectory to understand momentum",
            "Revenue growth rate data",
        ),
        # Step 3: Analyze profitability margins
        _plan(
            "fetch_margins",
            "financial_data",
            {"company": "NovaTech Corp", "metric": "gross_margin"},
            "Evaluate gross margin for business model quality",
            "Margin data for profitability assessment",
        ),
        # Step 4: Analyze 10-K filing
        _plan(
            "analyze_annual_filing",
            "filing_analyzer",
            {"filing_type": "10-K"},
            "Deep dive into annual filing for comprehensive picture",
            "Key insights from 10-K filing analysis",
        ),
        # Step 5: Search for recent news (news_search call 1)
        _plan(
            "search_recent_news",
            "news_search",
            {"query": "NovaTech Corp recent developments"},
            "Gather recent news to assess market sentiment",
            "Recent news and analyst coverage",
        ),
        # Step 6: Analyze quarterly filing
        _plan(
            "analyze_quarterly_filing",
            "filing_analyzer",
            {"filing_type": "10-Q"},
            "Review latest quarterly for trend confirmation",
            "Q4 performance details and forward indicators",
        ),
        # Step 7: Search for competitive landscape news (news_search call 2)
        _plan(
            "search_competitive_news",
            "news_search",
            {"query": "NovaTech competitive positioning enterprise software"},
            "Understand competitive dynamics and market position",
            "Competitive landscape intelligence",
        ),
        # Step 8: Broader news search (news_search call 3 -> LAWSUIT!)
        # After this action executes, thesis_state.lawsuit_discovered = True
        _plan(
            "search_market_news",
            "news_search",
            {"query": "NovaTech Corp market news analyst coverage"},
            "Comprehensive news sweep for material developments",
            "Latest market news and coverage",
        ),

        # --- Step 9: Revision triggered by lawsuit discovery ---
        # Evaluator sees lawsuit_discovered=True, risk_analyzed=False -> score=-0.5
        # Routes to revise node -> consumes RevisionOutput, then plan -> PlanningOutput

        RevisionOutput(
            should_revise=True,
            reasoning=(
                "Material litigation risk discovered: Vertex Systems filed a $500M "
                "patent infringement lawsuit against NovaTech. This significantly "
                "changes the risk profile and requires dedicated litigation risk "
                "analysis as part of the investment thesis. The original criteria "
                "did not account for legal risk assessment."
            ),
            new_description=(
                "Build comprehensive investment thesis for NovaTech Corp "
                "with litigation risk assessment"
            ),
            new_criteria=[
                "Analyze financial performance and growth trajectory",
                "Evaluate competitive positioning and market opportunity",
                "Assess management quality and corporate governance",
                "Identify key risks and catalysts",
                "Analyze litigation risk and potential financial impact",
            ],
        ),

        # Step 9 plan: Immediate risk analysis (sentiment_analyzer with
        # lawsuit text -> sets risk_analyzed=True on ThesisState)
        _plan(
            "analyze_lawsuit_sentiment",
            "sentiment_analyzer",
            {"text": "NovaTech patent infringement lawsuit $500M damages Vertex Systems"},
            "Analyze sentiment of lawsuit news for severity assessment",
            "Quantified sentiment around litigation risk",
            confidence=0.9,
        ),

        # --- Steps 10-25: Continued analysis (no more revision) ---
        # After step 9, risk_analyzed=True, so evaluator returns normal scores.

        # Step 10: Proxy analysis for governance
        _plan(
            "analyze_proxy",
            "filing_analyzer",
            {"filing_type": "proxy"},
            "Assess management quality and governance structure",
            "Management and governance assessment",
        ),
        # Step 11: More news context on lawsuit
        _plan(
            "search_lawsuit_context",
            "news_search",
            {"query": "NovaTech Vertex Systems patent lawsuit"},
            "Gather additional context on the litigation",
            "Detailed lawsuit coverage and analyst reactions",
        ),
        # Step 12: Financial resilience — free cash flow
        _plan(
            "fetch_free_cash_flow",
            "financial_data",
            {"company": "NovaTech Corp", "metric": "free_cash_flow"},
            "Assess cash position to evaluate litigation resilience",
            "Free cash flow data for litigation impact analysis",
        ),
        # Step 13: Debt analysis
        _plan(
            "fetch_debt_to_equity",
            "financial_data",
            {"company": "NovaTech Corp", "metric": "debt_to_equity"},
            "Check leverage to understand financial flexibility under litigation",
            "Debt metrics for balance sheet stress test",
        ),
        # Step 14: Analyst reactions
        _plan(
            "search_analyst_reactions",
            "news_search",
            {"query": "NovaTech analyst reaction lawsuit impact"},
            "Gather analyst sentiment post-lawsuit announcement",
            "Analyst reactions and price target revisions",
        ),
        # Step 15: R&D for IP defense capability
        _plan(
            "fetch_rd_spend",
            "financial_data",
            {"company": "NovaTech Corp", "metric": "r_and_d_spend"},
            "R&D investment level indicates patent portfolio strength",
            "R&D data for IP defense capability assessment",
        ),
        # Step 16: Industry patent trends
        _plan(
            "search_industry_trends",
            "news_search",
            {"query": "enterprise software patent litigation trends 2026"},
            "Contextualize lawsuit within industry patent landscape",
            "Industry-wide patent litigation benchmarks",
        ),
        # Step 17: EPS for valuation
        _plan(
            "fetch_eps",
            "financial_data",
            {"company": "NovaTech Corp", "metric": "eps"},
            "Get EPS for valuation modeling",
            "Earnings per share for DCF inputs",
        ),
        # Step 18: Liquidity check
        _plan(
            "fetch_current_ratio",
            "financial_data",
            {"company": "NovaTech Corp", "metric": "current_ratio"},
            "Verify short-term liquidity position",
            "Liquidity ratio for financial health check",
        ),
        # Step 19: Q4 revenue
        _plan(
            "fetch_q4_revenue",
            "financial_data",
            {"company": "NovaTech Corp", "metric": "q4_revenue"},
            "Confirm recent quarter execution",
            "Q4 revenue for momentum verification",
        ),
        # Step 20: Net income
        _plan(
            "fetch_net_income",
            "financial_data",
            {"company": "NovaTech Corp", "metric": "net_income"},
            "Net income for earnings quality analysis",
            "Bottom line profitability data",
        ),
        # Step 21: Growth catalyst news
        _plan(
            "search_growth_drivers",
            "news_search",
            {"query": "NovaTech growth catalysts cloud AI expansion"},
            "Identify forward-looking growth catalysts",
            "Growth drivers and expansion opportunities",
        ),
        # Step 22: Growth sentiment
        _plan(
            "analyze_growth_sentiment",
            "sentiment_analyzer",
            {"text": "NovaTech record revenue growth cloud expansion strong pipeline"},
            "Gauge overall market sentiment on growth thesis",
            "Sentiment score for growth narrative",
        ),
        # Step 23: Operating margin
        _plan(
            "fetch_operating_margin",
            "financial_data",
            {"company": "NovaTech Corp", "metric": "operating_margin"},
            "Complete profitability picture with operating margin",
            "Operating margin for cost structure analysis",
        ),
        # Step 24: Q4 growth
        _plan(
            "fetch_q4_growth",
            "financial_data",
            {"company": "NovaTech Corp", "metric": "q4_growth"},
            "Verify recent quarter growth acceleration",
            "Q4 growth rate for trend confirmation",
        ),
        # Step 25: Final news sweep
        _plan(
            "search_final_sweep",
            "news_search",
            {"query": "NovaTech Corp latest developments 2026"},
            "Final news check for any late-breaking developments",
            "Up-to-date news coverage",
        ),
    ]

    return MockStructuredChatModel(structured_responses=responses)


def build_thesis_agent(max_steps: int = 30):
    """Build a LangGraph investment thesis builder agent.

    Uses LLM mode with a custom evaluator.  The custom ThesisEvaluator
    reads simulated environment state, while the LLM (or mock) handles
    planning and revision.

    Returns ``(app, initial_state, thesis_state, knowledge_store, audit_trail)``
    tuple.
    """
    thesis_state = ThesisState()
    knowledge_store = KnowledgeStore()
    audit_trail = GoalAuditTrail(knowledge_store=knowledge_store)

    # --- Tools ---
    financial_tool = FinancialDataTool(thesis_state)
    news_tool = NewsSearchTool(thesis_state)
    filing_tool = FilingAnalyzerTool(thesis_state)
    sentiment_tool = SentimentTool(thesis_state)
    tools = [financial_tool, news_tool, filing_tool, sentiment_tool]

    # --- Strategies ---
    evaluator = ThesisEvaluator(thesis_state)
    source_checker = SourceDiversityChecker(thesis_state)

    # --- Working memory ---
    memory = WorkingMemory(
        initial_context=(
            "Investment Thesis Research — NovaTech Corp (NVTK)\n"
            "Target: Build comprehensive investment thesis\n"
            "Company: NovaTech Corp — enterprise software & cloud platform\n"
            "Market Cap: ~$25B | Sector: Technology | Industry: Enterprise Software\n"
            "Available tools: financial_data, news_search, filing_analyzer, sentiment_analyzer"
        ),
    )

    # --- Model ---
    model = _get_model() or _build_mock_model()

    # --- Build graph ---
    app, initial_state = (
        GraphBuilder("investment-thesis")
        .with_model(model)
        .with_goal(
            "Build comprehensive investment thesis for NovaTech Corp",
            criteria=[
                "Analyze financial performance and growth trajectory",
                "Evaluate competitive positioning and market opportunity",
                "Assess management quality and corporate governance",
                "Identify key risks and catalysts",
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

    return app, initial_state, thesis_state, knowledge_store, audit_trail
