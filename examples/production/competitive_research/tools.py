"""Simulated research tools for the Competitive Research agent.

Each tool follows the LangChain tool interface (name, description, invoke)
and mutates a shared ResearchState to simulate environment effects.
"""

import time
from typing import Any

from .models import ResearchFinding, ResearchState


class WebSearchTool:
    """Simulated web search tool.

    After enough API calls have accumulated (api_calls_used >= 7), if the
    query mentions competitor-related terms, returns news about a strategic
    pivot to AI -- triggering goal revision via the evaluator's pivot logic.

    Once the pivot has been discovered and a subsequent AI-related search
    is executed, marks the pivot as analyzed so the evaluator no longer
    forces a low score.
    """

    name: str = "web_search"
    description: str = (
        "Search the web for competitive intelligence. "
        "Input: {'query': 'search query string'}."
    )

    def __init__(self, research_state: ResearchState) -> None:
        self._state = research_state

    def invoke(self, params: dict[str, Any] | str) -> str:
        """Execute the web search and return results."""
        if isinstance(params, str):
            params = {"query": params}
        query = params.get("query", "")
        return self._search(query)

    def __call__(self, params: dict[str, Any] | str) -> str:
        return self.invoke(params)

    def _search(self, query: str) -> str:
        self._state.api_calls_used += 1
        now = time.time()

        pivot_terms = [
            "techrival", "competitor", "rival", "strategic",
            "direction", "pivot", "ai", "artificial intelligence",
        ]
        query_lower = query.lower()
        has_pivot_term = any(t in query_lower for t in pivot_terms)

        # Post-pivot: mark analyzed if searching for AI after discovery
        if self._state.pivot_discovered and not self._state.pivot_analyzed:
            ai_terms = ["ai", "artificial intelligence", "pivot", "strategy"]
            has_ai_term = any(t in query_lower for t in ai_terms)
            if has_ai_term:
                self._state.pivot_analyzed = True
                finding = ResearchFinding(
                    source="web_search:ai_analysis",
                    topic="ai_strategy",
                    content=(
                        "Deep analysis: TechRival's AI pivot includes $500M over 3 years, "
                        "hiring 200+ ML engineers, partnership with leading AI lab, "
                        "and restructuring CloudSuite as AI-first platform. "
                        "Timeline: AI features in Q2 2026, full platform in 2027. "
                        "Risk: execution uncertainty, talent competition."
                    ),
                    confidence=0.9,
                    timestamp=now,
                )
                self._state.findings.append(finding)
                self._state.topics_covered.add("ai_strategy")
                self._state.sources_consulted.add("web_search:ai_analysis")
                self._state.thesis_confidence = min(
                    1.0, self._state.thesis_confidence + 0.10
                )
                return finding.content

        # Trigger pivot discovery after sufficient research has accumulated
        if (
            self._state.api_calls_used >= 8
            and has_pivot_term
            and not self._state.pivot_discovered
        ):
            self._state.pivot_discovered = True
            finding = ResearchFinding(
                source="web_search:breaking_news",
                topic="strategic_pivot",
                content=(
                    "BREAKING: TechRival Inc announces major strategic pivot to AI. "
                    "CEO confirms $500M investment in generative AI platform, "
                    "restructuring 3 business units. Partnership with leading "
                    "AI research lab expected next quarter. Stock up 15%."
                ),
                confidence=0.95,
                timestamp=now,
            )
            self._state.findings.append(finding)
            self._state.topics_covered.add("strategic_pivot")
            self._state.sources_consulted.add("web_search:breaking_news")
            return finding.content

        # Normal search results based on query content
        result_map = {
            "product": (
                "TechRival Inc offers 4 main products: CloudSuite Enterprise, "
                "DataFlow Pro, SecureNet Plus, and DevOps Accelerator. "
                "CloudSuite holds 18% market share in mid-market segment."
            ),
            "market share": (
                "TechRival holds estimated 12-15% overall market share in "
                "cloud infrastructure. Growing at 22% YoY vs industry avg 15%. "
                "Primary competitors: AcmeTech (25%), GlobalSys (20%)."
            ),
            "revenue": (
                "TechRival FY2025 revenue estimated at $2.1B, up from $1.7B. "
                "Gross margins at 68%. R&D spend at 24% of revenue. "
                "Enterprise segment growing fastest at 35% YoY."
            ),
            "pricing": (
                "TechRival pricing strategy: freemium for DevOps Accelerator, "
                "tiered enterprise pricing for CloudSuite ($50-500/seat/month). "
                "Recent 10% price increase on SecureNet Plus."
            ),
            "leadership": (
                "TechRival CEO Sarah Chen (since 2022). CTO James Park, "
                "formerly VP Engineering at MegaCorp. Recent hire: Chief AI "
                "Officer Dr. Priya Sharma from DeepMind."
            ),
            "partnership": (
                "TechRival partnerships include AWS (preferred partner), "
                "Salesforce (integration), and 3 regional system integrators. "
                "Recently ended partnership with OracleDB."
            ),
            "customer": (
                "TechRival serves 5,000+ enterprise customers including "
                "Fortune 500 companies. NPS score of 42, churn rate 8%. "
                "Key verticals: finance, healthcare, retail."
            ),
            "weakness": (
                "Analyst reports highlight: limited international presence, "
                "aging SecureNet architecture, high customer acquisition cost. "
                "Employee reviews cite slow decision-making culture."
            ),
        }

        # Match based on query keywords
        for key, content in result_map.items():
            if key in query_lower:
                topic = key.replace(" ", "_")
                finding = ResearchFinding(
                    source=f"web_search:{topic}",
                    topic=topic,
                    content=content,
                    confidence=0.8,
                    timestamp=now,
                )
                self._state.findings.append(finding)
                self._state.topics_covered.add(topic)
                self._state.sources_consulted.add(f"web_search:{topic}")
                return content

        # Generic fallback
        content = (
            f"Search results for '{query}': TechRival Inc continues expansion "
            f"in enterprise cloud market. Recent quarterly earnings beat "
            f"estimates. Analyst consensus: 'outperform'."
        )
        finding = ResearchFinding(
            source="web_search:general",
            topic="general",
            content=content,
            confidence=0.6,
            timestamp=now,
        )
        self._state.findings.append(finding)
        self._state.topics_covered.add("general")
        self._state.sources_consulted.add("web_search:general")
        return content


class DocumentReaderTool:
    """Simulated document analysis tool.

    Reads and analyzes financial reports, press releases, and SEC filings.
    """

    name: str = "document_reader"
    description: str = (
        "Read and analyze documents (financial reports, press releases, filings). "
        "Input: {'url': 'document URL or identifier'}."
    )

    def __init__(self, research_state: ResearchState) -> None:
        self._state = research_state

    def invoke(self, params: dict[str, Any] | str) -> str:
        """Read and analyze a document."""
        if isinstance(params, str):
            params = {"url": params}
        url = params.get("url", "")
        return self._read(url)

    def __call__(self, params: dict[str, Any] | str) -> str:
        return self.invoke(params)

    def _read(self, url: str) -> str:
        self._state.api_calls_used += 1
        now = time.time()
        url_lower = url.lower()

        doc_map = {
            "10-k": (
                "annual_filing",
                "TechRival 10-K Analysis: Revenue $2.1B (+24% YoY). Operating "
                "income $315M. Cash reserves $890M. Goodwill from 3 acquisitions "
                "totaling $450M. Risk factors: concentration in North America (78%), "
                "key person dependency on CTO for AI strategy.",
                0.9,
            ),
            "10-q": (
                "quarterly_filing",
                "TechRival Q3 10-Q: Quarterly revenue $580M (+28% YoY). "
                "Subscription revenue now 72% of total. Customer count +800 "
                "net new. Deferred revenue up 35%, signaling strong pipeline.",
                0.9,
            ),
            "press": (
                "press_release",
                "TechRival press release: Launch of CloudSuite v5.0 with "
                "AI-powered analytics. New data center in Singapore. "
                "Strategic partnership with Accenture for enterprise deployments.",
                0.85,
            ),
            "earning": (
                "earnings_call",
                "TechRival earnings call summary: CEO highlighted 'transformative "
                "year ahead' with AI investments. CFO guided FY2026 revenue "
                "$2.5-2.7B. Analyst questions focused on AI monetization timeline.",
                0.85,
            ),
            "patent": (
                "patent_filing",
                "TechRival patent activity: 47 new patents filed in 2025, "
                "up from 28 in 2024. Clusters in: distributed AI inference (12), "
                "secure multi-tenant data processing (8), edge computing (7). "
                "Signals major technical investment in AI infrastructure.",
                0.75,
            ),
        }

        for key, (topic, content, conf) in doc_map.items():
            if key in url_lower:
                finding = ResearchFinding(
                    source=f"document_reader:{topic}",
                    topic=topic,
                    content=content,
                    confidence=conf,
                    timestamp=now,
                )
                self._state.findings.append(finding)
                self._state.topics_covered.add(topic)
                self._state.sources_consulted.add(f"document_reader:{topic}")
                self._state.thesis_confidence = min(
                    1.0, self._state.thesis_confidence + 0.08
                )
                return content

        # Fallback for unrecognized documents
        content = (
            f"Document analysis for '{url}': General corporate document. "
            f"Contains standard operational metrics and forward-looking statements."
        )
        finding = ResearchFinding(
            source="document_reader:general",
            topic="document_analysis",
            content=content,
            confidence=0.5,
            timestamp=now,
        )
        self._state.findings.append(finding)
        self._state.topics_covered.add("document_analysis")
        self._state.sources_consulted.add("document_reader:general")
        return content


class DataExtractorTool:
    """Simulated data extraction tool.

    Extracts structured data points (market share, revenue, growth rates)
    from specified sources.
    """

    name: str = "data_extractor"
    description: str = (
        "Extract specific data metrics from sources. "
        "Input: {'source': 'data source', 'metric': 'metric to extract'}."
    )

    def __init__(self, research_state: ResearchState) -> None:
        self._state = research_state

    def invoke(self, params: dict[str, Any] | str) -> str:
        """Extract data from a source."""
        if isinstance(params, str):
            params = {"source": params, "metric": "general"}
        source = params.get("source", "")
        metric = params.get("metric", "general")
        return self._extract(source, metric)

    def __call__(self, params: dict[str, Any] | str) -> str:
        return self.invoke(params)

    def _extract(self, source: str, metric: str) -> str:
        self._state.api_calls_used += 1
        now = time.time()
        metric_lower = metric.lower()

        extraction_map = {
            "market_share": (
                "market_data",
                "Market share data: TechRival 14.2% (up from 11.8%), "
                "AcmeTech 24.5% (stable), GlobalSys 19.1% (down from 20.3%), "
                "Others 42.2%. TechRival gaining share fastest in mid-market.",
                0.85,
            ),
            "revenue": (
                "financial_data",
                "Revenue breakdown: CloudSuite $920M (44%), DataFlow $480M (23%), "
                "SecureNet $380M (18%), DevOps $320M (15%). "
                "Cloud revenue CAGR 32% over 3 years.",
                0.9,
            ),
            "growth": (
                "growth_metrics",
                "Growth metrics: Revenue +24% YoY, ARR +28%, Customer count +19%, "
                "ACV (avg contract value) +12% to $42K. Net revenue retention 118%. "
                "Rule of 40 score: 52 (excellent).",
                0.85,
            ),
            "sentiment": (
                "sentiment_analysis",
                "Analyst sentiment: 12 Buy, 5 Hold, 1 Sell. Average price target "
                "$87 (current $72). Employee sentiment on Glassdoor: 3.8/5. "
                "Customer satisfaction (G2): 4.2/5.",
                0.7,
            ),
            "technology": (
                "tech_stack",
                "Technology assessment: Primary stack Kubernetes + Go microservices. "
                "AI/ML team grown from 45 to 120 engineers in 18 months. "
                "Internal LLM fine-tuning infrastructure confirmed. "
                "Edge computing pilot with 3 customers.",
                0.75,
            ),
            "competitive": (
                "competitive_position",
                "Competitive positioning: Strong in mid-market cloud (leader in "
                "Gartner Magic Quadrant). Weakness vs AcmeTech in enterprise. "
                "Differentiation through integrated security + ease of deployment. "
                "Threat: AI-native startups gaining traction in SMB.",
                0.8,
            ),
        }

        for key, (topic, content, conf) in extraction_map.items():
            if key in metric_lower:
                finding = ResearchFinding(
                    source=f"data_extractor:{topic}",
                    topic=topic,
                    content=content,
                    confidence=conf,
                    timestamp=now,
                )
                self._state.findings.append(finding)
                self._state.topics_covered.add(topic)
                self._state.sources_consulted.add(f"data_extractor:{topic}")
                self._state.thesis_confidence = min(
                    1.0, self._state.thesis_confidence + 0.06
                )
                return content

        # Fallback
        content = (
            f"Data extraction from '{source}' for metric '{metric}': "
            f"Limited data available. General performance indicators positive."
        )
        finding = ResearchFinding(
            source="data_extractor:general",
            topic="data_extraction",
            content=content,
            confidence=0.5,
            timestamp=now,
        )
        self._state.findings.append(finding)
        self._state.topics_covered.add("data_extraction")
        self._state.sources_consulted.add("data_extractor:general")
        return content
