"""Simulated financial research tools (LangChain-compatible).

Each tool has ``.name``, ``.description``, and ``.invoke(params)`` following
the LangChain tool interface.  Tools mutate a shared ``ThesisState`` to
simulate progressive research accumulation.
"""


from .models import FinancialMetric, NewsItem, ThesisState


class FinancialDataTool:
    """Retrieve financial metrics for a company.

    Returns revenue, margins, and growth rates.  Each call adds a
    ``FinancialMetric`` to the thesis state and registers the source.
    """

    name = "financial_data"
    description = "Retrieve financial metrics (revenue, margins, growth) for a company"

    _METRICS: dict[str, tuple[float, str, str]] = {
        "revenue": (2_450_000_000.0, "FY 2025", "Annual Report"),
        "revenue_growth": (0.18, "FY 2025", "Annual Report"),
        "gross_margin": (0.72, "FY 2025", "10-K Filing"),
        "operating_margin": (0.31, "FY 2025", "10-K Filing"),
        "net_income": (485_000_000.0, "FY 2025", "10-K Filing"),
        "eps": (4.85, "FY 2025", "Earnings Release"),
        "free_cash_flow": (620_000_000.0, "FY 2025", "Cash Flow Statement"),
        "r_and_d_spend": (380_000_000.0, "FY 2025", "10-K Filing"),
        "debt_to_equity": (0.45, "FY 2025", "Balance Sheet"),
        "current_ratio": (2.1, "FY 2025", "Balance Sheet"),
        "q4_revenue": (680_000_000.0, "Q4 2025", "Quarterly Earnings"),
        "q4_growth": (0.22, "Q4 2025", "Quarterly Earnings"),
    }

    def __init__(self, thesis_state: ThesisState) -> None:
        self._state = thesis_state

    def invoke(self, params: dict) -> str:
        metric = params.get("metric", "revenue")
        company = params.get("company", "NovaTech Corp")

        if metric in self._METRICS:
            value, period, source = self._METRICS[metric]
            fm = FinancialMetric(
                name=metric,
                value=value,
                period=period,
                source=source,
            )
            self._state.financials.append(fm)
            self._state.sources.add(source)
            return (
                f"{company} {metric}: {value:,.2f} ({period}, source: {source})"
            )

        return f"No data found for metric '{metric}' for {company}"

    def __call__(self, query: str) -> str:
        return self.invoke({"metric": query})


class NewsSearchTool:
    """Search financial news for a topic.

    Internal counter tracks calls.  At call 3+, returns lawsuit news and
    sets ``lawsuit_discovered=True`` on the thesis state.  Earlier calls
    return normal positive/neutral news.
    """

    name = "news_search"
    description = "Search financial news, press releases, and analyst reports"

    _NORMAL_NEWS: list[tuple[str, str, str]] = [
        ("NovaTech Reports Record Q4 Revenue", "Bloomberg", "positive"),
        ("NovaTech Expands Cloud Platform to APAC", "Reuters", "positive"),
        ("Analyst Upgrades NovaTech to Overweight", "Barron's", "positive"),
        ("NovaTech Partners with Azure for Enterprise AI", "TechCrunch", "positive"),
        ("NovaTech CEO Keynote at CES 2026", "The Verge", "neutral"),
        ("Enterprise Software Market Grows 15% YoY", "Gartner", "neutral"),
        ("NovaTech Hires Ex-Google VP as CTO", "Business Insider", "positive"),
        ("Cloud Infrastructure Spending Hits Record $73B", "IDC", "neutral"),
        ("NovaTech Beats Earnings Estimates by 12%", "CNBC", "positive"),
    ]

    _LAWSUIT_NEWS: tuple[str, str, str] = (
        "NovaTech Faces $500M Patent Infringement Lawsuit from Vertex Systems",
        "Reuters",
        "negative",
    )

    def __init__(self, thesis_state: ThesisState) -> None:
        self._state = thesis_state
        self._call_count = 0

    def invoke(self, params: dict) -> str:
        self._call_count += 1

        if self._call_count >= 3 and not self._state.lawsuit_discovered:
            headline, source, sentiment = self._LAWSUIT_NEWS
            item = NewsItem(
                headline=headline,
                source=source,
                sentiment=sentiment,
                content=(
                    "Vertex Systems has filed a $500M patent infringement lawsuit "
                    "against NovaTech Corp alleging unauthorized use of proprietary "
                    "distributed computing patents in NovaTech's cloud platform. "
                    "The case was filed in the Eastern District of Texas."
                ),
            )
            self._state.news.append(item)
            self._state.sources.add(source)
            self._state.lawsuit_discovered = True
            return f"BREAKING: {headline} (source: {source})"

        # Normal news rotation
        idx = (self._call_count - 1) % len(self._NORMAL_NEWS)
        headline, source, sentiment = self._NORMAL_NEWS[idx]
        item = NewsItem(
            headline=headline,
            source=source,
            sentiment=sentiment,
        )
        self._state.news.append(item)
        self._state.sources.add(source)
        return f"News: {headline} (source: {source}, sentiment: {sentiment})"

    def __call__(self, query: str) -> str:
        return self.invoke({"query": query})


class FilingAnalyzerTool:
    """Analyze SEC filings for a company.

    Supports 10-K, 10-Q, and proxy filing types.  Returns a summary of
    key findings from the filing.
    """

    name = "filing_analyzer"
    description = "Analyze SEC filings (10-K, 10-Q, proxy) for key insights"

    _ANALYSES: dict[str, str] = {
        "10-K": (
            "10-K Analysis: NovaTech FY2025 — Revenue $2.45B (+18% YoY), "
            "gross margin 72%, operating margin 31%. R&D spend $380M (15.5% of revenue). "
            "Key risk factors: market concentration (top 10 customers = 35% revenue), "
            "regulatory changes in data privacy, competitive pressure from hyperscalers."
        ),
        "10-Q": (
            "10-Q Analysis: NovaTech Q4 2025 — Revenue $680M (+22% QoQ). "
            "Enterprise segment grew 28%, SMB segment stable. "
            "Deferred revenue increased 15% indicating strong forward pipeline. "
            "Operating expenses well controlled at 41% of revenue."
        ),
        "proxy": (
            "Proxy Statement Analysis: CEO compensation aligned with revenue growth targets. "
            "Board composition: 8 independent directors, 2 insiders. "
            "Insider ownership: management holds 8.5% of shares. "
            "No material related-party transactions flagged."
        ),
    }

    def __init__(self, thesis_state: ThesisState) -> None:
        self._state = thesis_state

    def invoke(self, params: dict) -> str:
        filing_type = params.get("filing_type", "10-K")
        analysis = self._ANALYSES.get(
            filing_type,
            f"Filing type '{filing_type}' not available. Supported: 10-K, 10-Q, proxy",
        )

        self._state.sources.add(f"SEC {filing_type}")
        return analysis

    def __call__(self, filing_type: str) -> str:
        return self.invoke({"filing_type": filing_type})


class SentimentTool:
    """Analyze sentiment of a text passage.

    Returns negative sentiment if the text contains lawsuit-related keywords,
    otherwise returns positive/neutral.
    """

    name = "sentiment_analyzer"
    description = "Analyze sentiment of financial text (positive/negative/neutral)"

    _LAWSUIT_KEYWORDS = frozenset({
        "lawsuit", "litigation", "infringement", "patent", "legal action",
        "filing", "sued", "damages", "injunction", "defendant",
    })

    def __init__(self, thesis_state: ThesisState) -> None:
        self._state = thesis_state

    def invoke(self, params: dict) -> str:
        text = params.get("text", "").lower()

        if any(kw in text for kw in self._LAWSUIT_KEYWORDS):
            # Analyzing litigation text means risk has been assessed
            if self._state.lawsuit_discovered:
                self._state.risk_analyzed = True
            return (
                "Sentiment: NEGATIVE (0.15). Legal/litigation language detected. "
                "Significant downside risk identified. Recommend detailed risk analysis."
            )

        # Count positive/negative signals in text
        positive_words = {"growth", "record", "beat", "upgrade", "expand", "strong"}
        negative_words = {"decline", "miss", "risk", "concern", "weak", "loss"}
        pos_count = sum(1 for w in positive_words if w in text)
        neg_count = sum(1 for w in negative_words if w in text)

        if pos_count > neg_count:
            score = min(0.85, 0.5 + pos_count * 0.1)
            return f"Sentiment: POSITIVE ({score:.2f}). Favorable outlook indicated."
        if neg_count > pos_count:
            score = max(0.25, 0.5 - neg_count * 0.1)
            return f"Sentiment: CAUTIOUS ({score:.2f}). Some concerns noted."

        return "Sentiment: NEUTRAL (0.50). Balanced tone, no strong signals."

    def __call__(self, text: str) -> str:
        return self.invoke({"text": text})
