"""Domain model for the Investment Thesis Builder agent."""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class FinancialMetric:
    """A single financial metric data point."""

    name: str
    value: float
    period: str  # e.g. "Q4 2025", "FY 2025"
    source: str  # e.g. "SEC filing", "earnings call"
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True)
class NewsItem:
    """A news article or press release."""

    headline: str
    source: str  # e.g. "Reuters", "Bloomberg", "SEC"
    sentiment: str  # positive | negative | neutral
    date: datetime = field(default_factory=datetime.now)
    content: str = ""


@dataclass
class ThesisState:
    """Mutable research state accumulated during analysis."""

    financials: list[FinancialMetric] = field(default_factory=list)
    news: list[NewsItem] = field(default_factory=list)
    sources: set[str] = field(default_factory=set)
    thesis_components: dict[str, str] = field(default_factory=dict)
    risk_factors: list[str] = field(default_factory=list)
    lawsuit_discovered: bool = False
    risk_analyzed: bool = False
    confidence: float = 0.0
