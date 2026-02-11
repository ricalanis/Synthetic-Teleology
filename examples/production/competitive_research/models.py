"""Domain models for the Competitive Research agent."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ResearchFinding:
    """A single research finding from a tool invocation."""

    source: str
    topic: str
    content: str
    confidence: float = 0.8
    timestamp: float = 0.0


class ResearchState:
    """Mutable research state shared across tools and evaluator.

    Tracks findings, API usage, topic coverage, and pivot detection.
    """

    def __init__(self) -> None:
        self.findings: list[ResearchFinding] = []
        self.api_calls_used: int = 0
        self.topics_covered: set[str] = set()
        self.pivot_discovered: bool = False
        self.pivot_analyzed: bool = False
        self.thesis_confidence: float = 0.0
        self.sources_consulted: set[str] = set()
