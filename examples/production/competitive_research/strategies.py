"""Evaluation and constraint strategies for the Competitive Research agent.

The ResearchEvaluator uses a custom scoring formula that reads simulated
environment state, making the evaluation fully deterministic even when the
planner and reviser run through an LLM (real or mock).

When a competitor pivot is discovered but not yet analyzed, the score drops
to <= -0.3 to trigger the revision threshold.
"""

from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.values import ActionSpec, EvalSignal, StateSnapshot
from synthetic_teleology.services.constraint_engine import BaseConstraintChecker
from synthetic_teleology.services.evaluation import BaseEvaluator

from .models import ResearchState

# Topics the analyst must cover for a complete competitive analysis.
# These match the topic keys produced by the simulated tools.
REQUIRED_TOPICS: set[str] = {
    "product",
    "market_data",
    "annual_filing",
    "financial_data",
    "leadership",
    "quarterly_filing",
    "growth_metrics",
    "competitive_position",
}

MINIMUM_SOURCES = 5


class ResearchEvaluator(BaseEvaluator):
    """Score research progress based on topic coverage, confidence, and sources.

    Scoring formula (normalised to [-1, 1]):
        * Topic coverage   (40%): |topics_covered & required| / |required|
        * Thesis confidence (30%): thesis_confidence from ResearchState [0, 1]
        * Source diversity  (30%): |sources_consulted| / MINIMUM_SOURCES (capped at 1)

    CRITICAL: When ``pivot_discovered=True`` and ``pivot_analyzed=False``,
    the score is overridden to -0.35 so the graph's revision threshold
    (score <= -0.3) fires.
    """

    def __init__(self, research_state: ResearchState) -> None:
        self._state = research_state

    def evaluate(self, goal: Goal, state: StateSnapshot) -> EvalSignal:
        rs = self._state

        # --- Pivot override: force revision --------------------------------
        if rs.pivot_discovered and not rs.pivot_analyzed:
            return EvalSignal(
                score=-0.35,
                confidence=0.9,
                explanation=(
                    "ResearchEvaluator: PIVOT DETECTED — competitor strategic "
                    "pivot to AI discovered but not yet analyzed. Goal revision "
                    "required to incorporate this new intelligence."
                ),
                reasoning="Pivot discovered but not analyzed; forced below revision threshold.",
                criteria_scores={
                    "topic_coverage": self._topic_score(),
                    "thesis_confidence": rs.thesis_confidence,
                    "source_diversity": self._source_score(),
                    "pivot_coverage": -1.0,
                },
            )

        # --- Normal scoring ------------------------------------------------
        topic_score = self._topic_score()       # [0, 1]
        confidence_score = rs.thesis_confidence  # [0, 1]
        source_score = self._source_score()     # [0, 1]

        # Weighted composite in [0, 1], then map to [-1, 1]
        raw = 0.4 * topic_score + 0.3 * confidence_score + 0.3 * source_score
        score = raw * 2.0 - 1.0  # [0, 1] -> [-1, 1]
        score = max(-1.0, min(1.0, score))

        return EvalSignal(
            score=score,
            confidence=0.85,
            explanation=(
                f"ResearchEvaluator: topic={topic_score:.2f}, "
                f"confidence={confidence_score:.2f}, "
                f"sources={source_score:.2f} -> score={score:.4f}"
            ),
            reasoning=f"Composite research score across {len(rs.findings)} findings.",
            criteria_scores={
                "topic_coverage": topic_score,
                "thesis_confidence": confidence_score,
                "source_diversity": source_score,
            },
        )

    # -- helpers ---

    def _topic_score(self) -> float:
        """Fraction of required topics covered, [0, 1]."""
        if not REQUIRED_TOPICS:
            return 1.0
        covered = self._state.topics_covered & REQUIRED_TOPICS
        return len(covered) / len(REQUIRED_TOPICS)

    def _source_score(self) -> float:
        """Source diversity score, [0, 1]."""
        return min(1.0, len(self._state.sources_consulted) / MINIMUM_SOURCES)


class SourceDiversityChecker(BaseConstraintChecker):
    """Ensure no single source type exceeds 60% of all findings.

    Source types are identified by the prefix before the colon in
    ``ResearchFinding.source`` (e.g. ``web_search``, ``document_reader``).
    """

    def __init__(self, research_state: ResearchState) -> None:
        self._state = research_state

    def check(
        self,
        goal: Goal,
        state: StateSnapshot,
        action: ActionSpec | None = None,
    ) -> tuple[bool, str]:
        findings = self._state.findings
        if len(findings) < 3:
            return True, ""

        # Count findings per source type (prefix before ':')
        type_counts: dict[str, int] = {}
        for f in findings:
            source_type = f.source.split(":")[0] if ":" in f.source else f.source
            type_counts[source_type] = type_counts.get(source_type, 0) + 1

        total = len(findings)
        for source_type, count in type_counts.items():
            ratio = count / total
            if ratio > 0.60:
                return False, (
                    f"SourceDiversityChecker: {source_type} accounts for "
                    f"{ratio:.0%} of findings ({count}/{total}), exceeds 60% limit"
                )

        return True, ""
