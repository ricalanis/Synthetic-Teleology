"""Custom evaluator and constraint checker for investment thesis building."""

from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.values import (
    ActionSpec,
    EvalSignal,
    StateSnapshot,
)
from synthetic_teleology.services.constraint_engine import BaseConstraintChecker
from synthetic_teleology.services.evaluation import BaseEvaluator

from .models import ThesisState

# Required thesis sections to be considered complete
REQUIRED_SECTIONS = (
    "financial_performance",
    "competitive_positioning",
    "management_quality",
    "risk_factors",
    "catalysts",
)


class ThesisEvaluator(BaseEvaluator):
    """Score investment thesis completeness and quality.

    Factors:
    - Thesis completeness (40%): sections filled vs REQUIRED_SECTIONS
    - Source diversity (30%): distinct sources vs minimum (5)
    - Risk coverage (30%): risk factors identified vs minimum (3)

    CRITICAL: When ``lawsuit_discovered=True`` but ``risk_analyzed=False``,
    the score drops to <= -0.3 to trigger goal revision.  This models the
    real-world scenario of material information requiring thesis reassessment.
    """

    def __init__(
        self,
        thesis_state: ThesisState,
        min_sources: int = 5,
        min_risk_factors: int = 3,
    ) -> None:
        self._state = thesis_state
        self._min_sources = min_sources
        self._min_risk_factors = min_risk_factors

    def evaluate(self, goal: Goal, state: StateSnapshot) -> EvalSignal:
        ts = self._state

        # -- Lawsuit discovered but not yet analyzed: force revision --
        if ts.lawsuit_discovered and not ts.risk_analyzed:
            return EvalSignal(
                score=-0.5,
                confidence=0.9,
                explanation=(
                    "Material litigation risk discovered (patent lawsuit). "
                    "Thesis requires revision to include litigation risk analysis. "
                    "Risk has NOT been analyzed yet."
                ),
            )

        # -- Thesis completeness (40%) --
        filled = sum(
            1 for section in REQUIRED_SECTIONS
            if section in ts.thesis_components and ts.thesis_components[section]
        )
        completeness = filled / len(REQUIRED_SECTIONS)

        # -- Source diversity (30%) --
        source_score = min(len(ts.sources) / max(self._min_sources, 1), 1.0)

        # -- Risk coverage (30%) --
        risk_score = min(
            len(ts.risk_factors) / max(self._min_risk_factors, 1), 1.0
        )

        # Weighted raw score [0, 1]
        raw = (
            0.4 * completeness
            + 0.3 * source_score
            + 0.3 * risk_score
        )

        # Normalize to [-0.2, 1.0]: research in progress should NOT trigger
        # revision (threshold is -0.3).  Only the explicit lawsuit override
        # above returns scores low enough to trigger revision.
        score = max(-1.0, min(1.0, raw * 1.2 - 0.2))

        # Confidence grows with data volume
        data_count = len(ts.financials) + len(ts.news) + len(ts.thesis_components)
        confidence = min(1.0, data_count / 15)

        return EvalSignal(
            score=score,
            confidence=confidence,
            explanation=(
                f"Completeness: {filled}/{len(REQUIRED_SECTIONS)} sections, "
                f"Sources: {len(ts.sources)}/{self._min_sources}, "
                f"Risks: {len(ts.risk_factors)}/{self._min_risk_factors}, "
                f"Lawsuit: {'YES' if ts.lawsuit_discovered else 'no'}, "
                f"Risk analyzed: {'YES' if ts.risk_analyzed else 'no'}"
            ),
        )


class SourceDiversityChecker(BaseConstraintChecker):
    """Require at least N different source types before thesis is accepted.

    Prevents over-reliance on a single data source.
    """

    def __init__(
        self,
        thesis_state: ThesisState,
        min_source_types: int = 3,
    ) -> None:
        self._state = thesis_state
        self._min_source_types = min_source_types

    def check(
        self,
        goal: Goal,
        state: StateSnapshot,
        action: ActionSpec | None = None,
    ) -> tuple[bool, str]:
        source_count = len(self._state.sources)

        # Only enforce source diversity once substantial research is
        # underway.  During early gathering steps, allow actions to
        # proceed so the agent can diversify sources naturally.
        data_points = len(self._state.financials) + len(self._state.news)
        if data_points < 8:
            return (True, "")

        if source_count < self._min_source_types:
            return (
                False,
                f"Only {source_count} source(s) used; "
                f"minimum {self._min_source_types} required for a balanced thesis",
            )

        return (True, "")
