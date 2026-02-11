"""Custom evaluator and constraint checkers for the curriculum agent.

The CurriculumEvaluator reads CurriculumState directly for deterministic
scoring.  The LLM planner + reviser are handled by the mock model.
"""

from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.values import ActionSpec, EvalSignal, StateSnapshot
from synthetic_teleology.services.constraint_engine import BaseConstraintChecker
from synthetic_teleology.services.evaluation import BaseEvaluator

from .models import CurriculumState


class CurriculumEvaluator(BaseEvaluator):
    """Score curriculum progress from simulated learner state.

    Scoring formula:
    - Topic mastery (40%): average mastery across completed topics
    - Coverage (30%): topics_completed / total topics
    - Efficiency (30%): lessons_delivered vs quizzes_passed ratio

    When a quiz fails on a topic with difficulty >= 0.5
    (state_management or harder), the score drops to <= -0.3 **once** to
    trigger goal revision.  Subsequent evaluations revert to the normal
    formula so the revision is not re-triggered every step.

    Normalises to [-1, 1].
    """

    def __init__(self, state: CurriculumState) -> None:
        self._state = state
        self._revision_triggered_for: set[int] = set()  # indices already fired

    def evaluate(self, goal: Goal, snapshot: StateSnapshot) -> EvalSignal:
        s = self._state
        total_topics = len(s.available_topics) or 1

        # --- Check for NEW quiz failure on a hard topic -------------------
        if s.quiz_results:
            idx = len(s.quiz_results) - 1
            latest = s.quiz_results[-1]
            if not latest.passed and idx not in self._revision_triggered_for:
                # Look up the topic difficulty
                topic_diff = 0.0
                for t in s.available_topics:
                    if t.name == latest.topic:
                        topic_diff = t.difficulty
                        break
                if topic_diff >= 0.5:
                    # Hard topic failure -> force revision (once)
                    self._revision_triggered_for.add(idx)
                    return EvalSignal(
                        score=-0.5,
                        confidence=0.8,
                        explanation=(
                            f"Quiz FAILED on hard topic '{latest.topic}' "
                            f"(score {latest.score:.2f}, difficulty {topic_diff:.1f}). "
                            f"Knowledge gap detected — revision required."
                        ),
                    )

        # --- Normal scoring -----------------------------------------------

        # Grace period: before any quizzes are taken, return a neutral score
        # to avoid premature revision while the agent is still ramping up.
        if not s.quiz_results:
            progress = min(1.0, s.lessons_delivered / max(total_topics, 1))
            return EvalSignal(
                score=-0.2 + 0.4 * progress,
                confidence=0.3 + 0.3 * progress,
                explanation=(
                    f"Grace period: {s.lessons_delivered} lessons delivered, "
                    f"no quizzes yet. Progress: {progress:.2f}"
                ),
            )

        # 1) Topic mastery (40%) — average mastery of completed topics
        if s.topics_completed:
            completed_mastery = [
                s.learner.strengths.get(t, 0.0) for t in s.topics_completed
            ]
            mastery_score = sum(completed_mastery) / len(completed_mastery)
        else:
            mastery_score = 0.0

        # 2) Coverage (30%) — fraction of curriculum completed
        coverage_score = len(s.topics_completed) / total_topics

        # 3) Efficiency (30%) — ratio of passed quizzes to lessons delivered
        passed_quizzes = sum(1 for q in s.quiz_results if q.passed)
        if s.lessons_delivered > 0:
            efficiency_score = min(1.0, passed_quizzes / s.lessons_delivered)
        else:
            efficiency_score = 0.0

        raw = (
            0.4 * mastery_score
            + 0.3 * coverage_score
            + 0.3 * efficiency_score
        )

        # Normalise to [-1, 1]: 0.5 raw -> 0.0 normalised
        score = max(-1.0, min(1.0, (raw - 0.5) * 2))

        return EvalSignal(
            score=score,
            confidence=min(1.0, s.lessons_delivered / max(total_topics * 0.3, 1)),
            explanation=(
                f"Mastery: {mastery_score:.2f}, "
                f"Coverage: {len(s.topics_completed)}/{total_topics}, "
                f"Efficiency: {efficiency_score:.2f}, "
                f"Raw: {raw:.3f}, Score: {score:.3f}"
            ),
        )


class PrerequisiteChecker(BaseConstraintChecker):
    """Ensure the current topic's prerequisites are all completed with mastery >= 0.5."""

    def __init__(self, state: CurriculumState) -> None:
        self._state = state

    def check(
        self,
        goal: Goal,
        snapshot: StateSnapshot,
        action: ActionSpec | None = None,
    ) -> tuple[bool, str]:
        current = self._state.current_topic
        if current is None:
            return (True, "")

        topic = None
        for t in self._state.available_topics:
            if t.name == current:
                topic = t
                break
        if topic is None:
            return (True, "")

        for prereq in topic.prerequisites:
            if prereq not in self._state.topics_completed:
                return (
                    False,
                    f"Prerequisite '{prereq}' not completed for topic '{current}'",
                )
            mastery = self._state.learner.strengths.get(prereq, 0.0)
            if mastery < 0.5:
                return (
                    False,
                    f"Prerequisite '{prereq}' mastery {mastery:.2f} < 0.5 "
                    f"for topic '{current}'",
                )
        return (True, "")


class TimeBudgetChecker(BaseConstraintChecker):
    """Limit total learner time spent to max_time minutes.

    Each lesson costs ~5 min, each quiz ~3 min.
    """

    def __init__(
        self,
        state: CurriculumState,
        max_time: float = 120.0,
    ) -> None:
        self._state = state
        self._max_time = max_time

    def check(
        self,
        goal: Goal,
        snapshot: StateSnapshot,
        action: ActionSpec | None = None,
    ) -> tuple[bool, str]:
        if self._state.learner.total_time_spent >= self._max_time:
            return (
                False,
                f"Time budget exhausted: {self._state.learner.total_time_spent:.0f} min "
                f"/ {self._max_time:.0f} min",
            )
        return (True, "")
