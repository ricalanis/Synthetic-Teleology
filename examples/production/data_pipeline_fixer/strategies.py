"""Custom evaluator and constraint checkers for the data pipeline fixer.

PipelineEvaluator reads simulated PipelineState for deterministic scoring.
BudgetConstraintChecker and SafetyConstraintChecker enforce operational limits.
"""

from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.values import ActionSpec, EvalSignal, StateSnapshot
from synthetic_teleology.services.constraint_engine import BaseConstraintChecker
from synthetic_teleology.services.evaluation import BaseEvaluator

from .models import PipelineState


class PipelineEvaluator(BaseEvaluator):
    """Score pipeline health from simulated PipelineState.

    Weighted factors:
    - Health score (50%): pipeline_state.health_score
    - Error rate  (30%): 1.0 - (error_rate / 0.1) clamped to [0, 1]
    - Throughput  (20%): throughput / baseline clamped to [0, 1]

    When schema_drift_detected and health_score < 0.7, the raw score
    is forced below the -0.3 revision threshold so the LLM reviser fires.

    Final score is normalized from [0, 1] raw to [-1, 1].
    """

    def __init__(
        self,
        pipeline_state: PipelineState,
        baseline_throughput: float = 1000.0,
    ) -> None:
        self._state = pipeline_state
        self._baseline = baseline_throughput

    def evaluate(self, goal: Goal, state: StateSnapshot) -> EvalSignal:
        s = self._state

        # --- Component scores (each in [0, 1]) ---
        health_component = max(0.0, min(1.0, s.health_score))

        error_component = max(0.0, min(1.0, 1.0 - (s.error_rate / 0.1)))

        throughput_component = max(
            0.0, min(1.0, s.throughput / self._baseline)
        )

        raw = (
            0.50 * health_component
            + 0.30 * error_component
            + 0.20 * throughput_component
        )

        # When schema drift is active and health is degraded,
        # force the score into revision territory (<= -0.3).
        if s.schema_drift_detected and s.health_score < 0.70:
            raw = min(raw, 0.30)

        # Normalize [0, 1] raw -> [-1, 1]
        score = max(-1.0, min(1.0, (raw - 0.5) * 2.0))

        return EvalSignal(
            score=score,
            confidence=min(1.0, health_component),
            explanation=(
                f"Pipeline eval: health={s.health_score:.2f}, "
                f"error_rate={s.error_rate:.2f}, "
                f"throughput={s.throughput:.0f}, "
                f"schema_drift={s.schema_drift_detected}"
            ),
        )


class BudgetConstraintChecker(BaseConstraintChecker):
    """Limits total fix attempts to max_fixes.

    Prevents unbounded repair loops by capping the number of fixes.
    """

    def __init__(
        self,
        pipeline_state: PipelineState,
        max_fixes: int = 15,
    ) -> None:
        self._state = pipeline_state
        self._max_fixes = max_fixes

    def check(
        self,
        goal: Goal,
        state: StateSnapshot,
        action: ActionSpec | None = None,
    ) -> tuple[bool, str]:
        if self._state.fix_attempts_count >= self._max_fixes:
            return (
                False,
                f"Fix budget exhausted: {self._state.fix_attempts_count}"
                f"/{self._max_fixes} attempts used",
            )
        return (True, "")


class SafetyConstraintChecker(BaseConstraintChecker):
    """Blocks dangerous fix actions like drop_table.

    Inspects action parameters and rejects any action whose fix_type
    is 'drop_table'.
    """

    def check(
        self,
        goal: Goal,
        state: StateSnapshot,
        action: ActionSpec | None = None,
    ) -> tuple[bool, str]:
        if action is None:
            return (True, "")

        fix_type = action.parameters.get("fix_type", "")
        if fix_type == "drop_table":
            target = action.parameters.get("target", "unknown")
            return (
                False,
                f"Safety violation: drop_table on '{target}' is forbidden",
            )
        return (True, "")
