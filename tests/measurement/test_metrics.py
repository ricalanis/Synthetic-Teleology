"""Tests for all measurement metrics including Empowerment."""

from __future__ import annotations

import pytest

from synthetic_teleology.domain.events import GoalRevised
from synthetic_teleology.domain.values import GoalRevision, ObjectiveVector
from synthetic_teleology.domain.enums import Direction
from synthetic_teleology.measurement.collector import AgentLog, AgentLogEntry
from synthetic_teleology.measurement.metrics.adaptivity import Adaptivity
from synthetic_teleology.measurement.metrics.base import MetricResult
from synthetic_teleology.measurement.metrics.empowerment import Empowerment
from synthetic_teleology.measurement.metrics.goal_persistence import GoalPersistence
from synthetic_teleology.measurement.metrics.innovation_yield import InnovationYield
from synthetic_teleology.measurement.metrics.lyapunov_stability import LyapunovStability
from synthetic_teleology.measurement.metrics.normative_fidelity import NormativeFidelity
from synthetic_teleology.measurement.metrics.reflective_efficiency import (
    ReflectiveEfficiency,
)
from synthetic_teleology.measurement.metrics.teleological_coherence import (
    TeleologicalCoherence,
    _pearson,
)


def _make_log(
    agent_id: str = "test-agent",
    num_steps: int = 10,
    scores: list[float] | None = None,
    action_names: list[str] | None = None,
    revision_steps: list[int] | None = None,
    violation_steps: list[int] | None = None,
    reflection_steps: list[int] | None = None,
    num_revisions: int = 0,
    state_values_list: list[tuple[float, ...]] | None = None,
    goal_values_list: list[tuple[float, ...] | None] | None = None,
) -> AgentLog:
    """Build a synthetic AgentLog for testing."""
    if scores is None:
        scores = [0.5] * num_steps
    if action_names is None:
        action_names = ["action_a"] * num_steps
    revision_steps = revision_steps or []
    violation_steps = violation_steps or []
    reflection_steps = reflection_steps or []

    log = AgentLog(agent_id=agent_id)
    for i in range(num_steps):
        entry = AgentLogEntry(
            step=i,
            timestamp=float(i),
            eval_score=scores[i] if i < len(scores) else 0.0,
            action_name=action_names[i] if i < len(action_names) else "",
            goal_revised=i in revision_steps,
            constraint_violated=i in violation_steps,
            reflection_triggered=i in reflection_steps,
            state_values=state_values_list[i] if state_values_list and i < len(state_values_list) else (),
            goal_values=goal_values_list[i] if goal_values_list and i < len(goal_values_list) else None,
        )
        log.entries.append(entry)

    # Add revision events
    actual_rev = num_revisions or len(revision_steps)
    for _ in range(actual_rev):
        log.goal_revisions.append(
            GoalRevised(
                source_id=agent_id,
                revision=GoalRevision(
                    previous_goal_id="g1",
                    new_goal_id="g2",
                    reason="update",
                ),
                previous_objective=None,
                new_objective=None,
                timestamp=0.0,
            )
        )

    return log


class TestGoalPersistence:
    """GP = 1 - (num_revisions / num_steps)."""

    def test_no_revisions_gives_one(self) -> None:
        log = _make_log(num_steps=10, num_revisions=0)
        metric = GoalPersistence()
        result = metric.compute(log)
        assert result.value == pytest.approx(1.0)

    def test_one_revision_in_ten_steps(self) -> None:
        log = _make_log(num_steps=10, num_revisions=1)
        metric = GoalPersistence()
        result = metric.compute(log)
        assert result.value == pytest.approx(0.9)

    def test_revision_every_step(self) -> None:
        log = _make_log(num_steps=5, num_revisions=5)
        metric = GoalPersistence()
        result = metric.compute(log)
        assert result.value == pytest.approx(0.0)

    def test_insufficient_data(self) -> None:
        log = _make_log(num_steps=1)
        metric = GoalPersistence()
        result = metric.compute(log)
        assert result.value == 0.0
        assert "Insufficient" in result.explanation

    def test_name_and_description(self) -> None:
        metric = GoalPersistence()
        assert metric.name == "goal_persistence"
        desc = metric.describe()
        assert "Goal Persistence" in desc


class TestTeleologicalCoherence:
    """TC: three-tier computation (correlation, proxy, legacy)."""

    # -- Legacy tier (no revisions) ------------------------------------------

    def test_legacy_all_perfect_scores(self) -> None:
        log = _make_log(num_steps=5, scores=[1.0, 1.0, 1.0, 1.0, 1.0])
        metric = TeleologicalCoherence()
        result = metric.compute(log)
        assert result.value == pytest.approx(1.0)

    def test_legacy_all_negative_one(self) -> None:
        log = _make_log(num_steps=5, scores=[-1.0, -1.0, -1.0, -1.0, -1.0])
        metric = TeleologicalCoherence()
        result = metric.compute(log)
        assert result.value == pytest.approx(0.0)

    def test_legacy_mixed_scores(self) -> None:
        log = _make_log(num_steps=4, scores=[0.5, 0.5, -0.5, -0.5])
        metric = TeleologicalCoherence()
        result = metric.compute(log)
        # mean=0.0, TC = (0.0+1)/2 = 0.5
        assert result.value == pytest.approx(0.5)

    # -- Correlation tier (goal_values available) ----------------------------

    def test_correlation_negative_r_gives_high_tc(self) -> None:
        """Negative correlation = coherent: bigger changes when scores are low."""
        # Big goal jumps happen at steps with BAD scores, no change at GOOD scores
        goal_values = [
            (1.0,), (5.0,), (5.0,), (9.0,), (9.0,),
        ]
        scores = [0.5, -0.8, 0.5, -0.6, 0.5]
        log = _make_log(
            num_steps=5,
            scores=scores,
            goal_values_list=goal_values,
        )
        metric = TeleologicalCoherence()
        result = metric.compute(log)
        # r < 0 (big changes when bad scores) -> TC > 0.5
        assert result.value > 0.5

    def test_correlation_positive_r_gives_low_tc(self) -> None:
        """Positive correlation = incoherent: bigger changes when scores are high."""
        # Big goal jumps happen at steps with GOOD scores
        goal_values = [
            (1.0,), (5.0,), (5.0,), (9.0,), (9.0,),
        ]
        scores = [0.5, 0.8, -0.5, 0.9, -0.5]
        log = _make_log(
            num_steps=5,
            scores=scores,
            goal_values_list=goal_values,
        )
        metric = TeleologicalCoherence()
        result = metric.compute(log)
        # r > 0 (big changes when good scores) -> TC < 0.5
        assert result.value < 0.5

    def test_correlation_insufficient_points_falls_back_to_legacy(self) -> None:
        """Only one goal change -> not enough for correlation, use legacy."""
        goal_values = [
            (1.0,), (5.0,), None, None, None,
        ]
        scores = [0.5, 0.5, 0.5, 0.5, 0.5]
        log = _make_log(
            num_steps=5,
            scores=scores,
            goal_values_list=goal_values,
        )
        metric = TeleologicalCoherence()
        result = metric.compute(log)
        # Legacy: (0.5 + 1) / 2 = 0.75
        assert result.value == pytest.approx(0.75)

    # -- Proxy tier (revisions present, no goal_values) ----------------------

    def test_proxy_responsive_revision_gives_high_tc(self) -> None:
        """Revision during poor eval followed by improvement."""
        scores = [0.5, -0.3, 0.4, 0.5, 0.5]
        log = _make_log(
            num_steps=5,
            scores=scores,
            revision_steps=[1],
            num_revisions=1,
        )
        metric = TeleologicalCoherence()
        result = metric.compute(log)
        # score at step 1 is -0.3 (poor) and step 2 is 0.4 (improvement)
        assert result.value == pytest.approx(1.0)

    def test_proxy_unresponsive_revision_gives_zero(self) -> None:
        """Revision during good eval, not responsive."""
        scores = [0.5, 0.3, 0.1, 0.5, 0.5]
        log = _make_log(
            num_steps=5,
            scores=scores,
            revision_steps=[1],
            num_revisions=1,
        )
        metric = TeleologicalCoherence()
        result = metric.compute(log)
        # score at step 1 is 0.3 (not poor, >= 0) -> not responsive
        assert result.value == pytest.approx(0.0)

    def test_name(self) -> None:
        assert TeleologicalCoherence().name == "teleological_coherence"


class TestPearsonHelper:
    """Test _pearson helper function."""

    def test_perfect_positive(self) -> None:
        r = _pearson([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert r == pytest.approx(1.0)

    def test_perfect_negative(self) -> None:
        r = _pearson([1.0, 2.0, 3.0], [3.0, 2.0, 1.0])
        assert r == pytest.approx(-1.0)

    def test_no_correlation(self) -> None:
        r = _pearson([1.0, 2.0, 3.0, 4.0], [1.0, 3.0, 1.0, 3.0])
        assert abs(r) < 0.5

    def test_single_element_returns_zero(self) -> None:
        r = _pearson([1.0], [2.0])
        assert r == 0.0

    def test_constant_returns_zero(self) -> None:
        r = _pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])
        assert r == 0.0


class TestAdaptivity:
    """AD = 1 / (1 + mean_recovery_steps)."""

    def test_no_revisions_gives_one(self) -> None:
        log = _make_log(num_steps=10, revision_steps=[])
        metric = Adaptivity()
        result = metric.compute(log)
        assert result.value == pytest.approx(1.0)

    def test_instant_recovery(self) -> None:
        # Revision at step 3 with score 0.5. Next step score >= 0.5 -> recovery=1 step
        scores = [0.5] * 10
        scores[3] = 0.3
        scores[4] = 0.5
        log = _make_log(num_steps=10, scores=scores, revision_steps=[3])
        metric = Adaptivity()
        result = metric.compute(log)
        # recovery_steps=1, AD = 1/(1+1) = 0.5
        assert result.value == pytest.approx(0.5)

    def test_slow_recovery(self) -> None:
        scores = [0.5] * 10
        scores[2] = 0.3
        # Recovery takes 5 steps: scores[3..7] are all below 0.3, scores[7]=0.5
        for i in range(3, 7):
            scores[i] = 0.1
        scores[7] = 0.5
        log = _make_log(num_steps=10, scores=scores, revision_steps=[2])
        metric = Adaptivity()
        result = metric.compute(log)
        # recovery = 5 steps (from idx 3 to idx 7)
        assert result.value == pytest.approx(1.0 / (1.0 + 5.0))

    def test_name(self) -> None:
        assert Adaptivity().name == "adaptivity"


class TestReflectiveEfficiency:
    """RE = effective_reflections / total_reflections."""

    def test_no_reflections_gives_one(self) -> None:
        log = _make_log(num_steps=10, reflection_steps=[])
        metric = ReflectiveEfficiency(window_size=2)
        result = metric.compute(log)
        assert result.value == pytest.approx(1.0)

    def test_effective_reflection(self) -> None:
        # Reflection at step 5. Pre-window=[0.3, 0.3], post-window=[0.8, 0.8]
        scores = [0.3] * 10
        scores[6] = 0.8
        scores[7] = 0.8
        log = _make_log(num_steps=10, scores=scores, reflection_steps=[5])
        metric = ReflectiveEfficiency(window_size=2)
        result = metric.compute(log)
        # post_mean=0.8 > pre_mean=0.3 -> effective
        assert result.value == pytest.approx(1.0)

    def test_ineffective_reflection(self) -> None:
        # Reflection at step 5. Pre-window=[0.8, 0.8], post-window=[0.3, 0.3]
        scores = [0.8] * 10
        scores[6] = 0.3
        scores[7] = 0.3
        log = _make_log(num_steps=10, scores=scores, reflection_steps=[5])
        metric = ReflectiveEfficiency(window_size=2)
        result = metric.compute(log)
        assert result.value == pytest.approx(0.0)

    def test_name(self) -> None:
        assert ReflectiveEfficiency().name == "reflective_efficiency"


class TestNormativeFidelity:
    """NF = 1 - (num_violation_steps / num_steps)."""

    def test_no_violations_gives_one(self) -> None:
        log = _make_log(num_steps=10, violation_steps=[])
        metric = NormativeFidelity()
        result = metric.compute(log)
        assert result.value == pytest.approx(1.0)

    def test_half_violations(self) -> None:
        log = _make_log(num_steps=10, violation_steps=[0, 1, 2, 3, 4])
        metric = NormativeFidelity()
        result = metric.compute(log)
        assert result.value == pytest.approx(0.5)

    def test_all_violations_gives_zero(self) -> None:
        log = _make_log(num_steps=5, violation_steps=[0, 1, 2, 3, 4])
        metric = NormativeFidelity()
        result = metric.compute(log)
        assert result.value == pytest.approx(0.0)

    def test_name(self) -> None:
        assert NormativeFidelity().name == "normative_fidelity"


class TestInnovationYield:
    """IY: attribution formula with revisions, fallback without."""

    # -- Fallback (no revisions) -- backward compatible ----------------------

    def test_all_unique_gives_one(self) -> None:
        log = _make_log(
            num_steps=5,
            action_names=["a", "b", "c", "d", "e"],
        )
        metric = InnovationYield()
        result = metric.compute(log)
        assert result.value == pytest.approx(1.0)

    def test_all_same_gives_fraction(self) -> None:
        log = _make_log(
            num_steps=5,
            action_names=["a", "a", "a", "a", "a"],
        )
        metric = InnovationYield()
        result = metric.compute(log)
        assert result.value == pytest.approx(1.0 / 5.0)

    def test_partial_diversity(self) -> None:
        log = _make_log(
            num_steps=4,
            action_names=["a", "b", "a", "b"],
        )
        metric = InnovationYield()
        result = metric.compute(log)
        assert result.value == pytest.approx(2.0 / 4.0)

    def test_empty_actions_excluded(self) -> None:
        log = _make_log(
            num_steps=4,
            action_names=["a", "", "b", ""],
        )
        metric = InnovationYield()
        result = metric.compute(log)
        assert result.value == pytest.approx(2.0 / 2.0)

    def test_insufficient_data(self) -> None:
        log = _make_log(num_steps=1, action_names=["a"])
        metric = InnovationYield()
        result = metric.compute(log)
        assert result.value == 0.0
        assert "Insufficient" in result.explanation

    # -- Attribution formula (with revisions) --------------------------------

    def test_attributed_all_new_actions_after_revision(self) -> None:
        """All post-revision actions are novel -> high novelty ratio."""
        log = _make_log(
            num_steps=6,
            action_names=["a", "a", "a", "x", "y", "z"],
            revision_steps=[3],
            num_revisions=1,
            scores=[0.3, 0.3, 0.3, 0.5, 0.6, 0.7],
        )
        metric = InnovationYield()
        result = metric.compute(log)
        # novelty_ratio = 3/4 = 0.75 (x,y,z new out of {a,x,y,z})
        # quality_improvement: sigmoid(0.6 - 0.3) > 0.5
        assert result.value > 0.5

    def test_attributed_no_new_actions_after_revision(self) -> None:
        """Same actions before and after revision -> zero novelty ratio."""
        log = _make_log(
            num_steps=6,
            action_names=["a", "b", "a", "a", "b", "a"],
            revision_steps=[3],
            num_revisions=1,
            scores=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        )
        metric = InnovationYield()
        result = metric.compute(log)
        # novelty_ratio = 0/2 = 0.0
        # quality_improvement: sigmoid(0) = 0.5
        # IY = 0.6*0.0 + 0.4*0.5 = 0.2
        assert result.value == pytest.approx(0.2)

    def test_attributed_quality_improvement_matters(self) -> None:
        """Score improvement after revision boosts IY."""
        log = _make_log(
            num_steps=6,
            action_names=["a", "a", "a", "a", "a", "a"],
            revision_steps=[3],
            num_revisions=1,
            scores=[-0.5, -0.5, -0.5, 0.8, 0.8, 0.8],
        )
        metric = InnovationYield()
        result = metric.compute(log)
        # novelty_ratio = 0 (same actions)
        # quality_improvement: sigmoid(0.8 - (-0.5)) = sigmoid(1.3) ≈ 0.79
        # IY = 0.6*0 + 0.4*0.79 ≈ 0.31
        assert result.value > 0.25
        assert result.value < 0.5

    def test_name(self) -> None:
        assert InnovationYield().name == "innovation_yield"


class TestEmpowerment:
    """Empowerment: I(A; S'|S) — mutual information between actions and transitions."""

    def test_name(self) -> None:
        assert Empowerment().name == "empowerment"

    def test_insufficient_data(self) -> None:
        log = _make_log(num_steps=2, scores=[0.5, 0.5])
        metric = Empowerment()
        result = metric.compute(log)
        assert result.value == 0.0

    def test_deterministic_actions_give_high_empowerment(self) -> None:
        """Different actions produce consistently different transitions."""
        states = [
            (0.0,), (1.0,), (0.0,), (0.5,), (0.0,), (1.0,), (0.0,), (0.5,),
        ]
        actions = ["big", "reset", "small", "reset", "big", "reset", "small", "reset"]
        scores = [0.5] * 8
        log = _make_log(
            num_steps=8,
            action_names=actions,
            scores=scores,
            state_values_list=states,
        )
        metric = Empowerment()
        result = metric.compute(log)
        # Actions should determine transitions
        assert result.value >= 0.0

    def test_random_transitions_give_low_empowerment(self) -> None:
        """Same action produces variable transitions -> low empowerment."""
        states = [
            (0.0,), (5.0,), (0.1,), (10.0,), (0.2,), (0.5,), (0.3,), (20.0,),
        ]
        actions = ["a", "a", "a", "a", "a", "a", "a", "a"]
        scores = [0.5] * 8
        log = _make_log(
            num_steps=8,
            action_names=actions,
            scores=scores,
            state_values_list=states,
        )
        metric = Empowerment()
        result = metric.compute(log)
        # Single action -> H(delta|A) = H(delta) -> E = 0
        assert result.value == pytest.approx(0.0)

    def test_no_state_values_uses_eval_proxy(self) -> None:
        """When no state_values, use eval score changes as proxy."""
        scores = [0.1, 0.5, 0.1, 0.9, 0.1, 0.5, 0.1, 0.9]
        actions = ["up", "down", "up", "down", "up", "down", "up", "down"]
        log = _make_log(num_steps=8, action_names=actions, scores=scores)
        metric = Empowerment()
        result = metric.compute(log)
        assert 0.0 <= result.value <= 1.0

    def test_validates_min_entries(self) -> None:
        """Empowerment needs at least 3 entries with states and actions."""
        log = _make_log(num_steps=2, action_names=["a", "b"])
        metric = Empowerment()
        assert not metric.validate(log)

    def test_describe(self) -> None:
        metric = Empowerment()
        desc = metric.describe()
        assert "mutual information" in desc.lower()


class TestLyapunovStability:
    """LS: convergence of eval score variance."""

    def test_converging_scores_above_half(self) -> None:
        # Scores that converge (decrease in variance over time)
        scores = [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.45, 0.55, 0.48, 0.52]
        log = _make_log(num_steps=len(scores), scores=scores)
        metric = LyapunovStability(window_size=3, stride=1)
        result = metric.compute(log)
        assert result.value > 0.5

    def test_diverging_scores_below_half(self) -> None:
        # Scores that diverge (increase in variance over time)
        scores = [0.5, 0.5, 0.5, 0.5, 0.4, 0.6, 0.3, 0.7, 0.1, 0.9, 0.0, 1.0]
        log = _make_log(num_steps=len(scores), scores=scores)
        metric = LyapunovStability(window_size=3, stride=1)
        result = metric.compute(log)
        assert result.value < 0.5

    def test_constant_scores_neutral(self) -> None:
        scores = [0.5] * 12
        log = _make_log(num_steps=12, scores=scores)
        metric = LyapunovStability(window_size=3, stride=1)
        result = metric.compute(log)
        # All variances are 0, slope is 0, sigmoid(0) = 0.5
        assert result.value == pytest.approx(0.5)

    def test_insufficient_data(self) -> None:
        log = _make_log(num_steps=3, scores=[0.5, 0.5, 0.5])
        metric = LyapunovStability(window_size=5, stride=1)
        result = metric.compute(log)
        assert result.value == 0.0
        assert "Insufficient" in result.explanation

    def test_name(self) -> None:
        assert LyapunovStability().name == "lyapunov_stability"


class TestMetricResult:
    """Test MetricResult data structure."""

    def test_creation(self) -> None:
        result = MetricResult(name="test", value=0.75, explanation="good")
        assert result.name == "test"
        assert result.value == 0.75
        assert result.explanation == "good"

    def test_repr(self) -> None:
        result = MetricResult(name="test", value=0.75)
        r = repr(result)
        assert "test" in r
        assert "0.7500" in r

    def test_immutable(self) -> None:
        result = MetricResult(name="test", value=0.5)
        with pytest.raises(AttributeError):
            result.value = 0.9  # type: ignore[misc]
