"""Tests for action feedback emission and observation enrichment."""

from __future__ import annotations

from synthetic_teleology.domain.values import (
    ActionSpec,
    EvalSignal,
    PolicySpec,
    StateSnapshot,
)
from synthetic_teleology.graph.nodes import (
    _build_enriched_observation,
    act_node,
    perceive_node,
)


def _make_state(step: int = 0, **overrides) -> dict:
    """Minimal state dict for testing act_node / perceive_node."""
    policy = PolicySpec(
        actions=[ActionSpec(name="do_thing", parameters={"x": 1})],
        metadata={},
    )
    state = {
        "step": step,
        "filtered_policy": policy,
        "state_snapshot": StateSnapshot(timestamp=1.0, observation="obs"),
        "tools": [],
        "act_fn": None,
        "transition_fn": None,
        "action_feedback": [],
        "eval_history": [],
        "goal_history": [],
    }
    state.update(overrides)
    return state


class TestActNodeFeedback:
    def test_act_node_emits_feedback(self):
        state = _make_state()
        result = act_node(state)
        assert "action_feedback" in result
        assert len(result["action_feedback"]) == 1
        fb = result["action_feedback"][0]
        assert fb["action"] == "do_thing"
        assert "timestamp" in fb
        assert fb["step"] == 0

    def test_act_node_no_action_emits_empty(self):
        empty_policy = PolicySpec(actions=[], metadata={})
        state = _make_state(filtered_policy=empty_policy)
        result = act_node(state)
        assert result["action_feedback"] == []

    def test_act_node_feedback_includes_tool_result(self):
        class FakeTool:
            name = "calc"
            def invoke(self, args):
                return "42"

        action = ActionSpec(
            name="calculate",
            parameters={"expr": "6*7"},
            tool_name="calc",
        )
        policy = PolicySpec(actions=[action], metadata={})
        state = _make_state(filtered_policy=policy, tools=[FakeTool()])
        result = act_node(state)
        assert len(result["action_feedback"]) == 1
        assert result["action_feedback"][0]["result"] == "42"
        assert result["action_feedback"][0]["tool_name"] == "calc"


class TestObservationEnrichment:
    def test_step_1_no_enrichment(self):
        result = _build_enriched_observation("base obs", {"step": 1})
        assert result == "base obs"

    def test_step_0_no_enrichment(self):
        result = _build_enriched_observation("base obs", {"step": 0})
        assert result == "base obs"

    def test_step_2_with_feedback_enriches(self):
        state = {
            "step": 2,
            "action_feedback": [
                {"action": "search", "tool_name": "web", "result": "found it", "step": 1},
            ],
            "eval_history": [],
            "goal_history": [],
        }
        result = _build_enriched_observation("base", state)
        assert "--- History ---" in result
        assert "search" in result
        assert "found it" in result

    def test_enrichment_includes_eval_trend(self):
        signals = [
            EvalSignal(score=s, confidence=0.8)
            for s in [0.3, 0.5, 0.7]
        ]
        state = {
            "step": 3,
            "action_feedback": [],
            "eval_history": signals,
            "goal_history": [],
        }
        result = _build_enriched_observation("base", state)
        assert "Eval score trend" in result
        assert "0.30" in result
        assert "0.70" in result

    def test_enrichment_includes_goal_revisions(self):
        state = {
            "step": 3,
            "action_feedback": [],
            "eval_history": [],
            "goal_history": ["goal_v1", "goal_v2"],
        }
        result = _build_enriched_observation("base", state)
        assert "2 goal revision(s)" in result

    def test_enrichment_respects_max_action_results(self):
        feedback = [
            {"action": f"act_{i}", "tool_name": None, "result": f"r{i}", "step": i}
            for i in range(5)
        ]
        state = {
            "step": 6,
            "action_feedback": feedback,
            "eval_history": [],
            "goal_history": [],
        }
        result = _build_enriched_observation("base", state)
        # Only last 3 should appear
        assert "act_2" in result
        assert "act_3" in result
        assert "act_4" in result
        assert "act_0" not in result
        assert "act_1" not in result


class TestPerceiveNodeEnrichment:
    def test_enriched_snapshot_context_has_recent_results(self):
        feedback = [
            {"action": "a1", "tool_name": None, "result": "r1", "step": 1},
        ]
        state = {
            "step": 1,
            "perceive_fn": lambda: StateSnapshot(timestamp=1.0, observation="raw"),
            "action_feedback": feedback,
            "eval_history": [],
            "goal_history": [],
        }
        result = perceive_node(state)
        snap = result["state_snapshot"]
        assert "recent_action_results" in snap.context
        assert len(snap.context["recent_action_results"]) == 1

    def test_numeric_mode_minimal_enrichment(self):
        """No feedback = no enrichment, step 1 returns base observation."""
        state = {
            "step": 0,
            "perceive_fn": lambda: StateSnapshot(
                timestamp=1.0, values=(1.0, 2.0), observation=""
            ),
            "action_feedback": [],
            "eval_history": [],
            "goal_history": [],
        }
        result = perceive_node(state)
        # Step becomes 1, no enrichment on step 1
        assert "History" not in result["observation"]
        assert "State values at step 1" in result["observation"]
