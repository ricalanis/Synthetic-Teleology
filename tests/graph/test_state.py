"""Tests for TeleologicalState TypedDict."""

from __future__ import annotations

from synthetic_teleology.graph.state import TeleologicalState


class TestTeleologicalState:
    """Ensure the TypedDict can be constructed and has expected keys."""

    def test_empty_state(self) -> None:
        """TeleologicalState with total=False allows empty dict."""
        state: TeleologicalState = {}  # type: ignore[typeddict-item]
        assert isinstance(state, dict)

    def test_partial_state(self) -> None:
        """Can construct with a subset of keys."""
        state: TeleologicalState = {"step": 1, "max_steps": 100}  # type: ignore[typeddict-item]
        assert state["step"] == 1
        assert state["max_steps"] == 100

    def test_annotations_present(self) -> None:
        """Key annotations are available in __annotations__."""
        annotations = TeleologicalState.__annotations__
        assert "step" in annotations
        assert "goal" in annotations
        assert "evaluator" in annotations
        assert "events" in annotations
        assert "metadata" in annotations

    def test_accumulation_channels_exist(self) -> None:
        """The four append-reducer channels are annotated."""
        annotations = TeleologicalState.__annotations__
        for channel in ("events", "goal_history", "eval_history", "action_history"):
            assert channel in annotations
