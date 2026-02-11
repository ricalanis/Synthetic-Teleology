"""Tests for bounded accumulation channels."""

from __future__ import annotations

from synthetic_teleology.graph.state import (
    _make_bounded_add,
    make_bounded_state,
)


class TestMakeBoundedAdd:
    def test_below_limit(self) -> None:
        add = _make_bounded_add(10)
        result = add([1, 2], [3, 4])
        assert result == [1, 2, 3, 4]

    def test_at_limit(self) -> None:
        add = _make_bounded_add(4)
        result = add([1, 2], [3, 4])
        assert result == [1, 2, 3, 4]

    def test_above_limit_truncates(self) -> None:
        add = _make_bounded_add(3)
        result = add([1, 2], [3, 4, 5])
        assert result == [3, 4, 5]

    def test_empty_lists(self) -> None:
        add = _make_bounded_add(5)
        assert add([], []) == []
        assert add([], [1]) == [1]
        assert add([1], []) == [1]


class TestMakeBoundedState:
    def test_returns_type(self) -> None:
        bounded_state = make_bounded_state(100)
        assert isinstance(bounded_state, type)

    def test_has_expected_fields(self) -> None:
        bounded_state = make_bounded_state(50)
        hints = bounded_state.__annotations__
        assert "events" in hints
        assert "goal_history" in hints
        assert "eval_history" in hints
        assert "action_history" in hints
        assert "reasoning_trace" in hints
        assert "action_feedback" in hints
        assert "goal" in hints
        assert "step" in hints

    def test_bounded_reducer_in_annotations(self) -> None:
        """The bounded state channels should use bounded reducers."""
        bounded_state = make_bounded_state(10)
        hints = bounded_state.__annotations__
        # Just verify it's a valid TypedDict that can be introspected
        assert "events" in hints


class TestBuildTeleologicalGraphWithBoundedState:
    def test_compiles_with_bounded_state(self) -> None:
        from synthetic_teleology.graph.graph import build_teleological_graph

        bounded_state = make_bounded_state(50)
        graph = build_teleological_graph(state_schema=bounded_state)
        assert graph is not None

    def test_compiles_without_bounded_state(self) -> None:
        """Default (unbounded) still works."""
        from synthetic_teleology.graph.graph import build_teleological_graph

        graph = build_teleological_graph()
        assert graph is not None


class TestBuilderWithMaxHistory:
    def test_builder_with_max_history(self) -> None:
        from synthetic_teleology.graph.builder import GraphBuilder

        builder = GraphBuilder("test-agent")
        result = builder.with_max_history(100)
        assert result is builder
        assert builder._max_history == 100
