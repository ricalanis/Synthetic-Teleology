"""Tests for ComponentRegistry."""

from __future__ import annotations

import pytest

from synthetic_teleology.infrastructure.registry import ComponentRegistry


class TestComponentRegistry:
    """Test ComponentRegistry register, get, has, decorator, duplicates."""

    def test_register_instance_and_get(self) -> None:
        reg = ComponentRegistry()
        reg.register_instance("evaluator", "numeric", "NumericEvaluator")
        assert reg.get("evaluator", "numeric") == "NumericEvaluator"

    def test_get_missing_raises(self) -> None:
        reg = ComponentRegistry()
        with pytest.raises(KeyError, match="not registered"):
            reg.get("evaluator", "missing")

    def test_get_or_none(self) -> None:
        reg = ComponentRegistry()
        assert reg.get_or_none("evaluator", "missing") is None
        reg.register_instance("evaluator", "a", 42)
        assert reg.get_or_none("evaluator", "a") == 42

    def test_has(self) -> None:
        reg = ComponentRegistry()
        assert reg.has("evaluator", "numeric") is False
        reg.register_instance("evaluator", "numeric", "value")
        assert reg.has("evaluator", "numeric") is True

    def test_register_decorator(self) -> None:
        reg = ComponentRegistry()

        @reg.register("evaluator", "test")
        class TestEvaluator:
            pass

        assert reg.has("evaluator", "test")
        assert reg.get("evaluator", "test") is TestEvaluator

    def test_duplicate_registration_raises(self) -> None:
        reg = ComponentRegistry()
        reg.register_instance("evaluator", "numeric", "v1")
        with pytest.raises(ValueError, match="already registered"):
            reg.register_instance("evaluator", "numeric", "v2")

    def test_overwrite_allowed(self) -> None:
        reg = ComponentRegistry()
        reg.register_instance("evaluator", "numeric", "v1")
        reg.register_instance("evaluator", "numeric", "v2", overwrite=True)
        assert reg.get("evaluator", "numeric") == "v2"

    def test_list_category(self) -> None:
        reg = ComponentRegistry()
        reg.register_instance("evaluator", "a", 1)
        reg.register_instance("evaluator", "b", 2)
        names = reg.list_category("evaluator")
        assert set(names) == {"a", "b"}

    def test_list_category_empty(self) -> None:
        reg = ComponentRegistry()
        assert reg.list_category("missing") == []

    def test_list_categories(self) -> None:
        reg = ComponentRegistry()
        reg.register_instance("evaluator", "a", 1)
        reg.register_instance("planner", "b", 2)
        categories = reg.list_categories()
        assert set(categories) == {"evaluator", "planner"}

    def test_count(self) -> None:
        reg = ComponentRegistry()
        assert reg.count() == 0
        reg.register_instance("evaluator", "a", 1)
        reg.register_instance("evaluator", "b", 2)
        reg.register_instance("planner", "c", 3)
        assert reg.count("evaluator") == 2
        assert reg.count("planner") == 1
        assert reg.count() == 3

    def test_unregister(self) -> None:
        reg = ComponentRegistry()
        reg.register_instance("evaluator", "a", 42)
        removed = reg.unregister("evaluator", "a")
        assert removed == 42
        assert not reg.has("evaluator", "a")

    def test_unregister_missing_raises(self) -> None:
        reg = ComponentRegistry()
        with pytest.raises(KeyError, match="not found"):
            reg.unregister("evaluator", "missing")

    def test_clear_category(self) -> None:
        reg = ComponentRegistry()
        reg.register_instance("evaluator", "a", 1)
        reg.register_instance("planner", "b", 2)
        reg.clear("evaluator")
        assert reg.count("evaluator") == 0
        assert reg.count("planner") == 1

    def test_clear_all(self) -> None:
        reg = ComponentRegistry()
        reg.register_instance("evaluator", "a", 1)
        reg.register_instance("planner", "b", 2)
        reg.clear()
        assert reg.count() == 0

    def test_all_entries(self) -> None:
        reg = ComponentRegistry()
        reg.register_instance("evaluator", "a", 1)
        reg.register_instance("evaluator", "b", 2)
        entries = reg.all_entries()
        assert "evaluator" in entries
        assert entries["evaluator"]["a"] == 1
        assert entries["evaluator"]["b"] == 2

    def test_contains(self) -> None:
        reg = ComponentRegistry()
        reg.register_instance("evaluator", "a", 1)
        assert ("evaluator", "a") in reg
        assert ("evaluator", "b") not in reg

    def test_repr(self) -> None:
        reg = ComponentRegistry()
        reg.register_instance("evaluator", "a", 1)
        r = repr(reg)
        assert "ComponentRegistry" in r
        assert "evaluator" in r
