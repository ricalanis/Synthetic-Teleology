"""Tests for GoalAuditTrail and KnowledgeStore."""

from __future__ import annotations

import json

import pytest

from synthetic_teleology.domain.enums import GoalOrigin
from synthetic_teleology.domain.values import GoalAuditEntry, GoalProvenance, KnowledgeEntry
from synthetic_teleology.infrastructure.knowledge_store import KnowledgeStore
from synthetic_teleology.services.audit_trail import GoalAuditTrail


# ===================================================================== #
#  GoalAuditTrail tests                                                  #
# ===================================================================== #


class TestGoalAuditTrail:
    def test_record_creates_entry(self) -> None:
        trail = GoalAuditTrail()
        entry = trail.record(goal_id="g1", previous_goal_id="g0", revision_reason="test")
        assert entry.goal_id == "g1"
        assert entry.previous_goal_id == "g0"
        assert len(trail) == 1

    def test_query_by_goal_id(self) -> None:
        trail = GoalAuditTrail()
        trail.record(goal_id="g1", revision_reason="a")
        trail.record(goal_id="g2", revision_reason="b")
        trail.record(goal_id="g1", revision_reason="c")
        results = trail.query(goal_id="g1")
        assert len(results) == 2

    def test_query_by_reason(self) -> None:
        trail = GoalAuditTrail()
        trail.record(goal_id="g1", revision_reason="threshold")
        trail.record(goal_id="g2", revision_reason="gradient")
        trail.record(goal_id="g3", revision_reason="threshold")
        results = trail.query(reason="threshold")
        assert len(results) == 2

    def test_query_limit(self) -> None:
        trail = GoalAuditTrail()
        for i in range(10):
            trail.record(goal_id=f"g{i}")
        results = trail.query(limit=3)
        assert len(results) == 3

    def test_provenance_recorded(self) -> None:
        trail = GoalAuditTrail()
        prov = GoalProvenance(
            origin=GoalOrigin.USER,
            source_agent_id="agent-1",
            source_description="User directive",
        )
        entry = trail.record(goal_id="g1", provenance=prov)
        assert entry.provenance is not None
        assert entry.provenance.origin == GoalOrigin.USER

    def test_to_dict_and_from_dict(self) -> None:
        trail = GoalAuditTrail()
        prov = GoalProvenance(origin=GoalOrigin.ENDOGENOUS, source_agent_id="a1")
        trail.record(goal_id="g1", revision_reason="test", provenance=prov)
        trail.record(goal_id="g2", revision_reason="test2")

        data = trail.to_dict()
        assert len(data) == 2

        restored = GoalAuditTrail.from_dict(data)
        assert len(restored) == 2
        assert restored.entries[0].provenance is not None
        assert restored.entries[0].provenance.origin == GoalOrigin.ENDOGENOUS

    def test_to_json_and_from_json(self) -> None:
        trail = GoalAuditTrail()
        trail.record(goal_id="g1", eval_score=0.5, eval_confidence=0.8)

        json_str = trail.to_json()
        parsed = json.loads(json_str)
        assert len(parsed) == 1

        restored = GoalAuditTrail.from_json(json_str)
        assert len(restored) == 1
        assert restored.entries[0].eval_score == 0.5

    def test_integration_with_knowledge_store(self) -> None:
        store = KnowledgeStore()
        trail = GoalAuditTrail(knowledge_store=store)
        trail.record(goal_id="g1", revision_reason="test")
        # Audit trail should have cross-referenced in store
        assert len(store) >= 1

    def test_entries_property(self) -> None:
        trail = GoalAuditTrail()
        trail.record(goal_id="g1")
        entries = trail.entries
        assert len(entries) == 1
        # Verify it's a copy
        entries.clear()
        assert len(trail) == 1


# ===================================================================== #
#  KnowledgeStore tests                                                  #
# ===================================================================== #


class TestKnowledgeStore:
    def test_put_and_get(self) -> None:
        store = KnowledgeStore()
        entry = store.put(key="k1", value="hello", source="test")
        assert entry.key == "k1"
        assert entry.value == "hello"

        retrieved = store.get("k1")
        assert retrieved is not None
        assert retrieved.value == "hello"

    def test_get_missing_returns_none(self) -> None:
        store = KnowledgeStore()
        assert store.get("nonexistent") is None

    def test_put_overwrites(self) -> None:
        store = KnowledgeStore()
        store.put(key="k1", value="v1")
        store.put(key="k1", value="v2")
        assert store.get("k1").value == "v2"
        assert len(store) == 1

    def test_query_by_tags(self) -> None:
        store = KnowledgeStore()
        store.put("k1", "v1", tags=("a", "b"))
        store.put("k2", "v2", tags=("b", "c"))
        store.put("k3", "v3", tags=("a", "b", "c"))

        results = store.query_by_tags("a", "b")
        assert len(results) == 2  # k1 and k3

    def test_query_by_source(self) -> None:
        store = KnowledgeStore()
        store.put("k1", "v1", source="agent-1")
        store.put("k2", "v2", source="agent-2")
        store.put("k3", "v3", source="agent-1")

        results = store.query_by_source("agent-1")
        assert len(results) == 2

    def test_delete(self) -> None:
        store = KnowledgeStore()
        store.put("k1", "v1")
        assert store.delete("k1")
        assert "k1" not in store
        assert not store.delete("k1")

    def test_clear(self) -> None:
        store = KnowledgeStore()
        store.put("k1", "v1")
        store.put("k2", "v2")
        store.clear()
        assert len(store) == 0

    def test_contains(self) -> None:
        store = KnowledgeStore()
        store.put("k1", "v1")
        assert "k1" in store
        assert "k2" not in store

    def test_keys(self) -> None:
        store = KnowledgeStore()
        store.put("k1", "v1")
        store.put("k2", "v2")
        keys = store.keys()
        assert set(keys) == {"k1", "k2"}

    def test_to_dict_and_from_dict(self) -> None:
        store = KnowledgeStore()
        store.put("k1", "v1", source="s1", tags=("t1",), confidence=0.9)
        store.put("k2", {"nested": True}, source="s2")

        data = store.to_dict()
        restored = KnowledgeStore.from_dict(data)
        assert len(restored) == 2
        assert restored.get("k1").confidence == 0.9

    def test_to_json_and_from_json(self) -> None:
        store = KnowledgeStore()
        store.put("k1", "v1", source="test")
        json_str = store.to_json()

        restored = KnowledgeStore.from_json(json_str)
        assert restored.get("k1").value == "v1"

    def test_thread_safety(self) -> None:
        """Basic thread safety check."""
        import threading

        store = KnowledgeStore()
        errors: list[str] = []

        def writer(prefix: str) -> None:
            for i in range(50):
                try:
                    store.put(f"{prefix}_{i}", f"value_{i}")
                except Exception as e:
                    errors.append(str(e))

        threads = [threading.Thread(target=writer, args=(f"t{t}",)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(store) == 200  # 4 threads * 50 entries

    def test_confidence_field(self) -> None:
        store = KnowledgeStore()
        entry = store.put("k1", "v1", confidence=0.75)
        assert entry.confidence == 0.75
