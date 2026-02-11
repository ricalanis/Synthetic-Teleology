"""Tests for WorkingMemory utility."""

from __future__ import annotations

from synthetic_teleology.domain.values import ActionSpec, StateSnapshot
from synthetic_teleology.graph.working_memory import WorkingMemory


class TestWorkingMemory:
    def test_initial_perceive_returns_context(self):
        mem = WorkingMemory("Initial context here")
        snap = mem.perceive()
        assert isinstance(snap, StateSnapshot)
        assert "Initial context here" in snap.observation

    def test_perceive_with_empty_context(self):
        mem = WorkingMemory()
        snap = mem.perceive()
        assert isinstance(snap, StateSnapshot)
        assert snap.observation == "No observations yet."

    def test_perceive_after_record_includes_action(self):
        mem = WorkingMemory("ctx")
        action = ActionSpec(name="search_web", parameters={"q": "test"})
        mem.record(action)
        snap = mem.perceive()
        assert "search_web" in snap.observation
        assert "Working memory" in snap.observation

    def test_record_with_constraints_context(self):
        mem = WorkingMemory()
        action = ActionSpec(name="act1", parameters={})
        ctx = {"constraints_ok": True, "constraint_violations": []}
        mem.record(action, constraints_context=ctx)
        assert len(mem.entries) == 1
        assert mem.entries[0]["constraints_context"] == ctx

    def test_fifo_eviction(self):
        mem = WorkingMemory(max_entries=3)
        for i in range(5):
            mem.record(ActionSpec(name=f"act_{i}", parameters={}))
        assert len(mem.entries) == 3
        names = [e["action"] for e in mem.entries]
        assert names == ["act_2", "act_3", "act_4"]

    def test_perceive_returns_statesnapshot(self):
        mem = WorkingMemory("hello")
        snap = mem.perceive()
        assert isinstance(snap, StateSnapshot)
        assert snap.timestamp > 0

    def test_clear_resets_state(self):
        mem = WorkingMemory("ctx")
        mem.record(ActionSpec(name="x", parameters={}))
        assert mem.step == 1
        assert len(mem.entries) == 1
        mem.clear()
        assert mem.step == 0
        assert len(mem.entries) == 0

    def test_step_increments(self):
        mem = WorkingMemory()
        assert mem.step == 0
        mem.record(ActionSpec(name="a", parameters={}))
        assert mem.step == 1
        mem.record(ActionSpec(name="b", parameters={}))
        assert mem.step == 2
