"""Tests for domain aggregates (GoalTree, ConstraintSet, AgentIdentity)."""

from __future__ import annotations

import pytest

from synthetic_teleology.domain.aggregates import (
    AgentIdentity,
    ConstraintSet,
    GoalTree,
)
from synthetic_teleology.domain.entities import Constraint, Goal
from synthetic_teleology.domain.enums import (
    ConstraintType,
    Direction,
)
from synthetic_teleology.domain.values import (
    ConstraintSpec,
    GoalRevision,
    ObjectiveVector,
)

# ===================================================================== #
#  GoalTree                                                               #
# ===================================================================== #


class TestGoalTree:
    """Tests for GoalTree: add, remove, lineage, flatten, coherence, propagation."""

    @pytest.fixture
    def obj(self) -> ObjectiveVector:
        return ObjectiveVector(
            values=(5.0, 5.0),
            directions=(Direction.APPROACH, Direction.APPROACH),
        )

    @pytest.fixture
    def tree(self, obj: ObjectiveVector) -> GoalTree:
        root = Goal(goal_id="root", name="root", objective=obj)
        return GoalTree(root)

    def test_root(self, tree: GoalTree) -> None:
        assert tree.root.goal_id == "root"
        assert tree.size == 1

    def test_add_subgoal(self, tree: GoalTree, obj: ObjectiveVector) -> None:
        child = Goal(goal_id="child1", name="child1", objective=obj)
        tree.add_subgoal("root", child)
        assert tree.size == 2
        # Goal is frozen — retrieve from tree to see parent_id
        assert tree.get_goal("child1").parent_id == "root"
        children = tree.get_children("root")
        assert len(children) == 1
        assert children[0].goal_id == "child1"

    def test_add_subgoal_invalid_parent(self, tree: GoalTree, obj: ObjectiveVector) -> None:
        child = Goal(goal_id="child1", name="child1", objective=obj)
        with pytest.raises(KeyError, match="not found"):
            tree.add_subgoal("nonexistent", child)

    def test_remove_subgoal(self, tree: GoalTree, obj: ObjectiveVector) -> None:
        child = Goal(goal_id="child1", name="child1", objective=obj)
        tree.add_subgoal("root", child)
        assert tree.size == 2

        tree.remove_subgoal("child1")
        assert tree.size == 1

    def test_remove_root_raises(self, tree: GoalTree) -> None:
        with pytest.raises(ValueError, match="Cannot remove root"):
            tree.remove_subgoal("root")

    def test_remove_nonexistent_raises(self, tree: GoalTree) -> None:
        with pytest.raises(KeyError, match="not found"):
            tree.remove_subgoal("nonexistent")

    def test_remove_subtree(self, tree: GoalTree, obj: ObjectiveVector) -> None:
        child = Goal(goal_id="child1", name="child1", objective=obj)
        grandchild = Goal(goal_id="grandchild1", name="grandchild1", objective=obj)
        tree.add_subgoal("root", child)
        tree.add_subgoal("child1", grandchild)
        assert tree.size == 3

        tree.remove_subgoal("child1")
        assert tree.size == 1

    def test_lineage(self, tree: GoalTree, obj: ObjectiveVector) -> None:
        child = Goal(goal_id="child1", name="child1", objective=obj)
        grandchild = Goal(goal_id="gc1", name="gc1", objective=obj)
        tree.add_subgoal("root", child)
        tree.add_subgoal("child1", grandchild)

        lineage = tree.get_lineage("gc1")
        assert [g.goal_id for g in lineage] == ["root", "child1", "gc1"]

    def test_flatten(self, tree: GoalTree, obj: ObjectiveVector) -> None:
        child1 = Goal(goal_id="c1", name="c1", objective=obj)
        child2 = Goal(goal_id="c2", name="c2", objective=obj)
        tree.add_subgoal("root", child1)
        tree.add_subgoal("root", child2)

        flat = tree.flatten()
        ids = [g.goal_id for g in flat]
        assert "root" in ids
        assert "c1" in ids
        assert "c2" in ids
        assert len(flat) == 3

    def test_get_active_leaves(self, tree: GoalTree, obj: ObjectiveVector) -> None:
        child1 = Goal(goal_id="c1", name="c1", objective=obj)
        child2 = Goal(goal_id="c2", name="c2", objective=obj)
        tree.add_subgoal("root", child1)
        tree.add_subgoal("root", child2)

        leaves = tree.get_active_leaves()
        assert len(leaves) == 2  # c1, c2 are leaves

    def test_validate_coherence_clean(self, tree: GoalTree) -> None:
        issues = tree.validate_coherence()
        assert issues == []

    def test_validate_coherence_active_child_under_inactive(
        self, tree: GoalTree, obj: ObjectiveVector
    ) -> None:
        child = Goal(goal_id="c1", name="c1", objective=obj)
        tree.add_subgoal("root", child)
        # Goal is frozen — replace root with abandoned version
        abandoned_root = tree.root.abandon()
        tree._root = abandoned_root
        tree._all_goals["root"] = abandoned_root

        issues = tree.validate_coherence()
        assert len(issues) >= 1
        assert any("Active child" in issue for issue in issues)

    def test_propagate_revision(self, tree: GoalTree, obj: ObjectiveVector) -> None:
        child = Goal(goal_id="c1", name="c1", objective=obj)
        tree.add_subgoal("root", child)

        revisions = tree.propagate_revision(tree.root)
        assert len(revisions) == 1
        assert revisions[0].previous_goal_id == "c1"

    def test_get_goal(self, tree: GoalTree, obj: ObjectiveVector) -> None:
        child = Goal(goal_id="c1", name="c1", objective=obj)
        tree.add_subgoal("root", child)
        retrieved = tree.get_goal("c1")
        assert retrieved.goal_id == "c1"

    def test_get_goal_not_found(self, tree: GoalTree) -> None:
        with pytest.raises(KeyError):
            tree.get_goal("nonexistent")


# ===================================================================== #
#  ConstraintSet                                                          #
# ===================================================================== #


class TestConstraintSet:
    """Tests for ConstraintSet: add, remove, filter, iteration."""

    def test_empty(self) -> None:
        cs = ConstraintSet()
        assert len(cs) == 0
        assert not cs

    def test_add_and_len(self) -> None:
        cs = ConstraintSet()
        spec = ConstraintSpec(name="s", constraint_type=ConstraintType.HARD)
        c = Constraint(name="a", constraint_type=ConstraintType.HARD, spec=spec)
        cs.add(c)
        assert len(cs) == 1
        assert cs

    def test_remove(self) -> None:
        cs = ConstraintSet()
        spec = ConstraintSpec(name="s", constraint_type=ConstraintType.HARD)
        c = Constraint(name="a", constraint_type=ConstraintType.HARD, spec=spec)
        cs.add(c)
        cs.remove(c.constraint_id)
        assert len(cs) == 0

    def test_get_active(self) -> None:
        cs = ConstraintSet()
        spec = ConstraintSpec(name="s", constraint_type=ConstraintType.HARD)
        c1 = Constraint(name="a", constraint_type=ConstraintType.HARD, spec=spec)
        c2 = Constraint(name="b", constraint_type=ConstraintType.HARD, spec=spec)
        cs.add(c1)
        cs.add(c2)
        c2.deactivate()

        active = cs.get_active()
        assert len(active) == 1
        assert active[0].name == "a"

    def test_filter_by_type(self) -> None:
        cs = ConstraintSet()
        hard_spec = ConstraintSpec(name="h", constraint_type=ConstraintType.HARD)
        soft_spec = ConstraintSpec(name="s", constraint_type=ConstraintType.SOFT)
        c1 = Constraint(name="hard", constraint_type=ConstraintType.HARD, spec=hard_spec)
        c2 = Constraint(name="soft", constraint_type=ConstraintType.SOFT, spec=soft_spec)
        cs.add(c1)
        cs.add(c2)

        hard_only = cs.filter_by_type(ConstraintType.HARD)
        assert len(hard_only) == 1
        assert hard_only[0].name == "hard"

    def test_get_hard_constraints(self) -> None:
        cs = ConstraintSet()
        hard_spec = ConstraintSpec(name="h", constraint_type=ConstraintType.HARD)
        c = Constraint(name="hard", constraint_type=ConstraintType.HARD, spec=hard_spec)
        cs.add(c)
        assert len(cs.get_hard_constraints()) == 1

    def test_priority_ordering(self) -> None:
        cs = ConstraintSet()
        spec = ConstraintSpec(name="s", constraint_type=ConstraintType.HARD)
        c_low = Constraint(name="low", constraint_type=ConstraintType.HARD, spec=spec, weight=1.0)
        c_high = Constraint(name="high", constraint_type=ConstraintType.HARD, spec=spec, weight=10.0)
        cs.add(c_low)
        cs.add(c_high)

        items = list(cs)
        assert items[0].weight >= items[1].weight

    def test_iteration(self) -> None:
        cs = ConstraintSet()
        spec = ConstraintSpec(name="s", constraint_type=ConstraintType.HARD)
        for i in range(3):
            cs.add(Constraint(name=f"c{i}", constraint_type=ConstraintType.HARD, spec=spec))
        count = sum(1 for _ in cs)
        assert count == 3


# ===================================================================== #
#  AgentIdentity                                                          #
# ===================================================================== #


class TestAgentIdentity:
    """Tests for AgentIdentity aggregate."""

    def test_creation(self) -> None:
        goal = Goal(name="g1")
        identity = AgentIdentity("agent-1", goal)
        assert identity.agent_id == "agent-1"
        assert identity.current_goal is goal
        assert identity.revision_count == 0

    def test_record_revision(self) -> None:
        goal1 = Goal(name="g1")
        identity = AgentIdentity("agent-1", goal1)

        goal2 = Goal(name="g2", version=2)
        revision = GoalRevision(
            previous_goal_id=goal1.goal_id,
            new_goal_id=goal2.goal_id,
            reason="test",
        )
        identity.record_revision(goal2, revision)

        assert identity.current_goal is goal2
        assert identity.revision_count == 1
        assert len(identity.goal_history) == 2
        assert len(identity.revision_log) == 1

    def test_get_goal_at_version(self) -> None:
        goal1 = Goal(name="g1", version=1)
        identity = AgentIdentity("agent-1", goal1)

        goal2 = Goal(name="g2", version=2)
        revision = GoalRevision(
            previous_goal_id=goal1.goal_id,
            new_goal_id=goal2.goal_id,
        )
        identity.record_revision(goal2, revision)

        assert identity.get_goal_at_version(1) is goal1
        assert identity.get_goal_at_version(2) is goal2
        assert identity.get_goal_at_version(99) is None
