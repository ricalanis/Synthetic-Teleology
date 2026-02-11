#!/usr/bin/env python3
"""Example 06: Planning strategies — Greedy, Stochastic, and Hierarchical planners.

Demonstrates:
- GreedyPlanner: deterministic best-action selection
- StochasticPlanner: softmax temperature-based sampling with exploration
- HierarchicalPlanner: decomposes multi-dimensional goals into sub-plans
- Effect of temperature on exploration (high temp = more random)
- Comparing convergence speed and action diversity

Run:
    PYTHONPATH=src python examples/conceptual/06_planning_strategies.py
"""

from __future__ import annotations

import numpy as np

from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.enums import Direction
from synthetic_teleology.domain.values import ActionSpec, ObjectiveVector, StateSnapshot
from synthetic_teleology.environments.numeric import NumericEnvironment
from synthetic_teleology.graph import GraphBuilder, collect_stream_events
from synthetic_teleology.services.evaluation import NumericEvaluator
from synthetic_teleology.services.planning import (
    GreedyPlanner,
    HierarchicalPlanner,
    StochasticPlanner,
)


def _make_actions(step_size: float = 1.0) -> list[ActionSpec]:
    """Build a 2D action space with movements in 4 directions + noop."""
    actions: list[ActionSpec] = []
    for d in range(2):
        for sign, label in [(step_size, "pos"), (-step_size, "neg")]:
            effect = tuple(sign if i == d else 0.0 for i in range(2))
            actions.append(ActionSpec(
                name=f"move_d{d}_{label}",
                parameters={"effect": effect, "delta": effect},
            ))
    actions.append(ActionSpec(
        name="noop",
        parameters={"effect": (0.0, 0.0), "delta": (0.0, 0.0)},
    ))
    return actions


def _stochastic_act_fn(policy, _state):
    """Sample an action from a stochastic policy using probabilities."""
    if policy.size == 0:
        return None
    if policy.is_stochastic and policy.probabilities:
        rng = np.random.default_rng(None)
        idx = rng.choice(len(policy.actions), p=policy.probabilities)
        return policy.actions[idx]
    return policy.actions[0]


def _run_with_planner(label: str, planner, act_fn=None) -> dict:
    """Run a teleological agent with the given planner."""
    env = NumericEnvironment(dimensions=2, initial_state=(0.0, 0.0))

    def default_act(p, _s):
        return p.actions[0] if p.size > 0 else None

    chosen_act = act_fn or default_act

    app, initial_state = (
        GraphBuilder(f"plan-{label}")
        .with_objective((5.0, 5.0))
        .with_evaluator(NumericEvaluator(max_distance=10.0))
        .with_planner(planner)
        .with_max_steps(20)
        .with_goal_achieved_threshold(0.85)
        .with_environment(
            perceive_fn=lambda: env.observe(),
            act_fn=chosen_act,
            transition_fn=lambda a: env.step(a) if a else None,
        )
        .build()
    )

    stream = app.stream(initial_state, stream_mode="updates")
    events = collect_stream_events(stream)

    action_names = []
    scores = []
    for ev in events:
        if ev.get("node") == "act" and ev.get("executed_action"):
            action_names.append(ev["executed_action"].name)
        if ev.get("node") == "evaluate" and ev.get("eval_signal"):
            scores.append(ev["eval_signal"].score)

    final_step = 0
    stop_reason = "unknown"
    for ev in events:
        if ev.get("node") == "reflect":
            final_step = ev.get("step", final_step)
            stop_reason = ev.get("stop_reason", stop_reason) or "continuing"

    unique_actions = len(set(action_names)) if action_names else 0

    return {
        "label": label,
        "action_names": action_names,
        "scores": scores,
        "final_step": final_step,
        "stop_reason": stop_reason,
        "unique_actions": unique_actions,
        "total_actions": len(action_names),
        "env": env,
    }


def _demo_hierarchical_planner() -> None:
    """Demo the HierarchicalPlanner's decomposition logic separately."""
    print("--- HierarchicalPlanner Decomposition ---")

    # 1D action spaces for sub-planners
    actions_1d = [
        ActionSpec(name="move_pos", parameters={"effect": (1.0,)}),
        ActionSpec(name="move_neg", parameters={"effect": (-1.0,)}),
        ActionSpec(name="noop", parameters={"effect": (0.0,)}),
    ]

    hierarchical = HierarchicalPlanner(
        sub_planner=GreedyPlanner(action_space=actions_1d),
        sub_goal_groups=[[0], [1]],
    )

    # Create a 2D goal and state, then plan
    goal = Goal(
        name="demo-goal",
        objective=ObjectiveVector(
            values=(5.0, 3.0),
            directions=(Direction.APPROACH, Direction.APPROACH),
        ),
    )
    state = StateSnapshot(timestamp=0.0, values=(1.0, 1.0))

    policy = hierarchical.plan(goal, state)
    print(f"Goal: {goal.objective.values}")
    print(f"State: {state.values}")
    print(f"Decomposed into {len(policy.metadata.get('sub_plans', []))} sub-plans")
    for sp in policy.metadata.get("sub_plans", []):
        print(f"  Group {sp['group']}: dims={sp['dimensions']}, "
              f"actions={sp['num_actions']}")
    print(f"Merged policy: {policy.size} actions")
    for a in policy.actions:
        print(f"  {a.name} (effect={a.parameters.get('effect')})")
    print()


def main() -> None:
    print("=== Planning Strategies Comparison ===")
    print("Goal: approach (5.0, 5.0) from (0.0, 0.0)")
    print()

    actions = _make_actions(step_size=1.0)

    configs = [
        ("Greedy", GreedyPlanner(action_space=actions), None),
        (
            "Stochastic(T=0.5)",
            StochasticPlanner(
                action_space=actions, temperature=0.5, seed=42,
            ),
            _stochastic_act_fn,
        ),
        (
            "Stochastic(T=3.0)",
            StochasticPlanner(
                action_space=actions, temperature=3.0, seed=42,
            ),
            _stochastic_act_fn,
        ),
    ]

    results = []
    for label, planner, act_fn in configs:
        result = _run_with_planner(label, planner, act_fn)
        results.append(result)

    # -- Summary table --
    header = (
        f"{'Planner':<22} {'Steps':>5} {'Actions':>7} "
        f"{'Unique':>6} {'Stop':>12} Final State"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        vals = r["env"].observe().values
        state_str = f"({vals[0]:.1f}, {vals[1]:.1f})"
        print(
            f"{r['label']:<22} {r['final_step']:>5} "
            f"{r['total_actions']:>7} {r['unique_actions']:>6} "
            f"{r['stop_reason']:>12} {state_str}"
        )
    print()

    # -- Detail per strategy --
    for r in results:
        print(f"--- {r['label']} ---")
        if r["action_names"]:
            seq = " -> ".join(r["action_names"][:8])
            if len(r["action_names"]) > 8:
                seq += " ..."
            print(f"  Action sequence: {seq}")
        if r["scores"]:
            trajectory = " -> ".join(f"{s:.3f}" for s in r["scores"][:8])
            if len(r["scores"]) > 8:
                trajectory += " ..."
            print(f"  Score trajectory: {trajectory}")
        print()

    # -- HierarchicalPlanner decomposition demo --
    _demo_hierarchical_planner()

    print("Key insight: GreedyPlanner converges fastest but explores least.")
    print("StochasticPlanner(T=0.5) is near-greedy; T=3.0 explores broadly.")
    print("HierarchicalPlanner decomposes the 2D goal into two 1D sub-problems,")
    print("planning each dimension independently then merging policies.")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
