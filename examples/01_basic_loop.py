#!/usr/bin/env python3
"""Example 01: Basic end-to-end teleological loop.

Demonstrates:
- Creating a NumericEnvironment
- Wiring evaluator, updater, planner, and constraint pipeline
- Running the SyncAgenticLoop to completion
- Inspecting the RunResult

Run:
    PYTHONPATH=src python examples/01_basic_loop.py
"""

from __future__ import annotations

import time

from synthetic_teleology.domain.enums import Direction
from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.values import (
    ActionSpec,
    ObjectiveVector,
    StateSnapshot,
    PolicySpec,
)
from synthetic_teleology.environments.numeric import NumericEnvironment
from synthetic_teleology.services.evaluation import NumericEvaluator
from synthetic_teleology.services.goal_revision import ThresholdUpdater
from synthetic_teleology.services.planning import GreedyPlanner
from synthetic_teleology.services.constraint_engine import ConstraintPipeline
from synthetic_teleology.services.loop import SyncAgenticLoop, StopReason


def main() -> None:
    # -- Environment ----------------------------------------------------------
    env = NumericEnvironment(
        dimensions=2,
        initial_state=(0.0, 0.0),
        noise_std=0.0,
    )

    # -- Goal -----------------------------------------------------------------
    objective = ObjectiveVector(
        values=(5.0, 5.0),
        directions=(Direction.APPROACH, Direction.APPROACH),
    )
    goal = Goal(name="reach-target", objective=objective)

    # -- Services -------------------------------------------------------------
    evaluator = NumericEvaluator(max_distance=10.0)
    updater = ThresholdUpdater(threshold=0.8, learning_rate=0.1)

    # Build action space: 2*2 + 1 = 5 actions
    actions: list[ActionSpec] = []
    for d in range(2):
        eff_pos = tuple(1.0 if i == d else 0.0 for i in range(2))
        actions.append(ActionSpec(name=f"pos_{d}", parameters={"effect": eff_pos, "delta": eff_pos}))
        eff_neg = tuple(-1.0 if i == d else 0.0 for i in range(2))
        actions.append(ActionSpec(name=f"neg_{d}", parameters={"effect": eff_neg, "delta": eff_neg}))
    actions.append(ActionSpec(name="noop", parameters={"effect": (0.0, 0.0), "delta": (0.0, 0.0)}))

    planner = GreedyPlanner(action_space=actions)
    pipeline = ConstraintPipeline(checkers=[])

    # -- Callback functions for the loop --------------------------------------
    def perceive_fn() -> StateSnapshot:
        return env.observe()

    def act_fn(policy: PolicySpec, state: StateSnapshot) -> ActionSpec | None:
        if policy.size > 0:
            return policy.actions[0]
        return None

    def transition_fn(action: ActionSpec | None) -> None:
        if action is not None:
            env.step(action)

    # -- Loop -----------------------------------------------------------------
    loop = SyncAgenticLoop(
        evaluator=evaluator,
        goal_updater=updater,
        planner=planner,
        constraint_pipeline=pipeline,
        max_steps=20,
        goal_achieved_threshold=0.9,
        stop_on_empty_policy=True,
        perceive_fn=perceive_fn,
        act_fn=act_fn,
        transition_fn=transition_fn,
    )

    print("=== Basic Teleological Loop ===")
    print(f"Goal: approach {objective.values}")
    print(f"Start state: {env.observe().values}")
    print()

    result = loop.run(goal)

    print(f"Stopped: {result.stopped_reason.value}")
    print(f"Steps: {result.steps_completed}")
    print(f"Elapsed: {result.elapsed_seconds:.3f}s")
    if result.final_state is not None:
        print(f"Final state: {result.final_state.values}")
    print(f"Final eval score: {result.metadata.get('last_eval_score', 'N/A'):.4f}")
    print(f"Events emitted: {len(result.events)}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
