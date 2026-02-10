#!/usr/bin/env python3
"""Example 03: ReflectiveEvaluator confidence adjustment.

Demonstrates:
- Wrapping a NumericEvaluator in a ReflectiveEvaluator
- Observing how oscillating states reduce evaluation confidence
- Comparing raw vs reflective evaluation signals

Run:
    PYTHONPATH=src python examples/03_reflective_critic.py
"""

from __future__ import annotations

import time

from synthetic_teleology.domain.enums import Direction
from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.values import ObjectiveVector, StateSnapshot
from synthetic_teleology.services.evaluation import NumericEvaluator, ReflectiveEvaluator


def main() -> None:
    print("=== Reflective Critic ===\n")

    # -- Setup ----------------------------------------------------------------
    inner = NumericEvaluator(max_distance=10.0)
    reflective = ReflectiveEvaluator(inner, history_size=10, drift_threshold=0.1)

    objective = ObjectiveVector(
        values=(5.0,),
        directions=(Direction.APPROACH,),
    )
    goal = Goal(name="target", objective=objective)

    # -- Simulate oscillating observations ------------------------------------
    # The state oscillates between 0.0 and 10.0 -- should trigger drift detection
    oscillating_values = [0.0, 10.0, 0.0, 10.0, 0.0, 10.0, 0.0, 10.0]

    print("Oscillating state sequence:")
    print(f"  Goal target: {objective.values}")
    print(f"  States: {oscillating_values}\n")

    print(f"{'Step':>4s}  {'State':>6s}  {'Raw Score':>10s}  {'Raw Conf':>10s}  {'Refl Score':>10s}  {'Refl Conf':>10s}")
    print("-" * 65)

    for i, val in enumerate(oscillating_values):
        state = StateSnapshot(timestamp=time.time(), values=(val,))

        raw_signal = inner.evaluate(goal, state)
        refl_signal = reflective.evaluate(goal, state)

        print(
            f"{i:4d}  "
            f"{val:6.1f}  "
            f"{raw_signal.score:10.4f}  "
            f"{raw_signal.confidence:10.4f}  "
            f"{refl_signal.score:10.4f}  "
            f"{refl_signal.confidence:10.4f}"
        )

    print()
    print(f"Reflective history length: {len(reflective.history)}")
    print()

    # -- Simulate converging observations --------------------------------------
    print("Converging state sequence:")
    converging_values = [0.0, 2.0, 3.0, 4.0, 4.5, 4.8, 4.9, 5.0]
    print(f"  States: {converging_values}\n")

    reflective2 = ReflectiveEvaluator(inner, history_size=10, drift_threshold=0.1)

    print(f"{'Step':>4s}  {'State':>6s}  {'Score':>10s}  {'Confidence':>10s}")
    print("-" * 40)

    for i, val in enumerate(converging_values):
        state = StateSnapshot(timestamp=time.time(), values=(val,))
        signal = reflective2.evaluate(goal, state)
        print(f"{i:4d}  {val:6.1f}  {signal.score:10.4f}  {signal.confidence:10.4f}")

    print()
    print("Observation: Converging states maintain or increase confidence,")
    print("while oscillating states reduce confidence over time.")
    print("\nDone.")


if __name__ == "__main__":
    main()
