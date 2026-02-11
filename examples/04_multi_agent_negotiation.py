#!/usr/bin/env python3
"""Example 04: Multi-agent shared environment.

Demonstrates:
- SharedEnvironment with global + local state
- Multiple agents observing and acting in the same world
- Agent state isolation (local) vs shared state (global)
- Pairwise distance computation

Run:
    PYTHONPATH=src python examples/04_multi_agent_negotiation.py
"""

from __future__ import annotations

from synthetic_teleology.domain.values import ActionSpec
from synthetic_teleology.environments.shared import SharedEnvironment


def main() -> None:
    print("=== Multi-Agent Shared Environment ===\n")

    # -- Setup ----------------------------------------------------------------
    env = SharedEnvironment(
        global_dimensions=3,
        local_dimensions=2,
        initial_global_state=(0.0, 0.0, 0.0),
        observation_noise_std=0.0,
        action_effect_global=0.5,
    )

    # Register three agents
    env.register_agent("agent-A", initial_local_state=(0.0, 0.0))
    env.register_agent("agent-B", initial_local_state=(5.0, 5.0))
    env.register_agent("agent-C", initial_local_state=(10.0, 0.0))

    print(f"Registered agents: {sorted(env.registered_agents)}")
    print(f"Global dims: {env.global_dimensions}, Local dims: {env.local_dimensions}")
    print(f"Total observation dims: {env.total_observation_dimensions}")
    print()

    # -- Initial observations --------------------------------------------------
    print("Initial observations:")
    for agent_id in sorted(env.registered_agents):
        obs = env.observe(agent_id=agent_id)
        print(f"  {agent_id}: {tuple(round(v, 2) for v in obs.values)}")
    print()

    # -- Agent A modifies global state -----------------------------------------
    print("Agent-A pushes global state by (1.0, 0.0, 0.0)...")
    action_a = ActionSpec(
        name="push_global",
        parameters={"delta_global": (1.0, 0.0, 0.0)},
    )
    env.step(action_a, agent_id="agent-A")

    # -- Agent B modifies its local state -------------------------------------
    print("Agent-B pushes its local state by (-1.0, 1.0)...")
    action_b = ActionSpec(
        name="push_local",
        parameters={"delta_local": (-1.0, 1.0)},
    )
    env.step(action_b, agent_id="agent-B")
    print()

    # -- Observations after actions -------------------------------------------
    print("Observations after actions:")
    for agent_id in sorted(env.registered_agents):
        obs = env.observe(agent_id=agent_id)
        print(f"  {agent_id}: {tuple(round(v, 2) for v in obs.values)}")
    print()

    # -- Global state visible to all ------------------------------------------
    print(f"Global state: {tuple(round(v, 2) for v in env.global_state)}")
    print()

    # -- Pairwise distances ---------------------------------------------------
    print("Pairwise local-state distances:")
    distances = env.get_agent_distances()
    for (a, b), dist in sorted(distances.items()):
        print(f"  {a} <-> {b}: {dist:.4f}")
    print()

    # -- Perturbation ---------------------------------------------------------
    print("Injecting global shift perturbation (0, 0, 5.0)...")
    env.inject_perturbation({"global_shift": (0.0, 0.0, 5.0)})
    print(f"Global state after perturbation: {tuple(round(v, 2) for v in env.global_state)}")
    print()

    # -- Reset ----------------------------------------------------------------
    print("Resetting environment...")
    env.reset()
    print(f"Global state after reset: {tuple(round(v, 2) for v in env.global_state)}")
    print(f"Step count: {env.step_count}")

    print("\nDone.")


if __name__ == "__main__":
    main()
