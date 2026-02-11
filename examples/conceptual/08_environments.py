#!/usr/bin/env python3
"""Example 08: Environment types — ResourceEnvironment and ResearchEnvironment.

Demonstrates:
- ResourceEnvironment: competing resource allocation with regeneration + scarcity
- ResearchEnvironment: knowledge accumulation with diminishing returns + synthesis
- Running a teleological agent in each environment
- Perturbation injection via inject_perturbation()

Run:
    PYTHONPATH=src python examples/conceptual/08_environments.py
"""

from __future__ import annotations

from synthetic_teleology.domain.enums import Direction
from synthetic_teleology.domain.values import ActionSpec
from synthetic_teleology.environments.research import ResearchEnvironment
from synthetic_teleology.environments.resource import ResourceEnvironment
from synthetic_teleology.graph import GraphBuilder, collect_stream_events
from synthetic_teleology.services.evaluation import NumericEvaluator
from synthetic_teleology.services.planning import GreedyPlanner


def _run_resource_scenario() -> None:
    """Scenario A: Resource allocation with scarcity and regeneration."""
    print("--- Scenario A: Resource Environment ---")
    env = ResourceEnvironment(
        num_resources=3,
        total_resources=100.0,
        initial_levels=(80.0, 60.0, 90.0),
        regeneration_rate=0.1,
        consumption_efficiency=0.8,
    )

    print("Resources: 3 types, capacity=100 each")
    print(f"Initial levels: {tuple(float(v) for v in env.levels)}")
    print("Regeneration rate: 10%/step")
    print()

    # Goal: maintain resource levels around (50, 50, 50)
    goal_values = (50.0, 50.0, 50.0)

    # Action space: allocate/release per resource
    actions: list[ActionSpec] = []
    for r in range(3):
        # Consume 10 units of resource r
        actions.append(ActionSpec(
            name=f"consume_r{r}",
            parameters={
                "allocations": {r: 10.0},
                "effect": tuple(-10.0 if i == r else 0.0 for i in range(3)),
            },
        ))
        # Release 10 units of resource r
        actions.append(ActionSpec(
            name=f"release_r{r}",
            parameters={
                "allocations": {r: -10.0},
                "effect": tuple(10.0 if i == r else 0.0 for i in range(3)),
            },
        ))
    actions.append(ActionSpec(
        name="noop",
        parameters={"allocations": {}, "effect": (0.0, 0.0, 0.0)},
    ))

    app, initial_state = (
        GraphBuilder("resource-agent")
        .with_objective(
            goal_values,
            directions=(Direction.APPROACH, Direction.APPROACH, Direction.APPROACH),
        )
        .with_evaluator(NumericEvaluator(max_distance=150.0))
        .with_planner(GreedyPlanner(action_space=actions))
        .with_max_steps(12)
        .with_goal_achieved_threshold(0.9)
        .with_environment(
            perceive_fn=lambda: env.observe(),
            act_fn=lambda p, s: p.actions[0] if p.size > 0 else None,
            transition_fn=lambda a: env.step(a) if a else None,
        )
        .build()
    )

    # Inject a perturbation at step 6 (deplete resource 0)
    step_counter = [0]
    original_perceive = initial_state["perceive_fn"]

    def perceive_with_perturbation():
        step_counter[0] += 1
        if step_counter[0] == 6:
            env.inject_perturbation({
                "type": "resource_shock",
                "deplete": {0: 50.0},
                "magnitude": 0.5,
            })
            print("  [!] Perturbation injected: depleted 50 units of resource 0")
        return original_perceive()

    initial_state["perceive_fn"] = perceive_with_perturbation

    stream = app.stream(initial_state, stream_mode="updates")
    events = collect_stream_events(stream)

    # Print resource trajectory
    print("Resource levels over time:")
    step = 0
    for ev in events:
        if ev.get("node") == "perceive" and ev.get("state_snapshot"):
            step += 1
            vals = ev["state_snapshot"].values
            scarcity_meta = ev["state_snapshot"].metadata.get("scarcity", ())
            vals_str = ", ".join(f"{v:.1f}" for v in vals)
            print(f"  Step {step:2d}: [{vals_str}]", end="")
            if scarcity_meta:
                scar_str = ", ".join(f"{s:.2f}" for s in scarcity_meta)
                print(f"  scarcity=[{scar_str}]", end="")
            print()

    final_levels = tuple(float(v) for v in env.levels)
    print(f"Final levels: {final_levels}")
    print(f"Total consumed: {tuple(float(v) for v in env.total_consumed)}")
    print()


def _run_research_scenario() -> None:
    """Scenario B: Knowledge synthesis with diminishing returns."""
    print("--- Scenario B: Research Environment ---")
    env = ResearchEnvironment(
        topics=("ml", "physics", "biology"),
        max_knowledge=10.0,
        knowledge_decay=0.02,
        novelty_decay=0.1,
        synthesis_bonus=1.5,
        noise_std=0.0,  # deterministic for demonstration
    )

    print(f"Topics: {env.topics}")
    print("Max knowledge per topic: 10.0")
    print()

    # State: [knowledge_ml, knowledge_physics, knowledge_biology, synthesis, novelty]
    # Goal: build balanced knowledge + synthesis
    goal_values = (5.0, 5.0, 5.0, 2.0, 1.0)

    # Action space: research each topic, synthesize, or explore
    # ResearchEnvironment expects action.name in {"research", "synthesize", "explore"}
    # with "topic" param for research actions
    actions: list[ActionSpec] = []
    for t_idx in range(env.num_topics):
        # Estimate effect: research increases knowledge in that topic
        effect = [0.0] * 5
        effect[t_idx] = 0.5  # approximate gain
        actions.append(ActionSpec(
            name="research",
            parameters={
                "topic": t_idx,
                "effort": 0.5,
                "effect": tuple(effect),
            },
        ))
    # Synthesize: increases synthesis quality
    actions.append(ActionSpec(
        name="synthesize",
        parameters={
            "effort": 0.5,
            "effect": (0.0, 0.0, 0.0, 0.3, 0.0),
        },
    ))
    # Explore: increases novelty
    actions.append(ActionSpec(
        name="explore",
        parameters={
            "effort": 0.5,
            "effect": (0.0, 0.0, 0.0, 0.0, 0.4),
        },
    ))

    app, initial_state = (
        GraphBuilder("research-agent")
        .with_objective(
            goal_values,
            directions=(Direction.APPROACH,) * 5,
        )
        .with_evaluator(NumericEvaluator(max_distance=15.0))
        .with_planner(GreedyPlanner(action_space=actions))
        .with_max_steps(15)
        .with_goal_achieved_threshold(0.8)
        .with_environment(
            perceive_fn=lambda: env.observe(),
            act_fn=lambda p, s: p.actions[0] if p.size > 0 else None,
            transition_fn=lambda a: env.step(a) if a else None,
        )
        .build()
    )

    stream = app.stream(initial_state, stream_mode="updates")
    events = collect_stream_events(stream)

    # Print knowledge trajectory
    print("Knowledge trajectory:")
    step = 0
    for ev in events:
        if ev.get("node") == "act" and ev.get("executed_action"):
            step += 1
            action = ev["executed_action"]
            action_label = action.name
            topic_idx = action.parameters.get("topic")
            if topic_idx is not None:
                action_label = f"research(topic={topic_idx})"
            state = env.observe()
            knowledge = state.values[:3]
            synthesis = state.values[3]
            novelty = state.values[4]
            k_str = ", ".join(f"{k:.2f}" for k in knowledge)
            print(f"  Step {step:2d}: {action_label:<22} "
                  f"k=[{k_str}] synth={synthesis:.2f} nov={novelty:.2f}")

    final = env.observe()
    print(f"\nFinal state: {tuple(round(v, 2) for v in final.values)}")
    print(f"Knowledge diversity: {env.knowledge_diversity:.3f}")
    print()


def main() -> None:
    print("=== Environment Types ===")
    print()
    _run_resource_scenario()
    _run_research_scenario()
    print("Key insight: ResourceEnvironment models scarcity + regeneration dynamics,")
    print("while ResearchEnvironment models diminishing returns + knowledge synthesis.")
    print("Both support perturbation injection for testing agent adaptivity.")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
