#!/usr/bin/env python3
"""Example 03: Two agents with subgraphs + consensus negotiation.

Demonstrates:
- Building a multi-agent coordination graph
- Each agent runs its own teleological subgraph
- Agents negotiate a shared objective between rounds

Run:
    PYTHONPATH=src python examples/03_multi_agent_negotiation.py
"""

from __future__ import annotations

from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.enums import Direction
from synthetic_teleology.domain.values import ActionSpec, ObjectiveVector
from synthetic_teleology.environments.numeric import NumericEnvironment
from synthetic_teleology.graph import AgentConfig, build_multi_agent_graph
from synthetic_teleology.services.constraint_engine import ConstraintPipeline, PolicyFilter
from synthetic_teleology.services.coordination import ConsensusNegotiator
from synthetic_teleology.services.evaluation import NumericEvaluator
from synthetic_teleology.services.goal_revision import ThresholdUpdater
from synthetic_teleology.services.planning import GreedyPlanner


def _make_actions() -> list[ActionSpec]:
    actions: list[ActionSpec] = []
    for d in range(2):
        for sign, label in [(0.5, "pos"), (-0.5, "neg")]:
            effect = tuple(sign if i == d else 0.0 for i in range(2))
            actions.append(ActionSpec(
                name=f"s{d}_{label}",
                parameters={"effect": effect, "delta": effect},
            ))
    actions.append(ActionSpec(name="noop", parameters={"effect": (0.0, 0.0), "delta": (0.0, 0.0)}))
    return actions


def main() -> None:
    # Two agents with different goals, starting at different positions
    env_a = NumericEnvironment(dimensions=2, initial_state=(0.0, 0.0))
    env_b = NumericEnvironment(dimensions=2, initial_state=(10.0, 10.0))

    agent_configs = [
        AgentConfig(
            agent_id="explorer",
            goal=Goal(name="explore-high", objective=ObjectiveVector(
                values=(8.0, 8.0), directions=(Direction.APPROACH, Direction.APPROACH)
            )),
            perceive_fn=lambda: env_a.observe(),
            transition_fn=lambda a: env_a.step(a) if a else None,
            act_fn=lambda p, s: p.actions[0] if p.size > 0 else None,
            max_steps_per_round=5,
        ),
        AgentConfig(
            agent_id="conservator",
            goal=Goal(name="stay-low", objective=ObjectiveVector(
                values=(3.0, 3.0), directions=(Direction.APPROACH, Direction.APPROACH)
            )),
            perceive_fn=lambda: env_b.observe(),
            transition_fn=lambda a: env_b.step(a) if a else None,
            act_fn=lambda p, s: p.actions[0] if p.size > 0 else None,
            max_steps_per_round=5,
        ),
    ]

    negotiator = ConsensusNegotiator(max_rounds=50, tolerance=0.1, blending_rate=0.7)
    app = build_multi_agent_graph(agent_configs, negotiation_strategy=negotiator, max_rounds=3)

    actions = _make_actions()
    pipeline = ConstraintPipeline(checkers=[])

    print("=== Multi-Agent Negotiation ===")
    print("Explorer wants (8, 8), Conservator wants (3, 3)")
    print()

    result = app.invoke({
        "evaluator": NumericEvaluator(max_distance=15.0),
        "goal_updater": ThresholdUpdater(threshold=0.99),
        "planner": GreedyPlanner(action_space=actions),
        "constraint_pipeline": pipeline,
        "policy_filter": PolicyFilter(pipeline),
        "goal_achieved_threshold": 0.9,
        "max_rounds": 3,
        "events": [],
        "agent_results": {},
        "negotiation_round": 0,
    })

    print(f"Negotiation rounds: {result.get('negotiation_round', 0)}")
    if result.get("shared_objective"):
        print(f"Shared objective: {result['shared_objective'].values}")

    for agent_id, data in result.get("agent_results", {}).items():
        score = data["eval_signal"].score if data.get("eval_signal") else "N/A"
        print(f"  {agent_id}: steps={data['steps']}, eval_score={score}")

    print(f"Total events: {len(result.get('events', []))}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
