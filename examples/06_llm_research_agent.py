#!/usr/bin/env python3
"""Example 06: LLM-powered research agent configuration.

Demonstrates:
- Configuring an LLMAgent with LLMAgentConfig
- Using a mock LLM provider (no API keys required)
- Building an LLMAgent via the from_provider factory
- Running the teleological loop with LLM-backed strategies
- Inspecting the agent's evaluation, revision, and planning behaviour

Note: This example uses a mock provider that returns deterministic
JSON responses.  To use a real LLM, replace MockLLMProvider with
an actual provider (e.g., AnthropicProvider, OpenAIProvider).

Run:
    PYTHONPATH=src python examples/06_llm_research_agent.py
"""

from __future__ import annotations

import json
import time

from synthetic_teleology.agents.llm import LLMAgent, LLMAgentConfig
from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.enums import Direction
from synthetic_teleology.domain.values import (
    ActionSpec,
    ObjectiveVector,
    StateSnapshot,
)
from synthetic_teleology.infrastructure.event_bus import EventBus

# -- Mock LLM Provider (no API keys needed) --------------------------------


class MockLLMProvider:
    """A deterministic mock provider that returns pre-computed JSON responses.

    This allows the example to run without any external LLM service.
    The mock simulates reasonable evaluation, revision, and planning
    responses for a 3-dimensional APPROACH objective.
    """

    def __init__(self, name: str = "mock-llm") -> None:
        self._name = name
        self._call_count = 0

    @property
    def provider_name(self) -> str:
        return self._name

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.5,
    ) -> str:
        """Return a deterministic JSON response based on prompt content."""
        self._call_count += 1

        # Detect which phase is calling based on prompt keywords
        if "Goal Evaluation Request" in prompt:
            return self._eval_response()
        elif "Goal Revision Request" in prompt:
            return self._revision_response()
        elif "Planning Request" in prompt:
            return self._planning_response()
        else:
            return self._generic_response()

    def _eval_response(self) -> str:
        # Score improves slightly each call
        score = min(0.9, 0.3 + self._call_count * 0.1)
        return json.dumps({
            "score": round(score, 2),
            "confidence": 0.85,
            "explanation": f"Mock evaluation (call #{self._call_count}): "
            f"state shows moderate alignment with target.",
        })

    def _revision_response(self) -> str:
        # Suggest slightly adjusted values
        return json.dumps({
            "revised_values": [4.8, 5.1, 4.9],
            "reason": "Fine-tuning objective based on observed state drift.",
            "confidence": 0.75,
        })

    def _planning_response(self) -> str:
        return json.dumps({
            "actions": [
                {"name": "adjust_dim0", "parameters": {"delta": 0.5}, "cost": 0.1},
                {"name": "adjust_dim1", "parameters": {"delta": -0.2}, "cost": 0.1},
                {"name": "observe", "parameters": {}, "cost": 0.0},
            ],
            "reasoning": "Prioritise dimensions furthest from target; "
            "observe after adjustment.",
        })

    def _generic_response(self) -> str:
        return json.dumps({"status": "ok", "message": "Mock response"})


def main() -> None:
    print("=== LLM Research Agent ===\n")

    # -- Provider (mock) ------------------------------------------------------
    provider = MockLLMProvider(name="mock-research-llm")
    print(f"Provider: {provider.provider_name}")

    # -- Configuration --------------------------------------------------------
    config = LLMAgentConfig(
        eval_system_prompt=(
            "You are an evaluation critic for a research-oriented AI agent. "
            "Assess how well the current state aligns with the research goal."
        ),
        eval_temperature=0.2,
        revision_system_prompt=(
            "You are a goal-revision expert. Propose refined objective "
            "values that better capture the research target."
        ),
        revision_temperature=0.3,
        planning_system_prompt=(
            "You are a research planner. Propose a sequence of actions "
            "to advance the research objective."
        ),
        planning_temperature=0.5,
        available_actions=[
            ActionSpec(name="adjust_dim0", parameters={"delta": 0.5}, cost=0.1),
            ActionSpec(name="adjust_dim1", parameters={"delta": -0.2}, cost=0.1),
            ActionSpec(name="adjust_dim2", parameters={"delta": 0.3}, cost=0.1),
            ActionSpec(name="observe", parameters={}, cost=0.0),
        ],
    )
    print(f"Config: {config}")
    print()

    # -- Agent via factory ----------------------------------------------------
    agent = LLMAgent.from_provider(
        agent_id="research-agent-01",
        provider=provider,
        target_values=(5.0, 5.0, 5.0),
        directions=(Direction.APPROACH, Direction.APPROACH, Direction.APPROACH),
        goal_name="research-target",
        config=config,
    )

    print(f"Agent: {agent}")
    print(f"  Goal: {agent.current_goal.name} (id={agent.current_goal.goal_id})")
    print(f"  Evaluator: {type(agent.evaluator).__name__}")
    print(f"  Updater: {type(agent.updater).__name__}")
    print(f"  Planner: {type(agent.planner).__name__}")
    print(f"  Provider: {type(agent.provider).__name__}")
    print()

    # -- Manual step-by-step execution ----------------------------------------
    print("--- Step-by-step LLM teleological loop ---\n")

    # Simulate a state observation
    state = StateSnapshot(
        timestamp=time.time(),
        values=(3.0, 4.5, 2.0),
    )
    goal = agent.current_goal

    # 1. Evaluate
    print("1. Evaluate (LLMCriticEvaluator):")
    eval_signal = agent.evaluator.evaluate(goal, state)
    print(f"   Score:      {eval_signal.score:.4f}")
    print(f"   Confidence: {eval_signal.confidence:.4f}")
    print(f"   Explanation: {eval_signal.explanation}")
    print()

    # 2. Revise goal
    print("2. Revise goal (LLMGoalEditor):")
    revised = agent.updater.update(goal, state, eval_signal)
    if revised is not None:
        print(f"   Revised objective: {revised.objective.values if revised.objective else 'None'}")
    else:
        print("   No revision proposed.")
    print()

    # 3. Plan
    print("3. Plan (LLMPlanner):")
    active_goal = revised if revised is not None else goal
    policy = agent.planner.plan(active_goal, state)
    print(f"   Policy size: {policy.size}")
    for i, action in enumerate(policy.actions):
        print(f"   Action {i}: {action.name} (cost={action.cost})")
    print()

    # -- Direct construction (alternative) ------------------------------------
    print("--- Alternative: direct LLMAgent construction ---\n")

    objective = ObjectiveVector(
        values=(10.0, 8.0),
        directions=(Direction.MAXIMIZE, Direction.APPROACH),
    )
    goal_direct = Goal(name="direct-goal", objective=objective)

    agent_direct = LLMAgent(
        agent_id="direct-llm-agent",
        initial_goal=goal_direct,
        event_bus=EventBus(),
        provider=provider,
        config=LLMAgentConfig(
            eval_temperature=0.1,
            revision_temperature=0.2,
            planning_temperature=0.4,
        ),
    )
    print(f"Direct agent: {agent_direct}")
    print(f"  LLM config: {agent_direct.llm_config}")
    print()

    print("Done.")


if __name__ == "__main__":
    main()
