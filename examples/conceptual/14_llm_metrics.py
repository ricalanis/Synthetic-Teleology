#!/usr/bin/env python3
"""Example 14: Measuring LLM agent performance with metrics.

Demonstrates:
- Running an LLM agent and collecting metrics
- ReasoningQuality metric for LLM reasoning traces
- Combining LLM-specific and traditional metrics
- Using AgentLog with LLM-mode fields (reasoning, hypotheses_count)

This example uses mock data (no API key required).

Run:
    PYTHONPATH=src python examples/conceptual/14_llm_metrics.py
"""

from __future__ import annotations

from synthetic_teleology.measurement.collector import AgentLog, AgentLogEntry
from synthetic_teleology.measurement.metrics import (
    GoalPersistence,
    LyapunovStability,
    ReasoningQuality,
    TeleologicalCoherence,
)


def main() -> None:
    print("=== LLM Agent Metrics ===")
    print()

    # -- Simulate an LLM agent's log --
    log = _build_mock_llm_log()

    print(f"Agent: {log.agent_id}")
    print(f"Steps: {log.num_steps}")
    print(f"Goal revisions: {log.revision_count}")
    print()

    # -- Compute metrics --
    metrics = [
        ReasoningQuality(),
        TeleologicalCoherence(),
        GoalPersistence(),
        LyapunovStability(),
    ]

    print("Metric Results:")
    print(f"{'Metric':<30} {'Value':>8}  Description")
    print("-" * 80)

    for metric in metrics:
        result = metric.compute(log)
        desc = metric.describe()[:45]
        print(f"{result.name:<30} {result.value:>8.4f}  {desc}")

    print()

    # -- Show reasoning trace details --
    print("Reasoning Trace (per step):")
    for entry in log.entries:
        reasoning_preview = entry.reasoning[:60] if entry.reasoning else "(no reasoning)"
        print(
            f"  Step {entry.step}: score={entry.eval_score:.3f}, "
            f"hypotheses={entry.hypotheses_count}, "
            f"reasoning='{reasoning_preview}'"
        )
    print()

    print("Key insight: ReasoningQuality measures presence, diversity, and")
    print("depth of LLM reasoning. Numeric-mode agents score 0.0 since they")
    print("don't produce natural language reasoning traces.")
    print()
    print("Done.")


def _build_mock_llm_log() -> AgentLog:
    """Build a mock AgentLog simulating an LLM agent run."""
    log = AgentLog(agent_id="llm-productivity-agent")

    entries = [
        AgentLogEntry(
            step=0,
            timestamp=1000.0,
            goal_id="productivity-goal",
            eval_score=0.15,
            eval_confidence=0.6,
            action_name="research_bottlenecks",
            reasoning="Initial analysis suggests three main areas: meeting overhead, "
                      "tool fragmentation, and unclear priorities. Need data to confirm.",
            hypotheses_count=3,
        ),
        AgentLogEntry(
            step=1,
            timestamp=1001.0,
            goal_id="productivity-goal",
            eval_score=0.35,
            eval_confidence=0.7,
            action_name="survey_team",
            reasoning="Survey results confirm meeting overhead is the #1 bottleneck. "
                      "73% of team members report spending >15h/week in meetings.",
            hypotheses_count=3,
        ),
        AgentLogEntry(
            step=2,
            timestamp=1002.0,
            goal_id="productivity-goal",
            eval_score=0.55,
            eval_confidence=0.8,
            action_name="propose_solutions",
            reasoning="Three solutions proposed: (1) async standups via Slack, "
                      "(2) meeting-free Wednesdays, (3) decision docs instead of review meetings. "
                      "Expected combined impact: 20-30% productivity gain.",
            hypotheses_count=3,
            goal_revised=False,
        ),
        AgentLogEntry(
            step=3,
            timestamp=1003.0,
            goal_id="productivity-goal",
            eval_score=0.72,
            eval_confidence=0.85,
            action_name="estimate_impact",
            reasoning="Impact estimates: async standups saves ~5h/week per person, "
                      "meeting-free Wednesdays saves ~3h, decision docs saves ~2h. "
                      "Total: ~10h/week = 25% productivity gain. Goal achievable.",
            hypotheses_count=2,
        ),
        AgentLogEntry(
            step=4,
            timestamp=1004.0,
            goal_id="productivity-goal",
            eval_score=0.88,
            eval_confidence=0.9,
            action_name="finalize_plan",
            reasoning="Implementation plan complete with timelines: "
                      "Week 1: async standups pilot, Week 2-3: meeting audit, "
                      "Week 4: decision doc templates. All criteria met.",
            hypotheses_count=2,
        ),
    ]

    for entry in entries:
        log.entries.append(entry)

    return log


if __name__ == "__main__":
    main()
