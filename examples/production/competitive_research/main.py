"""Entry point for the Competitive Research agent.

Run:
    PYTHONPATH=src python -m examples.production.competitive_research.main

Options:
    --steps 25     Max research steps
    --verbose      Show detailed step output
"""

import argparse
import os

from .agent import build_research_agent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Competitive Research Analyst — Goal-directed competitive intelligence agent"
    )
    parser.add_argument("--steps", type=int, default=25, help="Max research steps")
    parser.add_argument("--verbose", action="store_true", help="Show detailed step output")
    args = parser.parse_args()

    # Detect mode
    has_key = bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY"))
    mode = "LIVE LLM" if has_key else "SIMULATED (MockStructuredChatModel)"

    print("=" * 64)
    print("  Competitive Research Analyst")
    print("=" * 64)
    print(f"  Mode:      {mode}")
    print(f"  Max steps: {args.steps}")
    print("  Target:    TechRival Inc")
    print()

    # --- Build and run ---
    app, initial_state, research_state, knowledge_store, audit_trail = (
        build_research_agent(max_steps=args.steps)
    )

    print("Initial goal:")
    goal = initial_state["goal"]
    print(f"  {goal.description}")
    for criterion in goal.success_criteria:
        print(f"    - {criterion}")
    print()

    print("Running teleological research loop...")
    print("-" * 64)

    result = app.invoke(initial_state)

    # --- Results ---
    print("-" * 64)
    print()

    # Verbose: step-by-step
    if args.verbose:
        eval_history = result.get("eval_history", [])
        action_history = result.get("action_history", [])
        feedback = result.get("action_feedback", [])

        for i, sig in enumerate(eval_history):
            step_num = i + 1
            print(f"--- Step {step_num} ---")
            print(f"  Eval: {sig.score:.3f} (confidence {sig.confidence:.2f})")
            if sig.explanation:
                print(f"  Detail: {sig.explanation[:100]}")

            if i < len(action_history):
                action = action_history[i]
                tool_info = f" [tool: {action.tool_name}]" if action.tool_name else ""
                print(f"  Action: {action.name}{tool_info}")

            if i < len(feedback):
                fb = feedback[i]
                if fb.get("result"):
                    print(f"  Result: {str(fb['result'])[:90]}")

            print()

    # Research findings summary
    print(f"Research Findings ({len(research_state.findings)} total):")
    for i, finding in enumerate(research_state.findings[:20], 1):
        content_preview = finding.content[:80].replace("\n", " ")
        print(
            f"  {i:3d}. [{finding.source:30s}] "
            f"{content_preview}..."
        )
    if len(research_state.findings) > 20:
        print(f"  ... and {len(research_state.findings) - 20} more findings")
    print()

    # Topics covered
    print(f"Topics Covered ({len(research_state.topics_covered)}):")
    for topic in sorted(research_state.topics_covered):
        print(f"  - {topic}")
    print()

    # Sources consulted
    print(f"Sources Consulted ({len(research_state.sources_consulted)}):")
    for source in sorted(research_state.sources_consulted):
        print(f"  - {source}")
    print()

    # Pivot discovery
    print("Pivot Discovery:")
    print(f"  Discovered: {research_state.pivot_discovered}")
    print(f"  Analyzed:   {research_state.pivot_analyzed}")
    print()

    # Goal revisions from audit trail
    if len(audit_trail) > 0:
        print(f"Goal Audit Trail ({len(audit_trail)} entries):")
        for entry in audit_trail.entries:
            print(f"  [{entry.entry_id}] goal={entry.goal_id[:12]}...")
            print(f"    reason: {entry.revision_reason[:80]}")
            print(f"    eval_score: {entry.eval_score:.4f}")
        print()
    else:
        print("Goal Audit Trail: no revisions recorded")
        print()

    # Knowledge store entries
    print(f"Knowledge Store ({len(knowledge_store)} entries):")
    for key in knowledge_store.keys()[:15]:
        entry = knowledge_store.get(key)
        if entry is not None:
            value_preview = str(entry.value)[:60].replace("\n", " ")
            print(f"  {key:40s} -> {value_preview}")
    if len(knowledge_store) > 15:
        print(f"  ... and {len(knowledge_store) - 15} more entries")
    print()

    # Final goal state (may have been revised)
    final_goal = result.get("goal")
    if final_goal is not None:
        print("Final Goal State:")
        print(f"  ID:          {final_goal.goal_id}")
        print(f"  Version:     {final_goal.version}")
        print(f"  Description: {final_goal.description}")
        if final_goal.success_criteria:
            print("  Criteria:")
            for c in final_goal.success_criteria:
                print(f"    - {c}")
        print()

    # Summary
    eval_signal = result.get("eval_signal")
    score_str = f"{eval_signal.score:.4f}" if eval_signal else "N/A"

    print("=" * 64)
    print(f"  Stop reason:        {result.get('stop_reason', 'none')}")
    print(f"  Steps completed:    {result['step']}")
    print(f"  Final eval score:   {score_str}")
    print(f"  API calls used:     {research_state.api_calls_used}")
    print(f"  Thesis confidence:  {research_state.thesis_confidence:.2f}")
    print(f"  Topics covered:     {len(research_state.topics_covered)}")
    print(f"  Sources consulted:  {len(research_state.sources_consulted)}")
    print(f"  Pivot discovered:   {research_state.pivot_discovered}")
    print(f"  Audit entries:      {len(audit_trail)}")
    print(f"  Knowledge entries:  {len(knowledge_store)}")
    print(f"  Events emitted:     {len(result.get('events', []))}")
    print("=" * 64)


if __name__ == "__main__":
    main()
