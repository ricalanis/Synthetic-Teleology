"""Entry point for the Investment Thesis Builder agent.

Run:
    PYTHONPATH=src python -m examples.production.investment_thesis.main

Options:
    --steps 30         Max research rounds
    --verbose          Show step-by-step output
"""

import argparse
import os

from .agent import build_thesis_agent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Investment Thesis Builder — Goal-directed investment analysis agent"
    )
    parser.add_argument("--steps", type=int, default=30, help="Max research rounds")
    parser.add_argument("--verbose", action="store_true", help="Show step-by-step output")
    args = parser.parse_args()

    has_api_key = bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY"))
    mode = "LIVE LLM" if has_api_key else "SIMULATED"

    print("=" * 65)
    print("  Investment Thesis Builder Agent")
    print("=" * 65)
    print(f"  Mode: {mode}")
    print("  Target: NovaTech Corp (NVTK)")
    print(f"  Max steps: {args.steps}")
    print()

    # --- Build and run agent ---
    app, initial_state, thesis_state, knowledge_store, audit_trail = build_thesis_agent(
        max_steps=args.steps,
    )

    # Show initial state
    print("Initial Goal:")
    print(f"  {initial_state['goal'].description}")
    print("  Criteria:")
    for c in initial_state["goal"].success_criteria:
        print(f"    - {c}")
    print()

    print("Available Tools:")
    for tool in initial_state.get("tools", []):
        print(f"  [{tool.name}] {tool.description}")
    print()

    print("Running teleological research loop...")
    print("-" * 65)

    result = app.invoke(initial_state)

    print("-" * 65)
    print()

    # --- Verbose: step-by-step ---
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

    # --- Findings summary ---
    print("Research Findings:")
    print(f"  Financial data points: {len(thesis_state.financials)}")
    print(f"  News items collected:  {len(thesis_state.news)}")
    print(f"  Unique sources:        {len(thesis_state.sources)}")
    if thesis_state.sources:
        for src in sorted(thesis_state.sources):
            print(f"    - {src}")
    print()

    # Thesis components
    print("Thesis Components:")
    if thesis_state.thesis_components:
        for section, content in thesis_state.thesis_components.items():
            preview = content[:80] + "..." if len(content) > 80 else content
            print(f"  [{section}] {preview}")
    else:
        print("  (components built via tool outputs, stored in working memory)")
    print()

    # Risk analysis
    print("Risk Analysis:")
    print(f"  Lawsuit discovered: {'YES' if thesis_state.lawsuit_discovered else 'No'}")
    print(f"  Risk analyzed:      {'YES' if thesis_state.risk_analyzed else 'No'}")
    if thesis_state.risk_factors:
        print("  Risk factors:")
        for rf in thesis_state.risk_factors:
            print(f"    - {rf}")
    print()

    # Key financial metrics
    if thesis_state.financials:
        print("Key Financial Metrics:")
        for fm in thesis_state.financials[:10]:
            if fm.value >= 1_000_000:
                val_str = f"${fm.value:,.0f}"
            elif fm.value < 1:
                val_str = f"{fm.value:.2%}"
            else:
                val_str = f"{fm.value:,.2f}"
            print(f"  {fm.name:25s}: {val_str:>15s}  ({fm.period}, {fm.source})")
        if len(thesis_state.financials) > 10:
            print(f"  ... and {len(thesis_state.financials) - 10} more metrics")
        print()

    # News highlights
    if thesis_state.news:
        print("News Highlights:")
        for item in thesis_state.news[:8]:
            sentiment_tag = f"[{item.sentiment.upper():8s}]"
            print(f"  {sentiment_tag} {item.headline[:65]}")
            print(f"             Source: {item.source}")
        if len(thesis_state.news) > 8:
            print(f"  ... and {len(thesis_state.news) - 8} more items")
        print()

    # Audit trail
    entries = audit_trail._entries
    if entries:
        print(f"Audit Trail ({len(entries)} entries):")
        for entry in entries[:5]:
            print(f"  [{entry.goal_id[:12]}...] reason: {entry.revision_reason[:60]}")
        if len(entries) > 5:
            print(f"  ... and {len(entries) - 5} more entries")
        print()

    # Knowledge store
    ks_entries = knowledge_store._entries
    if ks_entries:
        print(f"Knowledge Store ({len(ks_entries)} entries):")
        for key in list(ks_entries.keys())[:5]:
            print(f"  {key}: {str(ks_entries[key].value)[:60]}")
        if len(ks_entries) > 5:
            print(f"  ... and {len(ks_entries) - 5} more entries")
        print()

    # --- Final summary ---
    goal_history = result.get("goal_history", [])
    eval_history = result.get("eval_history", [])
    action_history = result.get("action_history", [])

    print("=" * 65)
    print(f"  Stop reason:        {result.get('stop_reason', 'none')}")
    print(f"  Steps completed:    {result['step']}")
    if eval_history:
        print(f"  Final eval score:   {result['eval_signal'].score:.4f}")
    print(f"  Actions taken:      {len(action_history)}")
    print(f"  Goal revisions:     {len(goal_history)}")
    print(f"  Events emitted:     {len(result.get('events', []))}")

    # Show goal revision if it happened
    if goal_history:
        final_goal = result.get("goal")
        if final_goal:
            print(f"\n  Final Goal: {final_goal.description}")
            print(f"  Final Criteria ({len(final_goal.success_criteria)}):")
            for c in final_goal.success_criteria:
                print(f"    - {c}")

    print("=" * 65)


if __name__ == "__main__":
    main()
