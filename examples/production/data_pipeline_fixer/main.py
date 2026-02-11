"""Entry point for the Data Pipeline Fixer agent.

Run:
    PYTHONPATH=src python -m examples.production.data_pipeline_fixer.main

Options:
    --steps 25     Max repair/monitoring rounds
    --verbose      Show detailed step output
"""

import argparse
import os

from .agent import build_pipeline_agent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Data Pipeline Fixer — Autonomous pipeline monitoring and repair agent"
    )
    parser.add_argument(
        "--steps", type=int, default=25, help="Max monitoring/repair rounds"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show detailed step output"
    )
    args = parser.parse_args()

    # Detect mode
    has_key = bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY"))
    mode = "LIVE LLM" if has_key else "SIMULATED (MockStructuredChatModel)"

    # --- Build ---
    app, initial_state, pipeline_state, knowledge_store, audit_trail = (
        build_pipeline_agent(max_steps=args.steps)
    )

    # --- Initial state ---
    print("=" * 65)
    print("  Data Pipeline Fixer Agent")
    print("=" * 65)
    print(f"  Mode:       {mode}")
    print(f"  Tables:     {', '.join(pipeline_state.tables)}")
    print(f"  Health:     {pipeline_state.health_score:.2f}")
    print(f"  Error rate: {pipeline_state.error_rate:.2f}")
    print(f"  Throughput: {pipeline_state.throughput:.0f} rec/s")
    print(f"  Schema:     {pipeline_state.schema_version}")
    print(f"  Max steps:  {args.steps}")
    print()

    # --- Evolving constraints ---
    ecm = initial_state.get("evolving_constraint_manager")
    if ecm is not None:
        print("Initial Evolving Constraints:")
        for c in ecm.constraints:
            print(f"  - {c}")
        print()

    print("Running teleological repair loop...")
    print("-" * 65)

    # --- Execute ---
    result = app.invoke(initial_state)

    # --- Results ---
    print("-" * 65)
    print()

    # Pipeline state after run
    print("Final Pipeline State:")
    print(f"  Health:         {pipeline_state.health_score:.2f}")
    print(f"  Error rate:     {pipeline_state.error_rate:.2f}")
    print(f"  Throughput:     {pipeline_state.throughput:.0f} rec/s")
    print(f"  Schema:         {pipeline_state.schema_version}")
    print(f"  Schema drift:   {pipeline_state.schema_drift_detected}")
    print(f"  Fix attempts:   {pipeline_state.fix_attempts_count}")
    print()

    # Fix history
    if pipeline_state.fix_history:
        print(f"Fix History ({len(pipeline_state.fix_history)} entries):")
        for i, fix in enumerate(pipeline_state.fix_history, 1):
            status = "OK" if fix.success else "FAIL"
            print(
                f"  {i:3d}. [{status:4s}] {fix.fix_type:20s} "
                f"-> {fix.target:15s} | {fix.details}"
            )
        print()

    # Schema drift details
    if pipeline_state.schema_drift_detected:
        print("Schema Drift Details:")
        for err in pipeline_state.errors:
            print(f"  ERROR:   {err}")
        for warn in pipeline_state.warnings:
            print(f"  WARNING: {warn}")
        print()

    # Evolving constraints
    if ecm is not None:
        print("Final Evolving Constraints:")
        for c in ecm.constraints:
            print(f"  - {c}")
        print()

        if ecm.evolution_history:
            print(f"Constraint Evolution History ({len(ecm.evolution_history)} rounds):")
            for i, evo in enumerate(ecm.evolution_history, 1):
                print(f"  Round {i}: {evo.reasoning}")
                for ev in evo.evolutions:
                    print(
                        f"    [{ev.evolution_type:6s}] {ev.constraint_text} "
                        f"(confidence={ev.confidence:.2f})"
                    )
                if evo.new_constraints:
                    print(f"    Added: {evo.new_constraints}")
                if evo.removed_constraints:
                    print(f"    Removed: {evo.removed_constraints}")
                if evo.modified_constraints:
                    for old, new in evo.modified_constraints:
                        print(f"    Modified: '{old}' -> '{new}'")
            print()

    # Goal audit trail
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

    # Knowledge store
    ks_keys = knowledge_store.keys()
    print(f"Knowledge Store ({len(knowledge_store)} entries):")
    for key in ks_keys[:15]:
        entry = knowledge_store.get(key)
        if entry is not None:
            value_preview = str(entry.value)[:60].replace("\n", " ")
            print(f"  {key:40s} -> {value_preview}")
    if len(knowledge_store) > 15:
        print(f"  ... and {len(knowledge_store) - 15} more entries")
    print()

    # Verbose: show eval history
    if args.verbose:
        eval_history = result.get("eval_history", [])
        if eval_history:
            print(f"Evaluation History ({len(eval_history)} entries):")
            for i, sig in enumerate(eval_history, 1):
                print(f"  Step {i:3d}: score={sig.score:+.4f}  {sig.explanation}")
            print()

        goal_history = result.get("goal_history", [])
        if goal_history:
            print(f"Goal History ({len(goal_history)} revisions):")
            for g in goal_history:
                print(f"  v{g.version}: {g.description}")
            print()

    # --- Summary ---
    print("=" * 65)
    print(f"  Stop reason:      {result.get('stop_reason', 'none')}")
    print(f"  Steps completed:  {result['step']}")
    eval_signal = result.get("eval_signal")
    if eval_signal is not None:
        print(f"  Final eval score: {eval_signal.score:+.4f}")
    print(f"  Pipeline health:  {pipeline_state.health_score:.2f}")
    print(f"  Error rate:       {pipeline_state.error_rate:.2f}")
    print(f"  Throughput:       {pipeline_state.throughput:.0f} rec/s")
    print(f"  Fix attempts:     {pipeline_state.fix_attempts_count}")
    print(f"  Events emitted:   {len(result.get('events', []))}")
    print(f"  Audit entries:    {len(audit_trail)}")
    print(f"  Knowledge entries:{len(knowledge_store)}")
    if ecm is not None:
        print(f"  Constraints:      {len(ecm.constraints)} active")
        print(f"  Evolutions:       {len(ecm.evolution_history)} rounds")
    print("=" * 65)


if __name__ == "__main__":
    main()
