"""Entry point for the Adaptive Learning Curriculum agent.

Run:
    PYTHONPATH=src python -m examples.production.learning_curriculum.main

Options:
    --steps 35         Max curriculum steps
    --verbose          Show detailed per-step output
"""

import argparse

from .agent import build_curriculum_agent
from .models import REACT_CURRICULUM


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adaptive Learning Curriculum — Goal-directed curriculum optimization agent"
    )
    parser.add_argument("--steps", type=int, default=35, help="Max curriculum steps")
    parser.add_argument("--verbose", action="store_true", help="Show detailed per-step output")
    args = parser.parse_args()

    # --- Build ---
    app, initial_state, curriculum_state, knowledge_store = build_curriculum_agent(
        max_steps=args.steps,
    )

    # --- Overview ---
    print("=" * 65)
    print("  Adaptive Learning Curriculum Agent")
    print("=" * 65)
    print(f"  Max steps:      {args.steps}")
    print(f"  Topics:         {len(REACT_CURRICULUM)}")
    print()

    print("Curriculum Topics:")
    for t in REACT_CURRICULUM:
        prereqs = ", ".join(t.prerequisites) if t.prerequisites else "(none)"
        print(f"  {t.name:20s}  difficulty={t.difficulty:.1f}  prereqs=[{prereqs}]")
    print()

    learner = curriculum_state.learner
    print("Initial Learner Profile:")
    for topic, mastery in sorted(learner.strengths.items(), key=lambda x: -x[1]):
        bar = "#" * int(mastery * 20)
        print(f"  {topic:20s}: {mastery:.2f}  {bar}")
    print(f"  Weaknesses: {', '.join(learner.weaknesses)}")
    print()

    print("Running teleological curriculum loop...")
    print("-" * 65)

    # --- Run ---
    result = app.invoke(initial_state)

    # --- Results ---
    print("-" * 65)
    print()

    # Quiz results
    print(f"Quiz Results ({len(curriculum_state.quiz_results)} quizzes):")
    for i, qr in enumerate(curriculum_state.quiz_results, 1):
        status = "PASS" if qr.passed else "FAIL"
        weak = f"  weak: {', '.join(qr.weak_areas)}" if qr.weak_areas else ""
        print(f"  {i:3d}. {qr.topic:20s}  {status}  score={qr.score:.3f}{weak}")
    print()

    # Mastery progression
    print("Final Mastery Levels:")
    for topic, mastery in sorted(
        curriculum_state.learner.strengths.items(), key=lambda x: -x[1]
    ):
        bar = "#" * int(mastery * 20)
        completed = "*" if topic in curriculum_state.topics_completed else " "
        print(f"  {completed} {topic:20s}: {mastery:.2f}  {bar}")
    print("  (* = topic completed)")
    print()

    # Topics completed
    completed = curriculum_state.topics_completed
    total = len(REACT_CURRICULUM)
    print(f"Topics Completed: {len(completed)}/{total}")
    for t in completed:
        print(f"    - {t}")
    print()

    # Knowledge store entries
    ks_keys = knowledge_store.keys()
    print(f"Knowledge Store ({len(ks_keys)} entries):")
    for key in ks_keys[:20]:
        entry = knowledge_store.get(key)
        if entry is not None:
            value_str = str(entry.value)[:60]
            print(f"  {key:35s}: {value_str}")
    if len(ks_keys) > 20:
        print(f"  ... and {len(ks_keys) - 20} more entries")
    print()

    # Goal revision history
    goal_history = result.get("goal_history", [])
    if goal_history:
        print(f"Goal Revisions ({len(goal_history)}):")
        for i, g in enumerate(goal_history, 1):
            desc = g.description[:70] if g.description else g.goal_id
            print(f"  {i}. {desc}")
        print()

    # Action history summary
    action_history = result.get("action_history", [])
    tool_counts: dict[str, int] = {}
    for action in action_history:
        tool = getattr(action, "tool_name", None) or "direct"
        tool_counts[tool] = tool_counts.get(tool, 0) + 1
    print(f"Action Summary ({len(action_history)} actions):")
    for tool, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
        print(f"  {tool:25s}: {count}")
    print()

    # Verbose step output
    if args.verbose:
        feedback = result.get("action_feedback", [])
        print("Step-by-Step Tool Results:")
        for fb in feedback:
            step = fb.get("step", "?")
            action = fb.get("action", "?")
            tool = fb.get("tool_name") or "direct"
            res = str(fb.get("result", ""))[:80]
            print(f"  [{step:3}] {action:30s} (via {tool}): {res}")
        print()

    # Final summary
    print("=" * 65)
    print(f"  Stop reason:          {result.get('stop_reason', 'none')}")
    print(f"  Steps completed:      {result['step']}")
    print(f"  Final eval score:     {result['eval_signal'].score:.4f}")
    print(f"  Topics completed:     {len(completed)}/{total}")
    print(f"  Lessons delivered:    {curriculum_state.lessons_delivered}")
    print(f"  Quizzes taken:        {curriculum_state.learner.quizzes_taken}")
    passed_q = sum(1 for q in curriculum_state.quiz_results if q.passed)
    print(f"  Quizzes passed:       {passed_q}/{len(curriculum_state.quiz_results)}")
    print(f"  Resources found:      {curriculum_state.resources_found}")
    print(f"  Time spent:           {curriculum_state.learner.total_time_spent:.0f} min")
    print(f"  Goal revisions:       {len(goal_history)}")
    print(f"  Knowledge entries:    {len(knowledge_store)}")
    print(f"  Events emitted:       {len(result.get('events', []))}")
    print("=" * 65)


if __name__ == "__main__":
    main()
