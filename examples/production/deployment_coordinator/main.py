"""Entry point for the Deployment Coordinator.

Run:
    PYTHONPATH=src python -m examples.production.deployment_coordinator.main

Options:
    --rounds 2       Number of negotiation rounds (agents run rounds+1 times)
    --verbose        Show full reasoning traces
"""

import argparse

from .agent import build_deployment_coordinator


def _fmt_score(score) -> str:
    """Format a score value safely."""
    if score is None:
        return "n/a"
    if isinstance(score, (int, float)):
        return f"{score:.2f}"
    return str(score)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deployment Coordinator — Multi-agent deployment orchestration"
    )
    parser.add_argument(
        "--rounds", type=int, default=2,
        help="Number of negotiation rounds (default: 2)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show full reasoning traces",
    )
    args = parser.parse_args()

    # --- Build ---
    app, initial_input, deployment_state, mode = build_deployment_coordinator(
        max_rounds=args.rounds,
    )

    print("=" * 70)
    print("  Deployment Coordinator")
    print("=" * 70)
    print(f"  Mode: {mode}")
    print(f"  Negotiation rounds: {args.rounds}")
    print("  Agents: release-agent, security-agent, sre-agent")
    print()

    # --- Deployment plan overview ---
    print("Deployment Plan:")
    print("  Release: v2.5")
    print("  Pipeline: planning -> canary -> staging -> production")
    print(f"  Services: {', '.join(s.service for s in deployment_state.services)}")
    print()

    print("Agent Goals:")
    print("  release-agent  : Ship release v2.5 on schedule with zero rollbacks")
    print("  security-agent : Ensure zero critical vulnerabilities before production")
    print("  sre-agent      : Maintain 99.9% uptime during deployment")
    print()

    print("Running multi-agent coordination loop...")
    print("-" * 70)

    # --- Run ---
    result = app.invoke(initial_input)

    print("-" * 70)
    print()

    # --- Per-agent results ---
    agent_results = result.get("agent_results", {})

    for agent_id in ["release-agent", "security-agent", "sre-agent"]:
        agent_result = agent_results.get(agent_id)
        if agent_result is None:
            continue

        goal = agent_result.get("final_goal")
        signal = agent_result.get("eval_signal")
        steps = agent_result.get("steps", 0)
        stop = agent_result.get("stop_reason", "unknown")

        print(f"--- {agent_id} ---")
        if goal:
            desc = getattr(goal, "description", str(goal))
            print(f"  Goal: {desc[:90]}")
        if signal:
            print(f"  Final score: {_fmt_score(signal.score)} "
                  f"(confidence: {_fmt_score(signal.confidence)})")
            if signal.criteria_scores and args.verbose:
                for criterion, score in signal.criteria_scores.items():
                    print(f"    {criterion}: {_fmt_score(score)}")
        print(f"  Steps: {steps}, Stop: {stop}")
        if args.verbose and agent_result.get("reasoning_trace"):
            for trace in agent_result["reasoning_trace"][-2:]:
                node = trace.get("node", "?")
                reasoning = trace.get("reasoning", "")[:120]
                print(f"    [{node}] {reasoning}")
        print()

    # --- Negotiation outcomes ---
    shared = result.get("shared_direction", "")
    neg_round = result.get("negotiation_round", 0)

    if shared:
        print("=" * 70)
        print("  Negotiation Outcome")
        print("=" * 70)
        print(f"  Rounds completed: {neg_round}")
        # Show shared direction, wrapping at 70 chars
        words = shared.split()
        lines = []
        current = "  "
        for word in words:
            if len(current) + len(word) + 1 > 68:
                lines.append(current)
                current = "  " + word
            else:
                current += " " + word if current.strip() else "  " + word
        if current.strip():
            lines.append(current)
        print("  Direction:")
        for line in lines:
            print(f"  {line}")
        print()

    # --- Deployment state ---
    print("=" * 70)
    print("  Final Deployment Status")
    print("=" * 70)
    print(f"  Stage:              {deployment_state.stage}")
    print(f"  Progress:           {deployment_state.deployment_progress:.0%}")
    print(f"  Canary passed:      {deployment_state.canary_passed}")
    print(f"  Staging passed:     {deployment_state.staging_passed}")
    print(f"  Rollbacks:          {deployment_state.rollback_count}")
    print(f"  CVE discovered:     {deployment_state.cve_discovered}")
    print(f"  Open CVEs:          {len(deployment_state.cve_reports)}")
    print(f"  Patches applied:    {deployment_state.patches_applied}")
    print(f"  SLA violations:     {deployment_state.sla_violations}")
    print()

    print("  Service Health:")
    for svc in deployment_state.services:
        print(f"    {svc.service:20s} status={svc.status:10s} "
              f"latency={svc.latency_ms:6.1f}ms  error_rate={svc.error_rate:.4f}")
    print()

    # --- Event timeline ---
    events = result.get("events", [])
    print("=" * 70)
    print(f"  Event Timeline ({len(events)} events)")
    print("=" * 70)

    for i, evt in enumerate(events, 1):
        etype = evt.get("type", "?")
        if etype == "agent_round_completed":
            agent = evt.get("agent_id", "?")
            score = evt.get("eval_score", 0)
            print(f"  {i:2d}. [{etype}] {agent}: score={score:.2f}")
        elif etype == "llm_negotiation_completed":
            neg_r = evt.get("round", "?")
            conf = evt.get("confidence", 0)
            direction = evt.get("shared_direction", "")[:60]
            print(f"  {i:2d}. [{etype}] round={neg_r}, "
                  f"confidence={conf:.2f}")
            if direction:
                print(f"      direction: {direction}...")
        else:
            print(f"  {i:2d}. [{etype}]")

    print()

    # --- CVE timeline ---
    if deployment_state.cve_discovered:
        print("CVE Incident Timeline:")
        print("  1. Pass 1: All scans clean. Deployment proceeds normally.")
        print("  2. Pass 2: CVE-2024-31337 discovered (critical RCE in api-gateway)")
        print("     -> Security agent revises goal to prioritize patching")
        print("     -> Negotiation: all agents agree to pause production promotion")
        print("     -> SRE scales capacity to maintain SLA")
        print("     -> Patch applied for CVE-2024-31337")
        patches = deployment_state.patches_applied
        print(f"  3. Pass 3: Post-patch scan clean. Patches applied: {patches}")
        print(f"     -> Production promotion: "
              f"{'COMPLETED' if deployment_state.stage == 'production' else 'PENDING'}")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
