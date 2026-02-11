"""Entry point for the Sales SDR agent.

Run:
    PYTHONPATH=src python -m examples.production.sales_sdr.main

Options:
    --leads 25         Number of leads in pipeline
    --target 5         Meeting booking target
    --steps 30         Max outreach rounds
    --live             Use real CRM API (needs HUBSPOT_API_KEY)
    --seed 42          Random seed for reproducibility
"""

import argparse

from .agent import build_sdr_agent
from .crm import get_crm


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sales SDR — Goal-directed sales development agent"
    )
    parser.add_argument("--leads", type=int, default=25, help="Number of leads in pipeline")
    parser.add_argument("--target", type=int, default=5, help="Meeting booking target")
    parser.add_argument("--steps", type=int, default=30, help="Max outreach rounds")
    parser.add_argument("--live", action="store_true", help="Use real CRM API")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # --- Setup ---
    crm = get_crm(num_leads=args.leads, seed=args.seed)
    mode = "LIVE (HubSpot)" if args.live else "SIMULATED"

    print("=" * 60)
    print(f"  Sales SDR Agent ({mode} mode)")
    print("=" * 60)
    print(f"  Leads: {args.leads}")
    print(f"  Meeting target: {args.target}")
    print(f"  Max steps: {args.steps}")
    print()

    # --- Build and run agent ---
    app, initial_state, leads, metrics, outreach_log = build_sdr_agent(
        crm=crm,
        meeting_target=args.target,
        max_steps=args.steps,
        seed=args.seed,
    )

    # Show initial pipeline
    status_counts: dict[str, int] = {}
    for lead in leads:
        status_counts[lead.status] = status_counts.get(lead.status, 0) + 1
    print("Initial Pipeline:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status:20s}: {count}")
    print()

    # Show top leads by ICP
    top_leads = sorted(leads, key=lambda ld: ld.icp_score, reverse=True)[:10]
    print("Top 10 Leads by ICP Score:")
    for lead in top_leads:
        print(
            f"  [{lead.id}] {lead.company:20s} | {lead.contact_name:20s} "
            f"| ICP: {lead.icp_score:.2f} | Status: {lead.status}"
        )
    print()

    print("Running teleological outreach loop...")
    print("-" * 60)

    result = app.invoke(initial_state)

    # --- Results ---
    print("-" * 60)
    print()

    # Outreach log
    actual_outreach = [o for o in outreach_log if o.get("channel")]
    print(f"Outreach actions: {len(actual_outreach)}")
    for o in actual_outreach[:25]:
        status_change = ""
        if o["old_status"] != o["new_status"]:
            status_change = f" [{o['old_status']} -> {o['new_status']}]"
        print(
            f"  Step {o['step']:3d}: {o['channel']:8s} -> {o['company']:20s} "
            f"| {o['subject'][:35]:35s}{status_change}"
        )
    if len(actual_outreach) > 25:
        print(f"  ... and {len(actual_outreach) - 25} more actions")
    print()

    # Channel effectiveness
    channel_counts: dict[str, int] = {}
    channel_conversions: dict[str, int] = {}
    for o in actual_outreach:
        ch = o["channel"]
        channel_counts[ch] = channel_counts.get(ch, 0) + 1
        if o["old_status"] != o["new_status"]:
            channel_conversions[ch] = channel_conversions.get(ch, 0) + 1

    print("Channel Effectiveness:")
    for ch in ["email", "call", "linkedin"]:
        total = channel_counts.get(ch, 0)
        conv = channel_conversions.get(ch, 0)
        rate = conv / total if total > 0 else 0
        print(f"  {ch:10s}: {total:3d} sent, {conv:3d} conversions ({rate:.0%})")
    print()

    # Final pipeline funnel
    final_status: dict[str, int] = {}
    for lead in leads:
        final_status[lead.status] = final_status.get(lead.status, 0) + 1
    print("Final Pipeline Funnel:")
    funnel_order = ["new", "contacted", "engaged", "qualified", "meeting_booked", "disqualified"]
    for status in funnel_order:
        count = final_status.get(status, 0)
        bar = "#" * count
        print(f"  {status:20s}: {count:3d} {bar}")
    print()

    # Summary
    print("=" * 60)
    print(f"  Stop reason:        {result.get('stop_reason', 'none')}")
    print(f"  Steps completed:    {result['step']}")
    print(f"  Final eval score:   {result['eval_signal'].score:.4f}")
    print(f"  Meetings booked:    {metrics.meetings_booked}/{args.target}")
    print(f"  Conversion rate:    {metrics.conversion_rate:.1%}")
    print(f"  Engagement rate:    {metrics.engagement_rate:.1%}")
    print(f"  Qualification rate: {metrics.qualification_rate:.1%}")
    print(f"  Events emitted:     {len(result.get('events', []))}")
    print("=" * 60)


if __name__ == "__main__":
    main()
