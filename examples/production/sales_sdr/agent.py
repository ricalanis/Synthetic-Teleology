"""Graph wiring for the Sales SDR agent.

Uses LLM mode (with_model + with_goal) for text-based goal description,
while keeping custom CRM-specific evaluator and planner that override
the LLM defaults.

In simulated mode (no API key), uses MockStructuredChatModel.
The mock only needs RevisionOutput responses since evaluator/planner
are custom implementations.
"""

import os
import time

from synthetic_teleology.domain.values import ActionSpec, PolicySpec, StateSnapshot
from synthetic_teleology.graph import GraphBuilder

from .crm import CRMProvider
from .models import Lead, OutreachAction, PipelineMetrics
from .strategies import (
    ContactFrequencyChecker,
    DailyLimitChecker,
    OutreachPlanner,
    PipelineEvaluator,
)


def _refresh_metrics(leads: list[Lead], metrics: PipelineMetrics) -> None:
    """Recompute pipeline metrics from current lead states."""
    metrics.total_leads = len(leads)
    metrics.contacted = sum(1 for ld in leads if ld.status != "new")
    metrics.engaged = sum(1 for ld in leads if ld.status == "engaged")
    metrics.qualified = sum(1 for ld in leads if ld.status == "qualified")
    metrics.meetings_booked = sum(1 for ld in leads if ld.status == "meeting_booked")
    metrics.disqualified = sum(1 for ld in leads if ld.status == "disqualified")


def _execute_outreach(
    crm: CRMProvider,
    leads_map: dict[str, Lead],
    metrics: PipelineMetrics,
    action: ActionSpec,
    outreach_log: list[dict],
) -> None:
    """Execute an outreach action through the CRM."""
    params = action.parameters
    act = params.get("action", "wait")

    if act != "outreach":
        return

    lead_id = params["lead_id"]
    channel = params["channel"]
    lead = leads_map.get(lead_id)
    if lead is None:
        return

    outreach = OutreachAction(
        lead_id=lead_id,
        channel=channel,
        subject=params.get("subject", ""),
        body=params.get("body", ""),
    )

    old_status = lead.status
    crm.log_activity(lead_id, outreach)

    # Update daily counters
    if channel == "email":
        metrics.emails_sent_today += 1
    elif channel == "call":
        metrics.calls_made_today += 1
    elif channel == "linkedin":
        metrics.linkedin_sent_today += 1

    outreach_log.append({
        "step": len(outreach_log) + 1,
        "lead_id": lead_id,
        "company": lead.company,
        "channel": channel,
        "subject": outreach.subject,
        "old_status": old_status,
        "new_status": lead.status,
        "contact_count": lead.contact_count,
    })


def _get_model():
    """Get a real or mock LLM model.

    In LLM mode, the model is used by the LLMReviser for goal revision.
    Custom evaluator and planner override the LLM defaults, so the model
    is only needed for revision reasoning.
    """
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0.5)
        except ImportError:
            pass
    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model="gpt-4o", temperature=0.5)
        except ImportError:
            pass

    # Simulated mode: mock provides RevisionOutput responses
    from synthetic_teleology.services.llm_revision import RevisionOutput
    from synthetic_teleology.testing import MockStructuredChatModel

    return MockStructuredChatModel(structured_responses=[
        RevisionOutput(
            should_revise=False,
            reasoning="Pipeline progressing, no goal revision needed",
        ),
    ])


def build_sdr_agent(
    crm: CRMProvider,
    meeting_target: int = 5,
    max_steps: int = 30,
    seed: int = 42,
):
    """Build a LangGraph sales SDR agent.

    Uses LLM mode (with_model + with_goal) for text-based goal with
    custom CRM-specific evaluator and planner.

    Returns ``(app, initial_state, leads, metrics, outreach_log)`` tuple.
    """
    leads = crm.get_leads()
    leads_map: dict[str, Lead] = {ld.id: ld for ld in leads}
    metrics = PipelineMetrics()
    _refresh_metrics(leads, metrics)

    outreach_log: list[dict] = []

    # --- Environment callbacks ---

    def perceive_fn() -> StateSnapshot:
        """Observe pipeline state as a StateSnapshot."""
        _refresh_metrics(leads, metrics)

        # Encode pipeline state as values tuple
        values = (
            float(metrics.total_leads),
            float(metrics.contacted),
            float(metrics.engaged),
            float(metrics.qualified),
            float(metrics.meetings_booked),
            float(metrics.disqualified),
            metrics.conversion_rate,
            metrics.engagement_rate,
        )

        observation = (
            f"Pipeline: {metrics.total_leads} leads, "
            f"{metrics.contacted} contacted, {metrics.engaged} engaged, "
            f"{metrics.qualified} qualified, {metrics.meetings_booked}/{meeting_target} meetings. "
            f"Conversion: {metrics.conversion_rate:.1%}, "
            f"Engagement: {metrics.engagement_rate:.1%}."
        )

        return StateSnapshot(
            timestamp=time.time(),
            values=values,
            observation=observation,
            metadata={
                "meetings": metrics.meetings_booked,
                "target": meeting_target,
                "contacted": metrics.contacted,
                "engaged": metrics.engaged,
                "qualified": metrics.qualified,
            },
        )

    def act_fn(policy: PolicySpec, state: StateSnapshot) -> ActionSpec | None:
        """Select the first action from the policy."""
        if policy.size > 0:
            return policy.actions[0]
        return None

    def transition_fn(action: ActionSpec) -> None:
        """Execute outreach and update CRM."""
        _execute_outreach(crm, leads_map, metrics, action, outreach_log)

    # --- Build strategies ---
    evaluator = PipelineEvaluator(leads, metrics, meeting_target)
    planner = OutreachPlanner(leads, metrics)
    freq_checker = ContactFrequencyChecker(leads)
    limit_checker = DailyLimitChecker(metrics)

    model = _get_model()

    app, initial_state = (
        GraphBuilder("sales-sdr")
        .with_model(model)
        .with_goal(
            f"Book {meeting_target} meetings from the sales pipeline",
            criteria=[
                f"Meeting booking rate >= {meeting_target}/{meeting_target}",
                "Pipeline engagement rate > 50%",
                "Respect daily contact limits and frequency constraints",
            ],
        )
        .with_evaluator(evaluator)
        .with_planner(planner)
        .with_constraint_checkers(freq_checker, limit_checker)
        .with_max_steps(max_steps)
        .with_goal_achieved_threshold(0.85)
        .with_environment(
            perceive_fn=perceive_fn,
            act_fn=act_fn,
            transition_fn=transition_fn,
        )
        .with_metadata(
            crm=crm,
            metrics=metrics,
            meeting_target=meeting_target,
        )
        .build()
    )

    return app, initial_state, leads, metrics, outreach_log
