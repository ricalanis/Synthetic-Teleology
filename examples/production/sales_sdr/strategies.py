"""Custom evaluator, planner, and constraint checkers for sales outreach."""

from datetime import datetime

from synthetic_teleology.domain.entities import Goal
from synthetic_teleology.domain.values import (
    ActionSpec,
    EvalSignal,
    PolicySpec,
    StateSnapshot,
)
from synthetic_teleology.services.constraint_engine import BaseConstraintChecker
from synthetic_teleology.services.evaluation import BaseEvaluator
from synthetic_teleology.services.planning import BasePlanner

from .models import Lead, OutreachAction, PipelineMetrics


class PipelineEvaluator(BaseEvaluator):
    """Score = weighted pipeline health.

    Factors:
    - Meeting booking rate vs target
    - Qualified lead ratio
    - Response rate trend
    - Pipeline velocity (leads moving through stages)
    """

    def __init__(
        self,
        leads: list[Lead],
        metrics: PipelineMetrics,
        meeting_target: int = 5,
    ) -> None:
        self._leads = leads
        self._metrics = metrics
        self._meeting_target = meeting_target

    def evaluate(self, goal: Goal, state: StateSnapshot) -> EvalSignal:
        m = self._metrics

        # Meeting progress (0-1, weight=0.4)
        meeting_score = min(m.meetings_booked / max(self._meeting_target, 1), 1.0)

        # Qualification rate (weight=0.2)
        qual_score = m.qualification_rate

        # Engagement rate (weight=0.2)
        engage_score = m.engagement_rate

        # Pipeline coverage: leads still active / total (weight=0.2)
        active = sum(
            1 for lead in self._leads
            if lead.status not in ("disqualified", "meeting_booked")
        )
        coverage_score = active / max(m.total_leads, 1)

        # Weighted score
        raw_score = (
            0.4 * meeting_score
            + 0.2 * qual_score
            + 0.2 * engage_score
            + 0.2 * coverage_score
        )
        # Normalize to [-1, 1] (0.5 raw → 0.0 normalized)
        score = max(-1.0, min(1.0, (raw_score - 0.5) * 2))

        return EvalSignal(
            score=score,
            confidence=min(1.0, m.contacted / max(m.total_leads * 0.3, 1)),
            explanation=(
                f"Meetings: {m.meetings_booked}/{self._meeting_target}, "
                f"Qual: {m.qualified}, Engaged: {m.engaged}, "
                f"Conversion: {m.conversion_rate:.1%}"
            ),
        )


class OutreachPlanner(BasePlanner):
    """Select which leads to contact and via which channel.

    Prioritization logic:
    1. Hot leads (recently engaged, high ICP) get priority
    2. Channel selection based on past response patterns
    3. Spread across channels to avoid email fatigue
    4. Follow-up timing based on last contact date
    """

    SUBJECTS = {
        "email": [
            "Quick question about {company}",
            "Idea for {company}'s {title} team",
            "{contact_name}, saw your recent work",
        ],
        "call": [
            "Discovery call with {contact_name}",
            "Follow-up on {company} discussion",
        ],
        "linkedin": [
            "Connection request for {contact_name}",
            "Shared insights for {company}",
        ],
    }

    BODIES = {
        "email": (
            "Hi {contact_name}, I noticed {company} is scaling "
            "and wanted to share how we help teams like yours..."
        ),
        "call": (
            "Call to discuss how we can help {company} "
            "achieve their goals in the {title} area."
        ),
        "linkedin": (
            "Hi {contact_name}, I'd love to connect and share "
            "some insights relevant to {company}."
        ),
    }

    def __init__(
        self,
        leads: list[Lead],
        metrics: PipelineMetrics,
        max_actions_per_step: int = 5,
    ) -> None:
        self._leads = leads
        self._metrics = metrics
        self._max_actions = max_actions_per_step

    def plan(self, goal: Goal, state: StateSnapshot) -> PolicySpec:
        # Prioritize leads
        scored_leads = []
        now = datetime.now()

        for lead in self._leads:
            if lead.status in ("meeting_booked", "disqualified"):
                continue

            priority = lead.icp_score

            # Boost engaged/qualified leads
            if lead.status == "qualified":
                priority *= 2.0
            elif lead.status == "engaged":
                priority *= 1.5

            # Recency penalty: don't contact too soon
            if lead.last_contact:
                days = (now - lead.last_contact).days
                if days < 2:
                    continue  # Too soon
                if days > 7:
                    priority *= 1.2  # Due for follow-up

            # Fatigue penalty
            if lead.contact_count > 5:
                continue  # Give up gracefully

            scored_leads.append((priority, lead))

        # Sort by priority (highest first)
        scored_leads.sort(key=lambda x: x[0], reverse=True)

        actions: list[ActionSpec] = []
        for _, lead in scored_leads[: self._max_actions]:
            channel = self._select_channel(lead)
            outreach = self._create_outreach(lead, channel)

            actions.append(
                ActionSpec(
                    name=f"outreach_{channel}_{lead.id}",
                    parameters={
                        "lead_id": lead.id,
                        "channel": channel,
                        "subject": outreach.subject,
                        "body": outreach.body,
                        "action": "outreach",
                        "lead_company": lead.company,
                        "lead_status": lead.status,
                    },
                    cost=1.0 if channel == "email" else 2.0,
                )
            )

        if not actions:
            actions.append(
                ActionSpec(name="wait", parameters={"action": "wait"}, cost=0.0)
            )

        return PolicySpec(actions=tuple(actions))

    def _select_channel(self, lead: Lead) -> str:
        """Pick the best channel for this lead."""
        # If we've never contacted: start with email
        if lead.contact_count == 0:
            return "email"

        # If lead responded to a specific channel, use it again
        if lead.response_count > 0 and lead.channel_history:
            # Find the channel that got responses (heuristic: last one)
            return lead.channel_history[-1]

        # Alternate channels to avoid fatigue
        used = set(lead.channel_history[-2:]) if lead.channel_history else set()
        for ch in ["email", "call", "linkedin"]:
            if ch not in used:
                return ch

        return "email"

    def _create_outreach(self, lead: Lead, channel: str) -> OutreachAction:
        """Create a personalized outreach action."""
        import random

        subjects = self.SUBJECTS[channel]
        subject_template = random.choice(subjects)
        subject = subject_template.format(
            company=lead.company,
            contact_name=lead.contact_name,
            title=lead.title,
        )
        body = self.BODIES[channel].format(
            company=lead.company,
            contact_name=lead.contact_name,
            title=lead.title,
        )

        return OutreachAction(
            lead_id=lead.id,
            channel=channel,
            subject=subject,
            body=body,
        )


class ContactFrequencyChecker(BaseConstraintChecker):
    """Max 2 contacts per lead per week, min 2-day gap between contacts."""

    def __init__(
        self,
        leads: list[Lead],
        max_contacts_per_week: int = 2,
        min_gap_days: int = 2,
    ) -> None:
        self._leads_map = {lead.id: lead for lead in leads}
        self._max_per_week = max_contacts_per_week
        self._min_gap_days = min_gap_days

    def check(
        self,
        goal: Goal,
        state: StateSnapshot,
        action: ActionSpec | None = None,
    ) -> tuple[bool, str]:
        if action is None or action.parameters.get("action") != "outreach":
            return (True, "")

        lead_id = action.parameters.get("lead_id", "")
        lead = self._leads_map.get(lead_id)
        if lead is None:
            return (True, "")

        # Check contact gap
        if lead.last_contact:
            days_since = (datetime.now() - lead.last_contact).days
            if days_since < self._min_gap_days:
                return (
                    False,
                    f"Lead {lead_id}: only {days_since}d since last contact "
                    f"(min {self._min_gap_days}d)",
                )

        # Approximate weekly frequency check
        if lead.contact_count >= self._max_per_week * 4:  # ~month limit
            return (
                False,
                f"Lead {lead_id}: {lead.contact_count} total contacts, "
                f"approaching fatigue threshold",
            )

        return (True, "")


class DailyLimitChecker(BaseConstraintChecker):
    """Max 50 emails/day, max 20 calls/day, max 30 linkedin/day."""

    def __init__(
        self,
        metrics: PipelineMetrics,
        max_emails: int = 50,
        max_calls: int = 20,
        max_linkedin: int = 30,
    ) -> None:
        self._metrics = metrics
        self._max_emails = max_emails
        self._max_calls = max_calls
        self._max_linkedin = max_linkedin

    def check(
        self,
        goal: Goal,
        state: StateSnapshot,
        action: ActionSpec | None = None,
    ) -> tuple[bool, str]:
        if action is None or action.parameters.get("action") != "outreach":
            return (True, "")

        channel = action.parameters.get("channel", "")

        if channel == "email" and self._metrics.emails_sent_today >= self._max_emails:
            return (False, f"Daily email limit reached ({self._max_emails})")
        if channel == "call" and self._metrics.calls_made_today >= self._max_calls:
            return (False, f"Daily call limit reached ({self._max_calls})")
        if channel == "linkedin" and self._metrics.linkedin_sent_today >= self._max_linkedin:
            return (False, f"Daily LinkedIn limit reached ({self._max_linkedin})")

        return (True, "")
