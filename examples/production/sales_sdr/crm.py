"""Data layer: simulated CRM + optional real API fallback."""

import os
import random
from abc import ABC, abstractmethod
from datetime import datetime, timedelta

from .models import Lead, OutreachAction


class CRMProvider(ABC):
    """Abstract interface for CRM operations."""

    @abstractmethod
    def get_leads(self) -> list[Lead]:
        """Fetch all leads in the pipeline."""

    @abstractmethod
    def update_lead(self, lead_id: str, updates: dict) -> Lead:
        """Update a lead with new data."""

    @abstractmethod
    def log_activity(self, lead_id: str, action: OutreachAction) -> None:
        """Log an outreach activity against a lead."""


class SimulatedCRM(CRMProvider):
    """In-memory CRM with realistic lead data.

    Generates N leads with varied ICP scores, statuses, contact histories.
    Simulates response probability based on:
    - ICP score (higher = more likely to respond)
    - Channel preference (some leads prefer email, others calls)
    - Contact fatigue (diminishing returns after 3+ contacts)
    - Time since last contact (recency effect)
    """

    COMPANIES = [
        ("Acme Corp", "John Smith", "VP Engineering"),
        ("TechFlow Inc", "Sarah Chen", "Head of Product"),
        ("DataVista", "Mike Johnson", "CTO"),
        ("CloudPeak", "Emily Zhang", "Director of Ops"),
        ("NexGen Labs", "David Kim", "VP Sales"),
        ("Quantum Sys", "Lisa Park", "CEO"),
        ("ByteScale", "Tom Brown", "Head of Growth"),
        ("PivotAI", "Anna White", "VP Marketing"),
        ("Hyperion Soft", "James Lee", "Director of Engineering"),
        ("Stellar Data", "Maria Garcia", "COO"),
        ("Forge Analytics", "Ryan Davis", "CRO"),
        ("Atlas Cloud", "Priya Patel", "VP Platform"),
        ("Nimbus Tech", "Chris Wilson", "Head of AI"),
        ("Horizon SaaS", "Diana Lopez", "VP Customer Success"),
        ("Cortex Labs", "Alex Turner", "Director of Innovation"),
        ("Meridian Sys", "Olivia Martin", "CTO"),
        ("Apex Digital", "Nick Brown", "VP Engineering"),
        ("Vertex AI", "Rachel Kim", "Head of Research"),
        ("Catalyst Co", "Dan Miller", "VP Strategy"),
        ("Synapse Tech", "Julia Adams", "CEO"),
        ("Radiant Soft", "Kevin Scott", "Director of Product"),
        ("Summit Data", "Helen Zhao", "VP Data"),
        ("Vector Labs", "Sam Taylor", "Head of Engineering"),
        ("Orbit Cloud", "Grace Liu", "CTO"),
        ("Beacon AI", "Mark Chen", "VP Technology"),
    ]

    CHANNEL_PREFERENCES = ["email", "call", "linkedin"]

    def __init__(self, num_leads: int = 25, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._leads: dict[str, Lead] = {}
        self._activity_log: list[dict] = []

        now = datetime.now()
        for i in range(min(num_leads, len(self.COMPANIES))):
            company, name, title = self.COMPANIES[i]
            lid = f"lead-{i:03d}"
            icp_score = round(self._rng.uniform(0.2, 1.0), 2)

            # Some leads have prior contact history
            status = "new"
            contact_count = 0
            response_count = 0
            last_contact = None
            channel_history: list[str] = []

            if self._rng.random() < 0.3:
                status = "contacted"
                contact_count = self._rng.randint(1, 3)
                channel_history = [
                    self._rng.choice(self.CHANNEL_PREFERENCES)
                    for _ in range(contact_count)
                ]
                last_contact = now - timedelta(days=self._rng.randint(1, 14))
                if self._rng.random() < icp_score * 0.5:
                    response_count = 1
                    status = "engaged"

            self._leads[lid] = Lead(
                id=lid,
                company=company,
                contact_name=name,
                email=f"{name.lower().replace(' ', '.')}@{company.lower().replace(' ', '')}.com",
                title=title,
                icp_score=icp_score,
                status=status,
                last_contact=last_contact,
                contact_count=contact_count,
                response_count=response_count,
                channel_history=channel_history,
            )

        # Each lead has a hidden channel preference
        self._lead_prefs: dict[str, str] = {
            lid: self._rng.choice(self.CHANNEL_PREFERENCES)
            for lid in self._leads
        }

    def get_leads(self) -> list[Lead]:
        return list(self._leads.values())

    def update_lead(self, lead_id: str, updates: dict) -> Lead:
        lead = self._leads[lead_id]
        for k, v in updates.items():
            if hasattr(lead, k):
                setattr(lead, k, v)
        return lead

    def log_activity(self, lead_id: str, action: OutreachAction) -> None:
        lead = self._leads[lead_id]
        lead.contact_count += 1
        lead.channel_history.append(action.channel)
        lead.last_contact = datetime.now()

        if lead.status == "new":
            lead.status = "contacted"

        self._activity_log.append({
            "lead_id": lead_id,
            "channel": action.channel,
            "subject": action.subject,
            "timestamp": datetime.now().isoformat(),
        })

        # Simulate response
        response_prob = self._compute_response_prob(lead, action.channel)
        if self._rng.random() < response_prob:
            lead.response_count += 1
            if lead.status == "contacted":
                lead.status = "engaged"
            elif lead.status == "engaged" and lead.response_count >= 2:
                lead.status = "qualified"
            elif lead.status == "qualified" and lead.response_count >= 3:
                lead.status = "meeting_booked"

    def _compute_response_prob(self, lead: Lead, channel: str) -> float:
        """Simulate realistic response probability."""
        base = lead.icp_score * 0.3

        # Channel preference bonus
        pref = self._lead_prefs.get(lead.id, "email")
        if channel == pref:
            base *= 1.5

        # Contact fatigue (diminishing returns)
        if lead.contact_count > 3:
            fatigue = 0.7 ** (lead.contact_count - 3)
            base *= fatigue

        # Recency effect: better if we waited at least 2 days
        if lead.last_contact:
            days_since = (datetime.now() - lead.last_contact).days
            if days_since < 2:
                base *= 0.3  # Too soon
            elif days_since > 7:
                base *= 0.8  # Getting cold

        return min(base, 0.8)


class HubSpotClient(CRMProvider):
    """Real HubSpot CRM API client.

    Requires ``HUBSPOT_API_KEY`` environment variable.
    """

    BASE_URL = "https://api.hubapi.com/crm/v3"

    def __init__(self) -> None:
        import httpx

        self._api_key = os.environ["HUBSPOT_API_KEY"]
        self._client = httpx.Client(
            base_url=self.BASE_URL,
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=15.0,
        )

    def get_leads(self) -> list[Lead]:
        resp = self._client.get(
            "/objects/contacts",
            params={"limit": 100, "properties": "email,firstname,lastname,company,jobtitle"},
        )
        resp.raise_for_status()
        leads = []
        for item in resp.json().get("results", []):
            props = item.get("properties", {})
            leads.append(Lead(
                id=str(item["id"]),
                company=props.get("company", "Unknown"),
                contact_name=f"{props.get('firstname', '')} {props.get('lastname', '')}".strip(),
                email=props.get("email", ""),
                title=props.get("jobtitle", ""),
                icp_score=0.5,  # Would need enrichment
                status="new",
            ))
        return leads

    def update_lead(self, lead_id: str, updates: dict) -> Lead:
        resp = self._client.patch(
            f"/objects/contacts/{lead_id}",
            json={"properties": updates},
        )
        resp.raise_for_status()
        # Return a stub; real implementation would parse response
        return Lead(id=lead_id, company="", contact_name="", email="", title="", icp_score=0.5)

    def log_activity(self, lead_id: str, action: OutreachAction) -> None:
        self._client.post(
            "/objects/notes",
            json={
                "properties": {
                    "hs_note_body": f"[{action.channel}] {action.subject}: {action.body}",
                    "hs_timestamp": datetime.now().isoformat(),
                },
                "associations": [{
                    "to": {"id": lead_id},
                    "types": [{"associationCategory": "HUBSPOT_DEFINED",
                               "associationTypeId": 202}],
                }],
            },
        )


def get_crm(num_leads: int = 25, seed: int | None = 42) -> CRMProvider:
    """Return real client if ``HUBSPOT_API_KEY`` is set, else simulated."""
    if os.environ.get("HUBSPOT_API_KEY"):
        return HubSpotClient()
    return SimulatedCRM(num_leads=num_leads, seed=seed)
