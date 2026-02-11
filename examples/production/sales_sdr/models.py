"""Domain model for the Sales SDR agent."""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Lead:
    """A sales lead in the pipeline."""

    id: str
    company: str
    contact_name: str
    email: str
    title: str
    icp_score: float  # Ideal Customer Profile fit 0-1
    status: str = "new"  # new | contacted | engaged | qualified | meeting_booked | disqualified
    last_contact: datetime | None = None
    contact_count: int = 0
    response_count: int = 0
    channel_history: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass(frozen=True)
class OutreachAction:
    """A single outreach action to perform on a lead."""

    lead_id: str
    channel: str  # email | call | linkedin
    subject: str
    body: str
    personalization: dict[str, str] = field(default_factory=dict)


@dataclass
class PipelineMetrics:
    """Aggregated metrics for the sales pipeline."""

    total_leads: int = 0
    contacted: int = 0
    engaged: int = 0
    qualified: int = 0
    meetings_booked: int = 0
    disqualified: int = 0
    emails_sent_today: int = 0
    calls_made_today: int = 0
    linkedin_sent_today: int = 0

    @property
    def conversion_rate(self) -> float:
        if self.contacted == 0:
            return 0.0
        return self.meetings_booked / self.contacted

    @property
    def engagement_rate(self) -> float:
        if self.contacted == 0:
            return 0.0
        return self.engaged / self.contacted

    @property
    def qualification_rate(self) -> float:
        if self.contacted == 0:
            return 0.0
        return self.qualified / self.contacted
