"""Typed shared state (state_schema), the triage verdict (response_schema — which is also what lets
model.cascade apply to triage_agent), and the postmortem timeline a spawned agent returns
(spawns.result_schema)."""

from typing import Literal

from pydantic import BaseModel, Field


class IncidentState(BaseModel):
    """Everything the agents share about the incident in progress. Every field is optional or
    defaulted so a brand-new session (nothing written yet) still validates."""

    alert_id: str | None = None
    severity: str | None = None
    blast_radius: int | None = None
    affected_services: list[str] = []
    suspected_root_cause: str | None = None
    root_cause_confidence: float | None = None
    investigation_notes: list[str] = []
    remediation_status: str | None = None
    postmortem_drafted: bool = False


class TriageVerdict(BaseModel):
    """triage_agent's final answer for an alert that doesn't route straight to the commander."""

    alert_id: str
    severity: Literal["sev1", "sev2", "sev3", "sev4"]
    blast_radius: int = Field(ge=0)
    affected_services: list[str]
    summary: str


class TimelineEvent(BaseModel):
    at: str
    event: str


class PostmortemTimeline(BaseModel):
    """What postmortem_agent's spawned timeline writer hands back via return_to_creator."""

    events: list[TimelineEvent]
    customer_impact: str
    detection_gap: str
    action_items: list[str] = []
