"""Typed shared state (state_schema) and structured spawn result (spawns.result_schema) for the
Support Desk Copilot."""

from pydantic import BaseModel


class TicketState(BaseModel):
    """Everything downstream agents need to see about the ticket currently in progress. Every
    field is optional/defaulted so a brand-new session (nothing written yet) still validates."""

    customer_id: str | None = None
    customer_tier: str | None = None
    ticket_summary: str | None = None
    escalation_notes: list[str] = []
    todays_review_subject: str | None = None
    resolved: bool = False


class IncidentResolution(BaseModel):
    """What a spawned enterprise incident specialist (priority_support_agent.spawns) hands back
    via return_to_creator — validated server-side instead of hand-parsed from free text."""

    root_cause: str
    affected_services: list[str] = []
    fix_applied: str
    follow_up_required: bool = False
    follow_up_notes: str = ""
