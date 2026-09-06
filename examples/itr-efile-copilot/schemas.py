"""Typed shared state (state_schema) and structured final output (filing_agent.response_schema)
for the ITR Filing Copilot."""

from pydantic import BaseModel


class TaxpayerState(BaseModel):
    """Everything downstream agents need about the return currently being prepared. Every field
    is optional/defaulted so a brand-new session (nothing written yet) still validates. Aadhaar
    and bank account details are deliberately NOT here — see tools/kyc_store.py.

    Field names past `age_band` mirror the official ITR-1 AY 2026-27 JSON schema's own PersonalInfo
    structure (fetched and verified 2026-09-05 — see generate_itr_json) rather than a looser
    "full_name"/"address" shape, so nothing gets lost or has to be re-split at generation time."""

    pan: str | None = None
    first_name: str | None = None
    middle_name: str = ""
    last_name: str | None = None
    father_name: str | None = None
    date_of_birth: str | None = None
    assessment_year: str | None = None  # e.g. "2026" (AY 2026-27) — the schema's own literal value
    age_band: str | None = None  # 'below_60' | '60_to_80' | 'above_80'
    regime_preference: str | None = None  # 'old' | 'new' | 'compare'

    employer_category: str | None = None  # CGOV|SGOV|PSU|PE|PESG|PEPS|PEO|OTH|NA (see resolve_employer_category)
    address_building: str | None = None
    address_locality: str | None = None
    address_city: str | None = None
    address_state_code: str | None = None  # 2-digit ITR state code (see resolve_state_code) — NOT the GST code
    address_pincode: str | None = None
    email: str | None = None
    mobile_no: str | None = None

    gross_salary: float | None = None
    exempt_allowances: float = 0.0
    house_property_income: float = 0.0
    other_sources_income: float = 0.0

    deduction_80c: float = 0.0
    deduction_80d: float = 0.0
    deduction_80tta_ttb: float = 0.0
    chapter_via_deductions: float = 0.0  # sum of the three above, via sum_deductions — never LLM-added

    tds_credit: float = 0.0

    filing_notes: list[str] = []


class FilingConfirmation(BaseModel):
    """filing_agent's final structured answer once a return has been generated (or explains why
    it couldn't be, if generation failed)."""

    status: str  # 'generated' | 'blocked'
    regime_used: str | None = None
    total_income: float | None = None
    total_tax_liability: float | None = None
    net_payable_or_refund: float | None = None
    output_path: str | None = None
    summary: str
