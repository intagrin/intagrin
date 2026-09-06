"""ITR-1 filing tools: PAN/Form 26AS lookups, statutory deduction caps, and final JSON generation.

SCHEMA FIDELITY (see blueprint.md for the full disclaimer): generate_itr_json's structure was
rebuilt against the Income Tax Department's own official AY 2026-27 ITR-1 JSON schema — fetched
directly from incometax.gov.in and verified via web search on 2026-09-05 — not a guessed
approximation. Every required top-level section (CreationInfo, Form_ITR1, PersonalInfo,
FilingStatus, ITR1_IncomeDeductions, ITR1_TaxComputation, TaxPaid, Refund, Verification) and their
required field names/nesting are reproduced faithfully. What's still NOT byte-exact, and known to
be so:

- CreationInfo.SWCreatedBy/JSONCreatedBy use a placeholder "SW00000000" — the real schema expects
  a software code issued by the Income Tax Department to a *registered* ERI (e-Return
  Intermediary). Generating a structurally valid JSON does not by itself make this tool a
  registered filing intermediary; this is a real, confirmed gap, not a formality.
- Refund.BankAccountDtls.AddtnlBankDetails' own exact sub-fields weren't retrieved at full depth —
  the shape below is a reasonable reconstruction (IFSCCode, BankAccountNo, ChooseRefundAccount),
  not independently confirmed against the schema file itself.
- ITR1_TaxComputation.TotalIntrstPay (interest under Sections 234A/B/C for late filing or short
  advance tax) is not computed by this demo and is always 0 — a real return with either condition
  needs this filled in correctly.
- ReturnFileSec is always 11 (original return, filed before the due date) — revised/belated return
  codes aren't handled.

Reconcile against the current official schema file directly, and have a qualified professional
review the return, before uploading anything this generates to a real e-filing portal.
"""

import json
import re
from datetime import date
from pathlib import Path

from intagrin.errors import AwaitingHumanInput

from . import kyc_store

_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

_PAN_RE = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$")
_IFSC_RE = re.compile(r"^[A-Z]{4}0[A-Z0-9]{6}$")

# Mock Form 26AS (the taxpayer's official TDS/TCS credit statement) fixture data, keyed by PAN, so
# this example runs with no external credentials. Swap for a real Form 26AS/AIS API integration
# when adapting this to production.
_FORM_26AS_FIXTURE: dict[str, dict] = {
    "ABCDE1234F": {"tds_on_salary": 42_000.0, "tds_on_interest": 1_500.0},
    "PQRSX5678K": {"tds_on_salary": 18_500.0, "tds_on_interest": 0.0},
}

# generate_itr_json's runtime AwaitingHumanInput threshold — a return claiming a refund above this
# amount gets a mandatory second look before the JSON is finalized, on top of the tool's own
# static requires_approval: true (every filing is reviewed; a large refund claim specifically is
# also flagged with why, mirroring docs/07_Human_In_The_Loop.md's dynamic-approval pattern).
_LARGE_REFUND_REVIEW_THRESHOLD = 50_000.0

# ITR-specific state codes — deliberately NOT the same numbering as GST state codes (e.g. Delhi is
# 09 here, 07 under GST; Maharashtra is 19 here, 27 under GST). Sourced from the Income Tax
# Department's own ITR schema documentation, verified via web search on 2026-09-05. Only the most
# populous states are listed; resolve_state_code fails closed (asks for the code directly) for
# anything not in this table rather than guessing.
_STATE_CODES: dict[str, str] = {
    "andhra pradesh": "02", "arunachal pradesh": "03", "assam": "04", "bihar": "05",
    "chandigarh": "06", "chhattisgarh": "33", "delhi": "09", "goa": "10", "gujarat": "11",
    "haryana": "12", "himachal pradesh": "13", "jammu and kashmir": "14", "jharkhand": "35",
    "karnataka": "15", "kerala": "16", "madhya pradesh": "18", "maharashtra": "19",
    "odisha": "24", "orissa": "24", "punjab": "26", "rajasthan": "27", "tamil nadu": "29",
    "telangana": "36", "uttar pradesh": "31", "uttarakhand": "34", "west bengal": "32",
}

# EmployerCategory enum — CGOV/SGOV/PSU/PE/PESG/PEPS/PEO/OTH/NA (Central Govt, State Govt, Public
# Sector Undertaking, Pensioners-CGOV/SGOV/PSU, Pensioners-Others, OTHer, Not Applicable).
_EMPLOYER_KEYWORDS = [
    (("central government", "cgov", "central govt"), "CGOV"),
    (("state government", "sgov", "state govt"), "SGOV"),
    (("public sector", "psu"), "PSU"),
    (("pension", "retired"), "PEO"),
    (("private", "self-employed", "self employed", "freelance"), "OTH"),
]


def validate_pan(pan: str) -> str:
    """Validates an Indian PAN's format (5 letters, 4 digits, 1 letter — e.g. ABCDE1234F).

    Args:
        pan: The PAN to validate.
    """
    pan = pan.strip().upper()
    if not _PAN_RE.match(pan):
        return f"'{pan}' is not a valid PAN format. Expected 5 letters, 4 digits, 1 letter (e.g. ABCDE1234F)."
    return f"'{pan}' is a validly formatted PAN."


def lookup_form26as_summary(pan: str) -> str:
    """Fetches the taxpayer's TDS/TCS credit summary from Form 26AS for the given PAN.

    Args:
        pan: The taxpayer's PAN.
    """
    record = _FORM_26AS_FIXTURE.get(pan.strip().upper())
    if record is None:
        return f"No Form 26AS record found for PAN '{pan}'. Ask the taxpayer for their TDS figures directly, or confirm the PAN is correct."
    total = sum(record.values())
    return json.dumps({"pan": pan.strip().upper(), **record, "total_tds_credit": total}, indent=2)


def check_deduction_limit(section: str, claimed_amount: float, is_senior_citizen: bool = False) -> str:
    """Checks a claimed Chapter VI-A deduction against its statutory cap and returns the allowed
    amount — never accept a claimed figure without checking it against this first.

    Args:
        section: The deduction section (e.g. '80C', '80D', '80TTA', '80TTB').
        claimed_amount: The amount the taxpayer wants to claim under this section.
        is_senior_citizen: Relevant for 80D/80TTA-vs-80TTB — whether the taxpayer is 60 or older.
    """
    caps = {
        "80C": 150_000.0,
        "80D": 50_000.0 if is_senior_citizen else 25_000.0,
        "80TTA": 10_000.0,
        "80TTB": 50_000.0,
    }
    cap = caps.get(section.upper())
    if cap is None:
        return f"Unknown section '{section}'. Known sections: {', '.join(caps)}."
    if section.upper() == "80TTA" and is_senior_citizen:
        return "Senior citizens claim savings/deposit interest under 80TTB (cap ₹50,000), not 80TTA."
    allowed = min(claimed_amount, cap)
    if allowed < claimed_amount:
        return f"Section {section.upper()} caps at ₹{cap:,.0f} — claimed ₹{claimed_amount:,.0f}, allowed ₹{allowed:,.0f}."
    return f"Section {section.upper()}: ₹{allowed:,.0f} is within the ₹{cap:,.0f} cap — fully allowed."


def sum_deductions(deduction_80c: float, deduction_80d: float, deduction_80tta_ttb: float) -> str:
    """Sums the three Chapter VI-A deduction amounts into the single total compute_tax_liability
    expects. Even a plain sum goes through a tool, never the LLM's own arithmetic.

    Args:
        deduction_80c: Allowed (post-cap) 80C amount from check_deduction_limit.
        deduction_80d: Allowed (post-cap) 80D amount from check_deduction_limit.
        deduction_80tta_ttb: Allowed (post-cap) 80TTA/80TTB amount from check_deduction_limit.
    """
    return str(deduction_80c + deduction_80d + deduction_80tta_ttb)


def resolve_state_code(state_name: str) -> str:
    """Resolves a state/UT name to its 2-digit ITR schema state code — NOT the same numbering as
    GST state codes. Only covers commonly-filed states; returns an error for anything else rather
    than guessing.

    Args:
        state_name: The state or union territory name (e.g. 'Maharashtra', 'Delhi').
    """
    code = _STATE_CODES.get(state_name.strip().lower())
    if code is None:
        return (
            f"'{state_name}' isn't in this demo's state code table. Ask the taxpayer for the "
            "2-digit ITR state code directly, or extend _STATE_CODES."
        )
    return code


def resolve_employer_category(description: str) -> str:
    """Maps a free-text employer description to the ITR schema's EmployerCategory enum
    (CGOV/SGOV/PSU/PE/PESG/PEPS/PEO/OTH/NA). Defaults to 'OTH' for anything unrecognized rather
    than guessing a more specific category.

    Args:
        description: How the taxpayer described their employer (e.g. 'private company', 'central government', 'retired').
    """
    text = description.strip().lower()
    for keywords, category in _EMPLOYER_KEYWORDS:
        if any(kw in text for kw in keywords):
            return category
    return "OTH"


def generate_itr_json(
    pan: str,
    first_name: str,
    middle_name: str,
    last_name: str,
    father_name: str,
    date_of_birth: str,
    assessment_year: str,
    employer_category: str,
    address_building: str,
    address_locality: str,
    address_city: str,
    address_state_code: str,
    address_pincode: str,
    email: str,
    mobile_no: str,
    regime_used: str,
    gross_salary: float,
    exempt_allowances: float,
    standard_deduction: float,
    house_property_income: float,
    other_sources_income: float,
    gross_total_income: float,
    total_deductions: float,
    total_income: float,
    tax_before_cess: float,
    rebate_87a: float,
    cess: float,
    total_tax_liability: float,
    tds_credit: float,
    net_payable_or_refund: float,
) -> str:
    """Generates the final ITR-1 JSON file for upload, after validating internal arithmetic
    consistency and that the taxpayer's refund bank details are on file. Requires human approval
    (ai.yaml's requires_approval: true) — this is the point-of-no-return step for this session.
    Structure matches the official AY 2026-27 ITR-1 JSON schema's required sections — see this
    module's own docstring for the specific, confirmed gaps that remain.

    Args:
        pan: The taxpayer's PAN (validated via validate_pan).
        first_name: First name as per PAN records.
        middle_name: Middle name as per PAN records (empty string if none).
        last_name: Surname as per PAN records.
        father_name: Father's name, required by the Verification section.
        date_of_birth: Date of birth, YYYY-MM-DD.
        assessment_year: The assessment year's schema literal (e.g. '2026' for AY 2026-27).
        employer_category: From resolve_employer_category.
        address_building: House/flat/building name or number.
        address_locality: Locality or area.
        address_city: City or town.
        address_state_code: 2-digit ITR state code from resolve_state_code.
        address_pincode: 6-digit PIN code.
        email: Contact email.
        mobile_no: 10-digit mobile number.
        regime_used: 'old' or 'new' — whichever regime was actually selected for filing.
        gross_salary: From compute_tax_liability's input for the selected regime.
        exempt_allowances: From compute_tax_liability's input.
        standard_deduction: 50000 for old regime, 75000 for new — see tools.tax_engine's constants.
        house_property_income: From compute_tax_liability's input.
        other_sources_income: From compute_tax_liability's input.
        gross_total_income: From compute_tax_liability's output for the selected regime.
        total_deductions: Chapter VI-A deductions actually applied (0 under the new regime).
        total_income: From compute_tax_liability's output.
        tax_before_cess: From compute_tax_liability's output.
        rebate_87a: From compute_tax_liability's output.
        cess: From compute_tax_liability's output (health_and_education_cess).
        total_tax_liability: From compute_tax_liability's output.
        tds_credit: From compute_tax_liability's output.
        net_payable_or_refund: From compute_tax_liability's output (negative means refund due).
    """
    pan = pan.strip().upper()
    if not _PAN_RE.match(pan):
        return f"Cannot generate: '{pan}' is not a valid PAN format."

    # Arithmetic reconciliation — never write a JSON whose own sections don't add up, regardless
    # of what the calling agent claims each field is. A $1 tolerance absorbs floating-point noise
    # from the upstream compute_tax_liability call, not real discrepancies.
    if abs((gross_total_income - total_deductions) - total_income) > 1:
        return (
            f"Cannot generate: gross_total_income ({gross_total_income}) - total_deductions "
            f"({total_deductions}) does not equal total_income ({total_income}). Re-run "
            "compute_tax_liability and use its own output values directly, don't restate them."
        )
    if abs((tax_before_cess - rebate_87a + cess) - total_tax_liability) > 1:
        return (
            f"Cannot generate: tax_before_cess ({tax_before_cess}) - rebate_87a ({rebate_87a}) + "
            f"cess ({cess}) does not equal total_tax_liability ({total_tax_liability})."
        )
    if abs((total_tax_liability - tds_credit) - net_payable_or_refund) > 1:
        return (
            f"Cannot generate: total_tax_liability ({total_tax_liability}) - tds_credit "
            f"({tds_credit}) does not equal net_payable_or_refund ({net_payable_or_refund})."
        )

    kyc = kyc_store.get_kyc()
    if not kyc or not kyc.get("confirmed"):
        return (
            "Cannot generate: no refund bank account on file. Ask the taxpayer to complete the "
            "KYC form (ui/server.py) with their bank account details before filing — do not ask "
            "for these in chat."
        )

    refund_due = -net_payable_or_refund if net_payable_or_refund < 0 else 0.0
    bal_tax_payable = net_payable_or_refund if net_payable_or_refund >= 0 else 0.0
    if refund_due > _LARGE_REFUND_REVIEW_THRESHOLD:
        raise AwaitingHumanInput(
            prompt=(
                f"This return claims a refund of ₹{refund_due:,.2f} for PAN {pan}, above the "
                f"₹{_LARGE_REFUND_REVIEW_THRESHOLD:,.0f} threshold for automatic review — please "
                "double-check the computation before this is finalized."
            ),
            context={"pan": pan, "refund_due": refund_due, "assessment_year": assessment_year},
        )

    income_from_salary = gross_salary - exempt_allowances - standard_deduction
    full_name = " ".join(p for p in (first_name, middle_name, last_name) if p)

    itr_json = {
        "ITR": {
            "ITR1": {
                "CreationInfo": {
                    "SWVersionNo": "1.0",
                    # Placeholder — see this module's docstring: a real integration needs an
                    # actual ERI-registered software code from the Income Tax Department here.
                    "SWCreatedBy": "SW00000000",
                    "JSONCreatedBy": "SW00000000",
                    "JSONCreationDate": date.today().isoformat(),
                    "IntermediaryCity": address_city,
                    "Digest": "-",
                },
                "Form_ITR1": {
                    "FormName": "ITR-1",
                    "Description": "For Individuals having Income from Salaries, One House Property, Other Sources",
                    "AssessmentYear": assessment_year,
                    "SchemaVer": "Ver1.0",
                    "FormVer": "Ver1.0",
                },
                "PersonalInfo": {
                    "AssesseeName": {
                        "FirstName": first_name,
                        "MiddleName": middle_name,
                        "SurNameOrOrgName": last_name,
                    },
                    "PAN": pan,
                    "Address": {
                        "ResidenceNo": address_building,
                        "LocalityOrArea": address_locality,
                        "CityOrTownOrDistrict": address_city,
                        "StateCode": address_state_code,
                        "CountryCode": "91",
                        "PinCode": address_pincode,
                        "CountryCodeMobile": 91,
                        "MobileNo": mobile_no,
                        "EmailAddress": email,
                    },
                    "SecondaryAdd": "N",
                    "DOB": date_of_birth,
                    "EmployerCategory": employer_category,
                    # AadhaarCardNo deliberately omitted — it's optional in the official schema,
                    # and this project never collects or stores Aadhaar at all (see
                    # tools/kyc_store.py's docstring for why).
                },
                "FilingStatus": {
                    "ReturnFileSec": 11,
                    "OptOutNewTaxRegime": "Y" if regime_used == "old" else "N",
                    "AsseseeRepFlg": "N",
                    "ItrFilingDueDate": f"{assessment_year}-07-31",
                },
                "ITR1_IncomeDeductions": {
                    "GrossSalary": gross_salary,
                    "NetSalary": gross_salary - exempt_allowances,
                    "DeductionUs16": standard_deduction,
                    "IncomeFromSal": income_from_salary,
                    "TotalIncomeChargeableUnHP": house_property_income,
                    "IncomeOthSrc": other_sources_income,
                    "GrossTotIncome": gross_total_income,
                    "GrossTotIncomeIncLTCG112A": gross_total_income,
                    "DeductUndChapVIA": {"TotalChapVIADeductions": total_deductions},
                    "TotalIncome": total_income,
                },
                "ITR1_TaxComputation": {
                    "TotalTaxPayable": tax_before_cess,
                    "Rebate87A": rebate_87a,
                    "TaxPayableOnRebate": tax_before_cess - rebate_87a,
                    "EducationCess": cess,
                    "GrossTaxLiability": (tax_before_cess - rebate_87a) + cess,
                    "Section89": 0,
                    "NetTaxLiability": total_tax_liability,
                    "TotalIntrstPay": 0,
                    "TotTaxPlusIntrstPay": total_tax_liability,
                },
                "TaxPaid": {
                    "TaxesPaid": {
                        "AdvanceTax": 0,
                        "TDS": tds_credit,
                        "TCS": 0,
                        "SelfAssessmentTax": 0,
                        "TotalTaxesPaid": tds_credit,
                    },
                    "BalTaxPayable": bal_tax_payable,
                },
                "Refund": {
                    "RefundDue": refund_due,
                    "BankAccountDtls": {
                        "AddtnlBankDetails": [
                            {
                                "IFSCCode": kyc["ifsc"],
                                "BankAccountNo": kyc["bank_account_number"],
                                "ChooseRefundAccount": "Y",
                            }
                        ]
                    },
                },
                "Verification": {
                    "Declaration": {
                        "AssesseeVerName": full_name,
                        "FatherName": father_name,
                        "AssesseeVerPAN": pan,
                    },
                    "Capacity": "S",
                    "Place": address_city,
                },
            }
        }
    }

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = _OUTPUT_DIR / f"ITR1_{pan}_{assessment_year}.json"
    output_path.write_text(json.dumps(itr_json, indent=2))

    outcome = "refund" if net_payable_or_refund < 0 else "payable"
    return (
        "⚠️ Demonstration output — structurally aligned with the official AY 2026-27 ITR-1 JSON "
        "schema, but see this tool's module docstring for confirmed remaining gaps (software "
        "registration code, 234A/B/C interest, bank-details sub-schema). Reconcile against the "
        "official schema file and have a qualified professional review it before uploading "
        "anywhere.\n\n"
        f"ITR-1 JSON generated at: {output_path}\n"
        f"Regime: {regime_used} | Total income: ₹{total_income:,.2f} | "
        f"Total tax liability: ₹{total_tax_liability:,.2f} | "
        f"{'Refund due' if outcome == 'refund' else 'Net payable'}: "
        f"₹{abs(net_payable_or_refund):,.2f}"
    )
