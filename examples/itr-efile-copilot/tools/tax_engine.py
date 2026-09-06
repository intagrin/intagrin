"""Deterministic Indian income tax computation for ITR-1 (Sahaj) — AY 2026-27 (FY 2025-26),
Old Regime vs. New Regime (Section 115BAC) as revised by the Finance Act, 2025 (the new-regime
slabs/rebate below were verified via web search on 2026-09-05 against the Income Tax Department's
own AY 2026-27 ITR-1 JSON schema — its Rebate87A field caps at exactly 60,000, which matches the
tax-before-cess this module computes at exactly ₹12,00,000 taxable income, corroborating both
figures independently).

IMPORTANT: slabs, standard deduction amounts, and rebate thresholds change almost every Union
Budget. This module hardcodes one specific assessment year's figures as a clearly-labeled
illustration of "tax math belongs in a deterministic Python tool, never in an LLM's own
arithmetic" — not as a maintained, always-current tax engine. Before relying on this for a real
filing, verify every figure below against the current Finance Act and update
_NEW_REGIME_SLABS/_OLD_REGIME_SLABS accordingly. Old-regime slabs are unchanged from AY 2025-26 —
no source found indicating the Finance Act, 2025 touched them.

Every function here is pure (no I/O, no LLM calls) and independently unit-testable — the agents
that call these tools never do tax arithmetic themselves; they collect inputs and report whatever
this module computes.
"""

STANDARD_DEDUCTION_OLD_REGIME = 50_000.0
STANDARD_DEDUCTION_NEW_REGIME = 75_000.0
HOUSE_PROPERTY_LOSS_SETOFF_CAP = 200_000.0
CESS_RATE = 0.04
NEW_REGIME_REBATE_THRESHOLD = 1_200_000.0
NEW_REGIME_REBATE_CAP = 60_000.0

# (upper_bound_exclusive, rate) — the last band's upper bound is None (no ceiling). Revised by the
# Finance Act, 2025, effective AY 2026-27 (FY 2025-26) — a 7-slab table replacing the prior 6-slab
# one, alongside the Section 87A rebate threshold moving from taxable income <= 7,00,000 to
# <= 12,00,000 (see NEW_REGIME_REBATE_THRESHOLD/_rebate_87a below).
_NEW_REGIME_SLABS = [
    (400_000, 0.0),
    (800_000, 0.05),
    (1_200_000, 0.10),
    (1_600_000, 0.15),
    (2_000_000, 0.20),
    (2_400_000, 0.25),
    (None, 0.30),
]

_OLD_REGIME_SLABS = {
    "below_60": [(250_000, 0.0), (500_000, 0.05), (1_000_000, 0.20), (None, 0.30)],
    "60_to_80": [(300_000, 0.0), (500_000, 0.05), (1_000_000, 0.20), (None, 0.30)],
    "above_80": [(500_000, 0.0), (1_000_000, 0.20), (None, 0.30)],
}


def _round_down_to_10(amount: float) -> float:
    """Section 288A: total income is rounded DOWN to the nearest multiple of 10."""
    return float(int(amount // 10) * 10)


def _round_to_nearest_10(amount: float) -> float:
    """Section 288B: tax payable/refundable is rounded to the nearest multiple of 10."""
    return float(round(amount / 10) * 10)


def _slab_tax(taxable_income: float, slabs: list[tuple[float | None, float]]) -> float:
    tax = 0.0
    lower = 0.0
    for upper, rate in slabs:
        if taxable_income <= lower:
            break
        band_top = taxable_income if upper is None else min(taxable_income, upper)
        if band_top > lower:
            tax += (band_top - lower) * rate
        lower = upper if upper is not None else taxable_income
    return tax


def compute_hra_exemption(
    basic_salary_plus_da: float, hra_received: float, rent_paid: float, is_metro_city: bool
) -> float:
    """Computes the exempt portion of House Rent Allowance under Section 10(13A) — the least of
    three amounts, per the standard HRA exemption rule.

    Args:
        basic_salary_plus_da: Annual basic salary plus dearness allowance (the base HRA exemption is computed against).
        hra_received: Annual HRA actually received from the employer.
        rent_paid: Annual rent actually paid by the employee.
        is_metro_city: True if residing in Mumbai, Delhi, Kolkata, or Chennai (50% limit); False otherwise (40% limit).
    """
    rent_over_10pct = max(0.0, rent_paid - 0.10 * basic_salary_plus_da)
    city_limit = (0.50 if is_metro_city else 0.40) * basic_salary_plus_da
    return max(0.0, min(hra_received, rent_over_10pct, city_limit))


def compute_house_property_income(
    is_self_occupied: bool, annual_rent_received: float, home_loan_interest: float
) -> float:
    """Computes net income (or loss) from the one house property ITR-1 permits, per Section 24.

    Args:
        is_self_occupied: True for a self-occupied property (no rental income; interest deduction capped at 200000). False for a let-out property.
        annual_rent_received: Annual rent received — 0 for a self-occupied property.
        home_loan_interest: Annual home loan interest paid, deductible under Section 24(b).
    """
    if is_self_occupied:
        return -min(home_loan_interest, HOUSE_PROPERTY_LOSS_SETOFF_CAP)
    standard_deduction = 0.30 * annual_rent_received
    net = annual_rent_received - standard_deduction - home_loan_interest
    return max(net, -HOUSE_PROPERTY_LOSS_SETOFF_CAP)


def _rebate_87a(taxable_income: float, tax_before_cess: float, regime: str) -> float:
    if regime == "new":
        if taxable_income <= NEW_REGIME_REBATE_THRESHOLD:
            return min(tax_before_cess, NEW_REGIME_REBATE_CAP)
        marginal_relief = tax_before_cess - (taxable_income - NEW_REGIME_REBATE_THRESHOLD)
        return max(0.0, min(marginal_relief, tax_before_cess))
    # Old regime: flat rebate up to 12,500 for taxable income <= 5,00,000. Deliberately does not
    # implement old-regime marginal relief for income just above 5,00,000 — a narrow, rarely
    # material edge case for a demo; document and add it before relying on this for real filings
    # near that threshold.
    if taxable_income <= 500_000:
        return min(12_500.0, tax_before_cess)
    return 0.0


def compute_tax_liability(
    regime: str,
    age_band: str,
    gross_salary: float,
    exempt_allowances: float,
    house_property_income: float,
    other_sources_income: float,
    chapter_via_deductions: float,
    tds_credit: float,
) -> str:
    """Computes the full ITR-1 tax liability breakdown for one regime. Returns a JSON string —
    never do this arithmetic yourself; always call this tool and report exactly what it returns.

    Args:
        regime: 'old' or 'new' — which regime to compute under.
        age_band: 'below_60', '60_to_80', or 'above_80' (irrelevant under the new regime, which has one slab set for all ages).
        gross_salary: Total gross salary from Form 16, before any exemptions/deductions.
        exempt_allowances: Total exempt allowances (e.g. HRA exemption from compute_hra_exemption, LTA) to subtract from gross salary.
        house_property_income: Net income/loss from house property (from compute_house_property_income) — 0 if none.
        other_sources_income: Interest, dividends, and other taxable income from sources other than salary/house property.
        chapter_via_deductions: Total Chapter VI-A deductions (80C + 80D + 80TTA/80TTB + others). Ignored under the new regime, which disallows nearly all of them.
        tds_credit: Total TDS/TCS credit from Form 26AS, to offset against the computed tax liability.
    """
    import json

    if regime not in ("old", "new"):
        return json.dumps({"error": f"Unknown regime '{regime}' — must be 'old' or 'new'."})

    standard_deduction = (
        STANDARD_DEDUCTION_OLD_REGIME if regime == "old" else STANDARD_DEDUCTION_NEW_REGIME
    )
    income_from_salary = max(0.0, gross_salary - exempt_allowances - standard_deduction)
    gross_total_income = income_from_salary + house_property_income + other_sources_income

    deductions_applied = chapter_via_deductions if regime == "old" else 0.0
    total_income = max(0.0, gross_total_income - deductions_applied)
    total_income = _round_down_to_10(total_income)

    slabs = _NEW_REGIME_SLABS if regime == "new" else _OLD_REGIME_SLABS[age_band]
    tax_before_cess = _slab_tax(total_income, slabs)
    rebate = _rebate_87a(total_income, tax_before_cess, regime)
    tax_after_rebate = tax_before_cess - rebate
    cess = tax_after_rebate * CESS_RATE
    total_tax_liability = _round_to_nearest_10(tax_after_rebate + cess)

    net_payable_or_refund = _round_to_nearest_10(total_tax_liability - tds_credit)

    return json.dumps(
        {
            "regime": regime,
            "income_from_salary": income_from_salary,
            "house_property_income": house_property_income,
            "other_sources_income": other_sources_income,
            "gross_total_income": gross_total_income,
            "chapter_via_deductions_applied": deductions_applied,
            "total_income": total_income,
            "tax_before_cess": round(tax_before_cess, 2),
            "rebate_87a": round(rebate, 2),
            "health_and_education_cess": round(cess, 2),
            "total_tax_liability": total_tax_liability,
            "tds_credit": tds_credit,
            "net_payable_or_refund": net_payable_or_refund,
            "outcome": "refund_due" if net_payable_or_refund < 0 else "payable",
        },
        indent=2,
    )


def compare_tax_regimes(
    age_band: str,
    gross_salary: float,
    exempt_allowances: float,
    house_property_income: float,
    other_sources_income: float,
    chapter_via_deductions: float,
    tds_credit: float,
) -> str:
    """Computes tax liability under BOTH regimes side by side and recommends whichever produces a
    lower total tax liability — always call this instead of guessing which regime is better.

    Args: same as compute_tax_liability, minus `regime` (both are computed).
    """
    import json

    old = json.loads(
        compute_tax_liability(
            "old",
            age_band,
            gross_salary,
            exempt_allowances,
            house_property_income,
            other_sources_income,
            chapter_via_deductions,
            tds_credit,
        )
    )
    new = json.loads(
        compute_tax_liability(
            "new",
            age_band,
            gross_salary,
            exempt_allowances,
            house_property_income,
            other_sources_income,
            chapter_via_deductions,
            tds_credit,
        )
    )
    recommended = "old" if old["total_tax_liability"] <= new["total_tax_liability"] else "new"
    return json.dumps(
        {"old_regime": old, "new_regime": new, "recommended_regime": recommended}, indent=2
    )
