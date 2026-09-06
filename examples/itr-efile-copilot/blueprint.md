# Product Blueprint: ITR Filing Copilot

## ⚠️ Read this before using it for a real filing

This is a demonstration of building a rigorous, deterministic-computation multi-agent system with
IntaGrin — **not** a certified e-filing product. Before relying on it for an actual return:

1. **Tax law changes every Union Budget.** `tools/tax_engine.py` targets AY 2026-27 (FY 2025-26) —
   new-regime slabs and the Section 87A rebate threshold were verified via web search on 2026-09-05
   against the Income Tax Department's own AY 2026-27 ITR-1 JSON schema (its `Rebate87A` field caps
   at exactly ₹60,000, matching what this module computes at exactly ₹12,00,000 taxable income —
   the two facts corroborate each other independently). Old-regime slabs are carried over unchanged
   from AY 2025-26; no source was found indicating the Finance Act, 2025 touched them. Verify every
   figure against the current Finance Act before relying on this for a different assessment year.
2. **The generated JSON's structure was rebuilt against the real official schema** — fetched
   directly from `incometax.gov.in` (AY 2026-27, schema version 1.1) rather than reconstructed from
   general knowledge. Every required section and field name/nesting is faithful to it. What's
   *still* known to be incomplete, confirmed by that same fetch — not guessed at:
   - `CreationInfo.SWCreatedBy`/`JSONCreatedBy` need a real software code issued by the Income Tax
     Department to a **registered e-Return Intermediary (ERI)**. A structurally valid JSON does not
     by itself make this tool an ERI — this is a real access-control gap, not a formality. Filing
     through the free government Offline Utility (fill in your own data, or import a JSON like
     this one, then let the utility itself submit) sidesteps needing ERI registration; a direct
     programmatic upload to the e-filing API does not.
   - `Refund.BankAccountDtls.AddtnlBankDetails`'s exact sub-fields weren't retrieved at full depth
     from the schema fetch — the shape used is a reasonable reconstruction, not independently
     confirmed field-by-field.
   - Interest under Sections 234A/B/C (late filing / short advance tax) is not computed — always 0.
   - Only an original return (`ReturnFileSec: 11`) is supported — no revised/belated return codes.

   See `tools/itr_tools.py`'s own module docstring for the same list, kept next to the code it
   describes.
3. **Have a qualified CA or tax professional review the return** before filing, especially for
   anything beyond the simplest salary-only case.
4. This only covers **ITR-1 (Sahaj)**: resident individuals, income from salary + at most one
   house property + other sources, total income up to ₹50 lakh, no capital gains and no foreign
   income/assets. It deliberately refuses to proceed if any of those apply.

## 1. Product Vision
A multi-agent copilot that walks a taxpayer through preparing an ITR-1 return end to end — intake,
income, deductions, old-vs-new regime tax computation, and a human-reviewed final JSON — and stops
there: **the copilot never submits anything to the government on the user's behalf.** The user
uploads the generated JSON to the e-filing portal (or the offline utility) themselves, keeping the
actual submission step under direct human control.

The one non-negotiable design principle throughout: **the LLM never does tax arithmetic.** Every
number in every agent's response is the direct, unmodified output of a deterministic Python tool
in `tools/tax_engine.py`. An LLM computing a tax slab correctly nine times out of ten is not good
enough when the tenth time produces a wrong number in a legal filing — so it's structurally not
allowed to try. Every agent prompt says this explicitly.

---

## 2. Technical Architecture & Routing

```
                    intake_agent (PAN, name, DOB, AY, regime preference)
                            |
                            v (handoff)
                    income_agent (salary, house property, other sources, Form 26AS)
                       |         \
         (handoff,     |          \ (deterministic router: regime_preference == 'new' —
          old/compare) |           \  skips deductions_agent entirely, no LLM round-trip)
                       v            v
              deductions_agent --> tax_computation_agent (old vs new, or one regime)
              (80C/80D/80TTA-B)          |
                                         v (handoff, on taxpayer confirmation)
                                  filing_agent
                                  (review -> generate_itr_json, requires_approval)
                                         |
                                         v
                              output/ITR1_<PAN>_<AY>.json
                              (the user uploads this themselves)
```

Backward handoffs exist at every stage (e.g. `filing_agent` -> `deductions_agent`) so the taxpayer
can correct something before finalizing — `inta verify` reports these as safely-bounded cycles,
not a design flaw; a real filing conversation rarely proceeds in a single unbroken pass.

### Default Agent & Models
* **Default Entry Agent:** `intake_agent`
* **Primary Model:** `gemini/gemini-3.5-flash-lite`
* **Temperature:** `0.2` — this is a tax assistant, not a creative one.
* **Guardrails:** PII masking, system safeguards, plus a custom module (below).

---

## 3. Keeping Sensitive Data Out of the LLM's Context — and Off Disk Entirely for Aadhaar

The refund bank account number is collected through a **separate, out-of-band HTML form**
(`ui/server.py`, run independently on its own port) backed by `tools/kyc_store.py` — a plain local
JSON file the chat engine never reads from or writes to. `generate_itr_json` reads it directly in
Python. It never becomes a chat message, a tool argument, or a tool result, so it never enters the
LLM's context at all — the same pattern `examples/travel-planner` uses for payment authorization.

**Aadhaar goes further: this project doesn't collect or store it anywhere, full stop** — not even
out-of-band. Storing an Aadhaar number is legally restricted under the Aadhaar Act, 2016 to
entities specifically authorized by UIDAI; an ordinary application isn't automatically permitted
to persist one just because a user consented to a form. The official ITR-1 schema's own
`PersonalInfo.AadhaarCardNo` field is optional, so `generate_itr_json` simply omits it rather than
work around the storage restriction. This was a real gap in an earlier version of this example —
Aadhaar used to flow through the KYC form and a local JSON file — fixed by removing the collection
path entirely rather than trying to store it "more safely" (hashing/masking at rest doesn't help
when the government form field ultimately needs the real value; the only fully safe answer, given
the field is optional, is to never take it on).

PAN is handled differently and deliberately: it **is** allowed in chat, because the taxpayer needs
to see and confirm it's correct before filing — a bank account number has no equivalent need.
`tools/custom_guardrails.py` (`model.guardrails.custom_module`) backstops this: it masks
Aadhaar-shaped and bank-account-shaped numbers on both incoming and outgoing messages, in case a
user pastes one into chat unprompted, while leaving PAN
untouched for `mask_pii`'s built-in checks (which don't recognize PAN's format anyway) to skip.

---

## 4. Deterministic Tax Computation (`tools/tax_engine.py`)

Every function is pure — no I/O, no LLM calls, independently unit-testable:

* `compute_hra_exemption` — the standard three-way-minimum HRA exemption formula (Section 10(13A)).
* `compute_house_property_income` — self-occupied (interest deduction capped at ₹2,00,000) vs.
  let-out (30% standard deduction + interest, loss set-off capped at ₹2,00,000 per Section 71(3A)).
* `compute_tax_liability` — full pipeline for one regime: standard deduction → gross total income
  → Chapter VI-A deductions (old regime only) → total income (rounded per Section 288A) → slab tax
  → Section 87A rebate (with new-regime marginal relief) → 4% health & education cess → total tax
  liability (rounded per Section 288B) → net payable/refund after TDS credit.
* `compare_tax_regimes` — runs both regimes and recommends whichever has the lower total tax
  liability. Every comparison an agent presents comes from this, never an LLM guess.

`tax_computation_agent`'s prompt states outright: restate tool output exactly, never recompute.

---

## 5. Human-in-the-Loop

| Trigger | Mechanism | Why |
|---|---|---|
| Every `generate_itr_json` call | static `requires_approval: true` | Finalizing a return is the point-of-no-return step for a session — always reviewed, no exceptions. |
| A claimed refund over ₹50,000 | runtime `AwaitingHumanInput` | A large refund claim gets an explicitly-flagged second look on top of the blanket approval above. |
| Mismatched arithmetic between sections | `generate_itr_json` fails closed | Never write an internally-inconsistent JSON, regardless of what the calling agent claims each field is — checked against a $1 floating-point tolerance, not trusted from the LLM's restated figures. |
| No KYC on file | `generate_itr_json` fails closed | Never generate a return with no refund account to pay into — points the taxpayer at the KYC form instead of asking for the details in chat. |

Set a **distinct** `ITR_COPILOT_APPROVER_KEY` from `ITR_COPILOT_API_KEY` in `.env` — otherwise the
same session that triggers filing could also approve it itself.

---

## 6. Running It

```bash
uv run inta dev                              # chat with the copilot
uv run uvicorn ui.server:app --port 8601     # separate KYC form, run alongside it
```

The generated JSON lands in `output/ITR1_<PAN>_<AY>.json` (gitignored — it contains PAN and
financial figures). Read the disclaimer at the top of this document again before uploading it
anywhere.
