# ITR Filing Copilot

A five-agent copilot that walks a taxpayer through preparing an Indian ITR-1 (Sahaj) income tax
return — intake, income, deductions, old-vs-new regime tax computation, and a human-reviewed final
JSON. **It never submits anything to the government itself** — you upload the generated JSON to
the e-filing portal (or the offline utility) yourself.

> ⚠️ **Not a certified filing product.** This is a demonstration of building deterministic,
> auditable computation into a multi-agent system — tax slabs change every Union Budget, the JSON
> structure has known, documented gaps, and Aadhaar is deliberately never collected at all (see
> below). **Read [blueprint.md](./blueprint.md)'s disclaimer section before using this for
> anything beyond exploring how it's built.**

## What this demonstrates

- **The LLM never does tax arithmetic, anywhere.** Every number — HRA exemption, house property
  income, slab tax, Section 87A rebate with marginal relief, cess, the final payable/refund — comes
  from a pure, independently-testable function in `tools/tax_engine.py`. Agents restate tool
  output; they never compute.
- **Schema-faithful output, not a guess.** `generate_itr_json`'s structure was built directly
  against the Income Tax Department's own official AY 2026-27 ITR-1 JSON schema (fetched and
  verified, not reconstructed from memory) — see the module's own docstring for the specific,
  named gaps that remain (an ERI software-registration code, Section 234A/B/C interest).
- **Aadhaar is never collected or stored — full stop.** Not even out-of-band. Storing Aadhaar
  numbers is legally restricted to UIDAI-authorized entities, and the schema's own Aadhaar field is
  optional — so the only fully safe design is to not take it on at all. See
  `tools/kyc_store.py`'s docstring.
- **Runtime `AwaitingHumanInput`** — `generate_itr_json` always requires approval, and additionally
  pauses with a specific reason when a claimed refund exceeds ₹50,000.
- **A custom guardrails module** — masks bank-account- and Aadhaar-shaped numbers if a user types
  one into chat anyway, while deliberately leaving PAN visible (the taxpayer must be able to
  confirm it).

## Setup

```bash
cp .env.example .env
# then fill in GEMINI_API_KEY, ITR_COPILOT_API_KEY, and a *distinct* ITR_COPILOT_APPROVER_KEY
```

## Running it

```bash
uv run inta dev
```

Before `generate_itr_json` will succeed, complete the refund-account form in a second terminal:

```bash
uv run uvicorn ui.server:app --port 8601
```

Then open `http://localhost:8601` and enter a bank account number + IFSC — no Aadhaar field
exists, by design.

## Try it

Walk through a full filing for PAN `ABCDE1234F` (one of two PANs `tools/itr_tools.py`'s mock Form
26AS fixture recognizes — the other is `PQRSX5678K`; any other PAN gets no TDS data back):

1. Give your PAN, name, DOB, address, and say you want the regimes compared.
2. Answer the salary/house-property/other-income questions `income_agent` asks.
3. If you're on the old regime (or comparing), answer `deductions_agent`'s 80C/80D/80TTA-TTB
   questions.
4. Confirm which regime to file under when `tax_computation_agent` presents the comparison.
5. Confirm with `filing_agent` — `generate_itr_json` will pause for approval, and again if the
   computed refund exceeds ₹50,000.

The generated file lands in `output/ITR1_<PAN>_<AY>.json` (gitignored — it contains PAN and
financial figures).
