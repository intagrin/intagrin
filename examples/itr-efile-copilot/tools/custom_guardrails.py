"""Backstop-only PII masking (see model.guardrails.custom_module) for anything a user pastes
directly into chat. A bank account number should have gone through the out-of-band KYC form
instead (tools/kyc_store.py, ui/server.py) — it never needs to be *stated back* in conversation the
way a PAN legitimately does, so it's masked here, not PAN. Aadhaar is masked too even though this
project never asks for it or stores it anywhere (see tools/kyc_store.py's docstring) — if a user
pastes their Aadhaar into chat anyway, unprompted, this is the last line of defense keeping it out
of the LLM's context and out of checkpointed conversation history.

Runs on both incoming user messages and outgoing assistant messages before mask_pii/banned_words
(engine.py's _apply_guardrails_to_text) — either direction reaching the LLM's own context at all,
even briefly, is the thing this exists to prevent.
"""

import re

_AADHAAR_RE = re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b")
# Indian bank account numbers run 9-18 digits. This form (ITR-1) caps total income at 50 lakh, so
# a legitimate rupee figure typed in this project's chat should never reach 9 digits — if you
# reuse this module for a form with larger income figures, narrow this pattern accordingly.
_BANK_ACCOUNT_RE = re.compile(r"\b\d{9,18}\b")


def apply_guardrails(text: str, guardrails) -> str:
    text = _AADHAAR_RE.sub("[REDACTED_AADHAAR]", text)
    text = _BANK_ACCOUNT_RE.sub("[REDACTED_ACCOUNT_NUMBER]", text)
    return text
