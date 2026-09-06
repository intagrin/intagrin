"""Stores the taxpayer's refund bank account details completely outside the chat conversation —
written only by ui/server.py's own form-submission endpoint, never by the LLM or a tool call.
generate_itr_json (tools/itr_tools.py) reads it directly here, in plain Python, so these values
never become a tool argument or a tool result and therefore never enter the LLM's own context.
Unlike PAN — which the taxpayer must be able to see and confirm in chat before filing — a bank
account number has no legitimate reason to ever appear there at all.

Deliberately does NOT collect or store Aadhaar. Two independent reasons, not just one:
1. Storing an Aadhaar number is restricted under the Aadhaar Act, 2016 to entities specifically
   authorized by UIDAI (AUAs/KUAs/Sub-AUAs and similar) — an ordinary application isn't
   automatically permitted to persist one just because a user typed it into a form, regardless of
   consent or good intent.
2. The official ITR-1 JSON schema's own `PersonalInfo.AadhaarCardNo` field is optional, not
   required — so omitting it entirely costs nothing in terms of producing a valid return.
Combined, there is no reason for this project to take on that storage risk at all. If a real
deployment genuinely needs to include Aadhaar in a generated return, that's a separate decision
requiring its own legal/security review (Aadhaar Data Vault tokenization, encryption at rest,
UIDAI authorization) — not something to default into.

This is NOT the same guarantee as write_state/read_state or Shared Typed State — those results
flow back into the conversation like any other tool result. This module is a genuinely separate
channel: a plain local file the chat engine never reads from or writes to.

Single current-taxpayer record, not session-keyed — right-sized for local/single-user testing,
which is what this project is. A real multi-tenant deployment would key this by an opaque
reference id issued at form-submit time, never raw account details themselves.
"""

import json
from pathlib import Path
from typing import Any

_STORE_PATH = Path(__file__).resolve().parent.parent / ".ai" / "kyc_details.json"


def save_kyc(bank_account_number: str, ifsc: str) -> None:
    _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STORE_PATH.write_text(
        json.dumps(
            {
                "bank_account_number": bank_account_number,
                "ifsc": ifsc,
                "confirmed": True,
            }
        )
    )


def get_kyc() -> dict[str, Any] | None:
    if not _STORE_PATH.exists():
        return None
    return json.loads(_STORE_PATH.read_text())


def clear_kyc() -> None:
    if _STORE_PATH.exists():
        _STORE_PATH.unlink()
