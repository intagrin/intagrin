"""Stores the traveler's contact/payment profile (email) completely outside the chat
conversation — written only by ui/server.py's own form-submission endpoint, never by the LLM or
a tool call. book_flight/book_hotel (travel_tools.py) read it directly here, in plain Python, so
the email never becomes a tool argument or a tool result and therefore never enters the LLM's own
context.

This is NOT the same guarantee as write_state/read_state or Shared Typed State — those results
flow back into the conversation like any other tool result (and, if state_schema is ever set, the
whole state dict gets dumped into the system prompt every turn). This module is a genuinely
separate channel: a plain local file the chat engine never reads from or writes to.

Single current-profile record, not session-keyed — right-sized for local/single-user testing,
which is what this project is. A real multi-tenant deployment would key this by an opaque profile
id issued at form-submit time (something safe to say back in chat, e.g. "profile ABC123 is
ready"), never the raw email itself.
"""

import json
from pathlib import Path
from typing import Any

_STORE_PATH = Path(__file__).resolve().parent.parent / ".ai" / "user_profile.json"


def save_profile(email: str) -> None:
    _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STORE_PATH.write_text(json.dumps({"email": email, "payment_authorized": True}))


def get_profile() -> dict[str, Any] | None:
    if not _STORE_PATH.exists():
        return None
    return json.loads(_STORE_PATH.read_text())


def clear_profile() -> None:
    if _STORE_PATH.exists():
        _STORE_PATH.unlink()
