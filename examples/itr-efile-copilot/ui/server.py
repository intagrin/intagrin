"""Standalone KYC form for the ITR filing copilot — deliberately a *separate* FastAPI app from
IntaGrin's own `inta serve`/`inta monitor`, run on its own port. This is the "channel outside the
chat conversation entirely" refund bank account details have to go through: the chat engine never
reads this app's code or its output, and this app never calls into IntaGrin's chat API. The only
thing connecting them is tools/kyc_store.py's plain JSON file, which generate_itr_json reads
directly.

Deliberately does NOT ask for Aadhaar — see tools/kyc_store.py's own docstring for why (storage is
legally restricted to UIDAI-authorized entities, and the official ITR-1 schema's Aadhaar field is
optional anyway, so there's no reason to collect it here at all).

Run it alongside `inta dev`/`inta serve`:
    uv run uvicorn ui.server:app --port 8601 --reload
"""

import re
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `tools` is importable regardless of which directory
# uvicorn is launched from.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse

from tools.kyc_store import clear_kyc, get_kyc, save_kyc

_IFSC_RE = re.compile(r"^[A-Z]{4}0[A-Z0-9]{6}$")

app = FastAPI(title="ITR Filing Copilot — Taxpayer KYC")


def _page(body: str) -> HTMLResponse:
    return HTMLResponse(f"""<!doctype html>
<html><head><title>ITR Filing Copilot — Taxpayer KYC</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 480px; margin: 4rem auto; padding: 0 1rem; }}
  input {{ width: 100%; padding: 0.5rem; margin: 0.5rem 0 1rem; box-sizing: border-box; }}
  button {{ padding: 0.6rem 1.2rem; background: #1d4ed8; color: white; border: none;
            border-radius: 6px; cursor: pointer; font-size: 1rem; }}
  .confirmed {{ padding: 1rem; background: #eff6ff; border: 1px solid #1d4ed8; border-radius: 6px; }}
  .error {{ padding: 0.75rem; background: #fef2f2; border: 1px solid #dc2626; border-radius: 6px; margin-bottom: 1rem; }}
</style></head>
<body>{body}</body></html>""")


@app.get("/", response_class=HTMLResponse)
def form_page():
    existing = get_kyc()
    if existing and existing.get("confirmed"):
        return _page("""
            <div class="confirmed">
              <strong>KYC on file.</strong><br>
              The filing copilot can now generate a return with your refund account details.
            </div>
            <p><a href="/reset">Use different KYC details</a></p>
        """)
    return _page("""
        <h2>Refund Account Details</h2>
        <p>These never get typed into the chat — the copilot only finds out a refund account is
           on file once you submit this form, never the account number itself. We deliberately
           don't ask for your Aadhaar here (or anywhere in this project) — see
           tools/kyc_store.py for why.</p>
        <form method="post" action="/submit">
          <label for="bank_account_number">Bank Account Number</label>
          <input type="text" id="bank_account_number" name="bank_account_number" required>
          <label for="ifsc">IFSC Code</label>
          <input type="text" id="ifsc" name="ifsc" required>
          <button type="submit">Save KYC Details</button>
        </form>
    """)


@app.post("/submit", response_class=HTMLResponse)
def submit(
    bank_account_number: str = Form(...),
    ifsc: str = Form(...),
):
    ifsc = ifsc.strip().upper()
    if not _IFSC_RE.match(ifsc):
        return _page(
            '<div class="error">IFSC must look like ABCD0123456.</div>'
            '<p><a href="/">Try again</a></p>'
        )
    save_kyc(bank_account_number=bank_account_number.strip(), ifsc=ifsc)
    return _page("""
        <div class="confirmed">
          <strong>KYC details saved.</strong><br>
          Go back to the chat and continue — the copilot can now generate your return.
        </div>
    """)


@app.get("/reset")
def reset():
    clear_kyc()
    return _page('<p>Cleared. <a href="/">Start over</a></p>')


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=8601)
