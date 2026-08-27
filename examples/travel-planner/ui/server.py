"""Standalone profile/payment form for the travel planner — deliberately a *separate* FastAPI
app from IntaGrin's own `inta serve`/`inta monitor`, run on its own port. This is the "channel
outside the chat conversation entirely" the email has to go through: the chat engine never reads
this app's code or its output, and this app never calls into IntaGrin's chat API. The only thing
connecting them is tools/user_profile_store.py's plain JSON file, which book_flight/book_hotel
read directly.

Run it alongside `inta dev`/`inta serve`:
    uv run uvicorn ui.server:app --port 8600 --reload
"""

from pathlib import Path
import sys

# Ensure the project root is on sys.path so `tools` is importable regardless of
# which directory uvicorn is launched from.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse

from tools.user_profile_store import get_profile, save_profile

PROJECT_DIR = Path(__file__).resolve().parent.parent

app = FastAPI(title="Travel Planner — Traveler Profile & Payment")


def _page(body: str) -> HTMLResponse:
    return HTMLResponse(f"""<!doctype html>
<html><head><title>Travel Planner — Traveler Profile</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 480px; margin: 4rem auto; padding: 0 1rem; }}
  input {{ width: 100%; padding: 0.5rem; margin: 0.5rem 0 1rem; box-sizing: border-box; }}
  button {{ padding: 0.6rem 1.2rem; background: #16a34a; color: white; border: none;
            border-radius: 6px; cursor: pointer; font-size: 1rem; }}
  .confirmed {{ padding: 1rem; background: #ecfdf5; border: 1px solid #16a34a; border-radius: 6px; }}
</style></head>
<body>{body}</body></html>""")


@app.get("/", response_class=HTMLResponse)
def form_page():
    existing = get_profile()
    if existing and existing.get("payment_authorized"):
        return _page(f"""
            <div class="confirmed">
              <strong>Profile on file.</strong><br>
              Payment authorized — the planner can now book flights and hotels.
            </div>
            <p><a href="/reset">Use a different traveler</a></p>
        """)
    return _page("""
        <h2>Before we book anything</h2>
        <p>Enter your email so we can send confirmations, then authorize payment.
           This never gets typed into the chat — the planner only finds out once it's done.</p>
        <form method="post" action="/submit">
          <label for="email">Email</label>
          <input type="email" id="email" name="email" required>
          <button type="submit">Proceed to Payment</button>
        </form>
    """)


@app.post("/submit", response_class=HTMLResponse)
def submit(email: str = Form(...)):
    # A real integration would run an actual payment authorization here before saving; this demo
    # treats form submission itself as authorization, matching book_flight/book_hotel's own
    # simulated (not real) booking behavior.
    save_profile(email)
    return _page("""
        <div class="confirmed">
          <strong>Payment authorized.</strong><br>
          Go back to the chat and continue — the planner can now book flights and hotels.
        </div>
    """)


@app.get("/reset")
def reset():
    from tools.user_profile_store import clear_profile

    clear_profile()
    return _page('<p>Cleared. <a href="/">Start over</a></p>')


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8600)