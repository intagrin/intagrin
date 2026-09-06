# Travel Planner

A single-entry-point travel assistant that plans itineraries and, once a human approves, books
flights and hotels. For a multi-city trip it spins up a narrowly-scoped research specialist per
destination instead of researching everything itself in one long turn.

See [blueprint.md](./blueprint.md) for the full architecture writeup.

## What this demonstrates

- **Dynamic agent spawning** (`spawns` / `spawn_agent`) — `planner` creates one specialist per
  destination at runtime, each returning a `schemas.ItineraryResult`-validated structured result
  instead of free text.
- **Static human-in-the-loop** (`requires_approval: true`) — `book_flight`/`book_hotel` always
  pause for approval; nothing gets booked without a human saying yes.
- **Keeping sensitive data out of the LLM's context** — the traveler's email and payment
  authorization are collected through a *separate* local web form (`ui/server.py`), never typed
  into the chat. `tools/user_profile_store.py` is the plain local file bridging the two; the chat
  engine never reads it directly.

## Setup

```bash
cp .env.example .env
# then fill in GEMINI_API_KEY and TRAVEL_PLANNER_API_KEY
```

## Running it

From this directory:

```bash
uv run inta dev
```

Before `book_flight`/`book_hotel` will succeed, complete the traveler profile form in a second
terminal:

```bash
uv run uvicorn ui.server:app --port 8600
```

Then open `http://localhost:8600`, enter an email, and submit — the planner finds out a profile
is on file the next time it tries to book, without ever seeing the email itself.

## Try it

- *"Plan a 4-day trip to Paris."* — a single-destination request `planner` handles directly.
- *"I want to visit Paris and then Dubai, 3 days each."* — watch it spawn one specialist per
  destination via `spawn_agent` instead of researching both itself.
- *"Book the flight and hotel for Paris."* — pauses for approval (`inta monitor`'s dashboard, or
  `POST /resume`) before anything is confirmed.

`tools/travel_tools.py` has canned data for Paris, Italy, Dubai, Malaysia, Singapore, and
Thailand — any other destination gets a generic placeholder itinerary.
