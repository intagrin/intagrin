# Social Media Manager

A three-agent pipeline that researches trending AI/IT topics, drafts a LinkedIn post, and
publishes it once a human approves — designed to run unattended on a daily schedule.

See [blueprint.md](./blueprint.md) for the full architecture writeup.

## What this demonstrates

- **Cyclic conversational handoffs** — `research_agent` ↔ `content_generator_agent` ↔
  `reviewer_agent` can hand control back and forth as a draft goes through revisions, not just a
  fixed one-way pipeline.
- **`lazy_load_tools`** on every agent — semantic tool retrieval so schema tokens don't grow with
  every tool this project adds.
- **Human-in-the-loop publishing** — `post_to_linkedin` requires approval; without a real
  `LINKEDIN_ACCESS_TOKEN` it runs in a safe sandbox/demo mode that returns a mock post ID instead
  of touching the real LinkedIn API.
- **Unattended scheduled runs** — `scripts/daily_cron_trigger.sh` drives the whole pipeline over
  HTTP (`/chat`) for a cron job, rather than an interactive chat session.

## Setup

```bash
cp .env.example .env
# then fill in GEMINI_API_KEY, INTAGRIN_API_KEY, and a *distinct* INTAGRIN_APPROVER_KEY
```

`LINKEDIN_ACCESS_TOKEN`/`LINKEDIN_AUTHOR_URN` are optional — omit them to stay in demo mode.

## Running it

From this directory:

```bash
uv run inta dev          # interactive chat
uv run inta serve        # HTTP API (what the cron script and docker-compose both target)
```

Or via Docker:

```bash
docker compose up --build
```

## Try it

*"Find the latest trending topic in AI and draft a LinkedIn post about it."* — watch it move
through research → draft → review, then pause for approval before `post_to_linkedin` runs.

## Running it on a schedule

`scripts/daily_cron_trigger.sh` posts a fixed research-and-draft prompt to a running `inta serve`
instance's `/chat` endpoint. Point `DEFIN_API_URL` at your deployment and add it to crontab — the
script itself documents the exact line to use.
