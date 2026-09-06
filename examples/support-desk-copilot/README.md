# Support Desk Copilot

A five-agent enterprise support-operations swarm: triage, technical diagnostics, billing/refunds,
an enterprise fast lane, and a nightly batch job that debates recurring failure patterns.

See [blueprint.md](./blueprint.md) for the full architecture writeup and an ASCII diagram of how
the agents connect.

## What this demonstrates

- **A deterministic router** — `is_enterprise_tier` fast-tracks enterprise/enterprise_plus
  accounts to `priority_support_agent` before the LLM even runs, no extra latency or cost.
- **RAG + episodic memory** — three knowledge-base articles under `kb/` are searchable via
  `search_knowledge_base`; recurring technical failure patterns are remembered across sessions via
  `remember_episode`/`recall_episodes`.
- **An Agent Skill** — `technical_agent` loads `skills/known_issues_runbook.md` on demand instead
  of carrying it in the system prompt every turn.
- **Sandboxed code execution + `delegate_to_many`** — isolated diagnostic scripts, and fanning out
  one `service_status_checker` per subscribed service concurrently.
- **`spawn_agent` + `on_complete`** — the enterprise lane spins up a one-off incident specialist
  that returns a `schemas.IncidentResolution`-validated result and auto-appends an escalation note.
- **The lethal-trifecta guardrail** — `auto_close_resolved_ticket` is only offered
  (`available_when`) once no untrusted diagnostic output is pending human review.

## Setup

```bash
cp .env.example .env
# then fill in GEMINI_API_KEY, SUPPORT_DESK_API_KEY, and a *distinct* SUPPORT_DESK_APPROVER_KEY
```

No external accounts needed — `tools/crm_store.py` is a small mock CRM/service-status fixture, so
this runs end to end with zero other credentials.

## Running it

```bash
uv run inta dev
```

## Try it

- *"Hi, I'm customer cus_1002, I can't log in."* — triage looks up the account, checks the
  knowledge base for the login-issue article, and may resolve it directly.
- *"Enterprise customer cus_1003 wants a status update."* — the deterministic router sends this
  straight to `priority_support_agent`, skipping the standard queue.
- *"Customer cus_1003 (subscribed to auth-api, billing-api, search-api, sync-worker) says nothing
  works."* — watch `technical_agent` fan out a status check per service via `delegate_to_many`.

`tools/crm_store.py`'s fixture only knows about `cus_1001`, `cus_1002`, and `cus_1003` — anything
else returns "no account found."

## The nightly batch job

`workflows.nightly_root_cause_review` isn't reachable conversationally — named workflows run
directly via `inta run`:

```bash
uv run inta run nightly_root_cause_review
```

`scripts/nightly_root_cause_review_cron.sh` wraps that for a crontab entry (see the script's own
header comment for the exact line).
