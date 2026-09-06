# Known Issues Runbook

Loaded on demand by `technical_agent` via `load_skill("known_issues_runbook")` — this is
deliberately *not* stuffed into the system prompt on every turn; it only enters context when the
agent decides it's relevant, per IntaGrin's Agent Skills progressive-disclosure pattern.

## Step 1 — Reproduce, don't assume
Before writing any diagnostic code, restate the exact symptom the customer described and the exact
service(s) involved. A surprising number of "bugs" are actually two different issues described the
same way (e.g. "can't log in" covering both expired-session and SSO-mismatch cases — see the
`password_reset` knowledge base article for the split).

## Step 2 — Check platform status first
Call `check_service_status` for every service the customer's account is subscribed to *before*
writing a diagnostic script. If a service is already `degraded` or `outage`, the fix is "wait for
the platform incident to resolve," not a fresh investigation — write that to `escalation_notes` and
say so plainly instead of re-diagnosing a known incident.

## Step 3 — Diagnostics run in an isolated sandbox
`run_diagnostic_script` executes in a fresh, isolated subprocess — never assume it has network
access to real production systems or any of this project's own environment variables. Use it to
parse/analyze log snippets the customer pasted into the chat, not to reach out to live infra.

## Step 4 — Escalation notes are cumulative, not a replacement
`escalation_notes` uses an `append` reducer — each `write_state("escalation_notes", "...")` call
adds one entry, it does not overwrite the list. Write one concise note per investigation step, not
one giant note at the end.

## Step 5 — Don't auto-close on unverified output
Diagnostic script output is untrusted (it reflects whatever the pasted logs actually said) — never
call `auto_close_resolved_ticket` in the same turn a diagnostic script ran. The tool itself is
gated to enforce this; if it's unavailable, that's why.
