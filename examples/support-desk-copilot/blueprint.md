# Product Blueprint: Support Desk Copilot

## 1. Product Vision
Support Desk Copilot is an enterprise support-operations swarm: it triages incoming tickets,
resolves known issues straight from a knowledge base, runs isolated diagnostics on technical
problems, handles billing and refunds with human oversight scaled to the actual risk of the
action, and fast-tracks enterprise-tier accounts to a concierge lane that can spin up a
narrowly-scoped specialist for a genuinely novel incident — without a human ever having to decide
which specialist should look at a ticket first.

---

## 2. Technical Architecture & Routing

```
                              incoming ticket
                                    |
                                    v
                          +-------------------+
                          |   triage_agent     |<-------------------+
                          | (KB lookup, tier   |                     |
                          |  classification)   |                     |
                          +-------------------+                     |
                             |      |      ^                        |
              (deterministic |      |      | (handoff)               |
                  router)    |      |      |                        |
       is_enterprise_tier    |      v      |                        |
                             |  +-----------------+   (handoff)      |
                             |  |  billing_agent  |<---------------->|
                             |  | (invoice, refund|   +--------------+
                             |  |  >$500 = HITL)  |   |
                             |  +-----------------+   |
                             |         ^              |
                             |         | (handoff)    |
                             |         v              v
                             |  +----------------------------+
                             |  |      technical_agent        |
                             |  | (service status, sandboxed  |
                             |  |  diagnostics, runbook skill) |
                             |  +----------------------------+
                             |         |
                             |         | delegate_to_many (N = subscribed services)
                             |         v
                             |  +-------------------------+
                             |  |  service_status_checker  |  (fan-out, one per service)
                             |  +-------------------------+
                             v
                  +--------------------------+
                  |  priority_support_agent   |  (enterprise / enterprise_plus only)
                  |  (concierge lane; can     |
                  |   spawn_agent a narrow    |----> ephemeral incident specialist
                  |   incident specialist)    |      (return_to_creator, structured
                  +--------------------------+       IncidentResolution result)
```

### Default Agent & Models
* **Default Entry Agent:** `triage_agent`
* **Primary Model:** `gemini/gemini-3.5-flash-lite`
* **Fallback Model:** `gemini/gemini-3.5-flash`
* **Temperature:** `0.3` (support responses should be consistent, not creative)
* **Guardrails:** PII masking, system safeguards, and a banned-phrase list
* **Circuit Breakers:** 12 handoffs/session, delegation depth/turns bounded, $2.00/session cost cap

---

## 3. Agent Specifications

### A. Triage Agent (`triage_agent`)
The single entry point for every ticket. Looks up the account, records `customer_tier` and
`ticket_summary` to shared state, and answers directly from the knowledge base
(`search_knowledge_base`, auto-registered by `rag:`) when it can. A **deterministic router**
(`is_enterprise_tier`, a `condition_functions` predicate — no LLM call, no latency) transparently
fast-tracks enterprise/enterprise_plus accounts to `priority_support_agent` before the model even
sees the next turn. Owns ticket closure via `auto_close_resolved_ticket`, which is only offered
(`available_when`) once no untrusted diagnostic output is pending human review — the lethal-trifecta
guardrail applied to an unglamorous but very real automation risk.

### B. Technical Agent (`technical_agent`)
Loads the `known_issues_runbook` **Agent Skill** on demand (progressive disclosure — it's not in
the system prompt on every turn, only when the agent decides it's relevant) before investigating.
Checks live platform status per service, and for an account subscribed to several services at
once, fans out with **`delegate_to_many`** — one `service_status_checker` instance per service,
concurrently, capped by `circuit_breakers.max_parallel_fan_out`. For anything the knowledge base
and status page can't explain, `run_diagnostic_script` (a `type: "sandbox"` tool) analyzes
customer-pasted logs in an isolated subprocess, gated by `requires_approval: true` since its output
is untrusted by default.

### C. Billing Agent (`billing_agent`)
Looks up the exact invoice before ever stating an amount. `issue_refund` demonstrates **dynamic,
runtime-decided human approval**: refunds of $500 or less complete immediately; refunds above that
raise `AwaitingHumanInput` from inside the tool body and pause for a reviewer via the same
`/resume` mechanism a statically-flagged `requires_approval: true` tool uses — the difference is
the decision is made in Python, per call, not for the whole tool.

### D. Priority Support Agent (`priority_support_agent`)
The enterprise/enterprise_plus concierge lane. Answers routine requests directly; for a genuinely
novel, account-specific incident, uses **`spawn_agent`** to create a single narrowly-scoped
specialist mid-session (tool pool restricted to a schema-enforced subset of its own tools),
which reports back a **`spawns.result_schema`**-validated `IncidentResolution` — structured data,
not a free-text summary a caller would have to re-parse. Completion automatically appends an
`escalation_notes` entry via `spawns.on_complete`, no `write_state` call required from the spawned
agent's own instruction.

### E. Service Status Checker (`service_status_checker`)
A deliberately narrow worker, reachable only via `technical_agent`'s `delegate_to_many` — never a
direct handoff target. Exists to make the "N services, one call each, concurrently" pattern
concrete rather than theoretical.

---

## 4. Memory, RAG & Episodic Recall

* **Shared Typed State** (`state_schema: schemas.TicketState`): every agent's prompt is JIT-injected
  with exactly the fields the schema declares — `customer_id`, `customer_tier`, `ticket_summary`,
  `escalation_notes` (an `append` reducer — each note is added, never overwritten), `resolved`.
* **RAG** (`rag: docs_dir: kb`): three knowledge-base articles (login/password issues, sync errors,
  billing FAQ) are automatically searchable via `search_knowledge_base` — no separate tool to wire
  up. `embedding_model` is pinned to `gemini/gemini-embedding-001` — both `rag` and
  `episodic_memory` default `embedding_model` to OpenAI's `text-embedding-3-small` regardless of
  `model.primary`'s provider, which would otherwise mean a second API key (`OPENAI_API_KEY`) is
  needed on top of `GEMINI_API_KEY` just for search/recall to work.
* **Episodic Memory** (`episodic_memory: scope: tenant`): `technical_agent`'s failure-loop
  breaker (three identical tool errors in a row) automatically records a `remember_episode` entry
  when it fires, so a recurring platform issue is recognized on a later ticket — and on the nightly
  batch review below — instead of being re-diagnosed from scratch every time.

---

## 5. Nightly Batch Workflow

`workflows.nightly_root_cause_review` (run via `inta run nightly_root_cause_review`, e.g. from
`scripts/nightly_root_cause_review_cron.sh` on a nightly cron schedule — named workflows run
directly, not conversationally through `/chat`) recalls the day's recurring failure patterns from
episodic memory, then runs a **`vote` task with `debate_rounds: 2`**: two independent hypotheses
(infrastructure vs. client-side) each get a fresh answer, then a revision round where each branch
sees the other's reasoning, before an `llm_judge` picks or synthesizes the final root-cause call —
the multi-agent-debate pattern applied to a genuinely useful ops task instead of a toy example.

---

## 6. Human-in-the-Loop Summary

| Trigger | Mechanism | Why |
|---|---|---|
| Any `run_diagnostic_script` call | static `requires_approval: true` | Executes agent-authored code; always needs a human, not just sometimes. |
| `issue_refund` over $500 | runtime `AwaitingHumanInput` | Most refunds are routine; only large ones need review. |
| `auto_close_resolved_ticket` right after untrusted diagnostic output | `available_when` gate | Don't let the model close a ticket based on output nobody's verified yet. |

Set a **distinct** `SUPPORT_DESK_APPROVER_KEY` from `SUPPORT_DESK_API_KEY` in `.env` — otherwise the
same session that triggers an approval-gated action could approve it itself.
