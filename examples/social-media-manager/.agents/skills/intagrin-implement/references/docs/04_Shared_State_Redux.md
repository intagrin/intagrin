# Shared Typed State (Redux for AI)

In multi-agent systems, agents need a way to communicate and share data. Instead of passing massive strings of text back and forth, IntaGrin implements a centralized, typed state-machine—much like Redux does for React frontends.

## 1. Global State
Every session in IntaGrin has a global `state` dictionary that is securely persisted to Postgres/SQLite. 

Agents have native access to two global tools:
- `read_state(key: str)`
- `write_state(key: str, value: str)`

If the `billing` agent looks up an invoice, it can call `write_state("invoice_id", "12345")`. When the user is transferred to the `support` agent, that agent can call `read_state("invoice_id")` and instantly have the context.

## 2. JIT State Injection
To prevent the LLM from "forgetting" to call the `read_state` tool, IntaGrin supports Just-In-Time (JIT) State Injection.

If you specify `state_schema` in your `ai.yaml`:

```yaml
state_schema: "schemas.UserState"
```

The engine will take the entire Shared State, JSON dump it, and automatically inject it into the active agent's system prompt at every single turn. 

```text
[SHARED TYPED STATE]:
{"invoice_id": "12345", "user_balance": -50.00}
```

If `state_schema` is set, every `write_state` call is also validated against that Pydantic model
before it's committed — a write that violates the schema is rejected and the error is returned to
the agent, so bad-shaped data can't silently enter the shared state.

**Sequential handoffs** (agent A hands off to agent B in the same turn chain) do share this live
state — B sees whatever A wrote. **Parallel workflow branches are different**: each branch runs
against its own fresh, isolated state (only `long_term_memory` is copied in), not a live view of
the parent's state — they can't see each other's writes mid-run. State only comes back together
when the parallel task completes and reducers merge each branch's final state into the parent, as
described below.

## 3. Declarative State Reducers
When you run agents concurrently using Parallel Workflows, they each mutate their own isolated copy of the state. How do you merge them back together without data loss?

Instead of writing custom merge functions, IntaGrin handles this declaratively. In your `ai.yaml`, define `reducers`:

```yaml
reducers:
  - key: "research_reports"
    strategy: "append"
  - key: "financial_stats"
    strategy: "deep_merge"
  - key: "executive_summary"
    strategy: "overwrite"
```

When a parallel workflow completes, the engine gathers the final state from each branch and merges
every key a branch actually changed back into the parent's global state: a key with a declared
`reducers` entry uses that strategy (`append`/`deep_merge`/`overwrite`); a changed key with no
declared reducer still merges back via a plain overwrite (last write wins) rather than being
silently dropped. A key a branch left untouched is never re-merged. Declare a reducer only for the
keys where plain overwrite isn't what you want (accumulating a list, merging a dict).

## 4. Vote Workflows (Fan-Out-and-Vote)

Sometimes you don't want every branch's answer — you want one consensus answer from several
independent attempts at the same task. A `vote` workflow task runs branches exactly like
`parallel` (same isolated per-branch state, same reducer merge-back above), but aggregates their
final answers into a single result instead of concatenating all of them:

```yaml
workflows:
  fact_check:
    - name: "consensus_check"
      type: "vote"
      vote:
        strategy: "majority"      # or "llm_judge"
        min_agreement: 0.6
      tasks:
        - name: "checker_1"
          agent: "fact_checker"
          instruction: "Is this claim true: ..."
        - name: "checker_2"
          agent: "fact_checker"
          instruction: "Is this claim true: ..."
        - name: "checker_3"
          agent: "fact_checker"
          instruction: "Is this claim true: ..."
```

`"majority"` (the default) compares branch answers directly and makes **zero extra LLM calls**. If
the winning answer's share of branches is below `min_agreement`, the task doesn't guess — it
reports "no consensus reached" plus every branch's output, the same "stop and ask rather than
guess" philosophy `inta compile` uses when a blueprint is ambiguous.

`"llm_judge"` makes one extra call (using `model.fallback`, or `model.primary` if no fallback is
set) to pick or synthesize the best answer from all branch outputs — useful when answers are
worded differently but semantically equivalent, where exact-text majority voting would see them as
disagreeing.

## 5. Cross-Session / Org-Level Shared Memory

Everything above is scoped to one session's own `state` and `long_term_memory`. `memory.shared_scope`
extends `long_term_memory` specifically — the summary `_compress_memory` produces once a
conversation exceeds `memory.max_messages` — to a broader scope:

```yaml
memory:
  type: "sqlite"          # or "postgres" — shared_scope is a no-op for redis/sliding_window/buffer
  shared_scope: "tenant"  # "session" (default) | "tenant" | "global"
```

- `"session"` (default): today's behavior — `long_term_memory` is private to one `session_id`.
- `"tenant"`: shared across every session under the same authenticated caller (the tenant prefix
  `server/api.py` already namespaces `session_id` with). A returning user gets their long-term
  profile even in a brand-new session.
- `"global"`: shared across every session in the project, any tenant — for a single-tenant
  deployment where all users should benefit from what any session has learned.

A session picks up the current shared content on every `initialize()` (every request) — so a
long-running session sees updates other sessions have made since it last checked — and contributes
back to it every time its own `_compress_memory` runs. **This is last-write-wins, not a merge**:
two sessions compressing at the same moment under the same scope will have one overwrite the
other's contribution, with no versioning or conflict resolution. Fine for "give returning users
their context back"; not a substitute for a real shared knowledge base if you need concurrent
writers to never lose data.
