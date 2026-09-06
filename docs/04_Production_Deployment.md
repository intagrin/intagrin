# Production Deployment

You can containerize your swarm with a single command:

```bash
inta deploy
```

This instantly generates a `python:3.12-slim` `Dockerfile`, a `.dockerignore`, and a `docker-compose.yml`. The generated container installs `intagrin`, starts with `inta serve`, and runs as a non-root `appuser`.

To spin it up locally:
```bash
docker compose up -d --build
```

To push it to AWS, GCP, or Railway, simply deploy the generated `Dockerfile`. Because IntaGrin relies on a standard FastAPI backend, it scales well horizontally for throughput — but read "Concurrency & Multi-Process Deployment" below before running more than one process/replica of the same project: one specific correctness guarantee (the per-session lock) is deliberately process-local, not distributed.

## Runtime Resource Pooling

Building an agent's tool schemas, connecting its MCP servers, indexing its RAG documents, and
loading its system prompts is real work — reconnecting subprocess-backed MCP servers or re-running
tool reflection on every single request would make a busy server or worker slow for no reason.
IntaGrin pools these resources at the process level instead of rebuilding them per request or per
job.

**`inta serve`** — the first request against a project builds a `SharedResources` pool (MCP
connections, tool schemas, the RAG index, agent prompts) once; every subsequent `/chat`,
`/chat/stream`, `/resume`, `/stream`, and `/ws/voice` request for that project reuses it. The pool
is invalidated and rebuilt automatically if `ai.yaml`'s modified time changes, so editing your
config and hitting the server again picks up the new tools/prompts without a restart. MCP
connections are torn down once, on server shutdown — not after every request.

**`inta worker`** — a `DistributedWorker` builds one pool per project on its first job and reuses
it for every job pulled off the queue afterward, for as long as the worker process runs. Cleanup
happens on graceful `stop()` or process exit, not per job.

This pooling is per-project and per-process: two different projects served from the same process,
or two separate `inta serve` processes, each get their own pool. Session state (conversation
history, `read_state`/`write_state`) is never pooled — only the parts of an engine that are the
same for every session against a given project.

**`inta run <workflow>`** (a one-shot CLI invocation) and delegated/parallel sub-agents spawned
*within* a single request also skip redundant rebuilds — a `delegate_task` call or a parallel
workflow branch inherits its parent engine's already-built tools/connections instead of
reconnecting from scratch mid-request.

## Concurrency & Multi-Process Deployment

**Per-session locking is process-local, not distributed.** `/chat`, `/chat/stream`, `/resume`, and
`/stream` serialize concurrent requests for the *same* `session_id` via an in-process lock
registry (`runtime/session_locks.py`) — closing the "double-click / client retry / two open tabs"
race where one request's checkpoint save silently clobbers another's. This lock only sees requests
handled by its own process. Running more than one `inta serve` process (multiple uvicorn workers,
multiple container replicas, multiple serverless instances) for the same project means two
concurrent requests for the same session *can* land on different processes and race — a lost
checkpoint write, or, in the worst case, a side-effecting tool call executing twice. If your
deployment fans out across processes, either route by `session_id` (sticky sessions at the load
balancer) so the same session always reaches the same process, or accept that cross-process
same-session collisions are a known, unresolved edge case for now — there is no distributed lock
built in. The lock registry itself is reference-counted, not a fixed-size cache: it holds an entry
only for sessions with a request genuinely in flight right now, so its memory footprint tracks
concurrent load, not lifetime session count.

**One-off Postgres writes are pooled, not opened-and-closed per call.** Audit logging
(`run_logs`, one row per API call), rate-limit checks, DB-backed approver lookups
(`runtime/approvers.py`), and shared/episodic memory reads and writes all borrow a connection from
the same process-wide pool `PostgresCheckpointer` itself uses (`runtime/memory.py`'s
`pooled_postgres_connection`, keyed by connection URL) instead of opening a fresh connection per
call — avoids adding a TCP+auth round-trip to every request and reduces pressure on Postgres's own
`max_connections` under bursty load.

**Audit-log and episode tables have no retention by default.** `run_logs` (one row per API call)
and, if `episodic_memory:` is configured, `episodes` (one row per `remember_episode` call) are both
append-only with no automatic cleanup unless you opt in — they will grow for as long as the project
runs. Set retention explicitly once you're running this in production for real:

```yaml
memory:
  run_log_retention_days: 90   # None (default) keeps every run_logs row forever

episodic_memory:
  retention_days: 90            # None (default) keeps every episode forever
```

Pruning is opportunistic (a small random fraction of writes also runs one indexed `DELETE ...
WHERE created_at < cutoff`) rather than a scheduled job — no extra cron/infrastructure needed, at
the cost of old rows disappearing "eventually" rather than exactly at the boundary. `run_logs` is
also what `server.rate_limit` queries on every single request, so retention keeps that query fast
as well as keeping the table itself bounded.

## Shadow Replay: `inta simulate`

Every framework can tell you whether an agent *ran*. None of them can tell you what a config
change would have done to conversations you already had — until you've deployed it and find out
from a support ticket. `inta simulate` answers that question before you ship the change, by
replaying real checkpointed sessions through a candidate `ai.yaml`.

```bash
inta simulate --config ai.yaml.new              # every session on record
inta simulate --config ai.yaml.new --since 30d  # only the last 30 days
inta simulate --config ai.yaml.new --session sess_123 --session sess_456
```

For each historical session, it reconstructs what really happened — state, active agent, handoff
count, tool-failure streak, cost — from the checkpointed message history, then re-evaluates the
*candidate* config's routers, circuit breakers, and `requires_approval` flags against that
reconstruction. No LLM is called and no tool is re-executed; a deterministic router's decision, a
circuit-breaker threshold, and an approval gate are all pure functions of state, so there's nothing
to call.

**Verdicts**, per session that would change:
- `ROUTING_DIVERGES` — a root or conditional router would now fire (or stop firing, or pick a
  different target) at a specific point in the real transcript.
- `NEW_CIRCUIT_BREAKER_TRIP` — `max_handoffs_per_session`, `max_tool_failures_in_a_row`, or
  `max_usd_cost_per_session` would now trip where the old config didn't.
- `NEW_APPROVAL_GATE` / `APPROVAL_GATE_REMOVED` — a tool the session actually called would now
  pause for human approval, or no longer would (local tools only — see below).

**What it deliberately won't guess at.** A config change that could alter what the LLM itself
generates — a different prompt, a different model, adding/removing/reconfiguring a tool,
`lazy_load_tools`, `auto_route`, `handoffs`/`delegations` — makes the whole run report
`not_simulatable` with the specific fields that blocked it, rather than silently pretending the
routing/circuit-breaker checks are still meaningful. Predicting *that* class of change needs a real
LLM call from the point of divergence onward — a deliberately separate, not-yet-built capability.
Approval-gate diffing is also scoped to `LocalToolConfig` entries only: an MCP server's tool names
aren't known without a live connection to it, so those are skipped rather than guessed.

**Memory backend requirements.** `inta simulate` needs to enumerate past sessions, not just load
one by id, so it requires `memory: {type: sqlite}` or `postgres` — both track `updated_at` per
session. `redis` works but only sees whatever hasn't already expired under its TTL (7 days by
default), and `--since` on Redis-backed history is best-effort. `buffer`/`sliding_window` memory
(no checkpointer at all) has nothing to replay.
