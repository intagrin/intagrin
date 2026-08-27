# REST API & SSE Streaming

When you run `inta serve`, IntaGrin spins up a production-ready FastAPI backend designed specifically for React, Vue, and Next.js consumption.

## The Streaming Endpoint (SSE)
`POST /chat/stream`

Instead of the client waiting for the full response, IntaGrin streams it as Server-Sent Events (SSE).

### Token-by-Token Tool Streaming (Next-Gen UI)
Previously, UIs had to wait for the LLM to finish generating an entire JSON blob of tool arguments before showing anything.
IntaGrin intercepts the LLM chunks and streams the tool arguments **token-by-token** to the frontend!

You will receive events like this in real-time:
```json
data: {"type": "content", "content": "Let me look that up for you."}

data: {"type": "tool_chunk", "index": 0, "name": "search_db", "arguments": "{\"query"}
data: {"type": "tool_chunk", "index": 0, "name": "", "arguments": "\": \"re"}
data: {"type": "tool_chunk", "index": 0, "name": "", "arguments": "fund\"}"}
```
This allows your frontend to render beautiful, terminal-like animations of the AI "typing out" its tool parameters live on the screen!

## Multi-Modal Input

`message` on `/chat`, `/chat/stream`, and `/stream` accepts either a plain string or an
OpenAI-style content-part list, so an image can be sent alongside text in the same request:
```json
{
  "session_id": "session_456",
  "message": [
    {"type": "text", "text": "What's odd about this chart?"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
  ]
}
```
Guardrails (PII masking, banned words) still apply to every `"text"` part; non-text parts (e.g.
`image_url`) pass through to the underlying model untouched. Whether the model itself can actually
see the image depends on the LiteLLM model string in `model.primary`/`model_override` — use a
vision-capable model (e.g. `openai/gpt-4o`, `anthropic/claude-3-5-sonnet`, `gemini/gemini-2.5-pro`).

## Standard Endpoints

`POST /chat`
A synchronous endpoint. If the agent hits a tool requiring human approval, it immediately aborts execution and returns:
```json
{
  "status": "awaiting_approval",
  "pending_action": {
    "tool": "post_to_twitter",
    "args": {"content": "Hello World"}
  }
}
```

`POST /resume`
Executes the reviewed tool call once after approval, then continues the agent and returns its result. Supply the caller-visible session ID; the server applies the authenticated tenant namespace.
```json
{
  "session_id": "session_456",
  "approved": true
}
```

## Per-Caller Rate Limiting

By default there's no limit on how much one authenticated caller (tenant) can call `/chat`,
`/chat/stream`, `/resume`, or `/stream` — opt in with `server.rate_limit` in `ai.yaml`:

```yaml
server:
  rate_limit:
    max_requests_per_window: 30
    window_seconds: 60
    max_cost_per_caller_per_day: 5.00
    max_tokens_per_caller_per_day: 500000
```

All four thresholds are `None` (unlimited) by default — set only the ones you need. Enforcement
reuses the `run_logs` audit table (see below) rather than a separate counter: a caller's usage is
an aggregate query over their own rows, filtered by their tenant prefix, so it requires no new
schema and only applies when `memory.type` is `sqlite` or `postgres` (the same scope `run_logs`
itself has). Exceeding a threshold raises error code `IG-RT-008` as an HTTP `429`, with a
`Retry`-relevant message naming which threshold was hit (see `docs/12_Error_Reference.md`). If the
audit database is temporarily unreachable, requests fail **open** (allowed through, not blocked) —
a monitoring outage shouldn't itself become a service outage.

## Debugging API Runs (the Monitor's Logs Page)

Every call to `/chat`, `/chat/stream`, `/resume`, and `/stream` writes a best-effort audit row —
timestamp, session, endpoint, agent, status, cost/token deltas, cumulative totals, message count
(context size), and latency — to a `run_logs` table living alongside your `checkpoints` (same
sqlite file or Postgres database). This is fully automatic whenever `memory.type` is `sqlite` or
`postgres`; no `ai.yaml` changes needed.

`postgres` and `redis` client libraries are optional dependencies — install them with `pip install
"intagrin[postgres]"` / `"intagrin[redis]"` (or `uv sync --extra postgres --extra redis`) before
pointing `memory.type` at either backend, otherwise the server raises a clear `IG-RT-004`/`IG-RT-005`
error at startup rather than a bare `ImportError`.

Browse it on the **Logs** tab of `inta monitor` (`GET /api/logs`, tenant-scoped, most recent 200
runs, searchable by session id/endpoint/status/error). Since the Agent Playground's own
`/api/chat`/`/api/stream`/`/api/resume` proxy directly to these same endpoints, Playground-driven
runs show up here too — one place to see everything that actually hit the engine, however it was
triggered.
