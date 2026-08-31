# A2A (Agent2Agent) Interoperability

`inta serve` exposes every IntaGrin app as an [A2A](https://a2a-protocol.org/) agent, no `ai.yaml`
configuration required — another framework's orchestrator (LangGraph, Google ADK, AutoGen, or
another IntaGrin app) can discover your agent's capabilities and call it exactly like it would call
any other A2A-compliant agent.

> **Not to be confused with:** A2A's Agent Card `skills` field is *protocol vocabulary* — a flat
> list describing what an agent can do, shown to external callers per the A2A spec, derived here
> from your default agent's *tools*. This is a different concept from IntaGrin's own ai.yaml
> `skills:` primitive (see [Agent Skills](/15_Agent_Skills)), an internal, progressive-disclosure
> mechanism for an agent's own prompt content. A project can use both at once.

## Why this matters

`handoffs`, `delegations`, `routers`, `auto_route`, `spawns`, and `workflows` (see
[Choosing an Orchestration Primitive](/03_Choosing_an_Orchestration_Primitive)) all move control
*within* one IntaGrin app. A2A is the opposite direction: it's how an IntaGrin app plugs into
someone else's multi-agent system, or how an IntaGrin app calls out to an agent built with a
different framework entirely. They're different layers, not competing choices — a single agent can
both `delegate_task` internally and be called externally over A2A.

## What's exposed

### `GET /.well-known/agent-card.json`

A static capability card, built from your `ai.yaml`'s `name`, `description`, and the
`default_agent`'s resolved tools. Deliberately unauthenticated (matching the A2A spec's public
discovery convention — a caller fetches this before it has any reason to hold a credential),
mirroring how an OpenAPI/Swagger document is typically served publicly even when the API it
describes requires auth. Actually calling the agent through `POST /a2a` still goes through
`server.auth` like every other endpoint.

If `server.auth.type` is `api_key` or `custom`, the card's `securitySchemes` advertises HTTP bearer
auth (`Authorization: Bearer <token>`) — the same header every `/chat` client already sends.

### `POST /a2a` — JSON-RPC 2.0

Three methods are supported:

- **`message/send`** — translates the A2A message into the same `ChatRequest` shape `/chat` uses
  (`message.contextId` becomes IntaGrin's `session_id`) and runs a real turn through the same
  `chat_endpoint` logic — handoffs, delegations, memory, and guardrails all apply exactly as they
  do over `/chat`. The response's task `status.state` is `completed`, or `input-required` when the
  turn paused for human approval — this is the one honest mapping the whole feature hinges on:
  IntaGrin's own human-in-the-loop pause *is* exactly what A2A's `input-required` state means.
- **`message/stream`** — the SSE equivalent, built on top of `/chat/stream`'s own event generator.
  IntaGrin's token-delta and tool-call-argument streaming events are reframed as a simplified
  text-only A2A status-update stream; tool-call bookkeeping events have no A2A equivalent and are
  dropped rather than forwarded. A caller wanting full visibility into tool-call streaming should
  use `message/send` plus `tasks/get` instead of `message/stream`.
- **`tasks/get`** — reports a session's current status without running a turn: `input-required` if
  a `requires_approval` tool call (or any `AwaitingHumanInput` pause) is outstanding, `working` if
  a long-running [MCP task](/05_Custom_Tools_and_MCP) is still in flight, otherwise `completed`.

An A2A "context" maps 1:1 onto an IntaGrin `session_id` — there is no separate A2A task store;
`tasks/get` reads straight from the same checkpointed engine state `/resume` already uses.

## Explicitly not implemented

- **Push notifications** (`tasks/pushNotificationConfig/*`) — no webhook delivery of task updates.
- **A2A-to-A2A delegated auth chains** — a caller authenticates with `server.auth` exactly like any
  other `/chat` client; there is no support for verifying an upstream agent's own delegated
  identity.
- **Multi-turn binary/file artifact streaming** — `message/stream` is text-only.
- **Any method beyond the three above** (`tasks/cancel`, `tasks/resubscribe`, etc.) — these return
  a JSON-RPC "method not found" error.

## Error shape

Because JSON-RPC 2.0 requires `error.code` to be a number, errors come back with the standard
reserved JSON-RPC codes (`-32600` invalid request, `-32601` method not found, `-32603` internal
error) as `error.code`, with the corresponding IntaGrin error code carried alongside as
`error.data.intagrin_code` (`IG-A2A-001`/`IG-A2A-002` — see the
[Error Code Reference](/12_Error_Reference)) for anyone cross-referencing it.
