# Dynamic Runtime Agent Creation

Every mechanism in [Routing & Handoffs](./03_Agent_Handoffs_and_Routing) — `handoffs`,
`auto_route`, `delegations`, conditional `routers` — moves control between agents you declared in
`ai.yaml` ahead of time. `spawns` is different: it lets an agent create a brand-new agent —
a new system prompt, a new subset of tools, a chosen model — **while a session is running**.

This is the one feature in IntaGrin that cuts against its own core rule ("everything declared,
everything statically verifiable"), so it's designed to be bounded, not free-form. Read this whole
page before enabling it — the defaults exist for a reason.

## What it is not

A dynamically-created agent can **never** get a tool implementation that wasn't already declared in
`ai.yaml` and already loaded — no new Python module, no new MCP server, no new OpenAPI URL supplied
at runtime. It can only be granted a subset of tools that already exist, already went through
`requires_approval` review, and already passed `inta verify`. If you're picturing an LLM writing
and running its own new tool code — that's not this feature, and IntaGrin doesn't have it.

## Configuring a factory agent

```yaml
agents:
  research_orchestrator:
    description: "Coordinates research across sub-topics"
    tools:
      - name: "web_search"
        module: "tools.search"
      - name: "summarize"
        module: "tools.summarize"
      - name: "issue_refund"
        module: "tools.billing"
        requires_approval: true
    spawns:
      tool_pool: ["web_search", "summarize"]   # NOT issue_refund — see below
      max_creations_per_session: 5
      requires_approval_on_first_action: true   # default — see below
      allow_recursive_spawning: false            # default
```

`spawns.tool_pool` **must be a subset of this agent's own `tools:`** — enforced when `ai.yaml` is
parsed (`inta verify` and every server start), not just at runtime. An agent can only hand a
sub-agent capabilities it already has itself; there's no way to configure a spawn factory that
escalates privilege through creation. In the example above, `research_orchestrator` has three
tools but can only grant `web_search`/`summarize` to anything it spawns — `issue_refund` (its own
approval-gated tool) is never eligible, on purpose.

When a factory should grant everything it has, write `tool_pool: "*"` instead of re-listing every
tool name — it's expanded to this agent's own tool list at parse time (the same subset validator
above), so it's shorthand for the common case, not a way around the ceiling.

At runtime, `research_orchestrator` gets a `spawn_agent` tool:
```json
{"role": "Climate policy specialist", "instruction": "Research EU carbon tariff policy", "tools": ["web_search"]}
```
The `tools` argument is schema-constrained (a JSON Schema `enum`) to exactly `spawns.tool_pool` —
the LLM cannot name anything outside it, and the engine re-validates server-side even if it did.

**Calling `spawn_agent` does not transfer control.** It creates the new agent, runs it to
completion in an isolated child engine (its own message history, its own turn loop, bounded by
`circuit_breakers.max_delegation_turns`), and returns the result as an ordinary tool result — the
creator's own turn is never interrupted. This is deliberate: the creator can call `spawn_agent`
multiple times in one turn and they run *concurrently*, each in its own isolated child, with no
shared mutable control-flow state to race on. (An earlier "transfer control" design was tried and
discarded for exactly this reason — concurrent spawns racing to become "the" active agent.)

## The star topology — no dynamic-to-dynamic edges

A spawned agent gets exactly three things: its granted tool subset, `read_state`/`write_state`,
and a single fixed `return_to_creator` tool — not open `handoffs` or `delegations` of its own. It
cannot hand off to another agent, static or dynamic, except back to whoever created it. This keeps
the graph shape a bounded star (creator → sub-agent → back to creator) instead of an open mesh,
which is what makes the safety story below tractable at all.

```
research_orchestrator ──spawn_agent (isolated, synchronous)──▶ research_orchestrator_dyn_a1b2c3d4
        ▲                                                                    │
        └───────────────────────return_to_creator───────────────────────────┘
                        (result flows back as spawn_agent's own tool result)
```

## Getting structured data back: `spawns.result_schema`

By default, `return_to_creator`'s only argument is a free-text `summary` — fine for a quick status
update, fragile if the creator actually needs to *act* on what the sub-agent found (e.g. book a
flight based on an itinerary a research sub-agent produced). Set `spawns.result_schema` to a dotted
Pydantic model path and `return_to_creator`'s tool-call schema is derived from that model instead:

```yaml
spawns:
  tool_pool: ["create_itinerary"]
  result_schema: "schemas.ItineraryResult"   # dotted path to a Pydantic BaseModel
```

```python
# schemas.py
from pydantic import BaseModel

class ItineraryResult(BaseModel):
    destination: str
    days: int
```

This is a stronger guarantee than asking the model to format free text correctly: the LLM
provider's own constrained tool-call decoding steers it toward the right shape, and the engine
re-validates the arguments server-side before accepting them regardless — self-healed via the same
corrector-model retry already used for malformed tool arguments elsewhere, up to two attempts,
before giving up. The validated result flows back as `spawn_agent`'s own tool result (as pretty-
printed JSON), so the creator can use it directly — no `write_state`/`read_state` handoff protocol
to write and hope the model follows.

Recursive spawning (a spawned agent itself getting `spawn_agent`) is **off by default**
(`allow_recursive_spawning: false`). Turning it on lets a spawned agent spawn further agents up to
`max_spawn_depth`, but every generation inherits exactly the same `tool_pool` its parent had — no
privilege growth at any depth, ever.

## Unlocking a gated tool automatically: `spawns.on_complete`

A common pattern is sequencing a creator's own tool use around what its sub-agents find — e.g. a
booking tool that should only become available once research has actually happened
(`tools[].available_when`, see [Routing & Handoffs](./03_Agent_Handoffs_and_Routing)). Without
`on_complete`, unlocking that gate means telling the sub-agent, in its spawn instruction, to call
`write_state` itself — prose standing in for something `ai.yaml` should own. `spawns.on_complete`
closes that gap declaratively:

```yaml
spawns:
  tool_pool: ["create_itinerary"]
  result_schema: "schemas.ItineraryResult"
  on_complete:
    - key: research_done
      value: true
```

Once a spawned agent from this factory genuinely completes — `return_to_creator`, or a final text
response — each `{key, value}` pair is written to state automatically, through the exact same
`apply_state_write` pipeline `write_state` itself uses (so `state_schema` validation and any
declared `reducers` strategy for that key apply identically; this is not a second, less-validated
write path). It does **not** fire on a pause awaiting approval, and does not fire if the spawn is
forcefully stopped by `circuit_breakers.max_delegation_turns` — only on a genuine completion. The
sub-agent's own instruction never needs to mention `write_state` at all.

## Safety defaults, and why they're defaults

- **`requires_approval_on_first_action: true`** — a spawned agent's very first tool call pauses for
  human approval (the same `/resume` mechanism and multi-approver chains described in
  [Human-In-The-Loop](./07_Human_In_The_Loop)), regardless of whether that specific tool is itself
  `requires_approval`-gated. Turn it off only once you trust a given factory's blast radius.
- **`max_creations_per_session`** (default 3) — a session-wide circuit breaker (same family as
  `circuit_breakers.max_handoffs_per_session`), tripped as `IntaGrinError` so it genuinely halts
  the session rather than becoming ordinary LLM-visible text a model could just retry against.
- **A spawned agent's tool calls are still gated by every `requires_approval` your granted tools
  already carry** — two independent checks stack: the tool-name-keyed gate every tool already has,
  plus the first-action gate above.
- Circuit breakers, guardrails, and session budget are session-wide and agent-agnostic already —
  a spawned agent inherits all of them automatically, with no extra configuration.

## When a spawned agent's tool needs human approval

Since `spawn_agent` runs the child synchronously, a pause deep inside it (its own `requires_approval`
tool, or the first-action gate above) pauses the **whole parent session**, not just the child — the
parent's own turn halts with `status: "awaiting_approval"` in the API response, exactly as if the
parent itself had called an approval-gated tool directly. Resolve it the same way, via `POST
/resume` on the parent session — the framework finds and continues the right child session
automatically; you never need to know a child's session id exists.

If more than one spawned agent pauses in the same turn (e.g. several concurrent `spawn_agent` calls
whose children each hit a gate), only one is ever the *current* `_pending_approval` — the rest queue
behind it in arrival order and are surfaced one at a time as each prior one resolves. `queued_approvals`
in the `/resume`/`/chat`/`/stream` response tells you how many more are waiting behind the current
one, so nothing is silently lost, just serialized.

## What `inta verify` says about it

`inta verify` walks the statically-declared graph — it cannot see an agent that doesn't exist yet.
An agent with `spawns` configured shows up as a **non-deterministic surface**, reported the same
way `auto_route` already is: not part of the cycle/cost analysis, bounded by
`max_creations_per_session` and the star-topology constraint (not by graph acyclicity), and by the
engine's 10-iteration hard turn cap once a sub-agent exists. This is an honest scope boundary, not
a gap being papered over — see [Production Deployment](./04_Production_Deployment) for the full
picture of what static verification does and doesn't cover.

`inta fuzz` additionally red-teams `spawns`-configured agents specifically: multi-turn attacks
trying to get `spawn_agent` misused (an over-privileged tool request, a creation-cap bypass), with
the resulting agent's actual state inspected afterward — not just its reply text — since a rogue
agent could exist without ever saying anything alarming.

## Watching it live

A spawned agent doesn't exist in `ai.yaml`, so it can't appear in the Monitor dashboard's normal
graph — instead it shows up as a dashed, "Ephemeral" node the moment it's created, connected back
to its creator, and disappears again once it returns control or the session ends. Nothing is ever
written back to `ai.yaml`.
