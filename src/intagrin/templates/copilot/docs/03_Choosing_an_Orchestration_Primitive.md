# Choosing an Orchestration Primitive

Six things move control between agents in `ai.yaml`: `handoffs`, `delegations`, `routers`,
`auto_route`, `spawns`, and `workflows`. They look similar in a config file but answer different
questions. Picking by the *question you're actually asking*, not by which one you remember first,
is the fast way to the right answer.

## Decision table

| You want to... | Use |
|---|---|
| Hand the whole conversation to a specialist, who now owns it | `handoffs` |
| Ask a specialist to do one sub-task and hand you back a result, then keep going yourself | `delegations` (`delegate_task`) |
| Run N instances of one already-known specialist concurrently, item count decided at runtime | `delegations` (`delegate_to_many`) |
| Skip the LLM entirely and branch on a fact already sitting in state | `routers` |
| Branch on logic too rich for bare comparisons/and/or/not, without an LLM handoff | `routers` + `condition_functions` |
| Let a loose group of agents pick whoever fits next, with no fixed graph of who-can-reach-whom | `auto_route` |
| Create a brand-new, narrowly-scoped agent at runtime for a task whose *specialist* you couldn't enumerate ahead of time | `spawns` |
| Run a fixed, non-conversational multi-step pipeline (`inta run <name>`), not a chat turn | `workflows` |

## Each one, in detail

**`handoffs: ["target_agent"]`** — declared on the agent giving up control. Compiles to a real
`transfer_agent` tool the LLM decides to call; once it fires, `target_agent` owns the
conversation from that point on — there's no implicit return trip. Use this for "the user's
request is really billing's problem now, not mine."

**`delegations: ["sub_agent"]`** — declared on the agent staying in control. Compiles to a
`delegate_task` tool that runs `sub_agent` to completion in an isolated child engine and returns
its result as an ordinary tool result — the delegating agent's own turn is never interrupted. Use
this for "I need an answer from the billing specialist before I can finish responding," not "hand
this whole thing off." The same `delegations:` list also compiles a `delegate_to_many` tool —
fan out N concurrent instances of one sub-agent, one per instruction, for a task whose item count
is only known at runtime (e.g. one sub-task per city the user mentioned); capped by
`circuit_breakers.max_parallel_fan_out` (default 10) to keep an LLM-chosen fan-out width bounded.

**`routers: [{condition: "...", target: "..."}]`** — deterministic, evaluated *before* the LLM
runs each turn, using a restricted, safe expression grammar (bare state-key names, comparisons,
`and`/`or`/`not` — no method calls, no `state.get(...)`). If a condition is already true from
earlier state, routing happens with zero LLM cost and zero judgment call. Use this when the
decision is really a fact lookup ("`user_status == 'banned'` → route to `compliance`"), not
something that needs interpretation. Don't reach for `routers` when the actual decision requires
reading the user's intent — that's what `handoffs` is for.

**`auto_route: true`** — semantic swarm routing: a lightweight LLM call picks the next agent from
a pool by matching the message against each candidate's `description`, with no `handoffs:` edges
declared between them. Use this for a loose "group chat" of peers where you don't want to
hand-wire every possible transition — the tradeoff is you give up `inta verify`'s ability to
reason about the graph shape, since there isn't a fixed one.

**`spawns:`** — the one primitive that creates an agent that didn't exist in `ai.yaml` at all.
Gives the declaring agent a `spawn_agent` tool that builds a new, narrowly-scoped specialist
mid-session (a subset of the declaring agent's own tools, parse-time-enforced so it can never
escalate privilege), runs it to completion in isolation, and returns the result — same
non-interrupting shape as `delegations`, but for a task where the *specialist itself* couldn't be
enumerated as a named agent ahead of time (e.g. a bespoke system prompt tailored to one user's
one-off request). Don't reach for this just because the *item count* is runtime-determined — "one
sub-agent per city in this trip" is `delegate_to_many` against an already-known agent, not
`spawns`, as long as every city needs the same kind of specialist. This is the one mechanism
`inta verify` cannot fully statically verify — reach for `delegations` instead whenever the set of
specialists you need is actually known upfront; `spawns` is for when it isn't.

**`workflows:`** — not part of the conversational turn loop at all. A named, fixed sequence of
tasks (`sequential`/`parallel`/`vote`) run via `inta run <workflow_name>`, for batch/autonomous
jobs with no user turn-by-turn interaction. Use this for "process these 500 records overnight,"
not "respond to what the user just said."

## Common confusions

- **`handoffs` vs `delegations`**: does the receiving agent *own* the rest of the conversation
  (`handoffs`), or does control come back to you with a result (`delegations`)? This is the single
  most common mix-up — if you're not sure, ask "after this specialist finishes, who talks to the
  user next?" If it's the specialist, that's a handoff. If it's you, that's a delegation.
- **`delegations` vs `spawns`**: is the specialist already a named agent in `ai.yaml` (`delegations`),
  or does it need to be invented on the spot for a task whose shape you can't predict at config
  time (`spawns`)? Default to `delegations` — only reach for `spawns` when you genuinely can't
  enumerate the specialists ahead of time.
- **`routers` vs `handoffs`**: is this a deterministic fact already in state (`routers`, zero LLM
  cost), or does it require judging the user's actual intent (`handoffs`, an LLM decision)? A
  router condition that references something the LLM would need to interpret is a sign you want
  `handoffs` instead.
- **`delegate_to_many` vs `spawns`**: is every item handled by the *same kind* of specialist
  (`delegate_to_many` — one already-known agent, run N times), or does each item actually need a
  *different* system prompt/tool set decided at runtime (`spawns`)? Item count alone being dynamic
  is not a reason to reach for `spawns`.
- **bare `routers` vs `routers` + `condition_functions`**: can the decision be written as a plain
  comparison on one or two state keys (bare `routers`), or does it need real logic — a regex, a
  multi-field rule, anything with a method call (`condition_functions`)? Both stay zero-LLM-cost
  and deterministic; `condition_functions` is not a step toward `handoffs`, it's a wider net for
  the same "skip the LLM" use case.

## Related, but a different axis: `skills` vs `tools` vs `lazy_load_tools`

Everything above is about *who talks to the user next* — a different question from *what an agent
can see/do this turn*. These three are complementary, not competing, and often used together:

| You want to... | Use |
|---|---|
| Give an agent a structured callable with typed arguments it invokes to take an action or fetch data | `tools:` |
| Keep reusable domain instructions/reference material out of context until the agent decides it actually needs them | `skills:` (Agent Skills — see `load_skill`) |
| Reduce context bloat from an agent having *many already-declared* `tools:` by semantically filtering which ones are shown each turn | `lazy_load_tools: true` |

`tools:` is for doing something (calling an API, running a calculation) — the LLM gets a schema
and a return value. `skills:` is for knowing something (a policy, a style guide, a worked example)
too bulky to keep in the system prompt on every turn — the LLM gets a name + one-line description
up front (via the auto-registered `load_skill` tool's schema) and pulls in the full text only when
it decides to. `lazy_load_tools` doesn't add new capabilities at all — it's a semantic pre-filter
over tools an agent already has, for when the sheer number of declared tools (not their content)
is what's bloating context. An agent can use all three at once: a modest `tools:` list,
`lazy_load_tools: true` because that list is long, and `skills:` for reference material none of
those tools need to have inlined into every prompt.
