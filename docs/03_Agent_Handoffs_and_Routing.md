# Routing & Handoffs

Unlike traditional group chat frameworks where agents debate endlessly, IntaGrin uses a deterministic state-machine handoff model. There is only ever one active agent. When that agent is done, it hands off the user to the next agent.

There are two ways to achieve this:

## 1. LLM-Driven Handoffs (Tool Based)
You can declare `handoffs` in `ai.yaml`. The framework will automatically register a `transfer_agent` tool and inject it into the agent's context. The LLM decides when to call it.

```yaml
agents:
  triage:
    description: "Routes users to the correct department"
    handoffs:
      - "billing"
      - "support"
```

If the user says "I need help with my invoice", the `triage` agent will dynamically call `transfer_agent(target_agent="billing")`.

## 2. Semantic Swarm Routing (LLM-Bypass Group Chat)
For classic "Group Chat" or Swarm behavior, you can enable `auto_route: true`. Instead of filling the main LLM's context window with the agent directory and burning massive tokens, IntaGrin intercepts the execution loop when an agent finishes speaking. It uses a cheap fallback model (e.g., `gemini-1.5-flash`) to semantically evaluate the last message and dynamically route to the next best agent.

```yaml
agents:
  manager:
    auto_route: true
```

## 3. Hierarchical Delegation
If an agent needs a sub-task completed without losing control of the conversation, use `delegations: ["sub_agent"]`. The main agent will spawn an isolated child engine (which shares the Typed State) and wait for the result.

```yaml
agents:
  manager:
    delegations:
      - "researcher"
```

## 4. Deterministic Conditional Routing (Procedural)
In highly regulated enterprise environments, you cannot always trust an LLM to route a conversation correctly. For strict compliance, you can define **Deterministic Conditional Routers**.

These execute *before* the LLM even sees the prompt. If the condition evaluates to true, the handoff happens instantaneously, costing you $0 in API tokens.

```yaml
agents:
  sales:
    routers:
      - condition: "user_balance < 0"
        target: "collections"
```

In this example, the engine evaluates the python condition against the global Shared State. If the user's balance is negative, the engine instantaneously transfers the session to the `collections` agent. Conditions are evaluated by a restricted AST walker (literals, comparisons, boolean logic, `in`/`not in`) — not Python's `eval()` — so this is safe to expose as user-authored config. That safety comes with a real constraint: reference state keys as bare names (`user_balance`, `intent`), not via method calls or attribute access — `state.get("user_balance", 0)` is **not** supported and raises `Unsupported syntax` at evaluation time (silently logged, the router just never fires). A missing key isn't `None` either; it raises `Unknown variable`, so guard with a boolean flag your tools always set (e.g. `has_balance and user_balance < 0`) rather than assuming a default.

Note the constraint is genuine Python syntax underneath — booleans are `True`/`False` (capitalized), not YAML's `true`/`false`; `research_done == True` works, `research_done == true` raises `Unknown variable: true` (it parses as a bare name, not a boolean literal). Simplest is usually to skip the comparison entirely and just reference the flag: `research_done`.

### Gating a tool's availability: `available_when`

The same restricted grammar also gates whether a *tool* is even offered to an agent this turn — useful for sequencing a single agent's own tool use (e.g. "don't offer `book_flight` until research is done"), as opposed to routing between agents:

```yaml
agents:
  planner:
    tools:
      - name: create_itinerary
      - name: book_flight
        available_when: "research_done"
```

This is qualitatively different from a prompt instruction asking the model to wait — the tool is structurally absent from `planner`'s schema until the condition is true, and the engine re-checks the condition again at execution time regardless of what schema the model was actually offered (the same defense-in-depth already applied to `spawns.tool_pool` and every other schema-driven gate). One deliberate asymmetry from `routers[].condition`: a malformed `available_when` expression fails **closed** — it hides/rejects the tool rather than silently skipping like a broken router does — since the whole point of the gate is to withhold access until it should be granted. `inta verify` and `validate_config_dict` (the same gate `inta compile` and Studio edits go through) both statically check `available_when` syntax alongside router conditions, so a typo like the `true`/`True` one above is caught before it ships, not discovered as a permanently-missing tool at runtime.

Fail-closed doesn't mean noisy: a condition referencing a state key that simply hasn't been set yet — the normal shape of `research_done` before any research has happened — hides the tool quietly, with nothing logged. Only a genuine syntax problem (the kind `inta verify` would have already caught) logs an error at runtime.

`available_when` is available on every tool-declaration shape — inline `LocalToolConfig`/`MCPToolConfig`/`OpenAPIToolConfig`, and the plain `- name: <tool>` reference shape most `agents.<name>.tools:` entries actually use — since availability is inherently per-agent (the same globally-declared tool might be gated for one agent and unrestricted for another).

## 5. Dynamic Agent Creation (`spawns`)

All four mechanisms above route between agents you declared in `ai.yaml` ahead of time. `spawns`
is a different, more powerful (and more carefully bounded) primitive: it lets an agent create a
brand-new agent — new prompt, new tool subset, new model — while a session is running. It's
substantial enough to warrant its own page: see
[Dynamic Agent Spawning](./03_Dynamic_Agent_Spawning).

Combining deterministic routers for compliance-critical transitions with LLM-driven handoffs for
judgment calls gives you a control point most pure-LLM-routing frameworks don't: `inta verify` can
statically prove the deterministic parts of your graph are acyclic. It can't do the same for
`auto_route` or LLM-driven handoffs — those are still runtime decisions, bounded by the engine's
turn cap rather than by static analysis.

`inta verify` proves your router *graph* is well-formed before you ever run it. It doesn't tell you
what a graph *change* does to traffic you already have. For that, `inta simulate --config
ai.yaml.new` replays real checkpointed sessions through a candidate router/circuit-breaker change
and reports exactly which ones would now route, trip a breaker, or gate a tool differently — with
zero new LLM calls, since a deterministic router's decision is a pure function of state and doesn't
need one. See [Production Deployment](./04_Production_Deployment.md) for the full picture of what
it can and can't evaluate.
