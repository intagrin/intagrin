# Agent Skills

Agent Skills are IntaGrin's `ai.yaml`-native answer to *context rot*: the well-documented pattern
where an LLM's tool choices and instruction-following degrade as its context window fills with
prompt bulk it doesn't need on every single turn. Instead of inlining a policy document, a style
guide, or a worked example into an agent's system prompt permanently, declare it as a **skill** —
a name and a one-line description the model sees cheaply, with the full content loaded into
context only when the model decides it actually needs it.

> **Not to be confused with:** `inta copilot`'s own "skill" bundle (`.claude/skills/`,
> `.cursor/rules/`, etc.) is an unrelated convention for teaching an *IDE coding agent* how to work
> on an IntaGrin project. This page is about the `skills:` key in `ai.yaml` — a runtime primitive
> your *deployed* agent uses, not a development-time tool.

## Declaring a skill

```yaml
skills:
  - name: refund_policy
    description: "How to evaluate and process a customer refund request"
    path: skills/refund_policy   # a directory, or a single .md/.txt file

agents:
  support:
    skills:
      - refund_policy
```

`path` is resolved relative to the project directory. It can point at:
- **A single file** — `load_skill` returns its full text content.
- **A directory** — `load_skill` looks for `SKILL.md` inside it first, falling back to any other
  `.md` file if `SKILL.md` isn't present. A directory can also hold additional resource files,
  readable via a second auto-registered tool, `read_skill_resource` (see below).

A skill referenced by an agent that isn't declared at the root — a typo'd name — is a parse-time
error (`ai.yaml` fails to load), not a silent no-op. A skill declared at the root whose `path`
doesn't exist on disk is caught by `inta verify`.

## What actually happens at runtime

When an agent's `skills:` list is non-empty, it automatically gets a `load_skill` tool. Its JSON
schema enum-constrains the `name` argument to exactly that agent's own declared skills, and its
*description* is built from every skill's own one-line `description` — so the "which skills exist
and when should I use one" information rides along in the tool schema itself, at the cost of one
tool definition, not a system-prompt paragraph that's always present whether or not it's relevant
this turn:

```json
{
  "name": "load_skill",
  "description": "Load the full instructions for one of your available Agent Skills — reusable domain guidance kept out of context until you decide you actually need it. Available skills:\nrefund_policy: How to evaluate and process a customer refund request",
  "parameters": {
    "properties": {
      "name": {"type": "string", "enum": ["refund_policy"]}
    }
  }
}
```

Calling `load_skill("refund_policy")` returns the skill's full body as an ordinary tool result —
it flows into the conversation exactly the way any other tool's return value does, with no new
context-injection mechanism to reason about. If the skill was never loaded, its full text never
enters context at all.

If any of an agent's skills is a directory (not a single file), a second tool,
`read_skill_resource(skill_name, resource_path)`, is also registered, for reading a file bundled
alongside the main skill instructions:

```
skills/refund_policy/
├── SKILL.md            # loaded by load_skill("refund_policy")
└── escalation_matrix.md  # read via read_skill_resource("refund_policy", "escalation_matrix.md")
```

`resource_path` is resolved relative to the skill's own directory and cannot escape it — a
`../../` traversal attempt is rejected, not silently followed.

## Skills that need to run code

A skill is deliberately just *text* — declarative instructions and reference material, not an
execution sandbox. If a skill's instructions describe a script the agent should actually run,
pair it with an existing [`type: "sandbox"` tool](./05_Custom_Tools_and_MCP) rather than expecting
IntaGrin to execute anything bundled inside the skill's own directory automatically.

## Gating a skill's availability

Like `tools[].available_when`, a skill reference can carry its own state-gated condition:

```yaml
agents:
  support:
    skills:
      - name: refund_policy
        available_when: "escalation_level >= 2"
```

The restricted condition grammar is identical to `routers[].condition`/`tools[].available_when`
(bare state-key comparisons, `and`/`or`/`not`, declared `condition_functions`) — see
[Choosing an Orchestration Primitive](./03_Choosing_an_Orchestration_Primitive) for the full
grammar. A gated skill simply doesn't appear in `load_skill`'s enum until its condition holds.

## Skills vs. tools vs. `lazy_load_tools`

These answer different questions and are meant to be combined, not chosen between:

| Question | Answer |
|---|---|
| Does the agent need to *do* something (call an API, run a calculation)? | `tools:` |
| Does the agent need to *know* something too bulky to keep in the system prompt every turn? | `skills:` |
| Does the agent have *many* already-declared tools and context is bloated by their sheer number? | `lazy_load_tools: true` |

See [Choosing an Orchestration Primitive](./03_Choosing_an_Orchestration_Primitive)'s "Related, but
a different axis" section for the full comparison.
