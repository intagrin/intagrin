# Getting Started with IntaGrin

Welcome to **IntaGrin**, a Python framework for declarative agentic orchestration.

IntaGrin lets you build multi-agent systems using a single YAML file (`ai.yaml`) for
routing/tools/guardrails and vanilla Python functions for tool logic, instead of wiring the
control flow by hand in Python.

This page walks through exactly what you get from the scaffolding command, piece by piece, so you
understand every line before you start changing it — not just "install this, run that."

## 1. Installation & Setup

To scaffold your first project, run the CLI command:
```bash
inta new my_project
cd my_project
cp .env.example .env   # then fill in your API keys
```

This generates a ready-to-run, three-agent swarm:
```text
my_project/
├── ai.yaml                        # Your declarative blueprint
├── ai.schema.json                 # JSON Schema for ai.yaml — see below
├── .env.example
├── prompts/
│   ├── triage_prompt.jinja2
│   ├── support_prompt.jinja2
│   └── billing_prompt.jinja2
├── tools/
│   └── custom_tools.py
└── tests/
    └── evals.yaml
```

It's not a "hello world" — it's already a small, working multi-agent system with a handoff, a
delegation, a deterministic router, a local tool, and an MCP tool wired in. The rest of this page
explains each piece.

`ai.yaml`'s first line is `# yaml-language-server: $schema=./ai.schema.json` — a modeline any
editor with a YAML language server (VS Code's `redhat.vscode-yaml` extension, Neovim, most
JetBrains IDEs) picks up automatically, with no settings to configure. It gets you autocomplete
for every field in this doc, inline docs on hover, and a red squiggle under a typo'd key (e.g.
`hand0ffs:`) or a value of the wrong type — before you ever run `inta verify`. `ai.schema.json` is
generated from the exact same `AppConfig` Pydantic schema `inta verify`/the parser validate
against, so it can't drift out of sync with what's actually accepted; regenerate it for an
existing project with `uv run python -m intagrin.config.json_schema > ai.schema.json` after
upgrading IntaGrin, or just re-run `inta new` in a scratch dir and copy the file over.

## 2. What `inta new` just built

Open `ai.yaml`. Trimmed to the parts that matter:

```yaml
default_agent: "triage"

agents:
  triage:
    system_prompt_file: "prompts/triage_prompt.jinja2"
    handoffs: ["support", "billing"]
    # Deterministic routers bypass the LLM entirely when their condition is already true —
    # e.g. once some earlier turn's tool call has written state via write_state():
    #   routers:
    #     - condition: "user_status == 'banned'"
    #       target: "billing"

  support:
    system_prompt_file: "prompts/support_prompt.jinja2"
    delegations: ["billing"]
    tools:
      - name: "get_user_account"
        module: "tools.custom_tools"
      # MCP servers plug in the same way as a local tool — e.g. GitHub (needs Node's `npx` on
      # PATH and a GITHUB_PERSONAL_ACCESS_TOKEN in .env; not required for this starter project):
      #   - name: "mcp_github"
      #     type: "mcp"
      #     command: "npx"
      #     args: ["-y", "@modelcontextprotocol/server-github"]
      #     requires_approval: true

  billing:
    system_prompt_file: "prompts/billing_prompt.jinja2"

workflows:
  daily_audit:
    - name: "check_account"
      agent: "support"
      instruction: "Look up the account for user_id '123' and report its status."
    - name: "report_billing"
      agent: "billing"
      instruction: "Summarize the findings and wait."
```

Reading it top to bottom:

- **`default_agent: "triage"`** — every new conversation starts here. `triage_prompt.jinja2`
  (in `prompts/`) tells it to figure out what the user wants and transfer them.
- **`triage.handoffs: ["support", "billing"]`** — this is the LLM-driven control flow. IntaGrin
  automatically gives the `triage` agent a `transfer_agent` tool that can only target `support` or
  `billing` — the model decides *when*, you constrain *where*. See
  [Routing & Handoffs](./03_Agent_Handoffs_and_Routing) for the full mechanism.
- **The commented-out `routers` block** — a *deterministic* alternative to a handoff: if
  `condition` evaluates true against the shared state, the transfer happens instantly with **zero
  LLM calls**, before the model ever sees a prompt. It ships commented out, on purpose, rather
  than as a live router that can never fire: nothing in this starter project calls
  `write_state("user_status", "banned")`, so an uncommented version of this router would sit dead
  from turn one — evaluated every turn, never firing, and not erroring either (routers fail silently
  open when the state they check doesn't exist yet). Uncomment it once `triage` (or a tool) actually
  calls `write_state("user_status", ...)`, and run `inta why user_status` any time to check who
  currently reads or (best-effort) writes a key before you rely on it — this is exactly the gotcha
  that command exists to catch instead of leaving you to discover it by watching a handoff not
  happen. Conditions are bare state keys, not Python expressions — `user_status == 'banned'` works,
  `state.get('user_status') == 'banned'` does not (see
  [Routing & Handoffs](./03_Agent_Handoffs_and_Routing) for exactly what the condition grammar
  supports).
- **`support.delegations: ["billing"]`** — a second, different control-flow mechanism. Where a
  handoff *transfers* the conversation, a delegation spawns an isolated `billing` sub-agent to
  finish one task and **hand the result back to `support`**, which stays in control. Compare this
  to `triage`'s `handoffs: ["billing"]` — same target agent, two different relationships.
- **`support.tools`** — one local Python tool (`get_user_account`, implemented in
  `tools/custom_tools.py` — open it, it's a few lines with a docstring the framework turns into a
  JSON schema automatically), plus a commented-out example of wiring in a Model Context Protocol
  server (`mcp_github`) the same way — IntaGrin spins it up as a subprocess and talks to it over
  JSON-RPC, no hand-written wrapper code either way. It ships commented out because it needs
  Node's `npx` on `PATH` and a `GITHUB_PERSONAL_ACCESS_TOKEN`, neither of which this starter
  project requires; `inta doctor` checks that any MCP command you do uncomment is actually on
  `PATH` before you find out the hard way at runtime. Uncommented, it'd also carry
  `requires_approval: true`, pausing any call for a human to approve before it actually runs.
- **`workflows.daily_audit`** — a third shape entirely: an *autonomous* pipeline, not a chat. No
  human types anything; `inta run daily_audit` drives `support` then `billing` through fixed
  instructions in sequence.

That's three distinct ways agents hand off control in one file — conversational (`handoffs`),
sub-task (`delegations`), and scripted (`workflows`) — because real systems need all three, not just
one.

Don't want to re-read `ai.yaml` line-by-line every time, or need to hand this system to someone
who won't? `inta explain` prints exactly this kind of walkthrough automatically — agent roles,
tools, handoffs/delegations/routers, and the safety limits — built from the parsed config plus
each `description:` field, in plain English:

```bash
inta explain
```

It only has a name to go on for any agent (or the app itself) missing a `description:` —
`inta doctor` (below) flags those gaps so `inta explain`'s output doesn't stay full of blanks.

## 3. Run it

Before your first real run, `inta doctor` checks the environment for you in one pass instead of
one `inta dev` crash at a time — that an API key is set for `model.primary`, that `state_schema`
actually loads, that any MCP command you've wired in is on `PATH`, and that the app and every
agent has a `description:` set (a warning, not a failure — `inta explain` above is what actually
consumes it):

```bash
inta doctor
```

Haven't filled in an API key yet and just want to see the loop work first? Set `model.primary`
(or an agent's `model_override`) to `"mock/echo"` in `ai.yaml` — it replies with a canned,
clearly-labeled message and costs nothing, no network call made, so `inta doctor` reports it as
needing no key at all. Swap in a real model string once you're ready.

```bash
inta dev
```

This starts an interactive terminal chat loop against `triage`. Try:

```
You: I need help with my account, user 123
```

`triage` should call `transfer_agent` to hand you to `support`, which can then call
`get_user_account("123")` to answer you. Ask about billing instead and you'll land on `billing`
directly. Every turn — the handoff decision, the tool call, the final answer — prints to your
terminal as it happens; nothing here is a black box. If a router along the way evaluates but
doesn't fire (see the commented-out one above), that shows up inline too, right when it happens,
not just later in `inta replay`.

For a single non-interactive message instead of the chat loop — useful for scripting, or quick
iteration on a prompt edit — use `--once`:

```bash
inta dev --once "I need help with my account, user 123"
```

## 4. Where to go next

You now have a working multi-agent system and a mental model for how control passes between
agents. From here, each of these extends one specific piece of it:

| I want to... | Read |
| :--- | :--- |
| Understand handoffs/delegation/routing in depth, and what `inta verify`/`inta simulate` can prove about them | [Routing & Handoffs](./03_Agent_Handoffs_and_Routing) |
| Share structured data between agents instead of stuffing it into prompts | [Shared Typed State](./04_Shared_State_Redux) |
| Add more local tools or MCP servers, and understand tool-level access control | [Tools & MCP Integration](./05_Custom_Tools_and_MCP) |
| Let agents answer from my own documents | [Advanced RAG & HyDE](./06_Advanced_RAG_and_HyDE) |
| Require a human to approve a risky action, like `mcp_github` above | [Human-In-The-Loop](./07_Human_In_The_Loop) |
| Add cost ceilings, PII masking, and loop protection | [Security & Guardrails](./08_Security_and_Reliability) |
| Ship this behind a real API, and pool resources so it's cheap under load | [Production Deployment](./04_Production_Deployment) |

## 5. Launch & Monitor

Once you're ready to run for real (not just `inta dev`'s terminal loop):

```bash
inta serve     # FastAPI server: /chat, /chat/stream, /resume, /ws/voice
inta monitor   # live visual dashboard — agent graph, execution traces, token/cost burn
```

Open `http://localhost:3000` to watch handoffs, tool calls, and cost accumulate on your swarm in
real time as you (or `inta serve`'s API) drive it.
