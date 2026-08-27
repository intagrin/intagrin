---
name: intagrin-implement
description: Instructions for building IntaGrin projects
---

# AI Coding Assistant Rules for IntaGrin

You are generating code for an application built on `IntaGrin`, an Agent-Native orchestration framework.

## Core Philosophy
1. DO NOT write programmatic graph routing logic (no LangChain, no LangGraph).
2. ALL orchestration, routing, memory, and guardrails MUST be defined declaratively in `ai.yaml`.
3. ALL custom logic and API integrations MUST be written as vanilla Python functions in `tools/custom_tools.py` with type hints and docstrings.

## Discovery & Clarification Protocol
Before generating code or scaffolding, DO NOT assume defaults. Always clarify:
1. **Model Selection:** Preferred provider & model (`gemini/`, `openai/`, `anthropic/`).
2. **Tooling & Integrations:** Free web scraping/RSS vs. paid search APIs vs. vanilla Python tools vs. MCP servers.
3. **Execution Pattern:** Autonomous sequential pipeline (`workflows:`) vs. Interactive chat routing (`handoffs:`).
4. **Storage & Observability:** Memory backend (`sqlite` vs `postgres`) and telemetry (`otel`, `langfuse`).
5. **Authentication:** If `inta serve`/`inta monitor` will be exposed beyond localhost, confirm `server.auth.type` — `api_key` (shared secret) or `custom` (project-supplied `verify_token`). `none` means completely unauthenticated; don't leave it unstated for a deployment-bound project. If `api_key` is chosen, ALWAYS explicitly define `env_var` in `ai.yaml` (e.g., `env_var: "MY_API_KEY"`).
6. **RAG / Vector Retrieval:** If knowledge-base search is needed, confirm `docs_dir` and the embedding model/provider before adding a `rag:` block — don't default it silently.

## YAML Syntax & Schema Rules
- **Mandatory Root Fields:** `name`, `version`, and `default_agent` are required.
- **Model Provider Prefixes:** Always use LiteLLM format (e.g., `gemini/gemini-2.5-flash`, `openai/gpt-4o`).
- **Agents:** Defined under `agents:`. Each agent requires `description` and `system_prompt_file`. Control flow: use `handoffs: ["other"]` (LLM transfer), `delegations: ["sub"]` (sub-agents), `auto_route: true` (semantic swarm routing), `routers:` (deterministic bypass), or `spawns:` (dynamic runtime agent creation — see below).
  - **Choosing between `handoffs` and `delegations` (the most common mix-up):** ask "after this specialist finishes, who talks to the user next?" If it's the specialist, that's a `handoff` (it now owns the conversation, no return trip). If it's you, that's a `delegation` (runs to completion in an isolated child engine, returns a result as an ordinary tool result, your own turn is never interrupted).
  - **`delegations` vs `spawns`:** is the specialist already a named agent in `ai.yaml`? Use `delegations`. Only reach for `spawns` when you can't enumerate the specialists ahead of time (e.g. "one sub-agent per city, however many the user mentions").
  - **`routers` vs `handoffs`:** is this a deterministic fact already in state (`user_status == 'banned'`, zero LLM cost)? Use `routers`. Does it require judging the user's actual intent? Use `handoffs` instead — a router condition that needs interpretation is a sign you picked the wrong primitive.
  - If `inta copilot` set up a `references/docs/` bundle in this project, `03_Choosing_an_Orchestration_Primitive.md` there has the full decision table.
- **Tools:** Use `module: "tools.custom_tools"` for Python, `type: "mcp"` for MCP servers, or `type: "openapi"` with `url: "..."` to auto-wire REST APIs. Set `lazy_load_tools: true` on an agent to enable semantic tool retrieval for massive schema lists.
- **Workflows:** Defined under `workflows:`. An array of tasks with `name`, `agent`, and `instruction`. `type: "parallel"` fans out to `tasks: [...]` concurrently; `type: "vote"` does the same but aggregates branch answers into one consensus result (`vote: {strategy: majority|llm_judge}`) instead of hand-written voting logic.
- **State Merging:** Use `reducers:` at the root level to declaratively merge parallel state (`strategy: append|overwrite|deep_merge`). A key with no declared reducer still merges back on completion via a plain overwrite (last write wins) — declare a reducer only where overwrite isn't what you want.
- **Human-In-The-Loop:** `requires_approval: true` gates every call to a tool. For a runtime decision (only some calls need review), raise `AwaitingHumanInput(prompt="...", context={...})` from inside the tool instead — same pause/resume mechanism, decided in Python at call time. For N-of-M sign-off instead of one approver, use `required_approvers: ["finance", "security"]` (matching ids in `server.auth.approvers`) or a bare `required_approvals: 2`.
- **Keeping a value out of the LLM's own context entirely:** a tool's arguments and its result are both part of what the LLM sees on every later turn — `read_state`/`write_state` included, their results flow back like any other tool result. None of that hides a value (e.g. a raw email/phone number) from the LLM; `model.guardrails.mask_pii` masks PII broadly in message content, but a value the LLM itself supplied as a tool argument or read back is not masked from its own future turns. To let a tool use a value the LLM must never see, the value has to enter session state through a channel outside the chat conversation entirely (a dedicated API endpoint, a form submission, a webhook writing to state directly — never an LLM tool call), and the consuming tool's own Python function reads it from state internally in its implementation, without accepting it as an LLM-provided argument or returning it in the tool's result content.
- **Per-Caller Rate Limiting:** `server.rate_limit` (`max_requests_per_window`/`max_cost_per_caller_per_day`/`max_tokens_per_caller_per_day`) caps one authenticated caller's usage — unlimited by default.
- **A/B Model Routing:** `model.variants: [{model, weight}, ...]` splits traffic across weighted models instead of always using `model.primary` — sticky per session; a per-agent `model_override` still wins over a variant.
- **Cross-Session Shared Memory:** `memory.shared_scope: tenant|global` (default `session`) extends the long-term-memory summary beyond one session — last-write-wins, not a merge.
- **Dynamic Agent Creation:** `spawns:` on an agent gives it a `spawn_agent` tool that creates a new agent (new prompt, a subset of `spawns.tool_pool`, an inherited model) mid-session. `spawn_agent` does not transfer control — it runs the new agent to completion in an isolated child engine and returns the result as an ordinary tool result, so multiple `spawn_agent` calls in one turn can run concurrently with nothing shared to race on. `tool_pool` MUST be a subset of that agent's own `tools:` (enforced at parse time). A spawned agent only gets its granted tools plus a fixed `return_to_creator` — no handoffs/delegations of its own, no recursive spawning unless `allow_recursive_spawning: true`. Safe defaults: `requires_approval_on_first_action: true`, `max_creations_per_session: 3`. If a spawned agent's tool needs approval, the whole parent session pauses too and resolves via `POST /resume` on the parent's own session id.
- **Structured spawn results, not prose handoffs:** `spawns.result_schema` (dotted Pydantic model path) makes `return_to_creator`'s own tool schema derive from that model — the validated result comes back automatically as `spawn_agent`'s tool result. Do not write a spawned agent's instruction to call `write_state`/`read_state` to hand data to its creator when `result_schema` already covers it.
- **Unlocking a tool once a spawn finishes:** `spawns.on_complete: [{key: "...", value: ...}]` writes state automatically the moment a spawned agent genuinely completes (never on a pause or a forced turn-cap abort) — pair with `tools[].available_when` (a state condition, same grammar as `routers[].condition`, gating whether a tool is even offered this turn) instead of instructing a spawned agent to call `write_state` itself to unlock something for its creator.

Always prioritize updating `ai.yaml` when asked to add new capabilities.


## CRITICAL INSTRUCTION
You MUST read the deep architectural blueprint located at `.agents/skills/intagrin-implement/references/architecture.md` to understand how to write agents, evals, telemetry, and tools for this framework — it has an index of full topic pages under `.agents/skills/intagrin-implement/references/docs/`; read the specific page for your question rather than guessing from the index's one-line summary. If you see an error formatted like `[IG-XXX-000]` in output, tracebacks, or API responses, look it up in `.agents/skills/intagrin-implement/references/error_codes.md`. For any question about what's configurable in `ai.yaml` (authentication, memory, guardrails, circuit breakers, server, RAG, etc.), check `.agents/skills/intagrin-implement/references/config_reference.md` FIRST — it's the complete, generated field reference and almost always has the answer directly, without needing to explore the project's own files.