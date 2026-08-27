# The ai.yaml Blueprint

The heart of IntaGrin is the `ai.yaml` file. Instead of writing messy Python code to wire up agents, you declare your entire system architecture here.

Here is a full production-ready example:

```yaml
name: "finance-swarm"
version: "1.0"
default_agent: "triage"
max_session_budget_usd: 5.00  # Global circuit breaker
state_schema: "schemas.UserState" # Typed Shared State

model:
  primary: "openai/gpt-4o"
  fallback: "gemini/gemini-1.5-pro"
  temperature: 0.1
  use_cache: true               # Semantic Caching
  guardrails:
    mask_pii: true

memory:
  type: "postgres"
  env_var: "DATABASE_URL"
  max_messages: 50

server:
  webhook_url: "https://hooks.slack.com/services/XXXX"
  webhook_secret_env_var: "SLACK_SECRET"
  auth:
    type: "custom"
    custom_module: "auth.jwt_verifier"

agents:
  triage:
    description: "Routes users to the correct department"
    system_prompt_file: "prompts/triage.jinja2"
    handoffs:
      - "billing"
      - "support"
```

## Global Configurations

- `max_session_budget_usd`: IntaGrin monitors token usage automatically via LiteLLM. If the session exceeds this dollar amount (e.g., during a malicious infinite loop), the engine safely aborts.
- `state_schema`: A dotted path to a Pydantic `BaseModel` (e.g. `"schemas.UserState"`), used two ways: (1) the full state is injected into each agent's prompt so the model can see it, and (2) every `write_state` call validates the resulting state against this model before committing it. A write that violates the schema is rejected — the real state is left untouched — and the validation error is returned to the calling agent so it can retry with corrected types.
- `response_schema` (per-agent, under `agents.<name>.response_schema`): Same idea for an agent's final structured output — a dotted path to a Pydantic model. When the agent's terminal response fails validation, IntaGrin asks a fast corrector model to fix it once before giving up and surfacing the validation error.
- `use_cache`: Passes LiteLLM's caching option to LLM completions. Configure a LiteLLM cache backend before relying on cache hits; RAG document and embedding lookups are not cached by this setting.

## A/B & Canary Model Routing

`model.variants` splits traffic across weighted model options instead of every session using
`primary`:
```yaml
model:
  primary: "openai/gpt-4o-mini"   # used only if variants is unset
  variants:
    - model: "openai/gpt-4o-mini"
      weight: 3
    - model: "openai/gpt-4o"
      weight: 1
```
Assignment is deterministic per `session_id` (a weighted hash, not a live coin flip) — a session
is assigned once, on its first turn, and stays on that variant for its whole conversation, even
across a checkpoint reload. Different sessions land on different variants according to the
configured weights (here, roughly 3:1 toward `gpt-4o-mini`). The Monitor dashboard's **Logs** page
records cost/tokens per session, so you can compare cost across sessions even though it doesn't
(yet) record which variant a given session used — cross-reference `session_id` against your own
records if you need a precise per-variant cost breakdown. An explicit per-agent `model_override`
still wins over a variant assignment — variants only apply where an agent hasn't hardcoded its own
model.

## The Memory Block (Checkpointers)
IntaGrin supports truly stateless API execution by flushing conversation history to a database.
Supported types:
- `sqlite`: Great for local development (`.ai/memory.db`).
- `postgres`: Enterprise-grade persistence. Provide a connection URL or use `env_var`.
- `redis`: Ultra-fast persistence for serverless environments.

## The Server Block
This block configures the internal FastAPI engine.
- You can configure custom authentication (which enables Tenant ID isolation and IDOR protection).
- You can configure Async Webhooks for Human-in-the-Loop notifications.

## YAML Imports (Modularity)
As your architecture scales to dozens of agents, you can split your `ai.yaml` into modular sub-graphs using the `imports` block.
```yaml
imports:
  - path: "departments/billing.yaml"
    namespace: "billing_"
```
During compilation, the engine will safely merge all sub-agents, tools, and workflows into the global AST graph and prefix their names to prevent collisions.
