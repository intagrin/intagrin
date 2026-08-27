# Security & Guardrails

IntaGrin includes several concrete mechanisms for limiting cost, loop, and content risk in an
agentic system — this page describes what each one actually does. See
[08_Security_Audit.md](./08_Security_Audit.md) for a dated third-party-style review of what's
remediated and what still needs deployment-time mitigation (network egress, prompt-injection
classification, etc.) — this framework reduces agentic risk, it doesn't eliminate the need for
your own security review.

## 1. Engine-Level Circuit Breakers
In `ai.yaml`, you can define strict runtime boundaries to prevent runaway costs, infinite handoff loops, and hallucination spirals:
```yaml
circuit_breakers:
  max_handoffs_per_session: 10   # defaults to 25 if omitted — never unlimited
  max_tool_failures_in_a_row: 3
  max_usd_cost_per_session: 1.50
  max_delegation_depth: 3        # defaults to 3
  max_delegation_turns: 15       # defaults to 15
```
If an agent exceeds these limits, the execution loop is instantly hard-killed. `inta verify` reports
which of these are backed by static graph analysis (handoffs, deterministic routers) versus only a
runtime cap (semantic `auto_route`, self-healing retries, memory compression) — read its output
before assuming a given config is fully bounded.

## 2. Self-Healing Error Loop Compression
If an LLM hallucinates bad python code, the tool will throw an error. IntaGrin contains an advanced `_compress_error_loops` garbage collector. If an agent fails 3 times in a row with the exact same error, IntaGrin deletes the redundant history and injects a hard system boundary into the prompt:
*"YOU ARE STUCK IN A LOOP. Formulate a completely different approach."*

## 3. Tenant Isolation (Anti-IDOR)
If you deploy IntaGrin behind a SaaS frontend, you must prevent User A from resuming User B's paused agent session. 
When custom authentication returns a stable tenant ID, the HTTP chat, streaming, resume, session-listing, monitor, and voice endpoints namespace their checkpoint IDs with that ID. Use custom authentication for multi-tenant deployments; `none` and `api_key` modes intentionally use the shared `global_tenant` namespace. Custom checkpointers remain responsible for applying the same isolation rule.

## 4. Trace-Based Swarm Evaluation (LLM-as-a-Judge)
Because IntaGrin flushes every turn to the SQLite checkpointer, you have a complete "flight data recorder" for every session.
You can rigorously evaluate complex, multi-agent trajectories offline using the CLI:
```bash
inta eval --judge --session-id "sess_123"
```
The framework retrieves the exact conversation trace from the database and feeds it to an LLM-as-a-judge to grade the swarm's efficiency and logical pathing.

## 5. Native Content Guardrails (PII & Toxicity)
Enterprise deployments cannot blindly pass user input to an LLM, nor can they blindly send LLM outputs to users. IntaGrin contains a native interceptor layer configured declaratively:
```yaml
model:
  primary: "openai/gpt-4o"
  guardrails:
    mask_pii: true
    system_safeguards: true
    banned_words: ["competitor_name", "ignore previous instructions"]
```
- `mask_pii`: Intercepts and redacts credit cards, emails, and SSNs before they leave your network.
- `system_safeguards`: Automatically injects strict boundary constraints into the agent's system prompt to prevent jailbreaks.
- You can even define a `custom_module: "security.my_guardrails"` to run custom Python interceptors!

## 6. Typed State & Response Validation
Malformed or wrongly-typed data in shared state is a common source of downstream tool failures.
Declare `state_schema: "schemas.AppState"` at the root and every `write_state` call is validated
against that Pydantic model before it's committed — a violating write is rejected and the
validation error is returned to the agent instead of corrupting state silently. Per-agent
`response_schema` does the same for an agent's final structured output, with one self-heal retry
via a corrector model before surfacing the validation failure.

## 7. OpenTelemetry (OTEL) Observability
Enterprise companies require tracing to debug LLM spans. IntaGrin integrates natively with OpenTelemetry (OTEL) and Langfuse. 
Simply add:
```yaml
telemetry:
  - "otel"
  - "langfuse"
```
The engine will automatically emit W3C OpenTelemetry spans for every LLM interaction and handoff event to your APM of choice.
