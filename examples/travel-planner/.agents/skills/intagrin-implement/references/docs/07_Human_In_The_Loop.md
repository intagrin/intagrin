# Human-In-The-Loop (HITL)

Enterprise applications cannot blindly trust AI to execute dangerous tools (like `refund_customer` or `drop_database`). IntaGrin provides a decoupled Human-in-the-Loop workflow: pause on a flagged tool, notify via webhook, resume via API once a human has reviewed (and optionally edited) the call.

## 1. Requiring Approval
In your `ai.yaml`, simply flag a tool:
```yaml
tools:
  - name: "execute_sql"
    module: "tools.db"
    requires_approval: true
```

When the LLM tries to call `execute_sql`, the engine **instantly aborts execution**, safely suspends the current conversation state to Postgres/SQLite, and returns a pending status to the API.

## 2. Async Webhook Notifications
If your agents are running in background cron jobs, you need a way to know they are waiting. 
In `ai.yaml`:
```yaml
server:
  webhook_url: "https://hooks.slack.com/services/XXXX"
  webhook_secret_env_var: "SLACK_WEBHOOK_SECRET"
```
IntaGrin asynchronously POSTs the tool payload to the configured webhook, adding an `Authorization: Bearer <secret>` header when configured. Your webhook receiver can present an approval UI and call `/resume` with the reviewer decision.

## 3. The Resume API
To resume the paused agent, your frontend (or Slack bot) hits the `/resume` REST API.

```json
POST /resume
{
  "session_id": "session456",
  "approved": true,
  "edited_args": {"query": "SELECT * FROM safe_table"}
}
```

**Powerful Feature:** Notice `edited_args`. If the LLM hallucinated a bad SQL query, the human reviewer can *edit* the arguments before approving it. The API directly executes those reviewed arguments exactly once, injects the tool result into the LLM context, and then continues the agent.

**Message threading:** the pause leaves a placeholder `tool` message ("...paused awaiting human
approval.") answering the original tool call. On resume, IntaGrin replaces that placeholder's
content with the real result in place — it does not append a second response to the same tool
call. This matters for providers (OpenAI in particular) that reject a message history where one
tool call is answered twice.

## 4. One-Time Execution Exemptions
When a human approves a tool via `/resume`, IntaGrin grants the agent a **one-time execution exemption** for that specific tool.
Once the tool runs, the exemption is removed from `_approved_tool_calls`. This prevents the LLM from re-triggering the same approved tool call again later in the same session without a fresh approval.

## 5. Dynamic Approval from Inside a Tool

`requires_approval: true` gates every call to a tool. Sometimes only *some* calls need a human —
the rest should just run. Raise `AwaitingHumanInput` from inside the tool's own Python body to
pause only that call, decided at runtime instead of declared statically in `ai.yaml`:

```python
from intagrin.errors import AwaitingHumanInput

def refund_customer(order_id: str, amount: float) -> str:
    if amount > 500:
        raise AwaitingHumanInput(
            prompt=f"Refund of ${amount} for order {order_id} exceeds the $500 auto-approve limit.",
            context={"order_id": order_id, "amount": amount},
        )
    # ... perform the refund ...
    return f"Refunded ${amount} for order {order_id}."
```

This pauses the session through the exact same mechanism as a statically gated tool — the same
`_pending_approval` state, the same `/resume` endpoint, the same webhook notification and message
threading described above. `prompt` and `context` are additive fields that ride along on
`pending_action`, so a reviewing client can show *why* this specific call paused instead of a
generic "approve this tool?" prompt.

**Resuming re-invokes the function from the start** (with `edited_args` if the reviewer supplied
them, otherwise the original `args`) — it is not a continuation that picks up mid-function. Avoid
non-idempotent side effects (charges, writes, external calls) before the point where you might
raise `AwaitingHumanInput`.

## 6. Multi-Approver Chains

By default a single `/resume` call with `approved: true` fully resolves a pending approval — one
approver is enough. For higher-risk tools, require several distinct approvers to each sign off
before the tool executes. Name your approvers and their credentials in `ai.yaml`:

```yaml
server:
  auth:
    approvers:
      finance: "FINANCE_APPROVER_KEY"
      security: "SECURITY_APPROVER_KEY"

tools:
  - name: "wire_transfer"
    module: "tools.banking"
    requires_approval: true
    required_approvers: ["finance", "security"]   # both must approve
```

Each approver calls `/resume` with their own `X-Approver-Key` header set to their credential:
```json
POST /resume
{"session_id": "session456", "approved": true}
```
with header `X-Approver-Key: <finance's secret>`. The response's `status` stays
`"awaiting_approval"` (and `pending_action` still reflects the paused call) until every required
approver has signed off — the tool only executes, and the agent loop only resumes, once the last
one calls `/resume`. A second `/resume` from the *same* approver id doesn't double-count.

If you only need a headcount rather than specific named approvers, use `required_approvals: 2`
instead of `required_approvers` — any 2 distinct approvers (from `server.auth.approvers`, or the
single `approver_env_var` approver counted as id `"default"`) satisfy it. `required_approvers`
takes precedence when both are set. Neither field changes anything for tools that don't set
them — `required_approvals` defaults to `1`, exactly today's single-approval behavior.
