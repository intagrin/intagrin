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

When the LLM tries to call `execute_sql`, the engine **instantly aborts execution**, suspends the current conversation state, and returns a pending status to the API.

**This requires a persistent `memory.type`** (`sqlite`, `postgres`, or `redis`) — the schema's own
default, `sliding_window` (in-process, lost the moment the request ends), can't actually suspend
anything durably. With it, the pause genuinely happens (the tool call really does return "paused
awaiting human approval") but there's no session left to resume from afterward, and Monitor's
dashboard has nothing to query either, so its Approve/Deny button never appears — `inta verify`
advises (never fails) when it finds a `requires_approval`/`required_approvers` tool alongside a
non-persistent `memory.type`.

## 2. Previewing Generated Media in Monitor

If a `requires_approval` tool's arguments (or a regular tool's own result) reference a generated
image, audio, or video file, `inta monitor`'s Playground renders it inline — the Approval card, a
tool-call bubble, and a tool-result bubble all detect a path ending in a recognized media
extension (`.png`/`.jpg`/`.jpeg`/`.gif`/`.webp`/`.svg` for images, `.mp3`/`.wav`/`.ogg`/`.m4a` for
audio, `.mp4`/`.webm`/`.mov` for video) and show it as a real `<img>`/`<audio>`/`<video>` element,
alongside — never instead of — the raw argument/result text. A tool doesn't need to do anything
special for this: return the file's path relative to the project root, exactly as you already
would.

```python
def review_content(caption: str, hashtags: list[str], image_path: str) -> str:
    """image_path: e.g. "generated_images/post_42.png", relative to the project root."""
    ...
```

This is served by `GET /api/files/{path}` (behind the same `verify_monitor_auth` as every other
Monitor endpoint), which only ever serves a fixed allowlist of media extensions from inside the
project directory — never a general file browser (the Architect chat's `read_file` tool already
covers arbitrary project files for an authenticated dashboard user) and never anything outside
`ai.yaml`'s own project root, regardless of a path containing `..` or being absolute.

## 3. Async Webhook Notifications
If your agents are running in background cron jobs, you need a way to know they are waiting. 
In `ai.yaml`:
```yaml
server:
  webhook_url: "https://hooks.slack.com/services/XXXX"
  webhook_secret_env_var: "SLACK_WEBHOOK_SECRET"
```
IntaGrin asynchronously POSTs the tool payload to the configured webhook, adding an `Authorization: Bearer <secret>` header when configured. Your webhook receiver can present an approval UI and call `/resume` with the reviewer decision.

## 4. The Resume API
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

## 5. One-Time Execution Exemptions
When a human approves a tool via `/resume`, IntaGrin grants a **one-time execution exemption** for that exact paused call — scoped to its `tool_call_id`, not just the tool's name. If an agent has two concurrent calls to the same `requires_approval` tool in one turn (e.g. two `refund_customer` calls for two different orders), approving one never lets the *other*, unapproved one ride along just because they share a name — each call needs its own approval.
Once the call runs, its exemption is removed from `_approved_tool_calls`. This prevents the LLM from re-triggering the same approved call again later in the same session without a fresh approval.

## 6. Dynamic Approval from Inside a Tool

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

## 7. Multi-Approver Chains

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
