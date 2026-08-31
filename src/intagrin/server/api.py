import asyncio
import importlib
import json
import os
import secrets
import time
from pathlib import Path
from typing import Any

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
    Security,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from rich.console import Console

from ..compiler.parser import parse_project
from ..errors import IntaGrinError
from ..runtime.approvers import add_approver, list_approvers, revoke_approver
from ..runtime.approvers import verify_secret as verify_approver_secret
from ..runtime.engine import RuntimeEngine, extract_final_answer
from ..runtime.rate_limiter import check_rate_limit
from ..runtime.run_logger import record_run_log
from ..runtime.session_locks import get_session_lock_registry
from ..runtime.shared_resources import get_shared_resources_cache
from ..tracing.console import Tracer
from .error_handlers import register_intagrin_error_handlers

console = Console()
app = FastAPI(title="IntaGrin Production Server")
register_intagrin_error_handlers(app)


@app.get("/health")
def health_check():
    """Plain, unauthenticated liveness check — deliberately returns no project/session data, just
    confirms the process is up. Used by the Dockerfile's HEALTHCHECK and any load balancer."""
    return {"status": "ok"}
@app.on_event("startup")
async def startup_event():
    from intagrin.db_migrations.auto_migrate import run_auto_migrations
    await asyncio.to_thread(run_auto_migrations)

    try:
        graph = parse_project(Path.cwd())
        if graph.config.server.auth.type == "none":
            console.print(
                "[bold yellow]⚠ Production server is running with server.auth.type: \"none\" — "
                "/chat, /stream, and /resume are unauthenticated. Set `server: {auth: {type: "
                "api_key}}` in ai.yaml before exposing this beyond localhost.[/bold yellow]"
            )
        has_approval_gated_tools = any(
            getattr(t, "requires_approval", False)
            for agent_cfg in graph.config.agents.values()
            for t in agent_cfg.tools
        )
        has_db_approvers = any(
            not row.get("revoked_at")
            for row in list_approvers(graph.config.memory, Path.cwd())
        )
        if has_approval_gated_tools and not (
            graph.config.server.auth.approver_env_var
            or graph.config.server.auth.approvers
            or has_db_approvers
        ):
            console.print(
                "[bold yellow]⚠ One or more tools declare requires_approval: true, but no "
                "approver credential is configured — the same credential that triggers a gated "
                "tool call can immediately approve it via /resume with no separate review. Set "
                "`server: {auth: {approver_env_var: ...}}` (single approver) or `approvers: "
                "{...}` (named/multi-approver) in ai.yaml, or run `inta approvers add <id>` to "
                "issue a DB-backed reviewer credential without an ai.yaml/.env edit.[/bold yellow]"
            )
    except Exception:
        pass


@app.on_event("shutdown")
async def shutdown_event():
    """Tear down pooled MCP server connections (SharedResourcesCache) at process exit. Requests
    no longer clean up their own engine's mcp_manager — it's shared across requests, so per-request
    cleanup would kill connections other in-flight or future requests still need."""
    await get_shared_resources_cache().cleanup()


security = HTTPBearer(auto_error=False)


def authenticate_token(token: str | None) -> str:
    """Authenticate an HTTP or WebSocket client and return its tenant namespace."""
    project_dir = Path.cwd()
    graph = parse_project(project_dir)
    auth_cfg = graph.config.server.auth

    if auth_cfg.type == "none":
        return "global_tenant"

    if not token:
        raise HTTPException(status_code=401, detail="Missing authentication")

    if auth_cfg.type == "api_key":
        expected = os.environ.get(auth_cfg.env_var)
        if not expected or not secrets.compare_digest(token, expected):
            raise HTTPException(status_code=401, detail="Invalid API Key")
        return "global_tenant"

    if auth_cfg.type == "custom":
        try:
            import sys

            if str(project_dir) not in sys.path:
                sys.path.insert(0, str(project_dir))
            mod = importlib.import_module(auth_cfg.custom_module)
            result = mod.verify_token(token)
            if not result:
                raise HTTPException(status_code=401, detail="Invalid Custom Token")
            # Fail closed, not open: a verify_token that returns a non-string truthy value
            # (True, a user object, an int ID — all common, reasonable implementations) used to
            # silently collapse every authenticated user into "global_tenant", defeating
            # multi-tenant session isolation with no error. A custom auth module must return the
            # tenant id explicitly as a string.
            if not isinstance(result, str):
                raise HTTPException(
                    status_code=401,
                    detail=(
                        "Custom auth module's verify_token() must return the tenant id as a "
                        "string, not a truthy non-string value — refusing to guess a tenant "
                        "namespace."
                    ),
                )
            return result
        except HTTPException:
            raise
        except Exception as e:
            Tracer.log_error(f"Custom Auth Error: {e}")
            raise HTTPException(status_code=500, detail=f"Custom Auth Error: {e}")


def verify_auth(credentials: HTTPAuthorizationCredentials = Security(security)):
    token = credentials.credentials if credentials else None
    return authenticate_token(token)


def verify_admin_auth(credentials: HTTPAuthorizationCredentials = Security(security)) -> None:
    """Gates the /approvers management endpoints (issue/list/revoke DB-backed reviewer
    credentials over HTTP — see runtime/approvers.py). Deliberately its own, even-more-privileged
    credential tier, checked against `server.auth.admin_env_var` — separate from both the main
    session auth (verify_auth) and any individual approver's own X-Approver-Key. Without this
    separation, whoever holds the main API key could hit this endpoint, mint themselves an
    approver credential, and immediately approve their own gated tool call — exactly the hole
    approver_env_var/approvers already closes for /resume itself.

    Unlike identify_approver's env-var path, there is no "if unset, same credential works"
    fallback: these endpoints didn't exist before, so the safe default is closed (503), not open."""
    graph = parse_project(Path.cwd())
    admin_env_var = graph.config.server.auth.admin_env_var
    if not admin_env_var:
        raise HTTPException(
            status_code=503,
            detail="Approver-management API is disabled. Set server.auth.admin_env_var in ai.yaml to enable it.",
        )
    expected = os.environ.get(admin_env_var)
    token = credentials.credentials if credentials else None
    if not expected or not token or not secrets.compare_digest(token, expected):
        raise HTTPException(status_code=401, detail="Missing or invalid admin credential.")


def identify_approver(request: Request, graph, project_dir: Path) -> str:
    """Checked only when /resume is approving (not denying) a requires_approval tool call —
    separately from the requester's own session auth via verify_auth. Without this, the same
    credential that triggered a gated call can immediately approve it, which is not a real
    privilege boundary. Opt-in via server.auth.approver_env_var / server.auth.approvers, or via
    DB-backed credentials issued through `inta approvers add` (runtime/approvers.py) — see the
    startup warning in startup_event when tools declare requires_approval but none of the three is
    set — when none is configured, today's behavior (same credential approves) is preserved rather
    than breaking existing deployments, and this returns the fixed id "default" without checking
    any header.

    Returns the id of the approver whose X-Approver-Key secret matched, so /resume can track
    which of a multi-approver chain's required approvers has signed off (see
    RuntimeEngine._pause_for_human's required_approvals/required_approvers). approver_env_var, if
    set, is checked as the implicit approver id "default" alongside any named `approvers`.

    DB-backed credentials are checked first (they're the deployment-recommended path — issued and
    revoked at runtime with no plaintext secret in the environment, see runtime/approvers.py's
    module docstring) and env-var candidates second — a project can use either or both. An
    optional X-Approver-Id header lets a caller that already knows its own approver_id skip
    verify_secret's per-row scrypt scan (see its docstring); omitting it still works, just without
    that optimization."""
    auth_cfg = graph.config.server.auth
    approver_env_var = auth_cfg.approver_env_var
    approvers = auth_cfg.approvers
    if not isinstance(approvers, dict):
        approvers = {}

    candidates = dict(approvers)
    if approver_env_var:
        candidates.setdefault("default", approver_env_var)

    provided = request.headers.get("X-Approver-Key")
    has_db_approvers = any(
        not row.get("revoked_at")
        for row in list_approvers(graph.config.memory, project_dir)
    )

    if not candidates and not has_db_approvers:
        return "default"

    if provided:
        id_hint = request.headers.get("X-Approver-Id")
        db_match = verify_approver_secret(graph.config.memory, project_dir, provided, id_hint)
        if db_match:
            return db_match
        for approver_id, env_var in candidates.items():
            expected = os.environ.get(env_var)
            if expected and secrets.compare_digest(provided, expected):
                return approver_id

    raise HTTPException(
        status_code=403,
        detail="Missing or invalid X-Approver-Key — approving a gated tool call requires a separate reviewer credential.",
    )



class ChatRequest(BaseModel):
    message: str | list[dict[str, Any]]
    session_id: str
    initial_state: dict = None


class ChatResponse(BaseModel):
    response: str
    active_agent: str
    status: str = "completed"
    pending_action: dict | None = None
    # How many more pauses are waiting behind pending_action (see _set_pending_approval's
    # queueing in runtime/engine.py) — without this, a caller has no way to know another
    # approval is coming next short of reading the raw checkpoint state directly.
    queued_approvals: int = 0


class ResumeRequest(BaseModel):
    session_id: str
    approved: bool
    reviewer_notes: str | None = None
    edited_args: dict | None = None


async def _log_run(
    graph,
    project_dir: Path,
    engine: RuntimeEngine,
    endpoint: str,
    session_id: str,
    status: str,
    error: str | None,
    pre_metrics: dict,
    run_start: float,
) -> None:
    """Shared per-call audit-log write for all four API endpoints — computes the delta/cumulative
    metrics and latency the same way at every call site, then hands off to record_run_log (which
    is itself best-effort and never raises)."""
    metrics = engine.state.get("_metrics", {}) or {}
    await asyncio.to_thread(
        record_run_log,
        graph.config.memory,
        project_dir,
        session_id=session_id,
        endpoint=endpoint,
        agent=engine.active_agent_name,
        status=status,
        error=error,
        tokens_delta=metrics.get("total_tokens", 0) - pre_metrics.get("total_tokens", 0),
        cost_delta=metrics.get("total_cost", 0.0) - pre_metrics.get("total_cost", 0.0),
        total_tokens=metrics.get("total_tokens", 0),
        total_cost=metrics.get("total_cost", 0.0),
        message_count=len(engine.messages),
        latency_ms=int((time.monotonic() - run_start) * 1000),
    )


async def _record_turn_failure(
    engine: RuntimeEngine | None, graph, error: Exception
) -> None:
    """Persists a clear, human-readable trace of an unhandled mid-turn exception directly into
    the checkpointed conversation. Without this, run_logs (the audit table behind the admin-only
    GET /api/logs) was the only place a genuine crash ever got recorded — a *normal* tool
    execution error (a rate limit, a bad argument) is already caught inside the turn loop itself
    and turned into a visible tool-result message the LLM gets to respond to, so this is
    specifically for the rarer case that escapes that: something breaks badly enough to abort the
    whole turn. Before this, anyone reviewing the session afterward — Monitor's Playground,
    `inta replay`, a reopened tab — saw nothing wrong, just an abrupt stop with no explanation.

    Best-effort and defensive on purpose: `engine`/`graph` can legitimately still be None if
    construction itself failed before either was assigned, and a failure while trying to persist
    this (e.g. the checkpoint backend is what's actually down) must never mask the original error
    from reaching the caller through the existing SSE/HTTPException paths — it's only ever logged,
    never re-raised."""
    if engine is None or graph is None:
        return
    try:
        summary = str(error)
        if len(summary) > 500:
            summary = summary[:500] + "... (truncated — see server logs for the full error)"
        engine.messages.append(
            {
                "role": "assistant",
                "content": (
                    f"⚠️ This request could not be completed due to an unexpected error: "
                    f"{summary}\n\nPlease try again."
                ),
            }
        )
        engine._save_checkpoint()
        await engine._await_last_checkpoint()
    except Exception as record_error:
        Tracer.log_error(f"Failed to persist turn-failure record: {record_error}")


def _check_approval_satisfied(
    pending_action: dict, approver_id: str | None
) -> tuple[bool, list[str] | None, list[str]]:
    """Records approver_id into pending_action's approvals_received (mutates in place) and
    returns (satisfied, outstanding, approvals_received) per required_approvers/
    required_approvals — the N-of-M approval-chain check shared by a direct resume and a resume
    that reaches into a spawned sub-agent's own pending approval (see _resume_nested_child_approval
    below)."""
    approvals_received = pending_action.setdefault("approvals_received", [])
    if approver_id not in approvals_received:
        approvals_received.append(approver_id)

    required_approvers = pending_action.get("required_approvers")
    required_approvals = pending_action.get("required_approvals", 1)
    if required_approvers:
        satisfied = all(a in approvals_received for a in required_approvers)
        outstanding = [a for a in required_approvers if a not in approvals_received]
    else:
        satisfied = len(approvals_received) >= required_approvals
        outstanding = None
    return satisfied, outstanding, approvals_received


def _pending_approval_block(engine: RuntimeEngine) -> dict | None:
    """/chat, /chat/stream, and /stream must not silently start a brand-new LLM turn on top of an
    already-unresolved _pending_approval — none of them checked this before, so a plain chat
    message sent while a session was paused would just plow ahead: the orchestrator could decide
    to spawn yet another sub-agent (spending circuit-breaker budget the user didn't intend to
    spend) while the original pause sat un-actioned, and — worse — a second pause created this way
    could jump the queue ahead of whatever was already waiting (see _set_pending_approval). Returns
    the response payload to send immediately instead of running a turn, or None if the session is
    clear to proceed."""
    pending = engine.state.get("_pending_approval")
    if not pending:
        return None
    return {
        "response": (
            "This session has an action awaiting human approval — resolve it via POST /resume "
            "before sending another message."
        ),
        "active_agent": engine.active_agent_name,
        "status": "awaiting_approval",
        "pending_action": pending,
        "queued_approvals": _queued_approvals_count(engine),
    }


async def _execute_approved_tool_and_replace_placeholder(
    engine: RuntimeEngine, pending: dict, req: "ResumeRequest"
) -> str:
    """Executes an approved tool call against `engine` and replaces its own paused placeholder
    tool-result message in place (same tool_call_id) instead of appending a second response to
    it — which would violate strict provider message-threading rules (a tool_call may be answered
    exactly once). Falls back to appending only if no placeholder can be found (e.g. a checkpoint
    saved before this existed). Shared by the direct /resume path and
    _resume_nested_child_approval, which do this identically against different engines (this
    session's own vs. a spawned child's) — kept in one place so a future change to approval
    execution can't accidentally diverge between the two. Returns the feedback message to append
    to the engine's own conversation."""
    tool_name = pending["tool"]
    tool_call_id = pending.get("tool_call_id")
    approved_args = req.edited_args if req.edited_args is not None else pending["args"]

    # Keyed by tool_call_id, not tool_name — see the matching comment in engine.py's execute_tool.
    # Approving this specific paused call must not grant a free pass to some *other* unapproved
    # call to the same-named tool sitting in the same batch. Falls back to the name only if this
    # pause genuinely has no tool_call_id (should not happen via the normal execution path).
    approved_list = engine.state.setdefault("_approved_tool_calls", [])
    approved_list.append(tool_call_id if tool_call_id is not None else tool_name)
    tool_result = await engine.execute_tool(
        tool_name, approved_args, interactive=False, tool_call_id=tool_call_id
    )

    replaced = False
    if tool_call_id:
        for m in reversed(engine.messages):
            if m.get("role") == "tool" and m.get("tool_call_id") == tool_call_id:
                m["content"] = tool_result
                replaced = True
                break
    if not replaced:
        tool_message = {"role": "tool", "name": tool_name, "content": tool_result}
        if tool_call_id:
            tool_message["tool_call_id"] = tool_call_id
        engine.messages.append(tool_message)

    return (
        "Human Reviewer Approved and executed the requested action with "
        f"the reviewed arguments {approved_args!r}. Tool result: {tool_result}"
    )


def _queued_approvals_count(engine: RuntimeEngine) -> int:
    """How many more pauses are waiting behind engine.state["_pending_approval"] — see
    _set_pending_approval's queueing in runtime/engine.py. Surfaced on every response that
    reports a pending_action so a caller (or a human reviewer) knows more are coming next,
    instead of having to inspect the raw checkpoint state to find out."""
    return len(engine.state.get("_pending_approval_queue") or [])


async def _resume_nested_child_approval(
    *,
    graph,
    project_dir: Path,
    shared,
    engine: RuntimeEngine,
    namespaced_session: str,
    pending_action: dict,
    req: "ResumeRequest",
    approver_id: str | None,
    pre_metrics: dict,
    run_start: float,
) -> "ChatResponse":
    """Resumes a pending approval that actually belongs to a spawned sub-agent's isolated child
    engine, not this session's own engine directly — see spawn_agent in runtime/engine.py, which
    persists the child under its own session_id and leaves a pointer (child_session_id) in this
    (parent) session's own _pending_approval instead of silently discarding the pause. Reloads
    that child session, applies the same approve/deny + N-of-M logic to *its* pending approval,
    continues the child exactly like spawn_agent's own completion loop would have, and — once the
    child is genuinely done (finished, or called return_to_creator) — merges its result back into
    this session and replaces the original spawn_agent tool-result placeholder, then lets this
    session's own turn loop continue so the orchestrator can react to the real outcome."""
    child_engine = RuntimeEngine(
        graph=graph,
        project_dir=project_dir,
        session_id=pending_action["child_session_id"],
        shared_resources=shared,
    )
    await child_engine.initialize()
    child_pending = child_engine.state.get("_pending_approval")
    if not child_pending:
        # Stale pointer (e.g. the child was already resumed and finished through some other
        # path) — clear the dangling pointer rather than leaving this session stuck forever.
        engine.state.pop("_pending_approval", None)
        engine._save_checkpoint()
        raise HTTPException(
            status_code=409,
            detail="The spawned sub-agent's pending approval no longer exists — it may have "
            "already been resolved.",
        )

    if req.approved:
        satisfied, outstanding, approvals_received = _check_approval_satisfied(
            child_pending, approver_id
        )
        if not satisfied:
            child_engine.state["_pending_approval"] = child_pending
            child_engine._save_checkpoint()
            pending_action["approvals_received"] = approvals_received
            engine.state["_pending_approval"] = pending_action
            engine._save_checkpoint()
            await child_engine._await_last_checkpoint()
            await engine._await_last_checkpoint()
            remaining = (
                len(outstanding)
                if outstanding is not None
                else child_pending.get("required_approvals", 1) - len(approvals_received)
            )
            response_msg = (
                f"Approved by '{approver_id}'. Still awaiting {remaining} more "
                "approval(s)" + (f" from {outstanding}" if outstanding else "") + "."
            )
            await _log_run(
                graph, project_dir, engine, "/resume", namespaced_session,
                "awaiting_approval", None, pre_metrics, run_start,
            )
            return ChatResponse(
                response=response_msg,
                active_agent=pending_action.get("agent", engine.active_agent_name),
                status="awaiting_approval",
                pending_action=pending_action,
                queued_approvals=_queued_approvals_count(engine),
            )

        child_engine.state.pop("_pending_approval", None)
        feedback_msg = await _execute_approved_tool_and_replace_placeholder(
            child_engine, child_pending, req
        )
    else:
        child_engine.state.pop("_pending_approval", None)
        feedback_msg = (
            f"Human Reviewer Denied: {req.reviewer_notes or 'Operation was rejected by operator.'}"
        )

    child_engine.messages.append({"role": "user", "content": feedback_msg})
    child_engine._save_checkpoint()

    # Continue the child to completion exactly like spawn_agent's own original loop — bounded by
    # the same circuit breaker, stopping early if it pauses again or calls return_to_creator.
    max_turns = graph.config.circuit_breakers.max_delegation_turns
    turn_count = 0
    dyn_name = pending_action.get("agent")
    while turn_count < max_turns:
        child_engine.is_transferring = False
        await child_engine._run_agent_turn(interactive=False)
        if "_pending_approval" in child_engine.state:
            break
        if not child_engine.is_transferring:
            break
        if child_engine.active_agent_name != dyn_name:
            break
        turn_count += 1

    if "_pending_approval" in child_engine.state:
        # Paused again on a later tool call — persist and re-surface exactly like the first
        # pause, so a follow-up /resume on this same (parent) session continues from here.
        child_engine.state.pop("_approved_tool_calls", None)
        child_engine._save_checkpoint()
        new_child_pending = child_engine.state["_pending_approval"]
        pending_action.update(
            {
                "tool": new_child_pending["tool"],
                "args": new_child_pending["args"],
                "required_approvals": new_child_pending.get("required_approvals", 1),
                "required_approvers": new_child_pending.get("required_approvers"),
                "approvals_received": [],
                # This is a genuinely new pause (a different tool) — the old created_at would
                # otherwise keep pointing at the *first* pause this child ever hit.
                "created_at": new_child_pending.get("created_at"),
            }
        )
        engine.state["_pending_approval"] = pending_action
        engine._save_checkpoint()
        await child_engine._await_last_checkpoint()
        await engine._await_last_checkpoint()
        await _log_run(
            graph, project_dir, engine, "/resume", namespaced_session,
            "awaiting_approval", None, pre_metrics, run_start,
        )
        return ChatResponse(
            response=(
                f"Sub-agent '{dyn_name}' resumed and is now paused again awaiting approval "
                f"for '{new_child_pending['tool']}'."
            ),
            active_agent=dyn_name,
            status="awaiting_approval",
            pending_action=pending_action,
            queued_approvals=_queued_approvals_count(engine),
        )

    # The child is genuinely done (or was forcefully stopped at the same turn cap spawn_agent's
    # own loop enforces) — merge its results back into the parent exactly like spawn_agent would
    # have on a synchronous completion, and replace the parent's own spawn_agent tool-result
    # placeholder with the real outcome.
    if turn_count >= max_turns:
        Tracer.log_error(
            f"Spawned agent '{dyn_name}' reached maximum turns ({max_turns}) and was forcefully "
            "stopped mid-resume."
        )
    final_answer = extract_final_answer(child_engine.messages)

    engine._merge_child_state(pending_action.get("pre_state", {}), child_engine.state)
    if turn_count < max_turns:
        dynamic_agent = engine.state.get("_dynamic_agents", {}).get(dyn_name)
        if dynamic_agent:
            engine._apply_spawn_completion_hooks(dynamic_agent)
    engine.state.pop("_pending_approval", None)

    result_msg = f"Sub-agent '{dyn_name}' completed.\nResult:\n{final_answer}"
    parent_tool_call_id = pending_action.get("parent_tool_call_id")
    replaced = False
    if parent_tool_call_id:
        for m in reversed(engine.messages):
            if m.get("role") == "tool" and m.get("tool_call_id") == parent_tool_call_id:
                m["content"] = result_msg
                replaced = True
                break
    if not replaced:
        tool_message = {"role": "tool", "name": "spawn_agent", "content": result_msg}
        if parent_tool_call_id:
            tool_message["tool_call_id"] = parent_tool_call_id
        engine.messages.append(tool_message)

    engine._save_checkpoint()

    while True:
        engine.is_transferring = False
        await engine._run_agent_turn(interactive=False)
        # A round can legitimately set BOTH is_transferring (a transfer_agent/return_to_creator
        # call) and _pending_approval (a requires_approval tool call in the same batch) —
        # _execute_tool_calls_with_healing runs normal tools, including approval-gated ones,
        # before the transfer tool. is_transferring alone used to keep this loop advancing into
        # the newly-transferred-to agent's own turn while the earlier pause sat unresolved.
        if not engine.is_transferring or "_pending_approval" in engine.state:
            break

    final_parent_answer = "No response"
    for msg in reversed(engine.messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            final_parent_answer = msg["content"]
            break

    final_pending_action = engine.state.get("_pending_approval", None)
    if not final_pending_action and engine._promote_next_queued_approval():
        # Another spawned child (or a directly-gated tool in the same original turn) also paused
        # concurrently and had to queue behind this one — surface it now instead of silently
        # dropping it once this one's resolved.
        final_pending_action = engine.state["_pending_approval"]
    status = "awaiting_approval" if final_pending_action else "completed"
    engine.state.pop("_approved_tool_calls", None)
    engine._save_checkpoint()
    await engine._await_last_checkpoint()

    await _log_run(
        graph, project_dir, engine, "/resume", namespaced_session,
        status, None, pre_metrics, run_start,
    )
    return ChatResponse(
        response=final_parent_answer,
        active_agent=engine.active_agent_name,
        status=status,
        pending_action=final_pending_action,
        queued_approvals=_queued_approvals_count(engine),
    )


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest, user_context: str = Depends(verify_auth)):
    project_dir = Path.cwd()
    engine = None
    graph = None
    try:
        graph = parse_project(project_dir)
        # check_rate_limit does blocking sqlite/psycopg I/O (runtime/rate_limiter.py) — must
        # not run directly on the event loop thread of this async endpoint.
        await asyncio.to_thread(
            check_rate_limit,
            graph.config.memory, project_dir, user_context, graph.config.server.rate_limit,
        )
        namespaced_session = f"{user_context}:{req.session_id}"
        # Serializes concurrent requests for the same session — without this, two overlapping
        # requests for one session_id (a double-click, a client retry, two tabs) each load the
        # whole session, run independently, and last-write-wins on save, silently dropping
        # whichever turn saved first.
        async with get_session_lock_registry().get_lock(namespaced_session):
            shared = await get_shared_resources_cache().get(graph, project_dir)
            engine = RuntimeEngine(
                graph=graph,
                project_dir=project_dir,
                session_id=namespaced_session,
                initial_state=req.initial_state,
                shared_resources=shared,
            )
            _run_start = time.monotonic()
            _pre_metrics: dict = {}
            await engine.initialize()
            _pre_metrics = dict(engine.state.get("_metrics", {}))

            blocked = _pending_approval_block(engine)
            if blocked is not None:
                await _log_run(
                    graph, project_dir, engine, "/chat", namespaced_session,
                    "awaiting_approval", None, _pre_metrics, _run_start,
                )
                return ChatResponse(**blocked)

            safe_input = engine._apply_guardrails(req.message)
            engine.messages.append({"role": "user", "content": safe_input})
            await engine._compress_memory()
            engine._save_checkpoint()

            while True:
                engine.is_transferring = False
                await engine._run_agent_turn(interactive=False)
                # See the identical comment in _resume_nested_child_approval — a single round can
                # set both is_transferring and _pending_approval, and the pause must win.
                if not engine.is_transferring or "_pending_approval" in engine.state:
                    break

            # Find the last assistant message
            final_answer = "No response"
            for msg in reversed(engine.messages):
                if msg.get("role") == "assistant" and msg.get("content"):
                    final_answer = msg["content"]
                    break

            pending_action = engine.state.get("_pending_approval", None)
            status = "awaiting_approval" if pending_action else "completed"
            engine._save_checkpoint()
            await engine._await_last_checkpoint()

            await _log_run(
                graph, project_dir, engine, "/chat", namespaced_session,
                status, None, _pre_metrics, _run_start,
            )
            return ChatResponse(
                response=final_answer,
                active_agent=engine.active_agent_name,
                status=status,
                pending_action=pending_action,
                queued_approvals=_queued_approvals_count(engine),
            )
    except IntaGrinError:
        # Without this, a rate-limit rejection (or any other codified error) below would be
        # caught by the generic handler and rewrapped into a misleading 500, losing its real
        # status code (429) and structured `code` field — the same class of bug fixed for
        # identify_approver's 403 in resume_endpoint.
        raise
    except Exception as e:
        Tracer.log_error(f"API Error: {e}")
        if engine is not None and graph is not None:
            await _log_run(
                graph, project_dir, engine, "/chat", namespaced_session,
                "error", str(e), _pre_metrics, _run_start,
            )
        await _record_turn_failure(engine, graph, e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream_endpoint(
    req: ChatRequest, user_context: str = Depends(verify_auth)
):
    """
    Stream Server-Sent Events (SSE) for real-time frontend integration.
    """
    project_dir = Path.cwd()
    namespaced_session = f"{user_context}:{req.session_id}"
    # Held from setup through the end of event_generator()'s iteration (released in its
    # finally), not scoped with `async with`, since StreamingResponse iterates the generator
    # after this function returns — the lock has to span that later iteration too, not just this
    # synchronous setup, to actually serialize two concurrent requests for the same session.
    lock = get_session_lock_registry().get_lock(namespaced_session)
    await lock.acquire()
    try:
        graph = parse_project(project_dir)
        # check_rate_limit does blocking sqlite/psycopg I/O (runtime/rate_limiter.py) — must
        # not run directly on the event loop thread of this async endpoint.
        await asyncio.to_thread(
            check_rate_limit,
            graph.config.memory, project_dir, user_context, graph.config.server.rate_limit,
        )
        shared = await get_shared_resources_cache().get(graph, project_dir)
        engine = RuntimeEngine(
            graph=graph,
            project_dir=project_dir,
            session_id=namespaced_session,
            initial_state=req.initial_state,
            shared_resources=shared,
        )
        _run_start = time.monotonic()
        await engine.initialize()
        _pre_metrics = dict(engine.state.get("_metrics", {}))

        blocked = _pending_approval_block(engine)
        if blocked is not None:
            await _log_run(
                graph, project_dir, engine, "/chat/stream", namespaced_session,
                "awaiting_approval", None, _pre_metrics, _run_start,
            )
            lock.release()

            async def blocked_generator():
                yield f"data: {json.dumps({'type': 'done', **blocked})}\n\n"

            return StreamingResponse(blocked_generator(), media_type="text/event-stream")

        safe_input = engine._apply_guardrails(req.message)
        engine.messages.append({"role": "user", "content": safe_input})
        await engine._compress_memory()
        engine._save_checkpoint()
    except Exception:
        lock.release()
        raise

    async def event_generator():
        try:
            import json

            while True:
                engine.is_transferring = False
                async for event in engine._run_agent_turn_stream(interactive=False):
                    yield f"data: {json.dumps(event)}\n\n"

                # See the identical comment in _resume_nested_child_approval — a single round can
                # set both is_transferring and _pending_approval, and the pause must win.
                if not engine.is_transferring or "_pending_approval" in engine.state:
                    break

            pending_action = engine.state.get("_pending_approval", None)
            status = "awaiting_approval" if pending_action else "completed"
            engine._save_checkpoint()
            await engine._await_last_checkpoint()

            await _log_run(
                graph, project_dir, engine, "/chat/stream", namespaced_session,
                status, None, _pre_metrics, _run_start,
            )
            final_event = {
                "type": "done",
                "active_agent": engine.active_agent_name,
                "status": status,
                "pending_action": pending_action,
                "queued_approvals": _queued_approvals_count(engine),
            }
            yield f"data: {json.dumps(final_event)}\n\n"
        except Exception as e:
            import json

            Tracer.log_error(f"Chat Stream Error: {e}")
            await _log_run(
                graph, project_dir, engine, "/chat/stream", namespaced_session,
                "error", str(e), _pre_metrics, _run_start,
            )
            await _record_turn_failure(engine, graph, e)
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
        finally:
            lock.release()

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/resume", response_model=ChatResponse)
async def resume_endpoint(
    req: ResumeRequest, request: Request, user_context: str = Depends(verify_auth)
):
    """
    Resume an asynchronous session that was suspended waiting for human-in-the-loop approval.
    """
    project_dir = Path.cwd()
    engine = None
    graph = None
    try:
        graph = parse_project(project_dir)
        # check_rate_limit does blocking sqlite/psycopg I/O (runtime/rate_limiter.py) — must
        # not run directly on the event loop thread of this async endpoint.
        await asyncio.to_thread(
            check_rate_limit,
            graph.config.memory, project_dir, user_context, graph.config.server.rate_limit,
        )
        # identify_approver does blocking sqlite/psycopg I/O (runtime/approvers.py) and, on a
        # DB-backed match attempt, a deliberately CPU/memory-heavy scrypt hash per active
        # approver — run on the event loop thread (this is `async def`), that would stall every
        # other in-flight request on this process for the duration, not just this one.
        approver_id = (
            await asyncio.to_thread(identify_approver, request, graph, project_dir)
            if req.approved
            else None
        )
        namespaced_session = f"{user_context}:{req.session_id}"
        async with get_session_lock_registry().get_lock(namespaced_session):
            shared = await get_shared_resources_cache().get(graph, project_dir)
            engine = RuntimeEngine(
                graph=graph,
                project_dir=project_dir,
                session_id=namespaced_session,
                shared_resources=shared,
            )
            _run_start = time.monotonic()
            _pre_metrics: dict = {}
            await engine.initialize()
            _pre_metrics = dict(engine.state.get("_metrics", {}))

            pending_action = engine.state.get("_pending_approval")
            if not pending_action:
                raise HTTPException(status_code=400, detail="No pending action to resume.")

            if pending_action.get("child_session_id"):
                # This session's own pending approval actually belongs to a sub-agent it spawned
                # (see spawn_agent in runtime/engine.py) — resolve it there instead of trying to
                # execute the tool directly against this engine.
                result = await _resume_nested_child_approval(
                    graph=graph,
                    project_dir=project_dir,
                    shared=shared,
                    engine=engine,
                    namespaced_session=namespaced_session,
                    pending_action=pending_action,
                    req=req,
                    approver_id=approver_id,
                    pre_metrics=_pre_metrics,
                    run_start=_run_start,
                )
                return result

            engine.active_agent_name = pending_action.get(
                "agent", engine.active_agent_name
            )

            if req.approved:
                satisfied, outstanding, approvals_received = _check_approval_satisfied(
                    pending_action, approver_id
                )

                if not satisfied:
                    # Not yet fully approved — persist the updated approvals_received and pause
                    # again rather than executing the tool or advancing the turn loop.
                    engine.state["_pending_approval"] = pending_action
                    engine._save_checkpoint()
                    await engine._await_last_checkpoint()
                    remaining = (
                        len(outstanding) if outstanding is not None
                        else pending_action.get("required_approvals", 1) - len(approvals_received)
                    )
                    response_msg = (
                        f"Approved by '{approver_id}'. Still awaiting {remaining} more "
                        "approval(s)" + (f" from {outstanding}" if outstanding else "") + "."
                    )
                    await _log_run(
                        graph, project_dir, engine, "/resume", namespaced_session,
                        "awaiting_approval", None, _pre_metrics, _run_start,
                    )
                    return ChatResponse(
                        response=response_msg,
                        active_agent=engine.active_agent_name,
                        status="awaiting_approval",
                        pending_action=pending_action,
                        queued_approvals=_queued_approvals_count(engine),
                    )

                engine.state.pop("_pending_approval", None)
                feedback_msg = await _execute_approved_tool_and_replace_placeholder(
                    engine, pending_action, req
                )
            else:
                engine.state.pop("_pending_approval", None)
                feedback_msg = f"Human Reviewer Denied: {req.reviewer_notes or 'Operation was rejected by operator.'}"

            engine.messages.append({"role": "user", "content": feedback_msg})
            engine._save_checkpoint()

            while True:
                engine.is_transferring = False
                await engine._run_agent_turn(interactive=False)
                # See the identical comment in _resume_nested_child_approval — a single round can
                # set both is_transferring and _pending_approval, and the pause must win.
                if not engine.is_transferring or "_pending_approval" in engine.state:
                    break

            final_answer = "No response"
            for msg in reversed(engine.messages):
                if msg.get("role") == "assistant" and msg.get("content"):
                    final_answer = msg["content"]
                    break

            pending_action = engine.state.get("_pending_approval", None)
            if not pending_action and engine._promote_next_queued_approval():
                # A concurrently-executed tool call (e.g. two spawn_agent calls in the same
                # original turn, each pausing independently) had to queue behind this one —
                # surface it now instead of silently dropping it once this one's resolved.
                pending_action = engine.state["_pending_approval"]
            status = "awaiting_approval" if pending_action else "completed"

            # Security Loophole Fix: Clear any unused exemptions so they don't persist
            engine.state.pop("_approved_tool_calls", None)
            engine._save_checkpoint()
            await engine._await_last_checkpoint()

            await _log_run(
                graph, project_dir, engine, "/resume", namespaced_session,
                status, None, _pre_metrics, _run_start,
            )
            return ChatResponse(
                response=final_answer,
                active_agent=engine.active_agent_name,
                status=status,
                pending_action=pending_action,
                queued_approvals=_queued_approvals_count(engine),
            )
    except (HTTPException, IntaGrinError):
        # Without this, both the pre-existing 400 ("No pending action to resume") and the 403
        # from identify_approver / 429 from check_rate_limit were caught by the generic handler
        # below and rewrapped into a misleading 500 — semantically wrong even though the client
        # still ends up rejected. Deliberately not logged here — a 400/403/429 is a client usage
        # error (nothing to resume, missing approver credential, or over quota), not an agent-run
        # outcome worth a cost/context debug row.
        raise
    except Exception as e:
        Tracer.log_error(f"Resume Error: {e}")
        if engine is not None and graph is not None:
            await _log_run(
                graph, project_dir, engine, "/resume", namespaced_session,
                "error", str(e), _pre_metrics, _run_start,
            )
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/sessions")
def get_sessions(user_context: str = Depends(verify_auth)):
    """
    List all sessions and their state (including pending approvals) for the authenticated tenant.
    """
    project_dir = Path.cwd()
    try:
        graph = parse_project(project_dir)
        sessions = []
        
        if graph.config.memory.type == "sqlite":
            import sqlite3
            
            db_path = project_dir / (graph.config.memory.db_path or ".ai/memory.db")
            if not db_path.exists():
                return []
            
            # Use LIKE query to isolate tenant sessions
            with sqlite3.connect(str(db_path), timeout=15.0) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT session_id, messages, state FROM checkpoints WHERE session_id LIKE ? ORDER BY updated_at DESC LIMIT 50",
                    (f"{user_context}:%",)
                )
                for row in cursor.fetchall():
                    try:
                        raw_session_id = row["session_id"]
                        # Strip the tenant namespace prefix for the client UI
                        client_session_id = raw_session_id.replace(f"{user_context}:", "", 1)
                        sessions.append({
                            "session_id": client_session_id,
                            "messages": json.loads(row["messages"]) if row["messages"] else [],
                            "state": json.loads(row["state"]) if row["state"] else {}
                        })
                    except Exception as row_err:
                        Tracer.log_error(
                            f"Skipping malformed session row while listing sessions: {row_err}"
                        )

        return sessions
    except Exception as e:
        Tracer.log_error(f"Sessions Listing Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class ApproverCreateRequest(BaseModel):
    approver_id: str


@app.post("/approvers")
def create_approver(req: ApproverCreateRequest, _: None = Depends(verify_admin_auth)):
    """
    Issues (or rotates, if approver_id already exists) a DB-backed X-Approver-Key credential —
    the HTTP equivalent of `inta approvers add`, for a consumer's own admin site/tooling. The
    generated secret is returned exactly once; it is not recoverable afterwards (only
    revoke-and-reissue). Gated by verify_admin_auth, a separate credential tier from both the
    main session auth and any individual approver's own key.
    """
    project_dir = Path.cwd()
    try:
        graph = parse_project(project_dir)
        secret = secrets.token_urlsafe(32)
        add_approver(graph.config.memory, project_dir, req.approver_id, secret)
        return {"approver_id": req.approver_id, "secret": secret}
    except IntaGrinError:
        raise
    except Exception as e:
        Tracer.log_error(f"Approver Create Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/approvers")
def list_approvers_endpoint(_: None = Depends(verify_admin_auth)):
    """Lists every DB-backed approver (active and revoked) — ids and timestamps only, never
    secrets/hashes."""
    project_dir = Path.cwd()
    try:
        graph = parse_project(project_dir)
        return {"approvers": list_approvers(graph.config.memory, project_dir)}
    except IntaGrinError:
        raise
    except Exception as e:
        Tracer.log_error(f"Approver List Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/approvers/{approver_id}")
def revoke_approver_endpoint(approver_id: str, _: None = Depends(verify_admin_auth)):
    """Revokes an approver's credential — it can no longer approve via /resume. Kept in the
    database for audit history rather than deleted."""
    project_dir = Path.cwd()
    try:
        graph = parse_project(project_dir)
        if not revoke_approver(graph.config.memory, project_dir, approver_id):
            raise HTTPException(
                status_code=404, detail=f"No active approver named '{approver_id}'."
            )
        return {"approver_id": approver_id, "revoked": True}
    except HTTPException:
        raise
    except IntaGrinError:
        raise
    except Exception as e:
        Tracer.log_error(f"Approver Revoke Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stream")
async def stream_endpoint(req: ChatRequest, user_context: str = Depends(verify_auth)):
    """
    Production SSE (Server-Sent Events) endpoint for token streaming.
    Allows React/Next.js frontends to stream the response typewriter-style.
    """
    project_dir = Path.cwd()
    namespaced_session = f"{user_context}:{req.session_id}"
    # Same acquire-here/release-in-generator's-finally pattern as chat_stream_endpoint — the lock
    # must span the generator's later iteration by StreamingResponse, not just this setup.
    lock = get_session_lock_registry().get_lock(namespaced_session)
    await lock.acquire()
    try:
        graph = parse_project(project_dir)
        # check_rate_limit does blocking sqlite/psycopg I/O (runtime/rate_limiter.py) — must
        # not run directly on the event loop thread of this async endpoint.
        await asyncio.to_thread(
            check_rate_limit,
            graph.config.memory, project_dir, user_context, graph.config.server.rate_limit,
        )
        shared = await get_shared_resources_cache().get(graph, project_dir)
        engine = RuntimeEngine(
            graph=graph,
            project_dir=project_dir,
            session_id=namespaced_session,
            initial_state=req.initial_state,
            shared_resources=shared,
        )
        _run_start = time.monotonic()
        await engine.initialize()
        _pre_metrics = dict(engine.state.get("_metrics", {}))

        blocked = _pending_approval_block(engine)
        if blocked is not None:
            await _log_run(
                graph, project_dir, engine, "/stream", namespaced_session,
                "awaiting_approval", None, _pre_metrics, _run_start,
            )
            lock.release()

            async def blocked_generator():
                yield f"data: {json.dumps({'type': 'awaiting_approval', 'pending_action': blocked['pending_action'], 'queued_approvals': blocked['queued_approvals']})}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(blocked_generator(), media_type="text/event-stream")

        safe_input = engine._apply_guardrails(req.message)
        engine.messages.append({"role": "user", "content": safe_input})
        await engine._compress_memory()
        engine._save_checkpoint()
    except IntaGrinError:
        lock.release()
        raise
    except Exception as e:
        lock.release()
        Tracer.log_error(f"Stream Setup Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    async def event_generator():
        try:
            yield f"data: {json.dumps({'type': 'agent', 'agent': engine.active_agent_name})}\n\n"

            engine.is_transferring = False
            async for chunk in engine._run_agent_turn_stream(interactive=False):
                yield f"data: {json.dumps(chunk)}\n\n"

            # See the identical comment in _resume_nested_child_approval — a single round can set
            # both is_transferring and _pending_approval, and the pause must win.
            while engine.is_transferring and "_pending_approval" not in engine.state:
                engine.is_transferring = False
                yield f"data: {json.dumps({'type': 'agent', 'agent': engine.active_agent_name})}\n\n"
                async for chunk in engine._run_agent_turn_stream(interactive=False):
                    yield f"data: {json.dumps(chunk)}\n\n"

            engine._save_checkpoint()
            await engine._await_last_checkpoint()
            # Regression fix: this used to hardcode "completed" unconditionally, unlike /chat and
            # /chat/stream — a turn that ended because a tool paused for human approval (directly,
            # or a spawned sub-agent's pause propagated up) was reported identically to one that
            # actually finished, with no client-visible signal that anything needed /resume.
            pending_action = engine.state.get("_pending_approval", None)
            status = "awaiting_approval" if pending_action else "completed"
            await _log_run(
                graph, project_dir, engine, "/stream", namespaced_session,
                status, None, _pre_metrics, _run_start,
            )
            if pending_action:
                yield f"data: {json.dumps({'type': 'awaiting_approval', 'pending_action': pending_action, 'queued_approvals': _queued_approvals_count(engine)})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            # Pre-existing gap fixed alongside this feature: this generator previously had no
            # except clause at all, so an error here just died silently mid-stream with no
            # client-visible event — and no chance to log it either.
            Tracer.log_error(f"Stream Error: {e}")
            await _log_run(
                graph, project_dir, engine, "/stream", namespaced_session,
                "error", str(e), _pre_metrics, _run_start,
            )
            await _record_turn_failure(engine, graph, e)
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
        finally:
            lock.release()

    return StreamingResponse(event_generator(), media_type="text/event-stream")



@app.websocket("/ws/voice")
async def voice_websocket_endpoint(
    websocket: WebSocket, session_id: str = "voice_session"
):
    """
    Bidirectional WebSocket gateway for real-time voice, speech-to-text, and conversational agents.
    """
    project_dir = Path.cwd()
    try:
        authorization = websocket.headers.get("authorization", "")
        token = (
            authorization.split(" ", 1)[1]
            if authorization.lower().startswith("bearer ")
            else None
        )
        user_context = authenticate_token(token)
    except HTTPException:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    namespaced_session = f"{user_context}:{session_id}"
    try:
        graph = parse_project(project_dir)
        shared = await get_shared_resources_cache().get(graph, project_dir)
        engine = RuntimeEngine(
            graph=graph,
            project_dir=project_dir,
            session_id=namespaced_session,
            shared_resources=shared,
        )
        await engine.initialize()

        await websocket.send_json(
            {
                "type": "ready",
                "active_agent": engine.active_agent_name,
                "message": f"Connected to {engine.active_agent_name} voice session.",
            }
        )

        while True:
            data = await websocket.receive_json()
            user_text = data.get("text", "")
            if not user_text.strip():
                continue

            # Same read-modify-write serialization every other stateful endpoint holds
            # (/chat, /chat/stream, /resume, /stream) — without it, this voice turn and a
            # concurrent /chat (or a second voice message) against the same session_id could
            # interleave load/mutate/save and lose an update. Scoped to just this turn, not the
            # whole connection, so it doesn't block while idly waiting on receive_json().
            lock = get_session_lock_registry().get_lock(namespaced_session)
            async with lock:
                if engine.state.get("_pending_approval"):
                    await websocket.send_json(
                        {
                            "type": "awaiting_approval",
                            "message": (
                                "This session has an action awaiting human approval — resolve "
                                "it via POST /resume before sending another message."
                            ),
                            "pending_action": engine.state["_pending_approval"],
                        }
                    )
                    continue

                safe_input = engine._apply_guardrails(user_text)
                engine.messages.append({"role": "user", "content": safe_input})
                engine._save_checkpoint()

                await websocket.send_json(
                    {"type": "agent_start", "agent": engine.active_agent_name}
                )

                engine.is_transferring = False
                async for chunk in engine._run_agent_turn_stream():
                    await websocket.send_json(chunk)

                # See the identical comment in _resume_nested_child_approval — a single round can
                # set both is_transferring and _pending_approval, and the pause must win.
                while engine.is_transferring and "_pending_approval" not in engine.state:
                    engine.is_transferring = False
                    await websocket.send_json(
                        {"type": "agent_start", "agent": engine.active_agent_name}
                    )
                    async for chunk in engine._run_agent_turn_stream():
                        await websocket.send_json(chunk)

                engine._save_checkpoint()
                await engine._await_last_checkpoint()

            await websocket.send_json({"type": "turn_complete"})

    except WebSocketDisconnect:
        console.print(
            f"[dim]Voice WebSocket client disconnected for session {session_id}.[/dim]"
        )
    except Exception as e:
        Tracer.log_error(f"Voice WebSocket error: {e}")
        await websocket.close()


# Registers GET /.well-known/agent-card.json and POST /a2a onto this same `app` — imported for
# its side effect (route registration) at the bottom of this file, since a2a.py imports
# chat_endpoint/stream_endpoint/verify_auth/app from this module and those names must already
# exist. See server/a2a.py for the full A2A (Agent2Agent protocol) surface this adds.
from . import a2a  # noqa: E402,F401
