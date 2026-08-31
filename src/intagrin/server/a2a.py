"""A2A (Agent2Agent) protocol surface — Agent Card discovery + a JSON-RPC endpoint.

Mounted directly onto server/api.py's existing, live FastAPI `app` (not a detached export like
compiler/exporter.py's STANDALONE_TEMPLATE) — an A2A caller gets a real, full-featured IntaGrin
agent: handoffs, delegations, memory, and human-in-the-loop pauses all behave exactly as they do
over /chat, because this module is a thin protocol-translation layer over `chat_endpoint`/
`stream_endpoint`, not a second implementation of the turn loop. An A2A "context" maps 1:1 onto an
IntaGrin `session_id` — there is no separate A2A task store; task status is read straight from the
same checkpointed engine state `/chat` and `/resume` already use.

SCOPE — implemented:
  - GET  /.well-known/agent-card.json  (public discovery — see note below on why this route is
    deliberately unauthenticated)
  - POST /a2a  (JSON-RPC 2.0), supporting exactly three methods: `message/send`, `message/stream`,
    `tasks/get`.

SCOPE — explicitly NOT implemented (mirrors compiler/exporter.py's own convention of stating scope
boundaries in the file header rather than implying full protocol coverage):
  - Push notifications (`tasks/pushNotificationConfig/*`) — no webhook delivery of task updates.
  - A2A-to-A2A delegated auth chains — a caller authenticates with this server's own
    `server.auth` exactly like any other /chat client; there is no support for verifying an
    upstream agent's own delegated identity/credentials.
  - Multi-turn artifact streaming beyond plain text parts — `message/stream` reframes IntaGrin's
    own token-delta/tool-call event stream as a simplified text-only A2A status-update stream (see
    `_translate_stream_event`'s docstring for the exact reasoning); it does not model partial
    binary/file artifacts.
  - Any JSON-RPC method other than the three listed above (`tasks/cancel`, `tasks/resubscribe`,
    etc.) — these return IG-A2A-002 (unsupported method).

Error shape note: JSON-RPC 2.0 requires `error.code` to be a Number, so a bare IntaGrin error code
string (e.g. "IG-A2A-001") cannot be used as the wire-level `code` field without violating the
spec. This module reports the standard JSON-RPC reserved codes (-32600 invalid request, -32601
method not found, -32603 internal error) as `error.code`, and carries the corresponding IntaGrin
error code alongside as `error.data.intagrin_code` for anyone cross-referencing
docs/12_Error_Reference.md.

Naming note: A2A's Agent Card `skills` field is *protocol vocabulary* — a flat list describing
what an agent can do, shown to external callers per the A2A spec. That is a different concept from
this framework's own ai.yaml `skills:` primitive (see docs/15_Agent_Skills.md), an internal,
progressive-disclosure mechanism for an agent's own prompt content. A project can have both; the
Agent Card's `skills` list here is derived from the default agent's *tools*, not from any ai.yaml
`skills:` entries.
"""

import json
from pathlib import Path
from typing import Any

from fastapi import Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..compiler.parser import parse_project
from ..config.schema import ToolReferenceConfig
from ..errors import IntaGrinError
from ..runtime.engine import RuntimeEngine
from ..runtime.shared_resources import get_shared_resources_cache
from ..tracing.console import Tracer
from .api import ChatRequest, app, chat_endpoint, stream_endpoint, verify_auth

_SUPPORTED_METHODS = {"message/send", "message/stream", "tasks/get"}

# Standard JSON-RPC 2.0 reserved error codes (spec-mandated numeric range) — see the module
# docstring's "Error shape note" for why these, rather than IntaGrin's own string codes, are what
# actually goes on the wire as `error.code`.
_JSONRPC_INVALID_REQUEST = -32600
_JSONRPC_METHOD_NOT_FOUND = -32601
_JSONRPC_INTERNAL_ERROR = -32603


def _resolve_tool(tool, root_tools_by_name: dict):
    """Same ToolReferenceConfig resolution pattern used by compiler/verifier.py's lethal-trifecta
    check — a per-agent `tools:` entry is usually a name-reference to a root-level tool, so a
    tool's real name/description live on the referenced root config, not the reference itself."""
    if isinstance(tool, ToolReferenceConfig):
        return root_tools_by_name.get(tool.name, tool)
    return tool


def _build_agent_card(cfg) -> dict:
    default_agent_cfg = cfg.agents.get(cfg.default_agent)
    root_tools_by_name = {t.name: t for t in cfg.tools}

    skills: list[dict] = []
    if default_agent_cfg is not None:
        for tool in default_agent_cfg.tools:
            resolved = _resolve_tool(tool, root_tools_by_name)
            name = getattr(tool, "name", None)
            if not name:
                continue
            description = getattr(resolved, "description", None) or f"IntaGrin tool: {name}"
            skills.append({"id": name, "name": name, "description": description, "tags": []})

    auth_cfg = cfg.server.auth
    if auth_cfg.type in ("api_key", "custom"):
        # IntaGrin's api_key/custom auth is sent as `Authorization: Bearer <token>` (FastAPI's
        # HTTPBearer, see server/api.py's verify_auth) — the accurate OpenAPI/A2A security scheme
        # for that is HTTP bearer, not a named `apiKey` header scheme. A deliberate correction
        # from an earlier assumption, made after reading verify_auth's actual implementation.
        security_schemes: dict[str, Any] = {"bearerAuth": {"type": "http", "scheme": "bearer"}}
        security = [{"bearerAuth": []}]
    else:
        security_schemes = {}
        security = []

    return {
        "protocolVersion": "0.2.0",
        "name": cfg.name,
        "description": cfg.description or f"IntaGrin agent '{cfg.name}'",
        "url": "/a2a",
        "preferredTransport": "JSONRPC",
        "capabilities": {"streaming": True, "pushNotifications": False},
        "defaultInputModes": ["text"],
        "defaultOutputModes": ["text"],
        "skills": skills,
        "securitySchemes": security_schemes,
        "security": security,
    }


@app.get("/.well-known/agent-card.json")
def get_agent_card():
    """Deliberately unauthenticated, matching the A2A spec's public-discovery convention (a
    caller fetches this before it has any reason to hold a credential yet) — actually calling the
    agent through POST /a2a still goes through the normal verify_auth check below. This mirrors
    how OpenAPI/Swagger spec documents are typically served publicly even when the API they
    describe requires auth."""
    graph = parse_project(Path.cwd())
    return _build_agent_card(graph.config)


def _jsonrpc_error(
    request_id: Any, jsonrpc_code: int, message: str, intagrin_code: str | None = None
) -> JSONResponse:
    error: dict[str, Any] = {"code": jsonrpc_code, "message": message}
    if intagrin_code:
        error["data"] = {"intagrin_code": intagrin_code}
    return JSONResponse(
        status_code=200,
        content={"jsonrpc": "2.0", "id": request_id, "error": error},
    )


def _extract_text(message: dict) -> str:
    parts = message.get("parts") or []
    texts = [
        p.get("text", "")
        for p in parts
        if isinstance(p, dict) and p.get("text") and (p.get("kind") or p.get("type")) in (None, "text")
    ]
    if texts:
        return "\n".join(texts)
    return message.get("text", "") or ""


def _require_context_id(message: dict, params: dict) -> str:
    context_id = message.get("contextId") or params.get("contextId")
    if not context_id:
        raise IntaGrinError(
            "IG-A2A-001",
            "message.contextId (or params.contextId) is required — mapped to IntaGrin's own "
            "session_id, so a caller must supply one to have a resumable conversation.",
        )
    return context_id


def _a2a_message(text: str, context_id: str) -> dict:
    return {
        "role": "agent",
        "parts": [{"kind": "text", "text": text}],
        "contextId": context_id,
        "kind": "message",
    }


async def _handle_message_send(params: dict, user_context: str) -> dict:
    message = params.get("message")
    if not isinstance(message, dict):
        raise IntaGrinError("IG-A2A-001", "params.message is required for message/send.")
    context_id = _require_context_id(message, params)
    text = _extract_text(message)

    # Deliberately omit initial_state rather than passing None explicitly: ChatRequest types it
    # as a bare `dict` (default None, but the annotation isn't Optional) — Pydantic only skips
    # validating a field against its annotation when the field is genuinely omitted and the
    # default is used as-is, not when None is passed as an explicit value. Passing it explicitly
    # raises a ValidationError; found via this feature's own end-to-end tests.
    chat_req = ChatRequest(message=text, session_id=context_id)
    chat_resp = await chat_endpoint(chat_req, user_context=user_context)

    # The one honest mapping this whole feature hinges on: IntaGrin's own HITL pause *is* exactly
    # what A2A's input-required task state means — a real human/external actor must act before
    # this task can continue. Any other non-approval status IntaGrin reports maps to completed;
    # there is no IntaGrin-side concept of a task that is still "working" by the time /chat
    # returns (chat_endpoint runs the turn to completion or to a pause before responding).
    task_state = "input-required" if chat_resp.pending_action else "completed"
    return {
        "id": context_id,
        "contextId": context_id,
        "status": {"state": task_state},
        "kind": "task",
        "history": [_a2a_message(chat_resp.response, context_id)],
    }


def _translate_stream_event(inner_event: dict, context_id: str) -> dict | None:
    """Best-effort reframing of one of IntaGrin's own SSE event dicts (see
    runtime/engine.py's `_run_agent_turn_stream` and server/api.py's `stream_endpoint` for the
    source shapes: {"type": "content", "content": <text delta>}, {"type": "tool_chunk", ...},
    {"type": "agent", "agent": ...}, {"type": "done", ...}, {"type": "error", "content": ...}) as
    an A2A TaskStatusUpdateEvent-shaped envelope.

    The A2A spec's exact nested Message/Part/Artifact streaming schema supports several valid
    encodings (incremental artifact chunks, structured data parts, etc.); this picks the simplest
    one — a single text-delta "working" status-update message per content chunk — rather than
    modeling partial-artifact streaming, since IntaGrin's own event stream already interleaves
    tool-call/handoff bookkeeping events that don't correspond to any A2A artifact at all. Those
    bookkeeping event types (tool_chunk, agent/handoff) are deliberately dropped here, not
    translated — a caller wanting full fidelity into tool-call streaming should use message/send
    plus tasks/get instead of message/stream. Returns None for an event type that should be
    dropped rather than forwarded."""
    event_type = inner_event.get("type")
    if event_type == "content":
        text_piece = inner_event.get("content") or ""
        if not text_piece:
            return None
        return {
            "kind": "status-update",
            "taskId": context_id,
            "contextId": context_id,
            "status": {"state": "working", "message": _a2a_message(str(text_piece), context_id)},
            "final": False,
        }
    if event_type == "done":
        pending = inner_event.get("pending_action")
        state = "input-required" if pending else "completed"
        return {
            "kind": "status-update",
            "taskId": context_id,
            "contextId": context_id,
            "status": {"state": state},
            "final": True,
        }
    if event_type == "error":
        return {
            "kind": "status-update",
            "taskId": context_id,
            "contextId": context_id,
            "status": {
                "state": "failed",
                "message": _a2a_message(str(inner_event.get("content", "Unknown error")), context_id),
            },
            "final": True,
        }
    # "tool_chunk", "agent" (handoff narration), or anything else — internal streaming detail
    # with no A2A status-update equivalent; dropped rather than forwarded.
    return None


async def _handle_message_stream(params: dict, user_context: str, request_id: Any) -> StreamingResponse:
    message = params.get("message")
    if not isinstance(message, dict):
        raise IntaGrinError("IG-A2A-001", "params.message is required for message/stream.")
    context_id = _require_context_id(message, params)
    text = _extract_text(message)

    # Deliberately omit initial_state rather than passing None explicitly: ChatRequest types it
    # as a bare `dict` (default None, but the annotation isn't Optional) — Pydantic only skips
    # validating a field against its annotation when the field is genuinely omitted and the
    # default is used as-is, not when None is passed as an explicit value. Passing it explicitly
    # raises a ValidationError; found via this feature's own end-to-end tests.
    chat_req = ChatRequest(message=text, session_id=context_id)
    inner_response = await stream_endpoint(chat_req, user_context=user_context)

    async def a2a_event_generator():
        async for raw_chunk in inner_response.body_iterator:
            if isinstance(raw_chunk, bytes):
                raw_chunk = raw_chunk.decode("utf-8")
            payload_str = raw_chunk.strip()
            if payload_str.startswith("data:"):
                payload_str = payload_str[len("data:"):].strip()
            if not payload_str or payload_str == "[DONE]":
                continue
            try:
                inner_event = json.loads(payload_str)
            except json.JSONDecodeError:
                continue
            a2a_event = _translate_stream_event(inner_event, context_id)
            if a2a_event is None:
                continue
            frame = {"jsonrpc": "2.0", "id": request_id, "result": a2a_event}
            yield f"data: {json.dumps(frame)}\n\n"

    return StreamingResponse(a2a_event_generator(), media_type="text/event-stream")


async def _handle_tasks_get(params: dict, user_context: str) -> dict:
    context_id = params.get("id") or params.get("taskId") or params.get("contextId")
    if not context_id:
        raise IntaGrinError(
            "IG-A2A-001",
            "tasks/get requires an 'id' (the task/context id — IntaGrin's own session_id).",
        )

    project_dir = Path.cwd()
    graph = parse_project(project_dir)
    namespaced_session = f"{user_context}:{context_id}"
    shared = await get_shared_resources_cache().get(graph, project_dir)
    engine = RuntimeEngine(
        graph=graph,
        project_dir=project_dir,
        session_id=namespaced_session,
        shared_resources=shared,
    )
    await engine.initialize()

    if engine.state.get("_pending_approval"):
        state = "input-required"
    elif engine.state.get("_pending_mcp_tasks"):
        state = "working"
    else:
        state = "completed"

    return {
        "id": context_id,
        "contextId": context_id,
        "status": {"state": state},
        "kind": "task",
    }


@app.post("/a2a")
async def a2a_endpoint(request: Request, user_context: str = Depends(verify_auth)):
    """JSON-RPC 2.0 entry point — see the module docstring for exactly which methods are
    supported and why errors come back with JSON-RPC's own numeric error codes rather than
    IntaGrin's string codes. Auth is the same `verify_auth` dependency (X-API-Key / Bearer token)
    every other endpoint in this file already uses — no new auth mechanism, and no bypass: a
    missing/invalid credential is rejected by the FastAPI dependency before this function body
    ever runs, identically to /chat."""
    try:
        body = await request.json()
    except Exception:
        return _jsonrpc_error(None, _JSONRPC_INVALID_REQUEST, "Request body is not valid JSON.", "IG-A2A-001")

    if not isinstance(body, dict) or body.get("jsonrpc") != "2.0" or not isinstance(body.get("method"), str):
        request_id = body.get("id") if isinstance(body, dict) else None
        return _jsonrpc_error(
            request_id,
            _JSONRPC_INVALID_REQUEST,
            "Request must be JSON-RPC 2.0 with a string 'method' field.",
            "IG-A2A-001",
        )

    request_id = body.get("id")
    method = body["method"]
    params = body.get("params")
    if params is None:
        params = {}
    if not isinstance(params, dict):
        return _jsonrpc_error(request_id, _JSONRPC_INVALID_REQUEST, "'params' must be an object.", "IG-A2A-001")

    if method not in _SUPPORTED_METHODS:
        return _jsonrpc_error(
            request_id,
            _JSONRPC_METHOD_NOT_FOUND,
            f"Unsupported method '{method}'. Supported: {sorted(_SUPPORTED_METHODS)}.",
            "IG-A2A-002",
        )

    try:
        if method == "message/send":
            result = await _handle_message_send(params, user_context)
            return JSONResponse(status_code=200, content={"jsonrpc": "2.0", "id": request_id, "result": result})
        if method == "message/stream":
            return await _handle_message_stream(params, user_context, request_id)
        result = await _handle_tasks_get(params, user_context)
        return JSONResponse(status_code=200, content={"jsonrpc": "2.0", "id": request_id, "result": result})
    except IntaGrinError as e:
        return _jsonrpc_error(request_id, _JSONRPC_INVALID_REQUEST, e.message, e.code)
    except Exception as e:
        # Calling chat_endpoint/stream_endpoint directly (not through FastAPI's own request
        # dispatch) means their internal HTTPException/generic-Exception handling never gets a
        # chance to run — any of their errors surface here as plain Python exceptions. Caught
        # broadly and reported as a generic internal error rather than ever leaking a raw
        # traceback/internal detail onto this authenticated-but-external-facing surface.
        Tracer.log_error(f"A2A Error: {e}")
        return _jsonrpc_error(request_id, _JSONRPC_INTERNAL_ERROR, "Internal error handling A2A request.")
