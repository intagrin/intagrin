import json
import os
import secrets
import sqlite3
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from rich.console import Console

from ..compiler.parser import parse_project
from ..config.orchestration_guide import GUIDE as ORCHESTRATION_GUIDE
from ..errors import IntaGrinError
from ..runtime.memory import postgres_connect, postgres_dict_cursor
from ..runtime.run_logger import ensure_schema
from ..tracing.console import Tracer
from .api import ChatRequest as APIChatRequest
from .api import ResumeRequest, chat_endpoint, resume_endpoint, stream_endpoint
from .error_handlers import register_intagrin_error_handlers

console = Console()
app = FastAPI(title="IntaGrin Monitor Dashboard")
register_intagrin_error_handlers(app)
@app.on_event("startup")
async def startup_event():
    import asyncio

    from intagrin.db_migrations.auto_migrate import run_auto_migrations
    await asyncio.to_thread(run_auto_migrations)

    try:
        graph = parse_project(Path.cwd())
        if graph.config.server.auth.type == "none":
            console.print(
                "[bold yellow]⚠ Monitor dashboard is running with server.auth.type: \"none\" — "
                "live agent telemetry, session memory, and chat endpoints are unauthenticated. "
                "Set `server: {auth: {type: api_key}}` in ai.yaml before exposing this beyond "
                "localhost.[/bold yellow]"
            )
        has_approval_gated_tools = any(
            getattr(t, "requires_approval", False)
            for agent_cfg in graph.config.agents.values()
            for t in agent_cfg.tools
        )
        if has_approval_gated_tools and not (
            graph.config.server.auth.approver_env_var or graph.config.server.auth.approvers
        ):
            console.print(
                "[bold yellow]⚠ One or more tools declare requires_approval: true, but neither "
                "server.auth.approver_env_var nor server.auth.approvers is set — the same "
                "credential that triggers a gated tool call can immediately approve it via "
                "/resume with no separate review. Set `server: {auth: {approver_env_var: ...}}` "
                "(single approver) or `approvers: {...}` (named/multi-approver) in ai.yaml to "
                "require a distinct approver credential.[/bold yellow]"
            )
    except Exception:
        pass

security = HTTPBasic(auto_error=False)


def verify_monitor_auth(credentials: HTTPBasicCredentials = Depends(security)):
    project_dir = Path.cwd()
    try:
        graph = parse_project(project_dir)
        auth_cfg = graph.config.server.auth

        if auth_cfg.type == "none":
            return "global_tenant"

        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Basic"},
            )

        if auth_cfg.type == "api_key":
            expected = os.environ.get(auth_cfg.env_var)
            if not expected:
                console.print(
                    f"[bold red]Warning: Auth is enabled but {auth_cfg.env_var} is not set in environment![/bold red]"
                )
                raise IntaGrinError("IG-SRV-001", "Server misconfiguration.")

            # Basic Auth requires *some* username, but IntaGrin has no user accounts here — one
            # shared secret gates the whole dashboard. The key belongs in the password field only;
            # username is a fixed, documented convention ("admin") and is not itself checked, so
            # any browser/curl client can fill it in without a second secret to manage.
            is_valid_pass = secrets.compare_digest(credentials.password, expected)

            if not is_valid_pass:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid API Key",
                    headers={"WWW-Authenticate": "Basic"},
                )
            return "global_tenant"

        if auth_cfg.type == "custom":
            import importlib
            import sys

            if str(project_dir) not in sys.path:
                sys.path.insert(0, str(project_dir))
            mod = importlib.import_module(auth_cfg.custom_module)
            result = mod.verify_token(credentials.password)
            if not result:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid Custom Token",
                    headers={"WWW-Authenticate": "Basic"},
                )
            # Fail closed, not open: see the identical fix in server/api.py's
            # authenticate_token — a non-string truthy return used to silently collapse every
            # authenticated user into "global_tenant", defeating multi-tenant isolation.
            if not isinstance(result, str):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=(
                        "Custom auth module's verify_token() must return the tenant id as a "
                        "string, not a truthy non-string value — refusing to guess a tenant "
                        "namespace."
                    ),
                    headers={"WWW-Authenticate": "Basic"},
                )
            return result

    except (HTTPException, IntaGrinError):
        raise
    except Exception as e:
        Tracer.log_error(f"Monitor Auth Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    return "global_tenant"


def resolve_templates_dir() -> str:
    possible_dirs = [
        Path(__file__).resolve().parent / "templates",
        Path(__file__).resolve().parents[3] / "src" / "intagrin" / "server" / "templates",
        Path.cwd() / "src" / "intagrin" / "server" / "templates",
    ]
    for d in possible_dirs:
        if d.exists() and any(d.glob("*.html*")):
            return str(d)
    return str(Path(__file__).resolve().parent / "templates")


TEMPLATES_DIR = resolve_templates_dir()
templates = Jinja2Templates(directory=TEMPLATES_DIR)


def get_logo_data_uri(filename: str = "logo3.png") -> str:
    possible_paths = [
        Path(__file__).resolve().parent / "static" / filename,
        Path(__file__).resolve().parents[3] / "docs" / "assets" / filename,
        Path(__file__).resolve().parent.parent.parent.parent / "docs" / "assets" / filename,
        Path.cwd() / "docs" / "assets" / filename,
        Path.cwd() / "static" / filename,
    ]
    import base64

    for p in possible_paths:
        if p.exists():
            b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
            return f"data:image/png;base64,{b64}"
    return ""


@app.get("/", dependencies=[Depends(verify_monitor_auth)])
async def serve_dashboard(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="monitor.html",
        context={
            "logo_data_uri": get_logo_data_uri("logo3.png"),
            # logo3.png's wordmark is styled for a dark background ("Inta" renders near-white) —
            # logo1.png is the same mark recolored for a light background ("Inta" in dark navy).
            # Falls back to the dark variant (via the template's `or` below) if not found, same
            # as get_logo_data_uri's own existing empty-string fallback.
            "logo_data_uri_light": get_logo_data_uri("logo1.png"),
        },
    )


@app.get("/api/config", dependencies=[Depends(verify_monitor_auth)])
def get_config():
    project_dir = Path.cwd()
    try:
        from ..runtime.model_info import resolve_context_window

        graph = parse_project(project_dir)
        payload = graph.config.model_dump()
        # Real context window for the configured model, not a hardcoded guess — the Playground's
        # Thermodynamic HUD uses this instead of assuming every model is 128k.
        payload["_context_window"] = resolve_context_window(graph.config.model.primary)
        return payload
    except Exception as e:
        Tracer.log_error(f"Config API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/docs", dependencies=[Depends(verify_monitor_auth)])
def get_docs():
    possible_paths = [
        Path(__file__).resolve().parents[3] / "docs",  # Repo root — only present in a dev checkout
        Path.cwd() / "docs",  # Current working directory
        # Always present, in every install (dev checkout or a regular pip/wheel install) — the
        # same docs/*.md bundle `inta copilot` scaffolds into consumer projects. Without this,
        # `inta monitor` run against any non-editable install (a built package, a container, a
        # real deployment) shows "No Documentation Found" — the two paths above only ever exist
        # in this repo's own dev checkout.
        Path(__file__).resolve().parent.parent / "templates" / "copilot" / "docs",
    ]

    docs_data = []
    try:
        for docs_dir in possible_paths:
            if docs_dir.exists() and any(docs_dir.glob("*.md")):
                for file in sorted(docs_dir.glob("*.md")):
                    docs_data.append(
                        {"filename": file.name, "content": file.read_text()}
                    )
                return docs_data

        # If we got here, no docs were found in any path
        debug_paths = [str(p) for p in possible_paths]
        return [
            {
                "filename": "Error.md",
                "content": "# No Documentation Found\n\nSearched the following paths:\n- "
                + "\n- ".join(debug_paths),
            }
        ]
    except Exception as e:
        Tracer.log_error(f"Docs API Error: {e}")
        return [{"filename": "Error.md", "content": f"# Error Loading Docs\n\n{e!s}"}]


class SyncGraphRequest(BaseModel):
    agent_id: str
    target_id: str
    edge_type: str = "handoff"  # handoff or delegation
    action: str = "add"


@app.post("/api/graph/sync", dependencies=[Depends(verify_monitor_auth)])
def sync_graph(req: SyncGraphRequest):
    project_dir = Path.cwd()
    yaml_file = project_dir / "ai.yaml"
    if not yaml_file.exists():
        raise IntaGrinError("IG-CFG-001", "ai.yaml not found.")

    try:
        import yaml

        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f)

        agents = data.get("agents", {})
        if req.agent_id not in agents:
            raise IntaGrinError("IG-SRV-002", f"Agent {req.agent_id} not found.")
        # Only a new ("add") edge needs its target to exist — removing a reference to an agent
        # that's already gone is cleanup, not new corruption, and must stay allowed.
        if req.action == "add" and req.target_id not in agents:
            raise IntaGrinError(
                "IG-SRV-002",
                f"Target agent '{req.target_id}' not found — cannot create a handoff/delegation to an agent that doesn't exist.",
            )

        agent = agents[req.agent_id]

        if req.edge_type == "handoff":
            handoffs = agent.get("handoffs", [])
            if req.action == "add" and req.target_id not in handoffs:
                handoffs.append(req.target_id)
            elif req.action == "remove" and req.target_id in handoffs:
                handoffs.remove(req.target_id)
            if not handoffs:
                agent.pop("handoffs", None)
            else:
                agent["handoffs"] = handoffs
        elif req.edge_type == "delegation":
            delegations = agent.get("delegations", [])
            if req.action == "add" and req.target_id not in delegations:
                delegations.append(req.target_id)
            elif req.action == "remove" and req.target_id in delegations:
                delegations.remove(req.target_id)
            if not delegations:
                agent.pop("delegations", None)
            else:
                agent["delegations"] = delegations

        from ..config.schema import validate_config_dict

        _config, errors = validate_config_dict(data)
        if errors:
            raise IntaGrinError(
                "IG-SRV-003",
                "This edit would leave ai.yaml invalid, so nothing was written:\n"
                + "\n".join(errors),
            )

        # Re-save YAML preserving order (using default_flow_style=False)
        with open(yaml_file, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        return {"status": "success"}
    except (HTTPException, IntaGrinError):
        raise
    except Exception as e:
        Tracer.log_error(f"Graph Sync Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class ArchitectRequest(BaseModel):
    messages: list[dict[str, str]]


def _read_dotenv_into_environ(env_file: Path) -> None:
    """Blocking disk I/O — run via asyncio.to_thread from the async run_architect below."""
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'").strip('"')
                if k and not os.environ.get(k):
                    os.environ[k] = v


def _list_root_files(project_dir: Path) -> list[str]:
    """Blocking disk I/O — run via asyncio.to_thread from the async run_architect below."""
    return [
        str(f.relative_to(project_dir))
        for f in project_dir.iterdir()
        if f.name not in [".venv", ".git", "__pycache__", ".ai"]
    ]


@app.post("/api/architect", dependencies=[Depends(verify_monitor_auth)])
async def run_architect(req: ArchitectRequest):
    import asyncio

    project_dir = Path.cwd()
    if not (project_dir / "ai.yaml").exists():
        raise IntaGrinError("IG-CFG-001", "No ai.yaml found in current directory.")

    import json

    await asyncio.to_thread(_read_dotenv_into_environ, project_dir / ".env")

    # Resolve active model from ai.yaml
    architect_model = "gemini/gemini-2.5-flash"
    try:
        from ..compiler.parser import parse_project

        graph = await asyncio.to_thread(parse_project, project_dir)
        if graph.config.model and graph.config.model.primary:
            architect_model = graph.config.model.primary
    except Exception:
        pass

    # The agent only gets the root directory structure initially
    root_files = await asyncio.to_thread(_list_root_files, project_dir)

    sys_prompt = f"""You are an expert conversational architect for the `IntaGrin` framework.
Your job is to chat with the user, answer clarifying questions, and modify their project files when requested.

{ORCHESTRATION_GUIDE}

When a task needs one of the six primitives above, use the decision table and confusions section
to pick the right one instead of guessing — this is the single most common source of a wrong
`ai.yaml`. Beyond primitive selection: prefer `spawns.result_schema` (a dotted Pydantic model
path) over instructing a spawned agent to hand data back via `write_state`/`read_state` — it
derives `return_to_creator`'s own tool schema from the model, and the validated result flows back
automatically. Prefer `spawns.on_complete` ([{{key, value}}] state writes applied once a spawn
genuinely completes) paired with `tools[].available_when` (a state condition gating whether a tool
is even offered, same grammar as router conditions) over instructing a spawned agent to call
`write_state` itself to unlock something for its creator. For higher-risk tools,
`required_approvers`/`required_approvals` on the tool config support N-of-M sign-off via named
`server.auth.approvers`, not just a single approval.

SCOPE BOUNDARY: `ai.yaml`/tools/prompts control backend agent orchestration only — there is no
declarative primitive for UI/frontend elements (a button, a screen, a form). If a request
describes UI behavior (e.g. "show a payment button"), say so directly in one turn and ask what
backend behavior it should actually trigger (a `requires_approval` gate before a booking tool
runs, a specific tool call, a state flag a separate UI layer can read) instead of exploring the
codebase hunting for something that structurally cannot exist in this framework.

KEEPING A VALUE OUT OF THE LLM'S OWN CONTEXT: a tool's arguments and its result are both part of
the conversation the LLM sees on every later turn — `read_state`/`write_state` are no exception,
their results flow back exactly like any other tool result. None of that is a way to hide a value
(e.g. a raw email address) from the LLM; `model.guardrails.mask_pii` masks PII broadly across
message content, but a value the LLM itself calls a tool with or reads back is not masked from
future turns. To let a tool use a value the LLM must never see, the value has to enter session
state through a channel outside the chat conversation entirely (a dedicated API endpoint, a form
submission, a webhook — something that writes to state directly, never through an LLM tool call),
and the consuming tool's own Python function must read that value from state internally in its
implementation, without accepting it as an LLM-provided argument and without ever including it in
the tool's returned result content.

DISCOVERY & CLARIFICATION PROTOCOL: Before including a technology decision in `files_to_write`, do not assume a default — ask the user, unless their own message already states the choice. This applies to:
1. Model/Provider: which LLM provider and model (LiteLLM prefix, e.g. `gemini/`, `openai/`, `anthropic/`).
2. Persistence: which memory backend (`sqlite` for local/dev, `postgres`/`redis` for production) — if the task needs cross-session memory or production durability, ask rather than defaulting to sqlite.
3. Authentication: if configuring `inta serve`/`inta monitor` for anything beyond local development, ask whether to use `server.auth.type: api_key` (a shared secret) or `custom` (a project-supplied verify_token function) — `none` means completely unauthenticated, which is rarely what's actually wanted once deployment comes up. If the deployment will be exposed to untrusted callers, also mention `server.rate_limit` (per-caller request/cost/token caps, unlimited by default) as an option rather than assuming it's needed.
4. RAG/Vector Retrieval: if the task needs knowledge-base search, ask what documents to index (`docs_dir`) and confirm the embedding model/provider before adding a `rag:` block.
5. Execution Pattern: autonomous pipeline (`workflows:`) vs interactive conversational routing (`handoffs:`).
6. Tooling & Integrations: vanilla Python functions, MCP servers, OpenAPI wrappers, or a `type: "sandbox"` tool if the agent needs to *execute* code it (or an LLM) generates rather than call a fixed function — and which specific external services/APIs.
If the user's task clearly implies remembering state across agents, still proactively suggest IntaGrin's Shared Typed State (Redux) and a checkpointer as options — but ask which memory backend fits their requirements rather than picking one and writing it.

MODULARITY & SAFETY: If the architecture grows beyond 5-10 agents, recommend splitting it into modular sub-graphs using `imports:` in the `ai.yaml`. You MUST always recommend adding `circuit_breakers:` (e.g. `max_handoffs_per_session: 10`) to prevent LLM infinite loops and runaway API costs. If an agent's tool set mixes an untrusted-output tool (RAG's `search_knowledge_base`, or an MCP/OpenAPI/sandbox tool — `untrusted_output` defaults to true for those) with a separate `requires_approval` tool, recommend gating the sensitive tool with `available_when: "not _untrusted_content_ingested"` — the lethal-trifecta guardrail: once any untrusted-output tool call succeeds this session, that state key stays true, so the gate withholds the sensitive tool for the rest of the session rather than just for one turn. `inta verify` also flags this combination if it's missed.

AUDIT & OPTIMIZATION: If the user asks you to "check implementation", "optimize", or "review", you MUST first explore the codebase and reply with a detailed markdown report of your suggested optimizations (e.g., semantic caching, lazy loading tools, deterministic routers, modular imports, circuit breakers). DO NOT write the files yet. Ask the user for confirmation. Once they approve, execute the changes.

PROJECT ROOT:
{root_files}

You have tools to explore the codebase (`list_directory`, `read_file`). Explore the codebase as needed to understand the context of the user's request. If the user's message or a file you read contains an IntaGrin error code (formatted like `[IG-XXX-000]`), call `lookup_error_code` to understand it before responding. For questions about what's configurable in `ai.yaml` (authentication, memory, guardrails, circuit breakers, server, RAG, routers, tools, etc.), call `lookup_config_reference` FIRST — before exploring project files — since the answer is almost always a schema fact, not something in this specific project's files.

IMPORTANT: When you are done exploring and ready to reply to the user, you MUST respond with a valid JSON object (and no other text) matching this schema:
{{
  "message": "The conversational text response to the user. Ask clarifying questions here, or confirm what you changed.",
  "files_to_write": [
    {{
      "filepath": "relative/path/to/file.ext",
      "content": "The complete updated file content"
    }}
  ]
}}
If you are not proposing any file changes, you can omit the files_to_write key or leave it empty."""

    messages = [{"role": "system", "content": sys_prompt}] + req.messages

    from ..config.reference import render_sections as render_config_sections

    config_reference_sections = sorted(render_config_sections().keys())

    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the contents of a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filepath": {
                            "type": "string",
                            "description": "Relative path to file",
                        }
                    },
                    "required": ["filepath"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_directory",
                "description": "List contents of a directory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dirpath": {
                            "type": "string",
                            "description": "Relative directory path (e.g. '.')",
                        }
                    },
                    "required": ["dirpath"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "lookup_error_code",
                "description": "Look up the title and likely causes for an IntaGrin error code (format IG-XXX-000) seen in an error message, traceback, or API response.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The error code, e.g. 'IG-CFG-003'",
                        }
                    },
                    "required": ["code"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "lookup_config_reference",
                "description": (
                    "Look up the fields, types, defaults, and descriptions for one section of "
                    "IntaGrin's ai.yaml schema — use this for any question about what's "
                    "configurable (auth, memory, guardrails, circuit breakers, server, RAG, "
                    "routers, tools, etc.) instead of exploring project files, since the answer "
                    "is a schema fact true for every project, not something local to this one."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "section": {
                            "type": "string",
                            "enum": config_reference_sections,
                            "description": (
                                "Which config section to look up, e.g. 'AuthConfig' for "
                                "authentication, 'ServerConfig' for inta serve/monitor, "
                                "'MemoryConfig' for conversation history, 'AppConfig' for "
                                "ai.yaml's own root-level fields."
                            ),
                        }
                    },
                    "required": ["section"],
                },
            },
        },
    ]

    try:
        import litellm

        # Agentic Loop
        max_turns = 10
        for _ in range(max_turns):
            response = await litellm.acompletion(
                model=architect_model, messages=messages, tools=tools
            )
            msg = response.choices[0].message

            if msg.tool_calls:
                # Add the assistant's tool call message
                messages.append(msg.model_dump())

                for tc in msg.tool_calls:
                    func_name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments)
                        if func_name == "read_file":
                            fpath = (project_dir / args["filepath"]).resolve()
                            if (
                                project_dir.resolve() not in fpath.parents
                                and fpath != project_dir.resolve()
                            ):
                                content = "Error: Access denied. Cannot read outside workspace."
                            elif fpath.exists() and fpath.is_file():
                                content = await asyncio.to_thread(
                                    fpath.read_text, errors="ignore"
                                )
                            else:
                                content = "Error: File not found."
                        elif func_name == "list_directory":
                            dpath = (project_dir / args["dirpath"]).resolve()
                            if (
                                project_dir.resolve() not in dpath.parents
                                and dpath != project_dir.resolve()
                            ):
                                content = "Error: Access denied. Cannot list outside workspace."
                            elif dpath.exists() and dpath.is_dir():
                                content = await asyncio.to_thread(
                                    lambda: "\\n".join(f.name for f in dpath.iterdir())
                                )
                            else:
                                content = "Error: Directory not found."
                        elif func_name == "lookup_error_code":
                            from ..errors import get_error_spec

                            try:
                                spec = get_error_spec(args["code"].strip())
                                content = f"{spec.code}: {spec.title}\nPossible causes: {spec.causes}"
                            except KeyError:
                                content = f"'{args.get('code')}' is not a recognized IntaGrin error code."
                        elif func_name == "lookup_config_reference":
                            section = args.get("section", "").strip()
                            if section in config_reference_sections:
                                content = render_config_sections()[section]
                            else:
                                content = (
                                    f"'{section}' is not a recognized config section. "
                                    f"Valid sections: {', '.join(config_reference_sections)}"
                                )
                        else:
                            content = "Error: Unknown tool."
                    except Exception as e:
                        Tracer.log_error(f"Architect tool '{func_name}' error: {e}")
                        content = f"Error executing tool: {e}"

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": func_name,
                            "content": content,
                        }
                    )
                continue  # Loop again to let LLM process tool results
            else:
                # No tool calls, must be the final JSON response
                content = msg.content or "{}"
                content = content.strip()
                content = content.removeprefix("```json")
                content = content.removeprefix("```")
                content = content.removesuffix("```")
                content = content.strip()

                try:
                    data = json.loads(content)
                except Exception:
                    # Fallback if LLM failed to output JSON
                    data = {
                        "message": f"I encountered an error formatting my response: {content}"
                    }

                # Make sure message exists
                if "message" not in data:
                    data["message"] = "I have updated the files."

                # Do NOT write files automatically! Pass them to UI for approval.
                return {
                    "status": "success",
                    "message": data["message"],
                    "files_to_write": data.get("files_to_write", []),
                }

        # Exhausted max_turns without ever producing the required no-tool-calls final answer —
        # rather than a dead-end 500 that throws away everything explored so far (every file
        # read, every lookup), force one last completion with `tools` omitted entirely so the
        # LLM has no choice but to answer in text: summarize what it found, or say what's still
        # unclear and ask the user directly. Still goes through the same JSON-parsing path as the
        # normal termination case, so the response shape the frontend expects is unchanged.
        messages.append(
            {
                "role": "user",
                "content": (
                    "You've used all available exploration turns and must stop calling tools "
                    "now. Respond with your final JSON answer based only on what you've already "
                    "found. If that isn't enough to make the change confidently, do not guess — "
                    "explain what's still unclear in the `message` field and ask the user "
                    "directly, with an empty files_to_write."
                ),
            }
        )
        response = await litellm.acompletion(model=architect_model, messages=messages)
        content = (response.choices[0].message.content or "{}").strip()
        content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            data = json.loads(content)
        except Exception:
            data = {"message": content} if content else {}
        if not data.get("message"):
            data["message"] = (
                "I ran out of exploration turns before finishing — could you narrow the "
                "request down, or ask again and I'll pick up from a smaller piece of it?"
            )
        return {
            "status": "success",
            "message": data["message"],
            "files_to_write": data.get("files_to_write", []),
        }

    except (HTTPException, IntaGrinError):
        raise
    except Exception as e:
        Tracer.log_error(f"Architect Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/api/stream/events")
async def stream_events(user_context: str = Depends(verify_monitor_auth)):
    import asyncio
    import json

    from intagrin.tracing.console import EventStreamer

    async def event_generator():
        q = EventStreamer.subscribe()
        try:
            while True:
                event = await q.get()
                yield f"data: {json.dumps(event)}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            EventStreamer.unsubscribe(q)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


class ApplyRequest(BaseModel):
    files_to_write: list[dict[str, str]] = []


@app.post("/api/architect/apply", dependencies=[Depends(verify_monitor_auth)])
def apply_architect(req: ApplyRequest):
    project_dir = Path.cwd()
    try:
        # Validate before writing anything, not per-file as we go: if one of the proposed files
        # is ai.yaml, its content must parse as a valid AppConfig before *any* file in this batch
        # is written. Partially applying (e.g. new tool/prompt files against an ai.yaml that
        # silently didn't update because it was invalid) can leave the project in a worse,
        # inconsistent state than not applying anything.
        for f in req.files_to_write:
            filepath = f.get("filepath")
            content = f.get("content")
            if not filepath or content is None:
                continue
            target = (project_dir / filepath).resolve()
            if target.name == "ai.yaml":
                import yaml as _yaml

                from ..config.schema import validate_config_dict

                try:
                    parsed = _yaml.safe_load(content)
                except Exception as e:
                    raise IntaGrinError(
                        "IG-SRV-003",
                        f"Proposed ai.yaml is not valid YAML, nothing was written: {e}",
                    )
                _config, errors = validate_config_dict(parsed or {})
                if errors:
                    raise IntaGrinError(
                        "IG-SRV-003",
                        "Proposed ai.yaml would be invalid, nothing in this batch was written:\n"
                        + "\n".join(errors),
                    )

        for f in req.files_to_write:
            filepath = f.get("filepath")
            content = f.get("content")
            if filepath and content:
                # Ensure we don't write outside project_dir
                target = (project_dir / filepath).resolve()
                if (
                    project_dir.resolve() in target.parents
                    or target == project_dir.resolve()
                ):
                    # SECURITY FIX: Block dangerous file overwrites (RCE/Credentials)
                    if target.name == ".env" or target.name.startswith("."):
                        raise HTTPException(
                            status_code=403,
                            detail=f"Forbidden: Cannot overwrite environment or hidden file {target.name}",
                        )
                    if "src/intagrin" in str(target.as_posix()):
                        raise HTTPException(
                            status_code=403,
                            detail="Forbidden: Cannot overwrite core framework internals",
                        )

                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(content)
        return {"status": "success"}
    except (HTTPException, IntaGrinError):
        raise
    except Exception as e:
        Tracer.log_error(f"Architect Apply Error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to sync graph: {e}")


@app.post("/api/blueprint/sync", dependencies=[Depends(verify_monitor_auth)])
def sync_blueprint():
    project_dir = Path.cwd()
    yaml_file = project_dir / "ai.yaml"
    blueprint_file = project_dir / "blueprint.md"

    if not yaml_file.exists():
        raise IntaGrinError("IG-CFG-001", "Missing ai.yaml")

    if not blueprint_file.exists():
        blueprint_file.write_text("# Product Blueprint\n\n")

    architect_model = "gemini/gemini-2.5-flash"
    import yaml

    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)
        primary = data.get("model", {}).get("primary")
        if primary:
            architect_model = primary

    import os

    import litellm

    env_file = project_dir / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if k and not os.environ.get(k.strip()):
                    os.environ[k.strip()] = v.strip().strip("'").strip('"')

    sys_prompt = "You are IntaGrin Architect. Read the ai.yaml architecture and completely rewrite the blueprint.md so that it accurately reflects the current technical architecture (agents, tools, handoffs, memory, routing). Maintain the original product vision, just update the technical specifics. ONLY output the raw markdown file content, nothing else."
    prompt = f"CURRENT AI.YAML:\n{yaml_file.read_text()}\n\nCURRENT BLUEPRINT.MD:\n{blueprint_file.read_text()}"

    try:
        resp = litellm.completion(
            model=architect_model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        new_content = resp.choices[0].message.content
        if new_content:
            new_content = (
                new_content.replace("```markdown", "").replace("```", "").strip()
            )
            blueprint_file.write_text(new_content)
        return {"status": "success"}
    except Exception as e:
        Tracer.log_error(f"Blueprint Sync Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/memory")
def get_memory(user_context: str = Depends(verify_monitor_auth)):
    project_dir = Path.cwd()
    try:
        graph = parse_project(project_dir)

        if graph.config.memory.type == "sqlite":
            from ..runtime.memory import SQLiteCheckpointer

            db_path = project_dir / (graph.config.memory.db_path or ".ai/memory.db")
            if not db_path.exists():
                return []

            # Initialize checkpointer to trigger any necessary schema migrations
            SQLiteCheckpointer(str(db_path))

            conn = sqlite3.connect(str(db_path), timeout=15.0)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                "SELECT session_id, messages, state FROM checkpoints "
                "WHERE session_id LIKE ? ORDER BY updated_at DESC LIMIT 10",
                (f"{user_context}:%",),
            )
            rows = cursor.fetchall()

            sessions = []
            for row in rows:
                try:
                    sessions.append(
                        {
                            "session_id": row["session_id"].replace(
                                f"{user_context}:", "", 1
                            ),
                            "messages": (
                                json.loads(row["messages"]) if row["messages"] else []
                            ),
                            "state": json.loads(row["state"]) if row["state"] else {},
                        }
                    )
                except Exception as parse_e:
                    console.print(
                        f"[bold red]Failed to parse session {row['session_id']}: {parse_e}[/bold red]"
                    )
            return sessions

        elif graph.config.memory.type == "postgres":
            mem_cfg = graph.config.memory
            conn_url = mem_cfg.connection_url
            if not conn_url and mem_cfg.env_var:
                conn_url = os.environ.get(mem_cfg.env_var)
            if not conn_url:
                conn_url = os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL")
            if not conn_url:
                return []
                
            try:
                conn = postgres_connect(conn_url)
                with conn:
                    with postgres_dict_cursor(conn) as cursor:
                        cursor.execute(
                            "SELECT session_id, messages, state FROM checkpoints "
                            "WHERE session_id LIKE %s ORDER BY updated_at DESC LIMIT 10",
                            (f"{user_context}:%",),
                        )
                        rows = cursor.fetchall()
                        sessions = []
                        for row in rows:
                            try:
                                sessions.append({
                                    "session_id": row["session_id"].replace(
                                        f"{user_context}:", "", 1
                                    ),
                                    "messages": row["messages"] if isinstance(row["messages"], list) else json.loads(row["messages"]),
                                    "state": row["state"] if isinstance(row["state"], dict) else json.loads(row["state"])
                                })
                            except Exception:
                                pass
                        return sessions
            except ImportError:
                return []
        
        elif graph.config.memory.type == "custom" and graph.config.memory.custom_module:
            import importlib
            import sys

            if str(project_dir) not in sys.path:
                sys.path.insert(0, str(project_dir))
            mod = importlib.import_module(graph.config.memory.custom_module)
            checkpointer = mod.CustomCheckpointer()
            if hasattr(checkpointer, "get_all_sessions"):
                return checkpointer.get_all_sessions(limit=10)
            else:
                return [
                    {
                        "session_id": "Custom Memory (No Monitor Support)",
                        "state": {
                            "info": "Your database is active but visual monitoring is disabled."
                        },
                        "messages": [
                            {
                                "role": "assistant",
                                "content": "To view your traces here, implement `def get_all_sessions(self, limit: int)` in your CustomCheckpointer class returning a list of dicts.",
                            }
                        ],
                    }
                ]
        return []
    except Exception as e:
        Tracer.log_error(f"Memory API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/logs")
def get_logs(user_context: str = Depends(verify_monitor_auth)):
    """Lists recent API-triggered run logs (see runtime/run_logger.py) for the authenticated
    tenant, most recent first — powers the Monitor dashboard's Logs page."""
    project_dir = Path.cwd()
    try:
        graph = parse_project(project_dir)
        mem_cfg = graph.config.memory
        logs = []

        if mem_cfg.type == "sqlite":
            db_path = project_dir / (mem_cfg.db_path or ".ai/memory.db")
            if not db_path.exists():
                return []

            # Ensures run_logs exists even if no run has ever been logged yet (e.g. a fresh
            # project) — without this, the SELECT below would fail with "no such table".
            ensure_schema(mem_cfg, project_dir)

            conn = sqlite3.connect(str(db_path), timeout=15.0)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM run_logs WHERE session_id LIKE ? ORDER BY created_at DESC LIMIT 200",
                (f"{user_context}:%",),
            )
            for row in cursor.fetchall():
                d = dict(row)
                d["session_id"] = (d.get("session_id") or "").replace(
                    f"{user_context}:", "", 1
                )
                logs.append(d)

        elif mem_cfg.type == "postgres":
            conn_url = mem_cfg.connection_url
            if not conn_url and mem_cfg.env_var:
                conn_url = os.environ.get(mem_cfg.env_var)
            if not conn_url:
                conn_url = os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL")
            if not conn_url:
                return []

            ensure_schema(mem_cfg, project_dir)

            try:
                conn = postgres_connect(conn_url)

                with conn, postgres_dict_cursor(conn) as cursor:
                    cursor.execute(
                        "SELECT * FROM run_logs WHERE session_id LIKE %s "
                        "ORDER BY created_at DESC LIMIT 200",
                        (f"{user_context}:%",),
                    )
                    for row in cursor.fetchall():
                        d = dict(row)
                        d["session_id"] = (d.get("session_id") or "").replace(
                            f"{user_context}:", "", 1
                        )
                        logs.append(d)
            except ImportError:
                return []

        return logs
    except Exception as e:
        Tracer.log_error(f"Logs API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def monitor_chat(
    req: APIChatRequest, user_context: str = Depends(verify_monitor_auth)
):
    return await chat_endpoint(req, user_context=user_context)


@app.post("/api/stream")
async def monitor_stream(
    req: APIChatRequest, user_context: str = Depends(verify_monitor_auth)
):
    return await stream_endpoint(req, user_context=user_context)


@app.post("/api/resume")
async def monitor_resume(
    req: ResumeRequest, request: Request, user_context: str = Depends(verify_monitor_auth)
):
    return await resume_endpoint(req, request, user_context=user_context)
