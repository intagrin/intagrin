"""Structured error codes for IntaGrin.

A single registry (`ERRORS`) is the source of truth for every codified error: the code itself,
its human category, a short title, and prose describing likely causes. `docs/12_Error_Reference.md`
and `templates/copilot/reference_error_codes.md` are both generated from this registry by
`scripts/generate_error_docs.py` — never hand-edit those files directly.

This module has no dependency on Rich, FastAPI, or Typer so it can be imported from the CLI,
runtime, and server layers alike without pulling in unrelated machinery.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ErrorSpec:
    code: str
    category: str
    title: str
    causes: str
    default_http_status: int | None = None


_SPECS: list[ErrorSpec] = [
    ErrorSpec(
        code="IG-CFG-001",
        category="Configuration",
        title="Missing ai.yaml configuration file",
        causes=(
            "The command was run from a directory that has no `ai.yaml`, or the working "
            "directory isn't the project root. Run `inta new <name>` to scaffold a project, or "
            "`cd` into the directory that contains `ai.yaml`."
        ),
        default_http_status=400,
    ),
    ErrorSpec(
        code="IG-CFG-002",
        category="Configuration",
        title="Invalid YAML syntax or import failure",
        causes=(
            "`ai.yaml` (or a file referenced via `imports:`) has invalid YAML syntax, or an "
            "`imports:` entry points at a file that fails to parse. Check indentation and run "
            "the file through a YAML linter; the underlying parser error is included in the "
            "message."
        ),
        default_http_status=400,
    ),
    ErrorSpec(
        code="IG-CFG-003",
        category="Configuration",
        title="ai.yaml failed schema validation",
        causes=(
            "One or more fields in `ai.yaml` don't match IntaGrin's schema — a missing required "
            "field, wrong type, or unrecognized value. The full list of field-level errors is "
            "included in the message; cross-reference `config/schema.py` or the AI YAML "
            "Blueprint doc."
        ),
        default_http_status=400,
    ),
    ErrorSpec(
        code="IG-CLI-001",
        category="CLI Usage",
        title="Target directory already exists",
        causes=(
            "`inta new <name>` was run with a name that already exists as a file or directory "
            "in the current location. Choose a different name, or remove/rename the existing "
            "directory first."
        ),
    ),
    ErrorSpec(
        code="IG-CLI-002",
        category="CLI Usage",
        title="Unrecognized --agent value",
        causes=(
            "`inta copilot --agent <value>` was given a value outside the supported set "
            "(`copilot`, `cursor`, `claude`, `antigravity`, `factory`). Check spelling and "
            "case — the flag is case-sensitive."
        ),
    ),
    ErrorSpec(
        code="IG-CLI-003",
        category="CLI Usage",
        title="Unrecognized --since duration",
        causes=(
            "`inta simulate --since <value>` was given a duration that doesn't match the "
            "supported `<number><unit>` grammar (e.g. `30d`, `12h`, `2w`)."
        ),
    ),
    ErrorSpec(
        code="IG-CLI-004",
        category="CLI Usage",
        title="Blueprint file not found",
        causes=(
            "`inta compile <file>` was pointed at a Markdown blueprint path that doesn't exist "
            "relative to the current directory. Check the path, or run `inta compile` without "
            "an argument to use the default location."
        ),
    ),
    ErrorSpec(
        code="IG-CLI-005",
        category="CLI Usage",
        title="Candidate config file not found",
        causes=(
            "`inta simulate --config <file>` references a candidate `ai.yaml` diff file that "
            "doesn't exist. Check the path passed to `--config`."
        ),
    ),
    ErrorSpec(
        code="IG-CLI-006",
        category="CLI Usage",
        title="Server dependencies not installed",
        causes=(
            "The `server` optional dependency group (FastAPI, Uvicorn) isn't installed. Run "
            "`pip install intagrin[server]`, or `pip install fastapi uvicorn` directly."
        ),
    ),
    ErrorSpec(
        code="IG-CLI-007",
        category="CLI Usage",
        title="--judge requires --session-id",
        causes=(
            "`inta eval --judge` was run without `--session-id`. LLM-judge evaluation grades "
            "one specific checkpointed session — pass `--session-id <id>` to select it."
        ),
    ),
    ErrorSpec(
        code="IG-CLI-008",
        category="CLI Usage",
        title="Missing LLM provider API key",
        causes=(
            "A command that calls an LLM before `ai.yaml` exists yet (e.g. `inta compile`) "
            "couldn't find the API key its model needs. Create a `.env` file next to the "
            "blueprint (or the current directory) with the right variable for your provider — "
            "`GEMINI_API_KEY` or `GOOGLE_API_KEY` for a `gemini/...` model, `OPENAI_API_KEY` for "
            "`openai/...`, `ANTHROPIC_API_KEY` for `anthropic/...` — or export it in your shell, "
            "then re-run the command."
        ),
    ),
    ErrorSpec(
        code="IG-CLI-009",
        category="CLI Usage",
        title="Built-in scaffold template failed schema validation",
        causes=(
            "`inta new --template <name>` generated an `ai.yaml` that doesn't pass "
            "`validate_config_dict` — a bug in IntaGrin's own built-in template, not something "
            "your project did. `tests/test_cli.py` asserts every built-in template validates, so "
            "this should only ever surface if you're running against a patched/development "
            "build; report it if you see it in a released version."
        ),
    ),
    ErrorSpec(
        code="IG-RT-001",
        category="Runtime",
        title="No PostgreSQL connection URL",
        causes=(
            "`memory: {type: postgres}` is configured but no `connection_url`, "
            "`env_var`-referenced variable, or `DATABASE_URL`/`POSTGRES_URL` environment "
            "variable resolves to a value. Only raised in strict mode (`inta replay`/"
            "`inta simulate`) — the live runtime instead falls back to SQLite with a logged "
            "warning."
        ),
    ),
    ErrorSpec(
        code="IG-RT-002",
        category="Runtime",
        title="No Redis connection URL",
        causes=(
            "`memory: {type: redis}` is configured but no `connection_url`, "
            "`env_var`-referenced variable, or `REDIS_URL` environment variable resolves to a "
            "value. Only raised in strict mode — the live runtime falls back to SQLite with a "
            "logged warning instead."
        ),
    ),
    ErrorSpec(
        code="IG-RT-003",
        category="Runtime",
        title="Unsupported memory type for strict tooling",
        causes=(
            "`memory.type` is set to a custom or unrecognized backend, and strict-mode CLI "
            "tooling (`inta replay`/`inta simulate`) has no built-in way to enumerate a "
            "user-supplied backend's sessions."
        ),
    ),
    ErrorSpec(
        code="IG-RT-004",
        category="Runtime",
        title="PostgreSQL driver not installed",
        causes=(
            "`memory: {type: postgres}` is configured but neither `psycopg[pool]` nor "
            "`psycopg2` is installed. Install one of them."
        ),
    ),
    ErrorSpec(
        code="IG-RT-005",
        category="Runtime",
        title="Redis package not installed",
        causes=(
            "`memory: {type: redis}` is configured but the `redis` package isn't installed. "
            "Run `pip install redis`."
        ),
    ),
    ErrorSpec(
        code="IG-RT-006",
        category="Runtime",
        title="Local tool failed to load",
        causes=(
            "A `type: local` tool in `ai.yaml` references a Python module or function that "
            "doesn't exist or fails to import — a typo in `module`/`name`, a missing "
            "dependency, or a syntax error inside `tools/*.py`."
        ),
    ),
    ErrorSpec(
        code="IG-RT-007",
        category="Runtime",
        title="Circuit breaker tripped — session halted",
        causes=(
            "A session hit `circuit_breakers.max_handoffs_per_session` or "
            "`max_tool_failures_in_a_row`. This is a deliberate stop, not a bug: the session "
            "was looping (repeated handoffs via transfer_agent, a conditional/root router, or "
            "auto_route semantic routing) or a tool kept failing on consecutive calls. Raise "
            "the relevant `circuit_breakers` threshold in `ai.yaml` if the limit is too tight "
            "for a legitimate long-running task, or fix the underlying loop/failing tool."
        ),
    ),
    ErrorSpec(
        code="IG-RT-008",
        category="Runtime",
        title="Rate limit exceeded",
        causes=(
            "One authenticated caller hit a `server.rate_limit` threshold — "
            "`max_requests_per_window`, `max_cost_per_caller_per_day`, or "
            "`max_tokens_per_caller_per_day` — measured from that caller's rows in the run_logs "
            "audit table. This is a deliberate stop, not a bug: either the caller is making more "
            "requests than the configured quota allows, or the threshold in `ai.yaml` is too "
            "tight for legitimate traffic and should be raised."
        ),
        default_http_status=429,
    ),
    ErrorSpec(
        code="IG-MCP-001",
        category="MCP Integration",
        title="MCP tool not found",
        causes=(
            "An agent (or the LLM mid-conversation) tried to call a tool name that no "
            "connected MCP server exposes — usually a stale or incorrect tool name in a "
            "prompt, or the MCP server process didn't register the tool that was expected."
        ),
    ),
    ErrorSpec(
        code="IG-SRV-001",
        category="Server & API",
        title="Server misconfigured: auth secret not set",
        causes=(
            "`server: {auth: {type: api_key}}` (or similar) is configured but the environment "
            "variable it references isn't set in the server process's environment. Export the "
            "secret before starting `inta serve`/`inta monitor`."
        ),
        default_http_status=500,
    ),
    ErrorSpec(
        code="IG-SRV-002",
        category="Server & API",
        title="Referenced agent does not exist",
        causes=(
            "A Studio graph edit (drag-and-drop handoff/delegation) referenced an agent id "
            "that isn't defined in the current `ai.yaml` — usually a stale UI state after a "
            "manual edit to the file. Reload the dashboard."
        ),
        default_http_status=400,
    ),
    ErrorSpec(
        code="IG-SRV-003",
        category="Server & API",
        title="Proposed ai.yaml change is invalid",
        causes=(
            "A Studio graph edit or an Architect-proposed file change, if applied, would "
            "produce an `ai.yaml` that fails schema/router-condition validation. Nothing was "
            "written — the validation errors are included in the message; fix the proposed "
            "change and retry."
        ),
        default_http_status=400,
    ),
]

ERRORS: dict[str, ErrorSpec] = {spec.code: spec for spec in _SPECS}


def get_error_spec(code: str) -> ErrorSpec:
    """Look up a registered error code. Raises KeyError for an unregistered code."""
    return ERRORS[code]


class IntaGrinError(Exception):
    """Base class for all codified IntaGrin errors.

    Subclasses (`ParserError`, `CheckpointerConfigError`) keep their existing name and import
    path for backward compatibility with existing `except` clauses — only their base class and
    constructor gain a code/message.
    """

    def __init__(self, code: str, message: str | None = None, *, http_status: int | None = None):
        spec = get_error_spec(code)
        self.code = code
        self.message = message or spec.title
        self.title = spec.title
        self.http_status = http_status if http_status is not None else spec.default_http_status
        super().__init__(f"[{code}] {self.message}")

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


class AwaitingHumanInput(Exception):
    """Raise from inside a local tool function to dynamically pause the session for human
    review — the runtime counterpart to statically flagging a whole tool with
    `requires_approval: true` in ai.yaml. Where requires_approval gates every call to a tool,
    this lets a single call decide at runtime that it specifically needs a human (a missing
    required value, a risky computed input, a result that should be confirmed before finishing).

    Caught in RuntimeEngine.execute_tool at the same point the static requires_approval headless
    path already sets `_pending_approval` — resumed identically, via POST /resume. Not an
    IntaGrinError: this isn't a failure to diagnose, it's an expected pause.

    IMPORTANT for tool authors: resuming re-invokes the tool function from the start (with
    edited_args if the reviewer supplied them, otherwise the original args) — this is not a
    continuation. Don't perform non-idempotent side effects before the point where you might
    raise this.
    """

    def __init__(self, prompt: str, context: dict | None = None):
        self.prompt = prompt
        self.context = context or {}
        super().__init__(prompt)
