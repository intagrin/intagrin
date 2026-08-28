# Error Code Reference

Every codified IntaGrin error is listed below, grouped by category. Errors not yet migrated to a code keep today's plain-text messages — this list grows incrementally, it is not exhaustive.

This file is generated from `src/intagrin/errors.py` by `scripts/generate_error_docs.py` — do not hand-edit it.

## CLI Usage

| Code | Title | Possible Causes |
|---|---|---|
| `IG-CLI-001` | Target directory already exists | `inta new <name>` was run with a name that already exists as a file or directory in the current location. Choose a different name, or remove/rename the existing directory first. |
| `IG-CLI-002` | Unrecognized --agent value | `inta copilot --agent <value>` was given a value outside the supported set (`copilot`, `cursor`, `claude`, `antigravity`, `factory`). Check spelling and case — the flag is case-sensitive. |
| `IG-CLI-003` | Unrecognized --since duration | `inta simulate --since <value>` was given a duration that doesn't match the supported `<number><unit>` grammar (e.g. `30d`, `12h`, `2w`). |
| `IG-CLI-004` | Blueprint file not found | `inta compile <file>` was pointed at a Markdown blueprint path that doesn't exist relative to the current directory. Check the path, or run `inta compile` without an argument to use the default location. |
| `IG-CLI-005` | Candidate config file not found | `inta simulate --config <file>` references a candidate `ai.yaml` diff file that doesn't exist. Check the path passed to `--config`. |
| `IG-CLI-006` | Server dependencies not installed | The `server` optional dependency group (FastAPI, Uvicorn) isn't installed. Run `pip install intagrin[server]`, or `pip install fastapi uvicorn` directly. |
| `IG-CLI-007` | --judge requires --session-id | `inta eval --judge` was run without `--session-id`. LLM-judge evaluation grades one specific checkpointed session — pass `--session-id <id>` to select it. |
| `IG-CLI-008` | Missing LLM provider API key | A command that calls an LLM before `ai.yaml` exists yet (e.g. `inta compile`) couldn't find the API key its model needs. Create a `.env` file next to the blueprint (or the current directory) with the right variable for your provider — `GEMINI_API_KEY` or `GOOGLE_API_KEY` for a `gemini/...` model, `OPENAI_API_KEY` for `openai/...`, `ANTHROPIC_API_KEY` for `anthropic/...` — or export it in your shell, then re-run the command. |
| `IG-CLI-009` | Built-in scaffold template failed schema validation | `inta new --template <name>` generated an `ai.yaml` that doesn't pass `validate_config_dict` — a bug in IntaGrin's own built-in template, not something your project did. `tests/test_cli.py` asserts every built-in template validates, so this should only ever surface if you're running against a patched/development build; report it if you see it in a released version. |

## Configuration

| Code | Title | Possible Causes |
|---|---|---|
| `IG-CFG-001` | Missing ai.yaml configuration file | The command was run from a directory that has no `ai.yaml`, or the working directory isn't the project root. Run `inta new <name>` to scaffold a project, or `cd` into the directory that contains `ai.yaml`. |
| `IG-CFG-002` | Invalid YAML syntax or import failure | `ai.yaml` (or a file referenced via `imports:`) has invalid YAML syntax, or an `imports:` entry points at a file that fails to parse. Check indentation and run the file through a YAML linter; the underlying parser error is included in the message. |
| `IG-CFG-003` | ai.yaml failed schema validation | One or more fields in `ai.yaml` don't match IntaGrin's schema — a missing required field, wrong type, or unrecognized value. The full list of field-level errors is included in the message; cross-reference `config/schema.py` or the AI YAML Blueprint doc. |

## MCP Integration

| Code | Title | Possible Causes |
|---|---|---|
| `IG-MCP-001` | MCP tool not found | An agent (or the LLM mid-conversation) tried to call a tool name that no connected MCP server exposes — usually a stale or incorrect tool name in a prompt, or the MCP server process didn't register the tool that was expected. |

## Runtime

| Code | Title | Possible Causes |
|---|---|---|
| `IG-RT-001` | No PostgreSQL connection URL | `memory: {type: postgres}` is configured but no `connection_url`, `env_var`-referenced variable, or `DATABASE_URL`/`POSTGRES_URL` environment variable resolves to a value. Only raised in strict mode (`inta replay`/`inta simulate`) — the live runtime instead falls back to SQLite with a logged warning. |
| `IG-RT-002` | No Redis connection URL | `memory: {type: redis}` is configured but no `connection_url`, `env_var`-referenced variable, or `REDIS_URL` environment variable resolves to a value. Only raised in strict mode — the live runtime falls back to SQLite with a logged warning instead. |
| `IG-RT-003` | Unsupported memory type for strict tooling | `memory.type` is set to a custom or unrecognized backend, and strict-mode CLI tooling (`inta replay`/`inta simulate`) has no built-in way to enumerate a user-supplied backend's sessions. |
| `IG-RT-004` | PostgreSQL driver not installed | `memory: {type: postgres}` is configured but neither `psycopg[pool]` nor `psycopg2` is installed. Install one of them. |
| `IG-RT-005` | Redis package not installed | `memory: {type: redis}` is configured but the `redis` package isn't installed. Run `pip install redis`. |
| `IG-RT-006` | Local tool failed to load | A `type: local` tool in `ai.yaml` references a Python module or function that doesn't exist or fails to import — a typo in `module`/`name`, a missing dependency, or a syntax error inside `tools/*.py`. |
| `IG-RT-007` | Circuit breaker tripped — session halted | A session hit `circuit_breakers.max_handoffs_per_session` or `max_tool_failures_in_a_row`. This is a deliberate stop, not a bug: the session was looping (repeated handoffs via transfer_agent, a conditional/root router, or auto_route semantic routing) or a tool kept failing on consecutive calls. Raise the relevant `circuit_breakers` threshold in `ai.yaml` if the limit is too tight for a legitimate long-running task, or fix the underlying loop/failing tool. |
| `IG-RT-008` | Rate limit exceeded | One authenticated caller hit a `server.rate_limit` threshold — `max_requests_per_window`, `max_cost_per_caller_per_day`, or `max_tokens_per_caller_per_day` — measured from that caller's rows in the run_logs audit table. This is a deliberate stop, not a bug: either the caller is making more requests than the configured quota allows, or the threshold in `ai.yaml` is too tight for legitimate traffic and should be raised. |

## Server & API

| Code | Title | Possible Causes |
|---|---|---|
| `IG-SRV-001` | Server misconfigured: auth secret not set | `server: {auth: {type: api_key}}` (or similar) is configured but the environment variable it references isn't set in the server process's environment. Export the secret before starting `inta serve`/`inta monitor`. |
| `IG-SRV-002` | Referenced agent does not exist | A Studio graph edit (drag-and-drop handoff/delegation) referenced an agent id that isn't defined in the current `ai.yaml` — usually a stale UI state after a manual edit to the file. Reload the dashboard. |
| `IG-SRV-003` | Proposed ai.yaml change is invalid | A Studio graph edit or an Architect-proposed file change, if applied, would produce an `ai.yaml` that fails schema/router-condition validation. Nothing was written — the validation errors are included in the message; fix the proposed change and retry. |
| `IG-SRV-004` | Project file cannot be previewed | `GET /api/files/{path}` (used to preview a tool's generated image/audio/video, e.g. in the Approval card or a tool-call/result bubble) was asked for a path outside the project directory, a file extension it doesn't recognize as previewable media, a file that doesn't exist, or a file larger than its preview size limit. This endpoint is intentionally narrower than the Architect chat's own `read_file` tool — it only ever serves a fixed allowlist of media types, inline, for display. |
