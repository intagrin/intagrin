from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictBaseModel(BaseModel):
    """Base for every ai.yaml config model: rejects unknown fields instead of silently dropping
    them (Pydantic v2's default `extra="ignore"`). Without this, a typo'd key (`toolz` instead of
    `tools`) or a stale/renamed field validated as a no-op — no error at parse time, no warning at
    runtime, the agent just silently didn't get the tool. The strict `additionalProperties: false`
    JSON Schema in `json_schema.py` only ever protected editors that enforce the
    `yaml-language-server` directive; this makes the same guarantee hold for `parse_project`,
    `validate_config_dict`, and every other real construction path, matching what
    `_generate_and_validate_wizard_config`'s own docstring already assumed was true. Also sharpens
    the untagged tool unions (`LocalToolConfig | ... | ToolReferenceConfig`, no `discriminator=`):
    with `extra="ignore"`, a `{name, module}` dict could validate against both LocalToolConfig
    -and- ToolReferenceConfig (the latter just silently dropping `module`), leaving Pydantic's
    union-member choice to its internal tie-breaking; forbidding extras means only the member
    whose actual fields match wins."""

    model_config = ConfigDict(extra="forbid")


class AuthConfig(StrictBaseModel):
    type: Literal["api_key", "custom", "none"] = Field(
        default="none",
        description=(
            "Authentication mode for `inta serve`/`inta monitor`: 'none' (no auth), 'api_key' "
            "(a single shared secret — sent as an `Authorization: Bearer` token for `inta "
            "serve`'s API, or as the password in HTTP Basic auth for `inta monitor`'s dashboard; "
            "the Basic auth username is not checked and can be any value, e.g. 'admin'), or "
            "'custom' (delegates to a project-supplied verify_token function)."
        ),
    )
    env_var: str | None = Field(
        default="INTAGRIN_API_KEY",
        description=(
            "Environment variable holding the API key, when type is 'api_key'. The server reads "
            "this at each auth check — export it before starting the server."
        ),
    )
    custom_module: str | None = Field(
        default=None,
        description=(
            "Python module path (e.g. 'auth.custom') exposing `verify_token(token: str) -> bool "
            "| str`, when type is 'custom'. Must return the tenant id as a string for per-tenant "
            "session isolation — any other truthy value is rejected (401), not silently treated "
            "as a single shared tenant."
        ),
    )
    approver_env_var: str | None = Field(
        default=None,
        description=(
            "Environment variable holding a separate secret required (via the X-Approver-Key "
            "header) to approve a requires_approval tool call through /resume. Without this, the "
            "same credential that triggered the gated call can immediately approve it — set this "
            "to require a distinct reviewer credential from the requester's own session auth. "
            "Acts as the single default approver (id 'default') when `approvers` isn't set."
        ),
    )
    approvers: dict[str, str] | None = Field(
        default=None,
        description=(
            "Named approvers for multi-approver chains: maps an approver id (e.g. 'finance', "
            "'security') to the environment variable holding that approver's own X-Approver-Key "
            "secret. A tool's `required_approvers` list names which of these ids must each sign "
            "off via /resume before it executes. Independent of `approver_env_var` — set both to "
            "keep a single default approver alongside named ones."
        ),
    )
    admin_env_var: str | None = Field(
        default=None,
        description=(
            "Environment variable holding the secret required (as an `Authorization: Bearer` "
            "token) to call the approver-management endpoints (POST/GET /approvers, DELETE "
            "/approvers/{id}) — issuing, listing, and revoking the DB-backed reviewer credentials "
            "from runtime/approvers.py over HTTP instead of only via `inta approvers` locally, so "
            "a consumer's own admin site/tooling can manage them. Deliberately its own credential "
            "tier, separate from both the main session auth (`env_var`) and any individual "
            "approver's own X-Approver-Key — without that separation, whoever holds the main API "
            "key could mint themselves an approver credential and immediately approve their own "
            "gated tool call. Unset (default) disables these endpoints entirely (503) rather than "
            "falling back to any other credential, since there is no pre-existing behavior on "
            "these endpoints to preserve."
        ),
    )


class RateLimitConfig(StrictBaseModel):
    max_requests_per_window: int | None = Field(
        default=None,
        description=(
            "Max /chat, /chat/stream, /resume, or /stream requests one authenticated caller "
            "(tenant) may make within `window_seconds`. None (default) means unlimited. Enforced "
            "by counting that caller's rows in the run_logs audit table, so it only applies when "
            "memory.type is 'sqlite' or 'postgres' — the same scope run_logs itself has."
        ),
    )
    window_seconds: int = Field(
        default=60,
        gt=0,
        description="Rolling window, in seconds, that max_requests_per_window is measured over.",
    )
    max_cost_per_caller_per_day: float | None = Field(
        default=None,
        description=(
            "Max total USD cost one authenticated caller may accrue in a rolling 24h window, "
            "summed from run_logs.cost_delta. None (default) means unlimited."
        ),
    )
    max_tokens_per_caller_per_day: int | None = Field(
        default=None,
        description=(
            "Max total tokens one authenticated caller may consume in a rolling 24h window, "
            "summed from run_logs.tokens_delta. None (default) means unlimited."
        ),
    )


class ServerConfig(StrictBaseModel):
    auth: AuthConfig = Field(
        default_factory=AuthConfig,
        description=(
            "Authentication configuration for `inta serve`/`inta monitor`. Defaults to no "
            "authentication — set this before exposing either beyond localhost."
        ),
    )
    webhook_url: str | None = Field(
        default=None, description="Webhook URL for async HITL notifications"
    )
    webhook_secret_env_var: str | None = Field(
        default=None, description="Env var containing secret to authenticate webhooks"
    )
    rate_limit: RateLimitConfig = Field(
        default_factory=RateLimitConfig,
        description=(
            "Per-caller rate limiting/usage quotas for the API server. All thresholds default to "
            "unlimited — opt in per project."
        ),
    )


class GuardrailsConfig(StrictBaseModel):
    banned_words: list[str] = Field(
        default_factory=list,
        description=(
            "Words/phrases that, if present in a user message or model output, are blocked "
            "before reaching the LLM or the user."
        ),
    )
    mask_pii: bool = Field(
        default=False,
        description=(
            "Redact common PII patterns (emails, SSNs, card numbers) from messages before "
            "they're sent to the LLM or logged."
        ),
    )
    system_safeguards: bool = Field(
        default=False,
        description=(
            "Append a built-in safety instruction to the system prompt discouraging "
            "harmful/off-policy behavior."
        ),
    )
    custom_module: str | None = Field(
        default=None,
        description=(
            "Python module path exposing a custom guardrail check function, run in addition to "
            "the built-in checks above."
        ),
    )


class ModelVariantConfig(StrictBaseModel):
    model: str = Field(description="LiteLLM model identifier for this variant.")
    weight: float = Field(
        gt=0.0, description="Relative weight for traffic splitting — weights don't need to sum to 1."
    )


class ModelConfig(StrictBaseModel):
    primary: str = Field(
        description=(
            "LiteLLM model identifier used for this agent/app (e.g. 'openai/gpt-4o-mini', "
            "'anthropic/claude-3-5-sonnet'). Required."
        )
    )
    fallback: str | None = Field(
        default=None,
        description="Model to retry with if the primary model call fails (rate limit, outage, etc.).",
    )
    variants: list[ModelVariantConfig] | None = Field(
        default=None,
        description=(
            "A/B or canary model routing: split traffic across weighted model variants instead "
            "of always using `primary`. Assignment is deterministic per session_id (sticky for "
            "the whole conversation, never flips mid-session) via a weighted hash. None "
            "(default) means every session uses `primary`, exactly today's behavior."
        ),
    )
    cascade: list[str] | None = Field(
        default=None,
        description=(
            "Ordered cheap-to-expensive LiteLLM model identifiers tried for an agent's "
            "response_schema-validated answer — the whole turn (including any tool calls it "
            "makes) runs on cascade[0] instead of `primary`, escalating to the next tier only "
            "when the terminal response fails response_schema validation. Ignored entirely for "
            "an agent without response_schema set: schema pass/fail is the only free, already-"
            "computed confidence signal this framework has for judging whether a cheaper model's "
            "answer was good enough — there's no generic way to judge an unstructured chat "
            "answer without spending another LLM call, which would eat into exactly the savings "
            "this exists to provide. Escalation only ever regenerates the already-terminal text "
            "response off the existing message history — it never re-executes any tool calls the "
            "turn already made, so it's safe regardless of what tools the agent used along the "
            "way. `primary` is always the final escalation tier regardless of what's listed here, "
            "so a cascade can never produce a worse ceiling than not configuring one at all. "
            "Lower priority than both `model_override` and a resolved `variants` assignment — an "
            "explicit per-agent override or an active A/B experiment both win over an automatic "
            "cost cascade."
        ),
    )
    temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="Sampling temperature passed to the LLM, 0.0 (deterministic) to 2.0 (most random).",
    )
    max_tokens: int = Field(
        default=1500, gt=0, description="Maximum tokens the LLM may generate in a single completion."
    )
    use_cache: bool = Field(
        default=False, description="Enable semantic caching to save API costs"
    )
    enable_prompt_caching: bool = Field(
        default=True,
        description=(
            "Automatically mark the system prompt and the trailing turn with Anthropic-style "
            "cache_control breakpoints (sets litellm.enable_anthropic_prompt_caching) so a "
            "stable system-prompt/tool-schema prefix is billed once and reused across turns "
            "instead of being re-priced as fresh input tokens every turn. Safe to leave on "
            "regardless of provider: LiteLLM only actually injects breakpoints for "
            "Anthropic/Bedrock Claude models that report prompt-caching support, and stands "
            "down entirely if a request already carries its own cache_control. Unrelated to "
            "`use_cache` (LiteLLM's semantic response cache, a full-response cache-hit "
            "feature) — this caches the prompt prefix itself, not the response."
        ),
    )
    guardrails: GuardrailsConfig = Field(
        default_factory=GuardrailsConfig,
        description="Content-safety checks applied to this model's inputs/outputs.",
    )


class MemoryConfig(StrictBaseModel):
    type: Literal[
        "sliding_window", "buffer", "sqlite", "postgres", "redis", "custom"
    ] = Field(
        default="sliding_window",
        description=(
            "Conversation history backend: 'sliding_window'/'buffer' (in-process, lost on "
            "restart), 'sqlite' (local file), 'postgres', 'redis', or 'custom'."
        ),
    )
    max_messages: int = Field(
        default=20,
        gt=0,
        description="Number of most-recent messages kept in context for sliding_window/buffer memory.",
    )
    db_path: str | None = Field(
        default=".ai/memory.db",
        description="SQLite database file path, relative to the project root, when type is 'sqlite'.",
    )
    connection_url: str | None = Field(
        default=None,
        description=(
            "Direct connection URL for postgres/redis, e.g. 'postgresql://...' or 'redis://...'. "
            "Takes precedence over env_var."
        ),
    )
    env_var: str | None = Field(
        default=None,
        description=(
            "Environment variable holding the connection URL for postgres/redis (e.g. "
            "'DATABASE_URL', 'REDIS_URL'), used if connection_url isn't set directly."
        ),
    )
    custom_module: str | None = Field(
        default=None,
        description="Python module path exposing a CustomCheckpointer class, when type is 'custom'.",
    )
    shared_scope: Literal["session", "tenant", "global"] = Field(
        default="session",
        description=(
            "Scope for the long_term_memory summary _compress_memory produces: 'session' "
            "(default — today's behavior, private to one session_id), 'tenant' (shared across "
            "every session under the same authenticated caller/tenant), or 'global' (shared "
            "across every session in the project, any tenant). Persisted to a new shared_memory "
            "table (sqlite/postgres only, same scope as run_logs) and merged into a session's "
            "long_term_memory on initialize(). Last-write-wins across concurrent sessions writing "
            "the same scope — no merge/versioning."
        ),
    )
    run_log_retention_days: int | None = Field(
        default=None,
        description=(
            "Age (in days) after which run_logs rows (one per API-triggered /chat, /chat/stream, "
            "/resume, /stream call — see runtime/run_logger.py) are deleted. None (default) means "
            "no automatic pruning — run_logs grows forever, which also slows down the rate "
            "limiter's own queries against it (runtime/rate_limiter.py scans this same table on "
            "every request) as the table grows over a long production lifetime. Enforced "
            "opportunistically (a small random chance on each record_run_log write triggers a "
            "single indexed DELETE) rather than needing an external cron job — sqlite/postgres "
            "only, same scope as run_logs itself."
        ),
    )


class ConditionFunctionConfig(StrictBaseModel):
    name: str = Field(
        description=(
            "Name a routers[].condition or tools[].available_when expression can call, e.g. "
            "`is_eligible(customer_tier, order_total)`. Must match the Python function name."
        )
    )
    module: str = Field(
        description=(
            "Python module path (e.g. 'tools.condition_functions') containing a pure, "
            "side-effect-free function named `name` that takes plain values and returns bool. "
            "Called with already-evaluated bare state-key names/literals as positional "
            "arguments — the function itself is never parsed as part of the condition grammar, "
            "only invoked by name, so this is the one way to express branching logic that the "
            "restricted condition grammar's comparisons/and/or/not can't reach (e.g. a regex "
            "match or a multi-field business rule) without falling back to an LLM handoff."
        )
    )


class LocalToolConfig(StrictBaseModel):
    name: str = Field(
        description="Tool name exposed to the LLM for tool-calling — must match the Python function name."
    )
    module: str = Field(
        description="Python module path (e.g. 'tools.custom_tools') containing the function named `name`."
    )
    requires_approval: bool = Field(
        default=False, description="Require human-in-the-loop approval before this tool actually executes."
    )
    required_approvals: int = Field(
        default=1,
        ge=1,
        description=(
            "Number of distinct approvers (see server.auth.approvers) who must each approve via "
            "/resume before this tool executes — 1 (default) matches today's single-approval "
            "behavior. Ignored unless requires_approval is true."
        ),
    )
    required_approvers: list[str] | None = Field(
        default=None,
        description=(
            "Specific approver ids (keys of server.auth.approvers) that must each sign off, "
            "instead of any `required_approvals` approvers. Ignored unless requires_approval is "
            "true; when set, overrides required_approvals with len(required_approvers)."
        ),
    )
    available_when: str | None = Field(
        default=None,
        description=(
            "State condition (same restricted grammar as routers[].condition — bare state-key "
            "names, comparisons, and/or/not; no method calls or attribute access) gating whether "
            "this tool is even offered to the agent this turn. Unlike a prompt instruction asking "
            "the model not to call a tool yet, the tool is structurally absent from its schema "
            "until the condition is true — re-checked server-side on every call regardless, the "
            "same defense-in-depth already applied to tool_pool and every other schema-driven "
            "gate in this codebase. None (default) means always available, today's behavior."
        ),
    )
    untrusted_output: bool = Field(
        default=False,
        description=(
            "Mark this tool's return value as untrusted (may contain LLM-directed instructions "
            "injected by a third party — the 'lethal trifecta' pattern: untrusted content + "
            "access to private data/state + a way to exfiltrate). False by default for local "
            "tools, since they're developer-authored Python — set true for one that fetches "
            "external content (e.g. a web scraper). The moment any tool call with "
            "untrusted_output=true succeeds, state['_untrusted_content_ingested'] is set true for "
            "the rest of the session; reference it from another tool's available_when (e.g. "
            "'not _untrusted_content_ingested') to withhold a sensitive tool once this session has "
            "seen untrusted content, or from a router condition to force a review handoff."
        ),
    )


class MCPToolConfig(StrictBaseModel):
    name: str = Field(description="Tool name exposed to the LLM for tool-calling.")
    type: Literal["mcp"] = Field(description="Discriminator — must be the literal string 'mcp'.")
    command: str = Field(
        description="Executable used to launch the MCP server subprocess (e.g. 'npx')."
    )
    args: list[str] = Field(
        description="Arguments passed to `command` when launching the MCP server."
    )
    env: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Environment variables set on the MCP server subprocess in addition to the parent "
            "process's own environment — e.g. an API key a specific MCP server needs that "
            "shouldn't be exported globally."
        ),
    )
    max_task_wait_seconds: int | None = Field(
        default=None,
        ge=1,
        description=(
            "For MCP servers that support the Tasks extension (long-running tool calls claimed "
            "instead of answered immediately): the longest the auto-registered "
            "check_mcp_task_status tool will consider a task still legitimately running before "
            "reporting it as failed (IG-MCP-002). None (default) means no cap — matches today's "
            "behavior for servers that don't use Tasks, and leaves genuinely long-running tasks "
            "unbounded on servers that do. Ignored entirely for a server that never claims a task."
        ),
    )
    requires_approval: bool = Field(
        default=False, description="Require human-in-the-loop approval before this tool actually executes."
    )
    required_approvals: int = Field(
        default=1,
        ge=1,
        description=(
            "Number of distinct approvers (see server.auth.approvers) who must each approve via "
            "/resume before this tool executes — 1 (default) matches today's single-approval "
            "behavior. Ignored unless requires_approval is true."
        ),
    )
    required_approvers: list[str] | None = Field(
        default=None,
        description=(
            "Specific approver ids (keys of server.auth.approvers) that must each sign off, "
            "instead of any `required_approvals` approvers. Ignored unless requires_approval is "
            "true; when set, overrides required_approvals with len(required_approvers)."
        ),
    )
    available_when: str | None = Field(
        default=None,
        description=(
            "State condition (same restricted grammar as routers[].condition) gating whether "
            "this tool is even offered to the agent this turn — see LocalToolConfig.available_when "
            "for the full explanation. None (default) means always available, today's behavior."
        ),
    )
    untrusted_output: bool = Field(
        default=True,
        description=(
            "Mark this tool's return value as untrusted — see LocalToolConfig.untrusted_output "
            "for the full explanation. True by default for MCP tools, since they reach an "
            "external server outside the project's own trust boundary; set false only for an MCP "
            "server you fully control and trust."
        ),
    )


class OpenAPIToolConfig(StrictBaseModel):
    name: str = Field(description="Tool name exposed to the LLM for tool-calling.")
    type: Literal["openapi"] = Field(
        description="Discriminator — must be the literal string 'openapi'."
    )
    url: str = Field(
        description="URL of the OpenAPI/Swagger spec IntaGrin generates a tool wrapper from."
    )
    auth_env: str | None = Field(
        default=None,
        description="Environment variable holding a bearer token/API key sent with requests to this API.",
    )
    requires_approval: bool = Field(
        default=False, description="Require human-in-the-loop approval before this tool actually executes."
    )
    required_approvals: int = Field(
        default=1,
        ge=1,
        description=(
            "Number of distinct approvers (see server.auth.approvers) who must each approve via "
            "/resume before this tool executes — 1 (default) matches today's single-approval "
            "behavior. Ignored unless requires_approval is true."
        ),
    )
    required_approvers: list[str] | None = Field(
        default=None,
        description=(
            "Specific approver ids (keys of server.auth.approvers) that must each sign off, "
            "instead of any `required_approvals` approvers. Ignored unless requires_approval is "
            "true; when set, overrides required_approvals with len(required_approvers)."
        ),
    )
    available_when: str | None = Field(
        default=None,
        description=(
            "State condition (same restricted grammar as routers[].condition) gating whether "
            "this tool is even offered to the agent this turn — see LocalToolConfig.available_when "
            "for the full explanation. None (default) means always available, today's behavior."
        ),
    )
    untrusted_output: bool = Field(
        default=True,
        description=(
            "Mark this tool's return value as untrusted — see LocalToolConfig.untrusted_output "
            "for the full explanation. True by default for OpenAPI-derived tools, since they call "
            "an external API outside the project's own trust boundary; set false only for an API "
            "you fully control and trust."
        ),
    )


class SandboxToolConfig(StrictBaseModel):
    """Runs agent-generated code in an isolated subprocess (see runtime/sandbox.py for exactly
    what is and isn't isolated — process/resource/environment isolation, NOT a filesystem or
    network security boundary). The missing piece between the coding-agent template's coder/
    verifier loop (which writes and reviews code) and actually running code an LLM produced."""

    name: str = Field(description="Tool name exposed to the LLM for tool-calling.")
    type: Literal["sandbox"] = Field(
        description="Discriminator — must be the literal string 'sandbox'."
    )
    language: Literal["python", "bash"] = Field(
        default="python", description="Interpreter used to run the submitted code."
    )
    timeout_seconds: int = Field(
        default=10,
        ge=1,
        le=300,
        description=(
            "Wall-clock limit before the sandboxed process is killed. Also used as the POSIX "
            "CPU-time rlimit (see runtime/sandbox.py) — a no-op on platforms without the "
            "`resource` module (Windows)."
        ),
    )
    max_memory_mb: int | None = Field(
        default=256,
        ge=1,
        description=(
            "POSIX address-space rlimit (RLIMIT_AS) applied to the sandboxed process, in "
            "megabytes. None disables the memory limit entirely. A no-op on platforms without "
            "the `resource` module (Windows)."
        ),
    )
    requires_approval: bool = Field(
        default=False, description="Require human-in-the-loop approval before this tool actually executes."
    )
    required_approvals: int = Field(
        default=1,
        ge=1,
        description=(
            "Number of distinct approvers (see server.auth.approvers) who must each approve via "
            "/resume before this tool executes — 1 (default) matches today's single-approval "
            "behavior. Ignored unless requires_approval is true."
        ),
    )
    required_approvers: list[str] | None = Field(
        default=None,
        description=(
            "Specific approver ids (keys of server.auth.approvers) that must each sign off, "
            "instead of any `required_approvals` approvers. Ignored unless requires_approval is "
            "true; when set, overrides required_approvals with len(required_approvers)."
        ),
    )
    available_when: str | None = Field(
        default=None,
        description=(
            "State condition (same restricted grammar as routers[].condition) gating whether "
            "this tool is even offered to the agent this turn — see LocalToolConfig.available_when "
            "for the full explanation. None (default) means always available, today's behavior."
        ),
    )
    untrusted_output: bool = Field(
        default=True,
        description=(
            "Mark this tool's return value as untrusted — see LocalToolConfig.untrusted_output "
            "for the full explanation. True by default: sandboxed code's stdout/stderr can be "
            "influenced by whatever the executed code does (including content it read from "
            "elsewhere), so it's treated the same as an external tool result by default."
        ),
    )


ToolConfig = Union[LocalToolConfig, MCPToolConfig, OpenAPIToolConfig, SandboxToolConfig]


class ToolReferenceConfig(StrictBaseModel):
    """Reference a tool or MCP/OpenAPI provider declared at the root level."""

    name: str = Field(
        description="Name of a tool or MCP/OpenAPI provider declared at the ai.yaml root."
    )
    available_when: str | None = Field(
        default=None,
        description=(
            "State condition (same restricted grammar as routers[].condition) gating whether "
            "this tool is even offered to this agent this turn — see LocalToolConfig.available_when "
            "for the full explanation. This is the field that actually matters in practice: a "
            "per-agent `tools:` entry is usually a name-reference to a root-level tool, and "
            "availability is inherently agent-specific (the same globally-declared tool might be "
            "gated for one agent and unrestricted for another). None (default) means always "
            "available, today's behavior."
        ),
    )


class SkillConfig(StrictBaseModel):
    """A reusable, progressive-disclosure instruction bundle — the ai.yaml-native "Agent Skills"
    primitive (distinct from the `inta copilot` IDE skill-folder convention under
    templates/copilot/skills, which is an unrelated developer-tooling concept). Declared once at
    the ai.yaml root, referenced by name from any agent's `skills:` list. Advertised to the model
    cheaply — just this name + description, folded into the auto-registered `load_skill` tool's
    JSON schema — with the full body only loaded into context on demand via `load_skill(name)`,
    to avoid the "always-on prompt bulk" that contributes to context rot."""

    name: str = Field(description="Skill name, referenced from agents[].skills.")
    description: str = Field(
        description=(
            "One-line summary of when this skill should be used — shown to the model in "
            "load_skill's tool schema so it can decide whether to load the full skill without "
            "paying the cost of loading it."
        )
    )
    path: str = Field(
        description=(
            "Path (relative to the project directory) to this skill's content — either a single "
            "Markdown/text file, or a directory containing a main instructions file plus "
            "resource files readable via read_skill_resource. Resolved and existence-checked by "
            "`inta verify`."
        )
    )


class SkillReferenceConfig(StrictBaseModel):
    """Reference a skill declared at the root level from a specific agent's `skills:` list."""

    name: str = Field(description="Name of a skill declared at the ai.yaml root (skills:).")
    available_when: str | None = Field(
        default=None,
        description=(
            "State condition (same restricted grammar as routers[].condition) gating whether "
            "this skill is even offered to this agent this turn — see "
            "LocalToolConfig.available_when for the full explanation. None (default) means "
            "always available, today's behavior."
        ),
    )


class RouterConfig(StrictBaseModel):
    condition: str | None = Field(
        default=None,
        description='Python evaluation string against state (e.g., "balance < 0")',
    )
    custom_module: str | None = Field(
        default=None, description="Module to execute deterministic routing logic"
    )
    target: str = Field(description="Agent to route to when this router fires.")


class StateWriteAction(StrictBaseModel):
    key: str = Field(description="State key to write.")
    value: Any = Field(description="Value to write — the same JSON-ish value write_state accepts.")


class AgentSpawningConfig(StrictBaseModel):
    tool_pool: list[str] = Field(
        description=(
            "Closed allow-list of already-declared tool names a dynamically-created sub-agent's "
            "tools are drawn from — never a new tool implementation supplied at runtime. Must be "
            "a subset of this agent's own `tools:` (enforced at parse time); an agent can only "
            "hand off capabilities it already has, never escalate through creation. The literal "
            "string `\"*\"` is shorthand for every tool this agent itself has — expanded to that "
            "concrete list at parse time (AgentConfig's own subset validator), so it still can "
            "never grant more than the agent already holds."
        ),
    )

    @field_validator("tool_pool", mode="before")
    @classmethod
    def _accept_wildcard_shorthand(cls, v):
        """Lets ai.yaml write `tool_pool: "*"` instead of re-listing every tool name already
        declared under this agent's own `tools:`. Normalized here to `["*"]` (list[str] would
        otherwise reject a bare string); expanded to the concrete tool list by AgentConfig's
        `_validate_spawns_tool_pool_is_a_subset_of_own_tools` below, which is where the agent's
        own tool names are actually known."""
        return ["*"] if v == "*" else v
    model_pool: list[str] | None = Field(
        default=None,
        description=(
            "LiteLLM model identifiers a spawned agent may be assigned. None (default) means "
            "every spawned agent inherits the spawning agent's own resolved model — the spawning "
            "LLM never picks a model tier itself unless this is explicitly set."
        ),
    )
    max_creations_per_session: int = Field(
        default=3,
        ge=1,
        description="Session-wide cap on how many agents this agent may dynamically create.",
    )
    requires_approval_on_first_action: bool = Field(
        default=True,
        description=(
            "Gate a spawned agent's very first tool call behind human approval (reuses the "
            "existing requires_approval/multi-approver /resume mechanism), regardless of whether "
            "that specific tool is itself approval-gated. Safe-by-default: opt out, not opt in."
        ),
    )
    allow_recursive_spawning: bool = Field(
        default=False,
        description=(
            "Whether an agent spawned by this factory may itself spawn further agents (same "
            "tool_pool — no privilege growth), up to max_spawn_depth. Off by default."
        ),
    )
    max_spawn_depth: int = Field(
        default=1,
        ge=1,
        description="Max recursive spawn depth, only consulted when allow_recursive_spawning is true.",
    )
    result_schema: str | None = Field(
        default=None,
        description=(
            "Dotted path to a Pydantic model a spawned agent's return_to_creator call must "
            "conform to. When set, return_to_creator's tool schema is derived from this model "
            "instead of a generic free-text `summary` field — steering the model via constrained "
            "tool-call decoding rather than just asking nicely — and the arguments are "
            "re-validated server-side before being accepted, self-healed via the same "
            "corrector-model retry already used for malformed tool arguments elsewhere. None "
            "(default) keeps today's free-text `summary` behavior."
        ),
    )
    on_complete: list[StateWriteAction] = Field(
        default_factory=list,
        description=(
            "State writes applied automatically once a spawned agent genuinely completes — "
            "return_to_creator or a final text response, never on a pause awaiting approval, "
            "never on a forced max-turns abort. Runs through the exact same reducer/state_schema "
            "pipeline as write_state, so validation and merge strategies apply identically — this "
            "is not a second, less-validated write path. Lets a tools[].available_when gate (or "
            "any other state-driven condition) unlock declaratively, instead of requiring the "
            "spawned agent's own instruction text to call write_state on the framework's behalf."
        ),
    )


class AgentConfig(StrictBaseModel):
    description: str | None = Field(
        default=None,
        description="Short human-readable summary of this agent's purpose — shown in the Monitor dashboard graph.",
    )
    model_override: str | None = Field(
        default=None,
        description="Use a different LiteLLM model for this agent instead of the app-level model.primary.",
    )
    system_prompt_file: str | None = Field(
        default=None, description="Path to a Jinja2 template file rendered as this agent's system prompt."
    )
    system_prompt_langfuse: str | None = Field(
        default=None,
        description="Langfuse prompt name to fetch as this agent's system prompt instead of a local file.",
    )
    system_prompt_module: str | None = Field(
        default=None,
        description="Python module path exposing a function that returns this agent's system prompt dynamically.",
    )
    prompt_key: str | None = Field(
        default=None,
        description="Key used to look up this agent's prompt when multiple agents share a prompt source.",
    )
    required_scopes: list[str] = Field(
        default_factory=list,
        description="Scopes/permissions a caller must have (checked against session state) before this agent can run.",
    )
    response_schema: str | None = Field(
        default=None,
        description=(
            "Dotted path to a Pydantic model this agent's final response is validated against; "
            "a corrector model retries once on violation."
        ),
    )
    auto_route: bool = Field(
        default=False,
        description="Enable semantic swarm routing (LLM-Bypass Group Chat)",
    )
    lazy_load_tools: bool = Field(
        default=False,
        description="Enable semantic tool retrieval to reduce context window bloat",
    )
    tools: list[
        LocalToolConfig | MCPToolConfig | OpenAPIToolConfig | SandboxToolConfig | ToolReferenceConfig
    ] = Field(
        default_factory=list,
        description=(
            "Tools available to this agent — local Python functions, MCP servers, OpenAPI "
            "wrappers, or references to root-level tools."
        ),
    )
    handoffs: list[str] = Field(
        default_factory=list,
        description="Agent names this agent may conversationally transfer control to (compiled to a transfer_agent tool).",
    )
    delegations: list[str] = Field(
        default_factory=list,
        description=(
            "Agent names this agent may delegate a sub-task to. Compiles two tools: "
            "`delegate_task` (one sub-task, one isolated child engine — the delegating agent's "
            "own turn is never interrupted) and `delegate_to_many` (fan out N concurrent "
            "instances of the same sub-agent, one per instruction, for an item count only known "
            "at runtime — capped by circuit_breakers.max_parallel_fan_out)."
        ),
    )
    routers: list[RouterConfig] = Field(
        default_factory=list,
        description="Deterministic Python conditions checked before the LLM runs, to bypass it entirely when they fire.",
    )
    spawns: AgentSpawningConfig | None = Field(
        default=None,
        description=(
            "Enables dynamic runtime agent creation: this agent gets a spawn_agent tool that "
            "creates a narrowly-scoped sub-agent (new system prompt, a subset of spawns.tool_pool, "
            "an inherited model) mid-session, runs it to completion in an isolated child engine, "
            "and returns the result as an ordinary tool result — it does not transfer control, so "
            "the creator's own turn is never interrupted and concurrent spawn_agent calls in one "
            "turn have nothing shared to race on. None (default) means this agent cannot spawn "
            "anything — today's behavior."
        ),
    )
    skills: list[SkillReferenceConfig | str] = Field(
        default_factory=list,
        description=(
            "Agent Skills (see SkillConfig) this agent may load — references to skills declared "
            "in the ai.yaml root `skills:` list, either as a bare name string or a "
            "SkillReferenceConfig with an `available_when` gate. When non-empty, auto-registers "
            "a load_skill tool (enum-constrained to these names) and, for any referenced skill "
            "whose path is a directory, a read_skill_resource tool."
        ),
    )

    @model_validator(mode="after")
    def _validate_spawns_tool_pool_is_a_subset_of_own_tools(self) -> "AgentConfig":
        if self.spawns is None:
            return self
        own_tool_names = {t.name for t in self.tools}
        if self.spawns.tool_pool == ["*"]:
            self.spawns.tool_pool = sorted(own_tool_names)
            return self
        foreign = [name for name in self.spawns.tool_pool if name not in own_tool_names]
        if foreign:
            raise ValueError(
                "spawns.tool_pool names tool(s) this agent doesn't itself have access to — "
                f"an agent can only grant a spawned sub-agent capabilities it already holds: "
                f"{foreign}. This agent's own tools: {sorted(own_tool_names)}."
            )
        return self


class VoteConfig(StrictBaseModel):
    strategy: Literal["majority", "llm_judge"] = Field(
        default="majority",
        description=(
            "How a 'vote' workflow task's branch answers become one result: 'majority' compares "
            "branch outputs directly with no extra LLM call; 'llm_judge' asks the model to pick "
            "or synthesize the best answer from all branch outputs."
        ),
    )
    min_agreement: float = Field(
        default=0.5,
        description=(
            "Minimum fraction (0.0-1.0) of branches that must agree for 'majority' to declare a "
            "winner. Below this, no answer is guessed — the task reports 'no consensus' plus all "
            "branch outputs instead."
        ),
    )
    debate_rounds: int = Field(
        default=1,
        ge=1,
        le=5,
        description=(
            "Number of rounds each branch's own answer goes through before `strategy` "
            "aggregates a final result — the multi-agent-debate pattern (Du et al., 2023). "
            "1 (default) is today's original behavior: every branch answers once, independently, "
            "with no visibility into the others. A value above 1 runs that many additional "
            "rounds where each branch is shown every other branch's current answer and asked to "
            "reconsider — its own isolated child engine/conversation continues across rounds, so "
            "it can see its own prior reasoning too, not just restart fresh. Orthogonal to "
            "`strategy`: debate changes how each branch's answer is produced before voting, not "
            "how the final vote itself is tallied. Capped at 5 — cost scales as branches x "
            "rounds, all full agent turns, so an unbounded value has no natural ceiling."
        ),
    )


class WorkflowTask(StrictBaseModel):
    name: str = Field(description="Task name, used for logging/tracing this step of the workflow.")
    type: Literal["sequential", "parallel", "vote"] = Field(
        default="sequential",
        description=(
            "'sequential' runs sub-tasks/agent one after another; 'parallel' runs them "
            "concurrently and appends every branch's result; 'vote' runs them concurrently like "
            "'parallel' but aggregates branch answers into one consensus result (see `vote`)."
        ),
    )
    agent: str | None = Field(default=None, description="Agent that executes this task.")
    instruction: str | None = Field(default=None, description="Instruction given to `agent` for this task.")
    tasks: list["WorkflowTask"] | None = Field(
        default=None,
        description="Nested sub-tasks, when this task is a grouping node rather than a leaf agent call.",
    )
    vote: VoteConfig | None = Field(
        default=None,
        description=(
            "Aggregation settings for a 'vote' task (ignored otherwise); defaults to majority "
            "voting with 50% minimum agreement when omitted."
        ),
    )


class RAGConfig(StrictBaseModel):
    docs_dir: str = Field(
        default="docs", description="Directory of documents to index for retrieval, relative to the project root."
    )
    embedding_model: str = Field(
        default="text-embedding-3-small", description="Embedding model used to vectorize documents and queries."
    )
    top_k: int = Field(default=4, description="Number of chunks retrieved per query.")
    chunk_size: int = Field(default=500, description="Maximum characters per indexed chunk.")
    chunk_overlap: int = Field(default=50, description="Characters of overlap between consecutive chunks.")
    hyde: bool = Field(
        default=False,
        description="Enable Hypothetical Document Embeddings for advanced retrieval",
    )


class EpisodicMemoryConfig(StrictBaseModel):
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description=(
            "Embedding model used to vectorize episode content and recall_episodes queries for "
            "semantic search. Defaults to match rag.embedding_model's default so a project "
            "using both features doesn't need two different embedding providers configured."
        ),
    )
    scope: Literal["session", "tenant", "global"] = Field(
        default="session",
        description=(
            "Visibility scope for recorded episodes, independent of memory.shared_scope (a "
            "project may want the long_term_memory summary private per-session while episodic "
            "events are shared globally, or vice versa). 'session': only this session_id's own "
            "episodes. 'tenant': every session under the same authenticated caller/tenant prefix "
            "(same convention as memory.shared_scope: tenant). 'global': every session in the "
            "project, any tenant."
        ),
    )
    default_limit: int = Field(
        default=5,
        gt=0,
        description="Default number of episodes recall_episodes returns when the caller doesn't pass an explicit limit.",
    )
    retention_days: int | None = Field(
        default=None,
        description=(
            "Age (in days) after which episodes rows are deleted. episodes is append-only (one "
            "row per remember_episode call, never updated in place — see "
            "runtime/episodic_memory.py) with no automatic pruning by default, so it grows "
            "forever over a long production lifetime. None (default) keeps every episode "
            "indefinitely. Enforced opportunistically (a small random chance on each "
            "remember_episode write triggers a single indexed DELETE), same mechanism as "
            "memory.run_log_retention_days."
        ),
    )


class StateReducerConfig(StrictBaseModel):
    key: str = Field(description="Shared state key this reducer applies to.")
    strategy: Literal["overwrite", "append", "deep_merge"] = Field(
        default="overwrite",
        description=(
            "How concurrent/repeated writes to `key` are combined: 'overwrite' (last write "
            "wins), 'append' (to a list), or 'deep_merge' (for dicts)."
        ),
    )


class ImportConfig(StrictBaseModel):
    path: str = Field(
        description="Path to another ai.yaml-shaped file to merge in (agents, tools, workflows, reducers)."
    )
    namespace: str | None = Field(
        default=None,
        description="Prefix added to imported agent/workflow names, to avoid collisions with the importing project.",
    )


class CircuitBreakersConfig(StrictBaseModel):
    max_handoffs_per_session: int | None = Field(
        default=25,
        description="Max allowed handoffs per session to prevent infinite loops",
    )
    max_tool_failures_in_a_row: int | None = Field(
        default=3, description="Max sequential tool failures before halting"
    )
    max_usd_cost_per_session: float | None = Field(
        default=None, description="Max USD cost per session before halting"
    )
    max_delegation_depth: int = Field(
        default=3,
        description="Max nested sub-agent delegation depth before rejecting further delegation",
    )
    max_delegation_turns: int = Field(
        default=15,
        description="Max turns a delegated sub-agent may take before it is forcefully aborted",
    )
    max_parallel_fan_out: int = Field(
        default=10,
        ge=1,
        description=(
            "Max instructions a single delegate_to_many call may fan out to concurrently — "
            "each one runs a full isolated sub-agent to completion, so an LLM-chosen item count "
            "with no ceiling could spawn an unbounded number of child engines/LLM calls at once. "
            "A call over this limit is rejected outright (asking the caller to split the work "
            "into smaller batches) rather than silently truncated."
        ),
    )
    max_parallel_tool_calls_per_turn: int = Field(
        default=10,
        ge=1,
        description=(
            "Max ordinary (non-transfer) tool calls a single LLM completion may request that "
            "actually get executed concurrently in one turn. A model can request an arbitrary "
            "number of tool calls in one completion with no natural ceiling; calls beyond this "
            "limit are not executed — each gets a synthetic tool-result telling the model to "
            "split the work across turns — rather than silently gathered anyway, mirroring how "
            "a duplicate control-transfer call in the same turn is rejected rather than run."
        ),
    )
    max_corrector_tokens: int = Field(
        default=1000,
        ge=1,
        description=(
            "max_tokens applied to every self-healing corrector-model call — both malformed "
            "tool-call argument repair and response_schema/result_schema repair. These calls "
            "previously had no max_tokens set at all, making their cost genuinely unbounded; "
            "capping the output side bounds each individual corrector call's worst-case cost."
        ),
    )
    max_compression_batch_messages: int = Field(
        default=50,
        ge=1,
        description=(
            "Max messages folded into long_term_memory per memory-compression LLM call. When "
            "more messages than this are evicted from the context window at once, compression "
            "runs in successive batches of this size, each folded into the summary in turn, "
            "instead of dumping an unbounded amount of evicted conversation into a single "
            "corrector-model prompt (the summary's own output side was already capped at 500 "
            "tokens; this bounds the input side)."
        ),
    )
    max_tool_result_chars: int | None = Field(
        default=20000,
        ge=1,
        description=(
            "Max characters kept from a single tool call's result before it's appended to "
            "conversation history — the rest is dropped with a truncation notice appended in "
            "its place. Character-based rather than a real per-model tokenizer, to stay cheap "
            "on this hot path (every tool call) — roughly 4 characters per token for English "
            "text. Protects against one oversized result (a scraped page, a large API/MCP "
            "response) silently consuming a large fraction of the context window in a single "
            "turn, the same way max_corrector_tokens bounds a corrector call's output side. "
            "None disables the cap entirely."
        ),
    )


class RootRouterConfig(StrictBaseModel):
    description: str | None = Field(
        default=None, description="What this router decides"
    )
    module: str = Field(description="Python module containing a route(state) function")
    possible_targets: list[str] = Field(
        default_factory=list,
        description="Strict list of allowed target agents to maintain determinism and UI visualization",
    )


class AppConfig(StrictBaseModel):
    version: str = Field(description="ai.yaml schema/format version.")
    name: str = Field(description="Project name.")
    description: str | None = Field(
        default=None, description="Short human-readable summary of what this app does."
    )
    state_schema: str | None = Field(
        default=None,
        description="Global JSON schema module path for Typed Shared State",
    )
    max_session_budget_usd: float | None = Field(
        default=None, description="Global hard cost ceiling per session"
    )
    imports: list[ImportConfig] = Field(
        default_factory=list,
        description="Other ai.yaml-shaped files to merge into this one.",
    )
    circuit_breakers: CircuitBreakersConfig = Field(
        default_factory=CircuitBreakersConfig,
        description="Thresholds that halt a runaway session — handoff loops, tool failure streaks, USD cost, delegation depth.",
    )
    model: ModelConfig = Field(
        description="Default LLM configuration, used by any agent that doesn't set model_override."
    )
    memory: MemoryConfig = Field(
        description="Conversation history backend used by all agents in this app."
    )
    rag: RAGConfig | None = Field(
        default=None,
        description="Vector retrieval configuration — when set, auto-registers a search_knowledge_base tool.",
    )
    episodic_memory: EpisodicMemoryConfig | None = Field(
        default=None,
        description=(
            "Discrete, structured, individually queryable event records (e.g. \"user prefers "
            "window seats\", \"booking BK-4471 failed: card declined\") — distinct from the "
            "single blended long_term_memory prose summary and from rag's document/knowledge-"
            "base retrieval. When set, auto-registers remember_episode/recall_episodes tools. "
            "Stored in a sqlite/postgres episodes table (memory.type must be 'sqlite' or "
            "'postgres' — same scope as shared_memory/run_logs); a no-op elsewhere."
        ),
    )
    tools: list[ToolConfig] = Field(
        default_factory=list,
        description="Tools declared at the root level, available to be referenced by name from any agent.",
    )  # Global tools
    skills: list[SkillConfig] = Field(
        default_factory=list,
        description=(
            "Agent Skills declared at the root level, available to be referenced by name from "
            "any agent's `skills:` list — see SkillConfig."
        ),
    )
    agents: dict[str, AgentConfig] = Field(
        default_factory=dict, description="The agents that make up this app, keyed by name."
    )
    routers: dict[str, RootRouterConfig] = Field(
        default_factory=dict,
        description="Root-level deterministic routers, keyed by name — evaluated before any agent runs.",
    )
    workflows: dict[str, list[WorkflowTask]] = Field(
        default_factory=dict,
        description="Named multi-step task sequences, run via `inta run <workflow>` rather than conversationally.",
    )
    reducers: list[StateReducerConfig] = Field(
        default_factory=list,
        description="How concurrent writes to shared state keys are combined.",
    )
    condition_functions: list[ConditionFunctionConfig] = Field(
        default_factory=list,
        description=(
            "Named, pure Python predicate functions that routers[].condition and "
            "tools[].available_when expressions may call by name (e.g. "
            "`is_eligible(customer_tier)`), for branching logic the restricted condition "
            "grammar's bare comparisons/and/or/not can't express on their own — without forcing "
            "an LLM handoff just to make a deterministic decision."
        ),
    )
    telemetry: list[Literal["langfuse", "otel"]] = Field(
        default_factory=list, description="Tracing backends to export spans to."
    )
    server: ServerConfig = Field(
        default_factory=ServerConfig,
        description="`inta serve`/`inta monitor` configuration — auth, webhooks.",
    )
    default_agent: str = Field(description="Agent that receives the first message in a new session.")

    @model_validator(mode="after")
    def _validate_agent_skill_references_exist_at_root(self) -> "AppConfig":
        known_skill_names = {s.name for s in self.skills}
        for agent_name, agent_cfg in self.agents.items():
            for skill_ref in agent_cfg.skills:
                name = skill_ref if isinstance(skill_ref, str) else skill_ref.name
                if name not in known_skill_names:
                    raise ValueError(
                        f"agents.{agent_name}.skills references skill {name!r}, which isn't "
                        f"declared in the root-level skills: list. Declared skills: "
                        f"{sorted(known_skill_names)}."
                    )
        return self


def validate_config_dict(data: dict) -> tuple[AppConfig | None, list[str]]:
    """Validates a raw config dict (from an LLM compile, a Studio graph edit, an architect-chat
    file proposal — anywhere something other than a human hand-writes ai.yaml) against the real
    AppConfig schema and the router-condition grammar (validate_condition_syntax,
    runtime/router.py) — the two ways a config can look like valid YAML/JSON but be silently
    broken (a missing required field, or a router condition like `state.get(...)` that will just
    never fire). Returns (parsed_config_or_None, list_of_human_readable_errors); every caller that
    might write `ai.yaml` on behalf of something other than a human should call this first and
    refuse to write on a non-empty error list, rather than writing unchecked and finding out later.
    """
    from pydantic import ValidationError

    errors: list[str] = []
    config: AppConfig | None = None
    try:
        config = AppConfig(**data)
    except ValidationError as e:
        errors.append(str(e))
    except Exception as e:
        errors.append(f"Could not parse as a valid AppConfig: {e}")

    if config is not None:
        from ..runtime.router import validate_condition_syntax

        known_functions = {cf.name for cf in config.condition_functions}

        for agent_name, agent_cfg in config.agents.items():
            for router in getattr(agent_cfg, "routers", []) or []:
                if router.condition:
                    reason = validate_condition_syntax(router.condition, known_functions)
                    if reason:
                        errors.append(
                            f"agents.{agent_name}.routers condition {router.condition!r}: {reason}"
                        )
            for tool in getattr(agent_cfg, "tools", []) or []:
                available_when = getattr(tool, "available_when", None)
                if available_when:
                    reason = validate_condition_syntax(available_when, known_functions)
                    if reason:
                        errors.append(
                            f"agents.{agent_name}.tools[{tool.name!r}].available_when "
                            f"{available_when!r}: {reason}"
                        )
            for skill_ref in getattr(agent_cfg, "skills", []) or []:
                available_when = getattr(skill_ref, "available_when", None)
                if available_when:
                    reason = validate_condition_syntax(available_when, known_functions)
                    if reason:
                        errors.append(
                            f"agents.{agent_name}.skills[{skill_ref.name!r}].available_when "
                            f"{available_when!r}: {reason}"
                        )
            spawns_cfg = getattr(agent_cfg, "spawns", None)
            for action in getattr(spawns_cfg, "on_complete", []) or []:
                if action.key.startswith("_"):
                    errors.append(
                        f"agents.{agent_name}.spawns.on_complete writes to {action.key!r} — "
                        "keys starting with '_' are reserved for internal engine state and are "
                        "silently rejected by write_state at runtime."
                    )

    return config, errors
