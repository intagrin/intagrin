"""A lighter-weight, heuristic cousin of test_error_docs_freshness.py /
test_config_reference_freshness.py: those are exact generator-diff checks, possible because
errors.py/schema.py are each a single source of truth. CLAUDE.md and templates/copilot are
hand-written narrative docs with no single generator — this can't prove they're *accurate*, only
that a shipped feature is mentioned *somewhere* across the doc set a coding agent actually reads.
Best-effort, not airtight: extend FEATURE_MARKERS whenever a new declarative `ai.yaml` feature
lands, as part of that change's own doc-update step.

Deliberately does not check a committed `.cursor/` bundle — that directory is the *output* of
`inta copilot --agent cursor` (for a consumer project's own ai.yaml/tools/prompts, e.g.
examples/social-media-manager/.agents/), not something this framework's own repo needs committed;
regenerate it on demand with `inta copilot` rather than keeping a copy in version control here.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DOC_LOCATIONS = [
    REPO_ROOT / "CLAUDE.md",
    *sorted((REPO_ROOT / "src" / "intagrin" / "templates" / "copilot").rglob("*.md")),
]

# Feature marker -> a one-line hint naming which doc-update step introduced it, so a failure
# message points at what to go add rather than just naming a missing string.
FEATURE_MARKERS = {
    "AwaitingHumanInput": "dynamic (runtime-triggered) human-in-the-loop approval",
    "run_logs": "the Logs page / per-API-call audit trail (runtime/run_logger.py)",
    "required_approvers": "multi-approver HITL chains (N-of-M sign-off via /resume)",
    "rate_limit": "per-caller rate limiting / usage quotas (server.rate_limit)",
    "model.variants": "A/B / canary model routing",
    "shared_scope": "cross-session / org-level shared memory (memory.shared_scope)",
    "spawn_agent": "dynamic runtime agent creation (agents.<name>.spawns)",
    "return_to_creator": "the star-topology control-flow constraint on spawned agents",
    "result_schema": "structured, validated spawn_agent/return_to_creator results (spawns.result_schema)",
    "available_when": "state-gated tool availability (tools[].available_when)",
    "on_complete": "declarative state writes on spawn completion (spawns.on_complete)",
    "remember_episode": "episodic memory — discrete queryable event records (runtime/episodic_memory.py, episodic_memory: config)",
    "_tool_call_scratch": "write-ahead tool-call durability / crash-safe exactly-once tool execution (engine.py's _recover_dangling_tool_calls)",
    "condition_functions": "named predicate functions callable from routers[].condition / tools[].available_when",
    "delegate_to_many": "runtime-determined concurrent fan-out delegation (AgentConfig.delegations)",
    "max_parallel_fan_out": "the circuit breaker capping delegate_to_many's fan-out width",
    "_router_trace": "persisted router/available_when evaluation history surfaced by inta replay (engine.py's _record_router_trace)",
    "_generate_and_validate_wizard_config": "schema-validated, self-healing inta new --withagent generation",
    "_untrusted_content_ingested": "the lethal-trifecta guardrail — tools[].untrusted_output tracking (config/schema.py, engine.py's execute_tool)",
    "runtime/sandbox.py": "isolated subprocess execution for agent-generated code (tools[].type: \"sandbox\", SandboxToolConfig)",
}


def _combined_doc_text() -> str:
    chunks = []
    for path in DOC_LOCATIONS:
        if path.exists() and path.is_file():
            chunks.append(path.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(chunks)


def test_every_feature_marker_is_mentioned_somewhere_in_the_doc_set():
    combined = _combined_doc_text()
    missing = [
        f"{marker!r} ({hint}) is not mentioned in CLAUDE.md or templates/copilot"
        for marker, hint in FEATURE_MARKERS.items()
        if marker not in combined
    ]
    assert not missing, "Stale docs — features shipped with no mention:\n" + "\n".join(missing)


def test_doc_locations_actually_resolved_at_least_one_file_per_directory_glob():
    """Guards the test above against silently checking nothing — an empty glob (e.g. a renamed
    directory) would make every marker check vacuously pass."""
    assert (REPO_ROOT / "CLAUDE.md").exists()
    assert any((REPO_ROOT / "src" / "intagrin" / "templates" / "copilot").rglob("*.md"))
