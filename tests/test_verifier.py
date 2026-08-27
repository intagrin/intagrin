import tempfile
from pathlib import Path

from intagrin.compiler.parser import parse_project
from intagrin.compiler.verifier import GraphVerifier
from intagrin.compiler.verifier import console as verifier_console


def test_verifier_acyclic_graph():
    with tempfile.TemporaryDirectory() as tmpdir:
        p_dir = Path(tmpdir)
        ai_yaml = """version: "1.0"
name: "acyclic-app"
default_agent: "agent_a"
model:
  primary: "gemini/gemini-2.5-flash"
memory:
  type: "sqlite"
agents:
  agent_a:
    handoffs: ["agent_b"]
  agent_b:
    handoffs: ["agent_c"]
  agent_c:
    handoffs: []
"""
        (p_dir / "ai.yaml").write_text(ai_yaml)
        verifier = GraphVerifier(project_dir=p_dir)
        # Should execute cleanly without errors
        verifier.verify()

def test_verifier_cyclic_graph_handled():
    with tempfile.TemporaryDirectory() as tmpdir:
        p_dir = Path(tmpdir)
        ai_yaml = """version: "1.0"
name: "cyclic-app"
default_agent: "agent_a"
model:
  primary: "openai/gpt-4o"
memory:
  type: "sqlite"
agents:
  agent_a:
    handoffs: ["agent_b"]
  agent_b:
    handoffs: ["agent_a"]
"""
        (p_dir / "ai.yaml").write_text(ai_yaml)
        verifier = GraphVerifier(project_dir=p_dir)
        verifier.verify()


FULL_COVERAGE_YAML = """version: "1.0"
name: "full-coverage-app"
default_agent: "agent_a"
model:
  primary: "gemini/gemini-2.5-flash"
memory:
  type: "sqlite"
circuit_breakers:
  max_delegation_depth: 2
  max_delegation_turns: 5
agents:
  agent_a:
    handoffs: ["agent_b"]
    delegations: ["agent_d"]
  agent_b:
    routers:
      - condition: "x > 0"
        target: "agent_c"
  agent_c:
    auto_route: true
  agent_d: {}
  agent_e:
    handoffs: []
routers:
  agent_e:
    module: "nonexistent.module"
    possible_targets: ["agent_b"]
"""


def test_verifier_adjacency_covers_handoffs_conditional_and_root_routers_but_not_auto_route():
    """auto_route is deliberately excluded from the cycle-checked adjacency graph — it's an LLM
    decision at runtime, not a static edge — and must not be silently dropped from the analysis
    either; it's asserted separately below via the rendered report."""
    with tempfile.TemporaryDirectory() as tmpdir:
        p_dir = Path(tmpdir)
        (p_dir / "ai.yaml").write_text(FULL_COVERAGE_YAML)
        graph = parse_project(p_dir)
        verifier = GraphVerifier(project_dir=p_dir)

        adj, edge_source = verifier._build_adjacency(graph.config)

        assert "agent_b" in adj["agent_a"]
        assert edge_source[("agent_a", "agent_b")] == "handoff"

        assert "agent_c" in adj["agent_b"]
        assert edge_source[("agent_b", "agent_c")] == "conditional router"

        assert "agent_b" in adj["agent_e"]
        assert edge_source[("agent_e", "agent_b")] == "root router"

        # Delegations never appear as cycle-checked edges — they're a bounded call tree, not a
        # graph-acyclicity concern (delegation always returns control to the caller).
        assert "agent_d" not in adj["agent_a"]

        # auto_route agents get no blanket edges injected into the adjacency graph at all.
        assert adj.get("agent_c", []) == []


def test_verifier_report_surfaces_auto_route_and_delegation_bounds():
    with tempfile.TemporaryDirectory() as tmpdir:
        p_dir = Path(tmpdir)
        (p_dir / "ai.yaml").write_text(FULL_COVERAGE_YAML)
        verifier = GraphVerifier(project_dir=p_dir)

        with verifier_console.capture() as capture:
            verifier.verify()
        output = capture.get()

        assert "auto_route" in output
        assert "agent_c" in output
        assert "Delegation subtree" in output
        assert "max_delegation_depth" in output


def test_verifier_reports_self_healing_compression_and_parallel_tool_calls_as_bounded():
    """Self-healing corrector calls, memory-compression input, and per-turn parallel tool calls
    used to be reported as explicitly uncapped in the worst-case cost table. All three are now
    bounded by circuit_breakers defaults and must render as such (green 'yes'), not red 'no'."""
    with tempfile.TemporaryDirectory() as tmpdir:
        p_dir = Path(tmpdir)
        (p_dir / "ai.yaml").write_text(FULL_COVERAGE_YAML)
        verifier = GraphVerifier(project_dir=p_dir)

        with verifier_console.capture() as capture:
            verifier.verify()
        output = capture.get()

        # Default circuit_breakers values (FULL_COVERAGE_YAML overrides neither): 2 retries x
        # max_corrector_tokens=1000, max_compression_batch_messages=50, max_parallel_tool_calls_
        # per_turn=10 — all three must render as their real bound, not "not capped".
        assert "not capped" not in output
        assert "…" not in output  # a truncated (not just wrapped) identifier would show as this
        assert "2,000" in output
        assert "50 msgs/batch" in output
        assert "10 concurrent" in output


def test_verifier_flags_a_router_condition_using_unsupported_syntax():
    """A condition written as `state.get(...)` never raises anywhere a user would notice — it's
    caught inside safe_eval's caller, logged, and the router just silently never fires. inta
    verify must catch this statically instead of leaving it to be discovered at runtime."""
    ai_yaml = """version: "1.0"
name: "bad-condition-app"
default_agent: "triage"
model:
  primary: "gemini/gemini-2.5-flash"
memory:
  type: "sqlite"
agents:
  triage:
    routers:
      - condition: "state.get('user_status', '') == 'banned'"
        target: "billing"
  billing: {}
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        p_dir = Path(tmpdir)
        (p_dir / "ai.yaml").write_text(ai_yaml)
        verifier = GraphVerifier(project_dir=p_dir)

        with verifier_console.capture() as capture:
            verifier.verify()
        output = capture.get()

        assert "Router condition syntax errors" in output
        assert "state.get" in output


def test_verifier_reports_valid_conditions_as_syntactically_valid():
    with tempfile.TemporaryDirectory() as tmpdir:
        p_dir = Path(tmpdir)
        (p_dir / "ai.yaml").write_text(FULL_COVERAGE_YAML)
        verifier = GraphVerifier(project_dir=p_dir)

        with verifier_console.capture() as capture:
            verifier.verify()
        output = capture.get()

        assert "Router condition syntax errors" not in output
        assert "syntactically valid" in output


SPAWNS_YAML = """version: "1.0"
name: "spawning-app"
default_agent: "orchestrator"
model:
  primary: "gemini/gemini-2.5-flash"
memory:
  type: "sqlite"
agents:
  orchestrator:
    tools:
      - name: "search"
        module: "tools.custom"
    spawns:
      tool_pool: ["search"]
      max_creations_per_session: 2
"""


def test_verifier_reports_the_unverifiable_dynamic_agent_creation_surface():
    with tempfile.TemporaryDirectory() as tmpdir:
        p_dir = Path(tmpdir)
        (p_dir / "ai.yaml").write_text(SPAWNS_YAML)
        verifier = GraphVerifier(project_dir=p_dir)

        with verifier_console.capture() as capture:
            verifier.verify()
        output = capture.get()

        assert "Non-deterministic agent creation" in output
        assert "orchestrator" in output
        assert "cycle/cost analysis" in output


def test_verifier_says_nothing_about_dynamic_agents_when_none_are_configured():
    with tempfile.TemporaryDirectory() as tmpdir:
        p_dir = Path(tmpdir)
        (p_dir / "ai.yaml").write_text(FULL_COVERAGE_YAML)
        verifier = GraphVerifier(project_dir=p_dir)

        with verifier_console.capture() as capture:
            verifier.verify()
        output = capture.get()

        assert "Non-deterministic agent creation" not in output


def test_verifier_flags_an_available_when_condition_using_unsupported_syntax():
    """The same failure mode as a bad router condition, but worse: available_when fails closed,
    so a syntax error permanently hides the tool rather than just never firing a router. inta
    verify must catch this statically."""
    ai_yaml = """version: "1.0"
name: "bad-available-when-app"
default_agent: "planner"
model:
  primary: "gemini/gemini-2.5-flash"
memory:
  type: "sqlite"
agents:
  planner:
    tools:
      - name: "create_itinerary"
        module: "tools.custom"
      - name: "book_flight"
        module: "tools.custom"
        available_when: "state.get('research_done')"
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        p_dir = Path(tmpdir)
        (p_dir / "ai.yaml").write_text(ai_yaml)
        verifier = GraphVerifier(project_dir=p_dir)

        with verifier_console.capture() as capture:
            verifier.verify()
        output = capture.get()

        assert "available_when condition syntax errors" in output
        assert "state.get" in output


def test_verifier_reports_valid_available_when_conditions_as_syntactically_valid():
    ai_yaml = """version: "1.0"
name: "good-available-when-app"
default_agent: "planner"
model:
  primary: "gemini/gemini-2.5-flash"
memory:
  type: "sqlite"
agents:
  planner:
    tools:
      - name: "create_itinerary"
        module: "tools.custom"
      - name: "book_flight"
        module: "tools.custom"
        available_when: "research_done"
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        p_dir = Path(tmpdir)
        (p_dir / "ai.yaml").write_text(ai_yaml)
        verifier = GraphVerifier(project_dir=p_dir)

        with verifier_console.capture() as capture:
            verifier.verify()
        output = capture.get()

        assert "available_when condition syntax errors" not in output
        assert "available_when conditions are syntactically valid" in output


def test_verifier_flags_on_complete_writing_to_a_reserved_key():
    """apply_state_write (the same pipeline write_state itself goes through) silently rejects any
    key starting with `_` — reserved for internal engine bookkeeping. inta verify must catch this
    statically instead of leaving a developer to discover it via a logged runtime error."""
    ai_yaml = """version: "1.0"
name: "bad-on-complete-app"
default_agent: "planner"
model:
  primary: "gemini/gemini-2.5-flash"
memory:
  type: "sqlite"
agents:
  planner:
    tools:
      - name: "search"
        module: "tools.custom"
    spawns:
      tool_pool: ["search"]
      on_complete:
        - key: "_pending_approval"
          value: true
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        p_dir = Path(tmpdir)
        (p_dir / "ai.yaml").write_text(ai_yaml)
        verifier = GraphVerifier(project_dir=p_dir)

        with verifier_console.capture() as capture:
            verifier.verify()
        output = capture.get()

        assert "on_complete writes to a reserved key" in output
        assert "_pending_approval" in output


def test_verifier_reports_on_complete_keys_as_writable_when_valid():
    ai_yaml = """version: "1.0"
name: "good-on-complete-app"
default_agent: "planner"
model:
  primary: "gemini/gemini-2.5-flash"
memory:
  type: "sqlite"
agents:
  planner:
    tools:
      - name: "search"
        module: "tools.custom"
    spawns:
      tool_pool: ["search"]
      on_complete:
        - key: "research_done"
          value: true
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        p_dir = Path(tmpdir)
        (p_dir / "ai.yaml").write_text(ai_yaml)
        verifier = GraphVerifier(project_dir=p_dir)

        with verifier_console.capture() as capture:
            verifier.verify()
        output = capture.get()

        assert "on_complete writes to a reserved key" not in output
        assert "spawns.on_complete keys are writable" in output


def test_verifier_nudges_toward_a_state_schema_when_none_is_set():
    ai_yaml = """version: "1.0"
name: "no-schema-app"
default_agent: "triage"
model:
  primary: "gemini/gemini-2.5-flash"
memory:
  type: "sqlite"
agents:
  triage: {}
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        p_dir = Path(tmpdir)
        (p_dir / "ai.yaml").write_text(ai_yaml)
        verifier = GraphVerifier(project_dir=p_dir)

        with verifier_console.capture() as capture:
            verifier.verify()
        output = capture.get()

        assert "No state_schema configured" in output


def test_verifier_does_not_nudge_when_a_state_schema_is_set():
    ai_yaml = """version: "1.0"
name: "with-schema-app"
default_agent: "triage"
state_schema: "schemas.AppState"
model:
  primary: "gemini/gemini-2.5-flash"
memory:
  type: "sqlite"
agents:
  triage: {}
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        p_dir = Path(tmpdir)
        (p_dir / "ai.yaml").write_text(ai_yaml)
        verifier = GraphVerifier(project_dir=p_dir)

        with verifier_console.capture() as capture:
            verifier.verify()
        output = capture.get()

        assert "No state_schema configured" not in output


def test_verifier_flags_a_condition_calling_an_undeclared_condition_function():
    ai_yaml = """version: "1.0"
name: "bad-condfn-app"
default_agent: "triage"
model:
  primary: "gemini/gemini-2.5-flash"
memory:
  type: "sqlite"
agents:
  triage:
    routers:
      - condition: "is_vip(tier)"
        target: "billing"
  billing: {}
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        p_dir = Path(tmpdir)
        (p_dir / "ai.yaml").write_text(ai_yaml)
        verifier = GraphVerifier(project_dir=p_dir)

        with verifier_console.capture() as capture:
            verifier.verify()
        output = capture.get()

        assert "Router condition syntax errors" in output
        assert "is_vip" in output


def test_verifier_accepts_a_condition_calling_a_declared_condition_function():
    ai_yaml = """version: "1.0"
name: "good-condfn-app"
default_agent: "triage"
model:
  primary: "gemini/gemini-2.5-flash"
memory:
  type: "sqlite"
condition_functions:
  - name: "is_vip"
    module: "tools.condition_functions"
agents:
  triage:
    routers:
      - condition: "is_vip(tier)"
        target: "billing"
  billing: {}
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        p_dir = Path(tmpdir)
        (p_dir / "ai.yaml").write_text(ai_yaml)
        verifier = GraphVerifier(project_dir=p_dir)

        with verifier_console.capture() as capture:
            verifier.verify()
        output = capture.get()

        assert "Router condition syntax errors" not in output
        assert "All conditional router conditions are syntactically valid" in output


def test_verifier_flags_required_approvals_ignored_when_required_approvers_also_set():
    ai_yaml = """version: "1.0"
name: "approval-conflict-app"
default_agent: "billing"
model:
  primary: "gemini/gemini-2.5-flash"
memory:
  type: "sqlite"
agents:
  billing:
    tools:
      - name: "issue_refund"
        module: "tools.custom"
        requires_approval: true
        required_approvals: 3
        required_approvers: ["alice", "bob"]
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        p_dir = Path(tmpdir)
        (p_dir / "ai.yaml").write_text(ai_yaml)
        verifier = GraphVerifier(project_dir=p_dir)

        with verifier_console.capture() as capture:
            verifier.verify()
        output = capture.get()

        assert "required_approvals is dead configuration" in output
        assert "issue_refund" in output
        assert "required_approvals=3" in output
        assert "effective count: 2" in output


def test_verifier_does_not_flag_required_approvals_left_at_default():
    """A tool that only sets required_approvers (leaving required_approvals at its default of 1)
    hasn't configured anything conflicting — nothing to warn about."""
    ai_yaml = """version: "1.0"
name: "approval-no-conflict-app"
default_agent: "billing"
model:
  primary: "gemini/gemini-2.5-flash"
memory:
  type: "sqlite"
agents:
  billing:
    tools:
      - name: "issue_refund"
        module: "tools.custom"
        requires_approval: true
        required_approvers: ["alice", "bob"]
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        p_dir = Path(tmpdir)
        (p_dir / "ai.yaml").write_text(ai_yaml)
        verifier = GraphVerifier(project_dir=p_dir)

        with verifier_console.capture() as capture:
            verifier.verify()
        output = capture.get()

        assert "required_approvals is dead configuration" not in output


def test_verifier_does_not_flag_required_approvals_matching_the_approver_count():
    """required_approvals happens to equal len(required_approvers) — functionally harmless even
    though required_approvers is still what actually wins, so nothing surprising to flag."""
    ai_yaml = """version: "1.0"
name: "approval-coincidental-match-app"
default_agent: "billing"
model:
  primary: "gemini/gemini-2.5-flash"
memory:
  type: "sqlite"
agents:
  billing:
    tools:
      - name: "issue_refund"
        module: "tools.custom"
        requires_approval: true
        required_approvals: 2
        required_approvers: ["alice", "bob"]
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        p_dir = Path(tmpdir)
        (p_dir / "ai.yaml").write_text(ai_yaml)
        verifier = GraphVerifier(project_dir=p_dir)

        with verifier_console.capture() as capture:
            verifier.verify()
        output = capture.get()

        assert "required_approvals is dead configuration" not in output


def test_verifier_flags_a_sensitive_tool_not_gated_against_untrusted_content():
    """An MCP tool is untrusted_output=true by default. An agent that also has a
    requires_approval tool with no available_when gate on _untrusted_content_ingested can't tell
    "clean session" from "this session already ingested untrusted content" — must be flagged."""
    ai_yaml = """version: "1.0"
name: "trifecta-app"
default_agent: "assistant"
model:
  primary: "gemini/gemini-2.5-flash"
memory:
  type: "sqlite"
agents:
  assistant:
    tools:
      - name: "fetch_docs"
        type: "mcp"
        command: "npx"
        args: ["-y", "some-mcp-server"]
      - name: "send_email"
        module: "tools.custom"
        requires_approval: true
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        p_dir = Path(tmpdir)
        (p_dir / "ai.yaml").write_text(ai_yaml)
        verifier = GraphVerifier(project_dir=p_dir)

        with verifier_console.capture() as capture:
            verifier.verify()
        output = capture.get()

        assert "not gated against ingested untrusted content" in output
        assert "assistant.send_email" in output
        assert "fetch_docs" in output


def test_verifier_does_not_flag_when_the_sensitive_tool_is_gated():
    """Same shape as above, but send_email's available_when already references
    _untrusted_content_ingested — nothing left to warn about."""
    ai_yaml = """version: "1.0"
name: "trifecta-gated-app"
default_agent: "assistant"
model:
  primary: "gemini/gemini-2.5-flash"
memory:
  type: "sqlite"
agents:
  assistant:
    tools:
      - name: "fetch_docs"
        type: "mcp"
        command: "npx"
        args: ["-y", "some-mcp-server"]
      - name: "send_email"
        module: "tools.custom"
        requires_approval: true
        available_when: "not _untrusted_content_ingested"
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        p_dir = Path(tmpdir)
        (p_dir / "ai.yaml").write_text(ai_yaml)
        verifier = GraphVerifier(project_dir=p_dir)

        with verifier_console.capture() as capture:
            verifier.verify()
        output = capture.get()

        assert "not gated against ingested untrusted content" not in output


def test_verifier_resolves_untrusted_output_and_gating_through_tool_references():
    """The common real-world shape: tools declared once at the root, referenced by name from
    agents. The per-agent available_when override (on the ToolReferenceConfig) is what actually
    matters here — must be resolved against the root tool's untrusted_output/requires_approval."""
    ai_yaml = """version: "1.0"
name: "trifecta-ref-app"
default_agent: "assistant"
model:
  primary: "gemini/gemini-2.5-flash"
memory:
  type: "sqlite"
tools:
  - name: "fetch_docs"
    type: "mcp"
    command: "npx"
    args: ["-y", "some-mcp-server"]
  - name: "send_email"
    module: "tools.custom"
    requires_approval: true
agents:
  no_gate:
    tools:
      - name: "fetch_docs"
      - name: "send_email"
  has_gate:
    tools:
      - name: "fetch_docs"
      - name: "send_email"
        available_when: "not _untrusted_content_ingested"
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        p_dir = Path(tmpdir)
        (p_dir / "ai.yaml").write_text(ai_yaml)
        verifier = GraphVerifier(project_dir=p_dir)

        with verifier_console.capture() as capture:
            verifier.verify()
        output = capture.get()

        assert "no_gate.send_email" in output
        assert "has_gate.send_email" not in output
