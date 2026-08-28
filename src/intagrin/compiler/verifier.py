from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..compiler.parser import parse_project
from ..config.schema import SandboxToolConfig, ToolReferenceConfig
from ..runtime.router import validate_condition_syntax

console = Console()


class GraphVerifier:
    """
    Static analysis over the declared control-flow graph: directed-cycle detection across
    handoffs and deterministic routers, a bounded worst-case cost/turn ceiling, and an explicit
    accounting of the routing/cost paths this analysis does — and does not — cover.

    This intentionally does not claim a single blanket "mathematically bound" guarantee: handoffs
    and deterministic routers are static edges and are cycle-checked; semantic (`auto_route`)
    routing is an LLM decision at runtime and can't be predicted statically, so it's reported
    separately as a non-deterministic path bounded only by the engine's hard turn cap, not by
    graph acyclicity.
    """

    def __init__(self, project_dir: Path):
        self.project_dir = project_dir

    def verify(self):
        graph = parse_project(self.project_dir)
        cfg = graph.config

        console.print(
            Panel(
                f"[bold cyan]🛡️ Control-Flow Graph Verifier: '{cfg.name}'[/bold cyan]\n"
                f"[dim]Cycle detection across handoffs + routers, delegation depth bounds, "
                f"and worst-case cost accounting...[/dim]",
                border_style="cyan",
            )
        )

        adj, edge_source = self._build_adjacency(cfg)
        cycles = self._detect_cycles(adj)

        console.print("[bold white]State Proof Metrics:[/bold white]")
        console.print(f" • Total Agent States: [bold green]{len(cfg.agents)}[/bold green]")
        console.print(f" • Entry Point: [bold green]{cfg.default_agent}[/bold green]")
        console.print(
            " • Max Allowed Turn Limit (per top-level turn): [bold green]10 iterations (hard capped)[/bold green]"
        )

        # --- Static routing surfaces: handoffs + deterministic routers ---
        if cycles:
            console.print(
                "\n[bold yellow]⚠️ Cycles detected across static handoffs/routers "
                "(safely bounded by the per-turn iteration cap, but review for intent):[/bold yellow]"
            )
            for c in cycles:
                console.print(f"   ↳ [dim]{c}[/dim]")
        else:
            console.print(
                "\n[bold green]✓ Acyclic across static handoffs and deterministic routers.[/bold green]"
            )

        # --- Router condition syntax: a condition using unsupported grammar (most commonly
        # `state.get(...)`, which reads naturally but isn't supported) never raises anywhere a
        # user would notice — it's caught at evaluation time, logged, and the router just silently
        # never fires. Catch it here instead, before that becomes a runtime surprise. ---
        known_functions = {cf.name for cf in cfg.condition_functions}
        bad_conditions = []
        for agent_name, agent_cfg in cfg.agents.items():
            for router in getattr(agent_cfg, "routers", []) or []:
                if router.condition:
                    reason = validate_condition_syntax(router.condition, known_functions)
                    if reason:
                        bad_conditions.append((agent_name, router.condition, reason))
        if bad_conditions:
            console.print(
                "\n[bold red]✗ Router condition syntax errors "
                "(these routers will silently never fire):[/bold red]"
            )
            for agent_name, condition, reason in bad_conditions:
                console.print(f"   ↳ [dim]{agent_name}: {condition!r} — {reason}[/dim]")
        else:
            has_conditions = any(
                router.condition
                for agent_cfg in cfg.agents.values()
                for router in (getattr(agent_cfg, "routers", []) or [])
            )
            if has_conditions:
                console.print(
                    "\n[bold green]✓ All conditional router conditions are syntactically valid.[/bold green]"
                )

        # --- available_when condition syntax: same failure mode as router conditions, and the
        # same restricted grammar (validate_condition_syntax/safe_eval) — but where a bad router
        # condition just silently never fires, a bad available_when condition fails CLOSED
        # (_tool_currently_available), permanently hiding the tool from the agent that needs it.
        # Catch it here instead of leaving a developer to discover a "missing" tool at runtime. ---
        bad_available_when = []
        for agent_name, agent_cfg in cfg.agents.items():
            for tool in getattr(agent_cfg, "tools", []) or []:
                available_when = getattr(tool, "available_when", None)
                if available_when:
                    reason = validate_condition_syntax(available_when, known_functions)
                    if reason:
                        bad_available_when.append((agent_name, tool.name, available_when, reason))
        if bad_available_when:
            console.print(
                "\n[bold red]✗ available_when condition syntax errors "
                "(these tools will be permanently hidden — fails closed):[/bold red]"
            )
            for agent_name, tool_name, condition, reason in bad_available_when:
                console.print(
                    f"   ↳ [dim]{agent_name}.{tool_name}: {condition!r} — {reason}[/dim]"
                )
        else:
            has_available_when = any(
                getattr(tool, "available_when", None)
                for agent_cfg in cfg.agents.values()
                for tool in (getattr(agent_cfg, "tools", []) or [])
            )
            if has_available_when:
                console.print(
                    "\n[bold green]✓ All available_when conditions are syntactically valid.[/bold green]"
                )

        # --- state_schema presence: a nudge, not a failure. Without it, write_state accepts any
        # key/type with zero validation — a typo'd key or a value of the wrong type sits silently
        # in state until something downstream (a prompt, a router condition, a reducer) trips over
        # it in a way that's hard to trace back to the actual write. inta new scaffolds this by
        # default; a project without one may simply not have needed it yet, or may have deleted it
        # deliberately, so this is only ever a suggestion. ---
        if not cfg.state_schema:
            console.print(
                "\n[dim]ℹ No state_schema configured — write_state accepts any key/type "
                "unchecked. Consider adding one (see docs/04_Shared_State_Redux.md) once your "
                "agents' write_state calls grow past a couple of keys.[/dim]"
            )

        # --- required_approvals + required_approvers set together: when required_approvers is
        # non-empty it overrides required_approvals entirely (approval count becomes
        # len(required_approvers)) — see config/schema.py's ToolConfig docstrings. Setting both to
        # DIFFERENT values is very likely a mistake (the author probably meant one or the other,
        # or expected them to combine), and required_approvals is silently dead configuration in
        # that case — nothing raises, it just quietly does nothing. Only flagged when
        # required_approvals was actually changed from its default of 1, so a project that just
        # left it at the default alongside a real required_approvers list isn't flagged for
        # something nobody actually configured. ---
        conflicting_approvals = []
        for agent_name, agent_cfg in cfg.agents.items():
            for tool in getattr(agent_cfg, "tools", []) or []:
                approvers = getattr(tool, "required_approvers", None)
                approvals = getattr(tool, "required_approvals", 1)
                if approvers and approvals != 1 and approvals != len(approvers):
                    conflicting_approvals.append((agent_name, tool.name, approvals, approvers))
        for tool in cfg.tools:
            approvers = getattr(tool, "required_approvers", None)
            approvals = getattr(tool, "required_approvals", 1)
            if approvers and approvals != 1 and approvals != len(approvers):
                conflicting_approvals.append((None, tool.name, approvals, approvers))
        if conflicting_approvals:
            console.print(
                "\n[bold yellow]⚠ required_approvals is dead configuration on these tools "
                "(required_approvers is set, so it wins and required_approvals is ignored):"
                "[/bold yellow]"
            )
            for agent_name, tool_name, approvals, approvers in conflicting_approvals:
                label = f"{agent_name}.{tool_name}" if agent_name else tool_name
                console.print(
                    f"   ↳ [dim]{label}: required_approvals={approvals} but "
                    f"required_approvers={approvers} (effective count: {len(approvers)})[/dim]"
                )

        # --- Lethal-trifecta guardrail: an agent with both an untrusted-output tool (RAG's
        # search_knowledge_base, or MCP/OpenAPI tools by default — see
        # LocalToolConfig.untrusted_output) and a separately-flagged sensitive tool
        # (requires_approval=true) can't distinguish "this session is clean" from "this session
        # already ingested untrusted content" unless something gates on
        # state["_untrusted_content_ingested"], which the runtime now tracks for free. Advisory
        # only — this can't know which tools are actually exfiltration-capable, only that a
        # developer already flagged one as sensitive without using the new signal. A tools: entry
        # is usually a name-reference to a root-level tool (ToolReferenceConfig), so both
        # untrusted_output and requires_approval are resolved from the referenced root tool;
        # available_when is checked on the per-agent reference first (the field that actually
        # matters in practice — see ToolReferenceConfig.available_when), falling back to the root
        # tool's own. ---
        root_tools_by_name = {t.name: t for t in cfg.tools}

        def _resolve(tool):
            if isinstance(tool, ToolReferenceConfig):
                return root_tools_by_name.get(tool.name, tool)
            return tool

        untrusted_gaps = []
        for agent_name, agent_cfg in cfg.agents.items():
            tools = getattr(agent_cfg, "tools", []) or []
            untrusted_names = {
                tool.name for tool in tools if getattr(_resolve(tool), "untrusted_output", False)
            }
            if not untrusted_names:
                continue
            for tool in tools:
                if tool.name in untrusted_names:
                    continue
                resolved = _resolve(tool)
                if not getattr(resolved, "requires_approval", False):
                    continue
                cond = getattr(tool, "available_when", None) or getattr(
                    resolved, "available_when", None
                )
                if not cond or "_untrusted_content_ingested" not in cond:
                    untrusted_gaps.append((agent_name, tool.name, sorted(untrusted_names)))

        if untrusted_gaps:
            console.print(
                "\n[bold yellow]⚠ Sensitive tool not gated against ingested untrusted "
                "content:[/bold yellow]"
            )
            for agent_name, tool_name, sources in untrusted_gaps:
                console.print(
                    f"   ↳ [dim]{agent_name}.{tool_name} requires_approval=true, but its "
                    f"available_when doesn't reference _untrusted_content_ingested — {agent_name} "
                    f"also has untrusted-output tool(s) {', '.join(sources)}. Consider "
                    f'available_when: "not _untrusted_content_ingested" so a compromised '
                    "document/tool result this session can't ride along to a high-risk action "
                    "without extra scrutiny.[/dim]"
                )

        # --- Sandbox tools (arbitrary code execution, even if resource-isolated — see
        # runtime/sandbox.py) without requires_approval: true. Advisory only: plenty of projects
        # will legitimately want an unattended sandbox tool (e.g. a coding agent's own test-run
        # step), so this nudges rather than blocks. ---
        ungated_sandboxes = []
        for agent_name, agent_cfg in cfg.agents.items():
            for tool in getattr(agent_cfg, "tools", []) or []:
                resolved = _resolve(tool)
                if isinstance(resolved, SandboxToolConfig) and not getattr(
                    resolved, "requires_approval", False
                ):
                    ungated_sandboxes.append((agent_name, tool.name))
        if ungated_sandboxes:
            console.print(
                "\n[dim]ℹ Sandbox tool(s) without requires_approval — code execution is "
                "process/resource-isolated but not a filesystem/network security boundary (see "
                "runtime/sandbox.py). Consider requires_approval: true if this agent's code "
                "isn't fully trusted:[/dim]"
            )
            for agent_name, tool_name in ungated_sandboxes:
                console.print(f"   ↳ [dim]{agent_name}.{tool_name}[/dim]")

        # --- requires_approval needs a persistent memory backend: a pause is held in
        # state["_pending_approval"], only resumed via a *separate* /resume request, and only
        # ever surfaced in Monitor's session list (the Approve/Deny button) if that session's
        # state can actually be looked up there — server/monitor.py's get_memory() only knows
        # how to list sqlite/postgres/custom-with-get_all_sessions backends. memory.type left at
        # its schema default (sliding_window) or explicitly buffer is in-process only: the pause
        # genuinely happens (the tool call really does return "awaiting human approval"), but
        # Monitor has nothing to query, so the button never renders, and nothing survives past
        # that process anyway to resume from. Advisory only — a project with its own resume path
        # that doesn't depend on Monitor's session list can ignore this. ---
        if cfg.memory.type in ("sliding_window", "buffer"):
            approval_tools = []
            for agent_name, agent_cfg in cfg.agents.items():
                for tool in getattr(agent_cfg, "tools", []) or []:
                    resolved = _resolve(tool)
                    if getattr(resolved, "requires_approval", False) or getattr(
                        resolved, "required_approvers", None
                    ):
                        approval_tools.append((agent_name, tool.name))
            if approval_tools:
                console.print(
                    f"\n[dim]ℹ requires_approval tool(s) with memory.type: {cfg.memory.type} — "
                    "a pause only shows up in Monitor's session list (the Approve/Deny button) "
                    "and only survives to be resumed with a persistent backend. Consider "
                    "memory.type: sqlite (or postgres/redis) instead:[/dim]"
                )
                for agent_name, tool_name in approval_tools:
                    console.print(f"   ↳ [dim]{agent_name}.{tool_name}[/dim]")

        # --- spawns.on_complete reserved keys: apply_state_write (the same pipeline write_state
        # itself goes through) silently rejects any key starting with `_` — reserved for internal
        # engine bookkeeping (_pending_approval, _dynamic_agents, _active_agent_name, ...). A
        # developer configuring on_complete against one of these would only find out via a logged
        # error the first time a spawn actually completes. Catch it here instead. ---
        bad_on_complete = []
        for agent_name, agent_cfg in cfg.agents.items():
            spawns_cfg = getattr(agent_cfg, "spawns", None)
            for action in getattr(spawns_cfg, "on_complete", []) or []:
                if action.key.startswith("_"):
                    bad_on_complete.append((agent_name, action.key))
        if bad_on_complete:
            console.print(
                "\n[bold red]✗ spawns.on_complete writes to a reserved key "
                "(silently rejected at runtime — write_state never accepts a leading `_`):[/bold red]"
            )
            for agent_name, key in bad_on_complete:
                console.print(f"   ↳ [dim]{agent_name}.spawns.on_complete: {key!r}[/dim]")
        else:
            has_on_complete = any(
                getattr(getattr(a, "spawns", None), "on_complete", None)
                for a in cfg.agents.values()
            )
            if has_on_complete:
                console.print(
                    "\n[bold green]✓ All spawns.on_complete keys are writable.[/bold green]"
                )

        # --- Non-deterministic surface: semantic auto_route ---
        auto_route_agents = [
            name for name, a in cfg.agents.items() if getattr(a, "auto_route", False)
        ]
        if auto_route_agents:
            console.print(
                f"\n[bold yellow]⚠ Non-deterministic routing: {len(auto_route_agents)} agent(s) "
                f"use `auto_route` (semantic swarm routing): {', '.join(auto_route_agents)}[/bold yellow]"
            )
            console.print(
                "   [dim]An LLM call picks the next agent at runtime — this cannot be statically "
                "verified. Worst case, treat these as fully connected to every other agent; the "
                "only thing bounding them is the 10-iteration hard turn cap above, not graph "
                "acyclicity.[/dim]"
            )

        # --- Non-deterministic surface: dynamic runtime agent creation (spawn_agent) ---
        spawning_agents = [name for name, a in cfg.agents.items() if getattr(a, "spawns", None)]
        if spawning_agents:
            console.print(
                f"\n[bold yellow]⚠ Non-deterministic agent creation: {len(spawning_agents)} "
                f"agent(s) can dynamically spawn sub-agents at runtime: "
                f"{', '.join(spawning_agents)}[/bold yellow]"
            )
            console.print(
                "   [dim]Sub-agents created via spawn_agent don't exist at verify-time and are "
                "not part of the cycle/cost analysis above — same reasoning as auto_route. Unlike "
                "auto_route, the graph shape is structurally bounded (a spawned agent's only "
                "control-flow option is return_to_creator, no dynamic-to-dynamic edges, no "
                "recursive spawning by default), and creation itself is capped by each agent's "
                "spawns.max_creations_per_session — but a spawned agent's own tool use is still "
                "bounded only by the 10-iteration hard turn cap, not by graph acyclicity.[/dim]"
            )

        # --- Delegation subtree ---
        delegating_agents = {
            name: a.delegations for name, a in cfg.agents.items() if a.delegations
        }
        max_depth = cfg.circuit_breakers.max_delegation_depth
        max_turns = cfg.circuit_breakers.max_delegation_turns
        if delegating_agents:
            console.print(
                f"\n[bold white]Delegation subtree:[/bold white] {len(delegating_agents)} agent(s) "
                f"can delegate. Bounded by circuit_breakers.max_delegation_depth="
                f"[bold green]{max_depth}[/bold green] and max_delegation_turns="
                f"[bold green]{max_turns}[/bold green] (each delegated sub-agent runs its own "
                f"independent chain, not cycle-checked against the parent — delegation always "
                f"returns control, so depth is the relevant bound, not cycles)."
            )

        # --- Cost ceiling ---
        cost_per_token = self._resolve_cost_per_token(cfg.model.primary)
        max_tokens_per_turn = cfg.model.max_tokens or 1500
        main_loop_tokens = max_tokens_per_turn * 10
        delegation_tokens = max_tokens_per_turn * max_turns * max_depth if delegating_agents else 0
        bounded_tokens = main_loop_tokens + delegation_tokens
        bounded_cost = bounded_tokens * cost_per_token

        table = Table(title="Worst-Case Cost Accounting", border_style="dim")
        table.add_column("Contribution", style="bold white", overflow="fold")
        table.add_column("Tokens (worst case)", justify="right", overflow="fold")
        table.add_column("Bounded?", justify="center")
        table.add_row("Main turn loop (10 iterations x max_tokens)", f"{main_loop_tokens:,}", "[green]yes[/green]")
        if delegating_agents:
            table.add_row(
                f"Delegation subtree ({max_depth} deep x {max_turns} turns)",
                f"{delegation_tokens:,}",
                "[green]yes[/green]",
            )
        cb = cfg.circuit_breakers
        corrector_tokens = 2 * cb.max_corrector_tokens
        table.add_row(
            "Self-healing corrector calls (per malformed tool/response, up to 2 retries)",
            f"{corrector_tokens:,} (max_corrector_tokens)",
            "[green]yes[/green]",
        )
        table.add_row(
            "Memory compression on eviction (input side)",
            f"{cb.max_compression_batch_messages} msgs/batch (max_compression_batch_messages)",
            "[green]yes[/green]",
        )
        table.add_row(
            "Parallel tool calls within one LLM turn",
            f"{cb.max_parallel_tool_calls_per_turn} concurrent (max_parallel_tool_calls_per_turn)",
            "[green]yes[/green]",
        )
        console.print()
        console.print(table)

        console.print(
            f"\n[bold white]Bounded floor:[/bold white] [bold green]${bounded_cost:.4f}[/bold green] "
            f"({bounded_tokens:,} tokens) at ~${cost_per_token * 1000:.5f}/1K tokens for "
            f"'{cfg.model.primary}'. This covers only the main turn loop and delegation subtree — "
            "every row above is now individually capped per call/batch too, but isn't folded into "
            "this total since self-healing retries, compression batches, and parallel-tool-call "
            "rounds scale with how many malformed calls/evictions/tool rounds actually happen in a "
            "turn, not with a fixed per-turn count. This is a floor, not a ceiling on total cost."
        )

        console.print(
            "\n[bold green]Verification complete.[/bold green] [dim]Acyclic (or safely-capped-cycle) "
            "across handoffs and deterministic routers; auto_route and per-turn tool/delegation "
            "fan-out are runtime-bounded by iteration/depth caps rather than statically verified.[/dim]"
        )

    def _build_adjacency(self, cfg) -> tuple[dict, dict]:
        """Builds the directed graph of statically-known transitions: handoffs, root routers, and
        per-agent conditional routers. auto_route is deliberately excluded — it isn't a static
        edge, it's resolved by an LLM call at runtime — and is reported separately."""
        adj: dict[str, list[str]] = {name: [] for name in cfg.agents}
        edge_source: dict[tuple[str, str], str] = {}

        for agent_name, agent_cfg in cfg.agents.items():
            for target in agent_cfg.handoffs or []:
                adj.setdefault(agent_name, []).append(target)
                edge_source[(agent_name, target)] = "handoff"
            for router in agent_cfg.routers or []:
                if router.target:
                    adj.setdefault(agent_name, []).append(router.target)
                    edge_source[(agent_name, router.target)] = "conditional router"

        for agent_name, root_router in cfg.routers.items():
            for target in root_router.possible_targets or []:
                adj.setdefault(agent_name, []).append(target)
                edge_source[(agent_name, target)] = "root router"

        return adj, edge_source

    def _detect_cycles(self, adj: dict) -> list[str]:
        visited, rec_stack, cycles = set(), set(), []

        def dfs(node, path):
            visited.add(node)
            rec_stack.add(node)
            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor, path + [neighbor])
                elif neighbor in rec_stack:
                    cycle_path = path[path.index(neighbor):] + [neighbor]
                    cycles.append(" ➔ ".join(cycle_path))
            rec_stack.remove(node)

        for agent in adj:
            if agent not in visited:
                dfs(agent, [agent])
        return cycles

    def _resolve_cost_per_token(self, model_name: str) -> float:
        import litellm

        try:
            cost_info = litellm.get_model_info(model=model_name)
            input_cost = cost_info.get("input_cost_per_token", 0.000005)
            output_cost = cost_info.get("output_cost_per_token", 0.000015)
            return (input_cost + output_cost) / 2.0
        except Exception:
            if "flash" in model_name or "mini" in model_name:
                return 0.0000005
            if "claude-3-7" in model_name or "gpt-4o" in model_name:
                return 0.000015
            return 0.00001
