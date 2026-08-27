from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..compiler.parser import parse_project
from ..runtime.model_info import resolve_context_window
from .eval_runner import EvalCaseResult, load_eval_cases, run_case

console = Console()

LATENCY_SLA_SECONDS = 3.0
CPI_ELEVATED_THRESHOLD = 0.65
CPI_CRITICAL_THRESHOLD = 0.85
TOOL_ERROR_RATE_THRESHOLD = 0.10


class AgentICUDiagnostics:
    """
    Agent Diagnostics: runs a batch of real requests against the swarm and reports aggregate
    health metrics computed from those runs — context utilization, tool error rate, cost, and
    latency — with a rules-based explanation keyed to whichever metric actually crossed its
    threshold. Every number shown is derived from the runs it just executed; nothing here is a
    fixed placeholder.

    Prefers `tests/evals.yaml` if present (real, user-authored or `inta synth`-generated cases,
    giving a statistically meaningful read of behavior); otherwise falls back to a small battery
    of generic probe prompts, one per agent as entry point.
    """

    def __init__(self, project_dir: Path, probe_battery_size: int = 5):
        self.project_dir = project_dir
        self.probe_battery_size = probe_battery_size

    async def run_diagnostics(self, custom_probe: str | None = None):
        graph = parse_project(self.project_dir)
        console.print(
            Panel(
                f"[bold cyan]🏥 IntaGrin Agent Diagnostics: '{graph.config.name}'[/bold cyan]\n"
                f"[dim]Running a probe battery and computing real health metrics...[/dim]",
                border_style="cyan",
            )
        )

        if custom_probe:
            cases = [{"name": "custom probe", "input": custom_probe, "starting_agent": graph.config.default_agent}]
            source = "1 custom probe (--prompt)"
        else:
            cases = load_eval_cases(self.project_dir)
            source = "tests/evals.yaml"
            if not cases:
                cases = self._build_probe_battery(graph)
                source = f"{len(cases)} generic probes (no tests/evals.yaml found)"

        results: list[EvalCaseResult] = []
        for case in cases:
            results.append(await run_case(graph, self.project_dir, case))

        console.print(f"[dim]Ran {len(results)} case(s) from: {source}[/dim]\n")

        metrics = self._aggregate(graph, results)
        self._render_vitals(metrics)
        self._render_diagnosis(metrics, results)

    def _build_probe_battery(self, graph) -> list[dict]:
        agent_names = list(graph.config.agents.keys())[: self.probe_battery_size]
        return [
            {
                "name": f"probe: {agent_name}",
                "input": "Please introduce yourself and describe what you can help with.",
                "starting_agent": agent_name,
            }
            for agent_name in agent_names
        ]

    def _aggregate(self, graph, results: list[EvalCaseResult]) -> dict:
        n = len(results)
        total_tokens = sum(r.tokens for r in results)
        total_cost = sum(r.cost for r in results)
        total_duration = sum(r.duration_seconds for r in results)
        total_tool_calls = sum(len(r.called_tools) for r in results)
        total_tool_errors = sum(r.tool_error_count for r in results)
        crashed = [r for r in results if r.crashed]

        avg_tokens_per_run = total_tokens / n if n else 0
        avg_latency = total_duration / n if n else 0.0
        max_latency = max((r.duration_seconds for r in results), default=0.0)
        tool_error_rate = (total_tool_errors / total_tool_calls) if total_tool_calls else 0.0
        crash_rate = (len(crashed) / n) if n else 0.0
        burn_rate = total_tokens / total_duration if total_duration > 0.1 else 0.0

        context_window = resolve_context_window(graph.config.model.primary)
        cpi = min(1.0, avg_tokens_per_run / context_window) if context_window else 0.0

        return {
            "n": n,
            "total_tokens": total_tokens,
            "avg_tokens_per_run": avg_tokens_per_run,
            "context_window": context_window,
            "cpi": cpi,
            "total_cost": total_cost,
            "total_tool_calls": total_tool_calls,
            "total_tool_errors": total_tool_errors,
            "tool_error_rate": tool_error_rate,
            "avg_latency": avg_latency,
            "max_latency": max_latency,
            "burn_rate": burn_rate,
            "crashed": crashed,
            "crash_rate": crash_rate,
        }

    def _render_vitals(self, m: dict):
        cpi_color = (
            "green"
            if m["cpi"] < CPI_ELEVATED_THRESHOLD
            else "yellow" if m["cpi"] < CPI_CRITICAL_THRESHOLD else "red"
        )
        cpi_status = (
            "NORMAL"
            if m["cpi"] < CPI_ELEVATED_THRESHOLD
            else "ELEVATED" if m["cpi"] < CPI_CRITICAL_THRESHOLD else "CRITICAL"
        )

        tea_color = "green" if m["tool_error_rate"] == 0 else "yellow" if m["tool_error_rate"] < TOOL_ERROR_RATE_THRESHOLD else "red"
        tea_status = f"{m['total_tool_errors']}/{m['total_tool_calls']} calls errored ({m['tool_error_rate']:.0%})" if m["total_tool_calls"] else "no tool calls observed"

        epistemic_color = "green" if m["crash_rate"] == 0 else "red"
        epistemic_status = "STABLE" if m["crash_rate"] == 0 else f"{len(m['crashed'])}/{m['n']} runs crashed"

        latency_pass = m["avg_latency"] <= LATENCY_SLA_SECONDS
        latency_color = "green" if latency_pass else "red"
        latency_status = f"{'within' if latency_pass else 'EXCEEDS'} {LATENCY_SLA_SECONDS:.0f}s SLA"

        table = Table(title=f"Agent Vital Signs — {m['n']} run(s)", border_style="dim")
        table.add_column("Metric", style="bold white")
        table.add_column("Value", style="cyan")
        table.add_column("Status", justify="center")

        table.add_row(
            "Context utilization (CPI)",
            f"{m['cpi']:.2f} ({m['avg_tokens_per_run']:.0f} avg tokens / {m['context_window']:,} window)",
            f"[{cpi_color}]{cpi_status}[/{cpi_color}]",
        )
        table.add_row("Tool error rate", tea_status, f"[{tea_color}]{'OK' if m['tool_error_rate'] < TOOL_ERROR_RATE_THRESHOLD else 'ELEVATED'}[/{tea_color}]")
        table.add_row("Token burn rate", f"{m['burn_rate']:.1f} tokens/sec" if m["burn_rate"] else "n/a", "[cyan]—[/cyan]")
        table.add_row("Crash rate", epistemic_status, f"[{epistemic_color}]{'PASS' if m['crash_rate'] == 0 else 'FAIL'}[/{epistemic_color}]")
        table.add_row(
            "Avg latency (wall-clock)",
            f"{m['avg_latency']:.2f}s avg, {m['max_latency']:.2f}s max",
            f"[{latency_color}]{latency_status}[/{latency_color}]",
        )
        table.add_row("Total cost (this run)", f"${m['total_cost']:.4f}", "[cyan]—[/cyan]")

        console.print(table)

    def _render_diagnosis(self, m: dict, results: list[EvalCaseResult]):
        suspects = []
        if m["crash_rate"] > 0:
            sample = next((r.crash_error for r in m["crashed"] if r.crash_error), "unknown")
            suspects.append(
                f"Engine crashes on {len(m['crashed'])}/{m['n']} run(s). Sample error: {sample}"
            )
        if m["tool_error_rate"] >= TOOL_ERROR_RATE_THRESHOLD:
            suspects.append(
                f"Tool error rate {m['tool_error_rate']:.0%} exceeds the "
                f"{TOOL_ERROR_RATE_THRESHOLD:.0%} threshold ({m['total_tool_errors']}/"
                f"{m['total_tool_calls']} calls) — check tool argument schemas and docstrings."
            )
        if m["cpi"] >= CPI_ELEVATED_THRESHOLD:
            suspects.append(
                f"Average context utilization {m['cpi']:.0%} of the model's context window — "
                "consider a tighter memory.max_messages or enabling lazy_load_tools."
            )
        if m["avg_latency"] > LATENCY_SLA_SECONDS:
            suspects.append(
                f"Average latency {m['avg_latency']:.2f}s exceeds the {LATENCY_SLA_SECONDS:.0f}s "
                "SLA — check model choice, tool network calls, or self-healing retries."
            )

        if suspects:
            console.print("\n[bold red]Pathology detected:[/bold red]")
            for i, s in enumerate(suspects, 1):
                console.print(f"  [bold yellow]{i}.[/bold yellow] {s}")
            console.print("\n[bold green]Suggested next steps:[/bold green]")
            console.print("  • Run [bold cyan]'inta tune'[/bold cyan] to auto-repair prompts against tests/evals.yaml.")
            console.print("  • Re-run [bold cyan]'inta diagnose'[/bold cyan] after changes to confirm the metric actually moved.")
        else:
            console.print(
                f"\n[bold green]No pathology detected across {m['n']} run(s) — all measured "
                "metrics are within threshold.[/bold green]"
            )
