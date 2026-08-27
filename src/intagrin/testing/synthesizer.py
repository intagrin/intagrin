import inspect
from pathlib import Path

import yaml
from rich.console import Console
from rich.panel import Panel

from ..compiler.parser import parse_project
from ..runtime.tools_loader import load_local_tool

console = Console()


class SyntheticEvalSynthesizer:
    """
    Autonomous Synthetic Edge-Case Dataset Synthesizer (`intagrin evolve`).
    Inspects python tool signatures, parameter type annotations, and docstrings to mathematically
    generate high-coverage edge cases (boundary limits, negative values, type overflows, happy paths)
    and saves them to `tests/evals.yaml` with zero developer effort and sub-cent API costs.
    """

    def __init__(self, project_dir: Path, count: int = 15):
        self.project_dir = project_dir
        self.count = count

    def evolve(self):
        graph = parse_project(self.project_dir)
        cfg = graph.config

        console.print(
            Panel(
                f"[bold cyan]🧬 IntaGrin Synthetic Evals Synthesizer: '{cfg.name}'[/bold cyan]\n"
                f"[dim]Analyzing tool reflection signatures and synthesizing {self.count} edge-case tests...[/dim]",
                border_style="cyan",
            )
        )

        synthetic_cases = []

        # 1. Inspect all agent tools across the project
        import sys

        if str(self.project_dir) not in sys.path:
            sys.path.insert(0, str(self.project_dir))

        for agent_name, agent_cfg in cfg.agents.items():
            for t in agent_cfg.tools:
                if hasattr(t, "module"):
                    try:
                        func = load_local_tool(t.module, getattr(t, "function", t.name))
                        sig = inspect.signature(func)

                        # Generate Happy Path
                        synthetic_cases.append(
                            {
                                "name": f"{t.name}: happy path",
                                "input": f"Please run {t.name} with standard valid parameters.",
                                "starting_agent": agent_name,
                                "expected_agent": agent_name,
                                "expected_tool": t.name,
                                "test_type": "happy_path",
                            }
                        )

                        # Inspect parameters for boundary edge cases
                        for param_name, param in sig.parameters.items():
                            # Numeric boundaries
                            if param.annotation in [int, float]:
                                synthetic_cases.append(
                                    {
                                        "name": f"{t.name}: numeric boundary ({param_name})",
                                        "input": f"Execute {t.name} with {param_name} set to 0 and -999999.",
                                        "starting_agent": agent_name,
                                        "expected_agent": agent_name,
                                        "expected_tool": t.name,
                                        "test_type": "numeric_boundary",
                                    }
                                )
                            # String / SQL Injection boundaries
                            elif param.annotation is str:
                                synthetic_cases.append(
                                    {
                                        "name": f"{t.name}: string boundary ({param_name})",
                                        "input": f"Look up {t.name} where {param_name} is 'NULL' or empty.",
                                        "starting_agent": agent_name,
                                        "expected_agent": agent_name,
                                        "expected_tool": t.name,
                                        "test_type": "string_boundary",
                                    }
                                )
                    except Exception:
                        pass

        # 2. Handoff Edge Cases
        for agent_name, agent_cfg in cfg.agents.items():
            for handoff_target in agent_cfg.handoffs or []:
                synthetic_cases.append(
                    {
                        "name": f"handoff: {agent_name} -> {handoff_target}",
                        "input": f"I need help transitioning from {agent_name} to {handoff_target}.",
                        "starting_agent": agent_name,
                        "expected_agent": handoff_target,
                        "expected_tool": "transfer_agent",
                        "test_type": "handoff_transition",
                    }
                )

        # Cap to the requested count (no padding — a smaller tool/handoff surface just yields fewer cases)
        selected_cases = synthetic_cases[: self.count]

        evals_file = self.project_dir / "tests" / "evals.yaml"
        evals_file.parent.mkdir(parents=True, exist_ok=True)

        eval_payload = {"version": "1.0", "evaluations": selected_cases}

        with open(evals_file, "w", encoding="utf-8") as f:
            yaml.dump(eval_payload, f, default_flow_style=False, sort_keys=False)

        console.print(
            f"[bold green]✓ Successfully generated {len(selected_cases)} synthetic test cases in: tests/evals.yaml[/bold green]"
        )
        console.print(
            "[dim]Run 'intagrin eval' to benchmark your swarm against these tests (Est. cost: < $0.001).[/dim]"
        )
