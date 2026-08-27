from pathlib import Path

import litellm
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..compiler.parser import parse_project
from .eval_runner import load_eval_cases, run_case

console = Console()


class AutoTuner:
    """
    Self-Healing Evals & Prompt Auto-Tuning Engine.
    Executes tests from `tests/evals.yaml`. When failures occur, it enters a meta-reflection
    loop to diagnose prompt flaws and automatically improves .jinja2 prompts until tests pass.

    Every prompt file this touches is snapshotted before its first edit. If tuning exhausts
    max_iterations without reaching zero failures, all touched files are rolled back to their
    original content — a non-converging run can never leave the project worse than it started.
    """

    def __init__(self, project_dir: Path, max_iterations: int = 3):
        self.project_dir = project_dir
        self.max_iterations = max_iterations
        self._original_prompts: dict[Path, str] = {}

    async def tune(self):
        test_cases = load_eval_cases(self.project_dir)
        if not test_cases:
            console.print(
                "[bold yellow]No evaluation cases found. Run 'inta synth' to generate some, or "
                "add them under `evaluations:` in tests/evals.yaml.[/bold yellow]"
            )
            return

        console.print(
            Panel(
                "[bold cyan]🧬 IntaGrin Self-Healing Auto-Tuner Initialized[/bold cyan]\n"
                "[dim]Running evaluations and iteratively repairing prompts on failure...[/dim]",
                border_style="cyan",
            )
        )

        iteration = 1
        while iteration <= self.max_iterations:
            console.print(
                f"\n[bold purple]── Tuning Iteration {iteration}/{self.max_iterations} ──[/bold purple]"
            )
            failures = await self._run_eval_pass(test_cases)

            if not failures:
                console.print(
                    "\n[bold green]🎉 All evaluation tests PASSED! Agent prompts are fully tuned and optimized.[/bold green]"
                )
                return

            console.print(
                f"\n[bold yellow]⚠️ Detected {len(failures)} test failure(s). Invoking Meta-Reflection Tuner...[/bold yellow]"
            )
            await self._auto_repair_prompts(failures)
            iteration += 1

        self._rollback()
        console.print(
            "\n[bold red]Auto-Tuning reached maximum iterations without converging. All prompt "
            "files touched during this run were rolled back to their original content — nothing "
            "was left in a worse state than before. Review the failures above manually.[/bold red]"
        )

    async def _run_eval_pass(self, test_cases: list[dict]) -> list[dict]:
        failures = []
        graph = parse_project(self.project_dir)

        table = Table(title="Evaluation Run Results", border_style="dim")
        table.add_column("Test Input", style="cyan")
        table.add_column("Active Agent", style="magenta")
        table.add_column("Status", justify="center")

        for case in test_cases:
            result = await run_case(graph, self.project_dir, case)

            if result.deterministic_pass:
                table.add_row(result.input[:40], result.final_agent, "[green]PASS ✓[/green]")
            else:
                table.add_row(
                    result.input[:40],
                    result.final_agent,
                    f"[red]FAIL: {'; '.join(result.reasons)}[/red]",
                )
                failures.append(
                    {
                        "case": case,
                        "reasons": result.reasons,
                        "agent": result.final_agent,
                        "actual_output": result.final_answer,
                    }
                )

        console.print(table)
        return failures

    async def _auto_repair_prompts(self, failures: list[dict]):
        graph = parse_project(self.project_dir)
        for fail in failures:
            agent_name = fail["agent"]
            agent_cfg = graph.config.agents.get(agent_name)
            if not agent_cfg or not agent_cfg.system_prompt_file:
                continue

            prompt_path = self.project_dir / agent_cfg.system_prompt_file
            if not prompt_path.exists():
                continue

            current_prompt = prompt_path.read_text(encoding="utf-8")
            if prompt_path not in self._original_prompts:
                self._original_prompts[prompt_path] = current_prompt

            console.print(
                f"[bold cyan]Reflecting on failure for agent '{agent_name}'...[/bold cyan]"
            )

            meta_prompt = f"""You are an expert AI Prompt Engineer and Meta-Optimizer.
Your job is to rewrite an agent system prompt to fix an evaluation test failure.

AGENT NAME: {agent_name}
CURRENT PROMPT:
{current_prompt}

FAILED TEST CASE:
Input: {fail['case'].get('input')}
Failure Reasons: {fail.get('reasons', [])}
Actual Agent Response: {fail.get('actual_output')}

INSTRUCTIONS:
1. Identify why the current prompt failed the test.
2. Rewrite the prompt clearly and concisely to guarantee deterministic execution of required tool calls, handoffs, or response structures.
3. Return ONLY the new prompt text. Do not wrap in markdown quotes or preamble.
"""
            try:
                model = graph.config.model.primary
                resp = await litellm.acompletion(
                    model=model,
                    messages=[{"role": "user", "content": meta_prompt}],
                    temperature=0.2,
                )
                new_prompt = resp.choices[0].message.content.strip()
                if new_prompt.startswith("```"):
                    lines = new_prompt.split("\n")
                    new_prompt = "\n".join(lines[1:-1])

                prompt_path.write_text(new_prompt, encoding="utf-8")
                console.print(
                    f"[bold green]✓ Auto-repaired prompt file: '{agent_cfg.system_prompt_file}'[/bold green]"
                )
            except Exception as e:
                console.print(f"[bold red]Failed to auto-repair prompt: {e}[/bold red]")

    def _rollback(self):
        for prompt_path, original in self._original_prompts.items():
            try:
                prompt_path.write_text(original, encoding="utf-8")
            except Exception as e:
                console.print(f"[bold red]Failed to roll back '{prompt_path}': {e}[/bold red]")
