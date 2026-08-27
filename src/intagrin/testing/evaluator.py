from pathlib import Path

from rich.console import Console
from rich.table import Table

from ..compiler.parser import parse_project
from .eval_runner import load_eval_cases, run_case

console = Console()


async def run_evals(project_dir: Path):
    cases = load_eval_cases(project_dir)
    if not cases:
        console.print(
            "[bold red]No evaluation cases found. Run 'inta synth' to generate some, or add "
            "them under `evaluations:` in tests/evals.yaml.[/bold red]"
        )
        return

    graph = parse_project(project_dir)

    table = Table(title="Agent Evaluation Results")
    table.add_column("Test Case", style="cyan")
    table.add_column("Agent", style="blue")
    table.add_column("Expected Tool", style="magenta")
    table.add_column("Status", style="bold")

    for case in cases:
        result = await run_case(graph, project_dir, case)

        if not result.deterministic_pass:
            status = f"[red]FAIL[/red] [dim]({'; '.join(result.reasons)})[/dim]"
        else:
            status = "[green]PASS[/green]"

            # Run LLM-as-a-Judge checks only once the deterministic assertions already passed
            evaluators = case.get("evaluators", [])
            if evaluators:
                context_data = "\n".join(str(t) for t in result.called_tools)
                import litellm

                for ev in evaluators:
                    if ev.get("type") == "llm_judge":
                        criteria = ev.get("criteria", "")
                        prompt = (
                            f"You are an expert RAG Evaluator.\nEvaluate based on this criteria: "
                            f"{criteria}\n\nUser Input: {result.input}\nTools called: "
                            f"{context_data}\nAgent Answer: {result.final_answer}\n\n"
                            "Respond EXACTLY with 'PASS' or 'FAIL: <reason>'."
                        )
                        try:
                            resp = await litellm.acompletion(
                                model=graph.config.model.primary,
                                messages=[{"role": "user", "content": prompt}],
                                temperature=0.0,
                            )
                            judge = resp.choices[0].message.content.strip()
                            if not judge.startswith("PASS"):
                                status = f"[red]{judge}[/red]"
                                break
                        except Exception as e:
                            status = f"[red]JUDGE ERROR: {e}[/red]"
                            break

        table.add_row(
            result.name, result.starting_agent, case.get("expected_tool", ""), status
        )

    console.print(table)
