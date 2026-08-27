import json
import urllib.request
from pathlib import Path

import litellm
from rich.console import Console
from rich.panel import Panel

console = Console()


class OpenAPISynthesizer:
    """
    Autonomous OpenAPI / Swagger to Multi-Agent Swarm Synthesizer.
    Parses OpenAPI 3.0 / Swagger specifications (local file or remote URL),
    clusters endpoints into logical agent personas using semantic grouping,
    and automatically scaffolds `ai.yaml`, `prompts/`, and `tools/` with typed Python functions.
    """

    def __init__(self, spec_source: str, project_name: str | None = None):
        self.spec_source = spec_source
        self.project_name = project_name or "api-swarm"

    def load_spec(self) -> dict:
        if self.spec_source.startswith("http://") or self.spec_source.startswith(
            "https://"
        ):
            console.print(
                f"[bold cyan]Fetching remote OpenAPI specification from {self.spec_source}...[/bold cyan]"
            )
            req = urllib.request.Request(
                self.spec_source, headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                content = resp.read().decode("utf-8")
                return json.loads(content)
        else:
            path = Path(self.spec_source)
            if not path.exists():
                raise FileNotFoundError(
                    f"OpenAPI spec file not found: {self.spec_source}"
                )
            return json.loads(path.read_text(encoding="utf-8"))

    async def synthesize(self, target_dir: Path):
        spec = self.load_spec()
        title = spec.get("info", {}).get("title", self.project_name)
        paths = spec.get("paths", {})

        console.print(
            Panel(
                f"[bold green]⚡ Reverse-Engineering OpenAPI: '{title}'[/bold green]\n"
                f"[dim]Found {len(paths)} endpoint paths. Performing semantic agent clustering...[/dim]",
                border_style="green",
            )
        )

        # Extract summary of endpoints
        endpoints_summary = []
        for path_url, methods in list(paths.items())[:25]:  # Sample top 25 endpoints
            for method, details in methods.items():
                if method.lower() in ["get", "post", "put", "delete", "patch"]:
                    endpoints_summary.append(
                        {
                            "method": method.upper(),
                            "path": path_url,
                            "summary": details.get("summary")
                            or details.get("description")
                            or path_url,
                            "operationId": details.get(
                                "operationId", f"{method}_{path_url.replace('/', '_')}"
                            ),
                        }
                    )

        # Ask LLM to cluster endpoints into 2-3 agents and generate tools
        prompt = f"""You are an expert AI Framework Architect.
Given this list of API endpoints from an OpenAPI spec titled '{title}':
{json.dumps(endpoints_summary, indent=2)}

TASK:
1. Group these endpoints into 2 or 3 logical Agent personas (e.g. 'billing_agent', 'support_agent', 'ops_agent').
2. Choose one agent as the 'default_agent'.
3. Write clean, production-grade Python functions in 'tools/custom_tools.py' with type hints, comprehensive docstrings, and enterprise HTTP error handling (status code checks, timeout guards, and clean error messages).
4. Output a JSON object with this exact structure:
{{
  "project_name": "{self.project_name}",
  "default_agent": "...",
  "agents": {{
    "agent_name": {{
      "description": "...",
      "handoffs": ["other_agent"],
      "tools": ["tool_function_name"],
      "system_prompt": "..."
    }}
  }},
  "tools_code": "# Python functions with docstrings, type hints, and timeout/error handling..."
}}
Output ONLY valid JSON. No markdown ticks, no preamble.
"""
        model = "gemini/gemini-2.5-flash"
        resp = await litellm.acompletion(
            model=model, messages=[{"role": "user", "content": prompt}], temperature=0.2
        )

        raw_json = resp.choices[0].message.content.strip()
        if raw_json.startswith("```"):
            lines = raw_json.split("\n")
            raw_json = "\n".join(lines[1:-1])

        data = json.loads(raw_json)

        # Scaffolding project files
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "prompts").mkdir(exist_ok=True)
        (target_dir / "tools").mkdir(exist_ok=True)
        (target_dir / "tests").mkdir(exist_ok=True)

        # Write tools/custom_tools.py
        (target_dir / "tools" / "custom_tools.py").write_text(
            data.get("tools_code", "# Tools"), encoding="utf-8"
        )

        # Generate ai.yaml
        agents_yaml_blocks = []
        for a_name, a_info in data.get("agents", {}).items():
            # Write prompt file
            prompt_file = f"prompts/{a_name}.jinja2"
            (target_dir / prompt_file).write_text(
                a_info.get("system_prompt", f"You are {a_name}."), encoding="utf-8"
            )

            tool_list = []
            for t_name in a_info.get("tools", []):
                tool_list.append(f"""      - name: "{t_name}"
        module: "tools.custom_tools"
        function: "{t_name}" """)

            handoff_str = "\n".join(
                [f'      - "{h}"' for h in a_info.get("handoffs", [])]
            )
            tool_list_str = "\n".join(tool_list)

            agents_yaml_blocks.append(f"""  {a_name}:
    description: "{a_info.get('description', a_name)}"
    system_prompt_file: "{prompt_file}"
    handoffs:
{handoff_str}
    tools:
{tool_list_str}""")

        agents_yaml_str = "\n\n".join(agents_yaml_blocks)
        ai_yaml_content = f"""name: "{self.project_name}"
version: "1.0"
default_agent: "{data.get('default_agent', list(data.get('agents', {}).keys())[0])}"

model:
  primary: "gemini/gemini-2.5-flash"
  fallback: "gemini/gemini-2.5-pro"
  temperature: 0.3
  guardrails:
    mask_pii: true

memory:
  type: "sqlite"
  db_path: ".ai/memory.db"

telemetry:
  - "otel"

agents:
{agents_yaml_str}
"""
        (target_dir / "ai.yaml").write_text(ai_yaml_content, encoding="utf-8")
        console.print(
            f"[bold green]✓ Successfully synthesized multi-agent project in '{target_dir.name}'![/bold green]"
        )
        console.print(
            f"[dim]Run 'cd {target_dir.name} && intagrin dev' to chat with your newly synthesized API swarm.[/dim]"
        )
