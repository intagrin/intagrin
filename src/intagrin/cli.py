import asyncio
import os
from datetime import UTC
from functools import cache
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from .compiler.parser import parse_project
from .errors import IntaGrinError
from .runtime.engine import RuntimeEngine
from .tracing.console import Tracer

app = typer.Typer(help="AI App Platform CLI - The Ruby on Rails for AI")
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        from importlib.metadata import version

        console.print(f"intagrin {version('intagrin')}")
        raise typer.Exit()


def _print_cli_error(e: Exception) -> None:
    """Prints an error to the console, prefixed with its IntaGrin error code when available.
    Uncoded exceptions keep today's plain formatting — codes are being migrated incrementally."""
    if isinstance(e, IntaGrinError):
        console.print(f"[bold red]Error [{e.code}]:[/bold red] {e.message}")
    else:
        console.print(f"[bold red]Error: {e}[/bold red]")


_PROVIDER_API_KEY_ENV_VARS = {
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "vertex_ai": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
}


def _check_llm_api_key(model: str) -> None:
    """Raises a clear IG-CLI-008 before an LLM call this CLI makes on its own (not through a
    project's already-validated `ai.yaml`) — e.g. `inta compile` bootstrapping ai.yaml before it
    exists. Without this, a missing key surfaces as a raw litellm traceback deep in a provider
    SDK instead of a one-line, actionable message. Local/self-hosted providers (ollama,
    llama.cpp, etc.) need no key and are silently allowed through."""
    provider = model.split("/", 1)[0] if "/" in model else None
    env_vars = _PROVIDER_API_KEY_ENV_VARS.get(provider)
    if env_vars and not any(os.environ.get(v) for v in env_vars):
        raise IntaGrinError(
            "IG-CLI-008",
            f"Model '{model}' needs one of {', '.join(env_vars)} set in the environment — "
            f"none were found. Add it to a `.env` file in this directory or export it in your "
            f"shell, then re-run the command.",
        )

def _prompt_for_llm_model(project_dir: Path | None = None) -> str:
    """Interactive provider/model picker for a CLI-driven LLM call with no `ai.yaml` yet to read
    `model.primary` from (e.g. `inta new --withagent`, `inta compile` on a brand-new blueprint).
    Shared by both so a provider added here (or a bug fixed here) doesn't need fixing twice —
    this used to be inlined separately in run_agent_wizard, missing Anthropic entirely despite
    _PROVIDER_API_KEY_ENV_VARS already knowing about it. "Custom" covers everything a fixed list
    can't enumerate — any other provider, any self-hosted/local endpoint — as a raw LiteLLM model
    string, so this never gates what a user can actually run on.

    A freshly-collected API key is set in os.environ for this process AND, when project_dir is
    given, appended to that project's .env so it's still there on the next run — appended, not
    written outright, so this can never silently wipe out other vars already in an existing .env
    (the previous inlined version in run_agent_wizard did exactly that)."""
    from rich.prompt import IntPrompt, Prompt

    console.print("\n[bold cyan]Which LLM provider?[/bold cyan]")
    console.print("  [1] OpenAI (GPT-4o)")
    console.print("  [2] Google Gemini")
    console.print("  [3] Anthropic Claude")
    console.print("  [4] Ollama (local)")
    console.print("  [5] Llama.cpp (local)")
    console.print("  [6] Other / custom (any LiteLLM model string)")

    choice = IntPrompt.ask(
        "\nChoose a provider", choices=["1", "2", "3", "4", "5", "6"], default=1
    )

    def _collect_key(prompt: str, env_var: str, model_str: str) -> str:
        key = Prompt.ask(prompt, password=True)
        os.environ[env_var] = key
        if project_dir is not None:
            with open(project_dir / ".env", "a") as f:
                f.write(f"{env_var}={key}\n")
        return model_str

    if choice == 1:
        return _collect_key("Enter your OpenAI API Key", "OPENAI_API_KEY", "openai/gpt-4o")
    if choice == 2:
        return _collect_key("Enter your Gemini API Key", "GEMINI_API_KEY", "gemini/gemini-2.5-flash")
    if choice == 3:
        return _collect_key(
            "Enter your Anthropic API Key", "ANTHROPIC_API_KEY", "anthropic/claude-sonnet-4-5"
        )
    if choice == 4:
        return Prompt.ask("Enter your Ollama model name", default="ollama/llama3")
    if choice == 5:
        return Prompt.ask("Enter your Llama.cpp model path", default="llama.cpp/llama3")
    return Prompt.ask(
        "Enter a LiteLLM model string (e.g. 'mistral/mistral-large-latest', "
        "'openai/my-self-hosted-endpoint')"
    )


# `inta copilot`'s generated agent/skill content lives as plain .md files under templates/copilot/
# instead of embedded Python string constants — easier to review/edit as markdown, and it ships
# with the package via [tool.setuptools.package-data] in pyproject.toml.
COPILOT_TEMPLATES_DIR = Path(__file__).parent / "templates" / "copilot"


@cache
def _load_copilot_template(relative_path: str) -> str:
    return (COPILOT_TEMPLATES_DIR / relative_path).read_text(encoding="utf-8")


# The JSON Schema for ai.yaml, generated from config/schema.py by scripts/generate_ai_schema.py
# (tests/test_ai_schema_freshness.py keeps it from drifting) — bundled the same way as the
# copilot templates above so `inta new` can scaffold it into every new project unchanged.
AI_SCHEMA_JSON_PATH = Path(__file__).parent / "templates" / "ai.schema.json"

AI_YAML_SCHEMA_DIRECTIVE = "# yaml-language-server: $schema=./ai.schema.json\n"


def _write_ai_yaml(project_dir: Path, content: str) -> None:
    """Every ai.yaml IntaGrin scaffolds gets the yaml-language-server schema directive as its
    first line — wires up autocomplete and inline validation in any editor with a YAML language
    server (VS Code's redhat.vscode-yaml, Neovim, etc.) with zero editor-side config. The one
    write site for all three places ai.yaml gets written (default template, --withagent's
    LLM-generated content, --withagent's fallback-on-error path) so a new one can't forget it."""
    if not content.lstrip().startswith("# yaml-language-server:"):
        content = AI_YAML_SCHEMA_DIRECTIVE + content
    (project_dir / "ai.yaml").write_text(content)

LOGO = r"""
 ___       _          ____      _       
|_ _|_ __ | |_ __ _  / ___|_ __(_)_ __  
 | || '_ \| __/ _` || |  _| '__| | '_ \ 
 | || | | | || (_| || |_| | |  | | | | |
|___|_| |_|\__\__,_| \____|_|  |_|_| |_|
"""


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the installed intagrin version and exit.",
    ),
    log_level: str = typer.Option(
        None,
        "--log-level",
        "-l",
        help="quiet | normal | debug | trace (env: INTAGRIN_LOG_LEVEL; default: normal). "
        "quiet suppresses step/tool/cost noise; debug adds full tracebacks and state dumps "
        "to every error; trace also logs every outbound LLM prompt and raw completion.",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Shorthand for --log-level debug. Also enables LiteLLM's own verbose HTTP logging.",
    ),
    json_logs: bool = typer.Option(
        False,
        "--json-logs",
        help="Emit one structured JSON object per log line instead of Rich console output — "
        "for piping to jq or a log aggregator (e.g. under `inta serve` / `inta worker`).",
    ),
):
    """IntaGrin CLI - The Ruby on Rails for AI"""
    resolved_level = log_level or os.environ.get("INTAGRIN_LOG_LEVEL")
    if debug and not resolved_level:
        resolved_level = "debug"
    if resolved_level:
        try:
            Tracer.set_level(resolved_level)
        except ValueError:
            console.print(
                f"[bold red]Invalid --log-level '{resolved_level}'. "
                "Use one of: quiet, normal, debug, trace.[/bold red]"
            )
            raise typer.Exit(1)
    if debug:
        os.environ["LITELLM_LOG"] = "DEBUG"
    if json_logs:
        Tracer.set_json_mode(True)

    if ctx.invoked_subcommand is None:
        console.print(
            Panel.fit(
                LOGO,
                title="The Ruby on Rails for AI",
                border_style="cyan",
                style="bold cyan",
            )
        )
        console.print(ctx.get_help())
    elif not json_logs:
        # Print logo before running the command (skip in --json-logs mode: it isn't JSON)
        console.print(f"[bold cyan]{LOGO}[/bold cyan]")


AI_YAML_TEMPLATE = """version: "1.0"
name: "{project_name}"
description: "Multi-agent support system"

# Validates every write_state call against schemas.AppState — catches a typo'd key or wrong type
# at the moment it's written instead of it silently sitting in state as an unvalidated string.
# Every field there is optional, so this costs nothing to keep even before you've filled it in;
# delete this line (and schemas.py) if you'd rather stay fully dynamic. See
# docs/04_Shared_State_Redux.md.
state_schema: "schemas.AppState"

model:
  primary: "anthropic/claude-3-5-sonnet-20241022"
  fallback: "openai/gpt-4o-mini"
  temperature: 0.2
  max_tokens: 1500
  guardrails:
    mask_pii: true
    system_safeguards: true
    banned_words: ["competitor_name"]

memory:
  type: "sqlite"
  db_path: ".ai/memory.db"
  max_messages: 20

telemetry: ["langfuse", "otel"]

default_agent: "triage"

agents:
  triage:
    system_prompt_file: "prompts/triage_prompt.jinja2"
    handoffs: ["support", "billing"]
    # Deterministic routers bypass the LLM entirely when their condition is already true —
    # e.g. once some earlier turn's tool call has written state via write_state():
    #   routers:
    #     - condition: "user_status == 'banned'"
    #       target: "billing"
    # Routers run *before* the LLM each turn, so the state they check has to already exist —
    # see docs/03_Agent_Handoffs_and_Routing.md for the full pattern.

  support:
    system_prompt_file: "prompts/support_prompt.jinja2"
    delegations: ["billing"]
    tools:
      - name: "get_user_account"
        module: "tools.custom_tools"
      # MCP servers plug in the same way as a local tool — e.g. GitHub (needs Node's `npx` on
      # PATH and a GITHUB_PERSONAL_ACCESS_TOKEN in .env; not required for this starter project):
      #   - name: "mcp_github"
      #     type: "mcp"
      #     command: "npx"
      #     args: ["-y", "@modelcontextprotocol/server-github"]
      #     requires_approval: true

  billing:
    system_prompt_file: "prompts/billing_prompt.jinja2"

workflows:
  daily_audit:
    - name: "check_account"
      agent: "support"
      instruction: "Look up the account for user_id '123' and report its status."
    - name: "report_billing"
      agent: "billing"
      instruction: "Summarize the findings and wait."
"""

TRIAGE_PROMPT = """You are the Triage Agent.
Your job is to understand the user's request and transfer them to the right department.
If they need account status or tech support, transfer to 'support'.
If they have a billing or payment question, transfer to 'billing'.
Always transfer as soon as you know the intent.
"""

EVALS_TEMPLATE = """version: "1.0"
evaluations:
  - name: "Test Handoff Logic"
    starting_agent: "triage"
    input: "I need to talk to support."
    expected_tool: "transfer_agent"

  - name: "Test Support Agent"
    starting_agent: "support"
    input: "Check my account 123"
    expected_tool: "get_user_account"

  - name: "RAG Faithfulness Test"
    starting_agent: "support"
    input: "Check my account 123"
    evaluators:
      - type: "llm_judge"
        criteria: "The final answer must accurately reflect the retrieved data without hallucinating."
"""

SUPPORT_PROMPT = """You are the Support Agent.
You can look up user accounts using your tools. Help the user with their technical issues.
"""

BILLING_PROMPT = """You are the Billing Agent.
You help users with their invoices and payments. 
"""

CUSTOM_TOOLS_TEMPLATE = """def get_user_account(user_id: str) -> str:
    \"\"\"
    Fetch user account details.
    
    Args:
        user_id: The ID of the user.
    \"\"\"
    return f"Account details for user {user_id}: Active, Premium Tier."
"""

SCHEMAS_TEMPLATE = '''"""Optional typed validation for state written via write_state — see ai.yaml's `state_schema`
and docs/04_Shared_State_Redux.md.

Every field is optional (defaults to None): write_state only rejects a write that gives an
EXISTING field the wrong type, it does not require every field to be set up front. Add real
fields here as your agents' write_state calls grow past what's comfortable to track by
convention alone; delete this file and ai.yaml's `state_schema` line if you'd rather stay fully
dynamic.
"""

from pydantic import BaseModel


class AppState(BaseModel):
    user_status: str | None = None
'''

# `inta new --template coding-agent` — a real, working version of docs/05_Example_Coding_Agent.md's
# plan -> code -> verify handoff loop, not just prose to copy from by hand. The other two Blueprint
# docs (SOC Analyst, Voice Agent) are deliberately NOT scaffolded here: they're prose sketches with
# no actual ai.yaml in them at all (external integrations — CrowdStrike, Cuckoo Sandbox, Twilio,
# Postgres — that would need real design decisions, not templating existing content), and writing
# one from scratch under this name would misrepresent it as a proven pattern rather than a new one.
CODING_AGENT_AI_YAML_TEMPLATE = """version: "1.0"
name: "{project_name}"
description: "Autonomous coding agent: plan -> code -> verify, self-healing on test failure"

state_schema: "schemas.AppState"

model:
  primary: "anthropic/claude-3-5-sonnet-20241022"
  fallback: "openai/gpt-4o-mini"
  temperature: 0.1
  max_tokens: 4000

memory:
  type: "sqlite"
  db_path: ".ai/memory.db"
  max_messages: 40

default_agent: "architect_agent"

agents:
  architect_agent:
    description: "Understands the codebase and formulates a plan before handing off to the coder."
    system_prompt_file: "prompts/architect_prompt.jinja2"
    handoffs: ["coder_agent"]
    tools:
      - name: "grep_search"
        module: "tools.custom_tools"
      - name: "list_directory"
        module: "tools.custom_tools"

  coder_agent:
    description: "Writes the actual code changes described by the architect's plan."
    system_prompt_file: "prompts/coder_prompt.jinja2"
    handoffs: ["verifier_agent"]
    tools:
      - name: "replace_file_content"
        module: "tools.custom_tools"

  verifier_agent:
    description: "Runs tests; hands back to the coder with the failure on a red run."
    system_prompt_file: "prompts/verifier_prompt.jinja2"
    handoffs: ["coder_agent"]  # Loop back on failure — the self-healing part of this pattern.
    tools:
      - name: "run_bash_command"
        module: "tools.custom_tools"
        # Crucial safety net (see docs/05_Example_Coding_Agent.md): this tool can run anything,
        # so every invocation pauses for human approval before it actually executes.
        requires_approval: true
"""

CODING_AGENT_ARCHITECT_PROMPT = """You are the Architect agent in a coding-agent swarm.

Understand the user's request by using `grep_search` and `list_directory` to explore the codebase
before proposing anything — never guess at code you haven't looked at. Once you have a concrete,
specific plan (which files change and how), transfer to `coder_agent` with clear, unambiguous
instructions for the change to make. Do not write code yourself.
"""

CODING_AGENT_CODER_PROMPT = """You are the Coder agent in a coding-agent swarm.

You receive a specific plan from the architect. Use `replace_file_content` to make the described
change — it overwrites a file's full contents, so always write the complete new file, not a diff
or a partial snippet. Once you've made the change, transfer to `verifier_agent` to confirm it
actually works. If `verifier_agent` hands a failing test back to you, read the traceback carefully
and fix the real bug it points to, then transfer back to `verifier_agent` again.
"""

CODING_AGENT_VERIFIER_PROMPT = """You are the Verifier agent in a coding-agent swarm.

Run the project's test suite with `run_bash_command` (e.g. `pytest`). If every test passes, tell
the user the change is complete and summarize what changed. If any test fails, transfer back to
`coder_agent` with the full failure output included in your handoff reason — it needs the actual
traceback, not a paraphrase, to fix its own mistake.
"""

CODING_AGENT_TOOLS_TEMPLATE = '''import subprocess


def grep_search(pattern: str, path: str = ".") -> str:
    """Search Python files under a directory for a regex pattern.

    Args:
        pattern: Regular expression to search for.
        path: Directory to search under, relative to the project root.
    """
    try:
        result = subprocess.run(
            ["grep", "-rn", "--include=*.py", pattern, path],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return result.stdout.strip() or "No matches found."
    except Exception as e:
        return f"Search failed: {e}"


def list_directory(path: str = ".") -> str:
    """List the files and subdirectories at a given path.

    Args:
        path: Directory to list, relative to the project root.
    """
    import os

    try:
        entries = sorted(os.listdir(path))
        return "\\n".join(entries) if entries else "(empty directory)"
    except Exception as e:
        return f"Could not list directory: {e}"


def replace_file_content(file_path: str, new_content: str) -> str:
    """Overwrite a file with new content — the coder agent's only way to make an edit.

    Args:
        file_path: Path to the file to write, relative to the project root.
        new_content: The complete new content for the file (not a diff).
    """
    try:
        with open(file_path, "w") as f:
            f.write(new_content)
        return f"Wrote {len(new_content)} characters to {file_path}."
    except Exception as e:
        return f"Write failed: {e}"


def run_bash_command(command: str) -> str:
    """Run a shell command (e.g. a test suite) and return its combined output.

    Gated by `requires_approval: true` in ai.yaml — every call pauses for human approval before
    it actually executes, since this can run anything. Do not remove that gate.

    Args:
        command: The shell command to run.
    """
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=60  # nosec B602
        )
        output = (result.stdout or "") + (result.stderr or "")
        return output.strip() or f"Command exited with code {result.returncode}, no output."
    except Exception as e:
        return f"Command failed: {e}"
'''

SYSTEM_PROMPT_TEMPLATE = """You are a helpful AI support agent.
Answer user queries clearly and concisely.
Use tools when necessary to look up real-time information.
"""

ENV_EXAMPLE_TEMPLATE = """OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-api03-...

# Only needed because this starter project's ai.yaml has `telemetry: ["langfuse", "otel"]` —
# remove that line (or these vars) if you don't want Langfuse tracing. otel needs no extra
# config here; it exports to whatever OTEL_EXPORTER_* endpoint you've already got configured.
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
"""

@app.command(name="init")
def init(
    project_name: str,
    withagent: bool = typer.Option(
        False, "--withagent", help="Use an AI assistant to generate your project"
    ),
):
    """
    Initialize a new IntaGrin project (Alias for new).
    """
    new(project_name, withagent)


TEMPLATE_CHOICES = ["default", "coding-agent"]


def _scaffold_coding_agent_template(base_dir: Path, project_name: str) -> None:
    """Writes `inta new --template coding-agent` — a real, working version of the plan -> code ->
    verify handoff loop from docs/05_Example_Coding_Agent.md, not just prose to copy from by hand.
    Runs the same `validate_config_dict` check the wizard's LLM-generated output goes through
    (see `_generate_and_validate_wizard_config`) as a defensive check against this hand-authored
    template itself ever drifting out of schema — cheap, and `tests/test_cli.py` also asserts this
    passes so a future edit can't silently break it."""
    import yaml

    from .config.schema import validate_config_dict

    ai_yaml_text = CODING_AGENT_AI_YAML_TEMPLATE.format(project_name=project_name)
    config, errors = validate_config_dict(yaml.safe_load(ai_yaml_text))
    if config is None or errors:
        raise IntaGrinError(
            "IG-CLI-009",
            f"The coding-agent template failed schema validation (this is a bug in IntaGrin "
            f"itself, not your project): {errors}",
        )

    _write_ai_yaml(base_dir, ai_yaml_text)
    (base_dir / "schemas.py").write_text(SCHEMAS_TEMPLATE)
    (base_dir / "tools" / "custom_tools.py").write_text(CODING_AGENT_TOOLS_TEMPLATE)
    (base_dir / "prompts" / "architect_prompt.jinja2").write_text(CODING_AGENT_ARCHITECT_PROMPT)
    (base_dir / "prompts" / "coder_prompt.jinja2").write_text(CODING_AGENT_CODER_PROMPT)
    (base_dir / "prompts" / "verifier_prompt.jinja2").write_text(CODING_AGENT_VERIFIER_PROMPT)
    (base_dir / ".env.example").write_text(ENV_EXAMPLE_TEMPLATE)


@app.command()
def new(
    project_name: str,
    withagent: bool = typer.Option(
        False, "--withagent", help="Use an AI assistant to generate your project"
    ),
    template: str = typer.Option(
        "default",
        "--template",
        "-t",
        help=f"Starting point to scaffold: one of {TEMPLATE_CHOICES}. Ignored if --withagent is set.",
    ),
):
    """
    Scaffold a new IntaGrin project.
    """
    if template not in TEMPLATE_CHOICES:
        _print_cli_error(
            IntaGrinError(
                "IG-CLI-002", f"--template must be one of {TEMPLATE_CHOICES}, got '{template}'"
            )
        )
        raise typer.Exit(1)

    base_dir = Path(project_name)
    if base_dir.exists():
        _print_cli_error(
            IntaGrinError("IG-CLI-001", f"Directory '{project_name}' already exists.")
        )
        raise typer.Exit(1)

    base_dir.mkdir()
    (base_dir / "tools").mkdir()
    (base_dir / "prompts").mkdir()
    (base_dir / "tests").mkdir()
    (base_dir / "tests" / "evals.yaml").write_text(EVALS_TEMPLATE)
    implement_skill_body = _load_copilot_template("implement_skill_body.md")
    (base_dir / ".cursorrules").write_text(implement_skill_body)
    (base_dir / "AGENT.md").write_text(implement_skill_body)
    (base_dir / "tools" / "__init__.py").touch()
    (base_dir / "ai.schema.json").write_text(
        AI_SCHEMA_JSON_PATH.read_text(encoding="utf-8"), encoding="utf-8"
    )

    if withagent:
        run_agent_wizard(base_dir, project_name)
    elif template == "coding-agent":
        _scaffold_coding_agent_template(base_dir, project_name)
    else:
        _write_ai_yaml(base_dir, AI_YAML_TEMPLATE.format(project_name=project_name))
        (base_dir / "schemas.py").write_text(SCHEMAS_TEMPLATE)
        (base_dir / "tools" / "custom_tools.py").write_text(CUSTOM_TOOLS_TEMPLATE)
        (base_dir / "prompts" / "triage_prompt.jinja2").write_text(TRIAGE_PROMPT)
        (base_dir / "prompts" / "support_prompt.jinja2").write_text(SUPPORT_PROMPT)
        (base_dir / "prompts" / "billing_prompt.jinja2").write_text(BILLING_PROMPT)
        (base_dir / ".env.example").write_text(ENV_EXAMPLE_TEMPLATE)

    console.print(
        Panel.fit(
            f"[bold green]Successfully created project: {project_name}[/bold green]\n\n"
            f"Next steps:\n"
            f"1. cd {project_name}\n"
            f"2. cp .env.example .env\n"
            f"3. Fill in your API keys in .env\n"
            f"4. run `inta dev`\n\n"
            f"[dim]Not sure whether to reach for handoffs, delegations, routers, auto_route, "
            f"spawns, or workflows? Run `inta monitor` and ask the IntaGrin Architect chat — it "
            f"already knows the full decision guide, including which one fits your specific "
            f"case.[/dim]",
            title="AI App Platform",
        )
    )


def _generate_and_validate_wizard_config(
    model: str, sys_prompt: str, idea: str, max_retries: int = 2
) -> tuple[str, str] | None:
    """Calls the wizard's LLM, then validates the generated `ai_yaml` against the real AppConfig
    schema via `validate_config_dict` — the same check every other LLM-generated config in this
    codebase (Studio's graph-sync, `inta compile`) is run through before being trusted. Without
    this, `run_agent_wizard` used to write whatever JSON the model returned straight to disk the
    moment it parsed as JSON at all — a schema violation (an unsupported router condition, a typo'd
    field) would only surface later, as a confusing `inta dev` crash, not here where it's cheap to
    catch and fix.

    Self-heals up to `max_retries` times by feeding the corrector model the exact validation
    errors and the previous attempt, mirroring the malformed-tool-argument corrector pattern in
    `runtime/engine.py`. Returns (ai_yaml_text, custom_tools_py_text) on success, or None if it
    still doesn't validate after every retry — the caller falls back to the safe default template
    rather than ever writing unchecked LLM output to disk.
    """
    import json

    import litellm
    import yaml

    from .config.schema import validate_config_dict

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": idea},
    ]

    for attempt in range(max_retries + 1):
        response = litellm.completion(model=model, messages=messages)
        content = response.choices[0].message.content
        if content.startswith("```json"):
            content = content.replace("```json\n", "").replace("```", "")
        if content.startswith("```"):
            content = content.replace("```\n", "").replace("```", "")

        errors: list[str] = []
        ai_yaml_text = ""
        custom_tools_py = "# Write tools here"
        try:
            data = json.loads(content)
            ai_yaml_text = data.get("ai_yaml", "")
            custom_tools_py = data.get("custom_tools_py", custom_tools_py)
            parsed = yaml.safe_load(ai_yaml_text)
            if not isinstance(parsed, dict):
                raise ValueError("ai_yaml did not parse to a YAML mapping")
            _config, errors = validate_config_dict(parsed)
        except Exception as e:
            errors = [f"Could not parse generated output: {e}"]

        if not errors:
            return ai_yaml_text, custom_tools_py

        console.print(
            f"[dim]Generated config failed validation (attempt {attempt + 1}/"
            f"{max_retries + 1}), asking the model to fix it...[/dim]"
        )
        if attempt == max_retries:
            break

        correction_prompt = (
            "The ai.yaml you generated failed validation against IntaGrin's real schema:\n"
            + "\n".join(errors)
            + f"\n\nHere is what you generated:\n{content}\n\n"
            'Fix it and return the SAME JSON shape as before (keys "ai_yaml" and '
            '"custom_tools_py"), with the corrected ai.yaml. Output ONLY valid JSON, no '
            "markdown, no explanation."
        )
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": idea},
            {"role": "assistant", "content": content},
            {"role": "user", "content": correction_prompt},
        ]

    return None


def run_agent_wizard(project_dir: Path, project_name: str):
    from rich.prompt import Prompt

    console.print(
        "\n[bold purple]🤖 Welcome to the AI Architect Wizard![/bold purple]\n"
    )

    model = _prompt_for_llm_model(project_dir)

    idea = Prompt.ask(
        "\n[bold blue]What kind of AI application do you want to build?[/bold blue]"
    )

    console.print(
        "[bold yellow]Thinking... (Generating ai.yaml and tools)[/bold yellow]"
    )

    sys_prompt = """You are an expert architect for the `IntaGrin` framework.
The framework uses a declarative `ai.yaml` file for orchestration and standard python files for tools.
Agents can use `handoffs: ["other_agent"]` to transfer control, `delegations: ["sub_agent"]` to spawn a child task, `routers: [{condition: "...", target: "..."}]` to route deterministically, `auto_route: true` for semantic swarm routing, or `spawns:` to let an agent dynamically create narrowly-scoped sub-agents at runtime. Workflows (`workflows:`) support `type: "sequential"` (default), `"parallel"`, and `"vote"` (fan-out-and-consensus). Mark risky tools `requires_approval: true` (or `required_approvers: [...]` for multi-person sign-off). Prefer a real `memory.type` (`sqlite`/`postgres`/`redis`) over the default in-process one for anything beyond a toy demo, and consider `model.guardrails` (`mask_pii`, `banned_words`) for user-facing agents.
Return EXACTLY a valid JSON object with two keys: "ai_yaml" (string of yaml content) and "custom_tools_py" (string of python code). Do not include any markdown outside the JSON."""

    try:
        generated = _generate_and_validate_wizard_config(model, sys_prompt, idea)

        if generated is None:
            console.print(
                "[bold red]The generated config didn't pass schema validation after "
                "retrying — falling back to the default template so you get a working "
                "project instead of a broken one.[/bold red]"
            )
            _write_ai_yaml(project_dir, AI_YAML_TEMPLATE.format(project_name=project_name))
            (project_dir / "schemas.py").write_text(SCHEMAS_TEMPLATE)
            (project_dir / "tools" / "custom_tools.py").write_text(CUSTOM_TOOLS_TEMPLATE)
        else:
            ai_yaml_text, custom_tools_py = generated
            _write_ai_yaml(project_dir, ai_yaml_text)
            (project_dir / "tools" / "custom_tools.py").write_text(custom_tools_py)
            console.print("[bold green]Project generated successfully via AI![/bold green]")

    except Exception as e:
        console.print(
            f"[bold red]Failed to generate via agent. Falling back to default template. Error: {e}[/bold red]"
        )
        _write_ai_yaml(project_dir, AI_YAML_TEMPLATE.format(project_name=project_name))
        (project_dir / "schemas.py").write_text(SCHEMAS_TEMPLATE)
        (project_dir / "tools" / "custom_tools.py").write_text(CUSTOM_TOOLS_TEMPLATE)


@app.command()
def dev():
    """
    Launch the local developer environment. Use the global --debug / --log-level flag
    (`inta --debug dev`) for verbose tracebacks and LiteLLM request/response logging.
    """
    project_dir = Path.cwd()
    try:
        graph = parse_project(project_dir)
        console.print(
            f"[bold green]Starting AI dev server for '{graph.config.name}'...[/bold green]"
        )

        async def run_dev():
            engine = RuntimeEngine(graph=graph, project_dir=project_dir)
            await engine.initialize()
            await engine.chat_loop()

        asyncio.run(run_dev())
    except Exception as e:
        Tracer.log_error(f"Dev Environment Error: {e}")
        raise typer.Exit(1)


@app.command()
def run(
    workflow: str = typer.Argument(..., help="Name of the workflow to run"),
):
    """
    Run an autonomous workflow from ai.yaml. Use the global --debug / --log-level flag
    (`inta --debug run <workflow>`) for verbose tracebacks and LiteLLM request/response logging.
    """
    project_dir = Path.cwd()
    try:
        graph = parse_project(project_dir)
    except Exception as e:
        Tracer.log_error(f"Configuration error: {e}")
        raise typer.Exit(1)
    if workflow not in graph.config.workflows:
        console.print(f"[bold red]Workflow '{workflow}' not found in ai.yaml[/bold red]")
        raise typer.Exit(1)

    console.print(f"[bold green]Starting Workflow: '{workflow}'...[/bold green]")
    try:

        async def run_wf():
            engine = RuntimeEngine(graph=graph, project_dir=project_dir)
            await engine.initialize()
            await engine.run_workflow(workflow)
            await engine.mcp_manager.cleanup()

        asyncio.run(run_wf())
    except Exception as e:
        Tracer.log_error(f"Workflow Error: {e}")
        raise typer.Exit(1)


@app.command(name="worker")
def worker_command(
    queue: str = typer.Option(
        "intagrin:tasks", "--queue", "-q", help="Name of the task queue"
    ),
    redis_url: str = typer.Option(
        None,
        "--redis-url",
        "-r",
        help="Redis URL for distributed queue (e.g. redis://localhost:6379/0)",
    ),
):
    """
    Start a distributed background worker to process async workflows and tasks.
    """
    project_dir = Path.cwd()
    import os

    resolved_redis = redis_url or os.environ.get("REDIS_URL")

    from .runtime.worker import DistributedWorker

    worker = DistributedWorker(
        project_dir=project_dir, queue_name=queue, redis_url=resolved_redis
    )
    try:
        asyncio.run(worker.start())
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Worker stopped by operator.[/bold yellow]")


@app.command(name="tune")
def tune_command(
    iterations: int = typer.Option(
        3, "--iterations", "-i", help="Maximum self-healing reflection iterations"
    )
):
    """
    Run the Self-Healing Auto-Tuner: iteratively repairs failing prompts in tests/evals.yaml.
    """
    project_dir = Path.cwd()
    from .testing.tuner import AutoTuner

    tuner = AutoTuner(project_dir=project_dir, max_iterations=iterations)
    asyncio.run(tuner.tune())


@app.command(name="export")
def export_command(
    output: str = typer.Option(
        "standalone_app.py",
        "--output",
        "-o",
        help="Target standalone Python output file",
    )
):
    """
    Export project to a standalone, zero-dependency FastAPI application with zero framework lock-in.
    """
    project_dir = Path.cwd()
    from .compiler.exporter import CodeExporter

    exporter = CodeExporter(project_dir=project_dir, output_file=output)
    exporter.export_fastapi()


@app.command(name="import")
def import_command(
    spec: str = typer.Argument(
        ..., help="Path to local OpenAPI JSON or URL (e.g. swagger.json)"
    ),
    name: str = typer.Option(
        "api-swarm", "--name", "-n", help="Name of the generated project"
    ),
):
    """
    Import any Swagger / OpenAPI spec and synthesize into a complete multi-agent swarm in seconds.
    """
    target_dir = Path.cwd() / name
    from .compiler.reverse import OpenAPISynthesizer

    synth = OpenAPISynthesizer(spec_source=spec, project_name=name)
    asyncio.run(synth.synthesize(target_dir))


@app.command(name="fuzz")
def fuzz_command(
    attacks: int = typer.Option(
        10, "--attacks", "-a", help="Number of adversarial attacks to simulate"
    )
):
    """
    Run the Adversarial Red-Team Fuzzer: stress-tests agents against injections, PII leaks, and boundary overflows.
    """
    project_dir = Path.cwd()
    from .testing.fuzzer import AdversarialFuzzer

    fuzzer = AdversarialFuzzer(project_dir=project_dir, num_attacks=attacks)
    asyncio.run(fuzzer.fuzz())


@app.command(name="verify")
def verify_command():
    """
    Run static control-flow verification: cycle detection across handoffs and deterministic
    routers, delegation depth bounds, and a worst-case cost accounting that explicitly reports
    which cost/routing paths are bounded and which (self-healing, memory compression, parallel
    tool fan-out, semantic auto_route) are not statically verifiable.
    """
    project_dir = Path.cwd()
    from .compiler.verifier import GraphVerifier

    verifier = GraphVerifier(project_dir=project_dir)
    verifier.verify()


@app.command(name="diagnose")
def diagnose_command(
    prompt: str = typer.Option(
        None,
        "--prompt",
        "-p",
        help="Run a single ad-hoc diagnostic probe with this text instead of tests/evals.yaml",
    )
):
    """
    Run agent diagnostics: executes tests/evals.yaml (or a small probe battery if none exists)
    and reports real, computed health metrics — context utilization, tool error rate, cost, and
    latency — with a rules-based explanation of whichever metric crossed its threshold.
    """
    project_dir = Path.cwd()
    from .testing.icu import AgentICUDiagnostics

    icu = AgentICUDiagnostics(project_dir=project_dir)
    asyncio.run(icu.run_diagnostics(custom_probe=prompt))


@app.command(name="synth")
def synth_command(
    count: int = typer.Option(
        15, "--count", "-c", help="Number of synthetic test cases to generate"
    )
):
    """
    Synthesize high-coverage edge cases and evaluations from tool signatures into tests/evals.yaml automatically.
    """
    project_dir = Path.cwd()
    from .testing.synthesizer import SyntheticEvalSynthesizer

    synth = SyntheticEvalSynthesizer(project_dir=project_dir, count=count)
    synth.evolve()


@app.command(name="eval")
def run_evals_command(
    judge: bool = typer.Option(
        False,
        "--judge",
        help="Use an LLM-as-a-judge to evaluate a complex swarm trajectory",
    ),
    session_id: str = typer.Option(
        None,
        "--session-id",
        help="The specific session ID to evaluate when using --judge",
    ),
):
    """
    Run automated agent evaluations against the datasets in tests/evals.yaml.
    """
    project_dir = Path.cwd()
    if judge:
        if not session_id:
            _print_cli_error(
                IntaGrinError("IG-CLI-007", "--session-id is required when using --judge")
            )
            raise typer.Exit(1)

        console.print(
            f"[bold cyan]Running LLM-as-a-Judge Evaluation on Session: {session_id}...[/bold cyan]"
        )

        try:
            from .compiler.parser import parse_project
            from .runtime.memory import SQLiteCheckpointer

            graph = parse_project(project_dir)
            if graph.config.memory.type == "sqlite":
                db_path = graph.config.memory.db_path
                if db_path:
                    checkpointer = SQLiteCheckpointer(
                        str((project_dir / db_path).resolve())
                    )
                    messages, state = checkpointer.load_checkpoint(session_id)

                    if not messages:
                        console.print(
                            f"[bold red]No trace found for session {session_id}.[/bold red]"
                        )
                        raise typer.Exit(1)

                    import json

                    import litellm

                    trace_text = json.dumps(messages, indent=2)
                    sys_prompt = "You are an expert AI-as-a-judge. Evaluate the following multi-agent swarm trajectory. Did they achieve the goal efficiently without loops or redundant tool calls? Grade from 1-10 and explain."

                    import asyncio

                    async def run_judge():
                        resp = await litellm.acompletion(
                            model=graph.config.model.primary,
                            messages=[
                                {"role": "system", "content": sys_prompt},
                                {
                                    "role": "user",
                                    "content": f"TRAJECTORY:\n{trace_text}",
                                },
                            ],
                        )
                        console.print(
                            Panel(
                                resp.choices[0].message.content,
                                title="[bold purple]LLM Judge Evaluation[/bold purple]",
                            )
                        )

                    asyncio.run(run_judge())
                    return
            console.print(
                "[bold red]Only SQLite checkpointers are currently supported for --judge offline evaluation.[/bold red]"
            )
        except Exception as e:
            Tracer.log_error(f"Judge Evaluation Error: {e}")
            raise typer.Exit(1)
    else:
        try:
            from .testing.evaluator import run_evals

            console.print("[bold green]Starting Agent Evaluations...[/bold green]")
            import asyncio

            asyncio.run(run_evals(project_dir))
        except Exception as e:
            Tracer.log_error(f"Evaluation Error: {e}")
            raise typer.Exit(1)


@app.command(name="deploy")
def deploy_command():
    """
    Generate a zero-config production Dockerfile and docker-compose.yml for your swarm.
    """
    project_dir = Path.cwd()
    if not (project_dir / "ai.yaml").exists():
        _print_cli_error(
            IntaGrinError("IG-CFG-001", "No ai.yaml found in the current directory.")
        )
        raise typer.Exit(1)

    dockerfile_content = """FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

# Create a non-root user for security
RUN useradd -m -s /bin/bash appuser

WORKDIR /app

# Install system dependencies if any required by tools (e.g. build-essential, git)
RUN apt-get update && apt-get install -y --no-install-recommends gcc curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir intagrin

COPY . .

# Ensure the non-root user owns the app directory
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["inta", "serve", "--port", "8000"]
"""

    dockerignore_content = """
.env
.venv
venv/
__pycache__/
.ai/
*.sqlite3
"""

    docker_compose_content = """version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    volumes:
      - ./.ai:/app/.ai
    restart: unless-stopped
"""

    with open(project_dir / "Dockerfile", "w") as f:
        f.write(dockerfile_content)

    with open(project_dir / ".dockerignore", "w") as f:
        f.write(dockerignore_content)

    with open(project_dir / "docker-compose.yml", "w") as f:
        f.write(docker_compose_content)

    # Check for requirements.txt
    if not (project_dir / "requirements.txt").exists():
        with open(project_dir / "requirements.txt", "w") as f:
            f.write("# Add your project-specific Python dependencies here\n")

    console.print(
        Panel.fit(
            "[bold green]Production Deployment Files Generated![/bold green]\n\n"
            "Created:\n"
            "- [cyan]Dockerfile[/cyan]\n"
            "- [cyan].dockerignore[/cyan]\n"
            "- [cyan]docker-compose.yml[/cyan]\n\n"
            "[bold yellow]Next steps:[/bold yellow]\n"
            "1. Add any custom python packages to [cyan]requirements.txt[/cyan]\n"
            "2. Run [bold]docker compose up -d --build[/bold] to deploy your agent swarm locally\n"
            "3. Or deploy the Dockerfile to AWS, GCP Cloud Run, or Railway.",
            title="Zero-Config Deploy",
        )
    )


@app.command(name="serve")
def serve_command(
    port: int = typer.Option(8000, "--port", "-p", help="Port to run the API server on")
):
    """
    Start the production FastAPI server.
    """
    try:
        import uvicorn

        from .server.api import app as fastapi_app

        console.print(
            f"[bold green]Starting Production API Server on port {port}...[/bold green]"
        )
        uvicorn.run(fastapi_app, host="0.0.0.0", port=port)  # nosec B104
    except ImportError:
        _print_cli_error(
            IntaGrinError(
                "IG-CLI-006",
                "Missing dependencies. Please install fastapi and uvicorn: pip install fastapi uvicorn",
            )
        )
        raise typer.Exit(1)


@app.command(name="monitor")
def monitor_command(
    port: int = typer.Option(
        3000, "--port", "-p", help="Port to run the visual dashboard on"
    )
):
    """
    Launch the visual Web Dashboard to monitor agents and memory in real-time.
    """
    from pathlib import Path

    if not (Path.cwd() / "ai.yaml").exists():
        _print_cli_error(
            IntaGrinError("IG-CFG-001", "No [yellow]ai.yaml[/yellow] found in the current directory.")
        )
        console.print(
            "Please navigate to your project directory (where ai.yaml is located) before running [bold cyan]intagrin monitor[/bold cyan]."
        )
        raise typer.Exit(1)

    try:
        import signal

        import uvicorn

        from .server.monitor import app as monitor_app

        url = f"http://localhost:{port}"
        console.print(
            "\n[bold purple]🚀 Launching AI Monitor Dashboard...[/bold purple]"
        )
        console.print(f"[bold cyan]Dashboard available at: {url}[/bold cyan]")
        console.print(
            "[dim]The dashboard will automatically sync with your configured memory store every 3 seconds.[/dim]\n"
        )
        console.print("[dim]Press Ctrl+C to stop the monitor server.[/dim]\n")

        config = uvicorn.Config(
            monitor_app,
            host="0.0.0.0",
            port=port,
            log_level="error",
            timeout_graceful_shutdown=1,
        )
        server = uvicorn.Server(config)

        def handle_exit(sig, frame):
            server.should_exit = True

        signal.signal(signal.SIGINT, handle_exit)
        signal.signal(signal.SIGTERM, handle_exit)

        try:
            server.run()
        except (KeyboardInterrupt, SystemExit):
            server.should_exit = True
        finally:
            console.print("\n[bold yellow]✓ Monitor dashboard stopped and port released.[/bold yellow]")
    except ImportError:
        _print_cli_error(
            IntaGrinError(
                "IG-CLI-006",
                "Missing dependencies. Please install fastapi and uvicorn: pip install fastapi uvicorn",
            )
        )
        raise typer.Exit(1)


@app.command(name="architect")
def architect_command():
    """
    Iteratively modify your project using an AI Assistant.
    """
    project_dir = Path.cwd()
    if not (project_dir / "ai.yaml").exists():
        _print_cli_error(
            IntaGrinError("IG-CFG-001", "No ai.yaml found in current directory.")
        )
        raise typer.Exit(1)

    import json

    import litellm
    from rich.prompt import Prompt

    console.print("\n[bold purple]🤖 Welcome to the AI Architect Editor![/bold purple]")
    console.print(
        "[dim]Describe what you want to add/change (e.g., 'Add a billing agent with a tool to fetch invoices'). Type 'exit' to quit.[/dim]\n"
    )

    # Load environment variables from .env if present
    env_file = project_dir / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'").strip('"')
                if k and not os.environ.get(k):
                    os.environ[k] = v

    from .compiler.parser import parse_project

    try:
        graph = parse_project(project_dir)
        architect_model = graph.config.model.primary
    except Exception:
        architect_model = "gemini/gemini-2.5-flash"

    while True:
        instruction = Prompt.ask("[bold blue]Instruction[/bold blue]")
        if instruction.lower() in ["exit", "quit"]:
            break

        current_yaml = (project_dir / "ai.yaml").read_text()
        tools_path = project_dir / "tools" / "custom_tools.py"
        current_tools = tools_path.read_text() if tools_path.exists() else ""

        # Read all existing prompt files for context
        prompts_context = ""
        prompts_dir = project_dir / "prompts"
        if prompts_dir.exists():
            for p in prompts_dir.glob("*.jinja2"):
                prompts_context += (
                    f"--- {p.relative_to(project_dir)} ---\n{p.read_text()}\n\n"
                )

        sys_prompt = """You are an expert architect for the `IntaGrin` framework.
You are given the current `ai.yaml`, `tools/custom_tools.py`, and existing prompt files.
Apply the user's requested modifications. If the user asks for a new agent, you MUST create a new prompt file for it.
Available orchestration primitives in `ai.yaml`: `handoffs`/`delegations`/`routers`/`auto_route` for control flow, `spawns:` for dynamic runtime agent creation, `workflows:` (`sequential`/`parallel`/`vote`) for autonomous pipelines, `requires_approval`/`required_approvers` for human-in-the-loop, `memory.type` (`sqlite`/`postgres`/`redis`) for persistence, and `model.guardrails` for content safety. Use whichever fits the user's request instead of defaulting to the simplest option.
You MUST output a valid JSON object with a "files" array.
Each item must have "path" (e.g. "ai.yaml", "tools/custom_tools.py", "prompts/billing.jinja2") and "content" (the full updated or new file string)."""

        user_content = f"CURRENT AI.YAML:\n{current_yaml}\n\nCURRENT CUSTOM_TOOLS.PY:\n{current_tools}\n\nEXISTING PROMPTS:\n{prompts_context}\n\nUSER REQUEST: {instruction}"

        console.print(
            f"[bold yellow]Architecting using {architect_model}...[/bold yellow]"
        )

        try:
            response = litellm.completion(
                model=architect_model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_content},
                ],
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            data = json.loads(content)

            files = data.get("files", [])
            for f in files:
                filepath = project_dir / f["path"]
                filepath.parent.mkdir(parents=True, exist_ok=True)
                filepath.write_text(f["content"])
                console.print(f"[green]✓ Wrote {f['path']}[/green]")

            console.print(
                "[bold green]Architecture successfully updated![/bold green]\n"
            )

        except Exception as e:
            console.print(
                f"[bold red]Failed to architect files. Error: {e}[/bold red]\n"
            )


def _ensure_local_tool_stub(project_dir: Path, tool_cfg) -> list[Path]:
    """If tool_cfg's module+function already resolves, does nothing. Otherwise creates any
    missing `__init__.py` package markers plus a minimal stub function (never overwrites an
    existing file/function — only fills in what's missing) so a compiled config never references
    a tool that silently doesn't exist until `initialize()` discovers it."""
    import sys

    from .runtime.tools_loader import load_local_tool

    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))

    created: list[Path] = []
    parts = tool_cfg.module.split(".")
    pkg_dir = project_dir
    for part in parts[:-1]:
        pkg_dir = pkg_dir / part
        pkg_dir.mkdir(parents=True, exist_ok=True)
        init_file = pkg_dir / "__init__.py"
        if not init_file.exists():
            init_file.touch()
            created.append(init_file)

    try:
        load_local_tool(tool_cfg.module, tool_cfg.name)
        return created
    except Exception:
        pass

    module_path = project_dir / (tool_cfg.module.replace(".", "/") + ".py")
    stub = (
        f'def {tool_cfg.name}(*args, **kwargs) -> str:\n'
        f'    """TODO: implement {tool_cfg.name} (scaffolded by `inta compile`)."""\n'
        f'    raise NotImplementedError("{tool_cfg.name} is not implemented yet")\n'
    )
    module_path.parent.mkdir(parents=True, exist_ok=True)
    if module_path.exists():
        module_path.write_text(module_path.read_text().rstrip("\n") + "\n\n\n" + stub)
    else:
        module_path.write_text(stub)
    created.append(module_path)
    return created


def _scaffold_referenced_files(project_dir: Path, config) -> list[Path]:
    """Creates minimal placeholder prompt files and stub tool functions for anything the
    compiled config references that doesn't exist yet. Never touches a file that already exists
    — a re-compile against manually-edited prompts/tools leaves them alone."""
    from .config.schema import LocalToolConfig

    created: list[Path] = []

    for tool_cfg in getattr(config, "tools", []) or []:
        if isinstance(tool_cfg, LocalToolConfig):
            created.extend(_ensure_local_tool_stub(project_dir, tool_cfg))

    for agent_name, agent_cfg in config.agents.items():
        prompt_file = getattr(agent_cfg, "system_prompt_file", None)
        if prompt_file:
            prompt_path = project_dir / prompt_file
            if not prompt_path.exists():
                prompt_path.parent.mkdir(parents=True, exist_ok=True)
                description = getattr(agent_cfg, "description", None) or f"the {agent_name} agent"
                prompt_path.write_text(f"You are {description}.\n")
                created.append(prompt_path)

        for tool_cfg in getattr(agent_cfg, "tools", []) or []:
            if isinstance(tool_cfg, LocalToolConfig):
                created.extend(_ensure_local_tool_stub(project_dir, tool_cfg))

    return created


@app.command(name="compile")
def compile_command(
    blueprint_file: str = typer.Argument(
        "blueprint.md", help="Path to the Markdown blueprint file"
    ),
    model: str = typer.Option(
        None,
        "--model",
        "-m",
        help=(
            "LiteLLM model string to compile with (e.g. 'openai/gpt-4o'). Skips the model "
            "resolution below entirely — set this for scripted/non-interactive use."
        ),
    ),
):
    """
    Compile a Natural Language Markdown blueprint into a production ai.yaml swarm.
    Performs bidirectional diff-merging to preserve existing manual configurations. Validates the
    compiled config against the real AppConfig schema and router-condition grammar before writing
    anything, self-healing up to 2 times if it doesn't validate — a config that still doesn't
    validate is never written, so a bad compile fails loudly instead of shipping something broken.
    """
    import json

    import litellm
    from dotenv import load_dotenv
    from rich.prompt import Confirm

    from .config import orchestration_guide
    from .config.schema import AppConfig

    project_dir = Path.cwd()
    bp_path = project_dir / blueprint_file

    if not bp_path.exists():
        _print_cli_error(
            IntaGrinError("IG-CLI-004", f"Blueprint file '{blueprint_file}' not found.")
        )
        raise typer.Exit(1)

    # parse_project() (the usual place .env gets loaded) only runs below once we know ai.yaml
    # exists — load it directly here too so a brand-new project's .env (an API key the model
    # resolution below might need) is picked up regardless of which branch runs.
    load_dotenv(project_dir / ".env")

    yaml_path = project_dir / "ai.yaml"
    existing_yaml = yaml_path.read_text() if yaml_path.exists() else ""

    # Model resolution, in order: an explicit --model always wins; otherwise reuse an already-
    # configured project's own model.primary (re-compiling a blueprint shouldn't silently switch
    # providers on you); otherwise — nothing to infer from yet, this is a first-ever compile —
    # ask interactively rather than forcing one specific provider on someone who may not even
    # have that key. Mirrors server/monitor.py's run_architect, which resolves the same way for
    # the same reason.
    if model:
        compile_model = model
    elif existing_yaml:
        try:
            compile_model = parse_project(project_dir).config.model.primary
        except Exception:
            compile_model = "gemini/gemini-2.5-flash"
    else:
        compile_model = _prompt_for_llm_model(project_dir)

    try:
        _check_llm_api_key(compile_model)
    except IntaGrinError as e:
        _print_cli_error(e)
        raise typer.Exit(1)

    blueprint_content = bp_path.read_text()

    console.print(f"[bold cyan]Compiling {blueprint_file} into ai.yaml...[/bold cyan]")

    sys_prompt = f"""You are the IntaGrin Architecture Compiler.
Your job is to read a Markdown Blueprint and compile it into a valid IntaGrin `ai.yaml` file.
If an existing `ai.yaml` is provided, you MUST preserve all existing configurations, environment variables, API keys, and custom tool attributes that are not explicitly removed by the blueprint. Perform a bidirectional merge.

{orchestration_guide.GUIDE}

Use the decision table and confusions section above to pick the right primitive for each
relationship the blueprint describes between agents — do not default to `handoffs` for
everything just because it's the most familiar one.

Router conditions (agents.<name>.routers[].condition) are evaluated by a restricted grammar, not Python's eval() — they may ONLY reference state keys as bare names, literals, comparisons (<, <=, >, >=, ==, !=, in, not in), and boolean logic (and/or/not). Method calls and attribute access are NOT supported: write `user_status == 'banned'`, never `state.get('user_status') == 'banned'`. `tools[].available_when` (gating whether a tool is even offered to an agent this turn) uses this exact same restricted grammar — the same rule applies: bare state-key names only, e.g. `available_when: "research_done"`, never `available_when: "state.get('research_done')"`.
If an agent spawns sub-agents (`agents.<name>.spawns`), prefer `spawns.result_schema` (a dotted Pydantic model path) over inventing a free-text handoff protocol, and prefer `spawns.on_complete` (a list of `{{key, value}}` state writes, applied once a spawn genuinely completes) over instructing a spawned agent's own prompt to call `write_state` itself — do not write prose instructions for either of these; declare them in `ai.yaml` instead. `on_complete[].key` must not start with `_` (reserved for internal engine state).
Some decisions are technology choices that depend on the user's actual requirements and must NOT be guessed: which memory backend (sqlite/postgres/redis), whether `inta serve`/`inta monitor` need authentication (and if so, `api_key` vs `custom`), and whether RAG/vector retrieval is needed (and if so, the docs directory and embedding model). If the blueprint requires one of these (e.g. it describes a production deployment, or persistence, or a knowledge base) but doesn't state the choice, do NOT pick a default — instead respond with ONLY this JSON shape and nothing else: {{"clarifications_needed": ["<plain-language question 1>", "<question 2>", ...]}}. Otherwise, output ONLY a valid JSON object matching the IntaGrin AppConfig schema, which we will convert to YAML. Do not output markdown code blocks in either case.

CRITICAL RULES:
1. EVERY agent in `agents` MUST specify a `system_prompt_file` pointing to `"prompts/<agent_name>_prompt.jinja2"`.
2. ALL model identifiers MUST use the valid LiteLLM prefix format (e.g., `gemini/gemini-2.5-flash`, `openai/gpt-4o`, `anthropic/claude-3-5-sonnet-20241022`). Never use `google/` or bare model names.
3. If `server.auth.type` is `api_key`, you MUST explicitly set `server.auth.env_var` to a custom environment variable name (e.g., `MY_APP_API_KEY`) so it is visible in the generated YAML and not hidden as a default.

You MUST conform to this exact JSON Schema for your output. Notice which fields are required (e.g. `version`, `name`, `model` with `primary`, `memory`, `default_agent`):
```json
{json.dumps(AppConfig.model_json_schema(), indent=2)}
```
"""

    user_prompt = f"### EXISTING AI.YAML\n```yaml\n{existing_yaml}\n```\n\n### UPDATED BLUEPRINT.MD\n```markdown\n{blueprint_content}\n```\n\nMerge these and return the final JSON representation of the ai.yaml configuration."


    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]

    max_attempts = 3  # 1 initial attempt + 2 self-heal retries
    validated_config = None
    compiled_json = None
    last_error = ""

    try:
        for attempt in range(max_attempts):
            response = litellm.completion(
                model=compile_model,
                messages=messages,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            if "{" in content:
                import re

                json_str = re.search(r"(\{.*\})", content.replace("\n", "@@@"))
                if json_str:
                    content = json_str.group(1).replace("@@@", "\n")

            try:
                compiled_json = json.loads(content)
            except json.JSONDecodeError as e:
                last_error = f"Response was not valid JSON: {e}"
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {
                        "role": "user",
                        "content": f"That was not valid JSON ({e}). Return ONLY the corrected JSON object, no other text.",
                    }
                )
                continue

            clarifications = compiled_json.get("clarifications_needed")
            if clarifications:
                console.print(
                    "[bold yellow]The blueprint doesn't specify enough to compile safely — "
                    "it needs answers to these questions, not a guess:[/bold yellow]"
                )
                for question in clarifications:
                    console.print(f"  • {question}")
                console.print(
                    "\n[dim]Add the answers to your blueprint and re-run `inta compile`. "
                    "ai.yaml was NOT written.[/dim]"
                )
                raise typer.Exit(1)

            from .config.schema import validate_config_dict

            validated_config, errors = validate_config_dict(compiled_json)
            if not errors:
                break

            last_error = "\n".join(errors)
            validated_config = None
            if attempt < max_attempts - 1:
                console.print(
                    f"[dim]Compiled config failed validation (attempt {attempt + 1}/{max_attempts}) — asking the model to fix it...[/dim]"
                )
            messages.append({"role": "assistant", "content": content})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "That configuration is invalid. Fix ONLY the following problems and "
                        f"return the complete corrected JSON object, no other text:\n{last_error}"
                    ),
                }
            )

        if validated_config is None:
            console.print(
                f"[bold red]Compilation failed after {max_attempts} attempt(s) — "
                "ai.yaml was NOT written.[/bold red]"
            )
            console.print(f"[dim]Last error:[/dim]\n{last_error}")
            raise typer.Exit(1)

        import yaml

        # Dump the validated model's canonical form, not the raw pre-validation JSON — what gets
        # written to disk must be exactly what was actually checked, including any type
        # coercion/normalization Pydantic applied during validation.
        dumped_dict = validated_config.model_dump(exclude_defaults=True, exclude_none=True)
        
        # Explicitly restore `env_var` if it was excluded as a default when `auth.type == api_key`
        # This guarantees it remains visible in the generated config for the user to see.
        if validated_config.server.auth.type == "api_key":
            if "server" not in dumped_dict:
                dumped_dict["server"] = {}
            if "auth" not in dumped_dict["server"]:
                dumped_dict["server"]["auth"] = {"type": "api_key"}
            dumped_dict["server"]["auth"]["env_var"] = validated_config.server.auth.env_var

        new_yaml = yaml.dump(
            dumped_dict,
            default_flow_style=False,
            sort_keys=False,
        )

        if existing_yaml:
            console.print(
                "\n[bold yellow]Existing ai.yaml detected. Proposed Changes:[/bold yellow]"
            )
            import difflib

            diff = difflib.unified_diff(
                existing_yaml.splitlines(),
                new_yaml.splitlines(),
                fromfile="current_ai.yaml",
                tofile="compiled_ai.yaml",
                lineterm="",
            )
            for line in diff:
                if line.startswith("+"):
                    console.print(f"[green]{line}[/green]")
                elif line.startswith("-"):
                    console.print(f"[red]{line}[/red]")
                else:
                    console.print(line)

            if not Confirm.ask("\nApply these architectural changes?"):
                console.print("[dim]Aborted compilation.[/dim]")
                return

        yaml_path.write_text(new_yaml)
        console.print(
            "[bold green]Successfully compiled blueprint into ai.yaml (schema + router syntax validated)![/bold green]"
        )

        scaffolded = _scaffold_referenced_files(project_dir, validated_config)
        for path in scaffolded:
            console.print(
                f"[green]✓ Scaffolded {path.relative_to(project_dir)}[/green] [dim](review and implement — this is a placeholder)[/dim]"
            )

        console.print("\n[bold cyan]Running `inta verify` on the compiled swarm...[/bold cyan]")
        from .compiler.verifier import GraphVerifier

        GraphVerifier(project_dir=project_dir).verify()

        if existing_yaml:
            console.print(
                "\n[dim]This updated an existing ai.yaml — if the project has real session "
                "history, `inta simulate --config ai.yaml` shows what this change would actually "
                "do to it before you rely on it.[/dim]"
            )

    except typer.Exit:
        raise
    except Exception as e:
        Tracer.log_error(f"Compilation failed: {e}")
        raise typer.Exit(1)


@app.command(name="replay")
def replay_command(
    session_id: str = typer.Argument(..., help="The session ID to replay."),
    brief: bool = typer.Option(
        False, "--brief", help="Truncate long content (system: 500 chars, tool: 200 chars) for quick scanning"
    ),
):
    """
    Time-Travel Debugging: Replay the exact sequence of events from a past session — full content
    by default, every handoff labeled with the mechanism that caused it (LLM tool call vs.
    conditional router vs. root router vs. semantic auto_route), tool failures highlighted
    distinctly from successful tool output, and session-total cost/tokens at the end.
    """
    from rich.panel import Panel

    project_dir = Path.cwd()
    if not (project_dir / "ai.yaml").exists():
        _print_cli_error(
            IntaGrinError("IG-CFG-001", "No ai.yaml found in current directory.")
        )
        raise typer.Exit(1)

    from .compiler.parser import parse_project

    try:
        graph = parse_project(project_dir)
    except Exception as e:
        Tracer.log_error(f"Configuration error: {e}")
        raise typer.Exit(1)

    from .runtime.memory import CheckpointerConfigError, build_checkpointer

    try:
        checkpointer = build_checkpointer(graph.config.memory, project_dir, strict=True)
    except CheckpointerConfigError as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(1)

    try:
        messages, state = checkpointer.load_checkpoint(session_id)
        if not messages:
            console.print(
                f"[bold yellow]No messages found for session '{session_id}'.[/bold yellow]"
            )
            return

        console.print(
            f"\n[bold green]=== Time-Travel Replay for Session: {session_id} ===[/bold green]"
        )
        console.print(f"[dim]Total Turns: {len(messages)}[/dim]\n")

        def _truncate(text: str, limit: int) -> str:
            text = str(text)
            if brief and len(text) > limit:
                return text[:limit] + "..."
            return text

        def _is_tool_error(content: str) -> bool:
            return "error" in str(content).lower() or "failed" in str(content).lower()

        # transfer_agent's outcome is a `tool` message, not the assistant's tool_call itself —
        # remember which tool_call_ids were transfer_agent calls so we can label that tool
        # message as a handoff (LLM tool call) rather than generic tool output.
        transfer_agent_call_ids = set()

        # _cost_trace entries are keyed by the message index the assistant response landed at
        # (see RuntimeEngine._record_usage) — sessions checkpointed before this field existed
        # simply have no entries here, so per-turn cost silently falls back to omitted.
        cost_by_turn = {
            entry["turn"]: entry for entry in (state or {}).get("_cost_trace", [])
        }

        # _router_trace (see RuntimeEngine._record_router_trace) is keyed by the same convention.
        # Only non-fired/errored entries are shown here — a router that DID fire already leaves a
        # "Router: Transferred to..." system message above, so repeating it would be pure noise.
        # This is the only place a router that was evaluated but silently didn't fire (or raised,
        # e.g. a typo'd state-key name) becomes visible after the fact — that data previously only
        # ever reached a live-connected Monitor dashboard and was gone the instant nobody was
        # watching. Sessions checkpointed before this field existed simply have no entries here.
        router_trace_by_turn: dict[int, list] = {}
        for entry in (state or {}).get("_router_trace", []):
            if not entry.get("fired"):
                router_trace_by_turn.setdefault(entry["turn"], []).append(entry)

        for idx, msg in enumerate(messages):
            for entry in router_trace_by_turn.get(idx, []):
                if entry.get("error"):
                    console.print(
                        f"[dim yellow]⚠ Router did not fire (condition raised):[/dim yellow] "
                        f"{entry['description']} — [dim]{entry['error']}[/dim]"
                    )
                else:
                    console.print(
                        f"[dim]· Router evaluated, did not fire:[/dim] {entry['description']}"
                    )
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            if role == "system":
                text = str(content)
                if text.startswith("Router: Transferred to") or text.startswith(
                    "Semantic Swarm Router: Control transferred to"
                ):
                    console.print(f"[bold cyan]↪ Handoff:[/bold cyan] {text}")
                else:
                    console.print(
                        Panel(_truncate(text, 500), title="[dim]System[/dim]", border_style="dim")
                    )
            elif role == "user":
                console.print(f"[bold blue]User:[/bold blue] {content}")
            elif role == "assistant":
                turn_cost = cost_by_turn.get(idx)
                cost_suffix = (
                    f" [dim]({turn_cost['tokens']:,} tokens, ${turn_cost['cost']:.6f})[/dim]"
                    if turn_cost
                    else ""
                )
                if msg.get("tool_calls"):
                    calls = []
                    for tc in msg.get("tool_calls"):
                        fn = tc.get("function", {})
                        fname = fn.get("name")
                        if fname == "transfer_agent":
                            transfer_agent_call_ids.add(tc.get("id"))
                        calls.append(f"{fname}({fn.get('arguments')})")
                    console.print(
                        f"[bold purple]Agent (Tool Call):[/bold purple] {', '.join(calls)}{cost_suffix}"
                    )
                if content:
                    console.print(f"[bold purple]Agent:[/bold purple] {content}{cost_suffix}")
            elif role == "tool":
                text = str(content)
                tool_name = msg.get("name")
                if msg.get("tool_call_id") in transfer_agent_call_ids:
                    console.print(f"[bold cyan]↪ Handoff (transfer_agent tool call):[/bold cyan] {text}")
                elif _is_tool_error(text):
                    console.print(
                        f"[bold red]✗ Tool '{tool_name}' FAILED:[/bold red] {_truncate(text, 200)}"
                    )
                else:
                    console.print(
                        f"[bold yellow]Tool '{tool_name}':[/bold yellow] [dim]{_truncate(text, 200)}[/dim]"
                    )

        metrics = (state or {}).get("_metrics", {})
        if metrics:
            per_turn_note = (
                "[dim](per-turn cost shown above)[/dim]"
                if cost_by_turn
                else "[dim](this session predates per-turn cost tracking — totals only)[/dim]"
            )
            console.print(
                f"\n[bold]Session totals:[/bold] {metrics.get('total_tokens', 0):,} tokens, "
                f"${metrics.get('total_cost', 0.0):.6f} {per_turn_note}"
            )

        console.print("\n[bold green]=== End of Replay ===[/bold green]\n")

    except Exception as e:
        Tracer.log_error(f"Replay error: {e}")


def _parse_duration_ago(spec: str):
    """Parses '30d' / '12h' / '2w' into a UTC datetime that many-of-that-unit ago. Returns None
    for anything else, so the caller can report a clear error instead of guessing."""
    import re
    from datetime import datetime, timedelta

    m = re.match(r"^(\d+)([dhw])$", spec.strip().lower())
    if not m:
        return None
    amount, unit = int(m.group(1)), m.group(2)
    delta = {
        "h": timedelta(hours=amount),
        "d": timedelta(days=amount),
        "w": timedelta(weeks=amount),
    }[unit]
    return datetime.now(UTC) - delta


@app.command(name="simulate")
def simulate_command(
    config: str = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to the candidate ai.yaml to simulate against real history",
    ),
    since: str = typer.Option(
        None,
        "--since",
        help="Only replay sessions updated within this window, e.g. '30d', '12h', '2w'",
    ),
    limit: int = typer.Option(
        200, "--limit", help="Maximum number of sessions to replay"
    ),
    session: list[str] = typer.Option(
        None,
        "--session",
        help="Replay only these specific session ids (repeatable) — overrides --since/--limit",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="List every unchanged session too, not just the ones that would differ"
    ),
):
    """
    Shadow Replay (Phase 1): re-evaluate a candidate ai.yaml's routers, circuit breakers, and
    requires_approval flags against real historical sessions — zero new LLM calls, zero
    re-executed tools — and report exactly what would change before you deploy it. Limited to
    config changes that can't alter what the LLM itself generates; anything else (prompts, models,
    tool identity, lazy_load_tools, auto_route, handoffs/delegations) is reported as
    not-yet-simulatable rather than guessed at.
    """
    import asyncio

    from .compiler.parser import ExecutionGraph, parse_project
    from .config.schema import AppConfig
    from .runtime.memory import CheckpointerConfigError
    from .testing.simulator import simulate

    project_dir = Path.cwd()

    try:
        old_graph = parse_project(project_dir)
    except Exception as e:
        Tracer.log_error(f"Configuration error: {e}")
        raise typer.Exit(1)

    candidate_path = Path(config)
    if not candidate_path.is_absolute():
        candidate_path = project_dir / candidate_path
    if not candidate_path.exists():
        _print_cli_error(
            IntaGrinError("IG-CLI-005", f"candidate config '{candidate_path}' not found.")
        )
        raise typer.Exit(1)

    try:
        import yaml

        with open(candidate_path) as f:
            raw = yaml.safe_load(f) or {}
        new_graph = ExecutionGraph(AppConfig(**raw), old_graph.env_vars)
    except Exception as e:
        Tracer.log_error(f"Failed to parse candidate config '{candidate_path}': {e}")
        raise typer.Exit(1)

    since_dt = None
    if since:
        since_dt = _parse_duration_ago(since)
        if since_dt is None:
            _print_cli_error(
                IntaGrinError(
                    "IG-CLI-003",
                    f"--since '{since}' isn't a recognized duration (e.g. '30d', '12h', '2w').",
                )
            )
            raise typer.Exit(1)

    try:
        report = asyncio.run(
            simulate(
                project_dir,
                old_graph,
                new_graph,
                since=since_dt,
                limit=limit,
                session_ids=list(session) if session else None,
            )
        )
    except CheckpointerConfigError as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(1)

    if not report.simulatable:
        console.print(
            "[bold yellow]Not simulatable: this diff could change what the LLM itself "
            "generates, which Phase 1 doesn't attempt to predict.[/bold yellow]"
        )
        for reason in report.not_simulatable_reasons:
            console.print(f"  [dim]-[/dim] {reason}")
        raise typer.Exit(1)

    if report.sessions_checked == 0:
        console.print(
            "[bold yellow]No historical sessions found to simulate against.[/bold yellow]"
        )
        return

    unchanged_count = sum(1 for r in report.results if r.unchanged)
    changed_count = report.sessions_checked - unchanged_count

    console.print(
        f"\n[bold green]=== Shadow Replay: {report.sessions_checked} session(s) checked "
        f"against '{candidate_path.name}' ===[/bold green]"
    )
    console.print(
        f"[green]{unchanged_count} unchanged[/green], "
        f"[yellow]{changed_count} would differ[/yellow]\n"
    )

    if changed_count:
        from rich.table import Table

        table = Table(title="Sessions That Would Differ")
        table.add_column("Session", style="cyan", overflow="fold")
        table.add_column("Verdict", style="bold")
        table.add_column("Turn", justify="right")
        table.add_column("Detail", overflow="fold")
        for r in report.results:
            if r.unchanged:
                continue
            for v in r.verdicts:
                table.add_row(r.session_id, v.kind, str(v.turn), v.detail)
        console.print(table)

    if verbose:
        for r in report.results:
            if r.unchanged:
                console.print(f"[dim]{r.session_id}: unchanged[/dim]")

    console.print()


@app.command(name="copilot")
def copilot_command(
    agent: str = typer.Option(
        None,
        "--agent",
        "-a",
        help="Skip the interactive prompt: one of copilot, cursor, claude, antigravity, factory",
    ),
):
    """
    Setup IntaGrin instructions/skills for external AI coding agents (Copilot, Cursor, Claude, Antigravity, Factory).
    """
    from rich.prompt import IntPrompt

    project_dir = Path.cwd()
    agents = ["copilot", "cursor", "claude", "antigravity", "factory"]

    console.print(
        "\n[bold purple]🤖 Welcome to the IntaGrin AIToolkit Setup![/bold purple]"
    )
    console.print(
        "[dim]This will generate the necessary instructions/skills for your AI coding agent to understand IntaGrin.[/dim]\n"
    )

    if agent:
        if agent not in agents:
            _print_cli_error(
                IntaGrinError("IG-CLI-002", f"--agent must be one of {agents}, got '{agent}'")
            )
            raise typer.Exit(1)
    else:
        console.print("[bold cyan]Supported AI Coding Agents:[/bold cyan]")
        console.print("  [1] GitHub Copilot")
        console.print("  [2] Cursor")
        console.print("  [3] Claude Code")
        console.print("  [4] Antigravity")
        console.print("  [5] Factory")

        choice = IntPrompt.ask(
            "\nSelect your AI coding agent", choices=["1", "2", "3", "4", "5"], default=2
        )
        agent = agents[choice - 1]
    try:

        def scaffold(agent_dir, skill_dir, agent_fname, skill_fname):
            a_dir = project_dir / agent_dir
            s_dir = project_dir / skill_dir / "intagrin-implement"
            r_dir = s_dir / "references"

            a_dir.mkdir(parents=True, exist_ok=True)
            r_dir.mkdir(parents=True, exist_ok=True)

            def write_text(path, content):
                # Always overwrite our specific generated files (by exact path) to prevent
                # duplication bloat on re-runs — we never touch other custom agents/skills the
                # user might have alongside these in the same directory. If a regenerate would
                # actually change a file the user may have hand-edited, say so before clobbering it.
                if path.exists():
                    existing = path.read_text(encoding="utf-8", errors="ignore")
                    if existing != content:
                        console.print(
                            f"[yellow]⚠ Overwriting existing '{path.relative_to(project_dir)}' "
                            "(content differs from what would be generated — any manual edits are lost)[/yellow]"
                        )
                with open(path, "w") as f:
                    f.write(content)

            ref_path = r_dir.relative_to(project_dir) / "architecture.md"
            error_ref_path = r_dir.relative_to(project_dir) / "error_codes.md"
            config_ref_path = r_dir.relative_to(project_dir) / "config_reference.md"
            docs_ref_dir = r_dir.relative_to(project_dir) / "docs"
            cross_ref_text = (
                f"\n\n## CRITICAL INSTRUCTION\nYou MUST read the deep architectural blueprint "
                f"located at `{ref_path}` to understand how to write agents, evals, telemetry, "
                f"and tools for this framework — it has an index of full topic pages under "
                f"`{docs_ref_dir}/`; read the specific page for your question rather than "
                f"guessing from the index's one-line summary. If you see an error formatted like "
                f"`[IG-XXX-000]` in output, tracebacks, or API responses, look it up in "
                f"`{error_ref_path}`. For any question about what's configurable in `ai.yaml` "
                f"(authentication, memory, guardrails, circuit breakers, server, RAG, etc.), "
                f"check `{config_ref_path}` FIRST — it's the complete, generated field reference "
                f"and almost always has the answer directly, without needing to explore the "
                f"project's own files."
            )

            implement_skill_body = _load_copilot_template("implement_skill_body.md")
            cursor_mdc = _load_copilot_template("cursor_mdc.md")

            skill_content = (
                _load_copilot_template("implement_skill_frontmatter.md")
                + implement_skill_body
                + cross_ref_text
            )

            agent_instructions = _load_copilot_template("architect_instructions.md").replace(
                "{ref_path}", str(ref_path)
            )

            if agent == "cursor":
                write_text(a_dir / agent_fname, cursor_mdc + "\n" + agent_instructions)
                skill_content = (
                    cursor_mdc
                    + "\n## Implementation Guide\n"
                    + implement_skill_body
                    + cross_ref_text
                )
            elif agent == "antigravity":
                write_text(
                    a_dir / agent_fname,
                    _load_copilot_template("frontmatter/antigravity.md") + agent_instructions,
                )
            elif agent == "copilot":
                write_text(
                    a_dir / agent_fname,
                    _load_copilot_template("frontmatter/copilot.md") + agent_instructions,
                )
            elif agent == "claude":
                write_text(
                    a_dir / agent_fname,
                    _load_copilot_template("frontmatter/claude.md") + agent_instructions,
                )
            elif agent == "factory":
                write_text(
                    a_dir / agent_fname,
                    _load_copilot_template("frontmatter/factory.md") + agent_instructions,
                )
            else:
                write_text(a_dir / agent_fname, agent_instructions)

            # Skill is common for all
            write_text(s_dir / skill_fname, skill_content)
            write_text(r_dir / "architecture.md", _load_copilot_template("reference_architecture.md"))
            write_text(r_dir / "error_codes.md", _load_copilot_template("reference_error_codes.md"))
            write_text(r_dir / "config_reference.md", _load_copilot_template("reference_config.md"))

            # Bundled verbatim copies of docs/*.md (the real, actively-maintained docs — see
            # scripts/generate_architecture_reference.py) — a loop over whatever's in the bundled
            # docs/ directory rather than one write_text per file, so a doc page added later needs
            # no cli.py change, only a regeneration.
            docs_dir = r_dir / "docs"
            docs_dir.mkdir(parents=True, exist_ok=True)
            bundled_docs_dir = COPILOT_TEMPLATES_DIR / "docs"
            for doc_file in sorted(bundled_docs_dir.glob("*.md")):
                write_text(docs_dir / doc_file.name, _load_copilot_template(f"docs/{doc_file.name}"))

            compile_skill_dir = project_dir / skill_dir / "intagrin-compile"
            compile_skill_dir.mkdir(parents=True, exist_ok=True)

            compile_skill_content = _load_copilot_template("compile_skill.md")

            if agent == "cursor":
                compile_skill_content = cursor_mdc + "\n" + compile_skill_content

            # The compile skill needs its own filename, not intagrin-implement's — reusing
            # skill_fname verbatim previously wrote intagrin-compile's content to a file literally
            # named "intagrin-implement.mdc" for name-specific filenames (Cursor). Generic
            # filenames (SKILL.md / skill.md, used by the other IDEs) are unaffected either way.
            compile_skill_fname = skill_fname.replace("intagrin-implement", "intagrin-compile")
            write_text(compile_skill_dir / compile_skill_fname, compile_skill_content)

            console.print(f"[green]✓ Created Agent in: {a_dir}/{agent_fname}[/green]")
            console.print(f"[green]✓ Created Skill in: {s_dir}/{skill_fname}[/green]")
            console.print(
                f"[green]✓ Created Sync Skill in: {compile_skill_dir}/{compile_skill_fname}[/green]"
            )
            console.print(
                f"[green]✓ Created References in: {r_dir}/architecture.md[/green]"
            )
            console.print(
                f"[green]✓ Created Error Reference in: {r_dir}/error_codes.md[/green]"
            )
            console.print(
                f"[green]✓ Created Config Reference in: {r_dir}/config_reference.md[/green]"
            )
            console.print(
                f"[green]✓ Created {len(list(bundled_docs_dir.glob('*.md')))} Deep Reference doc(s) in: {docs_dir}/[/green]"
            )

        if agent == "antigravity":
            scaffold(
                ".agents/agents/intagrin-architect",
                ".agents/skills",
                "agent.md",
                "SKILL.md",
            )
        elif agent == "cursor":
            scaffold(
                ".cursor/rules",
                ".cursor/skills",
                "intagrin-agent.mdc",
                "intagrin-implement.mdc",
            )
        elif agent == "copilot":
            scaffold(
                ".github/agents", ".github/skills", "intagrin-agent.agent.md", "SKILL.md"
            )
        elif agent == "claude":
            scaffold(".claude/agents", ".claude/skills", "intagrin-agent.md", "skill.md")
        elif agent == "factory":
            scaffold(".factory/droids", ".factory/skills", "intagrin-droid.md", "skill.md")

        console.print(
            f"\n[bold green]Success! Your {agent.title()} agent is now fully equipped with deep skills and references to build IntaGrin applications.[/bold green]"
        )

    except Exception as e:
        Tracer.log_error(f"Failed to setup AIToolkit: {e}")
        raise typer.Exit(1)


db_app = typer.Typer(help="Manage Enterprise Database Migrations")
app.add_typer(db_app, name="db")

@db_app.command("upgrade")
def db_upgrade(
    revision: str = typer.Argument("head", help="The revision to upgrade to")
):
    """
    Run enterprise database migrations (Alembic) to upgrade the schema.
    """
    try:
        from alembic import command
        from alembic.config import Config

        from intagrin.compiler.parser import parse_project
        
        project_dir = Path.cwd()
        graph = parse_project(project_dir)
        mem_cfg = graph.config.memory
        
        if mem_cfg.type == "sqlite":
            db_path = project_dir / (mem_cfg.db_path or ".ai/memory.db")
            url = f"sqlite:///{db_path}"
        elif mem_cfg.type == "postgres":
            url = mem_cfg.connection_url
            if not url and mem_cfg.env_var:
                url = os.environ.get(mem_cfg.env_var)
            if not url:
                url = os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL")
            if not url:
                raise ValueError("PostgreSQL URL not found in config or environment variables.")
        else:
            console.print(f"[bold yellow]Migrations are only supported for SQLite and Postgres, not '{mem_cfg.type}'.[/bold yellow]")
            return
            
        alembic_cfg = Config(str(Path(__file__).parent / "db_migrations" / "alembic.ini"))
        alembic_cfg.set_main_option("script_location", str(Path(__file__).parent / "db_migrations"))
        alembic_cfg.set_main_option("sqlalchemy.url", url)
        
        console.print(f"[bold cyan]Running database migrations for {mem_cfg.type}...[/bold cyan]")
        command.upgrade(alembic_cfg, revision)
        console.print("[bold green]Database schema is up to date![/bold green]")
    except Exception as e:
        Tracer.log_error(f"Migration Error: {e}")
        raise typer.Exit(1)
if __name__ == "__main__":
    app()
