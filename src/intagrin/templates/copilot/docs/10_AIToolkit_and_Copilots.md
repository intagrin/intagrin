# AI Toolkit & Copilots

IntaGrin is designed to pair well with your IDE's AI agents — `inta copilot` generates rule/skill
files that teach your assistant the framework's conventions instead of relying on it to infer them
from scratch.

## 1. Zero-Code Scaffolding (`inta copilot`)
If you use Cursor, GitHub Copilot, or Claude Code, do not write `ai.yaml` by hand. 

Run the following command in your project root:
```bash
inta copilot
```
The CLI will generate rule files (e.g., `.cursor/rules/intagrin-agent.mdc`) that describe IntaGrin's
YAML schema and conventions to your IDE's LLM — review the generated rules before relying on them.

## 2. The `intagrin-compile` Skill (Bidirectional Sync)
When you run `inta copilot`, it installs the **`intagrin-compile`** skill into your selected IDE integration.

### How to use it:
1. Create a plain-english file called `blueprint.md` in your project.
2. Describe your product vision: *"I need a triage agent that talks to a billing agent."*
3. Open your IDE's AI chat and say: **"Compile my architecture."**

The IDE Agent will read your `blueprint.md`, read your existing `ai.yaml`, and intelligently prompt you if they are out of sync. It acts as an interactive software architect, automatically writing your `ai.yaml` and scaffolding your `.jinja2` prompt files for you — and, per the skill's own instructions, always finishes by running `inta verify` and fixing anything it reports before calling the task done, rather than assuming the config it wrote is correct.

## 3. The CLI Compiler
If you prefer CI/CD or terminal workflows, you can bypass the IDE agent and use the native CLI compiler. `inta compile` writes `ai.yaml` itself, so — unlike `inta dev`/`inta run`, which read the key from a project's already-written `ai.yaml`/`.env` — it needs to resolve which model to compile *with* before that file necessarily exists. It picks, in order:

1. **`--model`/`-m`**, if you pass it — always wins, and is the one to use for CI/CD or any other non-interactive run: `inta compile blueprint.md --model openai/gpt-4o`.
2. **The project's own `model.primary`**, if `ai.yaml` already exists — re-compiling an existing project's blueprint reuses whatever provider it's already configured with, rather than silently switching to a different one for just this call.
3. **An interactive prompt**, only for a first-ever compile with neither of the above — asks which provider (OpenAI, Gemini, Anthropic, a local Ollama/Llama.cpp model, or any other LiteLLM model string), then the API key if that provider needs one. For OpenAI/Gemini/Anthropic, a follow-up prompt then offers a sensible default model (e.g. `openai/gpt-4o`) that you're free to type over with any other model from that same provider — picking a provider is never a commitment to one specific model from it. The key is both set for this run and appended to a `.env` next to `blueprint.md`, so you're not asked again next time.

Whichever way the model gets resolved, a missing API key fails fast with a clear `IG-CLI-008` error rather than a raw stack trace.
```bash
inta compile blueprint.md
```
This performs the same bidirectional diff-merge between your Markdown spec and your existing `ai.yaml`, preserving manual API keys and config it wasn't asked to change. Unlike free-form generation, the compiled result is never written to disk unchecked:

- **Validated, not hoped.** Every compile is checked against the real `AppConfig` schema and against the router-condition grammar (`routers[].condition` only supports bare state-key names and comparisons — a generated `state.get(...)` condition is caught here, not discovered later as a router that silently never fires). A config that doesn't validate is self-healed by feeding the exact error back to the model, up to 2 retries; if it still doesn't validate, **`ai.yaml` is not written** and the command exits with an error — a blocked compile beats a broken one.
- **Scaffolds what it references.** Any `system_prompt_file` or local tool the compiled config points at gets scaffolded if it doesn't already exist — never overwriting a file you've already edited by hand. A missing prompt file is drafted by the same compile model, using the agent's own description/tools/handoffs/delegations plus the full `blueprint.md` as context, so it starts as a real production-oriented system prompt rather than a one-line stub; if drafting itself fails (a network error, a bad response) it falls back to a minimal placeholder rather than blocking the compile. Review a scaffolded prompt before relying on it either way.
- **Verifies in the same flow.** On success, `inta compile` automatically runs `inta verify` against the result, so cycle/cost/condition-syntax feedback shows up immediately instead of requiring a separate command you have to remember to run. If you're updating an existing, already-running `ai.yaml`, it also points you at `inta simulate --config ai.yaml` to check what the change would actually do against real session history before you rely on it — see [Production Deployment](./04_Production_Deployment).
