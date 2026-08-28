---
name: intagrin-compile
description: Bidirectional Architecture Sync Skill
---

## Skill: IntaGrin-Compile (Bidirectional Architecture Sync)

When you are asked to 'compile', 'sync', or design the architecture, you act as the intelligent compiler.

1. **Read Context**: Always read `blueprint.md` (Product Requirements) and `ai.yaml` (Technical Truth).
2. **Analyze**: Check for discrepancies in agent roles, handoffs, tools, and state variables.
3. **Prompt the User**: If they are out of sync, DO NOT OVERWRITE blindly. Ask the user:
   > 'The blueprint and YAML are out of sync. Which direction should I sync? 
   > (A) Update YAML to match Blueprint
   > (B) Update Blueprint to match YAML'
4. **Execute Sync**: Based on the user's answer, rewrite the target file to align them perfectly.
5. **Scaffold Missing Code**: If you added new agents to the YAML, immediately scaffold their `.jinja2` prompt files or Python tools. Draft a real, production-quality system prompt from the blueprint's own description of that agent (persona, tools, when to hand off) — never a one-line placeholder. End a newly-drafted prompt file with its own line: `{# intagrin:blueprint-hash <hash> #}`, where `<hash>` is computed the same way `inta compile` computes it: `python3 -c "import hashlib; print(hashlib.sha256(open('blueprint.md','rb').read()).hexdigest()[:16])"`. This marks the file as machine-drafted, so a later sync — by you, or by `inta compile` itself — recognizes it as safe to reconsider, staying interchangeable between the two.
6. **Update Drafted Prompts, Never Hand-Written Ones**: Before touching an *existing* prompt file, check whether it ends with an `intagrin:blueprint-hash` comment. No marker means a human wrote or edited it — leave it alone entirely, even if the blueprint changed since. A marker whose hash no longer matches the current `blueprint.md` (recompute it with the same command as step 5) means it's safe to reconsider — but show the user a diff of the current text vs. your redraft and get explicit confirmation before overwriting it, exactly like step 3's sync confirmation, then stamp the new hash. Never silently regenerate an existing prompt file just because it's tracked as machine-drafted — and if the user declines, still update the marker's hash so you don't ask about the same unresolved diff again next time.
7. **Proactive Architecture**: You MUST proactively recommend framework features! If the architecture needs cross-agent state persistence, recommend `state_schema` and `reducers` (Shared Typed State Redux). If it requires long-term persistence, recommend configuring a checkpointer (e.g. `memory: {type: sqlite}`). Get approval before adding them to `ai.yaml`.
8. **Router condition syntax**: `routers[].condition` is evaluated by a restricted grammar, not Python's `eval()` — bare state-key names, literals, comparisons (`<`, `<=`, `>`, `>=`, `==`, `!=`, `in`, `not in`), and boolean logic (`and`/`or`/`not`) only. Write `user_status == 'banned'`, never `state.get('user_status') == 'banned'` — method calls and attribute access are not supported and the router will simply never fire, with no error anywhere you'd see it.
9. **Mandatory verification, not optional**: After writing or editing `ai.yaml`, always run `inta verify` and read its output before telling the user you're done. It statically validates the config schema (via `parse_project`), checks for cycles across handoffs/routers, and now also flags exactly the router-condition-syntax mistake in step 8. If it reports any errors, fix them and run `inta verify` again — do not report success until it's clean. This is the same gate the CLI's own `inta compile` command enforces in code (it won't write `ai.yaml` at all if the config doesn't validate); running `inta verify` here is how this skill holds itself to the same standard.
