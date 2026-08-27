# Example: Autonomous Coding Agent

Here's a pattern for building a coding-agent loop with IntaGrin's handoffs: 

### Architecture (`ai.yaml`)
```yaml
agents:
  architect_agent:
    description: "Formulates a plan and searches the codebase."
    handoffs: ["coder_agent"]
    tools:
      - name: "grep_search"
      - name: "list_directory"

  coder_agent:
    description: "Writes the actual code."
    handoffs: ["verifier_agent"]
    tools:
      - name: "replace_file_content"

  verifier_agent:
    description: "Runs tests (e.g. pytest)."
    handoffs: ["coder_agent"] # Loop back on failure!
    tools:
      - name: "run_bash_command"
        requires_approval: true # Crucial safety net!
```

### Self-Healing Loop
Because the `verifier_agent` can handoff back to the `coder_agent`, if a test fails, the error traceback is automatically passed back to the coder to fix its mistake until the code runs perfectly.
