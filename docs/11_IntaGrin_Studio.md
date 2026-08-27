# Bidirectional Visual Orchestration (IntaGrin Studio)

Writing YAML is fast, but visualizing complex multi-agent swarms is hard. IntaGrin includes a
visual node editor with no external services required.

## Launching the Studio
Run the following command:
```bash
inta monitor
```
The studio launches locally at `http://localhost:3000` by default. Use `inta monitor --port <port>` to choose another port.

## Features
1. **Drag-and-Drop Handoffs**: In the Dashboard tab, you will see a React Flow node graph of your `ai.yaml`. If you want your Triage agent to route to your Billing agent, simply drag a connecting arrow between the two nodes in the UI. 
2. **Instant Bidirectional Sync**: When you connect two nodes visually, the Python backend automatically detects the change, parses your `ai.yaml`, injects the new `handoffs: ["billing"]` array into the correct agent, and saves the file back to your disk in real-time.
3. **Execution Traces**: Switch to the "Execution Traces" tab to view real-time logs of your swarm's LLM reasoning — a self-hosted, open-source trace view in the spirit of LangSmith, though without its hosted analytics/eval tooling.
4. **Thermodynamic HUD**: See your live token burn rate and USD costs directly in the UI during playground test sessions.
