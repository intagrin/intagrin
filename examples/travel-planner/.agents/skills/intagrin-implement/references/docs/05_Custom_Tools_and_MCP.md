# Tools & MCP Integration

IntaGrin handles tool argument parsing, JSON decoding, and error trapping (with self-healing retries on malformed arguments) so your tool functions can stay plain Python.

## 1. Local Python Tools
To add a tool, write a standard Python async function in the `tools/` directory.

`tools/github.py`
```python
async def create_issue(title: str, body: str) -> str:
    """Create an issue on GitHub."""
    return f"Created issue: {title}"
```

Register it in `ai.yaml`:
```yaml
agents:
  developer:
    tools:
      - name: "create_issue"
        module: "tools.github"
```

## 2. Model Context Protocol (MCP)
IntaGrin natively supports the open standard **Model Context Protocol (MCP)**. This allows you to plug into hundreds of pre-built tool servers (like Postgres, Slack, or GitHub MCP servers) without writing a single line of python code!

Register an MCP server in `ai.yaml`:

```yaml
tools:
  - name: "github_mcp"
    type: "mcp"
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]

agents:
  developer:
    tools:
      - name: "github_mcp" # Explicit reference grants this agent access
```

The framework will automatically spin up the node process, negotiate the JSON-RPC streams, parse the available tools from the MCP server, and inject their JSON schemas into your agents!

Every MCP connection and OpenAPI spec fetch declared in `ai.yaml` — global and per-agent — is
established concurrently on startup, not one at a time. Ten MCP servers cost roughly as long as
the single slowest one to connect, not the sum of all ten.

## 3. Strict Tool Access Control (RBAC)
When you define a root-level tool (like an MCP server), it is **not** globally accessible to every agent by default. Register the provider at the root, then grant each agent access with a name-only reference under that agent's `tools` list. A complete tool definition under an agent is also scoped to that agent.

The runtime filters both the model-visible tool schemas and tool execution. An agent can only access a tool or provider explicitly listed in its own `tools` block, preventing a `Support` agent from accessing a `drop_database` tool assigned to the `DBA` agent.
