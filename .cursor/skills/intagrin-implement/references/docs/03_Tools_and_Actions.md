# Custom Tools & Actions

In IntaGrin, tools are just standard Python functions. 

## Writing a Tool
Open `tools/custom_tools.py` and write your function with type hints and a docstring. The framework automatically parses this into JSON schema for the LLM.

```python
def fetch_user_data(user_id: int) -> str:
    """
    Fetches the user's data from the CRM.
    
    Args:
        user_id: The ID of the user.
    """
    import requests
    return requests.get(f"https://api.crm.com/{user_id}").text
```

## Registering the Tool
In your `ai.yaml`, attach the tool to an agent:

```yaml
agents:
  support:
    tools:
      - name: "fetch_user_data"
        module: "tools.custom_tools"
```

## Human-in-the-Loop (`requires_approval`)
For dangerous actions (like writing to a database or executing code), IntaGrin has built-in safety pauses. 

```yaml
      - name: "delete_user"
        module: "tools.custom_tools"
        requires_approval: true
```
If an agent attempts to delete a user, the engine will pause execution and alert you in the `inta monitor` dashboard to approve the action.

## Asynchronous Thread-Pooling
IntaGrin automatically detects synchronous Python functions and pushes them to a background thread pool (`asyncio.to_thread`), so a long-running tool doesn't block the FastAPI event loop. Note this doesn't bound memory or wall-clock time per call — a tool that loads a huge payload into memory can still exhaust the process; add your own limits for untrusted or unbounded workloads.

## Running Agent-Generated Code (`type: "sandbox"`)
If an agent needs to actually *execute* code an LLM wrote — not just call a fixed Python function you authored — use a sandbox tool instead of shelling out from inside a regular tool:

```yaml
agents:
  coder:
    tools:
      - name: "run_python"
        type: "sandbox"
        language: "python"        # or "bash"
        timeout_seconds: 10       # wall-clock kill switch (also the CPU-time rlimit)
        max_memory_mb: 256
        requires_approval: true   # recommended unless the code is fully trusted
```

The LLM calls it with a single `code` argument. Each call runs in a fresh, isolated subprocess with a wall-clock timeout, POSIX CPU/memory limits, and an explicit minimal environment (never a copy of your process's own `os.environ` — a real API key sitting in your engine's environment won't leak into whatever the code prints or reads).

**Be clear about what this is and isn't.** It's process and resource isolation for a buggy or runaway script — it is *not* a filesystem or network sandbox: the subprocess runs as the same OS user as your engine and can read/write anything that user can, and there's no firewall blocking outbound network calls. For genuinely untrusted code, pair this with `requires_approval: true` (`inta verify` will nudge you if you forget) or swap in a real container/microVM-based executor. Its output is treated as `untrusted_output` by default (see [08_Security_and_Reliability.md](./08_Security_and_Reliability.md)), since what the code prints can be influenced by whatever it read or did.
