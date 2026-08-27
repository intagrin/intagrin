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
