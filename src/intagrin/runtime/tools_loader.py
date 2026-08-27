import importlib
import inspect
from collections.abc import Callable
from typing import Any

from ..errors import IntaGrinError


def load_local_tool(module_name: str, function_name: str) -> Callable:
    """Dynamically loads a function from a module."""
    try:
        module = importlib.import_module(module_name)
        func = getattr(module, function_name)
        return func
    except (ImportError, AttributeError) as e:
        raise IntaGrinError(
            "IG-RT-006", f"Failed to load tool {function_name} from {module_name}: {e}"
        )


def get_tool_schema(func: Callable) -> dict[str, Any]:
    """Extracts OpenAI-compatible JSON schema from a python function's signature and docstring."""
    from pydantic import Field, create_model

    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""

    # Parse docstring for descriptions (basic :param or arg: format)
    param_docs = {}
    for line in doc.split("\n"):
        line = line.strip()
        if line.startswith(":param "):
            parts = line[7:].split(":", 1)
            if len(parts) == 2:
                param_docs[parts[0].strip()] = parts[1].strip()
        elif ":" in line and not line.endswith(":"):
            parts = line.split(":", 1)
            arg_name = parts[0].split("(")[0].strip()
            if arg_name in sig.parameters:
                param_docs[arg_name] = parts[1].strip()

    fields = {}
    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue

        annotation = (
            param.annotation if param.annotation != inspect.Parameter.empty else str
        )
        desc = param_docs.get(name, f"Parameter {name}")

        if param.default == inspect.Parameter.empty:
            fields[name] = (annotation, Field(..., description=desc))
        else:
            fields[name] = (annotation, Field(param.default, description=desc))

    try:
        model = create_model(f"{func.__name__}Args", **fields)
        schema = model.model_json_schema()
    except Exception:
        schema = {"type": "object", "properties": {}}

    if "title" in schema:
        del schema["title"]

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": doc.split("\n")[0] if doc else f"Tool {func.__name__}",
            "parameters": schema,
        },
    }
