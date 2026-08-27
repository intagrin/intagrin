"""Loads Pydantic models referenced by dotted config paths (`state_schema`, `response_schema`)
so the framework can actually validate against them, instead of only checking that a path string
was set."""

import importlib
from functools import lru_cache

from pydantic import BaseModel


class SchemaLoadError(Exception):
    pass


@lru_cache(maxsize=64)
def load_model(dotted_path: str) -> type[BaseModel]:
    """Imports "package.module.ClassName" and returns the class, verifying it is a Pydantic
    BaseModel subclass. Cached by path since the same schema is resolved on every validation."""
    if "." not in dotted_path:
        raise SchemaLoadError(
            f"'{dotted_path}' is not a valid module path (expected 'module.ClassName')."
        )
    module_path, class_name = dotted_path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise SchemaLoadError(f"Could not import module '{module_path}': {e}") from e

    model = getattr(module, class_name, None)
    if model is None:
        raise SchemaLoadError(f"Module '{module_path}' has no attribute '{class_name}'.")
    if not (isinstance(model, type) and issubclass(model, BaseModel)):
        raise SchemaLoadError(
            f"'{dotted_path}' must be a Pydantic BaseModel subclass, got {type(model)!r}."
        )
    return model
