"""Small shared helpers for model metadata that more than one module needs (diagnostics, the
monitor dashboard's context-window meter). Single source of truth so they can't drift apart."""

from functools import lru_cache


@lru_cache(maxsize=64)
def resolve_context_window(model_name: str) -> int:
    """Best-effort lookup of a model's real input context window via LiteLLM's model catalog,
    falling back to a conservative 128k guess if the model isn't recognized."""
    import litellm

    try:
        info = litellm.get_model_info(model=model_name)
        window = info.get("max_input_tokens") or info.get("max_tokens")
        if window:
            return int(window)
    except Exception:
        pass
    return 128_000
