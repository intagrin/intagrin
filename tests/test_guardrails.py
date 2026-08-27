"""model.guardrails.custom_module's own field description says it runs "in addition to the
built-in checks" (mask_pii, banned_words) — but _apply_guardrails_to_text used to `return` the
custom module's result immediately, skipping the built-in checks entirely whenever a custom
module was configured. Fixed to fall through instead: custom module transforms the text first,
then the built-in checks still run on the result."""

import asyncio
import sys
import types

from intagrin.compiler.parser import ExecutionGraph
from intagrin.config.schema import (
    AgentConfig,
    AppConfig,
    GuardrailsConfig,
    MemoryConfig,
    ModelConfig,
)
from intagrin.runtime.engine import RuntimeEngine


def _graph(guardrails: GuardrailsConfig) -> ExecutionGraph:
    config = AppConfig(
        version="1.0",
        name="guardrails-test",
        default_agent="assistant",
        model=ModelConfig(primary="mock/model", guardrails=guardrails),
        memory=MemoryConfig(type="sqlite"),
        agents={"assistant": AgentConfig(description="Test agent")},
    )
    return ExecutionGraph(config, {})


def _install_module(name: str, source: str):
    """Installs a throwaway module into sys.modules directly — no tmp_path/sys.path project-dir
    machinery needed since _apply_guardrails_to_text just does importlib.import_module(name),
    which finds an already-imported module without touching disk."""
    mod = types.ModuleType(name)
    exec(source, mod.__dict__)
    sys.modules[name] = mod


async def _engine(tmp_path, guardrails, session_id):
    engine = RuntimeEngine(graph=_graph(guardrails), project_dir=tmp_path, session_id=session_id)
    await engine.initialize()
    return engine


def test_mask_pii_alone_still_works(tmp_path):
    async def _run():
        engine = await _engine(tmp_path, GuardrailsConfig(mask_pii=True), "s1")
        return engine._apply_guardrails_to_text("Contact me at a@b.com")

    result = asyncio.run(_run())
    assert "a@b.com" not in result
    assert "[REDACTED_EMAIL]" in result


def test_custom_module_runs_in_addition_to_mask_pii_not_instead_of_it(tmp_path):
    """The core regression: a custom_module that appends a marker must not suppress mask_pii —
    both effects must be present in the final text."""
    _install_module(
        "guardrail_addendum_test",
        "def apply_guardrails(text, guardrails):\n    return text + ' [custom-checked]'\n",
    )

    async def _run():
        engine = await _engine(
            tmp_path,
            GuardrailsConfig(mask_pii=True, custom_module="guardrail_addendum_test"),
            "s2",
        )
        return engine._apply_guardrails_to_text("Contact me at a@b.com")

    result = asyncio.run(_run())
    assert "a@b.com" not in result
    assert "[REDACTED_EMAIL]" in result
    assert "[custom-checked]" in result


def test_custom_module_output_is_itself_subject_to_mask_pii(tmp_path):
    """Proves the fall-through order: built-in checks run on the custom module's *returned* text,
    not the original — an email the custom module introduces still gets masked."""
    _install_module(
        "guardrail_injects_pii_test",
        "def apply_guardrails(text, guardrails):\n    return text + ' contact new@example.com'\n",
    )

    async def _run():
        engine = await _engine(
            tmp_path,
            GuardrailsConfig(mask_pii=True, custom_module="guardrail_injects_pii_test"),
            "s3",
        )
        return engine._apply_guardrails_to_text("hello")

    result = asyncio.run(_run())
    assert "new@example.com" not in result
    assert "[REDACTED_EMAIL]" in result


def test_a_failing_custom_module_still_leaves_built_in_checks_running(tmp_path):
    _install_module(
        "guardrail_raises_test",
        "def apply_guardrails(text, guardrails):\n    raise RuntimeError('boom')\n",
    )

    async def _run():
        engine = await _engine(
            tmp_path,
            GuardrailsConfig(mask_pii=True, custom_module="guardrail_raises_test"),
            "s4",
        )
        return engine._apply_guardrails_to_text("Contact me at a@b.com")

    result = asyncio.run(_run())
    assert "a@b.com" not in result
    assert "[REDACTED_EMAIL]" in result


def test_banned_words_also_still_applies_alongside_a_custom_module(tmp_path):
    _install_module(
        "guardrail_noop_test",
        "def apply_guardrails(text, guardrails):\n    return text\n",
    )

    async def _run():
        engine = await _engine(
            tmp_path,
            GuardrailsConfig(banned_words=["competitor_name"], custom_module="guardrail_noop_test"),
            "s5",
        )
        return engine._apply_guardrails_to_text("Ask about competitor_name pricing")

    result = asyncio.run(_run())
    assert "competitor_name" not in result
    assert "[REDACTED_BANNED]" in result
