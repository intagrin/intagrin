import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from intagrin.compiler.parser import ExecutionGraph
from intagrin.config.schema import (
    AgentConfig,
    AppConfig,
    MemoryConfig,
    ModelConfig,
    ModelVariantConfig,
)
from intagrin.runtime.engine import RuntimeEngine


def _variants():
    return [
        ModelVariantConfig(model="openai/gpt-4o-mini", weight=1.0),
        ModelVariantConfig(model="openai/gpt-4o", weight=1.0),
    ]


def test_select_model_variant_is_deterministic_for_a_given_session_id():
    variants = _variants()
    first = RuntimeEngine._select_model_variant(variants, "session-abc-123")
    second = RuntimeEngine._select_model_variant(variants, "session-abc-123")
    assert first == second
    assert first in {v.model for v in variants}


def test_select_model_variant_distribution_roughly_matches_weights():
    variants = [
        ModelVariantConfig(model="a", weight=1.0),
        ModelVariantConfig(model="b", weight=3.0),
    ]
    counts = {"a": 0, "b": 0}
    for i in range(2000):
        chosen = RuntimeEngine._select_model_variant(variants, f"session-{i}")
        counts[chosen] += 1

    # Expect roughly a 25/75 split (1:3 weight ratio) — generous tolerance since this is a hash
    # bucket, not a true RNG, and the test must not be flaky.
    b_fraction = counts["b"] / (counts["a"] + counts["b"])
    assert 0.65 < b_fraction < 0.85, counts


@pytest.fixture
def mock_graph_no_variants():
    config = AppConfig(
        version="1.0",
        name="test-swarm",
        default_agent="triage",
        model=ModelConfig(primary="openai/gpt-4o-mini"),
        memory=MemoryConfig(type="buffer"),
        agents={"triage": AgentConfig(description="Triage agent")},
    )
    return ExecutionGraph(config, {})


@pytest.fixture
def mock_graph_with_variants():
    config = AppConfig(
        version="1.0",
        name="test-swarm",
        default_agent="triage",
        model=ModelConfig(primary="openai/gpt-4o-mini", variants=_variants()),
        memory=MemoryConfig(type="buffer"),
        agents={"triage": AgentConfig(description="Triage agent")},
    )
    return ExecutionGraph(config, {})


@pytest.mark.anyio
async def test_initialize_does_not_set_model_variant_when_unconfigured(mock_graph_no_variants, tmp_path):
    engine = RuntimeEngine(graph=mock_graph_no_variants, project_dir=tmp_path, session_id="s1")
    await engine.initialize()
    assert "_model_variant" not in engine.state


@pytest.mark.anyio
async def test_initialize_sets_a_sticky_model_variant_when_configured(mock_graph_with_variants, tmp_path):
    engine = RuntimeEngine(graph=mock_graph_with_variants, project_dir=tmp_path, session_id="s1")
    await engine.initialize()
    assert engine.state["_model_variant"] in {"openai/gpt-4o-mini", "openai/gpt-4o"}


@pytest.mark.anyio
async def test_initialize_does_not_overwrite_an_already_persisted_model_variant(
    mock_graph_with_variants, tmp_path
):
    """A checkpoint reload must keep the session pinned to whichever variant it was already
    assigned, not recompute (which would be a no-op here since selection is deterministic, but
    this guards against a future change to the selection algorithm silently flipping variants
    mid-conversation for already-assigned sessions)."""
    engine = RuntimeEngine(graph=mock_graph_with_variants, project_dir=tmp_path, session_id="s1")
    engine.state["_model_variant"] = "some-other-model-from-an-older-config"
    await engine.initialize()
    assert engine.state["_model_variant"] == "some-other-model-from-an-older-config"


def test_run_agent_turn_actually_calls_litellm_with_the_assigned_variant_model(
    mock_graph_with_variants, tmp_path
):
    """End-to-end: the sticky variant assigned in state must be the model _run_agent_turn passes
    to litellm.acompletion — not just present in state but unused."""

    async def _run():
        engine = RuntimeEngine(
            graph=mock_graph_with_variants, project_dir=tmp_path, session_id="variant_test"
        )
        await engine.initialize()
        engine.active_agent_name = "triage"
        engine.state["_model_variant"] = "openai/gpt-4o"  # force a specific assignment
        engine.messages.append({"role": "user", "content": "hi"})

        mock_message = MagicMock(content="A reply.", tool_calls=None)
        mock_message.model_dump.return_value = {"role": "assistant", "content": "A reply."}
        mock_response = MagicMock(
            choices=[MagicMock(message=mock_message)],
            usage=MagicMock(prompt_tokens=5, completion_tokens=5, total_tokens=10),
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = mock_response
            await engine._run_agent_turn(interactive=False)

        assert mock_acompletion.call_args.kwargs["model"] == "openai/gpt-4o"

    asyncio.run(_run())
