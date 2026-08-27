import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from intagrin.compiler.parser import ExecutionGraph
from intagrin.config.schema import AgentConfig, AppConfig, MemoryConfig, ModelConfig
from intagrin.runtime.engine import RuntimeEngine


def test_extract_text_for_routing_passes_plain_strings_through():
    assert RuntimeEngine._extract_text_for_routing("hello world") == "hello world"


def test_extract_text_for_routing_joins_text_parts_of_a_multimodal_message():
    content = [
        {"type": "text", "text": "explain in detail what is in this image"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]
    assert (
        RuntimeEngine._extract_text_for_routing(content)
        == "explain in detail what is in this image"
    )


def test_extract_text_for_routing_never_returns_a_python_repr_string():
    """The bug this helper replaces: str(content) on a list produces a Python repr like
    "[{'type': 'text', ...}]" — trigger-phrase/word-count routing heuristics would then match
    against dict-repr noise (braces, quotes, key names) instead of the actual user text."""
    content = [{"type": "text", "text": "hi"}, {"type": "image_url", "image_url": {"url": "x"}}]
    result = RuntimeEngine._extract_text_for_routing(content)
    assert "{'type'" not in result
    assert "image_url" not in result


def test_extract_text_for_routing_returns_empty_string_for_image_only_message():
    content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]
    assert RuntimeEngine._extract_text_for_routing(content) == ""


@pytest.fixture
def mock_graph_with_auto_model():
    config = AppConfig(
        version="1.0",
        name="test-swarm",
        default_agent="triage",
        model=ModelConfig(primary="auto", fallback="gemini/gemini-2.5-flash"),
        memory=MemoryConfig(type="buffer"),
        agents={"triage": AgentConfig(description="Triage agent")},
    )
    return ExecutionGraph(config, {})


class _MockMessage:
    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self, exclude_none=True):
        return {"role": "assistant", "content": self.content}


class _MockChoice:
    def __init__(self, message):
        self.message = message


class _MockResponse:
    def __init__(self, message):
        self.choices = [_MockChoice(message)]
        self.usage = MagicMock(prompt_tokens=10, completion_tokens=10, total_tokens=20)


def test_run_agent_turn_routes_multimodal_input_on_real_text_not_a_repr_string(
    mock_graph_with_auto_model, tmp_path
):
    """End-to-end: a multi-modal user message (as the API's ChatRequest.message already accepts)
    reaches _run_agent_turn's "auto" model-resolution call with its real text extracted, not a
    Python repr of the content-part list."""

    async def _run():
        engine = RuntimeEngine(
            graph=mock_graph_with_auto_model, project_dir=tmp_path, session_id="mm_test"
        )
        await engine.initialize()
        engine.active_agent_name = "triage"
        engine.messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "explain in detail step by step how this works"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AAAA"},
                    },
                ],
            }
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion, patch(
            "intagrin.runtime.engine.SwarmRouter.resolve_model", wraps=lambda *a, **k: "mock/model"
        ) as mock_resolve:
            mock_acompletion.return_value = _MockResponse(_MockMessage("A plain text reply."))
            await engine._run_agent_turn(interactive=False)

        assert mock_resolve.called
        routed_text = mock_resolve.call_args.args[2]
        assert routed_text == "explain in detail step by step how this works"
        assert "{'type'" not in routed_text
        assert "image_url" not in routed_text

    asyncio.run(_run())


def test_run_agent_turn_stream_also_applies_auto_model_resolution(
    mock_graph_with_auto_model, tmp_path
):
    """_run_agent_turn_stream must resolve model.primary: "auto" through the same
    SwarmRouter.resolve_model call _run_agent_turn uses — otherwise the streaming path would
    hand LiteLLM the literal string "auto" instead of a real model id, while the blocking path
    (tested above) worked fine, a silent divergence between the two turn implementations."""

    async def _run():
        engine = RuntimeEngine(
            graph=mock_graph_with_auto_model, project_dir=tmp_path, session_id="mm_stream_test"
        )
        await engine.initialize()
        engine.active_agent_name = "triage"
        engine.messages.append(
            {"role": "user", "content": "explain in detail step by step how this works"}
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion, patch(
            "intagrin.runtime.engine.SwarmRouter.resolve_model", wraps=lambda *a, **k: "mock/model"
        ) as mock_resolve:
            mock_acompletion.return_value = _MockResponse(_MockMessage("A plain text reply."))
            async for _ in engine._run_agent_turn_stream(interactive=False):
                pass

        assert mock_resolve.called
        assert mock_resolve.call_args.args[2] == "explain in detail step by step how this works"

    asyncio.run(_run())
