"""Tests for model.cascade (FrugalGPT-style cheap-to-expensive escalation), scoped to
response_schema-validated agents only — see _resolve_cascade_entry_model and the cascade
escalation block in _apply_response_schema (runtime/engine.py)."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from intagrin.compiler.parser import ExecutionGraph
from intagrin.config.schema import AgentConfig, AppConfig, MemoryConfig, ModelConfig
from intagrin.runtime.engine import RuntimeEngine

RESPONSE_SCHEMA_MODULE = """
from pydantic import BaseModel

class InvoiceSummary(BaseModel):
    invoice_id: str
    total: float
"""


def _write_schema_module(tmp_path: Path):
    (tmp_path / "response_schemas.py").write_text(RESPONSE_SCHEMA_MODULE)


def _engine(tmp_path, cascade=None, response_schema="response_schemas.InvoiceSummary"):
    _write_schema_module(tmp_path)
    config = AppConfig(
        version="1.0",
        name="cascade-test",
        default_agent="billing",
        model=ModelConfig(primary="primary-model", cascade=cascade),
        memory=MemoryConfig(type="buffer"),
        agents={"billing": AgentConfig(response_schema=response_schema)},
    )
    engine = RuntimeEngine(ExecutionGraph(config, {}), tmp_path)
    return engine, config.agents["billing"]


# --- _resolve_cascade_entry_model: gating -------------------------------------------------------


def test_cascade_entry_model_used_when_both_response_schema_and_cascade_are_set(tmp_path):
    engine, agent_cfg = _engine(tmp_path, cascade=["cheap-model", "mid-model"])
    assert engine._resolve_cascade_entry_model(agent_cfg) == "cheap-model"


def test_cascade_entry_model_none_without_response_schema(tmp_path):
    engine, agent_cfg = _engine(tmp_path, cascade=["cheap-model"], response_schema=None)
    assert engine._resolve_cascade_entry_model(agent_cfg) is None


def test_cascade_entry_model_none_without_cascade_configured(tmp_path):
    engine, agent_cfg = _engine(tmp_path, cascade=None)
    assert engine._resolve_cascade_entry_model(agent_cfg) is None


# --- escalation behavior in _apply_response_schema -----------------------------------------------


def _bad_msg():
    msg = MagicMock()
    msg.tool_calls = None
    msg.content = '{"invoice_id": "INV-1"}'  # missing required "total"
    return msg


def test_cascade_escalates_to_the_next_tier_on_validation_failure(tmp_path):
    engine, agent_cfg = _engine(tmp_path, cascade=["cheap-model", "mid-model"])
    current_messages = [{"role": "user", "content": "invoice please"}]

    escalated_response = MagicMock()
    escalated_response.choices = [
        MagicMock(message=MagicMock(content='{"invoice_id": "INV-1", "total": 42.5}'))
    ]

    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs["model"])
        return escalated_response

    async def run():
        with patch("litellm.acompletion", side_effect=fake_acompletion):
            return await engine._apply_response_schema(agent_cfg, _bad_msg(), current_messages)

    result = asyncio.run(run())
    assert result.content == '{"invoice_id": "INV-1", "total": 42.5}'
    # cascade[0] ("cheap-model") already ran as the primary attempt outside this function — the
    # first thing _apply_response_schema itself should try is the next tier up ("mid-model"),
    # not "primary-model" or "cheap-model" again.
    assert calls == ["mid-model"]


def test_cascade_falls_through_every_tier_to_the_corrector_patch(tmp_path):
    """Every cascade tier and primary keep failing validation on regeneration — must still fall
    back to the pre-existing corrector-patch mechanism rather than giving up."""
    engine, agent_cfg = _engine(tmp_path, cascade=["cheap-model", "mid-model"])
    current_messages = [{"role": "user", "content": "invoice please"}]

    still_bad_response = MagicMock()
    still_bad_response.choices = [MagicMock(message=MagicMock(content='{"invoice_id": "INV-1"}'))]
    healed_response = MagicMock()
    healed_response.choices = [
        MagicMock(message=MagicMock(content='{"invoice_id": "INV-1", "total": 99.0}'))
    ]

    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs["model"])
        # Every cascade/primary escalation attempt still fails validation; the final call is the
        # corrector patch (model.fallback, unset here, so it defaults to "gemini/gemini-2.5-flash").
        if kwargs["model"] == "gemini/gemini-2.5-flash":
            return healed_response
        return still_bad_response

    async def run():
        with patch("litellm.acompletion", side_effect=fake_acompletion):
            return await engine._apply_response_schema(agent_cfg, _bad_msg(), current_messages)

    result = asyncio.run(run())
    assert result.content == '{"invoice_id": "INV-1", "total": 99.0}'
    # "mid-model" (remaining cascade tier) and "primary-model" (final cascade tier) both tried
    # and failed before the corrector patch ran.
    assert calls == ["mid-model", "primary-model", "gemini/gemini-2.5-flash"]


def test_cascade_deduplicates_primary_when_already_listed_in_cascade(tmp_path):
    """cascade=["cheap-model", "primary-model"] with primary="primary-model" must not try
    "primary-model" twice."""
    engine, agent_cfg = _engine(tmp_path, cascade=["cheap-model", "primary-model"])
    current_messages = [{"role": "user", "content": "invoice please"}]

    escalated_response = MagicMock()
    escalated_response.choices = [
        MagicMock(message=MagicMock(content='{"invoice_id": "INV-1", "total": 42.5}'))
    ]
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs["model"])
        return escalated_response

    async def run():
        with patch("litellm.acompletion", side_effect=fake_acompletion):
            return await engine._apply_response_schema(agent_cfg, _bad_msg(), current_messages)

    asyncio.run(run())
    assert calls == ["primary-model"]  # exactly once, not twice


def test_cascade_never_retries_tier_zero_even_if_it_reappears_later_in_the_list(tmp_path):
    """cascade[0] ("cheap-model") already ran as the primary attempt and failed — it must not be
    retried even if it's (redundantly) listed again later in `cascade`, since that would just
    repeat a call already known to fail."""
    engine, agent_cfg = _engine(tmp_path, cascade=["cheap-model", "cheap-model", "mid-model"])
    current_messages = [{"role": "user", "content": "invoice please"}]

    escalated_response = MagicMock()
    escalated_response.choices = [
        MagicMock(message=MagicMock(content='{"invoice_id": "INV-1", "total": 42.5}'))
    ]
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs["model"])
        return escalated_response

    async def run():
        with patch("litellm.acompletion", side_effect=fake_acompletion):
            return await engine._apply_response_schema(agent_cfg, _bad_msg(), current_messages)

    asyncio.run(run())
    assert calls == ["mid-model"]  # "cheap-model" skipped entirely, not retried


def test_cascade_skipped_without_current_messages_falls_straight_to_corrector(tmp_path):
    """A future call site that forgets to pass current_messages must degrade safely to today's
    exact corrector-patch behavior, not silently lose the response."""
    engine, agent_cfg = _engine(tmp_path, cascade=["cheap-model", "mid-model"])

    healed_response = MagicMock()
    healed_response.choices = [
        MagicMock(message=MagicMock(content='{"invoice_id": "INV-1", "total": 42.5}'))
    ]
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs["model"])
        return healed_response

    async def run():
        with patch("litellm.acompletion", side_effect=fake_acompletion):
            return await engine._apply_response_schema(agent_cfg, _bad_msg(), None)

    result = asyncio.run(run())
    assert result.content == '{"invoice_id": "INV-1", "total": 42.5}'
    assert calls == ["gemini/gemini-2.5-flash"]  # straight to the corrector, no cascade tiers tried


def test_cascade_never_touches_a_message_that_still_has_tool_calls(tmp_path):
    """The core safety property: cascade escalation must never run for a non-terminal message —
    msg.tool_calls being set already short-circuits _apply_response_schema before cascade logic
    is ever reached, so no tool call risks being re-executed."""
    engine, agent_cfg = _engine(tmp_path, cascade=["cheap-model", "mid-model"])
    msg = MagicMock()
    msg.tool_calls = [MagicMock()]
    msg.content = "irrelevant"

    async def run():
        with patch("litellm.acompletion", new=AsyncMock(side_effect=AssertionError("must not be called"))):
            return await engine._apply_response_schema(agent_cfg, msg, [{"role": "user", "content": "x"}])

    result = asyncio.run(run())
    assert result is msg  # returned untouched


def test_without_cascade_configured_behavior_is_unchanged(tmp_path):
    """Regression guard: an agent with response_schema but no model.cascade must behave exactly
    like before this feature existed — straight to the corrector patch, current_messages or not."""
    engine, agent_cfg = _engine(tmp_path, cascade=None)
    current_messages = [{"role": "user", "content": "invoice please"}]

    healed_response = MagicMock()
    healed_response.choices = [
        MagicMock(message=MagicMock(content='{"invoice_id": "INV-1", "total": 42.5}'))
    ]
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs["model"])
        return healed_response

    async def run():
        with patch("litellm.acompletion", side_effect=fake_acompletion):
            return await engine._apply_response_schema(agent_cfg, _bad_msg(), current_messages)

    result = asyncio.run(run())
    assert result.content == '{"invoice_id": "INV-1", "total": 42.5}'
    assert calls == ["gemini/gemini-2.5-flash"]


# --- turn-loop integration: the actual cost saving --------------------------------------------


def test_run_agent_turn_actually_calls_litellm_with_the_cascade_entry_model(tmp_path):
    """End-to-end: for an agent with response_schema + model.cascade configured, the primary
    completion call _run_agent_turn issues must use cascade[0], not model.primary — this is
    where the real cost saving comes from (the whole turn, not just escalation)."""
    engine, agent_cfg = _engine(tmp_path, cascade=["cheap-model", "mid-model"])

    async def _run():
        await engine.initialize()
        engine.active_agent_name = "billing"
        engine.messages.append({"role": "user", "content": "give me an invoice"})

        mock_message = MagicMock(
            content='{"invoice_id": "INV-1", "total": 42.5}', tool_calls=None
        )
        mock_message.model_dump.return_value = {
            "role": "assistant",
            "content": '{"invoice_id": "INV-1", "total": 42.5}',
        }
        mock_response = MagicMock(
            choices=[MagicMock(message=mock_message)],
            usage=MagicMock(prompt_tokens=5, completion_tokens=5, total_tokens=10),
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = mock_response
            await engine._run_agent_turn(interactive=False)

        assert mock_acompletion.call_args.kwargs["model"] == "cheap-model"

    asyncio.run(_run())


def test_run_agent_turn_uses_primary_model_when_response_schema_is_unset(tmp_path):
    """Regression guard for the gating itself: an agent WITHOUT response_schema must keep using
    model.primary even if model.cascade is configured at the app level — cascade only ever
    applies to schema-validated agents."""
    engine, agent_cfg = _engine(tmp_path, cascade=["cheap-model", "mid-model"], response_schema=None)

    async def _run():
        await engine.initialize()
        engine.active_agent_name = "billing"
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

        assert mock_acompletion.call_args.kwargs["model"] == "primary-model"

    asyncio.run(_run())
