import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from intagrin.compiler.parser import ExecutionGraph
from intagrin.config.schema import (
    AgentConfig,
    AppConfig,
    LocalToolConfig,
    MemoryConfig,
    ModelConfig,
    ToolReferenceConfig,
)
from intagrin.runtime.engine import RuntimeEngine


def test_extra_field_on_a_config_model_is_rejected_not_silently_dropped():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        AgentConfig(auto_route=True, tool_pool=["should not exist on AgentConfig"])


def test_tool_union_resolves_a_local_tool_dict_to_local_tool_config_only():
    # Before extra="forbid", a {name, module} dict could also validate as ToolReferenceConfig
    # (which just silently drops `module`), leaving the union match to Pydantic's own
    # tie-breaking instead of the shape of the data.
    agent = AgentConfig(tools=[{"name": "my_tool", "module": "tools.custom_tools"}])
    assert isinstance(agent.tools[0], LocalToolConfig)


def test_tool_union_resolves_a_bare_name_dict_to_tool_reference_config():
    agent = AgentConfig(tools=[{"name": "shared_tool"}])
    assert isinstance(agent.tools[0], ToolReferenceConfig)

STATE_SCHEMA_MODULE = """
from pydantic import BaseModel, ConfigDict

class AppState(BaseModel):
    model_config = ConfigDict(extra="allow")

    balance: int
"""

RESPONSE_SCHEMA_MODULE = """
from pydantic import BaseModel

class InvoiceSummary(BaseModel):
    invoice_id: str
    total: float
"""


def _write_module(tmp_path: Path, filename: str, content: str):
    (tmp_path / filename).write_text(content)


def test_write_state_rejects_a_write_that_violates_state_schema(tmp_path):
    _write_module(tmp_path, "state_schemas_reject.py", STATE_SCHEMA_MODULE)

    config = AppConfig(
        version="1.0",
        name="typed-state-test",
        default_agent="assistant",
        model=ModelConfig(primary="mock/model"),
        memory=MemoryConfig(type="buffer"),
        state_schema="state_schemas_reject.AppState",
        agents={"assistant": AgentConfig()},
    )
    engine = RuntimeEngine(ExecutionGraph(config, {}), tmp_path)

    result = engine.write_state("balance", "not-a-number")

    assert "rejected" in result
    assert "state_schema" in result
    # The real state must be untouched — the trial write was never committed
    assert "balance" not in engine.state


def test_write_state_accepts_a_write_that_satisfies_state_schema(tmp_path):
    _write_module(tmp_path, "state_schemas_accept.py", STATE_SCHEMA_MODULE)

    config = AppConfig(
        version="1.0",
        name="typed-state-test",
        default_agent="assistant",
        model=ModelConfig(primary="mock/model"),
        memory=MemoryConfig(type="buffer"),
        state_schema="state_schemas_accept.AppState",
        agents={"assistant": AgentConfig()},
    )
    engine = RuntimeEngine(ExecutionGraph(config, {}), tmp_path)

    result = engine.write_state("balance", "42")

    assert "successfully" in result
    assert engine.state["balance"] == 42


def test_write_state_ignores_schema_when_not_configured(tmp_path):
    config = AppConfig(
        version="1.0",
        name="untyped-state-test",
        default_agent="assistant",
        model=ModelConfig(primary="mock/model"),
        memory=MemoryConfig(type="buffer"),
        agents={"assistant": AgentConfig()},
    )
    engine = RuntimeEngine(ExecutionGraph(config, {}), tmp_path)

    result = engine.write_state("anything", '{"nested": true}')

    assert "successfully" in result
    assert engine.state["anything"] == {"nested": True}


def test_response_schema_heals_an_invalid_response(tmp_path):
    _write_module(tmp_path, "response_schemas.py", RESPONSE_SCHEMA_MODULE)

    config = AppConfig(
        version="1.0",
        name="typed-response-test",
        default_agent="billing",
        model=ModelConfig(primary="mock/model"),
        memory=MemoryConfig(type="buffer"),
        agents={"billing": AgentConfig(response_schema="response_schemas.InvoiceSummary")},
    )
    engine = RuntimeEngine(ExecutionGraph(config, {}), tmp_path)
    agent_cfg = config.agents["billing"]

    bad_msg = MagicMock()
    bad_msg.tool_calls = None
    bad_msg.content = '{"invoice_id": "INV-1"}'  # missing required "total"

    healed_response = MagicMock()
    healed_response.choices = [MagicMock(message=MagicMock(content='{"invoice_id": "INV-1", "total": 42.5}'))]

    async def run():
        with patch("litellm.acompletion", new=AsyncMock(return_value=healed_response)):
            return await engine._apply_response_schema(agent_cfg, bad_msg)

    result_msg = asyncio.run(run())
    assert result_msg.content == '{"invoice_id": "INV-1", "total": 42.5}'


def test_response_schema_passes_through_a_valid_response(tmp_path):
    _write_module(tmp_path, "response_schemas.py", RESPONSE_SCHEMA_MODULE)

    config = AppConfig(
        version="1.0",
        name="typed-response-test",
        default_agent="billing",
        model=ModelConfig(primary="mock/model"),
        memory=MemoryConfig(type="buffer"),
        agents={"billing": AgentConfig(response_schema="response_schemas.InvoiceSummary")},
    )
    engine = RuntimeEngine(ExecutionGraph(config, {}), tmp_path)
    agent_cfg = config.agents["billing"]

    good_msg = MagicMock()
    good_msg.tool_calls = None
    good_msg.content = '{"invoice_id": "INV-1", "total": 42.5}'

    async def run():
        with patch("litellm.acompletion", new=AsyncMock(side_effect=AssertionError("should not heal a valid response"))):
            return await engine._apply_response_schema(agent_cfg, good_msg)

    result_msg = asyncio.run(run())
    assert result_msg.content == '{"invoice_id": "INV-1", "total": 42.5}'
