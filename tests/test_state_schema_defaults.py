"""state_schema's own field defaults are seeded into a session's state on initialize, so a fresh
session starts in the schema's valid shape. Without this, a declared-but-not-yet-written key was
simply absent: a routers[].condition naming it raised "Unknown variable" on every evaluation
(logged as a critical error, surfaced in the Monitor as "Condition never fires" even though the
router later fired fine), and the first `append`-reducer write to a declared list replaced the
missing key with a bare scalar, which then failed the schema's own list validation."""

import asyncio

import pytest

from intagrin.compiler.parser import ExecutionGraph
from intagrin.config.schema import AgentConfig, AppConfig, MemoryConfig, ModelConfig, StateReducerConfig
from intagrin.runtime.engine import RuntimeEngine
from intagrin.runtime.router import safe_eval

DEFAULTS_SCHEMA = """
from pydantic import BaseModel

class IncidentState(BaseModel):
    severity: str | None = None
    notes: list[str] = []
"""

REQUIRED_SCHEMA = """
from pydantic import BaseModel

class StrictState(BaseModel):
    ticket_id: str
"""


def _engine(tmp_path, state_schema: str) -> RuntimeEngine:
    (tmp_path / "defaults_schemas.py").write_text(DEFAULTS_SCHEMA)
    (tmp_path / "required_schemas.py").write_text(REQUIRED_SCHEMA)
    config = AppConfig(
        version="1.0",
        name="state-defaults-test",
        default_agent="assistant",
        state_schema=state_schema,
        model=ModelConfig(primary="mock/model"),
        memory=MemoryConfig(type="buffer"),
        reducers=[StateReducerConfig(key="notes", strategy="append")],
        agents={"assistant": AgentConfig()},
    )
    engine = RuntimeEngine(graph=ExecutionGraph(config, {}), project_dir=tmp_path, session_id="s1")
    asyncio.run(engine.initialize())
    engine.active_agent_name = "assistant"
    return engine


def test_declared_keys_start_at_their_schema_defaults(tmp_path):
    engine = _engine(tmp_path, "defaults_schemas.IncidentState")
    assert engine.state["severity"] is None
    assert engine.state["notes"] == []


def test_condition_over_an_unwritten_declared_key_evaluates_instead_of_raising(tmp_path):
    engine = _engine(tmp_path, "defaults_schemas.IncidentState")
    assert safe_eval("severity == 'sev1'", engine.state, {}) is False


def test_a_typod_undeclared_key_still_raises(tmp_path):
    """Seeding only covers declared fields — typo detection in router conditions is preserved."""
    engine = _engine(tmp_path, "defaults_schemas.IncidentState")
    with pytest.raises(ValueError, match="Unknown variable"):
        safe_eval("sevrity == 'sev1'", engine.state, {})


def test_first_append_write_extends_the_declared_list(tmp_path):
    engine = _engine(tmp_path, "defaults_schemas.IncidentState")
    asyncio.run(engine.execute_tool("write_state", {"key": "notes", "value": "first"}, interactive=False))
    assert engine.state["notes"] == ["first"]


def test_schema_with_required_fields_seeds_nothing_and_still_boots(tmp_path):
    engine = _engine(tmp_path, "required_schemas.StrictState")
    assert "ticket_id" not in engine.state
