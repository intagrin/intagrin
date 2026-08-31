from pathlib import Path

import pytest

from intagrin.compiler.parser import ParserError, parse_project


def test_parse_valid_project(tmp_path: Path):
    ai_yaml = """version: "1.0"
name: "test-app"
model:
  primary: "openai/gpt-4o-mini"
  temperature: 0.5
memory:
  type: "sliding_window"
  max_messages: 5
default_agent: "agent1"
agents:
  agent1:
    system_prompt_file: "prompt.txt"
    tools:
      - name: "test_tool"
        module: "tools.custom"
"""
    (tmp_path / "ai.yaml").write_text(ai_yaml)
    (tmp_path / ".env").write_text("TEST_VAR=123\n")
    
    graph = parse_project(tmp_path)
    assert graph.config.name == "test-app"
    assert graph.config.model.temperature == 0.5
    assert len(graph.config.agents["agent1"].tools) == 1
    assert graph.env_vars.get("TEST_VAR") == "123"

def test_parse_invalid_schema(tmp_path: Path):
    ai_yaml = """version: "1.0"
name: "test-app"
model:
  primary: "openai/gpt-4o-mini"
  temperature: 5.0  # Invalid, max is 2.0
memory:
  type: "unknown_type" # Invalid enum
default_agent: "agent1"
agents:
  agent1:
    system_prompt_file: "prompt.txt"
"""
    (tmp_path / "ai.yaml").write_text(ai_yaml)
    
    with pytest.raises(ParserError) as exc:
        parse_project(tmp_path)
    
    err_str = str(exc.value)
    assert "model.temperature: Input should be less than or equal to 2" in err_str
    assert "memory.type: Input should be 'sliding_window', 'buffer', 'sqlite', 'postgres', 'redis' or 'custom'" in err_str

def test_missing_file(tmp_path: Path):
    with pytest.raises(ParserError) as exc:
        parse_project(tmp_path)
    assert "Missing configuration file" in str(exc.value)

def test_parse_rejects_a_typo_d_agent_field_instead_of_silently_dropping_it(tmp_path: Path):
    # `toolz` (typo of `tools`) must be a hard parse error, not a silently-ignored key that
    # leaves the agent missing the tool with zero indication why.
    ai_yaml = """version: "1.0"
name: "test-app"
model:
  primary: "openai/gpt-4o-mini"
memory:
  type: "buffer"
default_agent: "agent1"
agents:
  agent1:
    toolz:
      - name: "test_tool"
        module: "tools.custom"
"""
    (tmp_path / "ai.yaml").write_text(ai_yaml)

    with pytest.raises(ParserError) as exc:
        parse_project(tmp_path)

    err_str = str(exc.value)
    assert "agents.agent1.toolz" in err_str
    assert "Extra inputs are not permitted" in err_str

def test_parse_rejects_an_unknown_top_level_key(tmp_path: Path):
    ai_yaml = """version: "1.0"
name: "test-app"
model:
  primary: "openai/gpt-4o-mini"
memory:
  type: "buffer"
default_agent: "agent1"
agents:
  agent1: {}
routing:
  strategy: "round_robin"
"""
    (tmp_path / "ai.yaml").write_text(ai_yaml)

    with pytest.raises(ParserError) as exc:
        parse_project(tmp_path)

    assert "routing" in str(exc.value)
    assert "Extra inputs are not permitted" in str(exc.value)
