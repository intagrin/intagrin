import pytest
from pydantic import ValidationError

from intagrin.config.schema import AgentConfig, AgentSpawningConfig, LocalToolConfig


def _tool(name: str) -> LocalToolConfig:
    return LocalToolConfig(name=name, module="tools.custom")


def test_spawns_defaults_to_none_zero_behavior_change():
    agent = AgentConfig(description="test", tools=[_tool("search")])
    assert agent.spawns is None


def test_spawns_tool_pool_as_a_subset_of_own_tools_is_accepted():
    agent = AgentConfig(
        description="orchestrator",
        tools=[_tool("search"), _tool("summarize"), _tool("issue_refund")],
        spawns=AgentSpawningConfig(tool_pool=["search", "summarize"]),
    )
    assert agent.spawns.tool_pool == ["search", "summarize"]


def test_spawns_tool_pool_naming_a_tool_the_agent_lacks_is_rejected():
    with pytest.raises(ValidationError) as exc_info:
        AgentConfig(
            description="orchestrator",
            tools=[_tool("search")],
            spawns=AgentSpawningConfig(tool_pool=["search", "issue_refund"]),
        )
    assert "issue_refund" in str(exc_info.value)


def test_spawns_tool_pool_rejected_when_agent_has_no_tools_at_all():
    with pytest.raises(ValidationError) as exc_info:
        AgentConfig(description="orchestrator", spawns=AgentSpawningConfig(tool_pool=["search"]))
    assert "search" in str(exc_info.value)


def test_spawns_tool_pool_wildcard_expands_to_every_tool_the_agent_has():
    agent = AgentConfig(
        description="orchestrator",
        tools=[_tool("search"), _tool("summarize"), _tool("issue_refund")],
        spawns=AgentSpawningConfig(tool_pool="*"),
    )
    assert agent.spawns.tool_pool == ["issue_refund", "search", "summarize"]


def test_spawns_tool_pool_wildcard_on_a_toolless_agent_expands_to_empty():
    agent = AgentConfig(description="orchestrator", spawns=AgentSpawningConfig(tool_pool="*"))
    assert agent.spawns.tool_pool == []


def test_agent_spawning_config_safe_defaults():
    cfg = AgentSpawningConfig(tool_pool=["search"])
    assert cfg.requires_approval_on_first_action is True
    assert cfg.allow_recursive_spawning is False
    assert cfg.max_spawn_depth == 1
    assert cfg.max_creations_per_session == 3
    assert cfg.model_pool is None
