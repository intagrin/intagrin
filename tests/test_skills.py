import asyncio

from intagrin.compiler.parser import ExecutionGraph
from intagrin.config.schema import (
    AgentConfig,
    AppConfig,
    MemoryConfig,
    ModelConfig,
    SkillConfig,
    SkillReferenceConfig,
)
from intagrin.runtime.engine import RuntimeEngine


def _mock_graph(*, skill_refs=None, skills=None):
    config = AppConfig(
        version="1.0",
        name="skills-test",
        default_agent="support",
        model=ModelConfig(primary="mock/model"),
        memory=MemoryConfig(type="buffer"),
        skills=skills
        if skills is not None
        else [
            SkillConfig(
                name="refund_policy",
                description="How to handle refund requests",
                path="skills/refund_policy",
            ),
            SkillConfig(
                name="single_file_skill",
                description="A skill that is just one file",
                path="skills/single_file_skill.md",
            ),
        ],
        agents={
            "support": AgentConfig(
                description="Support agent",
                skills=skill_refs
                if skill_refs is not None
                else ["refund_policy", "single_file_skill"],
            ),
        },
    )
    return ExecutionGraph(config, {})


def _write_skill_files(project_dir):
    skill_dir = project_dir / "skills" / "refund_policy"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# Refund Policy\nAlways refund within 30 days.")
    (skill_dir / "extra_notes.md").write_text("Escalate anything over $500.")

    (project_dir / "skills").mkdir(exist_ok=True)
    (project_dir / "skills" / "single_file_skill.md").write_text("Just do the simple thing.")


async def _init_engine(tmp_path, graph, agent_name="support"):
    engine = RuntimeEngine(graph=graph, project_dir=tmp_path, session_id="s1")
    await engine.initialize()
    engine.active_agent_name = agent_name
    return engine


def test_load_skill_registered_with_enum_and_joined_description(tmp_path):
    async def _run():
        _write_skill_files(tmp_path)
        engine = await _init_engine(tmp_path, _mock_graph())
        agent_cfg = engine.graph.config.agents["support"]

        active_tools = await engine._get_active_tools(agent_cfg)
        load_skill = next(t for t in active_tools if t["function"]["name"] == "load_skill")

        name_param = load_skill["function"]["parameters"]["properties"]["name"]
        assert set(name_param["enum"]) == {"refund_policy", "single_file_skill"}
        assert "refund_policy: How to handle refund requests" in load_skill["function"]["description"]
        assert (
            "single_file_skill: A skill that is just one file"
            in load_skill["function"]["description"]
        )

    asyncio.run(_run())


def test_read_skill_resource_only_registered_for_directory_skills(tmp_path):
    async def _run():
        _write_skill_files(tmp_path)
        engine = await _init_engine(tmp_path, _mock_graph())
        agent_cfg = engine.graph.config.agents["support"]

        active_tools = await engine._get_active_tools(agent_cfg)
        names = {t["function"]["name"] for t in active_tools}
        assert "read_skill_resource" in names  # refund_policy is a directory

    asyncio.run(_run())


def test_no_skill_tools_registered_when_agent_has_no_skills(tmp_path):
    async def _run():
        _write_skill_files(tmp_path)
        engine = await _init_engine(tmp_path, _mock_graph(skill_refs=[]))
        agent_cfg = engine.graph.config.agents["support"]

        active_tools = await engine._get_active_tools(agent_cfg)
        names = {t["function"]["name"] for t in active_tools}
        assert "load_skill" not in names
        assert "read_skill_resource" not in names

    asyncio.run(_run())


def test_load_skill_returns_skill_md_body_for_directory_skill(tmp_path):
    async def _run():
        _write_skill_files(tmp_path)
        engine = await _init_engine(tmp_path, _mock_graph())

        result = await engine.execute_tool("load_skill", {"name": "refund_policy"})
        assert "Always refund within 30 days." in result

    asyncio.run(_run())


def test_load_skill_returns_file_body_for_single_file_skill(tmp_path):
    async def _run():
        _write_skill_files(tmp_path)
        engine = await _init_engine(tmp_path, _mock_graph())

        result = await engine.execute_tool("load_skill", {"name": "single_file_skill"})
        assert result == "Just do the simple thing."

    asyncio.run(_run())


def test_load_skill_rejects_a_skill_not_granted_to_this_agent(tmp_path):
    async def _run():
        _write_skill_files(tmp_path)
        engine = await _init_engine(tmp_path, _mock_graph(skill_refs=["single_file_skill"]))

        result = await engine.execute_tool("load_skill", {"name": "refund_policy"})
        assert "not available" in result.lower()

    asyncio.run(_run())


def test_read_skill_resource_reads_a_sibling_file(tmp_path):
    async def _run():
        _write_skill_files(tmp_path)
        engine = await _init_engine(tmp_path, _mock_graph())

        result = await engine.execute_tool(
            "read_skill_resource",
            {"skill_name": "refund_policy", "resource_path": "extra_notes.md"},
        )
        assert "Escalate anything over $500." in result

    asyncio.run(_run())


def test_read_skill_resource_blocks_path_traversal(tmp_path):
    """Verify-by-breaking target: a resource_path that escapes the skill's own directory must be
    rejected, not silently read (e.g. exfiltrating project secrets via '../../.env')."""

    async def _run():
        _write_skill_files(tmp_path)
        (tmp_path / "secret.txt").write_text("do-not-leak")
        engine = await _init_engine(tmp_path, _mock_graph())

        result = await engine.execute_tool(
            "read_skill_resource",
            {"skill_name": "refund_policy", "resource_path": "../../secret.txt"},
        )
        assert "do-not-leak" not in result
        assert "escapes" in result.lower() or "rejected" in result.lower()

    asyncio.run(_run())


def test_read_skill_resource_rejects_when_skill_is_a_single_file(tmp_path):
    async def _run():
        _write_skill_files(tmp_path)
        engine = await _init_engine(tmp_path, _mock_graph())

        result = await engine.execute_tool(
            "read_skill_resource",
            {"skill_name": "single_file_skill", "resource_path": "anything.md"},
        )
        assert "not" in result.lower()

    asyncio.run(_run())


def test_skill_gated_by_available_when_is_hidden_until_condition_holds(tmp_path):
    async def _run():
        _write_skill_files(tmp_path)
        graph = _mock_graph(
            skill_refs=[
                SkillReferenceConfig(name="refund_policy", available_when="escalated == True"),
                "single_file_skill",
            ]
        )
        engine = await _init_engine(tmp_path, graph)
        agent_cfg = engine.graph.config.agents["support"]

        names_before = {
            t["function"]["name"]
            for t in (await engine._get_active_tools(agent_cfg))
            if t["function"]["name"] == "load_skill"
        }
        load_skill_before = next(
            (t for t in await engine._get_active_tools(agent_cfg) if t["function"]["name"] == "load_skill"),
            None,
        )
        assert "refund_policy" not in load_skill_before["function"]["parameters"]["properties"]["name"]["enum"]

        engine.state["escalated"] = True
        load_skill_after = next(
            t for t in await engine._get_active_tools(agent_cfg) if t["function"]["name"] == "load_skill"
        )
        assert "refund_policy" in load_skill_after["function"]["parameters"]["properties"]["name"]["enum"]
        assert names_before  # sanity: load_skill was present both times (single_file_skill ungated)

    asyncio.run(_run())
