import asyncio
from unittest.mock import patch

from intagrin.compiler.parser import ExecutionGraph
from intagrin.config.schema import (
    AgentConfig,
    AppConfig,
    LocalToolConfig,
    MemoryConfig,
    ModelConfig,
)
from intagrin.runtime.worker import DistributedWorker


def _config():
    return AppConfig(
        version="1.0",
        name="worker-pool-test",
        default_agent="assistant",
        model=ModelConfig(primary="mock/model"),
        memory=MemoryConfig(type="buffer"),
        agents={"assistant": AgentConfig()},
        tools=[LocalToolConfig(name="noop", module="fake.module")],
    )


def test_process_task_reuses_pooled_resources_across_jobs(tmp_path):
    """A worker processing two sequential jobs against the same project must only load local
    tools once — the second job's engine should reuse the first job's pooled SharedResources
    instead of reloading tools / reconnecting MCP servers from scratch on every task."""

    async def run():
        graph = ExecutionGraph(_config(), {})
        worker = DistributedWorker(project_dir=tmp_path)

        load_calls = {"n": 0}

        def fake_load_local_tool(module, name):
            load_calls["n"] += 1
            return lambda: "ok"

        with patch(
            "intagrin.runtime.worker.parse_project", return_value=graph
        ), patch(
            "intagrin.runtime.engine.load_local_tool", side_effect=fake_load_local_tool
        ):
            await worker.process_task({"workflow": "nonexistent", "session_id": "job1"})
            await worker.process_task({"workflow": "nonexistent", "session_id": "job2"})

        assert load_calls["n"] == 1, "local tool should only be loaded once, by the pooled builder"
        assert len(worker._resources_cache._entries) == 1

        await worker.stop()

    asyncio.run(run())
