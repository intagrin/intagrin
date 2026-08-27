"""Pooling for the expensive, session-independent resources a RuntimeEngine normally rebuilds
from scratch on every construction: MCP server connections, the RAG index, tool schemas, local
tool functions, and agent system prompts. Verified (by grep, see engine.py) that none of these are
mutated after RuntimeEngine.initialize() finishes, so it's safe to build them once and share
read-only across concurrent sessions/engines for the same project.
"""

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any


class SharedResources:
    """Container for one project's pooled tools/connections/prompts."""

    def __init__(
        self,
        mcp_manager,
        global_tool_schemas: list[dict[str, Any]],
        local_tools: dict[str, Callable],
        agent_prompts: dict[str, str],
        tools_requiring_approval: dict[str, dict],
        untrusted_tools: set[str] | None = None,
    ):
        self.mcp_manager = mcp_manager
        self.global_tool_schemas = global_tool_schemas
        self.local_tools = local_tools
        self.agent_prompts = agent_prompts
        self.tools_requiring_approval = tools_requiring_approval
        self.untrusted_tools = untrusted_tools if untrusted_tools is not None else set()


class SharedResourcesCache:
    """Process-level cache of SharedResources keyed by project_dir. Lazily builds a
    SharedResources on first use by running one real RuntimeEngine.initialize() and lifting its
    resources out, guarded by an asyncio.Lock so concurrent first-requests don't double-build.
    Invalidated and rebuilt if ai.yaml's mtime changes, mirroring compiler/parser.py's own
    mtime-based cache."""

    def __init__(self):
        self._entries: dict[str, tuple[float, SharedResources]] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock_for(self, key: str) -> asyncio.Lock:
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock

    async def get(self, graph, project_dir: Path) -> SharedResources:
        key = str(project_dir)
        try:
            mtime = (project_dir / "ai.yaml").stat().st_mtime
        except OSError:
            mtime = 0.0

        cached = self._entries.get(key)
        if cached is not None and cached[0] == mtime:
            return cached[1]

        async with self._lock_for(key):
            cached = self._entries.get(key)
            if cached is not None and cached[0] == mtime:
                return cached[1]

            old = self._entries.pop(key, None)
            if old is not None and old[1].mcp_manager:
                try:
                    await old[1].mcp_manager.cleanup()
                except Exception:
                    pass

            from .engine import RuntimeEngine

            builder = RuntimeEngine(graph, project_dir)
            await builder.initialize()
            resources = SharedResources(
                mcp_manager=builder.mcp_manager,
                global_tool_schemas=builder.global_tool_schemas,
                local_tools=builder.local_tools,
                agent_prompts=builder.agent_prompts,
                tools_requiring_approval=builder.tools_requiring_approval,
                untrusted_tools=builder.untrusted_tools,
            )
            self._entries[key] = (mtime, resources)
            return resources

    async def cleanup(self):
        """Tear down every pooled MCP connection. Call once, on process shutdown."""
        for _, resources in self._entries.values():
            if resources.mcp_manager:
                try:
                    await resources.mcp_manager.cleanup()
                except Exception:
                    pass
        self._entries.clear()


_default_cache: SharedResourcesCache | None = None


def get_shared_resources_cache() -> SharedResourcesCache:
    global _default_cache
    if _default_cache is None:
        _default_cache = SharedResourcesCache()
    return _default_cache
