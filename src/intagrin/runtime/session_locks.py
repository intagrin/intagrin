"""Per-session concurrency control for the API server.

Every request builds a fresh RuntimeEngine, loads the whole checkpointed session, and overwrites
it whole on save (see runtime/memory.py) — nothing serializes two concurrent requests for the same
session_id. A double-click, a client retry-on-timeout, or two open tabs against the same session
silently drops one entire turn (last write wins). This registry gives server/api.py a lock keyed by
session_id to serialize processing of one session at a time, closing that race for a single
process. It does not provide cross-process/cross-replica mutual exclusion — that needs a
distributed lock or optimistic concurrency at the checkpoint-storage layer, out of scope here.

Same dict-of-locks-by-key shape as SharedResourcesCache._lock_for (shared_resources.py), keyed by
session_id instead of project_dir.
"""

import asyncio


class SessionLockRegistry:
    def __init__(self):
        self._locks: dict[str, asyncio.Lock] = {}

    def get_lock(self, session_id: str) -> asyncio.Lock:
        lock = self._locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[session_id] = lock
        return lock


_default_registry: SessionLockRegistry | None = None


def get_session_lock_registry() -> SessionLockRegistry:
    global _default_registry
    if _default_registry is None:
        _default_registry = SessionLockRegistry()
    return _default_registry
