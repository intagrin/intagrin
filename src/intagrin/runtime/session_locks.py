"""Per-session concurrency control for the API server.

Every request builds a fresh RuntimeEngine, loads the whole checkpointed session, and overwrites
it whole on save (see runtime/memory.py) — nothing serializes two concurrent requests for the same
session_id. A double-click, a client retry-on-timeout, or two open tabs against the same session
silently drops one entire turn (last write wins). This registry gives server/api.py a lock keyed by
session_id to serialize processing of one session at a time, closing that race for a single
process. It does not provide cross-process/cross-replica mutual exclusion — that needs a
distributed lock or optimistic concurrency at the checkpoint-storage layer, out of scope here.

Same dict-of-locks-by-key shape as SharedResourcesCache._lock_for (shared_resources.py), keyed by
session_id instead of project_dir — except SharedResourcesCache's keys (project directories) are
naturally bounded (a process serves a handful of projects at most), while a session_id is not: a
long-lived server sees an ever-growing set of distinct sessions over its lifetime. A plain
dict-of-locks that never removes an entry leaks one asyncio.Lock per session forever. get_lock()
here instead hands out a reference-counted wrapper (_RefCountedLock) that forgets a session's
entry the moment its last outstanding acquire()/`async with` has released it, so the registry's
size tracks concurrently in-flight sessions, not every session ever seen.
"""

import asyncio


class _RefCountedLock:
    """Wraps one session's real asyncio.Lock so the owning registry can tell when nobody still
    holds or is waiting on it, and forget it at that point. Delegates acquire/release/locked to
    the real Lock, so both existing call-site shapes in server/api.py keep working unchanged:
    `async with registry.get_lock(id):` and manual `lock = registry.get_lock(id); await
    lock.acquire(); ...; lock.release()`."""

    __slots__ = ("_lock", "_registry", "_session_id", "_acquired")

    def __init__(self, lock: asyncio.Lock, registry: "SessionLockRegistry", session_id: str):
        self._lock = lock
        self._registry = registry
        self._session_id = session_id
        self._acquired = False

    async def acquire(self) -> bool:
        try:
            result = await self._lock.acquire()
        except BaseException:
            # Acquisition itself failed or was cancelled (e.g. client disconnect mid-wait) — this
            # wrapper will never call release(), so drop its refcount contribution now instead of
            # leaking it forever (the exact bug this class exists to avoid, just for one session).
            self._registry._forget_one(self._session_id)
            raise
        self._acquired = True
        return result

    def release(self) -> None:
        self._lock.release()
        self._acquired = False
        self._registry._forget_one(self._session_id)

    def locked(self) -> bool:
        return self._lock.locked()

    async def __aenter__(self) -> "_RefCountedLock":
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.release()


class SessionLockRegistry:
    def __init__(self):
        self._locks: dict[str, asyncio.Lock] = {}
        self._refcounts: dict[str, int] = {}

    def get_lock(self, session_id: str) -> _RefCountedLock:
        lock = self._locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[session_id] = lock
        self._refcounts[session_id] = self._refcounts.get(session_id, 0) + 1
        return _RefCountedLock(lock, self, session_id)

    def _forget_one(self, session_id: str) -> None:
        """One outstanding get_lock() for this session_id is done with it (released, or failed to
        even acquire). Drops the entry once nothing else is still using it. Safe without extra
        synchronization: asyncio has no thread preemption between awaits, so this dict mutation
        can't race with another coroutine's."""
        remaining = self._refcounts.get(session_id, 1) - 1
        if remaining <= 0:
            self._refcounts.pop(session_id, None)
            self._locks.pop(session_id, None)
        else:
            self._refcounts[session_id] = remaining


_default_registry: SessionLockRegistry | None = None


def get_session_lock_registry() -> SessionLockRegistry:
    global _default_registry
    if _default_registry is None:
        _default_registry = SessionLockRegistry()
    return _default_registry
