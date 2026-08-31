"""Regression tests for runtime/session_locks.py's reference-counted eviction — a plain
dict-of-locks-by-session_id that never removed an entry would leak one asyncio.Lock per distinct
session forever on a long-running server. These confirm both that mutual exclusion still works
(the actual point of the registry) and that its memory footprint tracks concurrently in-flight
sessions rather than every session ever seen."""
import asyncio

from intagrin.runtime.session_locks import SessionLockRegistry


def test_lock_entry_is_forgotten_after_a_context_managed_use():
    async def scenario():
        registry = SessionLockRegistry()
        async with registry.get_lock("s1"):
            assert len(registry._locks) == 1
        assert len(registry._locks) == 0
        assert len(registry._refcounts) == 0

    asyncio.run(scenario())


def test_lock_entry_is_forgotten_after_manual_acquire_release():
    async def scenario():
        registry = SessionLockRegistry()
        lock = registry.get_lock("s1")
        await lock.acquire()
        assert len(registry._locks) == 1
        lock.release()
        assert len(registry._locks) == 0

    asyncio.run(scenario())


def test_many_distinct_sessions_do_not_accumulate_once_finished():
    """The actual regression this exists to catch: 10k distinct sessions used one after another
    on a long-lived registry must not leave 10k lock entries behind."""
    async def scenario():
        registry = SessionLockRegistry()
        for i in range(10_000):
            async with registry.get_lock(f"session-{i}"):
                pass
        assert len(registry._locks) == 0
        assert len(registry._refcounts) == 0

    asyncio.run(scenario())


def test_mutual_exclusion_still_holds_for_the_same_session():
    """The registry's whole purpose: two concurrent 'requests' for the same session must be
    serialized, not run concurrently."""
    async def scenario():
        registry = SessionLockRegistry()
        order = []

        async def worker(name, delay):
            async with registry.get_lock("shared-session"):
                order.append(f"{name}-start")
                await asyncio.sleep(delay)
                order.append(f"{name}-end")

        await asyncio.gather(worker("a", 0.05), worker("b", 0.0))
        # Whichever ran first must fully finish before the other starts — no interleaving.
        assert order in (
            ["a-start", "a-end", "b-start", "b-end"],
            ["b-start", "b-end", "a-start", "a-end"],
        )

    asyncio.run(scenario())


def test_concurrent_use_of_the_same_session_keeps_one_entry_until_all_finish():
    """While two callers are still using the same session_id's lock (one holding, one waiting),
    the registry must not have already forgotten it out from under the waiter."""
    async def scenario():
        registry = SessionLockRegistry()
        started_second = asyncio.Event()
        release_first = asyncio.Event()

        async def first():
            async with registry.get_lock("s1"):
                started_second.set()
                await release_first.wait()

        async def second():
            await started_second.wait()
            assert len(registry._locks) == 1  # first still holds it; entry must still exist
            async with registry.get_lock("s1"):
                pass

        task1 = asyncio.create_task(first())
        task2 = asyncio.create_task(second())
        await started_second.wait()
        assert registry._refcounts["s1"] == 2  # first (holding) + second (waiting)
        release_first.set()
        await asyncio.gather(task1, task2)
        assert len(registry._locks) == 0

    asyncio.run(scenario())


def test_failed_acquire_still_forgets_its_refcount_contribution():
    """If acquire() itself raises/cancels before ever succeeding, release() will never be called
    for that get_lock() — the wrapper must drop its own refcount contribution itself rather than
    leaking it."""
    async def scenario():
        registry = SessionLockRegistry()
        lock = registry.get_lock("s1")
        # Simulate the underlying lock's acquire() raising (e.g. cancellation mid-wait).
        real_lock = registry._locks["s1"]

        async def boom():
            raise asyncio.CancelledError()

        real_lock.acquire = boom
        try:
            await lock.acquire()
        except asyncio.CancelledError:
            pass
        assert len(registry._locks) == 0
        assert len(registry._refcounts) == 0

    asyncio.run(scenario())
