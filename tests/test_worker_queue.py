import asyncio

from intagrin.runtime.worker import DistributedWorker


def test_concurrent_claims_never_double_process_a_job(tmp_path):
    """Regression test for the SQLite job-queue race: N jobs seeded, N workers claiming
    concurrently against the same file must each get exactly one distinct job — no job claimed
    twice, no job left unclaimed."""

    async def run():
        worker = DistributedWorker(project_dir=tmp_path)
        job_count = 12

        for i in range(job_count):
            worker.enqueue_local({"workflow": f"job_{i}"})

        # job_count concurrent claim attempts racing against the same queue file
        claims = await asyncio.gather(
            *(asyncio.to_thread(worker._claim_next_job) for _ in range(job_count))
        )

        claimed = [c for c in claims if c is not None]
        job_ids = [job_id for job_id, _ in claimed]
        workflow_names = {task["workflow"] for _, task in claimed}

        assert len(claimed) == job_count, "every seeded job should be claimed exactly once"
        assert len(set(job_ids)) == job_count, "no job id was claimed by more than one worker"
        assert workflow_names == {f"job_{i}" for i in range(job_count)}

        # Queue is now empty — one more claim attempt must return nothing
        extra = await asyncio.to_thread(worker._claim_next_job)
        assert extra is None

    asyncio.run(run())


def test_claim_and_complete_round_trip(tmp_path):
    async def run():
        worker = DistributedWorker(project_dir=tmp_path)
        worker.enqueue_local({"workflow": "solo_job"})

        claim = await asyncio.to_thread(worker._claim_next_job)
        assert claim is not None
        job_id, task = claim
        assert task == {"workflow": "solo_job"}

        # Still no second job available while this one is "processing"
        assert await asyncio.to_thread(worker._claim_next_job) is None

        await asyncio.to_thread(worker._mark_completed, job_id)

    asyncio.run(run())
