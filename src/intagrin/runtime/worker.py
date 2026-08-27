import asyncio
import json
import sqlite3
from pathlib import Path

from rich.console import Console

from ..compiler.parser import parse_project
from ..runtime.engine import RuntimeEngine
from ..runtime.shared_resources import SharedResourcesCache

console = Console()

DEFAULT_MAX_RETRIES = 3


class DistributedWorker:
    """
    Background worker process for IntaGrin tasks and workflow pipelines. Uses a Redis queue
    (BRPOPLPUSH reliable-queue pattern) when `redis_url` is configured, otherwise a persistent
    local SQLite queue.

    A job that raises is retried up to `max_retries` times (requeued), then moved to a terminal
    'dead_letter' state — never silently marked 'completed'. Previously any exception inside
    process_task() was swallowed to a console print and the job was still marked 'completed'
    regardless of outcome, so a failed background workflow looked identical to a successful one
    in the job_queue table.
    """

    def __init__(
        self,
        project_dir: Path,
        queue_name: str = "intagrin:tasks",
        redis_url: str | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        self.project_dir = project_dir
        self.queue_name = queue_name
        self.redis_url = redis_url
        self.max_retries = max_retries
        self.running = False
        self.db_path = self.project_dir / ".ai" / "tasks.db"
        # Pooled across every process_task() call for this worker's lifetime — a long-running
        # worker processing many tasks would otherwise reconnect MCP servers, rebuild the RAG
        # index, and reload tool schemas/prompts from scratch on every single job.
        self._resources_cache = SharedResourcesCache()

    async def stop(self):
        """Graceful shutdown: stop claiming new jobs and tear down pooled MCP connections."""
        self.running = False
        await self._resources_cache.cleanup()

    async def start(self):
        self.running = True
        console.print(
            f"[bold purple]⚡ IntaGrin Worker online for queue: '{self.queue_name}'[/bold purple]"
        )
        console.print(
            "[dim]Listening for incoming background workflow tasks and evaluation jobs...[/dim]\n"
        )

        try:
            if self.redis_url:
                try:
                    import redis.asyncio as aioredis

                    client = aioredis.from_url(self.redis_url, decode_responses=True)
                    console.print(
                        f"[bold cyan]Connected to distributed Redis queue: {self.redis_url}[/bold cyan]"
                    )
                    await self._run_redis_loop(client)
                except ImportError:
                    console.print(
                        "[bold red]Redis client not installed. Falling back to local queue mode.[/bold red]"
                    )
                    await self._run_local_loop()
            else:
                await self._run_local_loop()
        finally:
            # Runs on graceful stop(), an unhandled task error, or KeyboardInterrupt — always in
            # this same event loop, before asyncio.run() unwinds it.
            await self._resources_cache.cleanup()

    # ---------------------------------------------------------------- local (SQLite) queue ----

    async def _run_local_loop(self):
        self._init_local_queue()
        console.print(
            f"[bold cyan]Local SQLite persistent queue initialized at: {self.db_path}[/bold cyan]"
        )

        while self.running:
            claim = await asyncio.to_thread(self._claim_next_job)
            if claim:
                job_id, task = claim
                try:
                    await self.process_task(task)
                except Exception as e:
                    console.print(f"[bold red]✗ Worker Task Failed: {e}[/bold red]")
                    await asyncio.to_thread(self._mark_failed, job_id, str(e))
                else:
                    await asyncio.to_thread(self._mark_completed, job_id)
            else:
                await asyncio.sleep(1)

    def _init_local_queue(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path, timeout=15.0) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS job_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    payload TEXT,
                    status TEXT DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            # ALTER TABLE ... ADD COLUMN has no "IF NOT EXISTS" in SQLite before 3.35, and this
            # table predates these columns — guard each against a pre-existing tasks.db that
            # already has the old, narrower schema.
            for ddl in (
                "ALTER TABLE job_queue ADD COLUMN attempts INTEGER DEFAULT 0",
                "ALTER TABLE job_queue ADD COLUMN error TEXT",
                "ALTER TABLE job_queue ADD COLUMN updated_at TIMESTAMP",
            ):
                try:
                    conn.execute(ddl)
                except sqlite3.OperationalError:
                    pass  # column already exists
            conn.commit()

    def _claim_next_job(self) -> tuple[int, dict] | None:
        """Atomically claims the oldest pending job via UPDATE ... RETURNING inside a single
        BEGIN IMMEDIATE transaction. The WHERE re-checks status='pending' at commit time, so two
        workers racing on the same row can never both "win" it, and RETURNING hands back exactly
        the row *this* call claimed — never a different worker's row, which a separate follow-up
        SELECT could race on. Safe to call concurrently from multiple worker processes/threads
        against the same SQLite file."""
        with sqlite3.connect(self.db_path, timeout=15.0) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    """
                    UPDATE job_queue SET status = 'processing'
                    WHERE id = (
                        SELECT id FROM job_queue WHERE status = 'pending' ORDER BY id ASC LIMIT 1
                    ) AND status = 'pending'
                    RETURNING id, payload
                    """
                ).fetchone()
                conn.commit()
            except Exception:
                conn.rollback()
                raise

        if not row:
            return None
        job_id, payload_str = row
        return job_id, json.loads(payload_str)

    def _mark_completed(self, job_id: int):
        with sqlite3.connect(self.db_path, timeout=15.0) as conn:
            conn.execute(
                "UPDATE job_queue SET status = 'completed', updated_at = CURRENT_TIMESTAMP "
                "WHERE id = ?",
                (job_id,),
            )
            conn.commit()

    def _mark_failed(self, job_id: int, error: str):
        """Requeues the job (status back to 'pending') if it still has retries left, otherwise
        moves it to the terminal 'dead_letter' status — a failed job is never silently
        indistinguishable from a completed one."""
        with sqlite3.connect(self.db_path, timeout=15.0) as conn:
            row = conn.execute(
                "SELECT attempts FROM job_queue WHERE id = ?", (job_id,)
            ).fetchone()
            attempts = (row[0] or 0) + 1 if row else 1
            next_status = "pending" if attempts <= self.max_retries else "dead_letter"
            conn.execute(
                "UPDATE job_queue SET status = ?, attempts = ?, error = ?, "
                "updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (next_status, attempts, error, job_id),
            )
            conn.commit()

    def enqueue_local(self, payload: dict):
        """Inserts a job directly into the local SQLite queue (used by tests and by callers that
        want to seed work without going through an external producer)."""
        self._init_local_queue()
        with sqlite3.connect(self.db_path, timeout=15.0) as conn:
            conn.execute(
                "INSERT INTO job_queue (payload) VALUES (?)", (json.dumps(payload),)
            )
            conn.commit()

    # ------------------------------------------------------------------------ Redis queue ----

    async def _run_redis_loop(self, client):
        """Reliable-queue pattern: BRPOPLPUSH atomically moves a job from the main queue into a
        per-worker `:processing` list, so a crash between pop and completion leaves the job
        recoverable there instead of vanishing (the previous plain BLPOP was destructive — a
        crash mid-job lost it irrecoverably, with no ack/retry of any kind). `attempts` travels
        inside the job's own JSON payload since Redis list items carry no side metadata."""
        processing_key = f"{self.queue_name}:processing"
        dead_letter_key = f"{self.queue_name}:dead_letter"

        while self.running:
            payload_str = await client.brpoplpush(self.queue_name, processing_key, timeout=2)
            if not payload_str:
                continue
            job = json.loads(payload_str)
            task = job.get("task", job)
            attempts = job.get("_attempts", 0)
            try:
                await self.process_task(task)
            except Exception as e:
                console.print(f"[bold red]✗ Worker Task Failed: {e}[/bold red]")
                attempts += 1
                if attempts <= self.max_retries:
                    requeued = json.dumps({"task": task, "_attempts": attempts})
                    await client.lpush(self.queue_name, requeued)
                else:
                    dead = json.dumps({"task": task, "_attempts": attempts, "error": str(e)})
                    await client.lpush(dead_letter_key, dead)
            finally:
                await client.lrem(processing_key, 1, payload_str)

    async def process_task(self, task: dict):
        """Runs one workflow to completion. Raises on failure — callers (the SQLite and Redis
        loops above) are responsible for retry/dead-letter bookkeeping; this method used to
        swallow every exception into a console print and return normally, which is why a failed
        job was previously indistinguishable from a successful one."""
        workflow_name = task.get("workflow")
        session_id = task.get("session_id", "worker_session")
        initial_state = task.get("state", {})

        console.print(
            f"[bold green]▶ Processing workflow '{workflow_name}' for session '{session_id}'...[/bold green]"
        )
        graph = parse_project(self.project_dir)
        shared = await self._resources_cache.get(graph, self.project_dir)
        engine = RuntimeEngine(
            graph=graph,
            project_dir=self.project_dir,
            session_id=session_id,
            initial_state=initial_state,
            shared_resources=shared,
        )
        await engine.initialize()
        await engine.run_workflow(workflow_name)
        console.print(f"[bold green]✓ Completed workflow '{workflow_name}'[/bold green]")
