import os
from pathlib import Path

from alembic import command
from alembic.config import Config
from rich.console import Console

from intagrin.compiler.parser import parse_project
from intagrin.runtime.memory import resolve_postgres_sqlalchemy_url

console = Console()

def run_auto_migrations():
    try:
        project_dir = Path.cwd()
        try:
            graph = parse_project(project_dir)
        except Exception as parse_e:
            console.print(f"[bold yellow]Skipping auto-migrations: Could not parse project config ({parse_e})[/bold yellow]")
            return

        mem_cfg = graph.config.memory
        
        if mem_cfg.type == "sqlite":
            db_path = project_dir / (mem_cfg.db_path or ".ai/memory.db")
            # A fresh `inta new` scaffold has no .ai/ directory yet — nothing creates it until
            # the first session is saved (SQLiteCheckpointer.__init__ does this lazily, per
            # request). Auto-migrations run at server startup, before any request has happened,
            # so without this SQLAlchemy fails with "unable to open database file" on a truly
            # fresh project's very first `inta serve`/`inta monitor` run.
            db_path.parent.mkdir(parents=True, exist_ok=True)
            url = f"sqlite:///{db_path}"
        elif mem_cfg.type == "postgres":
            url = mem_cfg.connection_url
            if not url and mem_cfg.env_var:
                url = os.environ.get(mem_cfg.env_var)
            if not url:
                url = os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL")
            if not url:
                console.print("[bold yellow]No PostgreSQL URL found. Skipping auto-migrations.[/bold yellow]")
                return
            url = resolve_postgres_sqlalchemy_url(url)
        else:
            # Custom or other memory types do not support alembic migrations natively
            return
            
        migrations_dir = Path(__file__).parent
        alembic_cfg = Config(str(migrations_dir / "alembic.ini"))
        alembic_cfg.set_main_option("script_location", str(migrations_dir))
        alembic_cfg.set_main_option("sqlalchemy.url", url)
        
        console.print(f"[bold cyan]Running automatic database migrations for {mem_cfg.type}...[/bold cyan]")
        command.upgrade(alembic_cfg, "head")
    except Exception as e:
        console.print(f"[bold red]Auto-Migration Error: {e}[/bold red]")
