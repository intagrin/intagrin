"""add_run_logs_index

Revision ID: d1a2b3c4e5f6
Revises: ccfa6ed12026
Create Date: 2026-08-23 00:00:01.000000

"""
from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'd1a2b3c4e5f6'
down_revision: str | Sequence[str] | None = 'ccfa6ed12026'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Speeds up the per-caller rate limiter's queries (runtime/rate_limiter.py), which filter
    run_logs by session_id prefix and a created_at window on every /chat, /chat/stream, /resume,
    and /stream call once server.rate_limit is configured."""
    op.create_index(
        "ix_run_logs_session_id_created_at",
        "run_logs",
        ["session_id", "created_at"],
        unique=False,
        if_not_exists=True,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_run_logs_session_id_created_at", table_name="run_logs", if_exists=True)
