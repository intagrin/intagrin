"""add_run_logs

Revision ID: ccfa6ed12026
Revises: 6308869d7c5d
Create Date: 2026-08-23 00:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'ccfa6ed12026'
down_revision: str | Sequence[str] | None = '6308869d7c5d'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    if not inspector.has_table("run_logs"):
        op.create_table(
            "run_logs",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("session_id", sa.String(255), nullable=True),
            sa.Column("endpoint", sa.Text(), nullable=True),
            sa.Column("agent", sa.Text(), nullable=True),
            sa.Column("status", sa.Text(), nullable=True),
            sa.Column("error", sa.Text(), nullable=True),
            sa.Column("tokens_delta", sa.Integer(), nullable=True),
            sa.Column("cost_delta", sa.Float(), nullable=True),
            sa.Column("total_tokens", sa.Integer(), nullable=True),
            sa.Column("total_cost", sa.Float(), nullable=True),
            sa.Column("message_count", sa.Integer(), nullable=True),
            sa.Column("latency_ms", sa.Integer(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        )

def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("run_logs")
