"""initial_schema

Revision ID: 6308869d7c5d
Revises: 
Create Date: 2026-08-21 21:43:22.820871

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '6308869d7c5d'
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    if not inspector.has_table("checkpoints"):
        op.create_table(
            "checkpoints",
            sa.Column("session_id", sa.String(255), primary_key=True),
            sa.Column("messages", sa.JSON(), nullable=True),
            sa.Column("state", sa.JSON(), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now())
        )

def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("checkpoints")
