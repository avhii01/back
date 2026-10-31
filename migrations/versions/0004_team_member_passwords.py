"""add password and password_hash to team_members

Revision ID: 0004
Revises: 0003
"""
from alembic import op
import sqlalchemy as sa

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("team_members", sa.Column("password", sa.String(length=100), nullable=True))
    op.add_column("team_members", sa.Column("password_hash", sa.String(length=200), nullable=True))


def downgrade() -> None:
    op.drop_column("team_members", "password_hash")
    op.drop_column("team_members", "password")
