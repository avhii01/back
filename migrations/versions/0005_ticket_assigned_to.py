"""add assigned_to column to support_tickets

Revision ID: 0005
Revises: 0004
"""
from alembic import op
import sqlalchemy as sa

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("support_tickets", sa.Column("assigned_to", sa.String(length=120), server_default="Unassigned", nullable=True))
    op.create_index("ix_support_tickets_assigned_to", "support_tickets", ["assigned_to"])
    # Backfill existing assigned_to from service_evidence JSON if available
    op.execute("""
        UPDATE support_tickets
        SET assigned_to = COALESCE(service_evidence->>'assigned_to', 'Unassigned')
        WHERE service_evidence IS NOT NULL
    """)


def downgrade() -> None:
    op.drop_index("ix_support_tickets_assigned_to", table_name="support_tickets")
    op.drop_column("support_tickets", "assigned_to")
