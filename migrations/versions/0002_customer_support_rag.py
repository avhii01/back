"""customer support: customers, conversations, rich tickets, durable jobs, knowledge chunks

Revision ID: 0002
Revises: 0001
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

JSON = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "customers",
        sa.Column("id", sa.String(length=40), nullable=False),
        sa.Column("email", sa.String(length=200), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("password_hash", sa.String(length=200), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_customers_email", "customers", ["email"], unique=True)

    op.create_table(
        "customer_sessions",
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("customer_id", sa.String(length=40), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["customer_id"], ["customers.id"]),
        sa.PrimaryKeyConstraint("token_hash"),
    )
    op.create_index("ix_customer_sessions_customer_id", "customer_sessions", ["customer_id"], unique=False)
    op.create_index("ix_customer_sessions_expires_at", "customer_sessions", ["expires_at"], unique=False)

    with op.batch_alter_table("orders", schema=None) as batch_op:
        batch_op.add_column(sa.Column("customer_id", sa.String(length=40), nullable=True))
        batch_op.create_foreign_key("fk_orders_customer_id_customers", "customers", ["customer_id"], ["id"])
        batch_op.create_index("ix_orders_customer_id", ["customer_id"], unique=False)

    op.create_table(
        "chat_conversations",
        sa.Column("id", sa.String(length=40), nullable=False),
        sa.Column("customer_id", sa.String(length=40), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("state", JSON, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["customer_id"], ["customers.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_chat_conversations_customer_id", "chat_conversations", ["customer_id"], unique=False)

    with op.batch_alter_table("support_tickets", schema=None) as batch_op:
        batch_op.add_column(sa.Column("customer_id", sa.String(length=40), nullable=True))
        batch_op.add_column(sa.Column("conversation_id", sa.String(length=40), nullable=True))
        batch_op.add_column(sa.Column("category", sa.String(length=40), nullable=True))
        batch_op.add_column(sa.Column("confirmed_summary", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("citations", JSON, nullable=True))
        batch_op.add_column(sa.Column("service_evidence", JSON, nullable=True))
        batch_op.add_column(sa.Column("resolution_summary", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("source", sa.String(length=20), nullable=False,
                                      server_default=sa.text("'agent'")))
        batch_op.add_column(sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False,
                                      server_default=sa.func.now()))
        batch_op.alter_column("status", existing_type=sa.String(length=12), type_=sa.String(length=20),
                              existing_nullable=False)
        batch_op.create_foreign_key("fk_support_tickets_customer_id_customers", "customers",
                                    ["customer_id"], ["id"])
        batch_op.create_foreign_key("fk_support_tickets_conversation_id_chat_conversations",
                                    "chat_conversations", ["conversation_id"], ["id"])
        batch_op.create_index("ix_support_tickets_customer_id", ["customer_id"], unique=False)
        batch_op.create_index("ix_support_tickets_conversation_id", ["conversation_id"], unique=False)
        batch_op.create_index("ix_support_tickets_status", ["status"], unique=False)

    op.create_table(
        "chat_messages",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("conversation_id", sa.String(length=40), nullable=False),
        sa.Column("role", sa.String(length=12), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("citations", JSON, nullable=True),
        sa.Column("meta", JSON, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["conversation_id"], ["chat_conversations.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_chat_messages_conversation_id", "chat_messages", ["conversation_id"], unique=False)

    op.create_table(
        "knowledge_chunks",
        sa.Column("chunk_id", sa.String(length=160), nullable=False),
        sa.Column("document_id", sa.String(length=80), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("version", sa.String(length=40), nullable=False),
        sa.Column("section", sa.String(length=200), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("embedding", JSON, nullable=True),
        sa.Column("embedding_model", sa.String(length=80), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("chunk_id"),
    )
    op.create_index("ix_knowledge_chunks_document_id", "knowledge_chunks", ["document_id"], unique=False)

    op.create_table(
        "background_jobs",
        sa.Column("id", sa.String(length=40), nullable=False),
        sa.Column("kind", sa.String(length=30), nullable=False),
        sa.Column("ticket_id", sa.String(length=40), nullable=False),
        sa.Column("run_id", sa.String(length=40), nullable=True),
        sa.Column("status", sa.String(length=12), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("max_attempts", sa.Integer(), nullable=False),
        sa.Column("lease_owner", sa.String(length=64), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["ticket_id"], ["support_tickets.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_background_jobs_ticket_id", "background_jobs", ["ticket_id"], unique=False)
    op.create_index("ix_background_jobs_status", "background_jobs", ["status"], unique=False)
    op.create_index(
        "ux_jobs_one_open_per_ticket", "background_jobs", ["ticket_id"], unique=True,
        postgresql_where=sa.text("status IN ('queued','running')"),
        sqlite_where=sa.text("status IN ('queued','running')"),
    )


def downgrade() -> None:
    op.drop_index("ux_jobs_one_open_per_ticket", table_name="background_jobs",
                  postgresql_where=sa.text("status IN ('queued','running')"),
                  sqlite_where=sa.text("status IN ('queued','running')"))
    op.drop_index("ix_background_jobs_status", table_name="background_jobs")
    op.drop_index("ix_background_jobs_ticket_id", table_name="background_jobs")
    op.drop_table("background_jobs")

    op.drop_index("ix_knowledge_chunks_document_id", table_name="knowledge_chunks")
    op.drop_table("knowledge_chunks")

    op.drop_index("ix_chat_messages_conversation_id", table_name="chat_messages")
    op.drop_table("chat_messages")

    with op.batch_alter_table("support_tickets", schema=None) as batch_op:
        batch_op.drop_index("ix_support_tickets_status")
        batch_op.drop_index("ix_support_tickets_conversation_id")
        batch_op.drop_index("ix_support_tickets_customer_id")
        batch_op.drop_constraint("fk_support_tickets_conversation_id_chat_conversations", type_="foreignkey")
        batch_op.drop_constraint("fk_support_tickets_customer_id_customers", type_="foreignkey")
        batch_op.alter_column("status", existing_type=sa.String(length=20), type_=sa.String(length=12),
                              existing_nullable=False)
        batch_op.drop_column("updated_at")
        batch_op.drop_column("source")
        batch_op.drop_column("resolution_summary")
        batch_op.drop_column("service_evidence")
        batch_op.drop_column("citations")
        batch_op.drop_column("confirmed_summary")
        batch_op.drop_column("category")
        batch_op.drop_column("conversation_id")
        batch_op.drop_column("customer_id")

    op.drop_index("ix_chat_conversations_customer_id", table_name="chat_conversations")
    op.drop_table("chat_conversations")

    with op.batch_alter_table("orders", schema=None) as batch_op:
        batch_op.drop_index("ix_orders_customer_id")
        batch_op.drop_constraint("fk_orders_customer_id_customers", type_="foreignkey")
        batch_op.drop_column("customer_id")

    op.drop_index("ix_customer_sessions_expires_at", table_name="customer_sessions")
    op.drop_index("ix_customer_sessions_customer_id", table_name="customer_sessions")
    op.drop_table("customer_sessions")

    op.drop_index("ix_customers_email", table_name="customers")
    op.drop_table("customers")
