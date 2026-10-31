"""Relay's own database schema.

Constraints that carry correctness guarantees (enforced by the DB, not by app code):
  * runs:               at most ONE active run per order (partial unique index)
  * idempotency_records: primary key = idempotency key
  * approvals:          at most one pending/approved refund approval per order
  * support_tickets:    one ticket per (order, reason_code)
  * background_jobs:    at most one queued/running recovery job per ticket
  * knowledge_chunks:   chunk_id is a stable citation identifier

Customer support additions are additive: the recovery engine, its tools and its
guarantees are unchanged. A customer ticket is a durable request that *dispatches*
the existing runner; it never executes recovery itself.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import TypeDecorator


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class UTCDateTime(TypeDecorator):
    """Timezone-aware UTC datetimes on every backend (SQLite drops tzinfo)."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def process_result_value(self, value, dialect):
        if value is not None and value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value


JsonType = JSON().with_variant(JSONB(), "postgresql")

ACTIVE_RUN_STATUSES = ("pending", "running", "paused", "awaiting_approval")
_ACTIVE_SQL = "status IN ('pending','running','paused','awaiting_approval')"


class Base(DeclarativeBase):
    pass


class Customer(Base):
    """A demo customer. Passwords are PBKDF2-hashed; sessions are server-side tokens."""

    __tablename__ = "customers"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    email: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(120))
    password_hash: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class CustomerSession(Base):
    """Only a SHA-256 hash of the bearer token is stored; the raw token is shown once."""

    __tablename__ = "customer_sessions"

    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(UTCDateTime, index=True)


class Order(Base):
    __tablename__ = "orders"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    customer_id: Mapped[str | None] = mapped_column(ForeignKey("customers.id"), nullable=True, index=True)
    customer_email: Mapped[str] = mapped_column(String(200))
    sku: Mapped[str] = mapped_column(String(80))
    quantity: Mapped[int] = mapped_column(Integer)
    total_cents: Mapped[int] = mapped_column(Integer)  # money is ALWAYS integer cents
    currency: Mapped[str] = mapped_column(String(3), default="USD")
    status: Mapped[str] = mapped_column(String(20), index=True)  # pending|paid|reserved|shipped|refunded
    payment_ref: Mapped[str | None] = mapped_column(String(80), nullable=True)
    reservation_id: Mapped[str | None] = mapped_column(String(80), nullable=True)
    shipment_id: Mapped[str | None] = mapped_column(String(80), nullable=True)
    refund_id: Mapped[str | None] = mapped_column(String(80), nullable=True)
    # UNTRUSTED free text (customer-entered / legacy workflow error). Never used by policy.
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    workflow_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Demo-only label shown in the dashboard. The agent never reads it.
    scenario_label: Mapped[str | None] = mapped_column(String(80), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow)


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    order_id: Mapped[str] = mapped_column(ForeignKey("orders.id"), index=True)
    correlation_id: Mapped[str] = mapped_column(String(64), index=True)
    mode: Mapped[str] = mapped_column(String(10))  # demo | llm  (fixed at creation)
    status: Mapped[str] = mapped_column(String(20), index=True)
    step_mode: Mapped[bool] = mapped_column(Boolean, default=False)
    state: Mapped[dict] = mapped_column(JsonType, default=dict)  # LangGraph state, persisted per node
    outcome: Mapped[str | None] = mapped_column(String(20), nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    steps: Mapped[int] = mapped_column(Integer, default=0)
    active_ms: Mapped[int] = mapped_column(Integer, default=0)
    lease_owner: Mapped[str | None] = mapped_column(String(64), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)

    __table_args__ = (
        Index(
            "ux_runs_one_active_per_order",
            "order_id",
            unique=True,
            postgresql_where=text(_ACTIVE_SQL),
            sqlite_where=text(_ACTIVE_SQL),
        ),
    )


class ToolCall(Base):
    """One row per tool *attempt* - this is the timeline shown in the dashboard."""

    __tablename__ = "tool_calls"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    seq: Mapped[int] = mapped_column(Integer)
    node: Mapped[str | None] = mapped_column(String(30), nullable=True)
    tool: Mapped[str] = mapped_column(String(60))
    mutating: Mapped[bool] = mapped_column(Boolean, default=False)
    purpose: Mapped[str | None] = mapped_column(String(30), nullable=True)  # e.g. "reconcile"
    attempt: Mapped[int] = mapped_column(Integer, default=1)
    idempotency_key: Mapped[str | None] = mapped_column(String(200), nullable=True)
    status: Mapped[str] = mapped_column(String(12))  # ok | error | ambiguous | replayed
    args: Mapped[dict | None] = mapped_column(JsonType, nullable=True)
    result: Mapped[dict | None] = mapped_column(JsonType, nullable=True)
    error: Mapped[dict | None] = mapped_column(JsonType, nullable=True)
    started_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)

    __table_args__ = (UniqueConstraint("run_id", "seq", name="uq_tool_calls_run_seq"),)


class IdempotencyRecord(Base):
    """Intent + outcome of every mutation. The primary key IS the idempotency key."""

    __tablename__ = "idempotency_records"

    key: Mapped[str] = mapped_column(String(200), primary_key=True)
    tool: Mapped[str] = mapped_column(String(60))
    order_id: Mapped[str] = mapped_column(String(40), index=True)
    status: Mapped[str] = mapped_column(String(12))  # in_progress | succeeded | failed
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    run_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    result: Mapped[dict | None] = mapped_column(JsonType, nullable=True)
    error: Mapped[dict | None] = mapped_column(JsonType, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow)


class Approval(Base):
    __tablename__ = "approvals"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    order_id: Mapped[str] = mapped_column(ForeignKey("orders.id"), index=True)
    run_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    kind: Mapped[str] = mapped_column(String(20), default="refund")
    amount_cents: Mapped[int] = mapped_column(Integer)
    currency: Mapped[str] = mapped_column(String(3), default="USD")
    status: Mapped[str] = mapped_column(String(12))  # pending|approved|rejected|expired
    reason: Mapped[str] = mapped_column(Text)
    evidence: Mapped[dict | None] = mapped_column(JsonType, nullable=True)
    requested_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(UTCDateTime)
    decided_by: Mapped[str | None] = mapped_column(String(80), nullable=True)
    decided_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    decision_note: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index(
            "ux_approvals_one_open_per_order_kind",
            "order_id",
            "kind",
            unique=True,
            postgresql_where=text("status IN ('pending','approved')"),
            sqlite_where=text("status IN ('pending','approved')"),
        ),
    )


class SupportTicket(Base):
    """One ticket per (order, reason_code).

    `source="agent"` tickets are the recovery engine's escalations. `source="customer"`
    tickets are created by the support chat after the customer confirms a summary; they
    dispatch the existing runner and are kept in sync with it.
    """

    __tablename__ = "support_tickets"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    order_id: Mapped[str] = mapped_column(ForeignKey("orders.id"), index=True)
    customer_id: Mapped[str | None] = mapped_column(ForeignKey("customers.id"), nullable=True, index=True)
    conversation_id: Mapped[str | None] = mapped_column(ForeignKey("chat_conversations.id"), nullable=True,
                                                        index=True)
    run_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    reason_code: Mapped[str] = mapped_column(String(40))
    category: Mapped[str | None] = mapped_column(String(40), nullable=True)
    priority: Mapped[str] = mapped_column(String(10), default="normal")
    summary: Mapped[str] = mapped_column(Text)
    confirmed_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    citations: Mapped[list | None] = mapped_column(JsonType, nullable=True)
    evidence: Mapped[dict | None] = mapped_column(JsonType, nullable=True)
    service_evidence: Mapped[dict | None] = mapped_column(JsonType, nullable=True)
    resolution_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(String(20), default="agent")
    status: Mapped[str] = mapped_column(String(20), default="open", index=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow)

    __table_args__ = (UniqueConstraint("order_id", "reason_code", name="uq_ticket_order_reason"),)


class ChatConversation(Base):
    __tablename__ = "chat_conversations"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.id"), index=True)
    status: Mapped[str] = mapped_column(String(20), default="open")
    # Persisted slot-filling state: stage, order id, draft ticket, last ticket id.
    state: Mapped[dict] = mapped_column(JsonType, default=dict)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow)


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("chat_conversations.id"), index=True)
    role: Mapped[str] = mapped_column(String(12))  # customer | assistant | system
    content: Mapped[str] = mapped_column(Text)
    citations: Mapped[list | None] = mapped_column(JsonType, nullable=True)
    meta: Mapped[dict | None] = mapped_column(JsonType, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class KnowledgeChunk(Base):
    """Ingested, chunked policy text. The embedding is stored as JSON for the DB fallback."""

    __tablename__ = "knowledge_chunks"

    chunk_id: Mapped[str] = mapped_column(String(160), primary_key=True)
    document_id: Mapped[str] = mapped_column(String(80), index=True)
    title: Mapped[str] = mapped_column(String(200))
    version: Mapped[str] = mapped_column(String(40))
    section: Mapped[str] = mapped_column(String(200))
    content: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64))
    embedding: Mapped[list | None] = mapped_column(JsonType, nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(80), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow)


class BackgroundJob(Base):
    """Durable ticket-dispatch queue. The chat request only enqueues; a worker runs recovery."""

    __tablename__ = "background_jobs"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    kind: Mapped[str] = mapped_column(String(30), default="ticket_recovery")
    ticket_id: Mapped[str] = mapped_column(ForeignKey("support_tickets.id"), index=True)
    run_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    status: Mapped[str] = mapped_column(String(12), default="queued", index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    lease_owner: Mapped[str | None] = mapped_column(String(64), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow)

    __table_args__ = (
        Index(
            "ux_jobs_one_open_per_ticket",
            "ticket_id",
            unique=True,
            postgresql_where=text("status IN ('queued','running')"),
            sqlite_where=text("status IN ('queued','running')"),
        ),
    )


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, index=True)
    correlation_id: Mapped[str] = mapped_column(String(64), index=True)
    run_id: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)
    order_id: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)
    actor: Mapped[str] = mapped_column(String(20))  # agent | llm | policy | tool | human | system
    event_type: Mapped[str] = mapped_column(String(40))
    payload: Mapped[dict | None] = mapped_column(JsonType, nullable=True)


class LLMCall(Base):
    """Real provider-reported usage. Rows exist only when an LLM call was actually made."""

    __tablename__ = "llm_calls"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    node: Mapped[str] = mapped_column(String(30))
    model: Mapped[str] = mapped_column(String(80))
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class TeamMember(Base):
    """A member of the operational support team who resolves customer tickets."""

    __tablename__ = "team_members"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    email: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    role: Mapped[str] = mapped_column(String(80))
    department: Mapped[str] = mapped_column(String(120))
    status: Mapped[str] = mapped_column(String(20), default="active")
    active_tickets_count: Mapped[int] = mapped_column(Integer, default=0)
    password: Mapped[str | None] = mapped_column(String(100), nullable=True)
    password_hash: Mapped[str | None] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
