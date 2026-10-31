from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def now() -> datetime:
    return datetime.now(timezone.utc)


class SimBase(DeclarativeBase):
    pass


class SimPayment(SimBase):
    __tablename__ = "sim_payments"
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    order_id: Mapped[str] = mapped_column(String(40), index=True)
    amount_cents: Mapped[int] = mapped_column(Integer)
    currency: Mapped[str] = mapped_column(String(3), default="USD")
    status: Mapped[str] = mapped_column(String(20))  # captured|authorized|failed|refunded
    memo: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class SimPaymentEvent(SimBase):
    __tablename__ = "sim_payment_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    order_id: Mapped[str] = mapped_column(String(40), index=True)
    payment_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    type: Mapped[str] = mapped_column(String(20))  # captured|failed|voided|chargeback|refunded
    amount_cents: Mapped[int] = mapped_column(Integer)
    source: Mapped[str] = mapped_column(String(20), default="webhook")
    memo: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class SimStock(SimBase):
    __tablename__ = "sim_stock"
    sku: Mapped[str] = mapped_column(String(80), primary_key=True)
    available: Mapped[int] = mapped_column(Integer)
    reserved: Mapped[int] = mapped_column(Integer, default=0)


class SimReservation(SimBase):
    __tablename__ = "sim_reservations"
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    order_id: Mapped[str] = mapped_column(String(40), index=True)
    sku: Mapped[str] = mapped_column(String(80))
    quantity: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(12), default="active")  # active|released
    idempotency_key: Mapped[str | None] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    __table_args__ = (UniqueConstraint("idempotency_key", name="uq_sim_res_idem"),)


class SimShipment(SimBase):
    __tablename__ = "sim_shipments"
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    order_id: Mapped[str] = mapped_column(String(40), index=True)
    status: Mapped[str] = mapped_column(String(12), default="created")  # created|cancelled
    carrier: Mapped[str] = mapped_column(String(40), default="SIM-CARRIER")
    tracking: Mapped[str] = mapped_column(String(40))
    idempotency_key: Mapped[str | None] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    __table_args__ = (UniqueConstraint("idempotency_key", name="uq_sim_ship_idem"),)


class SimRefund(SimBase):
    __tablename__ = "sim_refunds"
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    payment_id: Mapped[str] = mapped_column(String(40))
    order_id: Mapped[str] = mapped_column(String(40), index=True)
    amount_cents: Mapped[int] = mapped_column(Integer)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    idempotency_key: Mapped[str | None] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_sim_refund_idem"),
        UniqueConstraint("payment_id", name="uq_sim_refund_one_per_payment"),
    )


class SimFault(SimBase):
    """Failure injection. `remaining=-1` means 'forever'."""

    __tablename__ = "sim_faults"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    operation: Mapped[str] = mapped_column(String(30))  # reserve|create_shipment|refund|get_*
    order_id: Mapped[str] = mapped_column(String(40), default="*")
    mode: Mapped[str] = mapped_column(String(24))  # error_503|timeout_before_commit|timeout_after_commit
    remaining: Mapped[int] = mapped_column(Integer, default=1)
    delay_ms: Mapped[int] = mapped_column(Integer, default=0)
