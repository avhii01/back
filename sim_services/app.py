"""Simulated payment, inventory and shipping APIs + admin / failure-injection API.

Behaviours deliberately mimic real provider semantics that make recovery hard:
  * mutations accept an Idempotency-Key; replays return the original resource
  * faults can fire BEFORE the effect (503 / timeout_before_commit: nothing happened) or
    AFTER the effect is committed (timeout_after_commit: the 'lost response' case)
  * a payment can have several records and webhook events that disagree
"""
from __future__ import annotations

import time
import uuid
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import delete, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from relay.db import make_engine, make_session_factory
from sim_services.models import (
    SimBase,
    SimFault,
    SimPayment,
    SimPaymentEvent,
    SimRefund,
    SimReservation,
    SimShipment,
    SimStock,
)


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _iso(dt) -> str | None:
    return dt.isoformat() if dt else None


# ---------------------------------------------------------------- serialisers
def payment_json(p: SimPayment) -> dict:
    return {"id": p.id, "order_id": p.order_id, "amount_cents": p.amount_cents, "currency": p.currency,
            "status": p.status, "memo": p.memo, "created_at": _iso(p.created_at)}


def event_json(e: SimPaymentEvent) -> dict:
    return {"id": e.id, "payment_id": e.payment_id, "type": e.type, "amount_cents": e.amount_cents,
            "source": e.source, "memo": e.memo, "created_at": _iso(e.created_at)}


def reservation_json(r: SimReservation, replayed: bool = False) -> dict:
    return {"simulated": True, "id": r.id, "order_id": r.order_id, "sku": r.sku, "quantity": r.quantity,
            "status": r.status, "replayed": replayed, "created_at": _iso(r.created_at)}


def shipment_json(s: SimShipment, replayed: bool = False) -> dict:
    return {"simulated": True, "id": s.id, "order_id": s.order_id, "status": s.status, "carrier": s.carrier,
            "tracking": s.tracking, "replayed": replayed, "created_at": _iso(s.created_at)}


def refund_json(r: SimRefund, replayed: bool = False) -> dict:
    return {"simulated": True, "id": r.id, "payment_id": r.payment_id, "order_id": r.order_id,
            "amount_cents": r.amount_cents, "replayed": replayed, "created_at": _iso(r.created_at)}


# ---------------------------------------------------------------- request models
class ReserveBody(BaseModel):
    order_id: str
    sku: str
    quantity: int = Field(gt=0)


class ShipmentBody(BaseModel):
    order_id: str
    sku: str | None = None
    quantity: int | None = None


class RefundBody(BaseModel):
    amount_cents: int = Field(gt=0)
    reason: str | None = None


class PaymentSeed(BaseModel):
    status: Literal["captured", "authorized", "failed", "refunded"]
    amount_cents: int
    currency: str = "USD"
    memo: str | None = None


class EventSeed(BaseModel):
    type: str
    amount_cents: int
    payment_index: int | None = None
    source: str = "webhook"
    memo: str | None = None


class StockSeed(BaseModel):
    sku: str
    available: int


class ReservationSeed(BaseModel):
    sku: str
    quantity: int
    status: Literal["active", "released"] = "active"


class ShipmentSeed(BaseModel):
    status: Literal["created", "cancelled"] = "created"


class FaultSeed(BaseModel):
    operation: str
    mode: Literal["error_503", "timeout_before_commit", "timeout_after_commit"]
    remaining: int = 1
    delay_ms: int = 0
    order_id: str = "*"


class SeedBody(BaseModel):
    payments: list[PaymentSeed] = []
    events: list[EventSeed] = []
    stock: StockSeed | None = None
    reservations: list[ReservationSeed] = []
    shipments: list[ShipmentSeed] = []
    faults: list[FaultSeed] = []


# ---------------------------------------------------------------- fault injection
def take_fault(db: Session, operation: str, order_id: str) -> SimFault | None:
    """Atomically consume one unit of a matching fault (if any)."""
    rows = db.execute(
        select(SimFault)
        .where(SimFault.operation == operation, or_(SimFault.order_id == order_id, SimFault.order_id == "*"))
        .where(or_(SimFault.remaining > 0, SimFault.remaining == -1))
        .order_by(SimFault.id)
    ).scalars().all()
    for f in rows:
        if f.remaining == -1:
            return f
        res = db.execute(update(SimFault).where(SimFault.id == f.id, SimFault.remaining > 0)
                         .values(remaining=SimFault.remaining - 1))
        db.commit()
        if res.rowcount == 1:
            return f
    return None


def fault_before(db: Session, operation: str, order_id: str) -> SimFault | None:
    f = take_fault(db, operation, order_id)
    if f is None:
        return None
    if f.mode == "error_503":
        raise HTTPException(503, {"error": "service_unavailable", "simulated": True})
    if f.mode == "timeout_before_commit":
        if f.delay_ms:
            time.sleep(f.delay_ms / 1000)
        raise HTTPException(504, {"error": "gateway_timeout", "simulated": True})
    return f  # timeout_after_commit is applied by the caller after committing


def fault_after(f: SimFault | None) -> None:
    if f is not None and f.mode == "timeout_after_commit":
        if f.delay_ms:
            time.sleep(f.delay_ms / 1000)  # the client has likely given up by now
        else:
            raise HTTPException(504, {"error": "gateway_timeout_after_commit", "simulated": True})


# ---------------------------------------------------------------- app
def create_sim_app(database_url: str, *, create_tables: bool = True) -> FastAPI:
    engine = make_engine(database_url)
    session_factory = make_session_factory(engine)
    if create_tables:
        SimBase.metadata.create_all(engine)

    app = FastAPI(title="Relay SIMULATED services", version="0.1.0")
    app.state.engine = engine

    @app.middleware("http")
    async def label_simulated(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Relay-Simulated"] = "true"
        return response

    def get_db():
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    @app.get("/health")
    def health():
        return {"status": "ok", "simulated": True}

    # ------------------------------------------------------------ payments
    @app.get("/v1/payments/orders/{order_id}")
    def get_payments(order_id: str, db: Session = Depends(get_db)):
        fault_before(db, "get_payments", order_id)
        pays = db.scalars(select(SimPayment).where(SimPayment.order_id == order_id).order_by(SimPayment.created_at, SimPayment.id)).all()
        evs = db.scalars(select(SimPaymentEvent).where(SimPaymentEvent.order_id == order_id).order_by(SimPaymentEvent.id)).all()
        refunds = db.scalars(select(SimRefund).where(SimRefund.order_id == order_id)).all()
        return {"simulated": True, "order_id": order_id, "payments": [payment_json(p) for p in pays],
                "events": [event_json(e) for e in evs], "refunds": [refund_json(r) for r in refunds]}

    @app.post("/v1/payments/{payment_id}/refunds")
    def refund_payment(payment_id: str, body: RefundBody, idem: str = Header(..., alias="Idempotency-Key"),
                       db: Session = Depends(get_db)):
        pay = db.get(SimPayment, payment_id)
        if pay is None:
            raise HTTPException(404, {"error": "payment_not_found"})
        f = fault_before(db, "refund", pay.order_id)
        existing = db.scalar(select(SimRefund).where(SimRefund.idempotency_key == idem))
        if existing:
            return refund_json(existing, replayed=True)
        if pay.status != "captured":
            raise HTTPException(409, {"error": "payment_not_refundable", "status": pay.status})
        if body.amount_cents > pay.amount_cents:
            raise HTTPException(422, {"error": "refund_exceeds_capture"})
        refund = SimRefund(id=_id("re"), payment_id=pay.id, order_id=pay.order_id, amount_cents=body.amount_cents,
                           reason=body.reason, idempotency_key=idem)
        db.add(refund)
        pay.status = "refunded"
        db.add(SimPaymentEvent(order_id=pay.order_id, payment_id=pay.id, type="refunded",
                               amount_cents=body.amount_cents, source="ledger"))
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            existing = db.scalar(select(SimRefund).where(SimRefund.idempotency_key == idem))
            if existing:
                return refund_json(existing, replayed=True)
            raise HTTPException(409, {"error": "already_refunded"}) from None
        fault_after(f)
        return refund_json(refund)

    # ------------------------------------------------------------ inventory
    @app.get("/v1/inventory/stock/{sku}")
    def get_stock(sku: str, db: Session = Depends(get_db)):
        fault_before(db, "get_stock", "*")
        s = db.get(SimStock, sku)
        return {"simulated": True, "sku": sku, "available": s.available if s else 0, "reserved": s.reserved if s else 0}

    @app.get("/v1/inventory/reservations")
    def list_reservations(order_id: str, db: Session = Depends(get_db)):
        fault_before(db, "get_reservations", order_id)
        rows = db.scalars(select(SimReservation).where(SimReservation.order_id == order_id).order_by(SimReservation.created_at, SimReservation.id)).all()
        return {"simulated": True, "order_id": order_id, "reservations": [reservation_json(r) for r in rows]}

    @app.post("/v1/inventory/reservations")
    def create_reservation(body: ReserveBody, idem: str = Header(..., alias="Idempotency-Key"),
                           db: Session = Depends(get_db)):
        f = fault_before(db, "reserve", body.order_id)
        existing = db.scalar(select(SimReservation).where(SimReservation.idempotency_key == idem))
        if existing:
            if existing.order_id != body.order_id:
                raise HTTPException(422, {"error": "idempotency_key_reused_for_different_request"})
            return reservation_json(existing, replayed=True)
        dec = db.execute(update(SimStock).where(SimStock.sku == body.sku, SimStock.available >= body.quantity)
                         .values(available=SimStock.available - body.quantity, reserved=SimStock.reserved + body.quantity))
        if dec.rowcount == 0:
            db.rollback()
            raise HTTPException(409, {"error": "insufficient_stock", "sku": body.sku})
        res = SimReservation(id=_id("rsv"), order_id=body.order_id, sku=body.sku, quantity=body.quantity,
                             idempotency_key=idem)
        db.add(res)
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            existing = db.scalar(select(SimReservation).where(SimReservation.idempotency_key == idem))
            if existing:
                return reservation_json(existing, replayed=True)
            raise
        fault_after(f)
        return reservation_json(res)

    # ------------------------------------------------------------ shipping
    @app.get("/v1/shipping/shipments")
    def list_shipments(order_id: str, db: Session = Depends(get_db)):
        fault_before(db, "get_shipments", order_id)
        rows = db.scalars(select(SimShipment).where(SimShipment.order_id == order_id).order_by(SimShipment.created_at, SimShipment.id)).all()
        return {"simulated": True, "order_id": order_id, "shipments": [shipment_json(s) for s in rows]}

    @app.post("/v1/shipping/shipments")
    def create_shipment(body: ShipmentBody, idem: str = Header(..., alias="Idempotency-Key"),
                        db: Session = Depends(get_db)):
        f = fault_before(db, "create_shipment", body.order_id)
        existing = db.scalar(select(SimShipment).where(SimShipment.idempotency_key == idem))
        if existing:
            return shipment_json(existing, replayed=True)
        ship = SimShipment(id=_id("shp"), order_id=body.order_id, tracking="SIMTRK" + uuid.uuid4().hex[:8].upper(),
                           idempotency_key=idem)
        db.add(ship)
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            existing = db.scalar(select(SimShipment).where(SimShipment.idempotency_key == idem))
            if existing:
                return shipment_json(existing, replayed=True)
            raise
        fault_after(f)
        return shipment_json(ship)

    # ------------------------------------------------------------ admin / injection
    @app.post("/admin/reset")
    def reset():
        SimBase.metadata.drop_all(engine)
        SimBase.metadata.create_all(engine)
        return {"ok": True}

    @app.post("/admin/orders/{order_id}/seed")
    def seed(order_id: str, body: SeedBody, db: Session = Depends(get_db)):
        out: dict = {"payments": [], "reservations": [], "shipments": []}
        for p in body.payments:
            pay = SimPayment(id=_id("pay"), order_id=order_id, amount_cents=p.amount_cents, currency=p.currency,
                             status=p.status, memo=p.memo)
            db.add(pay)
            out["payments"].append(pay.id)
        db.flush()
        for e in body.events:
            pid = out["payments"][e.payment_index] if e.payment_index is not None else None
            db.add(SimPaymentEvent(order_id=order_id, payment_id=pid, type=e.type, amount_cents=e.amount_cents,
                                   source=e.source, memo=e.memo))
        if body.stock:
            stock = db.get(SimStock, body.stock.sku)
            if stock is None:
                stock = SimStock(sku=body.stock.sku, available=body.stock.available, reserved=0)
                db.add(stock)
            else:
                stock.available = body.stock.available
        for r in body.reservations:
            res = SimReservation(id=_id("rsv"), order_id=order_id, sku=r.sku, quantity=r.quantity, status=r.status,
                                 idempotency_key=f"legacy-{uuid.uuid4().hex[:12]}")
            db.add(res)
            out["reservations"].append(res.id)
            if r.status == "active":
                stock = db.get(SimStock, r.sku)
                if stock is not None:
                    stock.reserved += r.quantity
        for s in body.shipments:
            ship = SimShipment(id=_id("shp"), order_id=order_id, status=s.status,
                               tracking="SIMTRK" + uuid.uuid4().hex[:8].upper(),
                               idempotency_key=f"legacy-{uuid.uuid4().hex[:12]}")
            db.add(ship)
            out["shipments"].append(ship.id)
        for f in body.faults:
            db.add(SimFault(operation=f.operation, order_id=order_id if f.order_id == "*" else f.order_id, mode=f.mode,
                            remaining=f.remaining, delay_ms=f.delay_ms))
        db.commit()
        return {"simulated": True, **out}

    @app.post("/admin/faults")
    def add_fault(body: FaultSeed, db: Session = Depends(get_db)):
        f = SimFault(**body.model_dump())
        db.add(f)
        db.commit()
        return {"id": f.id}

    @app.delete("/admin/faults")
    def clear_faults(db: Session = Depends(get_db)):
        db.execute(delete(SimFault))
        db.commit()
        return {"ok": True}

    @app.get("/admin/orders/{order_id}/state")
    def ground_truth(order_id: str, db: Session = Depends(get_db)):
        """Ground truth for tests and the evaluation judge - NOT available to the agent's tools."""
        pays = db.scalars(select(SimPayment).where(SimPayment.order_id == order_id)).all()
        res = db.scalars(select(SimReservation).where(SimReservation.order_id == order_id)).all()
        ships = db.scalars(select(SimShipment).where(SimShipment.order_id == order_id)).all()
        refs = db.scalars(select(SimRefund).where(SimRefund.order_id == order_id)).all()
        skus = {r.sku for r in res}
        stock = {s: db.get(SimStock, s) for s in skus}
        return {
            "simulated": True,
            "payments": [payment_json(p) for p in pays],
            "reservations": [reservation_json(r) for r in res],
            "shipments": [shipment_json(s) for s in ships],
            "refunds": [refund_json(r) for r in refs],
            "stock": {k: {"available": v.available, "reserved": v.reserved} for k, v in stock.items() if v},
        }

    return app
