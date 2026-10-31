"""Tool implementations.

Two kinds of mutation, with different guarantees (see docs/ARCHITECTURE.md):

  LOCAL  (update_order_status, request_refund_approval, create_support_ticket)
         The business write, the idempotency record and the audit event commit in ONE database
         transaction - they either all happen or none do.

  REMOTE (reserve_inventory, create_shipment, execute_approved_refund)
         The effect happens in an independent service; no transaction spans both systems.
         We get at-least-once delivery + provider-side dedupe via Idempotency-Key, and recover
         from ambiguity by *looking the effect up* (the `reconcile_*` functions below).
"""
from __future__ import annotations

import uuid
from datetime import timedelta

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from relay.errors import PermanentToolError
from relay.models import Approval, Order, SupportTicket, utcnow
from relay.money import refundable_amount
from relay.tools import schemas as S
from relay.tools.base import ToolContext
from relay.untrusted import detect_injection, sanitize_text

TRANSITIONS = {
    ("pending", "paid"): "payment_id",
    ("paid", "reserved"): "reservation_id",
    ("reserved", "shipped"): "shipment_id",
    ("paid", "refunded"): "refund_id",
    ("reserved", "refunded"): "refund_id",
}
_REF_COLUMN = {"payment_id": "payment_ref", "reservation_id": "reservation_id",
               "shipment_id": "shipment_id", "refund_id": "refund_id"}


def _load_order(ctx: ToolContext, order_id: str) -> Order:
    order = ctx.session.get(Order, order_id)
    if order is None:
        raise PermanentToolError(f"order {order_id} not found", code="ORDER_NOT_FOUND")
    return order


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


# ============================================================== READ TOOLS
def get_order_status(ctx: ToolContext, args: S.OrderIn) -> S.OrderStatusOut:
    s = ctx.session
    o = _load_order(ctx, args.order_id)
    ap = s.scalars(select(Approval).where(Approval.order_id == o.id, Approval.kind == "refund")
                   .order_by(Approval.requested_at.desc())).first()
    if ap is not None and ap.status == "pending" and ap.expires_at < utcnow():
        # lazy expiry: an approval nobody answered in time can no longer authorise anything
        s.execute(update(Approval).where(Approval.id == ap.id, Approval.status == "pending").values(status="expired"))
        s.refresh(ap)
    tickets = s.scalars(select(SupportTicket.reason_code).where(
        SupportTicket.order_id == o.id,
        SupportTicket.status.in_(("open", "new", "investigating", "recovering", "awaiting_approval",
                                  "escalated")))).all()
    return S.OrderStatusOut(
        order_id=o.id, status=o.status, sku=o.sku, quantity=o.quantity, total_cents=o.total_cents,
        currency=o.currency, payment_ref=o.payment_ref, reservation_id=o.reservation_id,
        shipment_id=o.shipment_id, refund_id=o.refund_id,
        customer_note=sanitize_text(o.notes), workflow_error=sanitize_text(o.workflow_error),
        injection_suspected=bool(detect_injection(o.notes) or detect_injection(o.workflow_error)),
        approval=None if ap is None else S.ApprovalView(
            approval_id=ap.id, status=ap.status, amount_cents=ap.amount_cents, currency=ap.currency,
            expires_at=ap.expires_at.isoformat(), decided_by=ap.decided_by),
        open_ticket_reasons=list(tickets),
    )


def get_payment_status(ctx: ToolContext, args: S.OrderIn) -> S.PaymentStatusOut:
    r = ctx.app.sim.get_payments(args.order_id, correlation_id=ctx.correlation_id, timeout=ctx.timeout_s)
    return S.PaymentStatusOut(
        order_id=r.order_id,
        payments=[S.PaymentItem(payment_id=p.id, status=p.status, amount_cents=p.amount_cents,
                                currency=p.currency, memo=p.memo) for p in r.payments],
        events=[S.PaymentEventItem(payment_id=e.payment_id, type=e.type, amount_cents=e.amount_cents,
                                   source=e.source, memo=e.memo) for e in r.events],
        refunds=[S.RefundItem(refund_id=x.id, payment_id=x.payment_id, amount_cents=x.amount_cents)
                 for x in r.refunds])


def get_inventory_reservation(ctx: ToolContext, args: S.OrderIn) -> S.InventoryReservationOut:
    r = ctx.app.sim.get_reservations(args.order_id, correlation_id=ctx.correlation_id, timeout=ctx.timeout_s)
    return S.InventoryReservationOut(
        order_id=r.order_id,
        reservations=[S.ReservationItem(reservation_id=x.id, sku=x.sku, quantity=x.quantity, status=x.status)
                      for x in r.reservations])


def check_stock(ctx: ToolContext, args: S.CheckStockIn) -> S.CheckStockOut:
    r = ctx.app.sim.get_stock(args.sku, correlation_id=ctx.correlation_id, timeout=ctx.timeout_s)
    return S.CheckStockOut(sku=r.sku, available=r.available, requested=args.quantity,
                           sufficient=r.available >= args.quantity)


def get_shipment_status(ctx: ToolContext, args: S.OrderIn) -> S.ShipmentStatusOut:
    r = ctx.app.sim.get_shipments(args.order_id, correlation_id=ctx.correlation_id, timeout=ctx.timeout_s)
    return S.ShipmentStatusOut(
        order_id=r.order_id,
        shipments=[S.ShipmentItem(shipment_id=x.id, status=x.status, carrier=x.carrier, tracking=x.tracking)
                   for x in r.shipments])


# ============================================================== LOCAL MUTATIONS
def update_order_status(ctx: ToolContext, args: S.UpdateOrderStatusIn) -> S.UpdateOrderStatusOut:
    """Compare-and-set state transition. Deterministic validation lives here, not in the model."""
    s = ctx.session
    order = s.get(Order, args.order_id, with_for_update=True)
    if order is None:
        raise PermanentToolError(f"order {args.order_id} not found", code="ORDER_NOT_FOUND")
    if order.status == args.new_status:
        return S.UpdateOrderStatusOut(order_id=order.id, status=order.status, changed=False)
    if order.status != args.expected_from:
        raise PermanentToolError(f"order is '{order.status}', expected '{args.expected_from}'", code="STALE_STATE")
    ref_field = TRANSITIONS.get((args.expected_from, args.new_status))
    if ref_field is None:
        raise PermanentToolError(f"transition {args.expected_from}->{args.new_status} is not allowed",
                                 code="INVALID_TRANSITION")
    ref_value = getattr(args, ref_field)
    if not ref_value:
        raise PermanentToolError(f"{ref_field} is required for {args.new_status}", code="MISSING_REFERENCE")
    res = s.execute(update(Order).where(Order.id == order.id, Order.status == args.expected_from)
                    .values(status=args.new_status, **{_REF_COLUMN[ref_field]: ref_value}))
    if res.rowcount != 1:
        raise PermanentToolError("order changed concurrently", code="STALE_STATE")
    return S.UpdateOrderStatusOut(order_id=order.id, status=args.new_status, changed=True)


def _replay_valid_update(ctx: ToolContext, args: S.UpdateOrderStatusIn) -> bool:
    order = ctx.session.get(Order, args.order_id)
    return order is not None and order.status == args.new_status


def request_refund_approval(ctx: ToolContext, args: S.RequestRefundApprovalIn) -> S.RequestRefundApprovalOut:
    """Creates a PENDING approval. The amount is computed here from trusted data - never by the model."""
    s = ctx.session
    order = _load_order(ctx, args.order_id)
    pays = ctx.app.sim.get_payments(order.id, correlation_id=ctx.correlation_id, timeout=ctx.timeout_s)
    captured = [p for p in pays.payments if p.status == "captured"]
    if len(captured) != 1:
        raise PermanentToolError("refund requires exactly one captured payment", code="PAYMENT_NOT_REFUNDABLE")
    refunded = sum(r.amount_cents for r in pays.refunds)
    try:
        amount = refundable_amount(captured_cents=captured[0].amount_cents, refunded_cents=refunded,
                                   order_total_cents=order.total_cents)
    except ValueError as e:
        raise PermanentToolError(str(e), code="REFUND_AMOUNT_INVALID") from e
    approval = Approval(id=_new_id("apr"), order_id=order.id, run_id=ctx.run_id, kind="refund", amount_cents=amount,
                        currency=order.currency, status="pending", reason=sanitize_text(args.reason, 400) or "",
                        evidence=args.evidence,
                        expires_at=utcnow() + timedelta(seconds=ctx.app.settings.approval_ttl_seconds))
    try:
        with s.begin_nested():
            s.add(approval)
    except IntegrityError:
        # partial unique index: an open approval already exists -> idempotent return of it
        existing = s.scalars(select(Approval).where(Approval.order_id == order.id, Approval.kind == "refund",
                                                     Approval.status.in_(["pending", "approved"]))).first()
        if existing is None:
            raise
        approval = existing
    return S.RequestRefundApprovalOut(approval_id=approval.id, status=approval.status,
                                      amount_cents=approval.amount_cents, currency=approval.currency,
                                      expires_at=approval.expires_at.isoformat())


def create_support_ticket(ctx: ToolContext, args: S.CreateSupportTicketIn) -> S.CreateSupportTicketOut:
    s = ctx.session
    _load_order(ctx, args.order_id)
    # A customer ticket already owns this conversation/run. Escalating must update that
    # ticket, not create a second customer-visible ticket with a different reason code.
    if ctx.run_id:
        linked = s.scalars(select(SupportTicket).where(SupportTicket.run_id == ctx.run_id,
                                                       SupportTicket.source == "customer")).first()
        if linked is not None:
            linked.status = "escalated"
            linked.resolution_summary = sanitize_text(args.summary, 500) or "Escalated for human review."
            linked.evidence = args.evidence or linked.evidence
            linked.updated_at = utcnow()
            s.flush()
            return S.CreateSupportTicketOut(ticket_id=linked.id, reason_code=args.reason_code, created=False)
    ticket = SupportTicket(id=_new_id("tkt"), order_id=args.order_id, run_id=ctx.run_id, reason_code=args.reason_code,
                           category=args.reason_code, priority=args.priority,
                           summary=sanitize_text(args.summary, 500) or "", evidence=args.evidence,
                           source="agent")
    try:
        with s.begin_nested():
            s.add(ticket)
        return S.CreateSupportTicketOut(ticket_id=ticket.id, reason_code=ticket.reason_code, created=True)
    except IntegrityError:
        existing = s.scalars(select(SupportTicket).where(SupportTicket.order_id == args.order_id,
                                                          SupportTicket.reason_code == args.reason_code)).first()
        if existing is None:
            raise
        return S.CreateSupportTicketOut(ticket_id=existing.id, reason_code=existing.reason_code, created=False)


# ============================================================== REMOTE MUTATIONS
def reserve_inventory(ctx: ToolContext, args: S.ReserveInventoryIn) -> S.ReserveInventoryOut:
    order = _load_order(ctx, args.order_id)
    if order.sku != args.sku or order.quantity != args.quantity:
        raise PermanentToolError("sku/quantity do not match the order", code="INPUT_MISMATCH")
    r = ctx.app.sim.create_reservation(order_id=order.id, sku=order.sku, quantity=order.quantity,
                                       idem_key=ctx.idempotency_key or "", correlation_id=ctx.correlation_id,
                                       timeout=ctx.timeout_s)
    return S.ReserveInventoryOut(reservation_id=r.id, order_id=r.order_id, sku=r.sku, quantity=r.quantity,
                                 status=r.status, reused_by_provider=r.replayed)


def reconcile_reserve_inventory(ctx: ToolContext, args: S.ReserveInventoryIn) -> S.ReserveInventoryOut | None:
    r = ctx.app.sim.get_reservations(args.order_id, correlation_id=ctx.correlation_id, timeout=ctx.timeout_s)
    active = [x for x in r.reservations if x.status == "active" and x.sku == args.sku and x.quantity == args.quantity]
    if len(active) != 1:
        return None  # not found (or ambiguous: retry with the same key; provider dedupes)
    x = active[0]
    return S.ReserveInventoryOut(reservation_id=x.id, order_id=x.order_id, sku=x.sku, quantity=x.quantity,
                                 status=x.status, reused_by_provider=True)


def create_shipment(ctx: ToolContext, args: S.CreateShipmentIn) -> S.CreateShipmentOut:
    order = _load_order(ctx, args.order_id)
    r = ctx.app.sim.create_shipment(order_id=order.id, sku=order.sku, quantity=order.quantity,
                                    idem_key=ctx.idempotency_key or "", correlation_id=ctx.correlation_id,
                                    timeout=ctx.timeout_s)
    return S.CreateShipmentOut(shipment_id=r.id, order_id=r.order_id, status=r.status, tracking=r.tracking,
                               reused_by_provider=r.replayed)


def reconcile_create_shipment(ctx: ToolContext, args: S.CreateShipmentIn) -> S.CreateShipmentOut | None:
    r = ctx.app.sim.get_shipments(args.order_id, correlation_id=ctx.correlation_id, timeout=ctx.timeout_s)
    live = [x for x in r.shipments if x.status != "cancelled"]
    if len(live) != 1:
        return None
    x = live[0]
    return S.CreateShipmentOut(shipment_id=x.id, order_id=x.order_id, status=x.status, tracking=x.tracking,
                               reused_by_provider=True)


def execute_approved_refund(ctx: ToolContext, args: S.ExecuteApprovedRefundIn) -> S.ExecuteApprovedRefundOut:
    """Defence in depth: the approval is re-read from the DB HERE, so even a buggy caller
    (or a manipulated model) cannot trigger a refund without a persisted human approval."""
    ap = ctx.session.get(Approval, args.approval_id)
    if ap is None or ap.order_id != args.order_id:
        raise PermanentToolError("approval not found for this order", code="APPROVAL_NOT_FOUND")
    if ap.status != "approved" or not ap.decided_by:
        raise PermanentToolError(f"approval is '{ap.status}', not approved", code="APPROVAL_NOT_GRANTED")
    pays = ctx.app.sim.get_payments(args.order_id, correlation_id=ctx.correlation_id, timeout=ctx.timeout_s)
    captured = [p for p in pays.payments if p.status == "captured"]
    if len(captured) != 1 or captured[0].amount_cents != ap.amount_cents:
        raise PermanentToolError("payment state no longer matches the approval", code="PAYMENT_MISMATCH")
    r = ctx.app.sim.create_refund(payment_id=captured[0].id, amount_cents=ap.amount_cents,
                                  reason=f"approved refund {ap.id}", idem_key=ctx.idempotency_key or "",
                                  correlation_id=ctx.correlation_id, timeout=ctx.timeout_s)
    return S.ExecuteApprovedRefundOut(refund_id=r.id, payment_id=r.payment_id, amount_cents=r.amount_cents,
                                      approval_id=ap.id, reused_by_provider=r.replayed)


def reconcile_execute_approved_refund(ctx: ToolContext, args: S.ExecuteApprovedRefundIn) -> S.ExecuteApprovedRefundOut | None:
    ap = ctx.session.get(Approval, args.approval_id)
    if ap is None:
        return None
    pays = ctx.app.sim.get_payments(args.order_id, correlation_id=ctx.correlation_id, timeout=ctx.timeout_s)
    matching = [r for r in pays.refunds if r.amount_cents == ap.amount_cents]
    if len(matching) != 1:
        return None
    x = matching[0]
    return S.ExecuteApprovedRefundOut(refund_id=x.id, payment_id=x.payment_id, amount_cents=x.amount_cents,
                                      approval_id=ap.id, reused_by_provider=True)


# ============================================================== VERIFICATION
def verify_recovery(ctx: ToolContext, args: S.VerifyRecoveryIn) -> S.VerifyRecoveryOut:
    """Re-reads LIVE state from every system and checks business invariants.
    It does not trust the agent's own memory of what it did."""
    s = ctx.session
    order = _load_order(ctx, args.order_id)
    sim, cid, to = ctx.app.sim, ctx.correlation_id, ctx.timeout_s
    pays = sim.get_payments(order.id, correlation_id=cid, timeout=to)
    resv = sim.get_reservations(order.id, correlation_id=cid, timeout=to)
    ships = sim.get_shipments(order.id, correlation_id=cid, timeout=to)

    captured = [p for p in pays.payments if p.status == "captured"]
    money_moved = [p for p in pays.payments if p.status in ("captured", "refunded")]
    active = [r for r in resv.reservations if r.status == "active"]
    live_ships = [x for x in ships.shipments if x.status != "cancelled"]
    refunds = pays.refunds
    approved = s.scalars(select(Approval).where(Approval.order_id == order.id, Approval.status == "approved")).all()
    tickets = s.scalars(select(SupportTicket).where(SupportTicket.order_id == order.id)).all()

    checks: list[S.VerificationCheck] = []

    def check(cid_: str, desc: str, ok: bool, detail: str = "") -> None:
        checks.append(S.VerificationCheck(id=cid_, description=desc, passed=bool(ok), detail=detail))

    refund_authorised = all(any(a.amount_cents == r.amount_cents and a.decided_by for a in approved) for r in refunds)

    if args.expectation == "fulfilled":
        check("order_shipped", "Order status is 'shipped'", order.status == "shipped", f"status={order.status}")
        check("customer_charged_once", "Customer was charged exactly once for the order total",
              len(money_moved) == 1 and len(captured) == 1 and captured[0].amount_cents == order.total_cents,
              f"captures={len(captured)}, money_moved_records={len(money_moved)}")
        check("single_active_reservation", "Exactly one active reservation matching sku and quantity",
              len(active) == 1 and active[0].sku == order.sku and active[0].quantity == order.quantity,
              f"active_reservations={len(active)}")
        check("single_shipment", "Exactly one live shipment exists", len(live_ships) == 1,
              f"live_shipments={len(live_ships)}")
        check("order_references_match", "Order references the real reservation and shipment",
              len(active) == 1 and len(live_ships) == 1 and order.reservation_id == active[0].id
              and order.shipment_id == live_ships[0].id)
        check("no_refund_issued", "No refund was issued for a fulfilled order", len(refunds) == 0,
              f"refunds={len(refunds)}")
    elif args.expectation == "refunded":
        refunded_pays = [p for p in pays.payments if p.status == "refunded"]
        check("order_refunded", "Order status is 'refunded'", order.status == "refunded", f"status={order.status}")
        check("single_full_refund", "Exactly one refund equal to the order total",
              len(refunds) == 1 and refunds[0].amount_cents == order.total_cents, f"refunds={len(refunds)}")
        check("payment_marked_refunded", "The single payment is marked refunded and none remain captured",
              len(refunded_pays) == 1 and len(captured) == 0)
        check("refund_was_authorised", "Refund matches a persisted human approval", len(refunds) == 1 and refund_authorised)
        check("nothing_reserved_or_shipped", "No active reservation and no live shipment remain",
              len(active) == 0 and len(live_ships) == 0, f"active={len(active)}, shipments={len(live_ships)}")
        check("order_refund_reference", "Order references the real refund",
              len(refunds) == 1 and order.refund_id == refunds[0].id)
    else:  # escalated
        check("ticket_exists", "A support ticket exists for this order", len(tickets) > 0, f"tickets={len(tickets)}")
        check("no_unauthorised_refund", "No refund exists without a matching human approval", refund_authorised,
              f"refunds={len(refunds)}, approvals_granted={len(approved)}")
        check("order_state_consistent", "Order is not marked complete without the matching side effects",
              not (order.status == "shipped" and not live_ships) and not (order.status == "refunded" and not refunds))

    return S.VerifyRecoveryOut(order_id=order.id, expectation=args.expectation,
                               passed=all(c.passed for c in checks), checks=checks)
