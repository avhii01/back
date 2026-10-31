"""Customer tickets: creation, idempotent dispatch, and verified status sync.

The ticket is a durable request. It never performs recovery itself: it enqueues a
BackgroundJob, and the job invokes the existing runner/graph. Ticket status is derived
from that run plus its persisted verification - never from the model's claim.
"""
from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from relay.context import AppContext
from relay.models import (
    Approval,
    BackgroundJob,
    ChatConversation,
    ChatMessage,
    Customer,
    Order,
    Run,
    SupportTicket,
    ToolCall,
    utcnow,
)

TERMINAL = ("resolved", "escalated")
ACTIVE = ("new", "investigating", "recovering", "awaiting_approval")
RECOVERY_CATEGORIES = {"PAYMENT_PENDING", "PAYMENT_CONFLICT", "INVENTORY_ISSUE", "OUT_OF_STOCK",
                       "SHIPPING_DELAY", "REFUND_REQUEST", "GENERAL_ESCALATION"}


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def enqueue_ticket_job(session, ticket_id: str, *, max_attempts: int = 3) -> BackgroundJob:
    """Add at most one queued/running job per ticket (DB partial unique index)."""
    existing = session.scalars(select(BackgroundJob).where(
        BackgroundJob.ticket_id == ticket_id, BackgroundJob.status.in_(("queued", "running")))).first()
    if existing is not None:
        return existing
    job = BackgroundJob(id=_new_id("job"), ticket_id=ticket_id, status="queued", max_attempts=max_attempts)
    try:
        with session.begin_nested():
            session.add(job)
            session.flush()
    except IntegrityError:
        existing = session.scalars(select(BackgroundJob).where(
            BackgroundJob.ticket_id == ticket_id, BackgroundJob.status.in_(("queued", "running")))).first()
        if existing is None:
            raise
        return existing
    return job


def create_customer_ticket(app: AppContext, *, customer: Customer, order: Order, conversation_id: str,
                           category: str, confirmed_summary: str, citations: list[dict],
                           service_evidence: dict, dispatch: bool = True, status: str = "awaiting") -> tuple[SupportTicket, bool]:
    """Create (or return) the one ticket for this order/category.

    Ticket + job commit in one transaction, so a crash cannot leave a confirmed ticket
    without a durable dispatch request. Repeated confirmations return the same row.
    """
    reason = category[:40]
    with app.session() as s:
        existing = s.scalars(select(SupportTicket).where(SupportTicket.order_id == order.id,
                                                         SupportTicket.reason_code == reason)).first()
        if existing is not None:
            # Idempotent confirmation. An agent-created ticket for the same order/category
            # is linked to the customer so it appears in their ticket list.
            changed = False
            if existing.customer_id is None:
                existing.customer_id = customer.id
                changed = True
            if existing.conversation_id is None:
                existing.conversation_id = conversation_id
                changed = True
            if existing.confirmed_summary is None:
                existing.confirmed_summary = confirmed_summary
                existing.citations = citations or existing.citations
                existing.service_evidence = service_evidence or existing.service_evidence
                changed = True
            if changed:
                existing.updated_at = utcnow()
                s.commit()
            return existing, False
        evidence_data = dict(service_evidence or {})
        if "operator_status" not in evidence_data:
            evidence_data["operator_status"] = "awaiting"
        if "raised_at" not in evidence_data:
            evidence_data["raised_at"] = utcnow().isoformat()
        ticket = SupportTicket(
            id=_new_id("tkt"), order_id=order.id, customer_id=customer.id, conversation_id=conversation_id,
            reason_code=reason, category=category, priority="high" if category == "PAYMENT_CONFLICT" else "normal",
            summary=confirmed_summary, confirmed_summary=confirmed_summary, citations=citations,
            service_evidence=evidence_data, source="customer",
            assigned_to=evidence_data.get("assigned_to", "Unassigned"),
            status="awaiting" if dispatch else "escalated",
            resolution_summary=None if dispatch else "Waiting for a human specialist to review the request.",
        )
        s.add(ticket)
        try:
            s.flush()
        except IntegrityError:
            s.rollback()
            existing = s.scalars(select(SupportTicket).where(SupportTicket.order_id == order.id,
                                                             SupportTicket.reason_code == reason)).first()
            if existing is None:
                raise
            return existing, False
        if dispatch:
            enqueue_ticket_job(s, ticket.id, max_attempts=app.settings.job_max_attempts)
        s.commit()
        s.refresh(ticket)
        return ticket, True


def sync_ticket(app: AppContext, ticket_id: str) -> SupportTicket | None:
    """Derive ticket status from the linked run and its verification. Terminal is sticky."""
    with app.session() as s:
        ticket = s.get(SupportTicket, ticket_id)
        if ticket is None:
            return None
        evidence = dict(ticket.service_evidence or {})
        op_status = evidence.get("operator_status")
        if op_status:
            if ticket.status != op_status:
                ticket.status = op_status
                s.commit()
            return ticket
        if ticket.status in ("resolved", "escalated", "on_hold", "working", "awaiting_caller", "awaiting"):
            # Operator status overrides and terminal states are sticky.
            return ticket
        run = s.get(Run, ticket.run_id) if ticket.run_id else None
        if run is None:
            return ticket
        mutating = s.scalar(select(func.count()).select_from(ToolCall).where(
            ToolCall.run_id == run.id, ToolCall.mutating.is_(True))) or 0
        status = ticket.status
        summary = ticket.resolution_summary
        if run.status in ("pending", "running"):
            status = "recovering" if mutating else "investigating"
            summary = "Automated recovery is checking the order against live service records."
        elif run.status in ("paused", "awaiting_approval"):
            status = "awaiting_approval"
            summary = "Waiting for a support specialist to decide a refund request."
        elif run.status == "succeeded":
            verification = (run.state or {}).get("verification") or {}
            if verification.get("passed") and run.outcome in ("fulfilled", "refunded"):
                status = "resolved"
                summary = ("Recovery verified: the order is marked shipped." if run.outcome == "fulfilled"
                           else "Recovery verified: the refund was completed.")
            else:
                status = "escalated"
                summary = "Automated recovery finished without passing every verification check."
        else:  # failed or escalated
            status = "escalated"
            summary = "Automated recovery stopped; a support specialist is reviewing the case."
        if status != ticket.status or summary != ticket.resolution_summary:
            ticket.status = status
            ticket.resolution_summary = summary
            ticket.updated_at = utcnow()
            s.commit()
        return ticket


def sync_tickets_for_order(app: AppContext, order_id: str) -> None:
    with app.session() as s:
        ids = list(s.scalars(select(SupportTicket.id).where(
            SupportTicket.order_id == order_id, SupportTicket.source == "customer")))
    for ticket_id in ids:
        sync_ticket(app, ticket_id)


def ticket_customer_json(app: AppContext, ticket_id: str) -> dict | None:
    ticket = sync_ticket(app, ticket_id)
    if ticket is None:
        return None
    with app.session() as s:
        order = s.get(Order, ticket.order_id)
        run = s.get(Run, ticket.run_id) if ticket.run_id else None
        conversation = s.get(ChatConversation, ticket.conversation_id) if ticket.conversation_id else None
        citations = ticket.citations or []
        evidence = ticket.service_evidence or {}
    verification = ((run.state or {}).get("verification") if run else None) or {}
    verified = bool(run and run.status == "succeeded" and verification.get("passed")
                    and run.outcome in ("fulfilled", "refunded"))
    progress = [{"at": ticket.created_at.isoformat(), "status": "new", "label": "Ticket created"}]
    if run is not None:
        progress.append({"at": run.created_at.isoformat(), "status": "investigating",
                         "label": "Recovery investigation started"})
        if run.status in ("paused", "awaiting_approval"):
            progress.append({"at": run.updated_at.isoformat(), "status": "awaiting_approval",
                             "label": "Waiting for a refund approval"})
        elif ticket.status in ("recovering", "resolved", "escalated"):
            progress.append({"at": run.updated_at.isoformat(), "status": ticket.status,
                             "label": "Verified outcome" if ticket.status == "resolved" else "Case escalated"})
    return {
        "id": ticket.id, "order_id": ticket.order_id,
        "order": None if order is None else {"id": order.id, "status": order.status, "sku": order.sku,
                                             "quantity": order.quantity, "total_cents": order.total_cents,
                                             "currency": order.currency},
        "category": ticket.category or ticket.reason_code, "status": ticket.status,
        "summary": ticket.confirmed_summary or ticket.summary, "citations": citations,
        "service_evidence": evidence, "created_at": ticket.created_at.isoformat(),
        "updated_at": ticket.updated_at.isoformat(),
        "resolution_summary": ticket.resolution_summary,
        "resolution": ticket.resolution_summary if ticket.status == "resolved" else None,
        "verified": verified,
        "run": None if run is None else {"status": run.status, "outcome": run.outcome},
        "conversation_id": conversation.id if conversation else None,
        "progress": progress,
    }


def ticket_operator_json(app: AppContext, ticket_id: str) -> dict | None:
    ticket = sync_ticket(app, ticket_id)
    if ticket is None:
        return None
    with app.session() as s:
        customer = s.get(Customer, ticket.customer_id) if ticket.customer_id else None
        order = s.get(Order, ticket.order_id)
        run = s.get(Run, ticket.run_id) if ticket.run_id else None
        conversation = s.get(ChatConversation, ticket.conversation_id) if ticket.conversation_id else None
        messages = list(s.scalars(select(ChatMessage).where(
            ChatMessage.conversation_id == ticket.conversation_id).order_by(ChatMessage.id)).all()) if conversation else []
        approvals = list(s.scalars(select(Approval).where(Approval.order_id == ticket.order_id)
                                   .order_by(Approval.requested_at.desc())).all())
        pending = next((a.id for a in approvals if a.status == "pending"), None)
        verification = ((run.state or {}).get("verification") if run else None) or {}
    from relay.api import serializers as ser
    verified = bool(run and run.status == "succeeded" and verification.get("passed")
                    and run.outcome in ("fulfilled", "refunded"))
    evidence = ticket.service_evidence or {}
    dept = evidence.get("department") or get_department(ticket.category, ticket.reason_code)
    assigned = ticket.assigned_to or evidence.get("assigned_to") or "Unassigned"
    query_text = ticket.confirmed_summary or ticket.summary
    raised_at = evidence.get("raised_at") or ticket.created_at.isoformat()
    resolved_at = evidence.get("resolved_at")
    return {
        "ticket": ser.ticket_json(ticket),
        "department": dept,
        "assigned_to": assigned,
        "query": query_text,
        "raised_at": raised_at,
        "resolved_at": resolved_at,
        "operator_status": evidence.get("operator_status") or ticket.status or "awaiting",
        "resolution_notes": ticket.resolution_summary or evidence.get("resolution_notes") or "",
        "customer": None if customer is None else {"id": customer.id, "name": customer.name, "email": customer.email},
        "order": ser.order_json(order) if order else None,
        "conversation": None if conversation is None else {
            "id": conversation.id, "created_at": conversation.updated_at.isoformat(),
            "messages": [{"id": m.id, "role": m.role, "content": m.content, "citations": m.citations,
                          "meta": m.meta, "created_at": m.created_at.isoformat()} for m in messages],
        },
        "run": ser.run_detail(app, run.id) if run else None,
        "approvals": [ser.approval_json(a) for a in approvals],
        "verification": verification,
        "pending_approval_id": pending,
        "verified": verified,
    }

def get_department(category: str | None, reason_code: str | None) -> str:
    c = (category or reason_code or "").upper()
    if any(k in c for k in ("SHIP", "DELIVERY", "CARRIER", "TRACK")):
        return "Shipping & Logistics"
    if any(k in c for k in ("PAY", "CHARGE", "REFUND", "BILL")):
        return "Payments & Billing"
    if any(k in c for k in ("INVENTORY", "STOCK", "RESERV")):
        return "Inventory & Warehouse"
    return "Customer Support Operations"
