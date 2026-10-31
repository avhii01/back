"""Operator ticket queue. Internal detail is allowed here, never on the customer API."""
from __future__ import annotations

from datetime import timedelta
from pydantic import BaseModel, Field

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from relay.api import serializers as ser
from relay.context import AppContext
from relay.models import Approval, Customer, Order, Run, SupportTicket, utcnow
from relay.support.tickets import enqueue_ticket_job, sync_ticket, ticket_operator_json

class UpdateTicketStatusBody(BaseModel):
    status: str | None = Field(default=None, pattern="^(awaiting|blank|working|on_hold|hold|resolved|resolve|escalated|awaiting_caller|waiting_for_caller_confirmation)$")
    simulate_days_on_hold: int | None = None
    assigned_to: str | None = None
    department: str | None = None
    resolution_notes: str | None = None


ACTIVE_TICKET_STATUSES = ("new", "investigating", "recovering", "awaiting_approval")


def create_operator_support_router(ctx: AppContext) -> APIRouter:
    router = APIRouter(prefix="/api/tickets", tags=["tickets"])

    @router.get("")
    def list_tickets(status: str | None = None, limit: int = 100):
        with ctx.session() as s:
            ids = list(s.scalars(select(SupportTicket.id).order_by(SupportTicket.updated_at.desc())
                                 .limit(min(max(limit, 1), 200))))
        out = []
        for ticket_id in ids:
            ticket = sync_ticket(ctx, ticket_id)
            if ticket is None:
                continue
            evidence = ticket.service_evidence or {}
            op_status = evidence.get("operator_status") or ticket.status or "awaiting"
            if op_status in ("blank", "new", "new_ticket"):
                op_status = "awaiting"
            if status:
                clean_target = status.lower().strip()
                if clean_target in ("new", "new_ticket", "new tickets", "new ticket"):
                    clean_target = "awaiting"
                if op_status != clean_target and ticket.status != clean_target:
                    continue
            with ctx.session() as s:
                customer = s.get(Customer, ticket.customer_id) if ticket.customer_id else None
                order = s.get(Order, ticket.order_id)
                run = s.get(Run, ticket.run_id) if ticket.run_id else None
                pending = s.scalars(select(Approval.id).where(
                    Approval.order_id == ticket.order_id, Approval.status == "pending")).first()
                verification = ((run.state or {}).get("verification") if run else None) or {}
            from relay.support.tickets import get_department
            evidence = ticket.service_evidence or {}
            dept = evidence.get("department") or get_department(ticket.category, ticket.reason_code)
            assigned = ticket.assigned_to or evidence.get("assigned_to") or "Unassigned"
            query_text = ticket.confirmed_summary or ticket.summary
            raised_at = evidence.get("raised_at") or ticket.created_at.isoformat()
            resolved_at = evidence.get("resolved_at")
            out.append({
                "ticket": ser.ticket_json(ticket),
                "customer": None if customer is None else {"id": customer.id, "name": customer.name,
                                                           "email": customer.email},
                "order": ser.order_json(order) if order else None,
                "run": ser.run_brief(run) if run else None,
                "pending_approval_id": pending,
                "verified": bool(run and run.status == "succeeded" and verification.get("passed")
                                 and run.outcome in ("fulfilled", "refunded")),
                "department": dept,
                "assigned_to": assigned,
                "query": query_text,
                "raised_at": raised_at,
                "resolved_at": resolved_at,
                "operator_status": evidence.get("operator_status") or ticket.status or "awaiting",
                "resolution_notes": ticket.resolution_summary or evidence.get("resolution_notes") or "",
            })
        return out

    @router.get("/{ticket_id}")
    def get_ticket(ticket_id: str):
        detail = ticket_operator_json(ctx, ticket_id)
        if detail is None:
            raise HTTPException(404, "ticket not found")
        return detail

    @router.post("/{ticket_id}/dispatch")
    def dispatch(ticket_id: str):
        ticket = sync_ticket(ctx, ticket_id)
        if ticket is None:
            raise HTTPException(404, "ticket not found")
        if ticket.status == "resolved":
            raise HTTPException(409, "ticket is resolved")
        with ctx.session() as s:
            row = s.get(SupportTicket, ticket_id)
            job = enqueue_ticket_job(s, ticket_id, max_attempts=ctx.settings.job_max_attempts)
            row.status = "investigating"
            row.resolution_summary = None
            s.commit()
        return {"ticket_id": ticket_id, "job_id": job.id, "queued": True}

    @router.post("/{ticket_id}/status")
    def update_status(ticket_id: str, body: UpdateTicketStatusBody):
        with ctx.session() as s:
            row = s.get(SupportTicket, ticket_id)
            if row is None:
                raise HTTPException(404, "ticket not found")

            evidence = dict(row.service_evidence or {})

            # Once resolved, it cannot be reopened or edited!
            if evidence.get("operator_status") == "resolved" and body.status and body.status.lower() not in ("resolve", "resolved"):
                raise HTTPException(400, "Once a ticket is resolved, it cannot be reopened or changed.")

            if body.status:
                clean_status = body.status.lower()
                if clean_status in ("hold", "on_hold"):
                    clean_status = "on_hold"
                elif clean_status in ("resolve", "resolved"):
                    clean_status = "resolved"
                    if not evidence.get("resolved_at"):
                        evidence["resolved_at"] = utcnow().isoformat()
                elif clean_status in ("awaiting_caller", "waiting_for_caller_confirmation", "waiting for caller", "waiting_for_caller"):
                    clean_status = "awaiting_caller"
                elif clean_status in ("awaiting", "blank", "new", "new_ticket", "new ticket"):
                    clean_status = "awaiting"
                elif clean_status == "escalated":
                    clean_status = "escalated"
                else:
                    clean_status = "working"

                evidence["operator_status"] = clean_status
                row.status = clean_status

            if body.assigned_to is not None:
                clean_assignee = body.assigned_to.strip() or "Unassigned"
                evidence["assigned_to"] = clean_assignee
                row.assigned_to = clean_assignee
            if body.department is not None:
                evidence["department"] = body.department.strip()
            if body.resolution_notes is not None:
                notes_clean = body.resolution_notes.strip()
                row.resolution_summary = notes_clean
                evidence["resolution_notes"] = notes_clean

            row.service_evidence = evidence

            if body.simulate_days_on_hold is not None and body.simulate_days_on_hold > 0:
                row.updated_at = utcnow() - timedelta(days=body.simulate_days_on_hold)
            else:
                row.updated_at = utcnow()
            s.commit()
            s.refresh(row)
        return ticket_operator_json(ctx, ticket_id)

    return router
