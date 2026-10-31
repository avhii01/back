"""Response shaping. Nothing here can leak secrets: Settings.public() omits them and
tool args/results were already passed through logging_utils.bounded() before storage."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import func, select

from relay.context import AppContext
from relay.models import Approval, AuditEvent, LLMCall, Order, Run, SupportTicket, ToolCall
from relay.untrusted import sanitize_text


def iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def order_json(o: Order) -> dict:
    return {
        "id": o.id, "customer_id": o.customer_id, "customer_email": o.customer_email, "sku": o.sku,
        "quantity": o.quantity,
        "total_cents": o.total_cents, "currency": o.currency, "status": o.status, "payment_ref": o.payment_ref,
        "reservation_id": o.reservation_id, "shipment_id": o.shipment_id, "refund_id": o.refund_id,
        "customer_note": sanitize_text(o.notes), "workflow_error": sanitize_text(o.workflow_error),
        "scenario_label": o.scenario_label, "created_at": iso(o.created_at), "updated_at": iso(o.updated_at),
    }


def run_brief(r: Run) -> dict:
    return {"id": r.id, "order_id": r.order_id, "status": r.status, "outcome": r.outcome, "mode": r.mode,
            "step_mode": r.step_mode, "created_at": iso(r.created_at), "finished_at": iso(r.finished_at)}


def approval_json(a: Approval) -> dict:
    return {"id": a.id, "order_id": a.order_id, "run_id": a.run_id, "kind": a.kind, "amount_cents": a.amount_cents,
            "currency": a.currency, "status": a.status, "reason": a.reason, "evidence": a.evidence,
            "requested_at": iso(a.requested_at), "expires_at": iso(a.expires_at), "decided_by": a.decided_by,
            "decided_at": iso(a.decided_at), "decision_note": a.decision_note}


def ticket_json(t: SupportTicket) -> dict:
    return {"id": t.id, "order_id": t.order_id, "customer_id": t.customer_id,
            "conversation_id": t.conversation_id, "run_id": t.run_id, "reason_code": t.reason_code,
            "category": t.category, "priority": t.priority, "summary": t.summary,
            "confirmed_summary": t.confirmed_summary, "citations": t.citations, "evidence": t.evidence,
            "service_evidence": t.service_evidence, "resolution_summary": t.resolution_summary,
            "source": t.source, "status": t.status, "created_at": iso(t.created_at),
            "updated_at": iso(t.updated_at)}


def tool_call_json(t: ToolCall) -> dict:
    return {"seq": t.seq, "node": t.node, "tool": t.tool, "mutating": t.mutating, "purpose": t.purpose,
            "attempt": t.attempt, "idempotency_key": t.idempotency_key, "status": t.status, "args": t.args,
            "result": t.result, "error": t.error, "started_at": iso(t.started_at), "duration_ms": t.duration_ms}


def audit_json(e: AuditEvent) -> dict:
    return {"id": e.id, "ts": iso(e.ts), "correlation_id": e.correlation_id, "actor": e.actor,
            "event_type": e.event_type, "payload": e.payload}


def llm_metrics(app: AppContext, run_id: str, mode: str) -> dict | None:
    with app.session() as s:
        rows = s.scalars(select(LLMCall).where(LLMCall.run_id == run_id)).all()
    if mode != "llm" and not rows:
        return None
    pt = sum(r.prompt_tokens or 0 for r in rows)
    ct = sum(r.completion_tokens or 0 for r in rows)
    reported = all(r.total_tokens is not None for r in rows) and bool(rows)
    st = app.settings
    cost = None
    if reported and st.llm_price_input_per_mtok is not None and st.llm_price_output_per_mtok is not None:
        cost = round(pt * st.llm_price_input_per_mtok / 1e6 + ct * st.llm_price_output_per_mtok / 1e6, 6)
    return {"calls": len(rows), "errors": sum(1 for r in rows if r.error), "model": rows[0].model if rows else None,
            "prompt_tokens": pt if reported else None, "completion_tokens": ct if reported else None,
            "total_tokens": pt + ct if reported else None, "latency_ms": sum(r.latency_ms for r in rows),
            "estimated_cost_usd": cost, "usage_reported_by_provider": reported}


def run_detail(app: AppContext, run_id: str) -> dict | None:
    with app.session() as s:
        run = s.get(Run, run_id)
        if run is None:
            return None
        calls = s.scalars(select(ToolCall).where(ToolCall.run_id == run_id).order_by(ToolCall.seq)).all()
        events = s.scalars(select(AuditEvent).where(AuditEvent.run_id == run_id).order_by(AuditEvent.id)).all()
        approvals = s.scalars(select(Approval).where(Approval.order_id == run.order_id)
                              .order_by(Approval.requested_at)).all()
        tickets = s.scalars(select(SupportTicket).where(SupportTicket.order_id == run.order_id)).all()
        order = s.get(Order, run.order_id)
        n_calls = s.scalar(select(func.count()).select_from(ToolCall).where(ToolCall.run_id == run_id)) or 0
        state = run.state or {}
        end = run.finished_at or run.updated_at
        out = {
            **run_brief(run),
            "correlation_id": run.correlation_id, "summary": run.summary, "error": run.error, "steps": run.steps,
            "next_node": state.get("next_node"),
            "mode_label": ("LLM mode (model proposes actions; policy validates)" if run.mode == "llm"
                           else "Deterministic demo mode (rule-based planner - no LLM)"),
            "metrics": {"active_ms": run.active_ms, "wall_ms": int((end - run.created_at).total_seconds() * 1000),
                        "tool_calls": n_calls, "steps": run.steps, "llm": llm_metrics(app, run_id, run.mode)},
            "decisions": state.get("history", []),
            "verification": state.get("verification"),
            "evidence": {k: v.get("data") for k, v in (state.get("evidence") or {}).items()},
            "tool_calls": [tool_call_json(c) for c in calls],
            "audit_events": [audit_json(e) for e in events],
            "approvals": [approval_json(a) for a in approvals],
            "tickets": [ticket_json(t) for t in tickets],
            "order": order_json(order) if order else None,
        }
    return out
