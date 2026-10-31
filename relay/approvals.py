"""Human approval decisions.

Only this module (reachable only through the operator API) can move an approval out of
'pending'. The agent has no tool, no code path and no credentials that call it.

Concurrency: the transition is a compare-and-set (`UPDATE ... WHERE status='pending'`).
Exactly one writer wins; a repeated identical decision is an idempotent no-op that returns
the stored decision; a conflicting second decision is rejected with 409.
"""
from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import update

from relay.audit import record_event
from relay.context import AppContext
from relay.models import Approval, Run, utcnow
from relay.untrusted import sanitize_text

FORBIDDEN_DECIDERS = {"agent", "relay", "relay-agent", "system", "llm", "bot", ""}


class ApprovalError(Exception):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


@dataclass
class DecisionResult:
    approval: Approval
    applied: bool  # False => idempotent replay of an earlier identical decision
    run_id: str | None


def decide_approval(app: AppContext, approval_id: str, decision: str, decided_by: str, note: str | None,
                    correlation_id: str) -> DecisionResult:
    if decision not in ("approve", "reject"):
        raise ApprovalError(422, "decision must be 'approve' or 'reject'")
    who = (decided_by or "").strip()
    if who.lower() in FORBIDDEN_DECIDERS:
        raise ApprovalError(422, "a human operator identity is required; the agent cannot decide approvals")
    target = "approved" if decision == "approve" else "rejected"
    with app.session() as s:
        ap = s.get(Approval, approval_id)
        if ap is None:
            raise ApprovalError(404, "approval not found")
        now = utcnow()
        if ap.status == "pending" and ap.expires_at < now:
            s.execute(update(Approval).where(Approval.id == ap.id, Approval.status == "pending").values(status="expired"))
            s.commit()
            s.refresh(ap)
        res = s.execute(update(Approval).where(Approval.id == approval_id, Approval.status == "pending",
                                               Approval.expires_at >= now)
                        .values(status=target, decided_by=who[:80], decided_at=now,
                                decision_note=sanitize_text(note, 300)))
        if res.rowcount == 1:
            s.refresh(ap)
            record_event(s, correlation_id=correlation_id, run_id=ap.run_id, order_id=ap.order_id, actor="human",
                         event_type="approval_decided", payload={"approval_id": ap.id, "decision": target,
                                                                 "decided_by": who[:80], "amount_cents": ap.amount_cents})
            s.commit()
            return DecisionResult(ap, True, ap.run_id)
        s.rollback()
        s.refresh(ap)
        if ap.status == target:  # repeated identical submission: safe replay
            return DecisionResult(ap, False, ap.run_id)
        raise ApprovalError(409, f"approval is already '{ap.status}'")


def run_for_approval(app: AppContext, run_id: str | None) -> Run | None:
    if not run_id:
        return None
    with app.session() as s:
        return s.get(Run, run_id)
