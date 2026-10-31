"""Deterministic policy engine. The model PROPOSES; this module DECIDES.

Guarantees enforced here, regardless of what the planner (rules or LLM) asks for:
  * only whitelisted action tools can be proposed
  * required evidence must be fresh (re-read after the last relevant mutation)
  * every action has explicit preconditions over derived facts (never over free text)
  * risky arguments (amounts, ids, refund reason, ticket evidence) are REPLACED with values
    re-derived from trusted state - the model's version is discarded
  * a refund can only execute against a persisted, human-approved approval
  * the agent has no code path that approves anything

Rules never read memo/notes/workflow_error text, so injected instructions cannot affect them.
"""
from __future__ import annotations

from relay.agent.facts import Facts, derive, evidence_snapshot
from relay.agent.types import Decision, Proposal, RunState
from relay.tools.registry import AGENT_ACTION_TOOLS
from relay.tools.schemas import ReasonCode
from relay.untrusted import sanitize_text

REQUIRED_FRESH = {
    "update_order_status": ("order", "payment", "reservation", "shipment"),
    "reserve_inventory": ("order", "payment", "reservation"),  # stock freshness is checked in the rule
    "create_shipment": ("order", "payment", "reservation", "shipment"),
    "request_refund_approval": ("order", "payment", "reservation", "shipment", "stock"),
    "execute_approved_refund": ("order", "payment"),
    "create_support_ticket": (),
}
_VALID_REASONS = set(ReasonCode.__args__)  # type: ignore[attr-defined]


def deny(rule: str, reason: str) -> Decision:
    return Decision(allowed=False, rule_ids=[rule], reasons=[reason])


def allow(rule: str, reason: str, args: dict, requires_approval: bool = False) -> Decision:
    return Decision(allowed=True, rule_ids=[rule], reasons=[reason], canonical_args=args,
                    requires_approval=requires_approval)


def evaluate(state: RunState, proposal: Proposal) -> Decision:
    f = derive(state)
    if proposal.action == "finish":
        if f.status in ("shipped", "refunded"):
            return allow("P-FINISH", f"order is terminal ('{f.status}'); verification will follow", {})
        return deny("P-FINISH-NOT-TERMINAL", f"cannot finish: order status is '{f.status}'")
    if proposal.action == "wait_for_approval":
        if f.approval and f.approval["status"] == "pending":
            return allow("P-WAIT-APPROVAL", "a human approval is pending; pausing the run", {})
        return deny("P-WAIT-NO-PENDING", "no pending approval to wait for")

    tool = proposal.tool or ""
    if tool not in AGENT_ACTION_TOOLS:
        return deny("P-TOOL-NOT-ALLOWED", f"'{tool}' is not an action the agent may perform")
    stale = [k for k in REQUIRED_FRESH[tool] if k in state.get("stale", []) or k not in state.get("evidence", {})]
    if stale:
        return deny("P-EVIDENCE-STALE", f"evidence not fresh: {', '.join(stale)}")
    if f.order is None and tool != "create_support_ticket":
        return deny("P-NO-ORDER", "order evidence missing")
    return _RULES[tool](state, f, proposal)


def _rule_update_status(state: RunState, f: Facts, p: Proposal) -> Decision:
    new = p.args.get("new_status")
    cur = f.status
    base = {"order_id": f.order["order_id"], "expected_from": cur, "new_status": new, "reason": f"agent: {new}"}
    if new in ("paid", "reserved", "shipped") and f.payment_state != "captured_ok":
        return deny("P-PAYMENT-NOT-CONFIRMED", f"payment state is '{f.payment_state}', not a single matching capture")
    if (cur, new) == ("pending", "paid"):
        return allow("P-ORDER-PAID", "exactly one captured payment matches the order total",
                     {**base, "payment_id": f.capture["payment_id"]})
    if (cur, new) == ("paid", "reserved"):
        if f.reservation_state != "usable":
            return deny("P-RESERVATION-NOT-USABLE", f"reservation state is '{f.reservation_state}'")
        return allow("P-ORDER-RESERVED", "one active reservation matches sku and quantity",
                     {**base, "reservation_id": f.reservation["reservation_id"]})
    if (cur, new) == ("reserved", "shipped"):
        if f.shipment_state != "one" or f.reservation_state != "usable":
            return deny("P-SHIPMENT-NOT-CONFIRMED", f"shipment={f.shipment_state}, reservation={f.reservation_state}")
        return allow("P-ORDER-SHIPPED", "one live shipment and one usable reservation exist",
                     {**base, "shipment_id": f.shipment["shipment_id"]})
    if new == "refunded" and cur in ("paid", "reserved"):
        ap = f.approval
        if f.payment_state != "refunded" or len(f.refunds) != 1:
            return deny("P-REFUND-NOT-CONFIRMED", "provider does not show exactly one refund on a refunded payment")
        if not ap or ap["status"] != "approved" or ap["amount_cents"] != f.refunds[0]["amount_cents"] \
                or f.refunds[0]["amount_cents"] != f.order["total_cents"]:
            return deny("P-REFUND-UNAUTHORISED", "refund does not match an approved approval for the full total")
        return allow("P-ORDER-REFUNDED", "provider refund matches the approved amount",
                     {**base, "refund_id": f.refunds[0]["refund_id"]})
    return deny("P-TRANSITION", f"transition {cur}->{new} is not permitted for the agent")


def _rule_reserve(state: RunState, f: Facts, p: Proposal) -> Decision:
    if f.status != "paid":
        return deny("P-RESERVE-STATUS", f"order must be 'paid' (is '{f.status}')")
    if f.payment_state != "captured_ok":
        return deny("P-PAYMENT-NOT-CONFIRMED", f"payment state is '{f.payment_state}'")
    if f.reservation_state != "none":
        return deny("P-REUSE-EXISTING-RESERVATION",
                    f"reservation state is '{f.reservation_state}': reuse/escalate instead of reserving again")
    if "stock" in state.get("stale", []) or f.stock_sufficient is not True:
        return deny("P-NO-STOCK", "fresh stock evidence showing sufficient stock is required")
    return allow("P-RESERVE", "paid order, no existing reservation, stock available",
                 {"order_id": f.order["order_id"], "sku": f.order["sku"], "quantity": f.order["quantity"]})


def _rule_ship(state: RunState, f: Facts, p: Proposal) -> Decision:
    if f.status != "reserved":
        return deny("P-SHIP-STATUS", f"order must be 'reserved' (is '{f.status}')")
    if f.payment_state != "captured_ok":
        return deny("P-PAYMENT-NOT-CONFIRMED", f"payment state is '{f.payment_state}'")
    if f.reservation_state != "usable":
        return deny("P-RESERVATION-NOT-USABLE", f"reservation state is '{f.reservation_state}'")
    if f.shipment_state != "none":
        return deny("P-SHIPMENT-EXISTS", f"shipment state is '{f.shipment_state}'")
    return allow("P-SHIP", "reserved order with confirmed payment and no existing shipment",
                 {"order_id": f.order["order_id"], "reservation_id": f.reservation["reservation_id"]})


def _rule_request_refund(state: RunState, f: Facts, p: Proposal) -> Decision:
    if f.payment_state != "captured_ok":
        return deny("P-PAYMENT-NOT-CONFIRMED", f"payment state is '{f.payment_state}': no financial action")
    if f.status != "paid" or f.reservation_state != "none" or f.shipment_state != "none":
        return deny("P-REFUND-PRECONDITIONS", "refund path only applies to a paid order with nothing reserved/shipped")
    if f.stock_sufficient is not False:
        return deny("P-REFUND-NOT-JUSTIFIED", "refund is only proposed when stock is confirmed insufficient")
    if f.approval is not None:
        return deny("P-APPROVAL-EXISTS", f"an approval already exists (status '{f.approval['status']}')")
    s = f.stock or {}
    reason = (f"Out of stock after payment: requested {s.get('requested')}, available {s.get('available')}. "
              f"Full refund of order total.")
    return allow("P-REQUEST-REFUND-APPROVAL", "stock confirmed insufficient after captured payment",
                 {"order_id": f.order["order_id"], "reason": reason, "evidence": evidence_snapshot(state)},
                 requires_approval=True)


def _rule_execute_refund(state: RunState, f: Facts, p: Proposal) -> Decision:
    ap = f.approval
    if not ap or ap["status"] != "approved" or not ap.get("decided_by"):
        return deny("P-REFUND-NO-APPROVAL", "no persisted human approval: refund forbidden")
    if f.payment_state != "captured_ok":
        return deny("P-PAYMENT-NOT-CONFIRMED", f"payment state is '{f.payment_state}'")
    if ap["amount_cents"] != f.order["total_cents"] or f.capture["amount_cents"] != ap["amount_cents"]:
        return deny("P-REFUND-AMOUNT", "approved amount does not equal order total and captured amount")
    if f.refunds:
        return deny("P-REFUND-ALREADY-DONE", "provider already shows a refund")
    return allow("P-EXECUTE-REFUND", "human approval persisted; amounts match",
                 {"order_id": f.order["order_id"], "approval_id": ap["approval_id"]}, requires_approval=True)


def _rule_ticket(state: RunState, f: Facts, p: Proposal) -> Decision:
    code = p.args.get("reason_code")
    if code not in _VALID_REASONS:
        return deny("P-TICKET-REASON", f"unknown reason_code '{code}'")
    summary = sanitize_text(p.args.get("summary") or p.summary or f"Escalated: {code}", 500)
    return allow("P-ESCALATE", "escalation to a human is always permitted",
                 {"order_id": state["order_id"], "reason_code": code, "summary": summary,
                  "priority": "high" if code in ("PAYMENT_CONFLICT", "VERIFICATION_FAILED") else "normal",
                  "evidence": evidence_snapshot(state)})


_RULES = {
    "update_order_status": _rule_update_status,
    "reserve_inventory": _rule_reserve,
    "create_shipment": _rule_ship,
    "request_refund_approval": _rule_request_refund,
    "execute_approved_refund": _rule_execute_refund,
    "create_support_ticket": _rule_ticket,
}
