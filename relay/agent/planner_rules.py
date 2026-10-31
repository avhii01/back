"""Deterministic DEMO planner (no LLM).

This is a rule table, not reasoning. The dashboard labels runs using it as
"Deterministic demo mode" and every proposal carries source="rules" and a rule id.
"""
from __future__ import annotations

from relay.agent.facts import Facts, derive
from relay.agent.types import Proposal, RunState


def _ticket(order_id: str, code: str, summary: str, rule: str) -> Proposal:
    return Proposal(action="tool", tool="create_support_ticket",
                    args={"order_id": order_id, "reason_code": code, "summary": summary},
                    summary=summary, source="rules", rule_id=rule)


def _update(f: Facts, new: str, summary: str, rule: str, refs: list[str]) -> Proposal:
    return Proposal(action="tool", tool="update_order_status",
                    args={"order_id": f.order["order_id"], "expected_from": f.status, "new_status": new},
                    summary=summary, evidence_refs=refs, source="rules", rule_id=rule)


class RulesPlanner:
    source = "rules"


    def propose(self, state: RunState, feedback: list[str]) -> Proposal:
        f = derive(state)
        oid = state["order_id"]
        fail = state.get("last_failure")
        if fail:
            if fail.get("tool") == "create_shipment" and fail.get("code") == "RETRIES_EXHAUSTED":
                return _ticket(oid, "SHIPPING_UNAVAILABLE", "Shipping service unavailable; bounded retries exhausted.",
                               "R-SHIP-RETRIES-EXHAUSTED")
            return _ticket(oid, "TOOL_FAILURE", f"Tool {fail.get('tool')} failed: {fail.get('code')}", "R-TOOL-FAILURE")

        if f.order is None:
            return _ticket(oid, "UNEXPECTED_STATE", "Order evidence is unavailable.", "R-NO-ORDER")

        ap = f.approval
        if ap:  # human-in-the-loop path has priority once an approval exists
            if ap["status"] == "pending":
                return Proposal(action="wait_for_approval", summary="Refund approval is pending a human decision.",
                                evidence_refs=["order.approval"], source="rules", rule_id="R-WAIT")
            if ap["status"] == "rejected":
                return _ticket(oid, "REFUND_REJECTED", "Human rejected the refund; escalating for manual handling.",
                               "R-REFUND-REJECTED")
            if ap["status"] == "expired":
                return _ticket(oid, "REFUND_APPROVAL_EXPIRED", "Refund approval expired without a decision.",
                               "R-REFUND-EXPIRED")
            if ap["status"] == "approved":
                if f.status == "refunded":
                    return Proposal(action="finish", summary="Refund executed and order marked refunded.",
                                    source="rules", rule_id="R-FINISH")
                if f.refunds and f.payment_state == "refunded":
                    return _update(f, "refunded", "Provider confirms the approved refund; recording it on the order.",
                                   "R-ORDER-REFUNDED", ["payment.refunds"])
                return Proposal(action="tool", tool="execute_approved_refund",
                                args={"order_id": f.order["order_id"], "approval_id": ap["approval_id"]},
                                summary="Approval granted by a human; executing the approved refund.",
                                evidence_refs=["order.approval"], source="rules", rule_id="R-EXECUTE-REFUND")

        if f.status in ("shipped", "refunded"):
            return Proposal(action="finish", summary=f"Order is already '{f.status}'.", source="rules",
                            rule_id="R-FINISH")
        if f.payment_state == "conflict":
            return _ticket(oid, "PAYMENT_CONFLICT",
                           f"Conflicting payment evidence ({', '.join(f.payment_conflicts)}). No financial action taken.",
                           "R-PAYMENT-CONFLICT")
        if f.payment_state != "captured_ok":
            return _ticket(oid, "PAYMENT_NOT_CAPTURED", f"Payment state is '{f.payment_state}'; cannot confirm capture.",
                           "R-PAYMENT-NOT-CAPTURED")

        if f.status == "pending":
            return _update(f, "paid", "Provider shows exactly one captured payment matching the total; no new charge.",
                           "R-ORDER-PAID", ["payment.payments"])
        if f.status == "paid":
            if f.reservation_state == "usable":
                return _update(f, "reserved", "Existing active reservation found; reusing it instead of reserving again.",
                               "R-REUSE-RESERVATION", ["reservation.reservations"])
            if f.reservation_state in ("mismatch", "multiple"):
                return _ticket(oid, "INVENTORY_AMBIGUOUS", f"Reservation state is '{f.reservation_state}'.",
                               "R-RESERVATION-AMBIGUOUS")
            if f.stock_sufficient is True:
                return Proposal(action="tool", tool="reserve_inventory",
                                args={"order_id": f.order["order_id"], "sku": f.order["sku"],
                                      "quantity": f.order["quantity"]},
                                summary="No reservation exists and stock is available; reserving.",
                                evidence_refs=["reservation", "stock"], source="rules", rule_id="R-RESERVE")
            if f.stock_sufficient is False:
                return Proposal(action="tool", tool="request_refund_approval",
                                args={"order_id": f.order["order_id"], "reason": "out of stock after payment"},
                                summary="Stock is insufficient after payment; requesting human approval for a refund.",
                                evidence_refs=["stock"], source="rules", rule_id="R-REQUEST-REFUND")
        if f.status == "reserved":
            if f.shipment_state == "one":
                return _update(f, "shipped", "A shipment already exists; recording it on the order.",
                               "R-REUSE-SHIPMENT", ["shipment.shipments"])
            if f.shipment_state == "multiple":
                return _ticket(oid, "DUPLICATE_SHIPMENT", "Multiple live shipments exist for this order.",
                               "R-DUPLICATE-SHIPMENT")
            if f.reservation_state == "usable":
                return Proposal(action="tool", tool="create_shipment",
                                args={"order_id": f.order["order_id"],
                                      "reservation_id": f.reservation["reservation_id"]},
                                summary="Reservation confirmed and no shipment exists; creating one.",
                                evidence_refs=["reservation", "shipment"], source="rules", rule_id="R-SHIP")
        return _ticket(oid, "UNEXPECTED_STATE", f"No safe action for status '{f.status}'.", "R-UNEXPECTED")
