"""Derived facts from collected evidence.

Both the planner and the policy engine read facts from here, so "what does the evidence
say?" has exactly one definition. Note what is NOT used: free-text fields. Memos, notes and
workflow error strings never influence a fact.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from relay.agent.types import RunState

EVIDENCE_KEYS = ("order", "payment", "reservation", "shipment", "stock")


def initial_state(run_id: str, order_id: str, correlation_id: str, mode: str, step_mode: bool) -> RunState:
    return RunState(
        run_id=run_id, order_id=order_id, correlation_id=correlation_id, mode=mode, step_mode=step_mode,
        next_node="observe", halt=None, evidence={}, stale=list(EVIDENCE_KEYS), proposal=None, decision=None,
        denials=0, feedback=[], last_failure=None, last_result=None, escalated=False, outcome=None,
        summary=None, history=[],
    )


def data(state: RunState, key: str) -> dict | None:
    ev = state.get("evidence", {}).get(key)
    return ev["data"] if ev else None


@dataclass
class Facts:
    order: dict | None = None
    payment_state: str = "unknown"  # captured_ok | conflict | not_captured | absent | refunded | unknown
    payment_conflicts: list[str] = field(default_factory=list)
    capture: dict | None = None
    refunds: list[dict] = field(default_factory=list)
    reservation_state: str = "unknown"  # none | usable | mismatch | multiple | unknown
    reservation: dict | None = None
    shipment_state: str = "unknown"  # none | one | multiple | unknown
    shipment: dict | None = None
    stock_sufficient: bool | None = None
    stock: dict | None = None
    approval: dict | None = None

    @property
    def status(self) -> str | None:
        return self.order["status"] if self.order else None


def classify_payment(order: dict, pay: dict) -> tuple[str, list[str], dict | None, list[dict]]:
    payments, events, refunds = pay["payments"], pay["events"], pay["refunds"]
    captured = [p for p in payments if p["status"] == "captured"]
    refunded = [p for p in payments if p["status"] == "refunded"]
    conflicts: list[str] = []
    if len(captured) + len(refunded) > 1:
        conflicts.append("MULTIPLE_CHARGES")
    for p in captured + refunded:
        if p["amount_cents"] != order["total_cents"]:
            conflicts.append("AMOUNT_MISMATCH")
        if p["currency"] != order["currency"]:
            conflicts.append("CURRENCY_MISMATCH")
    bad = {"failed", "voided", "chargeback"}
    for p in captured:
        if any(e["type"] in bad and e["payment_id"] in (p["payment_id"], None) for e in events):
            conflicts.append("EVENT_CONTRADICTS_RECORD")
        if any(e["type"] == "refunded" for e in events) and not refunds:
            conflicts.append("UNRECORDED_REFUND_EVENT")
    if not captured and not refunded and any(e["type"] == "captured" for e in events):
        conflicts.append("EVENT_CONTRADICTS_RECORD")
    conflicts = sorted(set(conflicts))
    if conflicts:
        return "conflict", conflicts, None, refunds
    if len(captured) == 1:
        return "captured_ok", [], captured[0], refunds
    if len(refunded) == 1:
        return "refunded", [], refunded[0], refunds
    if any(p["status"] == "authorized" for p in payments):
        return "not_captured", [], None, refunds
    return "absent", [], None, refunds


def derive(state: RunState) -> Facts:
    f = Facts()
    f.order = data(state, "order")
    pay = data(state, "payment")
    if f.order and pay:
        f.payment_state, f.payment_conflicts, f.capture, f.refunds = classify_payment(f.order, pay)
    res = data(state, "reservation")
    if f.order and res:
        active = [r for r in res["reservations"] if r["status"] == "active"]
        if not active:
            f.reservation_state = "none"
        elif len(active) > 1:
            f.reservation_state = "multiple"
        elif active[0]["sku"] == f.order["sku"] and active[0]["quantity"] == f.order["quantity"]:
            f.reservation_state, f.reservation = "usable", active[0]
        else:
            f.reservation_state = "mismatch"
    ship = data(state, "shipment")
    if ship:
        live = [s for s in ship["shipments"] if s["status"] != "cancelled"]
        f.shipment_state = "none" if not live else ("one" if len(live) == 1 else "multiple")
        f.shipment = live[0] if len(live) == 1 else None
    stock = data(state, "stock")
    if stock:
        f.stock, f.stock_sufficient = stock, stock["sufficient"]
    f.approval = f.order.get("approval") if f.order else None
    return f


def evidence_snapshot(state: RunState) -> dict:
    """Compact, trusted-origin evidence attached to tickets/approvals (built by code, not by the model)."""
    f = derive(state)
    pay = data(state, "payment") or {}
    o = f.order or {}
    return {
        "order": {k: o.get(k) for k in ("order_id", "status", "sku", "quantity", "total_cents", "currency")},
        "payment": {"state": f.payment_state, "conflicts": f.payment_conflicts,
                    "records": pay.get("payments", []), "events": pay.get("events", []),
                    "refunds": pay.get("refunds", [])},
        "reservation": {"state": f.reservation_state},
        "shipment": {"state": f.shipment_state},
        "stock": f.stock,
        "approval": f.approval,
        "last_failure": state.get("last_failure"),
        "injection_suspected": bool(o.get("injection_suspected")),
    }
