"""Customer-safe live evidence.

Reads through the SAME read tools the recovery agent uses, but shapes the result for a
customer: no internal workflow errors, memos, traces, approvals decisions or tool calls.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from relay.agent.facts import classify_payment
from relay.context import AppContext
from relay.models import Order
from relay.money import format_cents
from relay.tools import impl
from relay.tools import schemas as S
from relay.tools.base import ToolContext

CATEGORY_LABELS = {
    "PAYMENT_PENDING": "Payment captured, order pending",
    "PAYMENT_CONFLICT": "Conflicting payment evidence",
    "INVENTORY_ISSUE": "Inventory reservation issue",
    "OUT_OF_STOCK": "Out of stock after payment",
    "SHIPPING_DELAY": "Shipping delay",
    "REFUND_REQUEST": "Refund or cancellation request",
    "REFUND_STATUS": "Refund status",
    "GENERAL_ESCALATION": "General support escalation",
}

POLICY_DOCS = {
    "PAYMENT_PENDING": ["payment-confirmation"],
    "PAYMENT_CONFLICT": ["payment-confirmation", "escalation"],
    "INVENTORY_ISSUE": ["inventory-reservation"],
    "OUT_OF_STOCK": ["inventory-reservation", "refunds-cancellations"],
    "SHIPPING_DELAY": ["shipping-delay"],
    "REFUND_REQUEST": ["refunds-cancellations"],
    "REFUND_STATUS": ["refunds-cancellations"],
    "GENERAL_ESCALATION": ["escalation"],
}


@dataclass
class CustomerEvidence:
    order: dict
    payment_state: str
    payment_conflicts: list[str]
    captured: dict | None
    refunds: list[dict]
    reservation_state: str
    active_reservations: list[dict]
    reservation: dict | None
    shipment_state: str
    live_shipments: list[dict]
    stock: dict | None
    category: str
    summary: str
    snapshot: dict = field(default_factory=dict)

    @property
    def category_label(self) -> str:
        return CATEGORY_LABELS.get(self.category, self.category)

    @property
    def policy_document_ids(self) -> list[str]:
        return POLICY_DOCS.get(self.category, ["escalation"])

    @property
    def eligible_for_recovery(self) -> bool:
        return self.order["status"] in ("pending", "paid", "reserved")


def _safe_order(row: Order) -> dict:
    return {"id": row.id, "status": row.status, "sku": row.sku, "quantity": row.quantity,
            "total_cents": row.total_cents, "currency": row.currency}


def _infer_category(order: dict, payment_state: str, reservation_state: str, shipment_state: str,
                    stock: dict | None, message: str) -> str:
    text = message.lower()
    # Evidence wins over wording: the customer may describe a symptom, not the root cause.
    if payment_state == "conflict":
        return "PAYMENT_CONFLICT"
    if order["status"] == "refunded" or payment_state == "refunded":
        return "REFUND_STATUS"
    if order["status"] == "pending":
        return "PAYMENT_PENDING"
    if order["status"] == "paid":
        if reservation_state == "none":
            if stock and stock.get("sufficient") is False:
                return "OUT_OF_STOCK"
            return "INVENTORY_ISSUE"
        if reservation_state == "usable":
            return "SHIPPING_DELAY"
        return "INVENTORY_ISSUE"
    if order["status"] == "reserved":
        return "SHIPPING_DELAY"
    if any(w in text for w in ("refund", "cancel", "return", "money back")):
        return "REFUND_REQUEST"
    if any(w in text for w in ("ship", "delivery", "deliver", "track", "parcel")):
        return "SHIPPING_DELAY"
    if any(w in text for w in ("stock", "inventory", "reserve", "unavailable")):
        return "INVENTORY_ISSUE"
    if any(w in text for w in ("payment", "charge", "charged", "deduct", "debited", "paid")):
        return "PAYMENT_PENDING"
    return "GENERAL_ESCALATION"


def _summary(order: dict, category: str, payment_state: str, capture: dict | None,
             active: list[dict], stock: dict | None) -> str:
    oid = order["id"]
    money = format_cents(order["total_cents"], order["currency"])
    if category == "PAYMENT_CONFLICT":
        return f"The payment records for order {oid} conflict, so no automatic financial action can be taken."
    if category == "REFUND_STATUS":
        return f"Order {oid} is marked refunded. The payment provider record is authoritative."
    if category == "PAYMENT_PENDING":
        if capture:
            return f"Payment of {money} was captured, but order {oid} is still pending."
        return f"Order {oid} is pending and a captured payment has not been confirmed yet."
    if category == "OUT_OF_STOCK":
        available = stock.get("available") if stock else 0
        return (f"Payment of {money} was captured, but the item for order {oid} is out of stock "
                f"({available} available).")
    if category == "INVENTORY_ISSUE":
        return f"Payment for order {oid} was captured, but the inventory reservation has not completed."
    if category == "SHIPPING_DELAY":
        if active:
            return f"Payment and inventory are confirmed, but no shipment exists yet for order {oid}."
        return f"Order {oid} is waiting for shipment and no live shipment exists yet."
    if category == "REFUND_REQUEST":
        return f"The customer requested a refund or cancellation for order {oid}."
    return f"Order {oid} is currently '{order['status']}' and the customer reported: {payment_state}."


def collect(app: AppContext, order_id: str, *, correlation_id: str, message: str = "") -> CustomerEvidence | None:
    with app.session() as s:
        row = s.get(Order, order_id)
        if row is None:
            return None
        safe_order = _safe_order(row)
        tc = ToolContext(app=app, session=s, run_id=None, correlation_id=correlation_id)
        payments = impl.get_payment_status(tc, S.OrderIn(order_id=order_id)).model_dump()
        reservations = impl.get_inventory_reservation(tc, S.OrderIn(order_id=order_id)).model_dump()
        shipments = impl.get_shipment_status(tc, S.OrderIn(order_id=order_id)).model_dump()

    payment_state, conflicts, capture, refunds = classify_payment(safe_order, payments)
    active = [r for r in reservations["reservations"] if r["status"] == "active"]
    usable = [r for r in active if r["sku"] == safe_order["sku"] and r["quantity"] == safe_order["quantity"]]
    reservation_state = "none" if not active else ("usable" if len(usable) == 1 and len(active) == 1 else
                                                    ("multiple" if len(active) > 1 else "mismatch"))
    live = [x for x in shipments["shipments"] if x["status"] != "cancelled"]
    shipment_state = "none" if not live else ("one" if len(live) == 1 else "multiple")

    stock = None
    if safe_order["status"] == "paid" and reservation_state == "none":
        with app.session() as s:
            tc = ToolContext(app=app, session=s, run_id=None, correlation_id=correlation_id)
            stock = impl.check_stock(tc, S.CheckStockIn(sku=safe_order["sku"],
                                                       quantity=safe_order["quantity"])).model_dump()
    category = _infer_category(safe_order, payment_state, reservation_state, shipment_state, stock, message)
    summary = _summary(safe_order, category, payment_state, capture, active, stock)
    snapshot = {
        "order": safe_order,
        "payment": {"state": payment_state, "conflicts": conflicts,
                    "captured_count": len([p for p in payments["payments"] if p["status"] == "captured"]),
                    "refund_count": len(refunds)},
        "inventory": {"state": reservation_state, "active_reservations": len(active),
                      "matching_reservation": bool(usable)},
        "shipment": {"state": shipment_state, "live_shipments": len(live)},
        "stock": None if stock is None else {"available": stock["available"], "sufficient": stock["sufficient"]},
    }
    return CustomerEvidence(order=safe_order, payment_state=payment_state, payment_conflicts=conflicts,
                            captured=capture, refunds=refunds, reservation_state=reservation_state,
                            active_reservations=active, reservation=usable[0] if usable else None,
                            shipment_state=shipment_state, live_shipments=live, stock=stock,
                            category=category, summary=summary, snapshot=snapshot)
