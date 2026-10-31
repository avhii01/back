"""Tool registry. The agent can only invoke tools listed here, with schema-validated input."""
from __future__ import annotations

from relay.tools import impl
from relay.tools import schemas as S
from relay.tools.base import ToolSpec

_SPECS = [
    ToolSpec("get_order_status", "Read the order record (status, amounts, references, approval state).",
             S.OrderIn, S.OrderStatusOut, impl.get_order_status, evidence_key="order", timeout_s=5),
    ToolSpec("get_payment_status", "Read payment records, payment events and refunds from the payment service.",
             S.OrderIn, S.PaymentStatusOut, impl.get_payment_status, remote=True, evidence_key="payment"),
    ToolSpec("get_inventory_reservation", "List inventory reservations for the order from the inventory service.",
             S.OrderIn, S.InventoryReservationOut, impl.get_inventory_reservation, remote=True,
             evidence_key="reservation"),
    ToolSpec("check_stock", "Check available stock for a SKU.", S.CheckStockIn, S.CheckStockOut,
             impl.check_stock, remote=True, evidence_key="stock"),
    ToolSpec("get_shipment_status", "List shipments for the order from the shipping service.",
             S.OrderIn, S.ShipmentStatusOut, impl.get_shipment_status, remote=True, evidence_key="shipment"),
    ToolSpec("update_order_status",
             "Compare-and-set order status transition (pending->paid->reserved->shipped, or ->refunded).",
             S.UpdateOrderStatusIn, S.UpdateOrderStatusOut, impl.update_order_status, mutating=True,
             idempotency_key=lambda a: f"order:{a.order_id}:{a.expected_from}->{a.new_status}",
             invalidates=("order",), replay_valid=impl._replay_valid_update),
    ToolSpec("reserve_inventory", "Reserve stock for the order (idempotent at the provider).",
             S.ReserveInventoryIn, S.ReserveInventoryOut, impl.reserve_inventory, mutating=True, remote=True,
             idempotency_key=lambda a: f"reserve:{a.order_id}", reconcile=impl.reconcile_reserve_inventory,
             invalidates=("reservation", "stock")),
    ToolSpec("create_shipment", "Create a shipment for the order (idempotent at the provider).",
             S.CreateShipmentIn, S.CreateShipmentOut, impl.create_shipment, mutating=True, remote=True,
             idempotency_key=lambda a: f"ship:{a.order_id}", reconcile=impl.reconcile_create_shipment,
             invalidates=("shipment",)),
    ToolSpec("request_refund_approval", "Create a PENDING refund approval for a human to decide.",
             S.RequestRefundApprovalIn, S.RequestRefundApprovalOut, impl.request_refund_approval, mutating=True,
             idempotency_key=lambda a: f"approval:{a.order_id}:refund", invalidates=("order",)),
    ToolSpec("execute_approved_refund", "Execute a refund that a human has approved (verified server-side).",
             S.ExecuteApprovedRefundIn, S.ExecuteApprovedRefundOut, impl.execute_approved_refund, mutating=True,
             remote=True, idempotency_key=lambda a: f"refund:{a.approval_id}",
             reconcile=impl.reconcile_execute_approved_refund, invalidates=("payment", "order")),
    ToolSpec("create_support_ticket", "Escalate to a human with a reason code and evidence snapshot.",
             S.CreateSupportTicketIn, S.CreateSupportTicketOut, impl.create_support_ticket, mutating=True,
             idempotency_key=lambda a: f"ticket:{a.order_id}:{a.reason_code}", invalidates=("order",)),
    ToolSpec("verify_recovery", "Re-read live state and check business invariants for the expected outcome.",
             S.VerifyRecoveryIn, S.VerifyRecoveryOut, impl.verify_recovery, remote=True),
]

REGISTRY: dict[str, ToolSpec] = {s.name: s for s in _SPECS}
READ_TOOLS = [s.name for s in _SPECS if not s.mutating and s.evidence_key]
AGENT_ACTION_TOOLS = ["update_order_status", "reserve_inventory", "create_shipment", "request_refund_approval",
                      "execute_approved_refund", "create_support_ticket"]
assert len(REGISTRY) == 12
