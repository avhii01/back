"""Typed inputs and outputs for every agent tool."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

OrderId = Field(min_length=1, max_length=40, pattern=r"^[A-Za-z0-9_\-]+$")
OrderStatus = Literal["pending", "paid", "reserved", "shipped", "refunded"]
Expectation = Literal["fulfilled", "refunded", "escalated"]
ReasonCode = Literal[
    "PAYMENT_CONFLICT", "PAYMENT_NOT_CAPTURED", "SHIPPING_UNAVAILABLE", "REFUND_REJECTED",
    "REFUND_APPROVAL_EXPIRED", "INVENTORY_AMBIGUOUS", "DUPLICATE_SHIPMENT", "VERIFICATION_FAILED",
    "POLICY_DENIED", "BUDGET_EXHAUSTED", "LLM_UNAVAILABLE", "LLM_INVALID_OUTPUT", "TOOL_FAILURE",
    "UNEXPECTED_STATE",
]


class _In(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _Out(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ------------------------------------------------------------------ inputs
class OrderIn(_In):
    order_id: str = OrderId


class CheckStockIn(_In):
    sku: str = Field(min_length=1, max_length=80)
    quantity: int = Field(gt=0, le=1000)


class UpdateOrderStatusIn(_In):
    order_id: str = OrderId
    expected_from: OrderStatus
    new_status: OrderStatus
    payment_id: str | None = Field(default=None, max_length=80)
    reservation_id: str | None = Field(default=None, max_length=80)
    shipment_id: str | None = Field(default=None, max_length=80)
    refund_id: str | None = Field(default=None, max_length=80)
    reason: str = Field(default="", max_length=300)


class ReserveInventoryIn(_In):
    order_id: str = OrderId
    sku: str = Field(min_length=1, max_length=80)
    quantity: int = Field(gt=0, le=1000)


class CreateShipmentIn(_In):
    order_id: str = OrderId
    reservation_id: str = Field(min_length=1, max_length=80)


class RequestRefundApprovalIn(_In):
    order_id: str = OrderId
    reason: str = Field(max_length=400)
    evidence: dict = Field(default_factory=dict)


class ExecuteApprovedRefundIn(_In):
    order_id: str = OrderId
    approval_id: str = Field(min_length=1, max_length=40)


class CreateSupportTicketIn(_In):
    order_id: str = OrderId
    reason_code: ReasonCode
    summary: str = Field(max_length=500)
    priority: Literal["low", "normal", "high"] = "normal"
    evidence: dict = Field(default_factory=dict)


class VerifyRecoveryIn(_In):
    order_id: str = OrderId
    expectation: Expectation


# ------------------------------------------------------------------ outputs
class ApprovalView(_Out):
    approval_id: str
    status: str
    amount_cents: int
    currency: str
    expires_at: str
    decided_by: str | None = None


class OrderStatusOut(_Out):
    order_id: str
    status: str
    sku: str
    quantity: int
    total_cents: int
    currency: str
    payment_ref: str | None
    reservation_id: str | None
    shipment_id: str | None
    refund_id: str | None
    customer_note: str | None = Field(description="UNTRUSTED free text", default=None)
    workflow_error: str | None = Field(description="UNTRUSTED free text", default=None)
    injection_suspected: bool = False
    approval: ApprovalView | None = None
    open_ticket_reasons: list[str] = []
    simulated: bool = False


class PaymentItem(_Out):
    payment_id: str
    status: str
    amount_cents: int
    currency: str
    memo: str | None = None


class PaymentEventItem(_Out):
    payment_id: str | None
    type: str
    amount_cents: int
    source: str
    memo: str | None = None


class RefundItem(_Out):
    refund_id: str
    payment_id: str
    amount_cents: int


class PaymentStatusOut(_Out):
    order_id: str
    payments: list[PaymentItem]
    events: list[PaymentEventItem]
    refunds: list[RefundItem]
    simulated: bool = True


class ReservationItem(_Out):
    reservation_id: str
    sku: str
    quantity: int
    status: str


class InventoryReservationOut(_Out):
    order_id: str
    reservations: list[ReservationItem]
    simulated: bool = True


class CheckStockOut(_Out):
    sku: str
    available: int
    requested: int
    sufficient: bool
    simulated: bool = True


class UpdateOrderStatusOut(_Out):
    order_id: str
    status: str
    changed: bool


class ReserveInventoryOut(_Out):
    reservation_id: str
    order_id: str
    sku: str
    quantity: int
    status: str
    reused_by_provider: bool = False
    simulated: bool = True


class ShipmentItem(_Out):
    shipment_id: str
    status: str
    carrier: str
    tracking: str


class ShipmentStatusOut(_Out):
    order_id: str
    shipments: list[ShipmentItem]
    simulated: bool = True


class CreateShipmentOut(_Out):
    shipment_id: str
    order_id: str
    status: str
    tracking: str
    reused_by_provider: bool = False
    simulated: bool = True


class RequestRefundApprovalOut(_Out):
    approval_id: str
    status: str
    amount_cents: int
    currency: str
    expires_at: str


class ExecuteApprovedRefundOut(_Out):
    refund_id: str
    payment_id: str
    amount_cents: int
    approval_id: str
    reused_by_provider: bool = False
    simulated: bool = True


class CreateSupportTicketOut(_Out):
    ticket_id: str
    reason_code: str
    created: bool


class VerificationCheck(_Out):
    id: str
    description: str
    passed: bool
    detail: str = ""


class VerifyRecoveryOut(_Out):
    order_id: str
    expectation: str
    passed: bool
    checks: list[VerificationCheck]
