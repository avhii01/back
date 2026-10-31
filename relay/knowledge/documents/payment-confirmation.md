---
document_id: payment-confirmation
title: Payment Confirmation and Pending Orders
version: "2026.1"
owner: Customer Support Policy
---
# Payment Confirmation and Pending Orders

## Captured payment, pending order

A payment is **captured** when the payment provider has taken the money. If the payment record shows `captured` and the order is still `pending`, the customer must not pay again. Relay checks that there is exactly one captured payment whose amount and currency match the order total, then records the order as paid and continues fulfilment.

The customer-facing status only changes after Relay re-reads the provider and verifies the order. Until then the safe message is: "Your payment is captured; the order is still being processed."

## Authorization versus capture

An `authorized` payment is not a captured payment. Authorization reserves funds but the money may not have moved, and an authorization can expire. Support must not tell a customer that an order is paid, shipped or refundable on the strength of an authorization alone.

If the provider shows only an authorization while a webhook claims a capture, that is conflicting evidence. The case is escalated for a human and no financial action is taken.

## Duplicate payment records and webhooks

Payment providers can deliver duplicate webhook events. Duplicate `captured` events are not proof of a double charge; the payment records are authoritative. Two separate captured payments, a captured amount that differs from the order total, or a capture in the wrong currency are conflicts.

When payment evidence conflicts, Relay does not charge again, refund, reserve stock or ship. It records the evidence and escalates the ticket for human review.

## What the customer should be told

- If one matching capture is confirmed: explain that the payment was received once and the order is being recovered; do not promise a delivery date until a shipment exists.
- If payment is only authorized: explain that the money may only be reserved and that capture is still pending.
- If records conflict: explain that the payment needs human review and that no additional charge or refund will be attempted automatically.
- Never ask the customer to pay again to "unblock" an order while a captured payment is being investigated.
