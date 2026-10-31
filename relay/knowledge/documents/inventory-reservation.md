---
document_id: inventory-reservation
title: Inventory Reservation Guidance
version: "2026.1"
owner: Customer Support Policy
---
# Inventory Reservation Guidance

## Reservation lifecycle

After payment is confirmed, Relay asks the inventory service to reserve stock for the order's SKU and quantity. A reservation has a status such as `active`, `released` or `consumed`. Fulfilment may continue only when there is exactly one active reservation that matches the order's SKU and quantity.

An existing active reservation is reused. Support must not create a second reservation for the same order merely because a response was slow or a page was refreshed.

## Lost or delayed reservation response

A timeout does not mean the reservation failed. The workflow may have committed the reservation and then lost the response on the way back. Relay looks the reservation up before retrying, and provider idempotency keys make a repeated reserve call safe.

If no reservation exists, a bounded retry is allowed while stock is available. If a reservation exists but has the wrong quantity or SKU, or if several active reservations exist, the evidence is ambiguous and the ticket is escalated instead of guessing.

## Insufficient stock after payment

If the item is out of stock after a captured payment, support must not promise a substitute item, a different colour or an unapproved partial refund. Relay requests a **refund approval** and pauses. A human operator must approve the refund before any money is returned.

The customer can be told: "Your payment was captured, but the item is currently unavailable. A refund has been requested and needs a support specialist to approve it." The ticket is not resolved until the provider confirms the refund and verification passes.

## Customer guidance

- Do not claim that stock is reserved until an active reservation matches the order.
- Do not ask the customer to order again while a captured payment is being recovered.
- For a delayed response, explain that the reservation is being checked rather than repeating the request.
- For confirmed insufficient stock, explain the approval step and avoid inventing a refund date.
