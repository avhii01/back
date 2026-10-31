---
document_id: refunds-cancellations
title: Refund and Cancellation Policy
version: "2026.1"
owner: Customer Support Policy
---
# Refund and Cancellation Policy

## When a refund can be requested

A refund can be requested when a payment was captured but the order cannot be fulfilled, for example because stock is unavailable after payment. In the demo policy, refunds are full refunds of the captured order total and are returned to the original payment method. Partial refunds and compensation are outside the automated flow.

A refund is never approved by the chatbot or by the recovery agent. Relay creates a **pending approval**; a human operator must approve it. The ticket shows "Awaiting approval" while the decision is pending.

## Refund execution and timing

After approval, Relay checks that the provider still shows exactly one captured payment matching the approved amount, executes the refund once with an idempotency key, and re-reads the provider. The order is marked refunded only when a matching refund record exists.

Once the provider confirms the refund, the demo policy says funds typically appear within 5 to 10 business days depending on the customer's bank. Support must say this is an estimate and must not claim the money has already arrived before the provider reports the refund.

If an approval is rejected or expires, the automated refund path stops and the ticket is escalated for manual handling. Support must not retry a rejected refund or pressure the customer to accept a different remedy.

## Cancellation requests

A cancellation can be handled automatically only before a shipment exists and only when the same refund preconditions hold. If a shipment already exists, the request becomes a returns or dispute case for a human specialist.

Customers may ask to cancel in chat, but the chat cannot cancel an order or approve money movement. It can create a ticket after confirmation and explain the approval step.

## What support may say

- "A refund needs a support specialist's approval."
- "After approval, refunds usually take 5 to 10 business days to appear."
- "The order will show as refunded only after the payment provider confirms it."
- Do not promise a refund amount that differs from the captured order total.
- Do not say a refund is complete until verification against the provider passes.
