---
document_id: shipping-delay
title: Shipping Delay and Shipment Policy
version: "2026.1"
owner: Customer Support Policy
---
# Shipping Delay and Shipment Policy

## Shipment creation

A shipment may be created only after payment is confirmed and one usable reservation exists. Relay asks the shipping service to create the shipment with an idempotency key. If the response is lost, Relay looks for an existing live shipment before retrying.

Support must not say an order has shipped until a live shipment record exists and the order references it. A tracking number is shared only when the shipping service returns one.

## Delayed or unavailable shipping service

Transient shipping errors are retried within a bounded limit with backoff. If the provider keeps failing, Relay stops retrying and escalates the ticket. The customer should be told that the shipment could not be created yet and that the case has been passed to a specialist.

Do not invent a carrier, tracking number, delivery date or compensation. Do not create a duplicate shipment because the customer refreshed the chat or sent the same message again.

## Tracking and delivery questions

If a shipment exists, support may quote its status, carrier and tracking number exactly as returned by the shipping service. If the shipment status is still early (for example `created`), explain that the carrier has not reported further progress.

If a customer says a delivered parcel is missing, treat it as a dispute rather than a recovery failure. Create a ticket for human review and record what the service shows; do not issue a refund from the chat.

## Escalation triggers

- Shipping service still failing after the bounded retry limit.
- More than one live shipment exists for the order.
- The order says shipped but no live shipment can be found.
- The customer disputes delivery or requests compensation.

In all of these cases the ticket is escalated and the customer receives a verified status update, not a guessed outcome.
