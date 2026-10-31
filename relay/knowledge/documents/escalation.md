---
document_id: escalation
title: Escalation Procedures
version: "2026.1"
owner: Customer Support Policy
---
# Escalation Procedures

## When to escalate

Create or escalate a ticket when:

- payment, inventory or shipping evidence conflicts or is ambiguous;
- the provider keeps failing after the bounded retry limit;
- the policy knowledge base does not contain a passage that clearly answers the question;
- the customer asks for a human specialist, a complaint review or compensation;
- post-recovery verification fails;
- a refund approval is rejected or expires.

Escalation is a safe outcome, not a failure. The recovery agent stops acting and a human reviews the recorded evidence.

## What an escalation contains

A customer ticket records the customer and order, the conversation, the customer-confirmed summary, the issue category, policy citations, a service-evidence snapshot, and the linked recovery run. The operator sees the run's tool timeline, approvals and verification checks.

Customer-facing views never include internal traces, tool arguments, operator credentials or another customer's data. Customers see ticket status, policy citations, and a resolution summary only after verification.

## Customer commitments

- A ticket is created only after the customer confirms the summary.
- Repeated messages do not create duplicate active tickets or duplicate recovery runs.
- "Resolved" means the recovery run succeeded **and** verification passed.
- "Escalated" means a human must review; it is not a promise of a refund or a delivery date.
- The chatbot cannot approve a refund, override policy, or change order state directly.

## Response targets (demo policy)

The demo does not provide a contractual service-level agreement. A specialist should review an escalated ticket on the next business day. Support must not invent a faster commitment or compensation. In a production deployment these targets would come from the real operations policy and be versioned in the knowledge base.
