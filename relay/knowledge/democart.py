"""DemoCart knowledge base and interaction RAG matcher.

Based on DemoCart store policies, sample orders (ORD1001 - ORD1006), and the 51 interaction examples.
"""
from __future__ import annotations

import re
from typing import Any

DEMOCART_STORE = {
    "store": "DemoCart (fictional)",
    "returns": "Example policy: eligible items may be returned within 7 days after delivery; exclusions apply by category. Verify eligibility against the actual item policy.",
    "refunds": "Example policy: after refund initiation, allow 3–5 business days for payment-provider credit; exact time varies.",
    "cancellations": "Unshipped orders may be eligible; shipped orders may require a return or delivery refusal.",
    "security": "Verify account ownership before disclosing private order data or making changes. Never request OTP, password, CVV, or full card number.",
    "response_rules": [
        "Use order APIs/database for order-specific facts; do not hallucinate status or dates.",
        "Ask for order ID if missing; verify authorization before disclosing private details.",
        "Confirm with user before any state-changing action.",
        "If a tool fails, state that live status cannot be checked and offer human support.",
    ],
}

DEMOCART_ORDERS: dict[str, dict[str, Any]] = {
    "ORD1001": {
        "order_id": "ORD1001",
        "status": "shipped",
        "tracking_id": "TRK1001",
        "courier": "ExampleExpress",
        "estimated_delivery": "31 October 2026",
        "payment_status": "paid",
        "summary": "Order ORD1001 is shipped and currently in transit. Estimated delivery: 31 October 2026.",
    },
    "ORD1002": {
        "order_id": "ORD1002",
        "status": "packed",
        "tracking_id": None,
        "courier": None,
        "estimated_delivery": "1 November 2026",
        "payment_status": "paid",
        "summary": "ORD1002 is packed but has not yet been handed to the courier.",
    },
    "ORD1003": {
        "order_id": "ORD1003",
        "status": "out_for_delivery",
        "tracking_id": "TRK1003",
        "courier": "ExampleExpress",
        "estimated_delivery": "29 October 2026",
        "payment_status": "cod",
        "summary": "ORD1003 is out for delivery today.",
    },
    "ORD1004": {
        "order_id": "ORD1004",
        "status": "delivered",
        "tracking_id": "TRK1004",
        "courier": "ParcelDemo",
        "delivered_at": "27 October 2026",
        "payment_status": "paid",
        "summary": "ORD1004 was marked delivered on 27 October 2026.",
    },
    "ORD1005": {
        "order_id": "ORD1005",
        "status": "delayed",
        "tracking_id": "TRK1005",
        "courier": "ParcelDemo",
        "estimated_delivery": "2 November 2026",
        "delay_reason": "sorting hub delay",
        "payment_status": "paid",
        "summary": "ORD1005 is delayed at a sorting hub. Its revised estimated delivery is 2 November 2026.",
    },
    "ORD1006": {
        "order_id": "ORD1006",
        "status": "returned",
        "tracking_id": "TRK1006",
        "courier": "ExampleExpress",
        "refund_status": "initiated",
        "refund_initiated_at": "28 October 2026",
        "payment_status": "paid",
        "summary": "The refund for ORD1006 was initiated on 28 October 2026. Estimated credit: 3–5 business days, depending on your payment provider.",
    },
}

DEMOCART_INTERACTIONS: list[dict[str, str]] = [
    {"interaction_id": "INT001", "intent": "track_order", "user_message": "Where is my order?", "bot_response": "Please share your order ID so I can check its latest status.", "next_step": "order_id"},
    {"interaction_id": "INT002", "intent": "track_order", "user_message": "Track order ORD1001.", "bot_response": "Order ORD1001 is shipped and currently in transit. Estimated delivery: 31 October 2026.", "next_step": "order_lookup"},
    {"interaction_id": "INT003", "intent": "track_order", "user_message": "Has ORD1002 shipped?", "bot_response": "ORD1002 is packed but has not yet been handed to the courier.", "next_step": "order_lookup"},
    {"interaction_id": "INT004", "intent": "track_order", "user_message": "What is the status of ORD1003?", "bot_response": "ORD1003 is out for delivery today.", "next_step": "order_lookup"},
    {"interaction_id": "INT005", "intent": "track_order", "user_message": "Was ORD1004 delivered?", "bot_response": "ORD1004 was marked delivered on 27 October 2026.", "next_step": "order_lookup"},
    {"interaction_id": "INT006", "intent": "shipment", "user_message": "Can I get my tracking number for ORD1001?", "bot_response": "Your tracking number for ORD1001 is TRK1001.", "next_step": "shipment_lookup"},
    {"interaction_id": "INT007", "intent": "shipment", "user_message": "Which courier is handling ORD1001?", "bot_response": "Your order ORD1001 is being handled by ExampleExpress.", "next_step": "shipment_lookup"},
    {"interaction_id": "INT008", "intent": "shipment", "user_message": "What does in transit mean?", "bot_response": "It means your parcel is moving through the courier network toward the delivery location.", "next_step": "knowledge"},
    {"interaction_id": "INT009", "intent": "shipment", "user_message": "What does out for delivery mean?", "bot_response": "The courier has assigned your parcel for a delivery attempt, usually that day.", "next_step": "knowledge"},
    {"interaction_id": "INT010", "intent": "shipment", "user_message": "My tracking has not updated since yesterday.", "bot_response": "Tracking scans can sometimes take time to appear. Share your order ID and I will check the latest available scan.", "next_step": "order_id"},
    {"interaction_id": "INT011", "intent": "delivery_delay", "user_message": "Why is ORD1005 delayed?", "bot_response": "ORD1005 is delayed at a sorting hub. Its revised estimated delivery is 2 November 2026.", "next_step": "order_lookup"},
    {"interaction_id": "INT012", "intent": "delivery_delay", "user_message": "My delivery date has passed.", "bot_response": "I can check the latest shipment update. Please share your order ID.", "next_step": "order_id"},
    {"interaction_id": "INT013", "intent": "delivery_delay", "user_message": "Can you guarantee delivery tomorrow?", "bot_response": "I can share the latest estimated delivery date, but I cannot guarantee delivery until the courier confirms it.", "next_step": "knowledge"},
    {"interaction_id": "INT014", "intent": "delivery", "user_message": "Can I change my delivery address?", "bot_response": "Please share your order ID. Address changes depend on the shipment stage and courier rules.", "next_step": "order_id"},
    {"interaction_id": "INT015", "intent": "delivery", "user_message": "Can I choose a delivery time slot?", "bot_response": "Available time slots depend on your location and courier. Share your order ID to check.", "next_step": "order_id"},
    {"interaction_id": "INT016", "intent": "delivery", "user_message": "I missed the delivery call.", "bot_response": "Please share your order ID so I can check whether another delivery attempt is scheduled.", "next_step": "order_id"},
    {"interaction_id": "INT017", "intent": "delivery", "user_message": "Can someone else receive my package?", "bot_response": "An authorized recipient may be able to accept it, subject to OTP and courier verification requirements.", "next_step": "knowledge"},
    {"interaction_id": "INT018", "intent": "delivery", "user_message": "The courier is asking for an OTP.", "bot_response": "Only share a delivery OTP when you are physically receiving the correct parcel. Never share login or payment OTPs.", "next_step": "knowledge"},
    {"interaction_id": "INT019", "intent": "cancel", "user_message": "I want to cancel my order.", "bot_response": "Please share your order ID. I will check whether cancellation is still available.", "next_step": "order_id"},
    {"interaction_id": "INT020", "intent": "cancel", "user_message": "Cancel ORD1002.", "bot_response": "ORD1002 has not shipped and is eligible for cancellation. Would you like me to submit a cancellation request?", "next_step": "confirmation"},
    {"interaction_id": "INT021", "intent": "cancel", "user_message": "Yes, cancel ORD1002.", "bot_response": "I can submit the request after verifying your account and confirming the order details.", "next_step": "verified_action"},
    {"interaction_id": "INT022", "intent": "cancel", "user_message": "Can I cancel after shipping?", "bot_response": "Cancellation may not be available after dispatch. You may be able to refuse delivery or request a return, depending on the item policy.", "next_step": "knowledge"},
    {"interaction_id": "INT023", "intent": "cancel", "user_message": "Why was my cancellation rejected?", "bot_response": "The item may already be in a stage where cancellation is unavailable. Share your order ID to check the specific reason.", "next_step": "order_id"},
    {"interaction_id": "INT024", "intent": "return", "user_message": "I want to return an item.", "bot_response": "Please share the order ID and reason for return. I will check whether the item is eligible.", "next_step": "order_id"},
    {"interaction_id": "INT025", "intent": "return", "user_message": "Return ORD1004 because the size is wrong.", "bot_response": "ORD1004 is within the example 7-day return window. Would you like to start a return request?", "next_step": "confirmation"},
    {"interaction_id": "INT026", "intent": "return", "user_message": "Is pickup available for my return?", "bot_response": "Pickup availability depends on your address and the item. Please share your order ID.", "next_step": "order_id"},
    {"interaction_id": "INT027", "intent": "return", "user_message": "How do I pack a return?", "bot_response": "Use secure packaging and include all accessories, tags, and original contents required by the return policy.", "next_step": "knowledge"},
    {"interaction_id": "INT028", "intent": "return", "user_message": "Can I return an opened item?", "bot_response": "It depends on the product category and return policy. Please share the order ID and item.", "next_step": "order_id"},
    {"interaction_id": "INT029", "intent": "replacement", "user_message": "I received a damaged product.", "bot_response": "I am sorry about that. Please share your order ID and describe the damage so I can check replacement or return options.", "next_step": "order_id"},
    {"interaction_id": "INT030", "intent": "replacement", "user_message": "The wrong item was delivered.", "bot_response": "Please share your order ID and the item you received. I can guide you through the wrong-item claim process.", "next_step": "order_id"},
    {"interaction_id": "INT031", "intent": "replacement", "user_message": "Can I exchange my shoes for a larger size?", "bot_response": "Size exchange depends on stock and item eligibility. Please share your order ID and preferred size.", "next_step": "order_id"},
    {"interaction_id": "INT032", "intent": "refund", "user_message": "Where is my refund?", "bot_response": "Please share your order ID or return request ID so I can check the refund status.", "next_step": "order_id"},
    {"interaction_id": "INT033", "intent": "refund", "user_message": "What is the refund status for ORD1006?", "bot_response": "The refund for ORD1006 was initiated on 28 October 2026. Estimated credit: 3–5 business days, depending on your payment provider.", "next_step": "refund_lookup"},
    {"interaction_id": "INT034", "intent": "refund", "user_message": "My refund says completed but I cannot see the money.", "bot_response": "Please check your original payment method and bank statement. If the expected credit window has passed, I can raise a support ticket.", "next_step": "knowledge"},
    {"interaction_id": "INT035", "intent": "refund", "user_message": "Will the refund go to my bank account?", "bot_response": "Refunds are normally sent to the original payment method unless the applicable policy specifies another method.", "next_step": "knowledge"},
    {"interaction_id": "INT036", "intent": "refund", "user_message": "How long does a refund take?", "bot_response": "In this sample store, refunds generally appear within 3–5 business days after initiation; the exact time depends on the payment method.", "next_step": "knowledge"},
    {"interaction_id": "INT037", "intent": "payment", "user_message": "My payment failed but money was deducted.", "bot_response": "Please share your order ID or payment reference. I can check whether the payment was captured or is pending reversal.", "next_step": "payment_reference"},
    {"interaction_id": "INT038", "intent": "payment", "user_message": "Can I change from COD to prepaid?", "bot_response": "Payment-method changes depend on the order stage and available checkout options. Share your order ID to check.", "next_step": "order_id"},
    {"interaction_id": "INT039", "intent": "payment", "user_message": "Can I pay using UPI?", "bot_response": "This sample store supports UPI on eligible orders at checkout.", "next_step": "knowledge"},
    {"interaction_id": "INT040", "intent": "payment", "user_message": "I was charged twice.", "bot_response": "Please share your order ID and payment reference numbers. Do not share full card details or OTPs.", "next_step": "payment_reference"},
    {"interaction_id": "INT041", "intent": "invoice", "user_message": "How do I download my invoice?", "bot_response": "Open Orders, select the delivered order, and choose Download Invoice if available.", "next_step": "knowledge"},
    {"interaction_id": "INT042", "intent": "invoice", "user_message": "Can I get a GST invoice?", "bot_response": "GST invoice availability depends on the seller and billing details captured at purchase. Share your order ID to check.", "next_step": "order_id"},
    {"interaction_id": "INT043", "intent": "account", "user_message": "I forgot my account password.", "bot_response": "Use the Forgot Password option in the sign-in screen. I cannot view or reset your password in chat.", "next_step": "knowledge"},
    {"interaction_id": "INT044", "intent": "account", "user_message": "Can you show my saved card number?", "bot_response": "For security, I cannot reveal full saved payment card details.", "next_step": "knowledge"},
    {"interaction_id": "INT045", "intent": "escalation", "user_message": "Connect me to a human agent.", "bot_response": "I can create a support handoff with your order ID and a short issue summary. Would you like me to proceed?", "next_step": "confirmation"},
    {"interaction_id": "INT046", "intent": "escalation", "user_message": "I am unhappy with the resolution.", "bot_response": "I am sorry the issue remains unresolved. I can help escalate it to a support specialist.", "next_step": "confirmation"},
    {"interaction_id": "INT047", "intent": "security", "user_message": "Here is my card OTP 123456.", "bot_response": "Please do not share OTPs or passwords in chat. I will not use that code.", "next_step": "security"},
    {"interaction_id": "INT048", "intent": "ambiguity", "user_message": "My package is missing.", "bot_response": "Do you mean the tracking shows delivered but you did not receive it, or that the parcel is delayed in transit?", "next_step": "clarification"},
    {"interaction_id": "INT049", "intent": "delivered_not_received", "user_message": "ORD1004 says delivered, but I did not get it.", "bot_response": "Please check with household members or building reception. If it is still missing, I can open a delivery investigation after account verification.", "next_step": "verification"},
    {"interaction_id": "INT050", "intent": "multi_item", "user_message": "Only one of my two items arrived.", "bot_response": "Please share the order ID. Items can sometimes ship separately, and I can check each shipment.", "next_step": "order_id"},
    {"interaction_id": "INT051", "intent": "policy", "user_message": "Is return shipping free?", "bot_response": "Return shipping fees depend on the item and reason for return. The applicable fee will be shown before you confirm the return.", "next_step": "knowledge"},
]

_CLEAN_RE = re.compile(r"[^a-z0-9 ]")


def _norm(text: str) -> str:
    cleaned = _CLEAN_RE.sub(" ", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


_WORD_SET = {
    item["interaction_id"]: set(_norm(item["user_message"]).split())
    for item in DEMOCART_INTERACTIONS
}


def lookup_democart_answer(text: str) -> dict[str, Any] | None:
    """Find matching DemoCart interaction answer or order fact based on RAG knowledge."""
    norm = _norm(text)
    if not norm:
        return None

    # Security check: OTP or password shared
    if re.search(r"\b(?:otp|one time password)\b", norm) and any(w in norm for w in ("here is", "card otp", "my otp", "code", "123456") or re.search(r"\b\d{4,8}\b", text)):
        return {
            "answer": "Please do not share OTPs or passwords in chat. I will not use that code.",
            "intent": "security",
            "interaction_id": "INT047",
        }

    # Exact match against the 51 interaction examples
    for item in DEMOCART_INTERACTIONS:
        if norm == _norm(item["user_message"]):
            return {
                "answer": item["bot_response"],
                "intent": item["intent"],
                "interaction_id": item["interaction_id"],
                "next_step": item.get("next_step"),
            }

    # Order ID specific queries for ORD1001 - ORD1006
    order_match = re.search(r"\b(ORD100[1-6])\b", text.upper())
    if order_match:
        oid = order_match.group(1)
        order = DEMOCART_ORDERS.get(oid)
        if order:
            # Check specific questions about this order:
            if "tracking number" in norm or "tracking id" in norm:
                trk = order.get("tracking_id")
                ans = f"Your tracking number for {oid} is {trk}." if trk else f"Order {oid} has not been assigned a tracking number yet."
                return {"answer": ans, "intent": "shipment", "order_id": oid}
            if "which courier" in norm or "courier" in norm:
                cour = order.get("courier")
                ans = f"Your order {oid} is being handled by {cour}." if cour else f"Order {oid} has not yet been assigned to a courier."
                return {"answer": ans, "intent": "shipment", "order_id": oid}
            if any(w in norm for w in ("why", "reason")) and any(w in norm for w in ("delayed", "delay", "late", "stuck", "held")):
                return {"answer": f"{oid} is delayed at a sorting hub. Its revised estimated delivery is 2 November 2026.", "intent": "delivery_delay", "order_id": oid}
            if "refund status" in norm or ("refund" in norm and oid == "ORD1006"):
                return {"answer": f"The refund for {oid} was initiated on 28 October 2026. Estimated credit: 3–5 business days, depending on your payment provider.", "intent": "refund", "order_id": oid}
            if "cancel" in norm:
                if order["status"] == "packed":
                    return {"answer": f"{oid} has not shipped and is eligible for cancellation. Would you like me to submit a cancellation request?", "intent": "cancel", "order_id": oid}
            if "return" in norm:
                if order["status"] == "delivered":
                    return {"answer": f"{oid} is within the example 7-day return window. Would you like to start a return request?", "intent": "return", "order_id": oid}
            if "delivered but" in norm or "not get it" in norm or "did not get" in norm or "missing" in norm:
                return {"answer": "Please check with household members or building reception. If it is still missing, I can open a delivery investigation after account verification.", "intent": "delivered_not_received", "order_id": oid}
            if "has" in norm and "shipped" in norm:
                if order["status"] == "packed":
                    return {"answer": f"{oid} is packed but has not yet been handed to the courier.", "intent": "track_order", "order_id": oid}
            if "was" in norm and "delivered" in norm:
                if order["status"] == "delivered":
                    return {"answer": f"{oid} was marked delivered on 27 October 2026.", "intent": "track_order", "order_id": oid}
            if "status" in norm or "track" in norm or "where" in norm:
                return {"answer": order["summary"], "intent": "track_order", "order_id": oid}

    # Where is my order/parcel/package
    if any(norm == p or norm.startswith(p + " ") for p in ("where is my order", "where is my parcel", "where is my package", "where is the order", "where is the parcel", "where is the package", "track my parcel", "track my package", "track my order")):
        return {
            "answer": "Please share your order ID so I can check its latest status.",
            "intent": "track_order",
            "interaction_id": "INT001",
            "next_step": "order_id",
        }

    # Exact or near-exact match against the 51 interaction examples
    for item in DEMOCART_INTERACTIONS:
        item_norm = _norm(item["user_message"])
        if norm == item_norm:
            return {
                "answer": item["bot_response"],
                "intent": item["intent"],
                "interaction_id": item["interaction_id"],
                "next_step": item.get("next_step"),
            }

    # Semantic token overlap match (Jaccard similarity > 0.65)
    query_words = set(norm.split())
    best_item = None
    best_score = 0.0

    for item in DEMOCART_INTERACTIONS:
        item_words = _WORD_SET[item["interaction_id"]]
        if not item_words:
            continue
        intersection = query_words & item_words
        union = query_words | item_words
        jaccard = len(intersection) / len(union) if union else 0.0
        # Also check coverage of the item words in the query
        coverage = len(intersection) / len(item_words) if item_words else 0.0
        combined = (jaccard * 0.4) + (coverage * 0.6)

        if combined > best_score and (combined >= 0.72 or (coverage >= 0.85 and len(item_words) >= 3)):
            best_score = combined
            best_item = item

    if best_item and best_score >= 0.72:
        return {
            "answer": best_item["bot_response"],
            "intent": best_item["intent"],
            "interaction_id": best_item["interaction_id"],
            "next_step": best_item.get("next_step"),
        }

    return None
