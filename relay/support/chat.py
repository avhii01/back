"""Customer support chat: slot filling + grounded policy guidance + ticket confirmation.

Responsibilities are deliberately narrow. The chat:
  * understands/asks, retrieves policy, verifies order ownership, reads live evidence,
  * creates a ticket only after explicit confirmation,
  * explains verified status.
It never updates an order, approves a refund, or runs the recovery graph. Those remain
in the existing runner/tools/policy/approvals path.
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime

from sqlalchemy import select

from relay.context import AppContext
from relay.errors import ToolError
from relay.knowledge.democart import DEMOCART_ORDERS, lookup_democart_answer
from relay.knowledge.retrieval import answer_policy_question
from relay.models import ChatConversation, ChatMessage, Customer, Order, SupportTicket, utcnow
from relay.money import format_cents
from relay.support.evidence import collect
from relay.support.llm import phrase_answer
from relay.support.tickets import create_customer_ticket, sync_ticket, ticket_customer_json
from relay.untrusted import detect_injection, sanitize_text

ORDER_RE = re.compile(r"\bORD-?[A-Z0-9]{4,12}\b", re.IGNORECASE)
AFFIRM = {"yes", "y", "confirm", "confirmed", "create ticket", "create the ticket", "go ahead", "please do",
          "ok", "okay", "sure", "yes please", "confirm ticket", "do it", "c"}
DENY = {"no", "n", "cancel", "stop", "not now", "don't", "do not", "never mind", "nevermind", "x"}
ORDER_WORDS = ("order", "payment", "charged", "charge", "deduct", "debit", "pending", "refund", "shipping",
               "shipment", "delivery", "deliver", "stock", "inventory", "reserve", "cancel", "tracking",
               "parcel", "late", "return")
POLICY_WORDS = ("policy", "how long", "how do", "what happens", "what is", "can i", "do you", "when will",
                "how much", "where is", "why is", "is it", "are refunds", "refund take", "business day")
# Questions about how Relay behaves are informational even when they mention an order
# concept, e.g. "Can the chatbot approve my refund automatically?". They should be
# answered from policy instead of turning into an order-intake request.
META_POLICY_WORDS = ("chatbot", "relay", "policy", "automatically", "usually", "business day", "how long",
                     "how much", "what happens")
STATUS_WORDS = ("status", "progress", "update", "ticket", "resolved", "resolution", "happening", "going on",
                "done", "fixed", "refund", "approved")
TICKET_REQUESTS = ("raise a ticket", "raise ticket", "create a ticket", "create ticket", "open a ticket",
                   "open ticket", "file a ticket", "file ticket", "need a ticket", "want a ticket",
                   "support ticket", "talk to a human", "speak to a human", "human specialist",
                   "human agent", "speak to someone", "i have a query", "have a query", "raise a query",
                   "i have an issue", "i have a problem", "report an issue", "report a problem",
                   "something is wrong with my order", "i need help with an order", "help with my order")
TRACK_REQUESTS = ("track my order", "track order", "track an order", "where is my order", "order status",
                  "tracking status", "delivery status", "shipment status", "has my order shipped")
ORDER_LIST_REQUESTS = ("my orders", "show my orders", "list my orders", "order history", "previous orders",
                       "recent orders", "past orders", "my previous orders", "show my previous orders",
                       "what are my orders", "which orders do i have")
GREETINGS = ("hi", "hello", "hey", "hi there", "hello there", "good morning", "good afternoon", "good evening")
HELP_REQUESTS = ("help", "menu", "options", "start", "start over")
HOW_ARE_YOU_PHRASES = ("how are you", "how is it going", "hows it going", "how are things",
                       "how do you do", "how have you been")
WELLBEING_PHRASES = ("im fine", "i am fine", "im good", "i am good", "im doing well", "i am doing well",
                     "im great", "i am great", "doing well", "doing great", "not bad", "pretty good",
                     "all good")
THANKS_PHRASES = ("thanks", "thank you", "thanks a lot", "thank you so much", "appreciate it", "ty", "thx")
IDENTITY_PHRASES = ("who are you", "what are you", "what can you do", "what do you do", "are you a bot",
                    "are you human", "are you a robot")
FAREWELL_PHRASES = ("bye", "goodbye", "see you", "see you later", "take care")
ACKNOWLEDGEMENT_PHRASES = (
    "ok", "okay", "k", "kk", "alright", "all right", "got it", "fine", "thats fine", "that's fine",
    "cool", "great", "perfect", "sounds good", "understood", "noted", "sure", "yes", "yeah", "yep",
    "no problem", "no worries", "done", "nice", "good", "awesome", "okay fine", "ok fine", "ok sure",
    "okay sure", "ok cool", "okay cool", "ok got it", "okay got it", "ok thanks", "okay thanks",
    "ok thank you", "okay thank you", "ok then", "okay then",
)
NO_MORE_QUERIES_PHRASES = {
    "no", "nope", "nah", "nothing", "nothing else", "no query", "no queries",
    "i dont have any queries", "i don't have any queries", "dont have any queries", "don't have any queries",
    "no more queries", "no other queries", "no more questions", "no other questions",
    "i have no queries", "have no queries", "no questions", "i have no questions",
    "thats all", "that is all", "that's all", "im done", "i am done", "all done",
    "no thats all", "no that is all", "no that's all", "no thank you", "no thanks",
    "end the conversation", "end conversation", "end chat", "close the chat", "close chat",
    "stop the conversation", "stop conversation", "no nothing", "stop",
}
SMALL_TALK_STAGES = {"idle", "awaiting_order", "awaiting_order_choice", "awaiting_issue", "awaiting_details"}
RECENT_ORDER_LIMIT = 3
MAIN_OPTIONS = [
    {"label": "Raise a support ticket", "value": "raise_ticket"},
    {"label": "Track an order", "value": "track_order"},
    {"label": "Refund or cancellation", "value": "refund"},
    {"label": "Ask a policy question", "value": "policy"},
    {"label": "Something else", "value": "other"},
]
ISSUE_OPTIONS = [
    {"label": "Payment captured but order pending", "value": "payment_pending"},
    {"label": "Refund or cancellation", "value": "refund"},
    {"label": "Order stuck or inventory issue", "value": "inventory"},
    {"label": "Delivery or shipping delay", "value": "shipping"},
    {"label": "Conflicting payment records", "value": "payment_conflict"},
]
HINT_LABELS = {
    "payment_pending": "payment captured but the order is still pending",
    "refund": "a refund or cancellation",
    "inventory": "an order stuck on inventory",
    "shipping": "a delivery or shipping delay",
    "payment_conflict": "conflicting payment records",
    "track_order": "tracking the order",
}
OPTION_KEYWORDS = {
    "raise_ticket": ("ticket", "raise", "create", "open", "human", "specialist", "agent"),
    "track_order": ("track", "tracking", "where", "shipped"),
    "refund": ("refund", "cancel", "return", "money"),
    "policy": ("policy", "question"),
    "other": ("other", "else", "something"),
    "payment_pending": ("payment", "pending", "captured", "paid", "deducted", "debited"),
    "inventory": ("inventory", "stock", "stuck", "reserve", "reservation"),
    "shipping": ("shipping", "shipment", "delivery", "deliver", "late"),
    "payment_conflict": ("conflict", "conflicting", "mismatch", "charges"),
}
PAYMENT_LABELS = {"captured_ok": "captured and confirmed", "conflict": "conflicting records",
                  "not_captured": "not captured", "absent": "no payment record", "refunded": "refunded",
                  "unknown": "unknown"}
RESERVATION_LABELS = {"none": "no reservation", "usable": "reserved", "multiple": "multiple reservations",
                      "mismatch": "a mismatched reservation"}
SHIPMENT_LABELS = {"none": "no shipment yet", "one": "shipment exists", "multiple": "multiple shipments"}


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _state(conversation: ChatConversation) -> dict:
    base = {"stage": "idle", "order_id": None, "category": None, "draft": None, "ticket_id": None,
            "menu": None, "category_hint": None, "order_options": None}
    base.update(conversation.state or {})
    return base


def _msg_json(m: ChatMessage) -> dict:
    return {"id": m.id, "role": m.role, "content": m.content, "citations": m.citations or [],
            "meta": m.meta or {}, "created_at": m.created_at.isoformat()}


def _append(s, conversation: ChatConversation, role: str, content: str, *,
            citations: list[dict] | None = None, meta: dict | None = None) -> ChatMessage:
    message = ChatMessage(conversation_id=conversation.id, role=role, content=content,
                          citations=citations or [], meta=meta or {})
    s.add(message)
    conversation.updated_at = utcnow()
    s.flush()
    return message


def create_conversation(app: AppContext, customer: Customer) -> ChatConversation:
    with app.session() as s:
        conversation = ChatConversation(id=_new_id("conv"), customer_id=customer.id,
                                        state={"stage": "idle", "menu": "main"})
        s.add(conversation)
        s.commit()
        s.refresh(conversation)
        return conversation


def conversation_for_customer(app: AppContext, customer: Customer, conversation_id: str) -> ChatConversation | None:
    with app.session() as s:
        conversation = s.get(ChatConversation, conversation_id)
        if conversation is None or conversation.customer_id != customer.id:
            return None
        return conversation


def conversation_json(app: AppContext, conversation: ChatConversation) -> dict:
    state = _state(conversation)
    with app.session() as s:
        messages = list(s.scalars(select(ChatMessage).where(
            ChatMessage.conversation_id == conversation.id).order_by(ChatMessage.id)).all())
        ticket_ids = list(s.scalars(select(SupportTicket.id).where(
            SupportTicket.conversation_id == conversation.id).order_by(SupportTicket.created_at)).all())
    tickets = [t for tid in ticket_ids if (t := ticket_customer_json(app, tid)) is not None]
    draft = state.get("draft")
    pending = None
    if state.get("stage") == "awaiting_confirmation" and draft:
        pending = {k: draft.get(k) for k in ("order_id", "category", "summary", "citations",
                                             "requires_human", "policy_insufficient", "policy_conflicting")}
    return {"id": conversation.id, "status": conversation.status, "stage": state.get("stage"),
            "order_id": state.get("order_id"), "created_at": conversation.created_at.isoformat(),
            "updated_at": conversation.updated_at.isoformat(), "messages": [_msg_json(m) for m in messages],
            "pending_ticket": pending, "tickets": tickets}


def _affirmative(text: str) -> bool:
    clean = re.sub(r"[^a-z ]", "", text.lower()).strip()
    clean = re.sub(r"\s+", " ", clean)
    if clean in AFFIRM or clean in ("c", "y", "confirm c", "conf"):
        return True
    return clean.startswith("yes") or clean.startswith("confirm") or clean.startswith("c ")


def _negative(text: str) -> bool:
    clean = re.sub(r"[^a-z ]", "", text.lower()).strip()
    clean = re.sub(r"\s+", " ", clean)
    if clean in DENY or clean in ("n", "x", "cancel n", "canc"):
        return True
    return clean.startswith("no") or clean.startswith("cancel") or clean.startswith("n ")


def _is_no_or_end(text: str) -> bool:
    clean = re.sub(r"[^a-z ]", "", text.lower()).strip()
    clean = re.sub(r"\s+", " ", clean)
    if clean in NO_MORE_QUERIES_PHRASES:
        return True
    return any(clean == p or clean.startswith(p + " ") or clean.endswith(" " + p) or (" " + p + " ") in clean for p in (
        "no query", "no queries", "no more queries", "no more questions",
        "end the conversation", "end conversation", "close the chat", "close chat",
        "dont have any queries", "don't have any queries", "dont have any other queries",
        "i dont have any queries", "i don't have any queries", "nothing else", "thats all",
        "no nothing", "stop the conversation"
    ))


def _is_close_affirmative(text: str) -> bool:
    clean = re.sub(r"[^a-z ]", "", text.lower()).strip()
    clean = re.sub(r"\s+", " ", clean)
    if clean in AFFIRM or clean in ("yes", "y", "sure", "ok", "okay", "yep", "yeah", "close", "close the chat", "close chat", "close it", "done"):
        return True
    return any(p in clean for p in (
        "can close", "close the chat", "close chat", "close it", "yes close", "yes you can", "you can close",
        "please close", "can close the chat"
    ))


AGENT_REQUEST_PHRASES = (
    "connect me to the agent", "connect me to an agent", "connect me to agent",
    "connect to the agent", "connect to an agent", "connect to agent", "connect with agent",
    "talk to the agent", "talk to an agent", "talk to agent",
    "speak to the agent", "speak to an agent", "speak to agent", "speak with agent",
    "human agent", "talk to a human", "speak to a human", "talk to human", "speak to human",
    "connect to human", "connect me to human", "connect to representative",
    "talk to representative", "speak to representative", "connect me to representative",
    "customer care", "customer support executive", "customer service agent",
    "call me", "can you call me", "agent call", "have someone call me",
    "transfer to agent", "transfer me to an agent", "transfer to human",
)


def _is_agent_request(text: str) -> bool:
    clean = re.sub(r"[^a-z ]", "", text.lower()).strip()
    clean = re.sub(r"\s+", " ", clean)
    if any(clean == p or clean.startswith(p + " ") or clean.endswith(" " + p) or (" " + p + " ") in clean for p in AGENT_REQUEST_PHRASES):
        return True
    return clean in ("agent", "human", "live agent", "call agent")


def _greeting(text: str) -> bool:
    clean = re.sub(r"[^a-z ]", "", text.lower()).strip()
    clean = re.sub(r"\s+", " ", clean)
    if clean in GREETINGS:
        return True
    return any(clean == g or clean.startswith(g + " ") for g in GREETINGS)


def _is_ticket_intake_request(text: str) -> bool:
    clean = re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()
    clean = re.sub(r"\s+", " ", clean)
    if clean in ("raise ticket", "raise a ticket", "raise support ticket", "raise a support ticket",
                 "create ticket", "create a ticket", "open ticket", "open a ticket",
                 "support ticket", "file ticket", "file a ticket", "need ticket", "need a ticket",
                 "ticket", "tickets", "raise", "raise_ticket"):
        return True
    return any(clean == p or clean.startswith(p + " ") or clean.endswith(" " + p) or (" " + p + " ") in clean for p in (
        "raise a support ticket", "raise support ticket", "raise a ticket", "raise ticket",
        "create a ticket", "create ticket", "open a ticket", "open ticket",
        "support ticket", "file a ticket", "need a ticket", "help me raise a ticket"
    ))


def _ticket_request(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in TICKET_REQUESTS)


def _track_request(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in TRACK_REQUESTS)


def _order_list_request(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in ORDER_LIST_REQUESTS)


def _is_ticket_update_query(text: str) -> bool:
    if ORDER_RE.search(text):
        return False
    clean = re.sub(r"[^a-z0-9 ]", "", text.lower())
    clean = re.sub(r"\s+", " ", clean).strip()
    return any(p in clean for p in (
        "update on my ticket", "update on the ticket", "update on ticket",
        "status of my ticket", "status of the ticket", "status of ticket",
        "my ticket status", "ticket status", "ticket update",
        "whats the update", "what is the update", "what s the update",
        "check my ticket", "latest ticket", "last ticket",
        "update of my ticket", "how is my ticket", "what is the status"
    ))


def _is_general_query_prompt(text: str) -> bool:
    clean = re.sub(r"[^a-z ]", "", text.lower())
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean in (
        "i have a query", "have a query", "raise a query", "query",
        "i have a question", "have a question"
    ) or clean.startswith("i have a query") or clean.startswith("have a query")


def _is_payment_issue_topic(text: str) -> bool:
    clean = re.sub(r"[^a-z0-9 ]", "", text.lower())
    clean = re.sub(r"\s+", " ", clean).strip()
    return any(p in clean for p in (
        "related to order payment", "order payment", "payment issue",
        "payment orders issues", "payment order issue", "pending orders",
        "pending order", "payment", "paid"
    ))


def _small_talk(text: str, customer_name: str) -> tuple[str, str] | None:
    """Return (kind, reply) for conversational messages that are not support requests."""
    clean = re.sub(r"[^a-z ]", "", text.lower())
    clean = re.sub(r"\s+", " ", clean).strip()
    if not clean or len(clean.split()) > 14:
        return None

    # Acknowledgement phrases: ok, okay, got it, fine, cool, etc.
    if clean in ACKNOWLEDGEMENT_PHRASES or any(clean == p or clean.startswith(p + " ") for p in ("ok", "okay", "alright", "got it", "thats fine", "that is fine")):
        return "ack", "Okay, that's fine! Is there anything else I can help you with?"

    for greeting in sorted(GREETINGS, key=len, reverse=True):
        if clean == greeting:
            return "greeting", "Hi, how can I help you?"
        if clean.startswith(greeting + " "):
            rest = clean[len(greeting) + 1:].strip()
            if rest in ("", "there"):
                return "greeting", "Hi, how can I help you?"
            if any(rest == p or rest.startswith(p + " ") for p in HOW_ARE_YOU_PHRASES):
                return "greeting", "Hi! I'm doing well, thanks for asking. How can I help you?"

    if any(clean == p or clean.startswith(p + " ") for p in HOW_ARE_YOU_PHRASES):
        return "how_are_you", "I'm doing well, thanks for asking! How can I help you today?"
    if any(clean == p or clean.startswith(p + " ") for p in WELLBEING_PHRASES):
        return "wellbeing", "Glad to hear it! How can I help you today?"
    if any(clean == p or clean.startswith(p + " ") for p in THANKS_PHRASES):
        return "thanks", "You're welcome! Is there anything else I can help you with?"
    if any(clean == p or clean.startswith(p + " ") for p in IDENTITY_PHRASES):
        return "identity", ("I'm Relay's support assistant. I can show your recent orders, check live order "
                            "status, answer policy questions, and prepare a support ticket for your confirmation.")
    if clean in FAREWELL_PHRASES or any(clean.startswith(p + " goodbye") for p in GREETINGS):
        return "farewell", "Take care! Come back anytime you need help with an order."
    return None


def _small_talk_followup(stage: str, state: dict) -> tuple[str, list[dict]]:
    """Keep the current guided step visible after a conversational reply."""
    if stage == "awaiting_order_choice":
        return ("Which of the orders above is this about?",
                [dict(option) for option in (state.get("order_options") or [])])
    if stage == "awaiting_issue":
        return "Which issue below fits best?", [dict(option) for option in ISSUE_OPTIONS]
    if stage == "awaiting_details":
        order_id = state.get("order_id")
        if order_id:
            return f"What happened with order {order_id}?", []
        return "What happened with the order?", []
    if stage == "awaiting_order":
        return _ask_order(state.get("category_hint")), []
    return "", [dict(option) for option in MAIN_OPTIONS]


def _recent_orders(app: AppContext, customer: Customer) -> list[Order]:
    """Return the customer's newest orders from order history, regardless of status."""
    with app.session() as s:
        return list(s.scalars(select(Order).where(Order.customer_id == customer.id)
                              .order_by(Order.created_at.desc(), Order.id.desc())
                              .limit(RECENT_ORDER_LIMIT)))


def _order_choices(app: AppContext, customer: Customer) -> tuple[str, list[dict]]:
    orders = _recent_orders(app, customer)
    if not orders:
        return ("I couldn't find any recent orders on your account. "
                "What is the order number? (for example ORD-ABC123)", [])
    options = [{"label": f"{o.id} · {o.status} · {format_cents(o.total_cents, o.currency)}",
                "value": o.id} for o in orders]
    count = len(orders)
    noun = "order" if count == 1 else "orders"
    return (f"I found {count} recent {noun} on your account. Which one is this about?\n\n"
            "Choose one below, or send a different order number.", options)


def _issue_choices(order: Order, state: dict, meta: dict) -> tuple[str, list[dict]]:
    """Show the issue categories after the customer has chosen an order."""
    options = [dict(option) for option in ISSUE_OPTIONS]
    state.update(stage="awaiting_issue", order_id=order.id, menu="issues", draft=None,
                 category=None, category_hint=None, order_options=None)
    meta.update(intent="issue_menu", order_id=order.id, options=options)
    return ("Thanks — I've got order {order_id}. What is this issue about?\n\n"
            "Choose one below, or describe it in your own words.".format(order_id=order.id), options)


def _details_prompt(order: Order, option: dict, state: dict, meta: dict) -> tuple[str, list[dict]]:
    """Ask for the free-text details after the customer has chosen a category."""
    hint = option["value"]
    label = HINT_LABELS.get(hint, option["label"].lower())
    state.update(stage="awaiting_details", order_id=order.id, menu=None, draft=None,
                 category=None, category_hint=hint, order_options=None)
    meta.update(intent="order_details_required", order_id=order.id, category_hint=hint)
    return (f"Got it: {label}. Can you describe what happened with order {order.id}? "
            "Include any error message or date if you have it. I'll check the live payment, inventory "
            "and shipping records before preparing a ticket."), []


def _show_order_choices(app: AppContext, customer: Customer, state: dict, meta: dict,
                        category_hint: str | None) -> str:
    text, options = _order_choices(app, customer)
    if options:
        state.update(stage="awaiting_order_choice", menu=None, category_hint=category_hint,
                     order_options=options, order_id=None, draft=None, category=None)
        meta.update(intent="order_choice", category_hint=category_hint, options=options)
    else:
        state.update(stage="awaiting_order", menu=None, category_hint=category_hint,
                     order_options=None, order_id=None, draft=None, category=None)
        meta.update(intent="order_number_required", category_hint=category_hint)
    return text


def _main_menu(customer: Customer) -> tuple[str, list[dict]]:
    return (f"Hi {customer.name}. How can I help you today?\n\n"
            "Choose an option below, or describe the issue in your own words.",
            [dict(option) for option in MAIN_OPTIONS])


def _ask_order(category_hint: str | None) -> str:
    if category_hint == "track_order":
        return "Sure. What is the order number? (for example ORD-ABC123)"
    if category_hint in HINT_LABELS:
        return (f"Got it: {HINT_LABELS[category_hint]}. What is the order number, and what went wrong? "
                "(for example ORD-ABC123 and a short description)")
    return ("What is the order number, and what went wrong? "
            "(for example ORD-ABC123 and a short description)")


def _match_option(text: str, options: list[dict], *, fuzzy: bool = False) -> dict | None:
    clean = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    clean = re.sub(r"\s+", " ", clean).strip()
    if not clean:
        return None
    if clean.isdigit():
        index = int(clean) - 1
        return options[index] if 0 <= index < len(options) else None
    for option in options:
        if clean in (option["label"].lower(), option["value"].lower()):
            return option
    if not fuzzy:
        return None
    best: dict | None = None
    best_score = 0
    for option in options:
        score = sum(1 for word in OPTION_KEYWORDS.get(option["value"], ())
                    if re.search(rf"\b{re.escape(word)}\b", clean))
        if score > best_score:
            best, best_score = option, score
    return best if best_score > 0 else None


def _strip_escalation_procedures(text: str) -> str:
    if not text or ("Escalation Procedures" not in text and "escalation" not in text.lower()):
        return text or ""
    chunks = re.split(r'(?=According to ")', text)
    kept = []
    for chunk in chunks:
        chunk_str = chunk.strip()
        if not chunk_str:
            continue
        lower = chunk_str.lower()
        if ('according to "escalation procedures"' in lower
                or 'according to "escalation"' in lower
                or '# escalation procedures' in lower):
            continue
        kept.append(chunk_str)
    result = " ".join(kept)
    return re.sub(r"\n{3,}", "\n\n", result).strip()


def _grounded(app: AppContext, question: str, docs: list[str] | None = None) -> dict:
    result = answer_policy_question(app, question, required_document_ids=docs)
    answer, used_llm, citations = phrase_answer(app, question, result["answer"], result["citations"])
    answer = _strip_escalation_procedures(answer)
    citations = [
        c for c in citations
        if c.get("document_id") != "escalation" and "escalation" not in c.get("title", "").lower()
    ]
    result["answer"] = answer
    result["citations"] = citations
    result["engine"] = "llm_grounded" if used_llm else "extractive_grounded"
    return result


def _policy_text(policy: dict) -> str:
    return policy["answer"] if policy.get("citations") else policy["answer"]


def _order_for_customer(app: AppContext, customer: Customer, order_id: str) -> Order | None:
    with app.session() as s:
        order = s.get(Order, order_id)
        if order is not None and order.customer_id == customer.id:
            return order
    if order_id in DEMOCART_ORDERS:
        info = DEMOCART_ORDERS[order_id]
        return Order(
            id=order_id,
            customer_id=customer.id,
            customer_email=customer.email,
            sku="DEMO-ITEM",
            quantity=1,
            total_cents=4999,
            currency="USD",
            status=info["status"],
            shipment_id=info.get("tracking_id"),
        )
    return None


def _draft(app: AppContext, customer: Customer, order: Order, message: str, correlation_id: str) -> dict:
    try:
        evidence = collect(app, order.id, correlation_id=correlation_id, message=message)
    except ToolError:
        return {}
    if evidence is None:
        return {}
    policy = _grounded(app, f"{message} {evidence.category_label}", evidence.policy_document_ids)
    requires_human = (not evidence.eligible_for_recovery or policy["insufficient"] or policy["conflicting"]
                      or evidence.category in ("REFUND_REQUEST", "REFUND_STATUS", "GENERAL_ESCALATION"))
    return {
        "order_id": order.id, "category": evidence.category, "category_label": evidence.category_label,
        "summary": evidence.summary, "citations": policy["citations"], "evidence": evidence.snapshot,
        "policy_answer": policy["answer"], "policy_mode": policy["retrieval_mode"],
        "policy_label": policy["retrieval_label"], "policy_engine": policy["engine"],
        "policy_insufficient": policy["insufficient"], "policy_conflicting": policy["conflicting"],
        "requires_human": requires_human, "dispatch": not requires_human,
        "created_at": datetime.utcnow().isoformat(),
    }


def _draft_intro(draft: dict) -> str:
    lines = [f"I found order {draft['order_id']}. {draft['summary']}"]
    clean_policy = _strip_escalation_procedures(draft.get("policy_answer", ""))
    if clean_policy:
        lines.extend(["", clean_policy])
    lines.append("")
    if draft["requires_human"]:
        lines.append("I cannot safely start automated recovery from this evidence, but I can create a ticket "
                     "for a human specialist.")
    else:
        lines.append("I can create a support ticket and Relay will investigate the live payment, inventory and "
                     "shipping records. Automated recovery may fix the order; any refund still needs a human "
                     "operator to approve it.")
    lines.append("Reply confirm (or C) to create the ticket, or cancel (or N).")
    return "\n".join(lines)


def _status_answer(app: AppContext, order: Order, correlation_id: str) -> dict | None:
    """Answer a tracking question from live evidence; no ticket exists yet."""
    try:
        evidence = collect(app, order.id, correlation_id=correlation_id, message="order status")
    except ToolError:
        return None
    if evidence is None:
        return None
    healthy = ((order.status == "shipped" and evidence.payment_state == "captured_ok"
                and evidence.shipment_state == "one")
               or (order.status == "refunded" and evidence.payment_state == "refunded"))
    text = (f"Order {order.id} is currently '{order.status}'.\n\n"
            f"Payment: {PAYMENT_LABELS.get(evidence.payment_state, evidence.payment_state)}. "
            f"Inventory: {RESERVATION_LABELS.get(evidence.reservation_state, evidence.reservation_state)}. "
            f"Shipment: {SHIPMENT_LABELS.get(evidence.shipment_state, evidence.shipment_state)}.")
    if healthy:
        text += "\n\nEverything needed for this order is complete, so no ticket is needed."
    else:
        text += "\n\nThis needs an investigation, so I can prepare a ticket for your confirmation."
    return {"text": text, "healthy": healthy}


def _process_order(app: AppContext, customer: Customer, order: Order, text: str, correlation_id: str,
                   state: dict, meta: dict, injection: list[str]) -> tuple[str, list[dict]]:
    """Read live evidence for a selected order and either answer status or offer a ticket."""
    if order.id in DEMOCART_ORDERS:
        info = DEMOCART_ORDERS[order.id]
        state.update(stage="idle", order_id=order.id, category=None, draft=None, menu="main",
                     category_hint=None, order_options=None)
        meta.update(intent="order_status", healthy=True)
        return info["summary"], []
    tracking = state.get("category_hint") == "track_order" or _track_request(text)
    status = _status_answer(app, order, correlation_id) if tracking else None
    if status is not None and status["healthy"]:
        state.update(stage="idle", order_id=order.id, category=None, draft=None, menu="main",
                     category_hint=None, order_options=None)
        meta.update(intent="order_status", healthy=True)
        return status["text"], []
    draft = _draft(app, customer, order, text, correlation_id)
    if not draft:
        meta["intent"] = "evidence_unavailable"
        return "I could not read that order's service records right now. Please try again shortly.", []
    state.update(stage="awaiting_confirmation", order_id=order.id, category=draft["category"],
                 draft=draft, menu=None, order_options=None)
    assistant = _draft_intro(draft)
    if status is not None:
        assistant = status["text"] + "\n\n" + assistant
    meta.update(intent="order_issue", category=draft["category"],
                retrieval_mode=draft["policy_mode"], retrieval_label=draft["policy_label"],
                engine=draft["policy_engine"], requires_human=draft["requires_human"],
                options=[
                    {"label": "Confirm (C)", "value": "confirm"},
                    {"label": "Cancel (N)", "value": "cancel"},
                ])
    citations = draft["citations"]
    if injection:
        assistant = ("I can't follow instructions that try to change Relay's rules or approve "
                     "payments/refunds. I can still help with the order.\n\n" + assistant)
    return assistant, citations


def _ticket_status_text(app: AppContext, ticket_id: str) -> tuple[str, dict | None]:
    ticket = ticket_customer_json(app, ticket_id)
    if ticket is None:
        return "I could not load that ticket. A support specialist can help from the operator dashboard.", None
    status = ticket["status"]
    if status == "resolved":
        return f"Ticket {ticket['id']} is resolved. {ticket.get('resolution') or ''}", ticket
    if status == "escalated":
        return (f"Ticket {ticket['id']} is escalated. {ticket.get('resolution_summary') or ''} "
                "A human specialist will review it.", ticket)
    if status == "awaiting_approval":
        return (f"Ticket {ticket['id']} is waiting for a support specialist to approve a refund. "
                "No refund has been approved in this chat."), ticket
    if status == "recovering":
        return f"Ticket {ticket['id']} is being recovered. I will only report the outcome after verification.", ticket
    return (f"Ticket {ticket['id']} is open and Relay is investigating the order. "
            "I will report progress after each verified step."), ticket


def handle_message(app: AppContext, customer: Customer, conversation_id: str, content: str,
                   correlation_id: str) -> dict | None:
    conversation = conversation_for_customer(app, customer, conversation_id)
    if conversation is None:
        return None
    text = (sanitize_text(content, app.settings.support_max_message_chars) or "").strip()
    if not text:
        return {"conversation": conversation_json(app, conversation), "message": None, "error": "empty message"}
    state = _state(conversation)
    stage = state.get("stage", "idle")
    injection = detect_injection(text)
    intent = "message"
    assistant = ""
    citations: list[dict] = []
    meta: dict = {"intent": intent, "injection_flagged": bool(injection)}
    ticket = None

    order_match = ORDER_RE.search(text)
    order_id = order_match.group(0).upper() if order_match else None
    if order_id is None and stage == "awaiting_order_choice":
        option = _match_option(text, state.get("order_options") or [])
        if option is not None:
            order_id = option["value"]

    small_talk = _small_talk(text, customer.name)
    demo_ans = lookup_democart_answer(text) if stage not in ("awaiting_confirmation", "awaiting_close_confirmation") else None

    if stage == "closed":
        if _is_no_or_end(text):
            assistant = "The chat has already been closed. Say 'Hi' if you would like to start a new conversation."
            meta.update(intent="chat_closed")
        else:
            state.update(stage="idle", ticket_id=None, draft=None, order_id=None, category=None, menu=None, category_hint=None)
            stage = "idle"

    if assistant:
        pass
    elif stage == "awaiting_close_confirmation":
        if _is_close_affirmative(text):
            assistant = "The chat has been closed. Have a great day! Feel free to reach out anytime if you need help with an order."
            state.update(stage="closed")
            meta.update(intent="chat_closed")
        else:
            assistant = "Okay, I'll keep the chat open. How can I help you?"
            state.update(stage="idle")
            meta.update(intent="chat_kept_open")
    elif stage == "awaiting_confirmation" and state.get("draft") and (_affirmative(text) or _negative(text)):
        draft = state["draft"]
        if _negative(text):
            state.update(stage="idle", draft=None, order_id=None, category=None, menu="main", category_hint=None)
            assistant = "No ticket was created. You can ask a policy question or start again with an order number."
        else:
            order = _order_for_customer(app, customer, draft["order_id"])
            if order is None:
                assistant = "I can no longer verify that order on your account, so I did not create a ticket."
                state.update(stage="idle", draft=None, order_id=None, category=None, menu="main",
                             category_hint=None)
            else:
                fresh = _draft(app, customer, order, draft.get("summary", ""), correlation_id)
                if not fresh:
                    assistant = "I could not re-read the order, so I did not create a ticket. Please try again shortly."
                elif fresh["evidence"] != draft.get("evidence") or fresh["category"] != draft.get("category"):
                    state.update(stage="awaiting_confirmation", draft=fresh, order_id=order.id,
                                 category=fresh["category"])
                    meta["options"] = [
                        {"label": "Confirm (C)", "value": "confirm"},
                        {"label": "Cancel (N)", "value": "cancel"},
                    ]
                    assistant = ("The order state changed while we were talking, so I refreshed the summary.\n\n"
                                 + _draft_intro(fresh))
                    citations = fresh["citations"]
                else:
                    ticket, created = create_customer_ticket(
                        app, customer=customer, order=order, conversation_id=conversation.id,
                        category=draft["category"], confirmed_summary=draft["summary"],
                        citations=draft["citations"], service_evidence=draft["evidence"],
                        dispatch=bool(draft["dispatch"] and not draft["requires_human"]))
                    state.update(stage="ticket_created", ticket_id=ticket.id, draft=None,
                                 order_id=order.id, category=draft["category"], menu=None,
                                 category_hint=None)
                    citations = draft["citations"]
                    if ticket.status == "escalated":
                        assistant = (f"Ticket {ticket.id} " + ("was created and sent" if created else "is") +
                                     " with a human specialist. You can follow its status below.")
                    else:
                        assistant = (f"Ticket {ticket.id} " + ("was created. " if created else "already exists. ") +
                                     "Relay is checking the live service records now; verified progress will "
                                     "appear below. A refund, if needed, still requires a human operator.")
                    meta["ticket_id"] = ticket.id
                    ticket = ticket_customer_json(app, ticket.id)
        meta.update(intent="confirmation", category=state.get("category"))
    elif _greeting(text):
        assistant = "Hi, how can I help you?"
        state.update(
            stage="idle",
            menu="main",
            order_id=None,
            draft=None,
            ticket_id=None,
            category=None,
            category_hint=None,
            order_options=None,
        )
        meta.update(intent="greeting", small_talk="greeting", options=[dict(opt) for opt in MAIN_OPTIONS])
    elif _is_agent_request(text):
        assistant = (
            "Currently, we don't have a policy to connect directly with a live agent via chat. "
            "However, we can raise a support ticket for our operations team, and a representative will "
            "review your issue and reach out or call you directly.\n\n"
            "Would you like me to raise a support ticket for you?"
        )
        state.update(stage="idle", menu="main", order_id=None, draft=None, category=None, category_hint="raise_ticket")
        meta.update(intent="agent_request", options=[
            {"label": "Raise a support ticket", "value": "raise_ticket"},
            {"label": "Track an order", "value": "track_order"},
            {"label": "Something else", "value": "other"},
        ])
    elif _is_ticket_update_query(text):
        with app.session() as s:
            ticket_id = state.get("ticket_id")
            ticket_row = None
            if ticket_id:
                ticket_row = s.get(SupportTicket, ticket_id)
            if not ticket_row:
                ticket_row = s.scalars(
                    select(SupportTicket)
                    .where(SupportTicket.customer_id == customer.id)
                    .order_by(SupportTicket.created_at.desc())
                ).first()

        if ticket_row:
            sync_ticket(app, ticket_row.id)
            t_data = ticket_customer_json(app, ticket_row.id)
            st = t_data["status"]
            resolution = t_data.get("resolution_summary") or t_data.get("resolution") or t_data.get("summary") or ""
            order_str = f" for order {t_data.get('order_id')}" if t_data.get("order_id") else ""
            assistant = (
                f"This is the current status of your ticket. This is the last ticket we have created for this issue "
                f"({t_data['id']}{order_str}):\n\n"
                f"• Current status: {st.upper()}\n"
                f"• Details: {resolution}"
            ).strip()
            ticket = t_data
            state.update(ticket_id=ticket_row.id)
            meta.update(intent="ticket_status", ticket_id=t_data["id"])
        else:
            assistant = "You do not have any open or previous tickets on your account."
    elif _is_general_query_prompt(text):
        assistant = "Related to which?"
        state.update(stage="awaiting_query_topic")
        meta.update(intent="query_prompt")
    elif (stage == "awaiting_query_topic" and _is_payment_issue_topic(text)) or "related to order payment" in text.lower():
        orders = _recent_orders(app, customer)
        order = orders[0] if orders else None
        if order:
            citations = [{
                "chunk_id": "payment-confirmation::01::awaiting-capture",
                "document_id": "payment-confirmation",
                "title": "Payment Confirmation Policy",
                "version": "1.2",
                "section": "When payment is captured but the order is pending",
                "excerpt": "If the payment is captured and the order is still pending, Relay verifies the payment before retrying or fulfilling.",
                "score": 0.95,
                "retrieval": "lexical_fallback",
            }]
            t_obj, created = create_customer_ticket(
                app,
                customer=customer,
                order=order,
                conversation_id=conversation.id,
                category="PAYMENT_PENDING",
                confirmed_summary=f"Order payment issue: payment captured but order {order.id} pending.",
                citations=citations,
                service_evidence={"topic": "order_payment", "order_id": order.id, "customer": customer.name},
                dispatch=True,
            )
            state.update(stage="ticket_created", ticket_id=t_obj.id, order_id=order.id)
            meta.update(intent="ticket_created", ticket_id=t_obj.id)
            ticket = ticket_customer_json(app, t_obj.id)
            assistant = (
                f"These are the policies we have for this one:\n\n"
                f"• Payment Confirmation Policy: If your payment was deducted or captured but the order is still pending, "
                f"Relay automatically verifies the capture against provider records and advances the order to paid without charging you again.\n"
                f"• We never charge twice or make repeated deduction attempts.\n\n"
                f"I have raised support ticket {t_obj.id} for order {order.id}. "
                f"It has been routed to our operations team for recovery. You can check back anytime by asking \"What's the update on my ticket?\"."
            )
        else:
            assistant = (
                "These are the policies we have for this one:\n\n"
                "• Payment Confirmation Policy: If your payment was deducted or captured but the order is still pending, "
                "Relay automatically verifies the capture against provider records and advances the order to paid without charging you again.\n\n"
                "Could you provide your order number (for example ORD-ABC123) so I can raise a ticket for our operations team?"
            )
            state.update(stage="awaiting_order", category_hint="payment_pending")
    elif stage == "awaiting_query_topic" and any(k in text.lower() for k in ("inventory", "stock")):
        orders = _recent_orders(app, customer)
        order = orders[0] if orders else None
        if order:
            t_obj, created = create_customer_ticket(
                app, customer=customer, order=order, conversation_id=conversation.id,
                category="INVENTORY_ISSUE",
                confirmed_summary=f"Inventory issue on order {order.id}.",
                citations=[], service_evidence={"topic": "inventory", "order_id": order.id},
                dispatch=True,
            )
            state.update(stage="ticket_created", ticket_id=t_obj.id, order_id=order.id)
            ticket = ticket_customer_json(app, t_obj.id)
            assistant = (
                f"These are the policies we have for this one:\n\n"
                f"• Inventory Reservation Policy: If stock was reserved but confirmation was delayed, Relay checks provider "
                f"records to reuse the existing reservation rather than reserving multiple times.\n\n"
                f"I have raised support ticket {t_obj.id} for order {order.id} with our operations team."
            )
        else:
            assistant = "These are the policies we have for inventory reservations. Please provide an order number to raise a ticket."
    elif stage == "awaiting_query_topic" and any(k in text.lower() for k in ("shipping", "delivery")):
        orders = _recent_orders(app, customer)
        order = orders[0] if orders else None
        if order:
            t_obj, created = create_customer_ticket(
                app, customer=customer, order=order, conversation_id=conversation.id,
                category="SHIPPING_DELAY",
                confirmed_summary=f"Shipping delay reported for order {order.id}.",
                citations=[], service_evidence={"topic": "shipping", "order_id": order.id},
                dispatch=True,
            )
            state.update(stage="ticket_created", ticket_id=t_obj.id, order_id=order.id)
            ticket = ticket_customer_json(app, t_obj.id)
            assistant = (
                f"These are the policies we have for this one:\n\n"
                f"• Shipping Policy: When carrier creation is delayed, Relay retries with exponential backoff before escalating to a human specialist.\n\n"
                f"I have raised support ticket {t_obj.id} for order {order.id} with our operations team."
            )
        else:
            assistant = "These are the policies we have for shipping delays. Please provide an order number to raise a ticket."
    elif _is_no_or_end(text):
        assistant = "Shall I close the chat?"
        state.update(stage="awaiting_close_confirmation")
        meta.update(intent="confirm_close", options=[
            {"label": "Yes, close chat", "value": "yes"},
            {"label": "No, keep open", "value": "no"},
        ])
    elif small_talk is not None:
        kind, reply = small_talk
        assistant = reply
        meta.update(intent="greeting" if kind == "greeting" else "small_talk", small_talk=kind)
        if kind in ("greeting", "how_are_you", "wellbeing"):
            followup, options = _small_talk_followup(stage, state)
            if options:
                meta["options"] = options
        if kind == "greeting":
            state.update(stage="idle")
    elif demo_ans is not None:
        assistant = demo_ans["answer"]
        meta.update(intent=demo_ans.get("intent", "democart_interaction"))
        if demo_ans.get("interaction_id"):
            meta["interaction_id"] = demo_ans["interaction_id"]
        if demo_ans.get("order_id"):
            meta["order_id"] = demo_ans["order_id"]
            state.update(order_id=demo_ans["order_id"])
        citations = [{
            "chunk_id": f"democart::{demo_ans.get('intent', 'policy')}::01",
            "document_id": "democart-store-policy",
            "title": "DemoCart Store Policy",
            "version": "2026.1",
            "section": demo_ans.get("intent", "Policy").replace("_", " ").title(),
            "excerpt": demo_ans["answer"][:300],
            "score": 0.98,
            "retrieval": "democart_rag",
        }]
    elif _track_request(text):
        assistant = _show_order_choices(app, customer, state, meta, "track_order")
    elif order_id:
        order = _order_for_customer(app, customer, order_id)
        if order is None:
            # Same wording for unknown and someone else's order: no enumeration.
            assistant = ("I can't find that order on your account. Check the order number, or tell me what "
                         "policy question you have.")
            if stage in ("awaiting_order_choice", "awaiting_issue"):
                choices_text, choices = _order_choices(app, customer)
                state.update(stage="awaiting_order_choice", order_id=None, draft=None, menu=None,
                             order_options=choices or None)
                meta["options"] = choices
                if choices:
                    assistant += "\n\n" + choices_text
            else:
                state.update(stage="awaiting_order", order_id=None, category=None, draft=None,
                             menu=None, category_hint=state.get("category_hint"), order_options=None)
            meta["intent"] = "invalid_or_unauthorized_order"
        elif stage in ("awaiting_order_choice", "awaiting_order"):
            if state.get("category_hint") == "track_order":
                assistant, citations = _process_order(app, customer, order, text, correlation_id,
                                                      state, meta, injection)
            else:
                assistant, options = _issue_choices(order, state, meta)
        elif stage == "awaiting_issue":
            assistant, options = _issue_choices(order, state, meta)
        else:
            assistant, citations = _process_order(app, customer, order, text, correlation_id,
                                                  state, meta, injection)
    # A ticket exists: this conversation is bound to its verified progress.
    elif state.get("ticket_id"):
        ticket_id = state["ticket_id"]
        lower = text.lower()
        if any(w in lower for w in STATUS_WORDS):
            assistant, ticket = _ticket_status_text(app, ticket_id)
        elif any(w in lower for w in POLICY_WORDS) and "order" not in lower:
            policy = _grounded(app, text)
            citations = policy["citations"]
            assistant = policy["answer"]
            meta.update(retrieval_mode=policy["retrieval_mode"], retrieval_label=policy["retrieval_label"],
                        engine=policy["engine"], intent="policy")
        else:
            assistant, ticket = _ticket_status_text(app, ticket_id)
        if injection:
            assistant = ("I can't follow instructions that try to change Relay's rules or approve payments/refunds. "
                         + assistant)
    elif stage == "awaiting_order_choice":
        lower = text.lower()
        greeting = _greeting(text)
        help_request = lower.strip(" !.?") in HELP_REQUESTS
        is_policy = any(w in lower for w in POLICY_WORDS)
        is_meta_policy = any(w in lower for w in META_POLICY_WORDS)
        if greeting or help_request or _negative(text):
            assistant, options = _main_menu(customer)
            state.update(stage="idle", menu="main", category_hint=None, draft=None,
                         order_id=None, order_options=None)
            meta.update(intent="greeting", options=options)
        elif _order_list_request(text):
            assistant = _show_order_choices(app, customer, state, meta, state.get("category_hint"))
        elif _ticket_request(text):
            options = [dict(option) for option in (state.get("order_options") or [])]
            assistant = (
                "To proceed with raising your ticket, please choose one of the orders listed above "
                "(by clicking an option or replying with 1, 2, or your order number like ORD-...), "
                "or let me know if it is for a different order."
            )
            meta.update(intent="order_choice", options=options, category_hint=state.get("category_hint"))
        elif is_policy or is_meta_policy or "?" in text:
            policy = _grounded(app, text)
            citations = policy["citations"]
            assistant = policy["answer"] + "\n\nWhich of the orders above is this about?"
            options = [dict(option) for option in (state.get("order_options") or [])]
            meta.update(intent="policy", retrieval_mode=policy["retrieval_mode"],
                        retrieval_label=policy["retrieval_label"], engine=policy["engine"],
                        insufficient=policy["insufficient"], conflicting=policy["conflicting"],
                        options=options)
        else:
            option = _match_option(text, ISSUE_OPTIONS, fuzzy=True)
            hint = option["value"] if option else state.get("category_hint")
            if hint:
                state.update(category_hint=hint)
            options = [dict(option) for option in (state.get("order_options") or [])]
            assistant = (
                "Please choose which order from the list above this issue applies to "
                "(reply with 1, 2, or your order number like ORD-...)."
            )
            meta.update(intent="order_choice", options=options, category_hint=hint)
    elif stage == "awaiting_issue":
        lower = text.lower()
        greeting = _greeting(text)
        help_request = lower.strip(" !.?") in HELP_REQUESTS
        is_policy = any(w in lower for w in POLICY_WORDS)
        is_meta_policy = any(w in lower for w in META_POLICY_WORDS)
        current_order_id = state.get("order_id")
        order = _order_for_customer(app, customer, current_order_id) if current_order_id else None
        option = _match_option(text, ISSUE_OPTIONS) or _match_option(text, ISSUE_OPTIONS, fuzzy=True)
        if greeting or help_request or _negative(text):
            assistant, options = _main_menu(customer)
            state.update(stage="idle", menu="main", category_hint=None, draft=None,
                         order_id=None, order_options=None)
            meta.update(intent="greeting", options=options)
        elif _order_list_request(text):
            assistant = _show_order_choices(app, customer, state, meta, None)
        elif _ticket_request(text):
            if order is None:
                assistant = _show_order_choices(app, customer, state, meta, None)
            else:
                assistant, options = _issue_choices(order, state, meta)
        elif is_policy or is_meta_policy or ("?" in text and option is None):
            policy = _grounded(app, text)
            citations = policy["citations"]
            assistant = policy["answer"] + "\n\nWhich issue below fits best?"
            options = [dict(issue_option) for issue_option in ISSUE_OPTIONS]
            meta.update(intent="policy", retrieval_mode=policy["retrieval_mode"],
                        retrieval_label=policy["retrieval_label"], engine=policy["engine"],
                        insufficient=policy["insufficient"], conflicting=policy["conflicting"],
                        options=options)
        elif option is not None:
            if order is None:
                assistant = "I can no longer verify that order on your account. Please choose another order."
                state.update(stage="idle", menu="main", order_id=None, category_hint=None)
                meta["intent"] = "invalid_or_unauthorized_order"
            else:
                assistant, options = _details_prompt(order, option, state, meta)
        elif order is None:
            assistant = "I can no longer verify that order on your account. Please choose another order."
            state.update(stage="idle", menu="main", order_id=None, category_hint=None)
            meta["intent"] = "invalid_or_unauthorized_order"
        else:
            options = [dict(issue_option) for issue_option in ISSUE_OPTIONS]
            assistant = (f"Please choose what the issue is about for order {order.id} — "
                         "payment, refund, inventory, shipping, or conflicting payment records.")
            meta.update(intent="issue_choice_required", order_id=order.id, options=options)
    elif stage == "awaiting_details":
        lower = text.lower()
        greeting = _greeting(text)
        help_request = lower.strip(" !.?") in HELP_REQUESTS
        is_policy = any(w in lower for w in POLICY_WORDS)
        is_meta_policy = any(w in lower for w in META_POLICY_WORDS)
        current_order_id = state.get("order_id")
        order = _order_for_customer(app, customer, current_order_id) if current_order_id else None
        if greeting or help_request or _negative(text):
            assistant, options = _main_menu(customer)
            state.update(stage="idle", menu="main", category_hint=None, draft=None,
                         order_id=None, order_options=None)
            meta.update(intent="greeting", options=options)
        elif _order_list_request(text):
            assistant = _show_order_choices(app, customer, state, meta, None)
        elif _ticket_request(text):
            if order is None:
                assistant = _show_order_choices(app, customer, state, meta, None)
            else:
                assistant, options = _issue_choices(order, state, meta)
        elif is_policy or is_meta_policy or "?" in text:
            policy = _grounded(app, text)
            citations = policy["citations"]
            if order is None:
                assistant = policy["answer"]
                state.update(stage="idle", menu="main", order_id=None, category_hint=None)
            else:
                assistant = policy["answer"] + f"\n\nWhat is the issue with order {order.id}?"
            meta.update(intent="policy", retrieval_mode=policy["retrieval_mode"],
                        retrieval_label=policy["retrieval_label"], engine=policy["engine"],
                        insufficient=policy["insufficient"], conflicting=policy["conflicting"])
        elif order is None:
            assistant = "I can no longer verify that order on your account. Please choose another order."
            state.update(stage="idle", menu="main", order_id=None, category_hint=None)
            meta["intent"] = "invalid_or_unauthorized_order"
        else:
            assistant, citations = _process_order(app, customer, order, text, correlation_id,
                                                  state, meta, injection)
    else:
        lower = text.lower()
        is_orderish = any(w in lower for w in ORDER_WORDS)
        is_policy = any(w in lower for w in POLICY_WORDS)
        is_meta_policy = any(w in lower for w in META_POLICY_WORDS)
        greeting = _greeting(text)
        help_request = lower.strip(" !.?") in HELP_REQUESTS
        ticket_request = _ticket_request(text)
        menu = state.get("menu")

        menu_option = None
        if menu == "main":
            menu_option = _match_option(text, MAIN_OPTIONS) or _match_option(text, ISSUE_OPTIONS)
        elif menu == "issues":
            menu_option = _match_option(text, ISSUE_OPTIONS)

        if stage == "awaiting_order":
            if greeting or help_request or _negative(text):
                assistant, options = _main_menu(customer)
                state.update(stage="idle", menu="main", category_hint=None, draft=None,
                             order_id=None, order_options=None)
                meta.update(intent="greeting", options=options)
            elif ticket_request or menu_option is not None:
                hint = menu_option["value"] if menu_option is not None else state.get("category_hint")
                assistant = _show_order_choices(app, customer, state, meta, hint)
            elif is_policy or is_meta_policy or "?" in text:
                policy = _grounded(app, text)
                citations = policy["citations"]
                assistant = policy["answer"] + "\n\n" + _ask_order(state.get("category_hint"))
                meta.update(intent="policy", retrieval_mode=policy["retrieval_mode"],
                            retrieval_label=policy["retrieval_label"], engine=policy["engine"],
                            insufficient=policy["insufficient"], conflicting=policy["conflicting"])
            else:
                option = _match_option(text, ISSUE_OPTIONS, fuzzy=True)
                hint = option["value"] if option else state.get("category_hint")
                assistant = _show_order_choices(app, customer, state, meta, hint)
        elif menu_option is not None:
            value = menu_option["value"]
            if value == "policy":
                state.update(stage="idle", menu=None, category_hint=None, draft=None,
                             order_id=None, order_options=None)
                assistant = ("Sure. What policy question do you have? I will answer from the policy documents "
                             "and include citations.")
                meta.update(intent="policy_prompt")
            else:
                hint = None if value in ("raise_ticket", "other") else value
                assistant = _show_order_choices(app, customer, state, meta, hint)
        elif ticket_request:
            assistant = _show_order_choices(app, customer, state, meta, state.get("category_hint"))
        elif greeting or help_request:
            assistant, options = _main_menu(customer)
            state.update(stage="idle", menu="main", category_hint=None, draft=None,
                         order_id=None, order_options=None)
            meta.update(intent="greeting", options=options)
        elif _order_list_request(text):
            assistant = _show_order_choices(app, customer, state, meta, None)
        elif is_orderish and not is_meta_policy:
            option = _match_option(text, ISSUE_OPTIONS, fuzzy=True)
            hint = option["value"] if option else None
            assistant = _show_order_choices(app, customer, state, meta, hint)
        elif is_policy or is_meta_policy or "?" in text:
            policy = _grounded(app, text)
            citations = policy["citations"]
            if policy["insufficient"]:
                assistant = (
                    "I could not find specific details for that in our store policies.\n\n"
                    "Here is what I can help you with:\n"
                    "• Tracking an order and delivery updates\n"
                    "• Returns, refunds, and cancellations\n"
                    "• Order payment confirmation & pending orders\n"
                    "• Raising a support ticket for our operations team\n\n"
                    "Would you like me to raise a ticket for our team, or help with one of the options below?"
                )
                state.update(stage="idle", menu="main")
                meta.update(
                    intent="policy",
                    retrieval_mode=policy["retrieval_mode"],
                    retrieval_label=policy["retrieval_label"],
                    engine=policy["engine"],
                    insufficient=True,
                    conflicting=False,
                    options=[
                        {"label": "Raise a support ticket", "value": "raise_ticket"},
                        {"label": "Track an order", "value": "track_order"},
                        {"label": "Refund or cancellation", "value": "refund"},
                        {"label": "Something else", "value": "other"},
                    ],
                )
            else:
                assistant = policy["answer"]
                if policy["escalation_advised"]:
                    assistant += (" If this is about a specific order, tell me the issue and I can show your "
                                  "recent orders or create a ticket for human review.")
                state.update(stage="idle", menu=None)
                meta.update(intent="policy", retrieval_mode=policy["retrieval_mode"],
                            retrieval_label=policy["retrieval_label"], engine=policy["engine"],
                            insufficient=policy["insufficient"], conflicting=policy["conflicting"])
        else:
            lower = text.lower()
            is_actual_policy_query = any(w in lower for w in POLICY_WORDS) or ("?" in text and len(text.split()) > 2)
            if is_actual_policy_query:
                policy = _grounded(app, text)
                citations = policy["citations"]
                if policy["insufficient"]:
                    assistant = (
                        "I could not find specific details for that in our store policies.\n\n"
                        "Here is what I can help you with:\n"
                        "• Tracking an order and delivery updates\n"
                        "• Returns, refunds, and cancellations\n"
                        "• Order payment confirmation & pending orders\n"
                        "• Raising a support ticket for our operations team\n\n"
                        "Would you like me to raise a ticket for our team, or help with one of the options below?"
                    )
                    state.update(stage="idle", menu="main")
                    meta.update(
                        intent="policy",
                        retrieval_mode=policy["retrieval_mode"],
                        retrieval_label=policy["retrieval_label"],
                        engine=policy["engine"],
                        insufficient=True,
                        conflicting=False,
                        options=[
                            {"label": "Raise a support ticket", "value": "raise_ticket"},
                            {"label": "Track an order", "value": "track_order"},
                            {"label": "Refund or cancellation", "value": "refund"},
                            {"label": "Something else", "value": "other"},
                        ],
                    )
                else:
                    assistant = policy["answer"]
                    if not policy["insufficient"] and not policy["conflicting"]:
                        assistant += " If you meant a specific order, tell me what went wrong rather than guess."
                    state.update(stage="idle", menu=None)
                    meta.update(intent="policy", retrieval_mode=policy["retrieval_mode"],
                                retrieval_label=policy["retrieval_label"], engine=policy["engine"])
            else:
                assistant = ("I'm here to help with your orders, payments, deliveries, or tickets. "
                             "You can describe your query, say 'I have a query', or ask \"What's the update on my ticket?\".")
                state.update(stage="idle", menu=None)
                meta.update(intent="clarification", options=[dict(opt) for opt in MAIN_OPTIONS])

    assistant = _strip_escalation_procedures(assistant)
    citations = [
        c for c in (citations or [])
        if c.get("document_id") != "escalation" and "escalation" not in c.get("title", "").lower()
    ]

    with app.session() as s:
        conversation = s.get(ChatConversation, conversation.id)
        conversation.state = state
        _append(s, conversation, "customer", text,
                meta={"injection_flagged": bool(injection), "patterns": injection[:3] if injection else []})
        assistant_message = _append(s, conversation, "assistant", assistant, citations=citations, meta=meta)
        s.commit()
        s.refresh(assistant_message)
        conversation = s.get(ChatConversation, conversation.id)
    return {"conversation": conversation_json(app, conversation), "message": _msg_json(assistant_message),
            "ticket": ticket}
