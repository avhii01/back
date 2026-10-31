"""Customer-facing API. Separate from the operator API and never returns internal traces."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import select

from relay.context import AppContext
from relay.models import ChatConversation, ChatMessage, Customer, Order, SupportTicket
from relay.support import auth
from relay.support.chat import (
    conversation_for_customer,
    conversation_json,
    create_conversation,
    handle_message,
)
from relay.support.tickets import sync_ticket, ticket_customer_json


class RegisterBody(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    email: str = Field(min_length=3, max_length=200)
    user_id: str | None = Field(default=None, max_length=40)
    password: str = Field(min_length=1, max_length=200)


class SignInBody(BaseModel):
    email: str = Field(min_length=3, max_length=200)
    password: str = Field(min_length=1, max_length=200)


class MessageBody(BaseModel):
    content: str = Field(min_length=1, max_length=4000)


class NewConversationBody(BaseModel):
    pass


def _bearer(request: Request) -> str | None:
    header = request.headers.get("Authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return request.cookies.get("relay_customer_session")


def _customer_json(customer: Customer) -> dict:
    return {"id": customer.id, "name": customer.name, "email": customer.email}


def _order_summary(order: Order) -> dict:
    return {"id": order.id, "status": order.status, "sku": order.sku, "quantity": order.quantity,
            "total_cents": order.total_cents, "currency": order.currency,
            "created_at": order.created_at.isoformat()}


def create_customer_router(ctx: AppContext) -> APIRouter:
    router = APIRouter(prefix="/api/customer", tags=["customer"])

    def current(request: Request) -> Customer:
        customer = auth.customer_from_token(ctx, _bearer(request))
        if customer is None:
            raise HTTPException(401, "customer sign-in required")
        return customer

    @router.post("/register")
    def register(body: RegisterBody):
        token, session, customer = auth.register_customer(
            ctx,
            name=body.name,
            email=body.email,
            user_id=body.user_id,
            password=body.password,
        )
        return {
            "token": token,
            "expires_at": session.expires_at.isoformat(),
            "simulated": False,
            "customer": _customer_json(customer),
            "orders": [],
        }

    @router.post("/session")
    def sign_in(body: SignInBody):
        customer = auth.authenticate(ctx, body.email, body.password)
        if customer is None:
            raise HTTPException(401, "invalid demo credentials")
        token, session = auth.create_session(ctx, customer.id)
        with ctx.session() as s:
            orders = list(s.scalars(select(Order).where(Order.customer_id == customer.id)
                                    .order_by(Order.created_at.desc()).limit(50)))
        return {"token": token, "expires_at": session.expires_at.isoformat(),
                "simulated": True, "customer": _customer_json(customer),
                "orders": [_order_summary(o) for o in orders]}

    @router.delete("/session")
    def sign_out(request: Request):
        auth.revoke_session(ctx, _bearer(request))
        return {"signed_out": True}

    @router.get("/me")
    def me(request: Request):
        customer = current(request)
        with ctx.session() as s:
            orders = list(s.scalars(select(Order).where(Order.customer_id == customer.id)
                                    .order_by(Order.created_at.desc()).limit(50)))
        return {"customer": _customer_json(customer), "orders": [_order_summary(o) for o in orders],
                "simulated_sign_in": True}

    @router.get("/orders")
    def my_orders(request: Request):
        customer = current(request)
        with ctx.session() as s:
            orders = list(s.scalars(select(Order).where(Order.customer_id == customer.id)
                                    .order_by(Order.created_at.desc()).limit(50)))
        return [_order_summary(o) for o in orders]

    @router.post("/conversations", status_code=201)
    def new_conversation(request: Request):
        customer = current(request)
        conversation = create_conversation(ctx, customer)
        return conversation_json(ctx, conversation)

    @router.get("/conversations")
    def list_conversations(request: Request):
        customer = current(request)
        with ctx.session() as s:
            rows = list(s.scalars(select(ChatConversation).where(ChatConversation.customer_id == customer.id)
                                  .order_by(ChatConversation.updated_at.desc()).limit(20)))
            out = []
            for row in rows:
                last = s.scalars(select(ChatMessage).where(ChatMessage.conversation_id == row.id)
                                 .order_by(ChatMessage.id.desc())).first()
                out.append({"id": row.id, "updated_at": row.updated_at.isoformat(),
                            "stage": (row.state or {}).get("stage", "idle"),
                            "last_message": last.content[:140] if last else ""})
        return out

    @router.get("/conversations/{conversation_id}")
    def get_conversation(conversation_id: str, request: Request):
        customer = current(request)
        conversation = conversation_for_customer(ctx, customer, conversation_id)
        if conversation is None:
            raise HTTPException(404, "conversation not found")
        return conversation_json(ctx, conversation)

    @router.post("/conversations/{conversation_id}/messages")
    def send_message(conversation_id: str, body: MessageBody, request: Request):
        customer = current(request)
        result = handle_message(ctx, customer, conversation_id, body.content, request.state.correlation_id)
        if result is None:
            raise HTTPException(404, "conversation not found")
        return result

    @router.get("/tickets")
    def my_tickets(request: Request):
        customer = current(request)
        with ctx.session() as s:
            ids = list(s.scalars(select(SupportTicket.id).where(SupportTicket.customer_id == customer.id)
                                 .order_by(SupportTicket.created_at.desc()).limit(50)))
        return [t for tid in ids if (t := ticket_customer_json(ctx, tid)) is not None]

    @router.get("/tickets/{ticket_id}")
    def my_ticket(ticket_id: str, request: Request):
        customer = current(request)
        sync_ticket(ctx, ticket_id)
        with ctx.session() as s:
            ticket = s.get(SupportTicket, ticket_id)
            if ticket is None or ticket.customer_id != customer.id:
                raise HTTPException(404, "ticket not found")
        return ticket_customer_json(ctx, ticket_id)

    return router
