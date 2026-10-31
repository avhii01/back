"""FastAPI application: operator dashboard API."""
from __future__ import annotations

import json
import uuid
import secrets
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy import select

from relay import scenarios
from relay.agent import runner
from relay.api import serializers as ser
from relay.api.customer import create_customer_router
from relay.api.support import create_operator_support_router
from relay.approvals import ApprovalError, decide_approval
from relay.config import Settings
from relay.context import AppContext, build_context
from relay.errors import ToolError
from relay.logging_utils import configure_logging
from relay.models import Approval, Customer, Order, Run, SupportTicket, TeamMember
from relay.support import jobs
from relay.support.demo import SCENARIO_OWNER


class AdminSignInBody(BaseModel):
    email: str = Field(min_length=1, max_length=200)
    password: str = Field(min_length=1, max_length=200)


class TeamMemberCreate(BaseModel):
    id: str | None = None
    name: str = Field(min_length=1, max_length=120)
    email: str = Field(min_length=1, max_length=200)
    role: str = Field(min_length=1, max_length=80)
    department: str = Field(min_length=1, max_length=120)
    status: str = Field(default="active", max_length=20)
    active_tickets_count: int = Field(default=0, ge=0)


class TeamMemberUpdate(BaseModel):
    name: str | None = None
    role: str | None = None
    department: str | None = None
    status: str | None = None
    active_tickets_count: int | None = None


class InjectBody(BaseModel):
    scenario: str
    variant: str | None = None


class StartRunBody(BaseModel):
    step_mode: bool = False


class DecisionBody(BaseModel):
    decision: str = Field(pattern="^(approve|reject)$")
    decided_by: str = Field(min_length=1, max_length=80)
    note: str | None = Field(default=None, max_length=300)


def create_app(ctx: AppContext | None = None) -> FastAPI:
    configure_logging()
    app_ctx = ctx or build_context(Settings.from_env())

    worker: list = [None, None]

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if app_ctx.settings.auto_resume_on_startup:
            runner.resume_interrupted(app_ctx)
        if app_ctx.settings.job_worker_enabled:
            jobs.recover_interrupted(app_ctx)
            worker[0], worker[1] = jobs.start_worker(app_ctx)
        yield
        if worker[0] is not None:
            jobs.stop_worker(worker[0], worker[1])

    app = FastAPI(title="Relay - AI Order Recovery Agent", version="0.1.0", lifespan=lifespan)
    app.state.ctx = app_ctx
    app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in app_ctx.settings.cors_origins.split(",") if o.strip()],
                       allow_methods=["*"], allow_headers=["*"], allow_credentials=True,
                       expose_headers=["X-Correlation-ID"])

    @app.middleware("http")
    async def correlation(request: Request, call_next):
        supplied = request.headers.get("X-Correlation-ID") or ""
        cid = supplied if 0 < len(supplied) <= 64 else uuid.uuid4().hex
        request.state.correlation_id = cid
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = cid
        return response

    @app.exception_handler(ApprovalError)
    async def approval_error(_: Request, exc: ApprovalError):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})

        # ------------------------------------------------------------------ admin session
    @app.post("/api/admin/session")
    def admin_session(body: AdminSignInBody):
        email_clean = body.email.strip().lower()
        if email_clean in ("admin@relay.test", "admin") and body.password.strip() == "admin-relay":
            token = "adm_" + secrets.token_urlsafe(24)
            return {
                "token": token,
                "admin": {
                    "id": "ADM-9041",
                    "email": "admin@relay.test",
                    "name": "Admin Operator",
                    "role": "Lead Recovery Specialist & Operations Admin",
                },
            }
        elif len(email_clean) > 0 and body.password.strip() in ("admin-relay", "password", "relay123", "admin"):
            token = "adm_" + secrets.token_urlsafe(24)
            user_part = email_clean.split("@")[0]
            admin_id = "ADM-" + user_part.upper()[:10]
            admin_name = user_part.replace(".", " ").replace("-", " ").title()
            return {
                "token": token,
                "admin": {
                    "id": admin_id,
                    "email": email_clean if "@" in email_clean else f"{email_clean}@relay.test",
                    "name": admin_name or "Support Operator",
                    "role": "Support Specialist & Operator",
                },
            }
        raise HTTPException(401, "Invalid admin credentials. Use admin@relay.test / admin-relay")

    # ------------------------------------------------------------------ meta
    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/config")
    def config():
        return app_ctx.settings.public()

    @app.get("/api/scenarios")
    def list_scenarios():
        return scenarios.list_scenarios()

    @app.post("/api/scenarios/inject", status_code=201)
    def inject(body: InjectBody):
        try:
            order = scenarios.inject(app_ctx, body.scenario, body.variant)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        except Exception as e:  # simulator unreachable etc.
            raise HTTPException(502, f"could not configure simulated services: {type(e).__name__}") from e
        if app_ctx.settings.demo_customer_auth:
            # Demo convenience: an operator-injected order is owned by a seeded customer so
            # it can be exercised through the customer support chat (server-side ownership).
            owner = SCENARIO_OWNER.get(body.scenario)
            if owner:
                with app_ctx.session() as s:
                    if s.get(Customer, owner) is not None:
                        row = s.get(Order, order.id)
                        if row is not None:
                            row.customer_id = owner
                            s.commit()
                            s.refresh(row)
                            order = row
        return ser.order_json(order)

    # ------------------------------------------------------------------ orders
    @app.get("/api/orders")
    def list_orders():
        with app_ctx.session() as s:
            orders = s.scalars(select(Order).order_by(Order.created_at.desc()).limit(200)).all()
            out = []
            for o in orders:
                run = s.scalars(select(Run).where(Run.order_id == o.id).order_by(Run.created_at.desc())).first()
                tickets = s.scalars(select(SupportTicket.reason_code).where(SupportTicket.order_id == o.id)).all()
                pending = s.scalars(select(Approval.id).where(Approval.order_id == o.id, Approval.status == "pending")).first()
                out.append({**ser.order_json(o), "latest_run": ser.run_brief(run) if run else None,
                            "ticket_reasons": list(tickets), "pending_approval_id": pending})
        return out

    @app.get("/api/orders/{order_id}")
    def get_order(order_id: str):
        with app_ctx.session() as s:
            o = s.get(Order, order_id)
            if o is None:
                raise HTTPException(404, "order not found")
            runs = s.scalars(select(Run).where(Run.order_id == o.id).order_by(Run.created_at.desc())).all()
            approvals = s.scalars(select(Approval).where(Approval.order_id == o.id).order_by(Approval.requested_at.desc())).all()
            tickets = s.scalars(select(SupportTicket).where(SupportTicket.order_id == o.id)).all()
            body = {"order": ser.order_json(o), "runs": [ser.run_brief(r) for r in runs],
                    "approvals": [ser.approval_json(a) for a in approvals], "tickets": [ser.ticket_json(t) for t in tickets]}
        body["services"] = _live_services(app_ctx, body["order"])
        return body

    @app.post("/api/orders/{order_id}/runs", status_code=202)
    def start_run(order_id: str, request: Request, body: StartRunBody | None = None):
        step_mode = bool(body and body.step_mode)
        try:
            run, created = runner.create_run(app_ctx, order_id, step_mode=step_mode,
                                             correlation_id=request.state.correlation_id)
        except runner.OrderNotFound:
            raise HTTPException(404, "order not found") from None
        if created and not step_mode:
            runner.submit(app_ctx, run.id)
        return {"run": ser.run_brief(run), "already_active": not created}

    # ------------------------------------------------------------------ runs
    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str):
        d = ser.run_detail(app_ctx, run_id)
        if d is None:
            raise HTTPException(404, "run not found")
        return d

    @app.post("/api/runs/{run_id}/step")
    def step_run(run_id: str):
        status = runner.execute_run(app_ctx, run_id, single_step=True)
        if status == "not_claimed":
            raise HTTPException(409, "run is busy, finished, or owned by another worker")
        return ser.run_detail(app_ctx, run_id)

    @app.post("/api/runs/{run_id}/resume", status_code=202)
    def resume_run(run_id: str):
        with app_ctx.session() as s:
            run = s.get(Run, run_id)
            if run is None:
                raise HTTPException(404, "run not found")
            if run.status not in runner.RESUMABLE:
                raise HTTPException(409, f"run is {run.status}")
        runner.submit(app_ctx, run_id)
        return {"accepted": True}

    @app.get("/api/runs/{run_id}/trace")
    def trace(run_id: str):
        d = ser.run_detail(app_ctx, run_id)
        if d is None:
            raise HTTPException(404, "run not found")
        doc = {"format": "relay.trace.v1", "generated_at": datetime.now(timezone.utc).isoformat(),
               "simulated_services": True, "config": app_ctx.settings.public(), "run": d}
        return Response(json.dumps(doc, indent=2, default=str), media_type="application/json",
                        headers={"Content-Disposition": f'attachment; filename="relay-trace-{run_id}.json"'})

    # ------------------------------------------------------------------ approvals
    @app.get("/api/approvals")
    def list_approvals(status: str | None = None):
        with app_ctx.session() as s:
            q = select(Approval).order_by(Approval.requested_at.desc()).limit(100)
            if status:
                q = q.where(Approval.status == status)
            return [ser.approval_json(a) for a in s.scalars(q)]

    @app.post("/api/approvals/{approval_id}/decision")
    def decide(approval_id: str, body: DecisionBody, request: Request):
        result = decide_approval(app_ctx, approval_id, body.decision, body.decided_by, body.note,
                                 request.state.correlation_id)
        resumed = False
        if result.applied and result.run_id:
            with app_ctx.session() as s:
                run = s.get(Run, result.run_id)
                resumed = bool(run and run.status in runner.RESUMABLE)
            if resumed:
                runner.submit(app_ctx, result.run_id)
            from relay.support.tickets import sync_tickets_for_order
            sync_tickets_for_order(app_ctx, result.approval.order_id)
        return {"approval": ser.approval_json(result.approval), "applied": result.applied, "run_resumed": resumed}

    # ------------------------------------------------------------------ team members
    @app.get("/api/team")
    def list_team_members():
        with app_ctx.session() as s:
            members = list(s.scalars(select(TeamMember).order_by(TeamMember.created_at.asc())))
            if not members:
                from relay.seed import DEFAULT_SEEDED_TEAM
                for data in DEFAULT_SEEDED_TEAM:
                    s.add(TeamMember(**data))
                s.commit()
                members = list(s.scalars(select(TeamMember).order_by(TeamMember.created_at.asc())))
            return [
                {
                    "id": m.id,
                    "name": m.name,
                    "email": m.email,
                    "role": m.role,
                    "department": m.department,
                    "status": m.status,
                    "active_tickets_count": m.active_tickets_count,
                    "created_at": m.created_at.isoformat() if m.created_at else None,
                }
                for m in members
            ]

    @app.post("/api/team", status_code=201)
    def create_team_member(body: TeamMemberCreate):
        with app_ctx.session() as s:
            member_id = body.id or f"SPT-{secrets.randbelow(9000) + 1000}"
            existing = s.scalar(select(TeamMember).where(TeamMember.email == body.email))
            if existing:
                raise HTTPException(status_code=409, detail="Team member with this email already exists")
            new_member = TeamMember(
                id=member_id,
                name=body.name,
                email=body.email,
                role=body.role,
                department=body.department,
                status=body.status,
                active_tickets_count=body.active_tickets_count,
            )
            s.add(new_member)
            s.commit()
            s.refresh(new_member)
            return {
                "id": new_member.id,
                "name": new_member.name,
                "email": new_member.email,
                "role": new_member.role,
                "department": new_member.department,
                "status": new_member.status,
                "active_tickets_count": new_member.active_tickets_count,
                "created_at": new_member.created_at.isoformat() if new_member.created_at else None,
            }

    @app.patch("/api/team/{member_id}")
    def update_team_member(member_id: str, body: TeamMemberUpdate):
        with app_ctx.session() as s:
            member = s.get(TeamMember, member_id)
            if not member:
                raise HTTPException(status_code=404, detail="Team member not found")
            if body.name is not None:
                member.name = body.name
            if body.role is not None:
                member.role = body.role
            if body.department is not None:
                member.department = body.department
            if body.status is not None:
                member.status = body.status
            if body.active_tickets_count is not None:
                member.active_tickets_count = body.active_tickets_count
            s.commit()
            s.refresh(member)
            return {
                "id": member.id,
                "name": member.name,
                "email": member.email,
                "role": member.role,
                "department": member.department,
                "status": member.status,
                "active_tickets_count": member.active_tickets_count,
                "created_at": member.created_at.isoformat() if member.created_at else None,
            }

    @app.delete("/api/team/{member_id}")
    def delete_team_member(member_id: str):
        with app_ctx.session() as s:
            member = s.get(TeamMember, member_id)
            if not member:
                raise HTTPException(status_code=404, detail="Team member not found")
            s.delete(member)
            s.commit()
            return {"status": "deleted", "id": member_id}

    app.include_router(create_customer_router(app_ctx))
    app.include_router(create_operator_support_router(app_ctx))
    return app


def _live_services(app_ctx: AppContext, order: dict) -> dict:
    """Live read of the simulated services for the dashboard panels (not an agent action)."""
    out: dict = {"simulated": True}
    probes = {
        "payment": lambda: app_ctx.sim.get_payments(order["id"]).model_dump(),
        "inventory": lambda: app_ctx.sim.get_reservations(order["id"]).model_dump(),
        "stock": lambda: app_ctx.sim.get_stock(order["sku"]).model_dump(),
        "shipment": lambda: app_ctx.sim.get_shipments(order["id"]).model_dump(),
    }
    for name, fn in probes.items():
        try:
            out[name] = fn()
        except ToolError as e:
            out[name] = {"error": e.code, "message": e.message}
    return out
