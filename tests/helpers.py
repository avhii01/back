from __future__ import annotations

import time

from sqlalchemy import select

from relay import scenarios
from relay.agent import runner
from relay.approvals import decide_approval
from relay.models import Approval, Order, Run, SupportTicket, ToolCall


def truth(app, order_id: str) -> dict:
    """Ground truth straight from the simulator's admin API (what really happened)."""
    return app.sim.admin("GET", f"/admin/orders/{order_id}/state")


def run_to_end(app, order_id: str, **kw) -> Run:
    run, _ = runner.create_run(app, order_id)
    runner.execute_run(app, run.id)
    with app.session() as s:
        return s.get(Run, run.id)


def inject_and_run(app, scenario: str, variant: str | None = None, overrides: dict | None = None):
    order = scenarios.inject(app, scenario, variant, overrides)
    run = run_to_end(app, order.id)
    return order.id, run


def order_row(app, order_id: str) -> Order:
    with app.session() as s:
        return s.get(Order, order_id)


def tool_calls(app, run_id: str) -> list[ToolCall]:
    with app.session() as s:
        return list(s.scalars(select(ToolCall).where(ToolCall.run_id == run_id).order_by(ToolCall.seq)))


def calls_named(app, run_id: str, name: str, status: str | None = None) -> list[ToolCall]:
    return [c for c in tool_calls(app, run_id) if c.tool == name and (status is None or c.status == status)]


def tickets(app, order_id: str) -> list[SupportTicket]:
    with app.session() as s:
        return list(s.scalars(select(SupportTicket).where(SupportTicket.order_id == order_id)))


def pending_approval(app, order_id: str) -> Approval | None:
    with app.session() as s:
        return s.scalars(select(Approval).where(Approval.order_id == order_id)).first()


def decide(app, approval_id: str, decision="approve", who="alice"):
    return decide_approval(app, approval_id, decision, who, None, "test-corr")


def expire_lease(app, run_id: str) -> None:
    from datetime import timedelta

    from relay.models import utcnow

    with app.session() as s:
        run = s.get(Run, run_id)
        run.lease_expires_at = utcnow() - timedelta(seconds=5)
        s.commit()


def wait_for(predicate, timeout=15.0, interval=0.05):
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(interval)
    return False
