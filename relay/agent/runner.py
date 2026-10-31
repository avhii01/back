"""Run lifecycle: create, claim (lease), execute, resume.

Concurrency model
  * One ACTIVE run per order - enforced by a partial unique index, so two concurrent
    "start recovery" requests cannot both win (the loser gets the existing run back).
  * One WORKER per run - a compare-and-set lease (`lease_owner`, `lease_expires_at`).
    A worker that dies keeps its lease until it expires; then any process may resume the run.
"""
from __future__ import annotations

import logging
import uuid
from datetime import timedelta

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from relay.agent.facts import initial_state
from relay.agent.graph import AgentNodes, build_graph
from relay.agent.llm_client import LLMError
from relay.agent.planner_llm import LLMPlanner
from relay.agent.planner_rules import RulesPlanner
from relay.audit import record_event
from relay.context import AppContext
from relay.errors import LeaseLost
from relay.models import ACTIVE_RUN_STATUSES, Order, Run, utcnow

log = logging.getLogger("relay.runner")
RESUMABLE = ("pending", "running", "paused", "awaiting_approval")


class OrderNotFound(Exception):
    pass


def create_run(app: AppContext, order_id: str, *, step_mode: bool = False,
               correlation_id: str | None = None) -> tuple[Run, bool]:
    """Returns (run, created). If an active run exists, returns it with created=False."""
    cid = correlation_id or uuid.uuid4().hex
    with app.session() as s:
        if s.get(Order, order_id) is None:
            raise OrderNotFound(order_id)
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        mode = app.settings.mode
        run = Run(id=run_id, order_id=order_id, correlation_id=cid, mode=mode,
                  status="paused" if step_mode else "pending", step_mode=step_mode,
                  state=dict(initial_state(run_id, order_id, cid, mode, step_mode)))
        s.add(run)
        try:
            s.flush()
        except IntegrityError:
            s.rollback()
            existing = s.scalars(select(Run).where(Run.order_id == order_id,
                                                   Run.status.in_(ACTIVE_RUN_STATUSES))).first()
            if existing is None:
                raise
            return existing, False
        record_event(s, correlation_id=cid, run_id=run.id, order_id=order_id, actor="system", event_type="run_created",
                     payload={"mode": run.mode, "step_mode": step_mode})
        s.commit()
        return run, True


def _claim(app: AppContext, run_id: str, worker_id: str) -> bool:
    now = utcnow()
    with app.session() as s:
        res = s.execute(
            update(Run)
            .where(Run.id == run_id, Run.status.in_(RESUMABLE))
            .where((Run.lease_owner.is_(None)) | (Run.lease_expires_at < now))
            .values(lease_owner=worker_id, status="running",
                    lease_expires_at=now + timedelta(seconds=app.settings.lease_seconds)))
        s.commit()
        return res.rowcount == 1


def _release(app: AppContext, run_id: str, worker_id: str) -> None:
    with app.session() as s:
        s.execute(update(Run).where(Run.id == run_id, Run.lease_owner == worker_id)
                  .values(lease_owner=None, lease_expires_at=None))
        s.commit()


def make_planner(app: AppContext, run: Run):
    if run.mode == "llm":
        return LLMPlanner(app, run.id)
    return RulesPlanner()


def execute_run(app: AppContext, run_id: str, *, single_step: bool = False) -> str:
    """Advance a run: to completion (or a human gate), or exactly one node when single_step.
    Returns the run status afterwards, or 'not_claimed' when another worker holds the lease."""
    worker_id = f"w_{uuid.uuid4().hex[:8]}"
    if not _claim(app, run_id, worker_id):
        return "not_claimed"
    with app.session() as s:
        run = s.get(Run, run_id)
        state = dict(run.state)
        mode = run.mode
    state.update(halt=None, step_mode=single_step)
    try:
        planner = make_planner(app, run) if mode == "llm" else RulesPlanner()
    except LLMError as e:
        _fail(app, run_id, worker_id, state, f"LLM mode unavailable: {e}")
        return "failed"
    nodes = AgentNodes(app, run_id, worker_id, planner)
    nodes.bind(state)  # type: ignore[arg-type]
    graph = build_graph(nodes)
    try:
        graph.invoke(state, config={"recursion_limit": app.settings.max_steps * 2 + 20})
    except LeaseLost:
        log.warning("lease lost for run %s", run_id)
        return "lease_lost"
    except Exception as e:  # unexpected bug: record it, never leave the run 'running' forever
        log.exception("run %s crashed", run_id)
        _fail(app, run_id, worker_id, state, f"unexpected error: {type(e).__name__}: {str(e)[:300]}")
        return "failed"
    # NOTE: BaseException (e.g. SimulatedCrash / SIGKILL) deliberately skips this: the lease stays
    # held until it expires, exactly like a real crashed worker.
    _release(app, run_id, worker_id)
    with app.session() as s:
        return s.get(Run, run_id).status


def _fail(app: AppContext, run_id: str, worker_id: str, state: dict, message: str) -> None:
    with app.session() as s:
        s.execute(update(Run).where(Run.id == run_id, Run.lease_owner == worker_id)
                  .values(status="failed", outcome="failed", error=message, summary=f"Run failed: {message}",
                          finished_at=utcnow(), lease_owner=None, lease_expires_at=None))
        record_event(s, correlation_id=state.get("correlation_id", "-"), run_id=run_id, order_id=state.get("order_id"),
                     actor="system", event_type="run_failed", payload={"error": message})
        s.commit()


def submit(app: AppContext, run_id: str, *, single_step: bool = False) -> None:
    """Run in the background thread pool (used by the HTTP API)."""
    app.pool.submit(execute_run, app, run_id, single_step=single_step)


def resume_interrupted(app: AppContext) -> list[str]:
    """Called on startup: pick up runs whose worker died (status 'running', lease expired)
    and runs created in auto mode but never started."""
    now = utcnow()
    with app.session() as s:
        ids = list(s.scalars(select(Run.id).where(
            ((Run.status == "running") & ((Run.lease_expires_at.is_(None)) | (Run.lease_expires_at < now)))
            | ((Run.status == "pending") & (Run.step_mode.is_(False))))))
    for rid in ids:
        submit(app, rid)
    return ids
