"""Failure boundaries: crashes, restarts, concurrency, DB-enforced idempotency, limits."""
import threading

import pytest
from sqlalchemy.exc import IntegrityError

from relay import scenarios
from relay.agent import runner
from relay.clients.sim_client import SimClient
from relay.context import AppContext, Hooks
from relay.db import make_engine, make_session_factory
from relay.errors import SimulatedCrash
from relay.models import Approval, IdempotencyRecord, Run, SupportTicket, utcnow
from relay.tools.executor import ToolExecutor
from tests.helpers import decide, expire_lease, order_row, pending_approval, tool_calls, truth


def restart(app: AppContext) -> AppContext:
    """Simulate a process restart: brand-new engine/context over the same databases."""
    engine = make_engine(app.settings.database_url)
    return AppContext(settings=app.settings, engine=engine, session_factory=make_session_factory(engine),
                      sim=SimClient(app.settings.sim_base_url, app.settings.sim_timeout_s), sleep=lambda s: None,
                      hooks=Hooks())


def one_each(app, oid):
    t = truth(app, oid)
    return len(t["payments"]), len(t["reservations"]), len(t["shipments"]), len(t["refunds"])


# ------------------------------------------------------------------ crash AFTER side effect, BEFORE ack
@pytest.mark.parametrize("tool", ["reserve_inventory", "create_shipment"])
def test_crash_after_side_effect_before_ack_resumes_without_duplicate(app, tool):
    order = scenarios.inject(app, "payment_captured_pending", "basic")
    fired = []

    def crash_once(name):
        if name == tool and not fired:
            fired.append(name)
            raise SimulatedCrash()

    app.hooks.after_remote_effect = crash_once
    run, _ = runner.create_run(app, order.id)
    with pytest.raises(SimulatedCrash):
        runner.execute_run(app, run.id)

    # the effect happened at the provider, but Relay never recorded it
    t = truth(app, order.id)
    assert len(t["reservations" if tool == "reserve_inventory" else "shipments"]) == 1
    with app.session() as s:
        assert s.get(IdempotencyRecord, f"{'reserve' if tool == 'reserve_inventory' else 'ship'}:{order.id}").status == "in_progress"

    # a live lease blocks other workers; once it expires, a "restarted" process resumes
    app2 = restart(app)
    assert runner.execute_run(app2, run.id) == "not_claimed"
    expire_lease(app, run.id)
    assert runner.execute_run(app2, run.id) == "succeeded"
    assert one_each(app, order.id) == (1, 1, 1, 0)                       # exactly one of each
    assert any(c.tool == f"{tool}.reconcile" and c.status == "ok" for c in tool_calls(app, run.id))
    assert order_row(app, order.id).status == "shipped"


# ------------------------------------------------------------------ crash at EVERY node boundary
@pytest.mark.parametrize("crash_at", range(1, 28, 2))
def test_crash_at_any_node_boundary_then_resume_is_safe(app, crash_at):
    order = scenarios.inject(app, "payment_captured_pending", "basic")
    counter = {"n": 0}

    def maybe_crash(node):
        counter["n"] += 1
        if counter["n"] == crash_at:
            raise SimulatedCrash()

    app.hooks.before_node = maybe_crash
    run, _ = runner.create_run(app, order.id)
    crashed = False
    try:
        runner.execute_run(app, run.id)
    except SimulatedCrash:
        crashed = True
    app.hooks.before_node = None
    if crash_at <= 20:
        assert crashed, "the run should have been long enough to crash here"
    if crashed:
        expire_lease(app, run.id)
        runner.execute_run(restart(app), run.id)
    with app.session() as s:
        final = s.get(Run, run.id)
    assert final.outcome == "fulfilled", (crash_at, final.summary)
    assert one_each(app, order.id) == (1, 1, 1, 0)


def test_resume_interrupted_picks_up_dead_workers(app):
    order = scenarios.inject(app, "payment_captured_pending", "basic")
    n = {"i": 0}

    def crash_third(_):
        n["i"] += 1
        if n["i"] == 3:
            raise SimulatedCrash()

    app.hooks.before_node = crash_third
    run, _ = runner.create_run(app, order.id)
    with pytest.raises(SimulatedCrash):
        runner.execute_run(app, run.id)
    app.hooks.before_node = None
    assert runner.resume_interrupted(app) == []          # lease still valid
    expire_lease(app, run.id)
    app2 = restart(app)
    assert runner.resume_interrupted(app2) == [run.id]
    app2.pool.shutdown(wait=True)
    assert order_row(app, order.id).status == "shipped"


# ------------------------------------------------------------------ concurrency
def test_concurrent_start_requests_create_exactly_one_active_run(app):
    order = scenarios.inject(app, "payment_captured_pending", "basic")
    results, barrier = [], threading.Barrier(8)

    def go():
        barrier.wait()
        results.append(runner.create_run(app, order.id))

    threads = [threading.Thread(target=go) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sum(1 for _, created in results if created) == 1
    assert len({r.id for r, _ in results}) == 1


def test_concurrent_executors_on_one_run_only_one_acts(app):
    order = scenarios.inject(app, "payment_captured_pending", "basic")
    run, _ = runner.create_run(app, order.id)
    out, barrier = [], threading.Barrier(5)

    def go():
        barrier.wait()
        out.append(runner.execute_run(app, run.id))

    threads = [threading.Thread(target=go) for _ in range(5)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert out.count("not_claimed") + out.count("succeeded") == 5 and out.count("succeeded") >= 1
    assert one_each(app, order.id) == (1, 1, 1, 0)


def test_concurrent_approval_decisions_have_a_single_winner(app):
    from relay.approvals import ApprovalError
    from tests.helpers import inject_and_run
    oid, run = inject_and_run(app, "out_of_stock_after_payment", "basic")
    ap = pending_approval(app, oid)
    results, barrier = [], threading.Barrier(6)

    def go(decision):
        barrier.wait()
        try:
            results.append(("ok", decide(app, ap.id, decision, "alice").applied))
        except ApprovalError as e:
            results.append(("err", e.status_code))

    threads = [threading.Thread(target=go, args=("approve" if i % 2 == 0 else "reject",)) for i in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sum(1 for k, v in results if k == "ok" and v is True) == 1        # exactly one writer won
    assert all(v == 409 for k, v in results if k == "err")
    with app.session() as s:
        final = s.get(Approval, ap.id).status
    assert final in ("approved", "rejected")
    runner.execute_run(app, run.id)
    expected_refunds = 1 if final == "approved" else 0
    assert len(truth(app, oid)["refunds"]) == expected_refunds


# ------------------------------------------------------------------ database-enforced guarantees
def test_db_rejects_second_active_run_for_order(app):
    order = scenarios.inject(app, "payment_captured_pending", "basic")
    runner.create_run(app, order.id)
    with app.session() as s:
        s.add(Run(id="run_dup", order_id=order.id, correlation_id="c", mode="demo", status="running", state={}))
        with pytest.raises(IntegrityError):
            s.commit()


def test_db_rejects_duplicate_idempotency_key_and_open_approvals_and_tickets(app):
    order = scenarios.inject(app, "out_of_stock_after_payment", "basic")
    with app.session() as s:
        s.add(IdempotencyRecord(key="k1", tool="x", order_id=order.id, status="in_progress"))
        s.commit()
        s.add(IdempotencyRecord(key="k1", tool="x", order_id=order.id, status="in_progress"))
        with pytest.raises(IntegrityError):
            s.commit()
    def mk(i):
        return Approval(id=f"a{i}", order_id=order.id, kind="refund", amount_cents=1, status="pending",
                        reason="r", expires_at=utcnow())

    with app.session() as s:
        s.add(mk(1))
        s.commit()
        s.add(mk(2))
        with pytest.raises(IntegrityError):
            s.commit()
    with app.session() as s:
        s.add(SupportTicket(id="t1", order_id=order.id, reason_code="X", summary="s"))
        s.commit()
        s.add(SupportTicket(id="t2", order_id=order.id, reason_code="X", summary="s"))
        with pytest.raises(IntegrityError):
            s.commit()


def test_executor_replays_prior_success_instead_of_repeating_side_effect(app):
    order = scenarios.inject(app, "inventory_response_lost", "lost_before_commit")
    run, _ = runner.create_run(app, order.id)
    ex = ToolExecutor(app, run.id, order.id, run.correlation_id)
    args = {"order_id": order.id, "sku": order.sku, "quantity": 1}
    first = ex.call("reserve_inventory", args)
    second = ex.call("reserve_inventory", args)
    assert first.ok and second.ok and second.replayed
    assert first.output["reservation_id"] == second.output["reservation_id"]
    assert len(truth(app, order.id)["reservations"]) == 1


def test_provider_dedupes_same_key_even_if_relay_lost_its_record(app):
    """Layer 2 of defence: the provider's own Idempotency-Key handling."""
    order = scenarios.inject(app, "inventory_response_lost", "lost_before_commit")
    kw = dict(order_id=order.id, sku=order.sku, quantity=1, idem_key="manual-key")
    a = app.sim.create_reservation(**kw)
    b = app.sim.create_reservation(**kw)
    assert a.id == b.id and b.replayed


# ------------------------------------------------------------------ limits
def test_tool_call_budget_stops_the_run_and_escalates(app_with_low_budget):
    app = app_with_low_budget
    order = scenarios.inject(app, "payment_captured_pending", "basic")
    run, _ = runner.create_run(app, order.id)
    runner.execute_run(app, run.id)
    with app.session() as s:
        final = s.get(Run, run.id)
    assert final.status == "escalated" and "tool_calls" in final.error
    assert [t.reason_code for t in __import__("tests.helpers", fromlist=["x"]).tickets(app, order.id)] == ["BUDGET_EXHAUSTED"]
    # the budget bounded real work (+ the exempt escalation/verification calls)
    assert len([c for c in tool_calls(app, run.id) if c.tool not in ("create_support_ticket", "verify_recovery")]) <= 5


def test_step_limit_stops_the_run(make_app):
    app = make_app(max_steps=4)
    order = scenarios.inject(app, "payment_captured_pending", "basic")
    run, _ = runner.create_run(app, order.id)
    runner.execute_run(app, run.id)
    with app.session() as s:
        final = s.get(Run, run.id)
    assert final.status == "escalated" and "steps" in final.error


@pytest.fixture()
def app_with_low_budget(make_app):
    return make_app(max_tool_calls=5)


def test_step_mode_advances_exactly_one_node_per_call(app):
    order = scenarios.inject(app, "payment_captured_pending", "basic")
    run, _ = runner.create_run(app, order.id, step_mode=True)
    assert run.status == "paused"
    seen = []
    for _ in range(3):
        assert runner.execute_run(app, run.id, single_step=True) == "paused"
        with app.session() as s:
            seen.append(s.get(Run, run.id).steps)
    assert seen == [1, 2, 3]
    assert runner.execute_run(app, run.id) == "succeeded"       # then let it finish
