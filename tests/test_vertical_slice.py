"""Phase 1: payment captured but order pending -> investigate -> correct order -> verify."""
from tests.helpers import calls_named, inject_and_run, order_row, tool_calls, truth


def test_payment_captured_pending_recovers_without_double_charge(app):
    order_id, run = inject_and_run(app, "payment_captured_pending", "basic")

    assert run.status == "succeeded" and run.outcome == "fulfilled", run.summary
    o = order_row(app, order_id)
    assert o.status == "shipped" and o.payment_ref and o.reservation_id and o.shipment_id

    t = truth(app, order_id)
    assert len(t["payments"]) == 1 and t["payments"][0]["status"] == "captured"   # not charged again
    assert t["refunds"] == []
    assert len(t["reservations"]) == 1 and len(t["shipments"]) == 1

    # the trace shows investigation BEFORE any mutation
    calls = tool_calls(app, run.id)
    names = [c.tool for c in calls]
    first_mutation = next(i for i, c in enumerate(calls) if c.mutating)
    assert {"get_order_status", "get_payment_status"} <= set(names[:first_mutation])
    assert calls[first_mutation].tool == "update_order_status"
    assert names[-1] == "verify_recovery"
    assert run.state["verification"]["passed"] is True


def test_run_is_labelled_deterministic_not_llm(app):
    _, run = inject_and_run(app, "payment_captured_pending", "basic")
    assert run.mode == "demo"
    assert all(h["proposal"]["source"] == "rules" for h in run.state["history"])
    with app.session() as s:
        from sqlalchemy import select

        from relay.models import LLMCall
        assert s.scalars(select(LLMCall)).first() is None   # no fabricated LLM usage


def test_mutations_are_audited_with_correlation_id(app):
    from sqlalchemy import select

    from relay.models import AuditEvent
    _, run = inject_and_run(app, "payment_captured_pending", "basic")
    with app.session() as s:
        events = list(s.scalars(select(AuditEvent).where(AuditEvent.run_id == run.id)))
    assert events and {e.correlation_id for e in events} == {run.correlation_id}
    assert {"observe", "proposal", "policy_decision", "tool_call", "verification"} <= {e.event_type for e in events}


def test_retry_only_calls_never_exceed_budget(app):
    _, run = inject_and_run(app, "payment_captured_pending", "basic")
    assert len(tool_calls(app, run.id)) <= app.settings.max_tool_calls
    assert calls_named(app, run.id, "get_payment_status")
