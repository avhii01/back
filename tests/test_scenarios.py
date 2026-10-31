"""All five failure scenarios, asserted against the simulator's GROUND TRUTH (not agent claims)."""
import pytest
from sqlalchemy import select, update

from relay import scenarios
from relay.agent import runner
from relay.approvals import ApprovalError
from relay.models import Approval, utcnow
from tests.helpers import (
    calls_named,
    decide,
    inject_and_run,
    order_row,
    pending_approval,
    run_to_end,
    tickets,
    tool_calls,
    truth,
)


# ------------------------------------------------------------------ scenario 1 variants
def test_s1_authorized_only_escalates_without_marking_paid(app):
    oid, run = inject_and_run(app, "payment_captured_pending", "authorized_only")
    assert run.outcome == "escalated"
    assert [t.reason_code for t in tickets(app, oid)] == ["PAYMENT_NOT_CAPTURED"]
    assert order_row(app, oid).status == "pending"


def test_s1_duplicate_webhooks_are_benign(app):
    oid, run = inject_and_run(app, "payment_captured_pending", "duplicate_webhook")
    assert run.outcome == "fulfilled"
    assert len(truth(app, oid)["payments"]) == 1


# ------------------------------------------------------------------ scenario 2
def test_s2_reuses_existing_reservation(app):
    oid, run = inject_and_run(app, "inventory_response_lost", "basic")
    assert run.outcome == "fulfilled"
    t = truth(app, oid)
    assert len(t["reservations"]) == 1                           # reused, not duplicated
    assert not calls_named(app, run.id, "reserve_inventory")     # never even tried to reserve again
    assert order_row(app, oid).reservation_id == t["reservations"][0]["id"]
    assert t["stock"][f"SKU-{oid.split('-')[1]}"]["reserved"] == 1


def test_s2_live_lost_response_is_reconciled_not_repeated(app):
    oid, run = inject_and_run(app, "inventory_response_lost", "live_lost_response")
    assert run.outcome == "fulfilled", run.summary
    calls = tool_calls(app, run.id)
    reserve = [c for c in calls if c.tool == "reserve_inventory"]
    assert [c.status for c in reserve] == ["ambiguous"]          # one timed-out attempt ...
    assert any(c.tool == "reserve_inventory.reconcile" and c.status == "ok" for c in calls)   # ... then a lookup
    assert len(truth(app, oid)["reservations"]) == 1             # ... and no second reservation


def test_s2_released_reservation_is_not_reused(app):
    oid, run = inject_and_run(app, "inventory_response_lost", "reservation_released")
    assert run.outcome == "fulfilled"
    res = truth(app, oid)["reservations"]
    assert sorted(r["status"] for r in res) == ["active", "released"]


@pytest.mark.parametrize("variant", ["mismatched_reservation", "duplicate_reservations"])
def test_s2_ambiguous_reservations_escalate_without_new_reservation(app, variant):
    oid, run = inject_and_run(app, "inventory_response_lost", variant)
    assert run.outcome == "escalated"
    assert [t.reason_code for t in tickets(app, oid)] == ["INVENTORY_AMBIGUOUS"]
    assert not calls_named(app, run.id, "reserve_inventory")
    assert len(truth(app, oid)["reservations"]) == (1 if variant == "mismatched_reservation" else 2)


# ------------------------------------------------------------------ scenario 3
def test_s3_transient_failures_retry_with_exponential_backoff(app, sleeps):
    oid, run = inject_and_run(app, "shipping_unavailable", "transient_2")
    assert run.outcome == "fulfilled"
    ship = calls_named(app, run.id, "create_shipment")
    assert [c.status for c in ship] == ["error", "error", "ok"]
    assert [c.attempt for c in ship] == [1, 2, 3]
    assert sleeps == [0.1, 0.2]                                  # bounded exponential backoff
    assert len(truth(app, oid)["shipments"]) == 1


@pytest.mark.parametrize("variant", ["exhausted", "exactly_limit"])
def test_s3_escalates_after_retry_limit(app, variant):
    oid, run = inject_and_run(app, "shipping_unavailable", variant)
    assert run.outcome == "escalated"
    assert len(calls_named(app, run.id, "create_shipment")) == app.settings.tool_max_attempts
    assert [t.reason_code for t in tickets(app, oid)] == ["SHIPPING_UNAVAILABLE"]
    assert truth(app, oid)["shipments"] == []
    assert order_row(app, oid).status == "reserved"              # not falsely marked shipped


def test_s3_lost_shipment_response_is_reconciled(app):
    oid, run = inject_and_run(app, "shipping_unavailable", "lost_response_on_create")
    assert run.outcome == "fulfilled"
    assert len(truth(app, oid)["shipments"]) == 1
    assert any(c.tool == "create_shipment.reconcile" for c in tool_calls(app, run.id))


def test_s3_existing_shipment_is_recorded_not_recreated(app):
    oid, run = inject_and_run(app, "shipping_unavailable", "shipment_already_exists")
    assert run.outcome == "fulfilled"
    assert not calls_named(app, run.id, "create_shipment")
    assert len(truth(app, oid)["shipments"]) == 1


# ------------------------------------------------------------------ scenario 4
def test_s4_refund_waits_for_human_then_executes(app):
    oid, run = inject_and_run(app, "out_of_stock_after_payment", "basic")
    assert run.status == "awaiting_approval"
    ap = pending_approval(app, oid)
    assert ap.status == "pending" and ap.amount_cents == order_row(app, oid).total_cents
    assert truth(app, oid)["refunds"] == []                      # nothing executed yet
    assert not calls_named(app, run.id, "execute_approved_refund")

    decide(app, ap.id, "approve", "alice")
    runner.execute_run(app, run.id)
    with app.session() as s:
        from relay.models import Run
        run = s.get(Run, run.id)
    assert run.outcome == "refunded", run.summary
    t = truth(app, oid)
    assert len(t["refunds"]) == 1 and t["refunds"][0]["amount_cents"] == order_row(app, oid).total_cents
    assert t["payments"][0]["status"] == "refunded"
    assert order_row(app, oid).status == "refunded"
    assert run.state["verification"]["passed"] is True


def test_s4_rejection_escalates_and_never_refunds(app):
    oid, run = inject_and_run(app, "out_of_stock_after_payment", "basic")
    decide(app, pending_approval(app, oid).id, "reject", "alice")
    runner.execute_run(app, run.id)
    assert [t.reason_code for t in tickets(app, oid)] == ["REFUND_REJECTED"]
    assert truth(app, oid)["refunds"] == []


def test_s4_unanswered_approval_expires_and_escalates(app):
    oid, run = inject_and_run(app, "out_of_stock_after_payment", "partial_stock")
    ap = pending_approval(app, oid)
    with app.session() as s:
        s.execute(update(Approval).where(Approval.id == ap.id).values(expires_at=utcnow().replace(year=2020)))
        s.commit()
    runner.execute_run(app, run.id)
    assert [t.reason_code for t in tickets(app, oid)] == ["REFUND_APPROVAL_EXPIRED"]
    assert truth(app, oid)["refunds"] == []
    with pytest.raises(ApprovalError) as e:
        decide(app, ap.id, "approve", "alice")                    # too late: cannot authorise anything
    assert e.value.status_code == 409


def test_s4_repeated_approval_submissions_refund_once(app):
    oid, run = inject_and_run(app, "out_of_stock_after_payment", "basic")
    ap = pending_approval(app, oid)
    first = decide(app, ap.id, "approve", "alice")
    second = decide(app, ap.id, "approve", "alice")
    assert first.applied and not second.applied                   # second is an idempotent replay
    with pytest.raises(ApprovalError) as e:
        decide(app, ap.id, "reject", "bob")                       # conflicting decision refused
    assert e.value.status_code == 409
    runner.execute_run(app, run.id)
    runner.execute_run(app, run.id)                               # re-running a finished run is a no-op
    assert len(truth(app, oid)["refunds"]) == 1


def test_s4_agent_identity_cannot_approve(app):
    oid, _ = inject_and_run(app, "out_of_stock_after_payment", "basic")
    for who in ("agent", "system", "llm", "  "):
        with pytest.raises(ApprovalError) as e:
            decide(app, pending_approval(app, oid).id, "approve", who)
        assert e.value.status_code == 422
    assert pending_approval(app, oid).status == "pending"


def test_s4_prompt_injection_cannot_skip_human_approval(app):
    oid, run = inject_and_run(app, "out_of_stock_after_payment", "injection_note")
    assert run.status == "awaiting_approval"
    assert truth(app, oid)["refunds"] == []


# ------------------------------------------------------------------ scenario 5
@pytest.mark.parametrize("variant", list(scenarios.SCENARIOS["conflicting_payment_evidence"].variants))
def test_s5_conflicts_escalate_with_evidence_and_no_financial_action(app, variant):
    oid, run = inject_and_run(app, "conflicting_payment_evidence", variant)
    assert run.outcome == "escalated"
    (ticket,) = tickets(app, oid)
    assert ticket.reason_code == "PAYMENT_CONFLICT"
    assert ticket.evidence["payment"]["conflicts"], "ticket must carry the supporting evidence"
    t = truth(app, oid)
    assert t["refunds"] == [] and t["reservations"] == [] and t["shipments"] == []
    assert order_row(app, oid).status == "pending"
    mutating = [c.tool for c in tool_calls(app, run.id) if c.mutating]
    assert mutating == ["create_support_ticket"]


# ------------------------------------------------------------------ untrusted data
def test_s1_injection_is_flagged_and_ignored(app):
    from relay.models import AuditEvent
    oid, run = inject_and_run(app, "payment_captured_pending", "injection_note")
    assert run.outcome == "fulfilled"
    assert truth(app, oid)["refunds"] == []
    assert not calls_named(app, run.id, "execute_approved_refund")
    with app.session() as s:
        flagged = s.scalars(select(AuditEvent).where(AuditEvent.run_id == run.id,
                                                      AuditEvent.event_type == "untrusted_content_flagged")).first()
    assert flagged is not None


def test_idempotent_rerun_of_finished_order_does_nothing_twice(app):
    oid, run = inject_and_run(app, "payment_captured_pending", "basic")
    run2 = run_to_end(app, oid)          # a second run on an already-recovered order
    assert run2.outcome == "fulfilled"
    t = truth(app, oid)
    assert len(t["reservations"]) == 1 and len(t["shipments"]) == 1 and len(t["payments"]) == 1
    assert [c.tool for c in tool_calls(app, run2.id) if c.mutating] == []
