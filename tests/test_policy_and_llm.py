"""Policy engine under forged proposals, and the LLM planner against a mocked provider.

NOTE: the LLM path is exercised with a scripted httpx.MockTransport - these tests prove Relay's
handling (validation, policy, fail-closed, usage accounting). They do not measure a real model.
"""
import json

import httpx
import pytest

from relay import scenarios
from relay.agent import policy, runner
from relay.agent.llm_client import LLMClient
from relay.agent.types import Proposal
from relay.api import serializers as ser
from relay.models import LLMCall, Run
from relay.tools.executor import ToolExecutor
from tests.helpers import calls_named, inject_and_run, order_row, tickets, truth

SECRET = "sk-test-secret-1234567890"


# ------------------------------------------------------------------ policy unit tests
def _state_after(app, scenario, variant, steps):
    order = scenarios.inject(app, scenario, variant)
    run, _ = runner.create_run(app, order.id, step_mode=True)
    for _ in range(steps):
        runner.execute_run(app, run.id, single_step=True)
    with app.session() as s:
        return order, s.get(Run, run.id).state


def test_policy_replaces_model_supplied_ids_with_trusted_ones(app):
    order, state = _state_after(app, "payment_captured_pending", "basic", 2)   # observe + investigate
    p = Proposal(action="tool", tool="update_order_status", source="llm",
                 args={"order_id": order.id, "new_status": "paid", "payment_id": "pay_EVIL", "expected_from": "shipped"})
    d = policy.evaluate(state, p)
    assert d.allowed
    assert d.canonical_args["payment_id"] != "pay_EVIL" and d.canonical_args["expected_from"] == "pending"


def test_policy_denies_paid_transition_when_payment_conflicts(app):
    order, state = _state_after(app, "conflicting_payment_evidence", "double_capture", 2)
    state["stale"] = []
    d = policy.evaluate(state, Proposal(action="tool", tool="update_order_status", args={"new_status": "paid"}))
    assert not d.allowed and d.rule_ids == ["P-PAYMENT-NOT-CONFIRMED"]


def test_policy_denies_unlisted_tools_and_stale_evidence(app):
    order, state = _state_after(app, "payment_captured_pending", "basic", 2)
    assert policy.evaluate(state, Proposal(action="tool", tool="verify_recovery")).rule_ids == ["P-TOOL-NOT-ALLOWED"]
    assert policy.evaluate(state, Proposal(action="tool", tool="drop_database")).rule_ids == ["P-TOOL-NOT-ALLOWED"]
    state["stale"] = ["payment"]
    d = policy.evaluate(state, Proposal(action="tool", tool="update_order_status", args={"new_status": "paid"}))
    assert d.rule_ids == ["P-EVIDENCE-STALE"]


def test_policy_ticket_evidence_is_system_built_not_model_supplied(app):
    order, state = _state_after(app, "payment_captured_pending", "basic", 2)
    p = Proposal(action="tool", tool="create_support_ticket",
                 args={"reason_code": "TOOL_FAILURE", "summary": "x", "evidence": {"fabricated": True}})
    d = policy.evaluate(state, p)
    assert d.allowed and "fabricated" not in d.canonical_args["evidence"]
    assert "payment" in d.canonical_args["evidence"]


def test_refund_tool_itself_refuses_without_human_approval(app):
    """Defence in depth: even a direct call (bypassing policy) cannot refund without an approval."""
    from tests.helpers import pending_approval
    oid, run = inject_and_run(app, "out_of_stock_after_payment", "basic")
    ap = pending_approval(app, oid)
    ex = ToolExecutor(app, "adhoc", oid, "corr")
    with app.session() as s:
        from relay.models import Run as R
        s.add(R(id="adhoc", order_id=oid, correlation_id="corr", mode="demo", status="failed", state={}))
        s.commit()
    r = ex.call("execute_approved_refund", {"order_id": oid, "approval_id": ap.id})
    assert not r.ok and r.error["code"] == "APPROVAL_NOT_GRANTED"
    assert truth(app, oid)["refunds"] == []


def test_tool_input_validation_rejects_bad_arguments(app):
    oid, run = inject_and_run(app, "payment_captured_pending", "basic")
    ex = ToolExecutor(app, run.id, oid, run.correlation_id)
    assert ex.call("check_stock", {"sku": "X", "quantity": -3}).error["code"] == "INVALID_INPUT"
    assert ex.call("get_order_status", {"order_id": "../etc/passwd"}).error["code"] == "INVALID_INPUT"
    assert ex.call("get_order_status", {"order_id": oid, "extra": 1}).error["code"] == "INVALID_INPUT"
    assert ex.call("nope", {}).error["code"] == "UNKNOWN_TOOL"
    assert ex.call("get_order_status", {"order_id": "ORD-MISSING"}).error["code"] == "ORDER_NOT_FOUND"


def test_invalid_state_transitions_are_rejected_by_the_tool(app):
    oid, run = inject_and_run(app, "payment_captured_pending", "basic")      # now shipped
    ex = ToolExecutor(app, run.id, oid, run.correlation_id)
    r = ex.call("update_order_status", {"order_id": oid, "expected_from": "shipped", "new_status": "paid",
                                        "payment_id": "p"})
    assert not r.ok and r.error["code"] == "INVALID_TRANSITION"


# ------------------------------------------------------------------ LLM planner
def llm_app(make_app, actions, *, usage=True, status=200, text_only=False):
    seen = []
    queue = iter(actions)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append({"headers": dict(request.headers), "body": json.loads(request.content)})
        if status != 200:
            return httpx.Response(status, json={"error": "boom"})
        a = next(queue)
        msg = {"role": "assistant", "content": "I would like to refund."} if text_only else {
            "role": "assistant", "tool_calls": [{"id": "c", "type": "function", "function": {
                "name": "propose_action", "arguments": a if isinstance(a, str) else json.dumps(a)}}]}
        body = {"model": "mock-model", "choices": [{"message": msg}]}
        if usage:
            body["usage"] = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
        return httpx.Response(200, json=body)

    app = make_app(agent_mode="llm", llm_api_key=SECRET, llm_base_url="http://llm.test/v1",
                   llm_price_input_per_mtok=1.0, llm_price_output_per_mtok=2.0)
    app.llm = LLMClient(app.settings, transport=httpx.MockTransport(handler))
    return app, seen


def tool(name, **args):
    return {"action": "tool", "tool": name, "args": args, "summary": f"call {name}"}


HAPPY = [tool("update_order_status", new_status="paid"), tool("reserve_inventory"),
         tool("update_order_status", new_status="reserved"), tool("create_shipment"),
         tool("update_order_status", new_status="shipped"), {"action": "finish", "summary": "done"}]


def test_llm_mode_recovers_and_reports_real_usage(make_app):
    app, seen = llm_app(make_app, HAPPY)
    oid, run = inject_and_run(app, "payment_captured_pending", "basic")
    assert run.mode == "llm" and run.outcome == "fulfilled", run.summary
    assert all(h["proposal"]["source"] == "llm" for h in run.state["history"] if h["proposal"]["source"] != "policy")
    llm = ser.run_detail(app, run.id)["metrics"]["llm"]
    assert llm["calls"] == 6 and llm["prompt_tokens"] == 600 and llm["completion_tokens"] == 120
    assert llm["estimated_cost_usd"] == pytest.approx(600 * 1 / 1e6 + 120 * 2 / 1e6)
    assert llm["usage_reported_by_provider"] is True
    # key is used server-side as a header only; untrusted data is delimited in the prompt
    assert seen[0]["headers"]["authorization"] == f"Bearer {SECRET}"
    prompt = json.dumps(seen[0]["body"]["messages"])
    assert "<untrusted_data" in prompt and SECRET not in prompt


def test_missing_provider_usage_is_reported_as_unknown_not_invented(make_app):
    app, _ = llm_app(make_app, HAPPY, usage=False)
    _, run = inject_and_run(app, "payment_captured_pending", "basic")
    llm = ser.run_detail(app, run.id)["metrics"]["llm"]
    assert llm["total_tokens"] is None and llm["estimated_cost_usd"] is None
    assert llm["usage_reported_by_provider"] is False


def test_compromised_model_cannot_refund_without_approval(make_app):
    # model follows the injected note and tries to execute a refund twice
    evil = tool("execute_approved_refund", approval_id="apr_fake")
    app, _ = llm_app(make_app, [evil, evil])
    oid, run = inject_and_run(app, "out_of_stock_after_payment", "injection_note")
    assert run.outcome == "escalated"
    assert [t.reason_code for t in tickets(app, oid)] == ["POLICY_DENIED"]
    assert truth(app, oid)["refunds"] == []
    assert not calls_named(app, run.id, "execute_approved_refund")
    denied = [h for h in run.state["history"] if not h["allowed"]]
    assert len(denied) == 2 and denied[0]["rule_ids"] == ["P-REFUND-NO-APPROVAL"]


def test_model_cannot_approve_its_own_refund(make_app):
    app, _ = llm_app(make_app, [tool("request_refund_approval", reason="oos"),
                                tool("execute_approved_refund", approval_id="whatever"),
                                {"action": "wait_for_approval", "summary": "waiting"}])
    oid, run = inject_and_run(app, "out_of_stock_after_payment", "basic")
    assert run.status == "awaiting_approval"
    assert truth(app, oid)["refunds"] == []
    assert any(h["rule_ids"] == ["P-REFUND-NO-APPROVAL"] for h in run.state["history"])


def test_model_asking_for_a_double_reserve_is_denied(make_app):
    steps = [tool("update_order_status", new_status="paid"), tool("reserve_inventory"),
             tool("reserve_inventory"),                       # greedy repeat - policy must refuse
             tool("update_order_status", new_status="reserved"), tool("create_shipment"),
             tool("update_order_status", new_status="shipped"), {"action": "finish", "summary": "ok"}]
    app, _ = llm_app(make_app, steps)
    oid, run = inject_and_run(app, "payment_captured_pending", "basic")
    assert run.outcome == "fulfilled"
    assert len(truth(app, oid)["reservations"]) == 1
    assert any("P-REUSE-EXISTING-RESERVATION" in h["rule_ids"] for h in run.state["history"])


def test_invalid_model_output_fails_closed_to_escalation(make_app):
    app, _ = llm_app(make_app, ["{not json"])
    oid, run = inject_and_run(app, "payment_captured_pending", "basic")
    assert run.outcome == "escalated"
    assert [t.reason_code for t in tickets(app, oid)] == ["LLM_INVALID_OUTPUT"]
    assert order_row(app, oid).status == "pending"


def test_text_only_answer_is_not_treated_as_an_action(make_app):
    app, _ = llm_app(make_app, [], text_only=True)
    # provider keeps answering in prose: no tool call => invalid output => escalate
    app.llm._client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={
        "choices": [{"message": {"role": "assistant", "content": "refund them all"}}]})))
    oid, run = inject_and_run(app, "payment_captured_pending", "basic")
    assert run.outcome == "escalated"


def test_provider_outage_fails_closed(make_app, monkeypatch):
    monkeypatch.setattr("relay.agent.llm_client.time.sleep", lambda s: None)
    app, _ = llm_app(make_app, [], status=503)
    oid, run = inject_and_run(app, "payment_captured_pending", "basic")
    assert run.outcome == "escalated"
    assert [t.reason_code for t in tickets(app, oid)] == ["LLM_UNAVAILABLE"]
    with app.session() as s:
        from sqlalchemy import select
        assert all(c.error for c in s.scalars(select(LLMCall)))        # failures recorded, no usage invented
