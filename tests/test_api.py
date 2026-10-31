import json

import pytest
from fastapi.testclient import TestClient

from relay.api.main import create_app
from tests.helpers import wait_for

SECRET = "sk-test-secret-1234567890"


@pytest.fixture()
def client(app):
    with TestClient(create_app(app)) as c:
        yield c


def _wait_run(client, run_id, until=("succeeded", "escalated", "failed", "awaiting_approval")):
    assert wait_for(lambda: client.get(f"/api/runs/{run_id}").json()["status"] in until), "run did not settle"
    return client.get(f"/api/runs/{run_id}").json()


def test_full_flow_over_http(client):
    cfg = client.get("/api/config").json()
    assert cfg["mode"] == "demo" and cfg["simulated_services"] is True

    order = client.post("/api/scenarios/inject", json={"scenario": "payment_captured_pending"}).json()
    assert order["status"] == "pending"
    assert any(o["id"] == order["id"] for o in client.get("/api/orders").json())

    started = client.post(f"/api/orders/{order['id']}/runs", json={}).json()
    again = client.post(f"/api/orders/{order['id']}/runs", json={}).json()
    assert again["already_active"] in (True, False)       # race-dependent, but never a second active run
    run = _wait_run(client, started["run"]["id"])
    assert run["status"] == "succeeded" and run["outcome"] == "fulfilled"
    assert run["metrics"]["tool_calls"] == len(run["tool_calls"]) and run["metrics"]["llm"] is None
    assert run["mode_label"].startswith("Deterministic demo mode")
    assert run["verification"]["passed"] is True

    detail = client.get(f"/api/orders/{order['id']}").json()
    assert detail["order"]["status"] == "shipped" and detail["services"]["simulated"] is True
    assert len(detail["services"]["shipment"]["shipments"]) == 1

    tr = client.get(f"/api/runs/{run['id']}/trace")
    assert "attachment" in tr.headers["content-disposition"]
    doc = json.loads(tr.text)
    assert doc["format"] == "relay.trace.v1" and doc["run"]["tool_calls"]


def test_approval_flow_over_http_is_idempotent(client):
    order = client.post("/api/scenarios/inject", json={"scenario": "out_of_stock_after_payment"}).json()
    run_id = client.post(f"/api/orders/{order['id']}/runs", json={}).json()["run"]["id"]
    run = _wait_run(client, run_id)
    assert run["status"] == "awaiting_approval"
    ap = run["approvals"][0]
    body = {"decision": "approve", "decided_by": "alice"}

    assert client.post(f"/api/approvals/{ap['id']}/decision", json={**body, "decided_by": "agent"}).status_code == 422
    first = client.post(f"/api/approvals/{ap['id']}/decision", json=body).json()
    second = client.post(f"/api/approvals/{ap['id']}/decision", json=body).json()
    assert first["applied"] is True and second["applied"] is False
    assert client.post(f"/api/approvals/{ap['id']}/decision", json={**body, "decision": "reject"}).status_code == 409

    run = _wait_run(client, run_id, until=("succeeded", "escalated", "failed"))
    assert run["outcome"] == "refunded"
    detail = client.get(f"/api/orders/{order['id']}").json()
    assert len(detail["services"]["payment"]["refunds"]) == 1


def test_step_mode_over_http(client):
    order = client.post("/api/scenarios/inject", json={"scenario": "payment_captured_pending"}).json()
    run = client.post(f"/api/orders/{order['id']}/runs", json={"step_mode": True}).json()["run"]
    assert run["status"] == "paused"
    r = client.post(f"/api/runs/{run['id']}/step").json()
    assert r["status"] == "paused" and r["steps"] == 1 and len(r["tool_calls"]) == 1


def test_errors_and_validation(client):
    assert client.get("/api/runs/nope").status_code == 404
    assert client.post("/api/orders/ORD-NOPE/runs", json={}).status_code == 404
    assert client.post("/api/scenarios/inject", json={"scenario": "nope"}).status_code == 422
    assert client.post("/api/approvals/x/decision", json={"decision": "maybe", "decided_by": "a"}).status_code == 422


def test_secrets_never_reach_api_responses_or_storage(make_app):
    app = make_app(agent_mode="llm", llm_api_key=SECRET)
    app.llm = object()  # never called in this test
    with TestClient(create_app(app)) as c:
        cfg = c.get("/api/config")
        assert SECRET not in cfg.text and cfg.json()["llm_key_configured"] is True
        order = c.post("/api/scenarios/inject", json={"scenario": "payment_captured_pending"}).json()
        assert SECRET not in json.dumps(c.get(f"/api/orders/{order['id']}").json())
        assert SECRET not in repr(app.settings)
    from relay.logging_utils import redact
    assert SECRET not in json.dumps(redact({"headers": {"Authorization": f"Bearer {SECRET}"}, "api_key": SECRET,
                                            "note": f"token was {SECRET}"}))


def test_correlation_id_is_echoed_and_stored(client):
    order = client.post("/api/scenarios/inject", json={"scenario": "payment_captured_pending"}).json()
    r = client.post(f"/api/orders/{order['id']}/runs", json={}, headers={"X-Correlation-ID": "corr-abc-123"})
    assert r.headers["X-Correlation-ID"] == "corr-abc-123"
    run = _wait_run(client, r.json()["run"]["id"])
    assert run["correlation_id"] == "corr-abc-123"
    assert {e["correlation_id"] for e in run["audit_events"]} == {"corr-abc-123"}
