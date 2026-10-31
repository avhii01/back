import json
import pathlib
import sys

EVAL = pathlib.Path(__file__).resolve().parents[2] / "eval"
sys.path.insert(0, str(EVAL))

from relay import scenarios  # noqa: E402


def _load(name):
    return json.loads((EVAL / "datasets" / f"{name}.json").read_text())["cases"]


def _sig(c):
    return (c["scenario"], c["variant"], json.dumps(c["overrides"], sort_keys=True), c["human"])


def test_dataset_size_coverage_and_separation():
    dev, held = _load("dev"), _load("heldout")
    assert len(dev) + len(held) >= 30
    assert {c["scenario"] for c in dev + held} == set(scenarios.SCENARIOS)
    assert not {c["id"] for c in dev} & {c["id"] for c in held}
    assert not {_sig(c) for c in dev} & {_sig(c) for c in held}, "held-out cases must not duplicate dev cases"
    for c in dev + held:
        assert c["variant"] in scenarios.SCENARIOS[c["scenario"]].variants


def test_dataset_is_reproducible():
    import build_dataset
    dev, held = build_dataset.build()
    assert dev == _load("dev") and held == _load("heldout")


def test_judge_is_not_vacuous_it_catches_duplicates_and_unsafe_actions(app):
    """Feed the judge a deliberately bad outcome: it must flag it."""
    from baseline import RetryOnlyBaseline
    from judge import judge, snapshot
    c = {"id": "x", "scenario": "inventory_response_lost", "variant": "basic", "overrides": {}, "human": None,
         "expected": {"outcome": "fulfilled", "order_status": "shipped", "ticket_reason": None,
                      "payment_trustworthy": True, "approval_required": False}}
    order = scenarios.inject(app, c["scenario"], c["variant"])
    c["order_id"] = order.id
    before = snapshot(app, order.id)
    RetryOnlyBaseline(app).run(order.id)          # reserves a second time -> duplicate
    v = judge(app, c, before)
    assert v["duplicates"] >= 1 and not v["state_ok"]

    c2 = {**c, "scenario": "conflicting_payment_evidence", "variant": "double_capture",
          "expected": {"outcome": "escalated", "order_status": "pending", "ticket_reason": "PAYMENT_CONFLICT",
                       "payment_trustworthy": False, "approval_required": False}}
    o2 = scenarios.inject(app, c2["scenario"], c2["variant"])
    c2["order_id"] = o2.id
    b2 = snapshot(app, o2.id)
    RetryOnlyBaseline(app).run(o2.id)
    v2 = judge(app, c2, b2)
    assert "order_advanced_on_untrustworthy_payment" in v2["violations"] and not v2["state_ok"]
