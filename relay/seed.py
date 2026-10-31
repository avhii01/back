"""Seed demo data: one order per failure scenario (idempotent - does nothing if orders exist).

    python -m relay.seed
"""
from __future__ import annotations

import logging
import sys
import time

from sqlalchemy import func, select

from relay import scenarios
from relay.config import Settings
from relay.context import build_context
from relay.knowledge.ingest import ingest
from relay.logging_utils import configure_logging
from relay.models import Order, TeamMember
from relay.support.auth import upsert_customer
from relay.support.demo import DEMO_CUSTOMERS, SCENARIO_OWNER

log = logging.getLogger("relay.seed")

DEMO = [("payment_captured_pending", "basic"), ("inventory_response_lost", "basic"),
        ("shipping_unavailable", "transient_2"), ("out_of_stock_after_payment", "basic"),
        ("conflicting_payment_evidence", "amount_mismatch")]

# Demo ownership: Alice can walk the main payment -> ticket -> recovery story; the
# other seeded customers own the remaining failure types.
OWNER = SCENARIO_OWNER

DEFAULT_SEEDED_TEAM = [
    {
        "id": "ADM-9041",
        "name": "Admin Operator",
        "email": "admin.operator@relay.internal",
        "role": "Operations Admin",
        "department": "Customer Operations & Reliability",
        "status": "active",
        "active_tickets_count": 4,
    },
    {
        "id": "SPT-1002",
        "name": "Sarah Jenkins",
        "email": "sarah.jenkins@relay.internal",
        "role": "Senior Specialist",
        "department": "Payments & Billing",
        "status": "active",
        "active_tickets_count": 2,
    },
    {
        "id": "SPT-1003",
        "name": "Michael Torres",
        "email": "michael.torres@relay.internal",
        "role": "Support Specialist",
        "department": "Customer Operations",
        "status": "active",
        "active_tickets_count": 3,
    },
    {
        "id": "SPT-1004",
        "name": "Elena Rostova",
        "email": "elena.rostova@relay.internal",
        "role": "Logistics Coordinator",
        "department": "Shipping & Logistics",
        "status": "active",
        "active_tickets_count": 1,
    },
]


def seed_team(app) -> None:
    try:
        with app.session() as s:
            for member in DEFAULT_SEEDED_TEAM:
                existing = s.scalar(select(TeamMember).where(TeamMember.id == member["id"]))
                if not existing:
                    s.add(TeamMember(**member))
            s.commit()
            log.info("seeded support team members into database")
    except Exception:
        log.exception("failed to seed team members")


def seed_customers(app) -> None:
    for customer in DEMO_CUSTOMERS:
        upsert_customer(app, customer_id=customer["id"], email=customer["email"], name=customer["name"],
                        password=customer["password"])
    with app.session() as s:
        orders = list(s.scalars(select(Order).where(Order.customer_id.is_(None))))
        for order in orders:
            scenario = (order.scenario_label or "").split("/", 1)[0]
            owner = OWNER.get(scenario)
            if owner:
                order.customer_id = owner
        s.commit()


def seed(wait_s: float = 15.0) -> int:
    configure_logging()
    app = build_context(Settings.from_env())
    try:
        seed_customers(app)
        seed_team(app)
        try:
            stats = ingest(app)
            log.info("knowledge ingested: %s", {k: v for k, v in stats.items() if k != "embedding_error"})
        except Exception:
            log.exception("knowledge ingestion failed; chat will use whatever is already stored")
        with app.session() as s:
            if (s.scalar(select(func.count()).select_from(Order)) or 0) > 0:
                log.info("orders already present; skipping scenario seed")
                return 0
        deadline = time.time() + wait_s
        while True:  # the simulated services may still be starting
            try:
                app.sim.get_stock("health-probe")
                break
            except Exception:
                if time.time() > deadline:
                    log.warning("simulated services unreachable; scenario seed skipped (app will start normally)")
                    return 0
                time.sleep(1)
        for scenario, variant in DEMO:
            order = scenarios.inject(app, scenario, variant)
            log.info("seeded %s (%s/%s)", order.id, scenario, variant)
        seed_customers(app)
        seed_team(app)
        return 0
    finally:
        app.close()


if __name__ == "__main__":
    sys.exit(seed())
