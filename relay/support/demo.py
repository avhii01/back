"""Clearly-labelled demo customers. Passwords are for the local demo only."""

DEMO_CUSTOMERS = [
    {"id": "cus_alice", "email": "alice@example.test", "name": "Alice Demo", "password": "demo-alice"},
    {"id": "cus_bob", "email": "bob@example.test", "name": "Bob Demo", "password": "demo-bob"},
    {"id": "cus_carol", "email": "carol@example.test", "name": "Carol Demo", "password": "demo-carol"},
]

# Seeded / newly injected scenarios are owned by a demo customer so the support chat
# can demonstrate server-side ownership checks. The recovery agent never reads this.
SCENARIO_OWNER = {
    "payment_captured_pending": "cus_alice",
    "out_of_stock_after_payment": "cus_alice",
    "inventory_response_lost": "cus_bob",
    "conflicting_payment_evidence": "cus_bob",
    "shipping_unavailable": "cus_carol",
}
