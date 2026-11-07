# Relay — Backend & Autonomous AI Recovery Engine

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.14-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Neon%20Serverless-4169E1.svg?logo=postgresql&logoColor=white)](https://neon.tech)
[![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-2.0-red.svg)](https://www.sqlalchemy.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com)
[![Pytest](https://img.shields.io/badge/Pytest-100%2B%20Tests-brightgreen.svg)]()

> **Live Backend API & Interactive Swagger UI:** [https://backend-pvco.onrender.com/docs](https://backend-pvco.onrender.com/docs)  
> **Frontend Repository:** [https://github.com/avhii01/front](https://github.com/avhii01/front)

---

## 📌 Problem Statement

In distributed e-commerce and logistics systems, fulfillment orders frequently fail mid-transaction due to transient network drops, inventory race conditions, or third-party carrier outages.

Traditional automated recovery mechanisms either blindly retry (causing duplicate charges or duplicate shipments) or halt entirely, flooding human operators with manual tickets. Meanwhile, directly hooking up LLM agents to mutating APIs risks hallucinations, unauthorized refunds, and security breaches.

---

## 💡 The Solution: Dual-Engine Recovery Architecture

Relay Backend solves this via a **hybrid dual-engine architecture**:
1. **Autonomous Reasoning Agent (LLM / ReAct Planner):** Observes microservice evidence, diagnoses failure causes, and proposes corrective action sequences.
2. **Deterministic Policy Engine:** An infallible gatekeeper (`P-ALLOW`, `P-REFUND-MAX`, `P-HOLD-SLA`, `P-RESOLVE-PERM`) that intercepts and verifies every tool proposal before execution.
3. **Idempotency Guarantees:** Every mutating tool execution generates a deterministic `idempotency_key` (e.g. `ship:ORD-ABC123`), ensuring safe retries without double-execution.
4. **Financial Safeguards:** Any refund exceeding **\$50.00 USD** automatically pauses the workflow and mandates human operator review.
5. **Grounded RAG Support Engine:** Customer support queries query chunked vector embeddings and return answers grounded in verified company policy with section citations.
6. **Strict Role-Based Access Control (RBAC):**
   * Passwords hashed with **PBKDF2-HMAC-SHA256 (120,000 iterations)**.
   * `GET /api/team` enforces field-level security: only administrators can view passwords; non-admins receive redacted values.
   * Ticket reassignment is locked to administrators only.

---

## 🏗️ Architecture & Component Overview

```
backend/
├── relay/
│   ├── agent/             # ReAct loop, LLM planner, rule-based planner, runner
│   ├── api/               # FastAPI route controllers (main, support, customer, serializers)
│   ├── knowledge/         # Vector embeddings, RAG ingestion, document chunking
│   ├── support/           # Background job dispatch, customer auth, ticket lifecycle
│   ├── config.py          # Pydantic v2 application settings
│   ├── models.py          # SQLAlchemy 2.0 ORM database schema definitions
│   └── seed.py            # Initial seed data (team specialists, demo customers, orders)
├── sim_services/          # Simulated microservices (Stripe payments, warehouse stock, carrier shipping)
├── migrations/            # Alembic schema migration versions (Neon PostgreSQL)
└── tests/                 # 100+ Pytest unit, integration, and scenario tests
```

---

## 🔐 Authentication & Data Protection

To protect system integrity and prevent unauthorized access:
* **PBKDF2 Hashing:** All operator and customer credentials are cryptographically secured using salted PBKDF2-HMAC-SHA256 with 120,000 hash iterations. Plaintext passwords are never stored in databases or exported to disk.
* **Role-Based Redaction:** Non-admin operators querying the team roster receive strictly redacted responses (`password: null`, `password_hash: null`). Only authenticated administrators can inspect and manage team credentials.
* **Customer Data Isolation:** Customer tokens resolve strictly to authenticated customer rows. Cross-tenant order inspections or modifications are blocked server-side.
* **Credential Seeding:** Initial demo accounts are initialized securely via `.env` configuration and the internal database seeder (`relay.seed`). Plaintext credentials are never committed to public repositories.

---

## 🚀 Local Development

### 1. Environment & Dependencies
```bash
python -m venv .venv
# Activate:
# Windows: .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

### 2. Database Migrations & Seeding
```bash
# Apply Alembic migrations
alembic upgrade head

# Seed initial database rows
python -m relay.seed
```

### 3. Run Development Server
```bash
uvicorn relay.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger Documentation will be available at: `http://localhost:8000/docs`

---

## 🧪 Testing

Run the automated test suite:
```bash
pytest backend
```

---

## 🐳 Docker Container Deployment

```bash
docker build -t relay-backend .
docker run -p 8000:8000 -e DATABASE_URL="your-postgres-url" relay-backend
```

---

## 📄 License
MIT License
