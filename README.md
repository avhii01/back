# Relay - AI Order Recovery Agent (Backend & Services)

Production FastAPI orchestrator, deterministic policy engine, and simulated external microservices (payments, inventory, and logistics carrier dispatch).

## Architecture & Services
- **Relay Core API Engine** (FastAPI): State machine, run graph executors, ticket queues, admin auth.
- **Deterministic Policy Engine**: Evaluates `P-ALLOW`, `P-REFUND-MAX`, `P-HOLD-SLA` without language model hallucination.
- **Simulated Gateways** (`sim_services`): Simulated payment capture & refund, warehouse stock reservation, carrier waybills.
- **Relational Persistence**: PostgreSQL 16 schema with Alembic migrations.

## Local Development
```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run migrations & seed data
alembic upgrade head
python -m relay.seed

# Start backend server
uvicorn relay.main:app --host 0.0.0.0 --port 8000
```

## Render Deployment
- **Environment**: Python 3
- **Build Command**: `pip install -r requirements.txt && pip install -e .`
- **Start Command**: `bash start.sh`
- **Required Environment Variables**:
  - `DATABASE_URL`: Connection string to PostgreSQL
  - `CORS_ORIGINS`: `*` (or your frontend domain)
  - `DEMO_CUSTOMER_AUTH`: `true`
  - `JOB_WORKER_ENABLED`: `true`
