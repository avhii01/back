#!/usr/bin/env bash
set -e

echo "==> Running database migrations..."
alembic upgrade head

echo "==> Seeding initial demo data..."
python -m relay.seed || true

echo "==> Starting Relay API server on port ${PORT:-8000}..."
exec uvicorn relay.main:app --host 0.0.0.0 --port "${PORT:-8000}"
