#!/usr/bin/env bash
set -e

echo "==> Running database migrations..."
alembic upgrade head

echo "==> Starting local simulation services on port 8001..."
uvicorn sim_services.main:app --host 0.0.0.0 --port 8001 &

echo "==> Waiting for simulation services to be ready..."
for i in $(seq 1 10); do
  if python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')" >/dev/null 2>&1; then
    echo "==> Simulation services ready on port 8001."
    break
  fi
  sleep 1
done

echo "==> Seeding initial demo data..."
python -m relay.seed || true

echo "==> Starting Relay API server on port ${PORT:-8000}..."
exec uvicorn relay.main:app --host 0.0.0.0 --port "${PORT:-8000}"
