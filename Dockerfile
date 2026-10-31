FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml ./
COPY relay ./relay
COPY sim_services ./sim_services
# The RAG extra is optional at runtime; the image includes it so the Compose
# `rag` profile (Chroma) works without rebuilding a custom image.
RUN pip install --no-cache-dir ".[rag]"

COPY alembic.ini ./
COPY migrations ./migrations

RUN useradd --create-home relay
USER relay
EXPOSE 8000 8001

# Default command = the Relay API. The compose file overrides it for the simulated services.
CMD ["sh", "-c", "alembic upgrade head && python -m relay.seed && uvicorn relay.main:app --host 0.0.0.0 --port 8000"]
