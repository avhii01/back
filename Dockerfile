FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml ./
COPY requirements.txt ./
COPY relay ./relay
COPY sim_services ./sim_services
RUN pip install --no-cache-dir ".[rag]"

COPY alembic.ini ./
COPY migrations ./migrations
COPY users.xlsx users.csv ./
COPY start.sh ./
RUN chmod +x start.sh

RUN useradd --create-home relay && chown -R relay:relay /app
USER relay
EXPOSE 8000 8001

CMD ["bash", "start.sh"]
