"""Test fixtures.

* The SIMULATED services run as a real uvicorn server in a background thread (real HTTP, real
  timeouts) with their own database.
* Relay gets a fresh database per test. By default SQLite; set RELAY_TEST_DATABASE_URL to a
  Postgres URL (CI does) to run the very same suite on Postgres.
"""
from __future__ import annotations

import os
import socket
import threading
import time
import uuid

import pytest
import uvicorn
from sqlalchemy import text

from relay.clients.sim_client import SimClient
from relay.config import Settings
from relay.context import AppContext, Hooks
from relay.db import make_engine, make_session_factory
from relay.models import Base
from sim_services.app import create_sim_app


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def sim_server(tmp_path_factory):
    db = tmp_path_factory.mktemp("sim") / "sim.db"
    app = create_sim_app(f"sqlite:///{db}")
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    assert server.started, "simulated services failed to start"
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture()
def sleeps():
    return []


@pytest.fixture()
def make_app(sim_server, tmp_path, sleeps):
    """Factory so tests can tweak settings / hooks. Returns an AppContext."""
    created: list[AppContext] = []

    def factory(**settings_overrides) -> AppContext:
        url = os.environ.get("RELAY_TEST_DATABASE_URL") or f"sqlite:///{tmp_path / 'relay.db'}"
        engine = make_engine(url)
        if url.startswith("postgres"):
            with engine.begin() as c:
                c.execute(text("DROP SCHEMA public CASCADE"))
                c.execute(text("CREATE SCHEMA public"))
        Base.metadata.create_all(engine)
        base = dict(database_url=url, sim_base_url=sim_server, sim_timeout_s=0.5, agent_mode="demo",
                    tool_max_attempts=3, backoff_base_s=0.1, backoff_max_s=1.0, lease_seconds=30,
                    auto_resume_on_startup=False, job_worker_enabled=False, job_poll_seconds=0.05,
                    job_lease_seconds=30)
        base.update(settings_overrides)
        settings = Settings(**base)
        ctx = AppContext(settings=settings, engine=engine, session_factory=make_session_factory(engine),
                         sim=SimClient(sim_server, settings.sim_timeout_s), sleep=sleeps.append, hooks=Hooks())
        created.append(ctx)
        return ctx

    yield factory
    for c in created:
        # Drain background runner tasks before the next test drops/recreates the schema.
        c.pool.shutdown(wait=True, cancel_futures=True)
        c.close()


@pytest.fixture()
def app(make_app):
    return make_app()


def new_order_id() -> str:
    return f"ORD-{uuid.uuid4().hex[:6].upper()}"
