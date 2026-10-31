"""Application context: the explicit dependency container (no hidden globals).

Tests build their own context with fakes (no-op sleep, crash hooks, mock LLM transport).
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from relay.clients.sim_client import SimClient
from relay.config import Settings
from relay.db import make_engine, make_session_factory


@dataclass
class Hooks:
    """Failure-boundary test hooks. All None in production."""

    # called after a REMOTE mutation returned successfully but BEFORE its result is persisted
    after_remote_effect: Callable[[str], None] | None = None
    # called at the start of every graph node
    before_node: Callable[[str], None] | None = None


@dataclass
class AppContext:
    settings: Settings
    engine: Engine
    session_factory: sessionmaker[Session]
    sim: SimClient
    llm: object | None = None  # relay.agent.llm_client.LLMClient (kept loose to avoid import cycles)
    sleep: Callable[[float], None] = time.sleep
    hooks: Hooks = field(default_factory=Hooks)
    pool: ThreadPoolExecutor = field(default_factory=lambda: ThreadPoolExecutor(max_workers=4, thread_name_prefix="relay-run"))

    def session(self) -> Session:
        return self.session_factory()

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.sim.close()
        self.engine.dispose()


def build_context(settings: Settings, **overrides) -> AppContext:
    from relay.agent.llm_client import LLMClient  # local import: avoids cycle

    engine = make_engine(settings.database_url)
    sim = overrides.pop("sim", None) or SimClient(settings.sim_base_url, settings.sim_timeout_s)
    llm = overrides.pop("llm", None)
    if llm is None and settings.mode == "llm":
        llm = LLMClient(settings)
    return AppContext(settings=settings, engine=engine, session_factory=make_session_factory(engine),
                      sim=sim, llm=llm, **overrides)
