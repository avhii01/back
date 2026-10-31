from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel
from sqlalchemy.orm import Session

from relay.context import AppContext


@dataclass
class ToolContext:
    app: AppContext
    session: Session
    run_id: str | None
    correlation_id: str
    idempotency_key: str | None = None
    timeout_s: float | None = None


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    fn: Callable[[ToolContext, BaseModel], BaseModel]
    mutating: bool = False
    remote: bool = False  # True: effect happens in an independent service (no shared transaction)
    timeout_s: float | None = None
    idempotency_key: Callable[[BaseModel], str] | None = None
    # remote mutations: "did it already happen?" lookup used after ambiguous timeouts / crashes
    reconcile: Callable[[ToolContext, BaseModel], BaseModel | None] | None = None
    # evidence key refreshed by a read tool / invalidated by a mutation (see agent/facts.py)
    evidence_key: str | None = None
    invalidates: tuple[str, ...] = ()
    # local mutations whose stored result may no longer describe reality
    replay_valid: Callable[[ToolContext, BaseModel], bool] | None = None
