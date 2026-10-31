"""Structured audit events. Every event carries the run's correlation id."""
from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from relay.logging_utils import bounded
from relay.models import AuditEvent

log = logging.getLogger("relay.audit")


def record_event(session: Session, *, correlation_id: str, event_type: str, actor: str,
                 run_id: str | None = None, order_id: str | None = None, payload: dict | None = None) -> None:
    """Add an audit event to the caller's transaction (so it commits atomically with the change)."""
    safe = bounded(payload or {})
    session.add(AuditEvent(correlation_id=correlation_id, run_id=run_id, order_id=order_id, actor=actor,
                           event_type=event_type, payload=safe))
    log.info(event_type, extra={"fields": {"correlation_id": correlation_id, "run_id": run_id,
                                           "order_id": order_id, "actor": actor, "event": event_type}})
