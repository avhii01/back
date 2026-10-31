"""Durable ticket-dispatch worker.

The chat request never runs recovery. It commits a ticket + job; this module claims a
job with a lease and calls the existing runner. If the process dies mid-run, the run's
own lease and the job's lease both expire; startup recovery resumes them.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from datetime import timedelta

from sqlalchemy import select, update

from relay.agent import runner
from relay.context import AppContext
from relay.models import BackgroundJob, Run, SupportTicket, utcnow
from relay.support.tickets import sync_ticket

log = logging.getLogger("relay.support.jobs")


class RetryJob(Exception):
    pass


def _claim(app: AppContext, worker_id: str) -> str | None:
    now = utcnow()
    with app.session() as s:
        candidates = list(s.scalars(select(BackgroundJob.id).where(
            BackgroundJob.status.in_(("queued", "running")),
            BackgroundJob.attempts < BackgroundJob.max_attempts,
            (BackgroundJob.status == "queued")
            | ((BackgroundJob.status == "running") & ((BackgroundJob.lease_expires_at.is_(None))
                                                      | (BackgroundJob.lease_expires_at < now))),
        ).order_by(BackgroundJob.created_at).limit(5)))
    for job_id in candidates:
        with app.session() as s:
            res = s.execute(update(BackgroundJob).where(
                BackgroundJob.id == job_id,
                BackgroundJob.status.in_(("queued", "running")),
                BackgroundJob.attempts < BackgroundJob.max_attempts,
                (BackgroundJob.status == "queued")
                | ((BackgroundJob.status == "running") & ((BackgroundJob.lease_expires_at.is_(None))
                                                          | (BackgroundJob.lease_expires_at < now))),
            ).values(status="running", lease_owner=worker_id,
                     lease_expires_at=now + timedelta(seconds=app.settings.job_lease_seconds),
                     attempts=BackgroundJob.attempts + 1, updated_at=now))
            s.commit()
            if res.rowcount == 1:
                return job_id
    return None


def process_once(app: AppContext, worker_id: str | None = None) -> str | None:
    """Claim and run one job. Returns the job id, or None when the queue is empty."""
    worker_id = worker_id or f"job_{uuid.uuid4().hex[:8]}"
    job_id = _claim(app, worker_id)
    if job_id is None:
        return None
    try:
        _run(app, job_id)
        return job_id
    except RetryJob as e:
        _retry(app, job_id, str(e))
        return job_id
    except Exception as e:  # unexpected worker bug: never leave the job running forever
        log.exception("ticket job %s crashed", job_id)
        _retry(app, job_id, f"{type(e).__name__}: {str(e)[:300]}")
        return job_id


def _run(app: AppContext, job_id: str) -> None:
    with app.session() as s:
        job = s.get(BackgroundJob, job_id)
        if job is None:
            return
        ticket = s.get(SupportTicket, job.ticket_id)
        if ticket is None:
            job.status = "failed"
            job.last_error = "ticket no longer exists"
            job.updated_at = utcnow()
            s.commit()
            return
        if ticket.status == "resolved":
            job.status = "done"
            job.lease_owner = None
            job.lease_expires_at = None
            job.updated_at = utcnow()
            s.commit()
            return
        order_id = ticket.order_id
        ticket_id = ticket.id
        existing_run_id = job.run_id

    run = None
    if existing_run_id:
        with app.session() as s:
            run = s.get(Run, existing_run_id)
    if run is None:
        # Reuse an active run for the order (unique index) instead of creating a duplicate.
        run, _ = runner.create_run(app, order_id, correlation_id=f"ticket:{ticket_id}")
    with app.session() as s:
        ticket = s.get(SupportTicket, ticket_id)
        ticket.run_id = run.id
        if ticket.status == "new":
            ticket.status = "investigating"
        ticket.updated_at = utcnow()
        job = s.get(BackgroundJob, job_id)
        job.run_id = run.id
        job.updated_at = utcnow()
        s.commit()

    if run.status in ("succeeded", "failed", "escalated"):
        # A retried job must never start a second recovery run for the same ticket.
        sync_ticket(app, ticket_id)
        _finish(app, job_id)
        return
    status = runner.execute_run(app, run.id)
    if status in ("not_claimed", "lease_lost"):
        raise RetryJob(f"run is owned by another worker ({status})")
    sync_ticket(app, ticket_id)
    _finish(app, job_id)


def _finish(app: AppContext, job_id: str) -> None:
    with app.session() as s:
        job = s.get(BackgroundJob, job_id)
        if job is not None:
            job.status = "done"
            job.lease_owner = None
            job.lease_expires_at = None
            job.updated_at = utcnow()
            s.commit()


def _retry(app: AppContext, job_id: str, error: str) -> None:
    with app.session() as s:
        job = s.get(BackgroundJob, job_id)
        if job is None:
            return
        job.last_error = error[:500]
        job.lease_owner = None
        job.lease_expires_at = None
        job.status = "queued" if job.attempts < job.max_attempts else "failed"
        job.updated_at = utcnow()
        if job.status == "failed":
            ticket = s.get(SupportTicket, job.ticket_id)
            if ticket is not None and ticket.status not in ("resolved", "escalated"):
                ticket.status = "escalated"
                ticket.resolution_summary = "Dispatch failed; a support specialist is reviewing the case."
                ticket.updated_at = utcnow()
        s.commit()


def recover_interrupted(app: AppContext) -> list[str]:
    """Requeue jobs whose worker lease expired (called on startup and periodically)."""
    now = utcnow()
    with app.session() as s:
        ids = list(s.scalars(select(BackgroundJob.id).where(
            BackgroundJob.status == "running",
            (BackgroundJob.lease_expires_at.is_(None)) | (BackgroundJob.lease_expires_at < now))))
        if ids:
            s.execute(update(BackgroundJob).where(BackgroundJob.id.in_(ids))
                      .values(status="queued", lease_owner=None, lease_expires_at=None, updated_at=now))
            s.commit()
    return ids


def worker_loop(app: AppContext, stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            recover_interrupted(app)
            worked = process_once(app)
        except Exception:  # keep the worker alive; the job row retains the failure
            log.exception("ticket job worker iteration failed")
            worked = None
        if worked is None:
            stop.wait(app.settings.job_poll_seconds)


def start_worker(app: AppContext) -> tuple[threading.Thread, threading.Event]:
    stop = threading.Event()
    thread = threading.Thread(target=worker_loop, args=(app, stop), daemon=True, name="relay-ticket-worker")
    thread.start()
    return thread, stop


def stop_worker(thread: threading.Thread, stop: threading.Event) -> None:
    stop.set()
    thread.join(timeout=2)


def wait_for_idle(app: AppContext, timeout: float = 15.0) -> bool:
    """Test/demo helper: process queued work until none remains or the timeout expires."""
    end = time.time() + timeout
    while time.time() < end:
        if process_once(app) is None:
            return True
    return False
