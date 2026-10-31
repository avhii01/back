"""Tool executor: the single choke point through which every tool call passes.

Order of operations for a call:
  1. validate input against the typed schema           -> INVALID_INPUT, nothing executed
  2. (mutations) claim/inspect the idempotency record   -> replay a prior success, never re-run it
  3. charge the tool-call budget                        -> BudgetExceeded
  4. attempt (with a per-call timeout)
       transient error  -> bounded retries with exponential backoff
       ambiguous error  -> RECONCILE first (did it already happen?) and only then retry
       permanent error  -> stop
  5. persist the outcome + timeline row + audit event

Guarantees (and their limits) are documented in docs/ARCHITECTURE.md.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from relay.audit import record_event
from relay.context import AppContext
from relay.errors import AmbiguousToolError, BudgetExceeded, PermanentToolError, ToolError, TransientToolError
from relay.logging_utils import bounded
from relay.models import IdempotencyRecord, ToolCall, utcnow
from relay.tools.base import ToolContext, ToolSpec
from relay.tools.registry import REGISTRY


@dataclass
class ToolResult:
    ok: bool
    tool: str
    output: dict | None = None
    error: dict | None = None
    attempts: int = 0
    replayed: bool = False
    reconciled: bool = False
    idempotency_key: str | None = None
    meta: dict = field(default_factory=dict)


class ToolExecutor:
    def __init__(self, app: AppContext, run_id: str, order_id: str, correlation_id: str):
        self.app = app
        self.run_id = run_id
        self.order_id = order_id
        self.correlation_id = correlation_id

    # ---------------------------------------------------------------- public
    def call(self, name: str, raw_args: dict, *, node: str | None = None, exempt_budget: bool = False) -> ToolResult:
        spec = REGISTRY.get(name)
        if spec is None:
            err = PermanentToolError(f"unknown tool '{name}'", code="UNKNOWN_TOOL")
            self._log_call(name, False, node, 0, None, raw_args, None, err.to_dict(), "error", 0)
            return ToolResult(False, name, error=err.to_dict())
        try:
            args = spec.input_model.model_validate(raw_args)
        except ValidationError as e:
            err = PermanentToolError("input validation failed", code="INVALID_INPUT",
                                     details={"errors": [{"loc": list(x["loc"]), "msg": x["msg"]} for x in e.errors()][:5]})
            self._log_call(name, spec.mutating, node, 0, None, raw_args, None, err.to_dict(), "error", 0)
            return ToolResult(False, name, error=err.to_dict())

        key = spec.idempotency_key(args) if spec.idempotency_key else None
        if key and spec.mutating:
            early = self._begin_idempotency(spec, args, key, node)
            if early is not None:
                return early

        settings = self.app.settings
        max_attempts = settings.tool_max_attempts
        attempt = self._attempts_so_far(key) if key else 0
        last_err: ToolError | None = None
        while True:
            if attempt >= max_attempts:
                err = (last_err or TransientToolError("retries exhausted")).to_dict()
                err.update(code="RETRIES_EXHAUSTED", message=f"gave up after {attempt} attempts: {err['message']}")
                self._finish_failed(key, err)
                self._audit("retries_exhausted", {"tool": name, "attempts": attempt, "key": key})
                return ToolResult(False, name, error=err, attempts=attempt, idempotency_key=key)
            self._charge_budget(exempt_budget)
            attempt += 1
            if key:
                self._bump_attempts(key)
            started = time.perf_counter()
            try:
                output = self._invoke(spec, args, key, attempt, node, raw_args, started)
                return ToolResult(True, name, output=output, attempts=attempt, idempotency_key=key)
            except AmbiguousToolError as e:
                last_err = e
                self._log_call(name, spec.mutating, node, attempt, key, args.model_dump(), None, e.to_dict(),
                               "ambiguous", _ms(started))
                if spec.mutating and spec.reconcile:
                    found = self._reconcile(spec, args, key, node, attempt)
                    if found is not None:
                        return ToolResult(True, name, output=found, attempts=attempt, reconciled=True,
                                          idempotency_key=key)
            except TransientToolError as e:
                last_err = e
                self._log_call(name, spec.mutating, node, attempt, key, args.model_dump(), None, e.to_dict(),
                               "error", _ms(started))
            except ToolError as e:  # permanent
                self._log_call(name, spec.mutating, node, attempt, key, args.model_dump(), None, e.to_dict(),
                               "error", _ms(started))
                self._finish_failed(key, e.to_dict())
                return ToolResult(False, name, error=e.to_dict(), attempts=attempt, idempotency_key=key)
            if attempt < max_attempts:
                delay = min(settings.backoff_base_s * (2 ** (attempt - 1)), settings.backoff_max_s)
                self._audit("retry_scheduled", {"tool": name, "attempt": attempt, "delay_s": delay,
                                                "after": last_err.code if last_err else None})
                self.app.sleep(delay)

    # ---------------------------------------------------------------- invocation
    def _invoke(self, spec: ToolSpec, args, key, attempt, node, raw_args, started) -> dict:
        """One attempt. Local mutations commit effect + idempotency record + timeline atomically."""
        with self.app.session() as s:
            ctx = ToolContext(self.app, s, self.run_id, self.correlation_id, key, spec.timeout_s)
            try:
                out = spec.fn(ctx, args)
            except BaseException:
                s.rollback()
                raise
            out_dict = out.model_dump(mode="json")
            if spec.mutating and spec.remote and self.app.hooks.after_remote_effect:
                s.rollback()  # nothing local to keep; crash happens "after the side effect, before ack"
                self.app.hooks.after_remote_effect(spec.name)
            if spec.mutating and key:
                self._mark_succeeded(s, key, out_dict)
            self._log_call(spec.name, spec.mutating, node, attempt, key, args.model_dump(), out_dict, None, "ok",
                           _ms(started), session=s)
            s.commit()
            return out_dict

    def _reconcile(self, spec: ToolSpec, args, key, node, attempt) -> dict | None:
        """After an ambiguous failure: look up whether the effect already happened."""
        started = time.perf_counter()
        try:
            self._charge_budget(False)
            with self.app.session() as s:
                ctx = ToolContext(self.app, s, self.run_id, self.correlation_id, key, spec.timeout_s)
                found = spec.reconcile(ctx, args)
        except BudgetExceeded:
            raise
        except ToolError as e:
            self._log_call(f"{spec.name}.reconcile", False, node, attempt, key, args.model_dump(), None, e.to_dict(),
                           "error", _ms(started), purpose="reconcile")
            return None
        out = found.model_dump(mode="json") if found is not None else None
        self._log_call(f"{spec.name}.reconcile", False, node, attempt, key, args.model_dump(),
                       out or {"found": False}, None, "ok", _ms(started), purpose="reconcile")
        if out is not None and key:
            with self.app.session() as s:
                self._mark_succeeded(s, key, out)
                s.commit()
            self._audit("reconciled_existing_effect", {"tool": spec.name, "key": key, "result": out})
        return out

    # ---------------------------------------------------------------- idempotency records
    def _begin_idempotency(self, spec: ToolSpec, args, key: str, node: str | None) -> ToolResult | None:
        with self.app.session() as s:
            rec = s.get(IdempotencyRecord, key)
            if rec is None:
                try:
                    s.add(IdempotencyRecord(key=key, tool=spec.name, order_id=args.order_id, status="in_progress",
                                            attempts=0, run_id=self.run_id))
                    s.commit()
                    return None
                except IntegrityError:  # lost an insert race: the DB constraint decided
                    s.rollback()
                    rec = s.get(IdempotencyRecord, key)
            if rec.status == "succeeded":
                ctx = ToolContext(self.app, s, self.run_id, self.correlation_id, key, spec.timeout_s)
                if spec.replay_valid is None or spec.replay_valid(ctx, args):
                    self._log_call(spec.name, True, node, 0, key, args.model_dump(), rec.result, None, "replayed", 0,
                                   session=s)
                    record_event(s, correlation_id=self.correlation_id, run_id=self.run_id, order_id=self.order_id,
                                 actor="tool", event_type="idempotent_replay", payload={"tool": spec.name, "key": key})
                    s.commit()
                    return ToolResult(True, spec.name, output=rec.result, replayed=True, idempotency_key=key)
                rec.status, rec.attempts = "in_progress", 0  # stale success: state moved on, execute again
            elif rec.status == "failed":
                same_run_exhausted = rec.run_id == self.run_id and (rec.error or {}).get("code") == "RETRIES_EXHAUSTED"
                if same_run_exhausted:
                    return ToolResult(False, spec.name, error=rec.error, attempts=rec.attempts, idempotency_key=key)
                rec.status, rec.attempts = "in_progress", 0  # new run / non-retry failure: allow a fresh attempt
            rec.run_id = self.run_id
            was_in_progress = rec.status == "in_progress" and rec.attempts > 0
            s.commit()
        if was_in_progress and spec.reconcile:
            # a previous worker may have died mid-call: check what actually happened before retrying
            found = self._reconcile(spec, args, key, node, rec.attempts)
            if found is not None:
                return ToolResult(True, spec.name, output=found, reconciled=True, idempotency_key=key)
        return None

    def _attempts_so_far(self, key: str) -> int:
        with self.app.session() as s:
            rec = s.get(IdempotencyRecord, key)
            return rec.attempts if rec else 0

    def _bump_attempts(self, key: str) -> None:
        with self.app.session() as s:
            rec = s.get(IdempotencyRecord, key)
            if rec is not None:
                rec.attempts += 1
                s.commit()

    def _mark_succeeded(self, s: Session, key: str, out: dict) -> None:
        rec = s.get(IdempotencyRecord, key)
        if rec is not None:
            rec.status, rec.result, rec.error = "succeeded", out, None

    def _finish_failed(self, key: str | None, err: dict) -> None:
        if not key:
            return
        with self.app.session() as s:
            rec = s.get(IdempotencyRecord, key)
            if rec is not None and rec.status != "succeeded":
                rec.status, rec.error = "failed", err
                s.commit()

    # ---------------------------------------------------------------- budget / logging
    def _charge_budget(self, exempt: bool) -> None:
        if exempt:
            return
        with self.app.session() as s:
            n = s.scalar(select(func.count()).select_from(ToolCall).where(ToolCall.run_id == self.run_id)) or 0
        if n >= self.app.settings.max_tool_calls:
            raise BudgetExceeded("tool_calls", self.app.settings.max_tool_calls)

    def _audit(self, event_type: str, payload: dict) -> None:
        with self.app.session() as s:
            record_event(s, correlation_id=self.correlation_id, run_id=self.run_id, order_id=self.order_id,
                         actor="tool", event_type=event_type, payload=payload)
            s.commit()

    def _log_call(self, tool, mutating, node, attempt, key, args, result, error, status, duration_ms,
                  *, session: Session | None = None, purpose: str | None = None) -> None:
        def write(s: Session) -> None:
            seq = (s.scalar(select(func.max(ToolCall.seq)).where(ToolCall.run_id == self.run_id)) or 0) + 1
            s.add(ToolCall(run_id=self.run_id, seq=seq, node=node, tool=tool, mutating=mutating, purpose=purpose,
                           attempt=max(attempt, 1) if attempt else 0, idempotency_key=key, status=status,
                           args=bounded(args), result=bounded(result) if result is not None else None,
                           error=bounded(error) if error is not None else None, started_at=utcnow(),
                           duration_ms=duration_ms))
            record_event(s, correlation_id=self.correlation_id, run_id=self.run_id, order_id=self.order_id,
                         actor="tool", event_type="tool_call",
                         payload={"tool": tool, "status": status, "attempt": attempt, "mutating": mutating,
                                  "error_code": (error or {}).get("code")})

        if session is not None:
            write(session)
        else:
            with self.app.session() as s:
                write(s)
                s.commit()


def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
