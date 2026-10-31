"""LangGraph workflow:  observe -> investigate -> propose -> validate -> execute -> (loop) -> verify.

Durability model
  * After EVERY node, the full RunState is written to `runs.state` with a compare-and-set on the
    lease owner. On restart the graph re-enters at `state["next_node"]`.
  * Nodes are written to be RE-ENTRANT: if the process dies after a node's side effect but before
    its state is saved, the node simply runs again - and the tool executor's idempotency records /
    reconciliation make that safe.
  * LangGraph provides the control-flow graph; durability is Relay's own table (queryable,
    identical on SQLite and Postgres). A LangGraph checkpointer would be a valid alternative.
"""
from __future__ import annotations

import json
import time
from datetime import timedelta

from langgraph.graph import END, START, StateGraph
from sqlalchemy import update

from relay.agent import policy
from relay.agent.facts import data, derive, evidence_snapshot
from relay.agent.llm_client import LLMError
from relay.agent.types import Proposal, RunState
from relay.audit import record_event
from relay.context import AppContext
from relay.errors import BudgetExceeded, LeaseLost
from relay.models import Run, utcnow
from relay.tools.executor import ToolExecutor, ToolResult
from relay.tools.registry import REGISTRY

NODES = ("observe", "investigate", "propose", "validate", "execute", "verify")
MAX_CONSECUTIVE_DENIALS = 2
MAX_HISTORY = 40


class AgentNodes:
    def __init__(self, app: AppContext, run_id: str, worker_id: str, planner):
        self.app, self.run_id, self.worker_id, self.planner = app, run_id, worker_id, planner
        self.ex: ToolExecutor | None = None

    # ------------------------------------------------------------------ plumbing
    def bind(self, state: RunState) -> None:
        self.ex = ToolExecutor(self.app, self.run_id, state["order_id"], state["correlation_id"])

    def _audit(self, state: RunState, event_type: str, actor: str, payload: dict) -> None:
        with self.app.session() as s:
            record_event(s, correlation_id=state["correlation_id"], run_id=self.run_id, order_id=state["order_id"],
                         actor=actor, event_type=event_type, payload=payload)
            s.commit()

    def wrap(self, name: str, fn):
        def node(state: RunState) -> dict:
            started = time.perf_counter()
            self._check_lease()
            if self.app.hooks.before_node:
                self.app.hooks.before_node(name)
            try:
                self._check_limits(name)
                updates = fn(state)
            except BudgetExceeded as e:
                updates = self._escalate_budget(state, e)
            updates = dict(updates)
            if not updates.get("halt") and state.get("step_mode"):
                updates["halt"] = "paused"
            merged = {**state, **updates}
            self._persist(merged, int((time.perf_counter() - started) * 1000))
            return updates

        return node

    def _check_lease(self) -> None:
        with self.app.session() as s:
            run = s.get(Run, self.run_id)
            if run is None or run.lease_owner != self.worker_id:
                raise LeaseLost()

    def _check_limits(self, node: str) -> None:
        """Step and active-time limits. Verification is exempt so a run can always end with a verdict."""
        if node == "verify":
            return
        s_ = self.app.settings
        with self.app.session() as s:
            run = s.get(Run, self.run_id)
            if run.steps >= s_.max_steps:
                raise BudgetExceeded("steps", s_.max_steps)
            if run.active_ms / 1000 >= s_.max_active_seconds:
                raise BudgetExceeded("active_seconds", s_.max_active_seconds)

    def _persist(self, state: RunState, duration_ms: int) -> None:
        halt = state.get("halt")
        values: dict = {"state": json.loads(json.dumps(state, default=str)), "steps": Run.steps + 1,
                        "active_ms": Run.active_ms + duration_ms, "updated_at": utcnow(),
                        "lease_expires_at": utcnow() + timedelta(seconds=self.app.settings.lease_seconds)}
        if halt == "finished":
            outcome = state.get("outcome") or "failed"
            values.update(status="succeeded" if outcome in ("fulfilled", "refunded") else
                          ("escalated" if outcome == "escalated" else "failed"),
                          outcome=outcome, summary=state.get("summary"), finished_at=utcnow(),
                          error=state.get("error"))
        elif halt in ("paused", "awaiting_approval"):
            values["status"] = halt
        else:
            values["status"] = "running"
        with self.app.session() as s:
            res = s.execute(update(Run).where(Run.id == self.run_id, Run.lease_owner == self.worker_id).values(**values))
            if res.rowcount != 1:
                s.rollback()
                raise LeaseLost()
            s.commit()

    # ------------------------------------------------------------------ evidence helpers
    def _store(self, state: RunState, key: str, result: ToolResult) -> tuple[dict, list]:
        ev = dict(state["evidence"])
        ev[key] = {"data": result.output, "tool": result.tool}
        stale = [k for k in state["stale"] if k != key]
        return ev, stale

    def _needed_reads(self, state: RunState) -> list[tuple[str, str, dict]]:
        oid, stale = state["order_id"], set(state["stale"])
        out: list[tuple[str, str, dict]] = []
        if "order" in stale:
            out.append(("order", "get_order_status", {"order_id": oid}))
        for key, tool in (("payment", "get_payment_status"), ("reservation", "get_inventory_reservation"),
                          ("shipment", "get_shipment_status")):
            if key in stale:
                out.append((key, tool, {"order_id": oid}))
        if "order" not in stale and "reservation" not in stale and "stock" in stale:
            f = derive(state)
            if f.order and f.status == "paid" and f.reservation_state == "none":
                out.append(("stock", "check_stock", {"sku": f.order["sku"], "quantity": f.order["quantity"]}))
        return out

    def _read_all(self, state: RunState, node: str) -> dict:
        """Deterministic evidence collection. Stops at the first failed read and records it."""
        updates: dict = {}
        for _ in range(3):
            plan = self._needed_reads({**state, **updates})
            if not plan:
                break
            for key, tool, args in plan:
                r = self.ex.call(tool, args, node=node)
                cur = {**state, **updates}
                if not r.ok:
                    return {**updates, "last_failure": {"tool": tool, "code": r.error["code"],
                                                         "message": r.error["message"]},
                            "stale": [k for k in cur["stale"] if k != key]}
                ev, stale = self._store(cur, key, r)
                updates.update(evidence=ev, stale=stale)
                if key == "order" and r.output.get("injection_suspected"):
                    self._audit(state, "untrusted_content_flagged", "system",
                                {"note": "free text on the order looks like prompt injection; treated as data only"})
        return updates

    # ------------------------------------------------------------------ nodes
    def observe(self, state: RunState) -> dict:
        r = self.ex.call("get_order_status", {"order_id": state["order_id"]}, node="observe")
        if not r.ok:
            return self._fail(state, f"cannot read order: {r.error['code']}")
        ev, stale = self._store(state, "order", r)
        o = r.output
        self._audit(state, "observe", "system", {"order_status": o["status"], "total_cents": o["total_cents"],
                                                  "mode": state["mode"], "injection_suspected": o["injection_suspected"]})
        if o["injection_suspected"]:
            self._audit(state, "untrusted_content_flagged", "system",
                        {"note": "free text on the order looks like prompt injection; treated as data only"})
        return {"evidence": ev, "stale": stale, "next_node": "investigate"}

    def investigate(self, state: RunState) -> dict:
        updates = self._read_all(state, "investigate")
        updates["next_node"] = "propose"
        return updates

    def propose(self, state: RunState) -> dict:
        try:
            proposal = self.planner.propose(state, state.get("feedback", []))
        except LLMError as e:
            code = "LLM_INVALID_OUTPUT" if e.code == "LLM_INVALID_OUTPUT" else "LLM_UNAVAILABLE"
            proposal = Proposal(action="tool", tool="create_support_ticket",
                                args={"order_id": state["order_id"], "reason_code": code,
                                      "summary": f"LLM planner failed closed: {e}"},
                                summary=f"LLM planner unavailable or invalid ({e.code}); escalating instead of guessing.",
                                source="policy", rule_id="P-FAIL-CLOSED")
        self._audit(state, "proposal", self.planner.source, proposal.model_dump())
        return {"proposal": proposal.model_dump(), "next_node": "validate"}

    def validate(self, state: RunState) -> dict:
        proposal = Proposal.model_validate(state["proposal"])
        decision = policy.evaluate(state, proposal)
        history = (state.get("history", []) + [{
            "proposal": proposal.model_dump(), "allowed": decision.allowed, "rule_ids": decision.rule_ids,
            "reasons": decision.reasons, "canonical_args": decision.canonical_args}])[-MAX_HISTORY:]
        self._audit(state, "policy_decision", "policy", {"tool": proposal.tool, "action": proposal.action,
                    "allowed": decision.allowed, "rule_ids": decision.rule_ids, "reasons": decision.reasons})
        base = {"decision": decision.model_dump(), "history": history}
        if decision.allowed:
            base.update(denials=0, feedback=[])
            if proposal.action == "finish":
                return {**base, "next_node": "verify"}
            if proposal.action == "wait_for_approval":
                return {**base, "halt": "awaiting_approval", "next_node": "investigate",
                        "stale": sorted(set(state["stale"]) | {"order"})}
            return {**base, "next_node": "execute"}
        denials = state.get("denials", 0) + 1
        if denials >= MAX_CONSECUTIVE_DENIALS:
            forced = Proposal(action="tool", tool="create_support_ticket",
                              args={"order_id": state["order_id"], "reason_code": "POLICY_DENIED",
                                    "summary": "Proposed actions were repeatedly rejected by policy: "
                                               + "; ".join(decision.reasons)[:300]},
                              summary="Repeated policy denials; escalating to a human.", source="policy",
                              rule_id="P-FORCED-ESCALATION")
            forced_decision = policy.evaluate(state, forced)
            history = (history + [{"proposal": forced.model_dump(), "allowed": True,
                                   "rule_ids": forced_decision.rule_ids, "reasons": forced_decision.reasons,
                                   "canonical_args": forced_decision.canonical_args}])[-MAX_HISTORY:]
            return {"proposal": forced.model_dump(), "decision": forced_decision.model_dump(), "history": history,
                    "denials": 0, "feedback": [], "next_node": "execute"}
        return {**base, "denials": denials, "feedback": decision.reasons, "next_node": "propose"}

    def execute(self, state: RunState) -> dict:
        proposal = Proposal.model_validate(state["proposal"])
        decision = state["decision"]
        assert decision and decision["allowed"] and proposal.tool, "execute reached without an allowed tool proposal"
        spec = REGISTRY[proposal.tool]
        r = self.ex.call(proposal.tool, decision["canonical_args"], node="execute",
                         exempt_budget=proposal.tool == "create_support_ticket")
        stale = sorted(set(state["stale"]) | set(spec.invalidates))
        base = {"stale": stale, "last_result": {"tool": proposal.tool, "ok": r.ok, "output": r.output,
                                                "error": r.error, "attempts": r.attempts, "replayed": r.replayed,
                                                "reconciled": r.reconciled}}
        if r.ok:
            base["last_failure"] = None
            if proposal.tool == "create_support_ticket":
                return {**base, "escalated": True, "next_node": "verify"}
            return {**base, "next_node": "investigate"}
        if proposal.tool == "create_support_ticket":
            return {**base, **self._fail(state, f"escalation ticket could not be created: {r.error['code']}")}
        return {**base, "last_failure": {"tool": proposal.tool, "code": r.error["code"], "message": r.error["message"]},
                "next_node": "investigate"}

    def verify(self, state: RunState) -> dict:
        f = derive(state)
        if state.get("escalated"):
            expectation = "escalated"
        elif f.status == "shipped":
            expectation = "fulfilled"
        elif f.status == "refunded":
            expectation = "refunded"
        else:
            return self._fail(state, f"verify reached with non-terminal status {f.status}")
        r = self.ex.call("verify_recovery", {"order_id": state["order_id"], "expectation": expectation},
                         node="verify", exempt_budget=True)
        verification = r.output if r.ok else {"order_id": state["order_id"], "expectation": expectation,
                                              "passed": False, "checks": [], "error": r.error}
        passed = bool(r.ok and r.output["passed"])
        self._audit(state, "verification", "system", {"expectation": expectation, "passed": passed})
        base = {"verification": verification}
        if expectation == "escalated":
            return {**base, "halt": "finished", "outcome": "escalated",
                    "summary": "Escalated to a human with an evidence snapshot."
                               + ("" if passed else " Post-escalation verification reported issues.")}
        if passed:
            word = "shipped" if expectation == "fulfilled" else "refunded"
            return {**base, "halt": "finished", "outcome": expectation,
                    "summary": f"Order recovered and {word}. All {len(verification['checks'])} verification checks passed."}
        failed = [c["id"] for c in verification.get("checks", []) if not c["passed"]]
        forced = Proposal(action="tool", tool="create_support_ticket",
                          args={"order_id": state["order_id"], "reason_code": "VERIFICATION_FAILED",
                                "summary": f"Post-recovery verification failed: {failed or r.error}"},
                          summary="Verification failed; escalating.", source="policy", rule_id="P-VERIFY-FAILED")
        d = policy.evaluate(state, forced)
        return {**base, "proposal": forced.model_dump(), "decision": d.model_dump(), "next_node": "execute"}

    # ------------------------------------------------------------------ terminal helpers
    def _fail(self, state: RunState, message: str) -> dict:
        self._audit(state, "run_failed", "system", {"error": message})
        return {"halt": "finished", "outcome": "failed", "error": message, "summary": f"Run failed: {message}"}

    def _escalate_budget(self, state: RunState, exc: BudgetExceeded) -> dict:
        """A deterministic limit fired: stop acting, open a ticket (exempt from the budget) and finish."""
        self._audit(state, "budget_exceeded", "system", {"limit": exc.which, "value": exc.limit})
        snapshot = evidence_snapshot(state) if data(state, "order") else {}
        r = self.ex.call("create_support_ticket", {
            "order_id": state["order_id"], "reason_code": "BUDGET_EXHAUSTED", "priority": "high",
            "summary": f"Agent stopped: {exc}", "evidence": snapshot}, node="budget", exempt_budget=True)
        msg = f"{exc}" + ("" if r.ok else f"; ticket creation failed: {r.error['code']}")
        return {"halt": "finished", "outcome": "escalated" if r.ok else "failed", "error": msg,
                "summary": f"Stopped by execution limit: {msg}", "escalated": r.ok}


def build_graph(nodes: AgentNodes):
    g = StateGraph(RunState)
    for name in NODES:
        g.add_node(name, nodes.wrap(name, getattr(nodes, name)))
    g.add_conditional_edges(START, lambda s: s["next_node"], {n: n for n in NODES})

    def route(s: RunState) -> str:
        return END if s.get("halt") else s["next_node"]

    for name in NODES:
        g.add_conditional_edges(name, route, {**{n: n for n in NODES}, END: END})
    return g.compile()


__all__ = ["AgentNodes", "build_graph", "NODES"]
