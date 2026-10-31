from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field


class Proposal(BaseModel):
    """What the planner (rules or LLM) wants to do next. It is only a REQUEST."""

    action: Literal["tool", "finish", "wait_for_approval"]
    tool: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    summary: str = Field(default="", max_length=500)  # short stated rationale - not chain-of-thought
    evidence_refs: list[str] = Field(default_factory=list)
    source: Literal["rules", "llm", "policy"] = "rules"
    rule_id: str | None = None


class Decision(BaseModel):
    """Deterministic policy verdict. `canonical_args` are what actually get executed."""

    allowed: bool
    rule_ids: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    canonical_args: dict[str, Any] | None = None
    requires_approval: bool = False


class RunState(TypedDict, total=False):
    run_id: str
    order_id: str
    correlation_id: str
    mode: str
    step_mode: bool
    next_node: str
    halt: str | None  # None | paused | awaiting_approval | finished
    evidence: dict[str, dict]
    stale: list[str]
    proposal: dict | None
    decision: dict | None
    denials: int
    feedback: list[str]
    last_failure: dict | None
    last_result: dict | None
    escalated: bool
    outcome: str | None
    summary: str | None
    error: str | None
    verification: dict | None
    history: list[dict]
