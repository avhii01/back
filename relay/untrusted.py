"""Handling of untrusted text.

Service responses and free-text records (customer notes, legacy error messages,
payment memos) are DATA. Defence in depth:

 1. Structural (the real defence): policy and the rules planner never parse free text,
    and risky tool arguments are re-derived from trusted fields (see policy.py).
 2. Sanitisation: strip control chars, cap length before storing/showing/prompting.
 3. Prompt isolation: when text reaches an LLM it is JSON-encoded inside a delimited
    <untrusted_data> block, and the system prompt says it can never change instructions.
 4. Detection: a heuristic flags likely injection for the audit log. It is informational
    only - nothing depends on it, because pattern matching can always be evaded.
"""
from __future__ import annotations

import json
import re

MAX_TEXT = 500
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_PATTERNS = [
    r"ignore (all|any|the|previous|prior|above).{0,40}(instruction|rule|polic)",
    r"disregard .{0,40}(instruction|rule|polic)",
    r"(^|\n|\s)system\s*:",
    r"you are now",
    # Directive sentences aimed at the assistant. Policy statements that describe a
    # restriction ("the chatbot cannot approve a refund") must not be mistaken for
    # instructions to the assistant.
    r"(?:^|[.!?]\s*)(?:please\s+)?(?:approve|authori[sz]e|issue|process|send)\s+"
    r"(?:this|the|a|my|their|that)\s+(?:refund|payment)\b",
    r"(?:^|[.!?]\s*)(?:please\s+)?refund\s+(?:me|the customer|this order)\b.{0,30}"
    r"\b(?:immediately|now|without)\b",
    r"(?:^|[.!?]\s*)(?:please\s+)?override\s+(?:the\s+|all\s+|any\s+)?(?:polic(?:y|ies)|rules?)\b",
    r"\byou (?:must|should|need to|are required to)\b.{0,60}"
    r"\b(?:tell|say|confirm|approve|refund|ignore|override)\b",
    r"</?\s*(system|assistant|tool|untrusted_data)\s*>",
    r"call (the )?(execute_approved_refund|create_shipment|update_order_status)",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in _PATTERNS]


def sanitize_text(value: object, max_len: int = MAX_TEXT) -> str | None:
    if value is None:
        return None
    text = _CTRL.sub("", str(value))
    if len(text) > max_len:
        text = text[: max_len - 1] + "…"
    return text


def detect_injection(value: object) -> list[str]:
    if not value:
        return []
    text = str(value)
    return [p.pattern for p in _COMPILED if p.search(text)]


def wrap_untrusted(label: str, payload: object) -> str:
    """Render data for an LLM prompt so it cannot close the delimiter."""
    body = json.dumps(payload, default=str, ensure_ascii=True)
    body = body.replace("<", "\\u003c").replace(">", "\\u003e")
    return f'<untrusted_data label="{label}">\n{body}\n</untrusted_data>'
