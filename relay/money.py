"""Monetary calculations. Integer cents only - never floats, never model-supplied."""
from __future__ import annotations


def format_cents(cents: int, currency: str = "USD") -> str:
    sign = "-" if cents < 0 else ""
    cents = abs(cents)
    return f"{sign}{cents // 100}.{cents % 100:02d} {currency}"


def refundable_amount(*, captured_cents: int, refunded_cents: int, order_total_cents: int) -> int:
    """Amount that may be refunded: what was captured minus what was already refunded,
    never more than the order total. Raises on inconsistent inputs."""
    if min(captured_cents, refunded_cents, order_total_cents) < 0:
        raise ValueError("negative amount")
    remaining = captured_cents - refunded_cents
    if remaining <= 0:
        raise ValueError("nothing left to refund")
    if captured_cents != order_total_cents:
        raise ValueError("captured amount does not match order total")
    return min(remaining, order_total_cents)
