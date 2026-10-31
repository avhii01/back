"""Error taxonomy shared by service clients, tools and the executor.

The distinction that matters most is TRANSIENT vs AMBIGUOUS:
  * Transient: we know the request did not take effect (connection refused, 503 before commit).
  * Ambiguous: the request MAY have taken effect (read timeout, 504). We must check
    (reconcile) before retrying a mutation.
"""
from __future__ import annotations


class ToolError(Exception):
    code = "TOOL_ERROR"
    retryable = False
    ambiguous = False

    def __init__(self, message: str, *, code: str | None = None, details: dict | None = None):
        super().__init__(message)
        if code:
            self.code = code
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "ambiguous": self.ambiguous,
            "details": self.details,
        }


class TransientToolError(ToolError):
    code = "SERVICE_UNAVAILABLE"
    retryable = True


class AmbiguousToolError(ToolError):
    code = "AMBIGUOUS_TIMEOUT"
    retryable = True
    ambiguous = True


class PermanentToolError(ToolError):
    code = "REJECTED"


class ToolInputError(PermanentToolError):
    code = "INVALID_INPUT"


class BudgetExceeded(Exception):
    """A deterministic execution limit (steps, tool calls, active time) was hit."""

    def __init__(self, which: str, limit):
        super().__init__(f"{which} limit exceeded ({limit})")
        self.which = which
        self.limit = limit


class LeaseLost(Exception):
    """Another worker owns this run now; stop without touching state."""


class SimulatedCrash(BaseException):
    """Test hook: emulates the process dying. Derives from BaseException so that
    no `except Exception` handler in the executor/runner can swallow it."""
