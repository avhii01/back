"""Structured logging + secret redaction."""
from __future__ import annotations

import json
import logging
import re
from typing import Any

_SECRET_KEY = re.compile(r"(api[_-]?key|token|secret|authorization|password|passwd|credential)", re.I)
_BEARER = re.compile(r"(bearer\s+)[A-Za-z0-9._\-]+", re.I)
_SK = re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}")
MAX_JSON_CHARS = 8000


def redact(obj: Any) -> Any:
    """Recursively mask values under secret-looking keys and secret-looking strings."""
    if isinstance(obj, dict):
        return {k: ("[REDACTED]" if _SECRET_KEY.search(str(k)) else redact(v)) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [redact(v) for v in obj]
    if isinstance(obj, str):
        return _SK.sub("[REDACTED]", _BEARER.sub(r"\1[REDACTED]", obj))
    return obj


def bounded(obj: Any, max_chars: int = MAX_JSON_CHARS) -> Any:
    """Redact, and replace pathologically large payloads with a marker."""
    obj = redact(obj)
    try:
        encoded = json.dumps(obj, default=str)
    except (TypeError, ValueError):
        return {"_unserialisable": str(obj)[:500]}
    if len(encoded) > max_chars:
        return {"_truncated": True, "preview": encoded[:max_chars]}
    return json.loads(encoded)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": redact(record.getMessage()),
        }
        extra = getattr(record, "fields", None)
        if extra:
            base.update(redact(extra))
        return json.dumps(base, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if any(isinstance(h.formatter, JsonFormatter) for h in root.handlers):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root.handlers = [handler]
    root.setLevel(level)
    logging.getLogger("httpx").setLevel(logging.WARNING)
