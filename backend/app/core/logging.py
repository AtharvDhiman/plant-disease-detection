"""Structured JSON logging shared by the API and the training scripts.

Every record carries a timestamp, level, logger name and message; extra fields
attached with ``logger.info(..., extra={"extra_fields": {...}})`` are merged into
the JSON object. Request bodies and uploaded file contents are never logged —
only sizes, content types and derived metrics.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class JsonFormatter(logging.Formatter):
    """Render log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "extra_fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class HumanFormatter(logging.Formatter):
    """Compact console format used when a TTY is attached."""

    def format(self, record: logging.LogRecord) -> str:
        base = f"{self.formatTime(record, '%H:%M:%S')} {record.levelname:<7} {record.name:<28} {record.getMessage()}"
        extra = getattr(record, "extra_fields", None)
        if isinstance(extra, dict) and extra:
            base += "  " + " ".join(f"{k}={v}" for k, v in extra.items())
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


_CONFIGURED = False


def configure_logging(level: str = "INFO", log_file: Path | None = None, json_console: bool = False) -> None:
    """Install handlers on the root logger exactly once per process."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger()
    root.setLevel(level.upper())
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(JsonFormatter() if json_console else HumanFormatter())
    root.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(JsonFormatter())
        root.addHandler(file_handler)

    # These libraries are noisy at INFO and say nothing we need.
    for noisy in ("PIL", "matplotlib", "matplotlib.font_manager", "urllib3", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.LoggerAdapter:
    """Return a logger that accepts ``**fields`` style structured extras."""
    return _FieldLogger(logging.getLogger(name), {})


class _FieldLogger(logging.LoggerAdapter):
    """Adapter turning keyword arguments into the ``extra_fields`` payload."""

    def process(self, msg, kwargs):
        fields = kwargs.pop("fields", None)
        if fields:
            kwargs.setdefault("extra", {})["extra_fields"] = fields
        return msg, kwargs
