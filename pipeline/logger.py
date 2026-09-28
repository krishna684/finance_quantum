from __future__ import annotations

"""Structured JSON logging for the pipeline.

Usage:
    from pipeline.logger import get_logger
    log = get_logger(__name__)
    log.info("window_started", query="add to cart", start_date="20240101", end_date="20240131")
    log.error("api_error", exc=str(e), query="add to cart")
"""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Extra fields attached via log.info("msg", extra={...})
        for key, value in record.__dict__.items():
            if key.startswith("_") or key in {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
                "taskName",
            }:
                continue
            payload[key] = value

        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, default=str)


def _build_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    return logger


class _PipelineLogger:
    """Thin wrapper that allows keyword-argument structured fields."""

    def __init__(self, name: str) -> None:
        self._log = _build_logger(name)

    def _emit(self, level: int, msg: str, **fields: Any) -> None:
        self._log.log(level, msg, extra=fields)

    def debug(self, msg: str, **fields: Any) -> None:
        self._emit(logging.DEBUG, msg, **fields)

    def info(self, msg: str, **fields: Any) -> None:
        self._emit(logging.INFO, msg, **fields)

    def warning(self, msg: str, **fields: Any) -> None:
        self._emit(logging.WARNING, msg, **fields)

    def error(self, msg: str, **fields: Any) -> None:
        self._emit(logging.ERROR, msg, **fields)


def get_logger(name: str) -> _PipelineLogger:
    return _PipelineLogger(name)


def configure_log_level(level: str = "INFO") -> None:
    """Set root pipeline log level.  Call once at startup."""
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.getLogger("pipeline").setLevel(numeric)
