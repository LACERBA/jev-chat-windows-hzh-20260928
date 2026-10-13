import atexit
import json
import logging
import re
import sys
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from logging.handlers import QueueHandler, QueueListener
from pathlib import Path

try:
    from .jev_client import redact_secrets
except ImportError:
    from jev_client import redact_secrets


_logger = logging.getLogger("jev")
_logger.addHandler(logging.NullHandler())
_logger.propagate = False
_context = ContextVar("log_context", default={})
_secrets = set()
_secret_lock = threading.Lock()
_listener = None
_handler = None
_original_excepthook = sys.excepthook
_original_thread_hook = threading.excepthook


def register_secret(value):
    if value:
        with _secret_lock:
            _secrets.add(str(value))


def redact_text(text):
    text = redact_secrets(str(text))
    with _secret_lock:
        secrets = sorted(_secrets, key=len, reverse=True)
    for secret in secrets:
        text = text.replace(secret, "[REDACTED]")
    text = re.sub(r"(?i)(Bearer\s+)[^\s\"',;]+", r"\1[REDACTED]", text)
    text = re.sub(r"(?i)((?:api[_-]?key|access[_-]?token|password|secret|token)[\"']?\s*[:=]\s*[\"']?)[^\s\"'&,;]+",
                  r"\1[REDACTED]", text)
    return re.sub(r"data:image/[^;\s]+;base64,[A-Za-z0-9+/=]+", "[IMAGE REDACTED]", text)


class SafeFormatter(logging.Formatter):
    def format(self, record):
        return redact_text(super().format(record))


class DailyFileHandler(logging.FileHandler):
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.day = datetime.now().strftime("%Y-%m-%d")
        super().__init__(self.directory / (self.day + ".log"), encoding="utf-8")

    def emit(self, record):
        try:
            day = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d")
            if day != self.day:
                if self.stream:
                    self.stream.close()
                self.stream = None
                self.day = day
                self.baseFilename = str(self.directory / (day + ".log"))
                self.stream = self._open()
            super().emit(record)
        except Exception:
            self.handleError(record)


@contextmanager
def log_context(**fields):
    token = _context.set({**_context.get(), **fields})
    try:
        yield
    finally:
        _context.reset(token)


def _safe_fields(fields):
    if isinstance(fields, dict):
        return {key: "[REDACTED]" if re.fullmatch(
                    r"(?i)(?:.*_)?(?:key|token|secret|password|authorization)", key)
                else _safe_fields(value) for key, value in fields.items()}
    if isinstance(fields, (list, tuple)):
        return [_safe_fields(value) for value in fields]
    return fields


def log_event(event, level="info", exc_info=False, **fields):
    fields = _safe_fields({**_context.get(), **fields})
    _logger.log(getattr(logging, level.upper(), logging.INFO), "%s %s", event,
                redact_text(json.dumps(fields, ensure_ascii=False, default=str)), exc_info=exc_info)


@contextmanager
def log_step(step, **fields):
    started = time.monotonic()
    log_event(step + ".started", **fields)
    try:
        yield
    except Exception as error:
        log_event(step + ".failed", level="error", exc_info=True,
                  elapsed_ms=round((time.monotonic() - started) * 1000),
                  error_type=type(error).__name__, http_status=getattr(error, "status", None), **fields)
        raise
    else:
        log_event(step + ".completed", elapsed_ms=round((time.monotonic() - started) * 1000), **fields)


def _unhandled(exc_type, exc, tb):
    log_event("exception.unhandled", level="critical", exc_info=(exc_type, exc, tb))
    _original_excepthook(exc_type, exc, tb)


def _thread_unhandled(args):
    log_event("exception.thread", level="critical",
              exc_info=(args.exc_type, args.exc_value, args.exc_traceback), thread=args.thread.name)
    _original_thread_hook(args)


def configure_logging(log_queue=None, worker=False, directory=None):
    global _listener, _handler
    shutdown_logging()
    _logger.setLevel(logging.DEBUG)
    if worker and log_queue is None:
        raise ValueError("采集子进程必须使用主进程日志队列")
    if not worker:
        root = (Path(sys.executable).parent if getattr(sys, "frozen", False)
                else Path(__file__).resolve().parent.parent)
        _handler = DailyFileHandler(directory or root / "log")
        _handler.setFormatter(SafeFormatter(
            "%(asctime)s.%(msecs)03d %(levelname)s [%(processName)s:%(process)d/%(threadName)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"))
    if log_queue is not None:
        producer = QueueHandler(log_queue)
        producer.setFormatter(SafeFormatter("%(message)s"))
        _logger.addHandler(producer)
        if not worker:
            _listener = QueueListener(log_queue, _handler)
            _listener.start()
    else:
        _logger.addHandler(_handler)
    sys.excepthook = _unhandled
    threading.excepthook = _thread_unhandled
    return _handler.directory if _handler else None


def shutdown_logging():
    global _listener, _handler
    for handler in list(_logger.handlers):
        if not isinstance(handler, logging.NullHandler):
            _logger.removeHandler(handler)
            if handler is not _handler:
                handler.close()
    if _listener is not None:
        _listener.stop()
        _listener = None
    if _handler is not None:
        _handler.close()
        _handler = None
    if sys.excepthook is _unhandled:
        sys.excepthook = _original_excepthook
    if threading.excepthook is _thread_unhandled:
        threading.excepthook = _original_thread_hook


atexit.register(shutdown_logging)
