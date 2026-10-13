import logging
import multiprocessing
import os
import queue
import tempfile
import threading
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from core import diagnostics


def emit_from_process(log_queue):
    diagnostics.configure_logging(log_queue, worker=True)
    with diagnostics.log_context(chat="测试会话", revision=7, message_id=42):
        diagnostics.log_event("capture.child_test", new_count=1)
        try:
            raise RuntimeError("子进程异常")
        except RuntimeError:
            diagnostics.log_event("capture.child_failed", level="error", exc_info=True)
    diagnostics.shutdown_logging()


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.addCleanup(diagnostics.shutdown_logging)
        self.path = Path(self.directory.name) / "log"

    def contents(self):
        return "\n".join(path.read_text(encoding="utf-8") for path in sorted(self.path.glob("*.log")))

    def test_utf8_daily_log_appends_and_keeps_context(self):
        diagnostics.configure_logging(directory=self.path)
        with diagnostics.log_context(chat="月仔", revision=3, message_id=22):
            diagnostics.log_event("analysis.started", message_count=10)
        diagnostics.log_event("application.stopped")
        diagnostics.shutdown_logging()
        diagnostics.configure_logging(directory=self.path)
        diagnostics.log_event("application.started")
        diagnostics.shutdown_logging()
        text = self.contents()
        self.assertEqual(len(list(self.path.glob("*.log"))), 1)
        self.assertIn('"chat": "月仔"', text)
        self.assertIn('"revision": 3', text)
        self.assertIn('"message_id": 22', text)
        self.assertEqual(text.count("application.started"), 1)
        self.assertIn("application.stopped {}", text)

    def test_disabled_logging_does_not_write_and_can_be_enabled_again(self):
        self.assertIsNone(diagnostics.configure_logging(directory=self.path, enabled=False))
        diagnostics.log_event("test.disabled")
        self.assertFalse(self.path.exists())
        diagnostics.configure_logging(directory=self.path, enabled=True, level="INFO")
        diagnostics.log_event("test.enabled")
        diagnostics.shutdown_logging()
        self.assertIn("test.enabled", self.contents())
        self.assertNotIn("test.disabled", self.contents())

    def test_level_filtering_and_step_detail_classification(self):
        diagnostics.configure_logging(directory=self.path, level="INFO")
        diagnostics.log_event("test.debug", level="debug")
        diagnostics.log_event("test.info")
        try:
            with diagnostics.log_step("test.failure"):
                raise ValueError("保留错误堆栈")
        except ValueError:
            pass
        diagnostics.shutdown_logging()
        text = self.contents()
        self.assertNotIn("test.debug", text)
        self.assertNotIn("test.failure.started", text)
        self.assertNotIn("test.failure.completed", text)
        self.assertIn("test.info", text)
        self.assertIn("test.failure.failed", text)
        self.assertIn("ValueError: 保留错误堆栈", text)

    def test_worker_follows_shared_enabled_state_and_level(self):
        log_queue = queue.Queue()
        control = multiprocessing.Value("i", logging.INFO)
        diagnostics.configure_logging(log_queue, worker=True, control=control)
        diagnostics.log_event("worker.info")
        control.value = 0
        diagnostics.log_event("worker.disabled", level="error")
        control.value = logging.WARNING
        diagnostics.log_event("worker.filtered")
        diagnostics.log_event("worker.warning", level="warning")
        diagnostics.shutdown_logging()
        messages = [log_queue.get_nowait().getMessage() for _ in range(log_queue.qsize())]
        self.assertTrue(any("worker.info" in message for message in messages))
        self.assertTrue(any("worker.warning" in message for message in messages))
        self.assertFalse(any("worker.disabled" in message for message in messages))
        self.assertFalse(any("worker.filtered" in message for message in messages))

    def test_secrets_are_redacted_in_fields_messages_and_tracebacks(self):
        diagnostics.configure_logging(directory=self.path)
        diagnostics.register_secret("provided-secret-987")
        with patch.dict(os.environ, {"JEV_API_KEY": "live-secret-123",
                                     "DEEPSEEK_API_KEY": "legacy-secret-456"}):
            diagnostics.log_event("test.secret", api_key="arbitrary-secret-321",
                                  draft_key_present=True,
                                  nested={"Authorization": "other-secret-654"},
                                  reason="live-secret-123 legacy-secret-456 provided-secret-987 Bearer bearer-secret-789")
            try:
                with diagnostics.log_step("test.failure"):
                    raise RuntimeError("live-secret-123 provided-secret-987 api_key=query-secret-012")
            except RuntimeError:
                pass
        text = self.contents()
        for value in ("live-secret-123", "legacy-secret-456", "provided-secret-987", "bearer-secret-789",
                      "arbitrary-secret-321", "other-secret-654", "query-secret-012"):
            self.assertNotIn(value, text)
        self.assertIn("[REDACTED]", text)
        self.assertIn('"draft_key_present": true', text)
        self.assertIn("Traceback (most recent call last)", text)
        self.assertIn("test.failure.failed", text)

    def test_day_change_routes_records_to_each_date(self):
        handler = diagnostics.DailyFileHandler(self.path)
        self.addCleanup(handler.close)
        handler.setFormatter(diagnostics.SafeFormatter("%(message)s"))
        for day in (12, 13, 12):
            record = logging.LogRecord("jev", logging.INFO, __file__, 1, f"day-{day}", (), None)
            record.created = datetime(2026, 10, day, 23, 59).timestamp()
            handler.handle(record)
        self.assertEqual((self.path / "2026-10-12.log").read_text(encoding="utf-8"), "day-12\nday-12\n")
        self.assertEqual((self.path / "2026-10-13.log").read_text(encoding="utf-8"), "day-13\n")

    def test_queue_flushes_thread_records_and_exception_before_shutdown(self):
        diagnostics.configure_logging(queue.Queue(), directory=self.path)

        def emit(index):
            with diagnostics.log_context(chat=f"会话-{index}"):
                for revision in range(10):
                    diagnostics.log_event("thread.record", revision=revision)
                try:
                    raise ValueError("完整异常")
                except ValueError:
                    diagnostics.log_event("thread.failed", level="error", exc_info=True)

        threads = [threading.Thread(target=emit, args=(index,)) for index in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        diagnostics.shutdown_logging()
        text = self.contents()
        self.assertEqual(text.count("thread.record"), 40)
        self.assertEqual(text.count("thread.failed"), 4)
        self.assertEqual(text.count("ValueError: 完整异常"), 4)

    def test_spawned_child_uses_parent_writer(self):
        context = multiprocessing.get_context("spawn")
        log_queue = context.Queue()
        diagnostics.configure_logging(log_queue, directory=self.path)
        process = context.Process(target=emit_from_process, args=(log_queue,))
        process.start()
        process.join(timeout=20)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        self.assertEqual(process.exitcode, 0)
        diagnostics.shutdown_logging()
        log_queue.close()
        log_queue.join_thread()
        text = self.contents()
        self.assertIn("capture.child_test", text)
        self.assertIn("RuntimeError: 子进程异常", text)
        self.assertIn('"message_id": 42', text)
        self.assertEqual(len(list(self.path.glob("*.log"))), 1)

    def test_default_directory_is_program_directory_not_current_directory(self):
        with patch.object(diagnostics, "DailyFileHandler", wraps=diagnostics.DailyFileHandler) as handler, \
                patch.object(diagnostics, "__file__", str(Path(self.directory.name) / "core" / "diagnostics.py")):
            diagnostics.configure_logging()
        self.assertEqual(handler.call_args.args[0], self.path)

    def test_frozen_directory_is_beside_executable(self):
        with patch.object(diagnostics.sys, "frozen", True, create=True), \
                patch.object(diagnostics.sys, "executable", str(Path(self.directory.name) / "chat.exe")):
            self.assertEqual(diagnostics.configure_logging(), self.path)

    def test_unhandled_exception_hooks_are_logged_and_restored(self):
        with patch.object(diagnostics, "_original_excepthook") as original, \
                patch.object(diagnostics, "_original_thread_hook") as thread_original:
            diagnostics.configure_logging(directory=self.path)
            try:
                raise RuntimeError("未捕获异常")
            except RuntimeError as error:
                diagnostics.sys.excepthook(type(error), error, error.__traceback__)
                args = type("Args", (), {"exc_type": type(error), "exc_value": error,
                                        "exc_traceback": error.__traceback__, "thread": threading.current_thread()})()
                diagnostics.threading.excepthook(args)
            original.assert_called_once()
            thread_original.assert_called_once()
            diagnostics.shutdown_logging()
        self.assertIn("exception.unhandled", self.contents())
        self.assertIn("exception.thread", self.contents())
        diagnostics.sys.excepthook = diagnostics._original_excepthook
        diagnostics.threading.excepthook = diagnostics._original_thread_hook


if __name__ == "__main__":
    unittest.main()
