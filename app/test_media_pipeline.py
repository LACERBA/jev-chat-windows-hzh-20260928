import queue
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import main
from app import settings
from app.media import media_token
from app.message_store import MessageStore
from core.jev_client import JevError


class MediaPipelineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "messages.sqlite3"
        self.config = {"media_mode": "hybrid", "vision_upload": True, "analysis_enabled": True}
        self.overlay = Mock()
        self.overlay.current_chat.return_value = "目标会话"
        self.state = {"app": "wechat", "chat": "目标会话", "raw_chat": "目标会话", "busy": False,
                      "rerun": None, "hwnd": None, "area": None}
        self.slots = threading.BoundedSemaphore(1)
        self.controller = Mock(active=None)
        patches = [patch.object(settings, "_read", side_effect=lambda name, default=None: self.config.get(name, default)),
                   patch.object(settings, "message_db_path", return_value=self.path),
                   patch.object(settings, "vision_config", return_value={}),
                   patch.object(main, "ov", self.overlay, create=True), patch.object(main, "state", self.state),
                   patch.object(main, "chats", {}), patch.object(main, "media_results", queue.Queue()),
                   patch.object(main, "media_slots", self.slots), patch.object(main, "media_inflight", set()),
                   patch.object(main, "conversation_controller", self.controller), patch.object(main, "request_read")]
        for target in patches:
            target.start()
            self.addCleanup(target.stop)
        self.store = MessageStore(self.path)
        self.addCleanup(self.store.close)
        self.item = {"fingerprint": "1" * 32, "kind": "image", "ocr_text": "", "who": "her", "sender": None}
        self.bubble = ("her", None, media_token(self.item["fingerprint"]))
        self.snapshot = self.store.record_snapshot("wechat", "目标会话", "目标会话", [self.bubble],
                                                   (0, 0, 300, 300), media=[self.item], vision=True)
        self.job = {**self.snapshot["media_jobs"][0], "image": b"test-image"}
        main.sync_media("目标会话", self.snapshot["media_state"])
        main.sync_media_history("目标会话", self.snapshot["history"], self.snapshot["history_ids"])
        self.chat = main.chat_of("目标会话")
        self.chat["rev"] = 1
        self.chat["media_waiting"] = True

    def finish(self, result=None):
        self.slots.acquire()
        with patch.object(main, "recognize_media", return_value=result or {
            "summary": "照片画面", "status": "ready", "source": "vision"}) as recognize:
            main.recognize_media_bg("wechat", "目标会话", self.job)
        return recognize

    def test_analysis_waits_for_pending_media_without_starting_network(self):
        with patch.object(main.threading, "Thread") as thread:
            self.assertTrue(main.start_analyze("目标会话", list(self.chat["history"])))
        thread.assert_not_called()
        self.assertTrue(self.chat["media_waiting"])

    def test_pending_media_wait_is_bounded_and_can_generate_question_reply(self):
        self.chat["media_wait_started"] = 100
        with patch.object(main.time, "monotonic", return_value=106), \
                patch.object(settings, "models_ready", return_value=True), patch.object(main.threading, "Thread") as thread:
            self.assertTrue(main.start_analyze("目标会话", list(self.chat["history"])))
        thread.assert_called_once()
        thread.return_value.start.assert_called_once()
        self.assertEqual(self.chat["analysis_requested_rev"], self.chat["rev"])

    def test_pending_media_timeout_starts_only_one_fallback_per_revision(self):
        self.chat["media_wait_started"] = 100

        def started(*args):
            self.state["busy"] = True
            return True

        with patch.object(main.time, "monotonic", return_value=106), \
                patch.object(main, "drain"), patch.object(main, "drain_media"), \
                patch.object(main, "results", queue.Queue()), patch.object(main, "update_result", queue.Queue()), \
                patch.object(main, "start_analyze", side_effect=started) as analyze:
            main.tick()
            main.tick()
        analyze.assert_called_once()
        self.assertIsNone(self.state["rerun"])

    def test_late_visual_result_does_not_repeat_an_already_sent_reply(self):
        self.chat["auto_sent_key"] = main.reply_key("目标会话")
        self.finish()
        with patch.object(main, "start_analyze") as analyze:
            main.drain_media()
        analyze.assert_not_called()

    def test_media_completion_during_settle_period_does_not_start_reply_analysis(self):
        self.controller.phase = "settling"
        self.finish()
        with patch.object(main, "start_analyze") as analyze:
            main.drain_media()
        analyze.assert_not_called()
        self.assertTrue(self.chat["media_waiting"])
        self.assertEqual(self.chat["rev"], 2)

    def test_media_timeout_cannot_bypass_settle_period(self):
        self.controller.phase = "settling"
        self.chat["media_wait_started"] = 100
        with patch.object(main.time, "monotonic", return_value=106), \
                patch.object(main, "drain"), patch.object(main, "drain_media"), \
                patch.object(main, "results", queue.Queue()), patch.object(main, "update_result", queue.Queue()), \
                patch.object(main, "start_analyze") as analyze:
            main.tick()
        analyze.assert_not_called()
        self.assertEqual(self.chat["media_fallback_rev"], -1)
        self.assertIsNone(self.state["rerun"])

    def test_disabled_upload_never_calls_visual_model(self):
        self.config["vision_upload"] = False
        recognize = self.finish()
        recognize.assert_not_called()
        self.assertEqual(self.store.media_state("wechat", "目标会话")[self.job["message_id"]]["status"], "ocr")
        self.assertNotIn("image", self.job)

    def test_baseline_visual_enrichment_does_not_add_auto_send_block(self):
        self.finish()
        with patch.object(main, "start_analyze") as analyze:
            main.drain_media()
        self.assertEqual(self.chat["rev"], 2)
        self.assertNotIn("auto_blocked_rev", self.chat)
        self.assertIn("照片画面", self.chat["history"][0][1])
        analyze.assert_called_once()

    def test_delayed_media_result_keeps_latest_text_and_analyzes_latest_context(self):
        latest = self.store.record_snapshot("wechat", "目标会话", "目标会话",
                                           [self.bubble, ("her", None, "看看刚发的图")], (0, 0, 300, 300), vision=True)
        main.sync_media("目标会话", latest["media_state"])
        main.sync_media_history("目标会话", latest["history"], latest["history_ids"])
        self.chat["rev"] = 2
        self.finish()
        with patch.object(main, "start_analyze") as analyze:
            main.drain_media()
        self.assertEqual(self.chat["history"][-1][1], "看看刚发的图")
        self.assertEqual(analyze.call_args.args[1][-1][1], "看看刚发的图")

    def test_reply_after_media_was_sent_by_me_does_not_trigger_new_analysis(self):
        latest = self.store.record_snapshot("wechat", "目标会话", "目标会话",
                                           [self.bubble, ("me", None, "我已经回复")], (0, 0, 300, 300), vision=True)
        main.sync_media_history("目标会话", latest["history"], latest["history_ids"])
        self.finish()
        with patch.object(main, "start_analyze") as analyze:
            main.drain_media()
        analyze.assert_not_called()

    def test_old_snapshot_cannot_overwrite_completed_visual_result(self):
        self.finish()
        with patch.object(main, "start_analyze"):
            main.drain_media()
        main.sync_media("目标会话", self.snapshot["media_state"])
        main.sync_media_history("目标会话", self.snapshot["history"], self.snapshot["history_ids"])
        self.assertFalse(main.media_pending("目标会话"))
        self.assertIn("照片画面", self.chat["history"][0][1])

    def test_media_outside_reply_context_does_not_invalidate_latest_analysis(self):
        visible = [self.bubble] + [("her", None, f"消息{number}") for number in range(12)]
        latest = self.store.record_snapshot("wechat", "目标会话", "目标会话", visible, (0, 0, 300, 300), vision=True)
        main.sync_media_history("目标会话", latest["history"], latest["history_ids"])
        self.chat["media_waiting"] = False
        self.chat["rev"] = 12
        self.finish()
        with patch.object(main, "start_analyze") as analyze:
            main.drain_media()
        self.assertEqual(self.chat["rev"], 12)
        analyze.assert_not_called()

    def test_switching_chat_does_not_write_to_other_chat_or_invalidate_its_ui(self):
        self.state.update(chat="其他会话", raw_chat="其他会话")
        self.overlay.current_chat.return_value = "其他会话"
        main.chat_of("其他会话")["history"].append(("her", "其他内容", None))
        self.finish()
        with patch.object(main, "start_analyze"):
            main.drain_media()
        self.assertEqual(main.chat_of("其他会话")["history"][-1][1], "其他内容")
        self.overlay.invalidate_replies.assert_not_called()
        self.assertIn("照片画面", self.store.history("wechat", "目标会话")[0][2])

    def test_auto_switch_without_chat_authorization_does_not_call_vision(self):
        self.config["auto_switch"] = True
        recognize = self.finish()
        recognize.assert_not_called()

    def test_visual_failure_falls_back_to_ocr_and_still_generates_reply(self):
        self.slots.acquire()
        with patch.object(main, "recognize_media", side_effect=JevError("offline", 529)):
            main.recognize_media_bg("wechat", "目标会话", self.job)
        with patch.object(main, "start_analyze", return_value=True) as analyze:
            main.drain_media()
        analyze.assert_called_once()
        self.assertEqual(self.store.media_state("wechat", "目标会话")[self.job["message_id"]]["source"], "ocr")


if __name__ == "__main__":
    unittest.main()
