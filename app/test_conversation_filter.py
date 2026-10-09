import multiprocessing
import unittest
from unittest.mock import Mock, mock_open, patch

import numpy as np

from app import ocr, settings
from app.capture import UnreadTracker
import main


class ConversationFilterTests(unittest.TestCase):
    def setUp(self):
        self.config = {"auto_switch": True, "ignored_chats": {"wechat": ["银行通知", "营销群"]}}
        self.read_patch = patch.object(settings, "_read", side_effect=lambda name, default=None:
                                       self.config.get(name, default))
        self.read_patch.start()
        self.addCleanup(self.read_patch.stop)
        for name in ("has_jev_key", "has_llm_key"):
            target = patch.object(settings, name, return_value=True)
            target.start()
            self.addCleanup(target.stop)
        self.overlay = Mock()
        self.overlay.current_chat.return_value = "月仔"
        self.overlay.after.side_effect = lambda *args: None
        self.capture = multiprocessing.Event()
        self.capture.set()
        self.state = {"app": "wechat", "chat": "月仔", "raw_chat": "月仔", "busy": False,
                      "rerun": None, "hwnd": 1, "area": (200, 60, 600, 500)}
        for name, value in (("ov", self.overlay), ("capture_on", self.capture),
                            ("state", self.state), ("chats", {})):
            target = patch.object(main, name, value, create=True)
            target.start()
            self.addCleanup(target.stop)
        self.controller = main.ConversationController()
        target = patch.object(main, "conversation_controller", self.controller)
        target.start()
        self.addCleanup(target.stop)

    def test_builtin_entries_are_ignored_without_affecting_similar_names(self):
        for title in ("公众号", "订阅号", "服务号", " 公 众 号 "):
            with self.subTest(title=title):
                self.assertTrue(settings.is_ignored_chat(title, "wechat"))
        for title in ("公众号讨论群", "服务号开发", "文件传输助手", "月仔"):
            with self.subTest(title=title):
                self.assertFalse(settings.is_ignored_chat(title, "wechat"))
        self.assertFalse(settings.is_ignored_chat("公众号", "kakaotalk"))

    def test_custom_entries_use_exact_names_and_app_scope(self):
        self.assertTrue(settings.is_ignored_chat(" 银行通知 ", "wechat"))
        self.assertFalse(settings.is_ignored_chat("银行通知讨论群", "wechat"))
        self.assertFalse(settings.is_ignored_chat("银行通知", "kakaotalk"))
        self.config["ignored_chats"]["wechat"] = ["Ann a"]
        self.assertTrue(settings.is_ignored_chat("Ann  a", "wechat"))
        self.assertFalse(settings.is_ignored_chat("Anna", "wechat"))

    def test_invalid_ignore_config_is_safe(self):
        for value in (None, [], "公众号", {"wechat": "银行通知"}, {"wechat": [None, 7]}):
            with self.subTest(value=value):
                self.config["ignored_chats"] = value
                self.assertEqual(settings.ignored_chats("wechat"), [])
                self.assertTrue(settings.is_ignored_chat("服务号", "wechat"))

    def test_ignored_event_does_not_enter_switch_queue(self):
        self.controller.on_unread({"base": "public", "id": "public:1", "at": 1,
                                   "name": "公众号"})
        self.assertFalse(self.controller.pending)

    def test_switch_skips_blocked_target_and_uses_next_valid_candidate(self):
        import time

        now = time.monotonic()
        for base, name, at in (("normal", "月仔", now), ("public", "公众号", now + 1)):
            event = {"base": base, "id": base + ":1", "name": name, "at": at}
            self.controller.pending[event["id"]] = event
            self.controller.badges[base] = {**event, "x": 120, "y": 90}
        with patch.object(main, "click") as click:
            self.controller.advance()
        click.assert_called_once_with(1, 120, 90)
        self.assertEqual(self.controller.active["name"], "月仔")
        self.assertNotIn("public:1", self.controller.pending)

    def test_unknown_or_changed_target_is_not_clicked(self):
        for name in ("", "银行通知"):
            with self.subTest(name=name):
                self.controller.pending = {"person:1": {"base": "person", "id": "person:1",
                                                       "name": "月仔", "at": 1}}
                self.controller.badges = {"person": {"name": name, "x": 120, "y": 90}}
                with patch.object(main, "click") as click:
                    self.controller.advance()
                click.assert_not_called()

    def test_first_frame_does_not_bypass_authorization_in_switch_mode(self):
        self.assertEqual(self.controller.route_lines("月仔", "月仔", 1, True, "her"), "skip")
        self.assertEqual(self.controller.route_lines("公众号", "公众号", 1, True, "her"), "skip")
        self.config["auto_switch"] = False
        self.assertEqual(self.controller.route_lines("月仔", "月仔", 1, True, "her"), "normal")
        self.assertEqual(self.controller.route_lines("公众号", "公众号", 1, True, "her"), "skip")

    def test_ignored_analysis_never_starts_model_thread(self):
        with patch.object(main.threading, "Thread") as thread:
            self.assertFalse(main.start_analyze("公众号", [("her", "文章标题", None)]))
        thread.assert_not_called()

    def test_ignored_chat_never_sends_even_if_authorized(self):
        self.state.update(chat="公众号", raw_chat="公众号")
        self.config["auto_reply_chats"] = {"wechat:公众号": {"enabled": True}}
        with patch.object(main, "fill_and_send") as send:
            self.assertFalse(main.auto_send_reply("公众号", {"candidates": ["收到"], "best_index": 0}, 0))
        send.assert_not_called()

    def test_ignore_list_is_preserved_when_saving_other_settings(self):
        self.config["ignored_chats"]["kakaotalk"] = ["通知"]
        with patch.object(settings, "_read_env", return_value=""), patch("builtins.open", mock_open()), \
                patch.object(settings.json, "dump") as dump:
            settings.save(debug_view_on=True)
        self.assertEqual(dump.call_args.args[0]["ignored_chats"], self.config["ignored_chats"])
        self.assertTrue(dump.call_args.args[0]["debug_view"])

    def test_ignore_list_save_normalizes_without_overwriting_other_apps(self):
        self.config["ignored_chats"]["kakaotalk"] = ["通知"]
        with patch.object(settings, "_read_env", return_value=""), patch("builtins.open", mock_open()), \
                patch.object(settings.json, "dump") as dump:
            settings.save(ignored_chats_value=[" 营销群 ", "", "营销群", "Ann  a"])
        self.assertEqual(dump.call_args.args[0]["ignored_chats"],
                         {"wechat": ["营销群", "Ann a"], "kakaotalk": ["通知"]})
        self.assertEqual(self.config["ignored_chats"]["wechat"], ["银行通知", "营销群"])

    def test_switch_confirmation_rejects_wrong_chat(self):
        self.controller.phase = "switching"
        self.controller.active = {"id": "normal:1", "base": "normal", "name": "月仔",
                                  "candidate_title": "其他人", "candidate_raw": "其他人"}
        self.controller._confirm_switch()
        self.assertEqual(self.controller.phase, "listening")
        self.assertIsNone(self.controller.active)

    def test_switch_confirmation_preserves_authorized_chat_flow(self):
        self.config["auto_reply_chats"] = {"wechat:月仔": {"enabled": True}}
        self.controller.phase = "switching"
        self.controller.active = {"id": "normal:1", "base": "normal", "name": "月仔",
                                  "candidate_title": "月仔", "candidate_raw": "月仔"}
        self.controller._confirm_switch()
        self.assertEqual(self.controller.phase, "waiting_lines")
        self.assertEqual(self.controller.route_lines("月仔", "月仔", 1, True, "her"), "analyze_fresh")

    def test_ignored_lines_are_not_stored_or_analyzed(self):
        queue = Mock()
        import queue as queue_module

        queue.get_nowait.side_effect = [("lines", "公众号", [("her", None, "文章标题")],
                                        self.state["area"], True, "公众号"), queue_module.Empty]
        with patch.object(main, "q", queue, create=True), patch.object(main, "start_analyze") as analyze:
            main.drain()
        self.assertNotIn("公众号", main.chats)
        analyze.assert_not_called()
        self.overlay.log_message.assert_not_called()


class UnreadNameTests(unittest.TestCase):
    def setUp(self):
        self.badge = {"base": "new", "box": (70, 80, 84, 94), "x": 120, "y": 87}
        self.frame = np.zeros((300, 400, 3), dtype=np.uint8)
        self.area = (200, 60, 400, 280, np.array([0, 0, 0]), 20)

    def test_baseline_unread_is_not_emitted_or_ocr_read(self):
        read_name = Mock(return_value="公众号")
        tracker = UnreadTracker(read_name=read_name)
        with patch("app.capture.unread_badges", return_value=[dict(self.badge)]):
            _, events = tracker.update(self.frame, self.area)
        self.assertEqual(events, [])
        read_name.assert_not_called()

    def test_new_unread_is_named_and_cached(self):
        read_name = Mock(return_value="月仔")
        tracker = UnreadTracker(read_name=read_name)
        with patch("app.capture.unread_badges", side_effect=[[], [dict(self.badge)], [dict(self.badge)]]):
            tracker.update(self.frame, self.area)
            badges, events = tracker.update(self.frame, self.area)
            cached, repeated = tracker.update(self.frame, self.area)
        self.assertEqual(events[0]["name"], "月仔")
        self.assertEqual(badges[0]["name"], "月仔")
        self.assertEqual(cached[0]["name"], "月仔")
        self.assertEqual(repeated, [])
        read_name.assert_called_once()

    def test_unknown_name_is_deferred_until_identified(self):
        read_name = Mock(side_effect=["", "月仔"])
        tracker = UnreadTracker(read_name=read_name)
        with patch("app.capture.unread_badges", side_effect=[[], [dict(self.badge)], [dict(self.badge)]]):
            tracker.update(self.frame, self.area)
            _, unknown = tracker.update(self.frame, self.area)
            _, identified = tracker.update(self.frame, self.area)
        self.assertEqual(unknown, [])
        self.assertEqual(identified[0]["name"], "月仔")
        self.assertEqual(identified[0]["id"], "new:1")

    def test_disappeared_unresolved_badge_is_discarded(self):
        tracker = UnreadTracker(read_name=Mock(return_value=""))
        with patch("app.capture.unread_badges", side_effect=[[], [dict(self.badge)], []]):
            tracker.update(self.frame, self.area)
            tracker.update(self.frame, self.area)
            _, events = tracker.update(self.frame, self.area)
        self.assertEqual(events, [])
        self.assertFalse(tracker.deferred)


class SessionNameOcrTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((300, 400, 3), dtype=np.uint8)
        self.area = (200, 60, 400, 280, np.array([0, 0, 0]), 20)
        self.badge = {"box": (70, 80, 84, 94)}

    def _item(self, text, x=0, y=5, score=0.99):
        return ([[x, y], [x + 40, y], [x + 40, y + 14], [x, y + 14]], text, score)

    def test_reads_name_not_timestamp_or_preview(self):
        result = [self._item("16:58", x=80, y=4), self._item("公众号"),
                  self._item("文章预览", y=24)]
        engine = Mock(return_value=(result, None))
        with patch.object(ocr, "_engine", return_value=engine):
            self.assertEqual(ocr.read_session_name(self.frame, self.area, self.badge), "公众号")
        self.assertLess(engine.call_args.args[0].shape[1], self.area[0])

    def test_rejects_unknown_low_confidence_or_truncated_names(self):
        for result in ([], [self._item("公众号", score=0.7)], [self._item("营销群…")],
                       [self._item("营销群...")]):
            with self.subTest(result=result), patch.object(ocr, "_engine", return_value=Mock(
                    return_value=(result, None))):
                self.assertEqual(ocr.read_session_name(self.frame, self.area, self.badge), "")

    def test_group_count_matches_header_normalization(self):
        with patch.object(ocr, "_engine", return_value=Mock(
                return_value=([self._item("工作群（12）")], None))):
            self.assertEqual(ocr.read_session_name(self.frame, self.area, self.badge), "工作群")

    def test_invalid_crop_does_not_call_ocr(self):
        with patch.object(ocr, "_engine") as engine:
            self.assertEqual(ocr.read_session_name(self.frame, self.area, {"box": (180, 80, 196, 94)}), "")
        engine.assert_not_called()


class OverlayFilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.config = {"ignored_chats": {"wechat": ["营销群"]}, "auto_switch": True}
        for name, value in (("_read", lambda name, default=None: self.config.get(name, default)),
                            ("has_key", lambda: True), ("has_jev_key", lambda: True),
                            ("has_llm_key", lambda: True), ("jev_key", lambda: "demo-key"),
                            ("llm_key", lambda: "demo-key")):
            target = patch.object(settings, name, value)
            target.start()
            self.addCleanup(target.stop)
        from app.overlay import Overlay
        from PySide6.QtWidgets import QWidget

        with patch.object(QWidget, "show"):
            self.overlay = Overlay(on_fill=Mock())
        self.addCleanup(self.overlay.win.close)

    def test_settings_load_ignore_list_and_save_entered_names(self):
        self.overlay._load_settings()
        self.assertEqual(self.overlay.ignoredChatsEdit.toPlainText(), "营销群")
        self.overlay.ignoredChatsEdit.setPlainText("银行通知\n文件传输助手")
        with patch.object(settings, "save") as save:
            self.overlay._save()
        self.assertEqual(save.call_args.kwargs["ignored_chats_value"], ["银行通知", "文件传输助手"])

    def test_ignored_chat_disables_auto_reply_and_suppresses_cached_result(self):
        self.overlay.set_chat("公众号")
        self.overlay.show({"candidates": ["不应该展示"]})
        self.assertFalse(self.overlay.autoReplySwitch.isEnabled())
        self.assertFalse(self.overlay.autoKeyBox.isEnabled())
        self.assertEqual(self.overlay.cands, [])
        self.assertEqual(self.overlay.emptyTitle.text(), "当前会话已忽略")

    def test_regular_chat_retains_auto_reply_controls(self):
        self.overlay.set_chat("月仔")
        self.assertTrue(self.overlay.autoReplySwitch.isEnabled())
        self.assertTrue(self.overlay.autoKeyBox.isEnabled())


if __name__ == "__main__":
    unittest.main()
