import multiprocessing
import unittest
from unittest.mock import Mock, mock_open, patch

import numpy as np

from app import ocr, settings
from app.capture import UnreadTracker
import main


class ConversationFilterTests(unittest.TestCase):
    def setUp(self):
        self.config = {"auto_switch": True, "send_enabled": True,
                       "ignored_chats": {"wechat": ["银行通知", "营销群"]}}
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

    def test_manual_switch_waits_three_seconds_and_buffers_latest_message(self):
        self.config["auto_reply_chats"] = {"wechat:其他人": {"enabled": True}}
        with patch.object(main.time, "monotonic", return_value=100), \
                patch.object(main, "start_analyze", return_value=True) as analyze:
            self._drain_packets([("chat", "其他人", "其他人"),
                                 ("snapshot", self._snapshot(title="其他人", raw_title="其他人",
                                     new=[("her", None, "你好")], visible=[("her", None, "你好")],
                                     history=[("her", None, "你好")])),
                                 ("snapshot", self._snapshot(title="其他人", raw_title="其他人",
                                     new=[("her", None, "在吗")], visible=[("her", None, "在吗")],
                                     history=[("her", None, "你好"), ("her", None, "在吗")]))])
            analyze.assert_not_called()
            self.assertEqual(self.controller.phase, "settling")
            self.assertEqual(self.controller.deadline, 103)
        with patch.object(main.time, "monotonic", return_value=102.9), \
                patch.object(main, "start_analyze") as analyze:
            self.controller.tick()
            analyze.assert_not_called()
        with patch.object(main.time, "monotonic", return_value=103), \
                patch.object(main, "start_analyze", return_value=True) as analyze:
            self.controller.tick()
            self.controller.tick()
        analyze.assert_called_once_with("其他人", [("her", "你好", None), ("her", "在吗", None)])

    def test_repeated_manual_switch_restarts_delay_and_does_not_click_pending_chat(self):
        event = {"base": "pending", "id": "pending:1", "name": "待处理", "at": 100}
        self.controller.pending = {event["id"]: event}
        self.controller.badges = {"pending": {**event, "x": 120, "y": 90}}
        with patch.object(main.time, "monotonic", return_value=100):
            self._drain_packets([("chat", "其他人", "其他人")])
        with patch.object(main.time, "monotonic", return_value=102):
            self._drain_packets([("chat", "第三人", "第三人")])
        with patch.object(main.time, "monotonic", return_value=104), patch.object(main, "click") as click:
            self.controller.tick()
        click.assert_not_called()
        self.assertEqual(self.controller.phase, "settling")
        self.assertEqual(self.controller.deadline, 105)
        self.assertEqual(self.controller.active["title"], "第三人")

    def test_manual_switch_invalidates_old_analysis_and_pending_send(self):
        chat = main.chat_of("月仔")
        chat.update(rev=4, analysis_requested_rev=4, auto_retry_rev=4, media_waiting=True)
        self.state.update(busy=True, rerun=("月仔", []), analysis_task=("月仔", 4))
        self.controller.phase = "analyzing"
        self.controller.active = {"id": "current:4", "title": "月仔", "raw": "月仔"}
        self._drain_packets([("chat", "其他人", "其他人")])
        self.assertEqual(chat["rev"], 5)
        self.assertEqual(chat["analysis_requested_rev"], -1)
        self.assertIsNone(chat["auto_retry_rev"])
        self.assertFalse(chat["media_waiting"])
        self.assertFalse(self.state["busy"])
        self.assertIsNone(self.state["rerun"])
        self.assertIsNone(self.state["analysis_task"])
        with patch.object(main, "auto_send_reply") as send:
            main.auto_send_and_track("月仔", {"candidates": ["旧回复"]}, 4)
        send.assert_not_called()

    def test_cancelled_results_do_not_unlock_or_overwrite_new_analysis(self):
        import queue

        self.state.update(chat="其他人", raw_chat="其他人", busy=True, analysis_task=("其他人", 2))
        self.controller.phase = "analyzing"
        self.controller.active = {"id": "current:2", "title": "其他人", "raw": "其他人"}
        results = queue.Queue()
        results.put(("ok", {"candidates": ["旧回复"]}, "月仔", 1))
        results.put(("err", "旧请求失败", "月仔", 1))
        with patch.object(main, "results", results), patch.object(main, "retry_notices", queue.Queue()), \
                patch.object(main, "update_result", queue.Queue()), \
                patch.object(main, "drain"), patch.object(main, "drain_media"):
            main.tick()
        self.assertTrue(self.state["busy"])
        self.assertEqual(self.state["analysis_task"], ("其他人", 2))
        self.assertEqual(self.controller.phase, "analyzing")
        self.overlay.show.assert_not_called()
        self.overlay.set_busy.assert_not_called()
        self.overlay.set_status.assert_not_called()
        self.assertIsNone(main.chat_of("月仔")["result"])

    def test_auto_switch_waits_three_seconds_even_after_badge_disappears(self):
        self.config["auto_reply_chats"] = {"wechat:其他人": {"enabled": True}}
        event = {"base": "other", "id": "other:1", "name": "其他人", "at": 100}
        self.controller.pending = {event["id"]: event}
        self.controller.badges = {"other": {**event, "x": 120, "y": 90}}
        with patch.object(main.time, "monotonic", return_value=100), patch.object(main, "click") as click:
            self.controller.advance()
        click.assert_called_once_with(1, 120, 90)
        with patch.object(main.time, "monotonic", return_value=101), \
                patch.object(main, "start_analyze", return_value=True) as analyze:
            self._drain_packets([("chat", "其他人", "其他人"), ("unread_state", []),
                                 ("snapshot", self._snapshot(title="其他人", raw_title="其他人"))])
            analyze.assert_not_called()
            self.assertEqual(self.controller.deadline, 104)
            self._settle_current_chat()
        analyze.assert_called_once_with("其他人", [("her", "是吗 哈哈", None)])

    def test_same_chat_notification_and_snapshot_do_not_restart_settle_timer(self):
        self.config["auto_reply_chats"] = {"wechat:其他人": {"enabled": True}}
        with patch.object(main.time, "monotonic", return_value=100):
            self._drain_packets([("chat", "其他人", "其他人")])
        with patch.object(main.time, "monotonic", return_value=102), \
                patch.object(main, "start_analyze", return_value=True) as analyze:
            self._drain_packets([("chat", "其他人", "其他人"),
                                 ("snapshot", self._snapshot(title="其他人", raw_title="其他人"))])
            self.assertEqual(self.controller.deadline, 103)
            self._settle_current_chat()
        analyze.assert_called_once()

    def test_old_send_stays_invalid_after_switching_away_and_back(self):
        self.config["auto_reply_chats"] = {"wechat:月仔": {"enabled": True}}
        chat = main.chat_of("月仔")
        chat["rev"] = 4
        self.controller.phase = "waiting_send"
        self.controller.active = {"id": "current:4", "title": "月仔", "raw": "月仔"}
        self.controller.send_revision = 4
        self._drain_packets([("chat", "其他人", "其他人"), ("chat", "月仔", "月仔")])
        with patch.object(main, "fill_and_send") as send:
            main.auto_send_and_track("月仔", {"candidates": ["旧回复"]}, 4)
            self.assertFalse(main.auto_send_reply("月仔", {"candidates": ["旧回复"]}, 4))
        send.assert_not_called()
        self.assertEqual(chat["rev"], 5)
        self.assertEqual(self.controller.phase, "settling")

    def test_cancelled_retry_does_not_make_another_model_request(self):
        import queue
        from core import retry

        main.chat_of("月仔")["rev"] = 4
        self._drain_packets([("chat", "其他人", "其他人")])
        request = Mock()

        def limited(*args, **kwargs):
            return retry.call_with_retry(request)

        with patch.object(main, "results", queue.Queue()), patch.object(main, "analyze", side_effect=limited), \
                patch.object(settings, "general_judge_config", return_value={
                    "provider": "deepseek", "model": "", "base_url": "", "api_key": "k"}):
            main.analyze_bg([("her", "你好", None)], "月仔", 4)
        request.assert_not_called()
        self.assertEqual(self.controller.phase, "settling")

    def test_unconfirmed_snapshot_at_settle_end_waits_for_fresh_confirmation(self):
        self.config["auto_reply_chats"] = {"wechat:月仔": {"enabled": True}}
        with patch.object(main, "start_analyze", return_value=True) as analyze:
            self._drain_packets([("chat", "其他人", "其他人"), ("chat", "月仔", "月仔"),
                                 ("snapshot", self._snapshot(new=[("her", None, "是吗 哈哈")])),
                                 ("snapshot", self._snapshot(title_confirmed=False))])
            self._settle_current_chat()
            analyze.assert_not_called()
            self.assertEqual(self.controller.phase, "waiting_lines")
            self._drain_packets([("snapshot", self._snapshot())])
        analyze.assert_called_once()

    def test_history_correction_and_discontinuity_keep_switch_protection(self):
        self.config["auto_reply_chats"] = {"wechat:月仔": {"enabled": True}}
        for changes in ({"corrected": True}, {"discontinuous": True}):
            with self.subTest(changes=changes), patch.object(main.time, "monotonic", return_value=100), \
                    patch.object(main, "start_analyze", return_value=True) as analyze:
                self._drain_packets([("chat", "其他人", "其他人"), ("chat", "月仔", "月仔"),
                                     ("snapshot", self._snapshot(**changes))])
                self.assertEqual(self.controller.phase, "settling")
                self.assertEqual(self.controller.deadline, 103)
                analyze.assert_not_called()

    def test_models_cannot_bypass_settle_period_through_direct_entry(self):
        self.config["auto_reply_chats"] = {"wechat:月仔": {"enabled": True}}
        self._drain_packets([("chat", "其他人", "其他人"), ("chat", "月仔", "月仔")])
        with patch.object(main.threading, "Thread") as thread, patch.object(main, "fill_and_send") as send:
            self.assertFalse(main.start_analyze("月仔", [("her", "你好", None)]))
            self.assertFalse(main.auto_send_reply("月仔", {"candidates": ["旧回复"]}, main.chat_of("月仔")["rev"]))
        thread.assert_not_called()
        send.assert_not_called()

    def test_media_jobs_are_deferred_and_discarded_on_another_switch(self):
        self.config["auto_reply_chats"] = {"wechat:月仔": {"enabled": True}}
        job = {"message_id": 7, "kind": "image", "image": b"image"}
        with patch.object(main.threading, "Thread") as thread, patch.object(main, "media_inflight", set()):
            self._drain_packets([("chat", "其他人", "其他人"), ("chat", "月仔", "月仔"),
                                 ("snapshot", self._snapshot(media_jobs=[job])),
                                 ("snapshot", self._snapshot())])
            thread.assert_not_called()
            self.assertEqual(list(self.controller.early_media_jobs), [7])
            self._drain_packets([("chat", "第三人", "第三人")])
            self.assertFalse(self.controller.early_media_jobs)
            thread.assert_not_called()

    def test_deferred_media_job_starts_once_after_settle_period(self):
        self.config["auto_reply_chats"] = {"wechat:月仔": {"enabled": True}}
        job = {"message_id": 7, "kind": "image", "image": b"image"}
        with patch.object(main.threading, "Thread") as thread, patch.object(main, "media_inflight", set()), \
                patch.object(main.media_slots, "acquire", return_value=False), \
                patch.object(main, "start_analyze", return_value=True):
            self._drain_packets([("chat", "其他人", "其他人"), ("chat", "月仔", "月仔"),
                                 ("snapshot", self._snapshot(media_jobs=[job])),
                                 ("snapshot", self._snapshot())])
            thread.assert_not_called()
            self._settle_current_chat()
        thread.assert_called_once()
        self.assertEqual(thread.call_args.kwargs["target"], main.recognize_media_bg)
        self.assertFalse(self.controller.early_media_jobs)

    def test_read_only_manual_switch_collects_without_model_calls(self):
        self.config["analysis_enabled"] = False
        with patch.object(main, "start_analyze") as analyze, patch.object(main.threading, "Thread") as thread:
            self._drain_packets([("chat", "其他人", "其他人"), ("chat", "月仔", "月仔"),
                                 ("snapshot", self._snapshot(new=[("her", None, "是吗 哈哈")]))])
            self._settle_current_chat()
        analyze.assert_not_called()
        thread.assert_not_called()
        self.assertEqual(self.controller.phase, "listening")
        self.assertEqual(list(main.chat_of("月仔")["history"]), [("her", "是吗 哈哈", None)])

    def test_processing_setting_change_does_not_end_settle_period_early(self):
        with patch.object(main.time, "monotonic", return_value=100):
            self._drain_packets([("chat", "其他人", "其他人"), ("chat", "月仔", "月仔")])
        self.config["auto_reply_chats"] = {"wechat:月仔": {"enabled": True}}
        with patch.object(main.time, "monotonic", return_value=101), \
                patch.object(main, "start_analyze", return_value=True) as analyze:
            main.on_processing_changed()
            self._drain_packets([("snapshot", self._snapshot())])
            self.assertEqual(self.controller.phase, "settling")
            self.assertEqual(self.controller.deadline, 103)
            analyze.assert_not_called()
            self._settle_current_chat()
        analyze.assert_called_once()

    def test_current_result_is_accepted_when_cancelled_results_arrive_before_and_after_it(self):
        import queue

        self.state.update(chat="其他人", raw_chat="其他人", busy=True, analysis_task=("其他人", 2))
        self.overlay.current_chat.return_value = "其他人"
        main.chat_of("其他人")["rev"] = 2
        self.controller.phase = "analyzing"
        self.controller.active = {"id": "current:2", "title": "其他人", "raw": "其他人"}
        result = {"candidates": ["新回复"]}
        results = queue.Queue()
        results.put(("ok", {"candidates": ["旧回复"]}, "月仔", 1))
        results.put(("ok", result, "其他人", 2))
        results.put(("err", "旧请求失败", "月仔", 1))
        with patch.object(main, "results", results), patch.object(main, "retry_notices", queue.Queue()), \
                patch.object(main, "update_result", queue.Queue()), \
                patch.object(main, "drain"), patch.object(main, "drain_media"):
            main.tick()
        self.overlay.show.assert_called_once_with(result)
        self.assertEqual(main.chat_of("其他人")["result"], result)
        self.assertEqual(self.controller.phase, "waiting_send")
        self.assertFalse(self.state["busy"])
        self.assertIsNone(self.state["analysis_task"])
        self.assertFalse(any(call.args[1] == "error" for call in self.overlay.set_status.call_args_list))

    def test_analysis_error_is_not_overwritten_by_generic_abandonment(self):
        import queue

        self.controller.phase = "analyzing"
        self.controller.active = {"id": "current:1", "title": "月仔", "raw": "月仔"}
        results = queue.Queue()
        results.put(("err", "分析失败: 起草 HTTP 429: 被限流", "月仔", 0))
        with patch.object(main, "results", results), \
                patch.object(main, "update_result", queue.Queue()), \
                patch.object(main, "drain"), patch.object(main, "drain_media"):
            main.tick()
        text, kind = self.overlay.set_status.call_args.args
        self.assertIn("起草 HTTP 429", text)
        self.assertEqual(kind, "error")
        self.assertNotIn("自动处理未完成", text)
        self.assertEqual(self.controller.phase, "listening")

    def test_stale_analysis_result_reports_both_revisions(self):
        import queue

        self.controller.phase = "analyzing"
        self.controller.active = {"id": "current:1", "title": "月仔", "raw": "月仔"}
        main.chat_of("月仔")["rev"] = 2
        results = queue.Queue()
        results.put(("ok", {"candidates": ["收到"]}, "月仔", 1))
        with patch.object(main, "results", results), \
                patch.object(main, "update_result", queue.Queue()), \
                patch.object(main, "drain"), patch.object(main, "drain_media"):
            main.tick()
        text, kind = self.overlay.set_status.call_args.args
        self.assertIn("版本 1 → 2", text)
        self.assertIn("过期", text)
        self.assertEqual(kind, "warning")
        self.assertEqual(self.controller.phase, "listening")

    def test_missing_model_configuration_keeps_specific_block_reason(self):
        self.config["auto_reply_chats"] = {"wechat:月仔": {"enabled": True}}
        with patch.object(settings, "models_ready", return_value=False):
            self._drain_packets([("lines", "月仔", [("her", None, "你好")],
                                  self.state["area"], False, "月仔")])
        text, kind = self.overlay.set_status.call_args.args
        self.assertIn("模型配置不完整", text)
        self.assertEqual(kind, "warning")
        self.assertNotIn("自动处理未完成", text)

    def test_analysis_passes_configured_local_timeout_to_engine(self):
        import queue

        main.chat_of("月仔")["rev"] = 4
        with patch.object(main, "results", queue.Queue()), \
                patch.object(main, "analyze", return_value={"candidates": []}) as analyze, \
                patch.object(settings, "laya_timeout", return_value=90), \
                patch.object(settings, "general_judge_config", return_value={
                    "provider": "deepseek", "model": "", "base_url": "", "api_key": "k"}):
            main.analyze_bg([("her", "你好", None)], "月仔", 4)
            self.assertEqual(main.results.get_nowait()[0], "ok")
        self.assertEqual(analyze.call_args.kwargs["laya_timeout"], 90)

    def test_analysis_retry_is_reported_to_ui_queue(self):
        from core import retry
        from core.jev_client import JevError

        main.chat_of("月仔")["rev"] = 4
        calls = Mock(side_effect=[JevError("限流", 429), "ok"])

        def limited(*args, **kwargs):
            retry.call_with_retry(calls, what="起草")
            return {"candidates": []}

        with patch.object(main, "analyze", side_effect=limited), patch.object(retry.time, "sleep", Mock()), \
                patch.object(settings, "general_judge_config", return_value={
                    "provider": "deepseek", "model": "", "base_url": "", "api_key": "k"}):
            main.analyze_bg([("her", "你好", None)], "月仔", 4)
        self.assertEqual(main.results.get_nowait()[0], "ok")  # 清掉结果，避免污染其他用例共用的全局队列
        title, revision, text = main.retry_notices.get_nowait()
        self.assertEqual((title, revision), ("月仔", 4))
        self.assertIn("起草被限流", text)
        self.assertIn("第 1/5 次", text)

    def test_stale_analysis_does_not_wait_for_model_retries(self):
        from core import retry
        from core.jev_client import JevError

        main.chat_of("月仔")["rev"] = 5

        def limited(*args, **kwargs):
            return retry.call_with_retry(Mock(side_effect=JevError("限流", 429)))

        sleep = Mock()
        with patch.object(main, "analyze", side_effect=limited), patch.object(retry.time, "sleep", sleep), \
                patch.object(settings, "general_judge_config", return_value={
                    "provider": "deepseek", "model": "", "base_url": "", "api_key": "k"}):
            main.analyze_bg([("her", "你好", None)], "月仔", 4)
        sleep.assert_not_called()
        kind, message, title, revision = main.results.get_nowait()
        self.assertEqual((kind, title, revision), ("err", "月仔", 4))
        self.assertIn("会话内容已更新", message)

    def test_no_reply_result_completes_round_without_sending_or_failure(self):
        import queue

        self.config["auto_reply_chats"] = {"wechat:月仔": {"enabled": True}}
        chat = main.chat_of("月仔")
        chat["rev"] = 3
        chat["auto_retry_rev"] = 3
        self.controller.phase = "analyzing"
        self.controller.active = {"id": "current:3", "title": "月仔", "raw": "月仔"}
        result = {"reply_required": False, "no_reply_reason": "对方已平和结束话题，本轮无需回复",
                  "candidates": [], "answers": {}}
        results = queue.Queue()
        results.put(("ok", result, "月仔", 3))
        with patch.object(main, "results", results), \
                patch.object(main, "update_result", queue.Queue()), \
                patch.object(main, "drain"), patch.object(main, "drain_media"), \
                patch.object(self.controller, "result_abandoned") as abandoned, \
                patch.object(main, "auto_send_reply") as send:
            main.tick()
        abandoned.assert_not_called()
        send.assert_not_called()
        self.assertEqual(self.controller.phase, "listening")
        self.assertIsNone(self.controller.active)
        self.assertIsNone(chat["auto_retry_rev"])
        self.assertEqual(chat["analysis_requested_rev"], 3)
        self.assertEqual(chat["auto_sent_rev"], -1)
        self.assertTrue(settings.send_enabled())
        self.assertFalse(any(call.args[1] == "error" for call in self.overlay.set_status.call_args_list))
        self.assertIn("无需回复", self.overlay.set_status.call_args.args[0])
        self.assertEqual(self.controller.route_lines("月仔", "月仔", 4, False, "her"), "analyze")

    def test_no_reply_result_blocks_send_even_if_candidates_are_present(self):
        self.config["auto_reply_chats"] = {"wechat:月仔": {"enabled": True}}
        result = {"reply_required": False, "candidates": ["回头聊"], "best_index": 0}
        with patch.object(main, "fill_and_send") as send:
            self.assertFalse(main.auto_send_reply("月仔", result, 0))
        send.assert_not_called()
        self.assertEqual(main.chat_of("月仔")["auto_sent_rev"], -1)

    def test_old_send_callback_is_blocked_after_no_reply_completion(self):
        self.config["auto_reply_chats"] = {"wechat:月仔": {"enabled": True}}
        self.controller.complete_without_reply("月仔", 0, "本轮无需回复")
        with patch.object(main, "fill_and_send") as send:
            self.assertFalse(main.auto_send_reply("月仔", {"candidates": ["回头聊"], "best_index": 0}, 0))
        send.assert_not_called()

    def test_no_reply_cached_message_does_not_restart_analysis(self):
        self.config["auto_reply_chats"] = {"wechat:月仔": {"enabled": True}}
        chat = main.chat_of("月仔")
        chat["history"].append(("her", "是吗 哈哈", None))
        chat["analysis_requested_rev"] = chat["rev"]
        chat["result"] = {"reply_required": False, "candidates": []}
        with patch.object(main, "start_analyze") as analyze:
            self.assertFalse(main.resume_cached_reply(self._snapshot()))
        analyze.assert_not_called()

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

    def test_existing_unread_switches_after_current_reply_without_new_messages(self):
        tracker = UnreadTracker(read_name=Mock(return_value="其他人"))
        badge = {"base": "other", "box": (70, 80, 84, 94), "x": 120, "y": 90}
        self.controller.phase = "verifying"
        self.controller.active = {"id": "current:1", "title": "月仔", "raw": "月仔"}
        self.controller.expected_text = "收到"
        with patch("app.capture.unread_badges", return_value=[badge]), \
                patch.object(main, "click") as click:
            badges, events = tracker.update(None, None)
            self.controller.on_unread_state(badges)
            for event in events:
                self.controller.on_unread(event)
            self.controller.advance()
            click.assert_not_called()
            badges, repeated = tracker.update(None, None)
            self.assertEqual(repeated, [])
            self.controller.on_unread_state(badges)
            self.assertTrue(self.controller.observe("月仔", [("me", None, "收到")]))
            self.controller.tick()
        click.assert_called_once_with(1, 120, 90)
        self.assertEqual(self.controller.phase, "switching")
        self.assertEqual(self.controller.active["name"], "其他人")

    def test_switch_skips_blocked_target_and_uses_next_valid_candidate(self):
        import time

        now = time.monotonic()
        for base, name, at in (("normal", "其他人", now), ("public", "公众号", now + 1)):
            event = {"base": base, "id": base + ":1", "name": name, "at": at}
            self.controller.pending[event["id"]] = event
            self.controller.badges[base] = {**event, "x": 120, "y": 90}
        with patch.object(main, "click") as click:
            self.controller.advance()
        click.assert_called_once_with(1, 120, 90)
        self.assertEqual(self.controller.active["name"], "其他人")
        self.assertNotIn("public:1", self.controller.pending)

    def test_unread_for_current_chat_does_not_click_or_enter_switching_phase(self):
        event = {"base": "current", "id": "current:1", "name": "月仔", "at": 1}
        self.controller.pending = {event["id"]: event}
        self.controller.badges = {"current": {**event, "x": 120, "y": 90}}
        with patch.object(main, "click") as click:
            self.controller.advance()
        click.assert_not_called()
        self.assertNotIn(event["id"], self.controller.pending)
        self.assertEqual(self.controller.phase, "listening")
        self.assertEqual(self.controller.processed[-1], event["id"])

    def test_current_chat_unread_is_skipped_before_choosing_other_pending_chat(self):
        current = {"base": "current", "id": "current:1", "name": "月仔", "at": 2}
        other = {"base": "other", "id": "other:1", "name": "其他人", "at": 1}
        self.controller.pending = {current["id"]: current, other["id"]: other}
        self.controller.badges = {
            "current": {**current, "x": 120, "y": 90},
            "other": {**other, "x": 120, "y": 180},
        }
        with patch.object(main, "click") as click:
            self.controller.advance()
        click.assert_called_once_with(1, 120, 180)
        self.assertNotIn(current["id"], self.controller.pending)
        self.assertEqual(self.controller.active["name"], "其他人")

    def test_same_snapshot_prioritizes_top_conversation_not_hash_order(self):
        for base, name, y in (("a-friend", "阿梅", 100), ("z-other", "其他人", 300)):
            event = {"base": base, "id": base + ":1", "name": name, "at": 1}
            self.controller.pending[event["id"]] = event
            self.controller.badges[base] = {**event, "x": 120, "y": y}
        with patch.object(main, "click") as click:
            self.controller.advance()
        click.assert_called_once_with(1, 120, 100)
        self.assertEqual(self.controller.active["name"], "阿梅")

    def test_unknown_or_changed_target_is_not_clicked(self):
        for name in ("", "银行通知"):
            with self.subTest(name=name):
                self.controller.pending = {"person:1": {"base": "person", "id": "person:1",
                                                       "name": "月仔", "at": 1}}
                self.controller.badges = {"person": {"name": name, "x": 120, "y": 90}}
                with patch.object(main, "click") as click:
                    self.controller.advance()
                click.assert_not_called()

    def test_waiting_send_timeout_keeps_active_chat_locked(self):
        import time

        event = {"base": "other", "id": "other:1", "name": "其他人", "at": time.monotonic()}
        self.controller.phase = "waiting_send"
        self.controller.active = {"id": "current:1", "title": "月仔", "raw": "月仔",
                                  "name": "月仔", "revision": 1}
        self.controller.send_revision = 1
        self.controller.deadline = 1
        self.controller.pending = {event["id"]: event}
        self.controller.badges = {"other": {**event, "x": 120, "y": 90}}
        with patch.object(main.time, "monotonic", return_value=2), \
                patch.object(main, "click") as click:
            self.controller.tick()
        click.assert_not_called()
        self.assertEqual(self.controller.phase, "waiting_send")
        self.assertEqual(self.controller.active["title"], "月仔")
        self.assertIn(event["id"], self.controller.pending)
        self.overlay.set_status.assert_called_with(
            "自动回复仍在等待发送，暂不切换其他会话", "warning")

    def test_expired_send_callback_is_cancelled_after_session_release(self):
        chat = main.chat_of("月仔")
        chat["rev"] = 3
        chat["result"] = {"candidates": ["收到"], "best_index": 0,
                          "ranking_valid": True, "scores": [0.8]}
        self.controller.phase = "listening"
        self.controller.active = None
        with patch.object(main, "auto_send_reply") as send:
            main.auto_send_and_track("月仔", chat["result"], 3)
        send.assert_not_called()

    def test_active_send_callback_requires_matching_session_and_revision(self):
        chat = main.chat_of("月仔")
        chat["rev"] = 3
        self.controller.phase = "waiting_send"
        self.controller.active = {"id": "current:1", "title": "月仔", "raw": "月仔"}
        self.controller.send_revision = 2
        with patch.object(main, "auto_send_reply") as send:
            main.auto_send_and_track("月仔", {"candidates": ["收到"], "best_index": 0}, 2)
        send.assert_not_called()

    def test_failed_auto_send_keeps_next_chat_pending_until_manual_switch(self):
        event = {"base": "other", "id": "other:1", "name": "其他人", "at": 1}
        self.controller.phase = "waiting_send"
        self.controller.active = {"id": "current:1", "title": "月仔", "raw": "月仔",
                                  "revision": 1}
        self.controller.send_revision = 1
        self.controller.pending = {event["id"]: event}
        self.controller.badges = {"other": {**event, "x": 120, "y": 90}}
        self.controller.on_send_attempt("月仔", 1, False)
        self.assertEqual(self.controller.phase, "send_failed")
        with patch.object(main, "click") as click:
            self.controller.advance()
            click.assert_not_called()
            self.state.update(chat="手动会话", raw_chat="手动会话")
            self.controller.on_chat("手动会话", "手动会话")
            self.controller.advance()
            click.assert_not_called()
            self._settle_current_chat()
            self.controller.advance()
        click.assert_called_once_with(1, 120, 90)

    def test_verification_timeout_releases_and_switches_to_next_chat(self):
        event = {"base": "other", "id": "other:1", "name": "其他人", "at": 1}
        self.controller.phase = "verifying"
        self.controller.active = {"id": "current:1", "title": "月仔", "raw": "月仔",
                                  "name": "月仔"}
        self.controller.pending = {event["id"]: event}
        self.controller.badges = {"other": {**event, "x": 120, "y": 90}}
        self.controller.deadline = 1
        with patch.object(main.time, "monotonic", return_value=2), \
                patch.object(main, "click") as click:
            self.controller.tick()
            click.assert_called_once_with(1, 120, 90)
        self.assertEqual(self.controller.phase, "switching")
        self.assertEqual(self.controller.active["name"], "其他人")
        self.assertNotIn(event["id"], self.controller.pending)
        self.overlay.set_status.assert_called_with("检测到未读消息，正在切换会话", "busy")

    def test_pending_other_chat_never_switches_during_active_reply_phases(self):
        event = {"base": "other", "id": "other:1", "name": "其他人", "at": 1}
        for phase in ("switching", "waiting_lines", "analyzing", "waiting_send", "verifying"):
            with self.subTest(phase=phase):
                self.controller.phase = phase
                self.controller.active = {"id": "current:1", "title": "月仔", "raw": "月仔",
                                          "name": "月仔", "previous_title": "月仔",
                                          "previous_raw": "月仔"}
                self.controller.pending = {event["id"]: dict(event)}
                self.controller.badges = {"other": {**event, "x": 120, "y": 90}}
                with patch.object(main, "click") as click:
                    self.controller.advance()
                click.assert_not_called()
                self.assertIn(event["id"], self.controller.pending)

    def test_unconfigured_conversation_releases_without_waiting_to_send(self):
        self.controller.phase = "switching"
        self.controller.active = {"id": "person:1", "base": "person", "name": "月仔",
                                  "candidate_title": "月仔", "candidate_raw": "月仔"}
        self.controller._confirm_switch()
        self._settle_current_chat()
        self.assertEqual(self.controller.phase, "listening")
        self.assertIsNone(self.controller.active)
        self.overlay.set_status.assert_called_with("已切换到「月仔」，该会话未开启自动回复", "idle")

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

    def test_debug_mode_disables_sending_for_authorized_normal_chat(self):
        self.config["auto_reply_chats"] = {"wechat:月仔": {"enabled": True}}
        with patch.dict(main.os.environ, {"JEV_DISABLE_AUTO_SEND": "1"}), \
                patch.object(main, "fill_and_send") as send:
            self.assertFalse(main.auto_send_reply("月仔", {"candidates": ["收到"], "best_index": 0}, 0))
        send.assert_not_called()

    def test_valid_ranking_keeps_existing_auto_send_flow(self):
        self.config["auto_reply_chats"] = {"wechat:月仔": {"enabled": True}}
        with patch.object(main, "fill_and_send") as send:
            sent = main.auto_send_reply(
                "月仔", {"candidates": ["收到"], "best_index": 0,
                         "ranking_valid": True, "scores": [0.8]}, 0)
        self.assertTrue(sent)
        send.assert_called_once()

    def test_unranked_first_choice_auto_sends_only_for_authorized_chat(self):
        self.config["auto_reply_chats"] = {"wechat:月仔": {"enabled": True}}
        result = {"candidates": ["第一条候选", "第二条候选"], "best_index": 0,
                  "scores": [0.0, 0.0], "ranking_valid": False, "auto_send_fallback": True}
        with patch.object(main, "fill_and_send") as send:
            self.assertTrue(main.auto_send_reply("月仔", result, 0))
        send.assert_called_once_with(1, self.state["area"], "第一条候选", "enter")
        self.assertEqual(main.chat_of("月仔")["last_auto_sent_text"], "第一条候选")
        self.assertNotIn("0%", str(self.overlay.set_status.call_args))

    def test_unranked_first_choice_respects_per_chat_authorization(self):
        result = {"candidates": ["第一条候选"], "best_index": 0,
                  "ranking_valid": False, "auto_send_fallback": True}
        with patch.object(main, "fill_and_send") as send:
            self.assertFalse(main.auto_send_reply("月仔", result, 0))
        send.assert_not_called()

    def test_unranked_result_uses_first_candidate_without_extra_fallback_flag(self):
        self.config["auto_reply_chats"] = {"wechat:月仔": {"enabled": True}}
        with patch.object(main, "fill_and_send") as send:
            self.assertTrue(main.auto_send_reply(
                "月仔", {"candidates": ["收到"], "best_index": 0, "ranking_valid": False}, 0))
        send.assert_called_once_with(1, self.state["area"], "收到", "enter")

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

    def test_each_timeout_names_the_stage_that_stalled(self):
        phases = {
            "switching": "微信会话切换确认超时，继续监听新消息",
            "waiting_lines": "会话已切换，但 OCR 尚未确认可见文本消息，继续监听",
            "waiting_send": "自动回复仍在等待发送，暂不切换其他会话",
            "verifying": "发送已执行但未确认，继续处理其他会话",
        }
        for phase, expected in phases.items():
            with self.subTest(phase=phase):
                self.controller.phase = phase
                self.controller.active = {"id": phase, "title": "月仔", "name": "月仔"}
                self.controller.deadline = 1
                with patch.object(main.time, "monotonic", return_value=2):
                    self.controller.tick()
                self.overlay.set_status.assert_called_with(expected, "warning")
                expected_phase = "waiting_send" if phase == "waiting_send" else "listening"
                self.assertEqual(self.controller.phase, expected_phase)

    def test_switch_confirmation_follows_manual_chat_without_target_mismatch_warning(self):
        self.controller.phase = "switching"
        self.controller.active = {"id": "normal:1", "base": "normal", "name": "月仔",
                                  "candidate_title": "其他人", "candidate_raw": "其他人"}
        self.state.update(chat="其他人", raw_chat="其他人")
        self.controller._confirm_switch()
        self.assertEqual(self.controller.phase, "settling")
        self.assertEqual(self.controller.active["title"], "其他人")
        self.assertIn("normal:1", self.controller.processed_set)
        self.assertNotIn("目标不符", self.overlay.set_status.call_args.args[0])
        self._settle_current_chat()
        self.assertEqual(self.controller.phase, "listening")
        self.assertIsNone(self.controller.active)

    def test_switch_confirmation_preserves_authorized_chat_flow(self):
        self.config["auto_reply_chats"] = {"wechat:月仔": {"enabled": True}}
        self.controller.phase = "switching"
        self.controller.active = {"id": "normal:1", "base": "normal", "name": "月仔",
                                  "candidate_title": "月仔", "candidate_raw": "月仔"}
        self.controller._confirm_switch()
        self.assertEqual(self.controller.phase, "settling")
        self._settle_current_chat()
        self.assertEqual(self.controller.phase, "waiting_lines")
        self.assertEqual(self.controller.route_lines("月仔", "月仔", 1, True, "her"), "analyze_fresh")

    def test_message_arriving_before_switch_confirmation_is_replayed_once(self):
        self.config["auto_reply_chats"] = {"wechat:月仔": {"enabled": True}}
        self.state.update(chat="月仔", raw_chat="月仔")
        self.controller.phase = "switching"
        self.controller.active = {"id": "person:1", "base": "person", "name": "月仔",
                                  "previous_title": "群聊", "previous_raw": "群聊"}
        self.controller.badges = {"person": {"name": "月仔"}}
        with patch.object(main, "start_analyze", return_value=True) as analyze:
            self.controller.on_chat("月仔", "月仔")
            self.assertEqual(self.controller.route_lines("月仔", "月仔", 1, True, "her"), "skip")
            chat = main.chat_of("月仔")
            chat["history"].append(("her", "刚发来的消息", None))
            chat["rev"] = 1
            self.controller.on_unread_state([])
            analyze.assert_not_called()
            self._settle_current_chat()
        analyze.assert_called_once_with("月仔", [("her", "刚发来的消息", None)])
        self.assertEqual(self.controller.phase, "analyzing")
        self.assertEqual(self.controller.active["revision"], 1)
        self.assertNotIn("early_lines", self.controller.active)

    def test_drain_replays_buffered_message_once_after_target_confirmation(self):
        import queue as queue_module

        self.config["auto_reply_chats"] = {"wechat:月仔": {"enabled": True}}
        self.state.update(chat="群聊", raw_chat="群聊")
        self.controller.phase = "switching"
        self.controller.active = {"id": "person:1", "base": "person", "name": "月仔",
                                  "previous_title": "群聊", "previous_raw": "群聊"}
        messages = [
            ("unread_state", [{"base": "person", "name": "月仔", "x": 120, "y": 90}]),
            ("chat", "月仔", "月仔"),
            ("snapshot", self._snapshot(new=[("her", None, "刚发来的消息")],
                                        visible=[("her", None, "刚发来的消息")],
                                        history=[("her", None, "刚发来的消息")], initial=True)),
            ("unread_state", []),
            queue_module.Empty,
        ]
        q = Mock()
        q.get_nowait.side_effect = messages
        with patch.object(main, "q", q, create=True), \
                patch.object(main, "start_analyze", return_value=True) as analyze:
            main.drain()
            analyze.assert_not_called()
            self._settle_current_chat()
        analyze.assert_called_once_with("月仔", [("her", "刚发来的消息", None)])
        self.assertEqual(self.controller.phase, "analyzing")
        self.assertEqual(main.chat_of("月仔")["auto_sent_rev"], -1)

    def test_buffered_message_is_discarded_after_newer_reply(self):
        self.config["auto_reply_chats"] = {"wechat:月仔": {"enabled": True}}
        self.state.update(chat="月仔", raw_chat="月仔")
        self.controller.phase = "switching"
        self.controller.active = {"id": "person:1", "base": "person", "name": "月仔",
                                  "previous_title": "群聊", "previous_raw": "群聊"}
        self.controller.badges = {"person": {"name": "月仔"}}
        self.controller.on_chat("月仔", "月仔")
        self.controller.route_lines("月仔", "月仔", 1, True, "her")
        chat = main.chat_of("月仔")
        chat["history"].extend([("her", "新问题", None), ("me", "已回复", None)])
        chat["rev"] = 2
        with patch.object(main, "start_analyze") as analyze:
            self.controller.on_unread_state([])
            self._settle_current_chat()
        analyze.assert_not_called()
        self.assertEqual(self.controller.phase, "listening")
        self.assertIsNone(self.controller.active)

    def test_early_message_from_wrong_session_is_not_replayed(self):
        self.config["auto_reply_chats"] = {"wechat:月仔": {"enabled": True}}
        self.state.update(chat="月仔", raw_chat="月仔")
        self.controller.phase = "switching"
        self.controller.active = {"id": "person:1", "base": "person", "name": "月仔",
                                  "previous_title": "群聊", "previous_raw": "群聊"}
        self.controller.badges = {"person": {"name": "月仔"}}
        with patch.object(main, "start_analyze", return_value=True) as analyze:
            self.controller.route_lines("另一个会话", "另一个会话", 1, False, "her")
            chat = main.chat_of("月仔")
            chat["history"].append(("her", "其他会话的消息", None))
            chat["rev"] = 1
            self.controller.on_chat("月仔", "月仔")
            self.controller.on_unread_state([])
            self._settle_current_chat()
        analyze.assert_not_called()
        self.assertEqual(self.controller.phase, "waiting_lines")

    def _settle_current_chat(self):
        self.assertEqual(self.controller.phase, "settling")
        with patch.object(main.time, "monotonic", return_value=self.controller.deadline):
            self.controller.tick()

    def _drain_packets(self, packets):
        import queue as queue_module

        q = Mock()
        q.get_nowait.side_effect = packets + [queue_module.Empty]
        with patch.object(main, "q", q, create=True):
            main.drain()

    def _snapshot(self, new=None, visible=None, **changes):
        new = [] if new is None else new
        snapshot = {"title": "月仔", "raw_title": "月仔", "new": new,
                    "visible": visible if visible is not None else [("her", None, "是吗 哈哈")],
                    "history": [("her", None, "是吗 哈哈")], "area": self.state["area"],
                    "read_at": 100, "initial": False, "title_confirmed": True, "status": "read"}
        snapshot.update(changes)
        return snapshot

    def test_cached_snapshot_completes_switch_read_without_false_timeout_or_model_call(self):
        self.controller.phase = "waiting_lines"
        self.controller.active = {"id": "person:1", "title": "月仔", "raw": "月仔",
                                  "authorized": True}
        with patch.object(main, "start_analyze") as analyze:
            self._drain_packets([("snapshot", self._snapshot())])
        analyze.assert_not_called()
        self.assertEqual(self.controller.phase, "listening")
        self.overlay.set_status.assert_called_with(
            "已读取「月仔」的可见消息，与本地历史一致，暂无新增", "idle")
        self.assertEqual(list(main.chat_of("月仔")["history"]), [("her", "是吗 哈哈", None)])

    def test_authorized_cached_snapshot_before_confirmation_is_replied_to(self):
        self.config["auto_reply_chats"] = {"wechat:月仔": {"enabled": True}}
        self.controller.phase = "switching"
        self.controller.active = {"id": "person:1", "base": "person", "name": "月仔",
                                  "previous_title": "群聊", "previous_raw": "群聊"}
        self.controller.badges = {"person": {"name": "月仔"}}
        with patch.object(main, "start_analyze", return_value=True) as analyze:
            self._drain_packets([("chat", "月仔", "月仔"), ("snapshot", self._snapshot()),
                                 ("unread_state", [])])
            analyze.assert_not_called()
            self._settle_current_chat()
        analyze.assert_called_once_with("月仔", [("her", "是吗 哈哈", None)])
        self.assertEqual(self.controller.phase, "analyzing")

    def test_later_cached_snapshot_does_not_discard_buffered_fresh_message(self):
        self.config["auto_reply_chats"] = {"wechat:月仔": {"enabled": True}}
        self.controller.phase = "switching"
        self.controller.active = {"id": "person:1", "base": "person", "name": "月仔",
                                  "previous_title": "群聊", "previous_raw": "群聊"}
        self.controller.badges = {"person": {"name": "月仔"}}
        new = [("her", None, "是吗 哈哈")]
        with patch.object(main, "start_analyze", return_value=True) as analyze:
            self._drain_packets([("chat", "月仔", "月仔"),
                                 ("snapshot", self._snapshot(new=new, initial=True)),
                                 ("snapshot", self._snapshot()), ("unread_state", [])])
            analyze.assert_not_called()
            self._settle_current_chat()
        analyze.assert_called_once_with("月仔", [("her", "是吗 哈哈", None)])
        self.assertEqual(self.controller.phase, "analyzing")
        self.assertEqual(main.chat_of("月仔")["auto_sent_rev"], -1)

    def test_read_only_mode_does_not_call_models_or_require_reply_authorization(self):
        self.config["analysis_enabled"] = False
        self.controller.phase = "switching"
        self.controller.active = {"id": "person:1", "base": "person", "name": "月仔",
                                  "candidate_title": "月仔", "candidate_raw": "月仔"}
        self.controller._confirm_switch()
        self._settle_current_chat()
        self.assertEqual(self.controller.phase, "waiting_lines")
        with patch.object(main, "start_analyze") as analyze:
            self._drain_packets([("snapshot", self._snapshot(new=[("her", None, "是吗 哈哈")]))])
        analyze.assert_not_called()
        self.assertEqual(self.controller.phase, "listening")
        self.assertEqual(len(main.chat_of("月仔")["history"]), 1)

    def test_global_send_switch_blocks_authorized_chat_without_stopping_analysis(self):
        self.config["send_enabled"] = False
        self.config["auto_reply_chats"] = {"wechat:月仔": {"enabled": True}}
        self.controller.phase = "analyzing"
        self.controller.active = {"id": "person:1", "title": "月仔", "raw": "月仔"}
        self.assertFalse(self.controller.mark_result_ready("月仔", 0))
        self.assertEqual(self.controller.phase, "listening")
        with patch.object(main, "fill_and_send") as send:
            self.assertFalse(main.auto_send_reply(
                "月仔", {"candidates": ["收到"], "best_index": 0, "ranking_valid": True}, 0))
        send.assert_not_called()

    def test_snapshot_preserves_auto_send_echo_protection(self):
        import time

        chat = main.chat_of("月仔")
        chat["last_auto_sent_text"] = "这是刚刚发送的回复"
        chat["last_auto_sent_at"] = time.monotonic()
        incoming = [("her", None, "这是刚刚发送的回复")]
        with patch.object(main, "start_analyze") as analyze:
            self._drain_packets([("snapshot", self._snapshot(new=incoming, visible=incoming,
                                                              history=incoming))])
        analyze.assert_not_called()
        self.assertEqual(list(chat["history"]), [("me", "这是刚刚发送的回复", None)])

    def test_changing_mode_invalidates_scheduled_send_revision(self):
        chat = main.chat_of("月仔")
        chat["rev"] = 3
        self.controller.phase = "waiting_send"
        self.controller.active = {"id": "person:1", "title": "月仔", "raw": "月仔"}
        self.controller.send_revision = 3
        main.on_processing_changed()
        with patch.object(main, "auto_send_reply") as send:
            main.auto_send_and_track("月仔", {}, 3)
        send.assert_not_called()
        self.assertEqual(chat["rev"], 4)
        self.assertEqual(self.controller.phase, "listening")

    def test_unconfirmed_or_discontinuous_snapshot_never_triggers_analysis(self):
        for changes in ({"title_confirmed": False}, {"discontinuous": True}):
            with self.subTest(changes=changes), patch.object(main, "start_analyze") as analyze:
                self._drain_packets([("snapshot", self._snapshot(new=[("her", None, "未知消息")],
                                                                 **changes))])
            analyze.assert_not_called()

    def test_rebaselined_snapshot_analyzes_but_blocks_auto_send_until_real_new_message(self):
        self.config["auto_reply_chats"] = {"wechat:月仔": {"enabled": True}}
        message = [("her", None, "滚动后停住的消息")]
        with patch.object(main, "start_analyze", return_value=True) as analyze:
            self._drain_packets([("snapshot", self._snapshot(new=message, baseline=True, rebaselined=True))])
        analyze.assert_called_once()  # 仍然生成建议
        self.assertTrue(main.chat_of("月仔")["manual_only"])
        result = {"candidates": ["收到"], "best_index": 0}
        with patch.object(main, "fill_and_send") as send:
            self.assertFalse(main.auto_send_reply("月仔", result, main.chat_of("月仔")["rev"]))
        send.assert_not_called()
        self.assertIn("仅生成建议", main.chat_of("月仔")["auto_send_block_reason"])
        with patch.object(main, "start_analyze", return_value=True):
            self._drain_packets([("snapshot", self._snapshot(new=[("her", None, "真正的新消息")],
                                                             baseline=False))])
        self.assertFalse(main.chat_of("月仔")["manual_only"])

    def test_switch_confirmation_requests_fresh_ocr_even_if_pixels_unchanged(self):
        self.config["auto_reply_chats"] = {"wechat:月仔": {"enabled": True}}
        self.controller.phase = "switching"
        self.controller.active = {"id": "person:1", "base": "person", "name": "月仔",
                                  "candidate_title": "月仔", "candidate_raw": "月仔"}
        event = Mock()
        with patch.object(main, "read_requested", event):
            self.controller._confirm_switch()
        event.set.assert_called_once()

    def test_corrected_history_invalidates_cached_result_and_delayed_send(self):
        chat = main.chat_of("月仔")
        chat["rev"] = 3
        chat["result"] = {"candidates": ["旧的错误回复"]}
        self.controller.phase = "waiting_send"
        self.controller.active = {"id": "person:1", "title": "月仔", "raw": "月仔"}
        self.controller.send_revision = 3
        self._drain_packets([("snapshot", self._snapshot(corrected=True))])
        self.assertEqual(chat["rev"], 4)
        self.assertIsNone(chat["result"])
        self.assertEqual(chat["auto_sent_rev"], -1)
        with patch.object(main, "auto_send_reply") as send:
            main.auto_send_and_track("月仔", {}, 3)
        send.assert_not_called()

    def test_initial_cached_snapshot_replies_once_for_authorized_chat(self):
        self.config["auto_reply_chats"] = {"wechat:月仔": {"enabled": True}}
        with patch.object(main, "start_analyze", return_value=True) as analyze:
            snapshot = self._snapshot(initial=True)
            self._drain_packets([("snapshot", snapshot), ("snapshot", snapshot)])
        analyze.assert_called_once_with("月仔", [("her", "是吗 哈哈", None)])
        chat = main.chat_of("月仔")
        with patch.object(main, "fill_and_send") as send, patch.object(main.time, "time", return_value=100):
            self.assertTrue(main.auto_send_reply("月仔", {"candidates": ["你是说这个吗？"]}, chat["rev"]))
        send.assert_called_once()

    def test_media_ocr_partial_failure_and_pending_do_not_block_authorized_send(self):
        self.config["auto_reply_chats"] = {"wechat:月仔": {"enabled": True}}
        chat = main.chat_of("月仔")
        chat["history"].append(("her", "[图片消息] 可见文字：游戏", None))
        chat["history_ids"] = [7]
        for status in ("ocr", "partial", "failed", "pending"):
            with self.subTest(status=status), patch.object(main, "fill_and_send") as send:
                chat["media_states"] = {7: {"status": status, "who": "her", "summary": ""}}
                chat["auto_sent_rev"], chat["auto_sent_key"], chat["last_auto_sent_at"] = -1, None, 0
                self.assertTrue(main.auto_send_reply("月仔", {"candidates": ["你发的是这个游戏吗？"]}, 0))
                send.assert_called_once()

    def test_same_message_is_not_sent_again_after_visual_revision(self):
        self.config["auto_reply_chats"] = {"wechat:月仔": {"enabled": True}}
        chat = main.chat_of("月仔")
        chat["history"].append(("her", "图片内容", None))
        chat["history_ids"] = [7]
        result = {"candidates": ["这是你在玩的游戏吗？"]}
        with patch.object(main, "fill_and_send") as send:
            self.assertTrue(main.auto_send_reply("月仔", result, 0))
            chat["rev"] = 1
            self.assertFalse(main.auto_send_reply("月仔", result, 1))
        send.assert_called_once()

    def test_cooldown_defers_send_then_retries_once(self):
        self.config["auto_reply_chats"] = {"wechat:月仔": {"enabled": True}}
        chat = main.chat_of("月仔")
        chat["last_auto_sent_at"] = 100
        result = {"candidates": ["你是想说这个吗？"]}
        with patch.object(main.time, "monotonic", return_value=102), patch.object(main, "fill_and_send") as send:
            self.assertIsNone(main.auto_send_reply("月仔", result, 0))
            self.assertIsNone(main.auto_send_reply("月仔", result, 0))
        send.assert_not_called()
        self.assertEqual(self.overlay.after.call_count, 1)
        delay, retry = self.overlay.after.call_args.args
        self.assertGreaterEqual(delay, 6000)
        with patch.object(main.time, "monotonic", return_value=109), patch.object(main, "fill_and_send") as send:
            retry()
            retry()
        send.assert_called_once()

    def test_deferred_tracked_send_does_not_enter_failure_phase(self):
        self.config["auto_reply_chats"] = {"wechat:月仔": {"enabled": True}}
        self.controller.phase = "waiting_send"
        self.controller.active = {"id": "current:1", "title": "月仔", "raw": "月仔"}
        self.controller.send_revision = 0
        main.chat_of("月仔")["last_auto_sent_at"] = 100
        with patch.object(main.time, "monotonic", return_value=102):
            main.auto_send_and_track("月仔", {"candidates": ["这是啥呀？"]}, 0)
        self.assertEqual(self.controller.phase, "waiting_send")
        self.assertIsNotNone(self.controller.active)

    def test_deferred_send_is_cancelled_when_conversation_changes(self):
        self.config["auto_reply_chats"] = {"wechat:月仔": {"enabled": True}}
        main.chat_of("月仔")["last_auto_sent_at"] = 100
        with patch.object(main.time, "monotonic", return_value=102):
            main.auto_send_reply("月仔", {"candidates": ["这是什么意思呀？"]}, 0)
        retry = self.overlay.after.call_args.args[1]
        self.state.update(chat="其他会话", raw_chat="其他会话")
        with patch.object(main, "fill_and_send") as send:
            retry()
        send.assert_not_called()

    def test_new_incoming_message_recovers_from_previous_send_failure(self):
        self.config["auto_reply_chats"] = {"wechat:月仔": {"enabled": True}}
        self.controller.phase = "send_failed"
        self.controller.active = {"id": "current:1", "title": "月仔", "raw": "月仔", "revision": 1}
        self.assertEqual(self.controller.route_lines("月仔", "月仔", 2, False, "her"), "analyze")
        self.assertEqual(self.controller.phase, "analyzing")
        self.assertEqual(self.controller.active["revision"], 2)

    def test_unconfirmed_snapshot_defers_instead_of_dropping_reply(self):
        self.config["auto_reply_chats"] = {"wechat:月仔": {"enabled": True}}
        self.state["ocr_confirmed"] = False
        with patch.object(main, "fill_and_send") as send, patch.object(main, "request_read") as read:
            self.assertIsNone(main.auto_send_reply("月仔", {"candidates": ["你是在问这个吗？"]}, 0))
        send.assert_not_called()
        read.assert_called_once()
        self.overlay.after.assert_called_once()

    def test_cached_scroll_does_not_reply_to_older_message_as_latest(self):
        self.config["auto_reply_chats"] = {"wechat:月仔": {"enabled": True}}
        with patch.object(main, "start_analyze") as analyze:
            snapshot = self._snapshot(visible=[("her", None, "过去的消息")])
            self._drain_packets([("snapshot", snapshot)])
        analyze.assert_not_called()

    def test_authorized_initial_new_lines_can_auto_reply(self):
        self.config["auto_reply_chats"] = {"wechat:月仔": {"enabled": True}}
        with patch.object(main, "start_analyze", return_value=True) as analyze:
            self._drain_packets([("lines", "月仔", [("her", None, "你看这张图")], self.state["area"], True, "月仔")])
        analyze.assert_called_once()
        chat = main.chat_of("月仔")
        with patch.object(main, "fill_and_send") as send:
            self.assertTrue(main.auto_send_reply("月仔", {"candidates": ["这是哪里呀？"]}, chat["rev"]))
        send.assert_called_once()

    def test_send_requires_recent_confirmed_snapshot(self):
        self.config["auto_reply_chats"] = {"wechat:月仔": {"enabled": True}}
        self.state["ocr_confirmed"] = True
        self.state["read_at"] = 100
        with patch.object(main.time, "time", return_value=103), \
                patch.object(main, "fill_and_send") as send:
            self.assertFalse(main.auto_send_reply(
                "月仔", {"candidates": ["收到"], "best_index": 0, "ranking_valid": True}, 0))
        send.assert_not_called()

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


class MessageSequenceTests(unittest.TestCase):
    def setUp(self):
        with patch.object(ocr, "_engine"):
            self.reader = ocr.Reader()

    def test_repeated_incoming_text_after_outgoing_reply_is_new(self):
        first = [("her", None, "是吗 哈哈", 10), ("me", None, "到底怎么了", 50)]
        self.reader.new_lines(first)
        self.assertEqual(self.reader.new_lines(first + [("her", None, "是吗 哈哈", 90)]),
                         [("her", None, "是吗 哈哈")])

    def test_identical_consecutive_bubbles_are_distinct(self):
        self.reader.new_lines([("her", None, "哈哈", 10)])
        self.assertEqual(self.reader.new_lines([("her", None, "哈哈", 10),
                                                ("her", None, "哈哈", 50)]),
                         [("her", None, "哈哈")])

    def test_unchanged_and_scrolled_history_is_not_new(self):
        lines = [("her", None, "你好", 10), ("me", None, "今天怎么样", 50),
                 ("her", None, "还好", 90)]
        self.reader.new_lines(lines)
        self.assertEqual(self.reader.new_lines(lines), [])
        self.assertEqual(self.reader.new_lines(lines[:2]), [])
        self.assertEqual(self.reader.new_lines(lines[1:]), [])


class UnreadNameTests(unittest.TestCase):
    def setUp(self):
        self.badge = {"base": "new", "box": (70, 80, 84, 94), "x": 120, "y": 87}
        self.frame = np.zeros((300, 400, 3), dtype=np.uint8)
        self.area = (200, 60, 400, 280, np.array([0, 0, 0]), 20)

    def test_existing_unread_is_named_and_emitted_once_without_changes(self):
        read_name = Mock(return_value="月仔")
        tracker = UnreadTracker(read_name=read_name)
        with patch("app.capture.unread_badges", return_value=[dict(self.badge)]):
            badges, events = tracker.update(self.frame, self.area)
            _, repeated = tracker.update(self.frame, self.area)
        self.assertEqual(badges[0]["name"], "月仔")
        self.assertEqual(events[0]["name"], "月仔")
        self.assertEqual(events[0]["id"], "new:1")
        self.assertEqual(repeated, [])
        read_name.assert_called_once()

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

    def test_existing_unread_name_is_retried_without_new_badge(self):
        read_name = Mock(side_effect=["", "月仔"])
        tracker = UnreadTracker(read_name=read_name)
        with patch("app.capture.unread_badges", return_value=[dict(self.badge)]):
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


class UnreadGeometryTests(unittest.TestCase):
    def _frame(self):
        frame = np.full((400, 600, 3), 245, dtype=np.uint8)
        frame[:, :80] = 215
        frame[:, 80:300] = 235
        return frame

    def _badge(self, frame, x, y):
        ys, xs = np.ogrid[:frame.shape[0], :frame.shape[1]]
        frame[(xs - x) ** 2 + (ys - y) ** 2 <= 7 ** 2] = (255, 80, 80)

    def test_numeric_unread_badge_at_200_percent_dpi_is_detected(self):
        from app.capture import unread_badges

        frame = self._frame()
        ys, xs = np.ogrid[:frame.shape[0], :frame.shape[1]]
        frame[(xs - 130) ** 2 + (ys - 180) ** 2 <= 16 ** 2] = (255, 80, 80)
        frame[173:187, 128:132] = 255
        area = (300, 80, 600, 380, np.array([245, 245, 245]), 20)
        badges = unread_badges(frame, area, scale=2)
        self.assertEqual(len(badges), 1)
        self.assertGreater(badges[0]["box"][2] - badges[0]["box"][0], 30)

    def test_red_avatar_decoration_outside_badge_column_is_ignored(self):
        from app.capture import unread_badges

        frame = self._frame()
        for x, y in ((130, 100), (130, 180), (130, 260), (190, 340)):
            self._badge(frame, x, y)
        area = (300, 80, 600, 380, np.array([245, 245, 245]), 20)
        badges = unread_badges(frame, area)
        self.assertEqual(len(badges), 3)
        self.assertTrue(all(badge["box"][0] < 150 for badge in badges))

    def test_navigation_badge_is_not_a_conversation_unread(self):
        from app.capture import unread_badges

        frame = self._frame()
        self._badge(frame, 66, 120)
        self._badge(frame, 130, 180)
        area = (300, 80, 600, 380, np.array([245, 245, 245]), 20)
        badges = unread_badges(frame, area)
        self.assertEqual(len(badges), 1)
        self.assertGreater(badges[0]["box"][0], 80)


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

    def test_message_preview_is_not_accepted_as_session_name(self):
        with patch.object(ocr, "_engine", return_value=Mock(
                return_value=([self._item("叶艺凡妈妈：我们", y=33)], None))):
            self.assertEqual(ocr.read_session_name(self.frame, self.area, self.badge), "")

    def test_valid_title_with_colon_is_preserved(self):
        for title in ("团队：研发", "JH2026秋自然拼读-T二四16：30"):
            with self.subTest(title=title), patch.object(ocr, "_engine", return_value=Mock(
                    return_value=([self._item(title)], None))):
                self.assertEqual(ocr.read_session_name(self.frame, self.area, self.badge), title)

    def test_cut_off_title_is_not_accepted_as_complete_name(self):
        def recognize(crop, **kwargs):
            return [self._item("公众", y=crop.shape[0] - 14)], None
        with patch.object(ocr, "_engine", return_value=recognize):
            self.assertEqual(ocr.read_session_name(self.frame, self.area, self.badge), "")

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
                            ("has_key", lambda: True), ("models_ready", lambda: True),
                            ("has_jev_key", lambda: True), ("has_judge_key", lambda: True),
                            ("has_llm_key", lambda: True), ("jev_key", lambda: "demo-key"),
                            ("laya_key", lambda: ""),
                            ("judge_key", lambda: "demo-key"), ("llm_key", lambda: "demo-key")):
            target = patch.object(settings, name, value)
            target.start()
            self.addCleanup(target.stop)
        from app.overlay import Overlay
        from PySide6.QtWidgets import QWidget

        with patch.object(QWidget, "show"):
            self.overlay = Overlay(on_fill=Mock())
        self.addCleanup(self.overlay.win.close)

    def test_laya_controls_save_load_and_test_without_key(self):
        from app import overlay

        group = self.overlay.jev
        group.keyEdit.setText("unsaved-official-key")
        group.providerBox.setCurrentIndex(group.ids.index("laya"))
        self.assertEqual(group.keyEdit.text(), "")
        self.assertEqual(group.stored_key(), "")
        self.assertFalse(group.baseEdit.isHidden())
        self.assertEqual(group.baseEdit.text(), "http://127.0.0.1:8000")
        self.assertEqual(group.modelBox.text(), "multilingual")
        self.assertEqual(group.keyLabel.text(), "密钥（可选）")
        self.assertEqual(group.fetchButton.text(), "模型预设")
        group.baseEdit.setText("http://127.0.0.1:18001")
        with patch.object(settings, "save", side_effect=lambda *args, **kwargs: self.config.update(
                jev_provider=kwargs["jev_provider_text"], laya_model=kwargs["jev_model_text"],
                laya_base_url=kwargs["jev_base_url_text"])) as save:
            self.overlay._save()
        self.assertEqual(save.call_args.kwargs["jev_provider_text"], "laya")
        self.assertEqual(save.call_args.kwargs["jev_base_url_text"], "http://127.0.0.1:18001")
        self.assertIsNone(save.call_args.kwargs["jev_key_text"])
        with patch.object(overlay.threading, "Thread") as thread:
            self.overlay._test_jev()
            config = thread.call_args.kwargs["args"][0]
        self.assertEqual(config["api_key"], "")
        self.assertEqual(config["base_url"], "http://127.0.0.1:18001")
        with patch.object(overlay.jev_client, "ask", return_value=self._jev_test_result()) as ask:
            self.overlay._test_jev_bg(config)
        self.assertEqual(ask.call_args.kwargs["provider"], "laya")
        self.assertIn("已通过验证", group.status.text())
        self.assertIn("1.1/9", group.status.text())
        self.config.update(jev_provider="laya", laya_base_url="http://127.0.0.1:18001", laya_model="english")
        self.overlay._load_settings()
        self.assertEqual(group.baseEdit.text(), "http://127.0.0.1:18001")
        self.assertEqual(group.modelBox.text(), "english")

    def test_laya_timeout_control_load_save_visibility_and_unsaved_test(self):
        from app import overlay

        group = self.overlay.jev
        self.assertTrue(group.timeoutBox.isHidden())
        group.providerBox.setCurrentIndex(group.ids.index("laya"))
        self.assertFalse(group.timeoutBox.isHidden())
        self.assertEqual(group.timeoutBox.minimum(), 60)
        self.assertEqual(group.timeoutBox.maximum(), 120)
        self.assertEqual(group.timeoutBox.value(), 120)
        group.timeoutBox.setValue(90)
        with patch.object(overlay.threading, "Thread") as thread:
            self.overlay._test_jev()
            config = thread.call_args.kwargs["args"][0]
        self.assertEqual(config["timeout"], 90)
        with patch.object(overlay.jev_client, "ask", return_value=self._jev_test_result()) as ask:
            self.overlay._test_jev_bg(config)
        self.assertEqual(ask.call_args.kwargs["timeout"], 90)
        with patch.object(settings, "save") as save:
            self.overlay._save()
        self.assertEqual(save.call_args.kwargs["laya_timeout_seconds"], 90)
        self.config.update(jev_provider="laya", laya_timeout_seconds=75)
        self.overlay._load_settings()
        self.assertEqual(group.timeoutBox.value(), 75)
        self.overlay.judgeModeBox.setCurrentIndex(2)
        self.assertTrue(group.timeoutBox.isHidden())
        self.assertFalse(group.timeoutBox.isEnabled())

    def test_laya_switch_preserves_separate_addresses_and_clears_unsaved_keys(self):
        group = self.overlay.jev
        group.providerBox.setCurrentIndex(group.ids.index("openai_decisions"))
        group.baseEdit.setText("https://quick-pivot.com/v1")
        group.keyEdit.setText("official-key")
        group.providerBox.setCurrentIndex(group.ids.index("laya"))
        group.baseEdit.setText("http://127.0.0.1:18001")
        group.keyEdit.setText("local-key")
        group.providerBox.setCurrentIndex(group.ids.index("openai_decisions"))
        self.assertEqual(group.baseEdit.text(), "https://quick-pivot.com/v1")
        self.assertEqual(group.keyEdit.text(), "")
        self.assertEqual(group.stored_key(), "demo-key")
        group.providerBox.setCurrentIndex(group.ids.index("laya"))
        self.assertEqual(group.baseEdit.text(), "http://127.0.0.1:18001")
        self.assertEqual(group.stored_key(), "")

    def test_laya_presets_are_offline_and_do_not_report_connection_success(self):
        from app import overlay

        group = self.overlay.jev
        group.providerBox.setCurrentIndex(group.ids.index("laya"))
        with patch.object(overlay.threading, "Thread") as thread, \
                patch.object(overlay.jev_client, "_check_openrouter_key") as auth:
            self.overlay._fetch_models(group)
        thread.assert_not_called()
        auth.assert_not_called()
        self.assertEqual(group.modelBox.count(), 3)
        self.assertIn("内置模型预设", group.status.text())
        self.assertNotIn("已通过验证", group.status.text())

    def test_laya_requires_address_for_saving_and_testing(self):
        from app import overlay

        group = self.overlay.jev
        group.providerBox.setCurrentIndex(group.ids.index("laya"))
        group.baseEdit.clear()
        with patch.object(settings, "save") as save, patch.object(overlay.threading, "Thread") as thread:
            self.overlay._save()
            self.overlay._test_jev()
        save.assert_not_called()
        thread.assert_not_called()
        self.assertIn("Laya 服务地址", group.status.text())

    def test_decisions_controls_switch_load_and_save_custom_configuration(self):
        group = self.overlay.jev
        self.assertTrue(group.baseEdit.isHidden())
        group.providerBox.setCurrentIndex(group.ids.index("openai_decisions"))
        self.assertFalse(group.baseEdit.isHidden())
        self.assertEqual(group.modelBox.text(), "gpt-6-luna")
        self.assertEqual(group.baseEdit.text(), "https://api.openai.com/v1")
        group.baseEdit.setText("https://quick-pivot.com/v1")
        group.keyEdit.setText("typed-decision-key")
        group.modelBox.setText("future-model")
        with patch.object(settings, "save") as save:
            self.overlay._save()
        self.assertEqual(save.call_args.kwargs["jev_provider_text"], "openai_decisions")
        self.assertEqual(save.call_args.kwargs["jev_base_url_text"], "https://quick-pivot.com/v1")
        self.assertEqual(save.call_args.kwargs["jev_key_text"], "typed-decision-key")
        self.assertEqual(save.call_args.kwargs["jev_model_text"], "future-model")
        self.config.update(jev_provider="openai_decisions", jev_base_url="https://quick-pivot.com/v1",
                           jev_model="future-model")
        self.overlay._load_settings()
        self.assertEqual(group.baseEdit.text(), "https://quick-pivot.com/v1")
        self.assertEqual(group.modelBox.text(), "future-model")
        self.overlay.judgeModeBox.setCurrentIndex(2)
        self.assertTrue(group.baseEdit.isHidden())
        self.assertFalse(group.fetchButton.isEnabled())

    def test_decisions_fetch_uses_typed_address_key_and_preserves_selected_model(self):
        from app import overlay

        group = self.overlay.jev
        group.providerBox.setCurrentIndex(group.ids.index("openai_decisions"))
        group.baseEdit.setText("https://quick-pivot.com/v1")
        group.keyEdit.setText("typed-decision-key")
        with patch.object(overlay.threading, "Thread") as thread, \
                patch.object(self.overlay, "_list_models") as listing:
            self.overlay._fetch_models(group)
            thread.call_args.kwargs["target"]()
        listing.assert_called_once_with(group, "openai_decisions", "typed-decision-key",
                                        "https://quick-pivot.com/v1")
        with patch.object(overlay.jev_client, "list_models", return_value=["future-model", "gpt-6-luna"]) as models:
            self.overlay._list_models(group, "openai_decisions", "typed-decision-key", "https://quick-pivot.com/v1")
        models.assert_called_once_with("openai_decisions", "typed-decision-key",
                                       base_url="https://quick-pivot.com/v1")
        self.assertEqual(group.modelBox.text(), "gpt-6-luna")
        self.assertTrue(group.fetchButton.isEnabled())

    def test_decisions_allows_empty_base_for_official_endpoint_and_retains_hidden_address(self):
        group = self.overlay.jev
        self.config["jev_base_url"] = "https://quick-pivot.com/v1"
        self.overlay._load_settings()
        group.providerBox.setCurrentIndex(group.ids.index("openai_decisions"))
        self.assertEqual(group.baseEdit.text(), "https://quick-pivot.com/v1")
        group.baseEdit.clear()
        with patch.object(settings, "save") as save:
            self.overlay._save()
        self.assertEqual(save.call_args.kwargs["jev_base_url_text"], "")

    def _jev_test_result(self):
        return {"answers": {
            "literal_question": {"type": "noul", "noul": 0.9},
            "true_intent": {"type": "choice", "choice": "casual_chat"},
            "danger_level": {"type": "score", "score": 1.1},
        }}

    def test_jev_test_button_uses_unsaved_configuration_without_saving(self):
        from app import overlay

        group = self.overlay.jev
        group.providerBox.setCurrentIndex(group.ids.index("openai_decisions"))
        group.baseEdit.setText("https://quick-pivot.com/v1")
        group.modelBox.setText("future-model")
        group.keyEdit.setText("unsaved-test-key")
        with patch.object(overlay.threading, "Thread") as thread, patch.object(settings, "save") as save:
            self.overlay.jevTestButton.click()
            config = thread.call_args.kwargs["args"][0]
            thread.return_value.start.assert_called_once()
            self.assertFalse(self.overlay.jevTestButton.isEnabled())
            self.overlay._sync_model_fields()
            self.assertFalse(self.overlay.jevTestButton.isEnabled())
            self.overlay._test_jev()
            thread.assert_called_once()
            save.assert_not_called()
        self.assertEqual(config, {"provider": "openai_decisions", "model": "future-model",
                                  "base_url": "https://quick-pivot.com/v1", "api_key": "unsaved-test-key"})

    def test_jev_test_validates_synthetic_text_and_returns_result_to_ui(self):
        from app import overlay

        group = self.overlay.jev
        group.providerBox.setCurrentIndex(group.ids.index("openai_decisions"))
        group.keyEdit.setText("unsaved-test-key")
        with patch.object(overlay.threading, "Thread") as thread:
            self.overlay._test_jev()
            config = thread.call_args.kwargs["args"][0]
        with patch.object(overlay.jev_client, "ask", return_value=self._jev_test_result()) as ask, \
                patch.object(settings, "save") as save:
            self.overlay._test_jev_bg(config)
        self.assertEqual(ask.call_args.kwargs["provider"], "openai_decisions")
        self.assertEqual(ask.call_args.kwargs["api_key"], "unsaved-test-key")
        self.assertEqual(ask.call_args.kwargs["model"], "gpt-6-luna")
        self.assertEqual(ask.call_args.kwargs["base_url"], "https://api.openai.com/v1")
        self.assertEqual(ask.call_args.kwargs["max_retries"], 1)
        self.assertEqual(set(ask.call_args.args[1]), {"literal_question", "true_intent", "danger_level"})
        messages = ask.call_args.args[0]["chat"]["messages"]
        self.assertEqual(len(messages), 1)
        self.assertNotIn("月仔", str(ask.call_args.args[0]))
        self.assertIn("已通过验证", group.status.text())
        self.assertIn("1.1/9", group.status.text())
        self.assertTrue(self.overlay.jevTestButton.isEnabled())
        save.assert_not_called()

    def test_jev_test_reports_invalid_response_and_redacts_failure_key(self):
        from app import overlay
        from core.jev_client import JevError

        group = self.overlay.jev
        group.providerBox.setCurrentIndex(group.ids.index("openai_decisions"))
        group.keyEdit.setText("unsaved-test-key")
        for result in ({"answers": {}}, JevError("bad unsaved-test-key", 401)):
            with self.subTest(result=result), patch.object(overlay.threading, "Thread") as thread:
                self.overlay._test_jev()
                config = thread.call_args.kwargs["args"][0]
            kwargs = {"side_effect": result} if isinstance(result, Exception) else {"return_value": result}
            with patch.object(overlay.jev_client, "ask", **kwargs):
                self.overlay._test_jev_bg(config)
            self.assertIn("验证失败", group.status.text())
            self.assertNotIn("unsaved-test-key", group.status.text())
            self.assertTrue(self.overlay.jevTestButton.isEnabled())

    def test_jev_test_checks_missing_key_model_and_strategy_before_starting(self):
        from app import overlay

        group = self.overlay.jev
        with patch.object(overlay.threading, "Thread") as thread, patch.object(group, "stored_key", return_value=""):
            self.overlay._test_jev()
            self.assertEqual(group.status.text(), "先填密钥")
            group.keyEdit.setText("typed-key")
            group.modelBox.setText("")
            self.overlay._test_jev()
            self.assertIn("模型", group.status.text())
            group.modelBox.setText("test-model")
            self.overlay.judgeModeBox.setCurrentIndex(2)
            self.assertFalse(self.overlay.jevTestButton.isEnabled())
            self.overlay._test_jev()
            thread.assert_not_called()

    def test_jev_test_uses_saved_key_and_does_not_show_stale_success(self):
        from app import overlay

        group = self.overlay.jev
        group.providerBox.setCurrentIndex(group.ids.index("openai_decisions"))
        group.baseEdit.clear()
        with patch.object(overlay.threading, "Thread") as thread:
            self.overlay._test_jev()
            config = thread.call_args.kwargs["args"][0]
        self.assertEqual(config["api_key"], "demo-key")
        self.assertEqual(config["base_url"], "")
        group.modelBox.setText("changed-model")
        with patch.object(overlay.jev_client, "ask", return_value=self._jev_test_result()):
            self.overlay._test_jev_bg(config)
        self.assertNotIn("已通过验证", group.status.text())
        self.assertIn("配置已变更", group.status.text())
        self.assertTrue(self.overlay.jevTestButton.isEnabled())

    def test_jev_test_runs_for_existing_providers_and_remains_disabled_in_llm_only_mode(self):
        from app import overlay

        group = self.overlay.jev
        for provider in ("openrouter", "typesafe"):
            with self.subTest(provider=provider):
                group.providerBox.setCurrentIndex(group.ids.index(provider))
                with patch.object(overlay.threading, "Thread") as thread:
                    self.overlay._test_jev()
                    config = thread.call_args.kwargs["args"][0]
                result = self._jev_test_result()
                result["answers"]["danger_level"]["score"] = 1
                self.overlay.judgeModeBox.setCurrentIndex(2)
                with patch.object(overlay.jev_client, "ask", return_value=result) as ask:
                    self.overlay._test_jev_bg(config)
                self.assertEqual(ask.call_args.kwargs["provider"], provider)
                self.assertIsNone(ask.call_args.kwargs["base_url"])
                self.assertFalse(self.overlay.jevTestButton.isEnabled())
                self.overlay.judgeModeBox.setCurrentIndex(0)

    def test_settings_load_ignore_list_and_save_entered_names(self):
        self.overlay._load_settings()
        self.assertEqual(self.overlay.ignoredChatsEdit.toPlainText(), "营销群")
        self.overlay.ignoredChatsEdit.setPlainText("银行通知\n文件传输助手")
        with patch.object(settings, "save") as save:
            self.overlay._save()
        self.assertEqual(save.call_args.kwargs["ignored_chats_value"], ["银行通知", "文件传输助手"])

    def test_logging_controls_save_and_apply_immediately(self):
        callback = Mock()
        self.overlay.on_logging_change = callback
        with patch.object(settings, "save") as save:
            self.overlay.loggingLevelBox.setCurrentIndex(0)
        save.assert_called_once_with(logging_enabled_on=True, logging_level_text="DEBUG")
        callback.assert_called_once()
        with patch.object(settings, "save") as save:
            self.overlay.loggingSwitch.setChecked(False)
        save.assert_called_once_with(logging_enabled_on=False, logging_level_text="DEBUG")
        self.assertEqual(callback.call_count, 2)
        self.assertFalse(self.overlay.loggingLevelBox.isEnabled())

    def test_judge_strategy_enables_only_relevant_model_groups(self):
        self.assertTrue(self.overlay.jev.providerBox.isEnabled())
        self.assertFalse(self.overlay.judge.providerBox.isEnabled())
        self.overlay.judgeModeBox.setCurrentIndex(2)
        self.overlay.judgeReuseSwitch.setChecked(False)
        self.assertFalse(self.overlay.jev.providerBox.isEnabled())
        self.assertTrue(self.overlay.judge.providerBox.isEnabled())
        self.assertTrue(self.overlay.draft.providerBox.isEnabled())

    def test_save_includes_judge_strategy(self):
        self.overlay.judgeModeBox.setCurrentIndex(2)
        self.overlay.judgeReuseSwitch.setChecked(True)
        with patch.object(settings, "save") as save:
            self.overlay._save()
        self.assertEqual(save.call_args.kwargs["judge_mode_text"], "llm_only")
        self.assertTrue(save.call_args.kwargs["judge_reuse_draft_on"])

    def test_ignored_chat_disables_auto_reply_and_suppresses_cached_result(self):
        self.overlay.set_chat("公众号")
        self.overlay.show({"candidates": ["不应该展示"]})
        self.assertFalse(self.overlay.autoReplySwitch.isEnabled())
        self.assertFalse(self.overlay.autoKeyBox.isEnabled())
        self.assertEqual(self.overlay.cands, [])
        self.assertEqual(self.overlay.emptyTitle.text(), "当前会话已忽略")

    def test_processing_switches_default_to_suggestions_without_sending(self):
        self.assertTrue(self.overlay.analysisSwitch.isChecked())
        self.assertFalse(self.overlay.sendSwitch.isChecked())

    def test_read_only_switch_persists_independently_and_notifies_controller(self):
        callback = Mock()
        self.overlay.on_processing_change = callback

        def save(**changes):
            self.config["analysis_enabled"] = changes["analysis_enabled_on"]
            self.config["send_enabled"] = changes["send_enabled_on"]

        with patch.object(settings, "save", side_effect=save) as persist:
            self.overlay.analysisSwitch.setChecked(False)
        persist.assert_called_once_with(analysis_enabled_on=False, send_enabled_on=False)
        callback.assert_called_once()
        self.assertTrue(self.overlay.captureSwitch.isChecked())
        self.assertEqual(self.overlay.emptyTitle.text(), "仅采集并保存消息")

    def test_global_send_enable_requires_explicit_confirmation(self):
        from PySide6.QtWidgets import QMessageBox

        with patch.object(QMessageBox, "warning", return_value=QMessageBox.No), \
                patch.object(settings, "save") as save:
            self.overlay.sendSwitch.setChecked(True)
        save.assert_not_called()
        self.assertFalse(self.overlay.sendSwitch.isChecked())

    def test_restored_history_and_read_status_are_isolated_by_chat(self):
        self.overlay.set_chat("月仔")
        self.overlay.replace_history("月仔", [("her", "是吗 哈哈", None), ("me", "已回复", None)])
        self.overlay.set_read_state("月仔", 100, 0, "read")
        self.overlay.replace_history("其他人", [("her", "另一段消息", None)])
        self.assertEqual(self.overlay.counts["月仔"], 2)
        self.assertEqual(self.overlay.counts["其他人"], 1)
        self.assertEqual(self.overlay.hers["月仔"], "是吗 哈哈")
        self.assertIn("新增 0 条", self.overlay.readState.text())
        self.overlay.set_chat("其他人")
        self.assertNotIn("新增 0 条", self.overlay.readState.text())

    def test_corrected_history_without_incoming_clears_previous_latest_text(self):
        self.overlay.set_chat("月仔")
        self.overlay.replace_history("月仔", [("her", "我叫MT4", None)])
        self.assertEqual(self.overlay.latest.text(), "我叫MT4")
        self.overlay.replace_history("月仔", [("me", "正常的自己消息", None)])
        self.assertEqual(self.overlay.latest.text(), "")
        self.assertTrue(self.overlay.context.isHidden())

    def test_no_reply_result_shows_successful_empty_state_and_judgment(self):
        self.overlay.set_chat("月仔")
        self.overlay.show({"reply_required": False, "no_reply_reason": "对方已结束话题，本轮无需回复",
                           "candidates": [], "best_index": None, "ranking_valid": False,
                           "answers": {"true_intent": {"choice": "close_topic"},
                                       "she_needs": {"choice": "nothing"},
                                       "best_action": {"choice": "say_less"}}})
        self.assertEqual(self.overlay.cands, [])
        self.assertFalse(self.overlay._current)
        self.assertEqual(self.overlay.emptyTitle.text(), "本轮无需回复")
        self.overlay.set_busy(False)
        self.assertEqual(self.overlay.emptyTitle.text(), "本轮无需回复")
        self.assertIn("新消息", self.overlay.emptyHint.text())
        self.assertIn("无需回复", self.overlay.status.text())
        self.assertFalse(self.overlay.insight.isHidden())
        self.assertIn("平和结束话题", self.overlay.intent.text())
        self.assertTrue(self.overlay.setupButton.isHidden())

    def test_new_reply_result_replaces_previous_no_reply_state(self):
        self.overlay.set_chat("月仔")
        self.overlay.show({"reply_required": False, "candidates": [], "answers": {}})
        self.overlay.show({"reply_required": True, "candidates": ["几点出发？"], "best_index": 0})
        self.assertEqual(self.overlay.cands, ["几点出发？"])
        self.assertTrue(self.overlay._current)
        self.assertTrue(self.overlay.empty.isHidden())
        self.assertIn("建议已更新", self.overlay.status.text())

    def test_regular_chat_retains_auto_reply_controls(self):
        self.overlay.set_chat("月仔")
        self.assertTrue(self.overlay.autoReplySwitch.isEnabled())
        self.assertTrue(self.overlay.autoKeyBox.isEnabled())


if __name__ == "__main__":
    unittest.main()
