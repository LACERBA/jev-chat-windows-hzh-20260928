import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.message_store import MessageStore, new_messages


class MessageStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "messages.sqlite3"
        self.store = MessageStore(self.path)
        self.addCleanup(self.store.close)
        self.area = (200, 60, 600, 500)

    def record(self, visible, title="月仔", app="wechat", **kwargs):
        return self.store.record_snapshot(app, title, title, visible, self.area, **kwargs)

    def test_repeated_incoming_message_is_saved_after_outgoing_reply(self):
        first = [("her", None, "是吗 哈哈"), ("me", None, "到底怎么了")]
        self.record(first)
        latest = self.record(first + [("her", None, "是吗 哈哈")])
        self.assertEqual(latest["new"], [("her", None, "是吗 哈哈")])
        self.assertFalse(latest["baseline"])
        self.assertEqual(self.store.history("wechat", "月仔"), first + latest["new"])
        self.assertEqual(self.record(first + latest["new"])["new"], [])

    def test_restart_and_read_only_connection_restore_without_duplicates(self):
        messages = [("her", None, "你好"), ("me", None, "今天怎么样")]
        self.record(messages)
        reopened = MessageStore(self.path)
        self.addCleanup(reopened.close)
        reader = MessageStore(self.path, read_only=True)
        self.addCleanup(reader.close)
        self.assertEqual(reader.history("wechat", "月仔"), messages)
        same = reopened.record_snapshot("wechat", "月仔", "月仔", messages, self.area)
        self.assertEqual(same["new"], [])
        self.record(messages + [("her", None, "还好")])
        self.assertEqual(reader.history("wechat", "月仔")[-1], ("her", None, "还好"))
        with self.assertRaises(sqlite3.OperationalError):
            reader.connection.execute("DELETE FROM messages")

    def test_conversation_and_app_isolation_with_quoted_titles(self):
        self.record([("her", None, "微信消息")])
        self.record([("her", None, "另一个应用")], app="kakaotalk")
        title = "名字' OR 1=1 --"
        self.record([("her", "发言人", "其他会话")], title=title)
        self.assertEqual(self.store.history("wechat", "月仔"), [("her", None, "微信消息")])
        self.assertEqual(self.store.history("kakaotalk", "月仔"), [("her", None, "另一个应用")])
        self.assertEqual(self.store.history("wechat", title), [("her", "发言人", "其他会话")])

    def test_consecutive_identical_bubbles_are_distinct(self):
        bubble = ("her", None, "哈哈")
        self.record([bubble])
        self.assertEqual(self.record([bubble, bubble])["new"], [bubble])
        self.assertEqual(self.record([bubble, bubble])["new"], [])
        self.assertEqual(self.store.history("wechat", "月仔"), [bubble, bubble])

    def test_scroll_does_not_trigger_new_messages_but_keeps_visible_snapshot(self):
        messages = [("her", None, "你好"), ("me", None, "今天怎么样"), ("her", None, "还好")]
        self.record(messages)
        visible = [("her", None, "更早的消息")] + messages[:2]
        snapshot = self.record(visible)
        self.assertEqual(snapshot["new"], [])
        self.assertEqual(self.store.history("wechat", "月仔"), messages)
        raw = self.store.connection.execute(
            "SELECT visible_messages FROM ocr_snapshots WHERE id = ?", (snapshot["id"],)).fetchone()[0]
        self.assertEqual(json.loads(raw), [list(message) for message in visible])

    def test_ocr_jitter_and_missing_sender_do_not_duplicate(self):
        self.record([("her", "小王", "今天真是不好意思"), ("me", None, "没事的")])
        same = self.record([("her", None, "今天真是不好竟思"), ("me", None, "没事的")])
        self.assertEqual(same["new"], [])
        self.assertEqual(len(same["history"]), 2)

    def test_different_group_speakers_are_not_collapsed(self):
        history = [("her", "甲", "收到")]
        new, _ = new_messages(history, history + [("her", "乙", "收到")])
        self.assertEqual(new, [("her", "乙", "收到")])

    def test_unconfirmed_title_and_empty_read_do_not_change_message_history(self):
        original = [("her", None, "正常消息")]
        self.record(original)
        unknown = self.record([("her", None, "不能归入旧会话")], title_confirmed=False)
        empty = self.record([])
        self.assertEqual(unknown["new"], [])
        self.assertEqual(unknown["status"], "unconfirmed")
        self.assertEqual(empty["status"], "empty")
        self.assertEqual(self.store.history("wechat", "月仔"), original)

    def test_unanchored_read_is_marked_as_baseline(self):
        self.record([("her", None, "之前的消息")])
        snapshot = self.record([("her", None, "完全没有交集的文本")])
        self.assertTrue(snapshot["baseline"])
        self.assertTrue(snapshot["discontinuous"])

    def test_invalid_batch_rolls_back_messages_and_conversation(self):
        with self.assertRaises(sqlite3.IntegrityError):
            self.record([("her", None, "正常"), ("invalid", None, "非法角色")])
        self.assertEqual(self.store.history("wechat", "月仔"), [])
        self.assertEqual(self.store.conversations("wechat"), [])

    def test_unanchored_batch_is_quarantined_and_does_not_block_latest_message(self):
        history = [("me", None, "claude code 确实猛"), ("me", None, "来进行图灵测试")]
        self.record(history)
        bad = self.record([("her", "ALiGN前端...", "我叫MT4")])
        self.assertEqual(bad["history"], history)
        latest = self.record(history + [("her", None, "6机器人都用上了")])
        self.assertEqual(latest["new"], [("her", None, "6机器人都用上了")])
        self.assertEqual(self.store.connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0], 4)

    def test_stable_ocr_correction_keeps_original_and_audit(self):
        self.record([("me", None, "刚刚怎么了"), ("her", None, "今天真是不好竟思"),
                     ("me", None, "没事的")])
        visible = [("me", None, "刚刚怎么了"), ("her", None, "今天真是不好意思"),
                   ("me", None, "没事的")]
        first = self.record(visible)
        second = self.record(visible)
        self.assertEqual(first["history"][1][2], "今天真是不好竟思")
        self.assertEqual(second["history"][1][2], "今天真是不好意思")
        self.assertTrue(second["corrected"])
        original = self.store.connection.execute("SELECT text FROM messages ORDER BY id LIMIT 1 OFFSET 1").fetchone()[0]
        self.assertEqual(original, "今天真是不好竟思")
        self.assertEqual(self.store.connection.execute("SELECT COUNT(*) FROM message_revisions").fetchone()[0], 1)

    def test_legacy_pollution_is_quarantined_without_deleting_original(self):
        original = [("me", None, "claude code 确实猛"), ("me", None, "图灵测试")]
        self.record(original)
        conversation_id = self.store.connection.execute("SELECT id FROM conversations").fetchone()[0]
        with self.store.connection:
            bad_id = self.store.connection.execute("""
                INSERT INTO messages(conversation_id, who, sender, text, observed_at, baseline)
                VALUES (?, 'her', 'ALiGN前端...', '我叫MT4', 999999, 1)
            """, (conversation_id,)).lastrowid
        reopened = MessageStore(self.path)
        self.addCleanup(reopened.close)
        self.assertEqual(reopened.history("wechat", "月仔"), original)
        self.assertEqual(reopened.connection.execute(
            "SELECT text FROM messages WHERE id = ?", (bad_id,)).fetchone()[0], "我叫MT4")
        revision = reopened.connection.execute(
            "SELECT reason FROM message_revisions WHERE message_id = ?", (bad_id,)).fetchone()
        self.assertEqual(revision[0], "legacy_unanchored_batch")
        latest = reopened.record_snapshot("wechat", "月仔", "月仔",
                                         original + [("her", None, "6机器人都用上了")], self.area)
        self.assertEqual(latest["new"], [("her", None, "6机器人都用上了")])

    def test_oscillating_ocr_does_not_revise_original(self):
        original = [("me", None, "刚刚怎么了"), ("her", None, "今天真是不好竟思"),
                    ("me", None, "没事的")]
        changed = [original[0], ("her", None, "今天真是不好意思"), original[2]]
        self.record(original)
        self.record(changed)
        self.record(original)
        snapshot = self.record(changed)
        self.assertFalse(snapshot["corrected"])
        self.assertEqual(snapshot["history"], original)

    def test_absent_messages_are_not_rejected_as_ocr_errors(self):
        original = [("her", None, "第一条"), ("me", None, "第二条"), ("her", None, "第三条")]
        self.record(original)
        self.record(original[1:])
        self.record(original[:2])
        self.assertEqual(self.store.history("wechat", "月仔"), original)
        self.assertEqual(self.store.connection.execute("SELECT COUNT(*) FROM message_revisions").fetchone()[0], 0)

    def test_identical_snapshot_is_not_inserted_repeatedly(self):
        visible = [("her", None, "你好")]
        first = self.record(visible)
        second = self.record(visible)
        self.assertEqual(first["id"], second["id"])
        self.assertEqual(self.store.connection.execute("SELECT COUNT(*) FROM ocr_snapshots").fetchone()[0], 1)

    def media(self, fingerprint="1" * 32, text="", who="her"):
        from app.media import media_token

        item = {"fingerprint": fingerprint, "kind": "image", "ocr_text": text, "who": who,
                "sender": None}
        return (who, None, media_token(fingerprint)), item

    def test_context_reads_message_order_once_for_ids_and_text(self):
        bubble, item = self.media()
        self.record([bubble, ("her", None, "最新文字")], media=[item])
        with patch.object(self.store, "_history_rows", wraps=self.store._history_rows) as rows:
            context = self.store.context("wechat", "月仔")
        rows.assert_called_once()
        self.assertEqual(len(context["history"]), len(context["history_ids"]))
        self.assertIn("图片消息", context["history"][0][2])
        self.assertIn(context["history_ids"][0], context["media_state"])

    def test_media_identity_is_not_based_on_generated_summary(self):
        bubble, item = self.media(text="通知正文")
        snapshot = self.record([bubble], media=[item], vision=True)
        message_id = snapshot["media_jobs"][0]["message_id"]
        self.assertTrue(self.store.update_media("wechat", "月仔", message_id, item["fingerprint"],
                                              {"summary": "通知图片", "status": "ready", "source": "hybrid"}))
        again = self.record([bubble], media=[item], vision=True)
        self.assertEqual(again["new"], [])
        self.assertEqual(again["media_jobs"], [])
        self.assertIn("通知正文", again["history"][0][2])
        self.assertIn("通知图片", again["history"][0][2])
        original = self.store.connection.execute("SELECT text FROM messages").fetchone()[0]
        self.assertEqual(original, bubble[2])

    def test_consecutive_identical_media_are_distinct_messages(self):
        bubble, item = self.media()
        self.record([bubble], media=[item])
        latest = self.record([bubble, bubble], media=[item])
        self.assertEqual(len(latest["new"]), 1)
        self.assertEqual(len(latest["media_state"]), 2)
        self.assertEqual(self.record([bubble, bubble], media=[item])["new"], [])

    def test_different_media_tokens_do_not_match_by_shared_prefix(self):
        first, item = self.media()
        second, other = self.media("2" * 32)
        self.record([first], media=[item])
        latest = self.record([first, second], media=[item, other])
        self.assertEqual(len(latest["new"]), 1)
        self.assertEqual(len(latest["history"]), 2)

    def test_only_latest_baseline_media_is_sent_to_vision(self):
        first, item = self.media()
        second, other = self.media("2" * 32)
        snapshot = self.record([first, second], media=[item, other], vision=True)
        self.assertEqual(len(snapshot["media_jobs"]), 1)
        self.assertEqual(snapshot["media_jobs"][0]["fingerprint"], other["fingerprint"])

    def test_media_result_is_scoped_to_conversation_and_fingerprint(self):
        bubble, item = self.media()
        snapshot = self.record([bubble], media=[item], vision=True)
        message_id = snapshot["media_jobs"][0]["message_id"]
        result = {"summary": "不应写入", "status": "ready"}
        self.assertFalse(self.store.update_media("wechat", "其他会话", message_id, item["fingerprint"], result))
        self.assertFalse(self.store.update_media("wechat", "月仔", message_id, "2" * 32, result))
        self.assertEqual(self.store.media_state("wechat", "月仔")[message_id]["status"], "pending")

    def test_disabling_vision_releases_pending_media_without_deleting_original(self):
        bubble, item = self.media()
        snapshot = self.record([bubble], media=[item], vision=True)
        latest = self.record([bubble], media=[item], vision=False)
        self.assertEqual(latest["new"], [])
        self.assertEqual(next(iter(latest["media_state"].values()))["status"], "ocr")
        message_id = snapshot["media_jobs"][0]["message_id"]
        self.assertFalse(self.store.update_media("wechat", "月仔", message_id, item["fingerprint"],
                                               {"summary": "过期结果", "status": "ready"}))

    def test_restart_can_retry_visible_pending_media(self):
        bubble, item = self.media()
        first = self.record([bubble], media=[item], vision=True)
        second = self.record([bubble], media=[item], vision=True)
        self.assertEqual(second["new"], [])
        self.assertEqual(first["media_jobs"][0]["message_id"], second["media_jobs"][0]["message_id"])


if __name__ == "__main__":
    unittest.main()
