import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

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

    def test_identical_snapshot_is_not_inserted_repeatedly(self):
        visible = [("her", None, "你好")]
        first = self.record(visible)
        second = self.record(visible)
        self.assertEqual(first["id"], second["id"])
        self.assertEqual(self.store.connection.execute("SELECT COUNT(*) FROM ocr_snapshots").fetchone()[0], 1)


if __name__ == "__main__":
    unittest.main()
