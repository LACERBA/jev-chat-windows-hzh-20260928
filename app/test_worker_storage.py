import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from app import settings, worker
from app.capture import Capture
from app.message_store import MessageStore


class WorkerStorageTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "messages.sqlite3"
        self.frame = np.full((20, 30, 3), 245, dtype=np.uint8)
        self.area = (10, 5, 30, 15, np.array([245, 245, 245]), 0)
        self.q = Mock()
        self.reader = Mock(last_ms=10)
        self.reader.read.return_value = [("her", None, "你好", 10)]
        self.capture = Mock(latest=self.frame)
        self.capture.alive.side_effect = [True, False]
        self.capture.settled.return_value = self.frame
        self.capture.snapshot.return_value = self.frame
        self.enabled = Mock()
        self.enabled.is_set.return_value = True
        self.debug = Mock()
        self.debug.is_set.return_value = False
        self.read_requested = threading.Event()
        windll = Mock()
        windll.user32.GetDpiForWindow.return_value = 96
        patches = [patch.object(worker, "Capture", return_value=self.capture),
                   patch.object(worker, "Reader", return_value=self.reader),
                   patch.object(worker, "chat_area", return_value=self.area),
                   patch.object(worker, "read_title", return_value="月仔"),
                   patch.object(worker, "UnreadTracker", return_value=Mock(update=Mock(return_value=([], [])))),
                   patch.object(worker, "unminimize"), patch.object(worker.time, "sleep"),
                   patch.object(worker.ctypes, "windll", windll),
                   patch.object(settings, "message_db_path", return_value=self.path),
                   patch.object(settings, "_read", side_effect=lambda name, default=None: default)]
        for target in patches:
            target.start()
            self.addCleanup(target.stop)

    def run_worker(self):
        worker.run(self.q, 1, self.enabled, self.debug, "wechat", self.read_requested)
        return [call.args[0][1] for call in self.q.put.call_args_list if call.args[0][0] == "snapshot"]

    def test_worker_saves_before_publishing_snapshot(self):
        snapshots = self.run_worker()
        store = MessageStore(self.path, read_only=True)
        self.addCleanup(store.close)
        self.assertEqual(snapshots[0]["new"], [("her", None, "你好")])
        self.assertEqual(store.history("wechat", "月仔"), snapshots[0]["history"])

    def test_requested_read_publishes_unchanged_persisted_snapshot(self):
        store = MessageStore(self.path)
        store.record_snapshot("wechat", "月仔", "月仔", [("her", None, "你好")], self.area[:4])
        store.close()
        self.read_requested.set()
        snapshots = self.run_worker()
        self.capture.snapshot.assert_called_once()
        self.capture.settled.assert_not_called()
        self.assertEqual(snapshots[0]["new"], [])
        self.assertEqual(snapshots[0]["visible"], [("her", None, "你好")])
        self.assertFalse(self.read_requested.is_set())

    def test_ocr_crossing_chat_switch_is_not_committed(self):
        different = self.frame.copy()
        different[:5, 10:] = 0
        self.capture.latest = different
        self.assertEqual(self.run_worker(), [])
        self.assertTrue(self.read_requested.is_set())
        store = MessageStore(self.path, read_only=True)
        self.addCleanup(store.close)
        self.assertEqual(store.history("wechat", "月仔"), [])

    def test_unknown_header_does_not_attribute_messages_to_previous_chat(self):
        with patch.object(worker, "read_title", return_value=""):
            snapshots = self.run_worker()
        self.assertFalse(snapshots[0]["title_confirmed"])
        self.assertEqual(snapshots[0]["new"], [])
        self.assertEqual(snapshots[0]["history"], [])

    def test_storage_failure_does_not_consume_frame_for_retry(self):
        self.capture.alive.side_effect = [True, True, False]
        real_record = MessageStore.record_snapshot
        attempts = []

        def record(store, *args, **kwargs):
            attempts.append(True)
            if len(attempts) == 1:
                raise OSError("storage unavailable")
            return real_record(store, *args, **kwargs)

        with patch.object(MessageStore, "record_snapshot", record):
            snapshots = self.run_worker()
        self.assertEqual(len(attempts), 2)
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0]["new"], [("her", None, "你好")])

    def test_media_image_packet_is_only_the_target_crop(self):
        from app.media import media_token

        item = {"fingerprint": "1" * 32, "kind": "image", "ocr_text": "", "who": "her", "sender": None,
                "rect": (0, 0, 4, 4), "pixels": np.full((4, 4, 3), 100, dtype=np.uint8)}
        self.reader.last_media = [item]
        self.reader.read.return_value = [("her", None, media_token(item["fingerprint"]), 10)]
        with patch.object(settings, "vision_upload", return_value=True), \
                patch.object(settings, "media_mode", return_value="hybrid"):
            snapshots = self.run_worker()
        job = snapshots[0]["media_jobs"][0]
        self.assertTrue(job["image"].startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertNotIn("pixels", job)
        self.assertNotIn("rect", job)
        self.assertEqual(next(iter(snapshots[0]["media_state"].values()))["status"], "pending")

    def test_returning_to_chat_requeues_media_abandoned_during_settle_period(self):
        from app.media import media_token

        frames = [self.frame.copy() for _ in range(3)]
        frames[1][:5, 10:] = 240
        self.capture.alive.side_effect = [True, True, True, False]

        def next_frame():
            self.capture.latest = frames.pop(0)
            return self.capture.latest

        def read(*args):
            self.reader.last_media = [{"fingerprint": "1" * 32, "kind": "image", "ocr_text": "",
                                      "who": "her", "sender": None, "rect": (0, 0, 4, 4),
                                      "pixels": np.full((4, 4, 3), 100, dtype=np.uint8)}]
            return [("her", None, media_token("1" * 32), 10)]

        self.capture.settled.side_effect = next_frame
        self.reader.read.side_effect = read
        with patch.object(worker, "read_title", side_effect=["月仔", "阿梅", "月仔"]), \
                patch.object(settings, "vision_upload", return_value=True), \
                patch.object(settings, "media_mode", return_value="hybrid"):
            snapshots = self.run_worker()
        self.assertEqual([item["title"] for item in snapshots], ["月仔", "阿梅", "月仔"])
        self.assertEqual([len(item["media_jobs"]) for item in snapshots], [1, 1, 1])
        self.assertEqual(snapshots[0]["media_jobs"][0]["message_id"], snapshots[2]["media_jobs"][0]["message_id"])

    def test_only_collection_does_not_queue_visual_requests(self):
        from app.media import media_token

        item = {"fingerprint": "1" * 32, "kind": "image", "ocr_text": "", "who": "her", "sender": None}
        self.reader.last_media = [item]
        self.reader.read.return_value = [("her", None, media_token(item["fingerprint"]), 10)]
        with patch.object(settings, "vision_upload", return_value=True), \
                patch.object(settings, "media_mode", return_value="hybrid"), \
                patch.object(settings, "analysis_enabled", return_value=False):
            snapshots = self.run_worker()
        self.assertEqual(snapshots[0]["media_jobs"], [])
        self.assertIn("图片消息", snapshots[0]["history"][0][2])


class CaptureSnapshotTests(unittest.TestCase):
    def test_static_frame_is_available_for_explicit_reread(self):
        capture = Capture.__new__(Capture)
        capture.pending = None
        capture.latest = np.zeros((10, 10, 3), dtype=np.uint8)
        self.assertIs(capture.snapshot(), capture.latest)

    def test_explicit_reread_still_waits_for_pending_frame_to_settle(self):
        capture = Capture.__new__(Capture)
        capture.pending = np.zeros((10, 10, 3), dtype=np.uint8)
        capture.latest = capture.pending
        with patch.object(capture, "settled", return_value=None) as settled:
            self.assertIsNone(capture.snapshot())
        settled.assert_called_once()


if __name__ == "__main__":
    unittest.main()
