import os
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
from PIL import Image

from app.media import MediaReader, encode_crop, same_media


class MediaGeometryTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((400, 600, 3), 245, dtype=np.uint8)
        self.frame[50:90, 15:55] = (20, 30, 50)
        self.frame[50:190, 75:255] = (160, 80, 60)
        self.ocr = Mock(return_value=([], None))
        self.reader = MediaReader()
        self.bg = np.array([245, 245, 245])

    def read(self, lines=None, boxes=None):
        return self.reader.read(self.frame, self.bg, lines or [], boxes or [], self.ocr)

    def test_textless_image_is_detected_without_including_avatar_or_other_message(self):
        self.frame[260:300, 15:55] = (20, 30, 50)
        self.frame[260:300, 75:240] = 235
        lines = [("her", None, "其他文字", 270)]
        messages, media = self.read(lines, [(80, 270, 200, 290, "her", "其他文字")])
        self.assertEqual(len(media), 1)
        self.assertEqual(media[0]["rect"], (75, 50, 255, 190))
        self.assertEqual(media[0]["kind"], "image")
        self.assertEqual(media[0]["pixels"].shape, (140, 180, 3))
        self.assertEqual(messages[-1], lines[0])

    def test_outgoing_image_uses_layout_not_image_color(self):
        self.frame[:] = 245
        self.frame[50:90, 545:585] = (20, 30, 50)
        self.frame[50:190, 350:525] = (100, 100, 100)
        messages, media = self.read()
        self.assertEqual(len(media), 1)
        self.assertEqual(messages[0][0], "me")
        self.assertEqual(media[0]["rect"], (350, 50, 525, 190))

    def test_plain_text_is_not_converted_to_media(self):
        self.frame[50:190, 75:255] = 245
        self.frame[50:90, 75:255] = 235
        lines = [("her", None, "普通文字", 60)]
        messages, media = self.read(lines, [(80, 60, 200, 80, "her", "普通文字")])
        self.assertEqual(media, [])
        self.assertEqual(messages, lines)

    def test_inline_emoji_stays_in_one_message_with_text(self):
        self.frame[50:190, 75:255] = 245
        self.frame[50:90, 75:255] = 235
        self.frame[60:80, 220:240] = (250, 180, 20)
        lines = [("her", None, "文字", 60)]
        self.ocr.return_value = ([[[[0, 0], [100, 0], [100, 20], [0, 20]], "文字", 0.99]], None)
        messages, media = self.read(lines, [(80, 60, 200, 80, "her", "文字")])
        self.assertEqual(len(messages), 1)
        self.assertEqual(media[0]["kind"], "mixed")
        self.assertEqual(media[0]["ocr_text"], "文字")

    def test_animation_changes_do_not_create_a_new_message(self):
        first, _ = self.read()
        self.frame[100:130, 120:150] = (10, 150, 200)
        second, _ = self.read()
        self.assertEqual(first[0][2], second[0][2])

    def test_consecutive_identical_images_remain_two_bubbles(self):
        self.frame[240:280, 15:55] = (20, 30, 50)
        self.frame[240:380, 75:255] = (160, 80, 60)
        messages, media = self.read()
        self.assertEqual(len(media), 2)
        self.assertEqual(len(messages), 2)

    def test_clipped_media_is_not_uploaded(self):
        self.frame[50:, 75:255] = (160, 80, 60)
        self.assertEqual(self.read()[1], [])

    def test_resize_and_restart_keep_same_media_identity(self):
        _, first = self.read()
        resized = np.array(Image.fromarray(self.frame).resize((750, 500)))
        _, second = MediaReader().read(resized, self.bg, [], [], self.ocr)
        self.assertEqual(len(second), 1)
        self.assertTrue(same_media(first[0]["fingerprint"], second[0]["fingerprint"]))

    def test_multiple_animations_do_not_repeat_ocr_or_create_new_messages(self):
        self.frame[240:280, 15:55] = (20, 30, 50)
        self.frame[240:380, 75:255] = (160, 80, 60)
        first, _ = self.read()
        calls = self.ocr.call_count
        self.frame[100:130, 120:150] = (10, 150, 200)
        self.frame[290:320, 120:150] = (10, 150, 200)
        second, _ = self.read()
        self.assertEqual([line[2] for line in first], [line[2] for line in second])
        self.assertEqual(self.ocr.call_count, calls)

    def test_payment_card_is_not_sent_as_media(self):
        self.ocr.return_value = ([[[[0, 0], [100, 0], [100, 20], [0, 20]], "微信红包", 0.99]], None)
        self.assertEqual(self.read()[1], [])

    def test_png_encoding_is_in_memory(self):
        encoded = encode_crop(self.frame[50:190, 75:255])
        self.assertTrue(encoded.startswith(b"\x89PNG\r\n\x1a\n"))

    @unittest.skipUnless(os.environ.get("JEV_MEDIA_TEST_DIR"), "需要用户提供的本地媒体截图目录")
    def test_four_reference_screenshots(self):
        from app.capture import chat_area
        from app.ocr import Reader

        directory = Path(os.environ["JEV_MEDIA_TEST_DIR"])
        for number, expected in ((1, "video_cover"), (2, "image"), (3, "image"), (4, "emoji")):
            with self.subTest(number=number):
                frame = np.array(Image.open(directory / f"{number}.png").convert("RGB"))
                area = chat_area(frame)
                self.assertIsNotNone(area)
                x0, y0, x1, y1, bg, _ = area
                reader = Reader()
                reader.read(frame[y0:y1, x0:x1], bg)
                self.assertTrue(reader.last_media, f"示例 {number} 没有检测到媒体区域")
                target = reader.last_media[-1]
                self.assertEqual(target["kind"], expected, target["rect"])
                left, top, right, bottom = target["rect"]
                self.assertGreater(x0 + left, frame.shape[1] * 0.32)
                self.assertLess(x0 + right, frame.shape[1] * 0.65)
                self.assertGreater(y0 + top, frame.shape[0] * 0.35)
                self.assertLessEqual(y0 + bottom, y1)


if __name__ == "__main__":
    unittest.main()
