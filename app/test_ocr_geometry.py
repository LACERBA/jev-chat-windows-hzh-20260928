import os
import unittest
from unittest.mock import Mock, patch

import numpy as np

from app.capture import chat_area
from app.ocr import Reader


class MessageGeometryTests(unittest.TestCase):
    def test_large_white_image_does_not_change_chat_pane_background(self):
        frame = np.full((600, 900, 3), 245, dtype=np.uint8)
        frame[:, :260] = 255
        frame[:, 260:264] = 210
        frame[450:452, 264:] = 210
        frame[80:435, 465:865] = 255
        area = chat_area(frame)
        self.assertIsNotNone(area)
        self.assertGreaterEqual(area[0], 260)
        self.assertLess(area[1], 120)
        self.assertEqual(area[3], 450)
        self.assertEqual(tuple(area[4]), (245, 245, 245))

    def test_right_hand_white_image_text_is_not_an_incoming_message(self):
        frame = np.full((300, 500, 3), 245, dtype=np.uint8)
        frame[20:120, 320:480] = 255
        frame[180:220, 30:180] = 235
        image_box = [[340, 40], [450, 40], [450, 60], [340, 60]]
        message_box = [[40, 190], [170, 190], [170, 210], [40, 210]]
        engine = Mock(return_value=([(image_box, "我叫MT4", 0.99),
                                    (message_box, "6机器人都用上了", 0.99)], None))
        with patch("app.ocr._engine", return_value=engine), \
                patch("app.ocr.who_said", return_value=("her", np.array([235, 235, 235]), 14)):
            reader = Reader()
            messages = reader.read(frame, np.array([245, 245, 245]))
        self.assertEqual([line[2] for line in messages], ["6机器人都用上了"])
        self.assertEqual(reader.last_boxes[0][-2], "image")

    @unittest.skipUnless(os.environ.get("JEV_OCR_TEST_IMAGE"), "需要用户提供的本地截图")
    def test_user_screenshot_contains_latest_bubble_not_sidebar_preview(self):
        from PIL import Image

        frame = np.array(Image.open(os.environ["JEV_OCR_TEST_IMAGE"]).convert("RGB"))[:, :701]
        area = chat_area(frame)
        self.assertIsNotNone(area)
        self.assertGreaterEqual(area[0], 170)
        reader = Reader()
        messages = reader.read(frame[area[1]:area[3], area[0]:area[2]], area[4])
        text = [line[2] for line in messages]
        self.assertIn("6机器人都用上了", text)
        self.assertNotIn("我叫MT4", text)


if __name__ == "__main__":
    unittest.main()
