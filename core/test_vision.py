import json
import unittest
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

from core import llm, vision
from core.jev_client import JevError


class VisionProtocolTests(unittest.TestCase):
    def setUp(self):
        output = BytesIO()
        Image.new("RGB", (20, 20), "red").save(output, format="PNG")
        self.image = output.getvalue()

    def test_openai_sends_only_target_png_as_base64_data_url(self):
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])
        with patch("openai.OpenAI", return_value=client):
            llm.chat("openai", "https://token.sensenova.cn/v1", "test-key", "test-model", "规则", ["目标图"],
                     image=self.image, extra_body={"reasoning_effort": "none"})
        call = client.chat.completions.create.call_args.kwargs
        parts = call["messages"][-1]["content"]
        self.assertEqual(parts[0], {"type": "text", "text": "目标图"})
        self.assertTrue(parts[1]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertEqual(call["extra_body"], {"reasoning_effort": "none"})
        self.assertEqual(len(parts), 2)

    def test_anthropic_uses_base64_image_source(self):
        client = Mock()
        client.messages.create.return_value = SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")])
        with patch("anthropic.Anthropic", return_value=client):
            llm.chat("anthropic", None, "test-key", "test-model", "规则", ["目标图"], image=self.image)
        source = client.messages.create.call_args.kwargs["messages"][-1]["content"][1]["source"]
        self.assertEqual(source["type"], "base64")
        self.assertEqual(source["media_type"], "image/png")

    def test_gemini_uses_inline_png_bytes(self):
        client = Mock()
        client.models.generate_content.return_value = SimpleNamespace(text="ok")
        with patch("google.genai.Client", return_value=client):
            llm.chat("gemini", None, "test-key", "test-model", "规则", ["目标图"], image=self.image)
        part = client.models.generate_content.call_args.kwargs["contents"][-1].parts[1]
        self.assertEqual(part.inline_data.data, self.image)
        self.assertEqual(part.inline_data.mime_type, "image/png")

    def test_invalid_image_is_rejected_before_request(self):
        with patch("openai.OpenAI") as client, self.assertRaises(JevError):
            llm.chat("openai", None, "test-key", "test-model", "规则", ["目标图"], image=b"not an image")
        client.assert_not_called()

    def test_hybrid_uses_ocr_as_data_and_sensenova_parameters(self):
        config = {"provider": "sensenova", "model": "sensenova-6.8-flash-lite", "api_key": "test-key", "mode": "hybrid"}
        raw = json.dumps({"summary": "红色矩形", "uncertainty": "", "readable": True})
        with patch.object(vision, "chat", return_value=raw) as chat:
            result = vision.recognize(self.image, "video_cover", "测试文字", config)
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["source"], "hybrid")
        self.assertIn("测试文字", chat.call_args.args[5][0])
        self.assertEqual(chat.call_args.kwargs["image"], self.image)
        self.assertEqual(chat.call_args.kwargs["extra_body"], {"reasoning_effort": "none"})

    def test_visual_only_does_not_send_ocr(self):
        config = {"provider": "sensenova", "model": "test-model", "api_key": "test-key", "mode": "vision"}
        with patch.object(vision, "chat", return_value='{"summary":"模糊图片","readable":false}') as chat:
            result = vision.recognize(self.image, "image", "不应传入的OCR", config)
        self.assertNotIn("不应传入的OCR", chat.call_args.args[5][0])
        self.assertEqual(result["status"], "partial")

    def test_request_errors_do_not_expose_unsaved_keys_or_image_payloads(self):
        config = {"provider": "sensenova", "model": "test-model", "api_key": "unsaved-secret", "mode": "vision"}
        with patch.object(vision, "chat", side_effect=JevError("unsaved-secret data:image/png;base64,xxx", 400)), \
                self.assertRaises(JevError) as error:
            vision.recognize(self.image, "image", "", config)
        self.assertEqual(error.exception.status, 400)
        self.assertNotIn("unsaved-secret", str(error.exception))
        self.assertNotIn("base64", str(error.exception))

    def test_invalid_structured_result_is_not_treated_as_success(self):
        config = {"provider": "sensenova", "model": "test-model", "api_key": "test-key", "mode": "vision"}
        for raw in ("随便一段话", "[]", '{"summary":"猜测","readable":"true"}'):
            with self.subTest(raw=raw), patch.object(vision, "chat", return_value=raw), self.assertRaises(JevError):
                vision.recognize(self.image, "image", "", config)


if __name__ == "__main__":
    unittest.main()
