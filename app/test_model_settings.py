import unittest
from unittest.mock import mock_open, patch

from app import settings
from core import jev_client
from core.providers import JEV_ENV, JUDGE_ENV, LLM_ENV, VISION_ENV


class ModelSettingsTests(unittest.TestCase):
    def _config(self, **changes):
        value = {
            "judge_mode": "jev_fallback",
            "judge_reuse_draft": True,
            "jev_provider": "openrouter",
            "jev_model": "typesafe/jev-1.13",
            "draft_provider": "deepseek",
            "draft_model": "deepseek-chat",
        }
        value.update(changes)
        return value

    def test_default_fallback_reuses_draft_and_requires_jev_and_llm_keys(self):
        config = self._config()
        keys = {JEV_ENV: "jev-key", LLM_ENV: "llm-key", JUDGE_ENV: ""}
        with patch.object(settings, "_read", side_effect=lambda name, default=None: config.get(name, default)), \
                patch.object(settings, "_read_env", side_effect=lambda name: keys.get(name, "")):
            self.assertEqual(settings.judge_mode(), "jev_fallback")
            self.assertEqual(settings.general_judge_config()["api_key"], "llm-key")
            self.assertTrue(settings.models_ready())
            keys[JEV_ENV] = ""
            self.assertFalse(settings.models_ready())

    def test_llm_only_independent_requires_third_key_but_not_jev_key(self):
        config = self._config(judge_mode="llm_only", judge_reuse_draft=False,
                              judge_provider="deepseek", judge_model="deepseek-chat")
        keys = {JEV_ENV: "", LLM_ENV: "llm-key", JUDGE_ENV: "judge-key"}
        with patch.object(settings, "_read", side_effect=lambda name, default=None: config.get(name, default)), \
                patch.object(settings, "_read_env", side_effect=lambda name: keys.get(name, "")):
            self.assertEqual(settings.general_judge_config()["api_key"], "judge-key")
            self.assertTrue(settings.models_ready())
            keys[JUDGE_ENV] = ""
            self.assertFalse(settings.models_ready())

    def test_saves_third_key_only_to_environment(self):
        config = self._config()
        with patch.object(settings, "_read", side_effect=lambda name, default=None: config.get(name, default)), \
                patch.object(settings, "_read_env", return_value=""), \
                patch.object(settings, "_set_key") as set_key, \
                patch.object(settings, "_notify_env"), patch("builtins.open", mock_open()), \
                patch.object(settings.json, "dump") as dump:
            settings.save(judge_mode_text="llm_only", judge_reuse_draft_on=False,
                          judge_provider_text="deepseek", judge_key_text="judge-secret",
                          judge_model_text="deepseek-chat")
        set_key.assert_any_call(JUDGE_ENV, "judge-secret")
        data = dump.call_args.args[0]
        self.assertEqual(data["judge_mode"], "llm_only")
        self.assertFalse(data["judge_reuse_draft"])
        self.assertNotIn("judge-secret", str(data))

    def test_processing_defaults_keep_sending_disabled(self):
        with patch.object(settings, "_read", side_effect=lambda name, default=None: default):
            self.assertTrue(settings.analysis_enabled())
            self.assertFalse(settings.send_enabled())

    def test_processing_mode_save_preserves_independent_switches_and_chat_authorization(self):
        config = self._config(analysis_enabled=False, send_enabled=True,
                              auto_reply_chats={"wechat:月仔": {"enabled": True}})
        with patch.object(settings, "_read", side_effect=lambda name, default=None: config.get(name, default)), \
                patch.object(settings, "_read_env", return_value=""), \
                patch("builtins.open", mock_open()), patch.object(settings.json, "dump") as dump:
            settings.save(send_enabled_on=False)
        data = dump.call_args.args[0]
        self.assertFalse(data["analysis_enabled"])
        self.assertFalse(data["send_enabled"])
        self.assertEqual(data["auto_reply_chats"], config["auto_reply_chats"])

    def test_judge_key_is_redacted(self):
        with patch.dict(jev_client.os.environ, {JUDGE_ENV: "judge-secret"}, clear=False):
            self.assertEqual(jev_client.redact_secrets("key=judge-secret"), "key=[REDACTED]")

    def test_media_defaults_do_not_upload_images(self):
        with patch.object(settings, "_read", side_effect=lambda name, default=None: default):
            self.assertEqual(settings.media_mode(), "ocr")
            self.assertFalse(settings.vision_upload())
            self.assertEqual(settings.vision_provider(), "sensenova")
            self.assertEqual(settings.vision_model(), "sensenova-6.8-flash-lite")

    def test_vision_key_is_saved_only_to_environment_and_redacted(self):
        config = self._config()
        with patch.object(settings, "_read", side_effect=lambda name, default=None: config.get(name, default)), \
                patch.object(settings, "_read_env", return_value=""), patch.object(settings, "_set_key") as set_key, \
                patch.object(settings, "_notify_env"), patch("builtins.open", mock_open()), \
                patch.object(settings.json, "dump") as dump:
            settings.save(media_mode_text="hybrid", vision_upload_on=True, vision_key_text="vision-secret")
        set_key.assert_any_call(VISION_ENV, "vision-secret")
        self.assertEqual(dump.call_args.args[0]["media_mode"], "hybrid")
        self.assertNotIn("vision-secret", str(dump.call_args.args[0]))
        with patch.dict(jev_client.os.environ, {VISION_ENV: "vision-secret"}, clear=False):
            self.assertEqual(jev_client.redact_secrets("key=vision-secret"), "key=[REDACTED]")

    def test_unrelated_save_preserves_vision_config(self):
        config = self._config(media_mode="hybrid", vision_upload=True, vision_provider="custom_openai",
                              vision_model="local-vision", vision_base_url="http://localhost:1234/v1")
        with patch.object(settings, "_read", side_effect=lambda name, default=None: config.get(name, default)), \
                patch.object(settings, "_read_env", return_value=""), patch("builtins.open", mock_open()), \
                patch.object(settings.json, "dump") as dump:
            settings.save(debug_view_on=True)
        data = dump.call_args.args[0]
        for name in ("media_mode", "vision_upload", "vision_provider", "vision_model", "vision_base_url"):
            self.assertEqual(data[name], config[name])


if __name__ == "__main__":
    unittest.main()
