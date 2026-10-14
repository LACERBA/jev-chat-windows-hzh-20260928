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

    def test_decisions_configuration_defaults_and_readiness(self):
        config = self._config(jev_provider="openai_decisions", jev_model="", judge_mode="jev_only")
        keys = {JEV_ENV: "decision-key", LLM_ENV: "draft-key"}
        with patch.object(settings, "_read", side_effect=lambda name, default=None: config.get(name, default)), \
                patch.object(settings, "_read_env", side_effect=lambda name: keys.get(name, "")):
            self.assertEqual(settings.jev_provider(), "openai_decisions")
            self.assertEqual(settings.jev_model(), "gpt-6-luna")
            self.assertEqual(settings.jev_base_url(), "")
            self.assertTrue(settings.models_ready())
            config["jev_base_url"] = " https://quick-pivot.com/v1 "
            config["jev_model"] = "future-model"
            self.assertEqual(settings.jev_base_url(), "https://quick-pivot.com/v1")
            self.assertEqual(settings.jev_model(), "future-model")
            keys[JEV_ENV] = ""
            self.assertFalse(settings.models_ready())

    def test_saves_decisions_address_and_model_but_not_key_to_config(self):
        config = self._config()
        with patch.object(settings, "_read", side_effect=lambda name, default=None: config.get(name, default)), \
                patch.object(settings, "_read_env", return_value=""), \
                patch.object(settings, "_set_key") as set_key, \
                patch.object(settings, "_notify_env"), patch("builtins.open", mock_open()), \
                patch.object(settings.json, "dump") as dump:
            settings.save(jev_provider_text="openai_decisions", jev_model_text="gpt-6-luna",
                          jev_base_url_text=" https://quick-pivot.com/v1 ", jev_key_text="decisions-secret")
        data = dump.call_args.args[0]
        self.assertEqual(data["jev_provider"], "openai_decisions")
        self.assertEqual(data["jev_model"], "gpt-6-luna")
        self.assertEqual(data["jev_base_url"], "https://quick-pivot.com/v1")
        self.assertNotIn("decisions-secret", str(data))
        set_key.assert_any_call(JEV_ENV, "decisions-secret")

    def test_decisions_address_survives_unrelated_save_and_provider_switch(self):
        config = self._config(jev_provider="openai_decisions", jev_model="gpt-6-luna",
                              jev_base_url="https://quick-pivot.com/v1")
        with patch.object(settings, "_read", side_effect=lambda name, default=None: config.get(name, default)), \
                patch.object(settings, "_read_env", return_value=""), patch("builtins.open", mock_open()), \
                patch.object(settings.json, "dump") as dump:
            settings.save(logging_level_text="DEBUG")
            self.assertEqual(dump.call_args.args[0]["jev_base_url"], config["jev_base_url"])
            settings.save(jev_provider_text="openrouter")
            self.assertEqual(dump.call_args.args[0]["jev_base_url"], config["jev_base_url"])
            settings.save(jev_base_url_text="")
            self.assertEqual(dump.call_args.args[0]["jev_base_url"], "")

    def test_laya_defaults_and_readiness_do_not_require_official_key(self):
        config = self._config(jev_provider="laya", judge_mode="jev_only")
        keys = {LLM_ENV: "draft-key"}
        with patch.object(settings, "_read", side_effect=lambda name, default=None: config.get(name, default)), \
                patch.object(settings, "_read_env", side_effect=lambda name: keys.get(name, "")):
            self.assertEqual(settings.jev_provider(), "laya")
            self.assertEqual(settings.jev_model(), "multilingual")
            self.assertEqual(settings.jev_base_url(), "http://127.0.0.1:8000")
            self.assertTrue(settings.models_ready())
            config.update(laya_model="english", laya_base_url=" http://127.0.0.1:18001 ")
            self.assertEqual(settings.jev_model(), "english")
            self.assertEqual(settings.jev_base_url(), "http://127.0.0.1:18001")
            keys[LLM_ENV] = ""
            self.assertFalse(settings.models_ready())

    def test_laya_save_preserves_remote_configuration_and_separates_secrets(self):
        config = self._config(jev_provider="openai_decisions", jev_model="remote-model",
                              jev_base_url="https://quick-pivot.com/v1")
        with patch.object(settings, "_read", side_effect=lambda name, default=None: config.get(name, default)), \
                patch.object(settings, "_read_env", return_value=""), patch.object(settings, "_set_key") as set_key, \
                patch.object(settings, "_notify_env"), patch("builtins.open", mock_open()), \
                patch.object(settings.json, "dump") as dump:
            settings.save(jev_provider_text="laya", jev_model_text="multilingual",
                          jev_base_url_text=" http://127.0.0.1:18001 ", jev_key_text="local-secret")
        data = dump.call_args.args[0]
        self.assertEqual(data["jev_provider"], "laya")
        self.assertEqual(data["laya_model"], "multilingual")
        self.assertEqual(data["laya_base_url"], "http://127.0.0.1:18001")
        self.assertEqual(data["jev_model"], "remote-model")
        self.assertEqual(data["jev_base_url"], "https://quick-pivot.com/v1")
        self.assertNotIn("local-secret", str(data))
        set_key.assert_any_call("LAYA_API_KEY", "local-secret")
        self.assertNotIn(unittest.mock.call(JEV_ENV, "local-secret"), set_key.call_args_list)

    def test_remote_save_and_unrelated_save_preserve_local_configuration(self):
        config = self._config(laya_model="multilingual", laya_base_url="http://127.0.0.1:18001")
        with patch.object(settings, "_read", side_effect=lambda name, default=None: config.get(name, default)), \
                patch.object(settings, "_read_env", return_value=""), patch("builtins.open", mock_open()), \
                patch.object(settings.json, "dump") as dump:
            settings.save(jev_provider_text="openai_decisions", jev_model_text="remote-model",
                          jev_base_url_text="https://quick-pivot.com/v1")
            self.assertEqual(dump.call_args.args[0]["laya_model"], "multilingual")
            self.assertEqual(dump.call_args.args[0]["laya_base_url"], "http://127.0.0.1:18001")
            settings.save(logging_level_text="DEBUG")
            self.assertEqual(dump.call_args.args[0]["laya_base_url"], "http://127.0.0.1:18001")

    def test_local_key_does_not_fall_back_to_official_and_is_redacted(self):
        with patch.object(settings, "_read_env", side_effect=lambda name: "official-secret" if name == JEV_ENV else ""):
            self.assertEqual(settings.laya_key(), "")
        with patch.dict(jev_client.os.environ, {"LAYA_API_KEY": "local-secret"}):
            self.assertEqual(jev_client.redact_secrets("key=local-secret"), "key=[REDACTED]")

    def test_logging_defaults_and_invalid_values(self):
        with patch.object(settings, "_read", side_effect=lambda name, default=None: default):
            self.assertTrue(settings.logging_enabled())
            self.assertEqual(settings.logging_level(), "INFO")
        config = {"logging_enabled": "false", "logging_level": "TRACE"}
        with patch.object(settings, "_read", side_effect=lambda name, default=None: config.get(name, default)):
            self.assertTrue(settings.logging_enabled())
            self.assertEqual(settings.logging_level(), "INFO")

    def test_logging_settings_are_saved_and_invalid_level_keeps_current_value(self):
        config = {"logging_enabled": True, "logging_level": "DEBUG"}
        with patch.object(settings, "_read", side_effect=lambda name, default=None: config.get(name, default)), \
                patch.object(settings, "_read_env", return_value=""), patch("builtins.open", mock_open()), \
                patch.object(settings.json, "dump") as dump:
            settings.save(logging_enabled_on=False, logging_level_text="warning")
            data = dump.call_args.args[0]
            self.assertFalse(data["logging_enabled"])
            self.assertEqual(data["logging_level"], "WARNING")
            settings.save(logging_level_text="TRACE")
            self.assertEqual(dump.call_args.args[0]["logging_level"], "DEBUG")

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
