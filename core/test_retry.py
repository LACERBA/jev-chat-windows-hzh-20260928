import io
import json
import threading
import unittest
import urllib.error
import urllib.request
from types import SimpleNamespace
from unittest.mock import Mock, patch

from core import jev_client, llm, llm_judge, retry
from core.jev_client import JevError
from core.questions import JUDGE_QUESTIONS


def transient(status=429, **kwargs):
    return JevError(f"限流 {status}", status, **kwargs)


class Flaky:
    """前 failures 次抛 error，之后返回 result。"""

    def __init__(self, failures, error, result="ok"):
        self.failures, self.error, self.result, self.calls = failures, error, result, 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error
        return self.result


class RetryCoreTests(unittest.TestCase):
    def setUp(self):
        self.sleeps = []
        for target in (patch.object(retry.time, "sleep", side_effect=self.sleeps.append),
                       patch.object(retry.random, "random", return_value=0.0)):
            target.start()
            self.addCleanup(target.stop)

    def test_succeeds_after_temporary_failures_with_increasing_delays(self):
        operation = Flaky(2, transient())
        self.assertEqual(retry.call_with_retry(operation), "ok")
        self.assertEqual(operation.calls, 3)
        self.assertEqual(round(sum(self.sleeps), 3), 9.0)  # 3 + 6

    def test_stops_after_five_retries_with_documented_intervals(self):
        operation = Flaky(99, transient())
        with self.assertRaises(JevError) as caught:
            retry.call_with_retry(operation, what="起草")
        self.assertEqual(operation.calls, 6)
        self.assertEqual(round(sum(self.sleeps), 3), 93.0)  # 3 + 6 + 12 + 24 + 48
        self.assertEqual(caught.exception.status, 429)
        self.assertIn("已自动重试 5 次", str(caught.exception))

    def test_retry_delay_schedule_and_jitter_bounds(self):
        self.assertEqual([retry.retry_delay(i) for i in range(5)], [3, 6, 12, 24, 48])
        self.assertEqual(retry.retry_delay(7), 48)
        with patch.object(retry.random, "random", return_value=1.0):
            self.assertAlmostEqual(retry.retry_delay(0), 3.6)

    def test_permanent_errors_are_not_retried(self):
        for error in (JevError("密钥被拒", 401), JevError("没有权限", 403), JevError("模型不对", 404),
                      JevError("请求被拒", 422), JevError("参数错", 400), JevError("未配置密钥"),
                      ValueError("程序错误")):
            operation = Flaky(99, error)
            with self.subTest(error=error), self.assertRaises(Exception):
                retry.call_with_retry(operation)
            self.assertEqual(operation.calls, 1)
        self.assertEqual(self.sleeps, [])

    def test_all_temporary_statuses_and_network_errors_are_retried(self):
        for error in (transient(408), transient(409), transient(425), transient(429), transient(500),
                      transient(502), transient(503), transient(504), transient(529),
                      JevError("超时", retryable=True)):
            operation = Flaky(1, error)
            with self.subTest(error=error):
                self.assertEqual(retry.call_with_retry(operation), "ok")
            self.assertEqual(operation.calls, 2)

    def test_retry_after_is_respected_and_capped(self):
        retry.call_with_retry(Flaky(1, transient(retry_after=20)))
        self.assertEqual(round(sum(self.sleeps), 3), 20.0)
        self.sleeps.clear()
        retry.call_with_retry(Flaky(1, transient(retry_after=3600)))
        self.assertEqual(round(sum(self.sleeps), 3), 60.0)
        self.sleeps.clear()
        retry.call_with_retry(Flaky(1, transient(retry_after=1)))
        self.assertEqual(round(sum(self.sleeps), 3), 3.0)  # 不会比自己的退避更短

    def test_per_call_limit_and_zero_disable_retry(self):
        operation = Flaky(99, transient())
        with self.assertRaises(JevError):
            retry.call_with_retry(operation, max_retries=1)
        self.assertEqual(operation.calls, 2)
        operation = Flaky(99, transient())
        with self.assertRaises(JevError):
            retry.call_with_retry(operation, max_retries=0)
        self.assertEqual(operation.calls, 1)

    def test_cancel_guard_stops_waiting_when_conversation_changed(self):
        operation = Flaky(99, transient())
        with retry.retry_cancel_guard(lambda: True), self.assertRaises(JevError) as caught:
            retry.call_with_retry(operation)
        self.assertEqual(operation.calls, 1)
        self.assertEqual(self.sleeps, [])
        self.assertIn("会话有新消息", str(caught.exception))

    def test_cancel_during_wait_stops_before_next_attempt(self):
        state = {"cancel": False}

        def sleeping(seconds):
            self.sleeps.append(seconds)
            state["cancel"] = len(self.sleeps) >= 3

        operation = Flaky(99, transient())
        with patch.object(retry.time, "sleep", side_effect=sleeping), \
                retry.retry_cancel_guard(lambda: state["cancel"]), self.assertRaises(JevError):
            retry.call_with_retry(operation)
        self.assertEqual(operation.calls, 1)
        self.assertLess(sum(self.sleeps), 3)

    def test_guard_is_scoped_to_the_thread(self):
        seen = {}

        def other():
            seen["calls"] = Flaky(1, transient())
            seen["result"] = retry.call_with_retry(seen["calls"])

        with retry.retry_cancel_guard(lambda: True):
            thread = threading.Thread(target=other)
            thread.start()
            thread.join()
        self.assertEqual(seen["result"], "ok")

    def test_retry_events_are_logged_without_secrets(self):
        with patch("core.diagnostics.log_event") as log:
            retry.call_with_retry(Flaky(1, transient(retry_after=2)), what="起草")
        event, = [call for call in log.call_args_list if call.args[0] == "model.retry"]
        self.assertEqual(event.kwargs["retry"], 1)
        self.assertEqual(event.kwargs["http_status"], 429)
        self.assertEqual(event.kwargs["what"], "起草")


class HttpError(Exception):
    def __init__(self, status, headers=None, attr="status_code"):
        super().__init__(f"error {status} sk-secret")
        setattr(self, attr, status)
        self.response = SimpleNamespace(headers=headers or {})


class SdkIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.sleeps = []
        for target in (patch.object(retry.time, "sleep", side_effect=self.sleeps.append),
                       patch.object(retry.random, "random", return_value=0.0)):
            target.start()
            self.addCleanup(target.stop)

    def test_openai_chat_retries_once_and_sdk_internal_retry_is_off(self):
        ok = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="好"))])
        client = Mock()
        client.chat.completions.create.side_effect = [HttpError(429, {"retry-after": "7"}), ok]
        with patch("openai.OpenAI", return_value=client) as factory:
            self.assertEqual(llm.chat("openai", None, "key", "m", "S", ["U"]), "好")
        self.assertEqual(client.chat.completions.create.call_count, 2)
        self.assertEqual(factory.call_args.kwargs["max_retries"], 0)
        self.assertEqual(round(sum(self.sleeps), 3), 7.0)

    def test_openai_gives_up_after_five_retries_with_clean_message(self):
        client = Mock()
        client.chat.completions.create.side_effect = HttpError(429)
        with patch("openai.OpenAI", return_value=client), self.assertRaises(JevError) as caught:
            llm.chat("openai", None, "key", "m", "S", ["U"], what="起草")
        self.assertEqual(client.chat.completions.create.call_count, 6)
        self.assertEqual(caught.exception.status, 429)
        self.assertIn("起草 HTTP 429: 被限流", str(caught.exception))
        self.assertIn("已自动重试 5 次", str(caught.exception))
        self.assertNotIn("sk-secret", str(caught.exception))

    def test_openai_auth_error_is_not_retried(self):
        client = Mock()
        client.chat.completions.create.side_effect = HttpError(401)
        with patch("openai.OpenAI", return_value=client), self.assertRaises(JevError):
            llm.chat("openai", None, "key", "m", "S", ["U"])
        self.assertEqual(client.chat.completions.create.call_count, 1)
        self.assertEqual(self.sleeps, [])

    def test_network_exceptions_without_status_are_retried(self):
        class APIConnectionError(Exception):
            pass

        ok = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="好"))])
        client = Mock()
        client.chat.completions.create.side_effect = [APIConnectionError("Connection error."),
                                                      TimeoutError("timed out"), ok]
        with patch("openai.OpenAI", return_value=client):
            self.assertEqual(llm.chat("openai", None, "key", "m", "S", ["U"]), "好")
        self.assertEqual(client.chat.completions.create.call_count, 3)

    def test_anthropic_and_gemini_use_same_retry_layer(self):
        anthropic_client = Mock()
        anthropic_client.messages.create.side_effect = [
            HttpError(529), SimpleNamespace(content=[SimpleNamespace(type="text", text="嗯")])]
        with patch("anthropic.Anthropic", return_value=anthropic_client) as factory:
            self.assertEqual(llm.chat("anthropic", None, "key", "m", "S", ["U"]), "嗯")
        self.assertEqual(factory.call_args.kwargs["max_retries"], 0)
        gemini_client = Mock()
        gemini_client.models.generate_content.side_effect = [
            HttpError(503, attr="code"), SimpleNamespace(text="好")]
        with patch("google.genai.Client", return_value=gemini_client) as factory:
            self.assertEqual(llm.chat("gemini", None, "key", "m", "S", ["U"]), "好")
        self.assertEqual(factory.call_args.kwargs["http_options"].retry_options.attempts, 1)
        self.assertEqual(gemini_client.models.generate_content.call_count, 2)

    def test_general_judge_and_draft_and_vision_inherit_retry(self):
        answers = {"answers": {}}
        with patch.object(llm_judge, "chat", wraps=llm.chat), \
                patch("openai.OpenAI") as factory:
            client = factory.return_value
            client.chat.completions.create.side_effect = [
                HttpError(429),
                SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(answers)))])]
            with self.assertRaises(JevError) as caught:  # 重试后拿到空答案，由校验报错而不是 429
                llm_judge.ask({"chat": {}}, JUDGE_QUESTIONS, provider="deepseek",
                              model="deepseek-chat", api_key="key")
        self.assertIsNone(caught.exception.status)
        self.assertEqual(client.chat.completions.create.call_count, 2)

    def test_openrouter_jev_retries_on_429_with_retry_after_then_succeeds(self):
        body = {"answers": {}, "usage": {}}
        calls = []

        def urlopen(request, timeout=None):
            calls.append(request)
            if len(calls) == 1:
                raise urllib.error.HTTPError(request.full_url, 429, "Too Many", {"Retry-After": "9"},
                                             io.BytesIO(b'{"error":"slow down"}'))
            return io.BytesIO(json.dumps(body).encode())

        with patch.dict(jev_client.os.environ, {"JEV_API_KEY": "jev-key"}), \
                patch.object(urllib.request, "urlopen", urlopen):
            self.assertEqual(jev_client.ask({"chat": {}}, {}), body)
        self.assertEqual(len(calls), 2)
        self.assertEqual(round(sum(self.sleeps), 3), 9.0)

    def test_openrouter_jev_exhausts_then_reports_rate_limit(self):
        def urlopen(request, timeout=None):
            raise urllib.error.HTTPError(request.full_url, 429, "Too Many", {},
                                         io.BytesIO(b'{"error":"slow down"}'))

        with patch.dict(jev_client.os.environ, {"JEV_API_KEY": "jev-key"}), \
                patch.object(urllib.request, "urlopen", Mock(side_effect=urlopen)) as opened, \
                self.assertRaises(JevError) as caught:
            jev_client.ask({"chat": {}}, {})
        self.assertEqual(opened.call_count, 6)
        self.assertEqual(caught.exception.status, 429)
        self.assertIn("已自动重试 5 次", str(caught.exception))

    def test_openrouter_network_errors_are_retried_but_bad_key_is_not(self):
        body = {"answers": {}, "usage": {}}
        responses = [urllib.error.URLError("连接被重置"), TimeoutError("timed out"),
                     io.BytesIO(json.dumps(body).encode())]

        def urlopen(request, timeout=None):
            item = responses.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item

        with patch.dict(jev_client.os.environ, {"JEV_API_KEY": "jev-key"}), \
                patch.object(urllib.request, "urlopen", urlopen):
            self.assertEqual(jev_client.ask({"chat": {}}, {}), body)
        unauthorized = Mock(side_effect=urllib.error.HTTPError("u", 401, "Unauthorized", {}, io.BytesIO(b"{}")))
        with patch.dict(jev_client.os.environ, {"JEV_API_KEY": "jev-key"}), \
                patch.object(urllib.request, "urlopen", unauthorized), self.assertRaises(JevError):
            jev_client.ask({"chat": {}}, {})
        self.assertEqual(unauthorized.call_count, 1)

    def test_typesafe_jev_retries_outside_sdk_and_sdk_retry_is_off(self):
        seen = {"inits": [], "calls": 0}

        class Client:
            def __init__(self, **kwargs):
                seen["inits"].append(kwargs)

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def system_one(self, state, questions, **kwargs):
                seen["calls"] += 1
                if seen["calls"] == 1:
                    error = HttpError(503, {"retry-after-ms": "1500"}, attr="status")
                    raise error
                return SimpleNamespace(answers={}, usage=SimpleNamespace(input_tokens=1, output_tokens=2))

        import typesafe_sdk

        with patch.dict(jev_client.os.environ, {"JEV_API_KEY": "jev-key"}), \
                patch.object(typesafe_sdk, "TypeSafeClient", Client):
            result = jev_client.ask({"chat": {}}, {}, provider="typesafe", model="jev-latest")
        self.assertEqual(result["usage"], {"input_tokens": 1, "output_tokens": 2})
        self.assertEqual(seen["calls"], 2)
        self.assertTrue(all(init["retry"].max_retries == 0 for init in seen["inits"]))
        self.assertEqual(round(sum(self.sleeps), 3), 3.0)  # 1.5 秒的 Retry-After 短于自己的退避

    def test_explicit_jev_retry_limit_is_honored_for_fallback_mode(self):
        def urlopen(request, timeout=None):
            raise urllib.error.HTTPError(request.full_url, 503, "Unavailable", {}, io.BytesIO(b"{}"))

        with patch.dict(jev_client.os.environ, {"JEV_API_KEY": "jev-key"}), \
                patch.object(urllib.request, "urlopen", Mock(side_effect=urlopen)) as opened, \
                self.assertRaises(JevError):
            jev_client.ask({"chat": {}}, {}, max_retries=1)
        self.assertEqual(opened.call_count, 2)


if __name__ == "__main__":
    unittest.main()
