import copy
import json
import unittest
from unittest.mock import patch

import httpx
import openai

from core import engine, jev_client, llm_judge, retry
from core.jev_client import JevError
from core.questions import JUDGE_QUESTIONS, build_rank_question, build_state


class DecisionsTests(unittest.TestCase):
    def setUp(self):
        self.requests = []
        self.sleeps = []
        self.state = build_state([("her", "你好", "小明")], "friends", reply_to="小明")
        self.questions = {"literal_question": JUDGE_QUESTIONS["literal_question"],
                          "true_intent": JUDGE_QUESTIONS["true_intent"],
                          "danger_level": JUDGE_QUESTIONS["danger_level"]}
        self.response = {"answers": [
            {"name": "danger_level", "type": "score", "score": 1.1, "confidence": 0.55,
             "probabilities": [{"value": 0, "probability": 0.1},
                               {"value": 1, "probability": 0.7},
                               {"value": 2, "probability": 0.2}]},
            {"name": "literal_question", "type": "predicate", "probability": 0.9},
            {"name": "true_intent", "type": "choice", "choice": "casual_chat", "confidence": 0.8,
             "probabilities": [{"value": "casual_chat", "probability": 0.8},
                               {"value": "close_topic", "probability": 0.2}]},
        ], "usage": {"input_tokens": 123, "output_tokens": 0}}
        for target in (patch.dict(jev_client.os.environ, {"JEV_API_KEY": "decision-test-key"}),
                       patch.object(retry.time, "sleep", side_effect=self.sleeps.append),
                       patch.object(retry.random, "random", return_value=0.0)):
            target.start()
            self.addCleanup(target.stop)

    def _transport(self, handler):
        original = openai.OpenAI

        def factory(**kwargs):
            client = httpx.Client(transport=httpx.MockTransport(handler))
            self.addCleanup(client.close)
            return original(http_client=client, **kwargs)

        return patch("openai.OpenAI", side_effect=factory)

    def _ok(self, request):
        self.requests.append(request)
        return httpx.Response(200, json=self.response)

    def _ask(self, **kwargs):
        return jev_client.ask(self.state, self.questions, provider="openai_decisions", **kwargs)

    def _wire_result(self, questions, choices=None):
        answers = []
        for question in questions:
            name, kind = question["name"], question["type"]
            answer = {"name": name, "type": kind}
            if kind == "predicate":
                answer["probability"] = 0.8
            elif kind == "score":
                answer["score"] = 1.1
            else:
                choice = (choices or {}).get(name, "reply_b" if name == "best_reply"
                                             else question["choices"][0]["value"])
                answer.update(choice=choice, confidence=0.8,
                              probabilities=[{"value": choice, "probability": 0.8}])
            answers.append(answer)
        return {"answers": answers, "usage": {"input_tokens": 10}}

    def test_custom_endpoint_key_model_and_question_conversion(self):
        before = copy.deepcopy(self.questions)
        with self._transport(self._ok) as factory:
            result = self._ask(base_url="https://quick-pivot.com/v1/", model="future-decision-model", timeout=17)
        request = self.requests[0]
        self.assertEqual(str(request.url), "https://quick-pivot.com/v1/decisions")
        self.assertEqual(request.headers["authorization"], "Bearer decision-test-key")
        payload = json.loads(request.content)
        self.assertEqual(payload["model"], "future-decision-model")
        self.assertEqual(json.loads(payload["input"]), self.state)
        self.assertNotIn("state", payload)
        questions = {q["name"]: q for q in payload["questions"]}
        predicate = questions["literal_question"]
        self.assertEqual(predicate["type"], "predicate")
        for text in before["literal_question"]["criteria"].values():
            self.assertIn(text, predicate["instructions"])
        self.assertEqual(questions["true_intent"]["choices"], [
            {"value": key, "description": text}
            for key, text in before["true_intent"]["criteria"].items()])
        self.assertEqual(questions["danger_level"]["levels"], [
            {"label": str(i), "description": text}
            for i, text in enumerate(before["danger_level"]["criteria"])])
        self.assertEqual(self.questions, before)
        self.assertEqual(result["answers"]["literal_question"], {"type": "noul", "noul": 0.9})
        self.assertEqual(result["answers"]["danger_level"]["score"], 1.1)
        self.assertEqual(result["answers"]["danger_level"]["probabilities"],
                         {"0": 0.1, "1": 0.7, "2": 0.2})
        self.assertEqual(result["answers"]["true_intent"]["probabilities"],
                         {"casual_chat": 0.8, "close_topic": 0.2})
        self.assertEqual(result["usage"], self.response["usage"])
        self.assertEqual(factory.call_args.kwargs["max_retries"], 0)
        self.assertEqual(factory.call_args.kwargs["timeout"], 17)

    def test_explicit_key_overrides_environment_without_mutating_it(self):
        before = jev_client.os.environ["JEV_API_KEY"]
        with self._transport(self._ok), patch.object(jev_client, "_api_key", side_effect=AssertionError("must not read")):
            self._ask(api_key="unsaved-test-key")
        self.assertEqual(self.requests[0].headers["authorization"], "Bearer unsaved-test-key")
        self.assertEqual(jev_client.os.environ["JEV_API_KEY"], before)

    def test_default_official_endpoint_and_model(self):
        with self._transport(self._ok):
            self._ask()
        self.assertEqual(str(self.requests[0].url), "https://api.openai.com/v1/decisions")
        self.assertEqual(json.loads(self.requests[0].content)["model"], "gpt-6-luna")

    def test_models_use_same_base_and_key_without_filtering_future_models(self):
        def handler(request):
            self.requests.append(request)
            return httpx.Response(200, json={"data": [
                {"id": "gpt-6-luna"}, {"id": "future-model"}, {"id": "gpt-6-luna"}]})

        with self._transport(handler) as factory:
            models = jev_client.list_models("openai_decisions", "model-list-key",
                                           base_url="https://quick-pivot.com/v1/", timeout=12)
        self.assertEqual(models, ["future-model", "gpt-6-luna"])
        self.assertEqual(str(self.requests[0].url), "https://quick-pivot.com/v1/models")
        self.assertEqual(self.requests[0].headers["authorization"], "Bearer model-list-key")
        self.assertEqual(factory.call_args.kwargs["max_retries"], 0)

    def test_retry_after_and_explicit_retry_limit(self):
        def handler(request):
            self.requests.append(request)
            if len(self.requests) == 1:
                return httpx.Response(429, headers={"Retry-After": "7"}, json={"error": "busy"})
            return httpx.Response(200, json=self.response)

        with self._transport(handler):
            self._ask(max_retries=1)
        self.assertEqual(len(self.requests), 2)
        self.assertAlmostEqual(sum(self.sleeps), 7)
        self.requests.clear()
        with self._transport(lambda request: httpx.Response(503, json={"error": "busy"})) as factory, \
                self.assertRaises(JevError) as caught:
            self._ask(max_retries=1)
        self.assertEqual(factory.call_count, 2)
        self.assertEqual(caught.exception.status, 503)

    def test_auth_failure_does_not_retry_or_expose_key(self):
        with self._transport(lambda request: httpx.Response(
                401, json={"error": "decision-test-key"})) as factory, self.assertRaises(JevError) as caught:
            self._ask()
        self.assertEqual(factory.call_count, 1)
        self.assertEqual(caught.exception.status, 401)
        self.assertNotIn("decision-test-key", str(caught.exception))
        self.assertEqual(self.sleeps, [])

    def test_error_response_cannot_echo_context_candidates_or_key(self):
        detail = "完整聊天内容及候选正文 decision-test-key"
        with self._transport(lambda request: httpx.Response(400, json={"error": detail})), \
                self.assertRaises(JevError) as caught:
            self._ask()
        self.assertEqual(caught.exception.status, 400)
        self.assertNotIn(detail, str(caught.exception))
        self.assertNotIn("完整聊天", str(caught.exception))
        self.assertNotIn("decision-test-key", str(caught.exception))

    def test_rejects_refusal_missing_duplicate_and_invalid_answers(self):
        cases = [None, {}, {"answers": {}}, {"answers": []}]
        for change in (
                {"type": "refusal"}, {"type": "predicate"}, {"score": 10}, {"score": True},
                {"score": float("nan")}, {"confidence": 1.1},
                {"probabilities": [{"value": 1, "probability": 2}]},
                {"probabilities": [{"value": 1, "probability": 0.5}, {"value": 1, "probability": 0.5}]},
                {"probabilities": {"1": 0.7}}):
            response = copy.deepcopy(self.response)
            response["answers"][0].update(change)
            cases.append(response)
        duplicate = copy.deepcopy(self.response)
        duplicate["answers"].append(copy.deepcopy(duplicate["answers"][0]))
        cases.append(duplicate)
        unknown = copy.deepcopy(self.response)
        unknown["answers"][0]["name"] = "unexpected"
        cases.append(unknown)
        for choice in ("unexpected", [], {}):
            invalid_choice = copy.deepcopy(self.response)
            invalid_choice["answers"][2]["choice"] = choice
            cases.append(invalid_choice)
        for response in cases:
            with self.subTest(response=response), self._transport(
                    lambda request: httpx.Response(200, content=json.dumps(response).encode(),
                                                   headers={"Content-Type": "application/json"})), \
                    self.assertRaises(JevError):
                self._ask()

    def test_generic_judgment_still_rejects_fractional_score(self):
        with self._transport(self._ok):
            result = self._ask()
        with self.assertRaises(JevError):
            llm_judge.validate_answers(result["answers"], self.questions)

    def test_engine_judgment_and_ranking_use_decisions_and_preserve_probabilities(self):
        candidates = ["候选甲", "候选乙", "候选丙"]

        def handler(request):
            self.requests.append(request)
            payload = json.loads(request.content)
            return httpx.Response(200, json=self._wire_result(payload["questions"]))

        with self._transport(handler), patch.object(engine, "draft_candidates", return_value=candidates), \
                patch.object(engine, "ask_llm_judge") as generic:
            result = engine.analyze([("her", "你好")], "friends", jev_provider="openai_decisions",
                                    jev_base_url="https://quick-pivot.com/v1", judge_mode="jev_only")
        self.assertEqual(len(self.requests), 2)
        rank = build_rank_question(candidates)["best_reply"]["criteria"]
        self.assertEqual(json.loads(self.requests[1].content)["questions"][0]["choices"],
                         [{"value": key, "description": text} for key, text in rank.items()])
        self.assertTrue(all(str(r.url) == "https://quick-pivot.com/v1/decisions" for r in self.requests))
        self.assertEqual(result["best_index"], 1)
        self.assertEqual(result["scores"], [0, 0.8, 0])
        self.assertEqual(result["answers"]["danger_level"]["score"], 1.1)
        self.assertTrue(result["ranking_valid"])
        self.assertEqual(result["usage"]["input_tokens"], 20)
        self.assertEqual(result["judge_backend"], "jev")
        generic.assert_not_called()

    def test_engine_decisions_close_topic_does_not_draft_or_rank(self):
        def handler(request):
            self.requests.append(request)
            questions = json.loads(request.content)["questions"]
            return httpx.Response(200, json=self._wire_result(questions, {
                "true_intent": "close_topic", "she_needs": "nothing", "reply_decision": "no_reply"}))

        with self._transport(handler), patch.object(engine, "draft_candidates") as draft:
            result = engine.analyze([("her", "谢谢，再见")], "friends",
                                    jev_provider="openai_decisions", judge_mode="jev_only")
        self.assertFalse(result["reply_required"])
        self.assertEqual(result["candidates"], [])
        self.assertEqual(len(self.requests), 1)
        draft.assert_not_called()

    def test_engine_decisions_failure_uses_existing_generic_fallback(self):
        questions = jev_client._decision_questions(JUDGE_QUESTIONS)
        judgment = jev_client._decision_answers(self._wire_result(questions), JUDGE_QUESTIONS)
        judgment["answers"]["danger_level"]["score"] = 1
        ranking = {"answers": {"best_reply": {"type": "choice", "choice": "reply_b"}}}
        with self._transport(lambda request: httpx.Response(503, json={"error": "busy"})) as factory, \
                patch.object(engine, "ask_llm_judge", side_effect=[judgment, ranking]) as generic, \
                patch.object(engine, "draft_candidates", return_value=["候选甲", "候选乙"]):
            result = engine.analyze([("her", "你好")], "friends", jev_provider="openai_decisions",
                                    judge_mode="jev_fallback", llm_judge_api_key="generic-key")
        self.assertEqual(factory.call_count, 2)
        self.assertEqual(generic.call_count, 2)
        self.assertEqual(result["judge_backend"], "llm")
        self.assertEqual(result["ranking_backend"], "llm")
        self.assertTrue(result["fallback_used"])
        self.assertTrue(result["ranking_valid"])
        self.assertEqual(result["best_index"], 1)

    def test_decisions_default_retry_limit_is_five_and_network_errors_retry(self):
        with self._transport(lambda request: httpx.Response(503, json={"error": "busy"})) as factory, \
                self.assertRaises(JevError) as caught:
            self._ask()
        self.assertEqual(factory.call_count, 6)
        self.assertIn("已自动重试 5 次", str(caught.exception))

        def handler(request):
            self.requests.append(request)
            if len(self.requests) == 1:
                raise httpx.ConnectError("connection reset", request=request)
            return httpx.Response(200, json=self.response)

        with self._transport(handler):
            self._ask()
        self.assertEqual(len(self.requests), 2)

    def test_model_list_uses_shared_retry_and_redacts_unsaved_key_on_error(self):
        def handler(request):
            self.requests.append(request)
            if len(self.requests) == 1:
                return httpx.Response(429, headers={"Retry-After": "9"}, json={"error": "busy"})
            return httpx.Response(200, json={"data": [{"id": "gpt-6-luna"}]})

        with self._transport(handler):
            self.assertEqual(jev_client.list_models("openai_decisions", "unsaved-model-key"), ["gpt-6-luna"])
        self.assertAlmostEqual(sum(self.sleeps), 9)
        self.assertEqual(len(self.requests), 2)
        with patch("openai.OpenAI", side_effect=ValueError("bad unsaved-model-key")), \
                self.assertRaises(JevError) as caught:
            jev_client.list_models("openai_decisions", "unsaved-model-key")
        self.assertNotIn("unsaved-model-key", str(caught.exception))


class LayaTests(unittest.TestCase):
    def setUp(self):
        self.state = build_state([("her", "你好")], "friends")
        self.questions = {name: JUDGE_QUESTIONS[name] for name in
                          ("literal_question", "true_intent", "danger_level")}
        self.requests = []
        self.sleeps = []
        for target in (patch.dict(jev_client.os.environ, {"JEV_API_KEY": "official-secret",
                                                       "LAYA_API_KEY": ""}),
                       patch.object(retry.time, "sleep", side_effect=self.sleeps.append),
                       patch.object(retry.random, "random", return_value=0.0)):
            target.start()
            self.addCleanup(target.stop)

    def _response(self, questions, choices=None):
        answers = {}
        for name, question in questions.items():
            kind = question["type"]
            answer = {"type": kind}
            if kind == "noul":
                answer["noul"] = 0.8
            elif kind == "score":
                answer.update(score=1.1, confidence=0.55, x_jev_confidence=0.65,
                              probabilities={"0": 0.1, "1": 0.7, "2": 0.2})
            else:
                choice = (choices or {}).get(name, "reply_b" if name == "best_reply"
                                             else next(iter(question["criteria"])))
                answer.update(choice=choice, confidence=0.8, probabilities={choice: 0.8})
            answers[name] = answer
        return {"answers": answers, "usage": {"input_tokens": 10, "output_tokens": 0}}

    def _transport(self, handler=None):
        import io

        def send(request, timeout):
            self.requests.append((request, timeout))
            result = handler(request) if handler else self._response(self.questions)
            return io.BytesIO(json.dumps(result, ensure_ascii=False).encode("utf-8"))

        opener = unittest.mock.Mock()
        opener.open.side_effect = send
        return patch.object(jev_client.urllib.request, "build_opener", return_value=opener)

    def _ask(self, **kwargs):
        return jev_client.ask(self.state, self.questions, provider="laya", **kwargs)

    def test_local_endpoint_uses_jev_payload_optional_key_and_fractional_scores(self):
        before = copy.deepcopy(self.questions)
        with self._transport() as factory:
            result = self._ask(base_url="http://127.0.0.1:18001/", timeout=17)
        request, timeout = self.requests[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:18001/v1/systemone")
        self.assertEqual(request.get_method(), "POST")
        self.assertIsNone(request.get_header("Authorization"))
        self.assertEqual(json.loads(request.data), {"model": "multilingual", "state": self.state,
                                                  "questions": self.questions})
        self.assertEqual(self.questions, before)
        self.assertEqual(timeout, 60)
        self.assertEqual(result["answers"]["danger_level"]["score"], 1.1)
        self.assertEqual(result["answers"]["danger_level"]["confidence"], 0.65)
        self.assertEqual(result["usage"]["input_tokens"], 10)
        self.assertTrue(any(isinstance(arg, jev_client.urllib.request.ProxyHandler) and arg.proxies == {}
                            for arg in factory.call_args.args))

    def test_local_timeout_defaults_and_bounds_apply_to_direct_requests(self):
        for value, expected in ((None, 120), (60, 60), (90, 90), (120, 120), (15, 60), (300, 120)):
            with self.subTest(value=value), self._transport():
                self._ask(**({} if value is None else {"timeout": value}))
            self.assertEqual(self.requests[-1][1], expected)

    def test_local_engine_uses_independent_timeout_for_judgment_and_ranking(self):
        def handler(request):
            return self._response(json.loads(request.data)["questions"])

        for seconds in (60, 90, 120):
            self.requests.clear()
            with self.subTest(seconds=seconds), self._transport(handler), \
                    patch.object(engine, "draft_candidates", return_value=["候选甲", "候选乙"]) as draft:
                result = engine.analyze([("her", "你好")], "friends", jev_provider="laya", judge_mode="jev_only",
                                        timeout=25, laya_timeout=seconds)
            self.assertTrue(result["ranking_valid"])
            self.assertEqual([seconds_sent for _, seconds_sent in self.requests], [seconds, seconds])
            self.assertEqual(draft.call_args.kwargs["timeout"], 25)

    def test_remote_judgment_timeout_is_not_changed_by_local_setting(self):
        def answer(state, questions, **kwargs):
            return self._response(questions)

        with patch.object(engine, "ask_jev", side_effect=answer) as ask, \
                patch.object(engine, "draft_candidates", return_value=["候选甲", "候选乙"]):
            engine.analyze([("her", "你好")], "friends", jev_provider="openai_decisions", judge_mode="jev_only",
                           timeout=25, laya_timeout=120)
        self.assertEqual([call.kwargs["timeout"] for call in ask.call_args_list], [25, 25])

    def test_default_endpoint_and_separate_saved_or_explicit_bearer_key(self):
        with self._transport(), patch.object(jev_client, "_api_key", side_effect=AssertionError("official key")):
            self._ask()
            with patch.dict(jev_client.os.environ, {"LAYA_API_KEY": "local-secret"}):
                self._ask()
                self._ask(api_key="typed-local-secret", model="english")
        self.assertEqual(self.requests[0][0].full_url, "http://127.0.0.1:8000/v1/systemone")
        self.assertIsNone(self.requests[0][0].get_header("Authorization"))
        self.assertEqual(self.requests[1][0].get_header("Authorization"), "Bearer local-secret")
        self.assertEqual(self.requests[2][0].get_header("Authorization"), "Bearer typed-local-secret")
        self.assertEqual(json.loads(self.requests[2][0].data)["model"], "english")
        self.assertEqual(jev_client.os.environ["JEV_API_KEY"], "official-secret")

    def test_models_are_local_presets_without_network_or_official_authentication(self):
        with patch.object(jev_client.urllib.request, "build_opener") as network, \
                patch.object(jev_client, "_check_openrouter_key") as auth:
            self.assertEqual(jev_client.list_models("laya", ""),
                             ["multilingual", "english", "typed-decisions"])
        network.assert_not_called()
        auth.assert_not_called()

    def test_endpoint_normalization_and_invalid_addresses(self):
        for suffix in ("", "/", "/v1", "/v1/", "/v1/systemone", "/v1/systemone/"):
            with self.subTest(suffix=suffix), self._transport():
                self._ask(base_url="http://127.0.0.1:18001/laya" + suffix)
                self.assertEqual(self.requests[-1][0].full_url, "http://127.0.0.1:18001/laya/v1/systemone")
        for address in ("127.0.0.1:8000", "ftp://localhost", "http://localhost:0", "http://localhost:65536",
                        "http://localhost:bad", "http://user:secret@localhost", "http://localhost?secret=key"):
            with self.subTest(address=address), self._transport(), self.assertRaises(JevError) as caught:
                self._ask(base_url=address)
            self.assertEqual(caught.exception.status, 422)
            self.assertNotIn(address, str(caught.exception))

    def test_authentication_and_redirect_errors_do_not_retry_or_echo_context(self):
        import io

        for status in (401, 403, 307, 422):
            error = jev_client.urllib.error.HTTPError("http://localhost", status, "private context local-secret",
                                                    {}, io.BytesIO(b"private context local-secret"))
            with self.subTest(status=status), self._transport(lambda request: (_ for _ in ()).throw(error)), \
                    self.assertRaises(JevError) as caught:
                self._ask(api_key="local-secret")
            self.assertEqual(caught.exception.status, status)
            self.assertNotIn("private context", str(caught.exception))
            self.assertNotIn("local-secret", str(caught.exception))
        self.assertEqual(len(self.requests), 4)
        self.assertEqual(self.sleeps, [])
        with self._transport() as factory:
            self._ask()
        redirect = next(arg for arg in factory.call_args.args if isinstance(arg, jev_client._LayaNoRedirect))
        self.assertIsNone(redirect.redirect_request(None, None, 307, "", {}, "http://other-host"))

    def test_http_retry_after_and_connection_failure_use_shared_retry(self):
        attempts = 0

        def handler(request):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise jev_client.urllib.error.HTTPError(request.full_url, 503, "busy", {"Retry-After": "7"}, None)
            return self._response(self.questions)

        with self._transport(handler):
            self._ask(max_retries=1)
        self.assertEqual(attempts, 2)
        self.assertAlmostEqual(sum(self.sleeps), 7)
        self.requests.clear()
        error = jev_client.urllib.error.URLError("private connection error")
        with self._transport(lambda request: (_ for _ in ()).throw(error)), self.assertRaises(JevError) as caught:
            self._ask(max_retries=1)
        self.assertEqual(len(self.requests), 2)
        self.assertTrue(caught.exception.retryable)
        self.assertNotIn("private connection error", str(caught.exception))

    def test_invalid_json_and_answers_fail_without_retry(self):
        invalid = [None, [], {}, {"answers": {}}, self._response(self.questions)]
        invalid[-1]["answers"]["danger_level"]["score"] = float("nan")
        for response in invalid:
            with self.subTest(response=response), self._transport(lambda request: response), self.assertRaises(JevError):
                self._ask()
        with self._transport() as factory:
            import io

            factory.return_value.open.side_effect = None
            factory.return_value.open.return_value = io.BytesIO(b"not JSON private context")
            with self.assertRaises(JevError) as caught:
                self._ask()
        self.assertNotIn("private context", str(caught.exception))
        self.assertEqual(self.sleeps, [])

    def test_local_failure_retains_existing_fallback_and_configuration_errors_stop_it(self):
        judgment = self._response(JUDGE_QUESTIONS)
        judgment["answers"]["danger_level"]["score"] = 1
        ranking = {"answers": {"best_reply": {"type": "choice", "choice": "reply_b"}}}
        error = jev_client.urllib.error.URLError("offline")
        with self._transport(lambda request: (_ for _ in ()).throw(error)), \
                patch.object(engine, "ask_llm_judge", side_effect=[judgment, ranking]) as generic, \
                patch.object(engine, "draft_candidates", return_value=["候选甲", "候选乙"]):
            result = engine.analyze([("her", "你好")], "friends", jev_provider="laya",
                                    judge_mode="jev_fallback", llm_judge_api_key="generic-key")
        self.assertTrue(result["fallback_used"])
        self.assertEqual(result["judge_backend"], "llm")
        self.assertEqual(result["ranking_backend"], "llm")
        self.assertEqual(result["best_index"], 1)
        generic.assert_any_call(unittest.mock.ANY, unittest.mock.ANY, timeout=30, provider="deepseek",
                                model=None, base_url=None, api_key="generic-key")
        with patch.object(engine, "ask_llm_judge") as generic, patch.object(engine, "draft_candidates") as draft, \
                self.assertRaises(JevError):
            engine.analyze([("her", "你好")], "friends", jev_provider="laya", jev_base_url="invalid")
        generic.assert_not_called()
        draft.assert_not_called()

    def test_local_only_failure_does_not_call_remote_judge(self):
        error = TimeoutError("private timeout")
        with self._transport(lambda request: (_ for _ in ()).throw(error)), \
                patch.object(engine, "ask_llm_judge") as generic, patch.object(engine, "select_best") as selector, \
                patch.object(engine, "draft_candidates", return_value=["候选甲", "候选乙"]):
            result = engine.analyze([("her", "你好")], "friends", jev_provider="laya", judge_mode="jev_only")
        self.assertEqual(len(self.requests), 6)
        self.assertFalse(result["ranking_valid"])
        self.assertFalse(result["fallback_used"])
        generic.assert_not_called()
        selector.assert_not_called()

    def test_local_close_topic_does_not_draft_or_rank(self):
        def handler(request):
            return self._response(json.loads(request.data)["questions"], {
                "true_intent": "close_topic", "she_needs": "nothing", "reply_decision": "no_reply"})

        with self._transport(handler), patch.object(engine, "draft_candidates") as draft:
            result = engine.analyze([("her", "谢谢，再见")], "friends", jev_provider="laya", judge_mode="jev_only")
        self.assertFalse(result["reply_required"])
        self.assertEqual(len(self.requests), 1)
        draft.assert_not_called()

    def test_engine_local_judgment_and_ranking_preserve_probabilities(self):
        def handler(request):
            return self._response(json.loads(request.data)["questions"])

        with self._transport(handler), patch.object(engine, "draft_candidates", return_value=["候选甲", "候选乙"]), \
                patch.object(engine, "ask_llm_judge") as generic:
            result = engine.analyze([("her", "你好")], "friends", jev_provider="laya",
                                    jev_base_url="http://127.0.0.1:18001", judge_mode="jev_only")
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(result["answers"]["danger_level"]["score"], 1.1)
        self.assertEqual(result["best_index"], 1)
        self.assertEqual(result["scores"], [0, 0.8, 0])
        self.assertTrue(result["ranking_valid"])
        self.assertEqual(result["usage"]["input_tokens"], 20)
        generic.assert_not_called()


if __name__ == "__main__":
    unittest.main()
