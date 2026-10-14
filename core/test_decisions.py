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


if __name__ == "__main__":
    unittest.main()
