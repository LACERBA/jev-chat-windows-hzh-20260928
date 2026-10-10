import json
import unittest
from unittest.mock import patch

from core import engine, llm_judge
from core.engine import analyze
from core.jev_client import JevError
from core.questions import JUDGE_QUESTIONS, build_rank_question


JUDGMENT = {
    "literal_question": {"type": "noul", "noul": 0.8},
    "true_intent": {"type": "choice", "choice": "casual_chat"},
    "danger_level": {"type": "score", "score": 1},
    "should_reply_now": {"type": "noul", "noul": 0.7},
    "best_action": {"type": "choice", "choice": "acknowledge"},
    "she_needs": {"type": "choice", "choice": "nothing"},
    "tension_resolved": {"type": "noul", "noul": 0.9},
}
CANDIDATES = ["候选甲", "候选乙", "候选丙"]


class LlmJudgeTests(unittest.TestCase):
    def test_validates_all_question_types(self):
        answers = {**JUDGMENT, "best_reply": {"type": "choice", "choice": "reply_b"}}
        questions = {**JUDGE_QUESTIONS, **build_rank_question(CANDIDATES)}
        self.assertEqual(llm_judge.validate_answers(answers, questions), answers)

    def test_rejects_missing_invalid_or_out_of_range_answers(self):
        cases = [
            ({}, JUDGE_QUESTIONS),
            ({**JUDGMENT, "danger_level": {"type": "score", "score": 10}}, JUDGE_QUESTIONS),
            ({"best_reply": {"type": "choice", "choice": "reply_z"}},
             build_rank_question(CANDIDATES)),
            ({**JUDGMENT, "literal_question": {"type": "noul", "noul": 1.5}}, JUDGE_QUESTIONS),
        ]
        for answers, questions in cases:
            with self.subTest(answers=answers), self.assertRaises(JevError):
                llm_judge.validate_answers(answers, questions)

    def test_final_selector_accepts_plain_or_json_choice(self):
        for content, expected in (("reply_b", "reply_b"),
                                  ('{"choice":"reply_c"}', "reply_c"),
                                  ("我选择 reply_a", "reply_a")):
            with self.subTest(content=content), patch.object(llm_judge, "chat", return_value=content):
                result = llm_judge.select_best({"chat": {}}, CANDIDATES, provider="deepseek",
                                               model="deepseek-chat", api_key="judge-key")
            self.assertEqual(result["answers"]["best_reply"]["choice"], expected)

    def test_final_selector_rejects_ambiguous_choice(self):
        with patch.object(llm_judge, "chat", return_value="reply_a 或 reply_b"):
            with self.assertRaises(JevError):
                llm_judge.select_best({"chat": {}}, CANDIDATES, provider="deepseek",
                                      model="deepseek-chat", api_key="judge-key")

    def test_parses_fenced_json_and_uses_judgment_operation(self):
        content = "```json\n" + json.dumps({"answers": JUDGMENT}, ensure_ascii=False) + "\n```"
        with patch.object(llm_judge, "chat", return_value=content) as chat:
            result = llm_judge.ask({"chat": {}}, JUDGE_QUESTIONS, provider="deepseek",
                                   model="deepseek-chat", api_key="judge-key")
        self.assertEqual(result["answers"], JUDGMENT)
        self.assertEqual(chat.call_args.kwargs["what"], "通用判断")
        self.assertEqual(chat.call_args.kwargs["temperature"], 0.1)
        self.assertNotIn("judge-key", chat.call_args.args[-1])


class EngineFallbackTests(unittest.TestCase):
    def _analyze(self, **kwargs):
        return analyze([("her", "你好")], "friends", llm_judge_provider="deepseek",
                       llm_judge_model="deepseek-chat", llm_judge_api_key="judge-key", **kwargs)

    def test_jev_initial_failure_switches_whole_round_to_llm(self):
        llm_results = [
            {"answers": JUDGMENT, "usage": {}},
            {"answers": {"best_reply": {"type": "choice", "choice": "reply_b"}}, "usage": {}},
        ]
        with patch.object(engine, "ask_jev", side_effect=JevError("overloaded", 529)) as jev, \
                patch.object(engine, "ask_llm_judge", side_effect=llm_results) as generic, \
                patch.object(engine, "draft_candidates", return_value=CANDIDATES):
            result = self._analyze(judge_mode="jev_fallback")
        self.assertEqual(jev.call_count, 1)
        self.assertEqual(jev.call_args.kwargs["max_retries"], 1)
        self.assertEqual(generic.call_count, 2)
        self.assertEqual(result["best_index"], 1)
        self.assertTrue(result["ranking_valid"])
        self.assertTrue(result["fallback_used"])
        self.assertEqual(result["ranking_backend"], "llm")
        self.assertEqual(result["scores"], [0.0, 0.0, 0.0])

    def test_jev_rank_failure_falls_back_without_redrafting(self):
        jev_results = [
            {"answers": JUDGMENT, "usage": {}},
            JevError("overloaded", 529),
        ]
        with patch.object(engine, "ask_jev", side_effect=jev_results) as jev, \
                patch.object(engine, "ask_llm_judge", return_value={
                    "answers": {"best_reply": {"type": "choice", "choice": "reply_c"}}, "usage": {}}) as generic, \
                patch.object(engine, "draft_candidates", return_value=CANDIDATES) as draft:
            result = self._analyze(judge_mode="jev_fallback")
        self.assertEqual(jev.call_count, 2)
        generic.assert_called_once()
        draft.assert_called_once()
        self.assertEqual(result["best_index"], 2)
        self.assertTrue(result["ranking_valid"])
        self.assertTrue(result["fallback_used"])

    def test_auth_error_is_not_hidden_by_fallback(self):
        with patch.object(engine, "ask_jev", side_effect=JevError("bad key", 401)), \
                patch.object(engine, "ask_llm_judge") as generic, \
                patch.object(engine, "draft_candidates") as draft:
            with self.assertRaises(JevError):
                self._analyze(judge_mode="jev_fallback")
        generic.assert_not_called()
        draft.assert_not_called()

    def test_llm_only_never_calls_jev(self):
        llm_results = [
            {"answers": JUDGMENT, "usage": {}},
            {"answers": {"best_reply": {"type": "choice", "choice": "reply_a"}}, "usage": {}},
        ]
        with patch.object(engine, "ask_jev") as jev, \
                patch.object(engine, "ask_llm_judge", side_effect=llm_results), \
                patch.object(engine, "draft_candidates", return_value=CANDIDATES):
            result = self._analyze(judge_mode="llm_only")
        jev.assert_not_called()
        self.assertTrue(result["ranking_valid"])
        self.assertFalse(result["fallback_used"])
        self.assertEqual(result["judge_backend"], "llm")

    def test_both_initial_judges_down_does_not_retry_or_auto_rank(self):
        with patch.object(engine, "ask_jev", side_effect=JevError("offline", 529)) as jev, \
                patch.object(engine, "ask_llm_judge", side_effect=JevError("offline")) as generic, \
                patch.object(engine, "draft_candidates", return_value=CANDIDATES):
            result = self._analyze(judge_mode="jev_fallback")
        jev.assert_called_once()
        generic.assert_called_once()
        self.assertFalse(result["ranking_valid"])
        self.assertEqual(result["judge_backend"], "")

    def test_strict_rank_failure_uses_final_selector(self):
        with patch.object(engine, "ask_jev", side_effect=[
                    {"answers": JUDGMENT, "usage": {}}, JevError("offline", 529)]), \
                patch.object(engine, "ask_llm_judge", side_effect=JevError("invalid JSON")), \
                patch.object(engine, "select_best", return_value={
                    "answers": {"best_reply": {"type": "choice", "choice": "reply_c"}},
                    "usage": {}}), \
                patch.object(engine, "draft_candidates", return_value=CANDIDATES):
            result = self._analyze(judge_mode="jev_fallback")
        self.assertTrue(result["ranking_valid"])
        self.assertEqual(result["best_index"], 2)
        self.assertEqual(result["ranking_backend"], "llm")

    def test_failed_ranking_keeps_manual_candidates_but_blocks_auto_send(self):
        with patch.object(engine, "ask_jev", side_effect=[
                    {"answers": JUDGMENT, "usage": {}}, JevError("offline", 529)]), \
                patch.object(engine, "ask_llm_judge", side_effect=JevError("offline")), \
                patch.object(engine, "select_best", side_effect=JevError("offline")), \
                patch.object(engine, "draft_candidates", return_value=CANDIDATES):
            result = self._analyze(judge_mode="jev_fallback")
        self.assertEqual(result["candidates"], CANDIDATES)
        self.assertEqual(result["best_index"], 0)
        self.assertFalse(result["ranking_valid"])
        self.assertTrue(result["warnings"])


if __name__ == "__main__":
    unittest.main()
