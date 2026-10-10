# -*- coding: utf-8 -*-
"""整条链的唯一入口：对话 → Jev 判断 → 带着判断起草 3 条 → Jev 排序 → 结构化结果。

平台无关。SSE 消费者、悬浮窗、命令行 demo 都只调 analyze()。
"""
from __future__ import annotations

try:
    from .draft import draft_candidates
    from .jev_client import JevError, ask as ask_jev
    from .llm_judge import ask as ask_llm_judge, validate_answers
    from .questions import JUDGE_QUESTIONS, build_rank_question, build_state, guidance_text
except ImportError:
    from draft import draft_candidates
    from jev_client import JevError, ask as ask_jev
    from llm_judge import ask as ask_llm_judge, validate_answers
    from questions import JUDGE_QUESTIONS, build_rank_question, build_state, guidance_text

_REPLY_IDX = {"reply_a": 0, "reply_b": 1, "reply_c": 2}


def _add_usage(total: dict, one: dict | None) -> None:
    """两次 Jev 调用的 usage 相加（tokens、cost）；非数字的字段后来的盖掉前面的。"""
    for k, v in (one or {}).items():
        total[k] = total.get(k, 0) + v if isinstance(v, (int, float)) else v


_CONFIG_STATUSES = {401, 403, 404, 422}


def _can_fallback(error: JevError) -> bool:
    return error.status is None or error.status in (408, 409, 425, 429, 529) or (error.status >= 500)


def analyze(messages: list, relationship: str, model: str | None = None,
            timeout: float = 30, context: int = 10, provider: str = "deepseek",
            base_url: str | None = None, reply_to: str | None = None, style: str = "",
            thinking: bool = False, jev_provider: str = "openrouter",
            jev_model: str | None = None, judge_mode: str = "jev_fallback",
            llm_judge_provider: str = "deepseek", llm_judge_model: str | None = None,
            llm_judge_base_url: str | None = None, llm_judge_api_key: str = "") -> dict:
    """按配置用 Jev 或普通模型判断，再起草候选并完成有效排序。"""
    state = build_state(messages, relationship, keep=context, reply_to=reply_to)
    usage: dict = {}
    answers: dict = {}
    warnings = []
    judge_backend = ""
    fallback_used = False
    generic_available = True

    def generic(questions):
        return ask_llm_judge(state, questions, timeout=timeout, provider=llm_judge_provider,
                             model=llm_judge_model, base_url=llm_judge_base_url,
                             api_key=llm_judge_api_key)

    try:
        if judge_mode == "llm_only":
            first = generic(dict(JUDGE_QUESTIONS))
            judge_backend = "llm"
        else:
            first = ask_jev(state, dict(JUDGE_QUESTIONS), timeout=timeout,
                            provider=jev_provider, model=jev_model,
                            max_retries=1 if judge_mode == "jev_fallback" else 3)
            first["answers"] = validate_answers(first.get("answers"), JUDGE_QUESTIONS)
            judge_backend = "jev"
        answers = first["answers"]
        _add_usage(usage, first.get("usage"))
    except JevError as error:
        if error.status in _CONFIG_STATUSES:
            raise
        first = None
        if judge_mode == "jev_fallback" and _can_fallback(error):
            fallback_used = True
            try:
                first = generic(dict(JUDGE_QUESTIONS))
                answers = first["answers"]
                judge_backend = "llm"
                _add_usage(usage, first.get("usage"))
                warnings.append("Jev 暂时不可用，本次已使用通用判断模型")
            except JevError as fallback_error:
                generic_available = False
                if fallback_error.status in _CONFIG_STATUSES:
                    raise
                warnings.append("判断服务暂时不可用，本次仅生成手动候选")
        else:
            generic_available = False
            warnings.append("判断服务暂时不可用，本次仅生成手动候选")

    candidates = draft_candidates(messages, relationship, provider=provider, model=model,
                                  base_url=base_url, timeout=timeout, keep=context,
                                  reply_to=reply_to, style=style, thinking=thinking,
                                  guidance=guidance_text(answers) if answers else None)
    if not candidates:
        raise JevError("起草结果没有可用候选回复")

    ranking_valid = len(candidates) == 1 and bool(judge_backend)
    ranking_backend = judge_backend if ranking_valid else ""
    if len(candidates) >= 2 and judge_backend:
        rank_question = build_rank_question(candidates)
        ranked = None
        use_generic = judge_mode == "llm_only" or judge_backend == "llm"
        try:
            if use_generic:
                if generic_available:
                    ranked = generic(rank_question)
                    ranking_backend = "llm"
            else:
                ranked = ask_jev(state, rank_question, timeout=timeout,
                                 provider=jev_provider, model=jev_model,
                                 max_retries=1 if judge_mode == "jev_fallback" else 3)
                ranked["answers"] = validate_answers(ranked.get("answers"), rank_question)
                ranking_backend = "jev"
        except JevError as error:
            if (judge_mode == "jev_fallback" and not use_generic and _can_fallback(error)):
                fallback_used = True
                try:
                    ranked = generic(rank_question)
                    ranking_backend = "llm"
                    warnings.append("Jev 排序暂时不可用，本次已使用通用判断模型排序")
                except JevError:
                    ranked = None
            if ranked is None:
                warnings.append("候选排序不可用，请手动选择；本次不会自动发送")
        if ranked:
            answers = {**answers, **ranked["answers"]}
            _add_usage(usage, ranked.get("usage"))
            ranking_valid = True
        elif not warnings or "候选排序不可用，请手动选择；本次不会自动发送" not in warnings:
            warnings.append("候选排序不可用，请手动选择；本次不会自动发送")
    if not ranking_valid and "候选排序不可用，请手动选择；本次不会自动发送" not in warnings:
        warnings.append("候选排序不可用，请手动选择；本次不会自动发送")

    best_key = (answers.get("best_reply") or {}).get("choice")
    best_index = _REPLY_IDX.get(best_key, 0)
    if best_index >= len(candidates):
        best_index = 0
        ranking_valid = False

    scores = [0.0, 0.0, 0.0]
    if ranking_backend == "jev" and ranking_valid:
        probabilities = (answers.get("best_reply") or {}).get("probabilities") or {}
        for key, idx in _REPLY_IDX.items():
            try:
                scores[idx] = float(probabilities.get(key, 0.0))
            except (TypeError, ValueError):
                scores[idx] = 0.0

    return {
        "candidates": candidates,
        "best_index": best_index,
        "best_reply": candidates[best_index],
        "scores": scores,
        "answers": answers,
        "usage": usage,
        "reply_to": reply_to,
        "ranking_valid": ranking_valid,
        "judge_backend": judge_backend,
        "ranking_backend": ranking_backend,
        "fallback_used": fallback_used,
        "warnings": list(dict.fromkeys(warnings)),
    }


if __name__ == "__main__":
    # 候选被过滤光时要抛 JevError，不能在取第一条时 IndexError。
    from unittest.mock import patch

    with patch("__main__.ask_jev", side_effect=JevError("offline", 529)), \
         patch("__main__.draft_candidates", return_value=[]):
        try:
            analyze([("her", "hello")], "friends", judge_mode="jev_only")
            raise SystemExit("应当抛错")
        except JevError as e:
            assert "没有可用候选" in str(e)
    print("engine ok")
