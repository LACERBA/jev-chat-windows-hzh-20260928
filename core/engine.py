# -*- coding: utf-8 -*-
"""整条链的唯一入口：对话 → Jev 判断 → 带着判断起草 3 条 → Jev 排序 → 结构化结果。

平台无关。SSE 消费者、悬浮窗、命令行 demo 都只调 analyze()。
"""
from __future__ import annotations

try:
    from .draft import draft_candidates
    from .diagnostics import log_event, log_step
    from .jev_client import JevError, ask as ask_jev
    from .llm_judge import ask as ask_llm_judge, select_best, validate_answers
    from .questions import JUDGE_QUESTIONS, build_rank_question, build_state, guidance_text
except ImportError:
    from draft import draft_candidates
    from diagnostics import log_event, log_step
    from jev_client import JevError, ask as ask_jev
    from llm_judge import ask as ask_llm_judge, select_best, validate_answers
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
            llm_judge_base_url: str | None = None, llm_judge_api_key: str = "",
            jev_base_url: str | None = None) -> dict:
    """按配置用 Jev 或普通模型判断，再起草候选并完成有效排序。"""
    state = build_state(messages, relationship, keep=context, reply_to=reply_to)
    usage: dict = {}
    answers: dict = {}
    warnings = []
    judge_backend = ""
    fallback_used = False
    generic_available = True

    def generic(questions):
        step = "ranking.llm" if "best_reply" in questions else "judgment.llm"
        with log_step(step, provider=llm_judge_provider, model=llm_judge_model, timeout=timeout):
            return ask_llm_judge(state, questions, timeout=timeout, provider=llm_judge_provider,
                                model=llm_judge_model, base_url=llm_judge_base_url,
                                api_key=llm_judge_api_key)

    def jev(questions):
        step = "ranking.jev" if "best_reply" in questions else "judgment.jev"
        with log_step(step, provider=jev_provider, model=jev_model, timeout=timeout):
            return ask_jev(state, questions, timeout=timeout, provider=jev_provider, model=jev_model,
                           base_url=jev_base_url,
                           max_retries=1 if judge_mode == "jev_fallback" else None)  # 有兜底模型时 Jev 快速失败，否则用统一的 5 次

    def final_select(candidates):
        with log_step("ranking.final_select", provider=llm_judge_provider,
                      model=llm_judge_model, candidate_count=len(candidates)):
            return select_best(state, candidates, timeout=timeout, provider=llm_judge_provider,
                               model=llm_judge_model, base_url=llm_judge_base_url,
                               api_key=llm_judge_api_key)

    try:
        if judge_mode == "llm_only":
            first = generic(dict(JUDGE_QUESTIONS))
            judge_backend = "llm"
        else:
            first = jev(dict(JUDGE_QUESTIONS))
            first["answers"] = validate_answers(first.get("answers"), JUDGE_QUESTIONS,
                                                allow_fractional_score=jev_provider == "openai_decisions")
            judge_backend = "jev"
        answers = first["answers"]
        _add_usage(usage, first.get("usage"))
    except JevError as error:
        log_event("judgment.failed", level="warning", exc_info=True,
                  judge_mode=judge_mode, http_status=error.status)
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
                if fallback_error.status in _CONFIG_STATUSES:
                    raise
                warnings.append("通用判断未成功，继续尝试从候选中直接选择")
        elif judge_mode == "llm_only" and _can_fallback(error):
            warnings.append("通用判断未成功，继续尝试从候选中直接选择")
        else:
            generic_available = False
            warnings.append("判断服务暂时不可用，本次仅生成手动候选")

    intent = (answers.get("true_intent") or {}).get("choice")
    need = (answers.get("she_needs") or {}).get("choice")
    decision = (answers.get("reply_decision") or {}).get("choice")
    reply_required = not (judge_backend and decision == "no_reply"
                          and intent == "close_topic" and need == "nothing")
    log_event("judgment.reply_decision", judge_backend=judge_backend, decision=decision,
              true_intent=intent, she_needs=need, reply_required=reply_required)
    if not reply_required:
        reason = "对方已平和结束话题，没有待回答或待确认事项，本轮无需回复"
        log_event("engine.no_reply", judge_backend=judge_backend, reason=reason,
                  fallback_used=fallback_used)
        return {
            "reply_required": False,
            "no_reply_reason": reason,
            "candidates": [],
            "best_index": None,
            "best_reply": "",
            "scores": [],
            "answers": answers,
            "usage": usage,
            "reply_to": reply_to,
            "ranking_valid": False,
            "auto_send_fallback": False,
            "judge_backend": judge_backend,
            "ranking_backend": "",
            "fallback_used": fallback_used,
            "warnings": list(dict.fromkeys(warnings)),
        }

    with log_step("draft", provider=provider, model=model, timeout=timeout, thinking=thinking):
        candidates = draft_candidates(messages, relationship, provider=provider, model=model,
                                      base_url=base_url, timeout=timeout, keep=context,
                                      reply_to=reply_to, style=style, thinking=thinking,
                                      guidance=guidance_text(answers) if answers else None)
    log_event("draft.candidates", candidate_count=len(candidates))
    if not candidates:
        raise JevError("起草结果没有可用候选回复")

    ranking_valid = len(candidates) == 1 and bool(judge_backend)
    ranking_backend = judge_backend if ranking_valid else ""
    final_selector_attempted = False
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
                ranked = jev(rank_question)
                ranked["answers"] = validate_answers(ranked.get("answers"), rank_question,
                                                     allow_fractional_score=jev_provider == "openai_decisions")
                ranking_backend = "jev"
        except JevError as error:
            log_event("ranking.failed", level="warning", exc_info=True,
                      backend="llm" if use_generic else "jev", http_status=error.status)
            if use_generic:
                final_selector_attempted = True
                try:
                    ranked = final_select(candidates)
                    ranking_backend = "llm"
                    warnings.append("严格排序不可用，通用模型已完成最终候选选择")
                except JevError:
                    ranked = None
            elif judge_mode == "jev_fallback" and _can_fallback(error):
                fallback_used = True
                try:
                    ranked = generic(rank_question)
                    ranking_backend = "llm"
                    warnings.append("Jev 排序暂时不可用，本次已使用通用判断模型排序")
                except JevError:
                    final_selector_attempted = True
                    try:
                        ranked = final_select(candidates)
                        ranking_backend = "llm"
                        warnings.append("Jev 排序暂时不可用，通用模型已完成最终候选选择")
                    except JevError:
                        ranked = None
            if ranked is None:
                warnings.append("通用模型未能排序，将以第一条候选作为默认选择")
        if ranked:
            answers = {**answers, **ranked["answers"]}
            _add_usage(usage, ranked.get("usage"))
            ranking_valid = True
    if (len(candidates) >= 2 and not ranking_valid and not final_selector_attempted
            and generic_available and judge_mode in ("jev_fallback", "llm_only")):
        try:
            ranked = final_select(candidates)
            answers = {**answers, **ranked["answers"]}
            _add_usage(usage, ranked.get("usage"))
            ranking_backend = "llm"
            ranking_valid = True
            fallback_used = fallback_used or judge_mode == "jev_fallback"
            warnings.append("已由通用模型直接从候选中选择最佳回答")
        except JevError:
            warnings.append("通用模型未能选出候选；本次不会自动发送")
    if not ranking_valid:
        warnings.append("候选排序不可用，将以第一条候选作为默认选择")

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

    log_event("engine.completed", candidate_count=len(candidates), best_index=best_index,
              judge_backend=judge_backend, ranking_backend=ranking_backend,
              ranking_valid=ranking_valid, fallback_used=fallback_used,
              warnings=list(dict.fromkeys(warnings)))
    return {
        "reply_required": True,
        "no_reply_reason": "",
        "candidates": candidates,
        "best_index": best_index,
        "best_reply": candidates[best_index],
        "scores": scores,
        "answers": answers,
        "usage": usage,
        "reply_to": reply_to,
        "ranking_valid": ranking_valid,
        "auto_send_fallback": bool(candidates) and not ranking_valid,
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
