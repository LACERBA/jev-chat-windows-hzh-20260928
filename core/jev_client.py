# -*- coding: utf-8 -*-
"""Jev 判断 API 客户端：OpenRouter、TypeSafe 直连或 OpenAI Decisions。

TypeSafe 直连走官方 `typesafe_sdk`；OpenRouter 这条是唯一自己拼 HTTP 的路——
SDK 把路径写死成 `/v1/systemone`，打不到 OpenRouter 的 `/api/alpha/decisions`。
各后端返回同一个 dict 形状，engine 不关心跑的是哪条。key 默认从环境变量读，验证时可显式传入，绝不打进日志。
"""

from __future__ import annotations

import json
import math
import os
import socket
import time
import urllib.error
import urllib.request
from email.utils import parsedate_to_datetime
from typing import NoReturn

try:  # 当模块导入 / 当脚本直接跑 都能用
    from .providers import (ENV_VARS, JEV_ENV, JEV_PROVIDERS, LEGACY,
                            OPENROUTER_DECISIONS, OPENROUTER_KEY_URL, TYPESAFE_BASE)
    from .retry import MAX_RETRIES, call_with_retry
except ImportError:
    from providers import (ENV_VARS, JEV_ENV, JEV_PROVIDERS, LEGACY,
                           OPENROUTER_DECISIONS, OPENROUTER_KEY_URL, TYPESAFE_BASE)
    from retry import MAX_RETRIES, call_with_retry


class JevError(Exception):
    def __init__(self, message: str, status: int | None = None, *,
                 retryable: bool = False, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable  # 没有状态码的网络类临时故障（超时、断网）
        self.retry_after = retry_after  # 服务端要求的等待秒数


def redact_secrets(text: str) -> str:
    """Strip every live key from any string before print or disk write."""
    if not isinstance(text, str):
        text = str(text)
    for env in ENV_VARS:
        key = os.environ.get(env) or ""
        if key:
            text = text.replace(key, "[REDACTED]")
    return text


def _status_of(exc: Exception) -> int | None:
    """各家 SDK 放 HTTP 状态码的属性名不一样：openai/anthropic 是 status_code，
    google-genai 是 code（它的 status 是 'NOT_FOUND' 这种字符串），typesafe 是 status。"""
    for name in ("status_code", "code", "status"):
        value = getattr(exc, name, None)
        if isinstance(value, int):
            return value
    return None


def _parse_retry_after(headers) -> float | None:
    """Retry-After（秒或日期）和 retry-after-ms → 秒；没有或无法解析返回 None。"""
    try:
        lowered = {str(key).lower(): value for key, value in headers.items()}
        for name, divisor in (("retry-after-ms", 1000), ("retry-after", 1)):
            raw = lowered.get(name)
            if raw is None:
                continue
            try:
                value = float(str(raw).strip()) / divisor
            except ValueError:
                if name == "retry-after":
                    return max(0.0, parsedate_to_datetime(str(raw)).timestamp() - time.time())
                continue
            if math.isfinite(value) and value >= 0:
                return value
    except Exception:
        pass
    return None


def _retry_after_of(exc: Exception) -> float | None:
    """各家 SDK 的异常把响应头放在 headers 或 response.headers 里。"""
    for source in (exc, getattr(exc, "response", None)):
        headers = getattr(source, "headers", None)
        if headers is not None and hasattr(headers, "get"):
            value = _parse_retry_after(headers)
            if value is not None:
                return value
    return None


_TRANSIENT_NAMES = ("Timeout", "Connection", "ConnectError", "ReadError", "WriteError", "RemoteProtocol")


def _is_transient(exc: Exception) -> bool:
    """没有 HTTP 状态码时，按异常类型判断是不是超时、断网这类临时故障。"""
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    return any(name in cls.__name__ for cls in type(exc).__mro__ for name in _TRANSIENT_NAMES)


def _fail(exc: Exception, what: str, *, api_key: str = "", include_detail: bool = True) -> NoReturn:
    """SDK 抛的异常 → 一句人话的 JevError。消息过脱敏，绝不把 key 带出来。"""
    if isinstance(exc, JevError):
        raise exc
    status = _status_of(exc)
    hint = {401: "密钥被拒", 403: "没有权限", 404: "模型或地址不对", 408: "请求超时", 422: "请求被拒",
            429: "被限流", 500: "服务内部错误", 502: "网关错误", 503: "服务暂不可用",
            504: "网关超时", 529: "服务过载"}.get(status, "")
    detail = str(exc) if include_detail else type(exc).__name__
    detail = detail.replace(api_key, "[REDACTED]") if api_key else detail
    detail = redact_secrets(detail).strip()[:300]
    head = f"{what} HTTP {status}" if status else f"{what}失败"
    raise JevError(f"{head}: {hint or detail or type(exc).__name__}", status,
                   retryable=status is None and _is_transient(exc),
                   retry_after=_retry_after_of(exc)) from None


def _api_key(env: str = JEV_ENV) -> str:
    """读取指定密钥槽；有老变量映射时保持向后兼容。"""
    key = ((os.environ.get(env) or "").strip()
           or (os.environ.get(LEGACY.get(env, "")) or "").strip())
    if not key:
        raise JevError(
            f"{env} is not set. Export it in the environment; "
            "do not put the key in a file."
        )
    return key


def _error_body(exc: urllib.error.HTTPError) -> str:
    try:
        raw = exc.read().decode("utf-8", errors="replace")
    except Exception:
        raw = ""
    return redact_secrets(raw)[:800]


def ask(state: dict, questions: dict, timeout: float = 20,
        provider: str = "openrouter", model: str | None = None,
        max_retries: int = MAX_RETRIES, base_url: str | None = None, api_key: str = "") -> dict:
    """问 Jev 一轮判断，返回 {"answers": {名字: 答案}, "usage": {...}}。

    provider ∈ JEV_PROVIDERS；model=None 用该来源的默认模型，base_url 仅供 OpenAI Decisions 使用。
    各后端返回的 dict 形状一模一样；限流、过载、超时等临时故障由 core.retry 统一重试，
    max_retries 是这次调用的重试上限；api_key 可用于验证未保存的密钥，不修改环境变量。绝不打印或写出 key。
    """
    spec = JEV_PROVIDERS.get(provider) or JEV_PROVIDERS["openrouter"]
    key = api_key.strip() or _api_key(JEV_ENV)  # 两家共用同一把 key，换来源不用重填
    model = model or spec.default
    if provider == "openai_decisions":
        return call_with_retry(
            lambda: _ask_decisions(state, questions, key, model, timeout, base_url or spec.base),
            what="Decisions 判断", max_retries=max_retries)
    send = _ask_typesafe if provider == "typesafe" else _ask_openrouter
    return call_with_retry(lambda: send(state, questions, key, model, timeout),
                           what="Jev 判断", max_retries=max_retries)


def _decision_questions(questions: dict) -> list[dict]:
    result = []
    for name, question in questions.items():
        kind, criteria = question["type"], question["criteria"]
        item = {"name": name, "type": kind, "instructions": question["instructions"]}
        if kind == "noul":
            item["type"] = "predicate"
            item["instructions"] += f"\nTrue: {criteria['true']}\nFalse: {criteria['false']}"
        elif kind == "choice":
            item["choices"] = [{"value": key, "description": text} for key, text in criteria.items()]
        elif kind == "score":
            item["levels"] = [{"label": str(i), "description": text} for i, text in enumerate(criteria)]
        else:
            raise JevError("Decisions 不支持此问题类型")
        result.append(item)
    return result


def _decision_answers(result: dict, questions: dict) -> dict:
    try:
        from .llm_judge import validate_answers
    except ImportError:
        from llm_judge import validate_answers

    if not isinstance(result, dict) or not isinstance(result.get("answers"), list):
        raise JevError("Decisions 结果 answers 不是数组")
    answers = {}
    for answer in result["answers"]:
        if not isinstance(answer, dict):
            raise JevError("Decisions 结果含无效答案")
        name = answer.get("name")
        if not isinstance(name, str) or name not in questions or name in answers:
            raise JevError("Decisions 结果含未知或重复的问题名")
        kind = questions[name]["type"]
        if answer.get("type") == "refusal":
            raise JevError(f"Decisions 拒绝回答问题 {name}")
        if answer.get("type") != ("predicate" if kind == "noul" else kind):
            raise JevError(f"Decisions 结果字段 {name} 的类型不匹配")
        if kind == "noul":
            answers[name] = {"type": "noul", "noul": answer.get("probability")}
            continue
        if kind == "choice" and not isinstance(answer.get("choice"), str):
            raise JevError(f"Decisions 结果字段 {name}.choice 不是字符串")
        item = {"type": kind, kind: answer.get(kind)}
        if "confidence" in answer:
            item["confidence"] = answer["confidence"]
        probabilities = answer.get("probabilities", [])
        if not isinstance(probabilities, list):
            raise JevError(f"Decisions 结果字段 {name}.probabilities 不是数组")
        mapped = {}
        for option in probabilities:
            if not isinstance(option, dict):
                raise JevError(f"Decisions 结果字段 {name}.probabilities 含无效选项")
            value = option.get("value")
            if (isinstance(value, bool) or not isinstance(value, int if kind == "score" else str)
                    or str(value) in mapped):
                raise JevError(f"Decisions 结果字段 {name}.probabilities 含无效或重复选项")
            mapped[str(value)] = option.get("probability")
        item["probabilities"] = mapped
        answers[name] = item
    # 保留 Decisions 概率加权的小数评分，输出结构仍与 Jev 一致。
    return {"answers": validate_answers(answers, questions, allow_fractional_score=True),
            "usage": result.get("usage") if isinstance(result.get("usage"), dict) else {}}


def _ask_decisions(state: dict, questions: dict, key: str, model: str,
                   timeout: float, base_url: str) -> dict:
    import openai

    payload = {"model": model, "input": json.dumps(state, ensure_ascii=False),
               "questions": _decision_questions(questions)}
    try:
        with openai.OpenAI(base_url=base_url.rstrip("/"), api_key=key,
                           timeout=timeout, max_retries=0) as client:
            # 兼容尚未提供 decisions.create 的 SDK，同时复用官方鉴权和连接管理。
            result = client.post("/decisions", body=payload, cast_to=dict)
    except Exception as exc:
        # 网关错误可能回显输入，不把原始响应正文带进日志或界面。
        _fail(exc, "Decisions 判断", api_key=key, include_detail=False)
    return _decision_answers(result, questions)


def _answer(answer) -> dict:
    """SDK 的答案对象 → OpenRouter 那条路 JSON 出来的同一个形状。"""
    if answer.type == "noul":
        return {"type": "noul", "noul": answer.noul}
    if answer.type == "choice":
        return {"type": "choice", "choice": answer.choice, "confidence": answer.confidence,
                "probabilities": dict(answer.probabilities)}
    # score：SDK 把概率的 key 转成了 int，这里转回字符串，跟 JSON 那条路对齐
    return {"type": "score", "score": answer.score, "confidence": answer.confidence,
            "probabilities": {str(k): v for k, v in answer.probabilities.items()}}


def _ask_typesafe(state: dict, questions: dict, key: str, model: str, timeout: float) -> dict:
    """官方 typesafe_sdk，单次请求。questions 原样传；SDK 自带重试已关，统一由 core.retry 负责。"""
    import typesafe_sdk

    try:
        retry = typesafe_sdk.RetryPolicy(max_retries=0, timeout=timeout)
        with typesafe_sdk.TypeSafeClient(api_key=key, base_url=TYPESAFE_BASE, model=model,
                                         timeout=timeout, retry=retry) as client:
            result = client.system_one(state, questions, model=model)
    except Exception as exc:
        _fail(exc, "Jev 判断")
    return {
        "answers": {name: _answer(a) for name, a in result.answers.items()},
        "usage": {"input_tokens": result.usage.input_tokens,
                  "output_tokens": result.usage.output_tokens},
    }


def _ask_openrouter(state: dict, questions: dict, key: str, model: str, timeout: float) -> dict:
    """OpenRouter 的 /api/alpha/decisions，手写 urllib，单次请求；重试统一由 core.retry 负责。"""
    payload = json.dumps(
        {"model": model, "state": state, "questions": questions},
        ensure_ascii=False,
    ).encode("utf-8")
    req = urllib.request.Request(
        OPENROUTER_DECISIONS,
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw)
    except urllib.error.HTTPError as exc:
        body = _error_body(exc)
        readable = {
            401: f"Jev HTTP 401: API key rejected. Check {JEV_ENV}.",
            422: f"Jev HTTP 422: request body rejected. {body}",
            429: f"Jev HTTP 429: rate limited. {body}",
            529: f"Jev HTTP 529: provider overloaded. {body}",
        }.get(exc.code, f"Jev HTTP {exc.code}: {body}")
        raise JevError(readable, exc.code, retry_after=_retry_after_of(exc)) from None
    except (TimeoutError, socket.timeout) as exc:
        raise JevError(f"Jev request timed out after {timeout}s", retryable=True) from exc
    except urllib.error.URLError as exc:
        reason = redact_secrets(getattr(exc, "reason", exc))
        raise JevError(f"Jev request failed: {reason}", retryable=True) from None


def _check_openrouter_key(key: str, timeout: float) -> None:
    """免费的 auth/key 探测：401/403 说明 key 不对，别的错（超时/断网）也如实上报。
    列表本身是写死的，key 对不对只有靠它才知道，别等第一次判断才暴露。"""
    req = urllib.request.Request(OPENROUTER_KEY_URL,
                                 headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
    except urllib.error.HTTPError as exc:
        hint = {401: "密钥被拒", 403: "没有权限"}.get(exc.code, _error_body(exc)[:200])
        raise JevError(f"取模型列表 HTTP {exc.code}: {hint}") from None
    except (TimeoutError, socket.timeout):
        raise JevError(f"取模型列表请求超时（{timeout}s）") from None
    except urllib.error.URLError as exc:
        raise JevError(
            f"取模型列表失败: {redact_secrets(getattr(exc, 'reason', exc))}") from None


def list_models(provider: str, key: str, timeout: float = 10,
                base_url: str | None = None) -> list[str]:
    """某家能用的 Jev 模型 id，去重排序。失败抛 JevError（设置页直接显示这句话）。"""
    if provider == "openai_decisions":
        import openai

        def fetch():
            try:
                with openai.OpenAI(base_url=(base_url or JEV_PROVIDERS[provider].base).rstrip("/"),
                                   api_key=key, timeout=timeout, max_retries=0) as client:
                    return sorted({model.id for model in client.models.list()})
            except Exception as exc:
                _fail(exc, "取模型列表", api_key=key, include_detail=False)

        return call_with_retry(fetch, what="取模型列表")
    if provider == "typesafe":
        import typesafe_sdk

        try:
            with typesafe_sdk.TypeSafeClient(api_key=key, base_url=TYPESAFE_BASE,
                                             timeout=timeout) as client:
                return sorted({m.name for m in client.models.list().models})
        except Exception as exc:
            _fail(exc, "取模型列表")
    # OpenRouter 的 Jev 是 Decisions API 专属模型，不在 /api/v1/models 目录里
    # （也没有列它的专用端点），列表按官方模型页写死，别名列排最前（永远指向最新版）。
    # key 对不对由探测兜着，别让坏密钥等到第一次判断才暴露。
    _check_openrouter_key(key, timeout)
    return ["~typesafe/jev-latest", "typesafe/jev-1.13"]


if __name__ == "__main__":
    # ponytail: 不联网。两条路各测一次：SDK 那条在 typesafe_sdk 边界换成假客户端，
    # urllib 那条 mock urlopen。会坏的地方就一个——答案对象 → dict 的映射得跟 JSON 那条一模一样。
    import io
    import types as _t
    from unittest.mock import patch

    import typesafe_sdk

    try:
        from .questions import JUDGE_QUESTIONS, build_rank_question
    except ImportError:
        from questions import JUDGE_QUESTIONS, build_rank_question

    os.environ.pop(JEV_ENV, None)
    os.environ["OPENROUTER_API_KEY"] = "or-key"  # 老名字：新名字没设时该退回它
    assert _api_key(JEV_ENV) == "or-key"
    os.environ[JEV_ENV] = "ts-key"  # 新名字在就用新的，两家来源共用这一把
    questions = dict(JUDGE_QUESTIONS)
    questions.update(build_rank_question(["甲", "乙", "丙"]))
    seen: dict = {}

    class _FakeClient:
        def __init__(self, **kw):
            seen["init"] = kw

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def system_one(self, state, qs, **kw):
            seen["state"], seen["questions"], seen["kw"] = state, qs, kw
            return _t.SimpleNamespace(
                answers={
                    "literal_question": _t.SimpleNamespace(type="noul", noul=0.9),
                    "best_reply": _t.SimpleNamespace(
                        type="choice", choice="reply_b", confidence=0.7,
                        probabilities={"reply_a": 0.2, "reply_b": 0.7, "reply_c": 0.1}),
                    "danger_level": _t.SimpleNamespace(
                        type="score", score=4.0, confidence=0.6, probabilities={4: 0.6, 5: 0.4}),
                },
                usage=_t.SimpleNamespace(input_tokens=11, output_tokens=22))

        @property
        def models(self):
            return _t.SimpleNamespace(list=lambda: _t.SimpleNamespace(models=(
                _t.SimpleNamespace(name="jev-preview"), _t.SimpleNamespace(name="jev-latest"))))

    with patch.object(typesafe_sdk, "TypeSafeClient", _FakeClient):
        got = ask({"chat": {}}, questions, timeout=15, provider="typesafe", model="jev-1.13.0")
        ask_init = seen["init"]
        assert list_models("typesafe", "ts-key") == ["jev-latest", "jev-preview"]
        assert seen["init"] == {"api_key": "ts-key", "base_url": TYPESAFE_BASE, "timeout": 10}
    assert ask_init["api_key"] == "ts-key" and ask_init["base_url"] == TYPESAFE_BASE
    assert ask_init["model"] == "jev-1.13.0" and ask_init["timeout"] == 15
    assert ask_init["retry"].max_retries == 0  # SDK 自带重试已关，统一走 core.retry
    assert seen["kw"] == {"model": "jev-1.13.0"}
    # 题目原样进 SDK：它们本身就是 NoulModel / ChoiceModel / ScoreModel，不用再包一层
    assert seen["questions"] is questions
    assert seen["questions"]["danger_level"]["type"] == "score"
    assert isinstance(seen["questions"]["danger_level"]["criteria"], list)
    assert seen["questions"]["best_reply"]["criteria"] == {
        "reply_a": "甲", "reply_b": "乙", "reply_c": "丙"}
    # 映射出来的形状跟 OpenRouter 那条路的 JSON 必须一致（engine 不关心跑的是哪条）
    assert got["answers"]["literal_question"] == {"type": "noul", "noul": 0.9}
    assert got["answers"]["best_reply"] == {
        "type": "choice", "choice": "reply_b", "confidence": 0.7,
        "probabilities": {"reply_a": 0.2, "reply_b": 0.7, "reply_c": 0.1}}
    assert got["answers"]["danger_level"] == {
        "type": "score", "score": 4.0, "confidence": 0.6,
        "probabilities": {"4": 0.6, "5": 0.4}}  # score 的概率 key 转回字符串
    assert got["usage"] == {"input_tokens": 11, "output_tokens": 22}

    class _Boom(Exception):
        status = 429

    with patch.object(typesafe_sdk, "TypeSafeClient", lambda **kw: (_ for _ in ()).throw(_Boom("x"))):
        try:
            ask({"chat": {}}, questions, provider="typesafe", max_retries=0)
            raise SystemExit("应当抛错")
        except JevError as e:
            assert e.status == 429 and "被限流" in str(e)

    # OpenRouter 那条没动：还是自己拼 body、打 /api/alpha/decisions
    body = {"answers": {"best_reply": {"type": "choice", "choice": "reply_a"}}, "usage": {}}

    def _fake_urlopen(req, timeout=None):
        seen["url"], seen["body"] = req.full_url, json.loads(req.data.decode("utf-8"))
        return io.BytesIO(json.dumps(body).encode("utf-8"))

    with patch.object(urllib.request, "urlopen", _fake_urlopen):
        assert ask({"chat": {}}, questions) == body
    assert seen["url"] == OPENROUTER_DECISIONS
    assert seen["body"]["model"] == "typesafe/jev-1.13" and seen["body"]["questions"] == questions

    # OpenRouter 路的列表是写死的，但 key 要过 auth/key 探测：mock urlopen 验两头
    def _fake_key_ok(req, timeout=None):
        seen["key_url"] = req.full_url
        assert req.headers["Authorization"] == "Bearer or-key"
        return io.BytesIO(b'{"data":{}}')

    with patch.object(urllib.request, "urlopen", _fake_key_ok):
        assert list_models("openrouter", "or-key") == [
            "~typesafe/jev-latest", "typesafe/jev-1.13"]
    assert seen["key_url"] == OPENROUTER_KEY_URL

    def _fake_key_rejected(req, timeout=None):
        raise urllib.error.HTTPError(OPENROUTER_KEY_URL, 401, "Unauthorized", {},
                                     io.BytesIO(b'{"error":{"message":"bad key or-key"}}'))

    with patch.object(urllib.request, "urlopen", _fake_key_rejected):
        try:
            list_models("openrouter", "or-key")
            raise SystemExit("应当抛错")
        except JevError as e:
            assert e.status is None and "密钥被拒" in str(e) and "or-key" not in str(e)

    # 401/403 以外的状态码要把响应体带出来（别提前 read 把流吃空）
    def _fake_key_429(req, timeout=None):
        raise urllib.error.HTTPError(OPENROUTER_KEY_URL, 429, "Too Many", {},
                                     io.BytesIO(b'{"error":{"message":"rate limited"}}'))

    with patch.object(urllib.request, "urlopen", _fake_key_429):
        try:
            list_models("openrouter", "or-key")
            raise SystemExit("应当抛错")
        except JevError as e:
            assert "HTTP 429" in str(e) and "rate limited" in str(e)

    assert redact_secrets("key=ts-key or-key") == "key=[REDACTED] [REDACTED]"
    print("jev_client ok")
