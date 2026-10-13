# -*- coding: utf-8 -*-
"""所有大模型调用共用的重试层：Jev、通用判断、起草、视觉识别都经过这里。

各 SDK 自带的重试已统一关闭，次数只由这里决定，避免叠加后实际次数失控。
只重试临时性故障（429 限流、408/409/425、5xx、超时、断网）；密钥、权限、模型名、
请求参数这类配置错误重试也不会好，立即抛出。
"""
from __future__ import annotations

import random
import time
from contextlib import contextmanager
from contextvars import ContextVar

MAX_RETRIES = 5  # 首次请求之后最多再试 5 次，即最多请求 6 次
BACKOFF_BASE = 3.0  # 第 1 次重试前等 3 秒，之后每次翻倍：3、6、12、24、48
BACKOFF_CAP = 48.0
JITTER_RATIO = 0.2  # 随机多等 0~20%，避免多个请求同时撞上限流窗口
RETRY_AFTER_CAP = 60.0  # 服务端要求的等待时间最多采纳 60 秒
_SLICE = 0.25  # 等待时分片检查是否已被取消
_RETRY_STATUSES = {408, 409, 425, 429}
_cancel_check = ContextVar("retry_cancel_check", default=None)
_notify = ContextVar("retry_notify", default=None)
CANCEL_NOTE = "会话内容已更新（新消息、媒体识别结果等）"


class RetryCancelled(Exception):
    """请求发出前就发现结果已过期（会话内容已更新），不再发起这次模型调用。"""


@contextmanager
def retry_cancel_guard(check):
    """当前线程内每次发请求前、重试等待期间都会调用 check()；返回 True 就放弃（比如会话内容已更新）。"""
    token = _cancel_check.set(check)
    try:
        yield
    finally:
        _cancel_check.reset(token)


@contextmanager
def retry_notify_guard(callback):
    """当前线程内每次进入重试等待前调用 callback(提示文案)，让界面能看到"正在重试"而不是干等。

    callback 在调用线程里执行，必须自己保证线程安全（比如只往队列里放东西）。"""
    token = _notify.set(callback)
    try:
        yield
    finally:
        _notify.reset(token)


def retry_notice(what: str, retry: int, max_retries: int, delay: float, error: BaseException) -> str:
    """生成给用户看的重试提示：失败原因 + 第几次重试，如"通用判断请求超时，正在自动重试（第 1/5 次，3 秒后发出）…"。"""
    status = getattr(error, "status", None)
    if status == 429:
        reason = "被限流"
    elif isinstance(status, int):
        reason = f"服务异常（HTTP {status}）"
    elif "timed out" in str(error).lower() or "超时" in str(error):
        reason = "请求超时"
    else:
        reason = "网络异常"
    return f"{what}{reason}，正在自动重试（第 {retry}/{max_retries} 次，{round(delay)} 秒后发出）…"


def is_retryable(error: BaseException) -> bool:
    """有 HTTP 状态码按状态码判断；没有状态码时只信任显式标记的网络类临时故障。"""
    status = getattr(error, "status", None)
    if isinstance(status, int):
        return status in _RETRY_STATUSES or status >= 500
    return bool(getattr(error, "retryable", False))


def retry_delay(retry_index: int, retry_after: float | None = None) -> float:
    """第 retry_index（从 0 起）次重试前的等待秒数；服务端给了 Retry-After 就不早于它。"""
    delay = min(BACKOFF_CAP, BACKOFF_BASE * 2 ** retry_index)
    delay *= 1 + JITTER_RATIO * random.random()
    if retry_after:
        delay = max(delay, min(float(retry_after), RETRY_AFTER_CAP))
    return delay


def _wait(delay: float, cancelled) -> bool:
    """睡 delay 秒；期间被取消返回 False。"""
    remaining = delay
    while remaining > 0:
        if cancelled is not None and cancelled():
            return False
        step = min(_SLICE, remaining)
        time.sleep(step)
        remaining -= step
    return not (cancelled is not None and cancelled())


def _log(event, **fields):
    """日志失败不能影响模型调用本身。"""
    try:
        try:
            from .diagnostics import log_event
        except ImportError:
            from diagnostics import log_event
        log_event(event, level="warning", **fields)
    except Exception:
        pass


def _notify_retry(what, retry, max_retries, delay, error) -> None:
    """通知界面即将重试；提示失败不能影响模型调用本身。"""
    callback = _notify.get()
    if callback is None:
        return
    try:
        callback(retry_notice(what, retry, max_retries, delay, error))
    except Exception:
        pass


def _annotate(error: BaseException, retries: int, note: str = "") -> None:
    """在最终错误信息后补上重试次数，界面和日志一眼能看出已经重试过。"""
    if not retries and not note:
        return
    suffix = note or f"（已自动重试 {retries} 次）"
    try:
        if error.args and suffix not in str(error.args[0]):
            error.args = (f"{error.args[0]}{suffix}", *error.args[1:])
    except Exception:
        pass


def call_with_retry(operation, *, what: str = "模型调用", max_retries: int | None = None):
    """执行 operation()；遇到临时性故障按退避间隔重试，成功就返回结果，否则抛出最后一次的异常。"""
    limit = MAX_RETRIES if max_retries is None else max(0, int(max_retries))
    cancelled = _cancel_check.get()
    retries = 0
    while True:
        if cancelled is not None and cancelled():  # 请求发出前先检查，过期的请求不再白占模型额度和时间
            _log("model.request.cancelled", what=what)
            raise RetryCancelled(CANCEL_NOTE)
        try:
            return operation()
        except Exception as error:
            if retries >= limit or not is_retryable(error):
                _annotate(error, retries)
                raise
            delay = retry_delay(retries, getattr(error, "retry_after", None))
            _notify_retry(what, retries + 1, limit, delay, error)
            if not _wait(delay, cancelled):
                _log("model.retry.cancelled", what=what, retry=retries + 1, max_retries=limit,
                     http_status=getattr(error, "status", None), error_type=type(error).__name__,
                     reason=str(error)[:200])
                _annotate(error, retries, f"（已自动重试 {retries} 次，因{CANCEL_NOTE}而停止等待）")
                raise
            retries += 1
            _log("model.retry", what=what, retry=retries, max_retries=limit,
                 delay_seconds=round(delay, 1), http_status=getattr(error, "status", None),
                 error_type=type(error).__name__, reason=str(error)[:200])
